from __future__ import annotations
"""
export_bundle_from_data.py
Deterministic FAISS bundle exporter (strict input layout).

Inputs (required):
    data_dir/chapter_1.jsonl
    data_dir/embeddings.npy

Outputs:
    out_bundle/
        chunks.jsonl      (one flattened chunk per line)
        id_map.pkl        (pickle of list of flattened chunk dicts)
        embeddings.bin    (normalized embeddings bytes, row-major float32)
        index.faiss       (FAISS IndexFlatIP of normalized embeddings)
        model.json
        manifest.json
        version.txt
"""
import json
import os
from typing import Any, Dict, List
from pathlib import Path
import sys
import traceback

# try to import build_prompt_teacher; provide clear fallback if missing
try:
    from src.rag.build_prompt_teacher import build_prompt_teacher
except Exception:
    try:
        from src.rag.build_prompt import build_prompt as build_prompt_teacher
    except Exception:
        def build_prompt_teacher(*args, **kwargs):
            raise ImportError(
                "build_prompt_teacher not found. Define build_prompt_teacher in src/rag/build_prompt_teacher.py "
                "or provide src/rag/build_prompt.build_prompt as a fallback."
            )

def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def collect_json_chunks_from_dir(data_root: Path) -> List[Dict[str, Any]]:
    """
    Collect JSON files under data_root (recursively) and turn them into flattened chunks.
    Heuristics to extract text from common JSON shapes.
    """
    data_root = Path(data_root).resolve()
    chunks: List[Dict[str, Any]] = []
    for p in sorted(data_root.rglob("*.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        rel = p.relative_to(data_root)
        parts = rel.parts
        subject = parts[0] if parts else "unknown"
        chapter = p.stem.replace(" ", "_").lower()

        def make_chunk(text: str, idx: Optional[int] = None) -> Dict[str, Any]:
            txt = (text or "").strip()
            if not txt:
                return {}
            h = _md5(txt)
            suffix = f"_{idx}" if idx is not None else ""
            cid = f"{subject}_{chapter}_{h[:8]}{suffix}"
            return {
                "id": cid,
                "class": 0,
                "subject": subject.lower(),
                "chapter": chapter,
                "language": "en",
                "textbook": p.stem,
                "tokens": len(txt.split()),
                "hash": h,
                "text": txt,
            }

        if isinstance(raw, list):
            for i, item in enumerate(raw):
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or json.dumps(item, ensure_ascii=False)
                else:
                    text = str(item)
                c = make_chunk(text, i)
                if c:
                    chunks.append(c)
        elif isinstance(raw, dict):
            if "chapters" in raw and isinstance(raw["chapters"], list):
                for i, ch in enumerate(raw["chapters"]):
                    t = ch.get("text") or ch.get("content") or json.dumps(ch, ensure_ascii=False)
                    c = make_chunk(t, i)
                    if c:
                        chunks.append(c)
            elif "sections" in raw and isinstance(raw["sections"], list):
                for i, s in enumerate(raw["sections"]):
                    t = s.get("text") or s.get("content") or json.dumps(s, ensure_ascii=False)
                    c = make_chunk(t, i)
                    if c:
                        chunks.append(c)
            else:
                text = raw.get("text") or raw.get("content") or raw.get("body") or json.dumps(raw, ensure_ascii=False)
                c = make_chunk(text)
                if c:
                    chunks.append(c)
        else:
            text = str(raw)
            c = make_chunk(text)
            if c:
                chunks.append(c)
    return chunks

def embed_texts(texts, model_name="all-mpnet-base-v2"):
    """
    Compute and L2-normalize embeddings using sentence-transformers.
    Uses conservative env vars and falls back to sequential encode if batch encoding fails.
    """
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available; install with: python -m pip install sentence-transformers")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    model = SentenceTransformer(model_name, device="cpu")
    try:
        embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=32)
    except Exception:
        embs_list = []
        for t in texts:
            vec = model.encode(t, convert_to_numpy=True)
            embs_list.append(np.asarray(vec))
        embs = np.vstack(embs_list)

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return embs.astype("float32")

def export_bundle_from_collected_chunks(out_bundle: Path, chunks: List[Dict[str, Any]], embeddings: Optional[np.ndarray] = None):
    out_bundle = Path(out_bundle)
    out_bundle.mkdir(parents=True, exist_ok=True)

    # write chunks.jsonl
    with (out_bundle / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    # write id_map.pkl (store minimal metadata)
    id_map = []
    for c in chunks:
        id_map.append({
            "id": c.get("id"),
            "subject": c.get("subject"),
            "chapter": c.get("chapter"),
            "textbook": c.get("textbook"),
            "tokens": c.get("tokens"),
            "hash": c.get("hash"),
            "text": c.get("text"),
        })
    with (out_bundle / "id_map.pkl").open("wb") as f:
        pickle.dump(id_map, f)

    # optional embeddings + index
    if embeddings is not None:
        if embeddings.shape[0] != len(chunks):
            raise RuntimeError("embeddings row count does not match number of chunks")
        (out_bundle / "embeddings.bin").write_bytes(embeddings.astype("float32").tobytes())
        if faiss is not None:
            d = int(embeddings.shape[1])
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            faiss.write_index(index, str(out_bundle / "index.faiss"))
        with (out_bundle / "model.json").open("w", encoding="utf-8") as f:
            json.dump({"name": "precomputed", "dim": int(embeddings.shape[1])}, f, ensure_ascii=False)

    manifest = {
        "class": chunks[0].get("class", 0) if chunks else 0,
        "subject": chunks[0].get("subject", "unknown") if chunks else "unknown",
        "chapter": chunks[0].get("chapter", "unknown") if chunks else "unknown",
        "language": chunks[0].get("language", "en") if chunks else "en",
        "textbook": chunks[0].get("textbook", "unknown") if chunks else "unknown",
        "chunk_count": len(chunks),
        "model": "precomputed" if embeddings is not None else "none",
        "version": "2025.01.00"
    }
    with (out_bundle / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with (out_bundle / "version.txt").open("w", encoding="utf-8") as f:
        f.write("2025.01.00\n")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Path to data dir (JSONL file or a folder containing JSON files)")
    p.add_argument("--out-bundle", required=True, help="Output bundle path")
    p.add_argument("--compute-embeddings", action="store_true", help="Compute embeddings using sentence-transformers (optional)")
    p.add_argument("--embed-model", default="all-mpnet-base-v2", help="Sentence-transformers model name")
    args = p.parse_args()

    data_path = Path(args.data_dir)
    out_bundle = Path(args.out_bundle)

    # JSONL/ndjson input support
    if data_path.is_file() and data_path.suffix in (".jsonl", ".ndjson"):
        try:
            export_bundle(data_path, out_bundle)  # try to reuse older flow if present
            return 0
        except NameError:
            chunks = []
            for line in data_path.read_text(encoding="utf-8").splitlines():
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = obj.get("text") or obj.get("content") or json.dumps(obj, ensure_ascii=False)
                if not text.strip():
                    continue
                h = _md5(text)
                cid = obj.get("id") or f"unknown_{data_path.stem}_{h[:8]}"
                chunks.append({
                    "id": cid,
                    "class": obj.get("class", 0),
                    "subject": obj.get("subject", "unknown"),
                    "chapter": obj.get("chapter", data_path.stem),
                    "language": obj.get("language", "en"),
                    "textbook": obj.get("textbook", data_path.stem),
                    "tokens": obj.get("tokens", len(text.split())),
                    "hash": h,
                    "text": text,
                })
            if not chunks:
                raise SystemExit(f"No valid chunks found in {data_path}")
            embeddings = None
            if args.compute_embeddings:
                texts = [c["text"] for c in chunks]
                embeddings = embed_texts(texts, model_name=args.embed_model)
            export_bundle_from_collected_chunks(out_bundle, chunks, embeddings)
            print("Export complete:", out_bundle)
            return 0

    # Directory input: collect .json files recursively
    if data_path.is_dir():
        chunks = collect_json_chunks_from_dir(data_path)
        if not chunks:
            raise SystemExit(f"No JSON files/chunks found under: {data_path}")
        embeddings = None
        if args.compute_embeddings:
            texts = [c["text"] for c in chunks]
            embeddings = embed_texts(texts, model_name=args.embed_model)
        export_bundle_from_collected_chunks(out_bundle, chunks, embeddings)
        print("Export complete:", out_bundle)
        return 0

    raise SystemExit(f"Path not found or unsupported: {data_path}")

if __name__ == "__main__":
    raise SystemExit(main())