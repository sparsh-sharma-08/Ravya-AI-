"""
Teacher-mode prompt builder.

Creates a strict instruction prompt that:
- Addresses the model as "CBSE teacher assistant"
- Requires using ONLY the provided context chunks
- Requires JSON output with exact schema:
  {"content":"<long notes>", "sources":["id1","id2",...]}
- Includes structured sections and a 200-300+ word requirement
- Adds the context chunks in the specified "[<chunk_id>]\n<chunk_text>" format
"""
from __future__ import annotations
from typing import List, Dict
import textwrap
import json
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# optional deps
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def collect_json_chunks_from_dir(data_root: Path) -> List[Dict[str, Any]]:
    """
    Collect JSON files under data_root (recursively) and turn them into flattened chunks.
    Simple heuristics to extract text from common JSON shapes.
    """
    data_root = data_root.resolve()
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

def embed_texts(texts: List[str], model_name: str = "all-mpnet-base-v2") -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available in this environment")
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return embs.astype("float32")

def export_bundle_from_collected_chunks(out_bundle: Path, chunks: List[Dict[str, Any]], embeddings: Optional[np.ndarray] = None):
    out_bundle.mkdir(parents=True, exist_ok=True)
    chunks_file = out_bundle / "chunks.jsonl"
    with chunks_file.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(out_bundle / "id_map.pkl", "wb") as f:
        pickle.dump(chunks, f)
    if embeddings is not None:
        if embeddings.shape[0] != len(chunks):
            raise RuntimeError("embeddings row count does not match number of chunks")
        (out_bundle / "embeddings.bin").write_bytes(embeddings.astype("float32").tobytes())
        if faiss is not None:
            d = embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            faiss.write_index(index, str(out_bundle / "index.faiss"))
        with open(out_bundle / "model.json", "w", encoding="utf-8") as f:
            json.dump({"name": "precomputed", "dim": int(embeddings.shape[1])}, f, ensure_ascii=False)
    manifest = {
        "class": chunks[0].get("class", 0) if chunks else 0,
        "subject": chunks[0].get("subject", "unknown") if chunks else "unknown",
        "chapter": chunks[0].get("chapter", "unknown") if chunks else "unknown",
        "language": chunks[0].get("language", "en") if chunks else "en",
        "textbook": chunks[0].get("textbook", "unknown") if chunks else "unknown",
        "chunk_count": len(chunks),
        "model": "precomputed",
        "version": "2025.01.00"
    }
    with open(out_bundle / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(out_bundle / "version.txt", "w", encoding="utf-8") as f:
        f.write("2025.01.00\n")

def build_prompt_teacher(question: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Construct teacher-mode prompt. Use only provided chunks.
    """
    # header with explicit rules
    header = (
        "You are a strict syllabus-aware assistant. Use ONLY the provided CONTEXT to answer.\n"
        "If the CONTEXT contains information that answers the QUESTION, produce a concise factual answer (2-5 lines) "
        "and include the exact chunk id(s) used in the \"sources\" array.\n"
        "If the CONTEXT does NOT contain information to answer, return an empty answer string and an empty sources array.\n"
        "Do NOT hallucinate. If chunks are empty or irrelevant, respond with out_of_syllabus (handled by the system).\n\n"
        "Output (EXACTLY one JSON object):\n"
        "  \"answer\": string (concise; empty string if not available),\n"
        "  \"sources\": array of chunk id strings. Use EXACTLY the chunk ids shown above. "
        "Do NOT add prefixes, suffixes, or paths. Copy-paste ids exactly.\n\n"
    )

    # list chunks as id + text
    ctx_parts = []
    for c in chunks:
        ctx_parts.append(f"{c.get('id')}\nTEXT:\n{(c.get('text') or '')}\n---")

    ctx = "\n".join(ctx_parts)

    prompt = f"{header}\nQUESTION:\n{question}\n\nCONTEXT (use only this):\n{ctx}\n\nReturn the JSON object now and nothing else."
    return prompt

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

    # If a JSONL/ndjson file is provided, try to use existing exporter flow if available
    if data_path.is_file() and data_path.suffix in (".jsonl", ".ndjson"):
        try:
            # existing function in this file (if present) may be named export_bundle
            export_bundle(data_path, out_bundle)
            return 0
        except NameError:
            # fall back to simple conversion: read jsonl lines and treat them as chunks
            chunks = []
            for line in data_path.read_text(encoding="utf-8").splitlines():
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                # basic normalization (ensure 'text' exists)
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

    # If a directory was provided, collect all .json files under it and build chunks
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