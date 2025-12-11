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
import argparse
import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

REQUIRED_FIELDS = {
    "text",
    "class",
    "subject",
    "chapter",
    "language",
    "textbook",
    "tokens",
}


def md5_text(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()


def normalize_embedding_matrix(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


def _coerce_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        raise RuntimeError(f"Cannot coerce value to int: {v!r}")


def _flatten_and_validate(obj: Dict[str, Any], line_no: int) -> Dict[str, Any]:
    """
    Flatten chunk object:
    - Merge metadata/meta into top-level (metadata overrides top-level)
    - Ensure required fields present
    - Enforce types/normalization:
        class, tokens -> int
        subject -> lowercase str
        chapter -> string (may be a title)
        text -> stripped str
        hash -> md5 string (preserve if given, else compute)
    - Compute deterministic id: "<class>_<subject>_<chapter>_<hash8>"
    Returns flattened dict.
    """
    if not isinstance(obj, dict):
        raise RuntimeError(f"Line {line_no}: chunk is not a JSON object")

    # Start with a shallow copy of top-level (excluding metadata/meta)
    top = {k: v for k, v in obj.items() if k not in ("metadata", "meta")}

    # Merge metadata (metadata overrides top-level)
    meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    # Apply meta over top
    merged = dict(top)
    merged.update(meta)

    # Validate presence of required fields
    missing = REQUIRED_FIELDS - set(merged.keys())
    if missing:
        raise RuntimeError(f"Line {line_no} missing required fields: {sorted(missing)}")

    # Normalize types/values
    text = (merged.get("text") or "").strip()
    subj = str(merged.get("subject") or "").strip().lower()
    cls = _coerce_int(merged.get("class"))
    # Chapter may be a title string (do NOT coerce to int)
    chap = str(merged.get("chapter") or "").strip().lower()
    lang = str(merged.get("language") or "").strip()
    tb = str(merged.get("textbook") or "").strip()
    tokens = _coerce_int(merged.get("tokens"))

    # Preserve provided hash if present, else compute from text
    h = merged.get("hash") or merged.get("sha") or md5_text(text)
    h = str(h)

    # Deterministic id per spec (use chapter string)
    cid = f"{cls}_{subj}_{chap}_{h[:8]}"

    flat = {
        "id": cid,
        "class": cls,
        "subject": subj,
        "chapter": chap,
        "language": lang,
        "textbook": tb,
        "tokens": tokens,
        "hash": h,
        "text": text,
    }

    # Preserve any other metadata keys (that are not in standard keys)
    for k, v in merged.items():
        if k in flat:
            continue
        if k in ("metadata", "meta"):
            continue
        flat[k] = v

    return flat


def export_bundle(data_dir: str, out_bundle: str) -> None:
    data_dir_p = Path(data_dir)
    out_dir = Path(out_bundle)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = data_dir_p / "chapter_1.jsonl"
    emb_path = data_dir_p / "embeddings.npy"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Expected JSONL at: {jsonl_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Expected embeddings.npy at: {emb_path}")

    # Load chunks in order (line order preserved)
    raw_chunks: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for i, ln in enumerate(fh, start=1):
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception as e:
                raise RuntimeError(f"Invalid JSON on line {i}: {e}")
            raw_chunks.append((i, obj))  # keep line number for better errors

    # Load embeddings and validate shape/order alignment
    embeddings = np.load(str(emb_path))
    if embeddings.ndim != 2:
        raise RuntimeError("embeddings.npy must be 2D (N x D)")
    if len(raw_chunks) != embeddings.shape[0]:
        raise RuntimeError(f"Chunk count {len(raw_chunks)} != embedding rows {embeddings.shape[0]}")

    dim = int(embeddings.shape[1])
    embeddings_norm = normalize_embedding_matrix(embeddings)

    # Build flattened chunks in same order
    flattened: List[Dict[str, Any]] = []
    for line_no, obj in raw_chunks:
        flat = _flatten_and_validate(obj, line_no)
        flattened.append(flat)

    # Write chunks.jsonl (one flattened JSON per line)
    chunks_file = out_dir / "chunks.jsonl"
    with chunks_file.open("w", encoding="utf-8") as fh:
        for c in flattened:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Write id_map.pkl as full flattened chunk objects (list)
    id_map_path = out_dir / "id_map.pkl"
    with id_map_path.open("wb") as fh:
        pickle.dump(flattened, fh)

    # Write embeddings.bin (normalized float32 bytes)
    emb_bin_path = out_dir / "embeddings.bin"
    emb_bytes = embeddings_norm.astype(np.float32).tobytes()
    emb_bin_path.write_bytes(emb_bytes)

    # Build and write FAISS index (inner-product on normalized vectors)
    if faiss is None:
        raise RuntimeError("faiss is required to build index (install faiss-cpu).")
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(embeddings_norm))
    faiss.write_index(index, str(out_dir / "index.faiss"))

    # model.json
    model_json = {"name": "precomputed", "dim": dim}
    (out_dir / "model.json").write_text(json.dumps(model_json, indent=2), encoding="utf-8")

    # manifest.json using first chunk metadata (chapter kept as string)
    first = flattened[0]
    manifest = {
        "class": first["class"],
        "subject": first["subject"],
        "chapter": first["chapter"],
        "language": first["language"],
        "textbook": first["textbook"],
        "chunk_count": len(flattened),
        "model": "precomputed",
        "version": "2025.01.00",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # version.txt
    (out_dir / "version.txt").write_text("2025.01.00\n", encoding="utf-8")

    print(f"Wrote bundle to {out_dir} (chunks={len(flattened)} dim={dim})")


def main() -> int:
    p = argparse.ArgumentParser(description="Export deterministic FAISS bundle")
    p.add_argument("--data-dir", required=True, help="Directory containing chapter_1.jsonl and embeddings.npy")
    p.add_argument("--out-bundle", required=True, help="Output bundle directory")
    args = p.parse_args()
    export_bundle(args.data_dir, args.out_bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())