from __future__ import annotations
"""
src/rag/export_bundle_from_data.py
Create a deterministic FAISS bundle from /data/chapter_1.jsonl and /data/embeddings.npy.

Writes:
./bundle/class_8_science_en/
  - chunks.jsonl
  - id_map.pkl
  - embeddings.bin
  - index.faiss
  - model.json
  - manifest.json
  - version.txt
"""
import json
import os
import pickle
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import faiss
except Exception:
    faiss = None


REQUIRED_FIELDS = {"text", "class", "subject", "chapter", "language", "textbook", "tokens"}


def _md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def validate_and_load_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    out: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Invalid JSON on line {i+1}: {e}")
            missing = REQUIRED_FIELDS - set(obj.keys())
            if missing:
                raise RuntimeError(f"Line {i+1} missing fields: {sorted(missing)}")
            out.append(obj)
    if not out:
        raise RuntimeError("No chunks found in JSONL")
    return out


def export_bundle(data_dir: str, out_bundle: str) -> None:
    data_dir_p = Path(data_dir)
    jsonl = data_dir_p / "chapter_1.jsonl"
    emb_np = data_dir_p / "embeddings.npy"

    if not emb_np.exists():
        raise FileNotFoundError(f"embeddings.npy not found: {emb_np}")
    embeddings = np.load(str(emb_np))
    if embeddings.ndim != 2:
        raise RuntimeError("embeddings.npy must be 2D (N x D)")

    chunks = validate_and_load_jsonl(jsonl)
    if len(chunks) != embeddings.shape[0]:
        raise RuntimeError(
            f"Chunk count {len(chunks)} != embeddings rows {embeddings.shape[0]}"
        )

    dim = int(embeddings.shape[1])

    # normalize embeddings (in-memory)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_norm = (embeddings / norms).astype(np.float32, copy=False)

    out_dir = Path(out_bundle)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids: List[str] = []
    chunks_out: List[Dict[str, Any]] = []
    for obj in chunks:
        text = obj["text"].strip()
        md5 = _md5_text(text)
        cid = f"{obj['class']}_{obj['subject']}_{obj['chapter']}_{md5}"
        ids.append(cid)
        metadata = {
            "id": cid,
            "class": obj["class"],
            "subject": obj["subject"],
            "chapter": obj["chapter"],
            "language": obj["language"],
            "textbook": obj["textbook"],
            "tokens": obj["tokens"],
            "hash": md5,
        }
        chunks_out.append({"metadata": metadata, "text": text})

    # write chunks.jsonl
    chunks_path = out_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for c in chunks_out:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    # write id_map.pkl
    id_map_path = out_dir / "id_map.pkl"
    with id_map_path.open("wb") as fh:
        pickle.dump(ids, fh)

    # write embeddings.bin
    emb_bin_path = out_dir / "embeddings.bin"
    embeddings_norm.tofile(str(emb_bin_path))

    # build FAISS index
    if faiss is None:
        raise RuntimeError("faiss not installed. Install faiss-cpu to build index.")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_norm)
    index_path = out_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    # model.json
    model_meta = {"name": "precomputed", "dim": dim}
    with (out_dir / "model.json").open("w", encoding="utf-8") as fh:
        json.dump(model_meta, fh, ensure_ascii=False, indent=2)

    # manifest.json
    first = chunks_out[0]["metadata"]
    manifest = {
        "class": first["class"],
        "subject": first["subject"],
        "chapter": first["chapter"],
        "language": first["language"],
        "textbook": first["textbook"],
        "chunk_count": len(chunks_out),
        "model": "precomputed",
        "version": "2025.01.00",
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    with (out_dir / "version.txt").open("w", encoding="utf-8") as fh:
        fh.write("2025.01.00\n")

    print(f"Wrote bundle to {out_dir} (chunks={len(chunks_out)} dim={dim})")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Export deterministic FAISS bundle from /data")
    p.add_argument("--data-dir", default="/data", help="Path to data directory containing chapter_1.jsonl and embeddings.npy")
    p.add_argument("--out-bundle", default="./bundle/class_8_science_en", help="Output bundle directory")
    args = p.parse_args()
    try:
        export_bundle(args.data_dir, args.out_bundle)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()