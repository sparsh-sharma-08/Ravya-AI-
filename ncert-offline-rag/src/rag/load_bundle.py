from __future__ import annotations
"""
src/rag/load_bundle.py
Load FAISS bundle and return structured data.
"""
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None


def load_bundle(bundle_path: str) -> Dict[str, Any]:
    if faiss is None:
        raise RuntimeError("faiss not available. Install faiss-cpu on this device.")
    p = Path(bundle_path)
    if not p.is_dir():
        raise FileNotFoundError(f"Bundle path not found: {bundle_path}")

    required = ["index.faiss", "embeddings.bin", "id_map.pkl", "chunks.jsonl", "model.json", "manifest.json"]
    for name in required:
        f = p / name
        if not f.exists():
            raise FileNotFoundError(f"Missing bundle file: {f}")

    # load index
    index = faiss.read_index(str(p / "index.faiss"))

    # load ids
    with open(p / "id_map.pkl", "rb") as fh:
        ids = pickle.load(fh)
    if not isinstance(ids, list):
        raise RuntimeError("id_map.pkl must contain a list of ids")

    # load chunks in order
    chunks: List[Dict[str, Any]] = []
    with open(p / "chunks.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Invalid JSON in chunks.jsonl: {e}")
            chunks.append(obj)

    # load model meta and embeddings
    with open(p / "model.json", "r", encoding="utf-8") as fh:
        model_meta = json.load(fh)
    dim = int(model_meta.get("dim", 0))

    emb = None
    emb_path = p / "embeddings.bin"
    if emb_path.exists():
        arr = np.fromfile(str(emb_path), dtype=np.float32)
        if dim > 0:
            try:
                emb = arr.reshape(len(ids), dim)
            except Exception as e:
                raise RuntimeError(f"Failed to reshape embeddings.bin: {e}")
        else:
            if len(ids) == 0:
                raise RuntimeError("Cannot infer embedding dim: no ids")
            if arr.size % len(ids) != 0:
                raise RuntimeError("embeddings.bin size not divisible by id count")
            emb = arr.reshape(len(ids), arr.size // len(ids))
            dim = emb.shape[1]

    with open(p / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    return {
        "index": index,
        "ids": ids,
        "chunks": chunks,
        "embeddings": emb,
        "model_dim": dim,
        "manifest": manifest,
    }


if __name__ == "__main__":
    import argparse, json as _json
    p = argparse.ArgumentParser()
    p.add_argument("bundle", help="Path to bundle directory")
    args = p.parse_args()
    try:
        b = load_bundle(args.bundle)
        print(_json.dumps({"chunks": len(b["chunks"]), "dim": b["model_dim"]}))
    except Exception as e:
        print(_json.dumps({"error": str(e)}))
        sys.exit(2)