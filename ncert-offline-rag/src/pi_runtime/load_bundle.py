"""
Load FAISS bundle for Pi runtime.

Returns a dict:
{
  "index": faiss.Index,
  "ids": [str],
  "chunks": [ { "metadata": {...}, "text": "..." } ],
  "embeddings": np.ndarray (N, dim),
  "model_dim": int,
  "manifest": {...}
}
"""

from __future__ import annotations
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import faiss
except Exception:
    faiss = None


def load_bundle(bundle_path: str) -> Dict[str, Any]:
    """
    Load the FAISS bundle from disk.
    Fail early if anything is missing.
    """

    if faiss is None:
        raise RuntimeError("faiss not installed. Run: pip install faiss-cpu")

    p = Path(bundle_path)
    if not p.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_path}")

    files = {
        "index": p / "index.faiss",
        "embeddings": p / "embeddings.bin",
        "id_map": p / "id_map.pkl",
        "chunks": p / "chunks.jsonl",
        "model": p / "model.json",
        "manifest": p / "manifest.json",
    }

    for name, f in files.items():
        if not f.exists():
            raise FileNotFoundError(f"Missing bundle file: {name} -> {f}")

    # Load index
    index = faiss.read_index(str(files["index"]))

    # Load ID map (ordered)
    with open(files["id_map"], "rb") as fh:
        ids = pickle.load(fh)
    if not isinstance(ids, list):
        raise RuntimeError("id_map.pkl must contain a list of ids")

    # Load chunks in order
    chunks: List[Dict[str, Any]] = []
    with open(files["chunks"], "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    # Load embedding dimension
    with open(files["model"], "r", encoding="utf-8") as fh:
        model_meta = json.load(fh)
    emb_dim = int(model_meta.get("dim", 0))

    # Load raw embeddings
    emb_arr = np.fromfile(str(files["embeddings"]), dtype=np.float32)

    if emb_dim <= 0:
        # fallback infer
        if len(ids) == 0:
            raise RuntimeError("Cannot infer embedding dimensions from bundle.")
        if emb_arr.size % len(ids) != 0:
            raise RuntimeError("Invalid embeddings.bin size")
        emb_dim = emb_arr.size // len(ids)

    try:
        emb_arr = emb_arr.reshape(len(ids), emb_dim)
    except Exception as e:
        raise RuntimeError(f"Failed to reshape embeddings.bin: {e}")

    # Load manifest
    with open(files["manifest"], "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    return {
        "index": index,
        "ids": ids,
        "chunks": chunks,
        "embeddings": emb_arr,
        "model_dim": emb_dim,
        "manifest": manifest,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load bundle + print summary")
    parser.add_argument("bundle", help="Path to bundle directory")
    args = parser.parse_args()

    b = load_bundle(args.bundle)
    print({
        "chunks": len(b["chunks"]),
        "dim": b["model_dim"]
    })
