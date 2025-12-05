#!/usr/bin/env python3
"""
Load a FAISS bundle for offline retrieval on Raspberry Pi.

Outputs an in-memory dict:
{
  "index": faiss index,
  "ids": [...],
  "chunks": [{"metadata": {...}, "text": "..."}, ...],
  "embeddings": np.ndarray (N, D),
  "model_dim": int,
  "manifest": {...}
}
"""
from __future__ import annotations
import json
import os
import pickle
import sys
from typing import Any, Dict, List

import numpy as np

try:
    import faiss
except Exception:
    faiss = None

def load_bundle(bundle_path: str) -> Dict[str, Any]:
    if faiss is None:
        print("faiss not installed. Install faiss-cpu on the Pi.", file=sys.stderr)
        sys.exit(2)

    if not os.path.isdir(bundle_path):
        print(f"Bundle path not found: {bundle_path}", file=sys.stderr)
        sys.exit(2)

    # required files
    paths = {
        "index": os.path.join(bundle_path, "index.faiss"),
        "embeddings": os.path.join(bundle_path, "embeddings.bin"),
        "id_map": os.path.join(bundle_path, "id_map.pkl"),
        "chunks": os.path.join(bundle_path, "chunks.jsonl"),
        "model": os.path.join(bundle_path, "model.json"),
        "manifest": os.path.join(bundle_path, "manifest.json"),
    }
    for name, p in paths.items():
        if not os.path.exists(p):
            print(f"Missing bundle file: {p}", file=sys.stderr)
            sys.exit(2)

    # load index
    index = faiss.read_index(paths["index"])

    # load ids
    with open(paths["id_map"], "rb") as fh:
        ids = pickle.load(fh)
    if not isinstance(ids, list):
        print("id_map.pkl must contain a list of ids", file=sys.stderr)
        sys.exit(2)

    # load chunks
    chunks: List[Dict[str, Any]] = []
    with open(paths["chunks"], "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(obj)

    # load embeddings: infer dim from model.json if present, otherwise from file size
    with open(paths["model"], "r", encoding="utf-8") as fh:
        model_meta = json.load(fh)
    embedding_dim = int(model_meta.get("dim", 0))
    emb_arr = np.fromfile(paths["embeddings"], dtype=np.float32)
    if embedding_dim <= 0:
        # try to infer
        if len(ids) == 0:
            print("Cannot infer embedding dim (no ids)", file=sys.stderr)
            sys.exit(2)
        if emb_arr.size % len(ids) != 0:
            print("embeddings.bin size not divisible by id count and model.json missing dim", file=sys.stderr)
            sys.exit(2)
        embedding_dim = emb_arr.size // len(ids)
    try:
        emb_arr = emb_arr.reshape(len(ids), embedding_dim)
    except Exception as e:
        print("Failed to reshape embeddings.bin:", e, file=sys.stderr)
        sys.exit(2)

    with open(paths["manifest"], "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    return {
        "index": index,
        "ids": ids,
        "chunks": chunks,
        "embeddings": emb_arr,
        "model_dim": embedding_dim,
        "manifest": manifest,
    }

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Load FAISS bundle and print summary")
    p.add_argument("bundle", help="Path to bundle directory")
    args = p.parse_args()
    bundle = load_bundle(args.bundle)
    print(f"Loaded bundle. chunks={len(bundle['chunks'])} dim={bundle['model_dim']}")