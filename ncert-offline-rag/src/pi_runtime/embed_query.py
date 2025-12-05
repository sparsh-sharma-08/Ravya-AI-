"""
ncert-offline-rag/src/pi_runtime/embed_query.py

Laptop-only helper: produce a query embedding using intfloat/e5-small-v2.
Writes JSON (.json) or numpy (.npy).

Usage (laptop/cloud only):
  python ncert-offline-rag/src/pi_runtime/embed_query.py --text "What is photosynthesis?" --output q.json

Notes:
- This MUST NOT be installed/run on Raspberry Pi.
- Requires sentence-transformers installed in the venv where you run this.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # handled below


def compute_embedding(texts: List[str], model_name: str) -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers not available in this environment. "
            "Install on laptop/cloud only: pip install sentence-transformers"
        )
    model = SentenceTransformer(model_name)
    # prefer convert_to_numpy and normalize if available
    try:
        emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    except TypeError:
        # older ST versions may not support normalize_embeddings arg
        emb = model.encode(texts, convert_to_numpy=True)
        # manual L2 normalization
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    return emb


def write_output(vec: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".npy":
        np.save(str(out_path), vec)
    elif out_path.suffix == ".json":
        # write the first vector as JSON list
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(vec[0].tolist(), fh, ensure_ascii=False)
    else:
        raise RuntimeError("Unsupported output extension. Use .json or .npy")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compute query embedding (laptop only)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Query text")
    group.add_argument("--file", help="Path to text file")
    p.add_argument("--output", required=True, help="Output file: .json or .npy")
    p.add_argument("--model", default="intfloat/e5-small-v2", help="Embedding model name")
    args = p.parse_args(argv)

    if args.file:
        pth = Path(args.file)
        if not pth.exists():
            print(f"ERROR: file not found: {args.file}", file=sys.stderr)
            return 2
        text = pth.read_text(encoding="utf-8").strip()
    else:
        text = (args.text or "").strip()

    if not text:
        print("ERROR: empty text", file=sys.stderr)
        return 2

    try:
        vec = compute_embedding([text], args.model)  # (1, D)
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        return 2

    out_path = Path(args.output)
    try:
        write_output(vec, out_path)
    except Exception as e:
        print("ERROR writing output:", str(e), file=sys.stderr)
        return 2

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())