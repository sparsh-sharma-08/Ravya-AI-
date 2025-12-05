"""
ncert-offline-rag/src/rag/check_embedding_match.py

Compute cosine similarity between a freshly-computed embedding for a chunk text
(using a local SentenceTransformer model) and the precomputed embedding stored in the bundle.

Usage (laptop with sentence-transformers installed):
.venv/bin/python ncert-offline-rag/src/rag/check_embedding_match.py \
  --bundle ./bundle/class_8_science_en \
  --index 5 \
  --model all-mpnet-base-v2

Or check by chunk id:
  --id "10_Science_..._hash"

Options:
--text can override the chunk text (will not read from bundle).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# robust import for load_bundle
try:
    from .load_bundle import load_bundle  # type: ignore
except Exception:
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from load_bundle import load_bundle  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def compute_embedding(text: str, model_name: str) -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available. Run this on your laptop venv.")
    model = SentenceTransformer(model_name)
    emb = model.encode([text], convert_to_numpy=True)
    # normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    return emb[0].astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="Path to bundle directory")
    p.add_argument("--index", type=int, help="Chunk index (0-based) to test")
    p.add_argument("--id", help="Chunk id to test (alternative to --index)")
    p.add_argument("--text", help="Override chunk text (do not read from bundle)")
    p.add_argument("--model", default="all-mpnet-base-v2", help="SentenceTransformer model name (should be 768-dim)")
    args = p.parse_args()

    try:
        b = load_bundle(args.bundle)
    except Exception as e:
        print("ERROR loading bundle:", e, file=sys.stderr)
        sys.exit(2)

    ids = b.get("ids", [])
    chunks = b.get("chunks", [])
    embeddings = b.get("embeddings")
    if embeddings is None:
        print("ERROR: bundle has no embeddings array loaded (embeddings.bin missing or unreadable)", file=sys.stderr)
        sys.exit(2)

    idx: Optional[int] = None
    if args.id:
        try:
            idx = ids.index(args.id)
        except ValueError:
            print(f"ERROR: id not found in bundle: {args.id}", file=sys.stderr)
            sys.exit(2)
    elif args.index is not None:
        idx = args.index
        if idx < 0 or idx >= len(ids):
            print(f"ERROR: index out of range: {idx}", file=sys.stderr)
            sys.exit(2)
    else:
        print("ERROR: supply --index or --id", file=sys.stderr)
        sys.exit(2)

    stored_emb = np.asarray(embeddings[idx], dtype=np.float32)
    # ensure normalized
    s_norm = np.linalg.norm(stored_emb)
    if s_norm == 0:
        print("WARNING: stored embedding norm is zero", file=sys.stderr)
    else:
        stored_emb = stored_emb / (s_norm if s_norm != 0 else 1.0)

    if args.text:
        text = args.text
    else:
        chunk_obj = chunks[idx]
        text = chunk_obj.get("text", "")
    print("Testing chunk index:", idx, "id:", ids[idx])
    print("Chunk text snippet:", text.replace("\n", " ")[:400])

    try:
        fresh = compute_embedding(text, args.model)
    except Exception as e:
        print("ERROR computing fresh embedding:", e, file=sys.stderr)
        sys.exit(2)

    # cosine similarity (dot since normalized)
    sim = float(np.dot(fresh, stored_emb))
    print(f"cosine_similarity(fresh_vs_stored) = {sim:.6f}")

    # extra: compute similarity between fresh and other top stored embeddings
    dots = (embeddings @ fresh).astype(np.float32)  # (N,)
    topk_idx = np.argsort(-dots)[:10]
    print("Top-10 bundle positions by similarity to fresh embedding:")
    for rank, pidx in enumerate(topk_idx):
        print(f" rank {rank} pos {int(pidx)} id {ids[int(pidx)]} score {float(dots[pidx]):.6f}")

    # interpretation hint
    if sim > 0.85:
        print("INTERPRETATION: fresh embedding and stored embedding likely from same/similar encoder.")
    elif sim > 0.6:
        print("INTERPRETATION: partial match, but consider verifying model/version/tokenization.")
    else:
        print("INTERPRETATION: embeddings are in different spaces (re-embed chunks or use query encoder used to create bundle).")

if __name__ == "__main__":
    main()