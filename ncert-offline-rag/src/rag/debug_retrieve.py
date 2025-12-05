"""
ncert-offline-rag/src/rag/debug_retrieve.py

Debug helper: print bundle/query dims, embedding norms, and top-k FAISS scores + chunk snippets.
Run as a script to inspect why retrieve returned refer_teacher.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# robust import for load_bundle
try:
    from .load_bundle import load_bundle  # type: ignore
except Exception:
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from load_bundle import load_bundle  # type: ignore


def load_query(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    if p.suffix == ".npy":
        arr = np.load(str(p)).astype(np.float32)
    else:
        with open(p, "r", encoding="utf-8") as fh:
            arr = np.asarray(json.load(fh), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--embed", required=True)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    try:
        b = load_bundle(args.bundle)
    except Exception as e:
        print("ERROR loading bundle:", str(e), file=sys.stderr)
        sys.exit(2)

    try:
        q = load_query(args.embed)
    except Exception as e:
        print("ERROR loading query embed:", str(e), file=sys.stderr)
        sys.exit(2)

    print("--- bundle summary ---")
    print("bundle path:", args.bundle)
    print("model_dim (bundle):", b.get("model_dim"))
    print("num_chunks (ids):", len(b.get("ids", [])))
    emb = b.get("embeddings")
    if emb is None:
        print("embeddings: NOT present in bundle (embeddings.bin not loaded)")
    else:
        print("embeddings.shape:", getattr(emb, "shape", None))
        norms = np.linalg.norm(emb, axis=1)
        print("emb norms: min %.6f  mean %.6f  max %.6f" % (float(norms.min()), float(norms.mean()), float(norms.max())))

    print("\n--- query summary ---")
    print("query.shape:", q.shape)
    qnorm = np.linalg.norm(q, axis=1)
    print("query.norm: min %.6f mean %.6f max %.6f" % (float(qnorm.min()), float(qnorm.mean()), float(qnorm.max())))

    if q.shape[1] != b.get("model_dim"):
        print("\nDIM MISMATCH: query dim != bundle dim -> re-generate query with same encoder or re-embed chunks.")
        print("query dim:", q.shape[1], "bundle dim:", b.get("model_dim"))
        # still continue to show results if possible

    # normalize query (retrieval expects normalized vectors)
    qn = _normalize(q.astype(np.float32))

    index = b.get("index")
    try:
        D, I = index.search(qn, args.k)
    except Exception as e:
        print("ERROR running index.search:", str(e), file=sys.stderr)
        sys.exit(2)

    scores = D[0].tolist()
    idxs = I[0].tolist()
    print("\n--- top-%d results ---" % args.k)
    for rank, (score, pos) in enumerate(zip(scores, idxs)):
        print("rank", rank, "pos", pos, "score", float(score))
        if pos >= 0:
            cid = b["ids"][pos]
            chunk = b["chunks"][pos]
            text = chunk.get("text", "")
            snippet = text.replace("\n", " ")[:300]
            print("  id:", cid)
            print("  snippet:", snippet)
    top_score = float(scores[0]) if scores else 0.0
    print("\nTop-1 score:", top_score)
    if top_score < 0.60:
        print("=> TOP-1 below threshold 0.60; retrieve will return refer_teacher")
        print("Possible causes: different embedding model used for chunks vs query, embeddings not normalized, or unrelated query.")
    else:
        print("=> TOP-1 meets threshold (>=0.60)")

    # print candidate check: are retrieved ids present in chunks?
    print("\nRetrieved ids sample:", [b["ids"][i] for i in idxs if i >= 0][:10])


if __name__ == "__main__":
    main()