"""
Retrieve top-k chunks from FAISS bundle using a precomputed query embedding.

Output JSON:
If refer teacher:
  {"status": "refer_teacher"}

If ok:
  {
    "status": "ok",
    "chunks": [
      {"id": "...", "rank":0,"score":0.88,"text": "...","meta": {...}}
    ]
  }
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from load_bundle import load_bundle

THRESHOLD = 0.60


def _load_query(path: str) -> np.ndarray:
    """
    Accepts .json or .npy query embedding
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Query embedding not found: {path}")

    if p.suffix == ".npy":
        arr = np.load(str(p)).astype(np.float32)
    else:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def retrieve(bundle: str, embed_path: str, k: int = 5) -> Dict[str, Any]:
    b = load_bundle(bundle)

    q = _load_query(embed_path)
    if q.shape[1] != b["model_dim"]:
        raise RuntimeError(f"Query embedding dim {q.shape[1]} != bundle dim {b['model_dim']}")

    q = _normalize(q)

    index = b["index"]
    D, I = index.search(q, k)

    scores = D[0].tolist()
    idxs = I[0].tolist()

    if not scores or idxs[0] < 0:
        return {"status": "refer_teacher"}

    top = float(scores[0])
    if top < THRESHOLD:
        return {"status": "refer_teacher"}

    out = []
    for rank, (score, pos) in enumerate(zip(scores, idxs)):
        if pos < 0:
            continue
        chunk = b["chunks"][pos]
        out.append({
            "id": b["ids"][pos],
            "rank": rank,
            "score": float(score),
            "text": chunk.get("text", ""),
            "meta": chunk.get("metadata", {}),
        })

    return {"status": "ok", "chunks": out}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="FAISS retrieve")
    p.add_argument("--bundle", required=True)
    p.add_argument("--embed", required=True)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    res = retrieve(args.bundle, args.embed, args.k)
    print(json.dumps(res, ensure_ascii=False, indent=2))
