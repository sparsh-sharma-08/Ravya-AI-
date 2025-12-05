from __future__ import annotations
"""
src/rag/retrieve.py
Load bundle, load precomputed query embedding, normalize, search top-k using IndexFlatIP.
Prints strict JSON:
- {"status":"refer_teacher"} on failure/threshold
- {"status":"ok","chunks":[{id,rank,score,text,meta},...]}
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# robust import: allow running file directly or as package
try:
    from .load_bundle import load_bundle  # type: ignore
except Exception:
    # when executed as a script, ensure this file's directory is on sys.path
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    try:
        from load_bundle import load_bundle  # type: ignore
    except Exception as e:  # pragma: no cover
        raise

THRESHOLD = 0.60


def _load_query_embedding(path: str) -> np.ndarray:
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


def retrieve(bundle: str, embed_path: str, k: int = 5) -> Dict[str, Any]:
    b = load_bundle(bundle)
    q = _load_query_embedding(embed_path)
    if q.shape[1] != b["model_dim"]:
        raise RuntimeError(f"Embedding dim {q.shape[1]} != bundle dim {b['model_dim']}")
    q = _normalize(q.astype(np.float32))

    index = b["index"]
    D, I = index.search(q, k)
    scores = D[0].tolist()
    idxs = I[0].tolist()

    if len(scores) == 0:
        return {"status": "refer_teacher"}

    if idxs[0] < 0:
        return {"status": "refer_teacher"}

    top_score = float(scores[0])
    if top_score < THRESHOLD:
        return {"status": "refer_teacher"}

    out: List[Dict[str, Any]] = []
    for rank, (score, pos) in enumerate(zip(scores, idxs)):
        if pos < 0:
            continue
        cid = b["ids"][pos]
        chunk = b["chunks"][pos]
        out.append({
            "id": cid,
            "rank": rank,
            "score": float(score),
            "text": chunk.get("text", ""),
            "meta": chunk.get("metadata", {}),
        })
    return {"status": "ok", "chunks": out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--embed", required=True)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()
    try:
        res = retrieve(args.bundle, args.embed, args.k)
        print(json.dumps(res, ensure_ascii=False))
    except Exception as e:
        # conservative fallback
        print(json.dumps({"status": "refer_teacher", "error": str(e)}, ensure_ascii=False))
        sys.exit(0)


if __name__ == "__main__":
    main()