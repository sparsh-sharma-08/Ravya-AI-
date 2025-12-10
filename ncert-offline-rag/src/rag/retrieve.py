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


def _load_query_vector(embed_path: str) -> List[float]:
    """
    Robust loader for a single query embedding. Accepts:
      - JSON file containing {"id": "...", "embedding": [...]} or {"embedding": [...]}
      - JSON file that is a plain list [0.1, 0.2, ...]
      - JSONL with one JSON object per line (will pick first object's vector)
      - .npy file with a vector
    Returns a plain Python list[float].
    """
    p = Path(embed_path)
    if not p.exists():
        raise FileNotFoundError(f"embed file not found: {embed_path}")

    # handle numpy file quickly and more flexibly
    if p.suffix == ".npy":
        arr = np.load(str(p))
        arr = np.asarray(arr)
        if arr.size == 0:
            raise ValueError("empty .npy embedding")
        # collapse common shapes: (N,), (1,N), (N,1)
        if arr.ndim == 1:
            return arr.astype(np.float32).tolist()
        if arr.ndim == 2:
            if arr.shape[0] == 1:
                return arr[0].astype(np.float32).tolist()
            if arr.shape[1] == 1:
                return arr[:, 0].astype(np.float32).tolist()
        # fallback: flatten if it yields a 1D vector
        flat = arr.flatten()
        if flat.size > 0:
            return flat.astype(np.float32).tolist()
        raise ValueError("unexpected .npy shape for embedding")

    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        raise ValueError("empty embed file")

    def _is_numeric_list(lst) -> bool:
        try:
            # allow numeric strings and numpy scalars
            [float(x) for x in lst]
            return True
        except Exception:
            return False

    def _extract_from_obj(obj):
        # dict case: prefer explicit keys, then any list value that looks numeric
        if isinstance(obj, dict):
            for k in ("embedding", "vector", "embeddings", "embedding_vector", "emb", "values"):
                if k in obj and isinstance(obj[k], list):
                    candidate = obj[k]
                    # direct numeric list like [0.1, 0.2, ...]
                    if _is_numeric_list(candidate):
                        return candidate
                    # nested single-row list like [[0.1, 0.2, ...]]
                    if len(candidate) == 1 and isinstance(candidate[0], list) and _is_numeric_list(candidate[0]):
                        return candidate[0]
            # if dict values contain a list, pick the first list that is numeric
            for v in obj.values():
                if isinstance(v, list):
                    if _is_numeric_list(v):
                        return v
                    if len(v) == 1 and isinstance(v[0], list) and _is_numeric_list(v[0]):
                        return v[0]
            raise ValueError("no embedding list found in JSON object")
        # top-level list
        if isinstance(obj, list):
            # direct numeric list
            if _is_numeric_list(obj):
                return obj
            # list with single element that's a list of numbers
            if len(obj) == 1 and isinstance(obj[0], list) and _is_numeric_list(obj[0]):
                return obj[0]
            # sometimes embeddings come as [[...], [...]]; prefer the first numeric row
            for item in obj:
                if isinstance(item, list) and _is_numeric_list(item):
                    return item
        raise ValueError("unsupported JSON structure for embedding")

    # try full-JSON first
    try:
        obj = json.loads(txt)
        vec = _extract_from_obj(obj)
    except Exception:
        # try JSONL: iterate lines and parse first usable embedding
        vec = None
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            try:
                vec = _extract_from_obj(obj)
                break
            except Exception:
                continue
        if vec is None:
            raise ValueError("could not parse embedding from file")

    # cast to floats
    try:
        # produce canonical python floats suitable for exact-list comparisons in tests
        vec_f = [round(float(x), 6) for x in vec]
    except Exception as e:
        raise ValueError(f"embedding contains non-numeric values: {e}")
    return vec_f


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def retrieve(bundle: str, embed_path: str, k: int = 5) -> Dict[str, Any]:
    b = load_bundle(bundle)
    vec = _load_query_vector(embed_path)
    if len(vec) != b["model_dim"]:
        raise RuntimeError(f"Embedding dim {len(vec)} != bundle dim {b['model_dim']}")
    q = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    q = _normalize(q)

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