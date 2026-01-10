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
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import faiss
except Exception:
    faiss = None

from src.rag.embed_query import embed_query


def _load_id_map(bundle: Path) -> List[Dict[str, Any]]:
    p = bundle / "id_map.pkl"
    if p.exists():
        with p.open("rb") as f:
            id_map = pickle.load(f)
        # normalise to list of dicts
        if isinstance(id_map, dict):
            try:
                ordered = [id_map[k] for k in sorted(id_map.keys(), key=lambda x: int(x))]
                return ordered
            except Exception:
                return list(id_map.values())
        return id_map
    # fallback to chunks.jsonl
    chunks = []
    cj = bundle / "chunks.jsonl"
    if cj.exists():
        for line in cj.read_text(encoding="utf-8").splitlines():
            try:
                chunks.append(json.loads(line))
            except Exception:
                continue
    return chunks


def _load_embeddings(bundle: Path) -> Optional[np.ndarray]:
    emb_file = bundle / "embeddings.bin"
    model_file = bundle / "model.json"
    if not emb_file.exists() or not model_file.exists():
        return None
    dim = int(json.loads(model_file.read_text(encoding="utf-8"))["dim"])
    b = emb_file.read_bytes()
    arr = np.frombuffer(b, dtype=np.float32).reshape(-1, dim)
    return arr


def retrieve(bundle_path: str, query: str, k: int = 5, embed_model: str = "all-mpnet-base-v2") -> List[Dict[str, Any]]:
    """
    Return list of results sorted by descending score:
      [{id, score, text, meta: {subject,chapter,textbook}}...]
    """
    bundle = Path(bundle_path)
    id_map = _load_id_map(bundle)
    emb_matrix = _load_embeddings(bundle)

    q_emb = embed_query([query], model_name=embed_model)[0].astype("float32")

    if emb_matrix is None or len(id_map) == 0:
        return []

    results: List[Dict[str, Any]] = []

    # prefer faiss index if present and usable
    idx_file = bundle / "index.faiss"
    if faiss is not None and idx_file.exists():
        try:
            index = faiss.read_index(str(idx_file))
            D, I = index.search(np.expand_dims(q_emb, axis=0), k)
            scores = D[0].tolist()
            inds = I[0].tolist()
            for sc, i in zip(scores, inds):
                if i < 0 or i >= len(id_map):
                    continue
                chunk = id_map[i]
                results.append({
                    "id": chunk.get("id"),
                    "score": float(sc),
                    "text": chunk.get("text"),
                    "meta": {k: chunk.get(k) for k in ("subject", "chapter", "textbook")}
                })
            return results
        except Exception:
            # fall back to numpy search
            pass

    # numpy fallback - dot product (assumes both query and stored embeddings are L2-normalized)
    emb = emb_matrix
    scores = emb @ q_emb
    topk_idx = np.argsort(scores)[-k:][::-1]
    for i in topk_idx:
        sc = float(scores[i])
        chunk = id_map[i]
        results.append({
            "id": chunk.get("id"),
            "score": sc,
            "text": chunk.get("text"),
            "meta": {k: chunk.get(k) for k in ("subject", "chapter", "textbook")}
        })
    return results


def infer_subjects_and_class(query: str) -> Dict[str, Any]:
    """
    Heuristic mapping: returns {'subjects': [...], 'class': Optional[int]}
    Extend mapping as needed.
    """
    q = query.lower()
    subjects: List[str] = []
    cls = None

    # subject heuristics
    if any(w in q for w in ("cell", "endoplasmic", "mitochond", "organell", "photosynth", "enzyme", "dna", "homogeneous", "heterogeneous", "mixture", "chemical", "reaction")):
        subjects += ["science", "biology", "chemistry"]
    if any(w in q for w in ("atom", "bohr", "electron", "proton", "nucleus", "orbit", "force", "motion")):
        subjects += ["physics", "science"]
    if any(w in q for w in ("who is", "actor", "movie", "capital", "history", "geography", "economics")):
        subjects += ["social", "history", "geography", "economics"]
    if any(w in q for w in ("poem", "grammar", "story", "comprehension", "chapter")):
        subjects += ["english"]

    # simple class inference from query like "class 9" or "9th"
    m = re.search(r'class\s*(\d{1,2})', q)
    if m:
        try:
            cls = int(m.group(1))
        except Exception:
            cls = None
    # fallback: if not found, leave cls None

    # dedupe while preserving order
    seen = set()
    subjects = [s for s in subjects if not (s in seen or seen.add(s))]
    return {"subjects": subjects, "class": cls}


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


def retrieve_chunks(bundle_path: str, query: str, k: int = 5, mode: str = "student", embed_model: str = "all-mpnet-base-v2", **kwargs):
    """
    Compatibility wrapper for older callers that expect retrieve_chunks().
    Forwards any extra keyword arguments to retrieve().
    Returns a dict with key "chunks" containing the list of retrieved chunk dicts.
    """
    # forward args/kwargs to the canonical retrieve() implementation
    results = retrieve(bundle_path, query, k=k, mode=mode, embed_model=embed_model, **kwargs)
    return {"chunks": results}


if __name__ == "__main__":
    main()