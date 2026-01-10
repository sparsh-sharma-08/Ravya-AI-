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
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import re
from typing import Dict, Any, List

# optional faiss (not required; we'll prefer numpy search for filtered sets)
try:
    import faiss
except Exception:
    faiss = None

# import subject inference helper from rag
try:
    from src.rag.retrieve import infer_subjects_and_class
except Exception:
    def infer_subjects_and_class(query: str) -> Dict[str, Any]:
        """
        Heuristic mapping: returns {'subjects': [...], 'class': Optional[int]}
        Expand subject aliases so domain tokens like "chemistry" map to "science" bundles.
        """
        q = query.lower()
        subjects: List[str] = []
        cls = None

        # raw heuristics
        if any(w in q for w in ("cell", "endoplasmic", "mitochond", "organell", "photosynth", "enzyme", "dna", "homogeneous", "heterogeneous", "mixture", "chemical", "reaction")):
            subjects += ["chemistry", "biology"]
        if any(w in q for w in ("atom", "bohr", "electron", "proton", "nucleus", "orbit", "force", "motion")):
            subjects += ["physics"]
        if any(w in q for w in ("who is", "actor", "movie", "capital", "history", "geography", "economics")):
            subjects += ["social"]
        if any(w in q for w in ("poem", "grammar", "story", "comprehension", "chapter")):
            subjects += ["english"]

        # expand aliases -> map subject-specific tags to bundle-level tags
        alias_map = {
            "chemistry": ["chemistry", "science"],
            "physics": ["physics", "science"],
            "biology": ["biology", "science"],
            "science": ["science"],
            "english": ["english"],
            "social": ["social"]
        }

        expanded: List[str] = []
        for s in subjects:
            for mapped in alias_map.get(s, [s]):
                if mapped not in expanded:
                    expanded.append(mapped)

        # simple class inference from query like "class 9" or "9th"
        m = re.search(r'class\s*(\d{1,2})', q)
        if m:
            try:
                cls = int(m.group(1))
            except Exception:
                cls = None

        # final dedupe (preserve order)
        return {"subjects": expanded, "class": cls}

def _load_id_map(bundle: Path) -> List[Dict[str, Any]]:
    p = bundle / "id_map.pkl"
    if p.exists():
        with p.open("rb") as f:
            id_map = pickle.load(f)
        # normalize to list of dicts
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

def _debug(msg: str):
    if os.environ.get("RAG_DEBUG") == "1":
        print("DEBUG:", msg)

def _token_overlap_scores(query: str, id_map: List[Dict[str, Any]], chunks_map: Dict[str, Dict[str, Any]]) -> np.ndarray:
    """
    Compute a lightweight token-overlap score between query and each chunk text.
    Returns a 1D numpy float32 array of length len(id_map) with higher = more relevant.
    Works offline without heavy models.
    """
    q_tokens = re.findall(r"\w+", (query or "").lower())
    if not q_tokens:
        return np.zeros((len(id_map),), dtype=np.float32)
    q_set = set(q_tokens)
    scores = np.zeros((len(id_map),), dtype=np.float32)
    for i, c in enumerate(id_map):
        cid = c.get("id")
        txt = c.get("text") or (chunks_map.get(cid) or {}).get("text") or ""
        if not txt:
            continue
        tks = set(re.findall(r"\w+", txt.lower()))
        if not tks:
            continue
        inter = len(q_set & tks)
        if inter == 0:
            continue
        # normalized overlap score (Jaccard-like)
        union = len(q_set | tks)
        scores[i] = float(inter) / float(union)
    # tiny smoothing
    if np.all(scores == 0.0):
        return scores
    # normalize to unit range
    maxv = float(scores.max())
    if maxv > 0:
        scores = scores / maxv
    return scores

def retrieve(bundle_path: str, query: str, k: int = 5, mode: str = "student", embed_model: str = "all-mpnet-base-v2") -> List[Dict[str, Any]]:
    """
    Metadata-aware retrieval:
      - Infers subject/class from query
      - Filters id_map + embeddings by subject/class before similarity search
      - If mode == 'teacher' enforce strict subject filtering (return [] if none)
      - Returns list of result dicts: {id, score, text, meta}
    """
    bundle = Path(bundle_path)
    id_map = _load_id_map(bundle)
    emb_matrix = _load_embeddings(bundle)

    total_chunks = len(id_map)
    _debug(f"id_map loaded length = {total_chunks}")

    # build quick lookup from chunks.jsonl (fallback) so we can always return text
    chunks_jsonl = bundle / "chunks.jsonl"
    chunks_map: Dict[str, Dict[str, Any]] = {}
    if chunks_jsonl.exists():
        try:
            for line in chunks_jsonl.read_text(encoding="utf-8").splitlines():
                try:
                    j = json.loads(line)
                    if isinstance(j, dict) and j.get("id"):
                        chunks_map[j["id"]] = j
                except Exception:
                    continue
        except Exception:
            # ignore read errors; chunks_map stays empty
            pass

    # infer desired subjects/classes from query
    info = infer_subjects_and_class(query)
    wanted_subjects = [s.lower() for s in (info.get("subjects") or []) if s]
    wanted_class = info.get("class")

    _debug(f"inferred subjects={wanted_subjects} class={wanted_class} mode={mode}")

    # apply metadata filter
    if wanted_subjects:
        idxs = []
        for i, c in enumerate(id_map):
            subj = (c.get("subject") or "").lower()
            cls = c.get("class")
            if subj in wanted_subjects or (wanted_class is not None and cls == wanted_class):
                idxs.append(i)
        if mode == "teacher" and not idxs:
            _debug("teacher mode: no chunks matched inferred subjects -> returning []")
            return []
        if idxs:
            filtered_map = [id_map[i] for i in idxs]
            filtered_emb = emb_matrix[np.array(idxs, dtype=int)] if emb_matrix is not None else None
            _debug(f"filtered chunks: {len(filtered_map)} of {total_chunks}")
            id_map = filtered_map
            emb_matrix = filtered_emb

    # if no embeddings file present, we will use text-overlap fallback
    if emb_matrix is None:
        _debug("no embeddings available for bundle -> using text-overlap fallback")
        scores = _token_overlap_scores(query, id_map, chunks_map)
        # pick top-k by score
        topk_idx = np.argsort(scores)[-k:][::-1]
        results = []
        for i in topk_idx:
            sc = float(scores[int(i)])
            chunk = id_map[int(i)]
            cid = chunk.get("id")
            text_val = chunk.get("text") or (chunks_map.get(cid) or {}).get("text")
            results.append({
                "id": cid,
                "score": sc,
                "text": text_val,
                "meta": {m: chunk.get(m) for m in ("subject", "chapter", "textbook", "class")}
            })
        _debug(f"final matched chunks = {len(results)} (text-overlap)")
        return results

    # Normalize embedding matrix rows to unit vectors and sanitize NaN/Inf to avoid overflow/NaN in matmul
    try:
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_matrix = emb_matrix / norms
        emb_matrix = np.nan_to_num(emb_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        _debug("failed to normalize emb_matrix; proceeding without normalization")

    # compute query embedding using runtime embed (try multiple fallbacks)
    _embed_query = None
    # 1) Try explicit name from src.pi_runtime.embed_query
    try:
        mod = __import__("src.pi_runtime.embed_query", fromlist=["embed_query"])
        _embed_query = getattr(mod, "embed_query", None)
    except Exception:
        _embed_query = None

    # 2) Try rag shim
    if _embed_query is None:
        try:
            mod = __import__("src.rag.embed_query", fromlist=["embed_query"])
            _embed_query = getattr(mod, "embed_query", None)
        except Exception:
            _embed_query = None

    # 3) Try alternate common names inside src.pi_runtime.embed_query (embed_texts, embed)
    if _embed_query is None:
        try:
            mod = __import__("src.pi_runtime.embed_query", fromlist=["*"])
            for cand in ("embed_texts", "embed", "encode", "compute_embeddings"):
                if hasattr(mod, cand):
                    _embed_query = getattr(mod, cand)
                    break
        except Exception:
            _embed_query = None

    # 4) Final fallback: attempt to use sentence-transformers if explicitly enabled.
    # NOTE: importing/initializing sentence-transformers can invoke native libs and
    # may crash or attempt network downloads when offline. Only enable when the
    # environment variable RAG_ENABLE_SENTENCE_TRANSFORMERS is set to "1".
    if _embed_query is None:
        if os.environ.get("RAG_ENABLE_SENTENCE_TRANSFORMERS", "0") == "1":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                def _embed_query(texts, model_name="all-mpnet-base-v2"):
                    # Try to load model offline-first (local_files_only=True) to avoid network when offline.
                    model_local = None
                    try:
                        model_local = SentenceTransformer(model_name, device="cpu", local_files_only=True)
                    except TypeError:
                        # Some SentenceTransformer wrappers may not accept local_files_only
                        model_local = SentenceTransformer(model_name, device="cpu")
                    try:
                        embs = model_local.encode(texts, show_progress_bar=False, convert_to_numpy=True, batch_size=32)
                    except Exception:
                        out = []
                        for t in texts:
                            out.append(model_local.encode(t, convert_to_numpy=True))
                        embs = np.vstack(out)
                    embs = np.asarray(embs, dtype=np.float32)
                    norms = np.linalg.norm(embs, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    embs = embs / norms
                    embs = np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)
                    return embs.astype("float32")
            except Exception as e:
                _debug(f"sentence-transformers fallback disabled/failed: {e}")
                _embed_query = None
        else:
            _debug("sentence-transformers disabled (RAG_ENABLE_SENTENCE_TRANSFORMERS!=1); using offline fallback")
            _embed_query = None

    q_emb = None
    used_text_fallback = False
    if _embed_query is not None:
        try:
            q_emb = _embed_query([query], model_name=embed_model)[0].astype("float32")
            # sanitize q_emb: if contains NaN/Inf or all zeros, treat as failed and fallback
            if np.isnan(q_emb).any() or np.isinf(q_emb).any() or np.allclose(q_emb, 0.0):
                _debug("query embedding contains NaN/Inf or is all-zero -> falling back to text-overlap")
                q_emb = None
            else:
                # robust normalization
                q_norm = float(np.linalg.norm(q_emb))
                if q_norm == 0 or np.isnan(q_norm) or np.isinf(q_norm):
                    _debug("query embedding norm invalid -> falling back to text-overlap")
                    q_emb = None
                else:
                    q_emb = (q_emb / q_norm).astype("float32")
        except Exception as e:
            _debug(f"embed_query failed: {e}; falling back to text-overlap")
            q_emb = None

    # If embeddings couldn't be computed or were invalid, fallback to lightweight text-overlap scoring
    if q_emb is None:
        _debug("using offline text-overlap fallback for query embedding")
        scores = _token_overlap_scores(query, id_map, chunks_map)
        topk_idx = np.argsort(scores)[-k:][::-1]
        results = []
        for i in topk_idx:
            sc = float(scores[int(i)])
            chunk = id_map[int(i)]
            cid = chunk.get("id")
            text_val = chunk.get("text") or (chunks_map.get(cid) or {}).get("text")
            results.append({
                "id": cid,
                "score": sc,
                "text": text_val,
                "meta": {m: chunk.get(m) for m in ("subject", "chapter", "textbook", "class")}
            })
        _debug(f"final matched chunks = {len(results)} (text-overlap fallback)")
        return results

    # use numpy dot-product over current emb_matrix
    scores = emb_matrix @ q_emb
    # sanitize scores (replace nan/inf with very small values)
    scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)
    topk_idx = np.argsort(scores)[-k:][::-1]
    results: List[Dict[str, Any]] = []
    for i in topk_idx:
        sc = float(scores[int(i)])
        chunk = id_map[int(i)]
        cid = chunk.get("id")
        # ensure text is provided: prefer chunk entry, fallback to chunks.jsonl mapping
        text_val = chunk.get("text")
        if not text_val and cid and cid in chunks_map:
            text_val = chunks_map[cid].get("text")
        results.append({
            "id": cid,
            "score": sc,
            "text": text_val,
            "meta": {k: chunk.get(k) for k in ("subject", "chapter", "textbook", "class")}
        })
    _debug(f"final matched chunks = {len(results)}")
    return results

def retrieve_chunks(bundle_path: str, query: str, k: int = 5, mode: str = "student", embed_model: str = "all-mpnet-base-v2"):
    """
    Compatibility wrapper for older callers that expect retrieve_chunks().
    Returns a dict with key "chunks" containing the list of retrieved chunk dicts.
    """
    results = retrieve(bundle_path, query, k=k, mode=mode, embed_model=embed_model)
    return {"chunks": results}

# --- small test CLI helper (keeps module import safe) ----------------------
if __name__ == "__main__":  # allow quick local test by running this module
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--index_file", default="unused")
    p.add_argument("--id_map_file", required=True)
    p.add_argument("--class_", type=int, default=8)
    p.add_argument("--subject", default="science")
    p.add_argument("--language", default="en")
    p.add_argument("--chapter", default=1)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    q = {"class": args.class_, "subject": args.subject, "language": args.language, "chapter": args.chapter}
    res = retrieve_chunks(q, index_file=args.index_file, id_map_file=args.id_map_file, k=args.k)
    print(res)
