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
from typing import Any, Dict, List, Optional
from src.utils.hashing import generate_md5

# Top-level helper to check debug mode
def _rag_debug() -> bool:
    return bool(os.getenv("RAG_DEBUG"))

# --- id_map loader ---------------------------------------------------------
def _load_id_map(path: str) -> List[Dict[str, Any]]:
    """
    Try to load id_map from JSON / JSONL / pickle files.
    Return list of chunk dicts or empty list on failure.
    """
    p = Path(path)
    if not p.exists():
        return []

    # Try read as text first
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        text = ""

    # Try JSON array or object
    if text:
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and "chunks" in obj and isinstance(obj["chunks"], list):
                return obj["chunks"]
        except Exception:
            # not plain JSON array/object
            pass

        # Try JSONL
        try:
            out = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    # skip malformed line
                    continue
            if out:
                return out
        except Exception:
            pass

    # Try pickle
    try:
        with p.open("rb") as fh:
            obj = pickle.load(fh)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                if "chunks" in obj and isinstance(obj["chunks"], list):
                    return obj["chunks"]
                # if dict mapping id->meta, return values
                vals = [v for v in obj.values() if isinstance(v, dict)]
                if vals:
                    return vals
    except Exception:
        pass

    return []

# --- chunk normalization helpers -------------------------------------------
def _ensure_chunk_fields(obj: Dict[str, Any]) -> None:
    """
    Ensure chunk has 'text','id','hash' and try to keep class/subject/chapter/language.
    Mutates obj in place.
    ID format: <class>_<subject>_<chapter>_<short_hash8>
    """
    # Ensure text
    text_seed = ""
    if isinstance(obj.get("text"), str) and obj.get("text").strip():
        text_seed = obj["text"].strip()
    else:
        # try other fields
        for key in ("content", "body", "title"):
            v = obj.get(key)
            if isinstance(v, str) and v.strip():
                text_seed = v.strip()
                break
    obj["text"] = text_seed

    # compute hash
    full_hash = generate_md5(text_seed or "")
    obj.setdefault("hash", full_hash)

    # normalize simple metadata
    cls = str(obj.get("class") or obj.get("grade") or "").strip()
    subj = str(obj.get("subject") or "").strip().lower()
    chap = str(obj.get("chapter") or obj.get("chap") or "").strip()
    lang = str(obj.get("language") or obj.get("lang") or "").strip().lower()

    obj["class"] = cls
    obj["subject"] = subj
    obj["chapter"] = chap
    obj["language"] = lang

    short_hash = full_hash[:8]
    cls_token = cls if cls else "na"
    subj_token = subj if subj else "na"
    chap_token = chap if chap else "na"
    default_id = f"{cls_token}_{subj_token}_{chap_token}_{short_hash}"
    obj.setdefault("id", default_id)

def _load_chunks_from_bundle_dir(bundle_dir: str) -> List[Dict[str, Any]]:
    """
    Load chunk dicts from bundle_dir/chunks/*.{json,jsonl,txt} or from
    bundle_dir/chunks.jsonl / chunks.json / chunks.txt at bundle root.
    Ensure each chunk has required fields via _ensure_chunk_fields.
    """
    out: List[Dict[str, Any]] = []
    p = Path(bundle_dir)

    # 1) directory fallback (existing behavior)
    chunks_dir = p / "chunks"
    if chunks_dir.exists() and chunks_dir.is_dir():
        for f in sorted(chunks_dir.iterdir()):
            if not f.is_file():
                continue
            suf = f.suffix.lower()
            if suf not in (".jsonl", ".json", ".txt"):
                continue
            try:
                txt = f.read_text(encoding="utf-8")
            except Exception:
                continue

            if suf == ".jsonl":
                for line in txt.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            _ensure_chunk_fields(obj)
                            out.append(obj)
                    except Exception:
                        continue
            elif suf == ".json":
                try:
                    obj = json.loads(txt)
                    if isinstance(obj, list):
                        for item in obj:
                            if isinstance(item, dict):
                                _ensure_chunk_fields(item)
                                out.append(item)
                    elif isinstance(obj, dict):
                        _ensure_chunk_fields(obj)
                        out.append(obj)
                except Exception:
                    chunk = {"text": txt.strip()}
                    _ensure_chunk_fields(chunk)
                    out.append(chunk)
            else:  # .txt
                chunk = {"text": txt.strip()}
                _ensure_chunk_fields(chunk)
                out.append(chunk)

    # 2) bundle-root files fallback: chunks.jsonl, chunks.json, chunks.txt
    if not out:
        for candidate in ("chunks.jsonl", "chunks.json", "chunks.txt"):
            fp = p / candidate
            if not fp.exists() or not fp.is_file():
                continue
            try:
                txt = fp.read_text(encoding="utf-8")
            except Exception:
                continue
            if candidate.endswith(".jsonl"):
                for line in txt.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            _ensure_chunk_fields(obj)
                            out.append(obj)
                    except Exception:
                        continue
            elif candidate.endswith(".json"):
                try:
                    obj = json.loads(txt)
                    if isinstance(obj, list):
                        for item in obj:
                            if isinstance(item, dict):
                                _ensure_chunk_fields(item)
                                out.append(item)
                    elif isinstance(obj, dict):
                        _ensure_chunk_fields(obj)
                        out.append(obj)
                except Exception:
                    chunk = {"text": txt.strip()}
                    _ensure_chunk_fields(chunk)
                    out.append(chunk)
            else:  # .txt
                chunk = {"text": txt.strip()}
                _ensure_chunk_fields(chunk)
                out.append(chunk)

            # if we found any file and parsed something, stop checking other candidates
            if out:
                break

    return out

# --- main retrieval --------------------------------------------------------
def retrieve_chunks(query: Dict[str, Any], index_file: str, id_map_file: str, k: int = 5) -> Any:
    """
    Load id_map or fallback to bundle/chunks, filter by query fields,
    return {"chunks":[...]} or "REFER_TEACHER".
    """
    RAG_DEBUG = _rag_debug()

    # 1) Try to load id_map (JSON/JSONL/pickle)
    id_map_list: List[Dict[str, Any]] = []
    try:
        if id_map_file:
            id_map_list = _load_id_map(id_map_file) or []
    except Exception:
        id_map_list = []

    if RAG_DEBUG:
        try:
            print(f"DEBUG: id_map loaded length = {len(id_map_list)}", file=os.sys.stderr)
        except Exception:
            print("DEBUG: id_map loaded length = ?", file=os.sys.stderr)

    chunks: List[Dict[str, Any]] = []
    # if id_map_list contains list of floats or something unexpected, guard
    if isinstance(id_map_list, list) and id_map_list and isinstance(id_map_list[0], dict):
        chunks = id_map_list

    # 2) Fallback to bundle/chunks if id_map produced no usable chunks
    fallback_count = 0
    if not chunks:
        bundle_dir: Optional[Path] = None
        try:
            p = Path(id_map_file) if id_map_file else None
            if p and p.exists():
                bundle_dir = p.parent.parent if p.parent.name == "data" else p.parent
        except Exception:
            bundle_dir = None
        if not bundle_dir:
            try:
                ip = Path(index_file) if index_file else None
                if ip and ip.exists():
                    bundle_dir = ip.parent
            except Exception:
                bundle_dir = None

        if bundle_dir:
            fb = _load_chunks_from_bundle_dir(str(bundle_dir))
            fallback_count = len(fb)
            if fb:
                chunks = fb

    if RAG_DEBUG:
        print(f"DEBUG: fallback chunk count = {fallback_count}", file=os.sys.stderr)
        try:
            print(f"DEBUG: final candidate chunks = {len(chunks)}", file=os.sys.stderr)
        except Exception:
            pass

    # 3) Filter chunks by query fields
    matched: List[Dict[str, Any]] = []
    try:
        q_class = query.get("class")
        q_subject = str(query.get("subject", "") or "").lower()
        q_language = str(query.get("language", "") or "").lower()
        q_chapter = query.get("chapter")
        for c in chunks:
            try:
                # ensure dict
                if not isinstance(c, dict):
                    continue
                if q_class is not None and str(c.get("class") or "") != str(q_class):
                    continue
                if q_subject and str(c.get("subject") or "").lower() != q_subject:
                    continue
                if q_language and str(c.get("language") or "").lower() != q_language:
                    continue
                if q_chapter and str(c.get("chapter") or "").strip() != str(q_chapter):
                    continue
            except Exception:
                continue
            matched.append(c)
    except Exception:
        matched = []

    if RAG_DEBUG:
        print(f"DEBUG: final matched chunks = {len(matched)}", file=os.sys.stderr)

    if not matched:
        if RAG_DEBUG:
            print("DEBUG: returning REFER_TEACHER because no matched chunks", file=os.sys.stderr)
        return "REFER_TEACHER"

    # Return top-k chunks (preserve dict structure)
    return {"chunks": matched[:k]}

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
