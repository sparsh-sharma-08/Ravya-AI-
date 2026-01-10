"""
Robust RAG orchestrator: retrieval -> prompt build -> model -> validate -> return.
Uses:
 - src.pi_runtime.retrieve.retrieve_chunks  (retrieval)
 - src.rag.call_gema.call_gema            (model call)
 - src.rag.build_prompt / build_prompt_teacher (prompt builders)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Union
from pathlib import Path   #Path importing
import re
import numpy as np
import subprocess
# add requests import where top-level imports are; fallback to subprocess if requests missing
try:
    import requests
except Exception:
    requests = None

# try to import build_prompt_teacher; provide clear fallback if missing
try:
    from src.rag.build_prompt_teacher import build_prompt_teacher
except Exception:
    try:
        from src.rag.build_prompt import build_prompt as build_prompt_teacher
    except Exception:
        def build_prompt_teacher(*args, **kwargs):
            raise ImportError(
                "build_prompt_teacher not found. Define build_prompt_teacher in src/rag/build_prompt_teacher.py "
                "or provide src/rag/build_prompt.build_prompt as a fallback."
            )

# Robustly import the model-call function from call_gema.py.
# Some files export `call_gema`, others `call_gemma`; support both.
try:
    from src.rag.call_gema import call_gema as _call_model_fn
except Exception:
    try:
        from src.rag.call_gema import call_gemma as _call_model_fn
    except Exception:
        import importlib
        mod = importlib.import_module("src.rag.call_gema")
        if hasattr(mod, "call_gema"):
            _call_model_fn = getattr(mod, "call_gema")
        elif hasattr(mod, "call_gemma"):
            _call_model_fn = getattr(mod, "call_gemma")
        else:
            raise ImportError("no call_gema/call_gemma function found in src.rag.call_gema")
from src.pi_runtime.retrieve import retrieve, retrieve_chunks


# ---------------- Validators -------------------------------------------------
def _ensure_chunk_ids(chunks_or_ids: Union[List[Dict[str, Any]], List[str], Dict]) -> List[str]:
    """
    Normalize retrieved chunk list or id-map into list of chunk ids.
    """
    if not chunks_or_ids:
        return []
    if isinstance(chunks_or_ids, dict):
        try:
            return [str(v.get("id")) for k, v in sorted(chunks_or_ids.items(), key=lambda x: int(x[0])) if v.get("id")]
        except Exception:
            return [str(v.get("id")) for v in chunks_or_ids.values() if isinstance(v, dict) and v.get("id")]
    if isinstance(chunks_or_ids, list) and len(chunks_or_ids) > 0 and isinstance(chunks_or_ids[0], dict):
        return [str(c.get("id")) for c in chunks_or_ids if c.get("id")]
    return [str(x) for x in chunks_or_ids]

def retrieve_chunks(bundle_path: str, query: str, k: int = 5, mode: str = "student", embed_model: str = "all-mpnet-base-v2"):
    """
    Compatibility wrapper used by src.rag.rag_answer: returns {'chunks': [...]}
    """
    results = retrieve(bundle_path, query, k=k, mode=mode, embed_model=embed_model)
    return {"chunks": results}

def _normalize_source_token(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r'^(text/|chunks/|chunk/|source:)\s*', '', s, flags=re.IGNORECASE)
    if "/" in s:
        s = s.split("/")[-1]
    return s

def _expand_source_tokens(raw_sources: List[Any], retrieved_chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Map model-declared source tokens to full chunk ids from retrieved_chunks.
    Preserves order and deduplicates.
    """
    if not raw_sources:
        return []
    candidate_ids = _ensure_chunk_ids(retrieved_chunks)
    cand_set = set(candidate_ids)
    normalized: List[str] = []
    seen = set()

    for src in raw_sources:
        if src is None:
            continue
        s = _normalize_source_token(src)
        # exact match
        if s in cand_set and s not in seen:
            normalized.append(s); seen.add(s); continue
        # try exact on original raw (maybe already full)
        if str(src) in cand_set and str(src) not in seen:
            normalized.append(str(src)); seen.add(str(src)); continue
        # suffix/substring match
        matched = None
        for cid in candidate_ids:
            if cid.endswith(s) or s in cid:
                matched = cid
                break
        if matched and matched not in seen:
            normalized.append(matched); seen.add(matched)
    return normalized

def _chunks_all_empty(chunks: List[Dict[str, Any]]) -> bool:
    if not chunks:
        return True
    return all(not (c.get("text") or "").strip() for c in chunks)

def _validate_student_response(obj: Any, retrieved_chunks: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """
    Non-blocking validation for student responses.

    Accepts:
      - dict with textual "answer" in common keys
      - optional "sources" (various key variants) normalized via _normalize_sources
      - plain string -> accepted as answer
    Returns normalized dict {"answer": str, "sources": [str,...]} or None.

    retrieved_chunks is accepted for callers that provide it but is unused by validator.
    """
    # plain string -> accept as answer
    if isinstance(obj, str):
        return {"answer": obj.strip(), "sources": []}

    if not isinstance(obj, dict):
        return None

    # find answer key (common variants)
    answer = None
    for k in ("answer", "Answer", "ANSWER"):
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            answer = obj[k].strip()
            break
    # fallback: accept other text-like keys
    if answer is None:
        for k in ("text", "result", "response"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                answer = v.strip()
                break

    if answer is None:
        # no textual answer present in parsed JSON -> signal caller to fallback to raw text
        return None

    # normalize sources using helper
    sources = _normalize_sources(obj)

    return {"answer": answer, "sources": sources}


def _validate_teacher_response(parsed: Dict[str, Any], chunks_or_ids: Union[List[Dict[str, Any]], List[str], Dict]) -> bool:
    """
    Accept 'content' OR 'answer' as the teacher text.
    Require non-empty content and a non-empty list of sources, and at least one source matching retrieved chunk ids.
    """
    if not isinstance(parsed, dict):
        return False

    content = (parsed.get("content") or parsed.get("answer") or "")
    sources = parsed.get("sources")

    if not content or not isinstance(content, str):
        return False
    if not isinstance(sources, list) or len(sources) == 0:
        return False

    chunk_ids = _ensure_chunk_ids(chunks_or_ids)
    if not chunk_ids:
        # no retrieved ids to validate against -> accept as long as content+sources present
        return True

    return any(str(s) in chunk_ids for s in sources)


# ---------------- Helpers ---------------------------------------------------
def _extract_json_from_text(text: str) -> Optional[str]:
    """
    Find first top-level JSON object in text and return as string, else None.
    Conservative: matches balanced braces.
    """
    if not text:
        return None
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            if start is None:
                continue
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    # continue searching
                    start = None
                    depth = 0
    return None


def _validate_query_embedding(embed_path: str) -> Optional[List[float]]:
    """
    Validate and normalize a query embedding file.

    Accepted:
      - .npy : 1D float array
      - .json : either {"embedding":[...]} or a raw JSON list [0.1, 0.2, ...]
      - raw file that is a JSON list

    Reject:
      - .pkl and other non-supported types
      - dict objects (other than {"embedding":[...]})
      - empty vectors, non-float values

    On invalid input prints:
      "Invalid embed file. Expected an embedding vector, got <file-type>."
    and returns None.
    """
    if not embed_path:
        return None

    p = Path(embed_path)
    if not p.exists():
        print(f"Invalid embed file. Expected an embedding vector, got missing file", file=sys.stderr)
        return None

    suffix = p.suffix.lower()
    try:
        if suffix == ".npy":
            try:
                arr = np.load(str(p))
            except Exception:
                print(f"Invalid embed file. Expected an embedding vector, got .npy (load failed)", file=sys.stderr)
                return None
            if arr.ndim != 1:
                print(f"Invalid embed file. Expected a 1D embedding vector, got shape={arr.shape}", file=sys.stderr)
                return None
            vec = arr.astype(float).tolist()
        elif suffix in (".json", ".txt", ""):
            # try parse JSON
            text = p.read_text(encoding="utf-8")
            try:
                obj = json.loads(text)
            except Exception:
                print(f"Invalid embed file. Expected an embedding vector, got malformed JSON", file=sys.stderr)
                return None
            # if dict, expect {"embedding": [...]}
            if isinstance(obj, dict):
                if "embedding" in obj and isinstance(obj["embedding"], list):
                    vec = obj["embedding"]
                else:
                    print(f"Invalid embed file. Expected an embedding vector, got JSON object", file=sys.stderr)
                    return None
            elif isinstance(obj, list):
                vec = obj
            else:
                print(f"Invalid embed file. Expected an embedding vector, got JSON type {type(obj).__name__}", file=sys.stderr)
                return None
        else:
            print(f"Invalid embed file. Expected an embedding vector, got {suffix or 'unknown'}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Invalid embed file. Expected an embedding vector, got error: {e}", file=sys.stderr)
        return None

    # Validate elements are floats (or coercible)
    if not isinstance(vec, list) or len(vec) == 0:
        print(f"Invalid embed file. Expected an embedding vector, got empty vector", file=sys.stderr)
        return None
    clean = []
    for i, v in enumerate(vec):
        try:
            fv = float(v)
            clean.append(fv)
        except Exception:
            print(f"Invalid embed file. Expected an embedding vector, got non-float value at index {i}", file=sys.stderr)
            return None
    return clean


def _normalize_sources(parsed: Dict[str, Any]) -> List[str]:
    """
    Normalize various source key variants into a clean list of strings.
    Accepts keys: "sources", "source", "source ", "Sources", "Source"
    Always returns a list of stripped non-empty strings.
    """
    s = (
        parsed.get("sources")
        or parsed.get("source")
        or parsed.get("source ")
        or parsed.get("Sources")
        or parsed.get("Source")
        or []
    )
    if isinstance(s, (str, bytes)):
        s = [s]
    if not isinstance(s, (list, tuple)):
        s = []
    return [str(x).strip() for x in s if x is not None and str(x).strip()]


def _extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract a JSON object from model text.
    Prefer parsing entire text; fall back to first {...} blob.
    """
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    m = re.search(r'(\{[\s\S]*\})', text)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(1))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _expand_source_tokens(sources: List[str], retrieved_chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Expand/resolve source tokens returned by the model to full chunk ids.
    - If token already equals a known chunk id, keep it.
    - If token matches an 7-32 hex string, try to find a chunk whose 'hash' startswith that hex.
    - If token matches the trailing 8 chars of an id, try to match that too.
    - Fall back to returning original token if no match found (but keep order & dedupe).
    """
    if not sources:
        return sources
    id_by_hash_prefix: Dict[str, str] = {}
    id_set = set()
    for c in retrieved_chunks:
        cid = c.get("id")
        chash = str(c.get("hash") or "")
        if cid:
            id_set.add(cid)
            # map full hash and short prefixes (8 chars) to id
            if chash:
                id_by_hash_prefix[chash] = cid
                id_by_hash_prefix[chash[:8]] = cid
                # also lower-case prefix
                id_by_hash_prefix[chash.lower()] = cid
                id_by_hash_prefix[chash[:8].lower()] = cid

    out: List[str] = []
    for s in sources:
        if not s:
            continue
        s_str = str(s).strip()
        if s_str in id_set:
            out.append(s_str)
            continue
        # hex-like short id (7-32 hex chars)
        if re.fullmatch(r"[0-9a-fA-F]{6,32}", s_str):
            key = s_str.lower()
            if key in id_by_hash_prefix:
                out.append(id_by_hash_prefix[key])
                continue
        # maybe model returned just the 8-char suffix (e.g. c30d44d7)
        if len(s_str) == 8 and re.fullmatch(r"[0-9a-fA-F]{8}", s_str):
            key = s_str.lower()
            if key in id_by_hash_prefix:
                out.append(id_by_hash_prefix[key])
                continue
            # try matching id suffix
            matched = None
            for cid in id_set:
                if cid.endswith(f"_{s_str}") or cid.endswith(s_str):
                    matched = cid
                    break
            if matched:
                out.append(matched)
                continue
        # fallback: keep as-is
        out.append(s_str)

    # dedupe preserving order
    seen = set()
    res = []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


def _heuristic_extract_answer_sources(text: str) -> Optional[Dict[str, Any]]:
    """
    Heuristic extraction of answer and sources from raw model text when strict JSON parsing fails.
    Looks for common keys: "answer", "content", "response" and source lists under "sources"/"source".
    Returns {"answer": str, "sources": [str,...]} or None.
    """
    if not text or not isinstance(text, str):
        return None
    # Attempt to find a JSON-like "sources": [...] list
    src_match = re.search(r'"sources"\s*:\s*\[([^\]]+)\]', text, re.IGNORECASE)
    if not src_match:
        src_match = re.search(r'"source"\s*:\s*\[([^\]]+)\]', text, re.IGNORECASE)
    sources = []
    if src_match:
        raw = src_match.group(1)
        # split by commas and strip quotes/spaces
        parts = re.split(r'\s*,\s*', raw.strip())
        for p in parts:
            p = p.strip().strip('"').strip("'").strip()
            if p:
                # if model returns short hash, keep; otherwise try to extract trailing 8 chars
                sources.append(p)
    else:
        # also accept simple tokens like "sources":["abc"] without quotes spaced
        src_match2 = re.search(r'sources\s*:\s*\[([^]]+)\]', text, re.IGNORECASE)
        if src_match2:
            for p in re.split(r'\s*,\s*', src_match2.group(1)):
                p = p.strip().strip('"').strip("'").strip()
                if p:
                    sources.append(p)

    # Attempt to find answer-like fields
    # Prefer "answer" then "content"
    ans = None
    for key in ("\"answer\"", "\"content\"", "\"response\"", "\"text\""):
        m = re.search(rf'{key}\s*:\s*"(.*?)"(?:,|\}})', text, re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            break
    if not ans:
        # fallback: if the model returned a top-level JSON string before any 'sources' token,
        # take the first long quoted substring
        m2 = re.search(r'"([^"]{20,3000})"', text, re.DOTALL)
        if m2:
            ans = m2.group(1).strip()

    if not ans:
        return None
    # If no sources found, try to extract any id-like tokens of 6-40 chars (hash or composite id)
    if not sources:
        found = re.findall(r'([0-9A-Za-z_]+_[0-9A-Za-z_ ]+_[0-9A-Za-z_]+_[0-9a-f]{8})', text)
        if found:
            sources.extend(found)
    if not sources:
        # try short hex ids
        found2 = re.findall(r'\b([0-9a-f]{7,32})\b', text)
        for f in found2:
            sources.append(f)

    # Deduplicate and normalize
    sources = [s for i, s in enumerate(sources) if s and s not in sources[:i]]
    if not ans or not sources:
        return None
    result = {"answer": ans, "sources": sources}
    return result


def _normalize_teacher_sources(raw_sources, retrieved_chunks):
    """
    Normalize model-declared sources to full chunk ids from retrieved_chunks.
    - strip common prefixes
    - match exact or by suffix
    - deduplicate while preserving order
    """
    if not raw_sources:
        return []
    # build lookup
    valid_ids = [c.get("id") for c in retrieved_chunks if isinstance(c, dict) and c.get("id")]
    valid_set = set(valid_ids)

    def normalize_str(s: str) -> str:
        s = str(s).strip()
        # remove common prefixes and whitespace
        s = re.sub(r'^(text/|chunks/|chunk/|source:)\s*', '', s, flags=re.IGNORECASE)
        # use last path component if contains '/'
        if "/" in s:
            s = s.split("/")[-1]
        return s

    seen = {}
    normalized = []
    for raw in raw_sources:
        s = normalize_str(raw)
        # exact
        if s in valid_set and s not in seen:
            normalized.append(s); seen[s] = True; continue
        # suffix match or contained
        for cid in valid_ids:
            if cid.endswith(s) or s in cid:
                if cid not in seen:
                    normalized.append(cid); seen[cid] = True
                break
    return normalized


# ---------------- Orchestrator ------------------------------------------------
def get_rag_answer(
    bundle: str,
    embed: str,
    query: str,
    k: int = 5,
    model: str = "2b",
    mode: str = "student",
) -> Dict[str, Any]:
    """
    High-level function used by CLI/interactive code.
    Debug output enabled when environment variable RAG_DEBUG is set (non-empty).
    """
    RAG_DEBUG = bool(os.getenv("RAG_DEBUG"))
    # Validate embed file strictly
    vec = None
    if embed:
        vec = _validate_query_embedding(embed)
        if vec is None:
            # invalid embed -> fail fast
            return {"status": "refer_teacher"}

    def _refer(reason: str, *, ret=None, chunks=None, prompt=None, model_output=None):
        """
        Centralized fallback logger + refer return.
        Prints helpful debug info only when RAG_DEBUG is truthy.
        """
        if RAG_DEBUG:
            print("=== RAG DEBUG ===", file=sys.stderr)
            print("REASON:", reason, file=sys.stderr)
            if ret is not None:
                print("retrieve_chunks returned (repr):", repr(ret), file=sys.stderr)
            if chunks is not None:
                try:
                    ids = [c.get("id") for c in chunks if isinstance(c, dict) and c.get("id")]
                except Exception:
                    ids = None
                print("chunk ids:", ids, file=sys.stderr)
                print("chunk count:", len(chunks) if chunks is not None else None, file=sys.stderr)
            if prompt is not None:
                print("prompt len:", len(prompt) if isinstance(prompt, str) else type(prompt), file=sys.stderr)
                # print only first 2000 chars
                print("prompt (head):", (prompt[:2000] + "...") if isinstance(prompt, str) and len(prompt) > 2000 else prompt, file=sys.stderr)
            if model_output is not None:
                print("model output (repr/head):", (repr(model_output)[:2000] + "...") if model_output is not None else None, file=sys.stderr)
            print("=== END DEBUG ===", file=sys.stderr)
        return {"status": "refer_teacher"}

    # Determine id_map_file and index_file to pass to retrieve_chunks
    bundle_p = Path(bundle) if bundle else Path(".")
    # embed is the query embedding file (Do NOT use it as id_map)
    id_map_file = None
    index_file = "unused"

    # determine id_map from bundle only (do NOT use --embed for id_map)
    if bundle_p.exists():
        candidates = [
            bundle_p / "id_map.pkl",
            bundle_p / "id_map.json",
            bundle_p / "id_map.jsonl",
            bundle_p / "id_map",
            bundle_p / "data" / "id_map.pkl",
        ]
        for cand in candidates:
            if cand.exists():
                id_map_file = str(cand)
                break

    # If still None, fall back to "unused" (retrieve_chunks may still locate bundle)
    if id_map_file is None:
        id_map_file = "unused"

    # Build qdict: include embedding vector so retrieve_chunks can use it
    if isinstance(query, str):
        qdict = {"query": query}
    else:
        qdict = query
    if vec is not None:
        qdict["embedding"] = vec

    # Call retrieval (retrieve_chunks expects query dict)
    try:
        # now call retrieve_chunks (it may now read qdict["embedding"])
        # pass the discovered id_map_file (from bundle) — do NOT pass embed here
        ret = retrieve_chunks(qdict, index_file=index_file, id_map_file=id_map_file, k=k)
    except Exception as e:
        if RAG_DEBUG:
            print("DEBUG: retrieve exception:", repr(e), file=sys.stderr)
        return {"status": "refer_teacher"}

    if not ret or ret == "REFER_TEACHER" or ret == {"status": "refer_teacher"}:
        return _refer("empty/REFER_TEACHER from retrieve", ret=ret)

    chunks = ret.get("chunks") if isinstance(ret, dict) else None
    if not chunks or not isinstance(chunks, list):
        return _refer("no chunks returned", ret=ret, chunks=chunks)

    chunks = chunks[:k]

    # Build prompt
    try:
        if mode == "teacher":
            prompt = build_prompt_teacher(query, chunks)
        else:
            prompt = build_prompt(query, chunks)
    except Exception as e:
        return _refer(f"prompt build failed: {e}", ret=ret, chunks=chunks)

    # Call model
    try:
        model_output = _call_model_fn(prompt, model_variant=model)
        if not isinstance(model_output, str):
            model_output = str(model_output)
    except Exception as e:
        return _refer(f"model call failed: {e}", ret=ret, chunks=chunks, prompt=prompt)

    # Extract JSON if present
    json_blob = _extract_json_from_text(model_output)
    if json_blob is None:
        # student fallback to raw text
        if mode == "student":
            answer_text = model_output.strip()
            if not answer_text:
                return _refer("empty model output in student mode", ret=ret, chunks=chunks, prompt=prompt, model_output=model_output)
            return {"status": "ok", "mode": "student", "answer": answer_text, "sources": []}
        return _refer("no JSON blob found in model output (teacher mode)", ret=ret, chunks=chunks, prompt=prompt, model_output=model_output)

    try:
        parsed = json.loads(json_blob)
    except Exception as e:
        return _refer(f"json parse failed: {e}", ret=ret, chunks=chunks, prompt=prompt, model_output=model_output)

    # Validate and format output
    if mode == "teacher":
        # 1) If all retrieved chunks are empty, return empty teacher response (no hallucination)
        if _chunks_all_empty(chunks):
            return {
                "status": "ok",
                "mode": "teacher",
                "content": "",
                "sources": []
            }

        # 2) Normalize/expand model-declared sources BEFORE validation
        raw_sources = parsed.get("sources", [])
        parsed["sources"] = _expand_source_tokens(raw_sources, chunks)

        # 3) Validate using robust validator
        if not _validate_teacher_response(parsed, chunks):
            return _refer("teacher validation failed", ret=ret, chunks=chunks, prompt=prompt, model_output=model_output)
        return {
            "status": "ok",
            "mode": "teacher",
            "content": parsed.get("content") or parsed.get("answer") or "",
            "sources": parsed.get("sources", [])
        }
    else:
        if not _validate_student_response(parsed, chunks):
            # try to extract fallback free-text
            answer_text = parsed.get("answer") or parsed.get("text") or parsed.get("content")
            if answer_text:
                return {"status": "ok", "mode": "student", "answer": answer_text, "sources": parsed.get("sources", [])}
            return _refer("student validation failed", ret=ret, chunks=chunks, prompt=prompt, model_output=model_output)
        return {"status": "ok", "mode": "student", "answer": parsed["answer"], "sources": parsed["sources"]}


def _handle_model_output(mode: str, model_output: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Unified handling of model output.
    - teacher mode: strict JSON required (existing behavior preserved)
    - student mode: prefer JSON if present, otherwise accept raw text as answer
    """
    # gather retrieved ids for fallback
    retrieved_ids = []
    try:
        retrieved_ids = [c.get("id") for c in retrieved_chunks if isinstance(c, dict) and c.get("id")]
    except Exception:
        retrieved_ids = []

    if mode == "teacher":
        parsed = _extract_json_blob(model_output)
        if not parsed:
            # try heuristic extraction before failing
            if os.getenv("RAG_DEBUG"):
                print("=== RAG DEBUG ===\nREASON: no JSON blob found in model output (teacher mode)\n", file=sys.stderr)
                print("retrieve_chunks returned (repr):", repr({"chunks": retrieved_chunks}), file=sys.stderr)
                print("model output (repr/head):", repr(model_output[:2000]), file=sys.stderr)
            h = _heuristic_extract_answer_sources(model_output)
            if h:
                # expand any short source tokens into full ids using retrieved_chunks
                h["sources"] = _expand_source_tokens(h["sources"], retrieved_chunks)
                return {"status": "ok", "mode": "teacher", "answer": h["answer"], "sources": h["sources"]}
            return {"status": "refer_teacher"}
        try:
            valid = _validate_teacher_response(parsed)
        except Exception:
            valid = False
        if not valid:
            # try heuristic extraction from raw text if strict validation failed
            h = _heuristic_extract_answer_sources(model_output)
            if h:
                h["sources"] = _expand_source_tokens(h["sources"], retrieved_chunks)
                return {"status": "ok", "mode": "teacher", "answer": h["answer"], "sources": h["sources"]}
            if os.getenv("RAG_DEBUG"):
                print("RAW MODEL OUTPUT (parsed JSON failed validation):", file=sys.stderr)
                try:
                    print(json.dumps(parsed, ensure_ascii=False, indent=2), file=sys.stderr)
                except Exception:
                    print(repr(parsed), file=sys.stderr)
                print("Full raw output:", file=sys.stderr)
                print(model_output, file=sys.stderr)
            return {"status": "refer_teacher"}
        sources = _normalize_sources(parsed)
        # normalize sources from parsed JSON as well (resolve short hashes)
        sources = _expand_source_tokens(sources, retrieved_chunks)
        return {"status": "ok", "mode": "teacher", "answer": parsed, "sources": sources}

    # Student mode: extract JSON blob but allow fallback to raw text
    parsed = _extract_json_blob(model_output)
    if parsed:
        valid = _validate_student_response(parsed, retrieved_chunks)
        if valid:
            return {"status": "ok", "mode": "student", "answer": parsed["answer"], "sources": parsed["sources"]}
    # Fallback to raw text
    answer_text = model_output.strip()
    if not answer_text:
        return {"status": "refer_teacher"}
    return {"status": "ok", "mode": "student", "answer": answer_text, "sources": []}


def _call_llama_cpp(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
    """
    Call a local Llama (llama-cpp-python) model. Requires:
      - pip install llama-cpp-python
      - a local GGML model file (set LLAMA_CPP_MODEL env var to path)
      - set RAG_ENABLE_LOCAL_LLM=1 to enable
    """
    model_path = os.environ.get("LLAMA_CPP_MODEL", "")
    if not model_path or not Path(model_path).exists():
        raise RuntimeError("LLAMA_CPP_MODEL not set or file not found")
    try:
        from llama_cpp import Llama  # type: ignore
        ctx = int(os.environ.get("LLAMA_CTX", "2048"))
        n_threads = int(os.environ.get("LLAMA_THREADS", "2"))
        llm = Llama(model_path=model_path, n_ctx=ctx, n_threads=n_threads)
        resp = llm(prompt, max_tokens=max_tokens, temperature=float(temperature))
        # llama-cpp-python returns {'choices':[{'text': '...'}], ...}
        return resp.get("choices", [{}])[0].get("text", "") or ""
    except Exception as e:
        raise RuntimeError(f"llama-cpp failed: {e}") from e

def _call_ollama(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
    """
    Call local Ollama server (preferred) or ollama CLI (fallback).
    Handles streaming JSON lines (SSE-like) by concatenating 'response' fragments.
    """
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL")
    if not model:
        raise RuntimeError("OLLAMA_MODEL not set (set to the model name you pulled with `ollama pull`).")

    # Try HTTP API first if requests available
    if requests is not None:
        try:
            url = f"{host}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
            }
            # stream=True to handle line-delimited JSON streaming responses
            resp = requests.post(url, json=payload, timeout=60, stream=True)
            resp.raise_for_status()

            result_text = ""
            # iterate over streamed lines (each line is a JSON object in many Ollama versions)
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                # try parse JSON line
                try:
                    j = None
                    import json as _json
                    j = _json.loads(line)
                    # common streaming field that contains generated fragment
                    if isinstance(j, dict):
                        # if model returns fragmented content under 'response'
                        if "response" in j and isinstance(j["response"], str):
                            result_text += j["response"]
                        # some versions use 'output' or 'text' for final payload
                        elif "output" in j and isinstance(j["output"], str):
                            result_text = j["output"]
                        elif "text" in j and isinstance(j["text"], str):
                            result_text = j["text"]
                        # break if stream indicates done
                        if j.get("done") is True:
                            break
                except Exception:
                    # not JSON line -> treat as raw text fragment
                    try:
                        result_text += line + ("\n" if not line.endswith("\n") else "")
                    except Exception:
                        pass

            # final fallback: if nothing assembled, try resp.text
            if not result_text:
                result_text = resp.text or ""

            return result_text
        except Exception as e:
            if os.environ.get("RAG_DEBUG") == "1":
                print("DEBUG: Ollama HTTP call failed (streaming):", e)

    # Fallback: use ollama CLI if available
    try:
        cmd = ["ollama", "run", model]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(prompt, timeout=60)
        if proc.returncode == 0 and out:
            return out.strip()
        raise RuntimeError(f"ollama CLI failed rc={proc.returncode} stderr={err.strip()}")
    except Exception as e:
        if os.environ.get("RAG_DEBUG") == "1":
            print("DEBUG: Ollama CLI fallback failed:", e)
        raise RuntimeError(f"Ollama backend failed: {e}") from e

def _call_any_llm(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
    # 0) Ollama local server (highest priority when enabled)
    try:
        if os.environ.get("RAG_ENABLE_OLLAMA", "0") == "1":
            return _call_ollama(prompt, max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        if os.environ.get("RAG_DEBUG") == "1":
            print("DEBUG: ollama backend error:", e)

    # 1) repo-local pi_runtime llm
    try:
        from src.pi_runtime import llm as _pi_llm  # type: ignore
        for fn in ("generate", "call", "run", "complete"):
            if hasattr(_pi_llm, fn):
                return getattr(_pi_llm, fn)(prompt, max_tokens=max_tokens, temperature=temperature)
        if hasattr(_pi_llm, "call_llm"):
            return _pi_llm.call_llm(prompt, max_tokens=max_tokens, temperature=temperature)
    except Exception:
        pass

    # 2) repo-local rag llm
    try:
        from src.rag import llm as _rag_llm  # type: ignore
        for fn in ("generate", "call", "run", "complete"):
            if hasattr(_rag_llm, fn):
                return getattr(_rag_llm, fn)(prompt, max_tokens=max_tokens, temperature=temperature)
    except Exception:
        pass

    # 3) OpenAI (if available)
    try:
        if os.environ.get("OPENAI_API_KEY"):
            import openai
            resp = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            return resp.choices[0].text
    except Exception:
        pass

    return f"<<NO_LLM_AVAILABLE>>\n{prompt}"

def _synthesize_answer_from_chunks(question: str, chunks: list, mode: str = "student"):
    """
    Lightweight offline synthesizer:
    - Extracts candidate sentences from chunk texts
    - Scores sentences by overlap with question tokens
    - Returns {'answer': str, 'sources': [chunk_ids]}
    """
    import re
    q_tokens = [w for w in re.findall(r"\w+", question.lower()) if len(w) > 2]
    candidates = []
    for c in chunks:
        cid = c.get("id")
        text = (c.get("text") or "").strip()
        if not text:
            continue
        sents = re.split(r'(?<=[\.\?\!])\s+', text)
        for s in sents:
            s_clean = s.strip()
            if not s_clean:
                continue
            lw = s_clean.lower()
            score = sum(lw.count(t) for t in q_tokens)
            # small length bias to prefer longer informative sentences
            score += min(1.0, len(s_clean) / 200.0) * 0.1
            candidates.append((score, cid, s_clean))
    # choose best sentences
    candidates = [c for c in candidates if c[0] > 0]
    if not candidates:
        # fallback: first sentence from first up to 3 chunks
        out = []
        used = []
        for c in chunks[:3]:
            t = (c.get("text") or "").strip()
            if not t:
                continue
            first_sent = re.split(r'(?<=[\.\?\!])\s+', t)[0].strip()
            if first_sent:
                out.append(first_sent)
                used.append(c.get("id"))
        answer = " ".join(out) if out else "No relevant information found in the provided chunks."
        return {"answer": answer, "sources": used}
    candidates.sort(reverse=True, key=lambda x: x[0])
    top = candidates[:3]
    answer = " ".join([s for _, _, s in top])
    sources = list(dict.fromkeys([cid for _, cid, _ in top]))
    return {"answer": answer, "sources": sources}


# modify get_rag_answer_with_llm behavior to synthesize when no LLM is available
def get_rag_answer_with_llm(bundle: str, question: str, mode: str = "student", k: int = 5, **call_opts) -> dict:
    """
    Retrieve top-k chunks from the bundle and call an LLM to generate an answer.
    Falls back to offline synthesizer if no LLM is available.
    """
    # retrieve chunks using compatibility wrapper
    try:
        from src.pi_runtime.retrieve import retrieve_chunks
        res = retrieve_chunks(bundle, question, k=k, mode=mode)
        chunks = res.get("chunks", []) if isinstance(res, dict) else res
    except Exception:
        from src.pi_runtime.retrieve import retrieve
        chunks = retrieve(bundle, question, k=k, mode=mode)

    # build simple prompt (kept for LLM paths)
    ctx = "\n\n".join([f"{c.get('id')}\n{(c.get('text') or '')}" for c in chunks])
    if mode == "teacher":
        prompt = (
            "You are a teacher. Use ONLY the context below to answer the question. "
            "Return a JSON object with keys: answer (string), sources (list of chunk ids).\n\n"
            f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}\n"
        )
    else:
        prompt = (
            "Answer the question using the context below. Be concise and student-friendly.\n\n"
            f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}\n"
        )

    raw = _call_any_llm(prompt, max_tokens=call_opts.get("max_tokens", 256), temperature=call_opts.get("temperature", 0.0))

    out = {"status": "ok", "answer": "", "sources": [], "raw": raw, "chunks": chunks}

    # If no LLM available, synthesize from chunks offline
    if isinstance(raw, str) and raw.startswith("<<NO_LLM_AVAILABLE>>"):
        synth = _synthesize_answer_from_chunks(question, chunks, mode=mode)
        out["answer"] = synth["answer"]
        out["sources"] = synth.get("sources", [])
        out["status"] = "synthesized_offline"
        return out

    # existing teacher/student processing (unchanged)
    if mode == "teacher":
        try:
            first = raw.find("{")
            candidate = raw[first:] if first != -1 else raw
            parsed = json.loads(candidate)
            out["answer"] = parsed.get("answer", "")
            out["sources"] = parsed.get("sources", []) or []
            if not out["answer"].strip() and not out["sources"]:
                out["status"] = "out_of_syllabus"
        except Exception:
            out["status"] = "refer_teacher"
            out["answer"] = ""
    else:
        out["answer"] = raw
        import re as _re
        ids = _re.findall(r"[A-Za-z0-9_\-]+_[A-Za-z0-9_\-]+_[0-9a-f]{8}_[0-9]+", raw)
        out["sources"] = list(dict.fromkeys(ids))[:10]

    return out


# ---------------- CLI -------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="rag_answer", description="RAG answer CLI (student/teacher modes)")
    p.add_argument("--bundle", required=True, help="path to bundle directory (e.g. ./bundles/class_8_science_en)")
    p.add_argument("--embed", required=False, default="", help="path to id_map / embed file (optional)")
    p.add_argument("--query", required=True, help="user query")
    p.add_argument("--k", type=int, default=5, help="number of chunks to retrieve")
    p.add_argument("--model", type=str, default="2b", help="model variant")
    p.add_argument("--mode", choices=["student", "teacher"], default="student", help="response mode")
    p.add_argument("--debug", action="store_true", help="print retrieval/prompt/model output for debugging")
    p.add_argument("--plain", action="store_true", help="print human-readable text instead of JSON")
    args = p.parse_args(argv)

    # Debug: show retrieval/prompt/model if requested
    if args.debug:
        print("DEBUG: bundle =", args.bundle, "embed =", args.embed)
        try:
            # build qdict minimal
            qdict = {"query": args.query}
            dbg_ret = None
            # determine id_map_file from the bundle (do NOT use args.embed as id_map)
            id_map_file = None
            bundle_p = Path(args.bundle) if args.bundle else Path(".")
            if bundle_p.exists():
                candidates = [
                    bundle_p / "id_map.pkl",
                    bundle_p / "id_map.json",
                    bundle_p / "id_map.jsonl",
                    bundle_p / "id_map",
                    bundle_p / "data" / "id_map.pkl",
                ]
                for cand in candidates:
                    if cand.exists():
                        id_map_file = str(cand)
                        break
            if id_map_file is None:
                id_map_file = "unused"
            print("DEBUG: using id_map_file =", id_map_file)
            try:
                dbg_ret = retrieve_chunks(qdict, index_file="unused", id_map_file=id_map_file, k=args.k)
            except Exception as e_inner:
                # try to infer structured query from bundle name and retry
                import re
                m = re.search(r"class[_-]?(\d+)[_\-]([a-z]+)[_\-]([a-z]{2,})", str(Path(args.bundle).name), re.I)
                if m:
                    inferred = {
                        "class": int(m.group(1)),
                        "subject": m.group(2).lower(),
                        "language": m.group(3).lower(),
                        "query": args.query,
                        "chapter": 1,
                    }
                    try:
                        print("DEBUG: attempted inferred query ->", inferred)
                        dbg_ret = retrieve_chunks(inferred, index_file="unused", id_map_file=id_map_file, k=args.k)
                    except Exception as e2:
                        print("DEBUG: retrieve failed after inference:", e2)
                else:
                    print("DEBUG: retrieve failed:", e_inner)
            if dbg_ret is not None:
                print("DEBUG: retrieve ->", dbg_ret)
        except Exception as e:
            print("DEBUG: debug retrieval unexpected error:", e)

    res = get_rag_answer(bundle=args.bundle, embed=args.embed, query=args.query, k=args.k, model=args.model, mode=args.mode)

    if args.plain:
        if res.get("status") == "ok":
            if res.get("mode") == "teacher":
                print(res.get("content", ""))
                if res.get("sources"):
                    print("\nSources:", ", ".join(res.get("sources")))
            else:
                print(res.get("answer", ""))
        else:
            print("I'm not sure, you need to refer your teacher")
    else:
        print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
