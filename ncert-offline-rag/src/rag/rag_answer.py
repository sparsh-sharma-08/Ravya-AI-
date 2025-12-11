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
from pathlib import Path

import numpy as np
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Union
import re
from typing import List, Dict, Any, Optional

from src.rag.build_prompt import build_prompt
from src.rag.build_prompt_teacher import build_prompt_teacher
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
from src.pi_runtime.retrieve import retrieve_chunks


# ---------------- Validators -------------------------------------------------
def _ensure_chunk_ids(chunks_or_ids: Optional[Union[Sequence[Dict[str, Any]], Sequence[str]]]) -> set:
    if not chunks_or_ids:
        return set()
    first = next(iter(chunks_or_ids), None)
    if isinstance(first, dict):
        return {c.get("id") for c in chunks_or_ids if isinstance(c, dict) and c.get("id")}
    return set(chunks_or_ids)  # assume list of ids


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


def _validate_teacher_response(parsed: Dict[str, Any], chunks_or_ids: Optional[Union[List[Dict[str, Any]], List[str]]]) -> bool:
    """
    Accepts {"content":"...","sources":[...]}
    Require non-empty content and at least one source matching chunk ids.
    """
    if not isinstance(parsed, dict):
        return False
    content = parsed.get("content")
    sources = parsed.get("sources")
    if not content or not isinstance(sources, list) or len(sources) == 0:
        return False
    chunk_ids = _ensure_chunk_ids(chunks_or_ids)
    if not chunk_ids:
        return True
    return any(s in chunk_ids for s in sources)


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
        if not _validate_teacher_response(parsed, chunks):
            return _refer("teacher validation failed", ret=ret, chunks=chunks, prompt=prompt, model_output=model_output)
        return {"status": "ok", "mode": "teacher", "content": parsed["content"], "sources": parsed["sources"]}
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