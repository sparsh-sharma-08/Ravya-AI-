from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# robust imports: allow running as package or direct script
try:
    from .retrieve import retrieve  # type: ignore
    from .build_prompt import build_prompt  # type: ignore
    from .gemma_call import call_gemma  # type: ignore
except Exception:
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    try:
        from retrieve import retrieve  # type: ignore
        from build_prompt import build_prompt  # type: ignore
        from gemma_call import call_gemma  # type: ignore
    except Exception:
        import importlib.util
        def _load_mod_from_path(name: str, path: Path):
            spec = importlib.util.spec_from_file_location(name, str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            return mod
        retrieve = _load_mod_from_path("retrieve", _here / "retrieve.py").retrieve
        build_prompt = _load_mod_from_path("build_prompt", _here / "build_prompt.py").build_prompt
        call_gemma = _load_mod_from_path("gemma_call", _here / "gemma_call.py").call_gemma

def _extract_json_from_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start:end+1]

def _validate_output(parsed: Dict, retrieved_ids: List[str]) -> bool:
    if not isinstance(parsed, dict):
        return False
    ans = parsed.get("answer")
    srcs = parsed.get("sources")
    if not isinstance(ans, str) or not ans.strip():
        return False
    if not isinstance(srcs, list) or len(srcs) == 0:
        return False

    # build helper sets: full ids and suffix hashes (last '_' segment)
    full_ids = set(retrieved_ids)
    hashes = set()
    for rid in retrieved_ids:
        if "_" in rid:
            hashes.add(rid.rsplit("_", 1)[-1])

    found = False
    for s in srcs:
        if not isinstance(s, str):
            return False
        s = s.strip()
        # exact full id match
        if s in full_ids:
            found = True
            continue
        # model may return only the chunk hash (suffix) or a suffix of the id
        if s in hashes:
            found = True
            continue
        # fallback: check if any retrieved id endswith the returned string
        if any(rid.endswith(s) for rid in retrieved_ids):
            found = True
            continue
    return found

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--embed", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model", default="2b", choices=["2b", "7b"])
    # add debug arg to the parser
    parser.add_argument("--debug", action="store_true", help="print retrieval, prompt and raw model output for debugging")
    parser.add_argument("--plain", action="store_true", help="print only the answer text (no json)")
    args = parser.parse_args()

    try:
        ret = retrieve(args.bundle, args.embed, args.k)
    except Exception:
        print(json.dumps({"status": "refer_teacher"}, ensure_ascii=False))
        sys.exit(0)

    if not isinstance(ret, dict) or ret.get("status") != "ok":
        print(json.dumps({"status": "refer_teacher"}, ensure_ascii=False))
        sys.exit(0)

    chunks = ret.get("chunks", [])[: args.k]
    retrieved_ids = [c["id"] for c in chunks]

    prompt = build_prompt(args.query, chunks)

    if args.debug:
        try:
            print("DEBUG: retrieve chunks (top-k):", file=sys.stderr)
            print(json.dumps(ret.get("chunks", []), indent=2, ensure_ascii=False), file=sys.stderr)
        except Exception as _e:
            print("DEBUG: failed to print retrieve:", _e, file=sys.stderr)

    if args.debug:
        try:
            print("DEBUG: prompt (first 2000 chars):", file=sys.stderr)
            print((prompt[:2000] if isinstance(prompt, str) else str(prompt)), file=sys.stderr)
        except Exception as _e:
            print("DEBUG: failed to print prompt:", _e, file=sys.stderr)

    try:
        out_text = call_gemma(prompt, model_variant=args.model)
    except Exception:
        print(json.dumps({"status": "refer_teacher"}, ensure_ascii=False))
        sys.exit(0)

    # debug: print actual returned text and the extracted json blob
    if args.debug:
        try:
            print("DEBUG: model returned (raw):", file=sys.stderr)
            print(out_text if out_text is not None else "<None>", file=sys.stderr)
            json_blob_dbg = _extract_json_from_text(out_text)
            print("DEBUG: extracted JSON blob:", file=sys.stderr)
            print(json_blob_dbg if json_blob_dbg else "<no json blob found>", file=sys.stderr)
        except Exception as _e:
            print("DEBUG: failed to print model output:", _e, file=sys.stderr)

    json_blob = _extract_json_from_text(out_text)
    if not json_blob:
        print(json.dumps({"status": "refer_teacher"}, ensure_ascii=False))
        sys.exit(0)
    try:
        parsed = json.loads(json_blob)
    except Exception:
        print(json.dumps({"status": "refer_teacher"}, ensure_ascii=False))
        sys.exit(0)

    if not _validate_output(parsed, retrieved_ids):
        print(json.dumps({"status": "refer_teacher"}, ensure_ascii=False))
        sys.exit(0)

    answer = parsed["answer"].strip()
    sources = parsed["sources"]
    result = {"status": "ok", "answer": answer, "sources": sources}

    if args.plain:
        try:
            if isinstance(result, dict) and result.get("status") == "ok" and result.get("answer"):
                print(result["answer"].strip())
            else:
                # production-friendly fallback when no confident answer
                print("I'm not sure, you need to refer your teacher")
        except Exception:
            print("I'm not sure, you need to refer your teacher")
    else:
        # original behaviour
        print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()