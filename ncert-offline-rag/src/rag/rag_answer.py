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
    found = False
    for s in srcs:
        if not isinstance(s, str):
            return False
        if s in retrieved_ids:
            found = True
    return found

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--embed", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--model", default="2b", choices=["2b", "7b"])
    args = p.parse_args()

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

    try:
        out_text = call_gemma(prompt, model_variant=args.model)
    except Exception:
        print(json.dumps({"status": "refer_teacher"}, ensure_ascii=False))
        sys.exit(0)

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
    print(json.dumps({"status": "ok", "answer": answer, "sources": sources}, ensure_ascii=False))

if __name__ == "__main__":
    main()