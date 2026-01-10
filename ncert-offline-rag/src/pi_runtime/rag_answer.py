"""
Simple wrapper:
retrieved.json --> answer.json
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List
from pathlib import Path
import sys
import traceback

from rag_generate import generate_answer


def answer(query: str, retrieved_payload: Dict, model_variant="2b"):
    if retrieved_payload.get("status") != "ok":
        return {"answer": "I don't know, ask your teacher.", "sources": []}

    chunks = retrieved_payload.get("chunks", [])[:5]
    return generate_answer(query, chunks, model_variant)


# safe-wrapper: if get_rag_answer is defined later in this module, replace it with a wrapped version
def _install_safe_get_rag_answer():
    if "get_rag_answer" not in globals():
        return
    _orig = globals()["get_rag_answer"]
    if getattr(_orig, "_is_safe_wrapped", False):
        return

    def _safe_get_rag_answer(*args, **kwargs):
        try:
            return _orig(*args, **kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            # ensure debug info is visible in CLI
            print("Error running RAG:", e, file=sys.stderr)
            print(tb, file=sys.stderr)
            return {"status": "error", "error": str(e), "traceback": tb}
    _safe_get_rag_answer._is_safe_wrapped = True
    globals()["get_rag_answer"] = _safe_get_rag_answer


# Install wrapper now if function already exists; otherwise it will be idempotent if called later.
_install_safe_get_rag_answer()

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="RAG answer wrapper")
    p.add_argument("--query", required=True)
    p.add_argument("--retrieved", required=True)
    p.add_argument("--model", default="2b", choices=["2b","7b"])
    args = p.parse_args()

    with open(args.retrieved, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    res = answer(args.query, payload, args.model)
    print(json.dumps(res, ensure_ascii=False, indent=2))