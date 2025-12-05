"""
Simple wrapper:
retrieved.json --> answer.json
"""

from __future__ import annotations
import json
from typing import List, Dict

from rag_generate import generate_answer


def answer(query: str, retrieved_payload: Dict, model_variant="2b"):
    if retrieved_payload.get("status") != "ok":
        return {"answer": "I don't know, ask your teacher.", "sources": []}

    chunks = retrieved_payload.get("chunks", [])[:5]
    return generate_answer(query, chunks, model_variant)


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
