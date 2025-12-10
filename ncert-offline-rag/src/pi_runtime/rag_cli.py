"""
End-to-end offline RAG CLI for Raspberry Pi.

Process:
- Load FAISS bundle
- Load precomputed query embedding (from laptop)
- Retrieve top-K chunks
- Generate local Gemma answer

Returns strict JSON:
- {"status": "refer_teacher"}
- {"answer": "...", "sources":[...]}
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

from retrieve import retrieve
from rag_generate import generate_answer


def main():
    import argparse
    p = argparse.ArgumentParser(description="Offline RAG CLI (Pi)")
    p.add_argument("--bundle", required=True, help="Path to FAISS bundle directory")
    p.add_argument("--embed", required=True, help="Query embedding (.json or .npy)")
    p.add_argument("--query", required=True, help="User question text")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--model", default="2b", choices=["2b","7b"], help="Gemma model variant")

    args = p.parse_args()

    try:
        r = retrieve(args.bundle, args.embed, args.k)
    except Exception as e:
        print(json.dumps({"status":"refer_teacher", "error":str(e)}))
        sys.exit(0)

    if r.get("status") != "ok":
        print(json.dumps({"status":"refer_teacher"}))
        sys.exit(0)

    chunks = r["chunks"][: args.k]

    try:
        out = generate_answer(args.query, chunks, args.model)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"answer":"I don't know, ask your teacher.","sources":[],"error":str(e)},ensure_ascii=False))


if __name__ == "__main__":
    main()
