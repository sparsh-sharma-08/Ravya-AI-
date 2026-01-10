#!/usr/bin/env python3
"""
scripts/print_retrieved_chunks.py

Usage:
  PYTHONPATH="$PWD" python scripts/print_retrieved_chunks.py \
     --bundle bundles/class_8_science_en \
     --query "What happens when magnesium is burned in air?" \
     --k 5

This prints top-k retrieved chunk id, score, metadata and a text snippet.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="Path to bundle directory")
    p.add_argument("--query", required=True, help="User query text")
    p.add_argument("--k", type=int, default=5, help="Top-k to print")
    p.add_argument("--mode", choices=["student","teacher"], default="student")
    p.add_argument("--embed_model", default="all-mpnet-base-v2")
    args = p.parse_args()

    # import the retrieval implementation
    try:
        from src.pi_runtime.retrieve import retrieve  # type: ignore
    except Exception as e:
        print("ERROR: failed to import retrieve from src.pi_runtime.retrieve:", e)
        return

    # call retrieval
    try:
        results = retrieve(bundle_path=args.bundle, query=args.query, k=args.k, mode=args.mode, embed_model=args.embed_model)
    except Exception as e:
        print("ERROR running retrieve():", e)
        return

    if not results:
        print("No results returned.")
        return

    # results expected as List[Dict] with keys id, score, text, meta
    print(f"Top {args.k} results for query: {args.query!r}\n")
    for rank, r in enumerate(results[: args.k]):
        cid = r.get("id") or r.get("metadata", {}).get("id") or "<no-id>"
        score = r.get("score") if r.get("score") is not None else r.get("sim") or r.get("distance") or 0.0
        meta = r.get("meta") or r.get("metadata") or {}
        subj = meta.get("subject") or meta.get("Subject") or ""
        chap = meta.get("chapter") or ""
        tokens = meta.get("tokens") or ""
        text = r.get("text") or ""
        snippet = text.replace("\n", " ").strip()
        if len(snippet) > 600:
            snippet = snippet[:600] + "..."
        print(f"--- rank {rank+1} ---")
        print("id:      ", cid)
        print("score:   ", float(score))
        print("subject: ", subj)
        print("chapter: ", chap)
        print("tokens:  ", tokens)
        print("snippet: ", snippet)
        print()

    # also print raw JSON if needed
    print("Raw JSON (first result):")
    print(json.dumps(results[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()