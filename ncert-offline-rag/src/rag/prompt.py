from __future__ import annotations
"""
src/rag/build_prompt.py
Build the strict prompt for Gemma given question and chunks.
"""
from typing import List, Dict


def build_prompt(query: str, chunks: List[Dict]) -> str:
    # Show chunk ids exactly as expected in "sources"
    ctx_lines = []
    for c in chunks:
        cid = c.get("id", "<no-id>")
        snippet = c.get("text", "").replace("\n", " ").strip()
        if len(snippet) > 1000:
            snippet = snippet[:1000] + "..."
        ctx_lines.append(f"{cid}\nTEXT: {snippet}\n")
    context = "\n---\n".join(ctx_lines)

    instructions = (
        "Using ONLY the provided CONTEXT, answer the QUESTION in a clear, teacher-like style. "
        "Begin with one concise direct sentence that answers the question, then provide a numbered set of steps or short paragraphs "
        "that explain the underlying process, important details, and one brief safety or practical note if relevant. Aim for a detailed "
        "response of about 6-10 short sentences presented as 3-6 numbered steps or short paragraphs. Do NOT invent facts — if the CONTEXT "
        "does not support a claim, omit it or say you cannot answer that part.\n\n"
        "Output requirements (EXACTLY one JSON object, nothing else):\n"
        '  "answer": string (the detailed, teacher-style answer; empty string if not available),\n'
        '  "sources": array of chunk id strings (MUST match the ids shown above exactly).\n\n'
        "If the CONTEXT does NOT contain enough information to answer, return: {\"answer\": \"\", \"sources\": []}.\n\n"
        "QUESTION:\n"
    )

    prompt = (
        f"{instructions}"
        f"{query}\n\n"
        "CONTEXT (use only this):\n"
        f"{context}\n\n"
        "Return the single JSON object now and nothing else."
    )
    return prompt


if __name__ == "__main__":
    # quick CLI to print prompt (for debugging)
    import argparse, json, sys
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--retrieved", required=True, help="Path to retrieve JSON")
    args = p.parse_args()
    with open(args.retrieved, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("status") != "ok":
        print("ERROR: retrieved status not ok", file=sys.stderr)
        sys.exit(2)
    chunks = payload.get("chunks", [])[:5]
    print(build_prompt(args.query, chunks))