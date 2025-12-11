from __future__ import annotations
from typing import List, Dict

def build_prompt(query: str, chunks: List[Dict]) -> str:
    # Show chunk ids exactly as expected in "sources"
    ctx_lines = []
    for c in chunks:
        cid = c.get("id", "<no-id>")
        snippet = c.get("text", "").replace("\n", " ").strip()
        if len(snippet) > 800:
            snippet = snippet[:800] + "..."
        ctx_lines.append(f"{cid}\nTEXT: {snippet}\n")
    context = "\n---\n".join(ctx_lines)

    instructions = (
        "Using ONLY the provided CONTEXT, answer the QUESTION in 2-5 lines. "
        "If the CONTEXT contains relevant information, produce a factual answer "
        "and include the exact chunk id(s) used in the \"sources\" array. "
        "If the CONTEXT does NOT contain information to answer, return an empty answer and an empty sources array.\n\n"
        "Output requirements (EXACTLY one JSON object, nothing else):\n"
        '  "answer": string (the concise answer; empty string if not available),\n'
        '  "sources": array of chunk id strings (MUST match the ids shown above exactly).\n\n'
        "Do NOT output any additional text, explanation, or templates. Do NOT invent facts — only use the CONTEXT.\n\n"
        "QUESTION:\n"
    )

    prompt = (
        "You are an expert assistant designed to answer questions based on provided context. "
        f"{instructions}"
        f"{query}\n\n"
        "CONTEXT (use only this):\n"
        f"{context}\n\n"
        "Return the JSON object now and nothing else."
    )
    return prompt

STUDENT_PROMPT = """You are an NCERT question-answering assistant.

You MUST answer ONLY using the provided context.
If the answer is not present in the context, return:

{"answer": "", "sources": []}

Otherwise return STRICTLY this JSON format (no markdown, no explanation):

{
  "answer": "<short answer to the question>",
  "sources": ["<id1>", "<id2>"]
}

Do NOT output anything outside the JSON object.
Do NOT include explanations.
Do NOT include markdown.

CONTEXT:
{{context}}

QUESTION:
{{query}}
"""