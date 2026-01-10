from __future__ import annotations
from typing import List, Dict, Any

def build_prompt(question: str, chunks: List[Dict[str, Any]], mode: str = "student") -> str:
    """
    Build a student-mode prompt. Be tolerant to weak context and allow a natural-language answer.
    """
    header = (
        "You are a helpful assistant. Use the provided CONTEXT to answer the QUESTION. "
        "If the CONTEXT contains relevant information, include it in your answer and list sources by chunk id. "
        "If not, you may answer briefly from general knowledge but prefer quoting NCERT content where available.\n\n"
    )
    ctx_parts = []
    for c in chunks:
        ctx_parts.append(f"{c.get('id')}\nTEXT:\n{(c.get('text') or '')}\n---")

    ctx = "\n".join(ctx_parts)

    prompt = f"{header}\nQUESTION:\n{question}\n\nCONTEXT:\n{ctx}\n\nAnswer conversationally; if you cite chunks, include their ids."
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