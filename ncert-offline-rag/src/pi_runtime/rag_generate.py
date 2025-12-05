"""
RAG generator using local Gemma via Ollama.

Input:
- query text
- retrieved chunks

Output:
{
  "answer": "...",
  "sources": ["chunk_id"]
}
"""

from __future__ import annotations
import subprocess
from typing import List, Dict


def build_prompt(query: str, chunks: List[Dict]) -> str:
    header = (
        "You are a CBSE teacher assistant.\n"
        "Use ONLY the context chunks to answer.\n"
        "DO NOT answer from memory.\n"
        "If uncertain say: \"I don't know, ask your teacher.\"\n"
        "Cite chunk IDs you use.\n\n"
        "Context:\n"
    )

    parts = [header]
    for c in chunks:
        cid = c.get("id")
        text = c.get("text", "").strip()
        parts.append(f"[{cid}]\n{text}\n\n")

    parts.append("Question:\n" + query + "\n\nAnswer:\n")
    return "".join(parts)


def _call_ollama(prompt: str, variant: str) -> str:
    model = f"gemma:{variant}"
    out = subprocess.run(
        ["ollama", "run", model, "--prompt", prompt],
        capture_output=True,
        text=True
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr or out.stdout)
    return out.stdout.strip()


def _extract_citations(text: str, candidate_ids: List[str]) -> List[str]:
    cited = []
    for cid in candidate_ids:
        if cid in text:
            cited.append(cid)

    seen = set()
    result = []
    for cid in cited:
        if cid not in seen:
            seen.add(cid)
            result.append(cid)
    return result


def generate_answer(query: str, retrieved_chunks: List[Dict], model_variant: str = "2b") -> Dict:
    if not retrieved_chunks:
        return {"answer": "I don't know, ask your teacher.", "sources": []}

    prompt = build_prompt(query, retrieved_chunks)
    out = _call_ollama(prompt, model_variant)

    ids = [c["id"] for c in retrieved_chunks]
    sources = _extract_citations(out, ids)

    if not sources:
        return {"answer": "I don't know, ask your teacher.", "sources": []}

    return {"answer": out, "sources": sources}
