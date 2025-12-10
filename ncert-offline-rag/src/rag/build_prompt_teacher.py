"""
build_prompt_teacher.py

Builds teacher-mode prompts for modes:
 - lecture_plan
 - detailed_notes
 - study_materials
 - assessment

Provides build_prompt(query, chunks, mode, max_chunks=5).
"""
from __future__ import annotations
from typing import List, Dict
import textwrap

CHUNK_TRUNC = 1500

MODE_INSTRUCTIONS = {
    "lecture_plan": (
        "Produce a structured 40-minute lecture plan. Include: a short title, 1-2 sentence summary, "
        "learning_outcomes (3+), duration_minutes (number), materials list, lesson_steps (time_min, activity, references), "
        "a short assessment (3 MCQs with answers) and teacher notes. Be concise in 'summary' and exhaustive in 'notes'."
    ),
    "detailed_notes": (
        "Produce detailed stepwise lecture notes with examples, suggested diagrams (text descriptions), "
        "and common misconceptions. Return structured JSON following the schema."
    ),
    "study_materials": (
        "Produce concise revision notes, key formulas, 5+ practice problems with answers, and quick tips for students."
    ),
    "assessment": (
        "Generate a question bank (with answers), marking scheme, expected time per question and difficulty tags."
    ),
}


SCHEMA_TEXT = textwrap.dedent("""
Output contract: Produce EXACTLY one JSON object and nothing else with the following keys:
{
  "status":"ok",
  "mode":"<mode>",
  "title":"<short title>",
  "summary":"<1-2 sentence summary>",
  "learning_outcomes":[ "<lo1>", ...],
  "duration_minutes": 40,
  "materials":["<item1>", ...],
  "lesson_steps":[ {"time_min":5,"activity":"...","references":["<chunk_id1>"]}, ... ],
  "assessment":[ {"q":"...","a":"...","difficulty":"easy"} ],
  "notes":"<long teacher notes / tips>",
  "sources":["id1","id2",...]
}
- All fields must be present (empty arrays allowed except 'sources').
- MUST use ONLY the CONTEXT below.
- ALWAYS cite chunk IDs inline (in lesson_steps.references) and include them in the top-level "sources" array.
- If the CONTEXT is insufficient to answer, return {"status":"refer_teacher"} (nothing else).
""").strip()


def _format_chunk(c: Dict) -> str:
    text = c.get("text", "") or ""
    if len(text) > CHUNK_TRUNC:
        text = text[:CHUNK_TRUNC] + " [TRUNCATED]"
    cid = c.get("id", "<no-id>")
    return f"{cid}\nTEXT: {text}"


def build_prompt(query: str, chunks: List[Dict], mode: str = "lecture_plan", max_chunks: int = 5) -> str:
    if mode not in MODE_INSTRUCTIONS:
        raise ValueError("unknown mode")
    used = chunks[:max_chunks]
    ctx_lines = [_format_chunk(c) for c in used]
    context = "\n---\n".join(ctx_lines)
    instructions = (
        f"You are an expert teacher assistant. MODE: {mode}\n\n"
        f"{MODE_INSTRUCTIONS[mode]}\n\n"
        f"{SCHEMA_TEXT}\n\n"
        "QUESTION:\n"
    )
    prompt = f"{instructions}{query}\n\nCONTEXT (use only this):\n{context}\n\nReturn the single JSON object now and nothing else."
    return prompt