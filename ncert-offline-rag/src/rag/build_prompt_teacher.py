"""
Teacher-mode prompt builder.

Creates a strict instruction prompt that:
- Addresses the model as "CBSE teacher assistant"
- Requires using ONLY the provided context chunks
- Requires JSON output with exact schema:
  {"content":"<long notes>", "sources":["id1","id2",...]}
- Includes structured sections and a 200-300+ word requirement
- Adds the context chunks in the specified "[<chunk_id>]\n<chunk_text>" format
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
    cid = c.get("id") or c.get("hash") or "<unknown>"
    text = c.get("text", "")
    return f"[{cid}]\n{text}\n"


def build_prompt_teacher(question: str, chunks: List[Dict]) -> str:
    """
    Build a teacher-mode prompt.

    Args:
      question: user question string
      chunks: list of chunk dicts (each with 'id' and 'text' at minimum)

    Returns:
      A single string prompt to send to the model.
    """
    header = (
        "You are a CBSE teacher assistant. "
        "Use ONLY the provided context chunks to answer the user's request. "
        "If the answer is not fully supported by the provided chunks, respond exactly:\n"
        "I don't know, ask your teacher\n\n"
    )

    instruction = (
        "Generate detailed lecture / teaching material targeted at older school students "
        "and teachers (not toddlers). Produce a structured output with these sections:\n\n"
        "Overview\n\n"
        "Learning objectives\n\n"
        "Key concepts and definitions\n\n"
        "Stepwise explanation\n\n"
        "Examples (numerical/real-life as appropriate)\n\n"
        "Classroom activities / experiments\n\n"
        "Summary / recap questions\n\n"
        "Requirements:\n"
        "- Use ONLY the provided context chunks. Do NOT use external knowledge beyond what is "
        "explicitly present in the chunks.\n"
        "- The content must be at least ~200-300 words and suitably detailed.\n"
        "- Target audience: older school students and teachers.\n\n"
    )

    json_schema = (
        "Return ONLY valid JSON with this exact schema (no extra text, no explanation before/after):\n"
        '{"content":"<long formatted notes as markdown or bullet points>", "sources":["<chunk_id_1>","<chunk_id_2>", ...]}\n\n'
        "The 'sources' array must include the ids of the chunks you used (at least one),\n"
        "and 'content' must be non-empty.\n\n"
    )

    prompt_parts = [header, instruction, json_schema, "User question:\n", f"{question}\n\n", "Context chunks:\n"]
    for c in chunks:
        prompt_parts.append(_format_chunk(c))

    prompt_parts.append("\nImportant: If you are not sure, respond exactly: I don't know, ask your teacher\n")
    return "".join(prompt_parts)