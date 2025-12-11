import json
import pytest
from src.rag.build_prompt_teacher import build_prompt_teacher
from src.rag.rag_answer import _validate_teacher_response

def test_build_prompt_teacher_contains_instructions():
    question = "Explain photosynthesis for class 7"
    chunks = [
        {"id": "chunk1", "text": "Photosynthesis is ..."},
        {"id": "chunk2", "text": "Chlorophyll absorbs light ..."},
    ]
    prompt = build_prompt_teacher(question, chunks)
    assert "CBSE teacher assistant" in prompt or "CBSE Teacher" in prompt
    assert "[chunk1]" in prompt
    assert "[chunk2]" in prompt
    assert "learning objectives" in prompt.lower()
    assert '"content":' in prompt or "Return ONLY valid JSON" in prompt

def test_validate_teacher_response_accepts_valid_and_rejects_invalid():
    chunks = [{"id": "chunk1", "text": "A"}, {"id": "chunk2", "text": "B"}]

    valid = {"content": "Long notes ...", "sources": ["chunk1"]}
    assert _validate_teacher_response(valid, chunks)

    no_content = {"content": "", "sources": ["chunk1"]}
    assert not _validate_teacher_response(no_content, chunks)

    no_sources = {"content": "Notes", "sources": []}
    assert not _validate_teacher_response(no_sources, chunks)

    unknown_source = {"content": "Notes", "sources": ["unknown"]}
    assert not _validate_teacher_response(unknown_source, chunks)