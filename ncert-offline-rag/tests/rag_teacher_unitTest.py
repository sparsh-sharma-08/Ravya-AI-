"""
Unit tests for rag_teacher runner using monkeypatching.

Tests:
- success path: high top1 score, gemma returns valid JSON with matching source suffix -> status ok
- refer_teacher when top1 < threshold
"""
from __future__ import annotations
import json
import argparse
import types
import pytest

# Import module under test by path
import importlib.util
from pathlib import Path
import sys

# helper to load module by path
def load_mod(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

RAG_TEACHER_PY = Path("ncert-offline-rag/src/rag/rag_teacher.py").resolve()
MOD = load_mod(str(RAG_TEACHER_PY), "rag_teacher_testable")
rag_teacher = MOD

class DummyCompleted:
    def __init__(self, stdout):
        self.stdout = stdout
        self.returncode = 0

def test_success_path(monkeypatch):
    # fake retrieve subprocess output: top1 score 0.82 and two chunks
    fake_retr = {
        "chunks": [
            {"id":"10_Science_..._abc123","score":0.82,"text":"Endothermic ..."},
            {"id":"10_Science_..._def456","score":0.70,"text":"More text..."}
        ]
    }
    monkeypatch.setattr(rag_teacher, "run", lambda *a, **k: DummyCompleted(stdout=json.dumps(fake_retr)))
    # fake gemma_call module loader to return object with call_gemma
    class FakeGemmaMod:
        @staticmethod
        def call_gemma(prompt, model_variant="2b"):
            # return a JSON blob that uses suffix id 'abc123' for sources
            return json.dumps({
                "title":"Photosynthesis - short",
                "summary":"Photosynthesis converts light energy.",
                "learning_outcomes":["LO1","LO2","LO3"],
                "duration_minutes":40,
                "materials":["leaf","chart"],
                "lesson_steps":[ {"time_min":10,"activity":"Explain", "references":["abc123"]} ],
                "assessment":[ {"q":"MCQ1","a":"A","difficulty":"easy"} ],
                "notes":"Teacher tips",
                "sources":["abc123"]
            })
    # loader must return different fake modules depending on requested path:
    def _fake_loader(path, name):
        p = str(path)
        if "build_prompt_teacher.py" in p:
            return __import__("types").SimpleNamespace(build_prompt=lambda q, ch, mode, max_chunks=5: "FAKEPROMPT")
        # gemma_call fallback
        return __import__("types").SimpleNamespace(call_gemma=FakeGemmaMod.call_gemma)