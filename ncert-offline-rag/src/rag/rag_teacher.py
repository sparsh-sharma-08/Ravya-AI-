"""
rag_teacher.py

CLI to run teacher-mode RAG:
 - uses retrieve.py (subprocess) to load top-k chunks
 - builds teacher prompt via build_prompt_teacher
 - calls local LLM via gemma_call.call_gemma
 - validates JSON + sources and prints strict JSON output
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# local imports via importlib when needed
import importlib.util

RETRIEVE_TIMEOUT = 60
TOP1_THRESHOLD = 0.60

def run(cmd: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env, timeout=RETRIEVE_TIMEOUT)

def _find_json_blob(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

def _sources_match(returned: List[str], retrieved_ids: List[str]) -> bool:
    if not returned:
        return False
    full = set(retrieved_ids)
    suffixes = {rid.rsplit("_", 1)[-1] for rid in retrieved_ids if "_" in rid}
    for s in returned:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if s in full or s in suffixes or any(rid.endswith(s) for rid in retrieved_ids):
            return True
    return False

def load_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def run_teacher(args_ns: argparse.Namespace) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    retrieve_py = project_root / "ncert-offline-rag" / "src" / "rag" / "retrieve.py"
    gemma_call_py = project_root / "ncert-offline-rag" / "src" / "rag" / "gemma_call.py"
    build_prompt_py = project_root / "ncert-offline-rag" / "src" / "rag" / "build_prompt_teacher.py"

    if not retrieve_py.exists() or not gemma_call_py.exists() or not build_prompt_py.exists():
        return {"status": "refer_teacher"}

    # 1) retrieve via subprocess (reuse existing CLI)
    try:
        proc = run([sys.executable, str(retrieve_py),
                    "--bundle", args_ns.bundle,
                    "--embed", args_ns.embed,
                    "--k", str(args_ns.k)],
                   cwd=str(project_root))
    except Exception:
        return {"status": "refer_teacher"}

    try:
        retrieved = json.loads(proc.stdout)
    except Exception:
        return {"status": "refer_teacher"}

    chunks = retrieved.get("chunks", [])
    if not chunks:
        return {"status": "refer_teacher"}
    top_score = chunks[0].get("score", 0.0)
    if top_score < TOP1_THRESHOLD:
        return {"status": "refer_teacher"}

    # build prompt
    bp = load_module_from_path(build_prompt_py, "build_prompt_teacher")
    prompt = bp.build_prompt(args_ns.query, chunks, args_ns.mode, max_chunks=min(5, args_ns.k))

    # call model
    gc = load_module_from_path(gemma_call_py, "gemma_call")
    try:
        raw = gc.call_gemma(prompt, model_variant=args_ns.model)
    except Exception:
        return {"status": "refer_teacher"}

    parsed = _find_json_blob(raw)
    if not isinstance(parsed, dict):
        return {"status": "refer_teacher"}

    # validate required fields
    if parsed.get("status") == "refer_teacher":
        return {"status": "refer_teacher"}

    # ensure at least one source matches retrieved ids
    ret_sources = parsed.get("sources", [])
    retrieved_ids = [c.get("id") for c in chunks if c.get("id")]
    if not _sources_match(ret_sources, retrieved_ids):
        return {"status": "refer_teacher"}

    # enforce schema minimal presence
    # add mode echo and status
    parsed_out = {"status": "ok", "mode": args_ns.mode}
    # copy permitted keys (if absent, provide defaults)
    keys = ["title", "summary", "learning_outcomes", "duration_minutes", "materials", "lesson_steps", "assessment", "notes", "sources"]
    for k in keys:
        parsed_out[k] = parsed.get(k, [] if k.endswith("s") or k in ("materials","learning_outcomes","lesson_steps","assessment","sources") else parsed.get(k, ""))

    return parsed_out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--embed", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--mode", required=True, choices=["lecture_plan","detailed_notes","study_materials","assessment"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--model", default="2b", choices=["2b","7b"])
    args = p.parse_args()

    out = run_teacher(args)
    # print JSON only
    print(json.dumps(out, ensure_ascii=False))
    sys.exit(0)

if __name__ == "__main__":
    main()