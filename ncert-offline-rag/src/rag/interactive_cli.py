from __future__ import annotations
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

def run(cmd: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=os.environ)

def _find_json_blob(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # try direct parse
    try:
        return json.loads(text)
    except Exception:
        # find first {...} JSON object in text
        m = re.search(r"\{.*\}", text, re.DOTALL)
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
            return False
        s = s.strip()
        if s in full or s in suffixes or any(rid.endswith(s) for rid in retrieved_ids):
            return True
    return False

def load_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def interactive_loop(bundle: str, model_variant: str, k: int, embed_model: str, threshold: float):
    repo_root = Path(__file__).resolve().parents[3]
    embed_py = repo_root / "ncert-offline-rag" / "src" / "pi_runtime" / "embed_query.py"
    retrieve_py = repo_root / "ncert-offline-rag" / "src" / "rag" / "retrieve.py"
    build_prompt_py = repo_root / "ncert-offline-rag" / "src" / "rag" / "build_prompt.py"
    gemma_call_py = repo_root / "ncert-offline-rag" / "src" / "rag" / "gemma_call.py"

    if not embed_py.exists() or not retrieve_py.exists() or not build_prompt_py.exists() or not gemma_call_py.exists():
        print("Required runtime scripts missing under ncert-offline-rag/src (run from repo root).")
        return

    bp = load_module_from_path(build_prompt_py, "build_prompt")
    gc = load_module_from_path(gemma_call_py, "gemma_call")

    print("Interactive RAG CLI — type a question, or 'quit' to exit.")
    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in ("quit", "exit"):
            break

        # embed
        tf = tempfile.NamedTemporaryFile(prefix="q_", suffix=".json", delete=False)
        embed_path = Path(tf.name)
        tf.close()

        proc = run([sys.executable, str(embed_py), "--text", q, "--output", str(embed_path), "--model", embed_model], cwd=str(repo_root))
        if proc.returncode != 0:
            print("I'm not sure, you need to refer your teacher")
            continue

        # small wait for FS
        for _ in range(10):
            if embed_path.exists() and embed_path.stat().st_size > 10:
                break
            time.sleep(0.02)

        # retrieve
        proc = run([sys.executable, str(retrieve_py), "--bundle", bundle, "--embed", str(embed_path), "--k", str(k)], cwd=str(repo_root))
        if proc.returncode != 0:
            print("I'm not sure, you need to refer your teacher")
            continue

        try:
            retrieved = json.loads(proc.stdout)
        except Exception:
            print("I'm not sure, you need to refer your teacher")
            continue

        chunks = retrieved.get("chunks", [])
        if not chunks:
            print("I'm not sure, you need to refer your teacher")
            continue

        top_score = chunks[0].get("score", 0.0)
        retrieved_ids = [c.get("id") for c in chunks if c.get("id")]

        if top_score < threshold:
            print("I'm not sure, you need to refer your teacher")
            continue

        # build prompt and call model directly
        try:
            prompt = bp.build_prompt(q, chunks)
        except Exception:
            print("I'm not sure, you need to refer your teacher")
            continue

        try:
            raw = gc.call_gemma(prompt, model_variant=model_variant)
        except Exception:
            print("I'm not sure, you need to refer your teacher")
            continue

        parsed = _find_json_blob(raw)
        if not isinstance(parsed, dict):
            print("I'm not sure, you need to refer your teacher")
            continue

        # validate answer + sources
        ans = parsed.get("answer", "")
        srcs = parsed.get("sources", [])
        if not ans or not _sources_match(srcs, retrieved_ids):
            print("I'm not sure, you need to refer your teacher")
            continue

        # production style: print only answer
        print("\nAnswer:\n" + ans.strip())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default="./bundle/class_8_science_en")
    p.add_argument("--model", default="2b")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--embed-model", default="all-mpnet-base-v2")
    p.add_argument("--threshold", type=float, default=0.60, help="min top-1 similarity to answer")
    args = p.parse_args()

    interactive_loop(args.bundle, args.model, args.k, args.embed_model, args.threshold)

if __name__ == "__main__":
    main()