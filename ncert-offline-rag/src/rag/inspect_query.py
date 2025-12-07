from __future__ import annotations
import argparse
import subprocess
import sys
import os
import json
import tempfile
import time
from pathlib import Path
import re

def run(cmd, cwd=None):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=os.environ)

def _find_json_blob(text: str):
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # fallback: find first {...} JSON object in text
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--bundle", default="./bundle/class_8_science_en")
    p.add_argument("--model", default="2b")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--embed-model", default="all-mpnet-base-v2")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    embed_py = project_root / "ncert-offline-rag" / "src" / "pi_runtime" / "embed_query.py"
    rag_answer_py = project_root / "ncert-offline-rag" / "src" / "rag" / "rag_answer.py"

    tf = tempfile.NamedTemporaryFile(prefix="q_", suffix=".json", delete=False)
    embed_path = Path(tf.name)
    tf.close()

    # create embedding
    proc = run([sys.executable, str(embed_py), "--text", args.query, "--output", str(embed_path), "--model", args.embed_model], cwd=str(project_root))
    if proc.returncode != 0:
        # silent fallback for production
        print("I'm not sure, you need to refer your teacher")
        sys.exit(0)

    # allow small delay for file system
    for _ in range(10):
        if embed_path.exists() and embed_path.stat().st_size > 10:
            break
        time.sleep(0.02)

    # run rag_answer
    proc = run([sys.executable, str(rag_answer_py),
                "--bundle", args.bundle,
                "--embed", str(embed_path),
                "--query", args.query,
                "--k", str(args.k),
                "--model", args.model],
               cwd=str(project_root))

    parsed = _find_json_blob(proc.stdout or "")
    if isinstance(parsed, dict) and parsed.get("status") == "ok" and parsed.get("answer"):
        # print only the plain answer text
        print(parsed["answer"].strip())
        sys.exit(0)

    # fallback message when out-of-context or invalid response
    print("I'm not sure, you need to refer your teacher")
    sys.exit(0)

if __name__ == "__main__":
    main()