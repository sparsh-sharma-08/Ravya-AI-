from __future__ import annotations
import argparse
import subprocess
import sys
import json
import time
from pathlib import Path
import tempfile

def run(cmd, cwd=None):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=None)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--bundle", default="./bundle/class_8_science_en")
    p.add_argument("--model", default="2b")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--embed-model", default="all-mpnet-base-v2")
    p.add_argument("--retries", type=int, default=3)
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
        print(proc.stderr.strip(), file=sys.stderr)
        sys.exit(proc.returncode)

    last_stdout = ""
    last_stderr = ""
    for attempt in range(1, args.retries + 1):
        proc = run([sys.executable, str(rag_answer_py),
                    "--bundle", args.bundle,
                    "--embed", str(embed_path),
                    "--query", args.query,
                    "--k", str(args.k),
                    "--model", args.model],
                   cwd=str(project_root))
        last_stdout = proc.stdout or ""
        last_stderr = proc.stderr or ""
        try:
            j = json.loads(last_stdout)
        except Exception:
            j = None

        if isinstance(j, dict) and j.get("status") == "ok":
            # Print only the returned output and the sources used
            # First print the JSON answer (as returned)
            print(json.dumps(j, ensure_ascii=False))
            # Then print the sources array on its own line for easy parsing
            print(json.dumps(j.get("sources", []), ensure_ascii=False))
            sys.exit(0)

        if attempt < args.retries:
            time.sleep(0.5 * attempt)

    # persistent failure: surface last stdout/stderr minimally
    if last_stdout:
        sys.stdout.write(last_stdout)
    if last_stderr:
        sys.stderr.write(last_stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()