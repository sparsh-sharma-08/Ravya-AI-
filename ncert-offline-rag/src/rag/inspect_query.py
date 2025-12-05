from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import tempfile

def run_cmd(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\nstderr:\n{r.stderr}")
    return r.stdout

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--bundle", default="./bundle/class_8_science_en")
    p.add_argument("--model", default="all-mpnet-base-v2")
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    # fix: embed_query lives under src/pi_runtime
    embed_py = Path(__file__).resolve().parents[2] / "src" / "pi_runtime" / "embed_query.py"
    retrieve_py = here / "retrieve.py"

    with tempfile.NamedTemporaryFile(prefix="q_", suffix=".json", delete=False) as tf:
        tmp_path = Path(tf.name)

    # create embedding using same python interpreter (venv)
    run_cmd([sys.executable, str(embed_py), "--text", args.text, "--output", str(tmp_path), "--model", args.model])

    out = run_cmd([sys.executable, str(retrieve_py), "--bundle", args.bundle, "--embed", str(tmp_path), "--k", str(args.k)])
    try:
        data = json.loads(out)
    except Exception:
        print("Failed to parse retrieve output:")
        print(out)
        sys.exit(1)

    if data.get("status") != "ok":
        print("retrieve status:", data)
        sys.exit(1)

    print(f"Top {len(data.get('chunks',[]))} results for: {args.text!r}\n")
    for c in data["chunks"]:
        print(f"rank {c['rank']}  score {c['score']:.4f}")
        print(f"  id: {c['id']}")
        text = c.get("text","").replace("\n"," ").strip()
        if len(text) > 300:
            text = text[:300] + "..."
        print(f"  snippet: {text}\n")

if __name__ == "__main__":
    main()