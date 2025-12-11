import argparse
import json
import tempfile
import sys
from pathlib import Path
from src.pi_runtime.retrieve import retrieve_chunks
from src.rag.rag_answer import _validate_query_embedding

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    # generate embed via embed_query.py
    tmp = Path(tempfile.NamedTemporaryFile(suffix=".json", delete=False).name)
    cmd = [sys.executable, str(Path(__file__).parent / "embed_query.py"), "--text", args.query, "--output", str(tmp)]
    try:
        import subprocess
        subprocess.run(cmd, check=True, capture_output=False, text=True, timeout=30)
    except Exception as e:
        print("embed generation failed:", e, file=sys.stderr)
        sys.exit(1)

    vec = _validate_query_embedding(str(tmp))
    if vec is None:
        print("refer_teacher")
        sys.exit(0)

    qdict = {"query": args.query, "embedding": vec}
    # try common id_map locations in bundle
    bundle_dir = Path(args.bundle)
    id_map_candidates = [
        bundle_dir / "id_map.json",
        bundle_dir / "id_map.jsonl",
        bundle_dir / "id_map.pkl",
        bundle_dir / "data" / "id_map.pkl",
    ]
    id_map = str(next((p for p in id_map_candidates if p.exists()), id_map_candidates[0]))
    res = retrieve_chunks(qdict, index_file="unused", id_map_file=id_map, k=args.k)
    print("retrieve ->", res)
    if isinstance(res, dict) and res.get("chunks"):
        print("OK")
    else:
        print("refer_teacher")

if __name__ == "__main__":
    main()