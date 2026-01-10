from __future__ import annotations
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

try:
    from src.rag.rag_answer import get_rag_answer_with_llm
except Exception:
    get_rag_answer_with_llm = None

def _find_embed_script() -> Optional[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "src" / "pi_runtime" / "embed_query.py",
        repo_root / "ncert-offline-rag" / "src" / "pi_runtime" / "embed_query.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def _generate_query_embed(embed_py: Path, question: str, timeout: int = 30) -> str:
    tf = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tf.close()
    out_path = tf.name
    cmd = [sys.executable, str(embed_py), "--text", question, "--output", out_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"embed_query failed: {proc.stderr.strip() or proc.stdout.strip()}")
    # tolerant validation / normalization
    try:
        raw = Path(out_path).read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid embed file: {e}")

    embed = None
    if isinstance(data, dict):
        if "embedding" in data and isinstance(data["embedding"], (list, tuple)):
            embed = list(data["embedding"])
        elif "embeddings" in data and isinstance(data["embeddings"], (list, tuple)):
            embed = list(data["embeddings"])
        elif "vector" in data and isinstance(data["vector"], (list, tuple)):
            embed = list(data["vector"])
    elif isinstance(data, (list, tuple)):
        embed = list(data)

    if embed is None:
        raise RuntimeError("Invalid embed file: missing embedding vector (accepted keys: embedding, embeddings, vector, or root list)")

    # rewrite normalized embed file to expected shape
    Path(out_path).write_text(json.dumps({"embedding": embed}, ensure_ascii=False), encoding="utf-8")
    return out_path

def interactive(bundle: str, model: str, k: int, embed_model: str, mode: str):
    embed_py = _find_embed_script()
    if embed_py is None:
        print("embed_query.py not found under repo; ensure src/pi_runtime/embed_query.py exists")
        return

    current_mode = mode
    print("Interactive RAG CLI — type a question, or '/quit' to exit.")
    print(f"Bundle: {bundle}  model: {model}  k: {k}  embed-model: {embed_model}")
    print(f"Starting mode: {current_mode}")
    print("Commands:")
    print("  /mode                Show current mode")
    print("  /mode student|teacher  Switch mode")
    print("  /quit                Exit")
    print("Enter questions directly; mode will remain until you switch it with /mode.")

    while True:
        try:
            user_input = input("\nQuestion (or command): ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.strip() == "":
            continue

        q = user_input.strip()

        # use internal retrieval + LLM glue (offline-safe) instead of spawning embed_query
        if get_rag_answer_with_llm is None:
            print("ERROR: internal RAG glue not available; ensure src.rag.rag_answer is importable")
            continue

        try:
            res = get_rag_answer_with_llm(bundle=bundle, question=q, mode=mode, k=k, embed_model=embed_model)
        except Exception as e:
            print("ERROR: get_rag_answer_with_llm failed:", e)
            continue

        status = res.get("status")
        answer = res.get("answer") or ""
        sources = res.get("sources") or []
        print(f"\n[status: {status}]\n")
        if answer:
            print(answer.strip())
        else:
            # show raw prompt / raw if no answer
            raw = res.get("raw","")
            print("No generated answer. Raw output:")
            print(raw)

        if sources:
            print("\nSources:")
            for s in sources:
                print(" -", s)

        if res.get("status") == "refer_teacher":
            print("\nTeacher mode returned 'refer_teacher'. To inspect raw model output run with RAG_DEBUG=1.")
            print("You can switch to 'teacher' mode and try again after adjusting prompts or model settings.")

def main():
    p = argparse.ArgumentParser(prog="interactive_cli")
    p.add_argument("--bundle", required=True, help="Path to bundle directory (e.g. bundles/class_8_science_en)")
    p.add_argument("--model", default="2b", help="Model variant")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--embed-model", default="all-mpnet-base-v2")
    p.add_argument("--mode", choices=["student", "teacher"], default="student")
    args = p.parse_args()
    # capture CLI args into locals so we don't reference `args` inside the REPL loop
    bundle = args.bundle
    k = args.k
    embed_model = args.embed_model
    mode = args.mode  # initial mode (can be changed interactively)

    interactive(bundle=args.bundle, model=args.model, k=args.k, embed_model=args.embed_model, mode=args.mode)

if __name__ == "__main__":
    main()