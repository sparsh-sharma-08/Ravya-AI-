from __future__ import annotations
import argparse
import json
from src.rag.rag_answer import get_rag_answer

def main():
    parser = argparse.ArgumentParser(prog="ask", description="Query the RAG system")
    parser.add_argument("--query", "-q", required=True, help="User query")
    parser.add_argument("--k", type=int, default=5, help="number of chunks to retrieve")
    parser.add_argument("--model", type=str, default="2b", help="model id/size to use")
    parser.add_argument(
        "--mode",
        choices=["student", "teacher"],
        default="student",
        help="Response mode: 'student' for short direct answers (default), 'teacher' for long structured notes"
    )
    args = parser.parse_args()

    result = get_rag_answer(args.query, k=args.k, model=args.model, mode=args.mode)

    if args.mode == "teacher":
        if isinstance(result, dict) and result.get("status") == "ok":
            content = result.get("content", "")
            sources = result.get("sources", [])
            print(content)
            if sources:
                print("\nSources:", ", ".join(sources))
        else:
            print("I'm not confident, please refer your teacher or textbook.")
    else:
        # student mode: keep existing behavior (short answer)
        if isinstance(result, dict) and result.get("status") == "ok":
            print(result.get("answer") or result.get("content") or "")
        else:
            print("I'm not sure, please refer your teacher or textbook.")

if __name__ == "__main__":
    main()

# Usage examples:
# student:
# python src/rag/ask.py --query "What is force?" --mode student
#
# teacher:
# python src/rag/ask.py --query "Prepare lecture notes on Newton's laws for class 9" --mode teacher --k 5 --model 2b