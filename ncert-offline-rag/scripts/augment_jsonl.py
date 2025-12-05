"""
ncert-offline-rag/scripts/augment_jsonl.py

Add missing required fields to a JSONL of chunks.
- Ensures 'textbook' exists (default "unknown")
- Ensures 'tokens' exists (computed as whitespace token count)
- Writes corrected JSONL to output path

Usage:
python ncert-offline-rag/scripts/augment_jsonl.py \
  --input ncert-offline-rag/data/chapter_1.jsonl \
  --output ncert-offline-rag/data_fixed/chapter_1.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REQUIRED = {"text", "class", "subject", "chapter", "language", "textbook", "tokens"}

def _ensure(obj: Dict[str, Any]) -> Dict[str, Any]:
    # ensure text exists
    if "text" not in obj:
        raise RuntimeError("Missing 'text' field")
    # defaults
    if "textbook" not in obj:
        obj["textbook"] = "unknown"
    if "tokens" not in obj:
        # conservative token estimate: whitespace split
        obj["tokens"] = int(len(str(obj.get("text", "")).split()))
    return obj

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input JSONL")
    p.add_argument("--output", required=True, help="Output JSONL (fixed)")
    args = p.parse_args(argv)
    inp = Path(args.input)
    outp = Path(args.output)
    if not inp.exists():
        print("ERROR: input not found:", inp, file=sys.stderr)
        return 2
    outp.parent.mkdir(parents=True, exist_ok=True)
    with inp.open("r", encoding="utf-8") as inf, outp.open("w", encoding="utf-8") as outf:
        for i, line in enumerate(inf, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"ERROR: invalid JSON line {i}: {e}", file=sys.stderr)
                return 2
            try:
                obj_fixed = _ensure(obj)
            except Exception as e:
                print(f"ERROR on line {i}: {e}", file=sys.stderr)
                return 2
            outf.write(json.dumps(obj_fixed, ensure_ascii=False) + "\n")
    print(str(outp))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())