#!/usr/bin/env python3
import argparse
import json
import tempfile
import os
from pathlib import Path

from src.ingestion.ingest_jsonl import ingest_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--class", dest="cls", required=False)
    p.add_argument("--subject", required=False)
    p.add_argument("--language", required=False)
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(args.input)

    tmp_path = None
    try:
        # Pre-process JSONL: coerce chapter fields to int when possible.
        # Write a temporary JSONL file that ingestion will consume.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".jsonl") as tf:
            tmp_path = tf.name
            for lineno, raw in enumerate(input_path.open("r", encoding="utf-8"), start=1):
                raw_strip = raw.strip()
                if not raw_strip:
                    continue
                obj = json.loads(raw_strip)
                # If chapter exists and is not int, attempt coercion.
                if "chapter" in obj and not isinstance(obj["chapter"], int):
                    try:
                        obj["chapter"] = int(obj["chapter"])
                    except Exception:
                        raise ValueError(f"Field 'chapter' coercion failed at line {lineno}: {obj.get('chapter')!r}")
                tf.write(json.dumps(obj, ensure_ascii=False) + "\n")

        default_cls = int(args.cls) if args.cls is not None else None
        print("Starting ingestion pipeline")
        chunks = ingest_jsonl(tmp_path,
                              default_class=default_cls,
                              default_subject=args.subject,
                              default_language=args.language)
        print(json.dumps({"ingested": len(list(chunks))}))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    main()