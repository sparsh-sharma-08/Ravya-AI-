"""Small test for the chunker module."""
import os
import sys
import json
# Ensure project root is on sys.path so `services` package can be imported when running the test directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from services.chunker.chunker import build_chunks_from_extracted, save_chunks_jsonl


def run_test():
    in_path = os.path.join(ROOT, "data", "extracted", "sample_book_ch01.json")
    out_path = os.path.join(ROOT, "data", "chunks", "sample_book_ch01.jsonl")
    with open(in_path, "r", encoding="utf-8") as f:
        extracted = json.load(f)
    chunks = build_chunks_from_extracted(extracted)
    save_chunks_jsonl(chunks, out_path)
    assert len(chunks) > 0, "No chunks produced"
    required_fields = ["id", "text", "source", "chapter", "language"]
    for c in chunks:
        for rf in required_fields:
            assert rf in c, f"Missing field {rf} in chunk"
    print("Chunker test passed. Wrote", len(chunks), "chunks to", out_path)


if __name__ == "__main__":
    run_test()
