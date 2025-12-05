"""
ncert-offline-rag/scripts/embed_chunks.py

Laptop-only: encode texts from a JSONL of chunks into embeddings.npy using SentenceTransformer.

Usage:
python ncert-offline-rag/scripts/embed_chunks.py \
  --input ncert-offline-rag/data_fixed/chapter_1.jsonl \
  --output ncert-offline-rag/data_fixed/embeddings.npy \
  --model all-mpnet-base-v2 \
  --batch 32
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def load_texts(jsonl_path: Path) -> List[str]:
    texts: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj:
                raise RuntimeError(f"Line {i}: missing 'text' field")
            texts.append(obj["text"])
    if not texts:
        raise RuntimeError("No texts found in JSONL")
    return texts


def encode_texts(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. Run on laptop and install sentence-transformers")
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    if embs.dtype != np.float32:
        embs = embs.astype(np.float32)
    return embs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input chunks.jsonl")
    p.add_argument("--output", required=True, help="Output embeddings.npy")
    p.add_argument("--model", default="all-mpnet-base-v2", help="SentenceTransformer model (768-dim)")
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")
    outp.parent.mkdir(parents=True, exist_ok=True)

    texts = load_texts(inp)
    embs = encode_texts(texts, args.model, args.batch)
    np.save(str(outp), embs)
    print(f"wrote {outp} shape={embs.shape}")


if __name__ == "__main__":
    main()