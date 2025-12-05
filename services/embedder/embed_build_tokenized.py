"""Compute embeddings for tokenized chunks directory and save embeddings.npy + id_map.pkl.
"""
import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def load_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def build(path_jsonl):
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers not installed')
    chunks = load_jsonl(path_jsonl)
    texts = [c['text'] for c in chunks]
    ids = [c['id'] for c in chunks]
    model = SentenceTransformer('intfloat/multilingual-e5-base')
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    out_dir = os.path.dirname(path_jsonl)
    np.save(os.path.join(out_dir, 'embeddings.npy'), embs)
    with open(os.path.join(out_dir, 'id_map.pkl'), 'wb') as f:
        pickle.dump({'ids': ids}, f)

    print(f'Saved {embs.shape} embeddings and id map to {out_dir}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python embed_build_tokenized.py data/chunks/tokenized/chapter_1.jsonl')
        sys.exit(1)
    build(sys.argv[1])
