"""Embedder: compute embeddings for chunks JSONL and save embeddings and id_map.

Default uses sentence-transformers `intfloat/multilingual-e5-base` if available locally.
Outputs: embeddings.npy and id_map.pkl next to chunks file.
"""
import os
import sys
import json
import pickle
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def load_chunks(jsonl_path):
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            chunks.append(json.loads(line))
    return chunks


def compute_embeddings(texts, model_name='intfloat/multilingual-e5-base'):
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers not installed in environment')
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embs


def build(jsonl_path: str):
    chunks = load_chunks(jsonl_path)
    texts = [c['text'] for c in chunks]
    ids = [c['id'] for c in chunks]
    embs = compute_embeddings(texts)

    out_dir = os.path.dirname(jsonl_path)
    emb_path = os.path.join(out_dir, 'embeddings.npy')
    id_map_path = os.path.join(out_dir, 'id_map.pkl')
    np.save(emb_path, embs)
    with open(id_map_path, 'wb') as f:
        pickle.dump({'ids': ids}, f)

    print(f"Saved embeddings to {emb_path} ({embs.shape}) and id_map to {id_map_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python embed_build.py data/chunks/chapter_1.jsonl')
        sys.exit(1)
    build(sys.argv[1])
