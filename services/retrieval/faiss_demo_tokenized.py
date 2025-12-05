"""FAISS demo for tokenized chunks directory.
"""
import os
import sys
import pickle
import numpy as np

try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

CHUNKS_JSONL = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'chunks', 'tokenized', 'chapter_1.jsonl'))
EMB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'chunks', 'tokenized', 'embeddings.npy'))
IDMAP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'chunks', 'tokenized', 'id_map.pkl'))


def load_chunks(jsonl_path):
    chunks = []
    import json
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def build_index(embs: np.ndarray):
    if faiss is None:
        raise RuntimeError('faiss not installed')
    d = embs.shape[1]
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    return index


def encode_query(query: str):
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers not installed')
    model = SentenceTransformer('intfloat/multilingual-e5-base')
    q = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q)
    return q


def demo(query='What happens when magnesium is burned in air?', k=5):
    embs = np.load(EMB_PATH)
    with open(IDMAP_PATH, 'rb') as f:
        idmap = pickle.load(f)
    ids = idmap.get('ids', [])
    chunks = load_chunks(CHUNKS_JSONL)

    index = build_index(embs.copy())
    q = encode_query(query)
    D, I = index.search(q, k)
    for score, idx in zip(D[0], I[0]):
        print(f'score={score:.4f} id={ids[idx]} section={chunks[idx].get("section_heading")}')
        print(chunks[idx].get('text'))
        print('-'*60)


if __name__ == '__main__':
    if not os.path.exists(EMB_PATH):
        print('Embeddings missing:', EMB_PATH)
        sys.exit(1)
    demo()
