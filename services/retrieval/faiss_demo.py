"""FAISS demo: build index from precomputed embeddings and run top-k retrieval for a sample query.

Usage:
  python faiss_demo.py

Requires: faiss-cpu, sentence-transformers, numpy
"""
import os
import sys
import pickle
import numpy as np
import json

try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


CHUNKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'chunks'))
CHUNKS_JSONL = os.path.join(CHUNKS_DIR, 'chapter_1.jsonl')
EMB_PATH = os.path.join(CHUNKS_DIR, 'embeddings.npy')
IDMAP_PATH = os.path.join(CHUNKS_DIR, 'id_map.pkl')


def load_chunks(jsonl_path):
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            chunks.append(json.loads(line))
    return chunks


def build_index(embs: np.ndarray):
    if faiss is None:
        raise RuntimeError('faiss not installed')
    d = embs.shape[1]
    # normalize for cosine similarity
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    return index


def encode_query(query: str, model_name='intfloat/multilingual-e5-base'):
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers not installed')
    model = SentenceTransformer(model_name)
    q = model.encode([query], convert_to_numpy=True)
    # normalize
    import numpy as _np
    _np.linalg.norm(q, axis=1, keepdims=True)
    faiss.normalize_L2(q)
    return q


def demo(query: str = 'What happens when magnesium is burned in air?', k: int = 5):
    print('Loading embeddings...')
    embs = np.load(EMB_PATH)
    with open(IDMAP_PATH, 'rb') as f:
        idmap = pickle.load(f)
    ids = idmap.get('ids', [])

    print('Building FAISS index...')
    index = build_index(embs.copy())

    print('Encoding query...')
    q = encode_query(query)

    D, I = index.search(q, k)
    D = D[0]
    I = I[0]

    # load chunk texts lazily
    texts = []
    import json
    with open(CHUNKS_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            texts.append(json.loads(line))

    print(f"Top {k} results for query: {query}\n")
    for score, idx in zip(D, I):
        cid = ids[idx]
        chunk = texts[idx]
        print(f"score={score:.4f} id={cid} section={chunk.get('section_heading')}")
        print(chunk.get('text'))
        print('-' * 60)


if __name__ == '__main__':
    # quick sanity checks
    if not os.path.exists(EMB_PATH):
        print('Embeddings not found at', EMB_PATH)
        sys.exit(1)
    if not os.path.exists(CHUNKS_JSONL):
        print('Chunks JSONL not found at', CHUNKS_JSONL)
        sys.exit(1)
    if not os.path.exists(IDMAP_PATH):
        print('ID map not found at', IDMAP_PATH)
        sys.exit(1)

    demo()
