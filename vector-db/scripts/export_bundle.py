#!/usr/bin/env python3
"""
Export bundle for a collection:
Produces bundle/
  chunks.jsonl
  embeddings.bin        (float32 contiguous)
  id_map.pkl            (pickle: list of ids in same order as embeddings)
  manifest.json
  index.faiss           (faiss index file)
  model.json            (embedding model metadata)
"""
import os
import sys
import json
import requests
import numpy as np
import faiss
import pickle
import hashlib
from datetime import datetime

CHROMA_URL = os.environ.get("CHROMA_URL", "http://localhost:8000")
OUT_DIR = os.environ.get("BUNDLE_OUT", "bundle")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/e5-small-v2")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))
VERSION = os.environ.get("BUNDLE_VERSION", "2025.11.26")

def fetch_all(collection_name):
    url = f"{CHROMA_URL}/collections/{collection_name}/get"
    body = {"include": ["ids", "embeddings", "metadatas", "documents"]}
    resp = requests.post(url, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json().get("result", {})

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def build_bundle(collection_name):
    r = fetch_all(collection_name)
    ids = r.get("ids", [[]])[0]
    embeddings = r.get("embeddings", [[]])[0]
    metadatas = r.get("metadatas", [[]])[0]
    documents = r.get("documents", [[]])[0]
    if not ids:
        print("No items in collection", collection_name); return

    ids = list(ids)
    embeddings = np.array(embeddings, dtype=np.float32)
    assert embeddings.shape[1] == EMBEDDING_DIM, f"expected dim {EMBEDDING_DIM} got {embeddings.shape[1]}"

    os.makedirs(OUT_DIR, exist_ok=True)

    # Build chunks.jsonl with chunk bounds and token mapping (simple whitespace tokenization)
    chunks_file = os.path.join(OUT_DIR, "chunks.jsonl")
    with open(chunks_file, "w", encoding="utf8") as fh:
        for iid, meta, doc in zip(ids, metadatas, documents):
            text = doc or ""
            # compute char bounds and token range
            char_start = 0  # we don't have original doc-level offsets; per-chunk we set start=0,end=len
            char_end = len(text)
            tokens = text.split()
            token_count = len(tokens)
            token_start = 0
            token_end = token_count
            meta_out = dict(meta)
            meta_out["id"] = iid
            meta_out["char_start"] = char_start
            meta_out["char_end"] = char_end
            meta_out["token_start"] = token_start
            meta_out["token_end"] = token_end
            entry = {"metadata": meta_out, "text": text}
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # embeddings.bin contiguous float32
    emb_file = os.path.join(OUT_DIR, "embeddings.bin")
    embeddings.tofile(emb_file)

    # id_map.pkl
    idmap_file = os.path.join(OUT_DIR, "id_map.pkl")
    with open(idmap_file, "wb") as fh:
        pickle.dump(ids, fh)

    # build FAISS index (normalize to unit vectors -> IndexFlatIP)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = embeddings / norms
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(emb_norm)
    faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))

    # model.json
    with open(os.path.join(OUT_DIR, "model.json"), "w", encoding="utf8") as f:
        json.dump({"name": EMBEDDING_MODEL, "dim": EMBEDDING_DIM}, f, indent=2)

    manifest = {
        "class": metadatas[0].get("class"),
        "subject": metadatas[0].get("subject"),
        "chapter": metadatas[0].get("chapter"),
        "language": metadatas[0].get("language"),
        "board": metadatas[0].get("textbook", "ncert"),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_count": len(ids),
        "chunk_strategy": "semantic + page anchors",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "version": VERSION,
        "hash_strategy": "md5(chunk_text)",
    }
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2)

    # version.txt
    with open(os.path.join(OUT_DIR, "version.txt"), "w", encoding="utf8") as f:
        f.write(VERSION + "\n")

    # checksum for the chunks.jsonl (and overall)
    checksum = sha256_file(chunks_file)
    manifest["chunks_sha256"] = checksum
    with open(manifest_path, "w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2)

    print("Bundle exported to", OUT_DIR)
    print("manifest:", manifest_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_bundle.py <collection_name>")
        sys.exit(2)
    build_bundle(sys.argv[1])