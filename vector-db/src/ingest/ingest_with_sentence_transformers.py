#!/usr/bin/env python3
"""
Ingest pipeline (Laptop/Cloud only)
- Reads JSONL chunks file (one JSON per line) with required fields:
  text, class, subject, chapter, textbook, language, tokens (optional)
- Produces semantic embeddings using sentence-transformers (intfloat/e5-small-v2)
- Batches upserts to Chroma REST: /collections/{name}/add
- Collection naming: class_<grade>_<subject>_<lang>
- Batches: default 256 (configurable)
"""
import os
import sys
import json
import hashlib
import requests
from sentence_transformers import SentenceTransformer
from datetime import datetime
from typing import List

CHROMA_URL = os.environ.get("CHROMA_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "intfloat/e5-small-v2")
BATCH_SIZE = int(os.environ.get("INGEST_BATCH", "256"))

def md5hex(s: str) -> str:
    return hashlib.md5(s.encode("utf8")).hexdigest()

def collection_name_for(cls: int, subject: str, lang: str) -> str:
    return f"class_{int(cls)}_{subject.strip().lower().replace(' ','_')}_{lang.strip().lower()}"

def ensure_collection(coll: str):
    url = f"{CHROMA_URL}/collections"
    try:
        requests.post(url, json={"name": coll}, timeout=10).raise_for_status()
    except requests.HTTPError as e:
        # ignore already exists / other 4xx if server implements idempotency
        if e.response is not None and e.response.status_code // 100 == 4:
            return
        raise

def chunk_iter_from_file(path: str):
    with open(path, "r", encoding="utf8") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)

def upsert_batch(coll: str, ids: List[str], embeddings: List[List[float]], metadatas: List[dict], documents: List[str]):
    body = {
        "ids": ids,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "documents": documents
    }
    url = f"{CHROMA_URL}/collections/{coll}/add"
    resp = requests.post(url, json=body, timeout=60)
    resp.raise_for_status()

def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest_with_sentence_transformers.py /path/to/chunks.jsonl")
        sys.exit(2)
    infile = sys.argv[1]
    if not os.path.exists(infile):
        print("File not found:", infile); sys.exit(2)

    print("Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    print("Embedding dimension:", embedding_dim)

    # Read all chunks, group by collection
    groups = {}  # coll -> list of (id,text,metadata)
    count = 0
    for obj in chunk_iter_from_file(infile):
        text = str(obj.get("text", "")).strip()
        if not text:
            continue
        cls = int(obj.get("class"))
        subject = str(obj.get("subject"))
        chapter = int(obj.get("chapter", 0))
        textbook = str(obj.get("textbook", "ncert"))
        language = str(obj.get("language", "en"))
        tokens = int(obj.get("tokens", max(1, len(text.split()))))
        h = md5hex(text)
        cid = f"{cls}_{subject}_{chapter}_{h}"
        metadata = {
            "id": cid,
            "class": cls,
            "subject": subject,
            "chapter": chapter,
            "textbook": textbook,
            "language": language,
            "tokens": tokens,
            "hash": h,
        }
        coll = collection_name_for(cls, subject, language)
        groups.setdefault(coll, []).append((cid, text, metadata))
        count += 1

    print(f"Prepared {count} chunks across {len(groups)} collections")

    # For each collection, ensure exists and ingest in batches
    for coll, items in groups.items():
        print("Ensuring collection:", coll)
        ensure_collection(coll)
        # process in batches
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i:i+BATCH_SIZE]
            ids = [b[0] for b in batch]
            texts = [b[1] for b in batch]
            metadatas = [b[2] for b in batch]
            # compute embeddings (normalized)
            embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=False, normalize_embeddings=True)
            # convert numpy arrays to lists if needed
            embeddings = [list(map(float, e)) for e in embeddings]
            documents = texts  # store text in documents
            upsert_batch(coll, ids, embeddings, metadatas, documents)
            print(f"Upserted batch {i}-{i+len(batch)-1} into {coll}")
    print("Ingestion complete. Total chunks:", count)

if __name__ == "__main__":
    main()