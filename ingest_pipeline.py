"""
Usage:
  python ingest_pipeline.py \
    --input data/chapters/chapter_3.jsonl \
    --class 8 \
    --subject science \
    --language en

Environment assumptions:
  - Run on laptop/cloud only (not Raspberry Pi).
  - Chroma REST API reachable at http://localhost:8000
  - Python venv active and packages installed:
      pip install sentence-transformers requests numpy faiss-cpu

Embedding model:
  intfloat/e5-small-v2

Batch sizes:
  embed_batch_size = 256
  chroma_upsert_batch = 256

Expected input JSONL (one JSON object per line):
{
  "text": "...chunk...",
  "class": 8,
  "subject": "science",
  "chapter": 3,
  "language": "en",
  "textbook": "ncert",
  "tokens": 127
}

FAISS rationale:
  - Normalize embeddings -> IndexFlatIP for cosine-similarity compatible search.
  - IndexFlatIP keeps runtime simple and deterministic for Pi usage.

Behavior:
  - Validate input strictly; exit non-zero on any missing field or malformed line.
  - Use sentence-transformers to compute real semantic embeddings in batches.
  - Batch upsert to Chroma (no one-by-one inserts).
  - Run a smoke test query; if no results returned -> fail.
  - Export bundle/<collection_name>/ with required artifacts:
      chunks.jsonl, embeddings.bin, index.faiss, id_map.pkl, manifest.json, model.json, version.txt
  - On any failure during ingestion, attempt to remove the partial collection and exit non-zero.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import faiss
import numpy as np
import pickle
import requests
from sentence_transformers import SentenceTransformer

CHROMA_BASE = os.environ.get("CHROMA_URL", "http://localhost:8000")
TENANT = "default"
DATABASE = "default"
EMBEDDING_MODEL = "intfloat/e5-small-v2"
EMBED_BATCH = 256
UPSERT_BATCH = 256
SMOKE_K = 5
BUNDLE_VERSION = "2025.01.00"
HASH_STRATEGY = "md5(chunk_text)"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def md5hex(s: str) -> str:
    return hashlib.md5(s.encode("utf8")).hexdigest()


def collection_name(cls: int, subject: str, lang: str) -> str:
    return f"class_{int(cls)}_{subject.strip().lower().replace(' ', '_')}_{lang.strip().lower()}"


def validate_and_load_input(path: str, expected_class: int, expected_subject: str, expected_language: str):
    if not os.path.exists(path):
        eprint("Input file not found:", path)
        sys.exit(2)
    records = []
    line_no = 0
    with open(path, "r", encoding="utf8") as fh:
        for line in fh:
            line_no += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as ex:
                eprint(f"Invalid JSON on line {line_no}: {ex}")
                sys.exit(2)
            # required fields
            for f, t in [
                ("text", str),
                ("class", int),
                ("subject", str),
                ("chapter", int),
                ("language", str),
                ("textbook", str),
                ("tokens", int),
            ]:
                if f not in obj:
                    eprint(f"Missing required field '{f}' at line {line_no}")
                    sys.exit(2)
                if not isinstance(obj[f], t):
                    # allow ints for numeric types that may be parsed as other numeric types
                    if t is int and isinstance(obj[f], (int,)):
                        pass
                    else:
                        eprint(f"Field '{f}' has wrong type at line {line_no}: expected {t.__name__}")
                        sys.exit(2)
            # ensure provided CLI params match records
            if int(obj["class"]) != int(expected_class):
                eprint(f"Record class mismatch at line {line_no}: expected {expected_class}, got {obj['class']}")
                sys.exit(2)
            if str(obj["subject"]).strip().lower() != str(expected_subject).strip().lower():
                eprint(f"Record subject mismatch at line {line_no}: expected {expected_subject}, got {obj['subject']}")
                sys.exit(2)
            if str(obj["language"]).strip().lower() != str(expected_language).strip().lower():
                eprint(f"Record language mismatch at line {line_no}: expected {expected_language}, got {obj['language']}")
                sys.exit(2)
            text = str(obj["text"]).strip()
            h = md5hex(text)
            cid = f"{int(obj['class'])}_{str(obj['subject']).strip().lower()}_{int(obj['chapter'])}_{h}"
            meta = {
                "id": cid,
                "class": int(obj["class"]),
                "subject": str(obj["subject"]).strip().lower(),
                "chapter": int(obj["chapter"]),
                "language": str(obj["language"]).strip().lower(),
                "textbook": str(obj["textbook"]).strip().lower(),
                "tokens": int(obj["tokens"]),
                "hash": h,
            }
            records.append({"id": cid, "text": text, "metadata": meta})
    if not records:
        eprint("No valid records found in input.")
        sys.exit(2)
    return records


def ensure_collection_exists(coll_name: str):
    url = f"{CHROMA_BASE}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections"
    body = {"name": coll_name, "get_or_create": True}
    resp = requests.post(url, json=body, timeout=30)
    if resp.status_code not in (200, 201):
        eprint("Failed to create/get collection:", resp.status_code, resp.text)
        raise RuntimeError("chroma create collection failed")
    return resp.json()


def delete_collection(coll_name: str):
    url = f"{CHROMA_BASE}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections/{coll_name}"
    try:
        resp = requests.delete(url, timeout=30)
        # Accept 200/204/404 as okay (404 means already gone)
        if resp.status_code not in (200, 204, 404):
            eprint("Failed to delete collection:", resp.status_code, resp.text)
    except Exception as ex:
        eprint("Error deleting collection:", ex)


def upsert_batch_to_chroma(coll_name: str, ids, embeddings, metadatas, documents):
    url = f"{CHROMA_BASE}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections/{coll_name}/add"
    payload = {
        "ids": ids,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "documents": documents,
    }
    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code not in (200, 201):
        eprint("Chroma upsert failed:", resp.status_code, resp.text)
        raise RuntimeError("chroma upsert failed")


def compute_embeddings(model, texts, batch_size=EMBED_BATCH):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        # ensure float32 numpy
        embs = np.asarray(embs, dtype=np.float32)
        all_embs.append(embs)
    return np.vstack(all_embs)


def smoke_test_query(coll_name: str, model, sample_query="What is photosynthesis?"):
    q_emb = model.encode([sample_query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
    url = f"{CHROMA_BASE}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections/{coll_name}/query"
    body = {
        "query_embeddings": [q_emb.tolist()],
        "n_results": SMOKE_K,
        "include": ["distances", "documents", "metadatas"],
    }
    resp = requests.post(url, json=body, timeout=30)
    if resp.status_code != 200:
        eprint("Smoke test query failed:", resp.status_code, resp.text)
        raise RuntimeError("smoke test failed")
    data = resp.json()
    # Response shape (per OpenAPI) contains 'ids', 'documents', 'distances' etc as lists-of-lists
    ids = data.get("ids", [[]])[0] if isinstance(data.get("ids"), list) else data.get("ids", [])
    distances = data.get("distances", [[]])[0] if isinstance(data.get("distances"), list) else data.get("distances", [])
    if not ids:
        eprint("Smoke test returned no results.")
        raise RuntimeError("smoke test returned no results")
    # distances may be similarity scores (IP) between -1 and 1; log top1
    top1 = distances[0] if distances else None
    return {"ids": ids, "distances": distances, "top1": top1}


def export_bundle(out_dir: str, coll_name: str, ids, embeddings: np.ndarray, metadatas, documents, model_name: str):
    os.makedirs(out_dir, exist_ok=True)
    chunks_path = os.path.join(out_dir, "chunks.jsonl")
    with open(chunks_path, "w", encoding="utf8") as fh:
        for iid, meta, doc in zip(ids, metadatas, documents):
            entry = {"metadata": meta, "text": doc}
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # embeddings.bin contiguous float32
    emb_file = os.path.join(out_dir, "embeddings.bin")
    embeddings.astype(np.float32).tofile(emb_file)
    # id_map.pkl
    id_map_file = os.path.join(out_dir, "id_map.pkl")
    with open(id_map_file, "wb") as fh:
        pickle.dump(list(ids), fh)
    # build FAISS index with normalized vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = embeddings / norms
    dim = emb_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_norm.astype(np.float32))
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    # model.json
    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf8") as fh:
        json.dump({"name": model_name, "dim": int(dim)}, fh, indent=2)
    # manifest.json
    manifest = {
        "class": metadatas[0].get("class"),
        "subject": metadatas[0].get("subject"),
        "chapter": metadatas[0].get("chapter"),
        "language": metadatas[0].get("language"),
        "textbook": metadatas[0].get("textbook"),
        "embedding_model": model_name,
        "embedding_dim": int(dim),
        "chunk_count": len(ids),
        "chunk_strategy": "semantic + page anchors",
        "version": BUNDLE_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "hash_strategy": HASH_STRATEGY,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf8") as fh:
        json.dump(manifest, fh, indent=2)
    # version.txt
    with open(os.path.join(out_dir, "version.txt"), "w", encoding="utf8") as fh:
        fh.write(BUNDLE_VERSION + "\n")
    return out_dir


def run_pipeline(args):
    print("Starting ingestion pipeline")
    records = validate_and_load_input(args.input, args.class_, args.subject, args.language)
    coll = collection_name(args.class_, args.subject, args.language)
    print(f"Collection name: {coll}; records: {len(records)}")
    # load model
    print("Loading embedding model:", EMBEDDING_MODEL)
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as ex:
        eprint("Failed to load embedding model:", ex)
        sys.exit(2)
    emb_dim = model.get_sentence_embedding_dimension()
    print("Embedding dimension:", emb_dim)
    # prepare arrays
    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]
    metadatas = [r["metadata"] for r in records]
    documents = texts  # store the text as 'documents'
    # create collection
    try:
        ensure_collection_exists(coll)
    except Exception as ex:
        eprint("Failed to ensure collection exists:", ex)
        sys.exit(2)
    # compute embeddings in batches
    print("Computing embeddings (batch size {})...".format(EMBED_BATCH))
    try:
        embeddings = compute_embeddings(model, texts, batch_size=EMBED_BATCH)
    except Exception as ex:
        eprint("Embedding computation failed:", ex)
        delete_collection(coll)
        sys.exit(2)
    if embeddings.shape[0] != len(texts):
        eprint("Embedding count mismatch")
        delete_collection(coll)
        sys.exit(2)
    # upsert to chroma in batches
    print("Upserting to Chroma in batches of", UPSERT_BATCH)
    try:
        for i in range(0, len(ids), UPSERT_BATCH):
            batch_ids = ids[i : i + UPSERT_BATCH]
            batch_emb = embeddings[i : i + UPSERT_BATCH]
            batch_meta = metadatas[i : i + UPSERT_BATCH]
            batch_docs = documents[i : i + UPSERT_BATCH]
            # convert embeddings to lists
            emb_lists = [list(map(float, v)) for v in batch_emb]
            upsert_batch_to_chroma(coll, batch_ids, emb_lists, batch_meta, batch_docs)
            print(f"Upserted items {i}-{i+len(batch_ids)-1}")
    except Exception as ex:
        eprint("Upsert to Chroma failed:", ex)
        delete_collection(coll)
        sys.exit(2)
    # small delay to ensure Chroma persisted
    time.sleep(1.0)
    # smoke test
    print("Running smoke test query...")
    try:
        result = smoke_test_query(coll, model)
        print("Smoke test OK. Top1 similarity (IP / cosine):", result.get("top1"))
    except Exception as ex:
        eprint("Smoke test failed:", ex)
        delete_collection(coll)
        sys.exit(2)
    # Export bundle
    out_dir = os.path.join("bundle", coll)
    print("Exporting bundle to", out_dir)
    try:
        export_bundle(out_dir, coll, ids, embeddings, metadatas, documents, EMBEDDING_MODEL)
    except Exception as ex:
        eprint("Bundle export failed:", ex)
        delete_collection(coll)
        sys.exit(2)
    print("Bundle exported successfully:", out_dir)
    print("Done.")
    return out_dir


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end ingestion -> Chroma -> FAISS bundle exporter")
    p.add_argument("--input", required=True, help="Path to input JSONL")
    p.add_argument("--class", dest="class_", required=True, type=int, help="Integer class/grade")
    p.add_argument("--subject", required=True, help="Subject (single word, e.g. science)")
    p.add_argument("--language", required=True, help="Language code (en, hi, bn...)")
    return p.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    try:
        bundle_path = run_pipeline(ARGS)
        print(bundle_path)
        sys.exit(0)
    except KeyboardInterrupt:
        eprint("Interrupted by user")
        sys.exit(2)
    except Exception as exc:
        eprint("Unhandled error:", exc)
        sys.exit(2)
# filepath: /Users/sparsh/Documents/Coding/Ravya/ingest_pipeline.py
"""
Usage:
  python ingest_pipeline.py \
    --input data/chapters/chapter_3.jsonl \
    --class 8 \
    --subject science \
    --language en

Environment assumptions:
  - Run on laptop/cloud only (not Raspberry Pi).
  - Chroma REST API reachable at http://localhost:8000
  - Python venv active and packages installed:
      pip install sentence-transformers requests numpy faiss-cpu

Embedding model:
  intfloat/e5-small-v2

Batch sizes:
  embed_batch_size = 256
  chroma_upsert_batch = 256

Expected input JSONL (one JSON object per line):
{
  "text": "...chunk...",
  "class": 8,
  "subject": "science",
  "chapter": 3,
  "language": "en",
  "textbook": "ncert",
  "tokens": 127
}

FAISS rationale:
  - Normalize -> IndexFlatIP for cosine-similarity compatible search.
  - IndexFlatIP keeps runtime simple and deterministic for Pi usage.

Behavior:
  - Validate input strictly; exit non-zero on any missing field or malformed line.
  - Use sentence-transformers to compute real semantic embeddings in batches.
  - Batch upsert to Chroma (no one-by-one inserts).
  - Run a smoke test query; if no results returned -> fail.
  - Export bundle/<collection_name>/ with required artifacts:
      chunks.jsonl, embeddings.bin, index.faiss, id_map.pkl, manifest.json, model.json, version.txt
  - On any failure during ingestion, attempt to remove the partial collection and exit non-zero.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import faiss
import numpy as np
import pickle
import requests
from sentence_transformers import SentenceTransformer

CHROMA_BASE = os.environ.get("CHROMA_URL", "http://localhost:8000")
TENANT = "default"
DATABASE = "default"
EMBEDDING_MODEL = "intfloat/e5-small-v2"
EMBED_BATCH = 256
UPSERT_BATCH = 256
SMOKE_K = 5
BUNDLE_VERSION = "2025.01.00"
HASH_STRATEGY = "md5(chunk_text)"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def md5hex(s: str) -> str:
    return hashlib.md5(s.encode("utf8")).hexdigest()


def collection_name(cls: int, subject: str, lang: str) -> str:
    return f"class_{int(cls)}_{subject.strip().lower().replace(' ', '_')}_{lang.strip().lower()}"


def validate_and_load_input(path: str, expected_class: int, expected_subject: str, expected_language: str):
    if not os.path.exists(path):
        eprint("Input file not found:", path)
        sys.exit(2)
    records = []
    line_no = 0
    with open(path, "r", encoding="utf8") as fh:
        for line in fh:
            line_no += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as ex:
                eprint(f"Invalid JSON on line {line_no}: {ex}")
                sys.exit(2)
            # required fields
            for f, t in [
                ("text", str),
                ("class", int),
                ("subject", str),
                ("chapter", int),
                ("language", str),
                ("textbook", str),
                ("tokens", int),
            ]:
                if f not in obj:
                    eprint(f"Missing required field '{f}' at line {line_no}")
                    sys.exit(2)
                if not isinstance(obj[f], t):
                    # allow ints for numeric types that may be parsed as other numeric types
                    if t is int and isinstance(obj[f], (int,)):
                        pass
                    else:
                        eprint(f"Field '{f}' has wrong type at line {line_no}: expected {t.__name__}")
                        sys.exit(2)
            # ensure provided CLI params match records
            if int(obj["class"]) != int(expected_class):
                eprint(f"Record class mismatch at line {line_no}: expected {expected_class}, got {obj['class']}")
                sys.exit(2)
            if str(obj["subject"]).strip().lower() != str(expected_subject).strip().lower():
                eprint(f"Record subject mismatch at line {line_no}: expected {expected_subject}, got {obj['subject']}")
                sys.exit(2)
            if str(obj["language"]).strip().lower() != str(expected_language).strip().lower():
                eprint(f"Record language mismatch at line {line_no}: expected {expected_language}, got {obj['language']}")
                sys.exit(2)
            text = str(obj["text"]).strip()
            h = md5hex(text)
            cid = f"{int(obj['class'])}_{str(obj['subject']).strip().lower()}_{int(obj['chapter'])}_{h}"
            meta = {
                "id": cid,
                "class": int(obj["class"]),
                "subject": str(obj["subject"]).strip().lower(),
                "chapter": int(obj["chapter"]),
                "language": str(obj["language"]).strip().lower(),
                "textbook": str(obj["textbook"]).strip().lower(),
                "tokens": int(obj["tokens"]),
                "hash": h,
            }
            records.append({"id": cid, "text": text, "metadata": meta})
    if not records:
        eprint("No valid records found in input.")
        sys.exit(2)
    return records


def ensure_collection_exists(coll_name: str):
    url = f"{CHROMA_BASE}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections"
    body = {"name": coll_name, "get_or_create": True}
    resp = requests.post(url, json=body, timeout=30)
    if resp.status_code not in (200, 201):
        eprint("Failed to create/get collection:", resp.status_code, resp.text)
        raise RuntimeError("chroma create collection failed")
    return resp.json()


def delete_collection(coll_name: str):
    url = f"{CHROMA_BASE}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections/{coll_name}"
    try:
        resp = requests.delete(url, timeout=30)
        # Accept 200/204/404 as okay (404 means already gone)
        if resp.status_code not in (200, 204, 404):
            eprint("Failed to delete collection:", resp.status_code, resp.text)
    except Exception as ex:
        eprint("Error deleting collection:", ex)


def upsert_batch_to_chroma(coll_name: str, ids, embeddings, metadatas, documents):
    url = f"{CHROMA_BASE}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections/{coll_name}/add"
    payload = {
        "ids": ids,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "documents": documents,
    }
    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code not in (200, 201):
        eprint("Chroma upsert failed:", resp.status_code, resp.text)
        raise RuntimeError("chroma upsert failed")


def compute_embeddings(model, texts, batch_size=EMBED_BATCH):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        # ensure float32 numpy
        embs = np.asarray(embs, dtype=np.float32)
        all_embs.append(embs)
    return np.vstack(all_embs)


def smoke_test_query(coll_name: str, model, sample_query="What is photosynthesis?"):
    q_emb = model.encode([sample_query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
    url = f"{CHROMA_BASE}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections/{coll_name}/query"
    body = {
        "query_embeddings": [q_emb.tolist()],
        "n_results": SMOKE_K,
        "include": ["distances", "documents", "metadatas"],
    }
    resp = requests.post(url, json=body, timeout=30)
    if resp.status_code != 200:
        eprint("Smoke test query failed:", resp.status_code, resp.text)
        raise RuntimeError("smoke test failed")
    data = resp.json()
    # Response shape (per OpenAPI) contains 'ids', 'documents', 'distances' etc as lists-of-lists
    ids = data.get("ids", [[]])[0] if isinstance(data.get("ids"), list) else data.get("ids", [])
    distances = data.get("distances", [[]])[0] if isinstance(data.get("distances"), list) else data.get("distances", [])
    if not ids:
        eprint("Smoke test returned no results.")
        raise RuntimeError("smoke test returned no results")
    # distances may be similarity scores (IP) between -1 and 1; log top1
    top1 = distances[0] if distances else None
    return {"ids": ids, "distances": distances, "top1": top1}


def export_bundle(out_dir: str, coll_name: str, ids, embeddings: np.ndarray, metadatas, documents, model_name: str):
    os.makedirs(out_dir, exist_ok=True)
    chunks_path = os.path.join(out_dir, "chunks.jsonl")
    with open(chunks_path, "w", encoding="utf8") as fh:
        for iid, meta, doc in zip(ids, metadatas, documents):
            entry = {"metadata": meta, "text": doc}
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # embeddings.bin contiguous float32
    emb_file = os.path.join(out_dir, "embeddings.bin")
    embeddings.astype(np.float32).tofile(emb_file)
    # id_map.pkl
    id_map_file = os.path.join(out_dir, "id_map.pkl")
    with open(id_map_file, "wb") as fh:
        pickle.dump(list(ids), fh)
    # build FAISS index with normalized vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = embeddings / norms
    dim = emb_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_norm.astype(np.float32))
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    # model.json
    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf8") as fh:
        json.dump({"name": model_name, "dim": int(dim)}, fh, indent=2)
    # manifest.json
    manifest = {
        "class": metadatas[0].get("class"),
        "subject": metadatas[0].get("subject"),
        "chapter": metadatas[0].get("chapter"),
        "language": metadatas[0].get("language"),
        "textbook": metadatas[0].get("textbook"),
        "embedding_model": model_name,
        "embedding_dim": int(dim),
        "chunk_count": len(ids),
        "chunk_strategy": "semantic + page anchors",
        "version": BUNDLE_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "hash_strategy": HASH_STRATEGY,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf8") as fh:
        json.dump(manifest, fh, indent=2)
    # version.txt
    with open(os.path.join(out_dir, "version.txt"), "w", encoding="utf8") as fh:
        fh.write(BUNDLE_VERSION + "\n")
    return out_dir


def run_pipeline(args):
    print("Starting ingestion pipeline")
    records = validate_and_load_input(args.input, args.class_, args.subject, args.language)
    coll = collection_name(args.class_, args.subject, args.language)
    print(f"Collection name: {coll}; records: {len(records)}")
    # load model
    print("Loading embedding model:", EMBEDDING_MODEL)
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as ex:
        eprint("Failed to load embedding model:", ex)
        sys.exit(2)
    emb_dim = model.get_sentence_embedding_dimension()
    print("Embedding dimension:", emb_dim)
    # prepare arrays
    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]
    metadatas = [r["metadata"] for r in records]
    documents = texts  # store the text as 'documents'
    # create collection
    try:
        ensure_collection_exists(coll)
    except Exception as ex:
        eprint("Failed to ensure collection exists:", ex)
        sys.exit(2)
    # compute embeddings in batches
    print("Computing embeddings (batch size {})...".format(EMBED_BATCH))
    try:
        embeddings = compute_embeddings(model, texts, batch_size=EMBED_BATCH)
    except Exception as ex:
        eprint("Embedding computation failed:", ex)
        delete_collection(coll)
        sys.exit(2)
    if embeddings.shape[0] != len(texts):
        eprint("Embedding count mismatch")
        delete_collection(coll)
        sys.exit(2)
    # upsert to chroma in batches
    print("Upserting to Chroma in batches of", UPSERT_BATCH)
    try:
        for i in range(0, len(ids), UPSERT_BATCH):
            batch_ids = ids[i : i + UPSERT_BATCH]
            batch_emb = embeddings[i : i + UPSERT_BATCH]
            batch_meta = metadatas[i : i + UPSERT_BATCH]
            batch_docs = documents[i : i + UPSERT_BATCH]
            # convert embeddings to lists
            emb_lists = [list(map(float, v)) for v in batch_emb]
            upsert_batch_to_chroma(coll, batch_ids, emb_lists, batch_meta, batch_docs)
            print(f"Upserted items {i}-{i+len(batch_ids)-1}")
    except Exception as ex:
        eprint("Upsert to Chroma failed:", ex)
        delete_collection(coll)
        sys.exit(2)
    # small delay to ensure Chroma persisted
    time.sleep(1.0)
    # smoke test
    print("Running smoke test query...")
    try:
        result = smoke_test_query(coll, model)
        print("Smoke test OK. Top1 similarity (IP / cosine):", result.get("top1"))
    except Exception as ex:
        eprint("Smoke test failed:", ex)
        delete_collection(coll)
        sys.exit(2)
    # Export bundle
    out_dir = os.path.join("bundle", coll)
    print("Exporting bundle to", out_dir)
    try:
        export_bundle(out_dir, coll, ids, embeddings, metadatas, documents, EMBEDDING_MODEL)
    except Exception as ex:
        eprint("Bundle export failed:", ex)
        delete_collection(coll)
        sys.exit(2)
    print("Bundle exported successfully:", out_dir)
    print("Done.")
    return out_dir


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end ingestion -> Chroma -> FAISS bundle exporter")
    p.add_argument("--input", required=True, help="Path to input JSONL")
    p.add_argument("--class", dest="class_", required=True, type=int, help="Integer class/grade")
    p.add_argument("--subject", required=True, help="Subject (single word, e.g. science)")
    p.add_argument("--language", required=True, help="Language code (en, hi, bn...)")
    return p.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    try:
        bundle_path = run_pipeline(ARGS)
        print(bundle_path)
        sys.exit(0)
    except KeyboardInterrupt:
        eprint("Interrupted by user")
        sys.exit(2)
    except Exception as exc:
        eprint("Unhandled error:", exc)
        sys.exit(2)