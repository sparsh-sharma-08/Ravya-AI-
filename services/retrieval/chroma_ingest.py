"""Ingest tokenized chunks into a local Chroma DB.

Writes DB to data/chroma/<collection_name>
"""
import os
import json
from pathlib import Path

def ingest(jsonl_path: str, collection_name: str = 'chapter_1'):
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except Exception as e:
        raise RuntimeError('chromadb not installed') from e

    # Use persistent directory via environment variable or default
    persist_directory = str(Path('data/chroma').absolute())
    # set env var used by chromadb for persistence
    os.environ.setdefault('CHROMA_PERSIST_DIRECTORY', persist_directory)
    client = chromadb.Client()

    # create collection or get
    if collection_name in [c.name for c in client.list_collections()]:
        collection = client.get_collection(collection_name)
    else:
        collection = client.create_collection(name=collection_name)

    ids = []
    texts = []
    metadatas = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ids.append(obj['id'])
            texts.append(obj['text'])
            src = obj.get('source') or {}
            source_file = src.get('file') if isinstance(src, dict) else src
            source_page = src.get('page') if isinstance(src, dict) else None
            # Chroma metadata values must be primitives (no None). Convert None -> empty string
            metadatas.append({
                'section': obj.get('section_heading') or '',
                'chapter': obj.get('chapter') or '',
                'source_file': source_file or '',
                'source_page': str(source_page) if source_page is not None else ''
            })

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    print(f'Ingested {len(ids)} documents into Chroma collection {collection_name} at data/chroma')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python chroma_ingest.py data/chunks/tokenized/chapter_1.jsonl [collection_name]')
        sys.exit(1)
    ingest(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'chapter_1')
