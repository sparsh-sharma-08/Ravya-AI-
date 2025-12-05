"""Query Chroma collection and print top-k documents.
"""
import os
import json
from pathlib import Path

def query(collection_name: str = 'chapter_1', q: str = 'balance water equation', k: int = 5):
    try:
        import chromadb
    except Exception:
        raise RuntimeError('chromadb not installed')

    os.environ.setdefault('CHROMA_PERSIST_DIRECTORY', str(Path('data/chroma').absolute()))
    client = chromadb.Client()
    # debug: list existing collections
    cols = client.list_collections()
    col_names = [c.name for c in cols]
    if collection_name not in col_names:
        print('Available collections:', col_names)
        raise RuntimeError(f'Collection {collection_name} not found in Chroma')
    collection = client.get_collection(collection_name)
    res = collection.query(query_texts=[q], n_results=k)
    print('Query results:')
    for i, doc_id in enumerate(res['ids'][0]):
        text = res['documents'][0][i]
        metadata = res['metadatas'][0][i]
        print(f'[{i+1}] id={doc_id} score=NA section={metadata.get("section")}')
        print(text)
        print('-'*60)
    return res


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        q = sys.argv[1]
    else:
        q = 'Balance the chemical equation for water formation'
    query('chapter_1', q)
