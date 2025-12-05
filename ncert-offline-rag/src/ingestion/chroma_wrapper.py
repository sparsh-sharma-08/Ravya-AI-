from chromadb import Client
from chromadb.config import Settings

class ChromaWrapper:
    def __init__(self, collection_name):
        self.client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="chroma_db"))
        self.collection = self.client.create_collection(name=collection_name)

    def add_chunks(self, chunks):
        ids = [chunk['hash'] for chunk in chunks]
        texts = [chunk['text'] for chunk in chunks]
        metadata = [{k: chunk[k] for k in chunk if k != 'text'} for chunk in chunks]
        self.collection.add(documents=texts, metadatas=metadata, ids=ids)

    def query(self, query_embedding, n_results=5):
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results

    def clear_collection(self):
        self.collection.delete()