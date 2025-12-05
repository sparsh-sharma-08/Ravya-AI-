import os
import json
import faiss
import numpy as np
import pickle
from datetime import datetime

def export_faiss_bundle(collection_name, index, embeddings, chunks, id_map):
    # Create the bundle directory if it doesn't exist
    bundle_dir = os.path.join('bundles', collection_name)
    os.makedirs(bundle_dir, exist_ok=True)

    # Export the FAISS index
    faiss.write_index(index, os.path.join(bundle_dir, 'index.faiss'))

    # Export the embeddings
    embeddings.tofile(os.path.join(bundle_dir, 'embeddings.bin'))

    # Export the chunks in JSONL format
    with open(os.path.join(bundle_dir, 'chunks.jsonl'), 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')

    # Export the ID map
    with open(os.path.join(bundle_dir, 'id_map.pkl'), 'wb') as f:
        pickle.dump(id_map, f)

    # Create and export the manifest
    manifest = {
        "class": chunks[0]['class'],
        "subject": chunks[0]['subject'],
        "chapter": chunks[0]['chapter'],
        "language": chunks[0]['language'],
        "embedding_model": "intfloat/e5-small-v2",  # or the chosen model
        "embedding_dim": embeddings.shape[1],
        "chunk_count": len(chunks),
        "chunk_strategy": "max_tokens_350",
        "created_at": datetime.now().isoformat(),
        "version": "1.0",
        "hash_strategy": "md5"
    }

    with open(os.path.join(bundle_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f)

    # Export the model information
    model_info = {
        "model_name": "intfloat/e5-small-v2",  # or the chosen model
        "model_version": "1.0"
    }

    with open(os.path.join(bundle_dir, 'model.json'), 'w') as f:
        json.dump(model_info, f)

    # Export the version information
    with open(os.path.join(bundle_dir, 'version.txt'), 'w') as f:
        f.write("1.0\n")

if __name__ == "__main__":
    # Example usage (to be replaced with actual data)
    collection_name = "class_8_science_en"
    index = faiss.IndexFlatL2(512)  # Example FAISS index
    embeddings = np.random.rand(10, 512).astype('float32')  # Example embeddings
    chunks = [{"text": "Example chunk", "class": 8, "subject": "science", "chapter": 3, "language": "en", "textbook": "ncert", "tokens": 127}]
    id_map = {0: "unique_hash_0"}

    export_faiss_bundle(collection_name, index, embeddings, chunks, id_map)