from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name='intfloat/e5-small-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)

    def save_embeddings(self, embeddings, file_path):
        with open(file_path, 'wb') as f:
            f.write(embeddings.numpy().tobytes())

    def load_embeddings(self, file_path):
        with open(file_path, 'rb') as f:
            return np.frombuffer(f.read(), dtype=np.float32)