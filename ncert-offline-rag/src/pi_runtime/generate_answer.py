import json
import os
from typing import List, Dict
from utils.validators import validate_output
from utils.hashing import generate_md5
from embed_query import convert_query_to_embedding
from retrieve import retrieve_chunks
from ollama import OllamaGemma  # Assuming Ollama Gemma is a class in an ollama module

class AnswerGenerator:
    def __init__(self, faiss_index_path: str, model_path: str):
        self.faiss_index_path = faiss_index_path
        self.model = OllamaGemma(model_path)

    def generate_answer(self, query: str, class_id: int, subject: str, language: str, chapter: int = None) -> str:
        # Convert the user query to an embedding
        query_embedding = convert_query_to_embedding(query)

        # Retrieve top-k chunks from FAISS
        retrieved_chunks = retrieve_chunks(query_embedding, class_id, subject, language, chapter)

        # Validate the retrieved chunks
        if not retrieved_chunks:
            return "No relevant information found. Please ask your teacher for help."

        # Prepare context for the model
        context = self.prepare_context(retrieved_chunks)

        # Generate answer using the model
        answer = self.model.generate(context)

        # Validate the output
        if not validate_output(answer):
            return "The answer could not be generated correctly. Please ask your teacher for help."

        return answer

    def prepare_context(self, chunks: List[Dict]) -> str:
        context = ""
        for chunk in chunks:
            context += f"{chunk['text']} "
        return context.strip()

# Example usage
if __name__ == "__main__":
    generator = AnswerGenerator(faiss_index_path="path/to/index.faiss", model_path="path/to/model")
    response = generator.generate_answer("What is photosynthesis?", class_id=8, subject="science", language="en")
    print(response)