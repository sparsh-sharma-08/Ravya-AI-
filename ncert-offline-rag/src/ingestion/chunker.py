from typing import List, Dict
import json

class Chunker:
    def __init__(self, max_tokens: int = 350, min_tokens: int = 220):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def chunk_text(self, text: str) -> List[str]:
        # Split the text into potential chunks based on headings and bullet points
        potential_chunks = self._split_on_headings_and_bullets(text)
        valid_chunks = []

        for chunk in potential_chunks:
            token_count = self._count_tokens(chunk)
            if self.min_tokens <= token_count <= self.max_tokens:
                valid_chunks.append(chunk)

        return valid_chunks

    def _split_on_headings_and_bullets(self, text: str) -> List[str]:
        # Split the text by new lines and bullet points
        return [part.strip() for part in text.split('\n') if part.strip()]

    def _count_tokens(self, text: str) -> int:
        # A simple token count based on whitespace
        return len(text.split())

def process_jsonl(input_file: str, output_file: str):
    chunker = Chunker()
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            chunks = chunker.chunk_text(data['text'])
            for chunk in chunks:
                data['text'] = chunk
                outfile.write(json.dumps(data) + '\n')