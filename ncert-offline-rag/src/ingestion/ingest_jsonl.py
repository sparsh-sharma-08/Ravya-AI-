import json
import hashlib
import os

def validate_jsonl_format(line):
    required_keys = {"text", "class", "subject", "chapter", "language", "textbook", "tokens"}
    data = json.loads(line)
    if not required_keys.issubset(data.keys()):
        raise ValueError("Missing required keys in JSONL format.")
    if not isinstance(data["class"], int) or not isinstance(data["chapter"], int):
        raise ValueError("Class and chapter must be integers.")
    if not isinstance(data["tokens"], int):
        raise ValueError("Tokens must be an integer.")
    return data

def generate_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def ingest_jsonl(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    chunks = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data = validate_jsonl_format(line)
                    data['hash'] = generate_hash(data['text'])
                    chunks.append(data)
                except ValueError as e:
                    print(f"Error processing line: {line}. {e}")
    
    return chunks

if __name__ == "__main__":
    input_file = "path/to/your/input.jsonl"  # Update this path as needed
    chunks = ingest_jsonl(input_file)
    print(f"Successfully ingested {len(chunks)} chunks.")