#!/bin/bash

# This script automates the ingestion process for NCERT/CBSE content.

# Define the input JSONL file
INPUT_FILE="examples/sample_input.jsonl"

# Run the ingestion script
python src/ingestion/ingest_jsonl.py "$INPUT_FILE"

# Run the chunking process
python src/ingestion/chunker.py

# Generate embeddings
python src/ingestion/embeddings.py

# Ingest into Chroma
python src/ingestion/chroma_wrapper.py

# Export the FAISS bundle
python src/ingestion/export_faiss.py

echo "Ingestion process completed successfully."