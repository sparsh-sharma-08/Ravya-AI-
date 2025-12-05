# RUNBOOK for NCERT Offline RAG Project

## Overview
This document provides operational instructions for running the NCERT Offline RAG project, which serves educational content to rural students using a Raspberry Pi as the local teaching engine. The project is designed to work offline and utilizes a two-phase architecture for data ingestion and usage.

## Phase A: Ingestion

### 1. Prepare Input Data
- Ensure your input data is in JSONL format, with each line containing a chunk of text structured as follows:
  ```json
  {
    "text": "...chunk...",
    "class": 8,
    "subject": "science",
    "chapter": 3,
    "language": "en",
    "textbook": "ncert",
    "tokens": 127
  }
  ```
- Validate that the text chunks adhere to the specified chunking rules (max 220–350 tokens, semantic continuity, etc.).

### 2. Run Ingestion Script
- Execute the ingestion script to process the JSONL file:
  ```bash
  ./scripts/run_ingest.sh <path_to_your_input_file.jsonl>
  ```
- This script will read the input file, chunk the text, generate embeddings, and store the data in Chroma.

### 3. Export FAISS Bundle
- After ingestion, export the FAISS bundle using:
  ```bash
  ./scripts/export_bundle.sh <class_subject_language>
  ```
- Replace `<class_subject_language>` with the appropriate identifier (e.g., `class_8_science_en`).

## Phase B: Usage on Raspberry Pi

### 1. Load FAISS Index
- Ensure the FAISS index and associated files are available on the Raspberry Pi in the `bundles` directory.

### 2. Querying
- Convert user queries into embeddings using the `embed_query.py` script.
- Retrieve relevant chunks from the FAISS index using the `retrieve.py` script.
- Generate answers using the `generate_answer.py` script, ensuring the output is child-friendly and validated.

### 3. Example Query Process
- A typical query process involves:
  1. User asks a question.
  2. Convert the question to an embedding.
  3. Retrieve top-k chunks from FAISS.
  4. Generate a response based on the retrieved context.

## Validation and Testing
- Ensure to run the unit tests located in the `tests` directory to validate the ingestion, retrieval, and bundle export processes:
  ```bash
  pytest tests/
  ```

## Troubleshooting
- If you encounter issues, check the logs generated during the ingestion and retrieval processes for errors.
- Ensure that all dependencies listed in `requirements.txt` are installed.

## Conclusion
This RUNBOOK serves as a guide for operating the NCERT Offline RAG project. Follow the instructions carefully to ensure successful ingestion and retrieval of educational content for students.