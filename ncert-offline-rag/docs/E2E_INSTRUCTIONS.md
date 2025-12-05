# End-to-End Instructions for NCERT Offline RAG Project

## Overview
This document provides step-by-step instructions for setting up and using the NCERT Offline RAG project. The project is designed to serve educational content to rural students using a Raspberry Pi as the local teaching engine.

## Prerequisites
- A laptop or cloud environment for the ingestion phase.
- A Raspberry Pi for the usage phase.
- Python 3.7 or higher installed on both the laptop and Raspberry Pi.
- Required Python packages as listed in `requirements.txt`.

## Phase A: Ingestion

### Step 1: Prepare Input Data
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
- Save your input data in a file, e.g., `input_data.jsonl`.

### Step 2: Run Ingestion Script
- Navigate to the `src/ingestion` directory.
- Execute the ingestion script:
  ```bash
  python ingest_jsonl.py --input_file path/to/input_data.jsonl
  ```

### Step 3: Validate Ingestion
- After ingestion, validate the output to ensure all chunks are correctly processed.
- Use the validation functions in `src/utils/validators.py`.

### Step 4: Export FAISS Bundle
- Once validation is complete, export the FAISS bundle:
  ```bash
  python export_faiss.py --output_dir path/to/bundles/class_8_science_en
  ```

## Phase B: Usage on Raspberry Pi

### Step 5: Transfer Bundle to Raspberry Pi
- Copy the exported bundle from your laptop to the Raspberry Pi. Ensure the following files are included:
  - `index.faiss`
  - `embeddings.bin`
  - `chunks.jsonl`
  - `id_map.pkl`
  - `manifest.json`
  - `model.json`
  - `version.txt`

### Step 6: Set Up Raspberry Pi Environment
- Install the required Python packages on the Raspberry Pi:
  ```bash
  pip install -r requirements.txt
  ```

### Step 7: Run the Query Process
- Use the following command to start the query process:
  ```bash
  python embed_query.py --query "Your question here"
  ```

### Step 8: Retrieve and Generate Answer
- The retrieval script will fetch the top-k chunks based on the query embedding.
- The answer generation will be handled by `generate_answer.py`, which will provide a child-friendly response.

## Testing and Validation
- Ensure to run the tests provided in the `tests` directory to validate the ingestion and retrieval processes:
  ```bash
  python -m unittest discover -s tests
  ```

## Conclusion
Follow these instructions to successfully set up and use the NCERT Offline RAG project. Ensure that all steps are completed in order, and validate each phase to ensure a smooth operation.