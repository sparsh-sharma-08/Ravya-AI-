# NCERT Offline RAG

This project implements an offline-first educational robot designed to serve NCERT/CBSE content to rural students using a Raspberry Pi as the local teaching engine. The architecture is built around a two-phase process: ingestion and usage.

## Project Structure

- **src/**: Contains the source code for ingestion and runtime processes.
  - **ingestion/**: Handles the ingestion of JSONL files, chunking, embedding generation, and exporting to FAISS.
  - **pi_runtime/**: Manages query embedding, retrieval from FAISS, and answer generation using the Ollama Gemma model.
  - **utils/**: Provides utility functions for validation and hashing.
  - **types.py**: Defines data types and constants used throughout the project.

- **bundles/**: Contains the exported FAISS bundles for different subjects and classes.

- **examples/**: Provides example input files for the ingestion process.

- **tests/**: Contains unit tests for various components of the project.

- **scripts/**: Includes shell scripts for automating ingestion, bundle export, and validation.

- **docs/**: Provides operational and end-to-end instructions for using the project.

- **requirements.txt**: Lists the Python dependencies required for the project.

- **pyproject.toml**: Contains project metadata and configuration for package management.

- **.gitignore**: Specifies files and directories to be ignored by version control.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ncert-offline-rag
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your JSONL files according to the specified format for ingestion.

4. Run the ingestion process:
   ```
   ./scripts/run_ingest.sh <path-to-your-jsonl-file>
   ```

5. Export the FAISS bundle:
   ```
   ./scripts/export_bundle.sh
   ```

## Usage Instructions

1. Load the FAISS index on the Raspberry Pi.
2. Use the `embed_query.py` to convert user queries into embeddings.
3. Retrieve relevant chunks using `retrieve.py`.
4. Generate answers using `generate_answer.py`, ensuring the output is child-friendly and validated.

## Contribution

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.