#!/bin/bash

# Validate the contents of the exported FAISS bundle

BUNDLE_DIR="bundles/class_8_science_en"

# Check if the bundle directory exists
if [ ! -d "$BUNDLE_DIR" ]; then
  echo "Bundle directory does not exist: $BUNDLE_DIR"
  exit 1
fi

# Check for required files
REQUIRED_FILES=(
  "index.faiss"
  "embeddings.bin"
  "chunks.jsonl"
  "id_map.pkl"
  "manifest.json"
  "model.json"
  "version.txt"
)

for FILE in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$BUNDLE_DIR/$FILE" ]; then
    echo "Missing required file: $FILE"
    exit 1
  fi
done

# Validate manifest.json
MANIFEST_FILE="$BUNDLE_DIR/manifest.json"
if ! jq empty "$MANIFEST_FILE" > /dev/null 2>&1; then
  echo "Invalid JSON format in manifest.json"
  exit 1
fi

# Check for required fields in manifest.json
REQUIRED_MANIFEST_FIELDS=(
  "class"
  "subject"
  "chapter"
  "language"
  "embedding_model"
  "embedding_dim"
  "chunk_count"
  "chunk_strategy"
  "created_at"
  "version"
  "hash_strategy"
)

for FIELD in "${REQUIRED_MANIFEST_FIELDS[@]}"; do
  if ! jq -e ".${FIELD}" "$MANIFEST_FILE" > /dev/null; then
    echo "Missing required field in manifest.json: $FIELD"
    exit 1
  fi
done

echo "Bundle validation successful."