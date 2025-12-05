#!/bin/bash

# This script automates the export of the FAISS bundle after ingestion.

# Define the bundle directory
BUNDLE_DIR="bundles/class_8_science_en"

# Check if the bundle directory exists
if [ ! -d "$BUNDLE_DIR" ]; then
  echo "Bundle directory does not exist: $BUNDLE_DIR"
  exit 1
fi

# Export the FAISS index
python3 src/ingestion/export_faiss.py --bundle_dir "$BUNDLE_DIR"

# Check if the export was successful
if [ $? -ne 0 ]; then
  echo "Failed to export FAISS bundle."
  exit 1
fi

echo "FAISS bundle exported successfully to $BUNDLE_DIR."