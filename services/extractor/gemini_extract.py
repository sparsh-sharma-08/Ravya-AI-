"""Gemini extractor stub.

Usage:
    python gemini_extract.py data/raw/sample.pdf

This script is a placeholder that will call Google Gemini Document API to extract structured
content from PDFs. For now it can run in 'stub' mode and write a sample JSON to data/extracted/.
"""
import json
import os
import sys
from datetime import datetime


def write_sample_output(out_path: str):
    sample = {
        "title": "Sample Science Chapter",
        "source_url": "local:file://data/raw/sample.pdf",
        "extracted_at": datetime.utcnow().isoformat() + "Z",
        "extractor_version": "0.1.0-stub",
        "pages": [
            {
                "page_no": 1,
                "blocks": [
                    {"type": "heading", "text": "Photosynthesis", "bbox": [0,0,600,40]},
                    {"type": "paragraph", "text": "Photosynthesis is the process by which green plants make their food.", "bbox": [0,50,600,120]},
                    {"type": "paragraph", "text": "Plants use sunlight, water and carbon dioxide to make glucose and oxygen.", "bbox": [0,130,600,200]},
                ],
            },
            {
                "page_no": 2,
                "blocks": [
                    {"type": "heading", "text": "Equation", "bbox": [0,0,600,40]},
                    {"type": "paragraph", "text": "6CO2 + 6H2O -> C6H12O6 + 6O2", "bbox": [0,50,600,80]},
                ],
            },
        ],
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)


def main(argv):
    # For now ignore input PDF; write to data/extracted/sample_book_ch01.json
    out_path = os.path.join("data", "extracted", "sample_book_ch01.json")
    write_sample_output(out_path)
    print(f"Wrote sample extracted JSON to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
