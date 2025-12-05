"""Chunker: load extracted JSON and produce chunks JSONL per chapter.

This is a simple implementation that splits paragraphs into sentence-based chunks
and keeps formulas together. It follows the chunk schema required in the project.
"""
import json
import os
import re
import uuid
from typing import List

import nltk

# Ensure punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt")


SENT_SPLIT_RE = re.compile(r"(?<=[\.!?])\s+")


def sentence_tokenize(text: str) -> List[str]:
    # Use nltk's sent_tokenize for robustness
    try:
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)
    except Exception:
        # Fallback: simple regex-based split
        return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]


def is_formula(text: str) -> bool:
    # naive formula detection: contains => or -> or chemical element patterns or math symbols
    if "->" in text or "=" in text or re.search(r"\b[A-Z][a-z]?\d", text):
        return True
    return False


def chunk_paragraph(paragraph: str, max_tokens: int = 250) -> List[str]:
    # token ~= word; simple splitter that keeps formulas intact
    if is_formula(paragraph):
        return [paragraph]
    sents = sentence_tokenize(paragraph)
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        words = s.split()
        if cur_len + len(words) > max_tokens and cur:
            chunks.append(" ".join(cur))
            cur = words
            cur_len = len(words)
        else:
            cur.extend(words)
            cur_len += len(words)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def build_chunks_from_extracted(extracted_json: dict, metadata_overrides: dict = None, max_tokens=250):
    chunks = []
    meta_base = metadata_overrides or {}
    title = extracted_json.get("title", "unknown")
    pages = extracted_json.get("pages", [])
    for p in pages:
        page_no = p.get("page_no")
        for block in p.get("blocks", []):
            btype = block.get("type")
            text = block.get("text", "").strip()
            if not text:
                continue
            if btype == "heading":
                section_heading = text
                # create small chunk for heading
                chunk = {
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "source": {"file": title, "page": page_no},
                    "subject": meta_base.get("subject", "Science"),
                    "board": meta_base.get("board", "NCERT"),
                    "class": meta_base.get("class", 8),
                    "chapter": meta_base.get("chapter", title),
                    "section_heading": section_heading,
                    "language": meta_base.get("language", "en"),
                    "script": meta_base.get("script", "Latin"),
                    "tags": [],
                    "bloom_level": "Understanding",
                    "difficulty": "medium",
                    "has_diagram": False,
                    "formula_list": []
                }
                chunks.append(chunk)
            elif btype in ("paragraph", "formula"):
                # chunk paragraph into sentences
                subchunks = chunk_paragraph(text, max_tokens=max_tokens)
                for sc in subchunks:
                    formulas = []
                    if is_formula(sc):
                        formulas.append(sc)
                    chunk = {
                        "id": str(uuid.uuid4()),
                        "text": sc,
                        "source": {"file": title, "page": page_no},
                        "subject": meta_base.get("subject", "Science"),
                        "board": meta_base.get("board", "NCERT"),
                        "class": meta_base.get("class", 8),
                        "chapter": meta_base.get("chapter", title),
                        "section_heading": meta_base.get("section_heading", ""),
                        "language": meta_base.get("language", "en"),
                        "script": meta_base.get("script", "Latin"),
                        "tags": [],
                        "bloom_level": "Understanding",
                        "difficulty": "medium",
                        "has_diagram": False,
                        "formula_list": formulas,
                    }
                    chunks.append(chunk)
            else:
                # ignore other block types for now
                continue
    return chunks


def save_chunks_jsonl(chunks: List[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def main(in_file: str, out_file: str, max_tokens: int = 250):
    with open(in_file, "r", encoding="utf-8") as f:
        extracted = json.load(f)
    chunks = build_chunks_from_extracted(extracted, max_tokens=max_tokens)
    save_chunks_jsonl(chunks, out_file)
    print(f"Wrote {len(chunks)} chunks to {out_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python chunker.py data/extracted/<file>.json data/chunks/<out>.jsonl")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
