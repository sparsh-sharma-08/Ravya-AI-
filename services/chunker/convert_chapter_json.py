"""Convert a chapter JSON (NCERT-like) into chunk JSONL following Ravya chunk schema.

Usage:
  python convert_chapter_json.py /path/to/chapter_1.json /path/to/output.jsonl
"""
import json
import os
import sys
import uuid
import re
from typing import List


def split_sentences(text: str) -> List[str]:
    # lightweight sentence splitter
    parts = re.split(r'(?<=[\.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def is_formula(text: str) -> bool:
    if '->' in text or '→' in text or re.search(r'[A-Za-z][0-9]', text) or re.search(r'\bCO2\b|CO_{2}|H2O|H_{2}O', text):
        return True
    return False


def paragraph_to_chunks(paragraph: str, max_tokens: int = 200) -> List[str]:
    if is_formula(paragraph):
        return [paragraph]
    sents = split_sentences(paragraph)
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        w = s.split()
        if cur_len + len(w) > max_tokens and cur:
            chunks.append(' '.join(cur))
            cur = w
            cur_len = len(w)
        else:
            cur.extend(w)
            cur_len += len(w)
    if cur:
        chunks.append(' '.join(cur))
    return chunks


def convert(chapter_json_path: str, out_jsonl_path: str, max_tokens: int = 200):
    with open(chapter_json_path, 'r', encoding='utf-8') as f:
        ch = json.load(f)

    book = ch.get('book_name', 'unknown')
    class_no = int(ch.get('class', 0)) if str(ch.get('class', '')).isdigit() else ch.get('class')
    subject = ch.get('subject', 'Science')
    chapter_number = ch.get('chapter_number', '')
    chapter_title = ch.get('chapter_title', '')

    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)
    out_lines = []
    for sec in ch.get('sections', []):
        heading = sec.get('heading', '')
        # collect text from text, examples, definitions
        texts = []
        if sec.get('text'):
            texts.append(sec.get('text'))
        for ex in sec.get('examples', []) or []:
            if ex.get('title'):
                texts.append(ex.get('title'))
            if ex.get('content'):
                texts.append(ex.get('content'))
        for d in sec.get('definitions', []) or []:
            if d.get('term') and d.get('meaning'):
                texts.append(f"{d.get('term')}: {d.get('meaning')}")

        for t in texts:
            chunks = paragraph_to_chunks(t, max_tokens=max_tokens)
            for c in chunks:
                formula_list = []
                if is_formula(c):
                    formula_list.append(c)
                chunk_obj = {
                    "id": str(uuid.uuid4()),
                    "text": c,
                    "source": {"file": os.path.basename(chapter_json_path), "page": None},
                    "subject": subject,
                    "board": "NCERT",
                    "class": class_no,
                    "chapter": chapter_title or chapter_number,
                    "section_heading": heading,
                    "language": "en",
                    "script": "Latin",
                    "tags": [],
                    "bloom_level": "Understanding",
                    "difficulty": "medium",
                    "has_diagram": bool(sec.get('diagrams')),
                    "formula_list": formula_list,
                }
                out_lines.append(chunk_obj)

    with open(out_jsonl_path, 'w', encoding='utf-8') as wf:
        for obj in out_lines:
            wf.write(json.dumps(obj, ensure_ascii=False) + '\n')

    print(f"Wrote {len(out_lines)} chunks to {out_jsonl_path}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python convert_chapter_json.py chapter_1.json data/chunks/chapter_1.jsonl')
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
