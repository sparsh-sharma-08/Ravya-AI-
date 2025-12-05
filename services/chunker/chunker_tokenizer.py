"""Tokenizer-aware chunker.

Creates chunks that respect a tokenizer's token counts, with sentence boundaries and sentence overlap.
Writes JSONL to output directory (e.g., data/chunks/tokenized/chapter_1.jsonl).
"""
import os
import sys
import json
import uuid
import re
from typing import List

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except Exception:
    import nltk
    nltk.download('punkt')


def sentence_tokenize(text: str) -> List[str]:
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        # Fallback: simple regex split on punctuation + space
        parts = re.split(r'(?<=[\.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]


def is_formula(text: str) -> bool:
    if '->' in text or '→' in text or re.search(r'[A-Za-z][0-9]', text) or re.search(r'CO2|H2O|H_{2}O', text):
        return True
    return False


def chunk_with_tokenizer(sentences: List[str], tokenizer, max_tokens: int, overlap_sentences: int = 2):
    chunks = []
    cur_sents = []
    cur_tokens = 0
    for sent in sentences:
        toks = tokenizer.encode(sent, add_special_tokens=False)
        tlen = len(toks)
        if tlen >= max_tokens:
            # sentence itself is too long: emit as its own chunk
            if cur_sents:
                chunks.append(' '.join(cur_sents))
                cur_sents = []
                cur_tokens = 0
            chunks.append(sent)
            continue
        if cur_tokens + tlen > max_tokens and cur_sents:
            chunks.append(' '.join(cur_sents))
            # start new with overlap
            cur_sents = cur_sents[-overlap_sentences:] if overlap_sentences > 0 else []
            cur_tokens = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in cur_sents)
        cur_sents.append(sent)
        cur_tokens += tlen
    if cur_sents:
        chunks.append(' '.join(cur_sents))
    return chunks


def convert(chapter_json_path: str, out_jsonl_path: str, max_tokens: int = 500, tokenizer_name: str = 'gpt2', overlap_sentences: int = 2):
    if AutoTokenizer is None:
        raise RuntimeError('transformers not installed; please install transformers')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    with open(chapter_json_path, 'r', encoding='utf-8') as f:
        ch = json.load(f)

    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)
    out = []
    for sec in ch.get('sections', []):
        heading = sec.get('heading', '')
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
            if is_formula(t):
                chunk_texts = [t]
            else:
                sents = sentence_tokenize(t)
                chunk_texts = chunk_with_tokenizer(sents, tokenizer, max_tokens=max_tokens, overlap_sentences=overlap_sentences)

            for ct in chunk_texts:
                chunk_obj = {
                    'id': str(uuid.uuid4()),
                    'text': ct,
                    'source': {'file': os.path.basename(chapter_json_path), 'page': None},
                    'subject': ch.get('subject', 'Science'),
                    'board': 'NCERT',
                    'class': int(ch.get('class', 0)) if str(ch.get('class','')).isdigit() else ch.get('class'),
                    'chapter': ch.get('chapter_title', ch.get('chapter_number','')),
                    'section_heading': heading,
                    'language': 'en',
                    'script': 'Latin',
                    'tags': [],
                    'bloom_level': 'Understanding',
                    'difficulty': 'medium',
                    'has_diagram': bool(sec.get('diagrams')),
                    'formula_list': [ct] if is_formula(ct) else []
                }
                out.append(chunk_obj)

    with open(out_jsonl_path, 'w', encoding='utf-8') as wf:
        for o in out:
            wf.write(json.dumps(o, ensure_ascii=False) + '\n')

    print(f'Wrote {len(out)} token-aware chunks to {out_jsonl_path}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python chunker_tokenizer.py chapter_1.json data/chunks/tokenized/chapter_1.jsonl')
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
