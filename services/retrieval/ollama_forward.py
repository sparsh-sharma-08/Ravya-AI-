"""Forward retrieved context to a local Ollama model (Gemma/Vendor) via REST API.

Usage: ollama_forward.py "Your question"
"""
import os
import sys
import requests
from retrieval import chroma_query


def build_prompt(context_texts, user_query):
    ctx = '\n\n'.join(context_texts)
    prompt = f"Context:\n{ctx}\n\nTask: Using ONLY the above context, answer the question: {user_query}\nOutput JSON only."
    return prompt


def send_to_ollama(prompt, model='gemma', host='http://localhost:11434'):
    url = f"{host}/api/generate"
    payload = {
        'model': model,
        'input': prompt,
        'max_tokens': 512
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main(query_text: str):
    res = chroma_query.query('chapter_1', query_text, k=5)
    contexts = res['documents'][0]
    prompt = build_prompt(contexts, query_text)
    print('Sending prompt to Ollama...')
    out = send_to_ollama(prompt)
    print('Model response:')
    print(out)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python ollama_forward.py "Your question"')
        sys.exit(1)
    main(' '.join(sys.argv[1:]))
