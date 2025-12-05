# Ravya — AI / RAG Component (updated)

Role
This README documents the current AI/RAG runtime and developer workflows in Ravya: ingestion, bundle export, Pi runtime retrieval, and the local wrapper scripts used to run queries end‑to‑end.

High level
- Ingestion/embedding runs on a laptop/cloud (not on Pi).
- The runtime under `ncert-offline-rag/src/pi_runtime`/`ncert-offline-rag/src/rag` loads exported FAISS bundles and performs offline retrieval + LLM prompting.
- A small CLI wrapper (`ncert-offline-rag/src/rag/ask.py`) performs: embed → retrieve → model call → prints a clean answer (designed for demos).

Quick links (current)
- Embedding (query / chunk): ncert-offline-rag/src/pi_runtime/embed_query.py
- Retrieval: ncert-offline-rag/src/rag/retrieve.py
- Debug retrieval (scores/snippets): ncert-offline-rag/src/rag/debug_retrieve.py
- Prompt builder: ncert-offline-rag/src/rag/build_prompt.py
- Model call helper: ncert-offline-rag/src/rag/gemma_call.py
- RAG entrypoint (retrieve + prompt + model): ncert-offline-rag/src/rag/rag_answer.py
- Convenience wrapper (auto‑embed + run + print clean answer): ncert-offline-rag/src/rag/ask.py
- FAISS bundle layout: bundle/<bundle-name>/* (chunks.jsonl, embeddings.bin, index.faiss, id_map.pkl, model.json, manifest.json, version.txt)

Quickstart — run a demo query (from repo root)
1) Activate venv:
```
.venv/bin/activate  # or: source .venv/bin/activate
```

2) One-command query (auto-embed, retrieve, model call). Use the repo-root python binary:
```
.venv/bin/python ncert-offline-rag/src/rag/ask.py --query "What is an endothermic reaction?" --k 5 --model 2b
```

Notes
- ask.py is implemented to run from the repo root and prints only the plain answer text (no ids or debug) for demonstrations.
- If you need the sources, run the inspector or retrieve CLI (below).

Inspect retrieval and debug
- Pretty top‑k JSON:
```
.venv/bin/python ncert-offline-rag/src/rag/retrieve.py --bundle ./bundle/class_8_science_en --embed /path/to/q.json --k 5 | python -m json.tool
```

- Human inspector (creates embedding + prints top snippets):
```
.venv/bin/python ncert-offline-rag/src/rag/inspect_query.py --text "Your question" --k 5
```

- Debug with scores / norms:
```
.venv/bin/python ncert-offline-rag/src/rag/debug_retrieve.py --bundle ./bundle/class_8_science_en --embed /path/to/q.json --k 10
```

Prompt length / style
- To change answer length or style, edit:
  - ncert-offline-rag/src/rag/build_prompt.py
  - The prompt controls whether answers are concise or teacher‑like (and whether JSON contains sources inline).
- The runtime enforces "use only CONTEXT" and a similarity gate: if top‑1 similarity < 0.60, the runtime returns REFER_TEACHER (to avoid hallucination).

Troubleshooting (common)
- Intermittent {"status":"refer_teacher"} when running wrappers:
  - Ensure you run ask.py from repo root (wrapper resolves scripts relative to repo root).
  - If model backend (Ollama/Gemma) is unavailable, rag_answer may fallback; restart the model service.
  - Re-create query embedding before calling retrieve (ask.py does this automatically).
- If ask.py returns nothing for demos, run the low-level steps manually to collect logs:
  1) embed_query.py → produce /tmp/q.json
  2) retrieve.py --bundle ... --embed /tmp/q.json --k 5
  3) use build_prompt.py + gemma_call.py to see the raw model output.

Development notes
- Embeddings and bundle export happen under the ingest pipeline (not on Pi).
- The Pi runtime (or demo wrapper) should never compute embeddings — embeddings are precomputed and shipped in the bundle.
- Keep `model.json` and `manifest.json` in bundles updated if you change embedding model or chunk rules.

Files & locations (summary)
- repo root
  - ingest_pipeline.py (ingest/export)
  - bundle/ (exported FAISS bundles)
  - ncert-offline-rag/src/pi_runtime/ (embed/retrieve helpers)
  - ncert-offline-rag/src/rag/ (prompting + model-call + wrappers)

Contact & license
- AI Lead: Sparsh Sharma
- Add LICENSE at repo root (MIT/Apache2)

If you want, I can:
- Produce a one‑paragraph demo README (short) for presentations, or
- Remove debug scripts and keep only the production wrapper and minimal CLIs.
