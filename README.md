# Ravya — AI / RAG Component

Role
This README documents the AI work in Ravya: ingestion, chunking, embedding, Chroma ingestion, FAISS bundle export, and Raspberry‑Pi offline retrieval + RAG. Use this as the single-source guide for running, reproducing, or extending the AI pipeline.

Project summary
- Convert NCERT/CBSE PDFs → validated semantic chunks (JSONL).
- Compute semantic embeddings (laptop/cloud only) with `intfloat/e5-small-v2`.
- Ingest embeddings + metadata into Chroma for indexing.
- Export deterministic FAISS bundles (index + embeddings + metadata) for Raspberry Pi.
- Pi runtime loads FAISS bundle, performs retrieval, and forwards context to an offline LLM (Ollama Gemma) for grounded answers.

Key guarantees
- Embeddings computed only during ingestion (laptop/cloud).
- Pi runtime is offline: no embedding computation, no Chroma, no cloud LLM.
- Metadata-filtered retrieval and similarity gate to avoid hallucination.
- Deterministic bundle export with explicit hashing rules.

Quick links
- `ingest_pipeline.py` — end‑to‑end ingestion (validate → embed → upsert → smoke test → export)
- `docker-compose.chroma.yml` — Chroma ingestion server (laptop)
- `src/ingest/` — helper ingestion scripts & tokenized chunking
- `vector-db/scripts/export_bundle.py` — FAISS exporter utilities
- `ncert-offline-rag/src/pi_runtime/` — Pi runtime: `load_bundle.py`, `retrieve.py`, `rag_generate.py`, `rag_answer.py`
- `services/chunker/` — chunker and token-aware chunking code
- `tests/` — unit tests (chunker test included)

Quickstart (developer / evaluator)
Prereqs: laptop/cloud (NOT Raspberry Pi), Docker, Python 3.11+, npm (if running Node server).

1) Start Chroma (ingestion server)
```
# from repo root
docker compose -f docker-compose.chroma.yml up -d
# verify
curl -sS http://localhost:8000/openapi.json | head -n 1
```
If Chroma is on another host, set `CHROMA_URL`.

2) Prepare sample JSONL (one JSON per line)
Example content:
```
{"text":"Photosynthesis converts light to chemical energy.","class":8,"subject":"science","chapter":3,"language":"en","textbook":"ncert","tokens":32}
{"text":"Respiration releases energy by breaking down glucose.","class":8,"subject":"science","chapter":3,"language":"en","textbook":"ncert","tokens":28}
```
Save as `examples/sample_chunks.jsonl`.

3) Python deps (laptop/cloud)
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-ai.txt
# or:
pip install sentence-transformers requests numpy faiss-cpu
```
Notes: `sentence-transformers` downloads model artifacts (CPU); `faiss-cpu` required for building/exporting FAISS.

4) Run ingest pipeline (end‑to‑end)
```
python ingest_pipeline.py \
  --input examples/sample_chunks.jsonl \
  --class 8 \
  --subject science \
  --language en \
  --textbook ncert \
  --chroma-url http://localhost:8000 \
  --bundle-root ./bundle
```
Expected output: path to bundle, e.g. `./bundle/class_8_science_en/` containing:
- `chunks.jsonl`
- `embeddings.bin`
- `index.faiss`
- `id_map.pkl`
- `manifest.json`
- `model.json`
- `version.txt`

If ingestion or smoke test fails the script will attempt best-effort cleanup of the partial Chroma collection and exit non-zero.

5) Copy bundle to Raspberry Pi
```
scp -r bundle/class_8_science_en pi@raspberrypi:/home/pi/ravya/bundles/
```

6) On Raspberry Pi: Pi runtime (offline retrieval)
Install minimal deps on Pi:
```
python3 -m venv .venv && source .venv/bin/activate
pip install numpy faiss-cpu requests
# DO NOT install sentence-transformers on Pi
```
Example retrieve:
```
# Prepare query embedding on laptop (embed_query.py). Copy to Pi.
python ncert-offline-rag/src/pi_runtime/retrieve.py \
  --bundle /home/pi/ravya/bundles/class_8_science_en \
  --embed q1.json \
  --k 5
```

Repository structure (AI‑focused)
```
/
├─ ingest_pipeline.py
├─ docker-compose.chroma.yml
├─ services/
│  ├─ chunker/
│  └─ embedder/
├─ vector-db/
│  └─ scripts/export_bundle.py
├─ ncert-offline-rag/
│  └─ src/
│     └─ pi_runtime/
├─ examples/
│  └─ sample_chunks.jsonl
├─ tests/
│  └─ test_chunker.py
├─ requirements-ai.txt
└─ README.md  <-- this file
```

Data formats & contracts

Input JSONL chunk (required fields and types)
```
{
  "text": "<chunk text>",         # str
  "class": 8,                     # int
  "subject": "science",           # str
  "chapter": 3,                   # int
  "language": "en",               # str
  "textbook": "ncert",            # str
  "tokens": 127                   # int
}
```
- Every line is validated. `--class`, `--subject`, `--language` must match CLI flags.
- Chunk ID rule: `id = "<class>_<subject>_<chapter>_<md5(text)>"`.

Chroma metadata schema
```
{
  "id": "<class_subject_chapter_md5hash>",
  "class": 8,
  "subject": "science",
  "chapter": 3,
  "language": "en",
  "textbook": "ncert",
  "tokens": 127,
  "hash": "<md5_of_chunk_text>"
}
```

Exported bundle files
- `chunks.jsonl` — lines: `{"metadata": {...}, "text": "..."}` (same order as ids)
- `embeddings.bin` — contiguous float32 array
- `id_map.pkl` — ordered list of ids (pickle)
- `index.faiss` — `IndexFlatIP` built on normalized vectors
- `model.json` — embedding model name and dim
- `manifest.json` — bundle metadata (class, subject, chunk_count, version, created_at, etc.)
- `version.txt` — semantic bundle version (e.g. `2025.01.00`)

RAG & safety rules
- Metadata-filtered retrieval: every query must provide `class`, `subject`, `language`.
- Similarity gate: if top1 cosine score < 0.60 → return `REFER_TEACHER` (no LLM call).
- LLM prompt rules (Pi): LLM must use ONLY provided context. If unsure, reply "I don't know" or "please ask your teacher." Always cite chunk IDs used.
- No student PII uploaded without consent.

Troubleshooting & common errors
- Chroma unreachable: ensure Docker is running and CHROMA_URL correct.
- Slow model download / memory errors: reduce `--embed-batch`.
- `faiss` import errors: `pip install faiss-cpu` (use prebuilt wheel for ARM).
- Bundle too large: reduce chunk granularity or quantize index.

Testing & validation
- Unit tests: `python tests/test_chunker.py`
- Smoke test ingestion: ingest sample and verify `manifest.json` fields.
- Acceptance: 10 sample QA queries → ≥7 correct retrievals, ≤3 REFER_TEACHER.

Development & contribution
- Work on feature branches.
- Add tests for chunk → embedding → id mapping logic.
- When changing model or chunk rules, produce new bundle version and update `manifest.json`.
- Record experiments/commands in `experiments/`.

Contact & license
- AI Lead: Sparsh Sharma — sparsh@example.com
- Add a LICENSE file at repo root (MIT/Apache2 as decided).

Final notes
Keep this README current as chunking rules, the embedding model, or safety checks change. Include exact commands and metrics used to produce exported bundles to aid reproducibility and debugging.
