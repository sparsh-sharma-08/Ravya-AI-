# Ravya — Raspberry Pi Offline RAG Runtime

This directory contains the offline inference system used by Ravya on the Raspberry Pi.
The Pi runtime does NOT generate embeddings and does NOT connect to Chroma or remote APIs.
It loads FAISS bundles, retrieves top-K chunks, and generates answers using the local Gemma LLM via Ollama.

Core idea
- Ingestion (laptop/cloud): chunking → embedding → Chroma ingestion → FAISS bundle export.
- Pi runtime: load bundle → retrieval → prompt → local LLM answer.
- No embedding computation on Pi. No external inference calls.

Offline guarantees
- Only precomputed embeddings are used.
- Metadata-filtered retrieval (class, subject, language, textbook).
- Similarity threshold gating (default 0.60).
- LLM must cite chunk IDs; if no citation, answer is refused.
- Local Gemma model only via Ollama CLI.
- Safe fallback: REFER_TEACHER.

Directory overview
ncert-offline-rag/
└─ src/
   └─ pi_runtime/
      ├─ load_bundle.py     # Load FAISS bundle into memory
      ├─ retrieve.py        # Vector search against FAISS index
      ├─ rag_generate.py    # Build prompt + call Ollama Gemma
      ├─ rag_answer.py      # Retrieval wrapper → final JSON answer
      ├─ rag_cli.py         # One-shot CLI for teachers / Pi users
      └─ README.md          # This file

Installation — Raspberry Pi (minimal)
1. Create Python env
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (Pi)
Do NOT install transformers or sentence-transformers on Pi.
```bash
pip install numpy faiss-cpu requests
```

3. Install Ollama (ARM)
Follow Ollama's Pi instructions; then pull model:
```bash
ollama pull gemma:2b
```
Recommendation: use Gemma 2b (≈1.2–1.5GB). Gemma 7b is not recommended on most Pi boards.

FAISS bundle requirements
Each subject bundle must contain:
```
class_x_subject_language/
├─ index.faiss
├─ embeddings.bin      # contiguous float32 NxD
├─ chunks.jsonl        # ordered [{"metadata":...,"text":...}]
├─ id_map.pkl          # ordered list of ids (pickle)
├─ model.json          # {"name":..., "dim":...}
├─ manifest.json       # bundle metadata
└─ version.txt
```
Bundles are produced by the ingestion pipeline on laptop/cloud:
```bash
python ingest_pipeline.py ...
```
Copy bundle to Pi:
```bash
scp -r bundle/class_8_science_en pi@raspberrypi:/home/pi/ravya/bundles/
```

1. Loading bundle
```bash
python load_bundle.py /home/pi/ravya/bundles/class_8_science_en
```
Expected output:
```json
{"chunks": 120, "dim": 384}
```
Missing files → fail early.

2. Retrieval
Precompute query embedding on laptop (same model as bundle) and copy to Pi (.json or .npy).

Example:
```bash
python retrieve.py \
  --bundle /home/pi/ravya/bundles/class_8_science_en \
  --embed q.json \
  --k 5
```

Response formats
- OK:
```json
{
  "status": "ok",
  "chunks": [
    {
      "id":"8_science_3_abcd123",
      "rank":0,
      "score":0.874,
      "text":"...",
      "meta":{...}
    }
  ]
}
```
- Not confident:
```json
{"status":"refer_teacher"}
```

3. Local LLM answer
Reads retrieved chunks → prompt → Ollama Gemma.

Example:
```bash
python rag_answer.py \
  --bundle /home/pi/ravya/bundles/class_8_science_en \
  --embed q.json \
  --query "Explain respiration" \
  --model 2b
```

Output on success:
```json
{
  "answer":"Respiration is ...",
  "sources":["8_science_3_a34bd1"]
}
```

If the model does not cite chunk IDs, the system returns:
```json
{"answer":"I don't know, ask your teacher.","sources":[]}
```

4. One-click end-to-end
Teachers/users:
```bash
python rag_cli.py \
  --bundle /home/pi/ravya/bundles/class_8_science_en \
  --embed q.json \
  --query "What is photosynthesis?" \
  --k 5 \
  --model 2b
```
Output either an `answer` object with `sources` or `{"status":"refer_teacher"}`.

Model choice — Gemma 2b vs 7b
- Gemma 2b: ~1.2–1.5GB RAM, fast, recommended for Pi.
- Gemma 7b: 4–6GB RAM, slower, risky on Pi.
Start with gemma:2b. Only try 7b on 8GB+ Pi with sufficient cooling.

Retrieval safety rules
- Cosine similarity threshold: top1 < 0.60 → refuse.
- LLM receives ONLY retrieved chunks as context.
- LLM must cite chunk IDs used.
- If uncertain, respond: "I don't know, ask your teacher."

Troubleshooting
- "faiss not installed": `pip install faiss-cpu`
- "Query dimension mismatch": ensure query embedding was produced with the same model as the bundle (check `model.json`).
- Retrieval always returns `refer_teacher`: verify correct bundle, correct query embedding, try smaller k.
- Ollama missing: `ollama pull gemma:2b` and ensure Ollama daemon is running.

Validation example
```bash
python rag_cli.py \
  --bundle ./bundles/class_8_science_en \
  --embed ./tests/q_photosynthesis.json \
  --query "Explain photosynthesis"
```
Expected: retrieval (<5 chunks) and answer citing chunk IDs. No network calls.

Contribution notes
- Never run embeddings on Pi.
- Bundle changes must bump `version.txt`.
- When changing model or chunk strategy, regenerate bundle.
- Always require citation; never return answers without citations.

Contact
AI Lead — Sparsh Sharma  
sparshsharma0825@gmail.com
```// filepath: /Users/sparsh/Documents/Coding/Ravya/ncert-offline-rag/src/pi_runtime/README.md

# Ravya — Raspberry Pi Offline RAG Runtime

This directory contains the offline inference system used by Ravya on the Raspberry Pi.
The Pi runtime does NOT generate embeddings and does NOT connect to Chroma or remote APIs.
It loads FAISS bundles, retrieves top-K chunks, and generates answers using the local Gemma LLM via Ollama.

Core idea
- Ingestion (laptop/cloud): chunking → embedding → Chroma ingestion → FAISS bundle export.
- Pi runtime: load bundle → retrieval → prompt → local LLM answer.
- No embedding computation on Pi. No external inference calls.

Offline guarantees
- Only precomputed embeddings are used.
- Metadata-filtered retrieval (class, subject, language, textbook).
- Similarity threshold gating (default 0.60).
- LLM must cite chunk IDs; if no citation, answer is refused.
- Local Gemma model only via Ollama CLI.
- Safe fallback: REFER_TEACHER.

Directory overview
ncert-offline-rag/
└─ src/
   └─ pi_runtime/
      ├─ load_bundle.py     # Load FAISS bundle into memory
      ├─ retrieve.py        # Vector search against FAISS index
      ├─ rag_generate.py    # Build prompt + call Ollama Gemma
      ├─ rag_answer.py      # Retrieval wrapper → final JSON answer
      ├─ rag_cli.py         # One-shot CLI for teachers / Pi users
      └─ README.md          # This file

Installation — Raspberry Pi (minimal)
1. Create Python env
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (Pi)
Do NOT install transformers or sentence-transformers on Pi.
```bash
pip install numpy faiss-cpu requests
```

3. Install Ollama (ARM)
Follow Ollama's Pi instructions; then pull model:
```bash
ollama pull gemma:2b
```
Recommendation: use Gemma 2b (≈1.2–1.5GB). Gemma 7b is not recommended on most Pi boards.

FAISS bundle requirements
Each subject bundle must contain:
```
class_x_subject_language/
├─ index.faiss
├─ embeddings.bin      # contiguous float32 NxD
├─ chunks.jsonl        # ordered [{"metadata":...,"text":...}]
├─ id_map.pkl          # ordered list of ids (pickle)
├─ model.json          # {"name":..., "dim":...}
├─ manifest.json       # bundle metadata
└─ version.txt
```
Bundles are produced by the ingestion pipeline on laptop/cloud:
```bash
python ingest_pipeline.py ...
```
Copy bundle to Pi:
```bash
scp -r bundle/class_8_science_en pi@raspberrypi:/home/pi/ravya/bundles/
```

1. Loading bundle
```bash
python load_bundle.py /home/pi/ravya/bundles/class_8_science_en
```
Expected output:
```json
{"chunks": 120, "dim": 384}
```
Missing files → fail early.

2. Retrieval
Precompute query embedding on laptop (same model as bundle) and copy to Pi (.json or .npy).

Example:
```bash
python retrieve.py \
  --bundle /home/pi/ravya/bundles/class_8_science_en \
  --embed q.json \
  --k 5
```

Response formats
- OK:
```json
{
  "status": "ok",
  "chunks": [
    {
      "id":"8_science_3_abcd123",
      "rank":0,
      "score":0.874,
      "text":"...",
      "meta":{...}
    }
  ]
}
```
- Not confident:
```json
{"status":"refer_teacher"}
```

3. Local LLM answer
Reads retrieved chunks → prompt → Ollama Gemma.

Example:
```bash
python rag_answer.py \
  --bundle /home/pi/ravya/bundles/class_8_science_en \
  --embed q.json \
  --query "Explain respiration" \
  --model 2b
```

Output on success:
```json
{
  "answer":"Respiration is ...",
  "sources":["8_science_3_a34bd1"]
}
```

If the model does not cite chunk IDs, the system returns:
```json
{"answer":"I don't know, ask your teacher.","sources":[]}
```

4. One-click end-to-end
Teachers/users:
```bash
python rag_cli.py \
  --bundle /home/pi/ravya/bundles/class_8_science_en \
  --embed q.json \
  --query "What is photosynthesis?" \
  --k 5 \
  --model 2b
```
Output either an `answer` object with `sources` or `{"status":"refer_teacher"}`.

Model choice — Gemma 2b vs 7b
- Gemma 2b: ~1.2–1.5GB RAM, fast, recommended for Pi.
- Gemma 7b: 4–6GB RAM, slower, risky on Pi.
Start with gemma:2b. Only try 7b on 8GB+ Pi with sufficient cooling.

Retrieval safety rules
- Cosine similarity threshold: top1 < 0.60 → refuse.
- LLM receives ONLY retrieved chunks as context.
- LLM must cite chunk IDs used.
- If uncertain, respond: "I don't know, ask your teacher."

Troubleshooting
- "faiss not installed": `pip install faiss-cpu`
- "Query dimension mismatch": ensure query embedding was produced with the same model as the bundle (check `model.json`).
- Retrieval always returns `refer_teacher`: verify correct bundle, correct query embedding, try smaller k.
- Ollama missing: `ollama pull gemma:2b` and ensure Ollama daemon is running.

Validation example
```bash
python rag_cli.py \
  --bundle ./bundles/class_8_science_en \
  --embed ./tests/q_photosynthesis.json \
  --query "Explain photosynthesis"
```
Expected: retrieval (<5 chunks) and answer citing chunk IDs. No network calls.

Contribution notes
- Never run embeddings on Pi.
- Bundle changes must bump `version.txt`.
- When changing model or chunk strategy, regenerate bundle.
- Always require citation; never return answers without citations.

Contact
AI Lead — Sparsh Sharma