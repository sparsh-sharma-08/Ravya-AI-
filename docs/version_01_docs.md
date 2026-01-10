# NCERT Offline RAG — Technical Documentation (Single Source of Truth)

Version: 2025.01.00  
Author: Automated patch & engineering notes (copilot)  
Date: 2025-12-12

Table of Contents
1. Executive Summary
2. Original System (Pre-fixes)
   - Original architecture summary
   - Components and intended behavior
   - Student vs Teacher modes (intended)
   - Major limitations and behavior observed
3. Issues Discovered (Detailed)
   - For each issue: root cause, symptoms, module, impact, reproduction
4. Fixes Applied (Chronological + Detailed)
   - Fix #1 — Exporter rewritten
   - Fix #2 — Retrieval / id_map & fallback repaired
   - Fix #3 — Student validator fixed
   - Fix #4 — Teacher mode robustness and JSON handling
   - Fix #5 — CLI improvements and embed handling
   - Fix #6 — End-to-end verification & FAISS index creation
5. Current Architecture (Post-fixes)
   - 5.1 Data ingestion & exporter
   - 5.2 Bundle layout (exact)
   - 5.3 Retrieval pipeline
   - 5.4 RAG answer pipeline
   - 5.5 Student vs Teacher differences (detailed)
6. What Works Now (Verified)
   - Examples (student & teacher)
   - Commands to reproduce verification
   - Sample outputs
7. What Remains / Recommended Roadmap
   - Phase 2 — Completion items
   - Phase 3 — UX & Packaging
   - Phase 4 — Future features
8. Appendix
   - Key commands (exact)
   - Important file list (locations)
   - Small text diagrams
9. Final Summary — Current State of Project

--------------------------------------------------------------------------------
1. Executive Summary
--------------------------------------------------------------------------------

This document records the full state of the NCERT Offline RAG project: what it used to be, what broke, how each breakage was diagnosed and fixed, and the current architecture, behavior and recommended next steps. It is intended as an authoritative, reproducible guide for future contributors, maintainers, or auditors.

In short: the exporter and retrieval pipeline were reworked to produce deterministic, flattened bundles (chunks.jsonl + id_map.pkl) consistent with the RAG consumers. Embeddings are normalized and written to embeddings.bin. A FAISS IndexFlatIP index is built and saved. Student and teacher flows are validated with robust parsing and fallback heuristics. An interactive CLI was added and improved to support persistent mode selection.

--------------------------------------------------------------------------------
2. Original System (Pre-fixes)
--------------------------------------------------------------------------------

2.1 Original architecture summary

- Purpose: Offline RAG for NCERT content. Provide retrieval over local bundles with a QA/teacher-styled pipeline.
- High-level components:
  - Data ingestion / exporter
    - export_bundle_from_data.py (created bundles from input JSONL + embeddings)
  - Stored bundle (expected bundle layout for RAG to consume)
    - chunks.jsonl (expected flattened chunk objects, one per line)
    - id_map.pkl (expected list of flattened chunk dicts)
    - embeddings.bin (normalized embeddings)
    - index.faiss
    - model.json / manifest.json / version.txt
  - Retrieval layer
    - FAISS search (IndexFlatIP over normalized vectors)
    - retrieve() function that loads id_map.pkl and maps indices → chunks
  - rag_answer.py
    - Builds prompts for student & teacher modes
    - Calls model wrapper (Gemma)
    - Extracts JSON blobs for teacher responses and validates them
  - CLI tools
    - rag_answer CLI
    - interactive_cli.py / ask.py (interactive user mode)
  - Embedding generator
    - embed_query.py (used from CLI to generate query embedding JSON)

2.2 Intended system behavior

- Input:
  - data_dir/chapter_1.jsonl (one chunk per line)
  - data_dir/embeddings.npy (N×D numpy array matching number of chunks)
- Exporter:
  - Flatten metadata into top-level fields
  - Compute deterministic id: "<class>_<subject>_<chapter>_<md5[:8]>"
  - Ensure types: class, tokens → int; subject → lowercase string; chapter allowed to be string (title)
  - Normalize embeddings and produce embeddings.bin
  - Build FAISS IndexFlatIP and save to index.faiss
  - Produce id_map.pkl as a list of flattened full chunk objects (not only ids)
  - Produce manifest.json, model.json, version.txt
- Retrieval:
  - Generate query embedding (embed_query.py)
  - Use FAISS IP search on normalized embeddings to get top-k
  - Map index → id_map.pkl entry and present chunk payloads to model
- RAG answer:
  - Student mode: generate concise answer, return sources (chunk ids/hashes)
  - Teacher mode: enforce JSON with "content"/"sources" schema, validate sources align to flattened chunk ids
  - If teacher validation fails -> "refer_teacher" (user told to ask human teacher)

2.3 Major limitations that existed (pre-fixes)
- The exporter produced nested "metadata" objects inside chunks.jsonl, not flat keys expected by retrieve() and other pipeline components.
- id_map.pkl sometimes contained only ids or malformed objects; downstream code expected full flattened dicts.
- Exporter did not always match number of chunks to embeddings; or re-ordering introduced mismatch.
- FAISS index creation was missing or broken in prior variant.
- Teacher validator was strict and failed when model output contained non-strict JSON (or repeated keys).
- CLI was prompting for mode on every question, which was tedious.
- Extracting valid "sources" from model output sometimes failed if the model returned just short hashes or different formats.

2.4 Behavior users were seeing (examples)
- "I'm not sure, refer to your teacher" (refer_teacher) returned often because the teacher validator did not accept model output formats it actually produced.
- chunks.jsonl contained:
  {"metadata": {...}, "text": "..."} — retrieval code expected top-level class/subject/chapter
- id_map.pkl could be a list of ids or partial metadata, causing KeyErrors in retrieve()
- FAISS index missing → retrieve() failing when loading index or returning wrong candidates
- CLI prompting for mode on every question (bad UX) and failing when embed file format variant was used

--------------------------------------------------------------------------------
3. All Issues Discovered (Detailed)
--------------------------------------------------------------------------------

For each issue we include:
- Root cause
- Symptoms
- Responsible module(s)
- How it affected retrieval or model output
- Reproduction commands and detection method

Issue A — Incorrect / outdated bundle format (flattening)
- Root cause: Exporter wrote "metadata" as nested object; flattening logic not applied or inconsistent.
- Symptoms: retrieve() could not find top-level class/subject/chapter fields; models received badly formatted chunks.
- Module: export_bundle_from_data.py
- Effect: RAG pipeline expected flat fields; mismatched keys caused absent metadata, wrong IDs, or fallbacks.
- Reproduce:
  - Inspect bundles: head -n 1 bundles/.../chunks.jsonl
  - Expected flat object, observed nested "metadata": {"metadata": {...}, "text": "..."}
- Detection: Unit tests / runtime failure in retrieve() complaining about missing keys.

Issue B — Malformed id_map.pkl
- Root cause: exporter saved only IDs or wrong structure.
- Symptoms: retrieve() attempted to load id_map.pkl and expected list of dicts; code later attempted to access chunk metadata but got strings.
- Module: export_bundle_from_data.py & retrieve()
- Effect: Index mapping fails; downstream code returns no chunks or crashes.
- Reproduce:
  - python -c "import pickle; print(type(pickle.load(open('bundles/.../id_map.pkl','rb'))[0]))"
- Detection: Code reading id_map.pkl hit AttributeError/TypeError.

Issue C — Metadata inconsistencies (class, chapter, subject mismatches)
- Root cause: inconsistent merging priorities, case handling, and numeric coercion for chapter.
- Symptoms:
  - subject sometimes capitalized; class as string not int; chapter coerced to int where it was a title.
- Module: exporter flattening code
- Effect: IDs differed, retrieval grouping wrong, manifest incorrect.
- Reproduce: Inspect chunks.jsonl for subject casing; check type(method) of chunk["class"].

Issue D — Exporter not flattening correctly / mixing files
- Root cause: earlier attempt scanned recursively for all JSON files; mixing content broke ordering.
- Symptoms: chunk ordering not equal to embeddings rows → shape mismatch or wrong mapping.
- Module: exporter (file scanning logic)
- Effect: embeddings aligned to wrong chunk; incorrect retrievals.
- Reproduce:
  - python src/rag/export_bundle_from_data.py --data-dir data_fixed --out-bundle ...
  - then compare len(chunks) vs embeddings shape.

Issue E — Teacher mode failing due to validator expecting wrong schema
- Root cause: validator strictly required well-formed JSON blob in model output; model sometimes returned repeated keys or plain text or JSON with minor formatting issues.
- Symptoms:
  - Teacher mode returned {"status":"refer_teacher"} even when model had valid answer content.
- Module: rag_answer.py validator and JSON extraction logic
- Effect: Unnecessary fallbacks, poor UX; teacher mode unusable.
- Reproduce:
  - RAG_DEBUG=1 python src/rag/rag_answer.py --bundle ... --embed q.json --query "..." --mode teacher --k 5 --debug
  - Observe "no JSON blob found" in debug output with model raw output printed.

Issue F — retrieve → REFER_TEACHER behavior mismatch
- Root cause: teacher validation strictness + sources mismatching (model returned short hex strings but validator expected full ids).
- Symptoms: sources empty or not recognized; refer_teacher triggered.
- Module: rag_answer.py (_normalize_sources, _validate_teacher_response)
- Effect: Teacher answers rejected.
- Reproduce: See earlier RAG_DEBUG logs.

Issue G — FAISS index mismatches / embeddings handling missing
- Root cause: the exporter variant removed embeddings handling, did not write embeddings.bin, and did not create index.faiss.
- Symptoms: retrieval failing to produce vector search results.
- Module: exporter
- Effect: Retrieval impossible or fallback poor.
- Reproduce:
  - Check for out_bundle/index.faiss presence after running exporter.

Issue H — Student/teacher parsing errors & model returns text without proper JSON
- Root cause: model outputs variety; extraction logic not robust enough.
- Symptoms: JSON extraction failed; refer_teacher returned; student answers OK (more tolerant).
- Module: rag_answer.py
- Effect: Teacher flow brittle.
- Reproduce: Use RAG_DEBUG to see raw model output.

Issue I — Mismatched chunk count between embeddings.npy and chunks
- Root cause: exporter allowed recursive multiple json files or filtered incorrectly.
- Symptoms: Runtime error "Chunk count N != embedding rows M".
- Module: exporter
- Effect: Export aborted or incorrect mapping.
- Reproduce: Run exporter and observe exception.

Each of the above problems was recorded with commands and logs during debugging sessions. The logs and the interactive conversation provide the exact commands used to reproduce failures (see Appendix).

--------------------------------------------------------------------------------
4. What Has Been Fixed (Chronological + Detailed)
--------------------------------------------------------------------------------

This section lists the fixes applied in order, each with before/after comparison and technical details.

Fix #1 — Exporter rewritten (flattening + deterministic id + correct metadata)
- Old behavior:
  - Exporter wrote nested "metadata" objects or inconsistent flattening.
  - In some variants it recursively scanned many JSON files and ignored the required layout.
  - Did not preserve id/hash consistently.
  - Did not handle chapter as string (forced int).
  - Did not write embeddings.bin nor build index.faiss (in one incorrect variant).
- New behavior:
  - Input strictly: data_dir/chapter_1.jsonl and data_dir/embeddings.npy only.
  - Each JSONL line is validated for REQUIRED_FIELDS: text, class, subject, chapter, language, textbook, tokens.
  - Flattening process:
    - Merge metadata/meta into top-level; metadata overrides top-level fields when keys overlap.
    - subject → lowercase string
    - class → int (coerce), tokens → int (coerce)
    - chapter → string (lowercased) — supports titles like "Chemical Reactions and Equations"
    - hash: preserve if present (hash / sha), else compute md5(text)
    - id: deterministic id computed as "{class}_{subject}_{chapter}_{hash8}"
    - Preserve any other metadata keys at top-level (no metadata nesting)
  - Write outputs:
    - chunks.jsonl — one flattened JSON per line (no metadata nesting)
    - id_map.pkl — pickle of list[flattened chunk dicts]
    - embeddings.bin — normalized embeddings (row-major float32 bytes)
    - index.faiss — IndexFlatIP built from normalized embeddings
    - model.json — {"name": "precomputed", "dim": D}
    - manifest.json — uses first chunk metadata:
      {
        "class": <int>,
        "subject": <str>,
        "chapter": <str>,
        "language": <str>,
        "textbook": <str>,
        "chunk_count": N,
        "model": "precomputed",
        "version": "2025.01.00"
      }
    - version.txt — 2025.01.00
- Files changed: src/rag/export_bundle_from_data.py (rewritten)
- Commands used:
  - Export: PYTHONPATH="$PWD" python src/rag/export_bundle_from_data.py --data-dir data_fixed --out-bundle bundles/class_8_science_en
  - Inspect: head -n 1 bundles/class_8_science_en/chunks.jsonl
- Why this fixes the issue:
  - Retrieval and prompt builders expect flat keys. Flattening resolves all missing key errors. Deterministic IDs ensure stable cross-bundle references.

Fix #2 — Retrieval fallback repaired; id_map.pkl format stabilized
- Old behavior:
  - id_map.pkl sometimes saved only ids or mismatched structure.
- New behavior:
  - id_map.pkl always contains the full flattened chunk dicts (list).
  - retrieve() logic uses id_map.pkl as canonical mapping from index to chunk (exact one-to-one mapping with embeddings).
  - Additional fallback uses chunks.jsonl if id_map.pkl missing or malformed.
- Files touched: src/rag/retrieve (internal retrieval functions) and code that loads id_map.pkl.
- Benefit: downstream modules receive full chunk metadata (board, source, tags preserved).

Fix #3 — Student validator fixed
- Old behavior: student validator had a bug (signature/return type mismatch) that caused incorrect strict checks or exceptions.
- New behavior:
  - Student mode validation is tolerant: accepts raw text outputs from model and normalizes them to {"status":"ok","mode":"student","answer": "<text>", "sources": [...]}
  - If model output contains JSON, prefer parsed JSON "answer" or "content" fields.
- Files changed: src/rag/rag_answer.py — validation functions corrected
- Benefit: student mode is robust and returns answers in expected structure.

Fix #4 — Teacher mode implemented robustly (JSON enforcement plus heuristics)
- Old behavior:
  - Strict JSON enforcement with minimal heuristics -> many refer_teacher responses.
- New behavior:
  - Primary path: strict JSON extraction with _extract_json_blob and validation of schema (content & sources).
  - Secondary path: heuristic extraction that attempts to parse answer and sources when strict JSON missing (uses regex heuristics).
  - Source normalization: expand short hex tokens into full chunk ids (match hash or id suffix).
  - RAG_DEBUG env var prints raw model output for debugging.
- Files changed: src/rag/rag_answer.py
- Benefit:
  - Teacher mode still enforces structure, but is resilient to model formatting variations; teacher uses validated "content" and expanded "sources".

Fix #5 — CLI improvements & embed handling
- Old behavior:
  - interactive CLI asked for mode on each question
  - embed generation was strict about the embed file schema (only "embedding")
- New behavior:
  - interactive_cli.py:
    - Mode selected at start; persists until user switches via /mode command
    - Commands: /mode, /quit
    - Calls embed_query.py to produce temporary embed JSON
    - Tolerant embed normalization: accepts keys "embedding", "embeddings", "vector", or raw list; rewrites temporary file to {"embedding": [...]}
  - Output: cleaner UX and fewer errors.
- Files changed: src/rag/interactive_cli.py
- Example usage:
  - PYTHONPATH="$PWD" python src/rag/interactive_cli.py --bundle bundles/class_8_science_en --mode student --k 5
- Benefit: easier interactive usage and consistent embed file format for rag_answer.

Fix #6 — Full successful end-to-end retrieval
- Verified scenario (example):
  - Query: "What happens when magnesium is burned in air?"
  - Student mode returned:
    - Answer: "The magnesium ribbon burns with a dazzling white flame and changes into a white powder called magnesium oxide (MgO)."
    - Sources: chunk id that included the activity description (e.g. '10_science_..._cc5c3e41' or '363c80fb' depending on selection).
  - Teacher mode generated structured lecture notes; sources expanded to full flattened chunk ids.
- Why this matters:
  - Confirmed retrieval uses normalized embeddings and FAISS IndexFlatIP.
  - Confirmed id_map.pkl and chunk alignment are correct.

--------------------------------------------------------------------------------
5. Current Architecture (After fixes)
--------------------------------------------------------------------------------

High-level text diagram:

Client
  └─ interactive_cli.py / CLI / rag_answer.py
      └─ generate query embedding (embed_query.py) -> tmp embed JSON
      └─ rag_answer.get_rag_answer => load bundle + id_map.pkl + index
          ├─ embeddings.bin -> FAISS IndexFlatIP (prebuilt index.faiss used if available)
          ├─ id_map.pkl -> list[flattened_chunks] (index ↔ chunk)
          ├─ chunks.jsonl -> fallback canonical chunk list
          └─ model.json / manifest.json (metadata)
      └─ Prompt constructed (student/teacher)
      └─ Model called (Gemma wrapper)
      └─ Output parsed, validated, normalized
      └─ CLI prints final JSON

5.1 Data ingestion and exporter
- Input expectations:
  - data_dir/chapter_1.jsonl (required; one chunk per line)
  - data_dir/embeddings.npy (required; N×D npy array matching chunk lines)
- Exporter behavior:
  - Validate each JSONL line contains REQUIRED_FIELDS
  - Flatten merged metadata (metadata overrides top-level)
  - Enforce deterministic rules:
    - class & tokens → int
    - subject → lowercased
    - chapter → string lowercased (title allowed)
    - hash → preserve or compute md5(text)
    - id → compute "{class}_{subject}_{chapter}_{hash8}"
  - Normalize embeddings:
    - row-wise L2 normalization, dtype float32
  - Outputs written deterministically to out_bundle

5.2 Bundle layout (exact)

out_bundle/
  chunks.jsonl        # one flat JSON object per line
  id_map.pkl          # pickle.dump(list_of_flattened_chunk_dicts)
  embeddings.bin      # normalized embeddings (float32 bytes), row-major
  index.faiss         # faiss.IndexFlatIP built from normalized embeddings
  model.json          # {"name":"precomputed","dim":D}
  manifest.json       # keys: class, subject, chapter, language, textbook, chunk_count, model, version
  version.txt         # "2025.01.00\n"

Example chunk (one line in chunks.jsonl):
{
  "id":"10_science_chemical reactions and equations_c30d44d7",
  "class":10,
  "subject":"science",
  "chapter":"chemical reactions and equations",
  "language":"en",
  "textbook":"unknown",
  "tokens":83,
  "hash":"c30d44d7022620f11487193b3136de26",
  "text":"When the nature and the identity of the initial substance change..."
  // plus any other metadata keys preserved at top-level (e.g., source, board, tags)
}

5.3 Retrieval pipeline

- Embedding generation:
  - embed_query.py is invoked to produce tmp JSON with key "embedding": [float,...]
  - interactive_cli normalizes accepted embed file variants.
- FAISS search:
  - Use precomputed index.faiss (IndexFlatIP)
  - Query vector must be normalized (the embed generator must match normalization used in exporter)
  - Top-k nearest neighbors are returned (inner-product on normalized vectors)
- Chunk selection:
  - Using id_map.pkl (list), mapping index -> flattened chunk dict is applied to produce retrieved chunks list.
  - Fallback rules:
    - If id_map is missing or malformed, chunks.jsonl is used to rebuild list.
    - If model returns short source tokens, _expand_source_tokens resolves them by matching hash prefixes or id suffixes.
- Debug:
  - RAG_DEBUG=1 prints internal debug information: retrieved chunk id lists, prompt head, model raw output for teacher flows.

5.4 RAG answer pipeline

- Prompt construction:
  - For teacher: strict prompt instructing JSON-only response with keys "content" and "sources"
  - For student: instruction to produce a concise answer and sources
  - Prompt is augmented with retrieved chunks content (text)
- Model call wrapper:
  - Handles model selection (gemma:2b)
  - Returns raw text output; rag_answer attempts to extract JSON
- JSON extraction:
  - _extract_json_blob finds a JSON substring in raw model output (if present)
  - For teacher: validate JSON schema via _validate_teacher_response
  - For student: tolerate raw text; prefer parsed JSON if present
- Validation rules:
  - Teacher: must have content (non-empty string) and sources (non-empty list). Sources normalized and expanded.
  - Student: non-empty answer accepted; sources normalized.
- Fallback:
  - If teacher parse fails:
    - Attempt heuristic extraction (regex-based) to pull answer & sources
    - Expand short sources to full chunk ids using retrieved_chunks
    - If failure still, return {"status":"refer_teacher"}.

5.5 Student vs Teacher flow differences

- Student:
  - Tolerant parsing
  - Returns concise answer string and sources (may be short hashes)
  - Good for direct Q&A with minimal format constraints

- Teacher:
  - Strict (but with robust fallback) JSON format requirement
  - Produces structured teaching materials with sections (Overview, Objectives, Stepwise explanation, etc.)
  - Expects sources list to contain valid chunk ids (full id strings preferred)
  - On strict failure: refer_teacher OR heuristic extraction accepted (to avoid blocking)

--------------------------------------------------------------------------------
6. What Works Now (Verified)
--------------------------------------------------------------------------------

6.1 Confirmed features

- Exporter:
  - Accepts data_dir/chapter_1.jsonl + embeddings.npy
  - Produces flattened chunks.jsonl, id_map.pkl, embeddings.bin, index.faiss, model.json, manifest.json, version.txt
  - Handles string chapter names correctly
- Embeddings:
  - Normalized L2, saved in embeddings.bin, index built via FAISS IndexFlatIP
- Retrieval:
  - FAISS search returns correct top-k
  - id_map mapping matches embeddings order
- Student mode:
  - Returns plain answer and sources
  - Example:
    {
      "status":"ok",
      "mode":"student",
      "answer":"A chemical reaction is when the nature and the identity of the initial substance change...",
      "sources":["363c80fb"]
    }
- Teacher mode:
  - Returns validated JSON with "content" and "sources" where possible
  - Heuristic extraction improves robustness when model formatting deviates
- CLI interactive mode:
  - Single mode selection at start; can switch with /mode
  - Tolerant embed normalization
  - RAG_DEBUG prints raw model output for dev debugging

6.2 Example (End-to-end verification)
- Export bundle:
  PYTHONPATH="$PWD" python src/rag/export_bundle_from_data.py --data-dir data_fixed --out-bundle bundles/class_8_science_en

- Inspect (first chunk):
  head -n 1 bundles/class_8_science_en/chunks.jsonl
  python -c "import json; print(json.loads(open('bundles/class_8_science_en/chunks.jsonl').read().splitlines()[0]))"

- Interactive student query:
  PYTHONPATH="$PWD" python src/rag/interactive_cli.py --bundle bundles/class_8_science_en --mode student --k 5
  Question: What is a chemical reaction?
  -> Returned concise answer with source id.

- Interactive teacher query:
  PYTHONPATH="$PWD" python src/rag/interactive_cli.py --bundle bundles/class_8_science_en --mode teacher --k 5
  Question: What is a chemical reaction?
  -> Returned JSON content + full chunk ids in "sources".

- Direct debug run for teacher:
  PYTHONPATH="$PWD" RAG_DEBUG=1 python src/rag/rag_answer.py --bundle bundles/class_8_science_en --embed q.json --query "what is a chemical reaction" --mode teacher --k 5 --debug

6.3 Sample outputs (already observed)
- Student:
  {"status":"ok","mode":"student","answer":"A chemical reaction is when the nature and the identity of the initial substance change...","sources":["363c80fb"]}

- Teacher:
  {"status":"ok","mode":"teacher","content":"Chemical reactions involve a change ...","sources":["10_science_chemical reactions and equations_c30d44d7","10_science_chemical reactions and equations_dfcde531"]}

--------------------------------------------------------------------------------
7. What still needs to be done (Recommended next phases)
--------------------------------------------------------------------------------

Phase 2 — Completion
- Expand teacher prompt templates to cover different syllabus goals (exams, practicals)
- Stronger schema validation & automated test vectors (unit tests for teacher JSON output)
- Add an automatic bundle validator that checks:
  - chunk_count == embeddings rows
  - presence of all required fields and types
  - index.faiss readability & consistency with embeddings.bin

Phase 3 — UX + Packaging
- Improve CLI:
  - Allow fallback embedding generation (if embed_query not present, call remote service)
  - Add --preload-bundle option
  - Add logging levels (INFO/DEBUG)
- Package the exporter into a pip-installable CLI
- Create simple Web UI for interactive usage

Phase 4 — Future features
- Multi-bundle search (cross-bundle retrieval)
- Better source provenance (passage offsets/page numbers)
- Evaluation harness for teacher answers (automated scoring)
- On-device caching & streaming large bundles
- Lightweight mobile UI or local electron app

--------------------------------------------------------------------------------
8. Appendix
--------------------------------------------------------------------------------

8.1 Exact Commands Executed / Useful for Repro
- Export:
  PYTHONPATH="$PWD" python src/rag/export_bundle_from_data.py --data-dir data_fixed --out-bundle bundles/class_8_science_en

- Inspect first chunk:
  head -n 1 bundles/class_8_science_en/chunks.jsonl
  python -c "import json;print(json.loads(open('bundles/class_8_science_en/chunks.jsonl').read().splitlines()[0]))"

- Check id_map:
  python - <<'PY'
  import pickle, json
  m=pickle.load(open('bundles/class_8_science_en/id_map.pkl','rb'))
  print('id_map length=', len(m))
  print(json.dumps(m[0], ensure_ascii=False, indent=2))
  PY

- Check embeddings:
  python - <<'PY'
  import numpy as np, os
  a=np.load('data_fixed/embeddings.npy')
  print('embeddings.npy shape=', a.shape)
  print('embeddings.bin size=', os.path.getsize('bundles/class_8_science_en/embeddings.bin'))
  PY

- Run interactive CLI:
  PYTHONPATH="$PWD" python src/rag/interactive_cli.py --bundle bundles/class_8_science_en --mode student --k 5

- Full teacher debug:
  PYTHONPATH="$PWD" RAG_DEBUG=1 python src/rag/rag_answer.py --bundle bundles/class_8_science_en --embed q.json --query "what is a chemical reaction" --mode teacher --k 5 --debug

8.2 Important Files (locations)
- Exporter: src/rag/export_bundle_from_data.py
- Retrieval & RAG: src/rag/rag_answer.py
- Interactive CLI: src/rag/interactive_cli.py
- Embed script: src/pi_runtime/embed_query.py (expected)
- Bundle root: bundles/class_8_science_en/

8.3 Text diagrams

Data flow (simplified):

data_fixed/
  ├─ chapter_1.jsonl  ──> exporter --> chunks.jsonl
  └─ embeddings.npy    ──> exporter --> embeddings.bin + index.faiss

Query flow:

User -> interactive_cli -> embed_query.py -> tmp embed.json -> rag_answer.get_rag_answer
  -> load bundles (id_map.pkl, index.faiss)
  -> FAISS search -> top-k ids
  -> build prompt -> call model -> parse & validate -> return result

--------------------------------------------------------------------------------
9. Final Summary — Current State of Project
--------------------------------------------------------------------------------

- The exporter has been rewritten to follow strict input expectations and produce deterministic, flattened bundle outputs consistent with the RAG pipeline.
- Embeddings are normalized and saved; FAISS index is built and saved.
- id_map.pkl now consistently contains full flattened chunk objects in order matching embeddings rows.
- Retrieval uses the prebuilt FAISS index and maps indices to chunk metadata reliably.
- RAG answer pipeline includes robust parsing for both student and teacher modes; teacher mode retains strictness but includes heuristic fallback to avoid unnecessary refer_teacher responses.
- Interactive CLI supports persistent mode selection and tolerant embed generation.
- End-to-end verification (example: magnesium burning) confirms the pipeline produces correct retrievals and model responses.

This document should be placed under RAVYA/docs and used as the canonical technical reference for the project state after the fixes described above.

If you want I can:
- Add a machine-readable bundle validator script under RAVYA/scripts
- Add unit tests for exporter flattening and rag_answer validators
- Generate a short slide-style README for onboarding new devs

---- End of document ----
```# filepath: RAVYA/docs/NCERT_Offline_RAG_Documentation.md
# NCERT Offline RAG — Technical Documentation (Single Source of Truth)

Version: 2025.01.00  
Author: Automated patch & engineering notes (copilot)  
Date: 2025-12-12

Table of Contents
1. Executive Summary
2. Original System (Pre-fixes)
   - Original architecture summary
   - Components and intended behavior
   - Student vs Teacher modes (intended)
   - Major limitations and behavior observed
3. Issues Discovered (Detailed)
   - For each issue: root cause, symptoms, module, impact, reproduction
4. Fixes Applied (Chronological + Detailed)
   - Fix #1 — Exporter rewritten
   - Fix #2 — Retrieval / id_map & fallback repaired
   - Fix #3 — Student validator fixed
   - Fix #4 — Teacher mode robustness and JSON handling
   - Fix #5 — CLI improvements and embed handling
   - Fix #6 — End-to-end verification & FAISS index creation
5. Current Architecture (Post-fixes)
   - 5.1 Data ingestion & exporter
   - 5.2 Bundle layout (exact)
   - 5.3 Retrieval pipeline
   - 5.4 RAG answer pipeline
   - 5.5 Student vs Teacher differences (detailed)
6. What Works Now (Verified)
   - Examples (student & teacher)
   - Commands to reproduce verification
   - Sample outputs
7. What Remains / Recommended Roadmap
   - Phase 2 — Completion items
   - Phase 3 — UX & Packaging
   - Phase 4 — Future features
8. Appendix
   - Key commands (exact)
   - Important file list (locations)
   - Small text diagrams
9. Final Summary — Current State of Project

--------------------------------------------------------------------------------
1. Executive Summary
--------------------------------------------------------------------------------

This document records the full state of the NCERT Offline RAG project: what it used to be, what broke, how each breakage was diagnosed and fixed, and the current architecture, behavior and recommended next steps. It is intended as an authoritative, reproducible guide for future contributors, maintainers, or auditors.

In short: the exporter and retrieval pipeline were reworked to produce deterministic, flattened bundles (chunks.jsonl + id_map.pkl) consistent with the RAG consumers. Embeddings are normalized and written to embeddings.bin. A FAISS IndexFlatIP index is built and saved. Student and teacher flows are validated with robust parsing and fallback heuristics. An interactive CLI was added and improved to support persistent mode selection.

--------------------------------------------------------------------------------
2. Original System (Pre-fixes)
--------------------------------------------------------------------------------

2.1 Original architecture summary

- Purpose: Offline RAG for NCERT content. Provide retrieval over local bundles with a QA/teacher-styled pipeline.
- High-level components:
  - Data ingestion / exporter
    - export_bundle_from_data.py (created bundles from input JSONL + embeddings)
  - Stored bundle (expected bundle layout for RAG to consume)
    - chunks.jsonl (expected flattened chunk objects, one per line)
    - id_map.pkl (expected list of flattened chunk dicts)
    - embeddings.bin (normalized embeddings)
    - index.faiss
    - model.json / manifest.json / version.txt
  - Retrieval layer
    - FAISS search (IndexFlatIP over normalized vectors)
    - retrieve() function that loads id_map.pkl and maps indices → chunks
  - rag_answer.py
    - Builds prompts for student & teacher modes
    - Calls model wrapper (Gemma)
    - Extracts JSON blobs for teacher responses and validates them
  - CLI tools
    - rag_answer CLI
    - interactive_cli.py / ask.py (interactive user mode)
  - Embedding generator
    - embed_query.py (used from CLI to generate query embedding JSON)

2.2 Intended system behavior

- Input:
  - data_dir/chapter_1.jsonl (one chunk per line)
  - data_dir/embeddings.npy (N×D numpy array matching number of chunks)
- Exporter:
  - Flatten metadata into top-level fields
  - Compute deterministic id: "<class>_<subject>_<chapter>_<md5[:8]>"
  - Ensure types: class, tokens → int; subject → lowercase string; chapter allowed to be string (title)
  - Normalize embeddings and produce embeddings.bin
  - Build FAISS IndexFlatIP and save to index.faiss
  - Produce id_map.pkl as a list of flattened full chunk objects (not only ids)
  - Produce manifest.json, model.json, version.txt
- Retrieval:
  - Generate query embedding (embed_query.py)
  - Use FAISS IP search on normalized embeddings to get top-k
  - Map index → id_map.pkl entry and present chunk payloads to model
- RAG answer:
  - Student mode: generate concise answer, return sources (chunk ids/hashes)
  - Teacher mode: enforce JSON with "content"/"sources" schema, validate sources align to flattened chunk ids
  - If teacher validation fails -> "refer_teacher" (user told to ask human teacher)

2.3 Major limitations that existed (pre-fixes)
- The exporter produced nested "metadata" objects inside chunks.jsonl, not flat keys expected by retrieve() and other pipeline components.
- id_map.pkl sometimes contained only ids or malformed objects; downstream code expected full flattened dicts.
- Exporter did not always match number of chunks to embeddings; or re-ordering introduced mismatch.
- FAISS index creation was missing or broken in prior variant.
- Teacher validator was strict and failed when model output contained non-strict JSON (or repeated keys).
- CLI was prompting for mode on every question, which was tedious.
- Extracting valid "sources" from model output sometimes failed if the model returned just short hashes or different formats.

2.4 Behavior users were seeing (examples)
- "I'm not sure, refer to your teacher" (refer_teacher) returned often because the teacher validator did not accept model output formats it actually produced.
- chunks.jsonl contained:
  {"metadata": {...}, "text": "..."} — retrieval code expected top-level class/subject/chapter
- id_map.pkl could be a list of ids or partial metadata, causing KeyErrors in retrieve()
- FAISS index missing → retrieve() failing when loading index or returning wrong candidates
- CLI prompting for mode on every question (bad UX) and failing when embed file format variant was used

--------------------------------------------------------------------------------
3. All Issues Discovered (Detailed)
--------------------------------------------------------------------------------

For each issue we include:
- Root cause
- Symptoms
- Responsible module(s)
- How it affected retrieval or model output
- Reproduction commands and detection method

Issue A — Incorrect / outdated bundle format (flattening)
- Root cause: Exporter wrote "metadata" as nested object; flattening logic not applied or inconsistent.
- Symptoms: retrieve() could not find top-level class/subject/chapter fields; models received badly formatted chunks.
- Module: export_bundle_from_data.py
- Effect: RAG pipeline expected flat fields; mismatched keys caused absent metadata, wrong IDs, or fallbacks.
- Reproduce:
  - Inspect bundles: head -n 1 bundles/.../chunks.jsonl
  - Expected flat object, observed nested "metadata": {"metadata": {...}, "text": "..."}
- Detection: Unit tests / runtime failure in retrieve() complaining about missing keys.

Issue B — Malformed id_map.pkl
- Root cause: exporter saved only IDs or wrong structure.
- Symptoms: retrieve() attempted to load id_map.pkl and expected list of dicts; code later attempted to access chunk metadata but got strings.
- Module: export_bundle_from_data.py & retrieve()
- Effect: Index mapping fails; downstream code returns no chunks or crashes.
- Reproduce:
  - python -c "import pickle; print(type(pickle.load(open('bundles/.../id_map.pkl','rb'))[0]))"
- Detection: Code reading id_map.pkl hit AttributeError/TypeError.

Issue C — Metadata inconsistencies (class, chapter, subject mismatches)
- Root cause: inconsistent merging priorities, case handling, and numeric coercion for chapter.
- Symptoms:
  - subject sometimes capitalized; class as string not int; chapter coerced to int where it was a title.
- Module: exporter flattening code
- Effect: IDs differed, retrieval grouping wrong, manifest incorrect.
- Reproduce: Inspect chunks.jsonl for subject casing; check type(method) of chunk["class"].

Issue D — Exporter not flattening correctly / mixing files
- Root cause: earlier attempt scanned recursively for all JSON files; mixing content broke ordering.
- Symptoms: chunk ordering not equal to embeddings rows → shape mismatch or wrong mapping.
- Module: exporter (file scanning logic)
- Effect: embeddings aligned to wrong chunk; incorrect retrievals.
- Reproduce:
  - python src/rag/export_bundle_from_data.py --data-dir data_fixed --out-bundle ...
  - then compare len(chunks) vs embeddings shape.

Issue E — Teacher mode failing due to validator expecting wrong schema
- Root cause: validator strictly required well-formed JSON blob in model output; model sometimes returned repeated keys or plain text or JSON with minor formatting issues.
- Symptoms:
  - Teacher mode returned {"status":"refer_teacher"} even when model had valid answer content.
- Module: rag_answer.py validator and JSON extraction logic
- Effect: Unnecessary fallbacks, poor UX; teacher mode unusable.
- Reproduce:
  - RAG_DEBUG=1 python src/rag/rag_answer.py --bundle ... --embed q.json --query "..." --mode teacher --k 5 --debug
  - Observe "no JSON blob found" in debug output with model raw output printed.

Issue F — retrieve → REFER_TEACHER behavior mismatch
- Root cause: teacher validation strictness + sources mismatching (model returned short hex strings but validator expected full ids).
- Symptoms: sources empty or not recognized; refer_teacher triggered.
- Module: rag_answer.py (_normalize_sources, _validate_teacher_response)
- Effect: Teacher answers rejected.
- Reproduce: See earlier RAG_DEBUG logs.

Issue G — FAISS index mismatches / embeddings handling missing
- Root cause: the exporter variant removed embeddings handling, did not write embeddings.bin, and did not create index.faiss.
- Symptoms: retrieval failing to produce vector search results.
- Module: exporter
- Effect: Retrieval impossible or fallback poor.
- Reproduce:
  - Check for out_bundle/index.faiss presence after running exporter.

Issue H — Student/teacher parsing errors & model returns text without proper JSON
- Root cause: model outputs variety; extraction logic not robust enough.
- Symptoms: JSON extraction failed; refer_teacher returned; student answers OK (more tolerant).
- Module: rag_answer.py
- Effect: Teacher flow brittle.
- Reproduce: Use RAG_DEBUG to see raw model output.

Issue I — Mismatched chunk count between embeddings.npy and chunks
- Root cause: exporter allowed recursive multiple json files or filtered incorrectly.
- Symptoms: Runtime error "Chunk count N != embedding rows M".
- Module: exporter
- Effect: Export aborted or incorrect mapping.
- Reproduce: Run exporter and observe exception.

Each of the above problems was recorded with commands and logs during debugging sessions. The logs and the interactive conversation provide the exact commands used to reproduce failures (see Appendix).

--------------------------------------------------------------------------------
4. What Has Been Fixed (Chronological + Detailed)
--------------------------------------------------------------------------------

This section lists the fixes applied in order, each with before/after comparison and technical details.

Fix #1 — Exporter rewritten (flattening + deterministic id + correct metadata)
- Old behavior:
  - Exporter wrote nested "metadata" objects or inconsistent flattening.
  - In some variants it recursively scanned many JSON files and ignored the required layout.
  - Did not preserve id/hash consistently.
  - Did not handle chapter as string (forced int).
  - Did not write embeddings.bin nor build index.faiss (in one incorrect variant).
- New behavior:
  - Input strictly: data_dir/chapter_1.jsonl and data_dir/embeddings.npy only.
  - Each JSONL line is validated for REQUIRED_FIELDS: text, class, subject, chapter, language, textbook, tokens.
  - Flattening process:
    - Merge metadata/meta into top-level; metadata overrides top-level fields when keys overlap.
    - subject → lowercase string
    - class → int (coerce), tokens → int (coerce)
    - chapter → string (lowercased) — supports titles like "Chemical Reactions and Equations"
    - hash: preserve if present (hash / sha), else compute md5(text)
    - id: deterministic id computed as "{class}_{subject}_{chapter}_{hash8}"
    - Preserve any other metadata keys at top-level (no metadata nesting)
  - Write outputs:
    - chunks.jsonl — one flattened JSON per line (no metadata nesting)
    - id_map.pkl — pickle of list[flattened chunk dicts]
    - embeddings.bin — normalized embeddings (row-major float32 bytes)
    - index.faiss — IndexFlatIP built from normalized embeddings
    - model.json — {"name": "precomputed", "dim": D}
    - manifest.json — uses first chunk metadata:
      {
        "class": <int>,
        "subject": <str>,
        "chapter": <str>,
        "language": <str>,
        "textbook": <str>,
        "chunk_count": N,
        "model": "precomputed",
        "version": "2025.01.00"
      }
    - version.txt — 2025.01.00
- Files changed: src/rag/export_bundle_from_data.py (rewritten)
- Commands used:
  - Export: PYTHONPATH="$PWD" python src/rag/export_bundle_from_data.py --data-dir data_fixed --out-bundle bundles/class_8_science_en
  - Inspect: head -n 1 bundles/class_8_science_en/chunks.jsonl
- Why this fixes the issue:
  - Retrieval and prompt builders expect flat keys. Flattening resolves all missing key errors. Deterministic IDs ensure stable cross-bundle references.

Fix #2 — Retrieval fallback repaired; id_map.pkl format stabilized
- Old behavior:
  - id_map.pkl sometimes saved only ids or mismatched structure.
- New behavior:
  - id_map.pkl always contains the full flattened chunk dicts (list).
  - retrieve() logic uses id_map.pkl as canonical mapping from index to chunk (exact one-to-one mapping with embeddings).
  - Additional fallback uses chunks.jsonl if id_map.pkl missing or malformed.
- Files touched: src/rag/retrieve (internal retrieval functions) and code that loads id_map.pkl.
- Benefit: downstream modules receive full chunk metadata (board, source, tags preserved).

Fix #3 — Student validator fixed
- Old behavior: student validator had a bug (signature/return type mismatch) that caused incorrect strict checks or exceptions.
- New behavior:
  - Student mode validation is tolerant: accepts raw text outputs from model and normalizes them to {"status":"ok","mode":"student","answer": "<text>", "sources": [...]}
  - If model output contains JSON, prefer parsed JSON "answer" or "content" fields.
- Files changed: src/rag/rag_answer.py — validation functions corrected
- Benefit: student mode is robust and returns answers in expected structure.

Fix #4 — Teacher mode implemented robustly (JSON enforcement plus heuristics)
- Old behavior:
  - Strict JSON enforcement with minimal heuristics -> many refer_teacher responses.
- New behavior:
  - Primary path: strict JSON extraction with _extract_json_blob and validation of schema (content & sources).
  - Secondary path: heuristic extraction that attempts to parse answer and sources when strict JSON missing (uses regex heuristics).
  - Source normalization: expand short hex tokens into full chunk ids (match hash or id suffix).
  - RAG_DEBUG env var prints raw model output for debugging.
- Files changed: src/rag/rag_answer.py
- Benefit:
  - Teacher mode still enforces structure, but is resilient to model formatting variations; teacher uses validated "content" and expanded "sources".

Fix #5 — CLI improvements & embed handling
- Old behavior:
  - interactive CLI asked for mode on each question
  - embed generation was strict about the embed file schema (only "embedding")
- New behavior:
  - interactive_cli.py:
    - Mode selected at start; persists until user switches via /mode command
    - Commands: /mode, /quit
    - Calls embed_query.py to produce temporary embed JSON
    - Tolerant embed normalization: accepts keys "embedding", "embeddings", "vector", or raw list; rewrites temporary file to {"embedding": [...]}
  - Output: cleaner UX and fewer errors.
- Files changed: src/rag/interactive_cli.py
- Example usage:
  - PYTHONPATH="$PWD" python src/rag/interactive_cli.py --bundle bundles/class_8_science_en --mode student --k 5
- Benefit: easier interactive usage and consistent embed file format for rag_answer.

Fix #6 — Full successful end-to-end retrieval
- Verified scenario (example):
  - Query: "What happens when magnesium is burned in air?"
  - Student mode returned:
    - Answer: "The magnesium ribbon burns with a dazzling white flame and changes into a white powder called magnesium oxide (MgO)."
    - Sources: chunk id that included the activity description (e.g. '10_science_..._cc5c3e41' or '363c80fb' depending on selection).
  - Teacher mode generated structured lecture notes; sources expanded to full flattened chunk ids.
- Why this matters:
  - Confirmed retrieval uses normalized embeddings and FAISS IndexFlatIP.
  - Confirmed id_map.pkl and chunk alignment are correct.

--------------------------------------------------------------------------------
5. Current Architecture (After fixes)
--------------------------------------------------------------------------------

High-level text diagram:

Client
  └─ interactive_cli.py / CLI / rag_answer.py
      └─ generate query embedding (embed_query.py) -> tmp embed JSON
      └─ rag_answer.get_rag_answer => load bundle + id_map.pkl + index
          ├─ embeddings.bin -> FAISS IndexFlatIP (prebuilt index.faiss used if available)
          ├─ id_map.pkl -> list[flattened_chunks] (index ↔ chunk)
          ├─ chunks.jsonl -> fallback canonical chunk list
          └─ model.json / manifest.json (metadata)
      └─ Prompt constructed (student/teacher)
      └─ Model called (Gemma wrapper)
      └─ Output parsed, validated, normalized
      └─ CLI prints final JSON

5.1 Data ingestion and exporter
- Input expectations:
  - data_dir/chapter_1.jsonl (required; one chunk per line)
  - data_dir/embeddings.npy (required; N×D npy array matching chunk lines)
- Exporter behavior:
  - Validate each JSONL line contains REQUIRED_FIELDS
  - Flatten merged metadata (metadata overrides top-level)
  - Enforce deterministic rules:
    - class & tokens → int
    - subject → lowercased
    - chapter → string lowercased (title allowed)
    - hash → preserve or compute md5(text)
    - id → compute "{class}_{subject}_{chapter}_{hash8}"
  - Normalize embeddings:
    - row-wise L2 normalization, dtype float32
  - Outputs written deterministically to out_bundle

5.2 Bundle layout (exact)

out_bundle/
  chunks.jsonl        # one flat JSON object per line
  id_map.pkl          # pickle.dump(list_of_flattened_chunk_dicts)
  embeddings.bin      # normalized embeddings (float32 bytes), row-major
  index.faiss         # faiss.IndexFlatIP built from normalized embeddings
  model.json          # {"name":"precomputed","dim":D}
  manifest.json       # keys: class, subject, chapter, language, textbook, chunk_count, model, version
  version.txt         # "2025.01.00\n"

Example chunk (one line in chunks.jsonl):
{
  "id":"10_science_chemical reactions and equations_c30d44d7",
  "class":10,
  "subject":"science",
  "chapter":"chemical reactions and equations",
  "language":"en",
  "textbook":"unknown",
  "tokens":83,
  "hash":"c30d44d7022620f11487193b3136de26",
  "text":"When the nature and the identity of the initial substance change..."
  // plus any other metadata keys preserved at top-level (e.g., source, board, tags)
}

5.3 Retrieval pipeline

- Embedding generation:
  - embed_query.py is invoked to produce tmp JSON with key "embedding": [float,...]
  - interactive_cli normalizes accepted embed file variants.
- FAISS search:
  - Use precomputed index.faiss (IndexFlatIP)
  - Query vector must be normalized (the embed generator must match normalization used in exporter)
  - Top-k nearest neighbors are returned (inner-product on normalized vectors)
- Chunk selection:
  - Using id_map.pkl (list), mapping index -> flattened chunk dict is applied to produce retrieved chunks list.
  - Fallback rules:
    - If id_map is missing or malformed, chunks.jsonl is used to rebuild list.
    - If model returns short source tokens, _expand_source_tokens resolves them by matching hash prefixes or id suffixes.
- Debug:
  - RAG_DEBUG=1 prints internal debug information: retrieved chunk id lists, prompt head, model raw output for teacher flows.

5.4 RAG answer pipeline

- Prompt construction:
  - For teacher: strict prompt instructing JSON-only response with keys "content" and "sources"
  - For student: instruction to produce a concise answer and sources
  - Prompt is augmented with retrieved chunks content (text)
- Model call wrapper:
  - Handles model selection (gemma:2b)
  - Returns raw text output; rag_answer attempts to extract JSON
- JSON extraction:
  - _extract_json_blob finds a JSON substring in raw model output (if present)
  - For teacher: validate JSON schema via _validate_teacher_response
  - For student: tolerate raw text; prefer parsed JSON if present
- Validation rules:
  - Teacher: must have content (non-empty string) and sources (non-empty list). Sources normalized and expanded.
  - Student: non-empty answer accepted; sources normalized.
- Fallback:
  - If teacher parse fails:
    - Attempt heuristic extraction (regex-based) to pull answer & sources
    - Expand short sources to full chunk ids using retrieved_chunks
    - If failure still, return {"status":"refer_teacher"}.

5.5 Student vs Teacher flow differences

- Student:
  - Tolerant parsing
  - Returns concise answer string and sources (may be short hashes)
  - Good for direct Q&A with minimal format constraints

- Teacher:
  - Strict (but with robust fallback) JSON format requirement
  - Produces structured teaching materials with sections (Overview, Objectives, Stepwise explanation, etc.)
  - Expects sources list to contain valid chunk ids (full id strings preferred)
  - On strict failure: refer_teacher OR heuristic extraction accepted (to avoid blocking)

--------------------------------------------------------------------------------
6. What Works Now (Verified)
--------------------------------------------------------------------------------

6.1 Confirmed features

- Exporter:
  - Accepts data_dir/chapter_1.jsonl + embeddings.npy
  - Produces flattened chunks.jsonl, id_map.pkl, embeddings.bin, index.faiss, model.json, manifest.json, version.txt
  - Handles string chapter names correctly
- Embeddings:
  - Normalized L2, saved in embeddings.bin, index built via FAISS IndexFlatIP
- Retrieval:
  - FAISS search returns correct top-k
  - id_map mapping matches embeddings order
- Student mode:
  - Returns plain answer and sources
  - Example:
    {
      "status":"ok",
      "mode":"student",
      "answer":"A chemical reaction is when the nature and the identity of the initial substance change...",
      "sources":["363c80fb"]
    }
- Teacher mode:
  - Returns validated JSON with "content" and "sources" where possible
  - Heuristic extraction improves robustness when model formatting deviates
- CLI interactive mode:
  - Single mode selection at start; can switch with /mode
  - Tolerant embed normalization
  - RAG_DEBUG prints raw model output for dev debugging

6.2 Example (End-to-end verification)
- Export bundle:
  PYTHONPATH="$PWD" python src/rag/export_bundle_from_data.py --data-dir data_fixed --out-bundle bundles/class_8_science_en

- Inspect (first chunk):
  head -n 1 bundles/class_8_science_en/chunks.jsonl
  python -c "import json; print(json.loads(open('bundles/class_8_science_en/chunks.jsonl').read().splitlines()[0]))"

- Interactive student query:
  PYTHONPATH="$PWD" python src/rag/interactive_cli.py --bundle bundles/class_8_science_en --mode student --k 5
  Question: What is a chemical reaction?
  -> Returned concise answer with source id.

- Interactive teacher query:
  PYTHONPATH="$PWD" python src/rag/interactive_cli.py --bundle bundles/class_8_science_en --mode teacher --k 5
  Question: What is a chemical reaction?
  -> Returned JSON content + full chunk ids in "sources".

- Direct debug run for teacher:
  PYTHONPATH="$PWD" RAG_DEBUG=1 python src/rag/rag_answer.py --bundle bundles/class_8_science_en --embed q.json --query "what is a chemical reaction" --mode teacher --k 5 --debug

6.3 Sample outputs (already observed)
- Student:
  {"status":"ok","mode":"student","answer":"A chemical reaction is when the nature and the identity of the initial substance change...","sources":["363c80fb"]}

- Teacher:
  {"status":"ok","mode":"teacher","content":"Chemical reactions involve a change ...","sources":["10_science_chemical reactions and equations_c30d44d7","10_science_chemical reactions and equations_dfcde531"]}

--------------------------------------------------------------------------------
7. What still needs to be done (Recommended next phases)
--------------------------------------------------------------------------------

Phase 2 — Completion
- Expand teacher prompt templates to cover different syllabus goals (exams, practicals)
- Stronger schema validation & automated test vectors (unit tests for teacher JSON output)
- Add an automatic bundle validator that checks:
  - chunk_count == embeddings rows
  - presence of all required fields and types
  - index.faiss readability & consistency with embeddings.bin

Phase 3 — UX + Packaging
- Improve CLI:
  - Allow fallback embedding generation (if embed_query not present, call remote service)
  - Add --preload-bundle option
  - Add logging levels (INFO/DEBUG)
- Package the exporter into a pip-installable CLI
- Create simple Web UI for interactive usage

Phase 4 — Future features
- Multi-bundle search (cross-bundle retrieval)
- Better source provenance (passage offsets/page numbers)
- Evaluation harness for teacher answers (automated scoring)
- On-device caching & streaming large bundles
- Lightweight mobile UI or local electron app

--------------------------------------------------------------------------------
8. Appendix
--------------------------------------------------------------------------------

8.1 Exact Commands Executed / Useful for Repro
- Export:
  PYTHONPATH="$PWD" python src/rag/export_bundle_from_data.py --data-dir data_fixed --out-bundle bundles/class_8_science_en

- Inspect first chunk:
  head -n 1 bundles/class_8_science_en/chunks.jsonl
  python -c "import json;print(json.loads(open('bundles/class_8_science_en/chunks.jsonl').read().splitlines()[0]))"

- Check id_map:
  python - <<'PY'
  import pickle, json
  m=pickle.load(open('bundles/class_8_science_en/id_map.pkl','rb'))
  print('id_map length=', len(m))
  print(json.dumps(m[0], ensure_ascii=False, indent=2))
  PY

- Check embeddings:
  python - <<'PY'
  import numpy as np, os
  a=np.load('data_fixed/embeddings.npy')
  print('embeddings.npy shape=', a.shape)
  print('embeddings.bin size=', os.path.getsize('bundles/class_8_science_en/embeddings.bin'))
  PY

- Run interactive CLI:
  PYTHONPATH="$PWD" python src/rag/interactive_cli.py --bundle bundles/class_8_science_en --mode student --k 5

- Full teacher debug:
  PYTHONPATH="$PWD" RAG_DEBUG=1 python src/rag/rag_answer.py --bundle bundles/class_8_science_en --embed q.json --query "what is a chemical reaction" --mode teacher --k 5 --debug

8.2 Important Files (locations)
- Exporter: src/rag/export_bundle_from_data.py
- Retrieval & RAG: src/rag/rag_answer.py
- Interactive CLI: src/rag/interactive_cli.py
- Embed script: src/pi_runtime/embed_query.py (expected)
- Bundle root: bundles/class_8_science_en/

8.3 Text diagrams

Data flow (simplified):

data_fixed/
  ├─ chapter_1.jsonl  ──> exporter --> chunks.jsonl
  └─ embeddings.npy    ──> exporter --> embeddings.bin + index.faiss

Query flow:

User -> interactive_cli -> embed_query.py -> tmp embed.json -> rag_answer.get_rag_answer
  -> load bundles (id_map.pkl, index.faiss)
  -> FAISS search -> top-k ids
  -> build prompt -> call model -> parse & validate -> return result

--------------------------------------------------------------------------------
9. Final Summary — Current State of Project
--------------------------------------------------------------------------------

- The exporter has been rewritten to follow strict input expectations and produce deterministic, flattened bundle outputs consistent with the RAG pipeline.
- Embeddings are normalized and saved; FAISS index is built and saved.
- id_map.pkl now consistently contains full flattened chunk objects in order matching embeddings rows.
- Retrieval uses the prebuilt FAISS index and maps indices to chunk metadata reliably.
- RAG answer pipeline includes robust parsing for both student and teacher modes; teacher mode retains strictness but includes heuristic fallback to avoid unnecessary refer_teacher responses.
- Interactive CLI supports persistent mode selection and tolerant embed generation.
- End-to-end verification (example: magnesium burning) confirms the pipeline produces correct retrievals and model responses.

This document should be placed under RAVYA/docs and used as the canonical technical reference for the project state after the fixes described above.

If you want I can:
- Add a machine-readable bundle validator script under RAVYA/scripts
- Add unit tests for exporter flattening and rag_answer validators
- Generate a short slide-style README for onboarding new devs

---- End