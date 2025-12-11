# Teacher Mode (ncert-offline-rag)

What it is
- Teacher mode instructs the RAG pipeline to generate long, structured lecture notes / a lesson plan using only the retrieved context chunks.
- Target audience: older school students and teachers.

How to run
- From repository root:
  export PYTHONPATH="$PWD"
  python src/rag/ask.py \
    --query "Prepare detailed lecture notes on force and laws of motion for class 9 CBSE" \
    --mode teacher \
    --k 5 \
    --model 2b

Output schema
- Successful teacher-mode response:
  {
    "status": "ok",
    "mode": "teacher",
    "content": "<long notes in markdown or bullets>",
    "sources": ["id1","id2",...]
  }

- On failure or low confidence:
  {"status":"refer_teacher"}

Behavior and safety
- Uses FAISS retrieval (unchanged).
- The teacher prompt explicitly instructs the model to "Use ONLY the provided context chunks."
- If the model cannot answer from the chunks, it must return exactly: "I don't know, ask your teacher"
  (rag_answer then returns {"status":"refer_teacher"}).
- The produced JSON must include at least one source id that matches a retrieved chunk.

Limitations
- Output quality depends on the underlying local model and prompt tuning.
- Teacher mode enforces strict JSON parsing and validation; improper model output will trigger a fallback to "refer_teacher".