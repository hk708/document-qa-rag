# Document Q&A — RAG API

A REST API that lets you upload documents and ask natural language questions about them. Built from scratch using a **Retrieval-Augmented Generation (RAG)** pipeline — no RAG framework, just the raw components.

---

## What the App Does

1. You upload a PDF, DOCX, or TXT file
2. The server extracts the text, splits it into overlapping chunks, and embeds each chunk into a 384-dimensional vector using a local sentence-transformer model
3. Those vectors are stored in a FAISS index (persisted to disk)
4. You POST a question — the server embeds it, retrieves the top 5 most semantically similar chunks, feeds them to GPT-4o-mini, and returns a grounded answer with sources

---

## Architecture Flow

```
POST /api/upload
  │
  ├─ parser.py       Extract raw text from PDF / DOCX / TXT
  ├─ chunker.py      Split into overlapping sentence-aware chunks (700 chars, 120 overlap)
  ├─ embeddings.py   Encode each chunk → 384-dim float32 vector (all-MiniLM-L6-v2, local)
  └─ vector_store.py Add vectors to FAISS IndexFlatL2 + save index to disk

POST /api/ask
  │
  ├─ embeddings.py   Encode the question → query vector (same model)
  ├─ vector_store.py FAISS nearest-neighbour search → top-5 chunks by L2 distance
  ├─ ask.py          Build grounded prompt: "Answer using only this context…"
  └─ OpenAI API      GPT-4o-mini generates the answer → return answer + sources
```

---

## Endpoints

### `POST /api/upload`

Upload a document to be indexed.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | file | PDF, DOCX, or TXT — max 20 MB |

**Response:**
```json
{
  "filename": "resume.pdf",
  "char_count": 4821,
  "chunk_count": 14,
  "message": "File uploaded and text extracted successfully."
}
```

---

### `POST /api/ask`

Ask a question against all indexed documents.

**Request:** `application/json`
```json
{
  "question": "What programming languages does the candidate know?"
}
```

**Response:**
```json
{
  "answer": "Based on the resume, the candidate is proficient in Python, JavaScript, and TypeScript.",
  "sources": [
    {
      "rank": 1,
      "score": 0.312,
      "doc_id": "resume",
      "chunk_index": 3,
      "text": "Skills: Python, JavaScript, TypeScript, React, FastAPI, SQL..."
    }
  ]
}
```

---

### `GET /health`

```json
{ "status": "ok" }
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI |
| Text extraction | pdfplumber (PDF), python-docx (DOCX) |
| Embedding model | `all-MiniLM-L6-v2` via sentence-transformers (local, 80 MB) |
| Vector index | FAISS `IndexFlatL2` (exact brute-force, CPU) |
| LLM | GPT-4o-mini via OpenAI API |
| Persistence | FAISS binary index + pickle metadata saved to `data/index/` |

---

## RAG Tradeoffs & Design Decisions

### Chunking strategy
Chunks are split **sentence-aware** at ~700 characters with a 120-character overlap. Splitting mid-sentence loses context; the overlap ensures that a phrase at the boundary of one chunk also appears at the start of the next, so retrieval doesn't miss it.

### FAISS IndexFlatL2
Uses exact **Euclidean (L2) distance** — no approximation, no training needed. Correct for corpora up to ~100k chunks. For larger scale you'd switch to an approximate index (e.g. `IndexIVFFlat`) or a managed vector DB (Pinecone, Weaviate).

### L2 vs cosine similarity
`IndexFlatL2` measures geometric distance between vectors (magnitude + direction). True **cosine similarity** (direction only) requires L2-normalised vectors + `IndexFlatIP`. `all-MiniLM-L6-v2` produces approximately unit-length vectors so L2 and cosine give near-identical rankings for this model — a deliberate tradeoff of simplicity over strictness.

### Grounding / hallucination prevention
The LLM prompt explicitly instructs: *"Answer using only the context below. If the answer is not in the context, say you don't know."* Combined with `temperature=0`, this keeps answers deterministic and strictly grounded in the uploaded documents.

### Embedding model choice
`all-MiniLM-L6-v2` runs entirely locally — no API cost or latency during indexing. Swapping to a higher-quality model (`all-mpnet-base-v2`, `text-embedding-3-small`) is a single constant change in `embeddings.py`.

### Duplicate prevention
`add_embeddings` checks whether the `doc_id` already exists in the metadata list before inserting. Re-uploading the same document is a no-op rather than silently doubling every vector in the index.

---

## What Was Improved During Development

| Area | Original behaviour | Improvement |
|---|---|---|
| Chunking | Fixed character splits | Sentence-aware splits with overlap |
| Duplicate indexing | Re-upload stacked vectors infinitely | Dedup guard on `doc_id` before insert |
| Index persistence | Lost on restart | `save_index` / `load_index` on startup |
| Grounding | No explicit instruction | Prompt forces context-only answers |
| LLM cost | — | GPT-4o-mini chosen over GPT-4o deliberately |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create a `.env` file
```
OPENAI_API_KEY=your_key_here
```

### 3. Start the server
```powershell
# PowerShell
.\run.ps1

# Command Prompt
run.bat

# Manual
uvicorn app.main:app --reload
```

API available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`
