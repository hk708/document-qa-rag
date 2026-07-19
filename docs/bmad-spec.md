# BMAD Specification — Document Q&A (RAG Application)

**Version:** 1.0  
**Date:** 2026-06-12  
**Status:** Draft  

---

## Table of Contents

1. [Project Brief](#1-project-brief)
2. [Product Requirements Document (PRD)](#2-product-requirements-document-prd)
3. [Architecture Document](#3-architecture-document)
4. [Frontend Architecture](#4-frontend-architecture)
5. [API Contract](#5-api-contract)
6. [Data Models](#6-data-models)
7. [Epics & User Stories](#7-epics--user-stories)
8. [Non-Functional Requirements](#8-non-functional-requirements)
9. [Out of Scope](#9-out-of-scope)
10. [Open Questions & Decisions Log](#10-open-questions--decisions-log)

---

## 1. Project Brief

### 1.1 Problem Statement

Knowledge workers frequently need to query the contents of large or numerous documents — resumes, policy PDFs, contracts, rule-books — without reading them in full. Generic LLMs hallucinate answers when asked about private documents they have never seen. A retrieval layer is required to ground the model's response in the actual file content.

### 1.2 Solution Summary

A full-stack **Retrieval-Augmented Generation (RAG)** application that:

1. Accepts document uploads (PDF, DOCX, TXT) via a browser UI.
2. Parses, chunks, and embeds each document locally — no text leaves the machine until a question is asked.
3. Stores embeddings in a persistent FAISS vector index.
4. Answers natural-language questions by retrieving the most semantically relevant chunks and grounding GPT-4o-mini's response in that context.

### 1.3 Goals

| # | Goal | Success Metric |
|---|------|----------------|
| G1 | Accurate, grounded answers | LLM answer references only content present in uploaded documents |
| G2 | Fast time-to-first-answer | End-to-end question latency < 5 s on typical hardware |
| G3 | Zero-duplication indexing | Re-uploading the same file returns a 409 without re-indexing |
| G4 | Persistence across restarts | FAISS index survives server restart; no re-upload required |
| G5 | Multi-document support | Multiple documents can be indexed and queried simultaneously |

### 1.4 Target Users

- **Primary:** Individual developers / knowledge workers evaluating RAG techniques on private documents.
- **Secondary:** Technical interviewers or recruiters assessing a candidate via their uploaded resume.

---

## 2. Product Requirements Document (PRD)

### 2.1 Functional Requirements

#### FR-01 — Document Upload
- The system MUST accept file uploads via a multipart/form-data POST request.
- Accepted formats: `.pdf`, `.docx`, `.txt`.
- Maximum file size: 20 MB.
- The system MUST reject unsupported file types with HTTP 400.
- The system MUST reject files exceeding the size limit with HTTP 413.
- The system MUST return HTTP 409 if the document (`doc_id` derived from filename stem) is already indexed, without performing any processing.
- On success the system MUST return `filename`, `char_count`, `chunk_count`, and a status `message`.

#### FR-02 — Text Extraction
- PDF files MUST be extracted page-by-page using `pdfplumber`.
- DOCX files MUST be extracted paragraph-by-paragraph using `python-docx`.
- TXT files MUST be read as UTF-8 (with replacement for undecodable bytes).
- If extraction yields empty text the system MUST return HTTP 422.

#### FR-03 — Chunking
- Extracted text MUST be split into sentence-aware, overlapping chunks.
- Default parameters: `chunk_size = 700` characters, `overlap = 120` characters.
- Each chunk MUST be assigned a unique UUID (`chunk_id`) and a sequential `chunk_index` within the document.

#### FR-04 — Embedding
- Each chunk MUST be encoded into a 384-dimensional float32 vector using the `all-MiniLM-L6-v2` sentence-transformer model.
- The embedding model MUST be loaded once as a module-level singleton (not per-request).
- Embeddings MUST be produced in batches of 32 chunks for memory efficiency.

#### FR-05 — Vector Indexing & Persistence
- Embeddings MUST be stored in a FAISS `IndexFlatL2` index.
- The index and its parallel metadata list MUST be persisted to disk after every upload (`data/index/faiss.index`, `data/index/metadata.pkl`).
- The index MUST be loaded from disk at application startup via the FastAPI lifespan handler.
- Deduplication MUST be enforced: if `doc_id` already appears in `_metadata` the batch MUST be skipped.

#### FR-06 — Question Answering
- A POST `/api/ask` endpoint MUST accept a plain-text `question`.
- The system MUST embed the question using the same model as FR-04.
- The system MUST retrieve the top 5 nearest chunks by L2 distance from the FAISS index.
- The system MUST return HTTP 404 if no documents are indexed.
- Retrieved chunks MUST be concatenated into a grounded context block and passed to GPT-4o-mini.
- The LLM system prompt MUST instruct the model to answer only from the provided context.
- The response MUST include `answer` (string) and `sources` (ranked list of chunk metadata).

#### FR-07 — Health Check
- `GET /health` MUST return `{"status": "ok"}` when the server is running.

### 2.2 User Interface Requirements

#### UI-01 — Upload Panel
- Displays a file picker accepting `.pdf`, `.docx`, `.txt`.
- Shows a spinner with "Indexing…" label during upload.
- Displays ✅ success message with filename and chunk count on completion.
- Displays ⚠️ amber warning (HTTP 409) when the file is already indexed — allows the user to still ask questions.
- Displays ❌ error message on any other failure.

#### UI-02 — Question Panel
- Disabled with hint text "Upload a document first…" until at least one document is successfully indexed.
- Free-text input with submit button.
- Shows "Thinking…" spinner while awaiting API response.
- Renders the LLM answer in a styled answer block.

#### UI-03 — Source Chunks
- Rendered below the answer, showing all returned source references.
- Each source card shows: rank, `doc_id`, `chunk_index`, L2 score (4 decimal places).
- Cards are collapsed by default; clicking expands to show the raw chunk text.

---

## 3. Architecture Document

### 3.1 System Overview

```
┌─────────────────────────────┐        ┌──────────────────────────────────┐
│      Browser (React/Vite)   │◄──────►│   FastAPI Backend (Python)       │
│  UploadPanel                │  HTTP  │                                  │
│  QuestionPanel              │        │  POST /api/upload                │
│  SourceChunks               │        │  POST /api/ask                   │
└─────────────────────────────┘        │  GET  /health                    │
                                       │                                  │
                                       │  ┌──────────────────────────┐   │
                                       │  │  RAG Pipeline             │   │
                                       │  │  parser → chunker →       │   │
                                       │  │  embeddings → vector_store│   │
                                       │  └──────────────────────────┘   │
                                       │                                  │
                                       │  ┌─────────┐  ┌──────────────┐  │
                                       │  │  FAISS  │  │  OpenAI API  │  │
                                       │  │  Index  │  │  GPT-4o-mini │  │
                                       │  │ (disk)  │  └──────────────┘  │
                                       │  └─────────┘                    │
                                       └──────────────────────────────────┘
```

### 3.2 Upload Pipeline

```
POST /api/upload
     │
     ├─ [validation]     Extension check · Size check · 409 dedup check
     │
     ├─ parser.py        Extract raw text from PDF / DOCX / TXT
     │
     ├─ chunker.py       Sentence-aware sliding window
     │                   (700 char window, 120 char overlap)
     │
     ├─ embeddings.py    Batch encode chunks → float32[384] vectors
     │                   (all-MiniLM-L6-v2, singleton model)
     │
     ├─ vector_store.py  add_embeddings() → FAISS IndexFlatL2
     │                   save_index() → data/index/{faiss.index, metadata.pkl}
     │
     └─ [response]       UploadResponse JSON
```

### 3.3 Query Pipeline

```
POST /api/ask
     │
     ├─ embeddings.py    Embed question → float32[384] query vector
     │
     ├─ vector_store.py  FAISS IndexFlatL2.search(query_vec, k=5)
     │                   → top-5 (distance, faiss_id) pairs
     │                   → lookup _metadata[faiss_id] per result
     │
     ├─ ask.py           Build grounded prompt:
     │                   "Answer using only the context below…"
     │                   + labelled chunk context block
     │                   + user question
     │
     ├─ OpenAI API       chat.completions.create(model="gpt-4o-mini")
     │
     └─ [response]       AskResponse { answer, sources[] }
```

### 3.4 Startup Sequence

```
FastAPI lifespan handler
     │
     └─ vector_store.load_index()
          ├─ Reads data/index/faiss.index  → restores FAISS index in memory
          └─ Reads data/index/metadata.pkl → restores _metadata list
          (no-op if files do not exist yet)
```

### 3.5 Deduplication Strategy

Deduplication is enforced at two layers:

| Layer | Mechanism | When |
|-------|-----------|------|
| Route (early) | `is_doc_indexed(doc_id)` check before any I/O | Upload request received |
| Service (guard) | `add_embeddings()` checks `_metadata` before FAISS `.add()` | Belt-and-suspenders |

`doc_id` is derived as `Path(filename).stem` — the filename without extension. This means `resume.pdf` and `resume.docx` share the same `doc_id` (`resume`) and the second upload will be rejected.

### 3.6 Directory Layout

```
himmi-personal/
├── app/
│   ├── config.py               Path constants, allowed extensions, size limit
│   ├── main.py                 FastAPI app, CORS middleware, lifespan handler
│   ├── models/
│   │   └── schemas.py          Pydantic request/response models
│   ├── routes/
│   │   ├── upload.py           POST /api/upload
│   │   └── ask.py              POST /api/ask
│   └── services/
│       ├── parser.py           Text extraction (PDF/DOCX/TXT)
│       ├── chunker.py          Sentence-aware sliding-window chunker
│       ├── embeddings.py       Sentence-transformer embedding service
│       └── vector_store.py     FAISS index management & search
├── data/
│   ├── raw/                    Original uploaded files
│   ├── processed/              Extracted .txt files + *_chunks.json snapshots
│   └── index/
│       ├── faiss.index         Persisted FAISS binary index
│       └── metadata.pkl        Parallel chunk metadata list
├── frontend/
│   ├── src/
│   │   ├── App.jsx             Root component, state management, API calls
│   │   ├── App.css             All styles
│   │   └── components/
│   │       ├── UploadPanel.jsx File upload UI
│   │       ├── QuestionPanel.jsx Q&A UI
│   │       └── SourceChunks.jsx Collapsible source references
│   └── package.json
├── .env                        OPENAI_API_KEY
├── requirements.txt
├── run.bat / run.ps1           Dev startup scripts
└── docs/
    └── bmad-spec.md            This document
```

---

## 4. Frontend Architecture

### 4.1 Technology Choices

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Framework | React 18 | Declarative UI with hooks; minimal overhead for this scale |
| Build tool | Vite 5 | Fast HMR, ESM-native, zero-config for React |
| Styling | Plain CSS (App.css) | No build-time CSS framework needed; single-file style sheet is maintainable at this scope |
| HTTP | Native `fetch` | No extra dependency; sufficient for two endpoints |

### 4.2 State Model

All shared state lives in `App.jsx` and flows down as props:

```
App
├── indexed: boolean           — true after first successful upload (or 409)
├── uploadState: {             — controls UploadPanel feedback
│     status: 'idle' | 'loading' | 'success' | 'warning' | 'error'
│     message: string
│   }
└── askState: {                — controls QuestionPanel / SourceChunks
      status: 'idle' | 'loading' | 'success' | 'error'
      answer: string
      sources: SourceChunk[]
    }
```

### 4.3 Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `App.jsx` | Global state, `handleUpload()`, `handleAsk()`, API calls |
| `UploadPanel` | File picker, extension validation, upload feedback display |
| `QuestionPanel` | Question form, answer display, delegates `SourceChunks` rendering |
| `SourceChunks` | Collapsible source card list with rank/score/text |

### 4.4 API Communication

```
POST http://localhost:8000/api/upload
  body: FormData { file }

POST http://localhost:8000/api/ask
  headers: { Content-Type: application/json }
  body: { question: string }
```

CORS is restricted to `http://localhost:5173` (Vite dev server) in the backend.

---

## 5. API Contract

### 5.1 `POST /api/upload`

**Request** — `multipart/form-data`

| Field | Type | Constraints |
|-------|------|-------------|
| `file` | binary | `.pdf` / `.docx` / `.txt`, max 20 MB |

**Success Response — 200**
```json
{
  "filename": "resume.pdf",
  "char_count": 4821,
  "chunk_count": 14,
  "message": "File uploaded and text extracted successfully."
}
```

**Error Responses**

| HTTP Status | Condition |
|-------------|-----------|
| 400 | Unsupported file extension |
| 409 | `doc_id` already present in FAISS metadata |
| 413 | File exceeds 20 MB |
| 422 | Text extraction succeeded but yielded empty content |
| 422 | Text extraction raised an exception |

---

### 5.2 `POST /api/ask`

**Request** — `application/json`
```json
{ "question": "string" }
```

**Success Response — 200**
```json
{
  "answer": "string",
  "sources": [
    {
      "rank": 1,
      "score": 0.3124,
      "doc_id": "resume",
      "chunk_index": 3,
      "text": "Skills: Python, JavaScript…"
    }
  ]
}
```

**Error Responses**

| HTTP Status | Condition |
|-------------|-----------|
| 404 | No documents have been indexed yet |
| 500 | FAISS search error |
| 500 | `OPENAI_API_KEY` not set |
| 500 | OpenAI API error |

---

### 5.3 `GET /health`

**Success Response — 200**
```json
{ "status": "ok" }
```

---

## 6. Data Models

### 6.1 Pydantic Schemas (`app/models/schemas.py`)

```python
class UploadResponse(BaseModel):
    filename:    str
    char_count:  int
    chunk_count: int
    message:     str

class AskRequest(BaseModel):
    question: str

class SourceChunk(BaseModel):
    rank:        int
    score:       float
    doc_id:      str
    chunk_index: int
    text:        str

class AskResponse(BaseModel):
    answer:  str
    sources: list[SourceChunk]
```

### 6.2 Internal Chunk Dataclass (`app/services/chunker.py`)

```python
@dataclass
class Chunk:
    chunk_id:    str   # UUID4
    doc_id:      str   # filename stem
    text:        str   # raw chunk text (≤ 700 chars)
    chunk_index: int   # 0-based position within document
```

### 6.3 Embedded Chunk Dict (`app/services/embeddings.py`)

```python
{
    "doc_id":      str,         # filename stem
    "chunk_id":    str,         # UUID4
    "chunk_index": int,         # 0-based position
    "text":        str,         # raw chunk text
    "embedding":   list[float]  # length 384 (float32)
}
```

### 6.4 FAISS Metadata Entry (`app/services/vector_store.py`)

Stored in `_metadata: list[dict]` where `_metadata[i]` maps to FAISS internal vector id `i`:

```python
{
    "doc_id":      str,
    "chunk_id":    str,
    "chunk_index": int,
    "text":        str
}
```

### 6.5 Processed File Artefacts (`data/processed/`)

| File pattern | Content |
|---|---|
| `{stem}.txt` | Extracted plain text from the original file |
| `{stem}_chunks.json` | JSON array of embedded chunk dicts (human-readable snapshot) |

---

## 7. Epics & User Stories

### Epic 1 — Document Ingestion

| ID | Story | Acceptance Criteria |
|----|-------|---------------------|
| 1.1 | As a user I can upload a PDF so its content is searchable | Upload returns 200 with correct `chunk_count`; vectors appear in FAISS index |
| 1.2 | As a user I can upload a DOCX or TXT file | Same as 1.1 for respective formats |
| 1.3 | As a user uploading an already-indexed file I see a clear warning | API returns 409; frontend shows ⚠️ amber message; `indexed` state remains true |
| 1.4 | As a user uploading a corrupt or image-only PDF I see an error | API returns 422; frontend shows ❌ message |
| 1.5 | As a user the index survives a server restart | Vectors present after restart without re-uploading |

### Epic 2 — Question Answering

| ID | Story | Acceptance Criteria |
|----|-------|---------------------|
| 2.1 | As a user I can ask a natural language question about an uploaded document | API returns a non-empty `answer` and at least one source |
| 2.2 | As a user the answer is grounded in the document content | Answer references only information present in the retrieved chunks |
| 2.3 | As a user asking about content not in any document I get an honest "I don't know" | LLM responds accordingly without hallucinating |
| 2.4 | As a user I can see which document and chunk the answer came from | Sources rendered with doc_id, chunk_index, score, and expandable text |
| 2.5 | As a user querying before any upload I see a helpful error | API returns 404; Question input is disabled with hint text |

### Epic 3 — Multi-Document Support

| ID | Story | Acceptance Criteria |
|----|-------|---------------------|
| 3.1 | As a user I can upload multiple documents and query across all of them | Retrieval returns chunks from any indexed document |
| 3.2 | As a user the source cards correctly identify which document each chunk came from | `doc_id` in source matches uploaded filename stem |

### Epic 4 — Developer Experience

| ID | Story | Acceptance Criteria |
|----|-------|---------------------|
| 4.1 | As a developer I can start the full stack with a single command | `run.ps1` / `run.bat` launches both backend and Vite dev server |
| 4.2 | As a developer the health endpoint confirms the server is alive | `GET /health` returns 200 `{"status": "ok"}` |
| 4.3 | As a developer chunk snapshots are saved to disk for inspection | `data/processed/*_chunks.json` written after every upload |

---

## 8. Non-Functional Requirements

### 8.1 Performance

| Requirement | Target |
|---|---|
| Upload + index latency (10-page PDF) | < 10 s on CPU |
| Question answering end-to-end latency | < 5 s (excludes OpenAI network variance) |
| Embedding model cold-start | < 3 s (model loaded once at first request) |
| FAISS search time (< 100k vectors) | < 10 ms |

### 8.2 Scalability

The current architecture is intentionally single-process. Scalability constraints and their implications:

| Constraint | Implication |
|---|---|
| `IndexFlatL2` brute-force | Suitable up to ~100k chunks; replace with `IndexIVFFlat` for larger corpora |
| Module-level singletons (`_model`, `_index`, `_metadata`) | Not safe for multi-worker deployments; a shared index store (e.g. Redis + Qdrant) would be needed |
| CORS locked to `localhost:5173` | Must be updated for any deployed environment |

### 8.3 Security

| Area | Current Posture | Recommendation |
|---|---|---|
| API key management | Read from `OPENAI_API_KEY` env var; never logged | Keep as-is; add `.env` to `.gitignore` |
| File upload path traversal | Filename used only for stem derivation; `raw_path` is always inside `RAW_DIR` | Acceptable |
| Input validation | Pydantic enforces request schemas; extension + size validated before I/O | Acceptable |
| CORS | Locked to `localhost:5173` | Tighten for production deployment |
| Rate limiting | None | Add `slowapi` or reverse-proxy rate limiting before any public deployment |

### 8.4 Reliability

- The FAISS index is persisted after every upload; partial writes are possible if the process crashes mid-`save_index`. A write-to-temp-then-rename pattern would make persistence atomic.
- There is no retry logic on the OpenAI call; transient failures surface as HTTP 500.

### 8.5 Observability

- No structured logging is currently implemented.
- The `/health` endpoint provides basic liveness.
- `vector_store.total_vectors()` is available internally for diagnostics.

---

## 9. Out of Scope

The following capabilities are explicitly deferred and not part of the current implementation:

| Feature | Rationale |
|---|---|
| Document deletion / index removal | FAISS `IndexFlatL2` does not support vector removal; would require index rebuild |
| User authentication | Single-user local tool; no auth needed |
| Streaming LLM responses | Would require SSE/WebSocket changes; not needed at current scale |
| Re-indexing a modified document | User must delete `data/index/` manually and restart to force re-index |
| Cloud / production deployment | Out of scope; environment variables and CORS must be updated first |
| Document listing endpoint | No `GET /api/documents` endpoint exists yet |
| Non-English documents | Untested; `all-MiniLM-L6-v2` has limited multilingual capability |

---

## 10. Open Questions & Decisions Log

| # | Question | Decision | Date |
|---|----------|----------|------|
| D1 | L2 distance vs cosine similarity for retrieval? | L2 (`IndexFlatL2`) chosen for simplicity; cosine would require L2-normalising vectors first. Acceptable for same-model embeddings. | — |
| D2 | Framework RAG (LangChain/LlamaIndex) vs custom? | Custom pipeline chosen for full transparency and learning value. | — |
| D3 | How to handle duplicate uploads? | 409 HTTP response with early check before any processing; metadata `doc_id` lookup is the source of truth. | 2026-06-11 |
| D4 | Where does `doc_id` come from? | `Path(filename).stem` — files with different extensions but same stem share a `doc_id`. Acceptable for current scope. | — |
| O1 | Should document deletion be supported? | Open — requires index rebuild strategy (copy-and-exclude or swap to a deletion-capable store like Qdrant). | — |
| O2 | Should `/api/documents` list indexed docs? | Open — useful for multi-doc UX; straightforward to derive from `set(m["doc_id"] for m in _metadata)`. | — |
| O3 | Should chunk parameters be configurable per-upload? | Open — currently hardcoded at 700/120. Could be query params on `/upload`. | — |
