# StackAI RAG

A Retrieval-Augmented Generation (RAG) backend built with FastAPI and Mistral AI. Upload PDF files and ask natural-language questions — the system retrieves the most relevant content and generates grounded, cited answers.

---

## System Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│           (upload PDFs · ask questions · view citations)        │
└────────────────────┬───────────────────┬────────────────────────┘
                     │  POST /ingest      │  POST /query
                     ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                            │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │ ingestion.py│   │ retrieval.py │   │    generation.py     │ │
│  │             │   │              │   │                      │ │
│  │ 1. Validate │   │ 1. Transform │   │ 1. Detect intent     │ │
│  │ 2. Extract  │   │ 2. Embed     │   │ 2. Build prompt      │ │
│  │ 3. Chunk    │   │ 3. Cosine    │   │ 3. Call Mistral      │ │
│  │ 4. Embed    │   │ 4. BM25      │   │ 4. Hallucin. filter  │ │
│  │ 5. Store    │   │ 5. RRF fuse  │   │ 5. Build citations   │ │
│  └──────┬──────┘   └──────┬───────┘   └──────────────────────┘ │
└─────────│─────────────────│──────────────────────────────────────┘
          ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      storage.py                                 │
│        NumpyVectorStore (embeddings.npy + metadata.json)        │
│        In-memory · disk-persisted · no third-party vector DB    │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Mistral AI API                               │
│   mistral-embed (embeddings)  ·  mistral-large-latest (chat)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### 1. Data Ingestion (`/ingest`)

When you upload a PDF:

1. **Validation** — MIME type, file size (max 20 MB), PDF magic bytes, filename sanitisation
2. **Text extraction** — PyMuPDF extracts text page by page; near-empty pages are skipped
3. **Chunking** — Sliding window (512 chars, 128 overlap) respecting sentence boundaries
   - *Why sliding window?* Fixed chunks risk cutting sentences in half. Overlap ensures every sentence appears fully in at least one chunk. Table-heavy pages are stored as a single enriched chunk to preserve row/column context.
4. **Embedding** — Chunks are sent to Mistral's `mistral-embed` in batches of 32
5. **Storage** — Vectors saved to `data/vectors/embeddings.npy`; metadata to `data/vectors/metadata.json`

---

### 2. Query Processing (`/query`)

When you ask a question:

1. **Intent detection** — LLM classifies the query into one of five intents:
   - `CHITCHAT` → respond conversationally, skip retrieval
   - `REFUSAL` → PII detected, financial/legal/medical advice → refuse
   - `LIST` → answer formatted as bullets
   - `TABLE` → answer formatted as Markdown table
   - `FACTUAL` → standard concise answer

2. **Query transformation** — HyDE-lite: rewrites the question as a declarative statement closer in style to document text, improving embedding match quality

3. **Hybrid retrieval**:
   - **Semantic search** — cosine similarity (numpy matrix multiply, no external lib)
   - **BM25 keyword search** — implemented from scratch; catches exact keyword/name/date matches that semantic search can miss
   - **RRF fusion** — Reciprocal Rank Fusion merges both ranked lists using rank position only (robust to score scale differences)

4. **LLM re-ranking** — top candidates are re-ranked by the LLM as a cross-encoder for final relevance ordering

5. **Threshold filter** — If the top cosine score < 0.35, returns "insufficient evidence" instead of hallucinating

6. **Generation** — Mistral `mistral-large-latest` called with:
   - Intent-specific system prompt (factual / list / table)
   - Numbered context block with source labels
   - Instruction to cite inline as `[source: filename, page N]`

7. **Hallucination filter** — Post-hoc LLM evidence check: the answer is sent back to the LLM alongside the source chunks; any sentences not supported by the context are flagged

8. **Citations** — Source file + page number + excerpt attached to every response

---

## Key Design Decisions & Tradeoffs

| Decision | Why |
|---|---|
| No third-party vector DB | Required, also removes operational complexity |
| numpy cosine similarity | Standard implementation; O(N) per query is fast enough for thousands of chunks |
| BM25 from scratch | Required by spec ("no external search library"); `rank-bm25` is an algorithm only |
| RRF over weighted averaging | Score scales differ between cosine and BM25; rank-based fusion needs no tuning |
| LLM intent detection | Rule-based keyword matching is brittle; LLM handles any phrasing naturally |
| Sliding window chunks | Prevents context loss at chunk boundaries |
| Abstract VectorStore | `NumpyVectorStore` is swappable — replace with Pinecone/Weaviate by implementing the same interface |
| Mistral-embed for both chunks and query | Vectors must be in the same embedding space for cosine similarity to be meaningful |

---

## Scalability

- **Storage** — `VectorStoreBase` abstract class means replacing numpy with a distributed vector DB is a one-line change
- **Async endpoints** — All FastAPI endpoints are `async`; Mistral calls are I/O-bound and yield the event loop
- **Stateless app layer** — All state is in `data/vectors/`. Point multiple app instances at a shared volume for horizontal scaling
- **Batch embedding** — 32 chunks per API call instead of N calls; reduces latency and API costs linearly

---

## Security

- API key loaded from `.env` — never hardcoded
- File upload validation: MIME whitelist, size limit, magic bytes check, filename sanitisation
- PII regex patterns (SSN, credit card, email) checked before any LLM call
- LLM intent detection handles financial/legal/medical refusals naturally
- Rate limiting: 10/min on `/ingest`, 30/min on `/query`
- CORS restricted to `localhost:8501` (Streamlit)
- Global exception handler — no stack traces exposed to clients

---

## Libraries Used

| Library | Purpose | Link |
|---|---|---|
| FastAPI | API framework | https://fastapi.tiangolo.com |
| Uvicorn | ASGI server | https://www.uvicorn.org |
| Mistral AI | LLM + embeddings | https://docs.mistral.ai |
| PyMuPDF | PDF text extraction | https://pymupdf.readthedocs.io |
| NumPy | Vector arithmetic, cosine similarity | https://numpy.org |
| rank-bm25 | BM25 algorithm | https://github.com/dorianbrown/rank_bm25 |
| slowapi | Rate limiting | https://github.com/laurentS/slowapi |
| Streamlit | UI | https://streamlit.io |
| python-dotenv | Env var loading | https://github.com/theskumar/python-dotenv |

---

## Setup & Running

### 1. Install dependencies

```bash
cd rag-pipeline
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and set MISTRAL_API_KEY=your_key_here
```

### 3. Start the backend

```bash
uvicorn main:app --reload
# API running at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

### 4. Start the UI

```bash
streamlit run ui/app.py
# UI running at http://localhost:8501
```

---

## API Reference

### `POST /ingest`
Upload one or more PDF files.

**Request**: `multipart/form-data` with field `files` (one or more PDFs)

**Response** `200 OK`:
```json
{
  "message": "2 file(s) ingested successfully.",
  "files_ingested": ["doc1.pdf", "doc2.pdf"],
  "total_chunks": 142
}
```

---

### `POST /query`
Ask a question against the knowledge base.

**Request**:
```json
{ "question": "What are the main findings?" }
```

**Response** `200 OK`:
```json
{
  "answer": "The main findings are... [source: doc1.pdf, page 3]",
  "intent": "factual",
  "citations": [
    { "source_file": "doc1.pdf", "page_number": 3, "excerpt": "..." }
  ],
  "insufficient_evidence": false,
  "hallucination_flags": []
}
```

---

### `GET /health`
```json
{ "status": "ok", "chunks_stored": 142 }
```

### `GET /files`
```json
{ "files": ["doc1.pdf", "doc2.pdf"] }
```

### `POST /remove`
```json
{ "filename": "doc1.pdf" }
```

### `POST /clear`
Wipes the entire knowledge base.

---

## curl Commands

```bash
# Health check
curl http://localhost:8000/health

# List ingested files
curl http://localhost:8000/files

# Ingest a PDF
curl -X POST http://localhost:8000/ingest -F "files=@/path/to/your.pdf"

# Query
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "What is the main topic?"}'

# Remove a file
curl -X POST http://localhost:8000/remove -H "Content-Type: application/json" -d '{"filename": "your.pdf"}'

# Clear everything
curl -X POST http://localhost:8000/clear
```

Visit `http://localhost:8000/docs` to explore all endpoints interactively via the Swagger UI **Try it out** feature.
