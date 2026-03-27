# =============================================================================
# ingestion.py
# -----------------------------------------------------------------------------
# Handles the full data ingestion pipeline:
#   1. Security validation  → reject invalid / malicious files
#   2. PDF text extraction  → pull raw text from each page (PyMuPDF)
#   3. Chunking             → sliding-window split into overlapping chunks
#   4. Embedding            → convert chunks to vectors via Mistral API
#   5. Storage              → persist chunks + vectors in VectorStore
#
# PDF Requirements satisfied:
#   - "Develop an API endpoint to upload a set of files." (wired in main.py)
#   - "Implement a text extraction and chunking algorithm from PDF files.
#      Write down your considerations." → see chunking section below
#   - Security: MIME type check, file size limit, filename sanitisation
#   - Scalability: batch embedding — one API call per 32 chunks, not per chunk
# =============================================================================

import hashlib
import re
from pathlib import Path

import fitz  # PyMuPDF — pip install pymupdf
from mistralai import Mistral

from config import (
    ALLOWED_MIME_TYPES,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_FILE_SIZE_BYTES,
    MISTRAL_API_KEY,
    MISTRAL_EMBED_MODEL,
)
from models import Chunk
from storage import vector_store

# Mistral client — instantiated once at module level (reused across calls)
_mistral = Mistral(api_key=MISTRAL_API_KEY)

# Batch size for embedding API calls.
# Mistral's embed endpoint accepts up to 32 texts per request.
# Batching cuts API round-trips from N to ceil(N/32).
EMBED_BATCH_SIZE = 32


# =============================================================================
# 1. Security validation
# =============================================================================

def validate_file(filename: str, content_type: str, file_bytes: bytes) -> None:
    """
    Validate an uploaded file before any processing.
    Raises ValueError with a descriptive message if validation fails.

    Checks:
      - MIME type is in the whitelist (PDF only)
      - File size does not exceed MAX_FILE_SIZE_BYTES
      - PDF magic bytes ("%PDF") confirm the file is actually a PDF,
        not a renamed executable — defence against MIME spoofing
      - Filename is sanitised (no path traversal like ../../etc/passwd)

    PDF Requirement: Security — validate uploads before touching disk or LLM.
    """
    # --- MIME type whitelist ---
    if content_type not in ALLOWED_MIME_TYPES:
        raise ValueError(
            f"Unsupported file type '{content_type}'. Only PDF files are accepted."
        )

    # --- File size limit ---
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        mb = len(file_bytes) / (1024 * 1024)
        raise ValueError(
            f"File size {mb:.1f} MB exceeds the {MAX_FILE_SIZE_BYTES // (1024*1024)} MB limit."
        )

    # --- Magic bytes check (first 4 bytes of every valid PDF = b"%PDF") ---
    if not file_bytes.startswith(b"%PDF"):
        raise ValueError(
            "File does not appear to be a valid PDF (missing %PDF header)."
        )

    # --- Filename sanitisation: strip directory components ---
    # Path("../../etc/passwd").name → "passwd"  (safe)
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename:
        raise ValueError(
            f"Invalid filename '{filename}'. Directory traversal is not allowed."
        )

    # --- Only allow alphanumeric, dash, underscore, dot in filename ---
    if not re.match(r"^[\w\-. ]+$", safe_name):
        raise ValueError(
            f"Filename '{safe_name}' contains invalid characters."
        )


# =============================================================================
# 2. PDF text extraction
# =============================================================================

def extract_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    """
    Extract text from every page of a PDF.

    Returns a list of (page_number, page_text) tuples.
    page_number is 1-based (matching human-readable page references).

    Implementation note:
      PyMuPDF (fitz) is used because it handles:
        - Multi-column layouts better than pdfminer
        - Embedded fonts and encoding issues
        - Scanned PDFs (returns empty string; OCR is out of scope here)

    Consideration: pages with < 20 characters are skipped — they are likely
    cover pages, intentionally blank pages, or image-only pages with no
    extractable text. Processing them would create noise chunks.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: list[tuple[int, str]] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")  # plain text extraction mode

        # Normalise whitespace: collapse multiple blank lines into one
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        # Skip near-empty pages (images, blanks, chapter dividers)
        if len(text) < 20:
            continue

        pages.append((page_index + 1, text))  # convert to 1-based page number

    doc.close()
    return pages


# =============================================================================
# 3. Chunking — sliding window with sentence-boundary awareness
# =============================================================================

# Consideration (as required by PDF):
# ─────────────────────────────────────────────────────────────────────────────
# WHY sliding window?
#   Fixed non-overlapping chunks risk splitting a sentence or idea across two
#   chunks. A query about that idea would then find neither chunk particularly
#   relevant. Overlapping windows ensure every sentence appears fully in at
#   least one chunk.
#
# WHY 512 chars / 128 overlap?
#   - 512 chars ≈ 100-130 words — enough semantic content for meaningful
#     embedding, small enough to stay focused on one idea.
#   - 128-char overlap ≈ 1-2 sentences — enough to preserve cross-boundary
#     context without doubling storage.
#
# WHY respect sentence boundaries?
#   Cutting mid-sentence produces incomplete thoughts that confuse the
#   embedding model. We always extend a chunk to the next sentence end (". ")
#   before splitting, so embeddings represent complete ideas.
# ─────────────────────────────────────────────────────────────────────────────

def _find_sentence_end(text: str, pos: int) -> int:
    """
    Starting from pos, scan forward to find the next sentence boundary.
    Returns the index just after the boundary (or len(text) if none found).
    A boundary is defined as ". ", "? ", or "! " followed by a capital letter.
    """
    boundary = re.search(r"[.?!]\s+[A-Z]", text[pos:])
    if boundary:
        # +1 to include the punctuation itself in the current chunk
        return pos + boundary.start() + 1
    return len(text)


def _is_table_heavy(text: str) -> bool:
    """
    Detect pages dominated by tables regardless of domain.
    These pages need larger chunks so row values stay near their row labels.

    Signals (any two of these → table-heavy):
      1. High ratio of short lines (< 60 chars) — table rows are narrow
      2. Many lines contain numeric values of any kind (counts, decimals,
         percentages, dollar amounts, measurements like "10 mg" or "3.5 cm")
      3. Many lines contain tab characters or multiple consecutive spaces —
         typical of PDF table extraction where columns are space-aligned
      4. Low average line length relative to page text length — prose has
         long lines; tables have many short ones

    Domain examples this covers:
      - Financial: "$9,133"  "1,429"  "(408)"
      - Medical:   "10 mg"   "0.05"   "< 0.001"   "N=234"
      - Scientific: "3.14"   "±0.02"  "p=0.043"
      - Engineering: "M12x1.75"  "2,400 rpm"  "±5%"
      - Legal/HR tables: "Article 3"  "Section 4.2"  "12 months"
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 4:
        return False

    short_lines = sum(1 for l in lines if len(l) < 60)
    numeric_lines = sum(
        1 for l in lines
        if re.search(
            r"\b\d+[.,]\d+\b"          # decimals: 3.14, 1,429
            r"|\b\d{2,}\b"             # standalone numbers ≥2 digits
            r"|\d+\s*(?:mg|kg|ml|cm|mm|rpm|%|pts?)\b"  # measurements
            r"|\(\d[\d,.]*\)"          # parenthesised numbers (financial negatives)
            r"|\$[\d,]+",              # dollar amounts
            l,
        )
    )
    spaced_col_lines = sum(
        1 for l in lines
        if "\t" in l or re.search(r"  {2,}", l)  # tabs or 2+ consecutive spaces
    )

    total = len(lines)
    signals = (
        (short_lines / total > 0.50) +       # >50% short lines
        (numeric_lines / total > 0.25) +      # >25% numeric lines
        (spaced_col_lines / total > 0.20)     # >20% column-aligned lines
    )
    return signals >= 2


def chunk_text(
    text: str,
    source_file: str,
    page_number: int,
) -> list[Chunk]:
    """
    Split page text into overlapping chunks with sentence-boundary awareness.

    Algorithm:
      start = 0
      while start < len(text):
          end = start + CHUNK_SIZE
          if end < len(text): extend end to next sentence boundary
          chunk = text[start:end]
          start = end - CHUNK_OVERLAP   ← next chunk starts OVERLAP chars back

    Special case — financial table pages:
      Pages where >25% of lines contain monetary values are kept in larger
      chunks (up to 2×CHUNK_SIZE) so that row labels stay with their numbers.
      Without this, "Total revenues" and "$9,133" end up in different chunks
      and neither chunk scores well for a "total revenue" query.

    Each chunk is assigned a deterministic chunk_id based on a hash of its
    content — avoids duplicate chunks if the same file is ingested twice.
    """
    chunks: list[Chunk] = []
    start = 0
    index = 0
    text_len = len(text)

    # Table-heavy pages: keep the whole page as one chunk if it fits within
    # 4× CHUNK_SIZE. Prepend a descriptive header so the embedding captures
    # what kind of data this page contains — without this, terse table text
    # (row labels + numbers on separate lines) scores poorly against natural
    # language queries even when it's the correct answer.
    if _is_table_heavy(text) and text_len <= CHUNK_SIZE * 4:
        # Build a natural-language header from the first few non-empty lines
        # (these are typically the section title and date, e.g.
        #  "Condensed Consolidated Balance Sheets  April 30, 2024")
        header_lines = [l.strip() for l in text.splitlines() if l.strip()][:6]
        header = " | ".join(header_lines)
        enriched_text = f"[Table: {header}]\n\n{text.strip()}"
        chunk_id = hashlib.md5(
            f"{source_file}_{page_number}_{text[:50]}".encode()
        ).hexdigest()[:12]
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=enriched_text,
            source_file=source_file,
            page_number=page_number,
            chunk_index=0,
        ))
        return chunks

    # For large table pages or prose pages: sliding window
    effective_chunk_size = CHUNK_SIZE * 2 if _is_table_heavy(text) else CHUNK_SIZE

    while start < text_len:
        end = start + effective_chunk_size

        if end < text_len:
            # Extend to the nearest sentence boundary so we don't cut mid-sentence
            end = _find_sentence_end(text, end)

        chunk_text_content = text[start:end].strip()

        # Skip chunks that are pure whitespace or too short to be meaningful
        if len(chunk_text_content) < 20:
            start = end - CHUNK_OVERLAP
            index += 1
            continue

        # Deterministic chunk_id: hash of (file + page + content)
        # This means re-ingesting the same file produces the same IDs,
        # which makes deduplication possible in future iterations.
        raw_id = f"{source_file}_{page_number}_{chunk_text_content[:50]}"
        chunk_id = hashlib.md5(raw_id.encode()).hexdigest()[:12]

        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=chunk_text_content,
            source_file=source_file,
            page_number=page_number,
            chunk_index=index,
        ))

        # Move start back by CHUNK_OVERLAP for the sliding window
        start = end - CHUNK_OVERLAP
        index += 1

    return chunks


# =============================================================================
# 4. Embedding — batch API calls to Mistral
# =============================================================================

def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """
    Convert chunk texts to embedding vectors via the Mistral embed API.

    Batching strategy:
      - Send EMBED_BATCH_SIZE (32) texts per API call.
      - This reduces round-trips from N to ceil(N/32).
      - Each embedding is a list of 1024 floats (mistral-embed dimension).

    The returned list is index-aligned with the input chunks list:
      embeddings[i] is the vector for chunks[i].

    PDF Requirement: Scalability — batch calls instead of one-per-chunk.
    """
    embeddings: list[list[float]] = []

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        texts = [c.text for c in batch]

        response = _mistral.embeddings.create(
            model=MISTRAL_EMBED_MODEL,
            inputs=texts,
        )

        # response.data is a list of EmbeddingObject, each has a .embedding field
        for embedding_obj in response.data:
            embeddings.append(embedding_obj.embedding)

    return embeddings


# =============================================================================
# 5. Top-level ingestion function — called by FastAPI endpoint
# =============================================================================

def ingest_file(filename: str, content_type: str, file_bytes: bytes) -> int:
    """
    Full ingestion pipeline for a single PDF file.
    Returns the number of chunks stored.

    Steps:
      1. validate_file()   → security checks
      2. extract_pages()   → raw text per page
      3. chunk_text()      → sliding-window chunks per page
      4. embed_chunks()    → Mistral embedding vectors
      5. vector_store.add()→ persist to in-memory + disk store

    PDF Requirements satisfied:
      - "Develop an API endpoint to upload a set of files" (pipeline entry point)
      - "Implement a text extraction and chunking algorithm from PDF files"
      - Security: validation before any processing
      - Scalability: batch embedding
    """
    # Step 1: Security validation — raises ValueError on failure
    validate_file(filename, content_type, file_bytes)

    # Step 2: Extract text page by page
    pages = extract_pages(file_bytes)

    if not pages:
        raise ValueError(
            f"No extractable text found in '{filename}'. "
            "The file may be a scanned image PDF."
        )

    # Step 3: Chunk every page using sliding window
    all_chunks: list[Chunk] = []
    for page_number, page_text in pages:
        page_chunks = chunk_text(page_text, filename, page_number)
        all_chunks.extend(page_chunks)

    if not all_chunks:
        raise ValueError(f"No chunks produced from '{filename}'.")

    # Step 4: Embed all chunks (batched API calls)
    embeddings = embed_chunks(all_chunks)

    # Step 5: Persist chunks + embeddings to the vector store
    vector_store.add(all_chunks, embeddings)

    return len(all_chunks)
