import hashlib
import re
from pathlib import Path
import fitz
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

_mistral = Mistral(api_key=MISTRAL_API_KEY)

EMBED_BATCH_SIZE = 32


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
    """
    if content_type not in ALLOWED_MIME_TYPES:
        raise ValueError(
            f"Unsupported file type '{content_type}'. Only PDF files are accepted."
        )

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        mb = len(file_bytes) / (1024 * 1024)
        raise ValueError(
            f"File size {mb:.1f} MB exceeds the {MAX_FILE_SIZE_BYTES // (1024*1024)} MB limit."
        )

    if not file_bytes.startswith(b"%PDF"):
        raise ValueError(
            "File does not appear to be a valid PDF (missing %PDF header)."
        )

    safe_name = Path(filename).name
    if not safe_name or safe_name != filename:
        raise ValueError(
            f"Invalid filename '{filename}'. Directory traversal is not allowed."
        )

    if not re.match(r"^[\w\-. ]+$", safe_name):
        raise ValueError(
            f"Filename '{safe_name}' contains invalid characters."
        )


def extract_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    """
    Extract text from every page of a PDF using PyMuPDF.

    Returns (page_number, page_text) tuples (1-based page numbers).
    Pages with fewer than 20 characters are skipped — likely blank,
    image-only, or cover pages that would produce noise chunks.

    PyMuPDF handles multi-column layouts, embedded fonts, and encoding
    issues better than alternatives. Scanned PDFs return empty strings
    (OCR is out of scope).
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: list[tuple[int, str]] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if len(text) < 20:
            continue

        pages.append((page_index + 1, text))

    doc.close()
    return pages


def _find_sentence_end(text: str, pos: int) -> int:
    """Return the index just after the next sentence boundary from pos, or len(text)."""
    boundary = re.search(r"[.?!]\s+[A-Z]", text[pos:])
    if boundary:
        return pos + boundary.start() + 1
    return len(text)


def _is_table_heavy(text: str) -> bool:
    """
    Detect pages dominated by tables using three signals (any 2 of 3 → table-heavy):
      1. >50% of lines are short (< 60 chars)
      2. >25% of lines contain numeric values
      3. >20% of lines contain tabs or multiple consecutive spaces (column alignment)
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 4:
        return False

    total = len(lines)
    short_lines = sum(1 for l in lines if len(l) < 60)
    numeric_lines = sum(
        1 for l in lines
        if re.search(
            r"\b\d+[.,]\d+\b"
            r"|\b\d{2,}\b"
            r"|\d+\s*(?:mg|kg|ml|cm|mm|rpm|%|pts?)\b"
            r"|\(\d[\d,.]*\)"
            r"|\$[\d,]+",
            l,
        )
    )
    spaced_col_lines = sum(
        1 for l in lines if "\t" in l or re.search(r"  {2,}", l)
    )

    signals = (
        (short_lines / total > 0.50) +
        (numeric_lines / total > 0.25) +
        (spaced_col_lines / total > 0.20)
    )
    return signals >= 2


def chunk_text(text: str, source_file: str, page_number: int) -> list[Chunk]:
    """
    Split page text into overlapping chunks with sentence-boundary awareness.

    Algorithm:
      start = 0
      while start < len(text):
          end = start + CHUNK_SIZE
          if end < len(text): extend end to next sentence boundary
          chunk = text[start:end]
          start = end - CHUNK_OVERLAP

    Table-heavy pages that fit within 4×CHUNK_SIZE are stored as a single chunk
    with a [Table: ...] header prefix built from the first few lines — this
    improves embedding quality for pages where terse row labels and values would
    otherwise score poorly against natural-language queries.

    Chunk IDs are deterministic (hash of file + page + content) so re-ingesting
    the same file produces the same IDs.
    """
    chunks: list[Chunk] = []
    text_len = len(text)

    if _is_table_heavy(text) and text_len <= CHUNK_SIZE * 4:
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

    effective_chunk_size = CHUNK_SIZE * 2 if _is_table_heavy(text) else CHUNK_SIZE
    start = 0
    index = 0

    while start < text_len:
        end = start + effective_chunk_size
        if end < text_len:
            end = _find_sentence_end(text, end)

        chunk_text_content = text[start:end].strip()

        if len(chunk_text_content) < 20:
            start = end - CHUNK_OVERLAP
            index += 1
            continue

        raw_id = f"{source_file}_{page_number}_{chunk_text_content[:50]}"
        chunk_id = hashlib.md5(raw_id.encode()).hexdigest()[:12]

        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=chunk_text_content,
            source_file=source_file,
            page_number=page_number,
            chunk_index=index,
        ))

        start = end - CHUNK_OVERLAP
        index += 1

    return chunks


def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """
    Embed chunks in batches of EMBED_BATCH_SIZE via the Mistral embed API.
    Batching reduces API round-trips from N to ceil(N / EMBED_BATCH_SIZE).
    """
    embeddings: list[list[float]] = []

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i: i + EMBED_BATCH_SIZE]
        texts = [c.text for c in batch]
        response = _mistral.embeddings.create(model=MISTRAL_EMBED_MODEL, inputs=texts)
        for embedding_obj in response.data:
            embeddings.append(embedding_obj.embedding)

    return embeddings


def ingest_file(filename: str, content_type: str, file_bytes: bytes) -> int:
    """
    Full ingestion pipeline: validate → extract → chunk → embed → store.
    Returns the number of chunks stored.
    """
    validate_file(filename, content_type, file_bytes)

    pages = extract_pages(file_bytes)
    if not pages:
        raise ValueError(
            f"No extractable text found in '{filename}'. "
            "The file may be a scanned image PDF."
        )

    all_chunks: list[Chunk] = []
    for page_number, page_text in pages:
        all_chunks.extend(chunk_text(page_text, filename, page_number))

    if not all_chunks:
        raise ValueError(f"No chunks produced from '{filename}'.")

    embeddings = embed_chunks(all_chunks)
    vector_store.add(all_chunks, embeddings)

    return len(all_chunks)
