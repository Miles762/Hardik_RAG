from enum import Enum
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single text chunk extracted from a PDF, with its metadata."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int


class Citation(BaseModel):
    """Source reference attached to a generated answer sentence."""
    source_file: str
    page_number: int
    excerpt: str = Field(..., description="Short text snippet from the chunk")


class QueryIntent(str, Enum):
    """Detected query intent used to shape retrieval and answer format."""
    CHITCHAT = "chitchat"
    FACTUAL = "factual"
    LIST = "list"
    TABLE = "table"
    REFUSAL = "refusal"


class IngestResponse(BaseModel):
    """Response returned after PDF ingestion completes."""
    message: str
    files_ingested: list[str]
    total_chunks: int


class QueryRequest(BaseModel):
    """User query sent to the /query endpoint."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's natural-language question",
    )


class QueryResponse(BaseModel):
    """Full response returned by the /query endpoint."""
    answer: str
    intent: QueryIntent
    citations: list[Citation]
    insufficient_evidence: bool = False
    hallucination_flags: list[str] = []


class HealthResponse(BaseModel):
    status: str
    chunks_stored: int
