from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import RATE_LIMIT_INGEST, RATE_LIMIT_QUERY
from generation import generate
from ingestion import ingest_file
from models import HealthResponse, IngestResponse, QueryRequest, QueryResponse
from retrieval import retrieve
from storage import vector_store


limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="StackAI RAG API",
    description="Retrieval-Augmented Generation backend using Mistral AI. Upload PDFs via /ingest, then query them via /query.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


@app.post("/clear", summary="Clear all chunks from the knowledge base", tags=["System"])
async def clear_endpoint() -> dict:
    vector_store.clear()
    return {"message": "Knowledge base cleared.", "chunks_stored": 0}


class RemoveRequest(BaseModel):
    filename: str


@app.post("/remove", summary="Remove all chunks for a specific file", tags=["System"])
async def remove_endpoint(body: RemoveRequest) -> dict:
    import numpy as np
    all_chunks = vector_store.get_all_chunks()
    keep_indices = [i for i, c in enumerate(all_chunks) if c.source_file != body.filename]
    if len(keep_indices) == len(all_chunks):
        raise HTTPException(status_code=404, detail=f"No chunks found for '{body.filename}'")

    kept_chunks = [all_chunks[i] for i in keep_indices]
    kept_embeddings = vector_store._embeddings[keep_indices].tolist() if vector_store._embeddings is not None and keep_indices else []

    vector_store.clear()
    if kept_chunks:
        vector_store.add(kept_chunks, kept_embeddings)

    return {"message": f"Removed '{body.filename}'", "chunks_stored": vector_store.count()}


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=200,
    summary="Upload PDF files for ingestion into the knowledge base",
    tags=["Ingestion"],
)
@limiter.limit(RATE_LIMIT_INGEST)
async def ingest_endpoint(
    request: Request,
    files: list[UploadFile] = File(...),
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    accepted_files: list[str] = []

    for upload in files:
        file_bytes = await upload.read()
        content_type = upload.content_type or "application/octet-stream"
        filename = upload.filename or "unknown.pdf"

        try:
            ingest_file(filename, content_type, file_bytes)
            accepted_files.append(filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ingestion failed for '{filename}': {str(e)}")

    return IngestResponse(
        message=f"{len(accepted_files)} file(s) ingested successfully.",
        files_ingested=accepted_files,
        total_chunks=vector_store.count(),
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the knowledge base with a natural-language question",
    tags=["Query"],
)
@limiter.limit(RATE_LIMIT_QUERY)
async def query_endpoint(
    request: Request,
    body: QueryRequest,
) -> QueryResponse:
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Knowledge base is empty. Please ingest PDF files first.",
        )

    retrieved_chunks, insufficient_evidence = retrieve(body.question)
    return generate(body.question, retrieved_chunks, insufficient_evidence)


@app.get("/files", summary="List all ingested files in the knowledge base", tags=["System"])
async def files_endpoint() -> dict:
    chunks = vector_store.get_all_chunks()
    return {"files": sorted(set(c.source_file for c in chunks))}


@app.get("/health", response_model=HealthResponse, summary="Health check", tags=["System"])
async def health_endpoint() -> HealthResponse:
    return HealthResponse(status="ok", chunks_stored=vector_store.count())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
