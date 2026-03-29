import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_CHAT_MODEL: str = "mistral-small-2506"
MISTRAL_EMBED_MODEL: str = "mistral-embed"
MISTRAL_FAST_MODEL: str = "mistral-small-latest"  


CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 200

TOP_K: int = 12                  
SIMILARITY_THRESHOLD: float = 0.20  
BM25_K1: float = 1.5
BM25_B: float = 0.75
RRF_K: int = 60  

ALLOWED_MIME_TYPES: list[str] = ["application/pdf"]
MAX_FILE_SIZE_MB: int = 20
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

VECTOR_STORE_DIR: str = "data/vectors"
VECTOR_STORE_EMBEDDINGS: str = f"{VECTOR_STORE_DIR}/embeddings.npy"
VECTOR_STORE_METADATA: str = f"{VECTOR_STORE_DIR}/metadata.json"

RATE_LIMIT_INGEST: str = "10/minute"
RATE_LIMIT_QUERY: str = "30/minute"

# PII refusal patterns (regex)
PII_PATTERNS: list[str] = [
    r"\b\d{3}-\d{2}-\d{4}\b",                                    # US SSN
    r"\b4[0-9]{12}(?:[0-9]{3})?\b",                              # Visa card
    r"\b5[1-5][0-9]{14}\b",                                      # MasterCard
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",          # email
]
