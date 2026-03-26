import json
import os
from abc import ABC, abstractmethod

import numpy as np

from config import VECTOR_STORE_EMBEDDINGS, VECTOR_STORE_METADATA
from models import Chunk


class VectorStoreBase(ABC):
    """Interface that any vector store implementation must satisfy."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Persist a batch of chunks together with their embedding vectors."""
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        """Return top_k most similar chunks as (Chunk, cosine_score) tuples, sorted descending."""
        ...

    @abstractmethod
    def get_all_chunks(self) -> list[Chunk]:
        """Return every chunk in the store (used by BM25 in retrieval.py)."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of chunks currently stored."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all chunks and embeddings from the store."""
        ...


class NumpyVectorStore(VectorStoreBase):
    """
    Vector store using a numpy matrix for embeddings and a list of dicts for metadata.
    Both are persisted to disk (data/vectors/) after every write.
    Cosine similarity is computed in one vectorised numpy operation.
    """

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._metadata: list[dict] = []
        self._load_from_disk()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )

        if not chunks:
            return

        new_vecs = np.array(embeddings, dtype=np.float32)

        if self._embeddings is None:
            self._embeddings = new_vecs
        else:
            self._embeddings = np.vstack([self._embeddings, new_vecs])

        for chunk in chunks:
            self._metadata.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            })

        self._save_to_disk()

    def clear(self) -> None:
        self._embeddings = None
        self._metadata = []
        self._save_to_disk()

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        if self._embeddings is None or len(self._metadata) == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normed = self._embeddings / norms

        scores = normed @ q

        k = min(top_k, len(scores))
        # argpartition is O(N) vs argsort's O(N log N)
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: list[tuple[Chunk, float]] = []
        for idx in top_indices:
            meta = self._metadata[idx]
            chunk = Chunk(
                chunk_id=meta["chunk_id"],
                text=meta["text"],
                source_file=meta["source_file"],
                page_number=meta["page_number"],
                chunk_index=meta["chunk_index"],
            )
            results.append((chunk, float(scores[idx])))

        return results

    def get_all_chunks(self) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=m["chunk_id"],
                text=m["text"],
                source_file=m["source_file"],
                page_number=m["page_number"],
                chunk_index=m["chunk_index"],
            )
            for m in self._metadata
        ]

    def count(self) -> int:
        return len(self._metadata)

    def _save_to_disk(self) -> None:
        os.makedirs(os.path.dirname(VECTOR_STORE_EMBEDDINGS), exist_ok=True)

        if self._embeddings is not None:
            np.save(VECTOR_STORE_EMBEDDINGS, self._embeddings)
        elif os.path.exists(VECTOR_STORE_EMBEDDINGS):
            os.remove(VECTOR_STORE_EMBEDDINGS)

        with open(VECTOR_STORE_METADATA, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

    def _load_from_disk(self) -> None:
        if os.path.exists(VECTOR_STORE_EMBEDDINGS) and \
           os.path.exists(VECTOR_STORE_METADATA):
            self._embeddings = np.load(VECTOR_STORE_EMBEDDINGS)
            with open(VECTOR_STORE_METADATA, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)


# Single shared instance used across the entire app (ingestion + retrieval).
vector_store = NumpyVectorStore()
