import math
import re
from collections import defaultdict

import numpy as np
from mistralai import Mistral

from config import (
    BM25_B,
    BM25_K1,
    MISTRAL_API_KEY,
    MISTRAL_EMBED_MODEL,
    MISTRAL_FAST_MODEL,
    RRF_K,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from models import Chunk
from storage import vector_store

_mistral = Mistral(api_key=MISTRAL_API_KEY)


def embed_query(query: str) -> list[float]:
    """
    Embed the query via Mistral API using the same model as chunk ingestion,
    ensuring both live in the same vector space for cosine comparison.
    """
    response = _mistral.embeddings.create(model=MISTRAL_EMBED_MODEL, inputs=[query])
    return response.data[0].embedding


def transform_query(query: str) -> str:
    """
    Rewrite the query as a declarative statement (HyDE-lite) to improve
    cosine match against document-style chunks. Falls back to original on error.
    """
    try:
        response = _mistral.chat.complete(
            model=MISTRAL_FAST_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the following question as a factual declarative "
                        "statement that would appear in a reference document. "
                        "Use domain-appropriate terminology matching the subject of the question. "
                        "Return only the rewritten statement, nothing else."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        transformed = response.choices[0].message.content.strip()
        return transformed if transformed else query
    except Exception:
        return query


def semantic_search(query_embedding: list[float], top_k: int = TOP_K) -> list[tuple[Chunk, float]]:
    """Retrieve top_k chunks by cosine similarity via NumpyVectorStore."""
    return vector_store.search(query_embedding, top_k=top_k)


class BM25:
    """
    BM25 scorer over a fixed corpus of chunks. Built fresh per query.

    Parameters (from config.py):
      k1 = 1.5  → term frequency saturation (diminishing returns on repetition)
      b  = 0.75 → length normalisation strength (0=none, 1=full)
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.k1 = BM25_K1
        self.b = BM25_B
        self.N = len(chunks)

        self.tokenised_corpus: list[list[str]] = [
            self._tokenise(c.text) for c in chunks
        ]
        self.avgdl: float = (
            sum(len(doc) for doc in self.tokenised_corpus) / self.N
            if self.N > 0 else 1.0
        )
        self.df: dict[str, int] = defaultdict(int)
        for doc_tokens in self.tokenised_corpus:
            for term in set(doc_tokens):
                self.df[term] += 1

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Lowercase and split on non-alphanumeric characters."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _idf(self, term: str) -> float:
        """IDF with smoothing: log((N - df + 0.5) / (df + 0.5) + 1)"""
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        """BM25 score for a single chunk given query tokens."""
        doc_tokens = self.tokenised_corpus[doc_index]
        doc_len = len(doc_tokens)

        tf_map: dict[str, int] = defaultdict(int)
        for token in doc_tokens:
            tf_map[token] += 1

        score = 0.0
        for term in query_tokens:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """Score all chunks and return top_k by BM25 score."""
        if self.N == 0:
            return []
        query_tokens = self._tokenise(query)
        scores = [(self.chunks[i], self.score(query_tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def reciprocal_rank_fusion(
    semantic_results: list[tuple[Chunk, float]],
    bm25_results: list[tuple[Chunk, float]],
    top_k: int = TOP_K,
) -> list[tuple[Chunk, float]]:
    """
    Merge semantic and BM25 result lists using Reciprocal Rank Fusion.

    RRF is preferred over weighted score averaging because cosine ([-1,1]) and
    BM25 ([0, ~20]) scores are on incompatible scales. RRF uses only rank position,
    making it robust without any tuning.

    Formula: RRF_score = Σ 1 / (RRF_K + rank)  (summed over both lists)
    RRF_K = 60 (standard value from the original RRF paper).
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, Chunk] = {}

    for rank, (chunk, _) in enumerate(semantic_results, start=1):
        rrf_scores[chunk.chunk_id] += 1.0 / (RRF_K + rank)
        chunk_map[chunk.chunk_id] = chunk

    for rank, (chunk, _) in enumerate(bm25_results, start=1):
        rrf_scores[chunk.chunk_id] += 1.0 / (RRF_K + rank)
        chunk_map[chunk.chunk_id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    return [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_ids][:top_k]


def _llm_rerank(
    query: str,
    candidates: list[tuple[Chunk, float]],
) -> list[tuple[Chunk, float]]:
    """
    Re-rank candidates using the LLM as a cross-encoder.

    Bi-encoder retrieval never sees query and chunk together, so terse table
    text can score poorly even when it contains the correct answer. The LLM
    reads query + chunk text jointly and re-orders by actual relevance.
    Falls back to RRF order on error.
    """
    if not candidates:
        return candidates

    numbered = "\n\n".join(
        f"[{i+1}] (Source: {c.source_file}, Page {c.page_number})\n{c.text[:400]}"
        for i, (c, _) in enumerate(candidates)
    )

    try:
        response = _mistral.chat.complete(
            model=MISTRAL_FAST_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a relevance ranker. Given a query and numbered text chunks, "
                        "return ONLY a comma-separated list of chunk numbers ordered from most "
                        "to least relevant to the query. Include all numbers. Example: 3,1,5,2,4"
                    ),
                },
                {"role": "user", "content": f"Query: {query}\n\nChunks:\n{numbered}"},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        indices = [int(x) - 1 for x in re.findall(r"\d+", raw)
                   if 0 < int(x) <= len(candidates)]
        seen = set(indices)
        indices += [i for i in range(len(candidates)) if i not in seen]
        return [candidates[i] for i in indices]
    except Exception:
        return candidates


def retrieve(query: str) -> tuple[list[tuple[Chunk, float]], bool]:
    """
    Full retrieval pipeline: transform → embed → semantic + BM25 → RRF → rerank → threshold.

    Returns (results, insufficient_evidence).
    insufficient_evidence is True when the best cosine score is below SIMILARITY_THRESHOLD.
    """
    transformed_query = transform_query(query)

    original_embedding = embed_query(query)
    transformed_embedding = embed_query(transformed_query)
    avg_embedding = (
        (np.array(original_embedding) + np.array(transformed_embedding)) / 2
    ).tolist()

    semantic_results = semantic_search(avg_embedding, top_k=TOP_K * 5)

    all_chunks = vector_store.get_all_chunks()
    bm25 = BM25(all_chunks)
    bm25_merged: dict[str, tuple[Chunk, float]] = {}
    for chunk, score in bm25.search(query, top_k=TOP_K * 5) + bm25.search(transformed_query, top_k=TOP_K * 5):
        if chunk.chunk_id not in bm25_merged or score > bm25_merged[chunk.chunk_id][1]:
            bm25_merged[chunk.chunk_id] = (chunk, score)
    bm25_results = sorted(bm25_merged.values(), key=lambda x: x[1], reverse=True)[:TOP_K * 5]

    fused_results = reciprocal_rank_fusion(semantic_results, bm25_results, top_k=TOP_K)
    fused_results = _llm_rerank(query, fused_results)

    top_cosine_score = semantic_results[0][1] if semantic_results else 0.0
    if top_cosine_score < SIMILARITY_THRESHOLD:
        return [], True

    return fused_results, False
