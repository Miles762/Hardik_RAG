import json
import re

from mistralai import Mistral

from config import MISTRAL_API_KEY, MISTRAL_CHAT_MODEL, MISTRAL_FAST_MODEL, PII_PATTERNS
from models import Citation, Chunk, QueryIntent, QueryResponse


_mistral = Mistral(api_key=MISTRAL_API_KEY)


def _contains_pii(text: str) -> bool:
    """Return True if the query contains PII matching any pattern in config.PII_PATTERNS."""
    return any(re.search(pattern, text) for pattern in PII_PATTERNS)


def detect_intent(query: str) -> QueryIntent:
    """
    Classify the query into one of five intents using the LLM.
    PII is still caught with regex before the LLM call to avoid sending sensitive data.
    Falls back to FACTUAL on any error.
    """
    if _contains_pii(query):
        return QueryIntent.REFUSAL

    try:
        response = _mistral.chat.complete(
            model=MISTRAL_FAST_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user query into exactly one of these intents:\n"
                        "- REFUSAL: requests personal/financial/medical/legal advice, "
                        "asks whether to invest, buy, sell, or take action based on the data\n"
                        "- CHITCHAT: greetings, small talk, questions about the assistant itself\n"
                        "- LIST: explicitly asks for a list, steps, or enumeration\n"
                        "- TABLE: asks for a table, comparison, or side-by-side view\n"
                        "- FACTUAL: any other question seeking information from documents\n\n"
                        "Reply with ONLY the intent word, nothing else."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip().upper()
        intent_map = {
            "REFUSAL": QueryIntent.REFUSAL,
            "CHITCHAT": QueryIntent.CHITCHAT,
            "LIST": QueryIntent.LIST,
            "TABLE": QueryIntent.TABLE,
            "FACTUAL": QueryIntent.FACTUAL,
        }
        return intent_map.get(raw, QueryIntent.FACTUAL)
    except Exception:
        return QueryIntent.FACTUAL


def _build_system_prompt(intent: QueryIntent) -> str:
    """
    Return the system prompt for the detected intent.
    Intent-specific suffixes shape answer format (list, table, factual).
    """
    base = (
        "You are a knowledgeable assistant that answers questions strictly based on "
        "the provided context excerpts from uploaded documents. "
        "IMPORTANT RULES:\n"
        "1. Use ONLY information explicitly stated in the context. Do not recall outside knowledge.\n"
        "2. If the question specifies a time period, scope, or version, match it exactly — "
        "do not substitute figures or content from a different period or scope. "
        "If a document reports both a 3-month and a 6-month figure, use only the one "
        "that matches the period asked about. If the exact period is unavailable, say so explicitly.\n"
        "3. If a context chunk contains a table with column headers, read the headers carefully "
        "and only use the column that matches what the question asks about.\n"
        "4. When comparing multiple documents or sections, extract information from each source "
        "separately and label them clearly.\n"
        "5. Always cite sources inline as [source: filename, page N].\n"
        "6. If the information is present in the context, never say it is unavailable."
    )

    if intent == QueryIntent.FACTUAL:
        return base + " Give a concise, direct answer with exact numbers where available."

    if intent == QueryIntent.LIST:
        return base + (
            " Format your answer as a numbered or bulleted list. "
            "Each item should be on its own line."
        )

    if intent == QueryIntent.TABLE:
        return base + (
            " Format your answer as a Markdown table with clear column headers. "
            "Only include information present in the context."
        )

    return base


def _build_context_block(chunks: list[tuple[Chunk, float]]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    lines = []
    for i, (chunk, score) in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] Source: {chunk.source_file}, Page {chunk.page_number} "
            f"(relevance: {score:.2f})\n{chunk.text}"
        )
    return "\n\n---\n\n".join(lines)


def generate_answer(
    query: str,
    intent: QueryIntent,
    chunks: list[tuple[Chunk, float]],
) -> str:
    """
    Call Mistral with an intent-specific system prompt and the retrieved context.
    Temperature = 0.1 for consistent, grounded answers.
    """
    system_prompt = _build_system_prompt(intent)
    context_block = _build_context_block(chunks)

    user_message = (
        f"Context excerpts from the knowledge base:\n\n"
        f"{context_block}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"IMPORTANT: If the question specifies a time period, scope, or version, "
        f"use only information matching that exact scope — do not substitute content from a different period.\n\n"
        f"Answer based only on the context above. "
        f"Cite sources inline as [source: filename, page N]."
    )

    response = _mistral.chat.complete(
        model=MISTRAL_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()


def _check_hallucinations(
    answer: str,
    chunks: list[tuple[Chunk, float]],
) -> list[str]:
    """
    Post-hoc evidence check: send the answer and top chunks to the LLM and ask
    it to identify any claims not supported by the context.

    Returns a list of flagged sentences (empty = no flags).
    Falls back to [] on error so a verifier failure never blocks the answer.
    """
    if not answer or not chunks:
        return []

    context_block = "\n\n".join(
        f"[{i+1}] {chunk.text[:300]}"
        for i, (chunk, _) in enumerate(chunks[:8])
    )

    try:
        response = _mistral.chat.complete(
            model=MISTRAL_FAST_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fact-checker. Given an answer and the source context "
                        "it was generated from, identify any sentences in the answer that "
                        "contain claims NOT supported by the context. "
                        "Return ONLY a JSON array of unsupported sentences. "
                        "If all claims are supported, return an empty array: []. "
                        "Example: [\"Sentence one.\", \"Sentence two.\"]"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nAnswer to verify:\n{answer}",
                },
            ],
            max_tokens=512,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            flagged = json.loads(match.group())
            return [s for s in flagged if isinstance(s, str) and s.strip()]
        return []
    except Exception:
        return []


def _make_excerpt(text: str) -> str:
    """Return a 150-char excerpt, stripping any [Table: ...] header prefix."""
    if text.startswith("[Table:"):
        end = text.find("]\n\n")
        if end != -1:
            text = text[end + 3:]
    return text[:150].strip() + "..."


def _build_citations(
    chunks: list[tuple[Chunk, float]],
    answer: str,
) -> list[Citation]:
    """
    Build Citation objects for chunks actually referenced in the answer.

    Parses inline [source: filename, page N] tags from the answer text.
    Falls back to top-3 chunks by retrieval score if no inline citations found.
    """
    cited_refs: list[tuple[str, int]] = []
    for match in re.finditer(
        r"\[source:\s*([^,\]]+),\s*page\s*(\d+)\]", answer, re.IGNORECASE
    ):
        fname = match.group(1).strip()
        page = int(match.group(2))
        if (fname, page) not in cited_refs:
            cited_refs.append((fname, page))

    citations: list[Citation] = []
    seen: set[tuple[str, int]] = set()

    if cited_refs:
        chunk_lookup: dict[tuple[str, int], Chunk] = {}
        for chunk, _ in chunks:
            key = (chunk.source_file, chunk.page_number)
            if key not in chunk_lookup:
                chunk_lookup[key] = chunk

        for fname, page in cited_refs:
            key = (fname, page)
            chunk = chunk_lookup.get(key)
            if chunk is None:
                for (cf, cp), c in chunk_lookup.items():
                    if cp == page and fname.lower() in cf.lower():
                        chunk = c
                        key = (cf, cp)
                        break
            if chunk and key not in seen:
                seen.add(key)
                citations.append(Citation(
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                    excerpt=_make_excerpt(chunk.text),
                ))
    else:
        for chunk, _ in chunks[:3]:
            key = (chunk.source_file, chunk.page_number)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                    excerpt=_make_excerpt(chunk.text),
                ))

    return citations


def generate(
    query: str,
    retrieved_chunks: list[tuple[Chunk, float]],
    insufficient_evidence: bool,
) -> QueryResponse:
    """
    Full generation pipeline: intent → guard rails → generate → post-process.
    Called by the /query endpoint after retrieval.
    """
    intent = detect_intent(query)

    if intent == QueryIntent.CHITCHAT:
        return QueryResponse(
            answer="Hello! I'm a document Q&A assistant. Ask me anything about the uploaded PDFs.",
            intent=intent,
            citations=[],
        )

    if intent == QueryIntent.REFUSAL:
        return QueryResponse(
            answer=(
                "I'm unable to answer this query. It may contain personal information "
                "or touch on topics requiring professional (legal/medical) advice. "
                "Please consult a qualified professional."
            ),
            intent=intent,
            citations=[],
        )

    if insufficient_evidence or not retrieved_chunks:
        return QueryResponse(
            answer=(
                "Insufficient evidence: the knowledge base does not contain "
                "information relevant enough to answer this question confidently."
            ),
            intent=intent,
            citations=[],
            insufficient_evidence=True,
        )

    answer = generate_answer(query, intent, retrieved_chunks)
    hallucination_flags = _check_hallucinations(answer, retrieved_chunks)
    citations = _build_citations(retrieved_chunks, answer)

    return QueryResponse(
        answer=answer,
        intent=intent,
        citations=citations,
        insufficient_evidence=False,
        hallucination_flags=hallucination_flags,
    )
