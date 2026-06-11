"""
ask.py — POST /ask endpoint.

Where this fits in the RAG pipeline:
    [upload.py already handles]
        Extract → Chunk → Embed → Index → Save

    [this file handles]
        Question → Embed → Retrieve → Build context → LLM → Answer
"""

import os

from fastapi import APIRouter, HTTPException
from openai import OpenAI, OpenAIError

from app.models.schemas import AskRequest, AskResponse, SourceChunk
from app.services.vector_store import search

router = APIRouter()

# Module-level singleton — the OpenAI client is created once and reused,
# same pattern used by the embedding model in embeddings.py.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Return the cached OpenAI client, initialising it on first call."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable is not set.",
            )
        _client = OpenAI(api_key=api_key)
    return _client


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    # ── RAG Step 1+2 ── Embed the question and retrieve top chunks ────────
    # search() internally calls get_embedding() on the question, then asks
    # FAISS for the nearest stored vectors.  Returns ranked chunk dicts.
    try:
        chunks = search(body.question, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No documents indexed yet. Please upload a document first.",
        )

    # ── RAG Step 3 ── Build the context string ────────────────────────────
    # Combine all retrieved chunk texts into one block.
    # Each chunk is labelled with its doc/position for traceability.
    context = "\n\n".join(
        f"[{c['doc_id']} — chunk {c['chunk_index']}]\n{c['text']}"
        for c in chunks
    )

    # ── RAG Step 4 ── Build the LLM prompt ───────────────────────────────
    # The prompt instructs the model to answer strictly from the context,
    # which prevents hallucination and keeps answers grounded in your docs.
    prompt = (
        "Answer the question using only the context below.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{body.question}"
    )

    # ── RAG Step 5 ── Call the LLM ───────────────────────────────────────
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # fast and cheap; swap to "gpt-4o" for higher quality
            messages=[{"role": "user", "content": prompt}],
            temperature=0,         # 0 = deterministic answers, better for factual Q&A
        )
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")

    if not response.choices:
        raise HTTPException(status_code=502, detail="LLM returned an empty response.")

    answer = response.choices[0].message.content

    # ── RAG Step 6+7 ── Return answer and the source chunks ──────────────
    # Returning sources lets the caller (or a frontend) show exactly which
    # parts of your documents the answer came from.
    sources = [
        SourceChunk(
            rank=c["rank"],
            score=c["score"],
            doc_id=c["doc_id"],
            chunk_index=c["chunk_index"],
            text=c["text"],
        )
        for c in chunks
    ]

    return AskResponse(answer=answer, sources=sources)
