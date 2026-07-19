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
from app.services.conversation_store import get_conversation_store
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


def _auto_title(question: str) -> str:
    q = question.strip().lower()
    if "resume" in q and ("summarize" in q or "summary" in q or "analyze" in q):
        return "Resume Analysis"
    if "bert" in q:
        return "BERT Questions"
    words = [w for w in question.strip().split() if w]
    if not words:
        return "New Chat"
    return " ".join(words[:4]).strip(" ?.!") + ""


def _maybe_update_summary(conversation_id: str, messages: list[dict], existing_summary: str) -> None:
    # Keep this simple: summarize periodically once the conversation is long.
    if len(messages) < 12 or len(messages) % 4 != 0:
        return

    older_messages = messages[:-8]
    if not older_messages:
        return

    transcript = "\n".join(
        f"{m['role']}: {m['content']}" for m in older_messages
    )

    prompt = (
        "Summarize this conversation history for future QA context.\n"
        "Keep it factual and compact (5-8 lines).\n"
        "Preserve key entities, definitions, constraints, and unresolved items.\n"
        f"Existing summary:\n{existing_summary or '(none)'}\n\n"
        f"Transcript:\n{transcript}"
    )

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=220,
        )
    except OpenAIError:
        return

    if not response.choices:
        return

    summary = response.choices[0].message.content or ""
    if summary.strip():
        get_conversation_store().update_summary(conversation_id, summary.strip())


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    mode = body.answer_mode
    store = get_conversation_store()

    if body.conversation_id:
        conv = store.get_conversation(body.conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        conversation_id = body.conversation_id
    else:
        conv = store.create_conversation(
            user_id=body.user_id,
            title=_auto_title(body.question),
        )
        conversation_id = conv["conversation_id"]

    messages = store.get_messages(conversation_id)
    summary = conv.get("summary", "")

    # ── RAG Step 1+2 ── Embed the question and retrieve top chunks ────────
    # search() internally calls get_embedding() on the question, then asks
    # FAISS for the nearest stored vectors.  Returns ranked chunk dicts.
    try:
        chunks = search(body.question, top_k=7)
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

    recent_user_messages = [m["content"] for m in messages if m["role"] == "user"][-4:]
    recent_assistant_messages = [m["content"] for m in messages if m["role"] == "assistant"][-4:]

    memory_block = (
        f"Conversation summary:\n{summary or '(no summary yet)'}\n\n"
        f"Recent user messages:\n" + "\n".join(f"- {m}" for m in recent_user_messages)
        + "\n\nRecent assistant messages:\n"
        + "\n".join(f"- {m}" for m in recent_assistant_messages)
    )

    # ── RAG Step 4 ── Build the LLM prompt ───────────────────────────────
    # The prompt instructs the model to answer strictly from the context,
    # which prevents hallucination and keeps answers grounded in your docs.
    if mode == "concise":
        mode_instruction = (
            "Answer in 2-4 sentences. Give only the direct answer. "
            "Use only the provided context and do not add extra commentary."
        )
    elif mode == "bullet_summary":
        mode_instruction = (
            "Return 4-8 bullet points with key facts only. "
            "Keep each bullet concise and grounded in the context."
        )
    else:  # detailed
        mode_instruction = (
            "Give a detailed answer with clear structure and direct evidence from context.\n"
            "When helpful, include short quoted phrases from the context."
        )

    prompt = (
        "You are a careful document analyst. Use only the context below.\n"
        "If the answer is not present in the context, explicitly say you don't know.\n"
        "Do not invent facts or external details.\n"
        f"{mode_instruction}\n\n"
        f"Conversation memory:\n{memory_block}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{body.question}"
    )

    max_tokens_by_mode = {
        "concise": 220,
        "bullet_summary": 550,
        "detailed": 1200,
    }

    # ── RAG Step 5 ── Call the LLM ───────────────────────────────────────
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # fast and cheap; swap to "gpt-4o" for higher quality
            messages=[{"role": "user", "content": prompt}],
            temperature=0,         # 0 = deterministic answers, better for factual Q&A
            max_tokens=max_tokens_by_mode[mode],
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

    store.add_message(conversation_id, "user", body.question)
    store.add_message(
        conversation_id,
        "assistant",
        answer or "",
        sources=[s.model_dump() for s in sources],
    )

    updated_messages = store.get_messages(conversation_id)
    _maybe_update_summary(conversation_id, updated_messages, summary)

    refreshed = store.get_conversation(conversation_id)
    title = refreshed["title"] if refreshed else conv.get("title", "New Chat")
    return AskResponse(
        answer=answer or "",
        sources=sources,
        conversation_id=conversation_id,
        conversation_title=title,
    )
