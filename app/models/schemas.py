from typing import Literal

from pydantic import BaseModel


class UploadIndexedResponse(BaseModel):
    """Returned when a document is successfully chunked and indexed for the first time."""
    status: str          # always "indexed"
    doc_id: str
    filename: str
    char_count: int
    chunk_count: int
    rename_detected: bool  # always False for new uploads
    message: str


class UploadAliasResponse(BaseModel):
    """Returned when uploaded content matches an already-indexed document (rename case)."""
    status: str          # always "already_exists"
    doc_id: str          # the existing doc_id whose content matched
    original_filename: str   # filename used when the document was first indexed
    uploaded_filename: str   # filename just uploaded
    rename_detected: bool    # always True
    message: str


class ChunkSchema(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int


# ---------------------------------------------------------------------------
# /ask endpoint models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    user_id: str = "local_user"
    answer_mode: Literal["concise", "detailed", "bullet_summary"] = "detailed"


class SourceChunk(BaseModel):
    rank: int
    score: float
    doc_id: str
    chunk_index: int
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    conversation_id: str
    conversation_title: str


class ConversationMeta(BaseModel):
    conversation_id: str
    user_id: str
    title: str
    summary: str
    created_at: str
    updated_at: str


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: str
    sources: list[SourceChunk] | None = None


class CreateConversationRequest(BaseModel):
    user_id: str = "local_user"
    title: str | None = None


class ConversationDetail(BaseModel):
    conversation: ConversationMeta
    messages: list[ConversationMessage]
