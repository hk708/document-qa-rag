from pydantic import BaseModel


class UploadResponse(BaseModel):
    filename: str
    char_count: int
    chunk_count: int
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


class SourceChunk(BaseModel):
    rank: int
    score: float
    doc_id: str
    chunk_index: int
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
