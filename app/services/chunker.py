from __future__ import annotations
import re
import uuid
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[Chunk]:
    sentences = _split_into_sentences(text)
    chunks: list[Chunk] = []
    current = ""
    index = 0

    for sentence in sentences:
        if current and len(current) + len(sentence) > chunk_size:
            chunks.append(_make_chunk(current.strip(), doc_id, index))
            index += 1
            current = _tail(current, overlap) + " " + sentence
        else:
            current = (current + " " + sentence).strip()

    if current.strip():
        chunks.append(_make_chunk(current.strip(), doc_id, index))

    return chunks


def _split_into_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _tail(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    tail = text[-n:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def _make_chunk(text: str, doc_id: str, index: int) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        text=text,
        chunk_index=index,
    )
