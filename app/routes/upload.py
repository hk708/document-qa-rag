import hashlib
import json
import re
from pathlib import Path
from typing import Union

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES, RAW_DIR, PROCESSED_DIR
from app.models.schemas import UploadIndexedResponse, UploadAliasResponse
from app.services.parser import extract_text
from app.services.chunker import chunk_text
from app.services.embeddings import embed_chunks
from app.services.vector_store import (
    add_embeddings, save_index, is_doc_indexed,
    get_doc_by_hash, register_document,
)

router = APIRouter()


def _normalize_text(text: str) -> str:
    """Lowercase, strip edges, collapse all whitespace to a single space."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _content_hash(text: str) -> str:
    """Return a SHA-256 hex digest of the normalized extracted text."""
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


@router.post("/upload", response_model=Union[UploadIndexedResponse, UploadAliasResponse])
async def upload_file(file: UploadFile = File(...)):
    # ── Validate extension ─────────────────────────────────────────────────
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # ── Validate size ──────────────────────────────────────────────────────
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit.",
        )

    # ── Persist raw file (needed for extraction) ───────────────────────────
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / file.filename
    raw_path.write_bytes(contents)

    # ── Extract text ───────────────────────────────────────────────────────
    try:
        text = extract_text(raw_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {exc}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    # ── Content-based duplicate detection (hash of normalized text) ────────
    hash_val = _content_hash(text)
    existing = get_doc_by_hash(hash_val)
    if existing:
        # Same content, different (or same) filename — treat as a rename/alias.
        return UploadAliasResponse(
            status="already_exists",
            doc_id=existing["doc_id"],
            original_filename=existing["original_filename"],
            uploaded_filename=file.filename,
            rename_detected=True,
            message="This file has the same content as an existing document. No new document was indexed.",
        )

    # ── doc_id conflict guard (same stem, genuinely different content) ─────
    doc_id = Path(file.filename).stem
    if is_doc_indexed(doc_id):
        raise HTTPException(
            status_code=409,
            detail=(
                f"A different document is already indexed under the id '{doc_id}'. "
                "Rename the file or remove the existing document before re-indexing."
            ),
        )

    # ── Persist extracted text ─────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DIR / (doc_id + ".txt")
    processed_path.write_text(text, encoding="utf-8")

    # ── Chunk ──────────────────────────────────────────────────────────────
    chunks = chunk_text(text, doc_id=doc_id)

    # ── Embed ──────────────────────────────────────────────────────────────
    embedded_chunks = embed_chunks(chunks)

    # ── Save chunk snapshot ────────────────────────────────────────────────
    chunks_path = PROCESSED_DIR / (doc_id + "_chunks.json")
    chunks_path.write_text(
        json.dumps(embedded_chunks, indent=2),
        encoding="utf-8",
    )

    # ── Index + register + persist ─────────────────────────────────────────
    add_embeddings(embedded_chunks)
    register_document(doc_id=doc_id, filename=file.filename, content_hash=hash_val)
    save_index()

    return UploadIndexedResponse(
        status="indexed",
        doc_id=doc_id,
        filename=file.filename,
        char_count=len(text),
        chunk_count=len(chunks),
        rename_detected=False,
        message="Document indexed successfully.",
    )
