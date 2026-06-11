import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES, RAW_DIR, PROCESSED_DIR
from app.models.schemas import UploadResponse
from app.services.parser import extract_text
from app.services.chunker import chunk_text
from app.services.embeddings import embed_chunks
from app.services.vector_store import add_embeddings, save_index

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    # Validate extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read and validate size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit.",
        )

    # Persist raw file
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / file.filename
    raw_path.write_bytes(contents)

    # Extract text
    try:
        text = extract_text(raw_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {exc}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    # Persist extracted text
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DIR / (Path(file.filename).stem + ".txt")
    processed_path.write_text(text, encoding="utf-8")

    # Chunk the extracted text
    doc_id = Path(file.filename).stem
    chunks = chunk_text(text, doc_id=doc_id)

    # Generate embeddings for every chunk.
    # Each item in embedded_chunks is a dict with:
    #   doc_id, chunk_id, text, embedding (list[float])
    embedded_chunks = embed_chunks(chunks)

    # Save chunks + embeddings to JSON for inspection.
    chunks_path = PROCESSED_DIR / (doc_id + "_chunks.json")
    chunks_path.write_text(
        json.dumps(embedded_chunks, indent=2),
        encoding="utf-8",
    )

    # RAG: Index → Save
    # Add all chunk vectors for this document into the shared FAISS index,
    # then persist the updated index to disk so it survives server restarts.
    add_embeddings(embedded_chunks)
    save_index()

    return UploadResponse(
        filename=file.filename,
        char_count=len(text),
        chunk_count=len(chunks),
        message="File uploaded and text extracted successfully.",
    )
