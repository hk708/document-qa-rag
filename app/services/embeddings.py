"""
embeddings.py — Embedding service for the RAG pipeline.

Responsibility: given a list of Chunk objects, produce a dense vector
(embedding) for each chunk's text so they can later be stored in a
FAISS (or any other) vector index for semantic retrieval.

Why sentence-transformers?
  - Pre-trained models that are fast and accurate out of the box.
  - 'all-MiniLM-L6-v2' is a great starter model: small (80 MB),
    fast, and produces 384-dimensional vectors that work well for
    many retrieval tasks.
  - Drop-in swap: swap _MODEL_NAME later without changing any other file.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.services.chunker import Chunk

# ---------------------------------------------------------------------------
# Model name — change this single constant to experiment with other models.
# Popular alternatives:
#   "all-mpnet-base-v2"   — higher quality, 768-dim, ~420 MB
#   "multi-qa-MiniLM-L6-cos-v1" — tuned for Q&A retrieval
# ---------------------------------------------------------------------------
_MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singleton: the model is loaded once when this module is first
# imported and reused for every subsequent call.  Loading a transformer model
# is expensive (~1–2 s); we never want to do it per-request.
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return the cached model, loading it on first call."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[Chunk]) -> list[dict]:
    """Generate an embedding for every chunk and return enriched records.

    Args:
        chunks: Output of ``chunker.chunk_text()``.

    Returns:
        A list of dicts, one per chunk, each containing:
          - doc_id    : document identifier (same across all chunks of a file)
          - chunk_id  : unique UUID for this specific chunk
          - text      : the raw text of the chunk
          - embedding : list[float] of length 384 (for MiniLM-L6-v2)

        The embedding field is a plain Python list so it can be JSON-serialised
        and later loaded directly into a FAISS index.
    """
    if not chunks:
        return []

    model = _get_model()

    # Extract just the text for batched encoding — much faster than encoding
    # one chunk at a time because the model can parallelise internally.
    texts = [chunk.text for chunk in chunks]

    # encode() returns a NumPy array of shape (num_chunks, embedding_dim).
    # batch_size controls memory usage; 32 is safe for most machines.
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,   # set True if you want a tqdm bar in the terminal
        convert_to_numpy=True,     # keeps output as ndarray for .tolist() below
    )

    # Build the final records that callers (upload route, pipeline, tests) consume.
    embedded_chunks: list[dict] = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append(
            {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                # Convert numpy float32 array → plain Python list[float] so the
                # result is JSON-serialisable and FAISS-ingestible without extra steps.
                "embedding": embedding.tolist(),
            }
        )

    return embedded_chunks


def get_embedding(text: str) -> list[float]:
    """Return a single embedding vector for a plain text string.

    Used by the vector store to embed a user query at search time using
    the exact same model that was used when indexing the chunks — this is
    critical for meaningful similarity comparisons.

    Args:
        text: Any string, e.g. a user question.

    Returns:
        A list of 384 floats (for all-MiniLM-L6-v2).
    """
    model = _get_model()
    # encode() with a single string still returns a 2-D array (1, dim);
    # [0] gives us the 1-D vector, .tolist() converts numpy → plain Python.
    return model.encode([text], convert_to_numpy=True)[0].tolist()
