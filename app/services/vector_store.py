"""
vector_store.py — FAISS-backed vector index for the RAG pipeline.

Pipeline position:
    Extract → Chunk → Embed → [Index & Store] → Retrieve → Generate

═══════════════════════════════════════════════════════════════
HOW FAISS WORKS (high-level)
═══════════════════════════════════════════════════════════════
FAISS (Facebook AI Similarity Search) stores a large collection of dense
vectors and finds the ones most "similar" to a query vector — very fast.

Each chunk of text was turned into a list of numbers (a vector) by the
embedding model.  Chunks about similar topics produce vectors that are
"close together" in high-dimensional space.

FAISS answers the question:
    "Which stored vectors are geometrically nearest to this query vector?"

──────────────────────────────────────────────────────────────
Distance metric — IndexFlatL2 (what we use here)
──────────────────────────────────────────────────────────────
L2 = Euclidean distance:

    d(a, b) = sqrt( Σ (a_i − b_i)² )

Smaller distance → vectors are more similar → chunk is more relevant.

Alternative: IndexFlatIP uses the inner-product (dot product).
When vectors are L2-normalised first, that equals cosine similarity —
popular in NLP because it ignores vector magnitude.

We choose IndexFlatL2 because:
  • Exact brute-force search — no approximation errors.
  • No configuration needed.
  • Perfect for corpora up to ~100 k chunks.

═══════════════════════════════════════════════════════════════
SMALL EXAMPLE
═══════════════════════════════════════════════════════════════
Imagine you indexed two chunks:

  chunk 0 — "I am a software engineer with 3 years of Python experience."
  chunk 1 — "I enjoy hiking and camping in the Pacific Northwest."

Query: "What programming skills do you have?"

The embedding model maps the query to a vector that is geometrically
close to chunk 0 (both are about programming) and far from chunk 1
(outdoors activity).  FAISS returns chunk 0 as rank-1.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from app.services.embeddings import get_embedding

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Where to persist the index between server restarts.
_INDEX_DIR = Path("data/index")
_FAISS_FILE = "faiss.index"
_META_FILE  = "metadata.pkl"

# Must match the model in embeddings.py ("all-MiniLM-L6-v2" → 384 dims).
_EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# In-memory state (module-level singletons)
# ---------------------------------------------------------------------------

# The FAISS index holding all stored chunk vectors.
# Created lazily so importing this module carries no cost.
_index: faiss.Index | None = None

# Parallel list: _metadata[i] describes the vector at FAISS internal id i.
# Stored keys: doc_id, chunk_id, chunk_index, text.
_metadata: list[dict] = []

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_index() -> faiss.Index:
    """Return the live FAISS index, initialising it on first use."""
    global _index
    if _index is None:
        # IndexFlatL2: exact L2-distance search, no training required.
        # The only required argument is the number of dimensions per vector.
        _index = faiss.IndexFlatL2(_EMBEDDING_DIM)
    return _index


def _to_float32(vectors: list[list[float]]) -> np.ndarray:
    """Convert a list of float lists to a float32 NumPy matrix.

    FAISS only accepts float32 input.  Python floats and NumPy's default
    float64 must be explicitly cast.
    """
    return np.array(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API — indexing
# ---------------------------------------------------------------------------

def add_embeddings(embedded_chunks: list[dict]) -> None:
    """Store a batch of embedded chunks in the FAISS index.

    Call this right after ``embeddings.embed_chunks()`` to index a document.

    Args:
        embedded_chunks: The list of dicts returned by ``embed_chunks()``.
            Each dict must contain: doc_id, chunk_id, chunk_index,
            text, embedding.

    What happens internally:
      1. All embedding lists are stacked into a single float32 NumPy matrix.
      2. FAISS assigns sequential internal ids (0, 1, 2, …) as it adds them.
      3. We append matching metadata to ``_metadata`` in the same order so
         that ``_metadata[faiss_id]`` always retrieves the right chunk info.
    """
    if not embedded_chunks:
        return

    # Dedup guard: skip the entire batch if this doc_id is already indexed.
    doc_id = embedded_chunks[0]["doc_id"]
    if any(m["doc_id"] == doc_id for m in _metadata):
        return

    index = _get_index()

    # Step 1 — Build the (num_chunks × embedding_dim) matrix, e.g. (12, 384).
    matrix = _to_float32([ec["embedding"] for ec in embedded_chunks])

    # Step 2 — Add all vectors in one batch call; FAISS assigns contiguous ids
    # starting from `index.ntotal` (the count before this call).
    index.add(matrix)  # type: ignore[arg-type]

    # Step 3 — Keep metadata in the same order so that:
    #   _metadata[i]  ↔  FAISS internal vector id i
    for ec in embedded_chunks:
        _metadata.append(
            {
                "doc_id":      ec["doc_id"],
                "chunk_id":    ec["chunk_id"],
                "chunk_index": ec["chunk_index"],
                "text":        ec["text"],
            }
        )


# ---------------------------------------------------------------------------
# Public API — retrieval
# ---------------------------------------------------------------------------

def search(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve the top-k most relevant chunks for a user query.

    Full retrieval pipeline executed here:
      1. Embed the query with the same model used during indexing.
      2. Ask FAISS for the `top_k` nearest vectors by L2 distance.
      3. Map each returned FAISS id back to its chunk metadata.
      4. Return results ordered from most to least relevant.

    Args:
        query:  The user's question or search phrase (plain text).
        top_k:  Maximum number of chunks to return.  Defaults to 5.

    Returns:
        A list of dicts (most relevant first), each containing:
          rank        — 1-based position (1 = best match)
          score       — L2 distance; lower means more similar
          doc_id      — which document the chunk came from
          chunk_id    — unique identifier for this chunk
          chunk_index — position of the chunk within its document
          text        — the actual chunk text to pass to the LLM
    """
    index = _get_index()

    # Guard: nothing has been indexed yet.
    if index.ntotal == 0:
        return []

    # Never request more results than vectors we have stored.
    k = min(top_k, index.ntotal)

    # ── Step 1 ── Embed the query ─────────────────────────────────────────
    # get_embedding() uses the same singleton model from embeddings.py,
    # so we're guaranteed to be in the same vector space as the stored chunks.
    query_vec = np.array([get_embedding(query)], dtype=np.float32)  # shape (1, 384)

    # ── Step 2 ── Search FAISS ────────────────────────────────────────────
    # index.search() returns two arrays, each of shape (1, k):
    #   distances — L2 distances to the k nearest vectors
    #   indices   — FAISS internal ids of those vectors
    distances, indices = index.search(query_vec, k)  # type: ignore[arg-type]

    # ── Step 3 ── Map ids → metadata ─────────────────────────────────────
    results: list[dict] = []
    for rank, (faiss_id, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        # FAISS returns -1 when fewer than k vectors exist; skip those slots.
        if faiss_id == -1:
            continue

        meta = _metadata[faiss_id]
        results.append(
            {
                "rank":        rank,
                "score":       float(dist),   # L2 distance (↓ = more relevant)
                "doc_id":      meta["doc_id"],
                "chunk_id":    meta["chunk_id"],
                "chunk_index": meta["chunk_index"],
                "text":        meta["text"],
            }
        )

    return results


# ---------------------------------------------------------------------------
# Persistence — save / load
# ---------------------------------------------------------------------------

def save_index(directory: str | Path = _INDEX_DIR) -> None:
    """Write the FAISS index and metadata to disk.

    Creates two files inside `directory`:
      faiss.index  — binary FAISS index (can be large)
      metadata.pkl — Python list of chunk metadata dicts

    Call after indexing a new document so the data survives a server restart.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # FAISS has its own binary serialisation format.
    faiss.write_index(_get_index(), str(directory / _FAISS_FILE))

    # Metadata is just a Python list; pickle is compact and fast.
    with open(directory / _META_FILE, "wb") as f:
        pickle.dump(_metadata, f)


def load_index(directory: str | Path = _INDEX_DIR) -> None:
    """Load a previously saved FAISS index and metadata from disk.

    Call once at application startup (e.g. in main.py lifespan handler)
    so that search works immediately without re-indexing.

    If the files don't exist yet, the function is a safe no-op.
    """
    global _index, _metadata

    index_path = Path(directory) / _FAISS_FILE
    meta_path  = Path(directory) / _META_FILE

    if not index_path.exists() or not meta_path.exists():
        return  # Nothing saved yet — start with an empty index.

    _index = faiss.read_index(str(index_path))

    with open(meta_path, "rb") as f:
        _metadata = pickle.load(f)


def total_vectors() -> int:
    """Return the number of vectors currently in the index.

    Useful for health checks and logging.
    """
    return _get_index().ntotal
