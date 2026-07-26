"""
eval/experiments.py

Run the same evaluation dataset under multiple configurations and compare results.

Why this file exists:
  Your production pipeline hard-codes chunk_size=700, overlap=120, top_k=7
  in chunker.py and ask.py. This file lets you ask: "What if I used smaller
  chunks? What if I retrieved more chunks?" without touching production code.

How it works:
  1. For each ExperimentConfig (chunk_size × overlap × top_k):
       a. Read all .txt files from data/processed/
       b. Re-chunk them using the config's chunk_size and overlap
          (calls the same chunker.chunk_text() your production pipeline uses)
       c. Re-embed each chunk using the same SentenceTransformer model
       d. Build a fresh faiss.IndexFlatL2 in memory — the production index
          on disk is never touched
       e. Create a search_fn that queries this temporary index
       f. Pass search_fn to runner.run_eval() — the runner doesn't know
          or care whether the index is production or experimental
       g. Save results as a CSV in eval/results/

  2. After all configs finish, print a summary table.

Why re-chunk .txt files instead of chunk JSON files?
  The _chunks.json files in data/processed/ were produced with chunk_size=700
  and overlap=120. To test different settings we must re-chunk from raw text.
  The .txt files are the extracted plain text — the same input your upload
  pipeline uses after parsing PDFs.

Usage:
    python -m eval.experiments

To run a subset of configs, edit the EXPERIMENTS list below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.services.chunker import chunk_text
from app.services.embeddings import embed_chunks, get_embedding
from eval.runner import run_eval, save_results

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """
    One combination of hyperparameters to evaluate.

    chunk_size:  Target character count per chunk (chunker.py uses this as
                 a soft ceiling — it never splits mid-sentence).
    overlap:     How many characters of the previous chunk's tail are
                 prepended to the next chunk, preserving sentence context
                 across boundaries.
    top_k:       How many chunks to retrieve per question.
    """
    chunk_size: int
    overlap: int
    top_k: int

    @property
    def name(self) -> str:
        """Stable string label used in filenames and the report table."""
        return f"chunk{self.chunk_size}_ovlp{self.overlap}_topk{self.top_k}"


# ── Edit this list to add or remove experiments ───────────────────────────
#
# Start with a focused grid. Running all 27 combinations (3×3×3) takes time
# and OpenAI tokens. The configs below cover the most informative corners:
#   - Small chunks (400) vs current default (700) vs large chunks (1000)
#   - Low overlap vs current default vs high overlap
#   - Low top-k (3) vs current default (7)
#
EXPERIMENTS: list[ExperimentConfig] = [
    # Small chunks — tight overlap — conservative retrieval
    ExperimentConfig(chunk_size=400,  overlap=50,  top_k=3),
    ExperimentConfig(chunk_size=400,  overlap=50,  top_k=5),

    # Current production defaults (baseline for comparison)
    ExperimentConfig(chunk_size=700,  overlap=120, top_k=5),
    ExperimentConfig(chunk_size=700,  overlap=120, top_k=7),  # ← production default

    # Large chunks — generous overlap — broad retrieval
    ExperimentConfig(chunk_size=1000, overlap=200, top_k=5),
    ExperimentConfig(chunk_size=1000, overlap=200, top_k=7),
]

# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors


def _load_txt_files() -> list[tuple[str, str]]:
    """
    Return (doc_id, text) pairs for every .txt file in data/processed/.

    We skip _chunks.json and other non-.txt files. The doc_id is the
    filename stem, matching how upload.py assigns doc_ids.
    """
    docs: list[tuple[str, str]] = []
    for path in sorted(PROCESSED_DIR.glob("*.txt")):
        doc_id = path.stem
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append((doc_id, text))
    return docs


def _build_temp_index(config: ExperimentConfig) -> tuple[faiss.Index, list[dict]]:
    """
    Re-chunk and re-embed all .txt files, then build a fresh FAISS index.

    Returns:
        (index, metadata) — same structure as vector_store.py globals,
        but held only in memory. The production index is not modified.
    """
    index = faiss.IndexFlatL2(_EMBEDDING_DIM)
    metadata: list[dict] = []

    docs = _load_txt_files()
    if not docs:
        raise FileNotFoundError(
            f"No .txt files found in {PROCESSED_DIR}. "
            "Make sure you have uploaded and processed at least one document."
        )

    print(f"  Loading {len(docs)} document(s) from {PROCESSED_DIR}")
    print(f"  Chunking: size={config.chunk_size}, overlap={config.overlap}")

    for doc_id, text in docs:
        # Reuse the exact same chunking function as the production pipeline.
        chunks = chunk_text(
            text,
            doc_id=doc_id,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
        )

        if not chunks:
            continue

        # Reuse the exact same embedding function as the production pipeline.
        embedded = embed_chunks(chunks)

        # Build the float32 matrix and add to FAISS.
        matrix = np.array([e["embedding"] for e in embedded], dtype=np.float32)
        index.add(matrix)  # type: ignore[arg-type]

        # Keep metadata in the same order as FAISS internal ids.
        for ec in embedded:
            metadata.append({
                "doc_id":      ec["doc_id"],
                "chunk_id":    ec["chunk_id"],
                "chunk_index": ec["chunk_index"],
                "text":        ec["text"],
            })

    print(f"  Index built: {index.ntotal} total chunks")
    return index, metadata


def _make_search_fn(index: faiss.Index, metadata: list[dict], top_k: int):
    """
    Return a search function that queries the temporary index.

    The returned function has the same signature as vector_store.search()
    so runner.run_eval() can accept it without any changes.
    """
    def search_fn(question: str, top_k: int = top_k) -> list[dict]:
        embedding = get_embedding(question)
        query = np.array([embedding], dtype=np.float32)
        distances, indices = index.search(query, top_k)  # type: ignore[arg-type]
        results: list[dict] = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 when fewer results exist than top_k
                continue
            entry = metadata[idx].copy()
            entry["rank"] = rank + 1
            entry["score"] = float(dist)
            results.append(entry)
        return results

    return search_fn


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiments(configs: list[ExperimentConfig] | None = None) -> None:
    """
    Run every config in sequence and save one CSV per config.

    Args:
        configs: List of ExperimentConfig objects. Defaults to EXPERIMENTS above.
                 Pass a shorter list for quick testing:
                   run_experiments([ExperimentConfig(700, 120, 7)])
    """
    if configs is None:
        configs = EXPERIMENTS

    completed: list[str] = []

    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"Experiment: {config.name}")
        print(f"{'=' * 60}")

        # Build a completely isolated in-memory index for this config.
        # The production index in data/index/ is not touched.
        temp_index, temp_metadata = _build_temp_index(config)
        search_fn = _make_search_fn(temp_index, temp_metadata, config.top_k)

        # Run evaluation using the temporary index.
        results = run_eval(
            config_name=config.name,
            top_k=config.top_k,
            search_fn=search_fn,
        )
        save_results(results, run_label=config.name)
        completed.append(config.name)

    # Print a summary so you know what to do next.
    print(f"\n{'=' * 60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Configs run: {len(completed)}")
    for name in completed:
        print(f"  - {name}")
    print()
    print("Next steps:")
    print("  1. Open eval/results/*.csv in Excel or Google Sheets")
    print("  2. For each row, fill in:")
    print("       retrieval_success  → 0 or 1")
    print("       answer_score       → 1-5  (see eval/metrics.py)")
    print("       groundedness_score → 1-5  (see eval/metrics.py)")
    print("  3. Save the CSVs")
    print("  4. Run: python -m eval.report")


if __name__ == "__main__":
    run_experiments()
