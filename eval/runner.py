"""
eval/runner.py

Runs one evaluation pass against your RAG system and saves results to CSV.

Design decisions:
  - Imports search() directly from app.services.vector_store — no RAG logic
    is duplicated here.
  - Accepts an optional search_fn parameter so that experiments.py can inject
    a fresh temporary index without touching the production FAISS index.
  - Saves results as CSV so you can open them in Excel/Numbers to fill in
    manual scores (retrieval_success, answer_score, groundedness_score).
  - The prompt mirrors what ask.py sends to the LLM — same model, same
    temperature=0, same "only use context" instruction — so eval results
    reflect real production behavior.

Usage (single eval against the current production index):
    python -m eval.runner

Usage (programmatic, with a custom search function):
    from eval.runner import run_eval, save_results
    results = run_eval(config_name="my_config", top_k=5, search_fn=my_fn)
    save_results(results, run_label="my_config")
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from openai import OpenAI

# Reuse the existing search function — this is the only RAG import needed.
from app.services.vector_store import search as _default_search

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
DATASET_PATH = EVAL_DIR / "dataset.json"

# ---------------------------------------------------------------------------
# OpenAI client (same singleton pattern as ask.py)
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Run: $env:OPENAI_API_KEY='your-key' (PowerShell) or "
                "export OPENAI_API_KEY='your-key' (bash)"
            )
        _client = OpenAI(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Prompt builder (mirrors ask.py style for consistent evaluation)
# ---------------------------------------------------------------------------

def _build_eval_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build the LLM prompt for evaluation.

    Mirrors the ask.py prompt style on purpose: if you change the production
    prompt, update this too so eval results stay representative.
    """
    context = "\n\n".join(
        f"[{c['doc_id']} — chunk {c['chunk_index']}]\n{c['text']}"
        for c in chunks
    )
    return (
        "You are a careful document analyst. Use only the context below.\n"
        "If the answer is not present in the context, say exactly: "
        "'Not found in context.'\n"
        "Do not invent facts or add external knowledge.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def run_eval(
    config_name: str = "default",
    top_k: int = 7,
    dataset_path: Path = DATASET_PATH,
    search_fn: Callable | None = None,
) -> list[dict]:
    """
    Run one evaluation pass and return a list of result dicts.

    Args:
        config_name:  Label for this run (e.g. "chunk700_ovlp120_topk7").
                      Appears in every CSV row so you can filter by config.
        top_k:        How many chunks to retrieve per question.
        dataset_path: Path to dataset.json (or a subset for quick runs).
        search_fn:    Optional replacement for vector_store.search().
                      experiments.py injects a temp-index search function here.
                      Signature must match: fn(question: str, top_k: int) -> list[dict]

    Returns:
        List of result dicts — one per question. Each dict has all fields
        needed for the CSV output (see save_results below).
    """
    questions = json.loads(dataset_path.read_text(encoding="utf-8"))
    client = _get_client()
    search = search_fn or _default_search
    results: list[dict] = []

    print(f"\nRunning evaluation — config: '{config_name}', top_k: {top_k}")
    print(f"Questions: {len(questions)}")
    print("-" * 50)

    for item in questions:
        question = item["question"]
        question_id = item.get("question_id", "?")

        # ── Step 1: Retrieval ──────────────────────────────────────────────
        # Time how long FAISS + embedding takes.
        t_ret = time.perf_counter()
        chunks = search(question, top_k=top_k)
        retrieval_ms = round((time.perf_counter() - t_ret) * 1000, 1)

        # ── Step 2: Generation ─────────────────────────────────────────────
        # Time the OpenAI API call separately so you can see where latency
        # is actually spent (retrieval vs generation).
        prompt = _build_eval_prompt(question, chunks)
        t_gen = time.perf_counter()
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,       # deterministic — same question always produces same answer
            max_tokens=400,
        )
        generation_ms = round((time.perf_counter() - t_gen) * 1000, 1)
        generated_answer = (response.choices[0].message.content or "").strip()

        # ── Step 3: Record result ──────────────────────────────────────────
        # Store each chunk text truncated to 300 chars, separated by |||
        # This keeps the CSV readable without losing chunk identity.
        chunk_texts = " ||| ".join(
            f"[{c['doc_id']} chunk {c['chunk_index']}] {c['text'][:300]}"
            for c in chunks
        )

        results.append({
            "question_id":          question_id,
            "question":             question,
            "category":             item.get("category", ""),
            "expected_answer":      item.get("expected_answer", ""),
            "generated_answer":     generated_answer,
            "retrieved_chunks":     chunk_texts,
            "chunk_count":          len(chunks),
            "retrieval_latency_ms": retrieval_ms,
            "generation_latency_ms": generation_ms,
            "total_latency_ms":     round(retrieval_ms + generation_ms, 1),
            "config_name":          config_name,
            # ── Manual scoring columns (fill these in after reviewing) ─────
            # retrieval_success: 1 = the retrieved chunks contain the answer,
            #                    0 = they do not
            # answer_score:      1-5 scale (see eval/metrics.py for rubric)
            # groundedness_score: 1-5 scale (see eval/metrics.py for rubric)
            "retrieval_success":    "",
            "answer_score":         "",
            "groundedness_score":   "",
        })

        print(
            f"  [{question_id}] retrieval {retrieval_ms}ms | "
            f"generation {generation_ms}ms | "
            f"chunks retrieved: {len(chunks)}"
        )

    return results


# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------

def save_results(results: list[dict], run_label: str) -> Path:
    """
    Write results to a timestamped CSV file in eval/results/.

    Why CSV (not JSON)?
      - You can open it directly in Excel or Google Sheets.
      - You can add manual scores in new columns without writing code.
      - report.py reads it back with csv.DictReader — simple and portable.

    Args:
        results:   List returned by run_eval().
        run_label: Short label used in the filename (e.g. "chunk700_topk7").

    Returns:
        Path to the saved CSV file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not results:
        print("No results to save.")
        return RESULTS_DIR / "empty.csv"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{run_label}_{timestamp}.csv"

    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved → {out_path}")
    print("Open the CSV and fill in: retrieval_success, answer_score, groundedness_score")
    print("Then run: python -m eval.report")
    return out_path


# ---------------------------------------------------------------------------
# Entry point — run against the current production index
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load the production FAISS index from disk before searching.
    # experiments.py does NOT call this — it builds its own index.
    from app.services.vector_store import load_index
    load_index()

    results = run_eval(config_name="default_topk7", top_k=7)
    save_results(results, run_label="default_topk7")
