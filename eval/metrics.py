"""
eval/metrics.py

Scoring rubrics and latency helpers for the evaluation framework.

This file has two purposes:
  1. Document what each manual score means so you apply them consistently
     when reviewing the output CSVs.
  2. Provide a latency_summary() helper used by report.py.

─────────────────────────────────────────────────────────────────────────────
HOW TO FILL IN MANUAL SCORES (open the CSV in Excel / Google Sheets)
─────────────────────────────────────────────────────────────────────────────

retrieval_success  (column: retrieval_success)
─────────────────
Binary: 0 or 1. Only one question to answer:

  "Do the retrieved chunks contain the information needed to answer the question?"

  1 = Yes. The supporting_evidence (from dataset.json) appears in one of the
      retrieved chunks, OR the chunks clearly contain enough information to
      construct the expected answer.
  0 = No. The retrieved chunks are from the wrong document, wrong topic,
      or completely missing the relevant passage.

  TIP: Compare the 'retrieved_chunks' column with the 'supporting_evidence'
  in dataset.json. If the evidence is there, score 1.


answer_score  (column: answer_score)
────────────
Scale: 1–5. How correct and complete is the generated answer?

  5 = Perfect. Matches expected_answer closely. No errors. No missing facts.
  4 = Good. Minor wording difference or one small detail missing.
  3 = Partial. Correct direction but incomplete or missing key facts.
  2 = Poor. Mostly wrong, off-topic, or contradicts the expected answer.
  1 = Completely wrong or hallucinated. Nothing matches expected_answer.

  TIP: Focus on factual correctness, not writing style. A short but accurate
  answer scores higher than a fluent but wrong one.


groundedness_score  (column: groundedness_score)
──────────────────
Scale: 1–5. Is every claim in the generated answer supported by the
retrieved context?

  5 = Fully grounded. Every sentence can be traced to a retrieved chunk.
  4 = Mostly grounded. One minor detail came from outside the context.
  3 = Mixed. About half the answer is grounded; the rest uses outside knowledge.
  2 = Mostly ungrounded. The answer is plausible but mostly not in the context.
  1 = Hallucinated. The answer contradicts the context or invents facts entirely.

  TIP: Read the 'retrieved_chunks' column and ask: "Did the LLM say something
  that isn't in those chunks?" Each unsupported claim lowers the score.

─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations


def latency_summary(results: list[dict]) -> dict:
    """
    Compute average latency statistics from a list of result dicts.

    Called by report.py after loading a CSV. Returns a dict with:
      avg_retrieval_ms   — average FAISS + embedding time per question
      avg_generation_ms  — average OpenAI API time per question
      avg_total_ms       — average end-to-end time per question
      avg_chunk_count    — average number of chunks retrieved

    Args:
        results: List of dicts as returned by runner.run_eval(), or
                 as read back from a CSV with csv.DictReader.
    """
    if not results:
        return {}

    def _avg(key: str) -> float:
        values = []
        for r in results:
            try:
                values.append(float(r[key]))
            except (ValueError, TypeError, KeyError):
                pass
        return round(sum(values) / len(values), 1) if values else 0.0

    return {
        "avg_retrieval_ms":  _avg("retrieval_latency_ms"),
        "avg_generation_ms": _avg("generation_latency_ms"),
        "avg_total_ms":      _avg("total_latency_ms"),
        "avg_chunk_count":   _avg("chunk_count"),
    }


def score_summary(results: list[dict]) -> dict:
    """
    Compute average manual scores from a list of result dicts.

    Returns None for each metric if no scores have been filled in yet.
    This lets report.py distinguish "not yet scored" from "scored 0".

    Args:
        results: List of dicts as returned by runner.run_eval(), or
                 as read back from a CSV with csv.DictReader.
    """
    def _avg_or_none(key: str) -> float | None:
        values = []
        for r in results:
            try:
                v = float(r[key])
                values.append(v)
            except (ValueError, TypeError, KeyError):
                pass  # empty string or missing = not yet scored
        return round(sum(values) / len(values), 2) if values else None

    return {
        "avg_retrieval_success": _avg_or_none("retrieval_success"),
        "avg_answer_score":      _avg_or_none("answer_score"),
        "avg_groundedness":      _avg_or_none("groundedness_score"),
    }
