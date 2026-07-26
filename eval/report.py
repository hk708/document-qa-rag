"""
eval/report.py

Reads all CSVs from eval/results/ and generates a summary report.

The report does three things:
  1. Shows a per-config table with average latency and average scores.
  2. Highlights the best config per metric.
  3. Writes plain-English conclusions you can quote during a portfolio review
     or engineering interview.

Run this after you have filled in manual scores in the CSV files:
    python -m eval.report

The report is printed to the terminal and also saved as a .txt file in
eval/reports/ with a timestamp so you can keep a history.
"""

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from eval.metrics import latency_summary, score_summary

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"
REPORTS_DIR = Path(__file__).parent / "reports"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_all_results() -> list[dict]:
    """Read every CSV in eval/results/ and return all rows combined."""
    all_rows: list[dict] = []
    for csv_path in sorted(RESULTS_DIR.glob("*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
    return all_rows


def _group_by_config(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        config = row.get("config_name") or "unknown"
        groups[config].append(row)
    return groups


# ---------------------------------------------------------------------------
# Table formatting helpers
# ---------------------------------------------------------------------------


def _fmt_ms(value: float | None) -> str:
    """Format a millisecond value for the table, or '-' if missing."""
    return f"{value:.0f}ms" if value is not None else "  -"


def _fmt_score(value: float | None) -> str:
    """Format a score value for the table, or '-' if not yet filled in."""
    return f"{value:.2f}" if value is not None else "  -"


def _best_config(stats_map: dict[str, dict], metric: str, prefer_max: bool) -> str | None:
    """Return the config name that is best on a given metric."""
    candidates = {
        name: stats[metric]
        for name, stats in stats_map.items()
        if stats.get(metric) is not None
    }
    if not candidates:
        return None
    return (max if prefer_max else min)(candidates, key=lambda k: candidates[k])


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def generate_report() -> str:
    rows = _load_all_results()
    if not rows:
        return (
            "No results found in eval/results/.\n"
            "Run: python -m eval.runner   (single config)\n"
            "  or python -m eval.experiments  (all configs)"
        )

    groups = _group_by_config(rows)

    # Compute stats for each config
    all_stats: dict[str, dict] = {}
    for config_name, group_rows in groups.items():
        lat = latency_summary(group_rows)
        scr = score_summary(group_rows)
        all_stats[config_name] = {**lat, **scr, "question_count": len(group_rows)}

    # ── Build the report string ───────────────────────────────────────────
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("RAG EVALUATION REPORT")
    lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Questions : {len(rows)} total across {len(groups)} config(s)")
    lines.append("=" * 72)

    # ── Per-config summary table ──────────────────────────────────────────
    lines.append("")
    lines.append("PER-CONFIG SUMMARY")
    lines.append("")

    col_config  = 36
    col_num     = 4
    col_lat     = 8
    col_score   = 6

    header = (
        f"{'Config':<{col_config}} "
        f"{'Qs':>{col_num}} "
        f"{'RetMS':>{col_lat}} "
        f"{'GenMS':>{col_lat}} "
        f"{'TotMS':>{col_lat}} "
        f"{'Chunks':>{col_lat}} "
        f"{'Ret%':>{col_score}} "
        f"{'Ans':>{col_score}} "
        f"{'Gnd':>{col_score}}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for config_name in sorted(all_stats):
        s = all_stats[config_name]
        lines.append(
            f"{config_name:<{col_config}} "
            f"{s['question_count']:>{col_num}} "
            f"{_fmt_ms(s.get('avg_retrieval_ms')):>{col_lat}} "
            f"{_fmt_ms(s.get('avg_generation_ms')):>{col_lat}} "
            f"{_fmt_ms(s.get('avg_total_ms')):>{col_lat}} "
            f"{_fmt_ms(s.get('avg_chunk_count')):>{col_lat}} "
            f"{_fmt_score(s.get('avg_retrieval_success')):>{col_score}} "
            f"{_fmt_score(s.get('avg_answer_score')):>{col_score}} "
            f"{_fmt_score(s.get('avg_groundedness')):>{col_score}}"
        )

    lines.append("")
    lines.append(
        "Columns: Ret% = retrieval success (0-1) | "
        "Ans = answer score (1-5) | Gnd = groundedness (1-5)"
    )
    lines.append("'-' means scores have not been filled in yet.")

    # ── Best config per metric ────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 72)
    lines.append("BEST CONFIG PER METRIC")
    lines.append("=" * 72)
    lines.append("")

    metric_definitions = [
        ("Fastest retrieval",      "avg_retrieval_ms",       False),
        ("Fastest generation",     "avg_generation_ms",      False),
        ("Lowest total latency",   "avg_total_ms",           False),
        ("Best retrieval success", "avg_retrieval_success",  True),
        ("Best answer score",      "avg_answer_score",       True),
        ("Best groundedness",      "avg_groundedness",       True),
    ]

    for label, metric, prefer_max in metric_definitions:
        best = _best_config(all_stats, metric, prefer_max)
        if best is not None:
            value = all_stats[best][metric]
            formatted = _fmt_ms(value) if "ms" in metric else _fmt_score(value)
            lines.append(f"  {label:<30} → {best}  ({formatted})")
        else:
            lines.append(
                f"  {label:<30} → (no data — fill in manual scores first)"
            )

    # ── Auto-generated engineering conclusions ────────────────────────────
    lines.append("")
    lines.append("=" * 72)
    lines.append("ENGINEERING CONCLUSIONS")
    lines.append("=" * 72)
    lines.append("")

    # Retrieval success conclusion
    ret_best = _best_config(all_stats, "avg_retrieval_success", True)
    if ret_best:
        ret_score = all_stats[ret_best]["avg_retrieval_success"]
        lines.append(
            f"  RETRIEVAL: '{ret_best}' achieved the highest retrieval success "
            f"({ret_score:.0%} of questions returned relevant chunks)."
        )
    else:
        lines.append(
            "  RETRIEVAL: Fill in retrieval_success scores to see conclusions."
        )

    # Answer quality conclusion
    ans_best = _best_config(all_stats, "avg_answer_score", True)
    if ans_best:
        ans_score = all_stats[ans_best]["avg_answer_score"]
        lines.append(
            f"  QUALITY:   '{ans_best}' produced the best answers "
            f"(avg score {ans_score:.2f}/5)."
        )
    else:
        lines.append(
            "  QUALITY:   Fill in answer_score scores to see conclusions."
        )

    # Latency conclusion
    lat_best = _best_config(all_stats, "avg_total_ms", False)
    lat_worst = _best_config(all_stats, "avg_total_ms", True)
    if lat_best and lat_worst and lat_best != lat_worst:
        fast_ms = all_stats[lat_best]["avg_total_ms"]
        slow_ms = all_stats[lat_worst]["avg_total_ms"]
        diff_pct = round(((slow_ms - fast_ms) / fast_ms) * 100)
        lines.append(
            f"  LATENCY:   '{lat_best}' was the fastest config ({fast_ms:.0f}ms avg). "
            f"'{lat_worst}' was the slowest ({slow_ms:.0f}ms avg, "
            f"{diff_pct}% slower)."
        )
    elif lat_best:
        lines.append(
            f"  LATENCY:   Fastest config: '{lat_best}' "
            f"({all_stats[lat_best]['avg_total_ms']:.0f}ms avg total latency)."
        )

    # Chunk size note (parse from config name if possible)
    chunk_scores: dict[int, list[float]] = defaultdict(list)
    for config_name, s in all_stats.items():
        ret_val = s.get("avg_retrieval_success")
        if ret_val is not None and "chunk" in config_name:
            try:
                # Config names look like: chunk700_ovlp120_topk7
                size = int(config_name.split("chunk")[1].split("_")[0])
                chunk_scores[size].append(ret_val)
            except (ValueError, IndexError):
                pass

    if len(chunk_scores) > 1:
        avg_by_size = {
            size: round(sum(vals) / len(vals), 2)
            for size, vals in chunk_scores.items()
        }
        best_size = max(avg_by_size, key=lambda k: avg_by_size[k])
        lines.append(
            f"  CHUNK SIZE: Chunk size {best_size} achieved the highest average "
            f"retrieval success ({avg_by_size[best_size]:.0%}) across all top-k settings."
        )

    lines.append("")
    lines.append("─" * 72)
    lines.append("SCORING REMINDER")
    lines.append("─" * 72)
    lines.append("  retrieval_success : 1 = chunks contain the answer, 0 = they do not")
    lines.append("  answer_score      : 1-5  (1=wrong, 3=partial, 5=perfect)")
    lines.append("  groundedness_score: 1-5  (1=hallucinated, 5=fully grounded)")
    lines.append("  See eval/metrics.py for the full rubric.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = generate_report()
    print(report)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"report_{timestamp}.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved → {report_path}")
