"""
experiments/benchmark.py
------------------------
Measures compression ratio and quality loss across all TokenGuard strategies
for each test case in tests/test_cases/.

Metrics computed per test case per strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  - original_token_count   : tokens before any optimization
  - optimized_token_count  : tokens after optimization
  - tokens_saved           : absolute reduction
  - savings_pct            : percentage reduction
  - rouge1_score           : ROUGE-1 F1 comparing compressed vs original text
                             (proxy for quality retention — higher is better)
  - latency_ms             : wall-clock time for the optimization step
  - hallucination_rate     : if source_doc provided (0.0 if not applicable)

Strategies benchmarked
~~~~~~~~~~~~~~~~~~~~~~
  1. Baseline             — no optimization (pass-through)
  2. Filler Removal       — Strategy 1 of PromptCompressor only
  3. Full Compression     — all 3 compressor strategies
  4. Context Pruning      — ContextPruner (keep_ratio=0.70)
  5. Compress + Prune     — compressor then pruner
  6. Summarization        — ConversationSummarizer (only on test_2)

Output
~~~~~~
  - Prints a formatted table to stdout
  - Logs all metrics to MLflow (run per test case)
  - Saves results/benchmark_results.csv

Run:
    python experiments/benchmark.py
    python experiments/benchmark.py --test test_1_rag_document.txt
    python experiments/benchmark.py --no-mlflow
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path fix
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_ROOT / ".env")

# ---------------------------------------------------------------------------
# Imports — project modules
# ---------------------------------------------------------------------------
from core.compressor import PromptCompressor
from core.pruner import ContextPruner
from core.summarizer import ConversationSummarizer

# ---------------------------------------------------------------------------
# Lazy ROUGE import
# ---------------------------------------------------------------------------
try:
    from rouge_score import rouge_scorer as _rouge_module
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False
    print("[WARN] rouge-score not installed — ROUGE metrics will be 0.0")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_CASES_DIR = _ROOT / "tests" / "test_cases"
RESULTS_DIR    = _ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_CSV     = RESULTS_DIR / "benchmark_results.csv"

_KEEP_RATIO    = 0.70
_TOKEN_BUDGET  = 4000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Word-count × 1.3 token estimate (no tiktoken dependency here)."""
    return int(len(text.split()) * 1.3)


def _rouge1(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-1 F1 between reference and hypothesis.

    Returns 0.0 if rouge-score is not installed or inputs are empty.
    """
    if not _ROUGE_AVAILABLE or not reference.strip() or not hypothesis.strip():
        return 0.0
    scorer = _rouge_module.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return round(scores["rouge1"].fmeasure, 4)


def _parse_test_file(path: Path) -> tuple[str, str]:
    """Parse a test case file.

    Expected format (first line optional):
        Query: <query text>
        <blank line>
        <document text>

    Returns
    -------
    tuple[str, str]
        (query, document_text)
    """
    content = path.read_text(encoding="utf-8").strip()
    lines   = content.splitlines()

    query   = "Summarize the key points."
    doc_lines: list[str] = []

    for i, line in enumerate(lines):
        if line.lower().startswith("query:"):
            query = line.split(":", 1)[1].strip()
        else:
            doc_lines = lines[i:]
            break

    document = "\n".join(doc_lines).strip()
    return query, document


def _parse_conversation_file(path: Path) -> tuple[str, list[dict]]:
    """Parse a conversation test file into (query, history_list).

    Lines starting with 'User:' and 'Assistant:' become history turns.
    The 'Query:' header becomes the final query.
    """
    query, content = _parse_test_file(path)
    history: list[dict] = []

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("User:"):
            history.append({"role": "user", "content": line[5:].strip()})
        elif line.startswith("Assistant:"):
            history.append({"role": "assistant", "content": line[10:].strip()})

    return query, history


def _print_table(rows: list[dict], title: str) -> None:
    """Print a formatted markdown-style table."""
    if not rows:
        return

    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    sep = "+-" + "-+-".join("-" * widths[c] for c in cols) + "-+"
    header = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"

    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols) + " |"
        print(line)
    print(sep)


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def run_baseline(query: str, document: str) -> dict:
    t0 = time.perf_counter()
    orig_tok = _estimate_tokens(query) + _estimate_tokens(document)
    latency  = (time.perf_counter() - t0) * 1000
    return {
        "strategy":              "1. Baseline",
        "original_tokens":       orig_tok,
        "optimized_tokens":      orig_tok,
        "tokens_saved":          0,
        "savings_pct":           "0.0%",
        "rouge1_score":          1.0,
        "latency_ms":            f"{latency:.1f}",
        "hallucination_rate":    "N/A",
    }


def run_filler_only(query: str, document: str) -> dict:
    compressor = PromptCompressor()
    orig_tok = _estimate_tokens(query) + _estimate_tokens(document)

    t0 = time.perf_counter()
    # Apply only strategy 1 (filler removal)
    doc_lines = document.split(". ")
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc_obj = nlp(document)
    sents = [s.text.strip() for s in doc_obj.sents if s.text.strip()]
    kept_sents, removed = compressor._remove_filler(sents)
    compressed = " ".join(kept_sents)
    latency = (time.perf_counter() - t0) * 1000

    opt_tok = _estimate_tokens(query) + _estimate_tokens(compressed)
    saved   = max(0, orig_tok - opt_tok)
    pct     = saved / orig_tok * 100 if orig_tok > 0 else 0.0
    r1      = _rouge1(document, compressed)

    return {
        "strategy":           "2. Filler Removal",
        "original_tokens":    orig_tok,
        "optimized_tokens":   opt_tok,
        "tokens_saved":       saved,
        "savings_pct":        f"{pct:.1f}%",
        "rouge1_score":       r1,
        "latency_ms":         f"{latency:.1f}",
        "hallucination_rate": "N/A",
    }


def run_full_compression(query: str, document: str) -> dict:
    compressor = PromptCompressor()
    orig_tok   = _estimate_tokens(query) + _estimate_tokens(document)

    t0     = time.perf_counter()
    result = compressor.compress(document)
    latency = (time.perf_counter() - t0) * 1000

    compressed = result["compressed_text"]
    opt_tok    = _estimate_tokens(query) + _estimate_tokens(compressed)
    saved      = max(0, orig_tok - opt_tok)
    pct        = saved / orig_tok * 100 if orig_tok > 0 else 0.0
    r1         = _rouge1(document, compressed)

    return {
        "strategy":           "3. Full Compression",
        "original_tokens":    orig_tok,
        "optimized_tokens":   opt_tok,
        "tokens_saved":       saved,
        "savings_pct":        f"{pct:.1f}%",
        "rouge1_score":       r1,
        "latency_ms":         f"{latency:.1f}",
        "hallucination_rate": "N/A",
    }


def run_pruning_only(query: str, document: str) -> dict:
    pruner   = ContextPruner(keep_ratio=_KEEP_RATIO)
    orig_tok = _estimate_tokens(query) + _estimate_tokens(document)

    t0     = time.perf_counter()
    result = pruner.prune(query, document)
    latency = (time.perf_counter() - t0) * 1000

    pruned  = result["pruned_context"]
    opt_tok = _estimate_tokens(query) + _estimate_tokens(pruned)
    saved   = max(0, orig_tok - opt_tok)
    pct     = saved / orig_tok * 100 if orig_tok > 0 else 0.0
    r1      = _rouge1(document, pruned)

    return {
        "strategy":           "4. Context Pruning",
        "original_tokens":    orig_tok,
        "optimized_tokens":   opt_tok,
        "tokens_saved":       saved,
        "savings_pct":        f"{pct:.1f}%",
        "rouge1_score":       r1,
        "latency_ms":         f"{latency:.1f}",
        "hallucination_rate": "N/A",
    }


def run_compress_plus_prune(query: str, document: str) -> dict:
    compressor = PromptCompressor()
    pruner     = ContextPruner(keep_ratio=_KEEP_RATIO)
    orig_tok   = _estimate_tokens(query) + _estimate_tokens(document)

    t0 = time.perf_counter()
    comp_result  = compressor.compress(document)
    prune_result = pruner.prune(query, comp_result["compressed_text"])
    latency      = (time.perf_counter() - t0) * 1000

    optimized = prune_result["pruned_context"]
    opt_tok   = _estimate_tokens(query) + _estimate_tokens(optimized)
    saved     = max(0, orig_tok - opt_tok)
    pct       = saved / orig_tok * 100 if orig_tok > 0 else 0.0
    r1        = _rouge1(document, optimized)

    return {
        "strategy":           "5. Compress + Prune",
        "original_tokens":    orig_tok,
        "optimized_tokens":   opt_tok,
        "tokens_saved":       saved,
        "savings_pct":        f"{pct:.1f}%",
        "rouge1_score":       r1,
        "latency_ms":         f"{latency:.1f}",
        "hallucination_rate": "N/A",
    }


def run_summarization(query: str, history: list[dict]) -> dict:
    summarizer = ConversationSummarizer(token_budget=_TOKEN_BUDGET)
    orig_text  = " ".join(t.get("content", "") for t in history)
    orig_tok   = _estimate_tokens(orig_text)

    t0           = time.perf_counter()
    compressed_h = summarizer.summarize(history)
    latency      = (time.perf_counter() - t0) * 1000

    comp_text = " ".join(t.get("content", "") for t in compressed_h)
    opt_tok   = _estimate_tokens(comp_text)
    saved     = max(0, orig_tok - opt_tok)
    pct       = saved / orig_tok * 100 if orig_tok > 0 else 0.0
    r1        = _rouge1(orig_text, comp_text)

    return {
        "strategy":           "6. Summarization",
        "original_tokens":    orig_tok,
        "optimized_tokens":   opt_tok,
        "tokens_saved":       saved,
        "savings_pct":        f"{pct:.1f}%",
        "rouge1_score":       r1,
        "latency_ms":         f"{latency:.1f}",
        "hallucination_rate": "N/A",
    }


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def _log_to_mlflow(test_name: str, rows: list[dict], use_mlflow: bool) -> None:
    if not use_mlflow:
        return
    try:
        import mlflow
        mlflow.set_experiment("TokenGuard Benchmark")
        with mlflow.start_run(run_name=test_name):
            for row in rows:
                strategy = row["strategy"].replace(" ", "_").replace(".", "")
                mlflow.log_metric(f"{strategy}_tokens_saved",     int(row["tokens_saved"]))
                mlflow.log_metric(f"{strategy}_rouge1",           float(row["rouge1_score"]))
                mlflow.log_metric(f"{strategy}_latency_ms",       float(row["latency_ms"]))
                mlflow.log_metric(f"{strategy}_optimized_tokens", int(row["optimized_tokens"]))
            mlflow.log_param("test_case",   test_name)
            mlflow.log_param("keep_ratio",  _KEEP_RATIO)
            mlflow.log_param("token_budget", _TOKEN_BUDGET)
        print(f"[MLflow] Run logged for '{test_name}'")
    except Exception as exc:
        print(f"[MLflow] Warning — could not log: {exc}")


# ---------------------------------------------------------------------------
# Per-test runners
# ---------------------------------------------------------------------------

def benchmark_test1(use_mlflow: bool) -> list[dict]:
    """RAG document — all strategies applicable."""
    path = TEST_CASES_DIR / "test_1_rag_document.txt"
    if not path.exists():
        print(f"[SKIP] {path.name} not found.")
        return []

    query, document = _parse_test_file(path)
    print(f"\n[TEST 1] {path.name}")
    print(f"  Query    : {query}")
    print(f"  Doc size : {_estimate_tokens(document)} tokens\n")

    rows = [
        run_baseline(query, document),
        run_filler_only(query, document),
        run_full_compression(query, document),
        run_pruning_only(query, document),
        run_compress_plus_prune(query, document),
    ]
    _print_table(rows, "TEST 1 — RAG Document (WWI)")
    _log_to_mlflow("test_1_rag_document", rows, use_mlflow)
    return rows


def benchmark_test2(use_mlflow: bool) -> list[dict]:
    """Long conversation — focus on summarization strategy."""
    path = TEST_CASES_DIR / "test_2_long_conversation.txt"
    if not path.exists():
        print(f"[SKIP] {path.name} not found.")
        return []

    query, history = _parse_conversation_file(path)
    # Reconstruct document from conversation for non-summarization strategies
    document = " ".join(t.get("content", "") for t in history)
    print(f"\n[TEST 2] {path.name}")
    print(f"  Query   : {query}")
    print(f"  Turns   : {len(history)}")
    print(f"  Doc size: {_estimate_tokens(document)} tokens\n")

    rows = [
        run_baseline(query, document),
        run_full_compression(query, document),
        run_summarization(query, history),
    ]
    _print_table(rows, "TEST 2 — Long Conversation (Python Debug)")
    _log_to_mlflow("test_2_long_conversation", rows, use_mlflow)
    return rows


def benchmark_test3(use_mlflow: bool) -> list[dict]:
    """Repeated context — compression should shine here."""
    path = TEST_CASES_DIR / "test_3_repeated_context.txt"
    if not path.exists():
        print(f"[SKIP] {path.name} not found.")
        return []

    query, document = _parse_test_file(path)
    print(f"\n[TEST 3] {path.name}")
    print(f"  Query    : {query}")
    print(f"  Doc size : {_estimate_tokens(document)} tokens\n")

    rows = [
        run_baseline(query, document),
        run_filler_only(query, document),
        run_full_compression(query, document),
        run_compress_plus_prune(query, document),
    ]
    _print_table(rows, "TEST 3 — Repeated Context (Photosynthesis)")
    _log_to_mlflow("test_3_repeated_context", rows, use_mlflow)
    return rows


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _export_csv(all_rows: list[dict], test_labels: list[str]) -> None:
    if not all_rows:
        return

    labelled: list[dict] = []
    for label, row in zip(test_labels, all_rows):
        labelled.append({"test_case": label, **row})

    fieldnames = list(labelled[0].keys())
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labelled)

    print(f"\n[CSV] Results saved to: {OUTPUT_CSV}")


# ---------------------------------------------------------------------------
# Summary table across all tests
# ---------------------------------------------------------------------------

def _print_summary(all_results: dict[str, list[dict]]) -> None:
    print(f"\n{'═'*70}")
    print("  OVERALL SUMMARY — Best Strategy Per Test Case")
    print(f"{'═'*70}")

    summary_rows = []
    for test_name, rows in all_results.items():
        if not rows:
            continue
        # Best = max tokens_saved
        best = max(rows, key=lambda r: int(r.get("tokens_saved", 0)))
        summary_rows.append({
            "Test Case":      test_name,
            "Best Strategy":  best["strategy"],
            "Tokens Saved":   best["tokens_saved"],
            "Savings %":      best["savings_pct"],
            "ROUGE-1":        best["rouge1_score"],
            "Latency (ms)":   best["latency_ms"],
        })

    _print_table(summary_rows, "Summary")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TokenGuard Benchmark")
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Run a specific test file (e.g. test_1_rag_document.txt). "
             "Omit to run all tests.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        default=False,
        help="Skip MLflow logging.",
    )
    args = parser.parse_args()

    use_mlflow = not args.no_mlflow

    print("╔══════════════════════════════════════════╗")
    print("║   TokenGuard Benchmark Suite  v1.0.0    ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  MLflow tracking : {'enabled' if use_mlflow else 'disabled'}")
    print(f"  Test cases dir  : {TEST_CASES_DIR}")
    print(f"  Output CSV      : {OUTPUT_CSV}")

    all_results: dict[str, list[dict]] = {}
    all_flat_rows: list[dict] = []
    all_labels: list[str] = []

    if args.test:
        # Single test
        name = args.test.replace(".txt", "")
        if "test_1" in name:
            rows = benchmark_test1(use_mlflow)
            all_results["test_1_rag_document"] = rows
            all_flat_rows.extend(rows)
            all_labels.extend(["test_1"] * len(rows))
        elif "test_2" in name:
            rows = benchmark_test2(use_mlflow)
            all_results["test_2_long_conversation"] = rows
            all_flat_rows.extend(rows)
            all_labels.extend(["test_2"] * len(rows))
        elif "test_3" in name:
            rows = benchmark_test3(use_mlflow)
            all_results["test_3_repeated_context"] = rows
            all_flat_rows.extend(rows)
            all_labels.extend(["test_3"] * len(rows))
        else:
            print(f"[ERROR] Unknown test: {args.test}")
            sys.exit(1)
    else:
        # All tests
        rows1 = benchmark_test1(use_mlflow)
        rows2 = benchmark_test2(use_mlflow)
        rows3 = benchmark_test3(use_mlflow)

        all_results["test_1_rag_document"]    = rows1
        all_results["test_2_long_conversation"] = rows2
        all_results["test_3_repeated_context"] = rows3

        for rows, label in [(rows1, "test_1"), (rows2, "test_2"), (rows3, "test_3")]:
            all_flat_rows.extend(rows)
            all_labels.extend([label] * len(rows))

    _print_summary(all_results)
    _export_csv(all_flat_rows, all_labels)

    print("\n✅ Benchmark complete.")
    if use_mlflow:
        print("   View MLflow UI: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
