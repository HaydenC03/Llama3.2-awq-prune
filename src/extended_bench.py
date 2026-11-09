"""Extended benchmarks (GSM8K / MS MARCO / BoolQ) + CSV comparison.

The original pipeline reports ARC-Easy, HellaSwag and PIQA. This module adds
three extra tasks and writes a CSV that puts every compressed variant next to
the FP16 baseline, together with the checkpoint / VRAM reduction.

Metrics
-------
* gsm8k    : numeric exact match on the final answer (generation)
* ms_marco : token F1 (and exact match) on the selected passage (generation)
* boolq    : multiple-choice accuracy for " no" / " yes" (teacher forcing)

The CSV is deliberately long-form (one row per variant x task) plus a compact
per-variant summary:

    results/extended/<variant>.json
    results/extended_scores.csv
    results/extended_summary.csv
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterable, List

from .common import RESULTS, VARIANT_LABELS, dir_size_bytes, gb, load_json, save_json

log = logging.getLogger("extended")

EXTENDED_TASKS = ["gsm8k", "ms_marco", "boolq"]
TASK_LABELS = {
    "gsm8k": "GSM8K",
    "ms_marco": "MS MARCO",
    "boolq": "BoolQ",
}
TASK_METRIC = {
    "gsm8k": "acc",
    "ms_marco": "f1",
    "boolq": "acc",
}
ORDER = ["baseline", "pruned", "awq", "pruned_awq"]

EXTENDED_DIR = RESULTS / "extended"
SCORES_CSV = RESULTS / "extended_scores.csv"
SUMMARY_CSV = RESULTS / "extended_summary.csv"


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def _memory_gb() -> tuple[float, float]:
    """(currently allocated, peak allocated) on the active CUDA device."""
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (torch.cuda.memory_allocated() / 1024 ** 3,
            torch.cuda.max_memory_allocated() / 1024 ** 3)


def evaluate_variant(variant: str, path: str | Path, *, tasks: Iterable[str],
                     limit: int = 100, max_len: int = 1024,
                     max_new_tokens: int = 256, device: str = "cuda",
                     token: str | None = None) -> Dict:
    """Load one variant, run the extended tasks, and record memory use."""
    import torch

    from .evaluate import load_for_eval
    from .tasks import evaluate_task

    tasks = list(tasks)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, tok = load_for_eval(variant, path, device=device, token=token)
    load_vram, _ = _memory_gb()

    results: Dict[str, Dict] = {}
    for task in tasks:
        results[task] = evaluate_task(
            model, tok, task, limit=limit, max_len=max_len,
            max_new_tokens=max_new_tokens, device=device,
        )

    _, peak_vram = _memory_gb()
    size = dir_size_bytes(path)
    record = {
        "variant": variant,
        "display_name": VARIANT_LABELS.get(variant, variant),
        "path": str(path),
        "tasks": tasks,
        "limit": limit,
        "max_len": max_len,
        "max_new_tokens": max_new_tokens,
        "weight_gb": round(gb(size), 3),
        "load_vram_gb": round(load_vram, 3),
        "peak_vram_gb": round(peak_vram, 3),
        "results": results,
    }
    log.info("[%s] extended: %s", variant, ", ".join(
        f"{t}={results[t].get('score')}" for t in tasks))
    return record


def run_variant(variant: str, path: str | Path, *, tasks: Iterable[str],
                limit: int = 100, max_len: int = 1024,
                max_new_tokens: int = 256, out_dir: str | Path = EXTENDED_DIR,
                force: bool = False, device: str = "cuda",
                token: str | None = None) -> Dict:
    """Evaluate + cache one variant, skipping an identical previous run."""
    tasks = list(tasks)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{variant}.json"

    if out.exists() and not force:
        previous = load_json(out)
        if (previous and previous.get("tasks") == tasks
                and previous.get("limit") == limit
                and previous.get("max_len") == max_len
                and previous.get("max_new_tokens") == max_new_tokens):
            log.info("[%s] extended results already cached (%s)", variant, out.name)
            return previous

    record = evaluate_variant(
        variant, path, tasks=tasks, limit=limit, max_len=max_len,
        max_new_tokens=max_new_tokens, device=device, token=token,
    )
    save_json(record, out)
    return record


def load_records(out_dir: str | Path = EXTENDED_DIR) -> Dict[str, Dict]:
    out_dir = Path(out_dir)
    records: Dict[str, Dict] = {}
    for path in sorted(out_dir.glob("*.json")):
        rec = load_json(path)
        if rec and rec.get("variant"):
            records[rec["variant"]] = rec
    return records


# --------------------------------------------------------------------------- #
# CSV rows (pure functions, easy to test without a GPU)
# --------------------------------------------------------------------------- #
def _ordered_variants(records: Dict[str, Dict]) -> List[str]:
    known = [v for v in ORDER if v in records]
    extra = sorted(v for v in records if v not in ORDER)
    return known + extra


def _score(record: Dict, task: str) -> float | None:
    value = record.get("results", {}).get(task, {}).get("score")
    return None if value is None else float(value)


def _reduction_pct(baseline: float | None, value: float | None) -> float | None:
    if baseline is None or value is None or baseline <= 0:
        return None
    return (baseline - value) / baseline * 100.0


def _retention_pct(baseline: float | None, value: float | None) -> float | None:
    if baseline is None or value is None or baseline <= 0:
        return None
    return value / baseline * 100.0


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def build_rows(records: Dict[str, Dict], tasks: Iterable[str]) -> List[Dict]:
    """One row per variant x task, with baseline delta and memory reduction."""
    tasks = list(tasks)
    baseline = records.get("baseline", {})
    base_weight = baseline.get("weight_gb")
    base_peak = baseline.get("peak_vram_gb")
    base_load = baseline.get("load_vram_gb")

    rows: List[Dict] = []
    for variant in _ordered_variants(records):
        rec = records[variant]
        weight = rec.get("weight_gb")
        peak = rec.get("peak_vram_gb")
        load = rec.get("load_vram_gb")
        for task in tasks:
            base_score = _score(baseline, task)
            score = _score(rec, task)
            detail = rec.get("results", {}).get(task, {})
            delta = None if base_score is None or score is None else score - base_score
            rows.append({
                "variant": variant,
                "display_name": rec.get("display_name", VARIANT_LABELS.get(variant, variant)),
                "task": task,
                "task_label": TASK_LABELS.get(task, task),
                "metric": detail.get("metric", TASK_METRIC.get(task, "score")),
                "n": detail.get("n"),
                "baseline_score": _round(base_score),
                "variant_score": _round(score),
                "score_delta": _round(delta),
                "score_retention_pct": _round(_retention_pct(base_score, score), 2),
                "score_reduction_pct": _round(_reduction_pct(base_score, score), 2),
                "baseline_weight_gb": _round(base_weight, 3),
                "variant_weight_gb": _round(weight, 3),
                "weight_reduction_pct": _round(_reduction_pct(base_weight, weight), 2),
                "baseline_peak_vram_gb": _round(base_peak, 3),
                "variant_peak_vram_gb": _round(peak, 3),
                "vram_reduction_pct": _round(_reduction_pct(base_peak, peak), 2),
                "baseline_load_vram_gb": _round(base_load, 3),
                "variant_load_vram_gb": _round(load, 3),
                "load_vram_reduction_pct": _round(_reduction_pct(base_load, load), 2),
            })
    return rows


def build_summary_rows(records: Dict[str, Dict], tasks: Iterable[str]) -> List[Dict]:
    """One row per variant: average primary score + memory reduction."""
    tasks = list(tasks)
    baseline = records.get("baseline", {})
    base_scores = [_score(baseline, t) for t in tasks]
    base_valid = [s for s in base_scores if s is not None]
    base_avg = sum(base_valid) / len(base_valid) if base_valid else None

    rows: List[Dict] = []
    for variant in _ordered_variants(records):
        rec = records[variant]
        scores = [s for s in (_score(rec, t) for t in tasks) if s is not None]
        avg = sum(scores) / len(scores) if scores else None
        weight = rec.get("weight_gb")
        peak = rec.get("peak_vram_gb")
        load = rec.get("load_vram_gb")
        rows.append({
            "variant": variant,
            "display_name": rec.get("display_name", VARIANT_LABELS.get(variant, variant)),
            "score_metrics": ";".join(f"{t}:{TASK_METRIC.get(t, 'score')}" for t in tasks),
            "avg_score": _round(avg),
            "baseline_avg_score": _round(base_avg),
            "avg_score_delta": _round(None if avg is None or base_avg is None else avg - base_avg),
            "avg_score_retention_pct": _round(_retention_pct(base_avg, avg), 2),
            "weight_gb": _round(weight, 3),
            "weight_reduction_pct": _round(_reduction_pct(baseline.get("weight_gb"), weight), 2),
            "peak_vram_gb": _round(peak, 3),
            "vram_reduction_pct": _round(_reduction_pct(baseline.get("peak_vram_gb"), peak), 2),
            "load_vram_gb": _round(load, 3),
            "load_vram_reduction_pct": _round(_reduction_pct(baseline.get("load_vram_gb"), load), 2),
        })
    return rows


def _write_csv(path: Path, rows: List[Dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return path
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s", path)
    return path


def write_csvs(tasks: Iterable[str] = EXTENDED_TASKS,
               out_dir: str | Path = EXTENDED_DIR,
               scores_csv: str | Path = SCORES_CSV,
               summary_csv: str | Path = SUMMARY_CSV,
               records: Dict[str, Dict] | None = None) -> tuple[Path, Path]:
    """Read the cached extended JSONs and write the two comparison CSVs."""
    records = records if records is not None else load_records(out_dir)
    if not records:
        raise SystemExit(f"no extended results in {out_dir}; run `--stage extended` first")

    rows = build_rows(records, tasks)
    summary = build_summary_rows(records, tasks)
    return _write_csv(Path(scores_csv), rows), _write_csv(Path(summary_csv), summary)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="rebuild the extended CSV comparison")
    ap.add_argument("--tasks", nargs="*", default=EXTENDED_TASKS)
    ap.add_argument("--out-dir", default=str(EXTENDED_DIR))
    ap.add_argument("--scores-csv", default=str(SCORES_CSV))
    ap.add_argument("--summary-csv", default=str(SUMMARY_CSV))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    write_csvs(args.tasks, args.out_dir, args.scores_csv, args.summary_csv)
