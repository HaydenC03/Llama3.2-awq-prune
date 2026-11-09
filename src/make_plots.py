"""All figures. Reads results/metrics.json, writes results/plots/*.png.

Figures
-------
01_accuracy_vs_compression.png : accuracy vs checkpoint compression ratio
02_accuracy_by_task.png        : per-task accuracy (the 3 benchmark tasks)
03_mixed_precision.png         : per-block bit-width + accuracy vs avg bits
04_complexity.png              : throughput / VRAM / FLOPs / prefill scaling
05_efficiency_frontier.png     : accuracy-vs-speed and accuracy-per-GB
06_perplexity.png              : WikiText-2 perplexity vs compression
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import PLOTS, RESULTS, load_json

log = logging.getLogger("plots")

ORDER = ["baseline", "pruned", "awq", "pruned_awq"]
COLORS = {
    "baseline": "#1f3b73",
    "pruned": "#e07b39",
    "awq": "#2a9d8f",
    "pruned_awq": "#c0392b",
}
MARKERS = {"baseline": "o", "pruned": "s", "awq": "^", "pruned_awq": "P"}
TASK_LABELS = {
    "arc_easy": "ARC-Easy\n(science reasoning)",
    "hellaswag": "HellaSwag\n(commonsense)",
    "piqa": "PIQA\n(physical commonsense)",
    "boolq": "BoolQ\n(yes/no reading)",
    "gsm8k": "GSM8K\n(math word problems)",
    "ms_marco": "MS MARCO\n(QA token F1)",
}
SHORT = {
    "baseline": "FP16",
    "pruned": "Pruned",
    "awq": "AWQ W4",
    "pruned_awq": "Pruned+AWQ",
}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 170, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.autolayout": False,
})


# --------------------------------------------------------------------------- #
def _variants(m: Dict) -> List[str]:
    return [v for v in ORDER if v in m.get("variants", {})]


def _label(m: Dict, v: str) -> str:
    return m["variants"][v].get("display_name", v)


def _watermark(fig, m: Dict) -> None:
    if m.get("meta", {}).get("synthetic"):
        fig.text(0.5, 0.5, "PREVIEW DATA — NOT MEASURED", ha="center", va="center",
                 fontsize=34, color="#b00020", alpha=0.10, rotation=22, fontweight="bold",
                 zorder=0)


def _save(fig, name: str) -> Path:
    out = PLOTS / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)
    return out


# --------------------------------------------------------------------------- #
# 01 accuracy vs compression ratio
# --------------------------------------------------------------------------- #
def plot_accuracy_vs_compression(m: Dict) -> Path:
    vs = _variants(m)
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    fig.subplots_adjust(bottom=0.30)
    base = m["variants"].get("baseline", {}).get("accuracy_avg")

    xs, ys = [], []
    for v in vs:
        d = m["variants"][v]
        x = d.get("compression_ratio", 1.0)
        y = d.get("accuracy_avg", 0) * 100
        xs.append(x); ys.append(y)

        label = f"{_label(m, v)}  ·  {x:.2f}×"
        if base and v != "baseline":
            label += f"  ·  {(d.get('accuracy_avg', 0) - base) * 100:+.1f} pt"

        ax.scatter(x, y, s=190, marker=MARKERS.get(v, "o"), color=COLORS.get(v, "#555"),
                   edgecolor="white", linewidth=1.4, zorder=3, label=label)
        # right-most points get their label below so it cannot clip or collide
        below = x > 0.85 * max(xs) and v != "baseline"
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, -20) if below else (0, 13), ha="center",
                    fontsize=9.5, fontweight="bold", color=COLORS.get(v, "#333"))

    if len(xs) > 1:
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "--", color="#999",
                linewidth=1.2, zorder=1, alpha=0.8)

    if base:
        ax.axhline(base * 100, color=COLORS["baseline"], lw=1, ls=":", alpha=0.55)
        ax.text(min(xs) * 0.97, base * 100 + 0.35, "baseline", va="bottom", ha="left",
                fontsize=8, color=COLORS["baseline"])

    ax.set_xscale("log")
    ticks = [t for t in (1, 1.5, 2, 2.5, 3, 3.5, 4) if min(xs) * 0.9 <= t <= max(xs) * 1.1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}×" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(min(xs) * 0.85, max(xs) * 1.18)
    ax.set_ylim(min(ys) - 3.5, max(ys) + 3.0)
    ax.set_xlabel("Compression ratio  (FP16 checkpoint size ÷ variant size)")
    ax.set_ylabel("Average accuracy over 3 tasks (%)")
    ax.set_title("Accuracy vs. compression — Llama-3.2-1B")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    _watermark(fig, m)
    return _save(fig, "01_accuracy_vs_compression.png")


# --------------------------------------------------------------------------- #
# 02 per-task accuracy
# --------------------------------------------------------------------------- #
def plot_accuracy_by_task(m: Dict) -> Path:
    vs = _variants(m)
    tasks = [t for t in TASK_LABELS if any(t in m["variants"][v].get("accuracy", {}) for v in vs)]
    if not tasks:
        raise SystemExit("no per-task accuracy in metrics.json")

    x = np.arange(len(tasks))
    w = 0.8 / max(len(vs), 1)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    for i, v in enumerate(vs):
        acc = m["variants"][v].get("accuracy", {})
        vals = [acc.get(t, np.nan) * 100 for t in tasks]
        bars = ax.bar(x + (i - (len(vs) - 1) / 2) * w, vals, w * 0.92,
                      color=COLORS.get(v, "#777"), label=_label(m, v),
                      edgecolor="white", linewidth=0.8)
        for b, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.6, f"{val:.1f}",
                        ha="center", fontsize=7.6)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(60, ax.get_ylim()[1] * 1.18))
    ax.set_title("Downstream task accuracy by variant")
    ax.legend(frameon=False, fontsize=8.5, ncol=min(len(vs), 3))
    _watermark(fig, m)
    return _save(fig, "02_accuracy_by_task.png")


# --------------------------------------------------------------------------- #
# 03 mixed precision
# --------------------------------------------------------------------------- #
def plot_mixed_precision(m: Dict) -> Path:
    vs = _variants(m)
    mp = next((v for v in ("awq",) if m["variants"].get(v, {}).get("blocks")), None)
    if mp is None:
        log.warning("no bit plan in metrics.json; skipping precision figure")
        return PLOTS / "03_mixed_precision.png"

    blocks = m["variants"][mp]["blocks"]
    names = list(blocks)
    bits = [blocks[b]["bits"] for b in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                                   gridspec_kw={"width_ratios": [2.1, 1]})

    # (a) per-block effective bit width
    colors = ["#8e6bbf" if blocks[b]["max_bits"] > 4 else "#2a9d8f" for b in names]
    ax1.bar(names, bits, color=colors, edgecolor="white")
    ax1.axhline(4, color="#2a9d8f", ls="--", lw=1, alpha=0.8)
    ax1.axhline(16, color="#333", ls=":", lw=1, alpha=0.6)
    ax1.text(len(names) - 0.5, 4.3, "uniform W4", ha="right", va="bottom", fontsize=8,
             color="#2a9d8f", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    ax1.text(len(names) - 0.5, 15.2, "FP16", ha="right", fontsize=8, color="#333")
    ax1.set_ylabel("Effective bits / weight")
    ax1.set_xlabel("Decoder block")
    ax1.set_ylim(0, 17)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=90, fontsize=8)
    ax1.set_title(f"Per-block bit-width — {_label(m, mp)}")

    # (b) accuracy vs precision budget. Only the quantisation ladder
    # (W4 -> FP16) gets a connecting line: pruned checkpoints are not points on
    # the precision axis, so joining them would imply a trend that isn't there.
    pts = []
    for v in vs:
        d = m["variants"][v]
        b = d.get("avg_bits")
        if b is None:
            b = 16.0
        if "accuracy_avg" in d:
            pts.append((b, d["accuracy_avg"], v))
    pts.sort()

    ladder = [(b, a, v) for b, a, v in pts if v in ("awq", "baseline")]
    if len(ladder) > 1:
        ax2.plot([p[0] for p in ladder], [p[1] * 100 for p in ladder],
                 "--", color="#aaa", lw=1, zorder=1)

    for b, a, v in pts:
        ax2.scatter(b, a * 100, s=150, marker=MARKERS.get(v, "o"),
                    color=COLORS.get(v, "#555"), edgecolor="white", zorder=3)
        right = b < 12
        ax2.annotate(_label(m, v), (b, a * 100), textcoords="offset points",
                     xytext=(7 if right else -7, 7), ha="left" if right else "right",
                     fontsize=7.5)
    ax2.set_xlabel("Average bits per weight (transformer body)")
    ax2.set_ylabel("Average accuracy (%)")
    ax2.set_title("Accuracy vs. precision budget")
    ax2.set_xlim(3.0, 18.5)
    ax2.margins(y=0.14)

    fig.tight_layout()
    _watermark(fig, m)
    return _save(fig, "03_mixed_precision.png")


# --------------------------------------------------------------------------- #
# 04 complexity
# --------------------------------------------------------------------------- #
def plot_complexity(m: Dict) -> Path:
    vs = _variants(m)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0))
    ax_tps, ax_vram, ax_flop, ax_pre = axes.ravel()

    def bar(ax, key, title, ylabel, fmt="{:.1f}"):
        vals, labels, cols = [], [], []
        for v in vs:
            val = m["variants"][v].get("complexity", {}).get(key)
            if val is None:
                continue
            vals.append(val); labels.append(SHORT.get(v, v)); cols.append(COLORS.get(v, "#777"))
        if not vals:
            ax.set_visible(False); return
        b = ax.bar(range(len(vals)), vals, color=cols, edgecolor="white")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=0, fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.margins(y=0.14)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v, fmt.format(v), ha="center",
                    va="bottom", fontsize=8.5, fontweight="bold")

    bar(ax_tps, "tokens_per_s", "Generation throughput (batch 1)", "tokens / s")
    bar(ax_vram, "peak_vram_gb", "Peak GPU memory", "GiB")
    bar(ax_flop, "gflops_per_token", "Analytic compute per token", "GFLOP / token", "{:.2f}")

    # prefill scaling
    any_line = False
    for v in vs:
        pre = m["variants"][v].get("complexity", {}).get("prefill_ms") or {}
        if not pre:
            continue
        lengths = sorted(pre, key=int)
        ax_pre.plot([int(x) for x in lengths], [pre[x] for x in lengths],
                    marker=MARKERS.get(v, "o"), color=COLORS.get(v, "#777"),
                    label=_label(m, v), lw=1.6)
        any_line = True
    if any_line:
        ax_pre.set_xlabel("Prompt length (tokens)")
        ax_pre.set_ylabel("Prefill latency (ms)")
        ax_pre.set_title("Prefill latency scaling  (attention is O(n²))")
        ax_pre.legend(frameon=False, fontsize=7.5, loc="upper left")
    else:
        ax_pre.set_visible(False)

    fig.suptitle("Complexity / cost analysis", fontweight="bold", y=0.995)
    fig.tight_layout()
    _watermark(fig, m)
    return _save(fig, "04_complexity.png")


# --------------------------------------------------------------------------- #
# 05 efficiency frontier
# --------------------------------------------------------------------------- #
def plot_frontier(m: Dict) -> Path:
    vs = _variants(m)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5))

    for v in vs:
        d = m["variants"][v]
        c = d.get("complexity", {})
        tps, acc = c.get("tokens_per_s"), d.get("accuracy_avg")
        if tps and acc is not None:
            ax1.scatter(tps, acc * 100, s=170, marker=MARKERS.get(v, "o"),
                        color=COLORS.get(v, "#555"), edgecolor="white", zorder=3,
                        label=_label(m, v))
            ax1.annotate(f"{acc*100:.1f}%", (tps, acc * 100), textcoords="offset points",
                         xytext=(9, -3), fontsize=8)
        size, acc = d.get("size_gb"), d.get("accuracy_avg")
        if size and acc is not None:
            ax2.scatter(size, acc * 100, s=170, marker=MARKERS.get(v, "o"),
                        color=COLORS.get(v, "#555"), edgecolor="white", zorder=3,
                        label=_label(m, v))
            ax2.annotate(f"{acc*100:.1f}%", (size, acc * 100), textcoords="offset points",
                         xytext=(9, -3), fontsize=8)

    ax1.set_xlabel("Throughput (tokens / s)")
    ax1.set_ylabel("Average accuracy (%)")
    ax1.set_title("Accuracy vs. speed")
    ax1.legend(frameon=False, fontsize=8)

    ax2.set_xlabel("Checkpoint size (GiB)")
    ax2.set_ylabel("Average accuracy (%)")
    ax2.set_title("Accuracy vs. disk footprint")
    fig.tight_layout()
    _watermark(fig, m)
    return _save(fig, "05_efficiency_frontier.png")


# --------------------------------------------------------------------------- #
# 06 perplexity
# --------------------------------------------------------------------------- #
def plot_perplexity(m: Dict) -> Path:
    vs = [v for v in _variants(m) if m["variants"][v].get("ppl_wikitext2")]
    if not vs:
        return PLOTS / "06_perplexity.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, v in enumerate(vs):
        d = m["variants"][v]
        ax.scatter(d.get("compression_ratio", 1.0), d["ppl_wikitext2"], s=170,
                   marker=MARKERS.get(v, "o"), color=COLORS.get(v, "#555"),
                   edgecolor="white", zorder=3, label=_label(m, v))
        ax.annotate(f"{d['ppl_wikitext2']:.2f}",
                    (d.get("compression_ratio", 1.0), d["ppl_wikitext2"]),
                    textcoords="offset points", xytext=(0, 11 if i % 2 == 0 else -17),
                    ha="center", fontsize=8)
    ax.set_xscale("log")
    ticks = [t for t in (1, 1.5, 2, 2.5, 3, 3.5, 4) if 0.9 <= t <= 4.4]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}×" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("WikiText-2 perplexity (lower is better)")
    ax.set_title("Intrinsic degradation: perplexity vs. compression")
    ax.margins(y=0.15, x=0.10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _watermark(fig, m)
    return _save(fig, "06_perplexity.png")


# --------------------------------------------------------------------------- #
ALL_PLOTS = {
    "01": plot_accuracy_vs_compression,
    "02": plot_accuracy_by_task,
    "03": plot_mixed_precision,
    "04": plot_complexity,
    "05": plot_frontier,
    "06": plot_perplexity,
}


def main(metrics_path: str | Path | None = None, only: List[str] | None = None) -> None:
    metrics_path = Path(metrics_path) if metrics_path else RESULTS / "metrics.json"
    m = load_json(metrics_path)
    if not m or "variants" not in m:
        raise SystemExit(f"no metrics at {metrics_path}; run the pipeline first")
    for key, fn in ALL_PLOTS.items():
        if only and key not in only:
            continue
        try:
            fn(m)
        except Exception as exc:  # keep going, one bad panel shouldn't kill the rest
            log.error("plot %s failed: %s", key, exc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="render figures from results/metrics.json")
    ap.add_argument("--metrics", default=str(RESULTS / "metrics.json"))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args.metrics, args.only)
