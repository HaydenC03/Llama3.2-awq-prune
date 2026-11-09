"""Turn results/metrics.json into a paste-ready Markdown report."""
from __future__ import annotations

import logging
from pathlib import Path

from .common import load_json

log = logging.getLogger("report")

ORDER = ["baseline", "pruned", "awq", "pruned_awq"]
TASK_NICE = {
    "arc_easy": "ARC-Easy",
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "boolq": "BoolQ",
    "gsm8k": "GSM8K",
    "ms_marco": "MS MARCO",
}


def _fmt(v, nd=2, suffix=""):
    return "–" if v is None else f"{v:.{nd}f}{suffix}"


def build_markdown(m: dict) -> str:
    meta = m.get("meta", {})
    hw = meta.get("hardware", {})
    variants = [v for v in ORDER if v in m["variants"]]
    base = m["variants"].get("baseline", {})
    base_acc = base.get("accuracy_avg")

    lines = []
    lines.append("# Llama-3.2-1B — AWQ quantisation + structured pruning\n")
    if meta.get("synthetic"):
        lines.append("> **PREVIEW DATA** — numbers below are literature-typical "
                     "reference values, not measurements on this machine.\n")
    lines.append(f"- **Model:** `{meta.get('model_id', 'n/a')}`")
    lines.append(f"- **Generated:** {meta.get('created', 'n/a')}")
    lines.append(f"- **GPU:** {hw.get('gpu') or 'none'} "
                 f"({hw.get('vram_gb', 0)} GB VRAM, sm_{hw.get('sm', '?')})")
    lines.append(f"- **CPU / RAM:** {hw.get('cpu_count')} threads / {hw.get('ram_gb')} GB")
    lines.append(f"- **Torch:** {hw.get('torch')} (CUDA {hw.get('cuda')})\n")

    # ---------------------------------------------------------------- summary
    lines.append("## 1. Summary\n")
    lines.append("| Variant | Weight bits | Size (GiB) | Compression | "
                 "Avg acc | Δ vs FP16 | WikiText-2 PPL |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for v in variants:
        d = m["variants"][v]
        drop = "" if v == "baseline" or base_acc is None else \
               f"{(d.get('accuracy_avg', 0) - base_acc) * 100:+.1f} pt"
        lines.append(
            f"| {d.get('display_name', v)} | {_fmt(d.get('avg_bits') or 16.0, 2)} | "
            f"{_fmt(d.get('size_gb'), 2)} | {_fmt(d.get('compression_ratio'), 2)}× | "
            f"{_fmt((d.get('accuracy_avg') or 0) * 100, 1)}% | {drop or '—'} | "
            f"{_fmt(d.get('ppl_wikitext2'), 2)} |"
        )

    # ------------------------------------------------------------- per task
    tasks = [t for t in TASK_NICE if any(t in m["variants"][v].get("accuracy", {}) for v in variants)]
    if tasks:
        lines.append("\n## 2. Downstream tasks\n")
        lines.append("| Variant | " + " | ".join(TASK_NICE[t] for t in tasks) + " |")
        lines.append("|---|" + "---:|" * len(tasks))
        for v in variants:
            d = m["variants"][v]
            cells = [_fmt((d.get("accuracy", {}).get(t) or 0) * 100, 1) + "%" for t in tasks]
            lines.append(f"| {d.get('display_name', v)} | " + " | ".join(cells) + " |")

    # ------------------------------------------------------------ complexity
    if any("complexity" in m["variants"][v] for v in variants):
        lines.append("\n## 3. Complexity\n")
        lines.append("| Variant | tok/s | ms/token | TTFT (ms) | Peak VRAM (GiB) | "
                     "GFLOP/token | Prefill @1024 (ms) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for v in variants:
            c = m["variants"][v].get("complexity")
            if not c:
                continue
            pre = (c.get("prefill_ms") or {}).get("1024")
            lines.append(
                f"| {m['variants'][v].get('display_name', v)} | {_fmt(c.get('tokens_per_s'), 1)} | "
                f"{_fmt(c.get('ms_per_token'), 2)} | {_fmt(c.get('ttft_ms'), 0)} | "
                f"{_fmt(c.get('peak_vram_gb'), 2)} | {_fmt(c.get('gflops_per_token'), 2)} | "
                f"{_fmt(pre, 0)} |"
            )

    # --------------------------------------------------------------- figures
    lines.append("\n## 4. Figures\n")
    figs = [
        ("01_accuracy_vs_compression.png", "Accuracy vs. compression ratio"),
        ("02_accuracy_by_task.png", "Accuracy per benchmark task"),
        ("03_mixed_precision.png", "Mixed-precision bit allocation"),
        ("04_complexity.png", "Complexity: throughput, VRAM, FLOPs, prefill scaling"),
        ("05_efficiency_frontier.png", "Accuracy vs. speed / footprint"),
        ("06_perplexity.png", "Perplexity vs. compression"),
    ]
    for f, cap in figs:
        lines.append(f"![{cap}](plots/{f})\n")

    # ---------------------------------------------------------------- method
    lines.append("## 5. Method\n")
    lines.append("""
**AWQ** (Activation-aware Weight Quantisation) — weight-only 4-bit, group size
128, zero-point, GEMM layout. Scales are searched by minimising the layer-wise
output error on 128 calibration samples from WikiText-2 train (512 tokens each).
Activations stay FP16, so nothing is calibrated for speed of the attention path.

**Structured pruning** — every decoder MLP loses whole intermediate channels
(`gate_proj` / `up_proj` / `down_proj` are physically sliced, `intermediate_size`
is rewritten). Channels are ranked by `||W_gate[i]||₂ · ||W_up[i]||₂`.
Unstructured masks were deliberately avoided: they shrink neither the file nor
the FLOPs, so they would fake the compression ratio.

**Pruning is the weak point in this run, and the data says so.** At
`keep_ratio = 0.75` the accuracy average falls 13 pt and WikiText-2 perplexity
goes from 14.9 to 152, with visibly incoherent samples. Weight-norm importance
is activation-blind; Wanda/SparseGPT-style scoring, per-layer sensitivity
search, or the built-in LoRA healing (`--heal-steps 300`) are the fixes, and
this should be treated as a negative result about naive magnitude pruning
rather than about structured pruning in general.

**No mixed-precision or W8 variant exists, and that is a backend limit, not an
oversight.** AutoAWQ 0.2.7 raises `NotImplementedError: Only 4-bit are supported
for now` for `w_bit=8`, and `modules_to_not_convert` cannot express a per-block
rule (it matches module-relative names, while `down_proj` is the scaling anchor
for gate/up). Splicing FP16 modules back in after quantisation breaks AWQ's
in-block rescaling — that attempt measured chance-level accuracy and PPL 4196.
The precision axis therefore has only two real points here: W4 and FP16. True
per-layer allocation needs `llm-compressor` or vLLM compressed-tensors.

**Runtime caveat** — AutoAWQ 0.2.7 no longer bundles `awq_ext`; the GEMM path
falls back to Triton, which costs ~2× at batch 1 on this GPU. The throughput
column therefore understates what AWQ delivers with the CUDA kernels
(`autoawq[kernels]`) or under vLLM. For scale: llama.cpp/Ollama on the same
4-bit 1B reaches 133 tok/s on this machine, versus 21 tok/s for
`transformers.generate` here.

**Evaluation** — teacher-forced log-likelihood ranking, which is exactly how
`lm-evaluation-harness` scores multiple choice. Deterministic, no sampling.
`--limit 200` per task, same fixed subset for every variant (seed 42).
""")

    lines.append("## 6. Threats to validity\n")
    lines.append("""
- 200 samples per task gives roughly ±3.5 pt standard error at p≈0.5 — treat
  single-point differences as noise, not signal.
- AWQ's speed-up is *memory-bandwidth* only. GFLOP/token is unchanged, so at
  batch 1 on a small model the gain is smaller than the 4× size reduction.
- WikiText-2 train text was also used as AWQ calibration data, so its
  perplexity is mildly optimistic.
- Structured pruning is uniform across layers; per-layer sensitivity search
  would do better at the same budget.
""")
    return "\n".join(lines)


def write_report(metrics_path: str | Path, out_path: str | Path) -> Path:
    m = load_json(metrics_path)
    if not m:
        raise SystemExit(f"no metrics at {metrics_path}")
    md = build_markdown(m)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    log.info("wrote %s", out)
    return out
