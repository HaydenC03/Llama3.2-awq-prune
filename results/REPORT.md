# Llama-3.2-1B — AWQ quantisation + structured pruning

- **Model:** `unsloth/Llama-3.2-1B-Instruct`
- **Generated:** 2026-11-09T07:35:31
- **GPU:** NVIDIA GeForce RTX 3050 (8.0 GB VRAM, sm_8.6)
- **CPU / RAM:** 8 threads / 10.7 GB
- **Torch:** 2.5.1+cu124 (CUDA 12.4)

## 1. Summary

| Variant | Weight bits | Size (GiB) | Compression | Avg acc | Δ vs FP16 | WikiText-2 PPL |
|---|---:|---:|---:|---:|---:|---:|
| Original (FP16) | 16.00 | 2.30 | 1.00× | 63.5% | — | 14.91 |
| Pruned (structured) | 16.00 | 1.93 | 1.20× | 50.5% | -13.0 pt | 152.13 |
| AWQ W4A16 | 4.00 | 0.96 | 2.40× | 62.7% | -0.8 pt | 16.24 |
| Pruned + AWQ W4 | 4.00 | 0.86 | 2.67× | 52.8% | -10.7 pt | 195.22 |

## 2. Downstream tasks

| Variant | ARC-Easy | HellaSwag | PIQA |
|---|---:|---:|---:|
| Original (FP16) | 66.0% | 43.5% | 81.0% |
| Pruned (structured) | 49.0% | 34.0% | 68.5% |
| AWQ W4A16 | 66.0% | 42.0% | 80.0% |
| Pruned + AWQ W4 | 54.0% | 34.0% | 70.5% |

## 3. Complexity

| Variant | tok/s | ms/token | TTFT (ms) | Peak VRAM (GiB) | GFLOP/token | Prefill @1024 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Original (FP16) | 51.7 | 19.35 | 20 | 2.81 | 2.61 | 188 |
| Pruned (structured) | 57.8 | 17.29 | 18 | 2.43 | 2.20 | 155 |
| AWQ W4A16 | 21.5 | 46.58 | 35 | 1.47 | 2.61 | 193 |
| Pruned + AWQ W4 | 37.0 | 27.05 | 22 | 1.37 | 2.20 | 172 |

## 4. Figures

![Accuracy vs. compression ratio](plots/01_accuracy_vs_compression.png)

![Accuracy per benchmark task](plots/02_accuracy_by_task.png)

![Mixed-precision bit allocation](plots/03_mixed_precision.png)

![Complexity: throughput, VRAM, FLOPs, prefill scaling](plots/04_complexity.png)

![Accuracy vs. speed / footprint](plots/05_efficiency_frontier.png)

![Perplexity vs. compression](plots/06_perplexity.png)

## 5. Method


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

## 6. Threats to validity


- 200 samples per task gives roughly ±3.5 pt standard error at p≈0.5 — treat
  single-point differences as noise, not signal.
- AWQ's speed-up is *memory-bandwidth* only. GFLOP/token is unchanged, so at
  batch 1 on a small model the gain is smaller than the 4× size reduction.
- WikiText-2 train text was also used as AWQ calibration data, so its
  perplexity is mildly optimistic.
- Structured pruning is uniform across layers; per-layer sensitivity search
  would do better at the same budget.
