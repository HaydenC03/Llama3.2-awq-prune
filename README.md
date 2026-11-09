# Llama-3.2-1B — AWQ quantisation + structured pruning

A self-contained Docker environment that takes `Llama-3.2-1B-Instruct` and
produces four checkpoints, measures what each one costs and what each one
loses, then renders the plots and a Markdown report.

Built for **RTX 3050 (8 GB) / 16 GB RAM / Windows 11 + WSL2**.

---

## What you get

| Variant | What it is | Why include it |
|---|---|---|
| `baseline` | FP16 original | reference point (1.00× size) |
| `pruned` | structured MLP width pruning, keep 75 % | shows accuracy cost of *removing* compute |
| `awq` | AWQ W4A16, group 128, zero-point | 4-bit weights, same FLOPs |
| `pruned_awq` | prune → AWQ | the two orthogonal tricks stacked |

Outputs, in `results/`:

```
results/
├── metrics.json                 # every number, one file
├── REPORT.md                    # tables + embedded figures (paste into a report)
├── extended_scores.csv          # baseline vs variant, per task + memory reduction
├── extended_summary.csv         # per-variant average + memory reduction
├── extended/<variant>.json      # raw GSM8K / MS MARCO / BoolQ results
└── plots/
    ├── 01_accuracy_vs_compression.png   # accuracy vs compression ratio
    ├── 02_accuracy_by_task.png          # the 3 tasks, per variant
    ├── 03_mixed_precision.png           # per-block bit width + acc-vs-bits
    ├── 04_complexity.png                # tok/s, VRAM, GFLOPs, prefill scaling
    ├── 05_efficiency_frontier.png       # accuracy vs speed / disk
    └── 06_perplexity.png                # WikiText-2 PPL vs compression
```

`results/` currently holds **real measurements**. To regenerate a watermarked
placeholder set (useful for checking figure layout before a long run), use
`python -m src.synthetic_metrics` then plot `results/metrics.preview.json`.

---

## The three benchmark tasks

Chosen to cover three different failure modes of compression:

| Task | Shape | What it probes |
|---|---|---|
| **ARC-Easy** (`allenai/ai2_arc`) | 4-way MC, 2376 test items | grade-school science reasoning / factual knowledge |
| **HellaSwag** (`Rowan/hellaswag`) | 4-way completion, 10 042 val | commonsense — the most quantisation-sensitive of the three |
| **PIQA** (`ybisk/piqa`) | 2-way MC, 1838 val | physical commonsense |

Scoring is teacher forcing: append each candidate answer, take the sum of its
token log-probabilities, pick the argmax. This is exactly how
`lm-evaluation-harness` reports `acc`, so numbers are comparable, but it needs
no sampling and finishes in minutes instead of hours. WikiText-2 perplexity is
added as the sensitive intrinsic signal.

### Extended benchmarks (CSV comparison)

The `extended` stage adds three more tasks and writes a comparison table:

| Task | Shape | Metric |
|---|---|---|
| **GSM8K** (`openai/gsm8k`, test) | generated chain-of-thought, final `#### N` | numeric exact match |
| **MS MARCO** (`microsoft/ms_marco` v2.1, validation) | answer the query from the selected passage | token F1 (+ EM) |
| **BoolQ** (`google/boolq`, validation) | 2-way ` no` / ` yes` | accuracy |

GSM8K and MS MARCO are generative (greedy decoding), so they are slower than
the multiple-choice suite. The default is 100 samples per task; override with
`--extended-limit` and `--max-new-tokens`.

```powershell
powershell -File scripts/run.ps1 --stage extended
powershell -File scripts/run.ps1 --stage extended --extended-limit 50 --max-new-tokens 128
```

Outputs:

- `results/extended_scores.csv` — one row per variant × task with
  `baseline_score`, `variant_score`, `score_delta`,
  `weight_reduction_pct`, `vram_reduction_pct`, `load_vram_reduction_pct`.
- `results/extended_summary.csv` — one row per variant with the average score
  and memory reductions.
- `results/extended/<variant>.json` — raw per-task results and samples.

To rebuild the CSVs from cached JSONs without re-running the models:

```powershell
python -m src.extended_bench
```

"Memory reduction" is reported three ways: checkpoint size on disk
(`weight_reduction_pct`), peak CUDA memory during evaluation
(`vram_reduction_pct`), and CUDA memory right after loading
(`load_vram_reduction_pct`). All are relative to the FP16 baseline.

---

## Requirements

- Windows 11 (you have it) with **Docker Desktop** ≥ 4.30, WSL2 backend
- NVIDIA driver, **no CUDA toolkit needed on Windows**
- ~12 GB free disk (7 GB image + 3 GB models + caches)

### Install Docker (one time)

```powershell
winget install -e --id Docker.DockerDesktop
# then reboot, start Docker Desktop, and in Settings:
#   General        -> Use the WSL 2 based engine        [on]
#   Resources      -> WSL integration                    [on]
#   Resources      -> Advanced -> Memory                 [8 GB or more]
```

Verify the GPU is reachable from a container:

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## Quickstart

```powershell
cd C:\Users\User\Documents\CDE\llama32-awq-prune
Copy-Item .env.example .env        # optional: add HF_TOKEN for the gated repo
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1   # build + smoke test
powershell -File scripts/run.ps1 --stage all                 # full study
```

Preview the figures **without** Docker or a GPU (needs only matplotlib):

```powershell
powershell -File scripts/preview.ps1
```

### Stages

Every stage is resumable; `results/state.json` remembers what is done.

```powershell
powershell -File scripts/run.ps1 --stage prepare     # download FP16 baseline
powershell -File scripts/run.ps1 --stage prune
powershell -File scripts/run.ps1 --stage awq
powershell -File scripts/run.ps1 --stage pruned_awq
powershell -File scripts/run.ps1 --stage eval
powershell -File scripts/run.ps1 --stage bench
powershell -File scripts/run.ps1 --stage extended   # GSM8K / MS MARCO / BoolQ -> CSV
powershell -File scripts/run.ps1 --stage merge       # -> results/metrics.json
powershell -File scripts/run.ps1 --stage plots
powershell -File scripts/run.ps1 --stage report      # -> results/REPORT.md
```

Useful flags: `--quick` (50 eval samples, fast), `--limit 500`,
`--keep-ratio 0.6`, `--heal-steps 300`, `--extended-limit 50`,
`--max-new-tokens 128`, `--extended-tasks boolq gsm8k`, `--reset`.

### Runtime on your machine (RTX 3050 8 GB)

| Step | Time |
|---|---|
| `docker compose build` (first time) | 8–10 min on this machine |
| download 2.5 GB baseline | ~1 min at 75 MB/s |
| prune + save | < 1 min |
| AWQ, per variant (128 calibration samples, 16 layers) | **10–12 min** |
| eval, per variant (3 × 200 items + PPL) | 3–5 min |
| bench, per variant | 1 min |
| extended, per variant (100 × GSM8K/MS MARCO/BoolQ) | 8–20 min |
| **full `--stage all`** | **≈ 2–3 h with extended** |

---

## Configuration

Everything lives in `configs/pipeline.yaml`; every knob is also a CLI flag or
env var (see `.env.example`).

```yaml
awq:
  w_bit: 4                # AutoAWQ 0.2.7 only supports 4-bit
  q_group_size: 128
  calib_samples: 128
prune:
  keep_ratio: 0.75        # fraction of MLP intermediate width kept
  heal_steps: 0           # >0 runs a short LoRA recovery pass
eval:
  limit: 200
  tasks: [arc_easy, hellaswag, piqa]
extended:
  tasks: [gsm8k, ms_marco, boolq]
  limit: 100
  max_new_tokens: 256      # generation budget for GSM8K / MS MARCO
```

To study more prune ratios, rerun with `--keep-ratio 0.5 --reset` (or copy the
`artifacts/pruned` dir between runs) and merge the JSONs by hand.

---

## Design notes (the parts that are easy to get wrong)

**Why structured pruning, not `torch.nn.utils.prune`.** Magnitude masking zeroes
weights but the dense kernel still reads them: the checkpoint stays byte-identical
and nothing gets faster, so the "compression ratio" would be a fiction. This repo
physically slices `gate_proj` / `up_proj` / `down_proj` and rewrites
`intermediate_size`, so the file, the parameter count and the FLOPs all shrink.

**Why importances are `‖W_gate[i]‖₂ · ‖W_up[i]‖₂`.** Channel *i* survives only if
both the gate and the up projection read it strongly; the product penalises
channels that one branch cares about and the other ignores. Activation-aware
(Wanda/SparseGPT) scoring would be better — swap `prune.scoring` and implement
`_row_scores` if you want that for the report.

**Why AWQ does not make the model 4× faster.** AWQ is *weight-only*: activations
stay FP16 and every FLOP still happens. It cuts memory traffic (hence
`04_complexity` shows VRAM down and tok/s up), but `gflops_per_token` is
unchanged. Pruning is the only variant here that moves the FLOPs bar. That
contrast is the interesting result, and the plots are built to show it.

**Why `avg_bits` excludes `lm_head`.** The output head stays 16-bit in every AWQ
build; including its 262 M parameters would inflate the average to ≈ 6.4 bits and
hide the actual body precision. The scope is recorded in the metrics file.

**Why there is no true mixed-precision variant.** This was attempted and does
not work with AutoAWQ 0.2.7, for two independent reasons, both verified on this
machine:

1. `modules_to_not_convert` is matched with `key in name` against
   **module-relative** names (`self_attn.q_proj`, `mlp.down_proj`). Full-path
   patterns such as `layers.0.` never match, so per-block protection is
   silently ignored — while still being written into `config.json`, which makes
   it look applied. **Check the file size to confirm.**
2. Excluding `down_proj` (the only meaningful module *type* to protect, since it
   writes to the residual stream) raises `KeyError: 'mlp.down_proj'`: it is the
   scaling anchor for `gate_proj`/`up_proj`.

Restoring modules from the FP16 reference after quantisation also fails, and
instructively: AWQ rescales neighbours *inside* each block (`up_proj *= s`,
`down_proj /= s`), so splicing in unscaled weights desynchronises the pair. That
variant measured 26.5 % on ARC-Easy (chance) and PPL 4196 — it was discarded
rather than reported as a result.

A W8 build was tried next and rejected outright: AutoAWQ raises
`NotImplementedError: Only 4-bit are supported for now`. The backend therefore
offers exactly one precision, so the precision axis in the figures has two real
points (W4 and FP16). True per-layer bit allocation needs `llm-compressor` or
vLLM compressed-tensors.

**Why 200 samples per task.** ±3.5 pt standard error at p ≈ 0.5. The `awq` run
measured −0.8 pt average against FP16, which is *inside* that error bar — so the
honest claim is "no significant average loss on these three tasks", not
"−0.8 pt". Raise `--limit` before making finer claims.

---

## Repo layout

```
llama32-awq-prune/
├── docker/Dockerfile            CUDA 12.4 devel, compiles awq_ext for sm_86
├── docker-compose.yml           GPU passthrough, model cache volume
├── configs/pipeline.yaml        all hyper-parameters
├── src/
│   ├── common.py                config, JSON, size accounting, hw fingerprint
│   ├── tasks.py                 3 MC tasks + WikiText-2 PPL
│   ├── prune_model.py           structured MLP pruning (+ optional LoRA healing)
│   ├── quantize_awq.py          AWQ W4 + bit plan
│   ├── evaluate.py              load any variant, score it
│   ├── bench_complexity.py      tok/s, VRAM, FLOPs, prefill scaling
│   ├── make_plots.py            6 figures
│   ├── make_report.py           Markdown report
│   ├── extended_bench.py        GSM8K / MS MARCO / BoolQ + CSV comparison
│   ├── synthetic_metrics.py     watermarked preview data
│   └── pipeline.py              stage orchestrator
├── scripts/                     setup / run / preview (PowerShell + bash)
└── tests/                       fast tests, no GPU or download needed
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | Docker Desktop → Settings → Resources → enable GPU support; update the NVIDIA driver; `wsl --update`. |
| `awq_ext` import error at runtime | Rebuild the image: the wheel compiled against a different CUDA/numpy. Keep `numpy<2`. |
| CUDA OOM during AWQ | It should not happen at 1B/8 GB, but close other GPU apps and set `MAX_JOBS=2` in the Dockerfile. |
| `401 / gated repo` on `meta-llama/...` | Either accept the licence and set `HF_TOKEN`, or use `unsloth/Llama-3.2-1B-Instruct` (default in the config). |
| Container dies silently during eval | WSL memory limit. `%UserProfile%\.wslconfig` → `memory=10GB`, then `wsl --shutdown`. |
| Container dies during **AWQ** with exit code **137** | Same cause, and the default 50 % WSL cap is not enough. AutoAWQ holds the FP16 model *plus* 128×512 calibration tensors; 7.9 GB gets OOM-killed at layer 1/16. Create `%UserProfile%\.wslconfig` with `[wsl2]` / `memory=11GB` / `swap=4GB`, run `wsl --shutdown`, restart Docker Desktop. Verify with `wsl -d Ubuntu -- free -m`. |
| Plots show `PREVIEW DATA` | You are rendering `metrics.preview.json`. Run the pipeline, then `--stage plots`. |

---

## Tests

```powershell
docker compose run --rm pipeline bash
pytest -q
```

They cover the dataset adapters (including BoolQ / GSM8K / MS MARCO), the
bit-plan arithmetic, prune tensor shapes on a 32-hidden tiny Llama, the
extended CSV comparison, and a full preview→figure→report round trip.
