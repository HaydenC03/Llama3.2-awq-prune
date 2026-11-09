# Experiment design (for the write-up)

## Question

For a 1B-parameter LLM on a single 8 GB consumer GPU, how do **weight
quantisation** and **structured pruning** trade accuracy against cost — and do
they stack?

## Hypotheses

| # | Hypothesis | Where it shows up |
|---|---|---|
| H1 | 4-bit AWQ keeps accuracy within ~2 pt of FP16 at ~3× smaller checkpoints | `01`, `02` |
| H2 | Structured pruning of 25 % of the MLP width costs more accuracy per byte saved than AWQ | `01` (slope between baseline→pruned vs baseline→awq) |
| H3 | Pruning cuts FLOPs per token; AWQ does not (weight-only) | `04_complexity`, GFLOP panel |
| H4 | W4 AWQ is within noise of FP16 on these three tasks, i.e. extra bits buy little | `03_mixed_precision` |
| H5 | Perplexity degrades earlier than multiple-choice accuracy | `06` vs `02` |

## Independent variables

1. **Compression method**: FP16 / struct-prune / AWQ W4 / prune+AWQ.
2. **Prune ratio** (`prune.keep_ratio` ∈ {0.9, 0.75, 0.6, 0.5}).
3. **Group size** (`awq.q_group_size`, 128 default). Per-module bit-width was
   attempted and is unsupported by AutoAWQ — see README "Design notes".

## Controls

- Same fixed evaluation subset for every variant (seed 42, `eval.limit`).
- Same tokenizer, same prompts, same scoring rule (length-normalised
  log-likelihood argmax).
- Benchmarks run best-of-N with a warm-up, batch size 1, greedy decoding.
- All measurements on one machine, one driver version; `meta.hardware` is
  stored in `metrics.json` so the numbers are attributable.

## Dependent variables

- Accuracy per task + average; WikiText-2 perplexity.
- Checkpoint size (bytes on disk) → compression ratio = baseline / variant.
- Tokens/s, ms/token, TTFT, peak VRAM, analytic GFLOP/token, prefill latency
  at 128 / 512 / 1024 / 2048 tokens.

## Producing the extra prune-ratio points

`pipeline.py` writes one artifact dir per variant, so a sweep is:

```powershell
foreach ($r in 0.9, 0.75, 0.6, 0.5) {
  powershell -File scripts/run.ps1 --stage prune --keep-ratio $r --reset
  Copy-Item artifacts\pruned artifacts\pruned_$r -Recurse
  powershell -File scripts/run.ps1 --stage eval
  Copy-Item results\runs\pruned.eval.json results\runs\pruned_$r.eval.json
}
```

Then merge the JSON files into `results/metrics.json` (the plot code only needs
the `variants` dict) and rerun `--stage plots`. Each extra point costs ~6 min.

## Additional benchmark suite (GSM8K / MS MARCO / BoolQ)

The `extended` stage complements the three multiple-choice tasks with a
generative and reading-comprehension suite:

| Task | Split | Metric | Why it is included |
|---|---|---|---|
| GSM8K | test | numeric exact match | multi-step arithmetic; tests whether compression damages reasoning |
| MS MARCO (v2.1) | validation | token F1 / EM | answer extraction from a selected passage; tests reading comprehension |
| BoolQ | validation | accuracy | yes/no inference; cheap and deterministic |

GSM8K and MS MARCO use greedy generation, so they are slower and noisier than
teacher-forced scoring. Keep `--extended-limit` and `--max-new-tokens` identical
across variants. The CSV (`results/extended_scores.csv`) reports the baseline
score, the variant score, and the checkpoint / VRAM reduction next to each
other. The per-variant average mixes accuracy and F1, so use it as a summary,
not as a formal statistic.

## Statistics you should report

With `n = 200` items and `p ≈ 0.5`, the standard error of one accuracy number is
`sqrt(0.25/200) ≈ 3.5 pt`. Paired comparisons on the *same* subset are tighter,
but still expect ±2 pt. Report the drop **with** the sample size, and never call
a 0.5 pt difference significant at this budget.
