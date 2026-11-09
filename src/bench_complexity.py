"""Complexity / cost measurements.

What "complexity" means here, and why all four numbers matter:

* ``ms_per_token``   - reciprocal of generation throughput (latency).
* ``tokens_per_s``   - throughput at batch 1, the realistic desktop case.
* ``peak_vram_gb``   - footprint; decides whether it fits at all.
* ``gflops_per_token`` - analytic compute per generated token
                         2 * N_non_embedding + 4 * n_layer * d * seq
                       (the second term is attention). This is the number that
                       explains *why* pruning helps and weight quantisation
                       mostly does not: 4-bit weights shrink bytes, not FLOPs.

Also measures prefill latency vs prompt length, so the O(n^2) attention term is
visible instead of hand-waved.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List

import torch

log = logging.getLogger("bench")


def _sync() -> None:
    """torch.cuda.synchronize() when there is a GPU, no-op otherwise."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
def analytic_gflops_per_token(model, seq_len: int = 1024) -> tuple[float, float]:
    """FLOPs and parameter count derived from the *config*, not from tensors.

    This must not count `model.parameters()`: a 4-bit checkpoint stores packed
    int32 `qweight`s, so tensor counting reports a 16x too-small model and
    makes AWQ look like it reduced compute. Quantisation changes bytes moved,
    not arithmetic -- the config-derived number is the honest one.
    """
    cfg = model.config
    d = cfg.hidden_size
    n_layer = cfg.num_hidden_layers
    ff = cfg.intermediate_size
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads) or n_heads
    dh = d // n_heads

    attn = d * d + 2 * d * (n_kv * dh) + d * d          # q, k, v, o
    mlp = 3 * d * ff
    body = n_layer * (attn + mlp)
    head = cfg.vocab_size * d                            # output projection
    n_params = body + head

    attention_term = 2 * 2 * n_layer * seq_len * d       # QK^T + AV
    gflops = (2 * n_params + attention_term) / 1e9
    return round(gflops, 4), round(n_params / 1e9, 4)


@torch.inference_mode()
def measure_generation(model, tokenizer, prompt_len: int = 64, new_tokens: int = 128,
                       repeats: int = 3, device: str = "cuda") -> Dict:
    ids = torch.randint(100, 30000, (1, prompt_len), device=device)
    attn = torch.ones_like(ids)

    # warm-up (kernel autotune + cache allocation)
    model.generate(ids, attention_mask=attn, max_new_tokens=4,
                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
    _sync()

    times, ttfts = [], []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        # TTFT: a single forward pass over the prompt
        model(input_ids=ids, attention_mask=attn, use_cache=True)
        _sync()
        ttft = time.perf_counter() - t0

        t0 = time.perf_counter()
        out = model.generate(ids, attention_mask=attn, max_new_tokens=new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
        _sync()
        dt = time.perf_counter() - t0
        generated = out.shape[1] - prompt_len
        times.append(dt / max(generated, 1))
        ttfts.append(ttft)

    ms_per_tok = min(times) * 1000  # best-of-n: least noise from other apps
    return {
        "ms_per_token": round(ms_per_tok, 3),
        "tokens_per_s": round(1000 / ms_per_tok, 1),
        "ttft_ms": round(min(ttfts) * 1000, 2),
        "prompt_len": prompt_len,
        "new_tokens": new_tokens,
    }


@torch.inference_mode()
def measure_prefill(model, prompt_lengths: List[int], repeats: int = 3,
                    device: str = "cuda") -> Dict[str, float]:
    """Latency of a single forward pass vs prompt length (shows O(n^2))."""
    out: Dict[str, float] = {}
    for n in prompt_lengths:
        ids = torch.randint(100, 30000, (1, n), device=device)
        attn = torch.ones_like(ids)
        model(input_ids=ids, attention_mask=attn, use_cache=False)  # warm-up
        _sync()
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(input_ids=ids, attention_mask=attn, use_cache=False)
            _sync()
            best = min(best, time.perf_counter() - t0)
        out[str(n)] = round(best * 1000, 2)   # ms
        log.info("prefill %5d tok -> %7.2f ms (%.0f tok/s)", n, best * 1000, n / best)
    return out


# --------------------------------------------------------------------------- #
def run_bench(model, tokenizer, *, prompt_lengths: List[int], new_tokens: int = 128,
              repeats: int = 3, device: str = "cuda", do_prefill: bool = True) -> Dict:
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    gen = measure_generation(model, tokenizer, new_tokens=new_tokens, repeats=repeats, device=device)
    gflops, n_params_b = analytic_gflops_per_token(model)
    prefill = measure_prefill(model, prompt_lengths, repeats, device) if do_prefill else {}
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
    return {
        **gen,
        "peak_vram_gb": round(peak, 3),
        "gflops_per_token": gflops,
        "params_b": n_params_b,
        "prefill_ms": prefill,
    }
