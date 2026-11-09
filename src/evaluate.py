"""Load any variant and measure accuracy + compression ratio."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import torch

from .common import VARIANT_LABELS, count_params, dir_size_bytes, gb, save_json
from .tasks import DEFAULT_TASKS, evaluate_ppl, evaluate_task

log = logging.getLogger("eval")


def load_for_eval(variant: str, path: str | Path, device: str = "cuda", token=None):
    """Return (hf_model, tokenizer) for a baseline / pruned / AWQ checkpoint."""
    path = str(path)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if variant.startswith("awq"):
        from awq import AutoAWQForCausalLM

        log.info("[%s] loading AWQ checkpoint %s", variant, path)
        wrapper = AutoAWQForCausalLM.from_quantized(
            path, fuse_layers=False, safetensors=True,
            # AutoAWQ only accepts the accelerate keywords here ("auto",
            # "balanced", ...), not a raw device string like "cuda".
            device_map="auto", trust_remote_code=False,
        )
        model = wrapper.model
    else:
        from transformers import AutoModelForCausalLM

        log.info("[%s] loading fp16 checkpoint %s", variant, path)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            device_map="auto", token=token,
        )

    model.eval()
    return model, tok


@torch.inference_mode()
def run_eval(variant: str, path: str | Path, *, tasks: List[str] | None = None,
             limit: int = 200, max_len: int = 1024, do_ppl: bool = True,
             baseline_bytes: int | None = None, device: str = "cuda",
             token=None, extras: Dict | None = None,
             max_new_tokens: int | None = None) -> Dict:
    """Evaluate one variant; returns a metrics dict ready for results/."""
    tasks = tasks or DEFAULT_TASKS
    model, tok = load_for_eval(variant, path, device=device, token=token)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        log.warning("CUDA not available: latency numbers will be CPU numbers")

    acc: Dict[str, Dict] = {}
    for t in tasks:
        acc[t] = evaluate_task(model, tok, t, limit=limit, max_len=max_len,
                               max_new_tokens=max_new_tokens, device=device)

    ppl = evaluate_ppl(model, tok, seq_len=min(1024, max_len), device=device) if do_ppl else None

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
    size = dir_size_bytes(path)

    payload = {
        "variant": variant,
        "display_name": VARIANT_LABELS.get(variant, variant),
        "path": str(path),
        "params": count_params(model),
        "size_bytes": size,
        "size_gb": round(gb(size), 3),
        "compression_ratio": round(baseline_bytes / size, 3) if baseline_bytes and size else 1.0,
        "accuracy": {k: v["acc"] for k, v in acc.items()},
        "accuracy_norm": {k: v["acc_norm"] for k, v in acc.items()},
        "accuracy_detail": acc,
        "accuracy_avg": round(sum(v["acc"] for v in acc.values()) / max(len(acc), 1), 4),
        "ppl_wikitext2": ppl["ppl"] if ppl else None,
        "peak_vram_gb": round(peak_vram, 3),
    }
    if extras:
        payload.update(extras)
    return payload


def eval_and_save(variant: str, path: str | Path, out_json: str | Path, **kwargs) -> Dict:
    metrics = run_eval(variant, path, **kwargs)
    save_json(metrics, out_json)
    return metrics
