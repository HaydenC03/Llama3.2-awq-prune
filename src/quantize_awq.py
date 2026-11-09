"""AWQ (Activation-aware Weight Quantisation) weight-only 4-bit quantisation.

Pipeline stage:  FP16 model  ->  AWQ  ->  safetensors W4A16 group-wise
Group size 128, zero-point on, GEMM kernel layout (what vLLM / AutoAWQ load).

"Mixed precision" is expressed the AWQ-native way: a list of modules that keep
a higher bit-width (``modules_to_not_convert``). Protecting the first block,
the last block and the down projections is the usual recipe, because those are
where activation outliers and the residual stream live.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List

import torch

log = logging.getLogger("awq")


# --------------------------------------------------------------------------- #
# Calibration data
# --------------------------------------------------------------------------- #
def build_calibration(tokenizer, n_samples: int = 128, seq_len: int = 512) -> List[str]:
    """Real text (WikiText-2 train) instead of random ids -> better scales."""
    from datasets import load_dataset

    log.info("building %d calibration samples x %d tokens", n_samples, seq_len)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    need = n_samples * seq_len
    if len(ids) < need:
        ids = (ids * (need // max(len(ids), 1) + 1))[:need]
    return [
        tokenizer.decode(ids[i * seq_len: (i + 1) * seq_len])
        for i in range(n_samples)
    ]


# --------------------------------------------------------------------------- #
# Bit plan (used by the mixed-precision plots)
# --------------------------------------------------------------------------- #
def build_bit_plan(model, protect_patterns: List[str], w_bit: int = 4,
                   protect_bits: int = 8) -> Dict:
    """Per-linear-module bit-width, derived from the same rules AWQ applies.

    ``avg_bits`` is averaged over the transformer body only (``lm_head`` is
    excluded): the output head stays 16-bit in every AWQ build and including
    it would flatter the effective precision number.
    """
    plan: Dict[str, Dict] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            keep = any(p in name for p in protect_patterns)
            plan[name] = {"bits": protect_bits if keep else w_bit,
                          "params": int(mod.weight.numel()),
                          "protected": bool(keep),
                          "in_avg": "lm_head" not in name}
    avg_pool = {k: v for k, v in plan.items() if v["in_avg"]}
    total_params = sum(v["params"] for v in avg_pool.values()) or 1
    avg_bits = sum(v["bits"] * v["params"] for v in avg_pool.values()) / total_params
    return {"modules": plan, "avg_bits": round(avg_bits, 3), "w_bit": w_bit,
            "protect_bits": protect_bits, "protect_patterns": list(protect_patterns),
            "avg_bits_scope": "transformer body linears (lm_head excluded)"}


def bit_plan_by_block(plan: Dict) -> Dict[str, Dict]:
    """Group the bit plan per decoder block for plotting."""
    blocks: Dict[str, Dict] = {}
    for name, v in plan["modules"].items():
        key = "embeddings/head"
        parts = name.split(".")
        if "layers" in parts:
            i = parts.index("layers")
            key = f"block {int(parts[i + 1]):02d}"
        b = blocks.setdefault(key, {"params": 0, "weighted_bits": 0.0, "max_bits": 0})
        b["params"] += v["params"]
        b["weighted_bits"] += v["bits"] * v["params"]
        b["max_bits"] = max(b["max_bits"], v["bits"])
    for b in blocks.values():
        b["bits"] = round(b["weighted_bits"] / max(b["params"], 1), 3)
    return dict(sorted(blocks.items()))


# --------------------------------------------------------------------------- #
# Quantisation
# --------------------------------------------------------------------------- #
def run_awq(src: str, out_dir: str | Path, *, w_bit: int = 4, group_size: int = 128,
            zero_point: bool = True, version: str = "GEMM",
            calib_samples: int = 128, calib_seq_len: int = 512,
            protect: List[str] | None = None, protect_bits: int = 8,
            token: str | None = None, device: str = "cuda") -> Dict:
    """Quantise ``src`` (HF id or local dir) and save to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "AutoAWQ is not installed. Build the image with `docker compose build`, "
            "or `pip install autoawq==0.2.7.post3` in a CUDA environment."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("AWQ needs a CUDA device (the calibration pass runs on GPU).")

    tok = AutoTokenizer.from_pretrained(src, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log.info("loading %s for quantisation", src)
    model = AutoAWQForCausalLM.from_pretrained(
        src, low_cpu_mem_usage=True, use_cache=False, token=token
    )
    plan = build_bit_plan(model.model, protect or [], w_bit, protect_bits)
    log.info("bit plan: %d linears, avg %.2f bits/module", len(plan["modules"]), plan["avg_bits"])

    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": w_bit,
        "version": version,
    }
    if protect:
        quant_config["modules_to_not_convert"] = list(protect)

    calib = build_calibration(tok, calib_samples, calib_seq_len)

    t0 = time.perf_counter()
    model.quantize(tok, quant_config=quant_config, calib_data=calib)
    quant_seconds = time.perf_counter() - t0

    model.save_quantized(str(out_dir))   # AutoAWQ 0.2.7 indexes the path: needs str
    tok.save_pretrained(out_dir)
    log.info("AWQ done in %.1f s -> %s", quant_seconds, out_dir)

    return {
        "quant_seconds": round(quant_seconds, 1),
        "quant_config": quant_config,
        "bit_plan": plan,
        "blocks": bit_plan_by_block(plan),
        "calib_samples": calib_samples,
        "calib_seq_len": calib_seq_len,
    }
