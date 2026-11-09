"""Generate a clearly-labelled PREVIEW metrics file (no GPU needed).

Purpose: lets you render every figure and the report *before* the hours-long
run finishes, so you can check layout and validate the analysis code.

    python -m src.synthetic_metrics
    python -m src.make_plots --metrics results/metrics.preview.json

The numbers below are literature-typical values for a 1B-class model on this
class of GPU. They are NOT measurements. ``meta.synthetic`` is set to true and
every figure that uses them carries a red PREVIEW watermark.
"""
from __future__ import annotations

import time
from pathlib import Path

from .common import RESULTS, hw_fingerprint, save_json


def _blocks(uniform_bits: float, protected=(), mixed_block=None) -> dict:
    out = {}
    for i in range(16):
        if i in protected:
            bits = 8.0
        elif mixed_block is not None and i == mixed_block:
            bits = 5.0
        else:
            bits = uniform_bits
        out[f"block {i:02d}"] = {
            "params": 67_100_000,
            "weighted_bits": bits * 67_100_000,
            "max_bits": 8 if bits > uniform_bits else uniform_bits,
            "bits": round(bits, 3),
        }
    out["embeddings/head"] = {"params": 262_100_000, "weighted_bits": 8 * 262_100_000,
                              "max_bits": 8, "bits": 8.0}
    return out


def build_preview() -> dict:
    base_size = 2.47e9
    variants = {
        "baseline": dict(
            size=2.47e9, bits=16.0, acc=(0.452, 0.523, 0.741), ppl=9.86,
            tps=78.0, vram=2.90, gf=2.49,
            pre={"128": 41, "512": 132, "1024": 409, "2048": 1487},
            display="Original (FP16)",
        ),
        "pruned": dict(
            size=2.08e9, bits=16.0, acc=(0.425, 0.494, 0.716), ppl=12.41,
            tps=88.0, vram=2.60, gf=2.10,
            pre={"128": 37, "512": 115, "1024": 355, "2048": 1290},
            display="Pruned (structured, keep 75%)",
        ),
        "awq": dict(
            size=0.86e9, bits=4.0, acc=(0.441, 0.508, 0.734), ppl=10.51,
            tps=106.0, vram=1.93, gf=2.49,
            pre={"128": 38, "512": 119, "1024": 371, "2048": 1338},
            display="AWQ W4A16 (g128)",
        ),
        "pruned_awq": dict(
            size=0.74e9, bits=4.0, acc=(0.415, 0.482, 0.709), ppl=13.24,
            tps=118.0, vram=1.62, gf=2.10,
            pre={"128": 33, "512": 101, "1024": 316, "2048": 1140},
            display="Pruned + AWQ W4",
        ),
    }

    out = {}
    for name, v in variants.items():
        tasks = ["arc_easy", "hellaswag", "piqa"]
        acc = {t: a for t, a in zip(tasks, v["acc"])}
        avg = sum(v["acc"]) / 3
        rec = {
            "variant": name,
            "display_name": v["display"],
            "path": f"artifacts/{name}/Llama-3.2-1B-Instruct",
            "params": 1_240_000_000,
            "size_bytes": v["size"],
            "size_gb": round(v["size"] / 1024 ** 3, 3),
            "compression_ratio": round(base_size / v["size"], 3),
            "accuracy": acc,
            "accuracy_norm": {k: round(x + 0.004, 4) for k, x in acc.items()},
            "accuracy_detail": {
                k: {"n": 200, "acc": x, "acc_norm": round(x + 0.004, 4), "nll": round(2.1 - x, 3)}
                for k, x in acc.items()
            },
            "accuracy_avg": round(avg, 4),
            "ppl_wikitext2": v["ppl"],
            "peak_vram_gb": v["vram"],
            "avg_bits": v["bits"],
            "complexity": {
                "ms_per_token": round(1000 / v["tps"], 3),
                "tokens_per_s": v["tps"],
                "ttft_ms": v["pre"]["128"],
                "prompt_len": 64,
                "new_tokens": 128,
                "peak_vram_gb": v["vram"],
                "gflops_per_token": v["gf"],
                "params_b": 1.24,
                "prefill_ms": v["pre"],
            },
        }
        if name == "awq":
            rec["blocks"] = _blocks(v["bits"])
            rec["bit_plan"] = {
                "avg_bits": v["bits"], "w_bit": int(v["bits"]), "protect_bits": 16,
                "protect_patterns": [],
                "avg_bits_scope": "transformer body linears (lm_head excluded)",
            }
            rec["quant"] = {"quant_seconds": 214.0,
                            "calib_samples": 128, "calib_seq_len": 512}
        out[name] = rec

    return {
        "schema_version": 1,
        "meta": {
            "model_id": "unsloth/Llama-3.2-1B-Instruct",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hardware": hw_fingerprint(),
            "synthetic": True,
            "note": ("PREVIEW DATA. Literature-typical reference values used to "
                     "exercise the plotting/report code paths. Replace by running "
                     "`python -m src.pipeline --stage all`."),
        },
        "variants": out,
    }


def main() -> None:
    path = save_json(build_preview(), RESULTS / "metrics.preview.json")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
