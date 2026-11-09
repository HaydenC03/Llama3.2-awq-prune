"""Orchestrator:  python -m src.pipeline --stage all

Stages (each is independently runnable and resumable):

    prepare      snapshot the FP16 baseline into artifacts/ (real on-disk size)
    pruned       structured MLP width pruning        -> artifacts/pruned
    awq          AWQ W4A16 group-128 (AutoAWQ is 4-bit only)  -> artifacts/awq
    pruned_awq   prune first, then AWQ               -> artifacts/pruned_awq
    eval         accuracy on 3 tasks + WikiText-2 PPL
    bench        throughput / VRAM / FLOPs / prefill
    merge        combine results/runs/*.json -> results/metrics.json
    plots        render results/plots/*.png
    report       write results/REPORT.md

State lives in results/state.json so a crashed run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

from .common import (ARTIFACTS, RESULTS, RUNS, Timer, VARIANT_LABELS, dir_size_bytes,
                     gb, hw_fingerprint, load_config, load_json, model_dir, save_json,
                     setup_logging, METRICS_SCHEMA_VERSION)

log = logging.getLogger("pipeline")
STATE_PATH = RESULTS / "state.json"


STAGES = ["prepare", "prune", "awq", "pruned_awq",
          "eval", "bench", "extended", "merge", "plots", "report"]
ALL_ORDER = ["prepare", "prune", "awq", "pruned_awq",
             "eval", "bench", "extended", "merge", "plots", "report"]


# --------------------------------------------------------------------------- #
def load_state() -> Dict:
    return load_json(STATE_PATH, default={"variants": {}}) or {"variants": {}}


def save_state(state: Dict) -> None:
    save_json(state, STATE_PATH)


def cleanup() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
def stage_prepare(cfg: Dict, state: Dict) -> None:
    """Download the baseline once so sizes and later stages are reproducible."""
    from huggingface_hub import snapshot_download

    model_id = cfg["model"]["id"]
    token = os.getenv("HF_TOKEN") or None
    dest = model_dir("baseline", model_id)
    if (dest / "config.json").exists():
        log.info("baseline already present at %s", dest)
    else:
        log.info("snapshotting %s -> %s (first run downloads ~2.5 GB)", model_id, dest)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(dest),
            token=token,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt"],
        )
    state["variants"]["baseline"] = {"path": str(dest), "kind": "fp16"}
    save_state(state)
    log.info("baseline on disk: %.2f GB", gb(dir_size_bytes(dest)))


# --------------------------------------------------------------------------- #
def stage_prune(cfg: Dict, state: Dict) -> None:
    from .prune_model import heal_with_lora, load_base, prune_mlp_width, save_model

    src = state["variants"]["baseline"]["path"]
    out = model_dir("pruned", cfg["model"]["id"])
    model, tok = load_base(src, device="cuda", token=os.getenv("HF_TOKEN") or None)

    with Timer("structured pruning", log):
        report = prune_mlp_width(model, cfg["prune"]["keep_ratio"], cfg["prune"]["scoring"])

    heal = None
    if int(cfg["prune"].get("heal_steps", 0)) > 0:
        model, heal = heal_with_lora(model, tok, steps=int(cfg["prune"].get("heal_steps", 0)),
                                     lr=float(cfg["prune"].get("heal_lr", 2e-4)))
    model.to("cpu")
    save_model(model, tok, out)
    del model
    cleanup()

    state["variants"]["pruned"] = {
        "path": str(out), "kind": "fp16", "avg_bits": 16,
        "prune": report, "heal": heal,
    }
    save_state(state)


# --------------------------------------------------------------------------- #
def _run_awq(cfg: Dict, state: Dict, variant: str, src_key: str, protect: List[str]) -> None:
    from .quantize_awq import run_awq

    src = state["variants"][src_key]["path"]
    out = model_dir(variant, cfg["model"]["id"])
    a = cfg["awq"]
    with Timer(f"AWQ {variant}", log):
        meta = run_awq(
            src, out,
            w_bit=int(a["w_bit"]), group_size=int(a["q_group_size"]),
            zero_point=bool(a["zero_point"]), version=a["version"],
            calib_samples=int(a["calib_samples"]), calib_seq_len=int(a["calib_seq_len"]),
            protect=protect, protect_bits=int(a.get("mixed_precision_bits", 8)),
            token=os.getenv("HF_TOKEN") or None,
        )
    state["variants"][variant] = {
        "path": str(out), "kind": "awq",
        "avg_bits": meta["bit_plan"]["avg_bits"],
        "blocks": meta["blocks"],
        "bit_plan": meta["bit_plan"],
        "quant": meta,
    }
    save_state(state)
    cleanup()


def stage_awq(cfg: Dict, state: Dict) -> None:
    _run_awq(cfg, state, "awq", "baseline", [])


def stage_pruned_awq(cfg: Dict, state: Dict) -> None:
    if "pruned" not in state["variants"]:
        raise SystemExit("run the `prune` stage first")
    _run_awq(cfg, state, "pruned_awq", "pruned", [])


# --------------------------------------------------------------------------- #
def stage_eval(cfg: Dict, state: Dict) -> None:
    from .evaluate import run_eval

    e = cfg["eval"]
    base_bytes = dir_size_bytes(state["variants"]["baseline"]["path"])
    for variant, info in list(state["variants"].items()):
        out_json = RUNS / f"{variant}.eval.json"
        if out_json.exists():
            log.info("[%s] already evaluated (%s), skipping", variant, out_json.name)
            continue
        with Timer(f"eval {variant}", log):
            metrics = run_eval(
                variant, info["path"],
                tasks=list(e["tasks"]), limit=int(e["limit"]), max_len=int(e["max_len"]),
                do_ppl=True, baseline_bytes=base_bytes,
                token=os.getenv("HF_TOKEN") or None,
                extras={k: v for k, v in info.items()
                        if k in ("avg_bits", "blocks", "bit_plan", "prune", "heal", "quant")},
            )
        save_json(metrics, out_json)
        cleanup()


# --------------------------------------------------------------------------- #
def stage_bench(cfg: Dict, state: Dict) -> None:
    from .bench_complexity import run_bench
    from .evaluate import load_for_eval

    b = cfg["bench"]
    for variant, info in list(state["variants"].items()):
        json_path = RUNS / f"{variant}.bench.json"
        eval_path = RUNS / f"{variant}.eval.json"
        if not eval_path.exists():
            log.warning("[%s] no eval yet, skipping bench", variant)
            continue
        if json_path.exists():
            log.info("[%s] already benchmarked, skipping", variant)
            continue
        model, tok = load_for_eval(variant, info["path"], token=os.getenv("HF_TOKEN") or None)
        with Timer(f"bench {variant}", log):
            res = run_bench(model, tok,
                            prompt_lengths=[int(x) for x in b["prompt_lengths"]],
                            new_tokens=int(b["new_tokens"]), repeats=int(b["repeats"]))
        save_json(res, json_path)
        del model
        cleanup()

        # splice complexity into the eval record so metrics.json stays flat
        ev = load_json(eval_path)
        ev["complexity"] = res
        ev["avg_bits"] = info.get("avg_bits", ev.get("avg_bits"))
        save_json(ev, eval_path)


# --------------------------------------------------------------------------- #
def stage_extended(cfg: Dict, state: Dict) -> None:
    """GSM8K / MS MARCO / BoolQ for every variant + comparison CSVs."""
    from .extended_bench import EXTENDED_TASKS, run_variant, write_csvs

    e = cfg.get("extended", {})
    tasks = list(e.get("tasks") or EXTENDED_TASKS)
    limit = int(e.get("limit", 100))
    max_len = int(e.get("max_len", cfg["eval"]["max_len"]))
    max_new_tokens = int(e.get("max_new_tokens", 256))

    if not state.get("variants"):
        raise SystemExit("no variants in results/state.json; run `prepare`/`prune`/`awq` first")

    for variant, info in list(state["variants"].items()):
        run_variant(
            variant, info["path"], tasks=tasks, limit=limit,
            max_len=max_len, max_new_tokens=max_new_tokens,
            token=os.getenv("HF_TOKEN") or None,
        )
        cleanup()

    write_csvs(tasks)


# --------------------------------------------------------------------------- #
def stage_merge(cfg: Dict, state: Dict) -> None:
    variants: Dict[str, Dict] = {}
    for variant in state["variants"]:
        ev = load_json(RUNS / f"{variant}.eval.json")
        if not ev:
            continue
        info = state["variants"][variant]
        for k in ("avg_bits", "blocks", "bit_plan", "prune", "heal", "quant"):
            if info.get(k) is not None and ev.get(k) is None:
                ev[k] = info[k]
        ev["display_name"] = VARIANT_LABELS.get(variant, variant)
        variants[variant] = ev

    if not variants:
        raise SystemExit("nothing to merge: run `eval` first")

    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "meta": {
            "model_id": cfg["model"]["id"],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hardware": hw_fingerprint(),
            "config": cfg,
            "synthetic": False,
        },
        "variants": variants,
    }
    save_json(metrics, RESULTS / "metrics.json")
    log.info("merged %d variants -> results/metrics.json", len(variants))


# --------------------------------------------------------------------------- #
def stage_plots(cfg: Dict, state: Dict) -> None:
    from . import make_plots

    make_plots.main(RESULTS / "metrics.json")


def stage_report(cfg: Dict, state: Dict) -> None:
    from .make_report import write_report

    write_report(RESULTS / "metrics.json", RESULTS / "REPORT.md")


# --------------------------------------------------------------------------- #
DISPATCH = {
    "prepare": stage_prepare, "prune": stage_prune, "awq": stage_awq,
    "pruned_awq": stage_pruned_awq,
    "eval": stage_eval, "bench": stage_bench, "extended": stage_extended,
    "merge": stage_merge,
    "plots": stage_plots, "report": stage_report,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Llama-3.2-1B AWQ + pruning study")
    ap.add_argument("--stage", default="all", choices=["all"] + STAGES,
                    help="stage to run (default: all)")
    ap.add_argument("--config", default=None, help="path to pipeline.yaml")
    ap.add_argument("--limit", type=int, default=None, help="eval samples per task")
    ap.add_argument("--keep-ratio", type=float, default=None, help="pruning keep ratio")
    ap.add_argument("--heal-steps", type=int, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke run: 50 samples, short bench")
    ap.add_argument("--extended-tasks", nargs="*", default=None,
                    help="extended benchmarks (default: gsm8k ms_marco boolq)")
    ap.add_argument("--extended-limit", type=int, default=None,
                    help="samples per extended benchmark")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="max generated tokens for GSM8K / MS MARCO")
    ap.add_argument("--reset", action="store_true", help="delete results/state.json and rerun")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    setup_logging(args.log_level)

    overrides: Dict = {}
    if args.quick:
        overrides = {"eval": {"limit": 50}, "bench": {"new_tokens": 32, "repeats": 1,
                                                      "prompt_lengths": [128, 512]},
                     "extended": {"tasks": ["boolq"], "limit": 20,
                                  "max_new_tokens": 32}}
    if args.limit is not None:
        overrides.setdefault("eval", {})["limit"] = args.limit
    if args.keep_ratio is not None:
        overrides.setdefault("prune", {})["keep_ratio"] = args.keep_ratio
    if args.heal_steps is not None:
        overrides.setdefault("prune", {})["heal_steps"] = args.heal_steps
    if args.extended_tasks:
        overrides.setdefault("extended", {})["tasks"] = args.extended_tasks
    if args.extended_limit is not None:
        overrides.setdefault("extended", {})["limit"] = args.extended_limit
    if args.max_new_tokens is not None:
        overrides.setdefault("extended", {})["max_new_tokens"] = args.max_new_tokens

    cfg = load_config(args.config, overrides)

    if args.reset or not STATE_PATH.exists():
        save_json({"variants": {}}, STATE_PATH)
    state = load_state()

    stages = ALL_ORDER if args.stage == "all" else [args.stage]
    t0 = time.perf_counter()
    log.info("model=%s | stages=%s | keep_ratio=%s | eval_limit=%s",
             cfg["model"]["id"], stages, cfg["prune"]["keep_ratio"], cfg["eval"]["limit"])

    for st in stages:
        log.info("=" * 60)
        log.info("STAGE: %s", st)
        log.info("=" * 60)
        DISPATCH[st](cfg, state)
        state = load_state()

    log.info("done in %.1f min", (time.perf_counter() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
