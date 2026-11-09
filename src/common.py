"""Shared helpers: config loading, JSON IO, size/parameter accounting, logging."""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "pipeline.yaml"
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
PLOTS = RESULTS / "plots"
ARTIFACTS = ROOT / "artifacts"
for _p in (RESULTS, RUNS, PLOTS, ARTIFACTS):
    _p.mkdir(parents=True, exist_ok=True)

METRICS_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class Timer:
    """Context manager that records wall-clock seconds."""

    def __init__(self, label: str = "", log: logging.Logger | None = None):
        self.label = label
        self.log = log

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        if self.log and self.label:
            self.log.info("%s finished in %.1fs", self.label, self.elapsed)
        return False


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load configs/pipeline.yaml, apply ``overrides`` (dotted keys), then env vars."""
    path = Path(path) if path else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    # environment wins over the file (Docker friendly)
    env_map = {
        "LLAMA_MODEL_ID": ("model", "id"),
        "AWQ_CALIB_SAMPLES": ("awq", "calib_samples"),
        "PRUNE_KEEP_RATIO": ("prune", "keep_ratio"),
        "EVAL_LIMIT": ("eval", "limit"),
        "EVAL_MAX_LEN": ("eval", "max_len"),
        "HEAL_STEPS": ("prune", "heal_steps"),
        "EXTENDED_LIMIT": ("extended", "limit"),
        "EXTENDED_MAX_NEW_TOKENS": ("extended", "max_new_tokens"),
    }
    for env, (sec, key) in env_map.items():
        if os.getenv(env):
            raw = os.environ[env]
            cfg.setdefault(sec, {})[key] = _coerce(raw)

    if overrides:
        _deep_update(cfg, overrides)

    cfg.setdefault("model", {})["id"] = os.getenv("LLAMA_MODEL_ID", cfg["model"]["id"])
    return cfg


def _coerce(raw: str):
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Size / parameter accounting
# --------------------------------------------------------------------------- #
_WEIGHT_SUFFIXES = (".bin", ".safetensors", ".pt", ".pth")


def dir_size_bytes(path: str | Path) -> int:
    """Total bytes of model weight files (ignores tokenizer/json noise)."""
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file() and f.suffix in _WEIGHT_SUFFIXES:
            total += f.stat().st_size
    return total


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_params_nonembed(model) -> int:
    emb = model.get_input_embeddings()
    n = count_params(model)
    if emb is not None:
        n -= emb.weight.numel()
    return n


def gb(num_bytes: float) -> float:
    return num_bytes / (1024 ** 3)


# --------------------------------------------------------------------------- #
# Hardware / environment fingerprint (stored in every metrics file)
# --------------------------------------------------------------------------- #
def hw_fingerprint() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "ram_gb": round(_total_ram_gb(), 1),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["vram_gb"] = round(props.total_memory / 1024 ** 3, 1)
            info["sm"] = f"{props.major}.{props.minor}"
        else:
            info["gpu"] = None
            info["vram_gb"] = 0
    except Exception:  # torch not installed (plot-only host)
        info["torch"] = None
    return info


def _total_ram_gb() -> float:
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                text=True, stderr=subprocess.DEVNULL,
            )
            digits = [ln.strip() for ln in out.splitlines() if ln.strip().isdigit()]
            return int(digits[0]) / 1024 ** 3 if digits else 0.0
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return 0.0


# --------------------------------------------------------------------------- #
# Variant naming
# --------------------------------------------------------------------------- #
VARIANT_LABELS = {
    "baseline": "Original (FP16)",
    "pruned": "Pruned (structured)",
    "awq": "AWQ W4A16",
    "pruned_awq": "Pruned + AWQ W4",
}


def iter_variants(variants: Iterable[str]) -> Iterable[str]:
    for v in variants:
        if v not in VARIANT_LABELS:
            raise ValueError(f"unknown variant {v!r}; known: {list(VARIANT_LABELS)}")
        yield v


def model_dir(variant: str, model_id: str) -> Path:
    return ARTIFACTS / variant / model_id.split("/")[-1]
