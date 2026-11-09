"""Structured (width) pruning of the MLP blocks.

Why structured and not ``torch.nn.utils.prune`` masks:
    Unstructured magnitude pruning zeroes individual weights but the dense
    kernel still touches every one of them, so the file stays the same size and
    nothing gets faster. That would make the "compressed ratio" column a lie.

    Here we physically drop whole intermediate channels of every Llama MLP
    (gate_proj / up_proj / down_proj), which shrinks the checkpoint, the
    parameter count, and the FLOPs. ``intermediate_size`` is rewritten in the
    config so the result loads with vanilla ``transformers``.

Importance score (per intermediate channel i):
    s_i = ||W_gate[i, :]||_2 * ||W_up[i, :]||_2
    -> channels that a large gate *and* a large up row read are kept.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

log = logging.getLogger("prune")


# --------------------------------------------------------------------------- #
def _row_scores(gate_w: torch.Tensor, up_w: torch.Tensor, mode: str) -> torch.Tensor:
    g = gate_w.detach().float()
    u = up_w.detach().float()
    if mode == "gate_up_norm":
        return g.norm(dim=1) * u.norm(dim=1)
    if mode == "gate_norm":
        return g.norm(dim=1)
    if mode == "up_norm":
        return u.norm(dim=1)
    if mode == "l1":
        return g.abs().sum(dim=1) + u.abs().sum(dim=1)
    raise ValueError(f"unknown pruning score {mode!r}")


def _linear(in_f: int, out_f: int, bias: bool) -> nn.Linear:
    lin = nn.Linear(in_f, out_f, bias=bias)
    lin.weight.data = torch.empty(out_f, in_f, dtype=lin.weight.dtype)
    return lin


# --------------------------------------------------------------------------- #
@torch.no_grad()
def prune_mlp_width(model, keep_ratio: float = 0.75, scoring: str = "gate_up_norm") -> Dict:
    """Uniformly shrink every decoder layer's MLP intermediate dimension."""
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")

    layers = model.model.layers
    old_i = model.config.intermediate_size
    new_i = max(1, int(round(old_i * keep_ratio)))
    if new_i == old_i:
        log.warning("keep_ratio=%s results in no pruning", keep_ratio)
        return {"old_intermediate": old_i, "new_intermediate": new_i, "removed_channels": 0}

    log.info("pruning MLP width %d -> %d per layer (%d layers)", old_i, new_i, len(layers))
    for idx, layer in enumerate(layers):
        mlp = layer.mlp
        scores = _row_scores(mlp.gate_proj.weight, mlp.up_proj.weight, scoring)
        keep = torch.topk(scores, new_i, largest=True).indices.sort().values.to(mlp.gate_proj.weight.device)
        keep = keep.to(torch.long)

        g_bias = mlp.gate_proj.bias is not None
        u_bias = mlp.up_proj.bias is not None
        d_bias = mlp.down_proj.bias is not None

        new_gate = _linear(mlp.gate_proj.in_features, new_i, g_bias)
        new_up = _linear(mlp.up_proj.in_features, new_i, u_bias)
        new_down = _linear(new_i, mlp.down_proj.out_features, d_bias)

        new_gate.weight.data = mlp.gate_proj.weight.data[keep].clone()
        new_up.weight.data = mlp.up_proj.weight.data[keep].clone()
        new_down.weight.data = mlp.down_proj.weight.data[:, keep].clone()
        if g_bias:
            new_gate.bias.data = mlp.gate_proj.bias.data[keep].clone()
        if u_bias:
            new_up.bias.data = mlp.up_proj.bias.data[keep].clone()
        if d_bias:
            new_down.bias.data = mlp.down_proj.bias.data.clone()

        device = mlp.gate_proj.weight.device
        layer.mlp.gate_proj = new_gate.to(device)
        layer.mlp.up_proj = new_up.to(device)
        layer.mlp.down_proj = new_down.to(device)

        if idx == 0:
            log.info("layer0 MLP %s -> (%d, %d)", tuple(mlp.gate_proj.weight.shape),
                     new_i, mlp.gate_proj.in_features)

    model.config.intermediate_size = new_i
    report = {
        "old_intermediate": old_i,
        "new_intermediate": new_i,
        "keep_ratio": keep_ratio,
        "scoring": scoring,
        "removed_channels": (old_i - new_i) * len(layers),
    }
    log.info("pruning done: removed %d channels total", report["removed_channels"])
    return report


# --------------------------------------------------------------------------- #
def heal_with_lora(model, tokenizer, steps: int = 300, lr: float = 2e-4,
                   seq_len: int = 512, batch_size: int = 1, device: str = "cuda"):
    """Short LoRA recovery pass on WikiText-2 after pruning.

    Optional. Costs ~10-20 min on a 3050 for 300 steps, usually recovers a
    good part of the accuracy lost to pruning.

    Returns ``(model, report)``; ``model`` has the adapters merged back in.
    """
    if steps <= 0:
        return model, None
    try:
        from peft import LoraConfig, get_peft_model
        from datasets import load_dataset
    except ImportError:  # pragma: no cover
        log.warning("peft/datasets missing; skipping healing")
        return model, None

    log.info("healing pruned model with LoRA for %d steps", steps)
    cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    chunks = ids.unfold(0, seq_len, seq_len).contiguous()[: max(steps * batch_size, 1)]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    model.to(device)
    losses = []
    for step in range(steps):
        batch = chunks[torch.randint(0, chunks.shape[0], (batch_size,))].to(device)
        out = model(input_ids=batch, labels=batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(out.loss.item())
        if (step + 1) % 50 == 0:
            log.info("  heal step %d/%d loss=%.4f", step + 1, steps, sum(losses[-50:]) / 50)

    # merge LoRA back into the base weights so the saved model is plain
    model = model.merge_and_unload()
    model.eval()
    return model, {"heal_steps": steps,
                   "final_loss": round(sum(losses[-50:]) / max(len(losses[-50:]), 1), 4)}


# --------------------------------------------------------------------------- #
def load_base(model_id: str, dtype=torch.float16, device: str = "cuda", token=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading %s (%s) ...", model_id, dtype)
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True, token=token,
    )
    model.to(device).eval()
    return model, tok


def save_model(model, tok, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    log.info("saved -> %s", out_dir)
    return out_dir
