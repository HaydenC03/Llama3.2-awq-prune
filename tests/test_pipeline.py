"""Fast tests. None of these need a GPU or a model download.

    pytest -q                      # from the repo root (or inside the container)

The torch/transformers tests skip themselves when the stack is not installed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src import synthetic_metrics
from src.common import RESULTS, load_json
from src.extended_bench import (EXTENDED_TASKS, build_rows, build_summary_rows,
                                write_csvs)
from src.tasks import (TASKS, _arc_easy, _boolq, _gsm8k, _hellaswag, _ms_marco,
                       _piqa, exact_match, extract_final_number, normalize_answer,
                       token_f1)


# --------------------------------------------------------------------------- #
# task adapters
# --------------------------------------------------------------------------- #
def test_arc_adapter():
    row = {"question": " What is 2+2? ", "choices": {"label": ["A", "B", "C", "D"],
                                                    "text": ["3", "4", "5", "6"]},
           "answerKey": "B"}
    ex = _arc_easy(row)
    assert ex["label"] == 1
    assert ex["choices"][1].strip() == "4"
    assert ex["prompt"].startswith("Question:")


def test_hellaswag_adapter_capitalises():
    row = {"ctx_a": "a man is holding", "ctx_b": "a ball", "endings": ["x", "y"],
           "label": "1"}
    ex = _hellaswag(row)
    assert ex["label"] == 1
    assert ex["prompt"][0].isupper()
    assert ex["choices"] == [" x", " y"]


def test_piqa_adapter():
    row = {"goal": "cut wood", "sol1": "use a saw", "sol2": "use a spoon", "label": 0}
    ex = _piqa(row)
    assert ex["label"] == 0 and len(ex["choices"]) == 2


# --------------------------------------------------------------------------- #
# extended benchmark adapters / metrics
# --------------------------------------------------------------------------- #
def test_boolq_adapter():
    ex = _boolq({"passage": "The sky is blue.", "question": "Is the sky blue?",
                 "answer": True})
    assert ex["label"] == 1
    assert ex["choices"] == [" no", " yes"]
    assert "Passage:" in ex["prompt"] and "Question:" in ex["prompt"]


def test_gsm8k_adapter_and_number_extraction():
    ex = _gsm8k({"question": "What is 2+2?", "answer": "2+2=4\n#### 4"})
    assert ex["answers"] == ["4"]
    assert extract_final_number("so #### 1,234.5") == 1234.5
    assert extract_final_number("the answer is 42") == 42.0
    assert extract_final_number("no number here") is None


def test_ms_marco_adapter_picks_selected_passage():
    row = {
        "query": "what is x?",
        "passages": {"passage_text": ["wrong passage", "right passage"],
                     "is_selected": [0, 1]},
        "answers": ["right answer"],
    }
    ex = _ms_marco(row)
    assert "right passage" in ex["prompt"]
    assert "wrong passage" not in ex["prompt"]
    assert ex["answers"] == ["right answer"]


def test_ms_marco_metrics():
    assert normalize_answer("The Cat!") == "cat"
    assert exact_match("The cat.", "cat") == 1.0
    assert token_f1("cat sat down", "cat sat") == pytest.approx(0.8)
    assert token_f1("dog", "cat") == 0.0


def test_extended_tasks_registered():
    assert EXTENDED_TASKS == ["gsm8k", "ms_marco", "boolq"]
    for task in EXTENDED_TASKS:
        assert task in TASKS


def test_evaluate_generation_task_scoring(monkeypatch):
    from src import tasks

    monkeypatch.setattr(tasks, "load_examples",
                        lambda task, limit: [{"prompt": "p", "answers": ["42"]}])
    monkeypatch.setattr(tasks, "generate_text", lambda *a, **k: "work #### 42")
    gsm = tasks.evaluate_generation_task(None, None, "gsm8k", limit=1, progress=False)
    assert gsm["score"] == 1.0 and gsm["metric"] == "acc"

    monkeypatch.setattr(tasks, "load_examples",
                        lambda task, limit: [{"prompt": "p", "answers": ["the cat sat"]}])
    monkeypatch.setattr(tasks, "generate_text", lambda *a, **k: "cat sat")
    marco = tasks.evaluate_generation_task(None, None, "ms_marco", limit=1, progress=False)
    assert marco["f1"] == 1.0 and marco["metric"] == "f1"


# --------------------------------------------------------------------------- #
# extended CSV comparison
# --------------------------------------------------------------------------- #
def _fake_extended_records():
    def rec(variant, display, weight, load, peak, scores):
        tasks = ["gsm8k", "ms_marco", "boolq"]
        metrics = {"gsm8k": "acc", "ms_marco": "f1", "boolq": "acc"}
        return {
            "variant": variant, "display_name": display,
            "weight_gb": weight, "load_vram_gb": load, "peak_vram_gb": peak,
            "results": {t: {"score": s, "metric": metrics[t], "n": 100}
                        for t, s in zip(tasks, scores)},
        }

    return {
        "baseline": rec("baseline", "Original (FP16)", 2.30, 2.00, 2.80,
                        (0.10, 0.40, 0.70)),
        "awq": rec("awq", "AWQ W4A16", 0.96, 1.00, 1.47,
                   (0.08, 0.38, 0.69)),
    }


def test_extended_csv_rows():
    records = _fake_extended_records()
    tasks = ["gsm8k", "ms_marco", "boolq"]
    rows = build_rows(records, tasks)
    assert len(rows) == 6

    awq_gsm = next(r for r in rows if r["variant"] == "awq" and r["task"] == "gsm8k")
    assert awq_gsm["baseline_score"] == 0.10
    assert awq_gsm["variant_score"] == 0.08
    assert awq_gsm["score_delta"] == pytest.approx(-0.02)
    assert awq_gsm["weight_reduction_pct"] == pytest.approx(
        (2.30 - 0.96) / 2.30 * 100, rel=1e-3)
    assert awq_gsm["vram_reduction_pct"] == pytest.approx(
        (2.80 - 1.47) / 2.80 * 100, rel=1e-3)

    summary = build_summary_rows(records, tasks)
    assert len(summary) == 2
    awq_summary = next(r for r in summary if r["variant"] == "awq")
    assert awq_summary["avg_score"] == pytest.approx((0.08 + 0.38 + 0.69) / 3, rel=1e-3)
    assert awq_summary["weight_reduction_pct"] == pytest.approx(
        (2.30 - 0.96) / 2.30 * 100, rel=1e-3)


def test_extended_write_csvs(tmp_path):
    out_dir = tmp_path / "extended"
    out_dir.mkdir()
    for name, rec in _fake_extended_records().items():
        (out_dir / f"{name}.json").write_text(json.dumps(rec), encoding="utf-8")

    scores, summary = write_csvs(
        ["gsm8k", "ms_marco", "boolq"], out_dir=out_dir,
        scores_csv=tmp_path / "scores.csv",
        summary_csv=tmp_path / "summary.csv",
    )
    assert scores.exists() and summary.exists()
    text = scores.read_text(encoding="utf-8-sig")
    assert "weight_reduction_pct" in text and "vram_reduction_pct" in text
    assert "MS MARCO" in text


# --------------------------------------------------------------------------- #
# bit plan
# --------------------------------------------------------------------------- #
class _FakeWeight:
    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n


def test_bit_plan_average_excludes_lm_head():
    pytest.importorskip("torch")
    import torch.nn as nn
    from src.quantize_awq import bit_plan_by_block, build_bit_plan

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(8, 8, bias=False)])
            self.lm_head = nn.Linear(8, 8, bias=False)

    plan = build_bit_plan(M(), ["lm_head"], w_bit=4, protect_bits=8)
    # body linear must be 4-bit, so the average over the body is exactly 4
    assert plan["avg_bits"] == 4.0
    assert plan["modules"]["lm_head"]["bits"] == 8
    assert not plan["modules"]["lm_head"]["in_avg"]

    blocks = bit_plan_by_block(plan)
    assert "block 00" in blocks and "embeddings/head" in blocks


def test_bit_plan_protect_pattern():
    pytest.importorskip("torch")
    import torch.nn as nn
    from src.quantize_awq import build_bit_plan

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(8, 8, bias=False)
            self.mlp = nn.Module()
            self.mlp.down_proj = nn.Linear(8, 8, bias=False)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])

    plan = build_bit_plan(M(), ["layers.0."], w_bit=4, protect_bits=8)
    assert plan["avg_bits"] == 6.0
    assert plan["modules"]["layers.0.self_attn.q_proj"]["protected"]
    assert not plan["modules"]["layers.1.mlp.down_proj"]["protected"]


# --------------------------------------------------------------------------- #
# preview metrics -> figures -> report
# --------------------------------------------------------------------------- #
def test_preview_metrics_are_consistent():
    m = synthetic_metrics.build_preview()
    assert m["meta"]["synthetic"] is True
    sizes = {k: v["size_bytes"] for k, v in m["variants"].items()}
    base = sizes["baseline"]
    for k, v in m["variants"].items():
        assert v["compression_ratio"] == pytest.approx(base / sizes[k], rel=1e-3)
    assert 0 < m["variants"]["awq"]["accuracy_avg"] <= 1


def test_plots_and_report(tmp_path):
    pytest.importorskip("matplotlib")
    from src import make_plots
    from src.make_report import write_report

    m = synthetic_metrics.build_preview()
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(m), encoding="utf-8")

    # redirect output dir
    make_plots.PLOTS = tmp_path / "plots"
    make_plots.PLOTS.mkdir()
    make_plots.main(metrics_path)

    pngs = sorted(p.name for p in make_plots.PLOTS.glob("*.png"))
    assert "01_accuracy_vs_compression.png" in pngs
    assert "04_complexity.png" in pngs
    assert all(p.stat().st_size > 10_000 for p in make_plots.PLOTS.glob("*.png"))

    report = write_report(metrics_path, tmp_path / "REPORT.md")
    text = report.read_text(encoding="utf-8")
    assert "Compression" in text and "ARC-Easy" in text


# --------------------------------------------------------------------------- #
# model-level tests (skip without torch/transformers)
# --------------------------------------------------------------------------- #
def _tiny_llama():
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    cfg = tf.LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256,
    )
    return tf.LlamaForCausalLM(cfg).eval()


def test_prune_mlp_shapes():
    torch = pytest.importorskip("torch")
    from src.prune_model import prune_mlp_width

    model = _tiny_llama()
    rep = prune_mlp_width(model, keep_ratio=0.5)
    assert rep["new_intermediate"] == 32
    layer = model.model.layers[0].mlp
    assert layer.gate_proj.out_features == 32
    assert layer.up_proj.out_features == 32
    assert layer.down_proj.in_features == 32
    assert model.config.intermediate_size == 32


class _DummyTok:
    """Character-level stub tokenizer: enough to exercise score_choices maths."""
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3 + (ord(c) % 61) for c in text]}


class _DecodeTok(_DummyTok):
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [3 + (ord(c) % 61) for c in text]
        if return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def test_generate_text_with_tiny_model():
    pytest.importorskip("torch")
    from src.tasks import generate_text

    model = _tiny_llama()
    tok = _DecodeTok()
    text = generate_text(model, tok, "abc", max_new_tokens=3, device="cpu")
    assert isinstance(text, str) and text


def test_score_choices_shapes():
    pytest.importorskip("torch")
    from src.tasks import score_choices

    model = _tiny_llama()
    tok = _DummyTok()
    scores = score_choices(model, tok, "abc", [" de", " fgh"], max_len=64, device="cpu")
    assert len(scores) == 2
    assert scores[0][0] <= 0 and scores[1][0] <= 0          # log-probs are <= 0
    assert scores[1][1] == len(" fgh")                      # token count of choice 2
    assert scores[1][2] == pytest.approx(scores[1][0] / scores[1][1])  # normalisation


def test_score_choices_truncates_prompt_not_choice():
    pytest.importorskip("torch")
    from src.tasks import score_choices

    model = _tiny_llama()
    tok = _DummyTok()
    scores = score_choices(model, tok, "x" * 100, [" abc"], max_len=16, device="cpu")
    assert scores[0][1] == len(" abc")                      # completion survived intact


def test_score_choices_rejects_impossible_budget():
    pytest.importorskip("torch")
    from src.tasks import score_choices

    model = _tiny_llama()
    tok = _DummyTok()
    with pytest.raises(ValueError):
        score_choices(model, tok, "abc", [" " + "y" * 50], max_len=8, device="cpu")


def test_score_choices_with_real_tokenizer():
    pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    if not os.getenv("RUN_NETWORK_TESTS"):
        pytest.skip("set RUN_NETWORK_TESTS=1 to download the test tokenizer")
    from src.tasks import score_choices

    model = _tiny_llama()
    tok = tf.LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    scores = score_choices(model, tok, "The capital of France is", [" Paris", " banana"],
                           max_len=64, device="cpu")
    assert len(scores) == 2
    assert scores[0][1] > 0 and scores[0][0] <= 0  # log-probabilities are non-positive
