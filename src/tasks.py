"""Three standard downstream tasks + WikiText-2 perplexity.

Why these three (the "common test tasks" asked for):

1. ``arc_easy``  - ARC-Easy: grade-school science, 4-way multiple choice.
                   Proxy for *reasoning / knowledge*.
2. ``hellaswag`` - HellaSwag: commonsense sentence completion, 4-way.
                   Proxy for *commonsense*; very sensitive to quantisation
                   damage, which makes it a good canary.
3. ``piqa``      - PIQA: physical commonsense, 2-way.
                   Proxy for *physical world knowledge*.

All three are scored the same way lm-evaluation-harness does it: teacher
forcing, pick the choice with the highest (length-normalised) log-likelihood.
No generation, so results are deterministic and fast.

``ppl_wikitext2`` is added as a cheap intrinsic signal: perplexity blows up
long before multiple-choice accuracy moves, so it is the most sensitive
indicator of quantisation / pruning damage.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

log = logging.getLogger("tasks")


# --------------------------------------------------------------------------- #
# Task registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaskSpec:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    description: str
    trust_remote_code: bool = False
    # "mc"  = teacher-forced multiple choice (one forward pass per example)
    # "gen" = greedy generation, scored by exact match / F1
    kind: str = "mc"
    gen_max_new_tokens: int = 256


TASKS: Dict[str, TaskSpec] = {
    "arc_easy": TaskSpec(
        "arc_easy", "allenai/ai2_arc", "ARC-Easy", "test",
        "Grade-school science multiple choice (reasoning/knowledge)",
    ),
    "hellaswag": TaskSpec(
        "hellaswag", "Rowan/hellaswag", None, "validation",
        "Commonsense sentence completion (commonsense reasoning)",
    ),
    "piqa": TaskSpec(
        "piqa", "ybisk/piqa", "plain_text", "validation",
        "Physical commonsense, 2-way (world knowledge)",
        # ybisk/piqa still ships a legacy loading script
        trust_remote_code=True,
    ),
    "boolq": TaskSpec(
        "boolq", "google/boolq", None, "validation",
        "Yes/no reading comprehension (passage + question)",
        trust_remote_code=True,
    ),
    "gsm8k": TaskSpec(
        "gsm8k", "openai/gsm8k", "main", "test",
        "Grade-school math word problems, scored by numeric exact match",
        kind="gen", gen_max_new_tokens=256,
    ),
    "ms_marco": TaskSpec(
        "ms_marco", "microsoft/ms_marco", "v2.1", "validation",
        "MS MARCO QA, answered from the selected passage (EM / token F1)",
        trust_remote_code=True, kind="gen", gen_max_new_tokens=64,
    ),
}

DEFAULT_TASKS = ["arc_easy", "hellaswag", "piqa"]
EXTENDED_TASKS = ["gsm8k", "ms_marco", "boolq"]


# --------------------------------------------------------------------------- #
# Dataset -> (prompt, choices, label)
# --------------------------------------------------------------------------- #
def _arc_easy(row) -> dict:
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    key = str(row["answerKey"]).strip()
    try:
        gold = labels.index(row["answerKey"])
    except ValueError:
        gold = int(key) - 1 if key.isdigit() else ["A", "B", "C", "D"].index(key)
    return {
        "prompt": f"Question: {row['question'].strip()}\nAnswer:",
        "choices": [f" {t.strip()}" for t in texts],
        "label": gold,
    }


def _hellaswag(row) -> dict:
    ctx = (row["ctx_a"] + " " + row["ctx_b"]).strip()
    if ctx and ctx[0].islower():
        ctx = ctx[0].upper() + ctx[1:]
    return {
        "prompt": ctx,
        "choices": [f" {e.strip()}" for e in row["endings"]],
        "label": int(row["label"]),
    }


def _piqa(row) -> dict:
    return {
        "prompt": f"Question: {row['goal'].strip()}\nAnswer:",
        "choices": [f" {row['sol1'].strip()}", f" {row['sol2'].strip()}"],
        "label": int(row["label"]),
    }


def _boolq(row) -> dict:
    answer = row["answer"]
    if isinstance(answer, str):
        answer = answer.strip().lower() in ("true", "1", "yes")
    return {
        "prompt": (f"Passage: {row['passage'].strip()}\n"
                   f"Question: {row['question'].strip()}\nAnswer:"),
        "choices": [" no", " yes"],
        "label": 1 if answer else 0,
    }


_GSM8K_FEWSHOT = """Q: There are 15 trees in the grove. 8 are cut down. How many remain?
A: 15 - 8 = 7. #### 7

Q: A shop has 12 apples and sells 5. How many are left?
A: 12 - 5 = 7. #### 7

Q: Tom has 3 boxes with 4 pencils each. How many pencils does he have?
A: 3 * 4 = 12. #### 12

Q: A train travels 60 km/h for 2 hours. How far does it go?
A: 60 * 2 = 120. #### 120
"""


def _gsm8k(row) -> dict:
    raw = str(row["answer"])
    gold = raw.split("####")[-1].strip() if "####" in raw else raw.strip()
    return {
        "prompt": (f"{_GSM8K_FEWSHOT}\nQ: {row['question'].strip()}\nA:"),
        "answers": [gold],
    }


def _ms_marco(row) -> dict:
    passages = row.get("passages")
    if passages is None:
        passages = {}
    if isinstance(passages, dict):
        texts = passages.get("passage_text")
        texts = list(texts) if texts is not None else []
        selected = passages.get("is_selected")
        selected = list(selected) if selected is not None else []
    else:  # older/newer dataset layouts expose a list of dicts
        texts = [str(p.get("passage_text", "")) for p in passages]
        selected = [int(p.get("is_selected", 0)) for p in passages]

    pick = 0
    for i, flag in enumerate(selected):
        if int(flag) == 1:
            pick = i
            break
    context = str(texts[pick]).strip() if texts else ""

    answers = row.get("answers")
    if answers is None:
        answers = []
    elif isinstance(answers, str):
        answers = [answers]
    else:
        answers = list(answers)
    answers = [str(a).strip() for a in answers if str(a).strip()]

    # A passage is required for this reading-comprehension formulation. Empty
    # answers are removed by load_examples for generation tasks.
    if not context:
        answers = []

    return {
        "prompt": (f"Passage: {context}\n"
                   f"Question: {str(row['query']).strip()}\n"
                   "Answer with a short phrase:"),
        "answers": answers,
    }


_BUILDERS = {
    "arc_easy": _arc_easy,
    "hellaswag": _hellaswag,
    "piqa": _piqa,
    "boolq": _boolq,
    "gsm8k": _gsm8k,
    "ms_marco": _ms_marco,
}


def _has_reference(ex: dict) -> bool:
    """Generation tasks carry an ``answers`` list; MC tasks do not."""
    return ex.get("answers") is None or bool(ex["answers"])


def load_examples(task: str, limit: int | None, seed: int = 42) -> List[dict]:
    """Deterministic subset of a task: shuffle(seed) then take ``limit``.

    Multiple-choice tasks keep the original, sorted random subset so existing
    results remain reproducible. Generation tasks (GSM8K / MS MARCO) reject
    rows without a reference answer; for those we draw a slightly larger pool
    and then keep the first ``limit`` valid examples.
    """
    from datasets import load_dataset

    spec = TASKS[task]
    ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.split,
                      trust_remote_code=spec.trust_remote_code)
    n = len(ds)
    builder = _BUILDERS[task]

    if limit and limit < n:
        g = torch.Generator().manual_seed(seed)
        pool_size = min(n, limit if spec.kind == "mc" else limit * 3)
        idx = torch.randperm(n, generator=g)[:pool_size].tolist()
        idx.sort()
    else:
        idx = list(range(n))

    examples = [builder(ds[i]) for i in idx]
    examples = [ex for ex in examples if _has_reference(ex)]
    if limit:
        examples = examples[:limit]
    return examples


# --------------------------------------------------------------------------- #
# Likelihood scoring
# --------------------------------------------------------------------------- #
def _encode(tok, text: str) -> List[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


@torch.inference_mode()
def score_choices(model, tokenizer, prompt: str, choices: Sequence[str],
                  max_len: int = 1024, device: str | torch.device = "cuda"):
    """Return (sum_logprob, n_tokens, mean_logprob) for every choice.

    All choices of one example are scored in a single forward pass by
    right-padding: right padding cannot change a token's own logits, and the
    attention mask hides it, so this is exact.
    """
    prompt_ids = _encode(tokenizer, prompt)
    choice_ids = [_encode(tokenizer, c) or [tokenizer.eos_token_id] for c in choices]

    # keep the longest choice intact, trim the prompt from the left if needed
    budget = max_len - max(len(c) for c in choice_ids)
    if budget < 1:
        raise ValueError(f"max_len={max_len} too small for choices of len "
                         f"{max(len(c) for c in choice_ids)}")
    if len(prompt_ids) > budget:
        prompt_ids = prompt_ids[-budget:]
    if tokenizer.bos_token_id is not None and (not prompt_ids or prompt_ids[0] != tokenizer.bos_token_id):
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids

    p = len(prompt_ids)
    rows = [prompt_ids + c for c in choice_ids]
    lens = [len(r) for r in rows]
    width = max(lens)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
    attn = torch.zeros_like(input_ids)
    for i, ln in enumerate(lens):
        attn[i, :ln] = 1

    input_ids = input_ids.to(device)
    attn = attn.to(device)

    logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, W, V]

    out = []
    for i, c in enumerate(choice_ids):
        n = len(c)
        # only the n positions that predict this completion, so the
        # [B, W, 128k] log-softmax never has to be materialised at full width
        seg = logits[i, p - 1: p - 1 + n].float()
        lp = torch.log_softmax(seg, dim=-1)
        target = torch.tensor(rows[i][p: p + n], dtype=torch.long, device=device)
        total = lp.gather(1, target.unsqueeze(1)).sum().item()
        out.append((total, n, total / n))
    return out


@torch.inference_mode()
def evaluate_task(model, tokenizer, task: str, limit: int | None = 200,
                  max_len: int = 1024, max_new_tokens: int | None = None,
                  device="cuda", progress: bool = True) -> dict:
    """Dispatch to multiple-choice or generative scoring based on the task."""
    spec = TASKS.get(task)
    if spec is None:
        raise ValueError(f"unknown task {task!r}; known: {list(TASKS)}")
    if spec.kind == "gen":
        return evaluate_generation_task(
            model, tokenizer, task, limit=limit, max_len=max_len,
            max_new_tokens=max_new_tokens, device=device, progress=progress,
        )
    return _evaluate_mc(model, tokenizer, task, limit=limit, max_len=max_len,
                        device=device, progress=progress)


@torch.inference_mode()
def _evaluate_mc(model, tokenizer, task: str, limit: int | None = 200,
                 max_len: int = 1024, device="cuda", progress: bool = True) -> dict:
    """Multiple-choice accuracy (raw + length-normalised)."""
    examples = load_examples(task, limit)
    correct = correct_norm = 0
    nll_gold = 0.0
    n_tok = 0

    iterator = examples
    if progress:
        from tqdm import tqdm
        iterator = tqdm(examples, desc=f"  {task}", leave=False)

    for ex in iterator:
        scores = score_choices(model, tokenizer, ex["prompt"], ex["choices"], max_len, device)
        raw = [s[0] for s in scores]
        norm = [s[2] for s in scores]
        if int(torch.tensor(raw).argmax()) == ex["label"]:
            correct += 1
        if int(torch.tensor(norm).argmax()) == ex["label"]:
            correct_norm += 1
        nll_gold += -scores[ex["label"]][0]
        n_tok += 1

    n = max(len(examples), 1)
    result = {
        "n": len(examples),
        "acc": round(correct / n, 4),
        "acc_norm": round(correct_norm / n, 4),
        "nll": round(nll_gold / n, 4),
        "score": round(correct / n, 4),
        "metric": "acc",
    }
    log.info("%-10s acc=%.4f acc_norm=%.4f (n=%d)", task, result["acc"], result["acc_norm"], n)
    return result


# --------------------------------------------------------------------------- #
# Generative scoring (GSM8K / MS MARCO)
# --------------------------------------------------------------------------- #
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_ARTICLES = {"a", "an", "the"}
_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _to_float(text: str | None) -> float | None:
    if text is None:
        return None
    cleaned = str(text).strip().replace(",", "").rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_final_number(text: str) -> float | None:
    """GSM8K answer extraction: prefer ``#### N``, else the last number."""
    if not text:
        return None
    marked = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if marked:
        return _to_float(marked.group(1))
    matches = _NUMBER_RE.findall(text)
    return _to_float(matches[-1]) if matches else None


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation used by the MS MARCO EM / F1 metrics."""
    lowered = "".join(ch.lower() for ch in str(text) if ch not in _PUNCT)
    return " ".join(tok for tok in lowered.split() if tok not in _ARTICLES)


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    from collections import Counter

    pred = normalize_answer(prediction).split()
    ref = normalize_answer(gold).split()
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 256,
                  device="cuda") -> str:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.inference_mode()
def evaluate_generation_task(model, tokenizer, task: str, limit: int | None = 100,
                             max_len: int = 1024, max_new_tokens: int | None = None,
                             device="cuda", progress: bool = True) -> dict:
    """Greedy-generation evaluation.

    * gsm8k    : numeric exact match on the final number (``#### N``).
    * ms_marco : SQuAD-style exact match and token F1 against the references;
                 the reported primary ``score`` is F1.
    """
    spec = TASKS[task]
    examples = load_examples(task, limit)
    max_new = int(max_new_tokens or spec.gen_max_new_tokens)
    # MS MARCO answers are short; do not let a global GSM8K budget make the
    # model ramble (which would only depress token F1).
    if task == "ms_marco":
        max_new = min(max_new, spec.gen_max_new_tokens)

    iterator = examples
    if progress:
        from tqdm import tqdm
        iterator = tqdm(examples, desc=f"  {task}", leave=False)

    samples = []
    gsm_correct = 0
    ems: List[float] = []
    f1s: List[float] = []

    for ex in iterator:
        text = generate_text(model, tokenizer, ex["prompt"], max_new, device)
        if task == "gsm8k":
            pred = extract_final_number(text)
            gold = extract_final_number(ex["answers"][0])
            ok = pred is not None and gold is not None and abs(pred - gold) < 1e-6
            gsm_correct += int(ok)
            if len(samples) < 5:
                samples.append({"prompt": ex["prompt"][:400], "prediction": text[:400],
                                "reference": ex["answers"][0], "correct": bool(ok)})
        else:  # ms_marco
            best_em = max((exact_match(text, g) for g in ex["answers"]), default=0.0)
            best_f1 = max((token_f1(text, g) for g in ex["answers"]), default=0.0)
            ems.append(best_em)
            f1s.append(best_f1)
            if len(samples) < 5:
                samples.append({"prompt": ex["prompt"][:400], "prediction": text[:400],
                                "reference": ex["answers"][0], "em": best_em, "f1": best_f1})

    n = max(len(examples), 1)
    if task == "gsm8k":
        score = gsm_correct / n
        result = {
            "n": len(examples),
            "acc": round(score, 4),
            "score": round(score, 4),
            "metric": "acc",
            "max_new_tokens": max_new,
            "samples": samples,
        }
    else:
        em = sum(ems) / n
        f1 = sum(f1s) / n
        result = {
            "n": len(examples),
            "acc": round(f1, 4),          # keep the generic field usable
            "em": round(em, 4),
            "f1": round(f1, 4),
            "score": round(f1, 4),
            "metric": "f1",
            "max_new_tokens": max_new,
            "samples": samples,
        }
    log.info("%-10s %s=%.4f (n=%d)", task, result["metric"], result["score"], n)
    return result


# --------------------------------------------------------------------------- #
# WikiText-2 perplexity (sliding window)
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def evaluate_ppl(model, tokenizer, dataset: str = "wikitext2",
                 seq_len: int = 1024, stride: int = 512,
                 max_tokens: int = 200_000, device="cuda") -> dict:
    from datasets import load_dataset

    if dataset not in ("wikitext2", "wikitext-2-raw-v1"):
        raise ValueError(f"unsupported ppl dataset {dataset!r}")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    ids = ids[:max_tokens]
    if ids.numel() < seq_len + 1:
        raise ValueError("not enough tokens for perplexity")

    total_nll, total_tokens = 0.0, 0
    windows = range(0, ids.numel() - seq_len, stride)
    from tqdm import tqdm
    for start in tqdm(windows, desc="  wikitext2 ppl", leave=False):
        chunk = ids[start: start + seq_len + 1].unsqueeze(0).to(device)
        out = model(input_ids=chunk, labels=chunk)
        # count only the newly-scored part of the window
        overlap = 0 if start == 0 else seq_len - stride
        n_scored = seq_len - overlap
        total_nll += out.loss.item() * n_scored
        total_tokens += n_scored

    ppl = math.exp(total_nll / total_tokens)
    log.info("wikitext2 ppl=%.3f over %d tokens", ppl, total_tokens)
    return {"ppl": round(ppl, 3), "tokens": total_tokens}
