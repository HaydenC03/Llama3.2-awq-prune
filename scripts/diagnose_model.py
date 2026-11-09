"""Sanity check a checkpoint: short-window perplexity + a greedy sample.

    python scripts/diagnose_model.py artifacts/baseline/Llama-3.2-1B-Instruct
"""
from __future__ import annotations

import math
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

path = sys.argv[1]
n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 3072

tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype=torch.float16, device_map="auto"
).eval()

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(t for t in ds["text"] if t and t.strip())
ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0][:n_tokens]

total, n = 0.0, 0
with torch.no_grad():
    for s in range(0, len(ids) - 1024, 1024):
        chunk = ids[s: s + 1025].unsqueeze(0).cuda()
        loss = model(input_ids=chunk, labels=chunk).loss.item()
        total += loss * 1024
        n += 1024

print(f"{path}")
print(f"  ppl@{n} tokens : {math.exp(total / n):.3f}")

prompt = "The capital of France is"
enc = tok(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=30, do_sample=False,
                         pad_token_id=tok.eos_token_id)
print("  sample        :", tok.decode(out[0], skip_special_tokens=True).replace("\n", " "))
