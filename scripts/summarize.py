"""Print a compact comparison table from results/metrics.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/metrics.json")
m = json.loads(path.read_text(encoding="utf-8"))
meta = m.get("meta", {})

print(f"model : {meta.get('model_id')}")
hw = meta.get("hardware", {})
print(f"gpu   : {hw.get('gpu')}  {hw.get('vram_gb')} GB  torch {hw.get('torch')}")
print(f"data  : {'SYNTHETIC PREVIEW' if meta.get('synthetic') else 'measured'}\n")

hdr = (f"{'variant':13}{'GiB':>7}{'ratio':>7}{'bits':>6}"
       f"{'ARC':>7}{'Hella':>7}{'PIQA':>7}{'avg':>7}{'PPL':>9}"
       f"{'tok/s':>8}{'VRAM':>7}{'GFLOP':>7}")
print(hdr)
print("-" * len(hdr))
for k, v in m["variants"].items():
    c = v.get("complexity") or {}
    a = v.get("accuracy") or {}
    print(f"{k:13}{v.get('size_gb', 0):7.2f}{v.get('compression_ratio', 0):7.2f}"
          f"{(v.get('avg_bits') or 16.0):6.2f}"
          f"{a.get('arc_easy', 0):7.3f}{a.get('hellaswag', 0):7.3f}{a.get('piqa', 0):7.3f}"
          f"{v.get('accuracy_avg', 0):7.3f}{v.get('ppl_wikitext2') or 0:9.2f}"
          f"{c.get('tokens_per_s', 0):8.1f}{c.get('peak_vram_gb', 0):7.2f}"
          f"{c.get('gflops_per_token', 0):7.2f}")

base = m["variants"].get("baseline")
if base:
    print("\ndelta vs baseline (accuracy points / perplexity):")
    for k, v in m["variants"].items():
        if k == "baseline":
            continue
        d = (v.get("accuracy_avg", 0) - base.get("accuracy_avg", 0)) * 100
        p = (v.get("ppl_wikitext2") or 0) - (base.get("ppl_wikitext2") or 0)
        print(f"  {k:13}{d:+7.1f} pt   ppl {p:+8.2f}")
