#!/usr/bin/env python3
"""Checks scripts/qwen3next-ref.py against transformers' own Qwen3NextForCausalLM (needs torch and transformers >= 4.57,
which are NOT needed by anything else; e.g. a venv with `pip install torch --index-url
https://download.pytorch.org/whl/cpu transformers`).

usage: qwen3next-hf-compare.py <tiny-model-dir>   (after: python3 scripts/qwen3next-ref.py <dir>, without REF_Q8)
Builds the model from config.json in float64, loads model.safetensors (per-expert tensors are packed into the fused
experts.gate_up_proj / down_proj parameters when the installed transformers uses them; mtp.* is skipped), runs the
ArchSmoke prompt (no cache: the chunked delta rule) and prints max|transformers - reference| over the last-token
logits; then generates 8 tokens greedily with the cache (single-token steps: the recurrent delta rule, the conv
update and the KV cache) and compares them with the reference's greedy tokens.
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

PROMPT = [1, 7, 42, 99, 13, 5, 250, 8, 77, 31, 600, 9]


def main():
    d = sys.argv[1]
    torch.set_default_dtype(torch.float64)
    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)
    config = Qwen3NextConfig(**{k: v for k, v in cfg.items() if k not in ("architectures", "torch_dtype")})
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"  # the grouped-matmul experts take no float64
    model = Qwen3NextForCausalLM(config).double().eval()
    sd = {k: v.double() for k, v in load_file(os.path.join(d, "model.safetensors")).items() if not k.startswith("mtp.")}
    own = model.state_dict()
    for l in range(config.num_hidden_layers):
        p = "model.layers.%d.mlp.experts." % l
        if p + "gate_up_proj" in own and p + "0.gate_proj.weight" in sd:
            e = config.num_experts
            sd[p + "gate_up_proj"] = torch.stack([torch.cat([sd.pop(p + "%d.gate_proj.weight" % i),
                                                              sd.pop(p + "%d.up_proj.weight" % i)], 0) for i in range(e)])
            sd[p + "down_proj"] = torch.stack([sd.pop(p + "%d.down_proj.weight" % i) for i in range(e)])
    if config.tie_word_embeddings:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        sys.exit("weights do not fit: missing %s unexpected %s" % (missing, unexpected))
    with open(os.path.join(d, "ref-logits.json")) as f:
        ref = json.load(f)
    if ref.get("q8"):
        sys.exit("ref-logits.json was written with REF_Q8=1; run qwen3next-ref.py without it")
    ids = torch.tensor([PROMPT])
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits[0, -1].numpy()
        gen = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True)[0, len(PROMPT):].tolist()
    r = np.array(ref["logits"])
    print("%s: max|transformers - reference| = %.3e (max|logit| %.3f); greedy transformers %s, reference %s -> %s" % (
        d, float(np.abs(logits - r).max()), float(np.abs(r).max()), gen, ref["greedy"],
        "same" if gen == ref["greedy"] else "DIFFERENT"))


if __name__ == "__main__":
    main()
