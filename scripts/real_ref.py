#!/usr/bin/env python3
"""Hugging Face transformers reference for RealWeightsCheck (scripts/real-weights-check.sh).

usage: real_ref.py <model-dir> <out-dir>

Loads the checkpoint in <model-dir> (config.json, *.safetensors, tokenizer.json) with transformers in float32 on
the CPU and, for a fixed set of prompts (chat-templated with tokenizer.apply_chat_template when the tokenizer has
a chat template, and raw text with the tokenizer's own special tokens), runs 16 greedy steps with generate()
(no end-of-sequence stop, no logits processors; every chosen id is checked to be the argmax of the raw logits).
It writes:

  <out-dir>/ref.json            model type, per prompt: the rendered template text, the prompt ids, the 16 greedy
                                ids, the top-20 ids / logits of every step; plus tokenizer probes (text -> ids
                                without special tokens)
  <out-dir>/<prompt>.f32        the full logits of every step, little-endian float32, steps x vocab
  <out-dir>/<prompt>.q8.f32     the same steps (teacher-forced on the greedy ids above) with every weight that the
                                engine quantizes rounded to Q8_0 first: what the engine at q8 should reproduce up to
                                float rounding (skipped with REF_Q8=0)

Needs torch and transformers (CPU build is enough). Environment: REF_DTYPE=upcast keeps the weights in the
checkpoint's dtype (bf16) and upcasts each matrix to float32 when it is used, so the activations are float32 as with
the default float32 load while the model takes half the memory (for checkpoints whose float32 copy does not fit; the
dtype-dependent casts of the modeling code then see bf16 weights, as when running the checkpoint as published).
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

STEPS = 16
TOPK = 20

CHATS = {
    "chat_user": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
    "chat_system": [{"role": "system", "content": "You are a concise assistant that answers in plain English."},
                    {"role": "user", "content": "Write a haiku about the sea."}],
    "chat_multi": [{"role": "user", "content": "Hi! Can you help me with arithmetic?"},
                   {"role": "assistant", "content": "Of course. What would you like to compute?"},
                   {"role": "user", "content": "What is 17 * 23? Show the steps."}],
}
RAWS = {
    "raw_text": "The history of the Roman Empire began when",
    "raw_code": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
}
PROBES = [
    "Hello, world! The quick brown fox jumps over the lazy dog.",
    "  leading spaces,\ttabs\tand\n\nnewlines  ",
    "Numbers: 1234567890 3.14159 -42 1e10 2024-10-21",
    "Unicode: café naïve 中文 日本語 한국어 русский \U0001F600\U0001F680",
    "def f(x):\n    return x ** 2  # square\n",
    "<|im_end|> <s> </s> [INST] literal markers in text",
]


def ids_of(x):
    if isinstance(x, dict) or hasattr(x, "keys"):
        x = x["input_ids"]
    if hasattr(x, "tolist"):
        x = x.tolist()
    if x and isinstance(x[0], list):
        x = x[0]
    return [int(i) for i in x]


def main():
    model_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "4")))
    mode = os.environ.get("REF_DTYPE", "float32")
    dtype = "auto" if mode == "upcast" else getattr(torch, mode)
    tok = AutoTokenizer.from_pretrained(model_dir)
    kw = {"attn_implementation": "eager", "low_cpu_mem_usage": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, **kw)
    model.eval()
    if mode == "upcast":
        upcast(model)
    cfg = json.load(open(os.path.join(model_dir, "config.json")))

    prompts = []
    if getattr(tok, "chat_template", None):
        for name, msgs in CHATS.items():
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ids = ids_of(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
            except Exception as e:  # e.g. Gemma 2 rejects a system message
                prompts.append({"name": name, "kind": "chat", "messages": msgs, "error": str(e)})
                continue
            prompts.append({"name": name, "kind": "chat", "messages": msgs, "text": text, "ids": ids})
    for name, text in RAWS.items():
        prompts.append({"name": name, "kind": "raw", "text": text, "ids": ids_of(tok(text))})

    # replaces the checkpoint's generation_config.json (whose eos_token_id would stop early and whose sampling
    # settings would be merged in)
    model.generation_config = GenerationConfig(max_new_tokens=STEPS, min_new_tokens=0, do_sample=False, num_beams=1,
                                               eos_token_id=None, bos_token_id=None,
                                               pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else 0)
    for p in prompts:
        if "ids" not in p:
            continue
        with torch.no_grad():
            out = model.generate(torch.tensor([p["ids"]]),
                                 attention_mask=torch.ones(1, len(p["ids"]), dtype=torch.long),
                                 output_logits=True, return_dict_in_generate=True)
        gen = out.sequences[0, len(p["ids"]):].tolist()
        steps = []
        with open(os.path.join(out_dir, p["name"] + ".f32"), "wb") as f:
            for s, logits in enumerate(out.logits):
                logits = logits[0].float()
                f.write(logits.numpy().astype("<f4").tobytes())
                v, i = torch.topk(logits, TOPK)
                if int(i[0]) != gen[s]:
                    raise SystemExit(f"{p['name']} step {s}: generate chose {gen[s]}, the raw argmax is {int(i[0])}")
                steps.append({"top_ids": i.tolist(), "top_vals": [round(float(x), 5) for x in v]})
        if len(gen) != STEPS:
            raise SystemExit(f"{p['name']}: generate stopped after {len(gen)} tokens")
        p["gen"] = gen
        p["steps"] = steps
        p["vocab"] = int(logits.shape[0])
        print(p["name"], len(p["ids"]), "prompt ids ->", repr(tok.decode(p["gen"])), flush=True)

    probes = [{"text": t, "ids": ids_of(tok(t, add_special_tokens=False))} for t in PROBES]
    out = {"model_type": cfg.get("model_type"), "dtype": mode,
           "transformers": __import__("transformers").__version__, "prompts": prompts, "probes": probes}
    with open(os.path.join(out_dir, "ref.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    if os.environ.get("REF_Q8", "1") != "0":
        if mode == "upcast":
            Q8[0] = True
            for m in model.modules():  # the embedding's own forward reads the parameter: round it once, in f32
                if isinstance(m, torch.nn.Embedding) and m.weight.shape[-1] % 32 == 0:
                    m.weight = torch.nn.Parameter(q8(m.weight.data.float()), requires_grad=False)
        else:
            round_q8(model)
        for p in prompts:
            if "ids" not in p:
                continue
            seq = p["ids"] + p["gen"][:-1]
            with torch.no_grad():
                logits = model(torch.tensor([seq])).logits[0, len(p["ids"]) - 1:].float()
            with open(os.path.join(out_dir, p["name"] + ".q8.f32"), "wb") as f:
                f.write(logits.numpy().astype("<f4").tobytes())


Q8 = [False]


def q8(w):
    """w rounded to Q8_0 like the engine's converter: blocks of 32 along the last dimension, scale amax / 127."""
    b = w.float().reshape(-1, 32)
    d = b.abs().amax(dim=1, keepdim=True) / 127
    inv = torch.where(d > 0, 1 / d, torch.zeros_like(d))
    return (torch.round(b * inv) * d).reshape(w.shape)


def upcast(model):
    """REF_DTYPE=upcast: Linear weights stay in the checkpoint dtype and are upcast (and Q8_0-rounded in the second
    pass) on each use; embeddings return float32; every other parameter and buffer becomes float32."""
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            def lin(x, m=m):
                w = m.weight.float()
                if Q8[0] and w.shape[-1] % 32 == 0:
                    w = q8(w)
                return F.linear(x.float(), w, None if m.bias is None else m.bias.float())
            m.forward = lin
            if m.bias is not None:
                m.bias.data = m.bias.data.float()
        elif isinstance(m, torch.nn.Embedding):
            orig = m.forward
            m.forward = lambda ids, orig=orig: orig(ids).float()
        else:
            for p in m.parameters(recurse=False):
                p.data = p.data.float()
            for name, b in m.named_buffers(recurse=False):
                if b.is_floating_point():
                    setattr(m, name, b.float())


def round_q8(model):
    """Rounds every weight the engine's converter quantizes to Q8_0 (blocks of 32 along the input dimension, one
    f32 scale amax / 127 each): the weight of each Linear and Embedding whose input width is a multiple of 32
    (Mamba's dt_proj, 48 wide in mamba-130m, stays f32 in the engine too); norms, biases and convolutions are kept.
    The second reference then differs from the engine only by the activations' float rounding."""
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)) and m.weight.shape[-1] % 32 == 0:
            m.weight.data = q8(m.weight.data).to(m.weight.dtype)


if __name__ == "__main__":
    main()
