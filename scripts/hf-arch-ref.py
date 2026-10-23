#!/usr/bin/env python3
"""Tiny random models of the long-context families, built with Hugging Face transformers, and the transformers
reference for them (scripts/hf-arch-check.sh drives it; the Java side is src/test/java/.../ArchRefCheck.java).

usage: hf-arch-ref.py <family> <out-dir> [--seed N]
  family: gptoss        GptOssForCausalLM: alternating sliding-window (4) / full attention, attention sinks, YaRN
                        (truncate false, original length 16), top-2 of 8 experts with biases and the clamped SwiGLU
                        (limit 3), the experts stored as bf16-style float tensors [E, in, out]
          gptoss-mxfp4  the same model with its experts stored as MXFP4 blocks / scales, as openai/gpt-oss-20b ships
                        them (the reference uses the dequantized MXFP4 values)
          llama4        Llama4ForCausalLM: chunked attention (chunk 8) on 3 of 4 layers, a NoPE layer with attention
                        temperature tuning (floor 8), L2 q/k norm, llama3-scaled interleaved RoPE, MoE (top 1 of 4,
                        sigmoid weight on the input, shared expert) on every layer
          llama4-mix    as llama4 with interleave_moe_layer_step 2: dense and MoE layers alternate
          granite       GraniteMoeHybridForCausalLM: 5 layers mamba / mamba / attention / mamba / attention, Mamba-2 with 8
                        heads of 32 and 2 groups (the gated norm covers the whole width), 4 experts top 2 plus the shared
                        MLP in every layer, the muP multipliers (embedding 2, residual 0.5, attention 0.2, logits / 4), no RoPE
          granite-rope  a dense Granite (no experts, only the shared MLP) whose attention layers rotate (RoPE)
          nemotronh     NemotronHForCausalLM: pattern "M-M*-M" (Mamba-2, ReLU-squared MLP, attention), 2 groups with a
                        group-wise gated norm, the step size clamped below at time_step_min
          jamba         JambaForCausalLM: Mamba-1 with the dt / B / C norms in 3 of 4 layers, attention in layer 2, a MoE
                        (4 experts top 2, softmax then top-k) in every other layer, a SwiGLU MLP in the rest

Writes <out-dir>/config.json + model.safetensors (a Hugging Face checkpoint the engine converts like any other) and
<out-dir>/ref/ref.json (prompt ids, the 12 greedy ids) + ref/logits.f32 (little-endian float32, one row of vocab logits
for each position of prompt + greedy ids, computed with every weight the engine quantizes rounded to Q8_0 first, so
the engine at q8 has the same weights up to float rounding) + ref/logits.f32exact (the same with the weights as saved).
Needs torch and transformers (scratchpad refvenv: torch 2.14, transformers 5.20).
"""
import json
import os
import struct
import sys

import numpy as np
import torch

FAMILIES = ("gptoss", "gptoss-mxfp4", "llama4", "llama4-mix", "granite", "granite-rope", "nemotronh", "jamba")
VOCAB = 1000
PROMPT_LEN = 40
STEPS = 12
FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)


def q8_round(w, axis=-1):
    """Rounds w to Q8_0 as the engine does: blocks of 32 along `axis`, scale amax / 127, q = Math.round(x / scale) (halves round up)."""
    w = w.detach().float()
    wm = w.movedim(axis, -1).contiguous()
    shape = wm.shape
    b = wm.reshape(-1, shape[-1] // 32, 32)
    d = b.abs().amax(dim=-1, keepdim=True) / 127.0
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    q = torch.floor(b * inv + 0.5)  # Java's Math.round (halves round up), as QTensor.quantize does
    return (q * d).reshape(shape).movedim(-1, axis)


def mxfp4_quantize(w):
    """w [rows..., cols] (cols % 32 == 0) -> (blocks uint8 [rows..., cols/32, 16], scales uint8 [rows..., cols/32],
    dequantized float32 w). Scale 2^(e - 127) with the smallest e such that max|block| / scale <= 6."""
    a = w.detach().float().numpy()
    lead, cols = a.shape[:-1], a.shape[-1]
    b = a.reshape(-1, cols // 32, 32)
    amax = np.abs(b).max(axis=-1)
    with np.errstate(divide="ignore"):
        e = np.where(amax > 0, np.ceil(np.log2(np.maximum(amax, 1e-30) / 6.0)), 0).astype(np.int64) + 127
    e = np.clip(e, 1, 254)
    scale = np.exp2((e - 127).astype(np.float64))[..., None]
    mag = np.abs(b) / scale
    idx = np.abs(mag[..., None] - FP4).argmin(axis=-1)
    code = idx | ((b < 0) << 3)
    deq = (np.sign(b) * FP4[idx] * scale).astype(np.float32)
    # checkpoint packing: element 2i in the low nibble of byte i, element 2i + 1 in the high nibble
    lo, hi = code[..., 0::2], code[..., 1::2]
    packed = (lo | (hi << 4)).astype(np.uint8)
    return (packed.reshape(*lead, cols // 32, 16), e.astype(np.uint8).reshape(*lead, cols // 32),
            torch.from_numpy(deq.reshape(a.shape)))


def randomize(model, gen, mxfp4=False):
    """Weights that make every piece matter: routing varies by token, sinks and biases are not negligible."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("layernorm.weight", "norm.weight", "norm_f.weight")):
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=gen))
            elif name.endswith("sinks"):
                p.copy_(torch.randn(p.shape, generator=gen))
            elif name.endswith("A_log"):
                p.copy_(torch.log(1.0 + 7.0 * torch.rand(p.shape, generator=gen)))
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.8 - 0.5)
            elif name.endswith(".D") or name == "D":
                p.copy_(1.0 + 0.3 * torch.randn(p.shape, generator=gen))
            elif "conv1d" in name and name.endswith("weight"):
                p.copy_(0.4 * torch.randn(p.shape, generator=gen))
            elif "bias" in name:
                p.copy_(0.2 * torch.randn(p.shape, generator=gen))
            elif "router" in name:
                p.copy_(0.5 * torch.randn(p.shape, generator=gen))
            elif "embed" in name or "lm_head" in name:
                p.copy_(torch.randn(p.shape, generator=gen))
            else:
                p.copy_(0.12 * torch.randn(p.shape, generator=gen))


def build(family):
    from transformers import GptOssConfig, GptOssForCausalLM, Llama4ForCausalLM, Llama4TextConfig
    if family.startswith("granite"):
        from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM
        rope = family == "granite-rope"
        cfg = GraniteMoeHybridConfig(
            vocab_size=VOCAB, hidden_size=128, intermediate_size=64, shared_intermediate_size=96, num_hidden_layers=5,
            num_attention_heads=4, num_key_value_heads=2, num_local_experts=0 if rope else 4, num_experts_per_tok=2,
            layer_types=["mamba", "mamba", "attention", "mamba", "attention"], mamba_n_heads=8, mamba_d_head=32,
            mamba_n_groups=2, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, max_position_embeddings=512,
            position_embedding_type="rope" if rope else None, embedding_multiplier=2.0, logits_scaling=4.0,
            residual_multiplier=0.5, attention_multiplier=0.2, rms_norm_eps=1e-5,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
        return GraniteMoeHybridForCausalLM(cfg)
    if family == "nemotronh":
        from transformers import NemotronHConfig, NemotronHForCausalLM
        cfg = NemotronHConfig(vocab_size=VOCAB, hidden_size=128, intermediate_size=192, hybrid_override_pattern="M-M*-M",
                              num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                              mamba_num_heads=8, mamba_head_dim=32, n_groups=2, ssm_state_size=16, conv_kernel=4,
                              max_position_embeddings=512, layer_norm_epsilon=1e-5, time_step_min=0.05)
        return NemotronHForCausalLM(cfg)
    if family == "jamba":
        from transformers import JambaConfig, JambaForCausalLM
        cfg = JambaConfig(vocab_size=VOCAB, hidden_size=128, intermediate_size=192, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, num_experts=4, num_experts_per_tok=2,
                          attn_layer_period=4, attn_layer_offset=2, expert_layer_period=2, expert_layer_offset=1,
                          mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, mamba_dt_rank=8, max_position_embeddings=512)
        return JambaForCausalLM(cfg)
    if family.startswith("gptoss"):
        cfg = GptOssConfig(vocab_size=VOCAB, hidden_size=128, intermediate_size=64, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_local_experts=8,
                           num_experts_per_tok=2, sliding_window=4, max_position_embeddings=512, swiglu_limit=3.0,
                           rope_parameters={"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
                                            "truncate": False, "original_max_position_embeddings": 16,
                                            "rope_theta": 150000.0})
        return GptOssForCausalLM(cfg)
    cfg = Llama4TextConfig(vocab_size=VOCAB, hidden_size=128, intermediate_size=64, intermediate_size_mlp=192,
                           num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                           num_local_experts=4, num_experts_per_tok=1, attention_chunk_size=8, floor_scale=8,
                           attn_scale=0.4, interleave_moe_layer_step=2 if family == "llama4-mix" else 1,
                           max_position_embeddings=512, eos_token_id=2,
                           rope_parameters={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                            "high_freq_factor": 4.0, "original_max_position_embeddings": 16,
                                            "rope_theta": 500000.0})
    return Llama4ForCausalLM(cfg)


def rounded_copy(model, family, state):
    """The model with the weights the engine quantizes at q8 rounded to Q8_0 (the engine's own weights); with
    `state` given, loads the (already prepared) state instead."""
    import copy
    m = copy.deepcopy(model)
    sd = m.state_dict()
    with torch.no_grad():
        for name, t in sd.items():
            if name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
                if family == "gptoss-mxfp4":
                    continue  # already the exact MXFP4 values
                # gpt-oss / Llama 4: [E, in, out], blocks along `in`; Granite / Jamba: [E, out, in], blocks along the last axis
                t.copy_(q8_round(t, axis=1 if family.startswith(("gptoss", "llama4")) else -1))
            elif t.dim() == 2 and "router" not in name and "bias" not in name and "dt_proj" not in name and not name.endswith("A_log"):  # the engine keeps biases, the router and dt_proj in f32
                t.copy_(q8_round(t, axis=-1))
    return m


def run(model, ids):
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0].float()


def greedy(model, prompt, steps):
    ids = list(prompt)
    for _ in range(steps):
        ids.append(int(run(model, ids)[-1].argmax()))
    return ids


def main():
    family, out = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1234
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    gen = torch.Generator().manual_seed(seed)
    model = build(family).float().eval()
    randomize(model, gen)
    os.makedirs(os.path.join(out, "ref"), exist_ok=True)
    mx = family == "gptoss-mxfp4"
    extra = {}
    if mx:  # the experts as the checkpoint ships them: MXFP4 blocks / scales, [E, out, in] (transposed)
        with torch.no_grad():
            for name, p in list(model.named_parameters()):
                if name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
                    blocks, scales, deq = mxfp4_quantize(p.detach().transpose(1, 2).contiguous())
                    base = name
                    extra[base + "_blocks"] = torch.from_numpy(blocks)
                    extra[base + "_scales"] = torch.from_numpy(scales)
                    p.copy_(deq.transpose(1, 2))
    from safetensors.torch import save_file
    if not mx:
        model.save_pretrained(out, safe_serialization=True)  # the checkpoint names transformers writes
        sd = model.state_dict()
    else:
        sd = {k: v.contiguous() for k, v in model.state_dict().items()}
        for k in [k for k in sd if k.endswith("experts.gate_up_proj") or k.endswith("experts.down_proj")]:
            del sd[k]
        sd.update(extra)
        model.config.save_pretrained(out)
    if mx:
        cfg = json.load(open(os.path.join(out, "config.json")))
        cfg["quantization_config"] = {"modules_to_not_convert": ["model.layers.*.self_attn", "model.layers.*.mlp.router",
                                                                 "model.embed_tokens", "lm_head"], "quant_method": "mxfp4"}
        json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    if mx:
        save_file(sd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})

    rng = np.random.default_rng(seed)
    prompt = [int(x) for x in rng.integers(3, VOCAB, PROMPT_LEN)]
    ref = rounded_copy(model, family, None).eval()
    ids = greedy(ref, prompt, STEPS)
    logits = run(ref, ids).numpy().astype("<f4")
    exact = run(model, ids).numpy().astype("<f4")
    open(os.path.join(out, "ref", "logits.f32"), "wb").write(logits.tobytes())
    open(os.path.join(out, "ref", "logits.f32exact"), "wb").write(exact.tobytes())
    json.dump({"family": family, "prompt": prompt, "ids": ids, "vocab": VOCAB,
               "greedy_exact": greedy(model, prompt, STEPS)[PROMPT_LEN:]},
              open(os.path.join(out, "ref", "ref.json"), "w"))
    print("%s: %d keys, prompt %d + %d greedy ids, q8 vs exact reference max |diff| %.3g" % (
        family, len(sd), PROMPT_LEN, STEPS, float(np.abs(logits - exact).max())))


if __name__ == "__main__":
    main()
