#!/usr/bin/env python3
"""Writes a tiny random-weight Hugging Face checkpoint (config.json + model.safetensors, float32) for
pipeline tests. Standard library only.

usage: make-tiny-model.py <out-dir> <family> [--seed N]
  family: llama | mistral | qwen2 | llama3rope | glm | glmedge | glm4

All three use headDim = 32 so the Metal backend (which needs a multiple of 32) can run them:
  llama      plain Llama: GQA 4/2, untied head, no scaling
  mistral    sliding_window=4 on every layer, GQA 4/2, untied head
  qwen2      q/k/v bias, tied embeddings, sliding window 6 on layers >= 2 (use_sliding_window)
  glm        GlmForCausalLM: q/k/v bias, fused gate_up_proj, interleaved RoPE on half of each 32-dim head
             (partial_rotary_factor 0.5, the default when the key is absent), two norms per layer, untied head
  glmedge    GlmForCausalLM like GLM-Edge: head_dim 64 (!= hidden/heads), no bias, full interleaved rotary
             (partial_rotary_factor 1.0), tied embeddings
  glm4       Glm4ForCausalLM: bias, head_dim 64, partial_rotary_factor 0.5, four norms per layer
             (post_self_attn_layernorm / post_mlp_layernorm), fused gate_up_proj
  llama3rope llama3 rope_scaling with original_max_position_embeddings=32 so every band of the rule is hit
"""
import json
import os
import random
import struct
import sys

FAMILIES = {
    "llama": dict(
        architectures=["LlamaForCausalLM"], model_type="llama", hidden_size=128, intermediate_size=192,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, vocab_size=1000,
        max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=500000.0, hidden_act="silu",
        attention_bias=False, tie_word_embeddings=False, bos_token_id=1, eos_token_id=2),
    "mistral": dict(
        architectures=["MistralForCausalLM"], model_type="mistral", hidden_size=128, intermediate_size=192,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, vocab_size=1000,
        max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=1000000.0, hidden_act="silu",
        sliding_window=4, tie_word_embeddings=False, bos_token_id=1, eos_token_id=2),
    "qwen2": dict(
        architectures=["Qwen2ForCausalLM"], model_type="qwen2", hidden_size=128, intermediate_size=192,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, vocab_size=1000,
        max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=1000000.0, hidden_act="silu",
        use_sliding_window=True, sliding_window=6, max_window_layers=2, tie_word_embeddings=True,
        bos_token_id=999, eos_token_id=[998, 999]),
    "llama3rope": dict(
        architectures=["LlamaForCausalLM"], model_type="llama", hidden_size=128, intermediate_size=192,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4, vocab_size=1000,
        max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=10000.0, hidden_act="silu",
        attention_bias=False, tie_word_embeddings=True, bos_token_id=1, eos_token_id=[2, 3],
        rope_scaling=dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
                          original_max_position_embeddings=32)),
    "glm": dict(
        architectures=["GlmForCausalLM"], model_type="glm", hidden_size=128, intermediate_size=192,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=1000,
        max_position_embeddings=512, rms_norm_eps=1.5625e-07, rope_theta=10000.0, hidden_act="silu",
        attention_bias=True, tie_word_embeddings=False, eos_token_id=[998, 999], pad_token_id=998),
    "glmedge": dict(
        architectures=["GlmForCausalLM"], model_type="glm", hidden_size=128, intermediate_size=192,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=64, vocab_size=1000,
        max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=10000.0, hidden_act="silu",
        attention_bias=False, partial_rotary_factor=1.0, tie_word_embeddings=True, eos_token_id=[998, 999],
        pad_token_id=998),
    "glm4": dict(
        architectures=["Glm4ForCausalLM"], model_type="glm4", hidden_size=128, intermediate_size=192,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=64, vocab_size=1000,
        max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=10000.0, hidden_act="silu",
        attention_bias=True, partial_rotary_factor=0.5, tie_word_embeddings=False, eos_token_id=[998, 999],
        pad_token_id=998),
}


def main():
    if len(sys.argv) < 3 or sys.argv[2] not in FAMILIES:
        sys.exit(__doc__)
    out, fam = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1234
    rnd = random.Random(seed)
    cfg = dict(FAMILIES[fam])
    d, heads, kv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim", d // heads)
    glm = cfg["model_type"] in ("glm", "glm4")
    ffn, vocab = cfg["intermediate_size"], cfg["vocab_size"]
    bias = cfg["model_type"] == "qwen2" or (glm and cfg.get("attention_bias", True))

    tensors = {}

    def mat(name, rows, cols, std=0.08):
        tensors[name] = ([rows, cols], [rnd.gauss(0, std) for _ in range(rows * cols)])

    def vec(name, n, base=1.0, std=0.1):
        tensors[name] = ([n], [base + rnd.gauss(0, std) for _ in range(n)])

    mat("model.embed_tokens.weight", vocab, d, 1.0)
    for l in range(cfg["num_hidden_layers"]):
        p = "model.layers.%d." % l
        vec(p + "input_layernorm.weight", d)
        vec(p + "post_attention_layernorm.weight", d)
        mat(p + "self_attn.q_proj.weight", heads * hd, d)
        mat(p + "self_attn.k_proj.weight", kv * hd, d)
        mat(p + "self_attn.v_proj.weight", kv * hd, d)
        mat(p + "self_attn.o_proj.weight", d, heads * hd)
        if bias:
            vec(p + "self_attn.q_proj.bias", heads * hd, 0.0, 0.15)
            vec(p + "self_attn.k_proj.bias", kv * hd, 0.0, 0.15)
            vec(p + "self_attn.v_proj.bias", kv * hd, 0.0, 0.15)
        if cfg["model_type"] == "glm4":
            vec(p + "post_self_attn_layernorm.weight", d)
            vec(p + "post_mlp_layernorm.weight", d)
        if glm:
            mat(p + "mlp.gate_up_proj.weight", 2 * ffn, d)  # gate rows first, then up rows
        else:
            mat(p + "mlp.gate_proj.weight", ffn, d)
            mat(p + "mlp.up_proj.weight", ffn, d)
        mat(p + "mlp.down_proj.weight", d, ffn)
    vec("model.norm.weight", d)
    if not cfg["tie_word_embeddings"]:
        mat("lm_head.weight", vocab, d, 0.5)

    header, blobs, off = {}, [], 0
    for name, (shape, vals) in tensors.items():
        raw = struct.pack("<%df" % len(vals), *vals)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * (-len(hj) % 8)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)
    cfg["torch_dtype"] = "float32"
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote %s (%s, %d tensors, %.1f MB)" % (out, fam, len(tensors), off / 1e6))


main()
