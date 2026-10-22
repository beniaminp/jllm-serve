#!/usr/bin/env python3
"""Writes a tiny random-weight Hugging Face checkpoint (config.json + model.safetensors, float32) for
pipeline tests. Standard library only.

usage: make-tiny-model.py <out-dir> <family> [--seed N] [--head-dim N] [--vocab N] [--attention-bias]
  family: llama | mistral | qwen2 | llama3rope | glm | glmedge | glm4 | phi3 | phi3short | phi3mha | gemma2 | gemma3 | gemma3mm | mamba | qwen3moe | deepseek_v2 | deepseek_v2q | deepseek_v3 | mamba3 | mamba130m | qwen3next | qwen3next2 | gptoss | gptoss_mxfp4 | llama4 | llama4mix | granite | nemotronh | jamba

All use headDim 32 (gemma2: 64) so the Metal backend (which needs a multiple of 32) can run them:
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
  phi3       Phi-3/4-mini layout: fused qkv_proj and gate_up_proj, longrope with original_max_position_embeddings=32
             (the smoke tests run with ctx 64, so the long factors apply), partial_rotary_factor 0.75 (24 of 32
             dims rotated), sliding_window=6 on all layers, GQA 4/2, untied head
  phi3short  as phi3 but original_max_position_embeddings=128 (short factors), tied embeddings, no sliding window
  phi3mha    Phi-3-mini / phi-4 style: headDim 64, MHA, plain RoPE (rope_scaling null), full rotation, sliding_window=6
  gemma2     Gemma 2: head_dim 64, 4 norms per layer, (1+w) norms, gelu_tanh, attention soft-cap 1.0 (small so it
             bites on random weights) and final soft-cap 30, query_pre_attn_scalar 48, local layers 0 and 2
             (sliding_window 6), embedding scaled by sqrt(hidden)
  gemma3     Gemma 3 text: head_dim 32, q/k norm, no soft-caps, 6 layers = 5 local + 1 global (pattern 6),
             rope_theta 1e6 with linear scaling 8 on the global layer, rope_local_base_freq 1e4 on local ones
  gemma3mm   the same decoder in the multimodal wrapper layout (Gemma3ForConditionalGeneration, text_config,
             tensors under language_model.model.*, plus dummy vision tensors that must be ignored)
  mamba      Mamba v1: d_model 64, d_inner 128, state 16, conv 4, dt_rank 4 (ceil(64/16)), 3 layers, tied head
  mamba130m  not tiny: random weights with the exact mamba-130m-hf shapes (24 layers, d_model 768, 515 MB) to
             benchmark the Mamba backends without downloading the real checkpoint (output is gibberish)
  mamba3     Mamba v1: d_model 128, d_inner 256, state 8, conv 3, dt_rank 8 (ceil(128/16)), 2 layers, untied head
  qwen3moe   Qwen3-MoE: q/k norm, 4 layers of which layer 1 is dense (mlp_only_layers), 8 experts of width 32,
             top 2 renormalized, per-expert tensors and router mlp.gate (no shared experts), untied head
  deepseek_v2  MLA with a single q_proj (q_lora_rank null, as DeepSeek-V2-Lite), 2 dense + 2 MoE layers, 8 routed
               experts in 2 groups (group_limited_greedy, softmax), 2 shared experts, yarn (mscale == mscale_all_dim)
  deepseek_v2q like deepseek_v2 with q_a_proj/q_a_layernorm/q_b_proj, norm_topk_prob, routed_scaling_factor 2.5
               and yarn mscale != mscale_all_dim
  deepseek_v3  MLA with q_lora_rank, sigmoid scores, noaux_tc with e_score_correction_bias, 1 shared expert,
               routed_scaling_factor 2.5, norm_topk_prob, plus a stray layer 4 (V3's MTP layer, must be ignored)
  qwen3next  Qwen3-Next hybrid: 4 layers, full_attention_interval 2 (layers 1 and 3 gated full attention, head_dim 64,
             partial_rotary_factor 0.25, GQA 4/2), Gated DeltaNet in layers 0 and 2 (2 key heads x 32, 4 value heads
             x 32, conv 4), layer 0 dense (mlp_only_layers), the others 8 experts top 2 + a gated shared expert of 64,
             untied head, zero-centred (1+w) norms, plus an mtp.* tensor that must be ignored
  qwen3next2 Qwen3-Next with explicit layer_types (L L F L L), one value head per key head with key dim 64 != value
             dim 32, conv 3, head_dim 32 with partial_rotary_factor 0.5, MoE every other layer (decoder_sparse_step 2,
             4 experts top 2, norm_topk_prob false), tied embeddings
  gptoss     gpt-oss: sliding window 4 on layers 0 and 2, full attention on 1 and 3, attention sinks, q/k/v/o biases, YaRN
             (legacy rope_scaling keys, truncate false, original length 16), 8 experts of width 64 top 2 with biased
             router, expert biases and the clamped SwiGLU (limit 3), experts stored as float tensors [E, in, out]
  gptoss_mxfp4  the same with the experts as MXFP4 blocks / scales as openai/gpt-oss-20b ships them
  llama4     Llama 4 text: 4 layers, chunked attention (chunk 8) on 3 of 4, a NoPE layer with attention temperature tuning
             (floor 8), L2 q/k norm, llama3-scaled interleaved RoPE, MoE (top 1 of 4, sigmoid on the input) with a shared
             expert on every layer
  llama4mix  as llama4 with dense and MoE layers alternating (interleave_moe_layer_step 2), the wrapper layout
             (Llama4ForConditionalGeneration, text_config, language_model.* tensors, a dummy vision tensor)
  granite    Granite 4 (GraniteMoeHybridForCausalLM, the released checkpoint's tensor names): 5 layers mamba / mamba /
             attention / mamba / attention, Mamba-2 with 8 heads of 32, 2 groups, state 16, conv 4, 4 experts top 2 plus the
             shared MLP in every layer, the muP multipliers, attention without RoPE, tied head
  nemotronh  Nemotron-H (NemotronHForCausalLM): pattern "M-M*-M" (Mamba-2 with 2 groups, ReLU-squared MLP, attention),
             one norm per layer, backbone.* names, untied head
  jamba      Jamba (JambaForCausalLM): 4 layers, Mamba-1 with the dt / B / C norms in 0, 1 and 3, attention in layer 2, a MoE
             (4 experts top 2, per-expert tensors) in layers 1 and 3 and a SwiGLU MLP in 0 and 2, untied head
MLA head sizes are nope 32 + rope 32 = 64 for q/k and v 32, so the attention kernel's separate q/k and v widths
are exercised.
--attention-bias sets attention_bias: true on a llama / qwen3 / gemma config, which (as in transformers) gives
q_proj, k_proj, v_proj AND o_proj a bias.
"""
import json
import math
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
    "gemma2": dict(
        architectures=["Gemma2ForCausalLM"], model_type="gemma2", hidden_size=128, intermediate_size=192,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=64, vocab_size=1000,
        max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=10000.0, hidden_act="gelu_pytorch_tanh",
        hidden_activation="gelu_pytorch_tanh", attn_logit_softcapping=1.0, final_logit_softcapping=30.0,
        query_pre_attn_scalar=48, sliding_window=6, attention_bias=False, tie_word_embeddings=True,
        bos_token_id=2, eos_token_id=[1, 107]),
    "gemma3": dict(
        architectures=["Gemma3ForCausalLM"], model_type="gemma3_text", hidden_size=128, intermediate_size=192,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=1, head_dim=32, vocab_size=1000,
        max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=1000000.0, rope_local_base_freq=10000.0,
        rope_scaling=dict(factor=8.0, rope_type="linear"), hidden_activation="gelu_pytorch_tanh",
        attn_logit_softcapping=None, final_logit_softcapping=None, query_pre_attn_scalar=48, sliding_window=6,
        sliding_window_pattern=6, attention_bias=False, tie_word_embeddings=True, bos_token_id=2,
        eos_token_id=106),
    "qwen3moe": dict(
        architectures=["Qwen3MoeForCausalLM"], model_type="qwen3_moe", hidden_size=128, intermediate_size=192,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=1000,
        max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=1000000.0, hidden_act="silu",
        attention_bias=False, tie_word_embeddings=False, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=32, norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[1],
        bos_token_id=1, eos_token_id=2),
    "mamba": dict(
        architectures=["MambaForCausalLM"], model_type="mamba", hidden_size=64, state_size=16, conv_kernel=4,
        expand=2, intermediate_size=128, time_step_rank=4, num_hidden_layers=3, vocab_size=1000,
        layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True, hidden_act="silu",
        bos_token_id=0, eos_token_id=0),
    "mamba130m": dict(
        architectures=["MambaForCausalLM"], model_type="mamba", hidden_size=768, state_size=16, conv_kernel=4,
        expand=2, intermediate_size=1536, time_step_rank=48, num_hidden_layers=24, vocab_size=50280,
        layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True, hidden_act="silu",
        bos_token_id=0, eos_token_id=0),
    "mamba3": dict(
        architectures=["MambaForCausalLM"], model_type="mamba", hidden_size=128, state_size=8, conv_kernel=3,
        expand=2, intermediate_size=256, time_step_rank="auto", num_hidden_layers=2, vocab_size=1000,
        layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True, hidden_act="silu",
        bos_token_id=0, eos_token_id=0, tie_word_embeddings=False),
}
FAMILIES["qwen3next"] = dict(
    architectures=["Qwen3NextForCausalLM"], model_type="qwen3_next", hidden_size=128, intermediate_size=192,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=64, vocab_size=1000,
    max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=1000000.0, partial_rotary_factor=0.25,
    hidden_act="silu", attention_bias=False, tie_word_embeddings=False, full_attention_interval=2,
    linear_conv_kernel_dim=4, linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=2,
    linear_num_value_heads=4, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
    shared_expert_intermediate_size=64, norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[0],
    bos_token_id=1, eos_token_id=2)
FAMILIES["qwen3next2"] = dict(
    architectures=["Qwen3NextForCausalLM"], model_type="qwen3_next", hidden_size=128, intermediate_size=96,
    num_hidden_layers=5, num_attention_heads=2, num_key_value_heads=1, head_dim=32, vocab_size=1000,
    max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=10000.0, partial_rotary_factor=0.5,
    hidden_act="silu", attention_bias=False, tie_word_embeddings=True,
    layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention"],
    linear_conv_kernel_dim=3, linear_key_head_dim=64, linear_value_head_dim=32, linear_num_key_heads=2,
    linear_num_value_heads=2, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
    shared_expert_intermediate_size=32, norm_topk_prob=False, decoder_sparse_step=2, mlp_only_layers=[],
    bos_token_id=1, eos_token_id=[2, 3])
FAMILIES["gptoss"] = dict(
    architectures=["GptOssForCausalLM"], model_type="gpt_oss", hidden_size=128, intermediate_size=64,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=1000,
    max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=150000.0, attention_bias=True,
    tie_word_embeddings=False, sliding_window=4,
    layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    num_local_experts=8, num_experts_per_tok=2, swiglu_limit=3.0, hidden_act="silu", bos_token_id=1, eos_token_id=2,
    rope_scaling=dict(rope_type="yarn", factor=32.0, beta_fast=32.0, beta_slow=1.0, truncate=False,
                      original_max_position_embeddings=16))
FAMILIES["gptoss_mxfp4"] = dict(FAMILIES["gptoss"], quantization_config=dict(quant_method="mxfp4"))
FAMILIES["llama4"] = dict(
    architectures=["Llama4ForCausalLM"], model_type="llama4_text", hidden_size=128, intermediate_size=64,
    intermediate_size_mlp=192, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    vocab_size=1000, max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=500000.0, attention_bias=False,
    tie_word_embeddings=False, num_local_experts=4, num_experts_per_tok=1, interleave_moe_layer_step=1,
    attention_chunk_size=8, floor_scale=8, attn_scale=0.4, attn_temperature_tuning=True, use_qk_norm=True,
    no_rope_layers=[1, 1, 1, 0], hidden_act="silu", bos_token_id=1, eos_token_id=2,
    rope_scaling=dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
                      original_max_position_embeddings=16))
FAMILIES["llama4mix"] = dict(FAMILIES["llama4"], interleave_moe_layer_step=2)
FAMILIES["granite"] = dict(
    architectures=["GraniteMoeHybridForCausalLM"], model_type="granitemoehybrid", hidden_size=128, intermediate_size=64,
    shared_intermediate_size=96, num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2, vocab_size=1000,
    max_position_embeddings=512, rms_norm_eps=1e-5, hidden_act="silu", attention_bias=False, tie_word_embeddings=True,
    layer_types=["mamba", "mamba", "attention", "mamba", "attention"], mamba_n_heads=8, mamba_d_head=32, mamba_n_groups=2,
    mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, mamba_conv_bias=True, mamba_proj_bias=False,
    num_local_experts=4, num_experts_per_tok=2, position_embedding_type="nope", embedding_multiplier=2.0,
    logits_scaling=4.0, residual_multiplier=0.5, attention_multiplier=0.2, bos_token_id=1, eos_token_id=2)
FAMILIES["nemotronh"] = dict(
    architectures=["NemotronHForCausalLM"], model_type="nemotron_h", hidden_size=128, intermediate_size=192,
    hybrid_override_pattern="M-M*-M", num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    vocab_size=1000, max_position_embeddings=512, layer_norm_epsilon=1e-5, mamba_num_heads=8, mamba_head_dim=32,
    n_groups=2, ssm_state_size=16, conv_kernel=4, mamba_hidden_act="silu", mlp_hidden_act="relu2", use_bias=False,
    use_conv_bias=True, mlp_bias=False, attention_bias=False, time_step_min=0.05, tie_word_embeddings=False,
    bos_token_id=1, eos_token_id=2)
FAMILIES["jamba"] = dict(
    architectures=["JambaForCausalLM"], model_type="jamba", hidden_size=128, intermediate_size=192, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, vocab_size=1000, max_position_embeddings=512, rms_norm_eps=1e-6,
    hidden_act="silu", num_experts=4, num_experts_per_tok=2, attn_layer_period=4, attn_layer_offset=2,
    expert_layer_period=2, expert_layer_offset=1, mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, mamba_dt_rank=8,
    mamba_conv_bias=True, mamba_proj_bias=False, tie_word_embeddings=False, bos_token_id=1, eos_token_id=2)
TEXT_KEYS_EXCLUDED_IN_WRAPPER = ("architectures", "bos_token_id", "eos_token_id")


def gemma_wrapper(text):
    inner = {k: v for k, v in text.items() if k not in TEXT_KEYS_EXCLUDED_IN_WRAPPER}
    return dict(architectures=["Gemma3ForConditionalGeneration"], model_type="gemma3", bos_token_id=2,
                eos_token_id=106, image_token_index=1001, mm_tokens_per_image=4, text_config=inner,
                vision_config=dict(model_type="siglip_vision_model", hidden_size=32, image_size=28,
                                   intermediate_size=64, num_attention_heads=2, num_hidden_layers=1,
                                   patch_size=14))

_DS_COMMON = dict(
    hidden_size=128, intermediate_size=192, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
    vocab_size=1000, max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=10000.0, hidden_act="silu",
    attention_bias=False, tie_word_embeddings=False, bos_token_id=998, eos_token_id=999,
    kv_lora_rank=64, qk_nope_head_dim=32, qk_rope_head_dim=32, v_head_dim=32,
    first_k_dense_replace=2, moe_layer_freq=1, n_routed_experts=8, moe_intermediate_size=32,
    num_experts_per_tok=2, n_group=2, topk_group=1)
FAMILIES["deepseek_v2"] = dict(
    _DS_COMMON, architectures=["DeepseekV2ForCausalLM"], model_type="deepseek_v2", q_lora_rank=None,
    n_shared_experts=2, norm_topk_prob=False, routed_scaling_factor=1.0, scoring_func="softmax",
    topk_method="group_limited_greedy",
    rope_scaling=dict(type="yarn", factor=4.0, original_max_position_embeddings=64, beta_fast=32, beta_slow=1,
                      mscale=0.707, mscale_all_dim=0.707))
FAMILIES["deepseek_v2q"] = dict(
    _DS_COMMON, architectures=["DeepseekV2ForCausalLM"], model_type="deepseek_v2", q_lora_rank=64,
    n_shared_experts=1, norm_topk_prob=True, routed_scaling_factor=2.5, scoring_func="softmax",
    topk_method="group_limited_greedy",
    rope_scaling=dict(type="yarn", factor=4.0, original_max_position_embeddings=64, beta_fast=32, beta_slow=1,
                      mscale=1.0, mscale_all_dim=0.707))
FAMILIES["deepseek_v3"] = dict(
    _DS_COMMON, architectures=["DeepseekV3ForCausalLM"], model_type="deepseek_v3", q_lora_rank=64,
    n_shared_experts=1, norm_topk_prob=True, routed_scaling_factor=2.5, scoring_func="sigmoid",
    topk_method="noaux_tc", n_group=4, topk_group=2,
    rope_scaling=dict(type="yarn", factor=4.0, original_max_position_embeddings=64, beta_fast=32, beta_slow=1,
                      mscale=1.0, mscale_all_dim=1.0))


def deepseek_tensors(cfg, rnd):
    """Tensor dict name -> (shape, values) of a tiny DeepSeek-V2/V3 checkpoint (HF names)."""
    d, heads, vocab = cfg["hidden_size"], cfg["num_attention_heads"], cfg["vocab_size"]
    nope, rope, vd = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
    ql, kvl = cfg["q_lora_rank"], cfg["kv_lora_rank"]
    tensors = {}

    def mat(name, rows, cols, std=0.08):
        tensors[name] = ([rows, cols], [rnd.gauss(0, std) for _ in range(rows * cols)])

    def vec(name, n, base=1.0, std=0.1):
        tensors[name] = ([n], [base + rnd.gauss(0, std) for _ in range(n)])

    def mlp(p, inter):
        mat(p + "gate_proj.weight", inter, d)
        mat(p + "up_proj.weight", inter, d)
        mat(p + "down_proj.weight", d, inter)

    mat("model.embed_tokens.weight", vocab, d, 1.0)
    for l in range(cfg["num_hidden_layers"]):
        p = "model.layers.%d." % l
        a = p + "self_attn."
        vec(p + "input_layernorm.weight", d)
        vec(p + "post_attention_layernorm.weight", d)
        if ql is None:
            mat(a + "q_proj.weight", heads * (nope + rope), d, 0.25)
        else:
            mat(a + "q_a_proj.weight", ql, d)
            vec(a + "q_a_layernorm.weight", ql)
            mat(a + "q_b_proj.weight", heads * (nope + rope), ql, 0.25)
        mat(a + "kv_a_proj_with_mqa.weight", kvl + rope, d, 0.25)
        vec(a + "kv_a_layernorm.weight", kvl)
        mat(a + "kv_b_proj.weight", heads * (nope + vd), kvl, 0.25)
        mat(a + "o_proj.weight", d, heads * vd)
        if l < cfg["first_k_dense_replace"] or l % cfg["moe_layer_freq"] != 0:
            mlp(p + "mlp.", cfg["intermediate_size"])
        else:
            e, m = cfg["n_routed_experts"], cfg["moe_intermediate_size"]
            mat(p + "mlp.gate.weight", e, d, 0.15)
            if cfg["topk_method"] == "noaux_tc":
                vec(p + "mlp.gate.e_score_correction_bias", e, 0.0, 0.3)
            for i in range(e):
                mlp(p + "mlp.experts.%d." % i, m)
            if cfg["n_shared_experts"]:
                mlp(p + "mlp.shared_experts.", m * cfg["n_shared_experts"])
    if cfg["model_type"] == "deepseek_v3":  # the multi-token-prediction layer: present in the checkpoint, unused
        mat("model.layers.%d.eh_proj.weight" % cfg["num_hidden_layers"], d, 2 * d)
    vec("model.norm.weight", d)
    mat("lm_head.weight", vocab, d, 0.5)
    return tensors


def mxfp4_quantize(vals, rows, cols):
    """Row-major float values -> (blocks bytes [rows, cols/32, 16], scales bytes [rows, cols/32], dequantized values) in
    the checkpoint layout of gpt-oss's MXFP4 experts: scale 2^(e - 127) with the smallest e such that max|block| / scale <= 6,
    e2m1 codes (sign bit 8), element 2i in the low nibble of byte i and 2i + 1 in the high one."""
    mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    blocks, scales, deq = bytearray(), bytearray(), []
    for r in range(rows):
        for b in range(cols // 32):
            blk = vals[r * cols + b * 32: r * cols + b * 32 + 32]
            amax = max(abs(v) for v in blk)
            e = 127 if amax == 0 else min(254, max(1, math.ceil(math.log2(amax / 6.0)) + 127))
            s = 2.0 ** (e - 127)
            codes = []
            for v in blk:
                i = min(range(8), key=lambda k: abs(abs(v) / s - mags[k]))
                codes.append(i | (8 if v < 0 else 0))
                deq.append(math.copysign(mags[i] * s, v) if i else 0.0)
            scales.append(e)
            for j in range(16):
                blocks.append(codes[2 * j] | (codes[2 * j + 1] << 4))
    return bytes(blocks), bytes(scales), deq


def gptoss_tensors(cfg, rnd, mxfp4):
    """Random gpt-oss weights with the released checkpoints' names: attention with biases and sinks, a biased router, the
    experts' gate_up_proj [E, dim, 2m] (gate and up columns alternating) / down_proj [E, m, dim] with their biases, or
    for MXFP4 the transposed gate_up_proj_blocks [E, 2m, dim/32, 16] / _scales and down_proj_blocks / _scales."""
    d, vocab, heads, kv, hd = cfg["hidden_size"], cfg["vocab_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    e, m = cfg["num_local_experts"], cfg["intermediate_size"]
    tensors = {}

    def mat(name, rows, cols, std=0.12):
        tensors[name] = ([rows, cols], [rnd.gauss(0, std) for _ in range(rows * cols)])

    def vec(name, n, base=0.0, std=0.2):
        tensors[name] = ([n], [base + rnd.gauss(0, std) for _ in range(n)])

    mat("model.embed_tokens.weight", vocab, d, 1.0)
    for l in range(cfg["num_hidden_layers"]):
        p = "model.layers.%d." % l
        vec(p + "input_layernorm.weight", d, 1.0)
        vec(p + "post_attention_layernorm.weight", d, 1.0)
        for n, rows in (("q", heads * hd), ("k", kv * hd), ("v", kv * hd)):
            mat(p + "self_attn.%s_proj.weight" % n, rows, d, 0.05 if n != "v" else 0.12)
            vec(p + "self_attn.%s_proj.bias" % n, rows)
        mat(p + "self_attn.o_proj.weight", d, heads * hd)
        vec(p + "self_attn.o_proj.bias", d)
        vec(p + "self_attn.sinks", heads, 0.0, 1.0)
        mat(p + "mlp.router.weight", e, d, 0.02)
        vec(p + "mlp.router.bias", e, 0.0, 2.0)
        x = p + "mlp.experts."
        tensors[x + "gate_up_proj_bias"] = ([e, 2 * m], [rnd.gauss(0, 0.2) for _ in range(e * 2 * m)])
        tensors[x + "down_proj_bias"] = ([e, d], [rnd.gauss(0, 0.2) for _ in range(e * d)])
        gu = [rnd.gauss(0, 0.12) for _ in range(e * d * 2 * m)]      # [E, dim, 2m]
        dn = [rnd.gauss(0, 0.12) for _ in range(e * m * d)]          # [E, m, dim]
        if not mxfp4:
            tensors[x + "gate_up_proj"] = ([e, d, 2 * m], gu)
            tensors[x + "down_proj"] = ([e, m, d], dn)
            continue
        for name, vals, rows_in, cols_out in (("gate_up_proj", gu, d, 2 * m), ("down_proj", dn, m, d)):
            # transposed to [E, out, in] (the matmul reads rows of `in`), quantized along `in`
            tr = []
            for ex in range(e):
                base = ex * rows_in * cols_out
                for o in range(cols_out):
                    tr.extend(vals[base + i * cols_out + o] for i in range(rows_in))
            blocks, scales, _ = mxfp4_quantize(tr, e * cols_out, rows_in)
            tensors[x + name + "_blocks"] = ([e, cols_out, rows_in // 32, 16], blocks, "U8")
            tensors[x + name + "_scales"] = ([e, cols_out, rows_in // 32], scales, "U8")
    vec("model.norm.weight", d, 1.0)
    mat("lm_head.weight", vocab, d, 0.5)
    return tensors


def llama4_tensors(cfg, rnd, wrapper):
    """Random Llama 4 text weights with the released checkpoints' names (language_model.* in the wrapper layout): dense
    layers feed_forward.{gate,up,down}_proj, MoE layers feed_forward.router, experts.gate_up_proj [E, dim, 2m] (gate columns
    then up columns), experts.down_proj [E, m, dim] and a shared_expert."""
    d, vocab, heads, kv, hd = cfg["hidden_size"], cfg["vocab_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    e, m, step = cfg["num_local_experts"], cfg["intermediate_size"], cfg["interleave_moe_layer_step"]
    pre = "language_model.model." if wrapper else "model."
    tensors = {}

    def mat(name, rows, cols, std=0.12):
        tensors[name] = ([rows, cols], [rnd.gauss(0, std) for _ in range(rows * cols)])

    def vec(name, n, base=1.0, std=0.2):
        tensors[name] = ([n], [base + rnd.gauss(0, std) for _ in range(n)])

    mat(pre + "embed_tokens.weight", vocab, d, 1.0)
    for l in range(cfg["num_hidden_layers"]):
        p = pre + "layers.%d." % l
        vec(p + "input_layernorm.weight", d)
        vec(p + "post_attention_layernorm.weight", d)
        mat(p + "self_attn.q_proj.weight", heads * hd, d)
        mat(p + "self_attn.k_proj.weight", kv * hd, d)
        mat(p + "self_attn.v_proj.weight", kv * hd, d)
        mat(p + "self_attn.o_proj.weight", d, heads * hd)
        f = p + "feed_forward."
        if (l + 1) % step == 0:
            mat(f + "router.weight", e, d, 0.5)
            tensors[f + "experts.gate_up_proj"] = ([e, d, 2 * m], [rnd.gauss(0, 0.12) for _ in range(e * d * 2 * m)])
            tensors[f + "experts.down_proj"] = ([e, m, d], [rnd.gauss(0, 0.12) for _ in range(e * m * d)])
            mat(f + "shared_expert.gate_proj.weight", m, d)
            mat(f + "shared_expert.up_proj.weight", m, d)
            mat(f + "shared_expert.down_proj.weight", d, m)
        else:
            mat(f + "gate_proj.weight", cfg["intermediate_size_mlp"], d)
            mat(f + "up_proj.weight", cfg["intermediate_size_mlp"], d)
            mat(f + "down_proj.weight", d, cfg["intermediate_size_mlp"])
    vec(pre + "norm.weight", d)
    mat("language_model.lm_head.weight" if wrapper else "lm_head.weight", vocab, d, 0.5)
    if wrapper:
        mat("vision_model.patch_embedding.linear.weight", 32, 24, 0.1)
    return tensors


def llama4_wrapper(text):
    inner = {k: v for k, v in text.items() if k not in TEXT_KEYS_EXCLUDED_IN_WRAPPER}
    return dict(architectures=["Llama4ForConditionalGeneration"], model_type="llama4", boi_token_index=996,
                eoi_token_index=997, image_token_index=998, tie_word_embeddings=False, text_config=inner,
                vision_config=dict(model_type="llama4_vision_model", hidden_size=32, num_hidden_layers=1))


def hybrid_tensors(cfg, rnd):
    """Random weights of Granite 4 / Nemotron-H / Jamba under the released checkpoints' names (docs/14.22): Mamba-2 with
    [z | x B C | dt] in_proj rows and a [heads] A_log / D / dt_bias, Mamba-1 with x_proj / dt_proj and the dt / B / C layer
    norms, attention without positions, the MoE as input_linear / output_linear (Granite) or per-expert tensors (Jamba)."""
    kind = cfg["model_type"]
    d, vocab, n = cfg["hidden_size"], cfg["vocab_size"], cfg["num_hidden_layers"] if kind != "nemotron_h" else len(cfg["hybrid_override_pattern"])
    tensors = {}

    def mat(name, rows, cols, std=0.12):
        tensors[name] = ([rows, cols], [rnd.gauss(0, std) for _ in range(rows * cols)])

    def vec(name, size, base=0.0, std=0.2):
        tensors[name] = ([size], [base + rnd.gauss(0, std) for _ in range(size)])

    def mamba2(p):
        h, hd, g, ns, k = (cfg.get("mamba_n_heads") or cfg.get("mamba_num_heads"), cfg.get("mamba_d_head") or cfg.get("mamba_head_dim"),
                           cfg.get("mamba_n_groups") or cfg.get("n_groups"), cfg.get("mamba_d_state") or cfg.get("ssm_state_size"),
                           cfg.get("mamba_d_conv") or cfg.get("conv_kernel"))
        di = h * hd
        cd = di + 2 * g * ns
        mat(p + "in_proj.weight", di + cd + h, d)
        tensors[p + "conv1d.weight"] = ([cd, 1, k], [rnd.gauss(0, 0.4) for _ in range(cd * k)])
        vec(p + "conv1d.bias", cd)
        vec(p + "dt_bias", h, -0.5, 0.8)
        tensors[p + "A_log"] = ([h], [math.log(1 + 7 * rnd.random()) for _ in range(h)])
        vec(p + "D", h, 1.0, 0.3)
        vec(p + "norm.weight", di, 1.0)
        mat(p + "out_proj.weight", d, di)

    def attention(p):
        heads, kv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg.get("head_dim", d // cfg["num_attention_heads"])
        mat(p + "q_proj.weight", heads * hd, d)
        mat(p + "k_proj.weight", kv * hd, d)
        mat(p + "v_proj.weight", kv * hd, d)
        mat(p + "o_proj.weight", d, heads * hd)

    root = "backbone." if kind == "nemotron_h" else "model."
    mat(root + ("embeddings.weight" if kind == "nemotron_h" else "embed_tokens.weight"), vocab, d, 1.0)
    if kind == "granitemoehybrid":
        e, m, s = cfg["num_local_experts"], cfg["intermediate_size"], cfg["shared_intermediate_size"]
        for l in range(n):
            p = "model.layers.%d." % l
            vec(p + "input_layernorm.weight", d, 1.0)
            vec(p + "post_attention_layernorm.weight", d, 1.0)
            if cfg["layer_types"][l] == "mamba":
                mamba2(p + "mamba.")
            else:
                attention(p + "self_attn.")
            tensors[p + "block_sparse_moe.input_linear.weight"] = ([e, 2 * m, d], [rnd.gauss(0, 0.12) for _ in range(e * 2 * m * d)])
            tensors[p + "block_sparse_moe.output_linear.weight"] = ([e, d, m], [rnd.gauss(0, 0.12) for _ in range(e * d * m)])
            mat(p + "block_sparse_moe.router.layer.weight", e, d, 0.5)
            mat(p + "shared_mlp.input_linear.weight", 2 * s, d)
            mat(p + "shared_mlp.output_linear.weight", d, s)
        vec("model.norm.weight", d, 1.0)
    elif kind == "nemotron_h":
        for l, ch in enumerate(cfg["hybrid_override_pattern"]):
            p = "backbone.layers.%d." % l
            vec(p + "norm.weight", d, 1.0)
            if ch == "M":
                mamba2(p + "mixer.")
            elif ch == "*":
                attention(p + "mixer.")
            else:
                mat(p + "mixer.up_proj.weight", cfg["intermediate_size"], d)
                mat(p + "mixer.down_proj.weight", d, cfg["intermediate_size"])
        vec("backbone.norm_f.weight", d, 1.0)
    else:  # jamba
        e, f = cfg["num_experts"], cfg["intermediate_size"]
        di, ns, r, k = cfg["mamba_expand"] * d, cfg["mamba_d_state"], cfg["mamba_dt_rank"], cfg["mamba_d_conv"]
        for l in range(n):
            p = "model.layers.%d." % l
            vec(p + "input_layernorm.weight", d, 1.0)
            vec(p + "pre_ff_layernorm.weight", d, 1.0)
            if l % cfg["attn_layer_period"] == cfg["attn_layer_offset"]:
                attention(p + "self_attn.")
            else:
                q = p + "mamba."
                mat(q + "in_proj.weight", 2 * di, d)
                tensors[q + "conv1d.weight"] = ([di, 1, k], [rnd.gauss(0, 0.4) for _ in range(di * k)])
                vec(q + "conv1d.bias", di)
                mat(q + "x_proj.weight", r + 2 * ns, di)
                mat(q + "dt_proj.weight", di, r, 0.5)
                vec(q + "dt_proj.bias", di, -0.5, 0.8)
                tensors[q + "A_log"] = ([di, ns], [math.log(1 + (i % ns)) + rnd.gauss(0, 0.05) for i in range(di * ns)])
                vec(q + "D", di, 1.0, 0.3)
                mat(q + "out_proj.weight", d, di)
                vec(q + "dt_layernorm.weight", r, 1.0)
                vec(q + "b_layernorm.weight", ns, 1.0)
                vec(q + "c_layernorm.weight", ns, 1.0)
            if l % cfg["expert_layer_period"] == cfg["expert_layer_offset"]:
                mat(p + "feed_forward.router.weight", e, d, 0.5)
                for i in range(e):
                    ep = p + "feed_forward.experts.%d." % i
                    mat(ep + "gate_proj.weight", f, d)
                    mat(ep + "up_proj.weight", f, d)
                    mat(ep + "down_proj.weight", d, f)
            else:
                mat(p + "feed_forward.gate_proj.weight", f, d)
                mat(p + "feed_forward.up_proj.weight", f, d)
                mat(p + "feed_forward.down_proj.weight", d, f)
        vec("model.final_layernorm.weight", d, 1.0)
    if not cfg.get("tie_word_embeddings"):
        mat("lm_head.weight", vocab, d, 0.5)
    return tensors


def phi3(hidden, heads, kv, orig, prf, window, tied, scaled, theta=10000.0):
    hd = hidden // heads
    rot = int(hd * prf)
    cfg = dict(
        architectures=["Phi3ForCausalLM"], model_type="phi3", hidden_size=hidden, intermediate_size=192,
        num_hidden_layers=3, num_attention_heads=heads, num_key_value_heads=kv, vocab_size=1000,
        max_position_embeddings=512, original_max_position_embeddings=orig, rms_norm_eps=1e-5, rope_theta=theta,
        hidden_act="silu", attention_bias=False, tie_word_embeddings=tied, sliding_window=window,
        bos_token_id=1, eos_token_id=[2, 3], embd_pdrop=0.0, resid_pdrop=0.0, rope_scaling=None)
    if prf != 1.0:
        cfg["partial_rotary_factor"] = prf
    if scaled:
        n = rot // 2
        cfg["rope_scaling"] = dict(type="longrope",
                                   short_factor=[round(1.0 + 0.05 * i, 4) for i in range(n)],
                                   long_factor=[round(1.0 + 0.6 * i, 4) for i in range(n)])
    return cfg


FAMILIES["phi3"] = phi3(128, 4, 2, 32, 0.75, 6, False, True)
FAMILIES["phi3short"] = phi3(128, 4, 2, 128, 0.75, None, True, True)
FAMILIES["phi3mha"] = phi3(256, 4, 4, 512, 1.0, 6, False, False)
def mamba_tensors(cfg, rnd):
    """Random Mamba v1 weights with the real checkpoint's names and shapes (A_log = log(1..d_state) like the
    S4D-real init, dt_proj.bias spread so softplus(dt) covers small and large steps)."""
    d, di, ds, dc = cfg["hidden_size"], cfg["intermediate_size"], cfg["state_size"], cfg["conv_kernel"]
    r = math.ceil(d / 16) if cfg["time_step_rank"] == "auto" else cfg["time_step_rank"]
    tensors = {}

    def mat(name, shape, std):
        n = 1
        for x in shape:
            n *= x
        tensors[name] = (shape, [rnd.gauss(0, std) for _ in range(n)])

    def vec(name, shape, fn):
        n = 1
        for x in shape:
            n *= x
        tensors[name] = (shape, [fn(i) for i in range(n)])

    mat("backbone.embeddings.weight", [cfg["vocab_size"], d], 1.0)
    for l in range(cfg["num_hidden_layers"]):
        p = "backbone.layers.%d." % l
        vec(p + "norm.weight", [d], lambda i: 1.0 + rnd.gauss(0, 0.1))
        q = p + "mixer."
        mat(q + "in_proj.weight", [2 * di, d], 0.15)
        mat(q + "conv1d.weight", [di, 1, dc], 0.4)
        vec(q + "conv1d.bias", [di], lambda i: rnd.gauss(0, 0.1))
        mat(q + "x_proj.weight", [r + 2 * ds, di], 0.2)
        mat(q + "dt_proj.weight", [di, r], 0.6)
        vec(q + "dt_proj.bias", [di], lambda i: rnd.uniform(-3.0, 0.5))
        vec(q + "A_log", [di, ds], lambda i: math.log(i % ds + 1) + rnd.gauss(0, 0.05))
        vec(q + "D", [di], lambda i: 1.0 + rnd.gauss(0, 0.2))
        mat(q + "out_proj.weight", [d, di], 0.12)
    vec("backbone.norm_f.weight", [d], lambda i: 1.0 + rnd.gauss(0, 0.1))
    if cfg.get("tie_word_embeddings") is False:
        mat("lm_head.weight", [cfg["vocab_size"], d], 0.5)
    return tensors


def qwen3next_tensors(cfg, rnd):
    """Random Qwen3-Next weights with the released checkpoints' names and layouts: in_proj_qkvz / in_proj_ba grouped per
    key head, q_proj holding [query | gate] per head, conv1d [conv_dim, 1, K] without bias, zero-centred norm gains
    (applied as 1 + w) except the gated DeltaNet norm, per-expert tensors plus shared_expert / shared_expert_gate."""
    d, vocab, n_layers = cfg["hidden_size"], cfg["vocab_size"], cfg["num_hidden_layers"]
    heads, kv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    dk, dv, kc = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"], cfg["linear_conv_kernel_dim"]
    conv_dim = 2 * nk * dk + nv * dv
    types = cfg.get("layer_types") or [
        "full_attention" if (i + 1) % cfg["full_attention_interval"] == 0 else "linear_attention" for i in range(n_layers)]
    tensors = {}

    def mat(name, rows, cols, std=0.08):
        tensors[name] = ([rows, cols], [rnd.gauss(0, std) for _ in range(rows * cols)])

    def vec(name, n, base=0.0, std=0.1):
        tensors[name] = ([n], [base + rnd.gauss(0, std) for _ in range(n)])

    def mlp(p, inter):
        mat(p + "gate_proj.weight", inter, d)
        mat(p + "up_proj.weight", inter, d)
        mat(p + "down_proj.weight", d, inter)

    # a tied head reads the embedding back: smaller rows keep the last token from dominating its own logits
    mat("model.embed_tokens.weight", vocab, d, 0.25 if cfg["tie_word_embeddings"] else 1.0)
    for l in range(n_layers):
        p = "model.layers.%d." % l
        vec(p + "input_layernorm.weight", d)
        vec(p + "post_attention_layernorm.weight", d)
        if types[l] == "full_attention":
            a = p + "self_attn."
            mat(a + "q_proj.weight", heads * hd * 2, d, 0.12)
            mat(a + "k_proj.weight", kv * hd, d, 0.12)
            mat(a + "v_proj.weight", kv * hd, d)
            mat(a + "o_proj.weight", d, heads * hd)
            vec(a + "q_norm.weight", hd)
            vec(a + "k_norm.weight", hd)
        else:
            a = p + "linear_attn."
            mat(a + "in_proj_qkvz.weight", 2 * nk * dk + 2 * nv * dv, d, 0.12)
            mat(a + "in_proj_ba.weight", 2 * nv, d, 0.3)
            tensors[a + "conv1d.weight"] = ([conv_dim, 1, kc], [rnd.gauss(0, 0.4) for _ in range(conv_dim * kc)])
            vec(a + "dt_bias", nv, 0.0, 0.8)
            tensors[a + "A_log"] = ([nv], [math.log(rnd.uniform(0.1, 8.0)) for _ in range(nv)])
            vec(a + "norm.weight", dv, 1.0, 0.1)
            mat(a + "out_proj.weight", d, nv * dv, 0.1)
        moe = cfg["num_experts"] > 0 and l not in cfg["mlp_only_layers"] and (l + 1) % cfg["decoder_sparse_step"] == 0
        if moe:
            e, m = cfg["num_experts"], cfg["moe_intermediate_size"]
            mat(p + "mlp.gate.weight", e, d, 0.3)
            for i in range(e):
                mlp(p + "mlp.experts.%d." % i, m)
            mlp(p + "mlp.shared_expert.", cfg["shared_expert_intermediate_size"])
            mat(p + "mlp.shared_expert_gate.weight", 1, d, 0.2)
        else:
            mlp(p + "mlp.", cfg["intermediate_size"])
    vec("model.norm.weight", d)
    if not cfg["tie_word_embeddings"]:
        mat("lm_head.weight", vocab, d, 0.5)
    mat("mtp.fc.weight", d, 2 * d)  # the multi-token-prediction head: present in the released checkpoints, unused
    return tensors


def main():
    if len(sys.argv) < 3 or sys.argv[2] not in FAMILIES and sys.argv[2] != "gemma3mm":
        sys.exit(__doc__)
    out, fam = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1234
    rnd = random.Random(seed)
    wrapper = fam == "gemma3mm"
    cfg = dict(FAMILIES["gemma3" if wrapper else fam])
    if cfg["model_type"] == "mamba":
        write(out, fam, cfg, mamba_tensors(cfg, rnd))
        return
    if cfg["model_type"] == "qwen3_next":
        write(out, fam, cfg, qwen3next_tensors(cfg, rnd))
        return
    if cfg["model_type"] in ("granitemoehybrid", "nemotron_h", "jamba"):
        write(out, fam, cfg, hybrid_tensors(cfg, rnd))
        return
    if cfg["model_type"] == "gpt_oss":
        write(out, fam, cfg, gptoss_tensors(cfg, rnd, fam == "gptoss_mxfp4"))
        return
    if cfg["model_type"] == "llama4_text":
        wrap = fam == "llama4mix"  # the mixed one in the wrapper layout, the other as a plain Llama4ForCausalLM
        write(out, fam, cfg, llama4_tensors(cfg, rnd, wrap), llama4=wrap)
        return
    d, heads, kv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    if "--head-dim" in sys.argv:  # e.g. 256 to exercise the largest head size the Metal attention kernel supports
        cfg["head_dim"] = int(sys.argv[sys.argv.index("--head-dim") + 1])
    if "--vocab" in sys.argv:  # e.g. 262144 for Gemma 3's vocabulary size
        cfg["vocab_size"] = int(sys.argv[sys.argv.index("--vocab") + 1])
    if "--attention-bias" in sys.argv:
        cfg["attention_bias"] = True
    hd = cfg.get("head_dim", d // heads)
    glm = cfg["model_type"] in ("glm", "glm4")
    ffn, vocab = cfg["intermediate_size"], cfg["vocab_size"]
    bias = cfg["model_type"] == "qwen2" or (glm and cfg.get("attention_bias", True))
    # Llama / Qwen3 / Gemma read attention_bias for all four projections (Qwen2 and GLM: q/k/v only)
    obias = bool(cfg.get("attention_bias")) and cfg["model_type"] in ("llama", "qwen3", "qwen3_moe", "gemma2",
                                                                        "gemma3_text")
    bias = bias or obias
    fused = cfg["model_type"] == "phi3"
    gemma = cfg["model_type"].startswith("gemma")
    # Gemma stores norm gains as w and applies 1 + w, so random gains are centered on 0
    nb = 0.0 if gemma else 1.0
    pre = "language_model.model." if wrapper else "model."

    tensors = {}
    if fam.startswith("deepseek"):
        tensors = deepseek_tensors(cfg, rnd)
        write(out, fam, cfg, tensors)
        return

    def mat(name, rows, cols, std=0.08):
        tensors[name] = ([rows, cols], [rnd.gauss(0, std) for _ in range(rows * cols)])

    def vec(name, n, base=1.0, std=0.1):
        tensors[name] = ([n], [base + rnd.gauss(0, std) for _ in range(n)])

    mat(pre + "embed_tokens.weight", vocab, d, 1.0)
    for l in range(cfg["num_hidden_layers"]):
        p = pre + "layers.%d." % l
        vec(p + "input_layernorm.weight", d, nb)
        vec(p + "post_attention_layernorm.weight", d, nb)
        if gemma:
            vec(p + "pre_feedforward_layernorm.weight", d, nb)
            vec(p + "post_feedforward_layernorm.weight", d, nb)
        if cfg["model_type"] in ("gemma3_text", "qwen3_moe"):
            vec(p + "self_attn.q_norm.weight", hd, nb)
            vec(p + "self_attn.k_norm.weight", hd, nb)
        if fused:  # rows: q (heads*hd), then k (kv*hd), then v (kv*hd)
            mat(p + "self_attn.qkv_proj.weight", (heads + 2 * kv) * hd, d)
        else:
            mat(p + "self_attn.q_proj.weight", heads * hd, d)
            mat(p + "self_attn.k_proj.weight", kv * hd, d)
            mat(p + "self_attn.v_proj.weight", kv * hd, d)
        mat(p + "self_attn.o_proj.weight", d, heads * hd)
        if bias:
            vec(p + "self_attn.q_proj.bias", heads * hd, 0.0, 0.15)
            vec(p + "self_attn.k_proj.bias", kv * hd, 0.0, 0.15)
            vec(p + "self_attn.v_proj.bias", kv * hd, 0.0, 0.15)
        if obias:
            vec(p + "self_attn.o_proj.bias", d, 0.0, 0.15)
        if cfg["model_type"] == "glm4":
            vec(p + "post_self_attn_layernorm.weight", d)
            vec(p + "post_mlp_layernorm.weight", d)
        if cfg.get("num_experts") and l not in cfg["mlp_only_layers"]:
            e, m = cfg["num_experts"], cfg["moe_intermediate_size"]
            mat(p + "mlp.gate.weight", e, d, 0.3)
            for i in range(e):
                mat(p + "mlp.experts.%d.gate_proj.weight" % i, m, d)
                mat(p + "mlp.experts.%d.up_proj.weight" % i, m, d)
                mat(p + "mlp.experts.%d.down_proj.weight" % i, d, m)
            continue
        if glm or fused:
            mat(p + "mlp.gate_up_proj.weight", 2 * ffn, d)
        else:
            mat(p + "mlp.gate_proj.weight", ffn, d)
            mat(p + "mlp.up_proj.weight", ffn, d)
        mat(p + "mlp.down_proj.weight", d, ffn)
    vec(pre + "norm.weight", d, nb)
    if not cfg["tie_word_embeddings"]:
        mat("lm_head.weight", vocab, d, 0.5)
    if wrapper:
        mat("vision_tower.vision_model.embeddings.patch_embedding.weight", 32, 24, 0.1)
        mat("multi_modal_projector.mm_input_projection_weight", 32, d, 0.1)

    write(out, fam, cfg, tensors, wrapper)


def write(out, fam, cfg, tensors, wrapper=False, llama4=False):
    header, blobs, off = {}, [], 0
    for name, t in tensors.items():
        shape, vals = t[0], t[1]
        if len(t) > 2 and t[2] == "U8":  # raw bytes (MXFP4 blocks and scales)
            raw, dtype = bytes(vals), "U8"
        else:
            raw, dtype = struct.pack("<%df" % len(vals), *vals), "F32"
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
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
    if llama4:
        cfg = llama4_wrapper(cfg)
        cfg["torch_dtype"] = "float32"
        cfg["text_config"]["torch_dtype"] = "float32"
    if wrapper:
        cfg = gemma_wrapper(cfg)
        cfg["torch_dtype"] = "float32"
        cfg["text_config"]["torch_dtype"] = "float32"
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote %s (%s, %d tensors, %.1f MB)" % (out, fam, len(tensors), off / 1e6))


main()
