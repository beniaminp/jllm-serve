# 14. Model families (Llama, Mistral, Qwen2, Phi-3/4, Gemma 2/3, DeepSeek-R1 distills, DeepSeek-V2/V3, Qwen3-Next)

Code: `model/Architecture` (`Decoder`, `MambaArch`, `Qwen3NextArch`), `model/Families` (one parser per `model_type`),
`model/LayerSpec`, `model/Attention`, `model/Mlp`, `model/Router`, `model/RopeTable`, `model/Scaling`, `model/Rope`,
`model/ModelConverter`, `cpu/CpuBackend`, `cpu/MlaCpu`, `metal/MetalBackend`, `metal/MlaMetal`, `kernels.metal`,
`Main.eosIds`, `tokenizer/Tokenizer.promptBos`, `engine/Engine.preparePrompt`

All these models are the same decoder-only transformer that [chapter 5](05-transformer-math.md) describes:
embedding, N × (RMSNorm → attention → RMSNorm → SwiGLU MLP), final RMSNorm, LM head. They differ in a handful of
switches, all read from `config.json` by the family's parser in `Families` and stored as typed values in an
`Architecture` (section 14.12). Qwen3 is the reference; this chapter lists what changes. Section map: 14.1 the
families at a glance, 14.2 – 14.6 what the Llama-like families have in common (and what they lack), 14.7 how it is
tested, 14.8 GLM, 14.9 Phi-3/4, 14.10 Gemma, 14.11 DeepSeek-V2/V3, 14.12 the typed model and how to add a family,
14.13 the comparison with transformers on real published weights.
14.14 Qwen3-Next (Gated DeltaNet linear attention + gated attention + sparse MoE).
Qwen3-MoE is [chapter 12](12-mixture-of-experts.md), Mamba [chapter 15](15-mamba.md).

## 14.1 Supported families

One row per `model_type`; the parser that produces each row is a short method in `Families` (`llama`, `mistral`,
`qwen2`, `qwen3`, `glm`, `phi3`, `gemma`, `deepseek`, `mamba`) over a shared `Draft` whose defaults are Llama's. "Config"
means the value comes from `config.json`; defaults are those of the family's transformers config class.

| `model_type` | Architecture class | q/k/v bias | QK-norm | Sliding window | RoPE (pairing; scaling) | Norms | MLP | BOS (from the tokenizer) | Tied head | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `llama` | `LlamaForCausalLM` | `attention_bias` (default false; also `o_proj`) | no | none | `HALF`; `rope_scaling` `llama3` (3.1+), `linear` | `PRE` | dense SiLU | yes | config (default false) | |
| `mistral` | `MistralForCausalLM` | no | no | `sliding_window` on every layer (null in v0.3) | `HALF`; none | `PRE` | dense SiLU | yes | config (false) | |
| `qwen2` | `Qwen2ForCausalLM` | **always** (the key is absent) | no | layers ≥ `max_window_layers`, only if `use_sliding_window` | `HALF`; none | `PRE` | dense SiLU | no (R1-Distill-Qwen: yes) | config | |
| `qwen3` | `Qwen3ForCausalLM` | as `llama` | **yes** | same rule as `qwen2` | `HALF`; none | `PRE` | dense SiLU | no | config | |
| `qwen3_moe` | `Qwen3MoeForCausalLM` | as `llama` | yes | same rule as `qwen2` | `HALF`; none | `PRE` | MoE, `Router.Softmax`, `norm_topk_prob` from config; dense in `mlp_only_layers` and off the `decoder_sparse_step` grid | no | config | chapter 12 |
| `glm` | `GlmForCausalLM` | `attention_bias`, **default true** | no | none | `INTERLEAVED`, `partial_rotary_factor` default 0.5 | `PRE` | dense SiLU, fused `gate_up_proj` | no (`[gMASK]<sop>`) | config | eps default 1.5625e-7; 14.8 |
| `glm4` | `Glm4ForCausalLM` | as `glm` | no | none | as `glm` | `SANDWICH_GLM4` (4 norms) | dense SiLU, fused `gate_up_proj` | no | config | 14.8 |
| `phi3` | `Phi3ForCausalLM` | no | no | `sliding_window` on every layer | `HALF`; `longrope` (short/long factors), `partial_rotary_factor` | `PRE` | dense SiLU; fused `qkv_proj`, `gate_up_proj` | no | config | eps default 1e-5; 14.9 |
| `gemma2` | `Gemma2ForCausalLM` | as `llama` | no | layers `i % 2 == 0` are local (`sliding_window`), the rest global | `HALF`; one table, theta default 1e4 | `SANDWICH_GEMMA`, gains `1 + w` | dense **GeLU-tanh** | yes | **default true** | embedding × `bf16(√dim)`, attention soft-cap 50, final soft-cap 30, `query_pre_attn_scalar`; 14.10 |
| `gemma3_text` (also inside `gemma3`) | `Gemma3ForCausalLM` (wrapper `Gemma3ForConditionalGeneration`) | as `llama` | **yes** | 5 local + 1 global (`sliding_window_pattern` 6, or `layer_types`) | `HALF`; global theta default 1e6 + `rope_scaling`, local table `rope_local_base_freq` (1e4) without scaling | `SANDWICH_GEMMA`, gains `1 + w` | dense GeLU-tanh | yes | default true | soft-caps off unless set; text part only; 14.10 |
| `gemma4_text` / `gemma4_unified_text` (also inside `gemma4` / `gemma4_unified`) | `Gemma4ForCausalLM` / `Gemma4UnifiedForCausalLM` (wrappers `Gemma4ForConditionalGeneration` / `Gemma4UnifiedForConditionalGeneration`) | as `llama` (no `v_proj` on full layers: K = V) | **yes** (+ a parameterless V norm) | 5 local + 1 full, the last always full (`layer_types`) | per layer: local `head_dim` 256 / full `global_head_dim` 512; full layers use proportional RoPE (a quarter of the head) at theta 1e6, local layers a plain table at `rope_freq_base_swa` (1e4) | `SANDWICH_GEMMA`, gains are the stored `w` (**no** `1 + w`) | dense GeLU-tanh | yes | default true | unscaled attention (`scale = 1`), per-layer `layer_scalar`, final soft-cap 30; 12B dense, CPU and Metal (512-wide heads; paged/batched KV not offered); 14.10.7 |
| `deepseek_v2` | `DeepseekV2ForCausalLM` | rejected | no | none | MLA rotary slice `INTERLEAVED`, `yarn` | `PRE` | dense for the first `first_k_dense_replace` layers, then MoE + shared experts; `Router.DeepSeek` softmax, `greedy` / `group_limited_greedy` | yes | config (false) | MLA latent KV cache; 14.11 |
| `deepseek_v3` | `DeepseekV3ForCausalLM` | rejected | no | none | as V2 | `PRE` | as V2; sigmoid, `noaux_tc` (8 groups keep 4), scaling 2.5, `norm_topk_prob` default true | tokenizer rule | config (false) | FP8 checkpoints dequantized by the converter (14.11.7) |
| `mamba` | `MambaForCausalLM` | n/a | n/a | n/a (no attention) | none (no positions) | one RMSNorm per block | no MLP: conv + selective scan mixer | no | default true | `MambaArch`; chapter 15 |
| `qwen3_next` | `Qwen3NextForCausalLM` | rejected | yes (`1 + w`) | none | `HALF` on `partial_rotary_factor` (0.25) of `head_dim` (256) | `PRE`, gains `1 + w` | MoE + gated shared expert (Qwen3-MoE layer rule) | no | config (false) | `Qwen3NextArch`: 3 of 4 layers Gated DeltaNet (no KV cache), the rest attention with an output gate; 14.14 |

Reading the table: *Norms* are the `LayerSpec.Norms` value of 14.12 (`PRE` = `input_layernorm` + `post_attention_layernorm`
before the MLP). *RoPE* `HALF` pairs element `i` with `i + rotaryDim/2`, `INTERLEAVED` pairs `2i` with `2i + 1` (14.8); every
family reads `rope_theta`, `partial_rotary_factor` and `rope_scaling` from the top level of `config.json` or, in
transformers 5 configs, from `rope_parameters`. *BOS* is not a family rule: see the BOS bullet in 14.2.

The checkpoints each row was verified against:

| `model_type` | Examples | Checked against |
|---|---|---|
| `llama` | Llama 3 / 3.1 / 3.2, DeepSeek-R1-Distill-Llama-8B, DeepSeek-Coder-1.3B | `unsloth/Llama-3.2-1B-Instruct`, `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| `mistral` | Mistral-7B-Instruct v0.1 – v0.3 | `unsloth/mistral-7b-instruct-v0.3` |
| `qwen2` | Qwen2 / Qwen2.5, DeepSeek-R1-Distill-Qwen | `Qwen/Qwen2.5-1.5B-Instruct`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| `qwen3` | Qwen3 dense | `Qwen/Qwen3-0.6B` |
| `qwen3_moe` | Qwen3-30B-A3B (chapter 12) | `Qwen/Qwen3-30B-A3B-Thinking-2507`, a tiny random checkpoint |
| `glm` | GLM-Edge-1.5B/4B-Chat, glm-4-9b-chat-hf (14.8) | `THUDM/GLM-Edge-1.5B-Chat`, `THUDM/glm-4-9b-chat-hf` |
| `glm4` | GLM-4-9B/32B-0414, GLM-Z1 (14.8) | `THUDM/GLM-4-9B-0414`, `THUDM/GLM-Z1-9B-0414` |
| `phi3` | Phi-3-mini, Phi-3.5-mini, Phi-4-mini, phi-4 (14B) (14.9) | `microsoft/Phi-3-mini-4k-instruct`, `Phi-3.5-mini-instruct`, `Phi-4-mini-instruct`, `phi-4` |
| `gemma2` | Gemma 2 2B / 9B / 27B (14.10) | `unsloth/gemma-2-2b-it` (mirror of the gated `google/gemma-2-2b-it`) |
| `gemma3_text` | Gemma 3 1B / 4B / 12B / 27B (14.10) | `unsloth/gemma-3-1b-it`, `unsloth/gemma-3-4b-it` |
| `deepseek_v2` | DeepSeek-V2-Lite, DeepSeek-Coder-V2-Lite-Instruct (14.11) | config, safetensors header and `modeling_deepseek.py` on the Hub; tiny random models |
| `deepseek_v3` | DeepSeek-V3 / R1 (FP8 or bf16; too large to run here, 14.11.7) | config and header only; tiny random models |
| `mamba` | `state-spaces/mamba-130m-hf` … `mamba-2.8b-hf` | chapter 15 |
| `qwen3_next` | Qwen3-Next-80B-A3B Instruct / Thinking (14.13) | config, tensor index and safetensors headers of `Qwen/Qwen3-Next-80B-A3B-Instruct`; tiny random models against transformers 5.20 |
| `gpt_oss` | gpt-oss-20b / 120b (14.18) | config, chat template, tokenizer of `openai/gpt-oss-20b`; tiny random models (bf16 and MXFP4 experts) against transformers 5.20 |
| `llama4_text` (wrapper `Llama4ForConditionalGeneration`) | Llama-4-Scout / Maverick text (14.19) | config, chat template, tokenizer of Scout; tiny random models against transformers 5.20 |
| `granitemoehybrid`, `nemotron_h`, `jamba` | Granite 4.0-H, Nemotron-H 8B, Jamba (14.22) | configs (and Granite's chat template / tokenizer); tiny random models against transformers 5.20 |

A different `model_type`, a `hidden_act` other than `silu` (Gemma reads `hidden_activation`, which must be
`gelu_pytorch_tanh`), `mlp_bias: true`, `attention_bias: true` for `mistral`, `phi3` or DeepSeek (transformers ignores
it for the first two; DeepSeek would also bias `q_a_proj` / `kv_a_proj_with_mqa`), or a `rope_scaling` type other than `default`/`linear`/`llama3`/`longrope`/`yarn` (`su` is the legacy spelling of
`longrope`; `yarn` follows 14.11.2 in every family) is rejected at load time with a message naming the field.

Other requirements: `hidden_size`, `intermediate_size` and `num_heads × head_dim` must be multiples of 32 (the
quantization block), and the Metal backend additionally needs `head_dim` to be a multiple of 32 and at most 512
(Llama 3, Mistral, Qwen2.5 and DeepSeek-Coder all have 64 or 128, Gemma 4's full-attention layers 512; DeepSeek's MLA has its own limits, 14.11.1).

## 14.2 What differs from Qwen3

| Feature | Llama 3.x | Mistral | Qwen2 / 2.5 | Qwen3 |
|---|---|---|---|---|
| q/k/v bias (`attention_bias`) | no (q/k/v/o if set) | no | **yes** (always, even though the key is absent; no `o_proj` bias) | no (q/k/v/o if set) |
| per-head q/k RMSNorm | no | no | no | **yes** |
| sliding window | no | `sliding_window` (null in v0.3), all layers | only if `use_sliding_window`, layers ≥ `max_window_layers` | same rule as Qwen2 |
| RoPE scaling | `llama3` (3.1+) | none | none | none |
| `head_dim` | `hidden/heads`, or explicit (3.2: 64) | explicit in v0.3 (128) | `hidden/heads` | explicit (128) |
| `tie_word_embeddings` | true for 1B/3B, false for 8B+ | false | true for ≤ 3B | per size |
| BOS in front of a plain prompt (the tokenizer's `add_bos_token`) | **yes** (`<\|begin_of_text\|>`) | **yes** (`<s>`) | no | no |
| chat template | `<\|start_header_id\|>` style | `[INST]` style | ChatML | ChatML |
| stop tokens | `<\|eot_id\|>`, `<\|end_of_text\|>`, `<\|eom_id\|>` | `</s>` | `<\|im_end\|>`, `<\|endoftext\|>` | same as Qwen2 |

How each is implemented:

- **Bias.** `Attention.Gqa.bias` is true for `qwen2` (always) and for `glm` / `glm4` (`attention_bias`, default true);
  the converter stores `self_attn.{q,k,v}_proj.bias` as f32 (they are tiny); the backends add them right after the
  q/k/v matmuls and *before* RoPE. CPU: `addBias`, a loop over the freshly written rows; Metal: the bias is an
  epilogue argument of the fused `matmul3` kernel that projects q, k and v, so it costs no dispatch.
- **`o_proj` bias.** In transformers' Llama, Qwen3 (and Qwen3-MoE) and Gemma 2 / 3 attention one flag,
  `attention_bias`, biases all four projections; Qwen2 and GLM hard-code `o_proj` without one. So for `llama`, `qwen3`
  and `gemma` (`Draft.Bias.CONFIG_QKVO`) `attention_bias: true` sets both `Attention.Gqa.bias` and `outBias`, the
  converter also keeps `self_attn.o_proj.bias [dim]` as f32 (`DecoderLayer.bo`), and the description gains `obias`.
  The bias belongs to the projection, so it is added *before* the residual add and, for Gemma's sandwich norms,
  before the post-attention norm:

  ```
  plain:      x = x + (W_o · att + b_o)
  sandwich:   x = x + rmsnorm(W_o · att + b_o) · g_post
  ```

  CPU (`CpuBackend.addPlain` / `addPostNormed`): with a bias the product is staged in `xb` and `x[t, i] += xb[t, i] +
  b_o[i]`, or `b_o` is added to `xb` before `addNormed`. Metal: `MatmulParams.hasBias` and a bias buffer (index 5) in
  `matmul`, `matmul_pf` and `matmul_mma`, added in the epilogue (`v + b`, then accumulated into `x` or stored for
  `rmsnorm_add`), so it costs no dispatch either. The MLA layers share this code (`DecoderLayer.bo`), but DeepSeek keeps
  rejecting `attention_bias`. Checked by the `LlamaBias` and `Gemma2Bias` tiny models of the arch smoke (CPU == Metal,
  14.7) and `SmokeTest.modelConfigs`.
- **QK-norm.** `Attention.Gqa.qkNorm` is true for `qwen3` / `qwen3_moe` and `gemma3_text`; `ModelWeights` checks that
  the `q_norm`/`k_norm` tensors exist exactly when it is set. The norm is applied per head right before RoPE (on Metal
  inside the `rope` kernel, so the same dispatch).
- **Tied embeddings.** `tie_word_embeddings: true` means there is no `lm_head.weight`; the LM head reuses
  `model.embed_tokens.weight` (quantized to Q8_0 once, shared by both uses).
- **BOS.** There is no per-family list. `Tokenizer.promptBos()` answers from the tokenizer's own files: the `bos_token`
  and `add_bos_token` of `tokenizer_config.json`, or, when that key is absent, whether the `tokenizer.json`
  post-processor template starts with a special token. `Engine.preparePrompt` prepends that id to a plain prompt unless
  it already starts with it (a chat template that emits the BOS token itself is therefore not doubled). So Llama 3,
  Mistral, Gemma and DeepSeek-V2 get one, Qwen, Phi-3/4, GLM and Mamba do not, and DeepSeek-R1-Distill-Qwen, whose
  `model_type` is `qwen2`, does because its tokenizer asks for `<｜begin▁of▁sentence｜>`. `Core.bosTokenId` is only
  the config's `bos_token_id`, kept for messages.
- **Stop tokens.** `Main.eosIds` takes the union of `eos_token_id` from `config.json` (an int or a list),
  `generation_config.json`, the end-of-turn tokens of the chat template (added by `Engine`) and, for base models without
  a template, whichever of `<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`, `<|end_of_text|>`, `</s>` exist in the vocabulary.

## 14.3 Sliding-window attention

Full attention lets token `p` read every key at positions `0 … p`. With a window `W`, it reads only the last `W`:

```
position q:      0 1 2 3 4 5 6 7 8 9
token p = 9, W = 4:            . . . . . . x x x x      attends to q in (p - W, p] = 6..9
```

In the code the score loop for token `p` (`len = p + 1` keys) starts at `first = max(0, len - W)` instead of 0; the
softmax and the weighted sum use the same range, so masked positions contribute exactly nothing (they are not
set to -inf and then softmaxed, they are never visited). On Metal the window is the `window` field of `AttnParams`
and every loop of the attention kernels starts at `first + simdgroup`.

The KV cache still holds the full context (`--ctx` positions); only the reads shrink. A ring buffer of `W`
entries per layer would also cut memory, but needs position-dependent indexing and is a later optimization.
Stacking `L` windowed layers lets information travel `L × W` tokens, which is why a 4096 window still sees far
more than 4096 tokens back in a 32-layer model.

`Attention.Gqa.window` (per layer, 0 = full context): Mistral and Phi-3 apply `sliding_window` to all layers; Qwen2 and
Qwen3 apply it to layers `l ≥ max_window_layers` only when `use_sliding_window` is true (Qwen2.5 checkpoints set it to
false, so the `sliding_window: 32768` in their config is inert); Gemma marks layers local or global (14.10). The window
follows transformers' mask `kv > q - window`: a token sees `window` positions including itself.

## 14.4 Llama 3 RoPE scaling

Llama 3.1+ extends the 8192-token training context to 128K by changing the rotary frequencies. Plain RoPE uses
`f_i = theta^(-2i/d)` for `i = 0 … d/2-1`. Each pair rotates with wavelength `λ_i = 2π / f_i` tokens. The `llama3`
rule (transformers' `_compute_llama3_parameters`), with `orig = original_max_position_embeddings`:

```
λ_high = orig / high_freq_factor          (e.g. 8192 / 4 = 2048)
λ_low  = orig / low_freq_factor           (e.g. 8192 / 1 = 8192)

λ_i < λ_high           f'_i = f_i                                  fast rotations: keep
λ_i > λ_low            f'_i = f_i / factor                         slow rotations: stretch
otherwise              s    = (orig / λ_i − low_freq_factor) / (high_freq_factor − low_freq_factor)
                       f'_i = (1 − s) · f_i / factor + s · f_i     smooth blend
```

Worked example, Llama-3.2-1B (`d = 64`, `theta = 500000`, `factor = 32`, `low = 1`, `high = 4`, `orig = 8192`):

| i | `f_i` | `λ_i` | band | `f'_i` |
|---|---|---|---|---|
| 0 | 1.0 | 6.3 | keep | 1.0 |
| 10 | 1.656e-2 | 379 | keep | 1.656e-2 |
| 15 | 2.131e-3 | 2948 | blend, `s = (8192/2948 − 1)/3 = 0.5928` | `0.4072·2.131e-3/32 + 0.5928·2.131e-3 = 1.2905e-3` |
| 17 | 9.385e-4 | 6695 | blend, `s = 0.0745` | 9.708e-5 |
| 18 | 6.228e-4 | 10089 | stretch | 6.228e-4 / 32 = 1.946e-5 |
| 31 | 3.014e-6 | 2.08e6 | stretch | 9.418e-8 |

`Rope(RopeTable, ctx)` computes the whole `[ctx × rotaryDim/2]` cos/sin table once from `f'` (`Scaling.Llama3` in the
`RopeTable`), so neither backend knows about scaling. `linear` scaling (`f' = f / factor`, `Scaling.Linear`) is also
accepted; it is the only scaling Gemma 3's global layers use. `config.json` may spell the key `rope_type` or the older
`type`. `SmokeTest.ropeScaling` checks the rule against values computed by hand.

## 14.5 Tensor names and shapes

All families use the Hugging Face layout (`N` = layer index, `d` = hidden, `q = heads·head_dim`,
`kv = kv_heads·head_dim`, `f` = intermediate size, `V` = vocab). Verified from the safetensors headers and index
files of the repositories listed in 14.1.

| Tensor | Shape | Llama | Mistral | Qwen2 | Qwen3 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | V × d | x | x | x | x |
| `model.layers.N.input_layernorm.weight` | d | x | x | x | x |
| `model.layers.N.self_attn.q_proj.weight` | q × d | x | x | x | x |
| `model.layers.N.self_attn.k_proj.weight` | kv × d | x | x | x | x |
| `model.layers.N.self_attn.v_proj.weight` | kv × d | x | x | x | x |
| `model.layers.N.self_attn.o_proj.weight` | d × q | x | x | x | x |
| `model.layers.N.self_attn.q_proj.bias` | q | | | x | |
| `model.layers.N.self_attn.k_proj.bias` | kv | | | x | |
| `model.layers.N.self_attn.v_proj.bias` | kv | | | x | |
| `model.layers.N.self_attn.q_norm.weight` / `k_norm.weight` | head_dim | | | | x |
| `model.layers.N.post_attention_layernorm.weight` | d | x | x | x | x |
| `model.layers.N.mlp.gate_proj.weight` | f × d | x | x | x | x |
| `model.layers.N.mlp.up_proj.weight` | f × d | x | x | x | x |
| `model.layers.N.mlp.down_proj.weight` | d × f | x | x | x | x |
| `model.norm.weight` | d | x | x | x | x |
| `lm_head.weight` | V × d | only if untied | x | only if untied | only if untied |

Concrete shapes seen on the Hub (all BF16):

| Repository | d | heads/kv | head_dim | f | V | layers | tied | lm_head |
|---|---|---|---|---|---|---|---|---|
| `unsloth/Llama-3.2-1B-Instruct` | 2048 | 32/8 | 64 | 8192 | 128256 | 16 | yes | absent |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 4096 | 32/8 | 128 | 14336 | 128256 | 32 | no | present |
| `unsloth/mistral-7b-instruct-v0.3` | 4096 | 32/8 | 128 | 14336 | 32768 | 32 | no | present |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1536 | 12/2 | 128 | 8960 | 151936 | 28 | yes | absent |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1536 | 12/2 | 128 | 8960 | 151936 | 28 | no | present |

In the quantized cache (`jllm-q4_0.bin`) gate and up are fused into `mlp.gate_up` (`2f × d`), biases keep their
checkpoint names as f32, and the embedding/LM head are always Q8_0.

## 14.6 Not supported (yet)

- State-space models are a different architecture and have their own chapter: **Mamba** (`model_type: mamba`,
  `state-spaces/mamba-*-hf`) is supported, see [15](15-mamba.md); so are the attention / state-space hybrids with
  Mamba-2 (or Mamba-1) layers, Granite 4, Nemotron-H and Jamba ([14.22](#1422-attention--state-space-hybrids-granite-4-nemotron-h-jamba)),
  and Qwen3-Next's Gated DeltaNet ([14.14](#1414-qwen3-next-hybrid-gated-deltanet--gated-attention--sparse-moe)). A plain
  Mamba-2 model (`model_type: mamba2`) and Falcon-Mamba are not.
- `dynamic` RoPE scaling, YaRN with `truncate: false`; `hidden_act`
  other than `silu` (Gemma's `gelu_pytorch_tanh` via `hidden_activation` is the one exception); MLP bias; Llama 4's vision tower (its text tower is supported, [14.19](#1419-llama-4-text-chunked-attention-nope-layers-temperature-tuning-sigmoid-moe)); a `head_dim` that is not a multiple of 32 (Mistral Nemo-style) on Metal (CPU works).
- Tokenizers: Llama 3 uses a tiktoken-style byte-level BPE (close to Qwen's); Mistral and Llama 2 use SentencePiece
  vocabularies with byte fallback. That is the tokenizer's job (chapter 4), not the model's.
- Hub test checkpoints: `hf-internal-testing/tiny-random-LlamaForCausalLM` has `hidden_size = 16`, which is not a
  multiple of the 32-wide quantization block; the Mistral one ships only `pytorch_model.bin` with
  `hidden_act: gelu`; there is no Qwen2 repo of that name. `scripts/make-tiny-model.py` therefore writes random
  float32 checkpoints with the same config keys, and `scripts/arch-smoke.sh` runs them (14.7).

## 14.7 Testing

```
scripts/arch-smoke.sh
```

`scripts/make-tiny-model.py` writes random float32 checkpoints with the real tensor names and the config keys of each
family; `scripts/arch-smoke.sh` generates 20 of them (21 directories are passed, the downloaded Hub
`tiny-random-LlamaForCausalLM` is skipped because its `hidden_size` 16 is not a multiple of the 32-wide quantization
block), converts each to **q4, q8 and q2k** (`ArchSmoke`), loads it, generates 8 tokens greedily from a fixed 12-token
prompt on the CPU backend and on Metal and requires equal token ids and final logits that agree to 1e-3 relative: **20
models × 3 quantizations = 60 combinations, 60/60 CPU == Metal**. Random weights give gibberish, but the computation is
deterministic, so any disagreement is a bug in a backend. The models:

| family | tiny models (what each isolates) |
|---|---|
| Llama / Mistral / Qwen2 | plain Llama (GQA, untied head); Llama 3 `rope_scaling` (MHA, tied); Llama with `attention_bias` (q/k/v and `o_proj` biases); Mistral with `sliding_window = 4`; Qwen2 with bias and a window on layers ≥ 2 |
| GLM (14.8) | `glm` (bias, `head_dim` 32, factor 0.5); `glmedge` (`head_dim` 64, factor 1.0, no bias, tied); `glm4` (four norms) |
| Phi-3 (14.9) | fused weights + longrope long factors + partial rotary + window; the same with short factors and tied embeddings; `head_dim` 64 MHA with plain RoPE |
| Gemma (14.10) | Gemma 2 (`head_dim` 64, cap 1.0 so that it matters on random weights, final cap 30, local layers 0 and 2 with window 6); Gemma 3 text (q/k norm, 5 local + 1 global, linear ×8 global and 10 000-base local RoPE); the multimodal wrapper layout; `head_dim` 256 variants of both; Gemma 2 with `attention_bias` (the `o_proj` bias before the post-attention norm) |
| DeepSeek (14.11) | `deepseek_v2` (`q_lora_rank` null, group-limited routing, 2 shared experts), `deepseek_v2q` (q LoRA, `norm_topk_prob`, scaling 2.5), `deepseek_v3` (sigmoid, `noaux_tc`) |
| Mamba (chapter 15) | `mamba` (d_model 64, tied head) and `mamba3` (d_model 128, state 8, conv 3, untied head), plus chunking-invariance and state checks |

`mvn test` adds the per-family unit checks that have a hand-computed or external reference: `SmokeTest.ropeScaling`
(llama3, 14.4), `glmRope` (14.8), `longRope` and `phi3Configs` (14.9), `GemmaChecks` (14.10), `yarnRope`,
`deepseekRouting`, `deepseekConfigs` (14.11), `modelConfigs`, and `ConverterTest` (conversion, resume, the tiny Qwen3-MoE
checkpoint of chapter 12). The DeepSeek models are in addition compared with an independent float64 reference
(`scripts/deepseek-ref.py`, `DeepSeekRef`, 14.11.8). Family-specific verification is described in each family's own
"Testing" subsection; the comparison with transformers on real published checkpoints is 14.13.

## 14.8 GLM (`glm`, `glm4`)

Code: `Families.glm`, `model/InterleavedRope`, `RopeTable.pairing/rotaryDim`, `LayerSpec.Norms.SANDWICH_GLM4`,
`CpuBackend.addNormed`, `kernels.metal` (`rope_il`, `rmsnorm_add`), `ModelConverter.fusedGateUp`

Checked against transformers' `modeling_glm.py` / `modeling_glm4.py` and the Hub `config.json` and safetensors
headers of `THUDM/GLM-Edge-1.5B-Chat`, `THUDM/glm-4-9b-chat-hf`, `THUDM/GLM-4-9B-0414`, `THUDM/GLM-Z1-9B-0414`
(no weights were downloaded).

### 14.8.1 Interleaved, partial RoPE

Every family so far uses the "rotate half" convention: element `i` of a head is paired with element `i + d/2`.
GLM pairs *adjacent* elements, and rotates only the first `partial_rotary_factor × head_dim` of them:

```
head_dim = 8, partial_rotary_factor = 0.5 (rotary_dim = 4), dimensions 0..7 of one head

rotate half (Llama, Qwen, ...)          interleaved + partial (GLM)
  pairs: (0,4) (1,5) (2,6) (3,7)          pairs: (0,1) (2,3)      elements 4..7 untouched
  [0 1 2 3 | 4 5 6 7]                     [0 1 | 2 3 | 4 5 6 7]
   `-----------'  angle f0                  `-'   angle f0  (pair 0)
     `-----------'  angle f1                      `-'  angle f1  (pair 1)
```

`x'[2i] = x[2i] cos - x[2i+1] sin`, `x'[2i+1] = x[2i+1] cos + x[2i] sin`; elements `>= rotary_dim` pass through.
The frequencies are `f_i = theta^(-2i / rotary_dim)` with `rotary_dim = head_dim × factor`, *not* divided by
`head_dim` (transformers builds `inv_freq` from `int(head_dim × factor)`). `Rope` therefore builds tables of
`rotary_dim / 2` entries per position, and the same tables serve both backends (`rope_il`, the `ADJACENT` variant of
the `rope` kernel, on Metal, one thread per rotated pair; `InterleavedRope.rotate` on the CPU). The table carries its
`RopeTable.Pairing.INTERLEAVED`; the rotate-half path of the other families is the same kernel with `HALF`.

`partial_rotary_factor` is 0.5 when the key is absent (the config class sets it for backward compatibility), which
is the case for `glm-4-9b-chat-hf`; GLM-Edge sets 1.0 explicitly and GLM-4-0414 sets 0.5. Newer transformers nest
`rope_theta` and `partial_rotary_factor` in `rope_parameters`; both layouts are read.

### 14.8.2 Norms

`glm` has the usual two norms. `glm4` has four per layer; the two extra ones are applied to the sub-layer output
*before* it is added to the residual stream:

```
glm :  x = x + o_proj(attn(input_layernorm(x)))              x = x + mlp(post_attention_layernorm(x))
glm4:  x = x + post_self_attn_layernorm(o_proj(attn(input_layernorm(x))))
       x = x + post_mlp_layernorm(mlp(post_attention_layernorm(x)))
```

The engine projects into the scratch buffer `xb`, then adds `rmsnorm(xb) * w` to `x` (`rmsnorm_add` on Metal).
Models without these norms keep the fused `matmul(..., accumulate)` path.

### 14.8.3 Other differences

- q/k/v bias when `attention_bias` is true (the config default if the key is missing); `o_proj` has no bias. GLM-Edge
  has `attention_bias: false`, glm-4-9b-chat-hf and GLM-4-0414 have `true`.
- MLP: `mlp.gate_up_proj.weight` is `[2f, d]`, gate rows first, then up rows (`gate, up = chunk(2)`): the layout of
  our `mlp.gate_up`, so it is quantized as is. `mlp.down_proj.weight` as usual, activation `silu`.
- `head_dim` is explicit (128) in all four checkpoints (GQA with 4 kv heads for Edge-1.5B, 2 for the 9B models); the
  engine takes it from the config rather than `hidden/heads`.
- No BOS token: the chat templates start with `[gMASK]<sop>` (GLM-4) or `<|user|>` (GLM-Edge) and the tokenizers do not
  ask for a BOS, so `Tokenizer.promptBos()` is -1 (14.2). Stop ids come from `eos_token_id` (`[59246, 59253, 59255]` for Edge, `[151329, 151336, 151338]` for the others)
  plus the template's stop tokens. `pad_token_id` is ignored.
- `tie_word_embeddings`: true for GLM-Edge-1.5B-Chat (the checkpoint nevertheless contains `lm_head.weight`, which
  the converter skips), false for the 9B models.
- `rms_norm_eps` differs per checkpoint: 1e-5 (Edge, 0414), 1.5625e-7 (glm-4-9b-chat-hf), which is also the value
  `Families.glm` uses when the key is absent.

### 14.8.4 Verified configuration

| Repository | model_type | d | heads/kv | head_dim | f | V | layers | rotary factor | bias | tied | norms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `THUDM/GLM-Edge-1.5B-Chat` | glm | 2048 | 16/4 | 128 | 6144 | 59264 | 28 | 1.0 (explicit) | no | yes | 2 |
| `THUDM/glm-4-9b-chat-hf` | glm | 4096 | 32/2 | 128 | 13696 | 151552 | 40 | 0.5 (default) | yes | no | 2 |
| `THUDM/GLM-4-9B-0414` | glm4 | 4096 | 32/2 | 128 | 13696 | 151552 | 40 | 0.5 | yes | no | 4 |
| `THUDM/GLM-Z1-9B-0414` | glm4 | 4096 | 32/2 | 128 | 13696 | 151552 | 40 | 0.5 | yes | no | 4 |

Tensors per layer (header of shard 1, bf16): `input_layernorm.weight` `[d]`, `self_attn.{q,k,v}_proj.weight` and
`.bias` (only with bias), `self_attn.o_proj.weight` `[d, q]`, `post_attention_layernorm.weight`,
`mlp.gate_up_proj.weight` `[2f, d]`, `mlp.down_proj.weight` `[d, f]`, and for `glm4` also
`post_self_attn_layernorm.weight` and `post_mlp_layernorm.weight` (both `[d]`); globally `model.embed_tokens.weight`,
`model.norm.weight`, `lm_head.weight` (untied models).

### 14.8.5 Testing

The three tiny models `glm`, `glmedge` and `glm4` of the arch smoke (14.7) cover bias or not, `head_dim` 32 or 64 with
`hidden/heads` = 32 (explicit `head_dim`), factor 0.5 or 1.0, tied or untied and two or four norms.
`SmokeTest.glmRope` checks the rotation on a hand-computed 6-element head and the configs above. A pure-Python float32
re-implementation of the transformers forward pass on the same tiny checkpoints gives the same first generated token
for all three.

## 14.9 Phi-3 / Phi-3.5 / Phi-4 (`model_type: phi3`)

Code: `Families.phi3`, `Scaling.LongRope` with `Rope.invFreq` / `Rope.attentionFactor`, `RopeTable.rotaryDim`,
`ModelConverter.planDecoder` (the `qkv_proj` row split), `kernels.metal` (`rope`).

Same decoder as Llama (RMSNorm, GQA, SwiGLU, no biases, no QK-norm); four things differ. Verified against
`modeling_phi3.py` and `_compute_longrope_parameters` in transformers (main) and the Hub files of the four
repositories below.

### 14.9.1 Fused projections

The checkpoint stores `self_attn.qkv_proj.weight` `[(heads + 2·kv_heads)·head_dim, d]` with the rows ordered
q (`heads·head_dim`), k (`kv_heads·head_dim`), v (`kv_heads·head_dim`) (`qkv[..., :q]`, `qkv[..., q:q+kv]`,
`qkv[..., q+kv:]` in the model code), and `mlp.gate_up_proj.weight` `[2·f, d]` with gate first, then up
(`up_states.chunk(2, dim=-1)` → `gate, up`). There is no `q_proj`/`k_proj`/`v_proj`/`gate_proj`/`up_proj`.

The converter splits qkv by rows into the usual `q_proj`/`k_proj`/`v_proj` cache tensors (so the backends see a
Llama layer) and quantizes `gate_up_proj` directly as `mlp.gate_up`, which is already the stacked gate/up layout.
No other Phi tensor needs special handling: `input_layernorm`, `post_attention_layernorm`, `o_proj`, `down_proj`,
`model.norm`, `model.embed_tokens` and (untied models) `lm_head` have the Llama names.

### 14.9.2 LongRoPE (`rope_scaling.type` `longrope`, legacy `su`)

`rope_scaling` carries `short_factor` and `long_factor`, one divisor per rotary pair (`rotary_dim/2` entries), and
`original_max_position_embeddings` (in `rope_scaling` or at the top level of `config.json`). With `rot` the number
of rotated dimensions and `ext` the chosen factor array:

```
inv_freq[i] = 1 / (ext[i] · theta^(2i/rot))                       i = 0 … rot/2 − 1
ext         = long_factor   if context > original_max_position_embeddings, else short_factor
factor      = rope_scaling.factor, or max_position_embeddings / original_max_position_embeddings
attn        = attention_factor                         if given
              1                                        if factor <= 1
              sqrt(1 + ln(factor) / ln(original_max_position_embeddings))   otherwise
cos, sin   *= attn      (applied to the table, so q·k scores are scaled by attn²)
```

transformers chooses `ext` per forward pass from the sequence length. This engine builds one table per backend,
so it chooses once from `--ctx`: contexts up to the original length (4096 for Phi-3.5 / Phi-4-mini) use the short
factors, longer ones use the long factors for *all* positions. That differs from transformers only for a
conversation shorter than the original length inside a long `--ctx` (the long factors were trained for long inputs;
the short ones are slightly better for short text), so use `--ctx 4096` when you do not need more. The attention
factor is always applied, as in transformers.

Worked example, Phi-3.5-mini (`rot = 96`, `theta = 10000`, `max_position_embeddings = 131072`, `original = 4096`):
`factor = 131072 / 4096 = 32`, `attn = sqrt(1 + ln 32 / ln 4096) = sqrt(1 + 3.4657/8.3178) = sqrt(17/12) = 1.19024`.

| pair | `theta^(-2i/96)` | `short_factor` | `long_factor` | `inv_freq` short | `inv_freq` long |
|---|---|---|---|---|---|
| 0 | 1.0 | 1.0 | 1.08 | 1.0 | 0.9259 |
| 47 | 1.2115e-4 | 2.84 | 64.84 | 4.266e-5 | 1.868e-6 |

Phi-4-mini uses the same rule with 48 pairs of `rot = 96` (see below) and long factors up to 47.77.
`SmokeTest.longRope` checks the formula against hand-computed values (`sqrt(17/12)`, short and long arrays, the table
entries), `SmokeTest.phi3Configs` parses the real config shapes.

### 14.9.3 Partial rotary (Phi-4-mini)

`partial_rotary_factor: 0.75` with `head_dim = 128` rotates only the first `rot = int(128 · 0.75) = 96` dimensions
of every query and key head, with rotate-half *inside that prefix* (dimension `i` pairs with `i + 48`); dimensions
96 … 127 pass through unchanged. The inverse frequencies use `rot` in the exponent (`theta^(2i/rot)`), not
`head_dim`. `RopeTable.rotaryDim` (from `partial_rotary_factor`) is shared with GLM (14.8); Phi keeps
`Pairing.HALF`. `Rope` builds `[ctx × rot/2]` tables, `CpuBackend.normRope` pairs `(i, i + rot/2)`, and Metal uses the
`rope` kernel (q and k heads together) with `rotHalf = rot/2`, so the pairs are `(i, i + rot/2)` instead of GLM's
`(2i, 2i+1)`. Partial rotation together with a QK-norm is rejected on Metal (no Phi model has one).

### 14.9.4 Other config keys

| Key | Rule |
|---|---|
| `sliding_window` | all layers. Phi-3-mini-4k: 2047 (a real window, since the context is 4096); Phi-3.5 and Phi-4-mini: 262144 (never reached); phi-4: `null` |
| `head_dim` | absent, so `hidden_size / num_attention_heads` (Phi-3-mini and 3.5: 96; Phi-4-mini and phi-4: 128). 96 is a multiple of 32, so Metal works |
| `tie_word_embeddings` | true for Phi-4-mini (no `lm_head.weight`), false for the others |
| `attention_bias`, `mlp_bias` | false, and rejected if true |
| `embd_pdrop`, `resid_pdrop`, `attention_dropout`, `initializer_range`, `full_attn_mod`, `interpolate_factor`, `lm_head_bias` | ignored (dropout is inference-off; the last three only appear in Phi-4-mini and are 1 / 1 / false) |
| `rms_norm_eps`, `rope_theta`, `hidden_act` | `rms_norm_eps` defaults to 1e-5 in `Families.phi3`; `rope_theta` and `hidden_act` (`silu` only) as for Llama; phi-4 has `rope_theta` 250000 |

### 14.9.5 BOS, EOS and templates

| | Phi-3-mini, Phi-3.5 | Phi-4-mini | phi-4 (14B) |
|---|---|---|---|
| tokenizer | SentencePiece-style BPE, 32064 ids | GPT-4o-style byte-level, 200064 ids | cl100k-style byte-level, 100352 ids |
| `bos_token_id` | 1 (`<s>`) | 199999 (`<\|endoftext\|>`) | 100257 (`<\|endoftext\|>`) |
| config `eos_token_id` | 32000 | 199999 | 100265 (`<\|im_end\|>`) |
| `generation_config.json` eos | 32000, 32001, 32007 (`<\|end\|>`) | 200020 (`<\|end\|>`), 199999 | 100257, 100265 |
| chat template | `<\|user\|>\n…<\|end\|>\n<\|assistant\|>\n` | `<\|user\|>…<\|end\|><\|assistant\|>` | `<\|im_start\|>user<\|im_sep\|>…<\|im_end\|>` |
| BOS prepended by this engine | no | no | no |

None of the published tokenizers adds a BOS: Phi-3's `tokenizer_config.json` has `add_bos_token: false`, its
`tokenizer.json` post-processor is empty and its chat template has no `<s>`; Phi-4-mini and phi-4 use
`<|endoftext|>` as `bos_token_id` but never prepend it. This is not a Phi rule in the engine: `Tokenizer.promptBos()`
(14.2) reads `add_bos_token` and returns -1 for all three tokenizers (`TokenizerFamiliesTest` asserts it), while
`Core.bosTokenId` still holds the config's `bos_token_id`. A template or prompt that contains the token itself is of
course left alone. The stop ids are the union of config,
`generation_config.json` and the template's `<|end|>` / `<|im_end|>` / `<|endoftext|>`. `ChatTemplate` picks the
phi-4 ChatML format when the vocabulary has `<|im_sep|>` (phi-4 14B), the Phi-4-mini format for other byte-level
vocabularies, and the Phi-3 one otherwise.

### 14.9.6 Verified on the Hub

Headers read with a ranged request (first 2 MB of shard 1, no weights downloaded), all BF16, layer 0:

| Repository | d | heads/kv | head_dim | f | V | layers | `qkv_proj` | `gate_up_proj` | `lm_head` |
|---|---|---|---|---|---|---|---|---|---|
| `microsoft/Phi-3-mini-4k-instruct` | 3072 | 32/32 | 96 | 8192 | 32064 | 32 | 9216 × 3072 | 16384 × 3072 | present |
| `microsoft/Phi-3.5-mini-instruct` | 3072 | 32/32 | 96 | 8192 | 32064 | 32 | 9216 × 3072 | 16384 × 3072 | present |
| `microsoft/Phi-4-mini-instruct` | 3072 | 24/8 | 128 | 8192 | 200064 | 32 | 5120 × 3072 | 16384 × 3072 | absent (tied) |
| `microsoft/phi-4` | 5120 | 40/10 | 128 | 17920 | 100352 | 40 | 7680 × 5120 | 35840 × 5120 | present |

(`9216 = 3072 + 2·3072`, `5120 = 3072 + 2·1024`, `7680 = 5120 + 2·1280`.) Per layer the only other tensors are
`input_layernorm.weight`, `post_attention_layernorm.weight`, `self_attn.o_proj.weight` and `mlp.down_proj.weight`.

### 14.9.7 Not supported for Phi

- Phi-3-small, Phi-3-vision, Phi-3.5-MoE (`phimoe`), Phi-4-multimodal: other `model_type`s / extra towers.
- Per-request switching between short and long RoPE factors (see above).
- Phi-1 / Phi-2 (`model_type: phi`: LayerNorm, GELU, parallel attention) are a different architecture.

### 14.9.8 Testing

The three Phi-3 tiny models of the arch smoke (14.7: fused weights with longrope long factors, partial rotary and a
window; the same with the short factors and tied embeddings; `head_dim` 64 MHA with plain RoPE) are also checked by
`ArchSmoke` for the fused split: each Q8_0 row of the converted `q_proj`/`k_proj`/`v_proj`/`gate_up` equals the
matching row range of the checkpoint's `qkv_proj` / `gate_up_proj` within the quantization step. `SmokeTest.longRope`
and `phi3Configs` are described in 14.9.2 and check the formula and the real config shapes.

## 14.10 Gemma 2 and Gemma 3

Code: `Families.gemma`, `model/GemmaMath`, the second `RopeTable` of `Decoder.ropeTables`, `ModelConverter.norm` /
`canonicalName`, `CpuBackend.geluMul`, `kernels.metal` (`rmsnorm_add` shared with GLM-4, the `gelu` flag of
`matmul_swiglu`, `softcap_f` in the attention kernels and in the LM-head matmul epilogue).
Behaviour checked against transformers `modeling_gemma2.py` / `modeling_gemma3.py` (v4.51.3).

### 14.10.1 What differs from the Llama-style layer

Same decoder skeleton, with these differences from the Llama-style layer of chapter 5:

| # | Gemma | Formula | Where |
|---|---|---|---|
| 1 | embedding scale | `x0 = embed[token] * normalizer`, `normalizer = bf16(sqrt(hidden_size))` | `Core.embedScale`, applied right after the row is dequantized (both backends) |
| 2 | RMSNorm gain `1 + w` | `y = x * rsqrt(mean(x²) + eps) * (1 + w)`, in f32 (`Gemma3RMSNorm`; the checkpoints store `w`, initialised to 0) | the converter stores `1 + w` as f32 (`GemmaMath.foldNormGain`), so the unchanged `rmsnorm` / `norm_rope` kernels apply it; this includes `q_norm`, `k_norm` and the final norm |
| 3 | four norms per layer | `h = x + post_attn_norm(attn(input_norm(x)))`, `x' = h + post_ffn_norm(mlp(pre_ffn_norm(h)))` | o_proj and down_proj write to `xb` (no accumulate); the GLM-4 post-norm path (`CpuBackend.addNormed` / `rmsnorm_add`) normalizes it and adds it to the residual `x`; `LayerSpec.Norms.SANDWICH_GEMMA.post()` is true like for GLM-4, only the tensor names differ |
| 4 | GeLU-tanh MLP | `down(gelu_tanh(gate(x)) * up(x))`, `gelu_tanh(x) = 0.5x(1 + tanh(sqrt(2/π)(x + 0.044715x³)))` | `Mlp.Dense.act` is `gelu_tanh` for Gemma, else `silu`; CPU `geluMul`; Metal: the fused `matmul_swiglu` kernels take a `gelu` flag in `SwigluParams` and apply `gelu_tanh` instead of `silu` in the epilogue |
| 5 | attention scale | `scores = q·k * query_pre_attn_scalar^-0.5` (not `head_dim^-0.5`; in the three checked configs the scalar equals `head_dim` = 256 so both give 1/16, the test checkpoints use 48 to tell them apart) | `Attention.Gqa.scale` goes into the scale parameter of every attention path |
| 6 | attention soft-cap (Gemma 2) | `s' = tanh(s / 50) * 50`, applied to the scaled score before the softmax; `null` in Gemma 3 | CPU f32 and f16 loops, Metal `attention` / `attention_h` (`AttnParams.softcap`, 0 = off, argument clamped to ±20 because fast-math `tanh` can overflow) |
| 7 | final logit soft-cap (Gemma 2) | `l' = tanh(l / 30) * 30` after the LM head; `null` in Gemma 3 | `Core.finalSoftcap`. CPU: `GemmaMath.softcap` / `VectorKernels.softcapFast` on the logits after the head matmul; Metal: fused into the epilogue of the LM-head matmul (`MatmulParams.softcap`) |
| 8 | QK-norm (Gemma 3 only) | per-head `(1+w)` RMSNorm of q and k before RoPE | reuses the Qwen3 `qNorm`/`kNorm` path (`qkNorm` is true for `gemma3_text`) |
| 9 | local / global layers | local layers use the sliding window, global layers the full context. Gemma 2: layer `i` is local if `i % 2 == 0`. Gemma 3: local unless `(i + 1) % sliding_window_pattern == 0` (5 local, 1 global). `layer_types` (`sliding_attention` / `full_attention`) overrides both rules | `Attention.Gqa.window` per layer |
| 10 | two RoPE tables (Gemma 3) | local layers: base `rope_local_base_freq` (10 000), no scaling; global layers: `rope_theta` (1 000 000) with `rope_scaling` (`linear`, factor 8 in the 4B, none in the 1B) | `Decoder.ropeTables` has a second `RopeTable` (index 1) when some layer is local; `Attention.Gqa.ropeTable` selects per layer; each backend builds one cos/sin table per `RopeTable` (Metal: two buffers). Gemma 2 has one table (`rope_theta` 10 000) |
| 11 | tied head, BOS, stops | `tie_word_embeddings` is absent from the configs and defaults to true (`Families.gemma`); `<bos>` (id 2) is prepended by the tokenizer rule (`add_bos_token`, 14.2; the chat template already emits it, so it is not doubled); stops: `eos_token_id` ([1, 107] for Gemma 2, 106 for Gemma 3) plus `<end_of_turn>` / `<eos>` from the template | |

The sliding window follows transformers' mask `kv > q - window` (a token sees `window` positions including itself),
the same convention as Mistral above. Gemma 2's `hidden_act` and `hidden_activation` are both
`gelu_pytorch_tanh`; Gemma 3 only has `hidden_activation`. The embedding normalizer is a tensor of the checkpoint
dtype in transformers, so `sqrt(1152) = 33.94` is really `34.0` in a bf16 model (`bf16(sqrt(2304)) = 48` exactly);
`Families.gemma` reads `torch_dtype` (or `dtype`) for this (`float32` checkpoints, such as the test ones, keep the exact
value).

### 14.10.2 Config keys read

`Families.gemma` and the shared `Draft` read: `hidden_size`, `intermediate_size`, `num_hidden_layers`,
`num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `rms_norm_eps`, `rope_theta`,
`rope_local_base_freq`, `rope_scaling` (`rope_type` linear; also `rope_parameters`, keyed by layer type
`full_attention` / `sliding_attention`, of transformers 5), `sliding_window`, `sliding_window_pattern`, `layer_types`, `query_pre_attn_scalar`,
`attn_logit_softcapping`, `final_logit_softcapping`, `hidden_activation`, `bos_token_id`, `eos_token_id`,
`torch_dtype`. Defaults when the key is absent: soft-caps 50 / 30 for `gemma2`, off for `gemma3_text`; an explicit
`null` always means off; `query_pre_attn_scalar` 256; `sliding_window_pattern` 6; `hidden_activation` `gelu_pytorch_tanh`.
For the multimodal `gemma3` wrapper (`Gemma3ForConditionalGeneration`) all of these are taken from
`text_config`, and `bos_token_id`, `eos_token_id` and `torch_dtype` fall back to the top level; `vision_config` and
the image token ids are ignored.

### 14.10.3 What the Hub says

Config.json and safetensors headers fetched with Range requests, no weights downloaded):

| | `unsloth/gemma-2-2b-it` | `unsloth/gemma-3-1b-it` | `unsloth/gemma-3-4b-it` (`text_config`) |
|---|---|---|---|
| `model_type` / class | `gemma2` / `Gemma2ForCausalLM` | `gemma3_text` / `Gemma3ForCausalLM` | `gemma3` / `Gemma3ForConditionalGeneration` (text: `gemma3_text`) |
| hidden / ffn / layers | 2304 / 9216 / 26 | 1152 / 6912 / 26 | 2560 / 10240 / 34 |
| heads / kv / `head_dim` | 8 / 4 / 256 | 4 / 1 / 256 | 8 / 4 / 256 |
| vocab (embedding rows) | 256000 | 262144 | 262208 |
| `query_pre_attn_scalar` | 256 | 256 | 256 |
| `sliding_window` | 4096 | 512 (pattern 6) | 1024 (pattern 6) |
| `rope_theta` / local / scaling | 10000 / - / none | 1e6 / 10000 / `null` | 1e6 / 10000 / linear ×8 |
| `attn_logit_softcapping` / `final_logit_softcapping` | 50.0 / 30.0 | null / null | null / null |
| `bos_token_id` / `eos_token_id` | 2 / [1, 107] | 2 / 106 (`generation_config`: [1, 106]) | 2 / 106 |
| `torch_dtype` | bfloat16 | bfloat16 | bfloat16 |
| `layer_types` | absent (rule above) | absent | absent |

(`google/gemma-2-2b-it` and `google/gemma-3-4b-it` are gated; the `unsloth` mirrors were used. The head dim is 256
in all three, so the Metal attention kernel (`headDim` ≤ 256, multiple of 32, 8 values per lane) covers them; the arch
smoke runs `head_dim` 256 variants.)

### 14.10.4 Tensors

All BF16; `d` = hidden, `q = heads·head_dim`, `kv = kv_heads·head_dim`, `f` = ffn, `V` = vocab; headers of
`unsloth/gemma-2-2b-it`: 288 tensors, `unsloth/gemma-3-1b-it`: 340 tensors.

| Tensor | Shape (2B / 1B) | Gemma 2 | Gemma 3 |
|---|---|---|---|
| `model.embed_tokens.weight` | V × d: 256000×2304 / 262144×1152 | x | x |
| `model.layers.N.input_layernorm.weight` | d | x | x |
| `model.layers.N.self_attn.{q,k,v}_proj.weight` | q × d, kv × d, kv × d: 2048/1024/1024 × 2304 / 1024/256/256 × 1152 | x | x |
| `model.layers.N.self_attn.o_proj.weight` | d × q | x | x |
| `model.layers.N.self_attn.{q,k}_norm.weight` | head_dim (256) | | x |
| `model.layers.N.post_attention_layernorm.weight` | d, applied to the attention output | x | x |
| `model.layers.N.pre_feedforward_layernorm.weight` | d, MLP input norm | x | x |
| `model.layers.N.post_feedforward_layernorm.weight` | d, applied to the MLP output | x | x |
| `model.layers.N.mlp.{gate,up}_proj.weight` | f × d | x | x |
| `model.layers.N.mlp.down_proj.weight` | d × f | x | x |
| `model.norm.weight` | d | x | x |
| `lm_head.weight` | absent (tied) | | |

Note that `post_attention_layernorm` is *not* the MLP input norm here, unlike Llama. In the 4B+ multimodal
checkpoints (`model.safetensors.index.json`, 883 entries in 2 shards) the same tensors are prefixed
`language_model.model.` (as released; transformers ≥ 4.52 writes `model.language_model.`), next to
`vision_tower.vision_model.*` (SigLIP, 27 layers) and `multi_modal_projector.{mm_input_projection_weight,
mm_soft_emb_norm.weight}`; the 4B embedding is `language_model.model.embed_tokens.weight` 262208 × 2560.
`ModelConverter.canonicalName` maps both spellings to the plain names and never reads the vision tensors. In the
quantized cache the norms are stored already as `1 + w` under their checkpoint names; delete a `jllm-*.bin` written
by a build without Gemma support before loading a Gemma model.

### 14.10.5 Testing and verification

`GemmaChecks` (called from `SmokeTest`) compares `gelu_tanh`, the soft-cap and the `(1 + w)` RMSNorm with
hand-computed values and parses the real Hub configs listed above; the five Gemma tiny models of the arch smoke (14.7)
add the CPU == Metal comparison. Beyond that, the tiny random checkpoints were
also run through an independent pure-Python implementation written line by line from `modeling_gemma2/3.py`
(sequential attention with an explicit mask, `rotate_half`, `(1 + w)` norms, soft-caps). Final logits of the last
prompt token agree with it to quantization noise (Q8_0: correlation 0.99997, max error 0.3–0.9 % of the largest
logit; Q4_0: 0.9994, 1–4.5 %) on both backends, for Gemma 2, Gemma 3 and the multimodal layout, with f32 and f16 KV
caches and `head_dim` 32, 64 and 256. A 262 144-row vocabulary also runs (CPU == Metal).

### 14.10.6 Not supported / not verified

Images (the vision tower and projector are skipped, so Gemma 3 4B+ is text-only);
the 4B+ sizes and Gemma 2 9B/27B were checked only through their configs and tensor indexes (only the 2B/1B/4B
configs above were read); real weights were run only for `gemma-2-2b-it` and `gemma-3-1b-it` (14.13);
`dynamic` RoPE scaling; Gemma 3n and Gemma 1 (`model_type: gemma`, no extra norms) are rejected.

### 14.10.7 Gemma 4 (12B)

Code: `Families.gemma4`, `Decoder.uniformShape`, the per-layer `KvCache[]` of `CpuBackend`, the `kEqV` / `vNorm` fields of
`Attention.Gqa`, `RopeTable.activePairs` (+ `Rope.invFreq`), `DecoderLayer.layerScalar`, `GgufArchitecture.GEMMA4` and
`GgufFamilies.gemma`.

Gemma 4 is not a Gemma 3 variant with new defaults; five things change:

| # | Gemma 4 | Gemma 2 / 3 |
|---|---|---|
| 1 | **head shape differs by layer**: local layers keep `head_dim` 256 and `num_key_value_heads` 8; full layers use `global_head_dim` 512 and `num_global_key_value_heads` 1 | one head shape for every layer |
| 2 | full layers have **no `v_proj`**: the raw key projection is the value (`attention_k_eq_v`); every layer has a **parameterless RMS norm on V** (`v_norm`, no gain) before the cache | `v_proj`, no V norm |
| 3 | **proportional RoPE** on full layers: `rotaryDim` is the whole 512-wide head (so pairs are `i, i+256`), but only `partial_rotary_factor` 0.25 (64 pairs) have a non-zero frequency; the rest are identity | full/partial RoPE over the head with every pair active |
| 4 | attention is **unscaled** (`self.scaling = 1.0`); q/k norm still applies | `query_pre_attn_scalar^-0.5` / `head_dim^-0.5` |
| 5 | norm gains are the stored weights (no `1 + w`), and each layer ends with a learned scalar `x *= layer_scalar` | gains `1 + w`, no layer scalar |

The 12B is dense with no per-layer input embeddings (`hidden_size_per_layer_input` 0); the parser rejects the E2B / E4B
(PLE) and 26B-A4B (MoE) shapes rather than loading them wrongly. The 12B runs on the CPU and on Metal: the backend has
512-wide attention kernels (`attention_g1_d512` / `_h` / `_q8`, `attention_dec_g1_d512`, `attention_fa` / `_fb_d512`,
`attention_split_d512`) and per-layer head-shape dispatch, and it applies `attention_k_eq_v` (V = the pre-RoPE key
projection), the parameterless per-head V-norm and the per-layer `layer_scalar` on the GPU. The paged, batched, mixed
and tiered KV passes, q4 KV and rotated q8 are not offered (each layer keeps its own flat KV cache). GGUF files name
the same decoder `gemma4` (E2B / E4B / 26B / 31B) or `gemma4_unified` (12B).

## 14.11 DeepSeek-V2 / V3 (Multi-head Latent Attention + shared experts)

Code: `Families.deepseek`, `model/Attention.Mla`, `model/Router.DeepSeek` + `model/DeepSeekRouter`, `model/AttnWeights.Mla`
(+ `kvBKeys` / `kvBValues`), `model/Scaling.Yarn` + `Rope.yarnInvFreq`, `model/ModelConverter.mlaRules` / `moeRules`,
`cpu/MlaCpu`, `metal/MlaMetal`, kernels `mla_lat_store`, `mla_absorb_q`, `mla_apply_v`, the `attention_mla` instantiation
(section 7 of `kernels.metal`), `rope_il` and `router_ds` (section 5), `tokenizer/ChatTemplate` (`deepseek-v2`).

| `model_type` | Architecture class | Examples | Weights |
|---|---|---|---|
| `deepseek_v2` | `DeepseekV2ForCausalLM` | `deepseek-ai/DeepSeek-V2-Lite` (15.7 B, 27 layers), `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | bf16, supported |
| `deepseek_v3` | `DeepseekV3ForCausalLM` | `deepseek-ai/DeepSeek-V3`, `DeepSeek-R1` (671 B) | FP8, dequantized while converting (14.11.7) |

The R1-*Distill* models are plain Qwen2/Llama checkpoints and were already supported (14.1).

What was checked on the Hub (config, `model.safetensors.index.json`, the first 2 MB of one shard for the safetensors
header, and `modeling_deepseek.py`; no weights were downloaded): the V2-Lite and Coder-V2-Lite-Instruct `config.json` are
identical except for `transformers_version`, and so are their `modeling_deepseek.py` files (byte for byte); V3's modeling
file differs in the router (`noaux_tc`, sigmoid) and in the V3-only `e_score_correction_bias`. Everything below follows
those files.

### 14.11.1 Multi-head Latent Attention

Standard attention projects `x` to per-head `q`, `k`, `v` and caches `k` and `v` of every head. MLA (DeepSeek-V2)
squeezes keys and values through a small shared *latent* `c` of `kv_lora_rank` numbers; the whole point of the design is
that the cache holds `c` plus one shared rotary key, `kv_lora_rank + qk_rope_head_dim` numbers per token, instead of
`H · (qk + v_dim)`. A head's query/key has two parts: `qk_nope_head_dim` dimensions without position and
`qk_rope_head_dim` dimensions that are rotated by RoPE; the rotated key part is a single vector shared by all heads.

The paper writes the per-head key and value as `k_nope_h = W_kb_k[h] · c` and `v_h = W_kb_v[h] · c` (`W_kb_k[h]`,
`W_kb_v[h]` the `nope` resp. `v_dim` rows of `kv_b_proj` that belong to head `h`). The engine never forms them: with the
*absorbed* formulation the two matrices move to the query side and to the output side, which is the same arithmetic in
a different order,

```
score_h(s) = q_nope_h · (W_kb_k[h] c_s) + q_pe_h · k_pe_s  =  (W_kb_k[h]ᵀ q_nope_h) · c_s + q_pe_h · k_pe_s
out_h      = Σ_s w_s (W_kb_v[h] c_s)                       =  W_kb_v[h] (Σ_s w_s c_s)
```

so attention runs over the latent directly: query `[W_kb_k[h]ᵀ q_nope_h | rope(q_pe_h)]` (`kv_lora + rope` wide), key
`[c_s | rope(k_pe_s)]` (one row per position, shared by all heads: multi-query attention with a single kv head), value
`c_s` (the first `kv_lora` entries of the same row), followed by one small `v_dim × kv_lora` matmul per head.

```
                      x [dim]
        ┌──────────────┼───────────────────────────────┐
        │ q path       │ kv path                       │
   q_proj (V2-Lite)    kv_a_proj_with_mqa [kv_lora + rope, dim]
   or q_a_proj [q_lora, dim]          ├────────────┬──────────────┐
     → q_a_layernorm                  │ latent     │ k_pe [rope]  │
     → q_b_proj [H·qk, q_lora]        │ [kv_lora]  │ (one vector) │
        │                             │ kv_a_layernorm   rope(k_pe) │
   q [H, qk] = q_nope | q_pe          ▼            ▼
        │ rope(q_pe)              cache row_s = [c_s | rope(k_pe_s)]       (kv_lora + rope per token, f32 or f16)
        │ q_abs_h = W_kb_kᵀ[h] q_nope_h    │
        ▼                                 │
   query_h = [q_abs_h | rope(q_pe_h)]     │  key_s = row_s      value_s = row_s[:kv_lora]
        └──── o_h = softmax(query_h · key_s · scale) · value_s  (causal; one kv head for all H) ────┘
                                   │
                         out_h = W_kb_v[h] · o_h   (per head v_dim × kv_lora, mla_apply_v)
                                   │
                         concat heads [H·v_dim] → o_proj [dim, H·v_dim]
```

Shapes for DeepSeek-V2-Lite (`dim` 2048, `H` = 16, `nope` 128, `rope` 64, `qk` = 192, `v_dim` 128, `kv_lora` 512,
`q_lora_rank: null`), as read from the safetensors header:

| Tensor | Shape |
|---|---|
| `self_attn.q_proj.weight` | `[3072, 2048]` = `[H·qk, dim]` |
| `self_attn.kv_a_proj_with_mqa.weight` | `[576, 2048]` = `[kv_lora + rope, dim]` |
| `self_attn.kv_a_layernorm.weight` | `[512]` |
| `self_attn.kv_b_proj.weight` | `[4096, 512]` = `[H·(nope + v_dim), kv_lora]` |
| `self_attn.o_proj.weight` | `[2048, 2048]` = `[dim, H·v_dim]` |

With `q_lora_rank` set (V3: 1536) `q_proj` is replaced by `q_a_proj [q_lora, dim]`, `q_a_layernorm [q_lora]` and
`q_b_proj [H·qk, q_lora]`. V3 has `H` = 128, so `q_b_proj` is `[24576, 1536]` and `kv_b_proj` `[32768, 512]`.

Implementation notes:

- **`kv_a_proj_with_mqa` is split** by the converter into `kv_a_latent.weight [kv_lora, dim]` and `k_pe_proj.weight
  [rope, dim]` (row slices; Q4_0/Q8_0 blocks run along a row, so slicing before quantization is exact). That makes the latent
  contiguous per token for the RMSNorm and the rotary key a separate small matmul.
- **Rotary layout.** The checkpoint stores the rotary dimensions interleaved. `apply_rotary_pos_emb` in
  `modeling_deepseek.py` first de-interleaves (`view(d/2, 2).transpose`) and then applies the usual rotate-half. Because the
  same permutation is applied to the query and the key, the dot product is identical to rotating the pairs `(2i, 2i+1)`
  in place by `pos · inv_freq[i]`, which is what `InterleavedRope.rotate` (CPU) and the interleaved `rope_il` kernel (Metal)
  do, with `RopeTable.Pairing.INTERLEAVED` (no permutation is needed).
- **Absorbed `kv_b_proj`.** `AttnWeights.Mla.kvBKeys` / `kvBValues` dequantize the two halves of `kv_b_proj` per head to f32
  once at load (`H · (nope + v_dim) · kv_lora` floats per layer: 8 MB for V2-Lite, 216 MB over 27 layers; on Metal they
  are uploaded as f32 buffers, laid out so that the absorb / apply kernels read them coalesced). The absorption
  `q_abs_h = W_kb_k[h]ᵀ q_nope_h` costs `H · nope · kv_lora` multiply-adds per token (1 M for V2-Lite, a sixth of
  `q_proj`), the output step `H · v_dim · kv_lora` the same; the expanded form's `kv_b_proj` matmul was exactly as large,
  so no work is added, it is only moved.
- **Attention widths.** The absorbed query / cached row is `kv_lora + rope` = 576 wide (V2-Lite and V3 alike) and the value
  is the `kv_lora` = 512 wide prefix of the same row. The CPU attention in `MlaCpu` is a tiled multi-query loop (every
  cached row is read, or expanded from f16, once per tile of tokens for all heads of the task); Metal uses the generic
  `attention_t` template as `attention_mla` / `attention_mla_h`: one kv head, runtime widths up to
  `MLA_MAX_PER · 32` = 576 lanes (the other instantiations keep `MAX_PER` = 8, i.e. head dims ≤ 256), the key and value
  buffers bound to the same cache. Metal needs `kv_lora` and `kv_lora + rope` to be multiples of 32 and ≤ 576.
- **Softmax scale** is `qk^-0.5` (192, not the 128 of an ordinary head) times `mscale²` with
  `mscale = 0.1 · mscale_all_dim · ln(factor) + 1` when `rope_scaling.mscale_all_dim` is set (V2-Lite: 0.707 → 1.2608²;
  V3: 1.0 → 1.3689²), see `Attention.Mla.scale`.
- **KV cache.** One row `[c | rope(k_pe)]` of `kv_lora + rope` values per token and layer, on both backends, f32 or
  (`backend.kv-dtype=f16`) half with the arithmetic in f32, exactly as the dense families' caches: V2-Lite needs
  `576 · 4 B` = 2.3 KB per layer and token, **62 KB per token over 27 layers (255 MB at `--ctx 4096`), 31 KB / 127 MB at
  f16, 16 KB / 64 MB at q8 (one scale per latent row, [07 §7.13](07-cpu-backend.md#713-the-f16-kv-cache-kvcache))**. Caching the expanded per-head keys and values instead would take `(16·192 + 16·128) · 4 B` = 20 KB per layer
  and token, **553 KB per token** (2.2 GB at `--ctx 4096`), 9× as much, and the value of MLA would be lost: the whole
  reason for the latent is a cache that does not grow with `H`. The absorbed order of the floating-point operations
  moves the logits in the last bits only: when the engine switched from the expanded to the latent layout, the largest
  difference on the three tiny models was 3.1e-5 (relative 1.7e-6 of the largest logit) on CPU and Metal, Q4_0 and
  Q8_0, with the same argmax, and the f16 cache differs from f32 by 2e-3 relative (the half rounding of the latent, as
  on the dense families).
- `Attention.Mla` reports `headDim = qk` (192), so `qDim = H·qk` and the Metal "multiple of 32, ≤ 256"
  check applies to 192 (the rope kernel's slice). The BOS (id 100000 for V2) is in the chat template; for raw prompts
  `Tokenizer.promptBos()` supplies it (V2's tokenizer sets `add_bos_token`, 14.2).

### 14.11.2 YaRN rotary scaling

`rope_scaling: {"type": "yarn", factor, original_max_position_embeddings, beta_fast, beta_slow, mscale, mscale_all_dim}`
applies only to the `rope` = 64 rotary dimensions (`Scaling.Yarn` in the layer's `RopeTable`, `Rope.yarnInvFreq`). Each of the 32 frequency pairs
`i` has the plain frequency `f_i = θ^(-2i/64)` ("extrapolation") and the stretched `f_i / factor` ("interpolation").
Which one is used depends on how many turns the pair makes over the original context `L` = 4096:

```
dim(β) = 64 · ln(L / (β · 2π)) / (2 ln θ)
low  = max(floor(dim(beta_fast)), 0)  = floor(10.47) = 10        (V2-Lite, θ = 10000, beta_fast 32)
high = min(ceil(dim(beta_slow)), 63)  = ceil(22.51)  = 23        (beta_slow 1)
ramp(i) = clamp((i - low) / (high - low), 0, 1)
inv_freq[i] = f_i / factor · ramp(i)  +  f_i · (1 - ramp(i))
```

Pairs 0 – 10 keep their frequency, pairs 23 – 31 are divided by 40, pairs 11 – 22 blend: e.g. pair 16 has
`f = 0.01`, ramp 6/13, `inv_freq = 0.00025 · 6/13 + 0.01 · 7/13 = 0.0055` (checked in `SmokeTest.yarnRope`).
Finally the cos/sin tables are multiplied by `yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)`
(exactly 1 for V2-Lite and V3, where the two are equal). Rope types other than `yarn` / none are rejected for DeepSeek.

**The other families** (Qwen2 / Qwen3 long context: `{"rope_type": "yarn", "factor": 4.0,
"original_max_position_embeddings": 32768}`) use the same frequencies over their rotary dimensions (`Families.yarn`,
transformers' `_compute_yarn_parameters`): `beta_fast` / `beta_slow` default to 32 / 1, the original length falls
back to `max_position_embeddings`, and the cos/sin amplitude is `attention_factor` if given, else
`yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)` when both are set, else
`0.1 · ln(factor) + 1` (1.1386 for factor 4). One deliberate difference from transformers, which replaces `factor`
by `max_position_embeddings / original_max_position_embeddings` when the latter is given: the map's `factor` is used,
as llama.cpp, vLLM and the GGUF metadata do (Qwen's own recipe adds the map without raising
`max_position_embeddings`, which would otherwise give 40960/32768 = 1.25 instead of 4). `SmokeTest.yarnRope` checks
that a Qwen3 config gives exactly the DeepSeek frequencies for the same parameters and the expected amplitude.

### 14.11.3 Mixture of experts: dense prefix, routed + shared experts

Layer `l` is MoE when `l ≥ first_k_dense_replace` and `l % moe_layer_freq == 0`, otherwise a dense SwiGLU MLP with
`intermediate_size` (V2-Lite: layer 0 dense, layers 1 – 26 MoE; V3: layers 0 – 2 dense). An MoE layer computes

```
y = Σ_{k ∈ top-k}  w_k · Expert_k(x)   +   SharedExperts(x)
```

- `n_routed_experts` routed experts (`mlp.experts.N.{gate,up,down}_proj`, width `moe_intermediate_size`), stacked into the
  same two tensors as Qwen3-MoE (chapter 12) and run by the same `moe_up` / `moe_down` kernels.
- `n_shared_experts` shared experts (`mlp.shared_experts.{gate,up,down}_proj`) that are *one* dense SwiGLU of width
  `n_shared_experts · moe_intermediate_size` (V2-Lite: 2 · 1408 = 2816, `[2816, 2048]`), applied to every token and added. They
  are stored as `mlp.shared.gate_up` / `mlp.shared.down` (`Mlp.Moe.shared`, `MlpWeights.Moe.shared`) and run after the
  routed experts with the dense SwiGLU + down-projection path (`CpuBackend.MoeFfn.run`, `MetalBackend.MoeStep.run`).
  Prefill groups the routed slots by expert on the CPU and Metal fuses SwiGLU into `moe_up`, as in
  [12.4](12-mixture-of-experts.md#124-cpu-implementation) and [12.5](12-mixture-of-experts.md#125-metal-implementation).
- The router `mlp.gate.weight [n_routed_experts, dim]` stays f32. V3 adds `mlp.gate.e_score_correction_bias [256]`
  (f32), stored as `mlp.router_bias`. V3 also has a layer 61 (multi-token prediction) in the checkpoint which the
  converter never reads (it is past `num_hidden_layers`).

### 14.11.4 The three routing rules

`scoring_func`: `softmax` (V2) or `sigmoid` (V3) over the router logits gives `scores`. Then `topk_method`
(`DeepSeekRouter.select` on the CPU, `router_ds` on Metal; ties go to the lower index):

| `topk_method` | Model | Selection |
|---|---|---|
| `greedy` | V2-Lite | the `k` highest scores |
| `group_limited_greedy` | V2 (236 B) | experts form `n_group` consecutive groups; group score = **max** score; keep the `topk_group` best groups; the `k` highest scores inside the kept groups |
| `noaux_tc` | V3 | choice = `scores + e_score_correction_bias`; group score = **sum of the two best** choices; keep `topk_group` groups; `k` highest *choices* inside them; the **weights are the uncorrected scores** |

Weights: if `k > 1 && norm_topk_prob` they are divided by `Σ + 1e-20`; then multiplied by `routed_scaling_factor`.
V3 always multiplies; the V2 code only multiplies in the branch where it did **not** normalize (the `else` of the
`norm_topk_prob` test), which `Mlp.Moe.normTopK`, `Router.DeepSeek.scaleAlways` and `DeepSeekRouter` reproduce.

Worked example for `noaux_tc` (8 experts, 4 groups of 2, `topk_group` 2, `k` 2, `routed_scaling_factor` 2.5):

```
expert            0    1  |  2    3  |  4    5  |  6    7
sigmoid score    .9   .1  | .6   .5  | .8   .7  | .3   .2
bias            -.5    0  |  0    0  |  0    0  | .45  .45
choice           .4   .1  | .6   .5  | .8   .7  | .75  .65
group score        .5     |   1.1    |   1.5    |   1.4          (sum of the two best choices)
kept groups:  2 and 3  →  best two choices inside: expert 4 (.8), expert 6 (.75)
weights: uncorrected .8 and .3  →  /1.1 → .727, .273  →  ×2.5 → 1.818, 0.682
```

Without the bias the kept groups would be 1 and 2 and the winners experts 4 and 5; with `group_limited_greedy` the group score would be the group maximum.
The same numbers (and a softmax example for `greedy` vs `group_limited_greedy`) are asserted in `SmokeTest.deepseekRouting`.

### 14.11.5 Config keys and tensor names (verified against the Hub)

Read from `config.json`: `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `vocab_size`,
`rms_norm_eps`, `rope_theta`, `rope_scaling` (yarn), `q_lora_rank` (null → 0), `kv_lora_rank`, `qk_nope_head_dim`,
`qk_rope_head_dim`, `v_head_dim`, `first_k_dense_replace`, `moe_layer_freq`, `n_routed_experts`, `n_shared_experts`,
`moe_intermediate_size`, `num_experts_per_tok`, `n_group`, `topk_group`, `topk_method`, `scoring_func`, `norm_topk_prob`,
`routed_scaling_factor`, `bos_token_id`, `eos_token_id`, `tie_word_embeddings` (false). Missing router keys default per
family in `Families.deepseek`: V3 `scoring_func` sigmoid, `topk_method` noaux_tc, `n_group` 8, `topk_group` 4,
`routed_scaling_factor` 2.5, `norm_topk_prob` true; V2 softmax, greedy, 1, 1, 1.0, false. `attention_bias: true`,
`hidden_act` ≠ `silu` and an unknown `topk_method` / `scoring_func` are rejected. `num_key_value_heads` is ignored
(MLA has no grouped KV heads); `aux_loss_alpha`, `seq_aux`, `ep_size`, `pretraining_tp` and `num_nextn_predict_layers` are training or
parallelism details.

Per layer `model.layers.N.`: `input_layernorm.weight`, `post_attention_layernorm.weight`,
`self_attn.{q_proj | q_a_proj, q_a_layernorm, q_b_proj}.weight`, `self_attn.kv_a_proj_with_mqa.weight`,
`self_attn.kv_a_layernorm.weight`, `self_attn.kv_b_proj.weight`, `self_attn.o_proj.weight`, then either
`mlp.{gate,up,down}_proj.weight` or `mlp.gate.weight`, `mlp.gate.e_score_correction_bias` (V3),
`mlp.experts.E.{gate,up,down}_proj.weight`, `mlp.shared_experts.{gate,up,down}_proj.weight`. Global:
`model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight` (separate). V2-Lite is split in 4 bf16 shards
(31.4 GB), V3 in 163 shards.

Tokens: the chat templates in `tokenizer_config.json` of V2-Lite and Coder-V2-Lite-Instruct are the
`User: …\n\nAssistant:` format (`<｜begin▁of▁sentence｜>` + optional system text + `\n\n`, each assistant turn closed by
`<｜end▁of▁sentence｜>`), implemented as `ChatTemplate` `deepseek-v2`; both set `add_bos_token`, BOS is 100000, EOS is 100001
and the only stop token is `<｜end▁of▁sentence｜>`. V3 and R1 (same `model_type`) use the `<｜User｜>` / `<｜Assistant｜>`
format that the `deepseek-r1` template implements; it always opens a `<think>` block, which is right for R1 but not for
the non-reasoning V3 chat model. V3's BOS/EOS ids are 0 and 1 (its vocabulary has 129280 entries).

### 14.11.6 Memory on this machine

DeepSeek-V2-Lite at Q4_0: 15.3 B linear weights at 5 bits (4-bit values plus an f32 scale per 32) = 9.6 GB, plus the
Q8_0 embedding and head (2 × 102400 × 2048 at 9 bits) = 0.47 GB: about **10 GB**, so it fits a 32 GB machine, plus the
dequantized `kv_b_proj` halves (216 MB f32) and the latent KV cache of `(512 + 64) · 4 B · 27` = **62 KB per token**
(255 MB at `--ctx 4096`; 31 KB per token and 127 MB with `backend.kv-dtype=f16`). Decode touches the 6 routed + 2 shared
experts of each of 26 layers, about 2.4 B parameters per token, so it is far cheaper than the weight size suggests (as for
Qwen3-MoE, chapter 12).

DeepSeek-V3 / R1 has 671 B parameters (the Hub lists 685 B including the multi-token-prediction module, which the converter skips): at Q4_0 that is 671 B × 5/8 B ≈ **420 GB** (+ 2 GB of
Q8_0 embedding and head); its latent KV cache is the same 576 numbers per token and layer, `576·4 B · 61` = **140 KB per
token** (f32; 574 MB at `--ctx 4096`), where the expanded per-head cache would have been `128·(192 + 128)·4 B · 61` =
10 MB per token (40 GB at `--ctx 4096`); the dequantized `kv_b_proj` halves are 64 MB per layer, 4 GB in all. It cannot
run on a 32 GB laptop (not even on 128 GB; the 1.37 TB FP8 download alone exceeds the disk).

### 14.11.7 FP8 checkpoints (V3, R1)

The published V3/R1 weights are block-quantized FP8: every large matrix is `F8_E4M3` (`float8_e4m3fn`) with a companion
`<name>.weight_scale_inv` tensor of dtype F32 holding one scale per 128 × 128 block (e.g. `q_a_proj.weight`
`[1536, 7168]` has `weight_scale_inv` `[12, 56]`); `config.json` carries `quantization_config: {quant_method: fp8,
weight_block_size: [128, 128]}`. The converter dequantizes them on the fly (`w = fp8_value · scale_inv[r/128][c/128]`,
`model/Fp8Quant`, exact decoding in `SafeTensors`, see [03 §3.8](03-safetensors.md)) and writes the engine's own format,
like any bf16 checkpoint; the embedding, head, norms and routers are bf16 in these files and read as they are. Running
V3 still needs a machine with more than 400 GB of memory; the path is tested on tiny FP8 checkpoints (`QuantFormatsTest`).

### 14.11.8 Testing

`scripts/make-tiny-model.py` writes three random checkpoints with the real tensor names and small dims (all multiples of 32:
`dim` 128, 4 heads, nope 32 + rope 32 = qk 64, v 32, kv_lora 64; 2 dense + 2 MoE layers; 8 routed experts):
`deepseek_v2` (`q_lora_rank: null`, 2 groups with `group_limited_greedy`, 2 shared experts, yarn with equal mscales),
`deepseek_v2q` (q LoRA, `norm_topk_prob`, `routed_scaling_factor` 2.5, yarn with `mscale` ≠ `mscale_all_dim`) and
`deepseek_v3` (sigmoid, `noaux_tc` with bias, 4 groups, the stray layer 4). They are part of the arch smoke (14.7: q4, q8
and q2k, CPU == Metal, final logits within 1e-3 relative, measured below 1e-4), and `src/test/.../DeepSeekRef` compares
the CPU and Metal logits of the 12-token prompt with `scripts/deepseek-ref.py`, an independent float64
re-implementation of `modeling_deepseek.py` (explicit de-interleave + rotate-half, `MoEGate`, YaRN, and the paper's
expanded `kv_b_proj` keys and values, so the engine's absorbed form is checked against the formulation it rearranges) run
on weights rounded to Q8_0 the same way the converter does: the largest difference is 1.3e-5 of the logit range.
(Without the rounding the engine is 1 – 6 % off, which is plain quantization noise: random weights with sharp attention
are sensitive.) `mvn test` checks the YaRN frequencies against hand-computed values (14.11.2, `SmokeTest.yarnRope`), the
routing rules on hand-made score matrices (14.11.4, `deepseekRouting`), the V2-Lite and V3 configs (`deepseekConfigs`),
the V2 chat template and V3's FP8 block size.

## 14.12 The typed model: `Architecture`

`config.json` is resolved once into a sealed `Architecture`; nothing after that looks at `model_type` (except the chat
template). The pieces:

```
Architecture (sealed)
├── Decoder(core, List<LayerSpec> layers, List<RopeTable> ropeTables)
│     LayerSpec(Attention attn, Mlp mlp, Norms norms)
│       Attention = Gqa(heads, kvHeads, headDim, window, ropeTable, qkNorm, bias, scale, softcap)
│                 | Mla(heads, qLoraRank, kvLoraRank, nope, rope, vDim, ropeTable, scale)
│       Mlp       = Dense(hidden, act) | Moe(experts, topK, moeDim, normTopK, Router, shared Dense)
│       Norms     = PRE | SANDWICH_GLM4 | SANDWICH_GEMMA   (also names the tensors)
│     RopeTable(rotaryDim, theta, Scaling, Pairing HALF | INTERLEAVED)
│       Scaling   = None | Linear | Llama3 | LongRope | Yarn
│       Router    = Softmax | DeepSeek(sigmoid, TopK greedy | group_limited_greedy | noaux_tc, nGroup, topkGroup, routedScale, scaleAlways)
└── MambaArch(core, dState, dConv, dtRank, dInner, convBias)
Core = architecture, modelType, dim, layers, vocab, norm eps, tied head, BOS/EOS ids, max positions,
       embed scale, final soft-cap, normPlusOne, description
```

A family's switches become values in this vocabulary: Qwen2's bias is `Gqa.bias`, Mistral's window is `Gqa.window`
per layer, Gemma 3's second RoPE base is a second `RopeTable`, DeepSeek's `first_k_dense_replace` is `Dense` for the
first layers and `Moe` for the rest. All layers of a `Decoder` share the attention variant and head shape
(`Attention.sameShape`), so a backend sizes its caches once; window, table, scale and MLP may differ per layer.

`ModelWeights.load` mirrors the spec with a sealed `LayerWeights` (`DecoderLayer` = norms + `AttnWeights.Gqa | Mla` +
`MlpWeights.Dense | Moe`, or `MambaLayer`), so shape checks are per variant: a GQA layer is checked for `q/k/v/o` shapes, an MLA layer for the latent
projections. The backends switch over `LayerSpec`, not over families.

`Core.description` is the one-line text (`Qwen3ForCausalLM dim=1024 ffn=3072 ...`) that `Architecture.toString()` prints
and the quantized cache index records; it is compared on resume, so for an existing family it must never change.

### 14.12.1 Adding a family

1. **One parser** in `Families` (a method returning a `Decoder`, registered in `SUPPORTED` and the `parse` switch). It
   holds the family's defaults (`rope_theta`, norm eps, activation, ...), reads its `config.json` keys and builds the
   `LayerSpec` list and `RopeTable`s. Append anything new to the description only when it is set, as the existing
   parsers do, so old caches stay valid.
2. **Converter rules** in `ModelConverter`: a cache tensor is one `Rule` (the checkpoint tensors it reads and a
   transform: quantize, split rows, stack, flatten a 3-D tensor, keep as f32, fold `1 + w`, `-exp`), and a layer's rule
   list is derived from its `LayerSpec` (`planDecoder`). A new checkpoint spelling (fused tensors, renamed prefixes) is
   detected from the shard map, not from the spec. The shard bookkeeping is derived from the rules.
3. **Chat template** entry in `ChatTemplate.forModelType` if the prompt format is new. The BOS needs no entry: it
   comes from the tokenizer (14.2).

If the family only combines existing `Attention`, `Mlp`, `Norms` and `Scaling` variants, no backend changes. A new
variant (a new attention kind, say) adds a case to the sealed type, which makes every `switch` over it a compile error
until the CPU and Metal backends, `LayerWeights` and the converter handle it.

### Loading a family from a GGUF file

The same parsers run on a GGUF file: `GgufArchitecture` writes the `config.json` that the file's llama.cpp metadata
describes (`qwen3.block_count` → `num_hidden_layers`, ...) and calls `Architecture.parse`, so the description and the
defaults are those of the safetensors load. Besides `llama`, `qwen2`, `qwen3`, `qwen3moe` (and the Qwen3.5 / Qwen3-Next
hybrids) the GGUF architectures `gemma2`, `gemma3`, `phi3`, `chatglm`, `glm4`, `mamba`, `deepseek2` and `mistral3` are
mapped ([16 §16.15](16-gguf.md#1615-gemma-23-phi-3-glm-mamba-deepseek-v2v3-and-mistral-3-as-gguf-files)); adding one means its
metadata keys in `GgufArchitecture` / `GgufFamilies`, its tensor names in `GgufWeights.resolve` and, if llama.cpp's converter
permutes or folds something for it (Llama's q/k rows, Gemma's `1 + w` norms, Mamba's `-exp(A_log)`), the matching switch in
the `Architecture`.

Where the GGUF layout differs from the checkpoint's, per family: **Gemma 2/3** the norms already hold `1 + w` (the cache's
convention), the attention scale is `1/sqrt(head_dim)` (no `query_pre_attn_scalar` in the file) and the embedding scale is
the float `sqrt(dim)`; **Phi-3 / ChatGLM / GLM-4** `attn_qkv` and `ffn_up` are the fused `qkv_proj` / `gate_up_proj` rows,
LongRoPE comes as two `rope_factors_*` tensors; **Mamba** `ssm_a` is `-exp(A_log)`; **DeepSeek** `kv_b_proj` is two tensors,
`attn_k_b` (the key half, transposed) and `attn_v_b`, and `yarn_log_multiplier = 0.1 × mscale_all_dim`.

## 14.13 Verified on real weights

Everything above was first checked on tiny random checkpoints (14.7), which exercise every code path but cannot show
that a real model computes the right thing: a wrong norm epsilon, a swapped RoPE pairing or a missing scale is
invisible on random weights. This section compares the engine with Hugging Face transformers on published
checkpoints.

```
scripts/real-weights-check.sh <hf-repo | model-dir> [q8|q4|q2k|q6k]
```

`scripts/real_ref.py` loads the checkpoint with transformers (float32, eager attention, CPU) and runs five prompts:
three chat conversations rendered with the tokenizer's own `apply_chat_template` (one user turn; system + user;
user / assistant / user) and two raw texts (prose, code) tokenized with the tokenizer's default special tokens. For
each it records 16 greedy ids (`generate` with the end-of-sequence stop and every logits processor removed) and the
full logits of every step. It then rounds every weight that the engine's converter quantizes to Q8_0 (the weight of
each `Linear` / `Embedding` whose input width is a multiple of 32: blocks of 32, scale `amax / 127`) and records the
logits of the same 16 steps again, teacher-forced on the same ids. `RealWeightsCheck` converts the checkpoint, runs the
CPU backend with float activations (`backend.cpu.int8-activations` off, `-Djllm.cpu.fastmath=false`, f32 KV cache)
on the reference's prompt ids and reports, per prompt:

- **ids / template**: the engine's own ids for the prompt (`ChatTemplate` for a conversation, `promptBos` +
  `encode` for raw text) equal the reference's, and the rendered template text equals `apply_chat_template`'s; plus
  six tokenizer probes (whitespace runs, numbers, CJK / Cyrillic / emoji, code, literal special-token text);
- **greedy**: free-running greedy decoding from the reference's prompt ids, the number of ids equal to the
  reference's before the first divergence, and the reference's logit margin there (its top logit minus its logit of
  the engine's choice: a small margin is a near tie that quantization noise can flip, after which the two
  continuations are different texts and no longer comparable);
- **teacher-forced**: one pass over the prompt plus the reference's 16 ids: the engine's argmax against the
  reference's id at each step, the mean top-20 overlap and the mean Pearson correlation of the full logit rows;
- **vs Q8-rounded reference**: the same rows against the reference that has the engine's quantized weights. Only
  the float rounding of the activations differs there, so this separates quantization noise (the previous columns)
  from a wrong computation (this one): Pearson correlation and the largest |engine − reference| relative to the
  largest |logit| over all steps.

Results at Q8_0 (head Q8_0), transformers 5.20.2, five prompts × 16 steps unless noted:

| Checkpoint | `model_type` | Prompt ids / probes / template | Greedy ids equal to the reference | First divergence: step (reference margin) | Teacher-forced argmax | Top-20 overlap | Pearson vs float32 | vs Q8-rounded reference: Pearson, max error, argmax |
|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` (control) | `qwen3` | 5/5, 6/6, 3/3 | 80/80 (5/5 prompts) | none | 80/80 | 0.979 | 0.9997 – 0.9998 | 0.999998, 0.74 %, 80/80 |
| `Qwen/Qwen2.5-0.5B-Instruct` | `qwen2` | 5/5, 6/6, 3/3 | 65/80 (4/5) | raw prose: step 1 (0.028) | 79/80 | 0.983 | 0.9996 – 0.9998 | 0.999997, 0.62 %, 79/80 |
| `unsloth/Llama-3.2-1B-Instruct` | `llama` | 5/5, 6/6, 3/3 | 66/80 (4/5) | system + user: step 2 (0.075) | 79/80 | 0.980 | 0.9993 – 0.9998 | 0.999996, 0.39 %, 80/80 |
| `Locutusque/TinyMistral-248M` | `mistral` | 2/2, 6/6, no template | 32/32 (2/2) | none | 32/32 | 0.997 | 0.99997 – 0.99998 | 1.000000, 0.03 %, 32/32 |
| `THUDM/glm-edge-1.5b-chat` | `glm` | 5/5, 6/6, 3/3 | 80/80 (5/5) | none | 80/80 | 0.984 | 0.9998 – 0.9999 | 0.999998, 0.31 %, 80/80 |
| `unsloth/gemma-3-1b-it` | `gemma3_text` | 5/5, 6/6, 3/3 | 72/80 (4/5) | one user turn: step 8 (0.617), after the answer's `<end_of_turn>` | 79/80 | 0.971 | 0.9988 – 0.9999 | 0.999986, 0.96 %, 80/80 |
| `unsloth/gemma-2-2b-it` | `gemma2` | 4/4, 6/6, 2/2 (the system conversation is refused by the official template) | 64/64 (4/4) | none | 64/64 | 0.991 | 0.9998 – 0.9999 | 0.999998, 0.58 %, 32/32 on the raw prompts (not recorded for the chats) |
| `state-spaces/mamba-130m-hf` | `mamba` | 2/2, 6/6, no template | 22/32 (1/2) | raw prose: step 6 (0.126) | 30/32 | 0.953 | 0.9984 – 0.9990 | 1.000000, 0.28 %, 32/32 |

- **No engine bug showed up.** Against the Q8-rounded reference every family agrees to a Pearson correlation of
  0.99998 or better, a largest error under 1 % of the largest logit, and the same argmax at every step but one (a
  Qwen2.5 near tie, below): once the weights are equal the engine computes what transformers computes. What remains
  against the float32 reference (Pearson 0.998 – 0.9999, top-20 overlap 0.95 – 0.997) is Q8_0 quantization noise; it
  is largest for the smallest model (mamba-130m) and for Gemma 3.
- **Divergences are near ties.** Every free-running divergence happened where the float32 reference itself had a
  margin of 0.03 – 0.13 logits between its choice and the engine's, after which the two continuations are different
  texts. The exception is Gemma 3's one-turn chat: the divergence comes after the answer ended (`<end_of_turn>`, where
  the reference goes on with a run of `<end_of_turn>` and stray tokens), and the Q8-rounded reference makes the same
  choice as the engine (80/80 argmax), so it is the quantized weights' choice, not a computation difference.
- **Tokenizer and templates.** The engine's prompt ids equal transformers' on all prompts and all six probes for every
  model, and the rendered chat text equals `apply_chat_template` for every conversation (Llama 3.2's template prints
  today's date; both sides render the same day). TinyMistral and mamba-130m-hf ship no chat template (raw prompts
  only). Gemma 2's official template raises `System role not supported`; the engine instead folds a system message
  into the first user turn as Gemma 3's template does, so that conversation was not compared.
- **What the prompts reach.** Llama 3 RoPE scaling; Qwen2 q/k/v bias; GLM-Edge's interleaved RoPE, fused
  `gate_up_proj` and tied head; Gemma 3's q/k norms, two RoPE tables and 5 local + 1 global layers; Gemma 2's
  attention and final soft-caps; TinyMistral's 32-token sliding window (the code prompt runs to 39 tokens); Mamba's
  convolution and selective scan. Not reached: Gemma's 512 / 4096-token windows (all prompts are at most 76 tokens
  plus 16).
- **Gemma embedding scale.** The engine multiplies the embedding by `bf16(√1152) = 34.0` (14.10.1, as transformers does
  for the bf16 checkpoint); a float32 load computes `√1152 = 33.94` instead. The 0.17 % difference is inside the
  Gemma 3 numbers above.
- **Reference precision.** float32, except Llama 3.2 and Gemma 2 with `REF_DTYPE=upcast` (the bf16 weights are kept and
  upcast per matmul, so the activations are float32; identical to a float32 load, max difference 0.0 on
  Qwen2.5-0.5B) to fit in memory next to other jobs.
- **Not checked.** Phi-3.5-mini / Phi-4-mini (3.8 B: a 7.6 GB checkpoint plus a 4 GB Q8_0 cache and a 15 GB float32
  reference did not fit the ~14 GB of free disk and the shared 15 GB of RAM of the machine used); an official
  Mistral (the smallest is 7 B; TinyMistral-248M is a community `mistral` checkpoint); `glm4`, Qwen3-MoE and
  DeepSeek (9 B and up); the other quantizations (the script takes `q4` / `q2k` / `q6k`); the int8-activation
  kernels (the default) and the Metal backend, whose agreement with this float path is the arch smoke's job (14.7).
- **Cost.** With float activations one prompt took from 2 minutes (TinyMistral) to 25 minutes (Gemma 2, longer for
  its 52-token chat) on 4 cores shared with other jobs. The script deletes the checkpoint and the cache afterwards
  unless `KEEP=1`; `LOW_DISK=1` deletes the safetensors as soon as they are converted, and
  `JAVA_OPTS=-Dreal.prompts=raw_text,raw_code` runs a subset of the prompts.
## 14.14 Qwen3-Next (hybrid: Gated DeltaNet + gated attention + sparse MoE)

Code: `model/Qwen3NextArch` (an `Architecture` of its own, with its parser), `model/Qwen3NextLayer`,
`ModelConverter.planQwen3Next`, `cpu/Qwen3NextCpuBackend`, `metal/Qwen3NextMetalBackend`, `kernels.metal` section 11
(`gdn_conv`, `gdn_rule`, `gdn_norm_gate`, `sigmoid_mul`, `shared_gate_add`), `scripts/qwen3next-ref.py`,
`scripts/qwen3next-hf-compare.py`, `src/test/.../Qwen3NextChecks`

`Qwen3NextForCausalLM` (`model_type: qwen3_next`; `Qwen/Qwen3-Next-80B-A3B-Instruct` / `-Thinking`) mixes two kinds of
token mixer. Three layers out of four are a **Gated DeltaNet**, a linear attention with a fixed-size memory per head
(like Mamba's state, [chapter 15](15-mamba.md)); every fourth layer is ordinary attention with a KV cache, plus an
output gate. Every layer's MLP is a mixture of 512 experts (10 per token) plus one always-on shared expert. The KV
cache exists only in the attention layers, so it is a quarter of a same-shape transformer's.

### 14.14.0 🧸 The simple story

A transformer remembers a conversation like a **diary**: every word gets a page (the KV cache) and, to write the
next word, the model flips through all the pages. The diary gets thicker with every word.

A DeltaNet layer remembers like a **whiteboard of fixed size**. For each word it writes a note under a label (the
*key*): "under *colour*, write *blue*". If something is already written under that label, it does not just scribble
on top: it first reads what is there, and writes only the **difference** ("it said *red*, it should say *blue*:
change it by *blue minus red*"). That correction is the *delta*. Before each word the whole board also fades a
little (the *gate*), so old, unimportant notes disappear by themselves. To answer, the layer reads the board with a
question (the *query*). The board never grows, however long the conversation.

Fading boards forget details, so every fourth layer still keeps a real diary (full attention). The model gets the
long memory of attention where it matters and the fixed cost of a whiteboard everywhere else.

### 14.14.1 One layer

Every layer is `x += mixer(rmsnorm(x)); x += mlp(rmsnorm(x))`. Which mixer a layer has comes from `layer_types`
(`linear_attention` / `full_attention`), or, when absent, from `full_attention_interval` (default 4): layer `l` is
full attention when `(l + 1) % interval == 0`, i.e. layers 3, 7, …, 47 of 48. Which MLP it has follows the Qwen3-MoE
rule (12.1): experts unless `l` is in `mlp_only_layers` or `(l + 1) % decoder_sparse_step != 0`, given `num_experts > 0`.

**Zero-centred norms.** `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm` and the final `norm` are
`Qwen3NextRMSNorm`, which computes `x / sqrt(mean(x²) + eps) · (1 + w)` with `w` initialised to 0 (like Gemma, 14.10).
The converter adds the 1 once (`f32PlusOne`), so the backends multiply by a plain gain. The DeltaNet's own output
norm (`Qwen3NextRMSNormGated`) is **not** zero-centred (`w` starts at 1) and is stored as is.

**Gated full attention** (`Qwen3NextAttention`):

```
[q | gate] = q_proj · h          per head: head_dim query rows, then head_dim gate rows (q_proj is [heads·2·hd, dim])
q, k       = rmsnorm(q) · (1+w_q), rmsnorm(k_proj · h) · (1+w_k)        per head, over the whole head
q, k       = RoPE on the first rotaryDim = partial_rotary_factor · head_dim dims (rotate-half: i with i + rotaryDim/2)
o          = softmax(q kᵀ / √head_dim) v                 GQA, causal, KV cache
x         += o_proj · (o · sigmoid(gate))                  the output gate, element-wise
```

The converter splits `q_proj` into `q_proj` (the query rows) and `q_gate` (the gate rows), so the attention itself is
the Qwen3 one with `partial_rotary_factor` 0.25 (64 of 256 dims rotate in the 80B).

**Sparse MoE with a gated shared expert** (`Qwen3NextSparseMoeBlock`): softmax router over the experts, top-k, the
weights renormalized when `norm_topk_prob` (true), each chosen expert a SwiGLU of `moe_intermediate_size`; plus a
SwiGLU `shared_expert` of `shared_expert_intermediate_size` on every token, scaled by a learned scalar gate:

```
y = Σ_k w_k · expert_k(h)  +  sigmoid(shared_expert_gate · h) · shared_expert(h)       shared_expert_gate is [1, dim]
```

**Gated DeltaNet** (`Qwen3NextGatedDeltaNet`), with `nk` key heads of `dk`, `nv` value heads of `dv` (`nv` a multiple
of `nk`; each q/k head serves `r = nv / nk` value heads):

```
[q k v z] = in_proj_qkvz · h,  [b a] = in_proj_ba · h        q, k: nk·dk;  v, z: nv·dv;  b, a: nv
[q k v]   = silu(causal_conv1d([q k v]))                     depthwise, conv kernel 4, no bias; conv state = last 3 inputs
q, k      = l2norm(q) / √dk,  l2norm(k)                      per head: x / sqrt(Σx² + 1e-6)
β         = sigmoid(b)                                       per value head: how much of the correction is written
g         = −exp(A_log) · softplus(a + dt_bias)              per value head, ≤ 0: the log of the decay
S, o      = gated delta rule (14.14.2), per value head with the q / k of its key head
o         = rmsnorm(o) · w · silu(z)                         per value head (the gated norm)
x        += out_proj · o
```

### 14.14.2 📐 The gated delta rule

Each value head keeps a state `S` of `dk × dv` numbers (transformers' `[num_v_heads, k_head_dim, v_head_dim]`).
`Sᵀ k` is what the state "says" for key `k`: a vector of `dv` values. One token with key `k` (unit length after the
L2 norm), value `v`, query `q`, decay `γ = exp(g)` ∈ (0, 1] and write strength `β` ∈ (0, 1):

```
S  ← γ S                              forget a little (the gate α_t of the papers)
d  = β (v − Sᵀ k)                     the correction: what should be there minus what is there
S  ← S + k dᵀ                         write the correction under the key
o  = Sᵀ q                             read with the query
```

Written as one formula, `S_t = γ_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ`. With the state transposed (`[dv, dk]`,
as in the Gated DeltaNet paper) that is `S_t = S_{t−1} (α_t (I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ` with `α_t = γ_t`: the
form in the task statement, checked term by term against `torch_recurrent_gated_delta_rule` (decay first, then
`kv_mem = Sᵀ k`, `delta = (v − kv_mem) β`, `S += k deltaᵀ`, `o = Sᵀ q`).

Why "delta": `d` is a gradient step of the regression loss `½ ‖Sᵀ k − v‖²` with learning rate `β` (the Widrow–Hoff
delta rule). A plain linear attention only ever adds `k vᵀ`, so two values written under similar keys pile up; the
delta rule replaces. The gate `γ` (as in Mamba-2) lets the model clear the whole state quickly when the context changes.

**Worked example** (checked by `Qwen3NextChecks.workedExample`): `dk = 2`, `dv = 1`, so the state is one column
`s = (1, 0)`; `γ = 0.5`, `k = q = (0.6, 0.8)`, `v = 2`.

```
s ← 0.5 s                = (0.5, 0)
s · k                    = 0.3                    what the key reads now
β = 1:   d = 2 − 0.3      = 1.7;   s = (0.5 + 1.7·0.6, 1.7·0.8) = (1.52, 1.36);   o = s · q = 0.912 + 1.088 = 2
β = 0.5: d = 0.85;  s = (1.01, 0.68);   o = 0.606 + 0.544 = 1.15 = 0.3 + 0.5 · (2 − 0.3)
```

With `β = 1` the key now reads exactly `v` (the old content under that key was overwritten, not added to); with
`β = 0.5` it reads half way between the old value and the new one.

The cost per token and head is a few passes over the `dk × dv` state (scale, one product for `Sᵀk`, the rank-1
update, one product for `Sᵀq`), the same at token 10 and token 100 000.

### 14.14.3 📐 The chunked form (prefill)

Token by token, every token reads and writes the whole state. For a prompt the same result can be computed a chunk
of `L` tokens (64) at a time, touching the state only at the chunk boundaries. Inside a chunk let `c_t = Σ_{u≤t} g_u`
(the cumulative log decay, so `γ` from token `u` to token `t` is `e^{c_t − c_u}` ≤ 1) and `S₀` the state before the
chunk. Unrolling the recurrence, the decayed state that token `t` reads is
`e^{c_t} S₀ + Σ_{u<t} e^{c_t−c_u} k_u d_uᵀ`, so the corrections satisfy

```
d_t = β_t ( v_t − e^{c_t} S₀ᵀ k_t − Σ_{u<t} e^{c_t−c_u} (k_t·k_u) d_u )      a unit lower-triangular system
o_t = e^{c_t} S₀ᵀ q_t + Σ_{u≤t} e^{c_t−c_u} (q_t·k_u) d_u
S   = e^{c_L} S₀ + Σ_u e^{c_L−c_u} k_u d_uᵀ
```

In matrix form `(I + A) D = β ⊙ (V − Γ K S₀)` with `A_{tu} = β_t e^{c_t−c_u} k_t·k_u` for `u < t`: transformers'
`torch_chunk_gated_delta_rule` builds `T = (I + A)⁻¹` (the "UT transform"; `T` applied to `β K` is the WY form of the
product of the `I − β k kᵀ` factors) with a triangular solve, and gets `D = T(βV) − T(βΓK) S₀`. `Qwen3NextCpuBackend.chunked`
solves the same system by forward substitution, row by row, which needs no explicit inverse:

1. `S₀ᵀ k_t` and `S₀ᵀ q_t` for every token (`2 L dk dv`), the Gram matrices `k_t·k_u` and `q_t·k_u` (`L² dk`);
2. `d_t` in token order from the earlier `d_u` (`L² dv / 2`), then `o_t`;
3. the new state from `S₀` and the chunk's `k_u d_uᵀ` (`L dk dv`).

Everything is a rearrangement of the token recurrence, so the two agree to float rounding: on the tiny models the
150-token prompt in chunks of 64 differs from the token loop by 1.2e-6 of the logit range (chunks of 7: 3.4e-7),
and `qwen3next-ref.py` gives 3e-14 between its two float64 forms. Only differences `c_t − c_u ≤ 0` are exponentiated
(`e^{c_t}` itself may underflow to 0 for a fast-decaying head, which is the right limit), so nothing overflows.

What the chunked form buys: the state (64 KB per head in the 80B) is read and written once per 64 tokens instead of
per token, and the work becomes small dense products that vectorize (and, on a GPU, map to matrix units). On the CPU
the arithmetic is about the same as the token loop; the Metal backend uses the token loop for prefill too (14.14.6).

### 14.14.4 Shapes of the 80B (verified on the Hub) and how they are stored

`config.json` of `Qwen/Qwen3-Next-80B-A3B-Instruct`: `hidden_size` 2048, 48 layers, `full_attention_interval` 4,
attention 16 heads / 2 KV heads × `head_dim` 256, `partial_rotary_factor` 0.25, `rope_theta` 1e7, DeltaNet 16 key heads ×
128, 32 value heads × 128, conv kernel 4, 512 experts of 512 (10 per token), shared expert 512, `intermediate_size`
5120 (unused: no `mlp_only_layers`), vocabulary 151936, untied head, `max_position_embeddings` 262144. 75 944 tensors,
41 shards; shapes read from the safetensors headers:

| Tensor (layer 0: DeltaNet, layer 3: attention) | Shape | Stored in jllm as |
|---|---|---|
| `linear_attn.in_proj_qkvz.weight` | `[12288, 2048]` = per key head `[q 128 \| k 128 \| v 256 \| z 256]` rows | with `in_proj_ba` one `linear_attn.in_proj` `[12352, 2048]`, rows reordered to `[q all heads \| k \| v \| z \| b \| a]` (`gather` rule) |
| `linear_attn.in_proj_ba.weight` | `[64, 2048]` = per key head `[b 2 \| a 2]` | (in `in_proj`) |
| `linear_attn.conv1d.weight` | `[8192, 1, 4]` | f32 |
| `linear_attn.A_log`, `dt_bias` | `[32]` each | f32, `A = −exp(A_log)` |
| `linear_attn.norm.weight` | `[128]` (gated norm, shared by the heads) | f32, as is |
| `linear_attn.out_proj.weight` | `[2048, 4096]` | quantized |
| `self_attn.q_proj.weight` | `[8192, 2048]` = per head `[query 256 \| gate 256]` | `q_proj` `[4096, 2048]` + `q_gate` `[4096, 2048]` |
| `self_attn.k_proj` / `v_proj` / `o_proj` | `[512, 2048]` / `[512, 2048]` / `[2048, 4096]` | quantized |
| `self_attn.q_norm` / `k_norm`, `input_layernorm`, `post_attention_layernorm`, `model.norm` | `[256]` / `[2048]` | f32, `1 + w` folded |
| `mlp.gate.weight` (router), `mlp.shared_expert_gate.weight` | `[512, 2048]`, `[1, 2048]` | f32 |
| `mlp.experts.N.{gate,up,down}_proj`, `mlp.shared_expert.*` | `[512, 2048]`, `[2048, 512]` | stacked like Qwen3-MoE (12.3); the shared expert as `mlp.shared.gate_up` / `.down` |
| `mtp.*` (one multi-token-prediction layer) | | never read |

The reorder makes the first `convDim = 2·nk·dk + nv·dv` outputs of `in_proj` the conv channels in the order of
transformers' `mixed_qkv` (all q heads, all k heads, all v heads), so the conv reads one contiguous slice per token.
Transformers ≥ 5 can also save the experts fused (`experts.gate_up_proj` 3-D); the converter reads both layouts.

### 14.14.5 State instead of most of the KV cache

Per DeltaNet layer the backends keep (f32)

```
conv state       convDim × (kernel − 1) = 8192 × 3 × 4 B            =    96 KB
recurrent state  nv × dv × dk          = 32 × 128 × 128 × 4 B      = 2 048 KB
```

so 36 layers hold **79 MB, whatever the context**. Only the 12 attention layers have a KV cache:

| per token of context | Qwen3-Next-80B-A3B | same attention shape in all 48 layers (= Qwen3-30B-A3B: 48 layers × 4 × 128) |
|---|---|---|
| KV cache f32 | 12 × 2 × 512 × 4 B = **48 KB** | 48 × 2 × 512 × 4 B = 192 KB |
| KV cache f16 | **24 KB** | 96 KB |
| at `--ctx 4096`, f32 | 201 MB + 79 MB of state = 280 MB | 805 MB |
| at the full 262 144 tokens, f16 | 6.4 GB + 79 MB | 25.8 GB |

The KV cache is 4× smaller per token; the fixed state equals the KV of ~1 600 tokens (f32), so the hybrid is
smaller as soon as the context exceeds ~550 tokens. (`docs/11` §11.3 lists the same numbers.)

The state follows Mamba's contract (15.5, 15.6): `pos == 0` resets it, any other `pos` must equal the number of tokens
fed, it cannot be rewound (`supportsPrefixReuse()` is false, so the engine resumes only when the new prompt extends
everything cached and otherwise prefills from 0, and speculative decoding is off). Unlike Mamba it can be **saved,
loaded and copied** (`stateFormat`, `saveState`, `loadState`, `copyStateFrom`: the prompt cache files and the
server's slot-to-slot prefix copies of chapter 10): the bytes are the KV rows of the attention layers in `KvCache.write`
order followed by each DeltaNet layer's conv and recurrent state, identical on both backends. Because a recurrent
state cannot be cut, saving `n` tokens is refused unless the state holds exactly `n` (the engine only ever asks for
that). `backend.kv-dtype=f16` applies to the attention layers; the recurrent state stays f32.

### 14.14.6 The implementations

**CPU (`Qwen3NextCpuBackend`).** The projections are `CpuMatmul` (int8 activations for single tokens when enabled),
the attention layers reuse `KvCache` (allocated for the attention layers only: `KvCache.create(dtype,
fullLayers, ...)`, layer index = rank among the attention layers), the experts reuse `MoeExperts`. The conv is
`MambaCpuBackend.convStep` per channel (parallel over channels). The recurrent state of a head is stored transposed,
`S[h][j][i]` (value dim major), so each of the `dv` columns that the token loop updates is a contiguous `dk`-vector:
decode runs `scale, dot, axpy, dot` on it with the Vector API, parallel over (head, column). Chunks of several tokens
use the chunked form, parallel over heads (`-Djllm.gdn.chunked=false` forces the token loop, `-Djllm.gdn.chunk=N`
sets the chunk length).

**Metal (`Qwen3NextMetalBackend`).** One command buffer per call. The projections, norms, RoPE, attention and experts
are the existing kernels (`matmul`, `matmul3`, `rmsnorm`, `rope`, `attention_t`, `router`, `moe_up` / `moe_down` or the
expert-grouped MMA kernels, `matmul_swiglu`); new kernels in section 11:

- `gdn_conv`: one thread per channel, the conv window and taps in registers for the chunk (as `mamba_conv`, but no bias
  and the input read from the fused projection row);
- `gdn_rule`: one **simdgroup per state column**: lane `l` holds rows `l, l + 32, …` of column `j` of head `h` in
  registers (`dk / 32` floats; `dk ≤ 256`), `Sᵀk` and `Sᵀq` are `simd_sum`s, and the simdgroup recomputes the q / k L2
  norms of its token with two more `simd_sum`s (cheaper than a separate pass). Grid `dv / 4 × nv` threadgroups of
  4 simdgroups: 4 096 simdgroups for the 80B, independent, each running the chunk's tokens in order. Prefill is
  therefore the token loop, not the chunked form (documented trade-off as for Mamba's scan, 15.7): simple and exactly
  the decode arithmetic; a chunked GPU kernel (matrix units for the Gram and state products) is the next step for long
  prompts;
- `gdn_norm_gate`: one threadgroup per (token, value head) row;
- `sigmoid_mul` (the attention output gate) and `shared_gate_add` (`x += sigmoid(w · xb) · shared`).

The per-head q / k norms of the attention layers run through the plain `rmsnorm` kernel on rows of `head_dim` (the
`rope` kernel's fused norm only supports a full rotation), then `rope` rotates the 64-dim slice and stores k into the
cache. No existing kernel or buffer index changed.

### 14.14.7 Verification

| check | result |
|---|---|
| `qwen3next-ref.py` (float64 numpy, written from `modeling_qwen3_next.py`) vs transformers 5.20 `Qwen3NextForCausalLM` in float64 (`qwen3next-hf-compare.py`): last-token logits of the 12-token prompt, then 8 greedy tokens with the cache (single-token decode: recurrent rule, conv update, KV cache) | max difference 3.3e-5 (`qwen3next`, logits up to 20) and 3.2e-6 (`qwen3next2`); identical tokens. The residue is transformers computing the delta rule in float32 even in a float64 model |
| reference recurrent form vs chunked form (chunks of 5 and 64) | 3e-14 |
| CPU backend (Q8_0) vs the reference on Q8_0-rounded weights (`REF_Q8=1`) | 3.7e-6 / 1.1e-6 of the logit range; the 8 greedy tokens identical |
| CPU, 150 tokens: chunks of 64, chunks of 7, one call per token vs the token loop | ≤ 1.4e-6 |
| CPU state saved + loaded / copied, f32 and f16 KV | bit-identical continuation; saving another length refused |
| Metal backend (on the emulator, doc 18) vs the reference | 2.2e-6 / 7.8e-7; the 8 greedy tokens identical |
| Metal, 150 tokens: one call, chunks of 16, token by token; vs CPU | ≤ 1.5e-6 between them, ≤ 1.8e-6 from the CPU |
| Metal state saved + loaded / copied (f32, f16 KV); the CPU's file loaded into Metal | bit-identical; CPU file: same format and size, continuation within 1e-5 |
| kernel tests (`kernel_tests.py`): `gdn_conv`, `gdn_rule` (dk 32 / 64 / 128, 1–3 value heads per key head), `gdn_norm_gate` vs numpy | ≤ 7.6e-7 |

The tiny models (`make-tiny-model.py qwen3next`: 4 layers, interval 2, head_dim 64 with 16 rotary dims, 2 × 32 keys /
4 × 32 values, 8 experts + shared, one dense layer; `qwen3next2`: explicit `layer_types` L L F L L, one value head per
key head with dk 64 ≠ dv 32, conv 3, head_dim 32, MoE every other layer with `norm_topk_prob` false, tied head) cover
the variants of the config. The real 80B config parses to the sizes above (`Qwen3NextChecks.realConfig`).

```
METAL_EMU=1 scripts/arch-smoke.sh          # includes both tiny models: CPU == Metal in every format, chunking, state
REF_Q8=1 python3 scripts/qwen3next-ref.py <dir>; java ... Qwen3NextChecks <metal-lib|-> <dir>...
python3 scripts/qwen3next-hf-compare.py <dir>   # needs torch + transformers
```

### 14.14.8 Not supported

- **Multi-token prediction in the Hugging Face checkpoints** (`mtp.*`, one extra layer that predicts the token after
  next from the last hidden state and the next token's embedding): ignored by the converter, as transformers does for
  inference. A **Qwen3.5 GGUF** carries the same layer (`nextn.*` tensors) and *is* used, as a built-in draft for
  speculative decoding: `engine.speculative=mtp`, [16.14](16-gguf.md#1614-multi-token-prediction-the-model-as-its-own-draft-model).
  The recurrent state cannot be rewound by overwriting, so that mode checkpoints it around each verification pass.
- **GGUF** (llama.cpp's `qwen3next` and `qwen35` architectures): supported, [16.13](16-gguf.md#1613-qwen35-and-qwen3-next-as-gguf-files).
  The `gguf` package lists both the fused DeltaNet input `ssm_in` + `ssm_ba` and the split `attn_qkv` + `attn_gate`; the
  split one (what current converters write) is mapped (and `qwen35moe`, on tiny models only). The legacy fused `ssm_in` is not.
- The chunked delta rule on Metal (prefill uses the token loop); a state snapshot to rewind to a shared prefix;
  `backend.metal.cpu-moe` (experts in CPU memory, 12.x) and flash-attention / flash decoding for the attention layers
  (they use `attention_t`); measured speed on real weights (an 80B checkpoint does not fit this machine).
- `attention_bias: true`; `rope_scaling` types other than those of 14.1 (the released models use none); a
  `linear_key_head_dim` that is not a multiple of 32 or above 256, or a conv kernel above 8, on Metal.

## 14.18 gpt-oss (attention sinks, sliding windows, clamped-SwiGLU experts with biases, MXFP4)

Code: `model/Families.gptOss` (parser), `model/Attention.Gqa` (`sinks`, `window`), `model/AttnWeights.Gqa.sinks`,
`model/Mlp.ExpertSpec`, `model/Router.TopLogits`, `model/ModelConverter` (`topLogitsRules`, `mxfp4()`,
`transposedExperts`), `cpu/KvCache` (sink-aware softmax), `cpu/MoeExperts` (`activate`), `cpu/CpuBackend`,
`metal/MetalBackend` (`GqaStep`, `MoeStep`), `metal/MetalKernels` (`AttnExtra`, `MoeMath`),
`kernels.metal` (`attention_t` with its `ext` buffer, `moe_act` / `moe_out` inside the `moe_*` kernels, `router_ds`
method 3), `tokenizer/ChatTemplate.harmony`, `scripts/hf-arch-ref.py` (families `gptoss`, `gptoss-mxfp4`)

`GptOssForCausalLM` (`model_type: gpt_oss`; `openai/gpt-oss-20b`, `openai/gpt-oss-120b`) is a grouped-query-attention
transformer with a mixture-of-experts MLP, and five differences from the Llama-style layer of 14.1:

| | gpt-oss-20b (read from the Hub config) | what is new |
|---|---|---|
| layers / hidden | 24 / 2880 (the 120B: 36) | |
| attention | 64 heads of 64 over 8 KV heads, q/k/v/o **with bias** | a learned **sink** per head |
| layer kinds | `layer_types`: `sliding_attention`, `full_attention` alternating, `sliding_window: 128` | **alternating windows** |
| position | RoPE theta 150000, YaRN factor 32 on 4096 (`truncate: false`), 131072 positions | YaRN **without truncation** |
| MLP | 32 experts, 4 per token, expert width 2880, router with bias | **clamped SwiGLU with biases**, softmax **after** top-k |
| expert storage | `quant_method: mxfp4` | **MXFP4** blocks read in place |

### 14.18.0 🧸 In plain words

Attention is a vote. To write the next word the model asks every earlier word "how much should I listen to you?",
and the answers are turned into percentages that must add up to 100. That has a flaw: even when nothing earlier is
useful, the 100 percent still has to go somewhere, so the model is forced to over-listen to some word. gpt-oss gives
every attention head **one extra, made-up listener** that never says anything: a head with nothing worth hearing parks its
percentage there. The extra listener ("sink") has no content to read; it only takes a share of the 100. Its size is
learned.

Half of the layers are short-sighted: they read only the last 128 words, like a reading lamp (cheap, local grammar and
spelling). The other half read everything. The feed-forward part is not one big brain but **32 small specialists**, and
for each word the router picks **4** of them; their answers are mixed with the weights the router gave. And the
specialists' numbers are stored in a very compact 4-bit code (32 numbers share one scale), which the engine reads
**as stored** rather than widening them first, so the 20B's specialists need about 10 GB, not 19.

### 14.18.1 📐 How it works

**The layer pattern** (24 layers, `L` = 128-token window, `G` = global; every layer also has sinks and an MoE):

```
layer:   0 1 2 3 4 5 6 7 8 9 ...                 23
         L G L G L G L G L G ...                  G
         each: x -> RMSNorm -> attention(+sinks) -> +x -> RMSNorm -> MoE(4 of 32) -> +x
```

**Sinks.** Each head `h` has a learned scalar `s_h` (`self_attn.sinks`, one per query head). With the usual scaled
scores `z_j = (q . k_j) * scale` over the visible keys `j`, the weights are

```
m   = max(s_h, max_j z_j)
p_j = exp(z_j - m) / ( exp(s_h - m) + sum_i exp(z_i - m) )         output = sum_j p_j v_j
```

and the sink's own share `exp(s_h - m) / (...)` is simply dropped: the `p_j` add up to less than 1. Worked example,
three visible keys with scores `z = [2.0, 1.0, 0.5]`:

| sink `s_h` | weights `p_1, p_2, p_3` | kept (sum) | parked on the sink |
|---|---|---|---|
| none (plain softmax) | 0.6285, 0.2312, 0.1402 | 1.0000 | |
| -4.0 (a head that wants its keys) | 0.6276, 0.2309, 0.1400 | 0.9984 | 0.0016 |
| 1.5 | 0.4551, 0.1674, 0.1015 | 0.7240 | 0.2760 |

and with nearly flat scores `[0.1, 0.0, -0.2]` and a sink of 3.0 the head keeps only 12.7% of its attention (weights
0.048, 0.044, 0.036): the head has effectively switched itself off for this token, which a plain softmax cannot do.

**Windows.** A sliding layer lets query position `i` see keys `j` with `i - 127 <= j <= i` (the first visible key is
`max(0, i - W + 1)`, `W` = 128). The window cuts the visible set; the sink is added after the cut. Diagram for `W = 4`,
query 9 (`.` hidden, `#` visible, `s` the sink):

```
key j:   0 1 2 3 4 5 6 7 8 9
query 9: . . . . . . # # # #   + s        (a global layer would show # for all of 0..9)
```

**YaRN without truncation.** The rotary frequencies of dimension pair `d` (`0 <= d < 32` for head size 64) are blended
between the original frequency and the one divided by `factor`, with a linear ramp between two correction dimensions
`low`, `high` computed from `beta_fast = 32` and `beta_slow = 1`:

```
corr(r) = dim * ln(L0 / (r * 2 pi)) / (2 ln(theta))        L0 = 4096, dim = 64, theta = 150000
low  = corr(32) = 8.09        high = corr(1) = 17.40
truncate = true  (every other YaRN model):  low = floor(8.09) = 8,  high = ceil(17.40) = 18
truncate = false (gpt-oss):                 low = 8.09,            high = 17.40
```

so the ramp is slightly different, and the frequencies differ from the truncated ones in the last digits of the angle
for the ramp dimensions; at long positions that is a different model, which is why the flag is read
(`Scaling.Yarn.truncate`, `Rope.yarnInvFreq(..., truncate)`). The amplitude factor `0.1 ln(32) + 1 = 1.347` multiplies
cos and sin as for every YaRN model (14.11.2).

**Router and experts.** The router is a linear layer with bias, `r = W_r x + b_r`, `r in R^32`. The 4 largest entries are
picked first and the softmax runs over **only those 4**: `w = softmax(r_top4)` (`Router.TopLogits`; Granite's MoE uses
the same rule, 14.22). With logits `[1.2, -0.3, 0.9, 0.4]`, bias `[0.1, 0, 0, 0.2]` and top-2 for the example: logits plus
bias `[1.3, -0.3, 0.9, 0.6]`, picked experts 0 and 2, weights `softmax(1.3, 0.9) = [0.5987, 0.4013]`.

Each picked expert has a gate projection `g` and an up projection `u` (both **with bias**), then

```
g' = min(g, limit)          u' = clamp(u, -limit, +limit)          limit = 7.0 (swiglu_limit)
act = (u' + 1) * g' * sigmoid(alpha * g')                          alpha = 1.702
out = down_proj(act) + b_down            result = sum over the picked experts of w_e * out_e
```

Worked numbers: `g = 9, u = -3` gives `g' = 7`, `u' = -3`, `act = (-3 + 1) * 7 * sigmoid(11.91) = -13.9999`;
`g = 0.5, u = 2` gives `act = 3 * 0.5 * sigmoid(0.851) = 1.0512`. Only the gate side is clamped from above (a large
negative gate is already squashed to zero by the sigmoid). `Mlp.ExpertSpec(clampLimit, alpha, bias, weightInput,
interleaved)` carries these numbers; the same record serves Llama 4 (14.19) with different values.

In the checkpoint the gate and up rows are **interleaved**: row `2i` is gate `i`, row `2i + 1` is up `i`; the converter
separates them into the engine's gate-then-up layout. The expert tensors are stored `[E, in, out]` (the transpose of a
Linear), so `ModelConverter.transposedExperts` re-lays them out per expert, quantizing each to Q4_0 / Q8_0 with
`QTensor.allocate` / `setRows`.

**MXFP4.** `*_blocks` are uint8 (two 4-bit codes per byte) and `*_scales` uint8 per 32 values; the scale byte `s` means
`2^(s - 127)` (an 8-bit exponent), the code is E2M1: a sign bit and three bits that index the magnitudes
`[0, 0.5, 1, 1.5, 2, 3, 4, 6]`. Example: scale byte 130 (factor 8) and code `0b0101` (magnitude index 5 = 3) is the value
`+24`; code `0b1011` (sign, index 3 = 1.5) is `-12`. Layout of one block of 32 values, HF against ggml:

```
HF  (blocks[..., 16] bytes):   byte j = [ value 2j+1 : high nibble | value 2j : low nibble ]
ggml MXFP4 (17 bytes):         [ scale byte | byte j = [ value j+16 : high nibble | value j : low nibble ] for j = 0..15 ]
```

`ModelConverter.mxfp4()` repacks the nibbles from the first order into the second and prepends the scale byte, so the same
block type as a GGUF `MXFP4` tensor results (`QType.MXFP4`, kernels on both backends, docs/16). The experts stay at 4.25
bits per weight in memory (17 bytes for 32 weights): the 20B's 19.1 billion expert weights occupy about 10.1 GB, a Q8_0 copy
would be 20.3 GB.

**Chat format (harmony).** `ChatTemplate.harmony(date)`: a system message with the knowledge cutoff, the date, the
reasoning effort and the channel list, the turns as `<|start|>role<|message|>...<|end|>`, and the assistant turn opened
with `<|start|>assistant`. The reasoning effort comes from the request's `enable_thinking`: true is `Reasoning: medium`,
false is `Reasoning: low` with the answer channel already open (`<|channel|>final<|message|>`). The generation stops on
`<|return|>`, `<|call|>` or `<|endoftext|>`. The render equals the model's Jinja template on 36 conversations
(`TokenizerFamiliesTest`, date pinned to 2025-08-05 in the fixtures, `src/test/python/gen_tokenizer_fixtures.py`).

**On Metal.** `attention_t` takes the sinks, the chunk size and the temperature through one extra buffer (`ext`, buffer 5:
three ints, one float, then one float sink per head; `MetalKernels.AttnExtra`), so the shared `AttnParams` struct and every
existing kernel binding are unchanged. The flash and decode attention kernels do not know the sinks: a layer that has them
always takes the generic kernel (`MetalBackend.GqaStep.special`). The MoE kernels got a `MoeXParams` buffer with the clamp,
alpha, the biases and a "weight on input" flag (used by Llama 4); `router_ds` method 3 is "top-k of the logits plus a linear
bias, then softmax".

### 14.18.2 Why this design

- **Sinks as one extra softmax term** instead of a fake key/value pair in the cache: no cache row is spent, the cache
  layout and its saved format are unchanged, and the term costs one `exp` per head.
- **One generic attention kernel with an extras buffer** instead of adding a sink argument to every flash/decode/prefill
  variant: those variants are the hot, tuned paths of every other family; a family-specific fork of each would multiply the
  code that has to stay bit-compatible. The price is that gpt-oss and Llama 4 layers do not get flash decoding on Metal
  (slower at long context; not measured here, see 14.18.5).
- **MXFP4 read in place** (not widened to Q8 at load): it halves the memory of the experts, which is 90% of the weights, and
  the same kernels read GGUF `MXFP4` files. Rejected: dequantizing to Q8_0 while converting (simple, but the 120B would then
  not fit a 64 GB machine).
- **`truncate` as data in `Scaling.Yarn`** instead of a gpt-oss special case: the transformers config carries the flag, so
  other models that set it are covered.

### 14.18.3 Settings

No new setting: the family is selected by `model_type`. Existing settings apply: `backend.kv-dtype` (f32 / f16 / q8; **q4 is
served as q8** for these layers, a note is printed, because the 4-bit cache kernels do not take sinks, 14.18.5),
`chat.thinking` (and a request's `enable_thinking`), which selects the harmony reasoning effort (true: medium, false: low), and `backend.metal.cpu-moe` for experts in CPU memory
(works with the extras; checked in `ArchSmoke.moeSplitCheck`). The `-Djllm.metal.*` tuner flags need no change. There is no
switch for the sink path itself because there is no "old path" for a model that has sinks.

### 14.18.4 Measured

Machine: Apple M1 Pro, 32 GB, 2026-10-22. Against transformers 5.20 (torch 2.14) on **tiny random** `GptOssForCausalLM` models
(4 layers `L G L G`, hidden 128, 4 heads of 32 over 2, 8 experts top 2, clamp 3, window 4, YaRN factor 32 on 16 positions,
`truncate` false, vocabulary 1000), weights rounded to the Q8_0 grid in the reference so the engine at q8 has the same
weights. Prompt of 40 ids plus 12 greedy ids, so 52 logit rows. Command: `REF_PYTHON=<python with torch> scripts/hf-arch-check.sh gptoss gptoss-mxfp4`.

| model | backend | max abs logit difference (relative to the largest logit) | greedy ids | argmax at 52 positions |
|---|---|---|---|---|
| gptoss, experts as float tensors | CPU | 6.4e-6 | 12 of 12 equal | equal |
| gptoss, experts as float tensors | Metal | 7.6e-6 | 12 of 12 equal | equal |
| gptoss-mxfp4 (experts MXFP4, read in place) | CPU | 9.1e-6 | 12 of 12 equal | equal |
| gptoss-mxfp4 | Metal | 6.5e-6 | 12 of 12 equal | equal |
| gptoss, Metal with the f16 KV cache | Metal | within 4e-3 | equal | equal |

One call, chunks of 7 and one token at a time give the same last-row logits (≤ 4e-6 on CPU), which crosses the window
and the tile boundaries. No tokens-per-second figure is claimed: the real weights were not downloaded (only the config,
tokenizer and chat template of `openai/gpt-oss-20b`), and a tiny model says nothing about speed.

### 14.18.5 Checks and limits

- **Checks:** the table above; `kernel_tests.py` (numpy) covers `attention_t` with sinks / window / chunk / temperature, the
  clamped-SwiGLU, bias and weight-on-input variants of `moe_up` / `moe_down` and the matrix (`_mma`, `_hmma`) forms in
  q4 / q8 / MXFP4, and `router_ds` method 3; `SmokeTest.kvCacheAttention` checks sink and chunk masks in f32 / f16 / q8 / q4
  caches; `ArchSmoke` runs `GptOss` and `GptOssMxfp4` tiny models through the whole battery (CPU == Metal, batching,
  q8 cache, replay, saved caches, experts on the CPU).
- **Batched rows equal single steps:** the arch smoke checks that three slots decoded in one batched pass give the logits of
  three lone steps. In its second pass the matrix-unit kernels are forced on from 2 tokens (`mmaMinTokens=2`,
  `moeMmaMin=2`), so the 3 rows took other kernels than a lone step and differed by 1e-5 (and by 2.6e-2 with the q8 cache,
  where one flipped int8 code is amplified by these models). `forwardBatch` now encodes fewer than 8 rows on the matrix-vector
  kernels a lone step uses (`MetalKernels.singleRows`); the default thresholds never put so few rows on the matrix kernels,
  so only the forced test configuration changes, and the check reads 0.0 (bit-identical) for gpt-oss and Llama 4 too.
- **A reference caveat found on the way:** the reference rounds weights to Q8_0 with round-half-up like the engine's
  `Math.round`; `torch.round` rounds half to even and made the first comparisons look like 1e-2 errors.
- **Not done:** q4 KV cache for these layers; flash / flash-decoding attention for layers with sinks on Metal; real-weight
  validation of the 20B / 120B; tool calling and separating the `analysis` channel into `reasoning_content` (the harmony
  render covers plain chats).

## 14.19 Llama 4 text (chunked attention, NoPE layers, temperature tuning, sigmoid MoE)

Code: `model/Families.llama4` (the `Llama4ForConditionalGeneration` wrapper and `llama4_text`), `model/Attention.Gqa`
(`chunk`, `l2Norm`, `tempFloor`, `tempScale`, `tempFactor`), `model/Scaling.Identity`, `model/Mlp.ExpertSpec.weightInput`,
`model/Router.TopLogits(sigmoid)`, `cpu/KvCache`, `cpu/MoeExperts`, `metal/MetalBackend`, `kernels.metal` (`attention_t`
chunk and temperature in `ext`, `moe_*` weight on input), `tokenizer/ChatTemplate.LLAMA4`, `scripts/hf-arch-ref.py`
(`llama4`, `llama4-mix`)

`meta-llama/Llama-4-Scout-17B-16E-Instruct` and `-Maverick-17B-128E-Instruct`, **text tower only** (the vision tower and the
projector are ignored). From the Scout config read on the Hub: 48 layers, hidden 5120, 40 heads of 128 over 8 KV heads,
16 experts with 1 per token plus a shared expert in every layer, `attention_chunk_size 8192`, llama3 RoPE scaling
`factor 16` with `low_freq_factor = high_freq_factor = 1`, `attn_temperature_tuning` with `attn_scale 0.1`,
`floor_scale 8192`, `use_qk_norm`, `max_position_embeddings 10485760`.

### 14.19.0 🧸 In plain words

Imagine reading a very long book. Three of every four readers (layers) read it **chapter by chapter**: they know exactly
where they are inside their chapter (8192 words) and never look into another chapter. The fourth reader skims the
**whole book at once**, but she has no page numbers at all, so she asks "what is this about?" rather than "where is it?".
When the book gets very long her questions are made a bit louder (the "temperature"), so that a faint match among a million
pages can still be heard; for short texts the volume is not touched.

The experts: each word goes to **one** specialist only, chosen by the router, and the router's confidence is applied to
the word **before** the specialist reads it (a doubtful router makes the specialist hear a whisper). A second, always-on
"shared" specialist reads every word too.

### 14.19.1 📐 How it works

**The layer pattern ("iRoPE")**, `no_rope_layers` of the Scout config (0 = no positional encoding):

```
layer:  0 1 2 3 | 4 5 6 7 | 8 9 10 11 | ...        47
kind:   C C C N | C C C N | C C C  N  | ...         N
C = RoPE + chunked attention (8192) + L2 qk-norm        N = NoPE + global attention + temperature
every layer: MoE (16 experts, top 1, sigmoid, weight on input) + shared expert
```

**Chunked attention.** With chunk size `C`, query `i` sees keys `j` with `j <= i` and `floor(j / C) = floor(i / C)`: the
first visible key is `floor(i / C) * C` (combined with a window with `max`, as `KvCache.firstKey` does). Diagram for
`C = 4`:

```
key j:    0 1 2 3 | 4 5 6 7 | 8 9
query 2:  # # # . | . . . . | . .        sees 0..2
query 5:  . . . . | # # . . | . .        sees 4..5   (nothing of chunk 0)
query 9:  . . . . | . . . . | # #        sees 8..9
```

A token at the start of a chunk therefore sees only itself; the NoPE layers are what carries information across chunks.

**NoPE.** A layer with `no_rope_layers[l] = 0` uses `Scaling.Identity`: a rotary table of cos 1 and sin 0. The same code
path serves it (on Metal the `rope` kernel then simply copies k into the cache). The RoPE layers use `llama3` scaling; with
`low_freq_factor = high_freq_factor` the usual smooth interpolation degenerates to a step at one wavelength, which the
implementation now allows (it used to reject equal factors).

**Temperature tuning** (NoPE layers only). The query of position `p` is multiplied by

```
t(p) = 1 + attn_scale * ln( floor((p + 1) / floor_scale) + 1 )         attn_scale 0.1, floor_scale 8192
```

| position `p` | `floor((p+1)/8192)` | `t(p)` |
|---|---|---|
| 100 | 0 | 1.0000 (untouched) |
| 8191 | 1 | 1.0693 |
| 20000 | 2 | 1.1099 |
| 100000 | 12 | 1.2565 |

so below 8191 tokens the temperature changes nothing, which is why short prompts are unaffected.
`Attention.Gqa.tempFactor(p)` computes it; the attention kernels take `tempFloor` and `tempScale` in `ext` and apply it to q
on the fly.

**L2 q/k norm** (`use_qk_norm`, RoPE layers only): after the rotation each head of q and of k is divided by its RMS,
`x / sqrt(mean(x^2) + eps)`, with no learned weight. It is the existing qk-norm path with a weight vector of ones
(`AttnWeights.Gqa.ones`). The subtle part: the head is rotated with the **interleaved** pairing, so on Metal the norm must
see the rotated head; `MetalBackend` keeps that order when the whole head rotates (`fullRotation`).

**MoE with the weight on the input.** `Router.TopLogits(sigmoid = true)`: top-1 of the router logits, the weight is
`sigmoid(logit)` (logit 1.2 gives 0.7685, logit -0.5 gives 0.3775; the weights of several picked experts are **not**
renormalized). `ExpertSpec.weightInput` multiplies the **input** of the expert by the weight, `down(act(gate(w x)) * up(w x))`,
instead of its output. The shared expert is a plain SwiGLU MLP added to the result. Experts are stored fused
`[E, in, out]` (`gate_up_proj`, `down_proj`), transposed like gpt-oss'; `interleave_moe_layer_step` selects the MoE layers
(Scout 1: every layer; Maverick 2: alternating with dense layers of width `intermediate_size_mlp`).

**Chat format** (`ChatTemplate.LLAMA4`): `<|begin_of_text|>`, an optional trimmed system turn
`<|header_start|>system<|header_end|>\n\n...<|eot|>`, each message as `<|header_start|>role<|header_end|>\n\n` + content +
`<|eot|>`, then the assistant header as the generation prompt. Stops: `<|eot|>`, `<|end_of_text|>`, `<|eom|>`. Equal to the Jinja
template on 18 conversations; tools and images are not rendered.

### 14.19.2 Why this design

- **Chunk and temperature as per-layer data on `Attention.Gqa`** (not a new layer class): the layers are otherwise ordinary
  GQA layers, and one record per layer keeps the CPU, the Metal backend and the GGUF reader on a single description.
- **Identity rotary table for NoPE** instead of a branch in every kernel: zero extra code in rope/attention, at the cost of
  one useless pass over q and k on those layers (a quarter of the layers; negligible next to the matmuls).
- **Full cache for chunked layers.** The cache keeps all positions, although a chunked layer never reads rows before its
  chunk. A ring/per-chunk cache would cut the KV memory of 3 of 4 layers at long context (Scout's headline 10M context)
  but changes the cache layout and the saved-cache format for every backend; not done, see 14.19.4.
- **Weight on input** is the model's definition (not an approximation), and it has to happen before the nonlinearity.

### 14.19.3 Settings

No new setting. Chunked and NoPE layers use the generic attention kernel; `backend.kv-dtype=q4` is served as q8 for models
with chunked attention, sinks or temperature (printed once). No chat setting beyond `chat.system`.

### 14.19.4 Measured and checked

Machine: Apple M1 Pro, 32 GB, 2026-10-22; transformers 5.20 on tiny random `Llama4ForCausalLM` models (4 layers, NoPE
every 4th, hidden 128, 4 heads of 32 over 2, 4 experts top 1 plus shared, chunk 8, `floor_scale 8`, `attn_scale 0.4` so
tuning is active inside the 52 positions, llama3 RoPE factor 8, L2 norm); `llama4-mix` uses the wrapper layout
(`language_model.model.*` names) with `interleave_moe_layer_step 2`. Command: `scripts/hf-arch-check.sh llama4 llama4-mix`.

| model | CPU | Metal | greedy ids | argmax |
|---|---|---|---|---|
| llama4 | 1.5e-6 | 1.7e-6 | 12 of 12 equal | equal at all 52 |
| llama4-mix | 1.7e-6 | 1.7e-6 | 12 of 12 equal | equal at all 52 |

(max abs logit difference relative to the largest logit.) Chunk sizes 7 and 1 agree with one call; the f16 cache is within
4e-3. Kernel tests cover the chunk and temperature variants of `attention_t`; `ArchSmoke` runs `Llama4` and `Llama4Mix`.

**Not done:** the vision tower and image inputs; a chunk-limited KV cache (the memory benefit of chunked attention); q4 KV;
Metal flash attention for chunked / tuned layers; real-weight validation (Scout is 109B parameters); long-context behaviour
beyond what a 40-token tiny model shows.

## 14.20 Dual Chunk Attention for Qwen2.5-1M: the design, not implemented

`Qwen/Qwen2.5-7B-Instruct-1M` and `-14B-Instruct-1M` load as ordinary `qwen2` models. The config read from the Hub is a plain
Qwen2.5 (28 layers, hidden 3584, 28 heads over 4 KV heads, `rope_theta 10000000`, `max_position_embeddings 1010000`,
`rope_scaling: null`) plus one extra key:

```
"dual_chunk_attention_config": {"chunk_size": 262144, "local_size": 8192, "original_max_position_embeddings": 262144}
```

Without that key honoured the model runs with ordinary attention: identical to the intended one up to 262144 tokens (see
below) and degraded beyond.

### 14.20.0 🧸 In plain words

A model that was trained on pages numbered 1 to 262144 gets confused when handed page 800000: it never saw a gap that
big between two words. Dual Chunk Attention keeps the model from ever seeing a gap bigger than it was trained on. It cuts the
text into big chunks and, instead of the true page numbers, tells the model different "relative" distances depending on how
far apart two words really are: neighbours inside a chunk get the true distance; a word and one in a *much earlier* chunk
are told "you are about one chunk apart" (they are a long way apart, but the model can't tell the exact length anyway); and
words just across a chunk boundary keep their exact small distance, so nearby text never gets blurry.

### 14.20.1 📐 How it would work

Let `s = chunk_size - local_size` (253952 for these models) be the chunk length used for positions and `w = local_size`
(8192). Standard attention scores a query at position `i` against a key at `j` through RoPE, which depends only on
`i - j`. DCA replaces the pair `(i, j)` by *three* pairs of "virtual" positions and picks one per key (design from An et al.,
*Training-Free Long-Context Scaling of Large Language Models*; the exact assignments below are the paper's description and
would have to be confirmed against vLLM's kernel, see "Why not implemented"):

| key `j` relative to query `i` | query position | key position | virtual distance |
|---|---|---|---|
| same chunk (**intra**), `floor(i/s) = floor(j/s)` | `i mod s` | `j mod s` | `i - j` (the true one, at most `s`) |
| an earlier chunk (**inter**) | a constant (about `s`, the largest distance trained on) | `j mod s` | between 1 and about `s` (it grows with `j mod s`) |
| the previous chunk, within `w` of `i` (**successive**) | `(i mod s) + s` | `j mod s` | exact for the last `w` tokens across the boundary |

A numeric toy (`s = 8`, `w = 2`): query 11 (chunk 1, offset 3) against key 10 is intra, distance 1; against key 7 (chunk 0,
offset 7, 4 tokens away) it is successive, virtual distance `(3 + 8) - 7 = 4`, the true one; against key 2 (chunk 0, offset 2,
9 away) it is inter, virtual distance about `8 - 2 = 6` instead of 9. The three score matrices are merged by **one softmax
over all keys** (not three), because the keys compete for the same weight.

```
                   key chunk 0                 key chunk 1               key chunk 2 (query in here)
query chunk 2:     [ inter: q fixed at ~s ]    [ successive near the boundary, inter further back ]   [ intra ]
```

The keys are cached rotated at `j mod s` (one rotation, so prefill and decode keep their present shape); only the queries are
rotated three times (three rotary tables), and the per-case partial maxima and sums are combined as in flash decoding.

**Equality below the training length.** If the whole sequence fits one chunk (length at most `s = 253952`) every pair is intra
with virtual distance `i - j`: DCA is exactly ordinary attention. So for prompts up to about 250k tokens the plain `qwen2`
path is already the intended model; only longer prompts differ.

### 14.20.2 Why it is not implemented

- **No independent reference.** The `Qwen2` class of transformers ignores the key; the only implementation is vLLM's custom
  attention backend (`dual_chunk_flash_attn`). This project's rule for a new family is to check it against transformers on a
  tiny random model; a port could not be checked that way, and a vLLM install (CUDA-oriented) is not available here.
- **It changes the cache contract.** Keys rotated at `j mod s` and queries rotated three times touch `KvCache`, the Metal
  attention kernels and the saved-cache format of every backend; doing that without a reference would risk regressions in
  every Qwen2 model for a feature only reachable above 250k tokens.
- **The memory is the real limit anyway.** The 7B at 262144 tokens needs `2 * 28 layers * 4 kv heads * 128 * 2 bytes * 262144`
  = 15 GB of f16 KV cache (7.5 GB at q8); at 1M tokens 60 GB. On this machine the practical range is well below the chunk size,
  where DCA is the identity.

### 14.20.3 What an implementation would need (in order)

1. A `dualChunk(chunkLen, local)` option on `Attention.Gqa`, parsed in `Families` from `dual_chunk_attention_config`.
2. Keys stored rotated at `j mod s` (a rope-position array per row instead of `pos`), queries rotated at the three positions by
   three tables in `Rope`.
3. `KvCache` (CPU) and `attention_t` (Metal, through the `ext` buffer of 14.18) gain the three-way selection per key from
   `i` and `j`, with a single combined softmax.
4. A reference: vLLM's backend on a tiny random Qwen2 with `dual_chunk_attention_config` (to pin the exact position
   assignments), plus a long-context test with `s` small enough (say 16) that the tiny model crosses many chunks.
5. The sparse variant shipped with the 1M models (`sparse_attention_config`, vertical-slash patterns) is a separate
   approximation and not needed for correctness.

### 14.20.4 Settings and limits

No setting. Nothing is measured. Run the 1M models with a context of at most 262144 tokens (`backend.ctx`) to stay in the region where plain attention is exact.

## 14.21 Checking the new families against transformers on tiny random models

`scripts/hf-arch-check.sh [family ...]` (families `gptoss gptoss-mxfp4 llama4 llama4-mix granite granite-rope nemotronh jamba`;
needs a Python with torch and transformers: `REF_PYTHON=/path/to/python`; `METAL_EMU=1` runs the Metal side on the CPU
emulator of the kernels) is the reusable form of the checks in 14.18, 14.19 and 14.22. It is also run from
`scripts/arch-smoke.sh` when `REF_PYTHON` is set.

### 14.21.0 🧸 In plain words

To know that a re-implementation of a model is right, you compare it with the original on the same input. Downloading a
20-billion-parameter model for that is slow and heavy. So we build a **miniature** of the same model with the original
library (the same layer types, the same options, 4 layers and 128 numbers wide instead of dozens of layers thousands wide),
fill it with random numbers, ask both programs the same questions, and compare the answers digit by digit. Random weights
are good at finding mistakes: a wrong mask, a missing factor or a swapped half of a vector changes the answer immediately.

### 14.21.1 📐 How it works

1. `scripts/hf-arch-ref.py <family> <dir>` builds the tiny model through transformers (config values of the real family: see the
   head of the script), initializes every weight randomly with a seeded generator (so that router logits, sinks, biases and
   the state-space parameters are non-trivial), **rounds every matrix to the Q8_0 grid** (the engine's q8 format; biases,
   router weights, `dt_proj` and `A_log` stay exact, because the engine keeps those in f32), saves it as a Hugging Face
   checkpoint (`save_pretrained`, so the tensor names are the ones transformers writes), and runs the model on a prompt of 40
   ids plus 12 greedy continuation ids, writing the logits of all 52 positions (`ref/logits.f32`).
2. `ArchRefCheck <dir> <metal-lib|-> q8` (`src/test/java/.../ArchRefCheck.java`) converts the checkpoint with the engine's own
   converter, then on the CPU backend and the Metal one (plus Metal with the f16 cache) reports: the largest
   `|logit - reference|` over all positions and relative to the largest logit, argmax disagreements, whether free-running
   greedy decoding from the prompt gives the reference's 12 ids, and the difference between the whole prompt in one call,
   in chunks of 7 and one token at a time (which exercises windows, chunks, sinks and recurrent states across chunk
   boundaries). It exits 1 on any failure; the tolerances are 5e-5 (CPU), 1e-4 (Metal), 1e-2 (Metal f16 cache) of the
   largest logit; the engine's results sit two orders of magnitude below them (14.18.4, 14.19.4, 14.22.6).

A pitfall worth recording: the reference must round to Q8_0 the way the engine does, half away from zero
(`floor(x + 0.5)`); `torch.round` rounds half to even, which differs on exact ties and made the first comparisons look
like 1e-2 errors. Another: biases must not be rounded, the engine keeps them in f32.

### 14.21.2 Why this design, settings, limits

- **Tiny models instead of real weights** (also the project's rule: no model downloads for architecture work): every code
  path of the family is exercised, a run takes seconds, and any machine can repeat it. What it cannot show is behaviour tied
  to real sizes: 128-token windows against 40-token prompts (the tiny models use a window of 4, a chunk of 8), long contexts,
  and quality.
- Q8_0 rounding in the reference instead of comparing at q4: the engine's q8 and the reference then differ only by float
  rounding, so a 1e-5 tolerance is meaningful; q4 would differ by quantization error.
- Settings: the script has none besides `REF_PYTHON`, `REF_WORK` (default `/tmp/jllm-hf`) and `METAL_EMU`. Results are
  printed per check; any `FAILED` fails the script.

## 14.22 Attention / state-space hybrids: Granite 4, Nemotron-H, Jamba

Code: `model/HybridArch` (the layer list and the Mamba-1/2 dimensions), `model/HybridParsers` (the three config parsers),
`model/HybridLayer` (the weights), `model/ModelConverter.planHybrid`, `cpu/HybridCpuBackend`, `metal/HybridMetalBackend`,
`kernels.metal` section 15 (`m2_conv`, `m2_scan`, `m2_gnorm`, `rms3`, `relu2`, `vec_axpy`, `vec_zero`) plus the Mamba-1
kernels of [chapter 15](15-mamba.md), `model/Mlp.Activation.RELU2`, `tokenizer/ChatTemplate.GRANITE`,
`scripts/hf-arch-ref.py` (`granite`, `granite-rope`, `nemotronh`, `jamba`)

Three open families interleave **state-space layers** (Mamba) with attention layers and an MLP or MoE in one decoder stack:
**IBM Granite 4.0** (`granitemoehybrid`), **NVIDIA Nemotron-H** (`nemotron_h`) and **AI21 Jamba** (`jamba`). They reuse the
Mamba machinery of chapter 15 and the hybrid state handling of Qwen3-Next (14.14.5): the recurrent state lives in the
backend, a call at `pos == 0` resets it, and a call at any other position than the one the state holds is refused.

| | mixers | feed-forward | positions | what is special |
|---|---|---|---|---|
| Granite 4.0-H (`-tiny`: 40 layers, hidden 1536, 36 Mamba-2 + 4 attention) | `layer_types`: `mamba` (Mamba-2: 48 heads of 64, 1 group, state 128) / `attention` (12 heads over 4 KV) | shared SwiGLU MLP (1024) + MoE of 64 experts, 6 per token (512 wide), in every layer | `nope` | muP multipliers: embedding 12, attention 0.0078125, residual 0.22, logits / 6 |
| Nemotron-H 8B (52 layers, hidden 4096) | pattern string `M-M-M-M*-...`: `M` Mamba-2 (128 heads of 64, 8 groups, state 128), `*` attention (32 heads over 8), `-` ReLU-squared MLP (21504) | only in `-` layers | none | one norm per layer; `time_step_min` 0.001 clamps `dt` |
| Jamba v0.1 (32 layers, hidden 4096) | Mamba-1 (state 16, dt rank 256) in 7 of 8 layers, attention in the 5th (32 heads over 8) | every second layer: 16 experts top 2 (14336 wide); SwiGLU MLP in the others | none | RMS norms on `dt`, `B`, `C` inside the Mamba layer |

### 14.22.0 🧸 In plain words

Chapter 15's Mamba reads text like a person with a **notepad of fixed size**: for every new word the old lines on the
pad fade a little and a new note is added, and the layer's answer is read off the pad. However long the text, the pad stays the
same size. Attention is the opposite habit: re-read the whole text for every word, with a diary (the KV cache) that grows
with the text. These models **alternate the two**: mostly cheap notepad layers, and now and then a full re-read, so the
growing cache is needed in only 1 layer out of 4 to 10. Granite-4.0-H-Tiny keeps a diary in 4 of its 40 layers; Jamba in 1 of 8.

Mamba-2 (Granite, Nemotron-H) is Mamba with a **simpler notepad**: one fade rate per head instead of one per channel, which lets
many channels share work and is faster on a GPU. Jamba uses the older Mamba-1 notepad and adds a safety measure: the three
signals that steer the pad (how fast to forget, what to write, what to read) are renormalised before use.

### 14.22.1 📐 One layer and the three stacks

```
every layer:   x ---+--> norm1 -> MIXER ---(* r)--->(+)---+--> norm2 -> FFN ---(* r)--->(+)--> x'
                    |                                 ^   |                              ^
                    +---------------------------------+   +------------------------------+
MIXER  = attention (KV cache)  |  Mamba-2 (conv state + SSM state)  |  Mamba-1 (conv state + SSM state)  |  none
FFN    = SwiGLU MLP | ReLU^2 MLP | MoE (+ shared MLP)  | none         r = residual multiplier (1 unless Granite)

Granite-4.0-H-Tiny:  M M M M M A M M M M ... (36 M, 4 A), FFN = MoE + shared MLP in every layer
Nemotron-H:          M - M - M - M * - M - ...   one letter = one layer, each layer has ONE mixer or ONE ReLU^2 MLP
Jamba:               [M F][M F-MoE][M F][M F-MoE][A F][M F-MoE][M F][M F-MoE] repeated 4 times
```

(`F` = feed-forward; in Nemotron-H a `-` layer is *only* the MLP with its single norm, `x + mlp(norm x)`.)

### 14.22.2 📐 Mamba-2 step by step

Per layer: `in_proj` makes one row `[ z | x B C | dt ]` with `z, x in R^(d_inner)`, `B, C in R^(groups * N)`,
`dt in R^(heads)`; `d_inner = heads * head_dim`, `N` = state size.

1. **Causal depthwise convolution** of width `k` (4) with bias over the channels `u = [x B C]` (`m2_conv`): for channel `c`,
   `v_t = b_c + sum_{i<k} w_{c,i} u_{t-k+1+i}`, using the last `k - 1` inputs saved as the **conv state**, then
   `silu(v)`. Worked example, `w = [0.1, 0.2, 0.3, 0.4]`, state `[1, 2, 3]`, new input `4`, bias `0.5`:
   `v = 0.5 + 0.1*1 + 0.2*2 + 0.3*3 + 0.4*4 = 3.5`, `silu(3.5) = 3.397`; the state becomes `[2, 3, 4]`.
2. **Selective state update** per head `h` (group `g = h / (heads / groups)`), `m2_scan`, one GPU thread per `(h, p)` row of the
   state with its `N` values:

   ```
   dt_h = clamp( softplus(dt_raw_h + dt_bias_h), dt_min, dt_max )         softplus(a) = ln(1 + e^a)
   A_h  = -exp(A_log_h)                                                    (a negative scalar per head)
   S_h  <- exp(dt_h * A_h) * S_h  +  dt_h * x_h (outer) B_g               S_h in R^(head_dim x N)
   y_h  =  S_h C_g  +  D_h * x_h
   ```
3. **Gate and group norm** (`m2_gnorm`): `y <- y * silu(z)`, then an RMS norm over each of `norm_groups` groups of channels with
   the learned weight `norm.weight`.
4. `out_proj` back to the hidden size.

Worked numbers for one head, `head_dim = 2`, `N = 2`, one group: `dt_raw = 0.3`, `dt_bias = -0.5`, `A_log = ln 2` (so `A = -2`),
`D = 1.5`, state `S = [[1, 0.5], [0, 2]]`, `x = [1, -2]`, `B = [0.5, 1]`, `C = [1, -1]`:
`dt = softplus(-0.2) = 0.5981`, decay `exp(0.5981 * -2) = 0.3023`;
`S' = 0.3023 * S + 0.5981 * x (outer) B = [[0.6014, 0.7493], [-0.5981, -0.5916]]`;
`y = S' C + D x = [0.6014 - 0.7493 + 1.5, -0.5981 + 0.5916 - 3.0] = [1.352, -3.006]`. With `z = [0.8, -1.2]`
(`silu = [0.552, -0.278]`) the gated vector is `[0.746, 0.835]`, whose RMS is `sqrt(0.627)`, giving the normed `[0.942, 1.054]`
(weights 1).

The recurrence is run token by token inside a chunk (`HybridCpuBackend` loops; `m2_scan` loops in each thread); the
"chunked SSD" matrix form of the Mamba-2 paper would speed up long prefills but gives the same numbers, so it was left out.
Data layout of the state, per layer: conv state `[conv_dim][k - 1]` floats, SSM state `[heads * head_dim][N]` floats.

### 14.22.3 📐 Mamba-1 with Jamba's norms, ReLU-squared, Granite's multipliers

- **Jamba's Mamba-1** is chapter 15's layer with one addition: `x_proj` produces `[dt_low | B | C]` (`dt_rank`, `N`, `N`
  values) and each part is RMS-normalised with its own weight (`dt_layernorm`, `b_layernorm`, `c_layernorm`, epsilon
  `rms_norm_eps`) before `dt_proj` and the scan. `rms3` does the three parts in one dispatch. Example for a 4-wide part
  `[3, -1, 2, 0]` with unit weight: mean square `3.5`, result `[1.604, -0.535, 1.069, 0]`.
- **ReLU-squared MLP** (Nemotron-H `-` layers): `down(relu(up(x))^2)`, no gate matrix (`Mlp.Activation.RELU2`, `gated() = false`;
  kernel `relu2`).
- **Granite's muP multipliers:** `embedding_multiplier` scales the embedding output, `attention_multiplier` **replaces**
  `1/sqrt(head_dim)` as the attention scale, `residual_multiplier` `r` scales each branch before it is added
  (`x <- x + r * mixer(norm x)`; `x <- x + r * (moe(norm x) + shared(norm x))`), and `logits_scaling` divides the final logits. On
  Metal the multiplier is applied with `vec_axpy` (`x += r * y`) unless `r = 1`, in which case the output projection accumulates
  into `x` directly. Granite's attention layers use no rotary table (`position_embedding_type: nope`, the identity table of 14.19).
- **Granite's MoE:** `block_sparse_moe.input_linear` is `[E, 2 * m, d]` (fused gate/up per expert), `output_linear` `[E, d, m]`,
  router `softmax(top-k(logits))` (`Router.TopLogits`, 14.18) with no bias; the shared MLP (`shared_mlp.input_linear` fused
  gate/up, `output_linear`) is added to it. Jamba's router is a plain `softmax` over all experts then top-k, **not** renormalised
  (`Router.SOFTMAX`).

### 14.22.4 Why this design

- **A separate `HybridArch` instead of widening `Decoder`:** the layer list has three mixers and per-layer residual multipliers;
  `Decoder`'s layer record assumes one attention shape. Rejected: encoding a Mamba layer as an "attention with nulls", which
  would have leaked mixer checks into every attention kernel call site.
- **Reuse over new kernels:** Mamba-1's conv/scan kernels, the MoE and SwiGLU kernels and the generic attention kernel are reused;
  new kernels are only the Mamba-2 ones (the scan is per head, not per channel) and tiny element-wise ones.
- **Token loop for the scan** instead of the chunked matrix form: exactly the arithmetic of the reference recurrence, trivially
  correct for any chunk size, and decode (one token) is the common case; the cost is slower long-prompt prefill.
- **Canonical tensor names in the converter** (`model.layers.N.mixer.*`): Granite, Nemotron-H and Jamba checkpoints use three
  different spellings (`mamba.`, `mixer.`, `feed_forward.`, `block_sparse_moe.`); `planHybrid` maps them all, including per-expert
  and fused expert tensors.

### 14.22.5 Settings

No new setting. `backend.kv-dtype` f32 or f16 (q8 and q4 are served as f16 on Metal for these, a note is printed; the CPU
backend takes the cache types of `KvCache`). The recurrent state cannot be rewound, so prefix reuse is off (`supportsPrefixReuse() = false`),
exactly as for Mamba and Qwen3-Next without an MTP draft.

### 14.22.6 Measured and checked

Machine: Apple M1 Pro, 32 GB, 2026-10-22; transformers 5.20, tiny random models with the real families' structure, Q8_0-rounded
reference weights, 52 logit rows. Command: `scripts/hf-arch-check.sh granite granite-rope nemotronh jamba`. Max abs logit
difference relative to the largest logit:

| model (tiny) | CPU | Metal | greedy ids | argmax at 52 positions |
|---|---|---|---|---|
| `granite`: 5 layers M M A M A, Mamba-2 8 heads of 32, 2 groups, state 16, MoE 4 experts top 2 + shared MLP, muP multipliers, tied head, no RoPE | 1.9e-6 | 1.4e-6 | 12 of 12 equal | equal |
| `granite-rope`: dense Granite whose attention layers rotate | 9.0e-7 | 9.5e-7 | equal | equal |
| `nemotronh`: pattern `M-M*-M`, 2 groups, ReLU-squared MLP, `time_step_min` 0.05 | 2.0e-6 | 3.7e-6 | equal | equal |
| `jamba`: Mamba-1 with dt/B/C norms in 3 of 4 layers, attention in layer 2, MoE every other layer | 7.7e-6 | 6.9e-6 | equal | equal |

Chunking (one call, chunks of 7, token by token) agrees to ≤ 4e-6 on the CPU and ≤ 3e-5 on Metal with an f16 cache. In the arch
smoke the three tiny models (`GraniteHybridForCausalLM`, `NemotronHForCausalLM`, `JambaForCausalLM`) pass the Mamba-style state
checks (chunk invariance 0 to 1.8e-5, wrong-position refusal, `supportsPrefixReuse` false) and CPU == Metal on the final logits
(≤ 3.6e-5 of a logit range of about 20) on both backends. `kernel_tests.py` checks `m2_conv` + `m2_scan` + `m2_gnorm` against a
float64 numpy Mamba-2 over two chunks (so the carried states are exercised; three shapes including a clamped `dt` and
`groups != 1`, relative error ≤ 3.2e-7), `rms3`, `relu2`, `vec_axpy`, `vec_zero`. The Granite chat format equals its Jinja template
on 18 conversations. No speed is claimed: the real hybrids (7B and up) were not downloaded.

### 14.22.7 Not supported

- **Nemotron-H's MoE layers** (`E` in the pattern; the 8B is dense): rejected with an explicit error.
- State save/restore, prefix reuse, rollback, replay and continuous batching for the hybrids (the same limits as Mamba,
  chapter 15): every request starts from position 0.
- The Mamba-2 chunked SSD matrix form (the token loop is used for prefill), q8 / q4 KV (f32 or f16 only on Metal),
  `backend.metal.cpu-moe`, speculative decoding with a hybrid as the target model.
- Chat formats for Nemotron-H and Jamba (no `ChatTemplate` entry: format the prompt yourself or use `/v1/completions`).
  Granite has one (`ChatTemplate.GRANITE`: `<|start_of_role|>role<|end_of_role|>...<|end_of_text|>\n`, the default system
  message when none is given, no tools or documents).
- GGUF files of `granitehybrid`, `nemotron_h`, `jamba` (llama.cpp's names).
- Real-weight validation (Granite-4.0-H-Tiny is 7B in total: a download this task did not do; its config, chat template and
  tokenizer were read).
