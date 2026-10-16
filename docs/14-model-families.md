# 14. Model families (Llama, Mistral, Qwen2, Phi-3/4, Gemma 2/3, DeepSeek-R1 distills, DeepSeek-V2/V3)

Code: `model/Architecture` (`Decoder`, `MambaArch`), `model/Families` (one parser per `model_type`),
`model/LayerSpec`, `model/Attention`, `model/Mlp`, `model/Router`, `model/RopeTable`, `model/Scaling`, `model/Rope`,
`model/ModelConverter`, `cpu/CpuBackend`, `cpu/MlaCpu`, `metal/MetalBackend`, `metal/MlaMetal`, `kernels.metal`,
`Main.eosIds`, `tokenizer/Tokenizer.promptBos`, `engine/Engine.preparePrompt`

All these models are the same decoder-only transformer that [chapter 5](05-transformer-math.md) describes:
embedding, N × (RMSNorm → attention → RMSNorm → SwiGLU MLP), final RMSNorm, LM head. They differ in a handful of
switches, all read from `config.json` by the family's parser in `Families` and stored as typed values in an
`Architecture` (section 14.12). Qwen3 is the reference; this chapter lists what changes. Section map: 14.1 the
families at a glance, 14.2 – 14.6 what the Llama-like families have in common (and what they lack), 14.7 how it is
tested, 14.8 GLM, 14.9 Phi-3/4, 14.10 Gemma, 14.11 DeepSeek-V2/V3, 14.12 the typed model and how to add a family.
Qwen3-MoE is [chapter 12](12-mixture-of-experts.md), Mamba [chapter 15](15-mamba.md).

## 14.1 Supported families

One row per `model_type`; the parser that produces each row is a short method in `Families` (`llama`, `mistral`,
`qwen2`, `qwen3`, `glm`, `phi3`, `gemma`, `deepseek`, `mamba`) over a shared `Draft` whose defaults are Llama's. "Config"
means the value comes from `config.json`; defaults are those of the family's transformers config class.

| `model_type` | Architecture class | q/k/v bias | QK-norm | Sliding window | RoPE (pairing; scaling) | Norms | MLP | BOS (from the tokenizer) | Tied head | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `llama` | `LlamaForCausalLM` | no | no | none | `HALF`; `rope_scaling` `llama3` (3.1+), `linear` | `PRE` | dense SiLU | yes | config (default false) | |
| `mistral` | `MistralForCausalLM` | no | no | `sliding_window` on every layer (null in v0.3) | `HALF`; none | `PRE` | dense SiLU | yes | config (false) | |
| `qwen2` | `Qwen2ForCausalLM` | **always** (the key is absent) | no | layers ≥ `max_window_layers`, only if `use_sliding_window` | `HALF`; none | `PRE` | dense SiLU | no (R1-Distill-Qwen: yes) | config | |
| `qwen3` | `Qwen3ForCausalLM` | no | **yes** | same rule as `qwen2` | `HALF`; none | `PRE` | dense SiLU | no | config | |
| `qwen3_moe` | `Qwen3MoeForCausalLM` | no | yes | same rule as `qwen2` | `HALF`; none | `PRE` | MoE, `Router.Softmax`, `norm_topk_prob` from config; dense in `mlp_only_layers` and off the `decoder_sparse_step` grid | no | config | chapter 12 |
| `glm` | `GlmForCausalLM` | `attention_bias`, **default true** | no | none | `INTERLEAVED`, `partial_rotary_factor` default 0.5 | `PRE` | dense SiLU, fused `gate_up_proj` | no (`[gMASK]<sop>`) | config | eps default 1.5625e-7; 14.8 |
| `glm4` | `Glm4ForCausalLM` | as `glm` | no | none | as `glm` | `SANDWICH_GLM4` (4 norms) | dense SiLU, fused `gate_up_proj` | no | config | 14.8 |
| `phi3` | `Phi3ForCausalLM` | no | no | `sliding_window` on every layer | `HALF`; `longrope` (short/long factors), `partial_rotary_factor` | `PRE` | dense SiLU; fused `qkv_proj`, `gate_up_proj` | no | config | eps default 1e-5; 14.9 |
| `gemma2` | `Gemma2ForCausalLM` | no | no | layers `i % 2 == 0` are local (`sliding_window`), the rest global | `HALF`; one table, theta default 1e4 | `SANDWICH_GEMMA`, gains `1 + w` | dense **GeLU-tanh** | yes | **default true** | embedding × `bf16(√dim)`, attention soft-cap 50, final soft-cap 30, `query_pre_attn_scalar`; 14.10 |
| `gemma3_text` (also inside `gemma3`) | `Gemma3ForCausalLM` (wrapper `Gemma3ForConditionalGeneration`) | no | **yes** | 5 local + 1 global (`sliding_window_pattern` 6, or `layer_types`) | `HALF`; global theta default 1e6 + `rope_scaling`, local table `rope_local_base_freq` (1e4) without scaling | `SANDWICH_GEMMA`, gains `1 + w` | dense GeLU-tanh | yes | default true | soft-caps off unless set; text part only; 14.10 |
| `deepseek_v2` | `DeepseekV2ForCausalLM` | rejected | no | none | MLA rotary slice `INTERLEAVED`, `yarn` | `PRE` | dense for the first `first_k_dense_replace` layers, then MoE + shared experts; `Router.DeepSeek` softmax, `greedy` / `group_limited_greedy` | yes | config (false) | MLA latent KV cache; 14.11 |
| `deepseek_v3` | `DeepseekV3ForCausalLM` | rejected | no | none | as V2 | `PRE` | as V2; sigmoid, `noaux_tc` (8 groups keep 4), scaling 2.5, `norm_topk_prob` default true | tokenizer rule | config (false) | FP8 checkpoints refused (14.11.7) |
| `mamba` | `MambaForCausalLM` | n/a | n/a | n/a (no attention) | none (no positions) | one RMSNorm per block | no MLP: conv + selective scan mixer | no | default true | `MambaArch`; chapter 15 |

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
| `deepseek_v3` | DeepSeek-V3 / R1 (**not loadable**, 14.11.7) | config and header only; tiny random models |
| `mamba` | `state-spaces/mamba-130m-hf` … `mamba-2.8b-hf` | chapter 15 |

A different `model_type`, a `hidden_act` other than `silu` (Gemma reads `hidden_activation`, which must be
`gelu_pytorch_tanh`), `mlp_bias: true`, `attention_bias: true` outside `qwen2` and `glm` (it would also add an `o_proj`
bias), or a `rope_scaling` type other than `default`/`linear`/`llama3`/`longrope` (`su` is the legacy spelling of
`longrope`) is rejected at load time with a message naming the field; `yarn` is accepted for the DeepSeek families only.

Other requirements: `hidden_size`, `intermediate_size` and `num_heads × head_dim` must be multiples of 32 (the
quantization block), and the Metal backend additionally needs `head_dim` to be a multiple of 32 and at most 256
(Llama 3, Mistral, Qwen2.5 and DeepSeek-Coder all have 64 or 128; DeepSeek's MLA has its own limits, 14.11.1).

## 14.2 What differs from Qwen3

| Feature | Llama 3.x | Mistral | Qwen2 / 2.5 | Qwen3 |
|---|---|---|---|---|
| q/k/v bias (`attention_bias`) | no | no | **yes** (always, even though the key is absent) | no |
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
  every other family rejects `attention_bias: true`, because transformers would then also add an `o_proj` bias, which
  is not implemented. The converter stores `self_attn.{q,k,v}_proj.bias` as f32 (they are tiny); the backends add
  them right after the q/k/v matmuls and *before* RoPE. CPU: `addBias`, a loop over the freshly written rows; Metal:
  the bias is an epilogue argument of the fused `matmul3` kernel that projects q, k and v, so it costs no dispatch.
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
  `state-spaces/mamba-*-hf`) is supported, see [15](15-mamba.md); Mamba-2 and hybrids are not.
- `yarn` outside the DeepSeek families (14.11.2), `dynamic` RoPE scaling; FP8 checkpoints (14.11.7); `hidden_act`
  other than `silu` (Gemma's `gelu_pytorch_tanh` via `hidden_activation` is the one exception); MLP bias; an `o_proj`
  bias; Llama 4; a `head_dim` that is not a multiple of 32 (Mistral Nemo-style) on Metal (CPU works).
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
| Llama / Mistral / Qwen2 | plain Llama (GQA, untied head); Llama 3 `rope_scaling` (MHA, tied); Mistral with `sliding_window = 4`; Qwen2 with bias and a window on layers ≥ 2 |
| GLM (14.8) | `glm` (bias, `head_dim` 32, factor 0.5); `glmedge` (`head_dim` 64, factor 1.0, no bias, tied); `glm4` (four norms) |
| Phi-3 (14.9) | fused weights + longrope long factors + partial rotary + window; the same with short factors and tied embeddings; `head_dim` 64 MHA with plain RoPE |
| Gemma (14.10) | Gemma 2 (`head_dim` 64, cap 1.0 so that it matters on random weights, final cap 30, local layers 0 and 2 with window 6); Gemma 3 text (q/k norm, 5 local + 1 global, linear ×8 global and 10 000-base local RoPE); the multimodal wrapper layout; `head_dim` 256 variants of both |
| DeepSeek (14.11) | `deepseek_v2` (`q_lora_rank` null, group-limited routing, 2 shared experts), `deepseek_v2q` (q LoRA, `norm_topk_prob`, scaling 2.5), `deepseek_v3` (sigmoid, `noaux_tc`) |
| Mamba (chapter 15) | `mamba` (d_model 64, tied head) and `mamba3` (d_model 128, state 8, conv 3, untied head), plus chunking-invariance and state checks |

`mvn test` adds the per-family unit checks that have a hand-computed or external reference: `SmokeTest.ropeScaling`
(llama3, 14.4), `glmRope` (14.8), `longRope` and `phi3Configs` (14.9), `GemmaChecks` (14.10), `yarnRope`,
`deepseekRouting`, `deepseekConfigs` (14.11), `modelConfigs`, and `ConverterTest` (conversion, resume, the tiny Qwen3-MoE
checkpoint of chapter 12). The DeepSeek models are in addition compared with an independent float64 reference
(`scripts/deepseek-ref.py`, `DeepSeekRef`, 14.11.8). Family-specific verification is described in each family's own
"Testing" subsection.

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
configs above were read), no real weights were run; `yarn` and `dynamic` RoPE scaling; Gemma 3n and
Gemma 1 (`model_type: gemma`, no extra norms) are rejected.

## 14.11 DeepSeek-V2 / V3 (Multi-head Latent Attention + shared experts)

Code: `Families.deepseek`, `model/Attention.Mla`, `model/Router.DeepSeek` + `model/DeepSeekRouter`, `model/AttnWeights.Mla`
(+ `kvBKeys` / `kvBValues`), `model/Scaling.Yarn` + `Rope.yarnInvFreq`, `model/ModelConverter.mlaRules` / `moeRules`,
`cpu/MlaCpu`, `metal/MlaMetal`, kernels `mla_lat_store`, `mla_absorb_q`, `mla_apply_v`, the `attention_mla` instantiation
(section 7 of `kernels.metal`), `rope_il` and `router_ds` (section 5), `tokenizer/ChatTemplate` (`deepseek-v2`).

| `model_type` | Architecture class | Examples | Weights |
|---|---|---|---|
| `deepseek_v2` | `DeepseekV2ForCausalLM` | `deepseek-ai/DeepSeek-V2-Lite` (15.7 B, 27 layers), `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | bf16, supported |
| `deepseek_v3` | `DeepseekV3ForCausalLM` | `deepseek-ai/DeepSeek-V3`, `DeepSeek-R1` (671 B) | **FP8, not supported** (14.11.7) |

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
  f16**. Caching the expanded per-head keys and values instead would take `(16·192 + 16·128) · 4 B` = 20 KB per layer
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

### 14.11.7 FP8 checkpoints (V3, R1) are not supported

The published V3/R1 weights are block-quantized FP8: every large matrix is `F8_E4M3` (`float8_e4m3fn`) with a companion
`<name>.weight_scale_inv` tensor of dtype F32 holding one inverse scale per 128 × 128 block (e.g. `q_a_proj.weight`
`[1536, 7168]` has `weight_scale_inv` `[12, 56]`); `config.json` carries `quantization_config: {quant_method: fp8,
weight_block_size: [128, 128]}`. `SafeTensors` has no FP8 dtype and the converter does not dequantize
(`w = fp8_value · scale_inv[block]`). The converter refuses such a checkpoint up front with an explanatory error (any
index containing `weight_scale_inv`). Supporting V3 needs an FP8 dequantization path in `ModelConverter` plus bf16
re-uploads (e.g. a community bf16 conversion works as is), and a machine with more than 400 GB of memory to run it.

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
the V2 chat template and the FP8 refusal.

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
defaults are those of the safetensors load. Four architectures are mapped today (`llama`, `qwen2`, `qwen3`,
`qwen3moe`); adding one means its metadata keys in `GgufArchitecture`, its tensor names in `GgufWeights.resolve` and,
if llama.cpp's converter permutes or folds something for it (Llama's q/k rows, Gemma's `1 + w` norms), the matching
switch in the `Architecture`. See [16](16-gguf.md).
