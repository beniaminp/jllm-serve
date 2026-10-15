# 14. Model families (Llama, Mistral, Qwen2, Phi-3/4, Gemma 2/3, DeepSeek-R1 distills, DeepSeek-V2/V3)

Code: `model/ModelConfig`, `model/Rope`, `model/ModelConverter` (biases, fused Phi-3 tensors), `cpu/CpuBackend.attention/addBias`,
`kernels.metal` (`attention`, `add_bias`), `Main.eosIds`, `engine/Engine.generate` (BOS)

All these models are the same decoder-only transformer that [chapter 5](05-transformer-math.md) describes:
embedding, N × (RMSNorm → attention → RMSNorm → SwiGLU MLP), final RMSNorm, LM head. They differ in a handful of
switches, all read from `config.json` into `ModelConfig`. Qwen3 is the reference; this chapter lists what changes.

## 14.1 Supported families

| `model_type` | Architecture class | Examples | Checked against |
|---|---|---|---|
| `llama` | `LlamaForCausalLM` | Llama 3 / 3.1 / 3.2, DeepSeek-R1-Distill-Llama-8B, DeepSeek-Coder-1.3B | `unsloth/Llama-3.2-1B-Instruct`, `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| `mistral` | `MistralForCausalLM` | Mistral-7B-Instruct v0.1 – v0.3 | `unsloth/mistral-7b-instruct-v0.3` |
| `qwen2` | `Qwen2ForCausalLM` | Qwen2 / Qwen2.5, DeepSeek-R1-Distill-Qwen | `Qwen/Qwen2.5-1.5B-Instruct`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| `qwen3` | `Qwen3ForCausalLM` | Qwen3 dense | `Qwen/Qwen3-0.6B` |
| `qwen3_moe` | `Qwen3MoeForCausalLM` | Qwen3-30B-A3B (chapter 12) | |
| `glm` | `GlmForCausalLM` | GLM-Edge-1.5B/4B-Chat, glm-4-9b-chat-hf (section 14.8) | `THUDM/GLM-Edge-1.5B-Chat`, `THUDM/glm-4-9b-chat-hf` |
| `glm4` | `Glm4ForCausalLM` | GLM-4-9B/32B-0414, GLM-Z1 (section 14.8) | `THUDM/GLM-4-9B-0414`, `THUDM/GLM-Z1-9B-0414` |
| `phi3` | `Phi3ForCausalLM` | Phi-3-mini, Phi-3.5-mini, Phi-4-mini, phi-4 (14B), see 14.9 | `microsoft/Phi-3-mini-4k-instruct`, `Phi-3.5-mini-instruct`, `Phi-4-mini-instruct`, `phi-4` |
| `gemma2` | `Gemma2ForCausalLM` | Gemma 2 2B / 9B / 27B (section 14.10) | `unsloth/gemma-2-2b-it` (mirror of the gated `google/gemma-2-2b-it`) |
| `gemma3_text` (also inside `gemma3`) | `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration` (text part) | Gemma 3 1B / 4B / 12B / 27B (section 14.10) | `unsloth/gemma-3-1b-it`, `unsloth/gemma-3-4b-it` |
| `deepseek_v2` | `DeepseekV2ForCausalLM` | DeepSeek-V2-Lite, DeepSeek-Coder-V2-Lite-Instruct (MLA, section 14.11) | config, safetensors header and `modeling_deepseek.py` on the Hub; tiny random models |
| `deepseek_v3` | `DeepseekV3ForCausalLM` | DeepSeek-V3 / R1 (FP8 weights: **not loadable**, section 14.11.7) | config and header only; tiny random models |

A different `model_type`, a `hidden_act` other than `silu` (Gemma reads `hidden_activation`, which must be
`gelu_pytorch_tanh`), `mlp_bias: true`, or a `rope_scaling` type other than
`default`/`linear`/`llama3`/`longrope` is rejected at load time with a message naming the field; `yarn` is accepted for the DeepSeek families only.

Other requirements: `hidden_size`, `intermediate_size` and `num_heads × head_dim` must be multiples of 32 (the
quantization block), and the Metal backend additionally needs `head_dim` to be a multiple of 32 and at most 256
(Llama 3, Mistral, Qwen2.5 and DeepSeek-Coder all have 64 or 128).

## 14.2 What differs from Qwen3

| Feature | Llama 3.x | Mistral | Qwen2 / 2.5 | Qwen3 |
|---|---|---|---|---|
| q/k/v bias (`attention_bias`) | no | no | **yes** (always, even though the key is absent) | no |
| per-head q/k RMSNorm | no | no | no | **yes** |
| sliding window | no | `sliding_window` (null in v0.3), all layers | only if `use_sliding_window`, layers ≥ `max_window_layers` | no |
| RoPE scaling | `llama3` (3.1+) | none | none | none |
| `head_dim` | `hidden/heads`, or explicit (3.2: 64) | explicit in v0.3 (128) | `hidden/heads` | explicit (128) |
| `tie_word_embeddings` | true for 1B/3B, false for 8B+ | false | true for ≤ 3B | per size |
| BOS in front of the prompt | **yes** (`<\|begin_of_text\|>`) | **yes** (`<s>`) | no | no |
| chat template | `<\|start_header_id\|>` style | `[INST]` style | ChatML | ChatML |
| stop tokens | `<\|eot_id\|>`, `<\|end_of_text\|>`, `<\|eom_id\|>` | `</s>` | `<\|im_end\|>`, `<\|endoftext\|>` | same as Qwen2 |

How each is implemented:

- **Bias.** `ModelConfig.attentionBias` is true for `qwen2`. The converter stores `self_attn.{q,k,v}_proj.bias` as
  f32 (they are tiny), the backends add them right after the q/k/v matmuls and *before* RoPE. CPU: a loop over
  the freshly written rows; Metal: an `add_bias` kernel, one thread per element (`v[i] += b[i % width]`).
  The same per-layer arrays work for `attention_bias: true` in `qwen2` only; for other families that flag would
  also add an `o_proj` bias, which is not implemented and is rejected.
- **QK-norm.** Present only if the checkpoint has `q_norm`/`k_norm` (Qwen3). `ModelConfig.qkNorm` is true for
  `qwen3*` and `ModelWeights` checks that the tensors exist exactly when it is set.
- **Tied embeddings.** `tie_word_embeddings: true` means there is no `lm_head.weight`; the LM head reuses
  `model.embed_tokens.weight` (quantized to Q8_0 once, shared by both uses).
- **BOS.** `Engine.generate` prepends `bos_token_id` for `llama` and `mistral` unless the prompt already starts
  with it (a chat template that emits the BOS token itself is therefore not doubled). Qwen models never get one.
- **Stop tokens.** The union of `eos_token_id` from `config.json` (an int or a list), `generation_config.json`,
  and the family end-of-turn tokens that exist in the vocabulary.

## 14.3 Sliding-window attention

Full attention lets token `p` read every key at positions `0 … p`. With a window `W`, it reads only the last `W`:

```
position q:      0 1 2 3 4 5 6 7 8 9
token p = 9, W = 4:            . . . . . . x x x x      attends to q in (p - W, p] = 6..9
```

In the code the score loop for token `p` (`len = p + 1` keys) starts at `first = max(0, len - W)` instead of 0; the
softmax and the weighted sum use the same range, so masked positions contribute exactly nothing (they are not
set to -inf and then softmaxed, they are never visited). On Metal the window is the ninth `AttnParams` field and
every loop of the `attention` kernel starts at `first + simdgroup`.

The KV cache still holds the full context (`--ctx` positions); only the reads shrink. A ring buffer of `W`
entries per layer would also cut memory, but needs position-dependent indexing and is a later optimization.
Stacking `L` windowed layers lets information travel `L × W` tokens, which is why a 4096 window still sees far
more than 4096 tokens back in a 32-layer model.

`ModelConfig.windowFor(layer)`: Mistral `sliding_window` applies to all layers; Qwen2 applies it to layers
`l ≥ max_window_layers` only when `use_sliding_window` is true (Qwen2.5 checkpoints set it to false, so the
`sliding_window: 32768` in their config is inert).

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

`Rope` computes the whole `[ctx × d/2]` cos/sin table once from `f'`, so neither backend knows about scaling.
`linear` scaling (`f' = f / factor`) is also accepted. `config.json` may spell the key `rope_type` or the older
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
- `yarn` outside the DeepSeek families (14.11.2), `dynamic` RoPE scaling; FP8 checkpoints (14.11.7); `hidden_act` other than `silu` (e.g. Gemma); MLP bias; an
  `o_proj` bias; Llama 4, Mistral Nemo-style `head_dim` that is not a multiple of 32 on Metal (CPU works).
- Tokenizers: Llama 3 uses a tiktoken-style byte-level BPE (close to Qwen's); Mistral and Llama 2 use SentencePiece
  vocabularies with byte fallback. That is the tokenizer's job (chapter 4), not the model's.
- Hub test checkpoints: `hf-internal-testing/tiny-random-LlamaForCausalLM` has `hidden_size = 16`, which is not a
  multiple of the 32-wide quantization block; the Mistral one ships only `pytorch_model.bin` with
  `hidden_act: gelu`; there is no Qwen2 repo of that name. `scripts/make-tiny-model.py` therefore writes random
  float32 checkpoints with the same config keys (`head_dim` 32), and `scripts/arch-smoke.sh` runs them.

## 14.7 Testing

```
scripts/arch-smoke.sh
```

generates tiny random models (plain Llama, llama3 RoPE scaling, Mistral with `sliding_window = 4`, Qwen2 with bias
and a window on layers ≥ 2), converts each to q4 and q8, loads it, generates 8 tokens greedily from a fixed 12-token
prompt on the CPU and on Metal, and requires equal token ids and final logits that agree to 1e-3 relative.
Random weights give gibberish, but the computation is deterministic, so any disagreement is a bug in a backend.
`mvn test` additionally checks the RoPE rule and the config parsing of each family. The Gemma part of that
(`GemmaChecks`, called from `SmokeTest`) compares `gelu_tanh`, the soft-cap and the `(1 + w)` RMSNorm with
hand-computed values and parses the real Hub configs listed in 14.10. The arch smoke adds Gemma 2 (`head_dim` 64,
attention cap 1.0 so that it matters on random weights, final cap 30, local layers 0 and 2 with a window of 6),
Gemma 3 text (q/k norm, 6 layers = 5 local + 1 global, linear-×8 global RoPE and 10 000-base local RoPE), the
multimodal wrapper layout and `head_dim` 256 variants of both.

The three Phi-3 models (fused weights, longrope with the long factors, partial rotary, sliding window; the same with
the short factors and tied embeddings; headDim 64 MHA with plain RoPE) are also checked by `ArchSmoke` for the
fused split: each Q8_0 row of the converted `q_proj`/`k_proj`/`v_proj`/`gate_up` equals the matching row range of
the checkpoint's `qkv_proj` / `gate_up_proj` within the quantization step.

## 14.8 GLM (`glm`, `glm4`)

Code: `model/InterleavedRope`, `ModelConfig.ropeInterleaved/rotaryDim/postNorms`, `CpuBackend.addNormed`,
`kernels.metal` (`norm_rope_il`, `rmsnorm_add`, appended at the end), `ModelConverter.fusedGateUp`

Checked against transformers' `modeling_glm.py` / `modeling_glm4.py` and the Hub `config.json` and safetensors
headers of `THUDM/GLM-Edge-1.5B-Chat`, `THUDM/glm-4-9b-chat-hf`, `THUDM/GLM-4-9B-0414`, `THUDM/GLM-Z1-9B-0414`
(no weights were downloaded).

### Interleaved, partial RoPE

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
`rotary_dim / 2` entries per position, and the same tables serve both backends (`norm_rope_il` on Metal, one thread
per rotated pair; `InterleavedRope.rotate` on the CPU). The existing rotate-half path is untouched.

`partial_rotary_factor` is 0.5 when the key is absent (the config class sets it for backward compatibility), which
is the case for `glm-4-9b-chat-hf`; GLM-Edge sets 1.0 explicitly and GLM-4-0414 sets 0.5. Newer transformers nest
`rope_theta` and `partial_rotary_factor` in `rope_parameters`; both layouts are read.

### Norms

`glm` has the usual two norms. `glm4` has four per layer; the two extra ones are applied to the sub-layer output
*before* it is added to the residual stream:

```
glm :  x = x + o_proj(attn(input_layernorm(x)))              x = x + mlp(post_attention_layernorm(x))
glm4:  x = x + post_self_attn_layernorm(o_proj(attn(input_layernorm(x))))
       x = x + post_mlp_layernorm(mlp(post_attention_layernorm(x)))
```

The engine projects into the scratch buffer `xb`, then adds `rmsnorm(xb) * w` to `x` (`rmsnorm_add` on Metal).
Models without these norms keep the fused `matmul(..., accumulate)` path.

### Other differences

- q/k/v bias when `attention_bias` is true (the config default if the key is missing); `o_proj` has no bias. GLM-Edge
  has `attention_bias: false`, glm-4-9b-chat-hf and GLM-4-0414 have `true`.
- MLP: `mlp.gate_up_proj.weight` is `[2f, d]`, gate rows first, then up rows (`gate, up = chunk(2)`): the layout of
  our `mlp.gate_up`, so it is quantized as is. `mlp.down_proj.weight` as usual, activation `silu`.
- `head_dim` is explicit (128) in all four checkpoints (GQA with 4 kv heads for Edge-1.5B, 2 for the 9B models); the
  engine takes it from the config rather than `hidden/heads`.
- No BOS token: the chat templates start with `[gMASK]<sop>` (GLM-4) or `<|user|>` (GLM-Edge), so `promptBosId` is -1.
  Stop ids come from `eos_token_id` (`[59246, 59253, 59255]` for Edge, `[151329, 151336, 151338]` for the others)
  plus the template's stop tokens. `pad_token_id` is ignored.
- `tie_word_embeddings`: true for GLM-Edge-1.5B-Chat (the checkpoint nevertheless contains `lm_head.weight`, which
  the converter skips), false for the 9B models.
- `rms_norm_eps` differs per checkpoint: 1e-5 (Edge, 0414), 1.5625e-7 (glm-4-9b-chat-hf).

### Verified configuration

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

### Testing

`scripts/arch-smoke.sh` adds three tiny random models: `glm` (bias, `head_dim` 32, default factor 0.5), `glmedge`
(`head_dim` 64 with `hidden/heads` = 32, factor 1.0, no bias, tied) and `glm4` (bias, `head_dim` 64, factor 0.5,
four norms); CPU and Metal must agree on tokens and logits for q4 and q8. `SmokeTest.glmRope` checks the rotation
on a hand-computed 6-element head and the configs above. A pure-Python float32 re-implementation of the
transformers forward pass on the same tiny checkpoints gives the same first generated token for all three.

## 14.9 Phi-3 / Phi-3.5 / Phi-4 (`model_type: phi3`)

Same decoder as Llama (RMSNorm, GQA, SwiGLU, no biases, no QK-norm); four things differ. Verified against
`modeling_phi3.py` and `_compute_longrope_parameters` in transformers (main) and the Hub files of the four
repositories below.

### Fused projections

The checkpoint stores `self_attn.qkv_proj.weight` `[(heads + 2·kv_heads)·head_dim, d]` with the rows ordered
q (`heads·head_dim`), k (`kv_heads·head_dim`), v (`kv_heads·head_dim`) (`qkv[..., :q]`, `qkv[..., q:q+kv]`,
`qkv[..., q+kv:]` in the model code), and `mlp.gate_up_proj.weight` `[2·f, d]` with gate first, then up
(`up_states.chunk(2, dim=-1)` → `gate, up`). There is no `q_proj`/`k_proj`/`v_proj`/`gate_proj`/`up_proj`.

The converter splits qkv by rows into the usual `q_proj`/`k_proj`/`v_proj` cache tensors (so the backends see a
Llama layer) and quantizes `gate_up_proj` directly as `mlp.gate_up`, which is already the stacked gate/up layout.
No other Phi tensor needs special handling: `input_layernorm`, `post_attention_layernorm`, `o_proj`, `down_proj`,
`model.norm`, `model.embed_tokens` and (untied models) `lm_head` have the Llama names.

### LongRoPE (`rope_scaling.type` `longrope`, legacy `su`)

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

### Partial rotary (Phi-4-mini)

`partial_rotary_factor: 0.75` with `head_dim = 128` rotates only the first `rot = int(128 · 0.75) = 96` dimensions
of every query and key head, with rotate-half *inside that prefix* (dimension `i` pairs with `i + 48`); dimensions
96 … 127 pass through unchanged. The inverse frequencies use `rot` in the exponent (`theta^(2i/rot)`), not
`head_dim`. `ModelConfig.partialRotaryFactor` / `rotaryDim()` are shared with GLM (14.8); Phi keeps
`ropeInterleaved()` false. `Rope` builds `[ctx × rot/2]` tables, `CpuBackend.normRope` pairs `(i, i + rot/2)`, and
Metal reuses the GLM `norm_rope_il` dispatch (q and k heads together) with its `rotateHalf` parameter set, so the
pairs are `(i, i + rot/2)` instead of `(2i, 2i+1)`. Partial rotation together with a QK-norm is rejected on Metal (no
Phi model has one).

### Other config keys

| Key | Rule |
|---|---|
| `sliding_window` | all layers. Phi-3-mini-4k: 2047 (a real window, since the context is 4096); Phi-3.5 and Phi-4-mini: 262144 (never reached); phi-4: `null` |
| `head_dim` | absent, so `hidden_size / num_attention_heads` (Phi-3-mini and 3.5: 96; Phi-4-mini and phi-4: 128). 96 is a multiple of 32, so Metal works |
| `tie_word_embeddings` | true for Phi-4-mini (no `lm_head.weight`), false for the others |
| `attention_bias`, `mlp_bias` | false, and rejected if true |
| `embd_pdrop`, `resid_pdrop`, `attention_dropout`, `initializer_range`, `full_attn_mod`, `interpolate_factor`, `lm_head_bias` | ignored (dropout is inference-off; the last three only appear in Phi-4-mini and are 1 / 1 / false) |
| `rms_norm_eps`, `rope_theta`, `hidden_act` | as for Llama (`silu` only); phi-4 has `rope_theta` 250000 |

### BOS, EOS and templates

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
`<|endoftext|>` as `bos_token_id` but never prepend it. So `ModelConfig.promptBosId` is -1 for `phi3` (a template
or prompt that contains the token itself is of course left alone). The stop ids are the union of config,
`generation_config.json` and the template's `<|end|>` / `<|im_end|>` / `<|endoftext|>`. `ChatTemplate` picks the
phi-4 ChatML format when the vocabulary has `<|im_sep|>` (phi-4 14B), the Phi-4-mini format for other byte-level
vocabularies, and the Phi-3 one otherwise.

### Verified on the Hub

Headers read with a ranged request (first 2 MB of shard 1, no weights downloaded), all BF16, layer 0:

| Repository | d | heads/kv | head_dim | f | V | layers | `qkv_proj` | `gate_up_proj` | `lm_head` |
|---|---|---|---|---|---|---|---|---|---|
| `microsoft/Phi-3-mini-4k-instruct` | 3072 | 32/32 | 96 | 8192 | 32064 | 32 | 9216 × 3072 | 16384 × 3072 | present |
| `microsoft/Phi-3.5-mini-instruct` | 3072 | 32/32 | 96 | 8192 | 32064 | 32 | 9216 × 3072 | 16384 × 3072 | present |
| `microsoft/Phi-4-mini-instruct` | 3072 | 24/8 | 128 | 8192 | 200064 | 32 | 5120 × 3072 | 16384 × 3072 | absent (tied) |
| `microsoft/phi-4` | 5120 | 40/10 | 128 | 17920 | 100352 | 40 | 7680 × 5120 | 35840 × 5120 | present |

(`9216 = 3072 + 2·3072`, `5120 = 3072 + 2·1024`, `7680 = 5120 + 2·1280`.) Per layer the only other tensors are
`input_layernorm.weight`, `post_attention_layernorm.weight`, `self_attn.o_proj.weight` and `mlp.down_proj.weight`.

### Not supported for Phi

- Phi-3-small, Phi-3-vision, Phi-3.5-MoE (`phimoe`), Phi-4-multimodal: other `model_type`s / extra towers.
- Per-request switching between short and long RoPE factors (see above).
- Phi-1 / Phi-2 (`model_type: phi`: LayerNorm, GELU, parallel attention) are a different architecture.

## 14.10 Gemma 2 and Gemma 3

Code: `model/ModelConfig.Gemma`, `model/GemmaMath`, `model/Rope.local`, `ModelConverter.normGain/canonicalName`,
`CpuBackend.geluMul`, `kernels.metal` (`rmsnorm_add` shared with GLM-4, `matmul_swiglu`'s `gelu` flag, `softcap`, `AttnParams.softcap`).
Behaviour checked against transformers `modeling_gemma2.py` / `modeling_gemma3.py` (v4.51.3).

Same decoder skeleton, with these differences from the Llama-style layer of chapter 5:

| # | Gemma | Formula | Where |
|---|---|---|---|
| 1 | embedding scale | `x0 = embed[token] * normalizer`, `normalizer = bf16(sqrt(hidden_size))` | `ModelConfig.embedScale`, applied right after the row is dequantized (both backends) |
| 2 | RMSNorm gain `1 + w` | `y = x * rsqrt(mean(x²) + eps) * (1 + w)`, in f32 (`Gemma3RMSNorm`; the checkpoints store `w`, initialised to 0) | the converter stores `1 + w` as f32 (`GemmaMath.foldNormGain`), so the unchanged `rmsnorm` / `norm_rope` kernels apply it; this includes `q_norm`, `k_norm` and the final norm |
| 3 | four norms per layer | `h = x + post_attn_norm(attn(input_norm(x)))`, `x' = h + post_ffn_norm(mlp(pre_ffn_norm(h)))` | o_proj and down_proj write to `xb` (no accumulate); the GLM-4 post-norm path (`rmsnormAdd` / `rmsnorm_add`) normalizes it and adds it to the residual `x`; `postNorms()` is true for Gemma too |
| 4 | GeLU-tanh MLP | `down(gelu_tanh(gate(x)) * up(x))`, `gelu_tanh(x) = 0.5x(1 + tanh(sqrt(2/π)(x + 0.044715x³)))` | `ModelConfig.activation()` is `gelu_tanh` for Gemma, else `silu`; CPU `geluMul`; Metal: the fused `matmul_swiglu` kernels take a `gelu` flag in `SwigluParams` and apply `gelu_tanh` instead of `silu` in the epilogue |
| 5 | attention scale | `scores = q·k * query_pre_attn_scalar^-0.5` (not `head_dim^-0.5`; in the three checked configs the scalar equals `head_dim` = 256 so both give 1/16, the test checkpoints use 48 to tell them apart) | `ModelConfig.attnScale()` goes into the scale parameter of every attention path |
| 6 | attention soft-cap (Gemma 2) | `s' = tanh(s / 50) * 50`, applied to the scaled score before the softmax; `null` in Gemma 3 | CPU f32 and f16 loops, Metal `attention` / `attention_h` (`AttnParams.softcap`, 0 = off, argument clamped to ±20 because fast-math `tanh` can overflow) |
| 7 | final logit soft-cap (Gemma 2) | `l' = tanh(l / 30) * 30` after the LM head; `null` in Gemma 3 | CPU: after the head matmul; Metal: `softcap` kernel on the logits buffer |
| 8 | QK-norm (Gemma 3 only) | per-head `(1+w)` RMSNorm of q and k before RoPE | reuses the Qwen3 `qNorm`/`kNorm` path (`qkNorm` is true for `gemma3_text`) |
| 9 | local / global layers | local layers use the sliding window, global layers the full context. Gemma 2: layer `i` is local if `i % 2 == 0`. Gemma 3: local unless `(i + 1) % sliding_window_pattern == 0` (5 local, 1 global). `layer_types` (`sliding_attention` / `full_attention`) overrides both rules | `ModelConfig.Gemma.local[]`, `windowFor(l)` |
| 10 | two RoPE tables (Gemma 3) | local layers: base `rope_local_base_freq` (10 000), no scaling; global layers: `rope_theta` (1 000 000) with `rope_scaling` (`linear`, factor 8 in the 4B, none in the 1B) | `Rope.of` (global) and `Rope.local`; each layer picks one (`usesLocalRope(l)`); Metal keeps two cos/sin buffers. Gemma 2 has one table (`rope_theta` 10 000) |
| 11 | tied head, BOS, stops | `tie_word_embeddings` is absent from the configs and defaults to true; `<bos>` (id 2) is prepended (`promptBosId`; the chat template already emits it, so it is not doubled); stops: `eos_token_id` ([1, 107] for Gemma 2, 106 for Gemma 3) plus `<end_of_turn>` / `<eos>` from the template | |

The sliding window follows transformers' mask `kv > q - window` (a token sees `window` positions including itself),
the same convention as Mistral above. Gemma 2's `hidden_act` and `hidden_activation` are both
`gelu_pytorch_tanh`; Gemma 3 only has `hidden_activation`. The embedding normalizer is a tensor of the checkpoint
dtype in transformers, so `sqrt(1152) = 33.94` is really `34.0` in a bf16 model (`bf16(sqrt(2304)) = 48` exactly);
`ModelConfig` reads `torch_dtype` for this (`float32` checkpoints, such as the test ones, keep the exact value).

**Config keys read** (`ModelConfig.load`): `hidden_size`, `intermediate_size`, `num_hidden_layers`,
`num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `rms_norm_eps`, `rope_theta`,
`rope_local_base_freq`, `rope_scaling` (`rope_type` linear; also `rope_parameters`, keyed by layer type, of
transformers 5), `sliding_window`, `sliding_window_pattern`, `layer_types`, `query_pre_attn_scalar`,
`attn_logit_softcapping`, `final_logit_softcapping`, `hidden_activation`, `bos_token_id`, `eos_token_id`,
`torch_dtype`. For the multimodal `gemma3` wrapper (`Gemma3ForConditionalGeneration`) all of these are taken from
`text_config`, and `bos_token_id`, `eos_token_id` and `torch_dtype` fall back to the top level; `vision_config` and
the image token ids are ignored.

**What the Hub says** (config.json and safetensors headers fetched with Range requests, no weights downloaded):

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

**Tensors** (all BF16; `d` = hidden, `q = heads·head_dim`, `kv = kv_heads·head_dim`, `f` = ffn, `V` = vocab; headers of
`unsloth/gemma-2-2b-it`: 288 tensors, `unsloth/gemma-3-1b-it`: 340 tensors):

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

**How it was verified.** Beyond the CPU == Metal comparison of the arch smoke, the tiny random checkpoints were
also run through an independent pure-Python implementation written line by line from `modeling_gemma2/3.py`
(sequential attention with an explicit mask, `rotate_half`, `(1 + w)` norms, soft-caps). Final logits of the last
prompt token agree with it to quantization noise (Q8_0: correlation 0.99997, max error 0.3–0.9 % of the largest
logit; Q4_0: 0.9994, 1–4.5 %) on both backends, for Gemma 2, Gemma 3 and the multimodal layout, with f32 and f16 KV
caches and `head_dim` 32, 64 and 256. A 262 144-row vocabulary also runs (CPU == Metal).

**Not supported / not verified**: images (the vision tower and projector are skipped, so Gemma 3 4B+ is text-only);
the 4B+ sizes and Gemma 2 9B/27B were checked only through their configs and tensor indexes (only the 2B/1B/4B
configs above were read), no real weights were run; `rope_scaling` types other than `linear`/`llama3`; Gemma 3n and
Gemma 1 (`model_type: gemma`, no extra norms) are rejected.

## 14.11 DeepSeek-V2 / V3 (Multi-head Latent Attention + shared experts)

Code: `model/DeepSeekConfig`, `model/DeepSeekRouter`, `model/MlaWeights`, `model/Rope.deepseek/yarnInvFreq`,
`model/ModelConverter.convertMla/convertDeepseekMoe`, `cpu/MlaCpu`, `metal/MlaMetal`, kernels `mla_rope`,
`mla_kv_store`, `attention_mla`, `router_ds` (appended to `kernels.metal`), `tokenizer/ChatTemplate` (`deepseek-v2`).

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

Standard attention projects `x` to per-head `q`, `k`, `v`. MLA (DeepSeek-V2) squeezes keys and values through a small
shared *latent* so that a cache of `kv_lora_rank + qk_rope_head_dim` numbers per token would suffice (the paper's
point). A head's query/key has two parts: `qk_nope_head_dim` dimensions without position and `qk_rope_head_dim`
dimensions that are rotated by RoPE. The rotated key part is a single vector shared by all heads.

```
                      x [dim]
        ┌──────────────┼───────────────────────────────┐
        │ q path       │ kv path                       │
   q_proj (V2-Lite)    kv_a_proj_with_mqa [kv_lora + rope, dim]
   or q_a_proj [q_lora, dim]          ├────────────┬──────────────┐
     → q_a_layernorm                  │ latent     │ k_pe [rope]  │
     → q_b_proj [H·qk, q_lora]        │ [kv_lora]  │ (one vector) │
        │                             │ kv_a_layernorm              │
        │                             │ kv_b_proj [H·(nope+v), kv_lora]
   q [H, qk]                          │ → per head: k_nope [nope] | v [v_dim]
   = q_nope | q_pe                    │              │                 │
        │ rope(q_pe)                  │              │  rope(k_pe)     │
        ▼                             ▼              ▼                 ▼
   query_h = [q_nope_h | rope(q_pe_h)]     key_h = [k_nope_h | rope(k_pe)]   value_h = v_h
        └──────────── softmax(query_h · key_s · scale) · value_s  (causal) ───────────┘
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
  in place by `pos · inv_freq[i]`, which is what `MlaCpu.rotate` and `mla_rope` do (no permutation is needed).
- **Different widths.** Query/key are `qk` = 192 wide and the value is `v_dim` = 128 wide. The CPU loops use the two widths
  directly; Metal has the separate kernel `attention_mla` (a copy of `attention` with `qkDim` and `vDim`, no grouping and no
  window, dispatched in the same 32-token tiles with `tokenOffset` as `attention`, each lane owns `qk/32 ≤ 8` resp. `v/32 ≤ 8` dimensions) rather than padding V to 192. Both widths must be
  multiples of 32 on Metal.
- **Softmax scale** is `qk^-0.5` (192, not the 128 of an ordinary head) times `mscale²` with
  `mscale = 0.1 · mscale_all_dim · ln(factor) + 1` when `rope_scaling.mscale_all_dim` is set (V2-Lite: 0.707 → 1.2608²;
  V3: 1.0 → 1.3689²), see `DeepSeekConfig.softmaxScale`.
- **KV cache.** The engine stores the *expanded* keys (`H·qk`) and values (`H·v_dim`) per layer in f32, like the other
  families, instead of the compressed latent. This is the simplest correct choice and costs memory: V2-Lite needs
  `(16·192 + 16·128) · 4 B` = 20 KB per layer and token, **553 KB per token over 27 layers** (2.2 GB at `--ctx 4096`),
  against 62 KB per token for a latent cache (`(512 + 64) · 4 B · 27`). `backend.kv-dtype=f16` is rejected for these
  models. (An absorbed-matrix latent cache, where `kv_b` is folded into the query and the output projection, is the known
  way to shrink this and is not implemented.)
- `ModelConfig` reports `headDim = qk` (192), `kvHeads = heads`, so `qDim = H·qk` and the Metal "multiple of 32, ≤ 256"
  check applies to 192. The BOS id (`bos_token_id`, 100000 for V2) is put in front of raw prompts; the chat template has it.

### 14.11.2 YaRN rotary scaling

`rope_scaling: {"type": "yarn", factor, original_max_position_embeddings, beta_fast, beta_slow, mscale, mscale_all_dim}`
applies only to the `rope` = 64 rotary dimensions (`Rope.deepseek`, `Rope.yarnInvFreq`). Each of the 32 frequency pairs
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
  are stored as `mlp.shared.gate_up` / `mlp.shared.down` and run with the dense fused-SwiGLU + down-projection path
  (`CpuBackend.forward`, `MlaMetal.sharedExperts`).
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
`norm_topk_prob` test), which `DeepSeekConfig.scaleAlways` reproduces.

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
`routed_scaling_factor`, `bos_token_id`, `eos_token_id`, `tie_word_embeddings` (false). `attention_bias: true`,
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
Q8_0 embedding and head (2 × 102400 × 2048 at 9 bits) = 0.47 GB: about **10 GB**, so it fits a 32 GB machine, plus the f32
KV cache of 553 KB per token. Decode touches the 6 routed + 2 shared experts of each of 26 layers, about 2.4 B parameters per
token, so it is far cheaper than the weight size suggests (as for Qwen3-MoE, chapter 12).

DeepSeek-V3 / R1 has 671 B parameters (the Hub lists 685 B including the multi-token-prediction module, which the converter skips): at Q4_0 that is 671 B × 5/8 B ≈ **420 GB** (+ 2 GB of
Q8_0 embedding and head), and its expanded f32 KV cache would be `128·(192 + 128)·4 B · 61` = **10 MB per token** (40 GB at
`--ctx 4096`). It cannot run on a 32 GB laptop (not even on 128 GB; the 1.37 TB FP8 download alone exceeds the disk).

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
`deepseek_v3` (sigmoid, `noaux_tc` with bias, 4 groups, the stray layer 4). `scripts/arch-smoke.sh` runs each on CPU and
Metal (q4 and q8; equal tokens, final logits within 1e-3 relative, measured below 1e-5) and `src/test/.../DeepSeekRef` compares the CPU and Metal logits of the
12-token prompt with `scripts/deepseek-ref.py`, an independent float64 re-implementation of `modeling_deepseek.py`
(explicit de-interleave + rotate-half, `MoEGate`, YaRN) run on weights rounded to Q8_0 the same way the converter does: the
largest difference is 1e-5 of the logit range. (Without the rounding the engine is 1 – 6 % off, which is plain quantization
noise: random weights with sharp attention are sensitive.) `mvn test` checks the YaRN frequencies against hand-computed values
(14.11.2), the routing rules on hand-made score matrices (14.11.4), the V2-Lite and V3 configs, the V2 chat template and the
FP8 refusal.
