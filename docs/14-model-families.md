# 14. Model families (Llama, Mistral, Qwen2, Phi-3/4, DeepSeek-R1 distills)

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

A different `model_type`, a `hidden_act` other than `silu`, `mlp_bias: true`, or a `rope_scaling` type other than
`default`/`linear`/`llama3`/`longrope` (e.g. `yarn`) is rejected at load time with a message naming the field.

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

- `yarn`, `dynamic` RoPE scaling; `hidden_act` other than `silu` (e.g. Gemma); MLP bias; an
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
`mvn test` additionally checks the RoPE rule and the config parsing of each family.

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
