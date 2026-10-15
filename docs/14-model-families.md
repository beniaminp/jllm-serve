# 14. Model families (Llama, Mistral, Qwen2, DeepSeek-R1 distills)

Code: `model/ModelConfig`, `model/Rope`, `model/ModelConverter` (biases), `cpu/CpuBackend.attention/addBias`,
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

A different `model_type`, a `hidden_act` other than `silu`, `mlp_bias: true`, or a `rope_scaling` type other than
`default`/`linear`/`llama3` (e.g. `yarn`, `longrope`) is rejected at load time with a message naming the field.

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

- `yarn`, `longrope`, `dynamic` RoPE scaling; `hidden_act` other than `silu` (e.g. Gemma, Phi); MLP bias; an
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
