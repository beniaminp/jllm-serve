# 16. GGUF: running llama.cpp's quantized files without converting them

Code: `gguf/GgufFile.java`, `gguf/GgmlType.java`, `gguf/GgufArchitecture.java`, `gguf/GgufWeights.java`,
`gguf/GgmlCodec.java`, `gguf/GgufModel.java`, `gguf/GgufQwenNext.java`, `model/Ggml.java`, `model/QType.java`, `model/QTensor.java`,
`model/TensorSource.java`, `tokenizer/GgufTokenizer.java`, `cpu/GgufKernels.java`, `cpu/GgufInt8Kernels.java`,
`cpu/KQuantKernels.java`, the native `src/main/native/jllm_cpu*.{c,inc}`, the two *GGUF block types* sections
of `metal/kernels.metal`, `hub/HuggingFaceHub.java` (file selection), `Main.java`

[03](03-safetensors.md) loads a Hugging Face checkpoint and **converts** it once into jllm's own quantized cache.
That is the wrong shape for the files most people actually have: the Hub is full of `*-GGUF` repositories
(`unsloth/Qwen3-0.6B-GGUF`, `Qwen/Qwen3-0.6B-GGUF`, `bartowski/...`) that hold a model *already quantized* by
llama.cpp, in one file, with its tokenizer inside. This chapter is how jllm-serve reads such a file **as it is** —
no conversion step, no cache, the quantized blocks used straight from the memory-mapped file — on both backends:

```sh
./jllm run --model unsloth/Qwen3-0.6B-GGUF --gguf-quant Q4_K_M --prompt "Why is the sky blue?"
./jllm chat --model /path/to/Llama-3.2-1B-Instruct-Q8_0.gguf
```

---

## 16.0 What GGUF is

**GGUF** ("GGML Universal Format", version 3) is a single file that holds, in this order:

1. a small fixed header,
2. **metadata**: a list of typed key/value pairs (`general.architecture = "qwen3"`, `qwen3.block_count = 28`, the
   whole vocabulary as an array of strings, ...),
3. a **tensor directory**: for each tensor its name, shape, element type and offset,
4. the **tensor data**, each tensor aligned, in the order the directory lists them.

Everything a safetensors checkpoint spreads over `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`generation_config.json` and the weight shards is in that one file. Elements are either plain floats or one of
llama.cpp's *block formats* (16.4): the file stores what the engine multiplies with, which is why it can be
mapped and used directly.

---

## 16.1 The file, byte by byte

All integers are little endian. The first bytes of `Qwen3-0.6B-Q8_0.gguf`:

```
00000000: 4747 5546 0300 0000 3601 0000 0000 0000  GGUF....6.......
00000010: 2000 0000 0000 0000 1400 0000 0000 0000   ...............
00000020: 6765 6e65 7261 6c2e 6172 6368 6974 6563  general.architec
00000030: 7475 7265 0800 0000 0500 0000 0000 0000  ture............
00000040: 7177 656e 33 ...                         qwen3
```

```
offset  size  field
0       4     magic            "GGUF"
4       4     version          3  (2 is read too; 1 used 32-bit counts and is rejected)
8       8     tensor_count     0x136 = 310
16      8     metadata_kv_count 0x20 = 32
24      ...   metadata_kv_count entries:
                string key        u64 length, then the bytes (UTF-8, no terminator)      "general.architecture"
                u32   value type  (table below)                                           8 = string
                value             "qwen3": u64 length 5, then the bytes
        ...   tensor_count tensor infos:
                string name       "blk.0.attn_k.weight"
                u32   n_dims      1..4
                u64   dims[n_dims]   innermost first: dims[0] is the row length, i.e. the COLUMNS of a matrix
                u32   ggml type   (16.4)
                u64   offset      from the start of the data section, a multiple of the alignment
        ...   zero padding up to a multiple of the alignment
data    ...   the tensors
```

| value type | id | | value type | id |
|---|---|---|---|---|
| uint8 / int8 | 0 / 1 | | bool (1 byte, 0 or 1) | 7 |
| uint16 / int16 | 2 / 3 | | string | 8 |
| uint32 / int32 | 4 / 5 | | **array**: `u32` element type, `u64` count, elements | 9 |
| float32 | 6 | | uint64 / int64 / float64 | 10 / 11 / 12 |

Arrays are how the vocabulary is stored (`tokenizer.ggml.tokens` is an array of 151 936 strings, `token_type` an
array of int32). The **alignment** is `general.alignment` (default 32): the data section starts at the next
multiple of it after the directory, and every tensor offset is a multiple of it. A model too big for one file is
split into `name-00001-of-0000N.gguf`; part 1 carries the metadata, every part its own tensors. Naming any part
opens them all.

Two details that matter later:

* `dims` is **innermost first** (ggml's `ne[]`). A weight matrix with 3072 rows of 1024 columns is stored as
  `[1024, 3072]`; a stack of 128 experts of 768×2048 as `[2048, 768, 128]`. `GgufFile.Tensor.cols()` is
  `dims[0]`, `rows()` the product of the others, so the row-major data is exactly the engine's
  `[rows, cols]` matrix.
* llama.cpp writes the tensors **sorted by name**: in a layer `ffn_down` comes before `ffn_gate`, and
  `ffn_norm` sits between `ffn_gate` and `ffn_up` (16.6 shows why that matters).

### What `GgufFile` checks

The reader is written like `SafeTensors` (03): the file is mapped, the header is parsed with bounds-checked reads
and a malformed file is an `IOException` naming the field, never an out-of-bounds read. It rejects: a wrong magic
or version, counts larger than the file could hold, a string or array longer than the bytes that remain, a
duplicate key or tensor name, more than 4 dimensions or a non-positive one, a shape or size that overflows `long`,
a row length that is not a multiple of the type's block size, an unknown ggml type id, an alignment that is not a
power of two, a tensor offset that is not aligned, a tensor extending past the end of the file (a truncated
download), and tensors whose byte ranges overlap. `GgufFileTest` truncates a valid file at every byte of its header
and requires an `IOException` each time. A vocabulary-only file (no tensors, e.g. llama.cpp's
`ggml-vocab-*.gguf` test files) ends at the header without padding and is accepted.

---

## 16.2 From metadata to an `Architecture`

jllm's `Architecture` ([14.12](14-model-families.md)) is parsed from a Hugging Face `config.json`. A GGUF has the
same information under llama.cpp's keys, so `GgufArchitecture` writes the `config.json` the model would have and
hands it to the same family parsers (`Families`). The defaults, the validation and the canonical
`Core.description` therefore come out exactly as for a safetensors load of the same model — for Qwen3-0.6B both
print

```
Qwen3ForCausalLM dim=1024 ffn=3072 layers=28 heads=16/8 headDim=128 vocab=151936 theta=1000000
```

| `general.architecture` | `model_type` | tensors differ by |
|---|---|---|
| `qwen3` | `qwen3` | q/k RMSNorm per head |
| `qwen3moe` | `qwen3_moe` | routed experts (`num_experts`, `num_experts_per_tok`, `moe_intermediate_size`; `norm_topk_prob` is always true, as in llama.cpp) |
| `qwen2` | `qwen2` | q/k/v biases |
| `llama` | `llama` (`mistral` when the file has a sliding window or "mistral" in its name) | the q/k row permutation, 16.5 |
| `qwen35` | `qwen3_next` (reported as `qwen3_5`: it has a chat template of its own) | Qwen3.5: Gated DeltaNet layers + gated attention, a dense MLP everywhere, value heads in tiled order, a multi-token-prediction block, 16.13 |
| `qwen3next` | `qwen3_next` | Qwen3-Next-80B-A3B: the same hybrid with experts and a gated shared expert, value heads grouped, 16.13 |
| `qwen35moe` | `qwen3_next` (`qwen3_5`) | Qwen3.5-35B-A3B and its siblings: Qwen3-Next's experts and gated shared expert with Qwen3.5's tiled value heads and separate beta / alpha; checked on tiny models against llama.cpp only, 16.13 |

| GGUF key (prefix `<arch>.`) | config.json key |
|---|---|
| `embedding_length` | `hidden_size` |
| `block_count` | `num_hidden_layers` |
| `feed_forward_length` | `intermediate_size` |
| `attention.head_count`, `attention.head_count_kv` | `num_attention_heads`, `num_key_value_heads` |
| `attention.key_length` | `head_dim` (Qwen3's 128 is not `hidden/heads` = 64) |
| `attention.layer_norm_rms_epsilon` | `rms_norm_eps` |
| `rope.freq_base` | `rope_theta` |
| `rope.dimension_count` (when smaller than the head) | `partial_rotary_factor` |
| `rope.scaling.type = linear`, `rope.scaling.factor` | `rope_scaling` linear |
| `rope.scaling.type = yarn`, `rope.scaling.factor`, `original_context_length`, `yarn_beta_fast`, `yarn_beta_slow`, `attn_factor` | `rope_scaling` yarn: `factor`, `original_max_position_embeddings`, `beta_fast`, `beta_slow`, `attention_factor = attn_factor · (0.1 ln factor + 1)` (below) |
| tensor `rope_freqs.weight` (Llama 3.x) | `rope_scaling` llama3 when the table is the Llama 3 one, else a longrope table |
| `context_length` | `max_position_embeddings` |
| `expert_count`, `expert_used_count`, `expert_feed_forward_length` | `num_experts`, `num_experts_per_tok`, `moe_intermediate_size` |
| `expert_shared_feed_forward_length` (`qwen3next`) | `shared_expert_intermediate_size` |
| `ssm.conv_kernel`, `ssm.state_size`, `ssm.group_count`, `ssm.time_step_rank` (`qwen35`, `qwen3next`) | `linear_conv_kernel_dim`, `linear_key_head_dim` = `linear_value_head_dim`, `linear_num_key_heads`, `linear_num_value_heads` (`ssm.inner_size` must be `state_size × time_step_rank`) |
| `full_attention_interval` (or the per-layer `attention.recurrent_layers`) | `full_attention_interval` (or `layer_types`) |
| `nextn_predict_layers` | `block_count - nextn_predict_layers` layers; the rest is the multi-token-prediction block (16.14) |
| rows of `token_embd.weight` | `vocab_size` |
| absence of `output.weight` | `tie_word_embeddings = true` |
| `tokenizer.ggml.bos_token_id`, `eos_token_id`, `eot_token_id` | `bos_token_id`, `eos_token_id` |

Per-layer head counts, other rope scaling types (and DeepSeek's `rope.scaling.yarn_log_multiplier`) and every other
architecture are refused with a message that names them. The model's chat template is the engine's own for the family ([04](04-tokenizer.md)); the Jinja string in
`tokenizer.chat_template` is ignored, but a warning is printed when it lacks the family's marker
(`<|im_start|>` for Qwen, `<|start_header_id|>` for Llama 3).

### 16.2.1 YaRN (long-context Qwen GGUFs)

🧸 A model learned to count positions up to 32 768. To read four times further, YaRN slows down only the slow
"clock hands" of the position code (the ones that never went all the way round during training) by 4 and leaves the
fast hands alone, then turns the volume of every clock up a little so attention stays as sharp as before.

📐 llama.cpp's converter writes a config's `rope_scaling: {"rope_type": "yarn", "factor": s,
"original_max_position_embeddings": L}` as `<arch>.rope.scaling.type = "yarn"`, `rope.scaling.factor = s`,
`rope.scaling.original_context_length = L` (plus `yarn_beta_fast` / `yarn_beta_slow` when the config sets the betas,
defaults 32 / 1). `GgufArchitecture` maps them back to that `rope_scaling` map, so the GGUF and the safetensors loads
go through the same `Families.yarn` → `Scaling.Yarn` → `Rope.yarnInvFreq` (14.11.2):

```
f_i       = θ^(-2i/d)                                  i = 0 … d/2 − 1, d = rotary dims
dim(β)    = d · ln(L / (β · 2π)) / (2 ln θ)
low, high = max(floor(dim(beta_fast)), 0), min(ceil(dim(beta_slow)), d − 1)
ramp(i)   = clamp((i − low) / (high − low), 0, 1)
inv_freq  = f_i / s · ramp(i) + f_i · (1 − ramp(i))
cos, sin *= attention_factor = attn_factor · (0.1 · ln s + 1)
```

The amplitude: ggml's `rope_yarn` multiplies cos and sin by `attn_factor · (1 + 0.1 ln(1/freq_scale))` with
`freq_scale = 1/s`, so a file's `rope.scaling.attn_factor` (default 1) is a multiplier *on top of* the paper's
`0.1 ln s + 1`, while transformers' `attention_factor` *replaces* it; the mapping above makes the two the same number.
Example, Qwen3 with `s = 4`, `L = 32768`, `θ = 10^6`, `d = 128`: `dim(32) = 128 · ln(32768/(32·2π)) / (2 ln 10^6)
= 23.60`, so `low = 23`; `dim(1) = 39.65`, so `high = 40`. Pairs 0 – 23 keep their frequency, pairs 40 – 63 are
divided by 4, pairs 24 – 39 blend linearly; `attention_factor = 0.1 · ln 4 + 1 = 1.1386`.

`GgufModelTest` writes tiny `qwen3` and `llama` models with `rope_scaling` yarn (factor 4 over 64 positions, betas
16 / 2, attention factor 1.2) as safetensors and as GGUF with those keys, and requires identical inverse frequencies,
the same amplitude (to the f32 rounding of `attn_factor`) and the same logits (3.5e-7 relative for the F32 file).

---

## 16.3 Tensor names

`GgufWeights` implements `TensorSource` — the interface `ModelWeights` reads its tensors through, which the
converted `QuantFile` also implements — and translates the canonical names the loader asks for into llama.cpp's:

| canonical (converter's) name | GGUF name | note |
|---|---|---|
| `model.embed_tokens.weight` | `token_embd.weight` | |
| `lm_head.weight` | `output.weight` | absent for tied embeddings |
| `model.norm.weight` | `output_norm.weight` | f32 |
| `model.layers.N.input_layernorm.weight` | `blk.N.attn_norm.weight` | f32 |
| `...post_attention_layernorm.weight` | `blk.N.ffn_norm.weight` | f32; the norm *before* the MLP |
| `...self_attn.q_norm.weight`, `k_norm` | `blk.N.attn_q_norm.weight`, `attn_k_norm` | Qwen3 |
| `...self_attn.q_proj.weight`, `k_proj`, `v_proj`, `o_proj` | `blk.N.attn_q.weight`, `attn_k`, `attn_v`, `attn_output` | `.bias` for Qwen2 |
| `...mlp.gate_up` (gate rows, then up rows) | `blk.N.ffn_gate.weight` + `blk.N.ffn_up.weight` | two tensors in the file, one stacked matrix in the engine (16.6) |
| `...mlp.down_proj.weight` | `blk.N.ffn_down.weight` | |
| `...mlp.router` | `blk.N.ffn_gate_inp.weight` | `[dim, experts]` f32 |
| `...mlp.experts.gate_up` | `blk.N.ffn_gate_exps.weight` + `blk.N.ffn_up_exps.weight` | `[dim, moeDim, experts]` each; per expert gate rows then up rows in the engine |
| `...mlp.experts.down` | `blk.N.ffn_down_exps.weight` | `[moeDim, dim, experts]`: rows `expert*dim + r`, exactly the stacked layout |

After translation `ModelWeights` runs the same shape checks as for a cache (`DecoderLayer.matches`), so a file that
does not match its own metadata fails at load with the tensor named, not at the first token.

---

## 16.4 The block formats

A ggml *block* is a group of consecutive weights of one row stored with its own scale(s); a row is a sequence of
blocks, a tensor a sequence of rows, and the type's `typeSize` is the bytes per block. The formats this engine
reads in place (`model/Ggml.java` is the scalar definition, `cpu/GgufKernels.java` and `kernels.metal` the kernels):

```
Q4_0    18 B / 32 w   4.5 bit/w    f16 d | 16 bytes: byte j = weight j (low nibble), weight j+16 (high nibble)
                                   w = (nibble - 8) d
Q8_0    34 B / 32 w   8.5 bit/w    f16 d | 32 x int8 q                          w = q d
Q4_K   144 B / 256 w  4.5 bit/w    f16 d | f16 dmin | 12 B scales/mins | 128 B nibbles     w = d sc q - dmin m
Q5_K   176 B / 256 w  5.5 bit/w    f16 d | f16 dmin | 12 B scales/mins | 32 B high bits | 128 B nibbles
Q6_K   210 B / 256 w  6.5625 bit/w 128 B low 4 bits | 64 B high 2 bits | 16 x int8 scales | f16 d   w = d sc (q - 32)
Q2_K    84 B / 256 w  2.625 bit/w  as in chapter 13 (already identical to llama.cpp's block)
```

Q4_0 and Q8_0 are chapter 06's formats with the scale **inside** the block as an f16 instead of in a separate f32
region; the engine's own caches keep their layout, the GGUF ones are new `QType`s (`Q4_0G`, `Q8_0G`) so that both
exist side by side. Because 18 and 34 are not multiples of 4, a block starts only on a 2-byte boundary: the CPU
reads scales with unaligned loads and the GPU reads the quants as `ushort`s.

### Q4_K and Q5_K: 8 sub-blocks of 32 inside a 256-weight super-block

Chapter 13's recipe with 4-bit codes: each 32-weight sub-block `j` has a 6-bit scale `sc_j` and a 6-bit min
`m_j`, themselves multiples of the super-block's `d` and `dmin`:

```
w = d·sc_j·q − dmin·m_j          q = 0..15 (Q4_K) or 0..31 (Q5_K)

scales[12]  byte:   0    1    2    3    4    5    6    7    8    9   10   11
            low 6:  sc0  sc1  sc2  sc3  m0   m1   m2   m3   sc4|m4 ... (nibbles)
            bits 7-6 of bytes 0..7 hold the top 2 bits of sc4..7 and m4..7

  j < 4:   sc_j = b[j]   & 63                       m_j = b[j+4] & 63
  j >= 4:  sc_j = (b[j+4] & 15) | (b[j-4] >> 6) << 4  m_j = (b[j+4] >> 4) | (b[j]   >> 6) << 4
```

The 128 quant bytes are four groups of 32: in group `g` the low nibble of byte `l` is weight `l` of sub-block
`2g` and the high nibble is weight `l` of sub-block `2g+1`. Q5_K adds `qh[32]`: the fifth bit of weight `l` of
sub-block `j` is bit `j` of `qh[l]`. The CPU dot product uses chapter 13's `Σ(d·sc·q − dmin·m)·x = d·Σ sc·(q·x) −
dmin·Σ m·(Σx)` per sub-block; the GPU kernel dequantizes a 32-weight unit into 8 `float4` and takes the dot product.

### Q6_K

Six bits per weight, 16 scales (int8, one per 16 weights) and one f16 `d`. `ql[128]` holds the low 4 bits (two per
byte) and `qh[64]` the high 2 bits (four per byte) of 256 weights in two halves of 128; `w = d·sc·(q − 32)`. It is
the format that `Q4_K_M` files use for half of `attn_v`, half of `ffn_down` and the embedding. It is the engine's own
`QType.Q6_K` (chapter 06), byte-compatible with the file, so these tensors are used in place too.

The other formats real files mix in are read in place as well since 16.4.1: Q4_1, Q5_0, Q5_1 (legacy and mixed files:
the Q4_0 file of Qwen3.8 has its `ffn_down` in Q4_1), Q3_K (`Q3_K_M/L`), IQ4_NL and IQ4_XS (imatrix quants), MXFP4
(gpt-oss) and plain F16 / BF16 matrices. The other IQ\* types and TQ\* are refused by name.

### 16.4.1 Every type at a glance

`bits/weight` is the file's, `CPU` the kernels (V = Vector API float path, I = int8-activation path in Java,
N = native `libjllm_cpu` on AArch64, [07](07-cpu-backend.md)), `Metal` the GPU kernels (decode matvec, tiled and
matrix-unit prefill, the MoE expert kernels, the embedding lookup: all through `gg_unit`). The exact layouts and
formulas of the types that are new here are in [06 §6.12](06-quantization.md#612-the-other-ggml-block-formats-read-in-place-from-gguf-files).

| type | bytes / weights | bits/weight | used | CPU | Metal |
|---|---|---|---|---|---|
| F32 vectors (norms, biases, router) | 4 / 1 | 32 | copied into `float[]` | | |
| F32 matrices | 4 / 1 | 32 | requantized to Q8_0 | | |
| F16, BF16 | 2 / 1 | 16 | **in place** | V | yes |
| Q8_0 | 34 / 32 | 8.5 | **in place** | V I N | yes |
| Q6_K | 210 / 256 | 6.5625 | **in place** | V I N | yes |
| Q5_1 | 24 / 32 | 6.0 | **in place** | V I N | yes |
| Q5_K | 176 / 256 | 5.5 | **in place** | V I N | yes |
| Q5_0 | 22 / 32 | 5.5 | **in place** | V I N | yes |
| Q4_1 | 20 / 32 | 5.0 | **in place** | V I N | yes |
| Q4_0 | 18 / 32 | 4.5 | **in place** | V I N | yes |
| Q4_K | 144 / 256 | 4.5 | **in place** | V I N | yes |
| IQ4_NL | 18 / 32 | 4.5 | **in place** | V I N | yes |
| IQ4_XS | 136 / 256 | 4.25 | **in place** | V I N | yes |
| MXFP4 | 17 / 32 | 4.25 | **in place** | V I N | yes |
| Q3_K | 110 / 256 | 3.4375 | **in place** | V | yes |
| Q2_K | 84 / 256 | 2.625 | **in place** | V | yes |
| IQ1\*, IQ2\*, IQ3\*, TQ1_0, TQ2_0, Q8_K, Q8_1 | | | refused (named in the error) | | |

`-Djllm.gguf.requantize=<types>` (ggml names separated by commas, `all`, or `none`) puts the listed types on the old
path: dequantized row by row and requantized to Q8_0 at load (8.5 bits per weight: more memory, and on the GPU slower
decode when decode is memory-bound). It is a switch for A/B comparisons (`scripts/mac-validate.sh --gguf-types DIR` runs both)
and a fallback; the load line says what happened (`GGUF weights: ... (requantized to Q8_0: ...)`).

**Default per backend.** The Metal backend reads every type in the table in place. The CPU backend requantizes **Q5_1,
Q3_K, F16 and BF16** by default, because their in-place decode is more than 25% slower than the Q8_0 kernels' (16.9 has the
measurements; F16 / BF16 would also read twice the bytes of Q8_0); the others (Q4_1, Q5_0, IQ4_NL, IQ4_XS, MXFP4 and the
K-quants) stay in place on the CPU too. The load prints `GGUF on the CPU backend: [Q3_K] matrices are requantized ...` when a
file has such tensors. An explicit `-Djllm.gguf.requantize=...` replaces the default, so `-Djllm.gguf.requantize=none` reads
everything in place on the CPU as well (the choice for a model that only fits in memory that way), and
`-Djllm.gguf.requantize=Q4_1,Q5_0,Q5_1` is the opt-out for those three types.
Requantizing F16 / BF16 to Q8_0 is lossy (about 0.5% per weight): the CPU default trades that for speed and memory, and
`-Djllm.gguf.requantize=none` keeps the full precision of the file.

---

## 16.5 The llama permutation: why Qwen needs nothing and Llama does

`convert_hf_to_gguf.py` writes the `q_proj` and `k_proj` weights of the **`llama` architecture** with the rows of
every head permuted (its `LlamaModel.permute`):

```python
def permute(weights, n_head, n_head_kv):
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                   .swapaxes(1, 2).reshape(weights.shape))
```

Hugging Face's Llama rotates dimension `j` of a head together with dimension `j + headDim/2` ("rotate half",
`RopeTable.Pairing.HALF`). llama.cpp's Llama implementation kept the original Meta checkpoint's convention:
it rotates **adjacent** dimensions `2j` and `2j+1`. The permutation moves HF row `c·half + j` of a head to row
`2j + c`, so after it the HF pair `(j, j+half)` is the adjacent pair `(2j, 2j+1)`:

```
head rows in the HF checkpoint:   0 1 2 3 | 4 5 6 7        (headDim 8, half 4; pairs (0,4) (1,5) (2,6) (3,7))
head rows in the GGUF:            0 4 1 5 2 6 3 7          (pairs (0,1) (2,3) (4,5) (6,7))
```

The qwen2 and qwen3 architectures use llama.cpp's NEOX rope — rotate half, the Hugging Face layout — and are *not*
permuted.

There are two ways to run a permuted file: un-permute the rows at load (a copy of two thirds of the attention
weights, and the file could no longer be used in place), or **rotate the pairs the file has**. The engine already has
the second pairing (`RopeTable.Pairing.INTERLEAVED`, used for GLM, with the `rope_il` kernel on Metal): for the
`llama` architecture `GgufArchitecture` sets it, and nothing else changes. It is exact because a permutation applied
to both `q` and `k` leaves every score `q·k` unchanged and the rotation of the pair `(2j, 2j+1)` uses the frequency
`θ^(-2j/headDim)` that the HF pair `(j, j+half)` had; `v` and `o_proj` are not permuted by the converter. The
description of such a model therefore carries `rope=interleaved/<rotaryDim>`.

`GgufModelTest` checks this on a tiny random Llama: the same weights written as a safetensors checkpoint
(converted by the engine to Q8_0) and as a GGUF with `permute` applied (F32 tensors, quantized to Q8_0 at load)
give logits that differ by 2.9e-7 relative; the permuted rows read with the half-pairing RoPE of the HF layout differ
by 0.74.

---

## 16.6 Zero-copy: how the kernels read the file's pages

`GgufFile` maps the whole file once (`FileChannel.map`, a page-aligned `MemorySegment`). `QTensor.mapped(type, rows,
cols, slice)` wraps a tensor's byte range with no copy: `rowOffset(r)` is `r · blocksPerRow · typeSize` and the
scales are inside the blocks (`QType.scaleBytes == 0`). The CPU kernels take `(segment, offset)` as always — a vector
load from a `MemorySegment` does not care about alignment — so they run on the file's bytes.

**Metal.** The GPU needs a buffer; Apple Silicon's unified memory lets `newBufferWithBytesNoCopy` wrap existing
pages, but only whole 16 KiB pages. A tensor is a slice somewhere inside the file, so `Metal.wrapRange` wraps the
pages the slice touches and the buffer carries the slice's offset in its first page; `MetalKernels` adds it to every
binding offset. (One buffer for the whole file would also work, but a 17 GB file can exceed
`maxBufferLength`; per-tensor buffers over overlapping pages cost nothing.) The kernels need no alignment beyond what
the file guarantees: tensor offsets are multiples of 32, so K-quant blocks (144 / 176 bytes) are 4-byte aligned for the
`half2` scale read, and the 18 / 34-byte legacy blocks are read as `ushort`s.

**gate and up without a copy.** The engine multiplies `[gate; up]` as one stacked matrix (`matmul_swiglu` computes
gate row `r` and up row `hidden + r` in one simdgroup), but a GGUF has two tensors and llama.cpp's name sort puts
`ffn_norm` between them, so they are not adjacent. `QTensor.split(gate, up, interleave)` represents the stack as two
views: `partOf(r)` / `partRow(r)` route a logical row to the view and row that hold it (the CPU entry points
`VectorKernels.dotRow` / `dequantRow` / `Int8Kernels.dotRow` do this first); the swiglu kernels read the up rows
from the second buffer, bound where the other formats bind their f32 scales — GGUF block types have none, so
`buffer(1)` is free. For the experts of a MoE layer the two `[experts · moeDim, dim]` tensors are a split pair with
`interleave = moeDim`: logical row `e·2m + k` is row `e·m + k` of the gate view for `k < m`, else row `e·m + k − m`
of the up view; `moe_up` reads gate and up rows from the two views. `ffn_down_exps` already is the stacked layout.
Only Q2_K, whose kernels read one block, copies its pair into one allocation.

**q, k and v of different types.** `Q4_K_M` quantizes `attn_v` as Q6_K in half the layers and `attn_q`/`attn_k` as
Q4_K, and a kernel instantiation has one block format. The fused q/k/v dispatch (`matmul3`) therefore falls back to
one single-matrix dispatch per projection, each with the kernel of its own type, when the three types differ.

**What is converted at load.** Anything the kernels cannot read is dequantized row by row and quantized to the
engine's own Q8_0 in memory (`QTensor.quantize`, in parallel), and reported:
`GGUF weights: ... (requantized to Q8_0: 3 x Q4_1, 1 x Q6_K)` (the line is absent when everything is used in place).

| in the file | used |
|---|---|
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, MXFP4, F16, BF16 | **in place** (zero copy, both backends); the table in 16.4.1 |
| F32 matrices, and the types named in `-Djllm.gguf.requantize` | quantized to Q8_0 in memory at load (a pass over all weights; the F32 / BF16 file of a small model becomes a Q8_0 model in RAM) |
| F32 vectors (norms, biases, router) | copied into `float[]` |

**Mixed files.** Real files mix types (a `Q4_0` file has `ffn_down` in Q4_1, `ssm_out` in Q5_K, `output` in Q6_K; gpt-oss has
MXFP4 experts and F32/BF16 elsewhere), and every kernel instantiation has one block format. The fused q/k/v dispatch
falls back to one dispatch per projection when the three types differ (above); the dense gate|up pair and the
experts' gate|up pair are split views of two tensors of the same type, and if the two differ the pair is requantized
(`GGUF weights:` says so).

The int8-activation CPU path (`backend.cpu.int8-activations`, [07](07-cpu-backend.md)) has kernels for Q4_0 and Q8_0
and, for these files, `Q4_0G` and `Q8_0G`, the K-quants Q4_K / Q5_K / Q6_K (`KQuantKernels`) and Q4_1, Q5_0, Q5_1, IQ4_NL,
MXFP4, IQ4_XS (`GgufInt8Kernels`; native AArch64 versions in `libjllm_cpu`); `CpuMatmul.useInt8` excludes Q2_K,
Q3_K, F16 and BF16, which stay on the float kernels.

---

## 16.7 The tokenizer in the file

`GgufTokenizer` writes the `tokenizer.json` that the model's Hugging Face tokenizer would have and loads it with the
existing encoder ([04](04-tokenizer.md)), so there is one BPE implementation:

* **`tokenizer.ggml.model = gpt2`**: byte-level BPE. `tokens` are the vocabulary and `merges` the merge list.
  `token_type` separates `CONTROL` tokens (added tokens flagged `special`: `<|im_start|>`, `<|endoftext|>`) and
  `USER_DEFINED` ones (added, *not* special: `<think>`, `<tool_call>`) from the vocabulary; as in llama.cpp,
  tokens it knows by name (`<|im_end|>`, `</s>`, the fill-in-the-middle markers...) are control tokens whatever the
  file says. `tokenizer.ggml.pre` picks the pre-tokenizer: a list of regular expressions applied one after the other
  plus `ignore_merges`, exactly llama.cpp's (81 values, 16.12; a missing key or `default` is llama.cpp's `default`,
  which is **not** the GPT-2 regex). Any other value is refused with its name; the loader then fetches the
  original model's `tokenizer.json` from the repository named in `general.base_model.0.repo_url` /
  `general.source.url`, and fails clearly if the file names none.
* **`tokenizer.ggml.model = llama`**: SentencePiece (Llama 2, Mistral 7B, TinyLlama, Gemma, Phi-3). The file has
  **scores**, not merges, and llama.cpp segments with them directly: the characters are the first symbols, then the
  adjacent pair whose concatenation is a token of the vocabulary (of any type) with the best score is merged, equal
  scores leftmost first, until none is; a character that is no token becomes its `<0xXX>` byte tokens (`SpmGreedy`).
  This differs from turning the scores into BPE merges, as `transformers` does, as soon as two scores are equal
  (Gemma's are all 0). Spaces become `▁`; a `▁` is put in front of the text *and of the text after every special
  token* when `add_space_prefix` (default true; Gemma: false).

Checked three ways (`GgufTokenizerTest`, `scripts/gguf-tokenizer-check.sh`): llama.cpp's own tokenizer tests
(`ggml-vocab-llama-spm`, `-phi-3`, `-deepseek-coder`: 47 strings each, ids must equal llama.cpp's; the other 11
fixtures are checked by hand), vocabularies generated for every pre value and compared with `llama-tokenize`
(16.12), and the Qwen3-0.6B GGUF against `tokenizer.json` and the fixture of the Python `tokenizers` package: **145 of
145** strings, the same decodings, and the **44** chat renders of `src/test/resources/tokenizers/qwen3.json`; 5 more
strings differ from the Hugging Face tokenizer by design, as llama.cpp does (16.12).

---

## 16.8 Downloading and selecting a file

A `*-GGUF` repository holds many files of one model (`Qwen3-0.6B-Q4_K_M.gguf`, `-Q8_0.gguf`, ...); loading all would
be absurd. `--model <repo>` with a repository that has `.gguf` files and no safetensors checkpoint (or with
`model.gguf-file` / `model.gguf-quant` set) downloads **only the chosen file** — and its `-0000i-of-0000N` siblings —
with the same resumable, SHA-256 verified `HuggingFaceHub.downloadFile` as every other download, into
`models/<repo with _>/`:

| setting (flag) | |
|---|---|
| `model.gguf-file` (`--gguf-file`) | the exact file name; any part of a split model selects all parts |
| `model.gguf-quant` (`--gguf-quant`) | a quantization label matched against the file names as a whole word, case-insensitively: `Q4_K_M`, `Q8_0`, `UD-Q4_K_XL`. `Q4_K` does **not** match `Q4_K_M` |
| neither | the only `.gguf` of the repository; with several, an error that lists them |

```
$ ./jllm download --model unsloth/Qwen3-0.6B-GGUF
unsloth/Qwen3-0.6B-GGUF: more than one GGUF file matches no model.gguf-file / model.gguf-quant; available:
  Qwen3-0.6B-BF16.gguf
  Qwen3-0.6B-IQ4_NL.gguf
  ...
set model.gguf-file to an exact file name (--gguf-file) or model.gguf-quant to a quantization label such as Q4_K_M (--gguf-quant)
```

`mmproj-*.gguf` (vision projectors) are never candidates. A file already in the directory is used without asking the
Hub. `--model /path/to/file.gguf` (any name ending in `.gguf`) loads a local file, and `model.quant` does not apply.
Everything after loading is as for a converted model: both backends, `serve`, `chat`, `bench`.

---

## 16.9 Measurements (Apple M1 Pro, Qwen3-0.6B)

**Which tensors are which and how good they are** (`GgufVsSafetensors`: every weight tensor dequantized by the
engine's own load path against the bf16 original in `model.safetensors`; relative RMS error):

| file | tensors | error |
|---|---|---|
| `Q8_0` | 197 Q8_0 | mean 0.55 %, worst 0.58 % |
| `Q4_K_M` (imatrix) | 168 Q4_K, 29 Q6_K | Q4_K mean 7.3 %, worst 7.6 %; Q6_K 1.9 % |
| `Q4_0` | 193 Q4_0, 3 Q4_1, 1 Q6_K | Q4_0 mean 8.5 %, worst 9.3 %; Q4_1 7.7 % |

All 310 tensors of every file match the original within its format's bound, which proves the name mapping and the
dimension order (a swapped `[rows, cols]` or a wrong tensor would give ~100 %).

**Byte parity with `gguf.quants`** (`GgufRealParity`, reference = llama.cpp's gguf-py): 13.4 M weights of real
blocks from each of the three files (the embedding and the tensors of layers 0, 5, 13, 20, 27) dequantized by the
engine are **bit-identical** to the reference for Q4_0, Q8_0, Q4_K, Q6_K and Q4_1. Q5_K, Q3_K, Q5_0, Q5_1 and Q2_K are
checked on random blocks with random packed scales (`GgufKernelsCheck`, 2048 weights per type, fixtures from
`src/test/python/gen_gguf_fixtures.py`): bit-identical too, and the SIMD dequantizers and dot products agree
with the scalar ones within 1e-8 relative. `GgufMetalCheck` runs the Metal kernels of the four in-place types
(`matmul`, the fused rmsnorm, tiled prefill, `matmul3` also with mixed types and offsets, swiglu and `moe_up` on split
views, `moe_down`) against a double-precision reference: relative error < 5e-5.

**End to end.**

| check | result |
|---|---|
| haiku and "capital of France" prompts, greedy, Qwen3-0.6B Q8_0 GGUF: Metal, CPU (float activations), and the engine's own `--quant q8` conversion of the safetensors, on both backends | **all four outputs identical** per prompt |
| a 160-token greedy generation (hash-map prompt), the same four runs | identical (md5 `8090eb37...`) |
| Q4_K_M GGUF, haiku: Metal vs CPU | identical |
| random GGUF models of every family (llama, qwen2, qwen3, qwen3moe) × Q4_0 / Q8_0 / Q4_K / Q5_K / mixed per tensor, prefill and decode (`GgufBackendsTest`) | CPU vs Metal max logit difference ≤ 3.5e-6 relative, same greedy tokens |
| tiny models, GGUF F32 vs safetensors converted to Q8_0 (`GgufModelTest`) | qwen2, qwen3, qwen3moe: 0 (identical); llama: 2.9e-7 |

The GGUF Q8_0 file and the engine's Q8_0 differ only in the f16 versus f32 block scale (relative 2^-11), which did
not change a single token in these runs.

**Load time and memory** (Q4_K_M GGUF, 397 MB, entirely in place now that Q6_K is native, versus the converted `q4`
cache, 452 MB; Metal; the machine was shared with other jobs, load average 12-30, so read the ratios):

| | map + tokenizer | wall to first token | peak RSS |
|---|---|---|---|
| GGUF Q4_K_M | 0.3-0.5 s (0.3 s of it the tokenizer; the weights take 30 ms) | 2.0-2.1 s | ~195 MB |
| converted q4 cache | 0.04 s | 1.9 s | ~300 MB |
| *conversion from safetensors, once* | *3 s, peak RSS 1.6 GB, 452 MB written* | | |

Before `QType.Q6_K` existed the 29 Q6_K tensors of this file were requantized to Q8_0 at load: 1.4 s to map, 510 MB
RSS. The output is token for token the same (md5 `1a5dc189...` for the haiku prompt, Metal and CPU). Q4_1, Q5_0,
Q5_1, Q3_K, IQ4_NL, IQ4_XS, MXFP4, F16 and BF16 are now read in place too, except where a backend's default
requantizes them (§16.4.1); `GGUF weights:` reports what was converted, and §16.9.1 has their numbers.

### 16.9.1 The newer block types in place (Apple M1 Pro, Qwen3-0.6B requantized with `llama-quantize`)

Every file below is the Qwen3-0.6B `Q8_0` GGUF converted with `llama-quantize --allow-requantize --pure <TYPE>` (all matrices and the
tied embedding in that type; Q3_K_M is the usual mix), so the weights are not the original model's, but the file is exactly what
llama.cpp reads. Speeds are `./jllm bench` medians (413-token prompt, 5 rounds Metal / 3 rounds CPU, speculation off) under the heavy lock,
**native** = read in place, **Q8_0** = the same file with `-Djllm.gguf.requantize` naming its type (the behaviour before these types existed:
dequantized at load, requantized to Q8_0 in memory). Other jobs were running on the machine (load average 9 – 15), so differences of a few
percent are noise.

| type | file = in place | requantized (RAM) | Metal prefill tok/s native / Q8_0 | Metal decode tok/s native / Q8_0 | CPU decode tok/s native / Q8_0 | CPU default |
|---|---|---|---|---|---|---|
| Q4_1 | 379 MB | 670 MB | 1426 / 1489 | 101.0 / 99.9 | 83.7 / 92.2 (-9%) | in place |
| Q5_0 | 416 MB | 670 MB | 1558 / 1466 | 103.3 / 103.4 | 85.6 / 91.8 (-7%) | in place |
| Q5_1 | 453 MB | 670 MB | 1375 / 1469 | 105.1 / 102.8 | 66.2 / 91.9 (-28%) | **Q8_0** |
| Q3_K | 262 MB | 670 MB | 1508 / 1469 | 101.5 / 105.1 | 58.0 / 92.4 (-37%) | **Q8_0** |
| Q3_K_M (mix) | 347 MB | 297 MB | 1400 / 1371 | 90.7 / 89.5 | 69.7 / 90.1 (-23%) | Q3_K tensors Q8_0 |
| IQ4_NL | 341 MB | 670 MB | 1356 / 1492 | 92.0 / 107.6 | 100.1 / 95.1 (+5%) | in place |
| IQ4_XS | 323 MB | 670 MB | 1358 / 1493 | 93.0 / 107.8 | 115.3 / 93.6 (+23%) | in place |
| MXFP4 | 323 MB | 670 MB | 1389 / 1467 | 92.0 / 102.8 | 120.9 / 92.7 (+30%) | in place |
| F16 | 1198 MB | 670 MB | 1428 / 1468 | 81.2 / 102.9 | 22.4 / 91.7 (-76%) | **Q8_0** (lossy) |

What the table says:

* **Memory**: the in-place file is 1.5 – 2.6x smaller than the Q8_0 copy for every type but F16 (twice as big). This is what lets a
  27B – 30B GGUF fit in 32 GB, and why Metal keeps everything in place. F16 / BF16 in place is exact but reads 16 bits per weight.
* **Metal**: prefill is the same within noise; decode is the same for Q4_1 / Q5_x / Q3_K, 10 – 14% lower for IQ4_NL / IQ4_XS / MXFP4 (a
  table lookup per weight; on this 0.6B model decode is not bandwidth-bound, a larger model would profit from the smaller weights) and
  21% lower for F16. The raw kernel (`GgufMetalBench`, an 8192 x 4096 matrix, single token, µs): Q4_0 290, Q8_0 310, Q4_1 287, Q5_0 304,
  Q5_1 302, Q3_K 372, IQ4_NL 348, IQ4_XS 338, MXFP4 343, F16 433.
* **CPU** (int8 activations, native NEON dot-product kernels): IQ4_XS and MXFP4 are faster in place (4.25 bits per weight against
  Q8_0's 8.5), Q4_1 / Q5_0 are 7 – 9% slower, Q5_1 (-28%), Q3_K (-37%) and F16 (-76%) are slower by more than the 25% the default
  tolerates, so **the CPU backend requantizes Q5_1, Q3_K, F16 and BF16 by default** and `-Djllm.gguf.requantize=none` keeps them in
  place. `-Djllm.gguf.requantize=Q4_1,Q5_0,Q5_1` is the opt-out for the legacy 4/5-bit formats on any backend. With
  `-Djllm.cpu.native=false` (Java kernels only, e.g. x86-64): Q4_1 29 tok/s, IQ4_NL 68 (Q8_0: 26 – 29).
* **The Metal prefill dip that was found on the way**: chunks of 8 – 31 tokens (the tail of every prompt) ran the tiled `matmul_pf`
  kernels, 3 – 4x slower than the matrix-unit kernels for the bit-field formats Q5_0 and Q3_K *and for the existing Q4_K / Q5_K* (Q4_K,
  1024 x 3072, 8 / 16 / 24 tokens: 394 / 815 / 1081 µs against 240 / 243 / 248 µs). `MetalKernels.tiledIsSlow` sends these four
  formats to the matrix units from 8 tokens (`jllm.metal.prefillMinTokens`). End to end, **`Qwen3-0.6B-Q4_K_M.gguf` on Metal, 413-token
  prompt: prefill 750 tok/s before (main) → 1414 – 1443 tok/s, decode 90 tok/s unchanged, and the greedy text is byte-identical** (md5 of
  two raw 64-token completions and of the chat haiku, 3 alternating runs each): `Qwen3-0.6B-Q4_K_M` is mostly Q4_K, so the same
  holds for any `Q4_K_M` file, including the Q5_K / Q6_K tensors of the Q4_0 file of Qwen3.8 (not run here: its architecture lands in another branch).

**Exactness against llama.cpp.** Dequantization is bit-identical for all nine types (`GgufKernelsCheck`: 2048 – 12 288 weights per type
from random blocks, gguf-py and `ggml-quants.c` compiled and called through ctypes: zero differing floats; the Metal kernels are checked
against the same numpy decoders to 1e-5 – 1e-7 relative). Greedy generation of 32 tokens on two prompts ("The capital of France is",
"def fibonacci(n):") against `llama-completion` (Homebrew 0.7.0, Metal): jllm on **Metal and on the CPU float path produced llama.cpp's text for
every file** (Q4_1, Q5_0, Q5_1, Q3_K, Q3_K_M, IQ4_NL, IQ4_XS, an IQ4_XS mix, MXFP4, F16, BF16: 22 of 22 runs, the last character apart: llama.cpp prints a
trailing newline). With int8 activations on the CPU the text agrees on 20 of 22 (Q4_1 prompt 1 and the IQ4_XS mix prompt 1 diverge
after 6 – 7 characters at a near tie, the documented effect of rounding the activation). Reading the same files with the old
requantize path changes the text in 3 of 22 runs on each backend (Q4_1 prompt 1 at once, Q3_K and the IQ4_XS mix or IQ4_XS later): the in-place kernels are
closer to llama.cpp than Q8_0 requantization is. (Two first Metal runs died with `Impacting Interactivity`, the GPU watchdog, while other jobs were using the GPU; run
alone they produced llama.cpp's text.)

---

## 16.10 Limits and what is next

* **Architectures**: `llama`, `mistral3`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35`, `qwen3next`, `qwen35moe` (16.13), `gemma2`,
  `gemma3`, `phi3`, `chatglm`, `glm4`, `mamba` and `deepseek2` (16.15). Other architectures are refused, with the reason where
  it is known (16.15).
* **Formats**: IQ1\*, IQ2\*, IQ3\* and the ternary TQ\* types are refused (the 16-entry-codebook IQ4 types are in place; the
  others use 256-entry grids and are not done); Q2_K, Q3_K, F16 and BF16 have no int8-activation CPU kernel; the x86-64
  native library has no kernels for the formats added in 16.4.1 (the Java int8 kernels serve them there).
* **RoPE**: `linear`, `yarn` (16.2.1), Llama 3 and a longrope-style table are mapped; YaRN's `truncate = false`
  variant and `dynamic` scaling are not.
* **MoE** layouts are tested on tiny models only (no MoE GGUF fits the download budget used here); shared experts
  of DeepSeek are mapped (16.15); the gated shared expert of Qwen2-MoE is not.
* **Tokenizer**: the pre-tokenizers that are not a list of regexes (`kimi-k2`, `afmoe`, `tiny_aya`, `superbpe`, `chameleon`,
  the SPM-style BPE `gemma4`, `sarvam-moe`, `mmbert`, `whitespace`) and `exaone-moe` need the original repository's
  `tokenizer.json`; `tokenizer.ggml.add_eos_token` and the WordPiece (`bert`) vocabularies are not applied (16.12).

---

## 16.11 Running the checks

`mvn test` runs the checks that need no download: `GgufFileTest` (reader, malformed files), `GgufKernelsCheck`
(block formats against the `gguf.quants` fixtures, SIMD against scalar, int8 activations), `GgufMetalCheck` (every
Metal dispatch family; `GgufMetalBench` times them), `NativeCpuCheck` (the native int8 kernels against the Java ones), `GgufModelTest` (tiny models of four families written as safetensors and as GGUF, the llama
permutation, metadata mapping), `GgufFamiliesCheck` (24 tiny models of the other families, 16.15), `GgufQwenNextTest` (tiny Qwen3.5 and Qwen3-Next hybrids, 16.13), `GgufBackendsTest` (CPU == Metal on random GGUF models), `GgufHubTest` (file
selection, settings) and `GgufTokenizerTest` (three of llama.cpp's vocabulary fixtures and the stored llama.cpp ids of 29 generated
vocabularies, 16.12). With the Qwen3-0.6B files downloaded by
`./jllm download --model unsloth/Qwen3-0.6B-GGUF --gguf-quant Q8_0` (and `Q4_K_M`, `Q4_0`), run by hand:

```sh
J="$JAVA_HOME/bin/java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -cp target/classes:target/test-classes org.robomiri.jllm"
G=models/unsloth_Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf
$J.GgufVsSafetensors $G models/Qwen_Qwen3-0.6B                       # every tensor against the bf16 original
python3 -m venv v && v/bin/pip install gguf numpy                     # byte parity with gguf.quants on real blocks
mkdir -p /tmp/parity && v/bin/python src/test/python/gguf_parity.py $G /tmp/parity 256
$J.GgufRealParity $G /tmp/parity
$J.metal.GgufMetalReal $G                                             # Metal matmul of every tensor vs its dequantization
$J.GgufTokenizerTest --gguf $G --hf models/Qwen_Qwen3-0.6B --fixture src/test/resources/tokenizers/qwen3.json
$J.GgufTokenizerTest --vocab DIR       # llama.cpp's models/ggml-vocab-*.gguf with their .inp / .out files
scripts/gguf-tokenizer-check.sh        # generated vocabularies for every pre value against llama-tokenize (16.12)
```

---

## 16.12 Tokenizer pre-tokenizers: which values are supported and how they were checked

🧸 **Cut first, glue second.** A tokenizer does not look at a sentence all at once. First it cuts the text into little
words with a cutting rule, then it glues the letters of each word into pieces it knows. Every family of models cuts a
little differently: Llama 3 chops a long number into groups of three digits, Qwen takes the digits one at a time,
DeepSeek also puts every Chinese phrase in a box of its own. The GGUF file only carries the *name* of the rule
(`tokenizer.ggml.pre`); the rule itself is written out, character by character, inside llama.cpp. If we cut even one
place differently, the model receives numbers it was never taught, and answers a little worse without telling anyone. So
we copied llama.cpp's rule for every name, and then we checked, with over a million strings, that llama.cpp
itself and jllm cut and glue them to exactly the same ids.

📐 **What is implemented** (`GgufPre`, `GgufTokenizer`, `SpmGreedy`; reference: llama.cpp b11669, `llama-vocab.cpp`).

* A pre value is a **list of regexes applied one after the other**, each cutting the pieces the previous one left (matches
  *and* the text between them are kept: a Hugging Face `Sequence` of `Split`/`Isolated`), then GPT-2 byte-level spelling,
  plus the flags `ignore_merges` (a piece that is a token is not merged) and the default of `add_bos_token`. The patterns
  are llama.cpp's strings, with four changes so that `java.util.regex` does what llama.cpp's C++ does: `\p{Lu}`... are
  "any letter" there; non-ASCII whitespace never matches a character class of a regex without `\p{..}` (it is
  replaced by U+000B first); `$` is the end of the piece; the look-ahead groups of GPT-4o / Tekken are one class.
* An absent key or `default` is llama.cpp's own default (4 regexes, digits in threes), **not** GPT-2's; an unknown value is refused.
* Control tokens are found in the text by type and by name (`</s>`, `<|im_end|>`, the FIM markers: control even if the
  file says normal), user-defined ones are found even in untrusted text (`encode(text, false)`), Phi-3 files
  (`general.name`) swallow the ASCII whitespace after a special token. A BOS is added as llama.cpp does: the key if present, else
  yes for SentencePiece, `llama-bpe` and `tekken`, no otherwise.
* SentencePiece (`model = llama`) is llama.cpp's greedy merge by score, not BPE merges (`SpmGreedy`).

| `tokenizer.ggml.pre` | cutting rules (llama.cpp `regex_exprs`) | flags | checked against |
|---|---|---|---|
| `llama-bpe`, `llama3`, `llama-v3`, `falcon3`, `falcon-h1`, `pixtral`, `midm-2.0`, `lfm2`, `jina-v5-nano` | Llama 3: `'s..`, `[^\r\n\p{L}\p{N}]?\p{L}+`, `\p{N}{1,3}`, ` ?[^\s\p{L}\p{N}]+[\r\n]*`, whitespace | ignore_merges, BOS | F llama-bpe 47/47 and 14 871/14 871; S 14 871/14 871 each |
| `dbrx`, `smaug-bpe`, `chatglm-bpe`; `glm4`, `glm5` (+ ignore_merges) | the same regex | no BOS default | S 14 871/14 871 each |
| `qwen2`, `deepseek-r1-qwen`, `kormo`, `f2llmv2`, `megrez`, `stablelm2`, `hunyuan`, `solar-open`, `grok-2` | Llama 3's with `\p{N}` (one digit) | | F qwen2 47/47 and 14 871/14 871 (+ Qwen3-0.6B Q8_0 and 2 795 doc paragraphs); S 14 871/14 871 each |
| `qwen35` | the same with `[\p{L}\p{M}]+` | | F qwen35 51/51 and 14 871/14 871; S |
| `gpt-2`, `phi-2`, `olmo`, `mpt`, `jais`, `trillion`, `granite-docling`, `exaone4`, `gigachat`, `a.x-4.0`, `mellum`, `modern-bert`, `roberta-bpe`, `jina-*` | GPT-2: `'s..`, ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`, `\s+(?!\S)` | | F gpt-2, mpt 47/47 and 14 871/14 871; S each |
| `starcoder`, `refact`, `command-r`, `smollm`, `codeshell`, `exaone`, `minerva-7b`, `mellum2` | `\p{N}`, then GPT-2's | | F starcoder, refact, command-r 47/47 and 14 871/14 871; S each |
| `default`, key absent (GPT-NeoX, Mamba) | `[\p{P}$+<=>^~\|]+`, GPT-2's, `\p{N}+`, `[0-9]{3}` | | F gpt-neox 14 871/14 871; S |
| `falcon` | `[\p{P}$+<=>^~\|`]+`, GPT-2's, `[0-9]{3}` | | F falcon 47/47 and 14 871/14 871; S |
| `deepseek-llm` / `deepseek-coder` | 6 / 5 rules: `[\r\n]`, letters / `\s?\p{L}+`, punctuation, `\s+$`, CJK ranges, `\p{N}+` / `\p{N}` | | F 47/47 and 14 871/14 871 each; S |
| `deepseek-v3`, `hunyuan-dense`, `hy_v4`, `joyai-llm`, `spark2_5` | `\p{N}{1,3}`, CJK + kana, symbol-then-letters rule | | S 14 871/14 871 each |
| `tekken`; `gpt-4o`, `llama4`, `kanana2`, `talkie`, `minimax-m2`; `granite-embed-multi-97m` | case-aware letters (look-ahead rewritten as classes), `\p{N}` / `\p{N}{1,3}` | tekken: ignore_merges, BOS | S 14 871/14 871 each |
| `jais-2`, `youtu`, `poro-chat`, `bloom`, `gpt3-finnish`, `viking`, `bailingmoe`, `bailingmoe2`, `llada-moe`, `seed-coder`, `ufakzeka`, `laguna`, `minicpm5`, `k2-horizon` | as in `llama-vocab.cpp` | per the table in `GgufPre` | S 14 871/14 871 each |
| SentencePiece (`model = llama`): Llama 2, Mistral 7B, Gemma 2/3, Phi-3 | greedy merge by score, `<0xXX>` bytes, `▁` prefix | Gemma: no prefix; Phi-3: rstrip | F llama-spm, phi-3 47/47 and 14 871/14 871; S 4 kinds (distinct scores, ties, Gemma-like, Phi-3-like) 14 871/14 871 each |

**How it was checked.** `scripts/gguf-tokenizer-check.sh` writes a vocabulary-only GGUF per pre value with all 65 536
merges of two byte symbols in a pseudo-random order (so a cut between two pieces shows as a pair that was not merged),
a few merge chains, tokens without merges (to see `ignore_merges`) and control / user-defined tokens, and gives the same
14 871 strings to `llama-tokenize` and to `GgufModel.tokenizer`: 191 hand-written (contractions in both cases, digit runs, CJK, accents,
emoji, every kind of whitespace, code, special tokens in the text), 2 000 random over a Unicode alphabet and a sweep over
code points (12 680 strings); for byte-level BPE also with `--no-parse-special` against `encode(text, false)`, and the BOS rules.
F = llama.cpp's own `ggml-vocab-*.gguf`. S = generated vocabulary. Three other merge orders (seeds 1-3) give the same results. `GgufTokenizerTest` keeps the ids of 19 691 strings
on 29 vocabularies (`synthetic-llama-cpp.txt.gz`, 68 KB), so `mvn test` needs no llama.cpp.

**Differences that remain** (all deliberate or unreachable in real text):
* Characters assigned after llama.cpp's Unicode tables (Unicode 14-17; 8 443 code points, e.g. Egyptian Hieroglyphs
  Extended-A) are letters / marks here and "unassigned" there.
* llama.cpp does not normalize to NFC (the previous qwen2 special case is gone) and makes `</s>` a control token, so
  these two differ from the Hugging Face tokenizer.json of the same model (Qwen3: 5 of 150 fixture strings).
* Two special tokens that overlap (`<|ab` and `b|c>x` in `<|ab|c>x`): llama.cpp takes the longer first, this scanner the
  leftmost. SentencePiece with special tokens not parsed: control tokens stay text here, llama.cpp merges them back.
* `add_eos_token`, WordPiece (`bert`) and the pre values that are not regex lists (`kimi-k2`, `afmoe`, `tiny_aya`,
  `superbpe`, `chameleon`, `gemma4`, `sarvam-moe`, `mmbert`, `whitespace`; `exaone-moe` would overflow the regex stack) are refused.

---

## 16.13 Qwen3.5 and Qwen3-Next as GGUF files

Code: `gguf/GgufQwenNext.java` (the tensor mapping), `gguf/GgufArchitecture.java` (`hybrid`), `model/Qwen3NextArch.java`
(`DeltaNet.tiled`), `model/Qwen3NextLayer.java` (`DeltaNet.z`, `.ba`), `cpu/Qwen3NextCpuBackend.java`,
`metal/Qwen3NextMetalBackend.java`, `gdn_rule` in `kernels.metal`, `tokenizer/ChatTemplate.java` (`QWEN35`),
`src/test/.../GgufQwenNextTest`, `GgufTinyWrite`, `GgufGreedy`, `GgufProfile`, `scripts/qwen35-llamacpp-check.sh`

🧸 **The simple story.** Qwen3.5 is built like Qwen3-Next ([14.14](14-model-families.md#1414-qwen3-next-hybrid-gated-deltanet--gated-attention--sparse-moe)):
three layers out of four keep a *whiteboard* of fixed size (the Gated DeltaNet) and every fourth keeps a real *diary*
(attention); every layer has an ordinary feed-forward block. The model is the same, only the **file** is new, and
llama.cpp stores two things in an order jllm's own cache does not.

First, the whiteboard has several *helpers* per label ("head 0's three helpers, then head 1's three helpers..."). The
converter of llama.cpp lays them out the other way round ("every head's first helper, then every head's second
helper..."), because that is how its tensor library repeats things. Putting a quantized matrix's columns in another
order means unpacking and packing it again, which loses a little, so jllm does not: it keeps the file's order and just
looks up *which label a helper belongs to* the other way (one flag, `tiled`). Second, the diary layer's *question* and
its *door* (an output gate that can shut part of the answer) sit interleaved in one matrix; jllm splits them when it
loads, copying only those compressed rows.

📐 **What the file holds** (`qwen35`; `qwen3next` and `qwen35moe` differ where said). Notation: `dim` = `embedding_length`
(5120), `nk` = `ssm.group_count` (16), `nv` = `ssm.time_step_rank` (48), `dk` = `ssm.state_size` (128, also the value
head width), `r = nv / nk` (3), `H`, `hd` = `attention.head_count`, `key_length` (24, 256).

| GGUF tensor (`blk.N.`) | shape (ggml order) | engine name | handling |
|---|---|---|---|
| `attn_norm`, `post_attention_norm` | `[dim]` f32 | `input_layernorm`, `post_attention_layernorm` | the converter already added the `1 +` of the zero-centred norms; the engine's cache holds them folded the same way |
| `attn_q` | `[dim, 2·H·hd]` | `self_attn.q_proj` + `q_gate` | per head `hd` query rows then `hd` gate rows: **gathered** into two matrices (a copy of the compressed rows, page aligned for Metal) |
| `attn_k`, `attn_v`, `attn_output`, `attn_q_norm`, `attn_k_norm` | | as Qwen3 | in place / f32 |
| `attn_qkv` | `[dim, 2·nk·dk + nv·dk]` | `linear_attn.in_proj` | rows `[q (nk·dk) \| k (nk·dk) \| v (nv·dk)]`, **in place** |
| `attn_gate` | `[dim, nv·dk]` | `linear_attn.in_proj_z` | the gate `z` of the gated norm, in place |
| `ssm_beta`, `ssm_alpha` | `[dim, nv]` **f32** | `linear_attn.in_proj_ba` | rows `[b (nv) \| a (nv)]` as one small Q8_0 matrix (`qwen3next`: the fused `ssm_ba`, rows `[b a]` per key head, regrouped) |
| `ssm_conv1d` | `[kernel, conv_dim]` f32 | `conv1d.weight` | channel-major `[conv_dim, kernel]`, as the engine wants |
| `ssm_a` | `[nv]` | `A` | already `-exp(A_log)` |
| `ssm_dt.bias`, `ssm_norm` | `[nv]`, `[dk]` | `dt_bias`, `norm.weight` | `ssm_norm` is *not* zero-centred |
| `ssm_out` | `[nv·dk, dim]` | `out_proj` | in place (Q5_K in the unsloth Q4_0 file) |
| `ffn_gate`, `ffn_up`, `ffn_down` | | `mlp.gate_up`, `mlp.down_proj` | a dense SwiGLU in every layer (`qwen35`); `qwen3next`, `qwen35moe`: `ffn_gate_inp` + `ffn_*_exps` + the shared expert `ffn_*_shexp` and `ffn_gate_inp_shexp` |
| `nextn.*`, and the attention / FFN tensors of block `block_count - 1` | | | the multi-token-prediction layer, **not** a layer of the model (`block_count - nextn_predict_layers` layers): 16.14 |

The input of a DeltaNet layer is therefore *not* one matrix as in jllm's cache (`[q k v | z | b a]` in one matmul): the
file's `attn_qkv` and `attn_gate` are read in place (no copy of 2.3 GB for the 27B), `b | a` is f32 in the file and tiny, and
three products go to three buffers. `DeltaNet.z` and `.ba` are null for a model loaded from the cache, which keeps the
fused layout.

### 16.13.1 Value heads: grouped or tiled

Hugging Face's Qwen3-Next gives every key head `r` value heads: value head `j` reads key head `⌊j / r⌋` (**grouped**).
llama.cpp (`convert_hf_to_gguf.py`, `_LinearAttentionVReorderBase`) writes Qwen3.5 with the value heads reordered so that
position `p = i·nk + g` holds the old head `g·r + i` (**tiled**: head `p` reads key head `p mod nk`), which lets ggml
broadcast q and k with one `repeat`. The reorder touches everything indexed by a value head: the `v` rows of `attn_qkv`,
`attn_gate`, `ssm_beta` / `ssm_alpha`, the `v` channels of `ssm_conv1d`, `ssm_a`, `ssm_dt` and the **columns** of
`ssm_out`. The columns of a Q5_K matrix come in blocks of 256 = two heads, so they cannot be moved without
requantizing. jllm moves nothing: `Qwen3NextArch.DeltaNet.tiled` (set for `qwen35` and `qwen35moe`) makes the key head of
value head `h` `h mod nk` instead of `h / r` (`keyHeadOf`) in the CPU recurrence and in `gdn_rule` (parameter `tiled`;
checked against numpy in `kernel_tests.py`). `qwen3next` files are not reordered, so grouped.

### 16.13.2 RoPE, tokenizer, chat template

Only the first `rope.dimension_count` = 64 of the 256 dims of a head rotate (`partial_rotary_factor` 0.25) with
`rope.freq_base` = 1e7. The file's `rope.dimension_sections = [11, 11, 10, 0]` are Qwen's multimodal RoPE (the temporal,
height and width positions interleaved over the frequencies): for text all three position streams are the same
number, so every dimension gets the angle of the ordinary rotate-half RoPE and nothing else is needed (the vision
tower is not loaded).

`tokenizer.ggml.pre = qwen35` is one of the pre-tokenizers of [16.12](#1612-tokenizer-pre-tokenizers-which-values-are-supported-and-how-they-were-checked)
(Qwen2's regex with `[\p{L}\p{M}]+`, no normalization, as llama.cpp): the ids of the 27B file equal `llama-tokenize`'s on
four test texts (English, code, accents / CJK / emoji / a combining mark / digits / tabs, and a chat prompt with special tokens).

The chat template (`ChatTemplate.QWEN35`, model type `qwen3_5`, selected by `Qwen3NextArch` for a tiled model) follows the
Jinja template stored in the file, without vision and tool calls: leading `system` / `developer` messages are merged
into one system turn that starts with the line `Reasoning effort is set to xhigh. ...` while thinking is on; every
assistant turn carries its `<think>` block; the generation prompt is `<|im_start|>assistant\n<think>\n`, or, with
thinking off, `<think>\n\n</think>\n\n`. Stop tokens `<|im_end|>` and `<|endoftext|>`. It renders the same prompts as
llama.cpp's Jinja engine does with the template of the 27B file on nine conversations (a system message, merged
system / developer messages, whitespace to trim, assistants with and without reasoning, consecutive tool results:
`GgufQwenNextTest`, `src/test/resources/gguf/qwen35-chat-*`).

### 16.13.3 Memory on this machine

The unsloth `Qwen3.8-27B-Q4_0.gguf` (16 056 MB: Q4_0 and Q5_K / Q6_K / Q8_0 for a few tensors, `ffn_down` of 8 layers Q4_1)
loads in 1.4 s because nothing is read: **14 893 MB are used in place**; 1 142 MB are converted or copied at load (the 96
f32 `b` / `a` matrices to Q8_0, the q / q-gate row copies of the 16 attention layers and of the head's block, and the 8 Q4_1
`ffn_down` tensors requantized to Q8_0; measured before Q4_1 was read in place (16.4.1), which now leaves them as they are).
Metal's recommended working set on this machine
is 26 800 MB; the model plus a 1024-token f32 KV cache and the scratch buffers fit with room to spare.

### 16.13.4 Verification

**The 27B file against llama.cpp 0.7.0 (build 11669, Metal, `llama-server` with `n_probs`)**, greedy, same file, same
token ids (`GgufGreedy`, f16 KV cache, Metal):

| prompt (32 greedy tokens) | greedy ids | log-probabilities of the first position, llama.cpp vs jllm |
|---|---|---|
| `The capital of France is` (raw) | 32 of 32 identical | ` Paris` −0.4861 / −0.4871, ` not` −3.2200 / −3.2204, ` a` −3.4227 / −3.4235, ` ______` −3.8898 / −3.8923, ` known` −3.9603 / −3.9594 |
| a Python function head (raw, 27 tokens) | 32 of 32 | `    ` −0.0003 / −0.0004, `else` −9.2255 / −9.2278 |
| chat, thinking off | 8 of 8 (ends at `<|im_end|>`) | `The` −0.1728 / −0.1729, `Paris` −1.8759 / −1.8759 |
| chat, thinking on (the system turn with the effort line) | 32 of 32, through `</think>` | `We` −0.3132 / −0.3261, `The` −1.6865 / −1.6497, `User` −2.6181 / −2.5889 |

The differences of 1e-3 to 4e-2 in log-probability come from the activations: llama.cpp quantizes them to Q8_0 for every
matrix product, jllm's Metal kernels use floats. **Tiny hybrids against llama.cpp** (`scripts/qwen35-llamacpp-check.sh`,
no download): `GgufTinyWrite` writes random Qwen3.5, Qwen3-Next and Qwen3.5-MoE models the way the converter does, with
F32 tensors holding Q8_0-representable values, so both sides compute with the same weights and exact activations: the
top-5 log-probabilities agree to 1e-3 (mostly 1e-4) and all 36 greedy ids per model are identical (CPU; Qwen3.5 also on Metal). In
`mvn test`, `GgufQwenNextTest` loads the same models as a Hugging Face checkpoint through the converter and as GGUF
(F32: logits equal to 3e-6; Q8_0 blocks in place against the same values requantized: 6e-6; CPU against Metal: 8e-6).

### 16.13.5 Speed (Apple M1 Pro, 32 GB, under the heavy lock)

`llama-bench -m Qwen3.8-27B-Q4_0.gguf -p 128,512 -n 32 -r 2 -ngl 99` (llama.cpp 0.7.0, build 11669, Metal + BLAS) and
jllm on the same file (`GgufProfile` and `./jllm run --temp 0`, Metal, f32 KV cache, `-Xmx6g`):

| | llama.cpp | jllm |
|---|---|---|
| prefill, 128 / 512 tokens | 77.8 / 78.1 tok/s | 40.6 / 40.5 tok/s |
| prefill, 32 tokens | | 38.0 tok/s |
| decode | 9.83 tok/s | 4.3 tok/s (231.6 ms per step, `GgufProfile`; 4.2-4.3 in `./jllm run`) |

So decode runs at 44 % and prefill at 52 % of llama.cpp's speed, the same ~2x gap as on the small dense models
([README](../README.md#beyond-a-plain-port-of-llamacpp)); the 16 GB file reads at about 70 GB/s against llama.cpp's 150. The
Gated DeltaNet recurrence runs token by token in prefill on Metal (no chunked kernel there, 14.14.8), and the 8 Q4_1
`ffn_down` tensors were requantized to Q8_0 in these runs (Q4_1 is read in place since 16.4.1). On the CPU the 27B is not
usable (it was only run on tiny models).


---

## 16.14 Multi-token prediction: the model as its own draft model

Code: `model/Qwen3NextMtp.java` (the weights), `model/MtpFeeder.java` (bookkeeping), the `Backend` methods `hasMtp`,
`enableMtp`, `mtpReady`, `mtpDraft`, `checkpointState`, `rollbackState`, `cpu/Qwen3NextCpuBackend.java` and
`metal/Qwen3NextMetalBackend.java` (`mtpRows`), `engine/Speculator.java` (`Mode.MTP`), `gdn_conv` / `gdn_rule` (the snapshot
parameter), `src/test/.../GgufQwenNextTest` (`mtp`)

🧸 Speculative decoding ([10.9](10-engine-and-server.md#109-speculative-decoding)) needs someone to guess the next word
so the big model can check the guess and the word after it in the same pass. Usually that someone is a small second
model. A Qwen3.5 file comes with its own tiny guesser: one extra layer, the **multi-token-prediction** layer (`nextn`),
trained to look at what the big model was thinking and the word it just said, and guess the word *after* that. It costs
about one layer per word, a sixtieth of the big model, and needs no second download. The only difficulty is the
whiteboard layers: the big model's whiteboards are *written on* by every word it reads, so after a wrong guess they
must be put back as they were after the last right word. jllm keeps a second copy of the whiteboards for that.

📐 **The head** (`blk.<layers>.nextn.*` plus the layer's own attention and MLP tensors; llama.cpp `qwen35.cpp`,
`graph_mtp`). With `h_t` the main model's hidden state after its final `output_norm` at position `t` and `x_{t+1}` the
token chosen after it:

```
x   = eh_proj · [ rmsnorm(embedding(x_{t+1}), enorm) ; rmsnorm(h_t, hnorm) ]      [2·dim] -> [dim]
x  += gated_attention(rmsnorm(x, attn_norm))     its own KV cache, RoPE at position t, q-gate as a main layer
x  += swiglu(rmsnorm(x, post_attention_norm))
logits = output_head · rmsnorm(x, shared_head_norm)                                the guess for x_{t+2}
```

`shared_head_norm`, `shared_head_head` and `embed_tokens` are the main model's own when the file has none (the unsloth
file has none of the last two). Row `t` of the head needs `h_t` and `x_{t+1}`, so a pass that feeds
`x_p .. x_{p+n-1}` completes the rows `p-1 .. p+n-2`; the last position's row waits for the next sampled token, and
`mtpDraft(token)` runs exactly that row early and returns the guess (`MtpFeeder`).

**One step with `engine.speculative=mtp`** (`Speculator`, one draft per step, `m = 1`):

```
tok  = sample(logits)                           # the target's own choice, to be fed at position pos
d1   = argmax(head(h_{pos-1}, tok))             # the head's guess for the token after tok
checkpoint                                      # arm: the pass saves the recurrent state after tok
rows = forwardAll([tok, d1], pos)               # ONE pass; the head's rows for pos-1 and pos ride along
s0   = verify(rows[0], d1)                      # greedy: d1 is kept iff it is the argmax
if s0 == d1: emit d1, then sample(rows[1])      # two tokens for one pass
else:        rollback(1); emit s0               # the state after tok only
```

The KV cache needs no undo (the next pass overwrites position `pos+1`), but the DeltaNet's recurrent state and conv state
have run through `d1`. On Metal `gdn_conv` and `gdn_rule` take a snapshot parameter and write their state as it is after
token 0 of the pass into a second set of buffers while they compute token 1 (48 layers x 3.1 MB); a rejection swaps the
two sets (no replay, no copy). The CPU backend copies the state before the pass and replays the kept token on a
rejection. The head's own KV rows beyond the kept position are recomputed by the next pass. Greedy output is the plain
greedy output by construction (`GgufQwenNextTest`: an oracle head that is right two times out of three, so both paths run,
and the real head, on CPU and Metal: identical ids). A model with a head uses this mode by default
(`engine.speculative` left at its default); `--engine.speculative off` turns it off.

**Checked against llama.cpp.** The tiny qwen35 file of `scripts/qwen35-llamacpp-check.sh` has a head and an output
layer focused on four tokens, so guesses are often right: at each of 60 positions, whether llama.cpp's own head
(`llama-server --spec-type draft-mtp --spec-draft-n-max 1`) guesses the next token is the same bit as jllm's, 60 of 60
(16 hits). On the 27B file, jllm's head guesses the next token **111 times out of 124** (90 %) in greedy runs of four
prompts (`GgufGreedy --mtp`: 31/31 and 28/31 on two raw prompts, 28/31 on a chat answer, 26/31 on a thinking trace).

**Measured** (Apple M1 Pro, 32 GB, Qwen3.8-27B Q4_0, `./jllm run --temp 0 --ctx 1024`, Metal, under the heavy lock;
the output text is identical with and without):

| prompt | `engine.speculative off` | `mtp` | accepted |
|---|---|---|---|
| one-sentence answer (7 tokens) | 4.2 tok/s | 4.6 tok/s | 4 of 4 |
| a Python function with a docstring (82 tokens) | 4.3 tok/s | 5.4 tok/s (**+26 %**) | 41 of 41 |

**Where the time goes** (`GgufProfile`, median of 15, same machine): a plain step 231.6 ms (234.7 ms with the head on: one
head row per step); the two-token verification pass 347 ms (353 ms with the head's two rows) -- **1.5 plain steps, not 1**:
the multi-token matmul kernels (`matmul`, `TOK_TILE` tokens per simdgroup) dequantize each Q4_0 block's weights once but
spend the dot products per token, so a second token adds half of the compute on a kernel that is not bandwidth-bound
here (a reading of the kernel, not a measurement of it); the draft (one head row and the logits through the Q6_K output head, 1.5 GB of weights) 20 ms. With acceptance `a`
a cycle takes `353 + 20` ms for `1 + a` tokens against 232 ms per token: break-even at `a = 0.61`, `+18 %` at the 90 %
measured on the 27B (`1.9 / 373 ms = 5.1 tok/s`), `+26 %` at 100 % (the table). The lever is the cost of a two-token pass:
the head is cheap, the verification is not yet.

---

## 16.15 Gemma 2/3, Phi-3, GLM, Mamba, DeepSeek-V2/V3 and Mistral 3 as GGUF files

Code: `gguf/GgufFamilies.java` (the metadata of each family), `gguf/GgufArchitecture.java` (the `Family` enum),
`gguf/GgufWeights.java` (names, fused tensors), `model/AttnWeights.java` (`Mla`: split `kv_b`); tests
`src/test/java/org/robomiri/jllm/HfToGguf.java`, `GgufFamiliesCheck.java`, `GgufArchSmoke.java`,
`scripts/gguf-llamacpp-check.sh`.

🧸 A GGUF file is a box that llama.cpp packed. Until now we knew how to unpack four kinds of box (Llama, Qwen2, Qwen3,
Qwen3-MoE). Every other family we can already run from a Hugging Face folder has its own way of being packed: Gemma adds 1
to its norms and keeps two kinds of layers, Phi-3 glues two matrices into one, Mamba stores a different number, DeepSeek
cuts one matrix in two and turns half of it around. To be sure that we unpack each of them the way llama.cpp packed it, we
did the packing ourselves following llama.cpp's own recipe, on tiny random models, and then asked **llama.cpp** to unpack
the same box and run it: both programs give the same numbers to the last digits.

📐 **Status** (`general.architecture` → what jllm runs; the numbers are from `GgufFamiliesCheck` /
`scripts/gguf-llamacpp-check.sh` on tiny models of that family written as `convert_hf_to_gguf.py` writes them; the last
column is `GgufArchSmoke` on the real GPU: CPU and Metal generate the same 8 greedy tokens from the Q8_0 file):

| GGUF architecture | jllm family (`model_type`) | F32 GGUF vs safetensors | jllm vs llama.cpp (`llama-debug` logits, same file, same ids) | Metal == CPU |
|---|---|---|---|---|
| `llama` (Llama 2/3.x, Mistral 7B, TinyLlama, DeepSeek-R1 Llama distills; Llama-3 `rope_freqs`, linear/YaRN scaling, q/k/v/o biases, fused `attn_qkv`) | `llama` / `mistral` | 2.4e-7 | 2.0e-7 … 3.0e-7 | yes |
| `mistral3` (Mistral Small 3.x, Ministral without attention temperature, Devstral) | `mistral` | 2.2e-7 | 2.6e-7 | yes (as `llama`) |
| `qwen2`, `qwen3moe` (and `qwen3`) | as before | ≤ 3e-7 | 2.5e-7 … 3.1e-7 | yes |
| `gemma2` | `gemma2` | 0 | 6.7e-5 | yes |
| `gemma3` (also from the text part of the multimodal checkpoint) | `gemma3_text` | 0 | 1.8e-5 | yes |
| `phi3` (long, short and no LongRoPE factors) | `phi3` | ≤ 4e-7 | 2.0e-7 … 1.1e-6 | yes |
| `chatglm` (`GlmForCausalLM`: GLM-4-9B-chat-hf, GLM-Edge) | `glm` | 0 | 2.8e-7 | yes |
| `glm4` (GLM-4-0414) | `glm4` | 0 | 4.7e-7 | yes |
| `mamba` (Mamba v1) | `mamba` | ≤ 3e-7 | 7.5e-7, 1.1e-6 | yes |
| `deepseek2` (V2-Lite shape without query low-rank, V2 with, V2-236B router, V3/R1) | `deepseek_v2` / `deepseek_v3` | ≤ 9e-7 (unsplit `kv_b`) | 1.4e-6 … 1.9e-5 | yes |

(Relative logit differences: the largest absolute difference of the last prompt token's logits over the largest logit.
"jllm vs llama.cpp" uses the file holding the Q8_0-rounded weights as F32, so both engines multiply the same numbers; llama.cpp
runs on the CPU with an f32 cache and no flash attention, `llama-debug --save-logits`, build b11669.)

Refused with a message that names the reason (math the engine does not have): `granite*` (scales the embeddings, residual
branches and logits), `olmo2` (norms after the attention and the MLP, full-width q/k norm), `smollm3` (layers without RoPE),
`qwen2moe` (a sigmoid-gated shared expert in a decoder layer), the LayerNorm families (`stablelm`, `phi2`, `starcoder2`,
`command-r`, `cohere2`, `falcon`, `olmo`), Gemma 1, Mamba 2, and `internlm2` / `exaone*` (llama-like, but no chat template
here). `mistral3` with `attention.temperature_scale` (Ministral 3's position-dependent query scaling) is refused too.

### 16.15.1 What each converter writes, and how it is read

The tensor name table of 16.3 holds; these are the differences. All of them were taken from llama.cpp's converter
(`conversion/*.py`, `gguf-py/gguf/constants.py`, `tensor_mapping.py`) and loader (`src/models/*.cpp`) at tag b11669, then
checked by running that loader.

| family | what the file has | read as |
|---|---|---|
| Gemma 2/3 | every norm is stored as `w + 1`; `post_attention_norm` is the norm on the attention *output*, `ffn_norm` the one before the MLP, `post_ffw_norm` the one after it; no `output.weight` (tied); `rope.freq_base_swa`, `attention.sliding_window`, optional `attention.sliding_window_pattern` (an int, default 2 / 6, or a bool array); `attn_logit_softcapping` / `final_logit_softcapping`; **no `query_pre_attn_scalar`** | the cache's convention already (the `+1` is folded), so the floats are used as they are. The scale is `1/sqrt(head_dim)` as in llama.cpp (only the 27B shapes, 46 / 62 layers, use `1/sqrt(dim/heads)`); the embedding is multiplied by `sqrt(dim)` *in float* (llama.cpp's `ggml_scale`), with no rounding to bf16; a file without a window has no local layers |
| Phi-3 | `attn_qkv` (q, k, v rows in this order) and `ffn_up` (**gate rows, then up rows**, `2 × ff`); `rope_factors_long/short.weight` + `rope.scaling.attn_factor` for LongRoPE (the long table when the context exceeds `rope.scaling.original_context_length`, as llama.cpp does); `rope.dimension_count = partial_rotary_factor × head`; llama.cpp ignores the sliding window | q, k, v are row slices of `attn_qkv`, `mlp.gate_up` is `ffn_up` (on the GPU two views of its halves). The engine *applies* `attention.sliding_window` when the file has one (as transformers does; llama.cpp does not) |
| ChatGLM / GLM-4 | `ffn_up` fused as Phi-3's; q/k/v biases; `rope.dimension_count = 0.5 × head`, no `key_length` for ChatGLM (so `head_dim` must be `hidden / heads`; the converter reads `layernorm_epsilon`, default 1e-5, and ignores `rms_norm_eps`); GLM-4 adds `post_attention_norm` (= `post_self_attn_layernorm`) and `post_ffw_norm` and keeps `key_length`; RoPE type NORM on the *unpermuted* rows | the family's interleaved pairing; `attention_bias` is set from the presence of `attn_q.bias` |
| Mamba | `ssm_in`, `ssm_conv1d` (`[d_conv, d_inner]`: the HF `[d_inner, 1, d_conv]` squeezed), `ssm_x`, `ssm_dt` (+`.bias`), **`ssm_a` = `-exp(A_log)`** and `ssm_d` (no `.weight`), `ssm_out`; `head_count 0`, `feed_forward_length 0`, `context_length 2^20`; the output is dropped when equal to the embedding | `mixer.A` is `ssm_a` as it is (the cache stores `-exp(A_log)` too); `dt_proj` and the conv are F32/F16 (a `dt_rank` below 32 cannot be block-quantized, so the converter writes F16); Falcon-Mamba (`ssm.dt_b_c_rms`) is refused |
| DeepSeek-2 | MLA written as MQA: `head_count_kv 1`, `key_length = kv_lora + rope`, `value_length = kv_lora`, **`key_length_mla = nope + rope`, `value_length_mla = v`**; `attn_kv_a_mqa` (latent rows then the rotary key rows), `attn_kv_a_norm`, `attn_q_a` / `attn_q_a_norm` / `attn_q_b` or `attn_q`, **`attn_k_b` `[nope, kv_lora, heads]` (the key half of `kv_b`, transposed) and `attn_v_b` `[kv_lora, v, heads]`**; `leading_dense_block_count`, `expert_*`, `expert_gating_func` (1 softmax, 2 sigmoid), `expert_group_count` / `_used_count`, `expert_weights_scale` (absent = 1), `expert_weights_norm`, `exp_probs_b.bias` (V3's selection bias), shared experts as `ffn_*_shexp`; `rope.scaling.yarn_log_multiplier = 0.1 × mscale_all_dim` | the MLA of 14.11 with `attn_k_b` / `attn_v_b` in the layouts the absorbed kernels want (`AttnWeights.Mla.kB / vB`, no transposing copy of the key half); a file with only `key_length` / `value_length` and an unsplit `attn_kv_b` (before March 2025) is read as the checkpoint is. `mscale = mscale_all_dim` (what V2 and V3 ship; llama.cpp cannot store another), so the rotary amplitude is 1 and the softmax scale gets `yarn_get_mscale(factor, mscale_all_dim)²` |
| Mistral 3 | the llama architecture under another name | as `llama` |

Fused variants that llama.cpp's converter writes with `--fuse-qkv` / `--fuse-gate-up-exps` (`attn_qkv` with its fused bias,
`ffn_gate_up_exps`: per expert the gate rows, then the up rows, which is the engine's layout) are read for every family;
`ffn_gate_up_exps` of the GGUF block types is copied apart once at load (the GPU kernels read two interleaved views).

**Two judgement calls where the file is silent.** (1) DeepSeek's group routing: llama.cpp scores a group by the sum of its two
best experts (V3's rule) for every file with `expert_group_count > 1`; V2-236B's `group_limited_greedy` scores by the best. The
engine uses V3's rule when the file has the `exp_probs_b` bias and V2's rule when it has not. (2) V2's router scales its
weights only when it does not normalize them, V3's after normalizing; llama.cpp always normalizes if asked and then scales, so a
file with both `expert_weights_norm` and a scale ≠ 1 gets V3's rule (real V2 files never have both).

### 16.15.2 How the layouts were checked

`HfToGguf` is a Java port of the converter's rules (names, `+1`, `-exp`, `permute`, the split of `kv_b`, merged experts,
`rope_freqs` / `rope_factors_*`, the tensors forced to F32, the Q8_0 quantization of `gguf.quants.Q8_0` and its F16 fallback for
rows that are not a multiple of 32) applied to the tiny random checkpoints of `scripts/make-tiny-model.py` (with `--set key=json`
overrides for the variants llama.cpp can express), plus a synthetic byte-level vocabulary so that llama.cpp can load the file.
It refuses what the real converter cannot express (a GLM-Edge `head_dim` ≠ `hidden / heads`, Qwen3-MoE with dense layers,
DeepSeek with `mscale ≠ mscale_all_dim`). `GgufFamiliesCheck` then checks, for 24 tiny variants (the families above with and
without bias, windows, fused tensors, all three Phi-3 rope modes and four DeepSeek routers):

1. the F32 GGUF and the safetensors load (converted to Q8_0) of the same weights give the same logits: ≤ 9e-7 relative; this
   pins the metadata mapping and every tensor layout;
2. the Q8_0 file read in place equals the F32 file of its dequantized values (≤ 4e-6) and is within quantization noise of the
   safetensors load (≤ 4e-3; for DeepSeek 2e-2, see below);
3. llama.cpp's logits for the same ids, table above (`scripts/gguf-llamacpp-check.sh`; skipped when `llama-debug` is absent).

llama.cpp pads `-c 64` to 256, which decides the short / long LongRoPE table (`n_ctx > original`): the short-table variant
therefore has `original_max_position_embeddings = 256`. llama.cpp recognises "lite" DeepSeek models by their layer count
(26 / 27), so the tiny V2 without query low-rank has 27 layers.

**The DeepSeek quantization floor.** The unsplit and the split form are the same weights, but the Q8_0 blocks of `attn_k_b` run
along `nope` and the checkpoint's `kv_b` along `kv_lora`, so a Q8_0 GGUF differs from the Q8_0 conversion of the safetensors by
more than usual: 2e-2 of the largest logit on these tiny, very sensitive models (4 layers; 0.2 with 27), against 1e-3 for the
unsplit form. The check prints the same figure for llama.cpp itself (float weights against jllm's Q8_0: 4e-2 … 2e-1), so it is
the models' sensitivity to quantization, not the layout: with the same numbers in both engines the difference is 1e-6.

### 16.15.3 Running them

```sh
scripts/gguf-llamacpp-check.sh                   # every tiny family against llama.cpp (needs llama-debug on PATH, python3)
scripts/gguf-llamacpp-check.sh gemma3 glm4       # some
mvn -o test                                      # GgufFamiliesCheck runs without llama.cpp (16 s)
scripts/arch-smoke.sh                            # ends with GgufArchSmoke: Q8_0 GGUFs of the families, CPU == Metal
```

Limits: no real Gemma / Phi / GLM / Mamba / DeepSeek GGUF was used (no weights are downloaded for this work), so what is proved
is the layout against llama.cpp's own loader, not a model's text; the `tokenizer.ggml.pre` values of these families are covered
by 16.12. DeepSeek's group-limited routing for V2 files and the sliding window of Phi-3 follow the model, not llama.cpp.
