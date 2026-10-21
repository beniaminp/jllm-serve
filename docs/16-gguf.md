# 16. GGUF: running llama.cpp's quantized files without converting them

Code: `gguf/GgufFile.java`, `gguf/GgmlType.java`, `gguf/GgufArchitecture.java`, `gguf/GgufWeights.java`,
`gguf/GgmlCodec.java`, `gguf/GgufModel.java`, `model/Ggml.java`, `model/QType.java`, `model/QTensor.java`,
`model/TensorSource.java`, `tokenizer/GgufTokenizer.java`, `cpu/GgufKernels.java`, the *GGUF block types* section
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

The other llama.cpp formats that small or old files contain (Q4_1, Q5_0, Q5_1, Q3_K) have scalar dequantizers
(`Ggml.dequantQ4_1` ...), used only to convert at load (16.6). IQ\*, TQ\* and MXFP4 are refused by name.

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
| Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q2_K | **in place** (zero copy, both backends) |
| Q3_K, Q4_1, Q5_0, Q5_1 | requantized to Q8_0 in memory (rare: files mix them in for tensors whose row length is not a multiple of 256) |
| F16, BF16, F32 matrices | quantized to Q8_0 in memory — meant for small models: a BF16 file becomes a Q8_0 model in RAM, and loading it costs a pass over all weights |
| F32 vectors (norms, biases, router) | copied into `float[]` |

The int8-activation CPU path (`backend.cpu.int8-activations`, [07](07-cpu-backend.md)) has kernels for Q4_0 and Q8_0
and, for these files, `Q4_0G` and `Q8_0G`; `CpuMatmul.useInt8` excludes every other type (Q2_K, Q4_K, Q5_K), which
stay on the float kernels.

---

## 16.7 The tokenizer in the file

`GgufTokenizer` writes the `tokenizer.json` that the model's Hugging Face tokenizer would have and loads it with the
existing encoder ([04](04-tokenizer.md)), so there is one BPE implementation:

* **`tokenizer.ggml.model = gpt2`**: byte-level BPE. `tokens` are the vocabulary and `merges` the merge list.
  `token_type` separates `CONTROL` tokens (added tokens flagged `special`: `<|im_start|>`, `<|endoftext|>`) and
  `USER_DEFINED` ones (added, *not* special: `<think>`, `<tool_call>`) from the vocabulary; as in llama.cpp,
  tokens it knows by name (`<|im_end|>`, `</s>`, the fill-in-the-middle markers...) are control tokens whatever the
  file says. `tokenizer.ggml.pre` picks the pre-tokenizer: a list of regular expressions applied one after the other
  plus `ignore_merges`, exactly llama.cpp's (80 values, 16.12; a missing key or `default` is llama.cpp's `default`,
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
RSS. The output is token for token the same (md5 `1a5dc189...` for the haiku prompt, Metal and CPU). With a file
that holds Q3_K, Q4_1/Q5_x or float matrices those tensors are still converted (`GGUF weights:` reports it).

---|---|---|---|
| converted q4 cache | 0.02 s | 1.7 s | ~230 MB |
| *conversion from safetensors, once* | *3 s, peak RSS 1.6 GB, 452 MB written* | | |

*Measured before `QType.Q6_K` was merged, when the 29 Q6_K tensors of this file were requantized to Q8_0 at load.* That
fallback (176 MB of copies) accounted for the difference in resident memory and about a second of the load; with Q6_K
used in place a `Q4_K_M` file is mapped entirely as it is (see the numbers re-measured below), and with a `Q8_0` file
the load is dominated by the tokenizer.

---

## 16.10 Limits and what is next

* **Architectures**: `llama`, `qwen2`, `qwen3`, `qwen3moe` only. Gemma 2/3, Phi-3, GLM-4, DeepSeek-2 and Mamba are
  supported from safetensors (14) but their GGUF tensor naming, norm and rotary conventions are not mapped yet.
* **Formats**: the IQ\* family, ternary TQ\* and MXFP4 are refused; Q4_K / Q5_K have no int8-activation CPU kernel.
* **RoPE**: `linear`, `yarn` (16.2.1), Llama 3 and a longrope-style table are mapped; YaRN's `truncate = false`
  variant and `dynamic` scaling are not.
* **MoE** layouts are tested on tiny models only (no MoE GGUF fits the download budget used here); shared experts
  (Qwen2-MoE, DeepSeek) are not mapped.
* **Tokenizer**: the pre-tokenizers that are not a list of regexes (`kimi-k2`, `afmoe`, `tiny_aya`, `superbpe`, `chameleon`,
  the SPM-style BPE `gemma4`, `sarvam-moe`, `mmbert`, `whitespace`) and `exaone-moe` need the original repository's
  `tokenizer.json`; `tokenizer.ggml.add_eos_token` and the WordPiece (`bert`) vocabularies are not applied (16.12).

---

## 16.11 Running the checks

`mvn test` runs the checks that need no download: `GgufFileTest` (reader, malformed files), `GgufKernelsCheck`
(block formats against the `gguf.quants` fixtures, SIMD against scalar, int8 activations), `GgufMetalCheck` (every
Metal dispatch family), `GgufModelTest` (tiny models of four families written as safetensors and as GGUF, the llama
permutation, metadata mapping), `GgufBackendsTest` (CPU == Metal on random GGUF models), `GgufHubTest` (file
selection, settings) and `GgufTokenizerTest` on the `llama-spm` vocabulary. With the Qwen3-0.6B files downloaded by
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
```

