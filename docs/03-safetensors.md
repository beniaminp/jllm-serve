# 3. Tensors, the safetensors format, and memory mapping

Code: `safetensors/SafeTensors.java`, `model/QuantFile.java`, `model/ModelConverter.java`

## 3.1 What is a tensor?

A **tensor** is just a multi-dimensional array of numbers with a known shape. Nothing more mysterious than that:

| Dimensions | Name | Example in this project | Java equivalent |
|---|---|---|---|
| 0 | scalar | the RMSNorm epsilon `1e-6` | `float` |
| 1 | vector | a norm weight `[1024]`; the embedding of one token | `float[1024]` |
| 2 | matrix | `q_proj.weight [2048, 1024]` | `float[2048][1024]` |
| 3 | 3-D tensor | fused MoE experts `gate_up_proj [128, 1536, 2048]` | `float[128][1536][2048]` |

A model is a collection of named tensors ("parameters" or "weights"). Qwen3-0.6B has 311 of them;
Qwen3-30B-A3B has 18 867. Each has:

- a **name** that tells you where it is used: `model.layers.3.self_attn.k_proj.weight` is the key projection
  matrix of layer 3;
- a **dtype** — how each number is stored: `BF16` (2 bytes), `F32` (4 bytes), `F16`, `I8`, …;
- a **shape** — the size of each dimension, e.g. `[2048, 1024]` = 2048 rows × 1024 columns;
- the **data** — `product(shape)` numbers written one after the other.

### Shapes follow the math

A linear layer computes `y = W · x`. If `x` has 1024 entries and `y` must have 2048, then `W` has shape
`[2048, 1024]` — *[output size, input size]*, which is how PyTorch's `nn.Linear` stores it and therefore how it
appears in the file. Reading shapes tells you the architecture without any documentation:

```
model.embed_tokens.weight            [151936, 1024]   151 936 tokens, each a 1024-vector
model.layers.0.self_attn.q_proj      [2048, 1024]     1024 → 2048 = 16 heads × 128
model.layers.0.self_attn.k_proj      [1024, 1024]     1024 → 1024 =  8 heads × 128  (GQA: fewer KV heads)
model.layers.0.mlp.gate_proj         [3072, 1024]     1024 → 3072
model.layers.0.mlp.down_proj         [1024, 3072]     3072 → 1024
model.layers.0.self_attn.q_norm      [128]            one gain per head dimension
```

### Row-major layout: how a shape becomes a flat list of bytes

Memory is one-dimensional, so a `[rows, cols]` matrix is flattened **row by row** ("row-major" or "C order"):
element `(r, c)` lives at flat index `r * cols + c`. A `[2, 3]` matrix

```
┌ a b c ┐
└ d e f ┘      is stored as   a b c d e f
```

For a 3-D `[E, R, C]` tensor the same rule nests: index `(e, r, c)` is at `e*R*C + r*C + c`. A useful
consequence used by the converter: a `[E, R, C]` tensor *is already* an `[E·R, C]` matrix in memory — "reshaping"
costs nothing (`ModelConverter.flatten3d`).

Row-major also means **a row is contiguous**. Every kernel in this project works a row at a time (one dot
product per output), which is why rows are the unit of quantization, of thread scheduling, and of GPU simdgroups.

## 3.2 The safetensors file format

`safetensors` is Hugging Face's replacement for Python pickles: trivially simple, impossible to hide code in,
and designed for exactly the zero-copy loading we do.

```
offset 0        ┌──────────────────────────┐
                │ N : u64, little-endian   │   length of the JSON header
offset 8        ├──────────────────────────┤
                │ JSON header, N bytes     │   names → dtype, shape, data_offsets
offset 8 + N    ├──────────────────────────┤
                │ tensor data              │   all tensors' bytes, back to back, no separators
                └──────────────────────────┘
```

The real header of `Qwen3-0.6B/model.safetensors` begins like this (whitespace added):

```json
{
  "__metadata__": {"format": "pt"},
  "lm_head.weight"            : {"dtype": "BF16", "shape": [151936, 1024], "data_offsets": [0, 311164928]},
  "model.embed_tokens.weight" : {"dtype": "BF16", "shape": [151936, 1024], "data_offsets": [311164928, 622329856]},
  "model.layers.0.input_layernorm.weight" : {"dtype": "BF16", "shape": [1024], "data_offsets": [622329856, 622331904]},
  "model.layers.0.mlp.down_proj.weight"   : {"dtype": "BF16", "shape": [1024, 3072], "data_offsets": [622331904, 628623360]},
  ...
}
```

(The names are sorted alphabetically, which is why `lm_head.weight` comes first. Qwen3-0.6B ties the head to the
embedding, `tie_word_embeddings = true`, yet the file still stores a second copy of it — 311 MB that jllm never
reads, because the converter only takes `lm_head.weight` from checkpoints whose head is untied.)

Check the arithmetic: the embedding has 151 936 × 1024 = 155 582 464 numbers × 2 bytes (BF16) = 311 164 928
bytes, which is exactly `data_offsets[1] − data_offsets[0]`. `data_offsets` are **relative to the start of the
data section**, so the absolute file position of a tensor is `8 + N + begin`. `SafeTensors.readFile` verifies
this invariant for every entry (`end − begin == numel × dtype.bytes`, `end ≤ data size`) and rejects the file
otherwise — a truncated download fails here, not as silent garbage.

### Reading one number by hand

Suppose you want element `(r = 5, c = 7)` of `q_proj.weight [2048, 1024]` whose `data_offsets` begin at `B`:

1. flat index = `5 × 1024 + 7 = 5127`
2. byte position in file = `8 + N + B + 5127 × 2`
3. read 2 bytes little-endian → a `BF16` bit pattern
4. convert to `float` (next section)

This is literally what `Tensor.toFloat(start, dst, off, n)` does, in a loop, for `n` consecutive elements.

### Sharded checkpoints

Big models are split into several files of ~4 GB so uploads and downloads can resume:

```
model-00001-of-00016.safetensors
...
model-00016-of-00016.safetensors
model.safetensors.index.json       {"metadata": {"total_size": 61064245248},
                                    "weight_map": {"model.embed_tokens.weight": "model-00001-of-00016.safetensors", ...}}
```

Each shard is a complete safetensors file; the index says which shard holds which tensor. `SafeTensors.open(dir)`
opens all shards; `ModelConverter` opens them one at a time (section 3.5).

## 3.3 bfloat16

Weights are stored as **bfloat16**: a 32-bit IEEE float with the low 16 bits of the mantissa dropped.

```
float32:  1 sign | 8 exponent | 23 mantissa      ~7 significant digits, range ±3.4e38
bfloat16: 1 sign | 8 exponent |  7 mantissa      ~3 significant digits, same range as float32
float16:  1 sign | 5 exponent | 10 mantissa      ~4 significant digits, range only ±65504
```

Neural networks care more about *range* than precision (activations span many orders of magnitude, individual
weights do not need 7 digits), so bf16 became the training standard. Because the sign and exponent fields are
laid out identically to float32, conversion is a 16-bit shift and no arithmetic:

```java
short bits = …;                                            // the two bytes from the file
float f = Float.intBitsToFloat((bits & 0xFFFF) << 16);     // bf16 → f32: pad the mantissa with zeros
```

Example: `0x3F80` → shifted `0x3F800000` → `1.0f`. `0xC000` → `0xC0000000` → `−2.0f`.
`F16` needs real conversion (`Float.float16ToFloat`), `F32` is a plain copy.

## 3.4 Memory mapping with the FFM API

A 1.5 GB (or 61 GB) file must not be read into a `byte[]`: arrays are capped at 2 GB, the copy is slow and
doubles memory. Instead we **memory-map** it:

```java
try (FileChannel ch = FileChannel.open(file, READ)) {
    MemorySegment mapped = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
}
```

`mmap` asks the kernel to make the file *appear* at an address range. No bytes are read yet; the first access to
each 16 KiB page triggers a page fault and the kernel loads that page from disk (and keeps it in the page cache,
sharing it with any other process mapping the same file). Pages can be dropped under memory pressure since they
are backed by the file. Opening a model therefore takes milliseconds, and memory usage is whatever you actually
touch.

`MemorySegment` (from `java.lang.foreign`, the Foreign Function & Memory API) is the handle: a bounds-checked
view of native memory of any size (the old `ByteBuffer` was limited to 2 GB). Each tensor becomes
`mapped.asSlice(8 + N + begin, end − begin)` — a sub-view, still no copy. Values are read with *unaligned
little-endian layouts* (`JAVA_SHORT_UNALIGNED.withOrder(LITTLE_ENDIAN)`) because tensor data in the file is
packed with no alignment guarantees and safetensors is always little-endian.

All slices share one `Arena.ofShared()` ("shared" = usable from any thread, which the parallel quantizer
needs); closing the `SafeTensors` object unmaps everything at once.

## 3.5 From safetensors to jllm's quantized cache

The safetensors file is read exactly once, by `ModelConverter`, which writes a second file in jllm's own layout:

```
models/Qwen_Qwen3-0.6B/
├── model.safetensors      1503 MB   bf16, as downloaded
├── jllm-q4_0.bin           452 MB   quantized blobs, one per tensor, each 16 KiB-aligned
└── jllm-q4_0.json           32 KB   {"version", "layout", "quant", "model", "repo", "revision", "sha",
                                      "tensors": {name: {type, rows, cols, offset, size}}}
```

The names follow `--quant`: `jllm-q8_0.bin/.json` for Q8_0, `jllm-q2_k.bin/.json` for Q2_K, so several formats of
one model can sit side by side and be switched with a flag. In the index, `"model"` is the architecture's
one-line description (`Architecture.core().description()`, e.g. `Qwen3ForCausalLM dim=1024 ffn=3072 layers=28 ...`),
and `"repo"`, `"revision"` and `"sha"` are the Hub name, the revision you asked for, and the commit it resolved to.

Besides the tensor table the index records what the cache was made from: the file format `version`, the
converter `layout` version (bumped whenever the converter changes what it writes — fused tensors, folded norms,
names), the config description, and the `repo@revision` the model was requested as plus the Hub commit it
resolved to. `ModelWeights.load` compares all of these (and every tensor shape against the `Architecture` parsed from
`config.json`) and, when anything differs — an older converter, another revision in the same directory, a
different config — prints `Reconverting …`, deletes the cache and converts again instead of loading stale
weights. The same happens to an index that cannot be parsed (`QuantFile.StaleCacheException`).

Why a cache file instead of quantizing at every start?

- quantizing 30 billion weights takes minutes; mapping a file takes milliseconds;
- the blobs are written in **exactly the in-memory layout of `QTensor`** (blocks, then scales, padded to a page),
  so after `mmap` a tensor is just `QTensor.view(type, rows, cols, mapped.asSlice(offset, size))` — and because
  every offset is a multiple of 16 KiB, the Metal backend can hand the mapped pages to the GPU with
  `newBufferWithBytesNoCopy` (zero copy, [08](08-metal-backend.md));
- it decouples the download from running: once the cache exists the original shards can be deleted.

The converter processes the model **one layer at a time** so peak memory is one layer, and opens shards lazily:

```
for each unit (embedding, layer 0 … layer L−1, final norm, lm_head):
    for each rule of the unit (one cache tensor: its source tensors and a transform):
        open the sources' shards if not open yet (ShardProvider.ensure → may download them)
        quantize / split / stack / keep-as-f32 into a temporary off-heap arena, append to jllm-q4_0.bin
    fsync the .bin, rewrite jllm-q4_0.partial.json (units and tensors written so far; tmp + fsync + rename)
    for each shard whose tensors are all consumed: close it, ShardProvider.release(shard)
write jllm-q4_0.json (tmp + fsync + rename)      ← only now does the cache count as complete
```

Each unit is a list of *rules*, one per cache tensor, naming the checkpoint tensors it reads; the set of shards
a unit needs is derived from those rules, so the shard bookkeeping cannot disagree with what the conversion
actually reads.

With `--low-disk`, `release` deletes the shard. That is how a 61 GB checkpoint is converted on a machine with
50 GB free: at most two ~4 GB shards exist at any moment next to the growing 19 GB cache. The JSON index is
written last and atomically, so a crash mid-way leaves no half-valid cache. A `.bin` without a `.json` is *not*
thrown away: the sidecar `jllm-q4_0.partial.json` records the units committed so far, and the next run resumes
after them (truncating whatever the interrupted unit had appended), provided the sidecar's layout version,
quant, config description and `repo@revision` match the current conversion — otherwise it starts over.

The cache also normalizes naming differences between checkpoint generations — e.g. the dense MLP is stored as one
fused `mlp.gate_up` tensor and MoE experts as `mlp.experts.gate_up` / `mlp.experts.down` whichever of the two
Hugging Face layouts the source used ([12](12-mixture-of-experts.md)).

## 3.6 Validation summary

| Check | Where | Catches |
|---|---|---|
| header length sane, offsets inside file, `end − begin == numel × bytes` | `SafeTensors.readFile` | truncated / corrupt downloads |
| SHA-256 of every LFS file | `HuggingFaceHub` | corrupted transfer |
| every matrix shape against the `Architecture` | `ModelWeights` constructor, `DecoderLayer.matches` | wrong model / stale cache (reconverted automatically) |
| duplicate tensor names (also across shards), overlapping `data_offsets` | `SafeTensors.readFile` | a header that does not describe one tensor per name |
| cache version (`QuantFile.VERSION`), converter layout version (`QuantFile.LAYOUT_VERSION`), `repo@revision`, config description | `QuantFile.open`, `ModelWeights.load` | format / converter changes, another revision in the same directory (reconverted automatically) |
| index and sidecar written to a temporary file, fsynced, renamed | `QuantFile.Writer` | a crash can leave a half-written index; instead the cache is simply incomplete and the conversion resumes |

## 3.7 MLX quantized checkpoints

🧸 Some people ship their models already squeezed small, like a sweater vacuum-packed for a suitcase. Apple's MLX
community does this for Macs. jllm-serve unpacks the sweater once, then folds it its own way for the cupboard.

📐 The `mlx-community/*-4bit` (and `-8bit`, `-3bit`…) repositories are safetensors files written by
`mlx_lm.convert -q`. Each quantized linear layer `X` is three tensors:

| tensor | dtype | shape | holds |
|---|---|---|---|
| `X.weight` | U32 | `[out, in · bits / 32]` | the `bits`-wide codes `q ∈ [0, 2^bits)`, packed little-endian, lowest bits first |
| `X.scales` | f16 / bf16 | `[out, in / g]` | one scale `s` per group of `g` columns |
| `X.biases` | f16 / bf16 | `[out, in / g]` | one offset `z` per group (the group's minimum) |

and `config.json` carries `"quantization": {"group_size": 64, "bits": 4}`. The value of a weight is **affine**:

```
w[r][c] = s[r][c / g] · q[r][c] + z[r][c / g]
q[r][c] = (word[r][c / (32/bits)] >>> ((c mod (32/bits)) · bits)) & (2^bits − 1)
```

Example (4 bits, so 8 codes per word): the word `0x76543210` holds `q = 0, 1, 2, …, 7` for columns 0–7; with
`s = 0.01, z = −0.08` those are `−0.08, −0.07, …, −0.01`. Q4_0 ([06](06-quantization.md)) stores only a scale
(`w = d · (q − 8)`, symmetric around zero); the extra offset lets MLX fit groups that are not centred on zero.

**Which bits?** Layers may differ from the config (MLX often keeps embeddings or the LM head at 6 or 8 bits). The
shapes tell: `in = (in/g) · g` and `packed_cols · 32 = in · bits`. One ambiguity: an 8-bit layer of width `in` with groups of 64 has
`in / 4` packed columns and `in / 64` scales, and so does a 4-bit layer of width `2·in` with groups of 128. So
`MlxQuant.dequantized` tries the config's group size first, then 64, 32, 128, and accepts the
first that gives 2, 4 or 8 bits.

**How it is loaded.** `MlxQuant.read` recognises the checkpoint from `config.json` (only `mode: affine`), and
`ModelConverter` reads each packed matrix through a `SafeTensors.Tensor` whose reader dequantizes on the fly, like
any bf16 tensor. The converter then writes the engine's own format (`--quant`) into the usual cache
(§3.5), so every family and both backends work unchanged and the MLX files are read only once. `q8` or `q6k` keep a
4-bit checkpoint's values almost exactly; `q4` re-quantizes them a second time. Measured on Qwen3-0.6B: the
dequantized 4-bit weights differ from the bf16 originals by about 9 % RMS (the 4-bit rounding itself), and the model
answers correctly end to end.

```
./jllm chat --model mlx-community/Qwen3-0.6B-4bit --quant q8
```
