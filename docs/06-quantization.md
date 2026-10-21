# 6. Quantization: Q4_0 and Q8_0 — with all the math

Code: `model/QType.java`, `model/QTensor.java`, `model/QuantFile.java`, `model/ModelConverter.java`,
`model/ModelWeights.java`; kernels in `cpu/VectorKernels.java`, `cpu/Int8Kernels.java`, `cpu/CpuMatmul.java`

This chapter derives the two 32-weight-block formats, Q4_0 and Q8_0, and the memory layout, cache file and
kernels that every format shares. The third format, Q2_K (`--quant q2k`, 256-weight super-blocks with
two-level scales), has its own chapter: [13](13-q2k-quantization.md).

The model's weights are 596 million numbers (Qwen3-0.6B) or 30 billion (Qwen3-30B-A3B). How many bytes each
number takes decides how much memory the model needs **and how fast it runs**. This chapter explains why,
then builds the two storage formats used in this project from first principles, with a full worked example
you can recompute by hand.

---

## 6.0 The toolbox

### Bits, bytes, and what "4-bit" means

A **bit** is a 0 or a 1. A **byte** is 8 bits and can hold 2⁸ = 256 different values. The number of bits
decides how many distinct values a slot can distinguish:

| bits | distinct values | as an unsigned integer | as a signed integer (two's complement) |
|---|---|---|---|
| 4 (a "nibble") | 16 | 0 … 15 | −8 … +7 |
| 8 | 256 | 0 … 255 | −128 … +127 |
| 16 | 65 536 | | bf16: ~3 significant digits |
| 32 | 4.3 billion | | f32: ~7 significant digits |

"4-bit quantization" therefore means: each weight is stored as one of only **16** possible values. That sounds
hopeless — a weight is a real number like −0.0613 — until you add one extra number per group of weights, the
*scale*, which is the whole trick (6.2).

Note the asymmetry of signed integers: 4 bits give **−8 … +7**, one more negative value than positive. We will
use that.

### Floating point vs fixed point

A `float` stores a number as `mantissa × 2^exponent`: the exponent lets it represent 10⁻³⁰ and 10³⁰ with the
same relative precision. Weights in one row do not need that: they are all roughly the same size (typically
±0.05). For such data a **fixed-point** representation — an integer times one shared scale — wastes no bits on
an exponent that would be the same for every value anyway. Quantization is converting from floating point to
fixed point, group by group.

### Memory bandwidth: why size is speed

Generating one token (decode) reads every weight exactly once ([05 §5.9](05-transformer-math.md#59-prefill-decode-and-the-kv-cache)),
and does only ~2 arithmetic operations per weight read. Any modern chip can do far more arithmetic than that
per byte fetched, so the time per token is simply

```
time per token  ≈  bytes of weights / memory bandwidth

Qwen3-0.6B, bf16 (2 bytes/weight):   1 190 MB / 200 GB/s  =  6.0 ms  →  ≤ 168 tok/s   (M1 Pro ceiling)
Qwen3-0.6B, Q4_0 (0.625 B/weight):     450 MB / 200 GB/s  =  2.3 ms  →  ≤ 440 tok/s
```

Halving the bytes doubles the ceiling. (The measured Q4_0 decode rates are well below the ceilings, ~111 tok/s on
Metal and ~51 tok/s on the CPU, for reasons covered in [11](11-performance.md), but the ratio holds.) The second reason is capacity: the 30B MoE model is 61 GB in bf16
and does not fit in 32 GB of RAM at all; at ~0.65 bytes/weight it is 19.4 GB and does.

---

## 6.1 The idea: one scale per block of 32

Pick a block of 32 consecutive weights in a row. Find the largest magnitude in the block, call it `amax`.
Choose a scale `d` so that `amax / d` lands exactly on the largest representable integer. Every weight is then
stored as the nearest integer to `w / d`, and reconstructed as `q × d`. Error is at most half a step, `d / 2`.

Why 32 and not the whole row? Because weights are not uniformly sized: a row of 1024 may contain one 0.3
among values around 0.02. A single row-wide scale would be set by the 0.3 and squash the 0.02s into one or two
levels. With 32-weight blocks, that outlier only hurts its 31 neighbours. The cost is 4 bytes of scale per
block — 1 bit per weight on top of the 4 or 8 data bits. (These are exactly llama.cpp's `Q4_0` and `Q8_0`
formats, so the numbers are comparable with the rest of the ecosystem.) In code: `QType.BLOCK = 32` (Q2_K is the exception: its unit is a 256-weight super-block, `QType.blockSize`, see [13](13-q2k-quantization.md)).

---

## 6.2 Q8_0: 8 bits per weight

**Formula** (`QTensor.quantizeBlockQ8`):

```
amax = max |x_j|                       j = 0 … 31
d    = amax / 127                      the scale: amax maps to ±127
q_j  = round(x_j / d)                  an integer in [−127, 127]  (one signed byte)
x_j  ≈ q_j · d
```

Storage: 32 bytes of `q` + one 4-byte float `d` = **36 bytes per 32 weights = 1.125 bytes/weight** (bf16 is 2).
Maximum error per weight: `d / 2 = amax / 254`, i.e. ~0.2 % of the block's largest value.

---

## 6.3 Q4_0: 4 bits per weight

**Formula** (`QTensor.quantizeBlockQ4`):

```
m    = the x_j with the largest |x_j|, sign kept     (e.g. −0.0613, not 0.0613)
d    = m / −8                                        so that m itself maps to exactly −8
q_j  = clamp( round(x_j / d) + 8, 0, 15 )            stored as an unsigned nibble; 8 means zero
x_j  ≈ (q_j − 8) · d
```

Storage: 16 bytes (two nibbles per byte) + 4 bytes scale = **20 bytes per 32 weights = 0.625 bytes/weight**.
Maximum error per weight: `|d| / 2 = amax / 16`.

### Full worked example

A realistic block (values of a Qwen3 attention matrix are typically ±0.02, with occasional outliers):

```
j :    0       1       2       3       4       5       6       7       8       9      10      11      12      13      14      15
x : −0.0051  0.0102 −0.0045 −0.0063 −0.0186 −0.0613  0.0222  0.0085  0.0207  0.0050  0.0079  0.0037 −0.0333  0.0171  0.0101  0.0100
j :   16      17      18      19      20      21      22      23      24      25      26      27      28      29      30      31
x : −0.0338 −0.0349 −0.0178 −0.0094  0.0061 −0.0009  0.0104 −0.0128  0.0062  0.0079 −0.0132  0.0344  0.0111  0.0239 −0.0124 −0.0148
```

**Step 1 — scale.** The largest magnitude is `x₅ = −0.0613`, so `m = −0.0613` and

```
d = m / −8 = −0.0613 / −8 = 0.0076625            (positive here because m was negative)
```

**Step 2 — quantize.** `q_j = round(x_j / d) + 8`, clamped to 0…15. (The code computes
`(int)(x · (1/d) + 8.5)`, which is the same rounding for the values that matter.)

```
x₀ = −0.0051 → −0.0051 / 0.0076625 = −0.666 → round = −1 → q = 7
x₁ =  0.0102 →  1.331 →  1 → q = 9
x₅ = −0.0613 → −8.000 → −8 → q = 0          ← the extreme value lands exactly on the end of the range
x₆ =  0.0222 →  2.897 →  3 → q = 11
x₂₇=  0.0344 →  4.489 →  4 → q = 12
x₁₁=  0.0037 →  0.483 →  0 → q = 8          ← "zero"
```

All 32:

```
q : 7 9 7 7 6 0 11 9 11 9 9 8 4 10 9 9 | 4 3 6 7 9 8 9 6 9 9 6 12 9 11 6 6
```

**Step 3 — reconstruct and measure.** `x̂_j = (q_j − 8) · d`:

```
j      x        q   x̂ = (q−8)·d    error
0   −0.0051     7    −0.0077       −0.0026
1    0.0102     9     0.0077       −0.0025
5   −0.0613     0    −0.0613        0.0000      exact by construction
6    0.0222    11     0.0230        0.0008
11   0.0037     8     0.0000       −0.0037
27   0.0344    12     0.0307       −0.0037
...
max |error| over the block = 0.00375  ≤  d/2 = 0.00383   ✓
```

Every reconstructed value is one of only 16 numbers (`−8d … 7d` = −0.0613 … 0.0536), yet each is within
0.0038 of the original. Relative to the typical weight (0.02) that is a ~15 % perturbation per weight; the
reason the model survives it is that each output is a sum of 1024 such products whose errors are independent
and mostly cancel ([6.9](#69-what-the-error-does-to-the-model)).

### Why the extreme value maps to −8, not ±7

A naive scheme would use a symmetric range: `d = amax / 7`, `q ∈ [−7, 7]`, wasting the 16th level. Mapping the
signed extreme to **−8** uses the full asymmetric range of 4-bit two's complement and gives a finer step. On the
block above:

| scheme | scale `d` | levels used | max error | RMS error |
|---|---|---|---|---|
| symmetric ±7 | 0.0613 / 7 = 0.00876 | 15 | 0.00431 | 0.00243 |
| Q4_0 (extreme → −8) | 0.0613 / 8 = 0.00766 | 16 | 0.00375 | 0.00234 |

The step is 12.5 % smaller, so every error bound is 12.5 % smaller, for free. (If the extreme is positive, `d`
comes out negative and the roles flip: the positive extreme still lands on `q = 0`, i.e. `−8 · d`.)

### Nibble packing

Sixteen bytes hold the 32 nibbles. The layout is **not** "weight 0 and 1 in byte 0"; it is:

```
byte j  (j = 0 … 15):   low nibble (bits 0–3)  = q_j        high nibble (bits 4–7) = q_{j+16}

byte 0 = q₀  | q₁₆ << 4 = 7  | 4 << 4 = 0x47
byte 1 = q₁  | q₁₇ << 4 = 9  | 3 << 4 = 0x39
byte 5 = q₅  | q₂₁ << 4 = 0  | 8 << 4 = 0x80
byte 11= q₁₁ | q₂₇ << 4 = 8  | 12 << 4 = 0xC8
...
block bytes: 47 39 67 77 96 80 9B 69 9B 99 69 C8 94 BA 69 69      then d = 0.0076625 lives in the scales region
```

Why this split? A SIMD kernel loads the 16 bytes once, and with one `AND 0x0F` gets weights 0…15 as 16 lanes,
and with one `SHIFT >> 4` gets weights 16…31 as 16 lanes — two full vectors from one load, no shuffling
([07 §7.3](07-cpu-backend.md#73-the-core-kernel-dotq4-block-by-block)). Unpacking: `lo = (b & 0xF) − 8`,
`hi = (b >>> 4) − 8` in `QTensor.dequantizeRow`.

---

## 6.4 The memory layout of a `QTensor`

A quantized `[rows, cols]` matrix is one contiguous off-heap `MemorySegment`. The diagram and arithmetic below are
the Q4_0 / Q8_0 case; each `QType` carries its own `blockSize` (weights per block: 32, or 256 for Q2_K),
`blockBytes` (16 / 32 / 84) and `scaleBytes` (4 for the separate f32 scale of Q4_0 / Q8_0, **0** for Q2_K, whose
scales live inside its blocks), and the formulas are written with those fields:

```
offset 0 ───────────────────────────────────────────────────────────────────────┐
│ quantized blocks, row-major:                                                 │
│   row 0: block 0 | block 1 | … | block nb−1       nb = cols / 32 (Q2_K: ⌈cols / 256⌉) │
│   row 1: …                                        blockBytes = 16 (Q4) / 32 (Q8) │
│   …                                                                           │
│ size = rows × nb × blockBytes                                                 │
├── scalesOffset  (rounded up to a multiple of 64 bytes) ───────────────────────┤
│ f32 scales, row-major: rows × nb floats                                        │
│ size = rows × nb × 4                                                           │
├── padding to a multiple of 16 384 bytes (one Apple Silicon page) ─────────────┤
└───────────────────────────────────────────────────────────────────────────────┘
```

The helper arithmetic (`QTensor.rowOffset`, `scaleOffset`, `scalesOffset`, `byteSize`):

```
nb             = ⌈cols / blockSize⌉                  QTensor.blocksPerRow()
rowOffset(r)   = r × nb × blockBytes                 first block of row r
scaleOffset(r) = scalesOffset + r × nb × 4           first scale of row r
scalesOffset   = align(rows × nb × blockBytes, 64)
byteSize       = align(scalesOffset + rows × nb × scaleBytes, 16384)
```

For Q2_K `scaleBytes = 0`, so the scale region is empty and `byteSize` is just the blocks rounded up to a page
([13 §13.6](13-q2k-quantization.md#136-a-q2_k-qtensor)).

**Worked example, `k_proj.weight` of Qwen3-0.6B, `[1024, 1024]` in Q4_0:**

```
nb            = 1024 / 32 = 32 blocks per row
blocks region = 1024 rows × 32 × 16 B = 524 288 B            (already a multiple of 64)
scalesOffset  = 524 288
scales region = 1024 × 32 × 4 B = 131 072 B
end           = 655 360 B  = exactly 40 pages of 16 384 → no padding
total         = 655 360 B  = 0.625 B/weight ✓     (bf16 would be 2 097 152 B)

row 700, block 5:   quantized bytes at 700×32×16 + 5×16 = 358 480
                    scale (float)    at 524 288 + 700×32×4 + 5×4 = 613 908
```

**The embedding `[151936, 1024]` in Q8_0:**

```
blocks region = 151 936 × 32 × 32 B = 155 582 464 B
scalesOffset  = 155 582 464
scales region = 151 936 × 32 × 4 B =  19 447 808 B
total         = 175 030 272 B ≈ 175 MB = 1.125 B/weight ✓
```

Scales are kept in their own region rather than interleaved with the data (as llama.cpp does) for two reasons:
both regions stay naturally aligned for vector loads, and the GPU can bind them as two separate buffer
arguments (`wq`, `ws` in `kernels.metal`) at two offsets of the same allocation (`MetalKernels.scaleOff`; for
Q2_K it binds offset 0 because there is nothing to point at).

The 16 KiB page alignment (`QTensor.PAGE`) is what allows Metal to wrap the memory with
`newBufferWithBytesNoCopy` instead of copying it ([08](08-metal-backend.md)); it costs at most 16 KiB of padding
per tensor.

---

## 6.5 Which tensors get which format

| tensor | format | why |
|---|---|---|
| embedding / LM head | **Q8_0** by default; `model.head-quant=q6k` (`--head-quant q6k`) stores them as Q6_K | the LM head's error goes straight into the logits; small models are measurably worse with a Q4 head. Also only read once per token (one row for the embedding; all rows for the head). Q6_K is 6.5625 instead of 8.5 bits per weight and near-lossless; see 6.5.1 |
| attention `q/k/v/o` (or the MLA projections), FFN `gate/up/down`, MoE experts incl. DeepSeek's shared experts | `--quant` (`model.quant`): Q4_0 default, `q8`, `q2k` or `q6k` | the bulk of the bytes; errors average out over many dot products. The converter applies the one chosen format to **every** one of these tensors; there is no mixed recipe (e.g. `down_proj` at 4 bits, the rest at 2) |
| norms (incl. QK-norm and post-norm gains), q/k/v biases, MoE router (and DeepSeek's router bias) | f32 | tiny (1 KB – 1 MB each); the router decides which experts run, so its precision matters |
| activations | f32; on the CPU decode path quantized on the fly to Q8_0 (6.8) | computed per token, never stored on disk |
| KV cache | f32, or f16 with `backend.kv-dtype=f16` | in RAM only; for DeepSeek's MLA it holds just the latent row ([05 §5.9](05-transformer-math.md#59-prefill-decode-and-the-kv-cache)) |

Sizes that result (`ModelWeights.byteSize()`):

| model | parameters | bf16 | jllm Q4_0 cache | jllm Q8_0 cache |
|---|---|---|---|---|
| Qwen3-0.6B (tied embedding) | 596 M | 1 190 MB | **452 MB** (175 Q8 embed + 275 Q4 layers + padding) | ~670 MB |
| Qwen3-30B-A3B (separate `lm_head`) | 30.5 B | 61 000 MB | **≈ 19.4 GB** (2 × 350 MB Q8 embed/head + 18.1 GB Q4 experts + 0.6 GB Q4 attention) | would not fit in 32 GB |

With `--quant q2k` the same 0.6B cache is 321 MB and the 30B's about 10.5 GB ([13 §13.6](13-q2k-quantization.md#136-a-q2_k-qtensor)).

### 6.5.1 The head format and Q6_K

`model.head-quant` (`q8`, the default, or `q6k`) selects the format of the embedding and the LM head. The head is
read in full for every decoded token, so for Qwen3-0.6B its 175 MB (Q8_0) are 39 % of the 452 MB read per token;
as Q6_K (`model/Q6K.java`, llama.cpp's `block_q6_K`: 256 weights in 210 bytes = `ql[128]` low nibbles, `qh[64]`
high 2-bit pairs, 16 int8 sub-block scales, one f16 `d`; `w = d·scale_j·(q − 32)`) it is 128 MB and the cache
405 MB. The format is byte-compatible with llama.cpp (the dequantizer is bit-identical to `gguf.quants`; the
quantizer is a port of `quantize_row_q6_K_ref` / `make_qx_quants`). `q6k` is also accepted by `model.quant` for every
linear layer, and rows that are not a multiple of 256 end with a partial super-block as for Q2_K.

The head format is part of the cache identity: a Q6_K head is converted into `jllm-q4_0-h6k.bin/.json` (the index
records `"head": "Q6_K"`) next to the unchanged Q8_0-head `jllm-q4_0.bin`, and a cache whose embedding is not the
requested format is reconverted. `LAYOUT_VERSION` is untouched, so existing caches stay valid.

Measured on Qwen3-0.6B (Q4_0 layers; M1 Pro, 10 cores, a machine that was heavily loaded by other jobs, so the
timings are best-of-many micro-benchmarks, not averages):

| | Q8_0 head | Q6_K head |
|---|---|---|
| head tensor / cache | 175 MB / 452 MB | 128 MB / 405 MB |
| Metal head matmul, 1 token (151 936 × 1 024) | 1.2 – 1.3 ms | 1.8 – 2.0 ms |
| CPU head matmul, 1 token, default `int8-activations` | 3.4 – 4.0 ms (integer dot) | 18 ms (float path) |
| teacher-forced perplexity, 514-token English passage | 35.738 | 35.837 (+0.28 %) |
| greedy top-1 vs the Q8_0 head, same passage | — | 494 / 513, mean KL 0.0017 |
| greedy text (haiku, capital of France, process vs thread) | — | identical; one binary-search answer adds "O(log n)" |

The head is *not* bandwidth-bound at decode (a Metal decode step reads ~450 MB in ~10 – 16 ms, a quarter of the
memory bandwidth), so the 47 MB saved do not buy speed, while Q6_K's extra unpacking costs more than they save: about
0.6 ms per token on Metal, and on the CPU the int8 path (Q4_0 / Q8_0 only, see 6.8) is lost for the head, which makes it
~14 ms slower per token. `q6k` is therefore a memory saving (and the basis for a Q6_K-all-layers recipe), not a speed
setting, and the default stays `q8`.

Kernels: `cpu/Q6KKernels.java` (byte-vector codes `lo | hi<<4 − 32` times the activation, scaled per sub-block; no int8
activations: `CpuMatmul.useInt8` is Q4_0 / Q8_0 only) and a `Q6_K` section at the end of `kernels.metal` whose matmul
lane unit is a *column quad* (four weights of each of a half's four 32-weight groups = 12 quant bytes) and whose MoE
`block_dot<6>` walks 16-weight sub-blocks.
---

## 6.6 Stacked matrices and row arithmetic

Several logical matrices with the same `cols` can live in one `QTensor`; because rows are contiguous, matrix
`i` is just a row range:

```
dense FFN   wGateUp  [2·ffn, dim]      rows [0, ffn) = Wgate,   rows [ffn, 2·ffn) = Wup
MoE         expGateUp [E·2m, dim]      expert e: rows [e·2m, e·2m+m) = gate_e, [e·2m+m, (e+1)·2m) = up_e
MoE         expDown   [E·dim, m]       expert e: rows [e·dim, (e+1)·dim)
```

Selecting expert `e` costs nothing: the kernel computes `row = e × rowsPerExpert + r` and reads
`rowOffset(row)` ([12](12-mixture-of-experts.md)). `ModelConverter.stack` builds these by quantizing a
`RowSource` that maps global row `r` to the right source tensor with a binary search over the row boundaries.

---

## 6.7 Quantizing: when and how

`QTensor.quantize(arena, type, rows, cols, rowSource)` allocates the segment (`cols` must be a multiple of 32),
then runs `IntStream.range(0, rows).parallel()` — each row is quantized independently, reading its source values
through the memory-mapped safetensors ([03 §3.4](03-safetensors.md#34-memory-mapping-with-the-ffm-api)). A worker
**stages** the row in thread-local on-heap arrays — the `float[cols]` source row, a `byte[]` for the quantized
blocks and a `float[]` for the scales, grown once and reused for every row the thread handles — and copies the
result into the segment with two bulk `MemorySegment.copy` calls, instead of one bounds-checked segment access
per quantized byte. All cores participate; Qwen3-0.6B converts in ~2 s, the 30B MoE in a few minutes dominated by
reading 61 GB from disk.

This happens **once**: `ModelConverter` walks the checkpoint one unit at a time (embedding, each layer, final
norm and head), writes every tensor's segment verbatim into `jllm-<quant>.bin` and the offsets into
`jllm-<quant>.json` (`<quant>` is the lower-cased `QType` name: `jllm-q4_0.bin`, `jllm-q8_0.bin`,
`jllm-q2_k.bin`; [03 §3.5](03-safetensors.md#35-from-safetensors-to-jllms-quantized-cache)). The JSON is written
last and is the "conversion complete" marker; a crash in the middle leaves a `jllm-<quant>.partial.json` sidecar
from which the next run resumes. The index records the file-format `version`, the converter's `LAYOUT_VERSION`
(which tensors exist under which names, which are fused or split; currently 2), the model description and the
**identity** of the checkpoint — `repo`, `revision` and the Hub commit `sha`.

The cache is therefore self-invalidating. `ModelWeights.load` converts when there is no cache, and when an
existing one cannot be used — another `LAYOUT_VERSION`, an unparseable or truncated index, a blob that does not
fit the file, tensor shapes that do not match the architecture, or an identity that differs from the
`repo@revision` now requested — `QuantFile` raises `StaleCacheException`, the files are deleted (`Reconverting
jllm-q4_0.bin: …` on stderr) and the conversion runs again. A freshly converted cache that still does not match
is reported as an error instead of looping. Afterwards `ModelWeights` memory-maps the `.bin` and wraps each
tensor with

```java
QTensor.view(type, rows, cols, mapped.asSlice(offset, size))
```

— no quantization, no copy, no RAM until a row is touched. `view` checks that `size == byteSize(type, rows,
cols)`, so a stale or truncated cache fails loudly (and, as above, triggers a reconversion).

---

## 6.8 Dequantizing: never materialize the matrix

Reading values back is `x̂ = (q − 8) · d` per weight (`QTensor.dequantizeRow`), but the hot kernels never expand
a whole matrix to floats: that would multiply memory traffic by 6. Instead:

All CPU matmuls go through `CpuMatmul.mul`, which picks the kernel by the number of tokens `n` and the weight format:

- **decode, float** (`VectorKernels.dotRow` → `dotQ4`): load 16 bytes, unpack to two 16-byte vectors *in
  registers*, convert to float lanes, multiply with the activation, accumulate; apply the scale once per block.
  Weight bytes are read exactly once. (`dotQ8` for Q8_0, `Q2KKernels.dot` for Q2_K.)
- **decode, int8 activations** (default, `backend.cpu.int8-activations=true`; `CpuMatmul.useInt8` → `Int8Kernels.dotRow`,
  Q4_0 and Q8_0 weights only): the activation vector is quantized **once per matmul** to Q8_0 — blocks of 32,
  `d = max|x| / 127`, one scale per block — and each weight row is multiplied with it in integer lanes: the exact
  integer sum `Σ w_i·q_i` of a block is scaled by `weightScale × activationScale` and accumulated in float. No
  byte→float conversion is left in the inner loop; the price is a ~0.9 % RMS relative error on each dot product,
  and the gain is about 4× decode speed on the CPU (13 → 54 tok/s in the A/B run of
  [11 §11.1](11-performance.md#111-measured); the current build decodes ~51 tok/s). Q2_K rows keep the float path.
- **prefill** (`n > 1`: `VectorKernels.dequantRow` + `dot4x4` / `dot4`): expand 4 rows into a thread-local float
  buffer, then reuse them for every token of the chunk (up to `backend.batch`, 64), four tokens at a time, so the
  unpacking cost is amortized. Prefill always uses float activations.
- **Metal** (`block_dot<QT>` / `mm_row<QT>` in `kernels.metal`): same as float decode, one `uchar4` at a time;
  activations stay f32 on the GPU.

Details in [07](07-cpu-backend.md) and [08](08-metal-backend.md).

---

## 6.9 What the error does to the model

Each quantized weight is off by up to `amax/16` (Q4_0). An output element is `Σ_c W[r][c] · x[c]` over 1024
terms; the errors `ε_c` are roughly independent with zero mean, so the error of the sum grows like `sqrt(1024)
= 32` while the sum itself grows like 1024: the *relative* error of a dot product is ~30× smaller than that of
an individual weight. That is why a 15 % per-weight perturbation yields outputs that are still mostly correct.

The standard way to measure the damage is **perplexity**: run a text through the model and compute how
surprised it is by the actual next tokens (lower is better). For Q4_0 on small models the increase is typically
a few percent (7.16 → 7.37 on the 97-token paragraph of [13 §13.7](13-q2k-quantization.md#137-measurements)); for Q8_0 it
is below measurement noise. In practice:

- greedy outputs of this project's Q4_0 Qwen3-0.6B give the same kind of answers as the full-precision model;
  individual token choices can differ from bf16 on close calls. Metal and the CPU with
  `backend.cpu.int8-activations=false` produce identical greedy text; the default int8 CPU path perturbs the
  last bits of the logits, so its text can diverge after a near-tie ([11 §11.1](11-performance.md#111-measured));
- `--quant q8` is the near-lossless reference when you want to check whether an odd answer is a Q4 artifact;
- the embedding/LM head stay Q8_0 because the logits see those errors with no averaging across layers.

Activations (`x`, `q`, `k`, `v`) stay f32 everywhere except the int8 decode matmuls of 6.8, where the activation
is rounded to Q8_0 for the duration of one matmul only (error ≈ 0.4 % per element, ~0.9 % per dot product); the
residual stream, norms, softmax and the attention scores are always f32. They are recomputed per token and never
stored, so there is no memory to save. The one activation-like thing that *is* stored, the KV cache, is f32 by
default and optionally f16 (`backend.kv-dtype`, [05 §5.9](05-transformer-math.md#59-prefill-decode-and-the-kv-cache)).

---

## 6.10 Other formats, for orientation

| format | idea | not in this project because |
|---|---|---|
| **Q2_K** (llama.cpp "K-quant") | super-blocks of 256 with 4-bit sub-block scales *and mins*, 2.625 bits/weight | *done*: `--quant q2k`, see [13](13-q2k-quantization.md) (applied to every linear layer; pure Q2_K breaks a 0.6B model) |
| **Q6_K** (llama.cpp "K-quant") | super-blocks of 256, 6-bit codes with int8 sub-block scales, 6.5625 bits/weight | *done*: `--head-quant q6k` for the embedding / LM head, `--quant q6k` for all layers (6.5.1) |
| **Q4_K / Q5_K** (llama.cpp "K-quants") | super-blocks of 256 with 6-bit sub-scales and minimums, ~0.56–0.81 B/weight with lower error than Q4_0 | the engine does not *produce* them (the simplest format that works is Q4_0, [13 §13.8](13-q2k-quantization.md#138-what-the-other-k-quants-change) compares them), but it *reads* Q4_K and Q5_K from llama.cpp GGUF files in place, and Q6_K in place as well ([16](16-gguf.md)) |
| **Q4_1, Q5_0, Q5_1, Q3_K, IQ4_NL, IQ4_XS, MXFP4, F16, BF16** (llama.cpp) | affine and 5-bit legacy blocks, 3-bit K-quant, non-linear 4-bit codebooks, e2m1 floats with a power-of-two scale, plain halves | the engine does not produce them either, but it reads them in place from GGUF files on both backends: layouts and exact dequantization in [6.12](#612-the-other-ggml-block-formats-read-in-place-from-gguf-files) |
| **GPTQ / AWQ** | choose the quantized values by minimizing the *output* error on calibration data, not per-weight rounding | needs calibration runs and a solver at conversion time |
| **int8 activations** (Q8_0 for `x` too) | integer × integer dot products, twice the lanes per SIMD register and no byte→float conversions | *done* on the CPU decode path for Q4_0 / Q8_0 weights, `backend.cpu.int8-activations` ([07 §7.10](07-cpu-backend.md#710-int8-activations-integer-dot-products-int8kernels)) |
| **fp8 / NF4** | 8-bit floats; 4-bit non-uniform "normal float" levels | GPU-oriented; no hardware fp8 on Apple M1 |

---

## 6.11 Summary

- A weight is stored as `integer × block scale`; 32 weights share a 4-byte scale. Q8_0: 1.125 B/weight, error
  ≤ amax/254. Q4_0: 0.625 B/weight, error ≤ amax/16, with the extreme value pinned to −8 to use all 16 levels.
- Nibbles are packed as (j, j+16) pairs so SIMD code gets two vectors from one 16-byte load.
- A `QTensor` is `[blocks][64-aligned scales][page padding]`; offsets are pure arithmetic, so a row, a block, or
  an expert is addressed without any lookup.
- Quantization happens once into `jllm-<quant>.bin` (+ `.json` with the layout version and the checkpoint's
  repo / revision / sha, so a stale cache is reconverted automatically); afterwards the file is memory-mapped
  and used in place by both the CPU (Vector API, optionally with int8 activations) and the GPU (zero-copy Metal
  buffers).
- Decode speed ∝ 1 / bytes, which is the whole reason to do this.

---

## 6.12 The other ggml block formats, read in place from GGUF files

[16](16-gguf.md) runs llama.cpp's `.gguf` files without converting them, and those files hold more block formats than
this chapter's two. The engine reads Q4_0, Q8_0, Q4_K, Q5_K, Q6_K and Q2_K directly (6.10); this section is the
block layout and the exact dequantization of the rest it reads in place: **Q4_1, Q5_0, Q5_1, Q3_K, IQ4_NL, IQ4_XS,
MXFP4** and the plain **F16 / BF16** matrices. All of them are byte-compatible with llama.cpp's `block_*` structs and
decode to exactly the floats of its `dequantize_row_*` functions (`ggml-quants.c`; the checks compare bit for bit).
Little endian everywhere, `d` and `m` are **f16**.

### 🧸 The idea, with cups and rulers

Think of a row of weights as a long shelf of cups, each filled to a different height. To save room we do not write
every height down; we pack the cups in small **boxes** and put a ruler in each box. A weight is then just "which mark
of the ruler".

* **Q4_0** (6.3) is the ruler with 16 evenly spaced marks that always starts at the same place, so every box says only
  how far apart the marks are (`d`).
* **Q4_1** gives the box a second number, where the ruler *starts* (`m`). A box whose cups are all tall can then use
  all its marks for the tall cups instead of wasting half of them on heights it never needs.
* **Q5_0 / Q5_1** are the same rulers with 32 marks instead of 16: one extra bit per cup, which is kept in a separate
  little strip of 32 bits so the nibbles stay where they were.
* **IQ4_NL** keeps 16 marks but spaces them unevenly: close together near zero, where most cups are, and far apart at
  the extremes. The mark numbers are looked up in a small fixed table instead of being counted.
* **IQ4_XS** is IQ4_NL for big boxes of 256 cups divided into 8 rooms; each room has its own step size, itself stored
  in 6 bits relative to one shared number, so the labels cost almost nothing.
* **MXFP4** writes each cup as a tiny floating-point number (a sign, a 2-bit exponent and 1 mantissa bit: 0, 0.5, 1,
  1.5, 2, 3, 4, 6) and the box's step is only a power of two, a single byte. It is what OpenAI's gpt-oss experts
  are stored in.
* **Q3_K** is the 3-bit cousin of the K-quants (13): 256 cups per box, 16 small rooms with their own 6-bit step sizes,
  and each cup is 2 bits plus one "is it in the lower half" bit kept in a separate mask.
* **F16 / BF16** are not boxes at all: every cup's height is simply written down with 16 bits.

### 📐 Layouts and formulas

`qs[j]` below is byte `j` of a block's nibble bytes: its low nibble is weight `j`, its high nibble weight `j + 16` (the
Q4_0 convention), for every 32-weight block type.

```
type    bytes / weights  bits/w   layout (offset: field)                                        weight
Q4_1    20 / 32          5.0      0: f16 d | 2: f16 m | 4: u8 qs[16]                            q d + m                q = nibble
Q5_0    22 / 32          5.5      0: f16 d | 2: u8 qh[4] | 6: u8 qs[16]                         (q5 - 16) d            q5 = nibble | bit << 4
Q5_1    24 / 32          6.0      0: f16 d | 2: f16 m | 4: u8 qh[4] | 8: u8 qs[16]              q5 d + m
IQ4_NL  18 / 32          4.5      0: f16 d | 2: u8 qs[16]                                       d kvalues_iq4nl[nibble]
MXFP4   17 / 32          4.25     0: u8 e (e8m0) | 1: u8 qs[16]                                 kvalues_mxfp4[nibble] 2^(e - 128)
Q3_K    110 / 256        3.4375   0: u8 hmask[32] | 32: u8 qs[64] | 96: u8 scales[12] | 108: f16 d     d (sc_s - 32) (((qs >> 2j) & 3) - (hmask bit ? 0 : 4))
IQ4_XS  136 / 256        4.25     0: f16 d | 2: u16 scales_h | 4: u8 scales_l[4] | 8: u8 qs[128]  d (ls_i - 32) kvalues_iq4nl[nibble]
F16     2 / 1            16       the IEEE half                                                  itself
BF16    2 / 1            16       the upper 16 bits of an f32                                    itself
```

**The fifth bit (Q5_0, Q5_1).** `qh` is a little-endian `u32`; weight `j < 16` (low nibble of `qs[j]`) takes bit `j` and
weight `j + 16` (high nibble) takes bit `j + 16`: `q5 = nibble | (((qh >> bit) & 1) << 4)`. Because the low half of the
weights uses `qh`'s low 16 bits and the high half its upper 16, a kernel that decodes one half of the block reads
only 2 of the 4 bytes.

**IQ4_NL.** The 16 levels are the signed bytes
`kvalues_iq4nl = -127 -104 -83 -65 -49 -35 -22 -10 1 13 25 38 53 69 89 113`
(denser around zero than a uniform grid); `w = d · kvalues[nibble]` and `d` is chosen at quantization time so the
extreme level lands on the block's extreme value. Same nibble order as Q4_0.

**IQ4_XS.** A 256-weight super-block of 8 sub-blocks of 32. Sub-block `i` has the 6-bit integer
`ls_i = (scales_l[i/2] >> 4(i mod 2)) & 15 | ((scales_h >> 2i) & 3) << 4` and weights
`d (ls_i - 32) kvalues_iq4nl[nibble]`, nibbles in the Q4_0 order within each 16 bytes of `qs`.

**MXFP4.** The OCP microscaling FP4: each value is an `e2m1` number (sign, 2 exponent bits, 1 mantissa bit) with the
nine magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6; llama.cpp stores them **doubled** as integers
`kvalues_mxfp4 = 0 1 2 3 4 6 8 12 0 -1 -2 -3 -4 -6 -8 -12` (nibble 8 is "-0") so that the scale can be a bare
exponent: `w = kvalues[nibble] · 2^(e - 128)`, where `2^(e-128)` is `2^(e-127)` (the shared e8m0 scale) times the
factor 1/2 that undoes the doubling. The float with the bits `(e - 1) << 23` is that power of two for `e >= 2`; for
`e = 0, 1` it is a subnormal (`0x00200000 << e`) which the GPU may flush to zero, a case no real model has.

**Q3_K.** Sixteen 6-bit scales `sc_s` (`s = 0 … 15`, 16 weights each) are packed into 12 bytes: scale
`s = 4g + i` takes its low four bits from byte `4 (g & 1) + i` (low nibble for `g < 2`, high nibble for `g >= 2`)
and its upper two bits from bits `2g, 2g+1` of byte `8 + i`. The quantized value of weight `l` of the 32-weight
unit `u = 4n + j` (`n = 0, 1` the 128-weight half, `j = 0 … 3` the bit pair) is
`((qs[32 n + l] >> 2j) & 3) - (hmask[l] bit u ? 0 : 4)`, i.e. `-4 … 3`; the 16 weights `l < 16` use scale
`sc_{2u}` and `l >= 16` use `sc_{2u+1}`.

**Worked examples** (the first weights of a block, decoded by the same reference the tests use):

```
Q4_1    d = 0.5,  m = -1,  qs[0] = 0x73, qs[1] = 0xF0
          w0  = 3 · 0.5 - 1   =  0.5      w1  =  0 · 0.5 - 1  = -1       (low nibbles of qs[0], qs[1])
          w16 = 7 · 0.5 - 1   =  2.5      w17 = 15 · 0.5 - 1  =  6.5     (high nibbles)
Q5_0    d = 0.25, qh = bit 0 and bit 16 set, qs[0] = 0x73
          w0  = (3 + 16 - 16) · 0.25 = 0.75     w1 = (0 - 16) · 0.25 = -4  (its bit 1 is clear)
          w16 = (7 + 16 - 16) · 0.25 = 1.75
IQ4_NL  d = 0.01, qs[0] = 0x9F, qs[1] = 0x30
          w0 = kvalues[15] d = 113 · 0.01 = 1.13      w1 = kvalues[0] d = -127 · 0.01 = -1.27
          w16 = kvalues[9] d = 13 · 0.01 = 0.13       w17 = kvalues[3] d = -65 · 0.01 = -0.65
MXFP4   e = 127 (2^-1), qs[0] = 0xA5, qs[1] = 0x7F
          w0 = kvalues[5] · 0.5 = 6 · 0.5 = 3          w1 = kvalues[15] · 0.5 = -12 · 0.5 = -6
          w16 = kvalues[10] · 0.5 = -2 · 0.5 = -1      w17 = kvalues[7] · 0.5 = 12 · 0.5 = 6
```

**Is the arithmetic exact?** Every product in these formulas is exactly representable in an f32 (`q`, `kvalues`, `sc`
are integers of at most 7 bits and `d`, `m` have an 11-bit mantissa; IQ4_XS's `d (ls - 32) · kvalues` needs 24 bits,
which just fits), so a fused multiply-add and a separate multiply and add give the same float. That is why the
dequantizers of the Java reference (`model/Ggml.java`), the Vector API kernels, the numpy references of
`scripts/metal-emu/ggml_ref.py` and llama.cpp's C code agree bit for bit
(`GgufKernelsCheck`, `ggml_ref.py <libggmlq>`: 2048–12 288 random-block weights per type, zero differences), and why
the GPU kernels can differ from them only in the order in which a dot product is summed.

**Dot products.** On the CPU the float path dequantizes in registers like Q4_0 does (6.8); the int8-activation path
(6.8, `cpu/GgufInt8Kernels.java`) multiplies the integer codes with the quantized activation and applies the scales
per block: Q4_1 / Q5_1 use `Σ (d q + m) x = d Σ q x + m Σ x` with the activation block sum the K-quants already compute;
the table formats look the nibbles up with a byte shuffle (`tbl`) and multiply them as Q8_0 bytes. On Metal each of
these formats is one more branch of `gg_unit` (`kernels.metal`, "GGUF block types, second set"), which decodes 32
consecutive weights into 8 `float4`, so every kernel family (decode matvec, tiled and matrix-unit prefill, the MoE
expert kernels, the embedding lookup) has them without further code. Which type is used in place and which is converted
is in [16.4](16-gguf.md#164-the-block-formats) and [16.6](16-gguf.md#166-zero-copy-how-the-kernels-read-the-files-pages).
