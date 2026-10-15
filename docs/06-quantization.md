# 6. Quantization: Q4_0 and Q8_0 — with all the math

Code: `model/QType.java`, `model/QTensor.java`, `model/QuantFile.java`, `model/ModelConverter.java`

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

Halving the bytes doubles the ceiling. (The measured numbers are below the ceilings for reasons covered in
[11](11-performance.md), but the ratio holds.) The second reason is capacity: the 30B MoE model is 61 GB in bf16
and does not fit in 32 GB of RAM at all; at ~0.65 bytes/weight it is 20 GB and does.

---

## 6.1 The idea: one scale per block of 32

Pick a block of 32 consecutive weights in a row. Find the largest magnitude in the block, call it `amax`.
Choose a scale `d` so that `amax / d` lands exactly on the largest representable integer. Every weight is then
stored as the nearest integer to `w / d`, and reconstructed as `q × d`. Error is at most half a step, `d / 2`.

Why 32 and not the whole row? Because weights are not uniformly sized: a row of 1024 may contain one 0.3
among values around 0.02. A single row-wide scale would be set by the 0.3 and squash the 0.02s into one or two
levels. With 32-weight blocks, that outlier only hurts its 31 neighbours. The cost is 4 bytes of scale per
block — 1 bit per weight on top of the 4 or 8 data bits. (These are exactly llama.cpp's `Q4_0` and `Q8_0`
formats, so the numbers are comparable with the rest of the ecosystem.) In code: `QType.BLOCK = 32`.

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

A quantized `[rows, cols]` matrix is one contiguous off-heap `MemorySegment`:

```
offset 0 ───────────────────────────────────────────────────────────────────────┐
│ quantized blocks, row-major:                                                 │
│   row 0: block 0 | block 1 | … | block nb−1       nb = cols / 32              │
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
rowOffset(r)   = r × nb × blockBytes                 first block of row r
scaleOffset(r) = scalesOffset + r × nb × 4           first scale of row r
scalesOffset   = align(rows × nb × blockBytes, 64)
byteSize       = align(scalesOffset + rows × nb × 4, 16384)
```

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
arguments (`wq`, `ws` in `kernels.metal`) at two offsets of the same allocation.

The 16 KiB page alignment (`QTensor.PAGE`) is what allows Metal to wrap the memory with
`newBufferWithBytesNoCopy` instead of copying it ([08](08-metal-backend.md)); it costs at most 16 KiB of padding
per tensor.

---

## 6.5 Which tensors get which format

| tensor | format | why |
|---|---|---|
| embedding / LM head | **Q8_0 always** | the LM head's error goes straight into the logits; small models are measurably worse with a Q4 head. Also only read once per token (one row for the embedding; all rows for the head). |
| attention `q/k/v/o`, FFN `gate/up/down`, MoE experts | `--quant` (Q4_0 default, Q8_0 optional) | the bulk of the bytes; errors average out over many dot products |
| norms, QK-norm gains, MoE router | f32 | tiny (1 KB – 1 MB each); the router decides which experts run, so its precision matters |
| activations, KV cache | f32 | computed on the fly, never stored on disk; quantizing them is a separate (future) optimization |

Sizes that result (`ModelWeights.byteSize()`):

| model | parameters | bf16 | jllm Q4_0 cache | jllm Q8_0 cache |
|---|---|---|---|---|
| Qwen3-0.6B (tied embedding) | 596 M | 1 190 MB | **452 MB** (175 Q8 embed + 275 Q4 layers + padding) | ~670 MB |
| Qwen3-30B-A3B (separate `lm_head`) | 30.5 B | 61 000 MB | **≈ 19.7 GB** (2 × 350 MB Q8 embed/head + 18.1 GB Q4 experts + 0.9 GB Q4 attention) | would not fit in 32 GB |

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

`QTensor.quantize(arena, type, rows, cols, rowSource)` allocates the segment, then runs
`IntStream.range(0, rows).parallel()` — each row is quantized independently (one thread-local `float[cols]`
buffer per worker), reading its source values through the memory-mapped safetensors
([03 §3.4](03-safetensors.md#34-memory-mapping-with-the-ffm-api)). All cores participate; Qwen3-0.6B converts
in ~2 s, the 30B MoE in a few minutes dominated by reading 61 GB from disk.

This happens **once**: `ModelConverter` writes every tensor's segment verbatim into `jllm-q4_0.bin` and the
offsets into `jllm-q4_0.json` ([03 §3.5](03-safetensors.md#35-from-safetensors-to-jllms-quantized-cache)).
Afterwards `ModelWeights` memory-maps the `.bin` and wraps each tensor with

```java
QTensor.view(type, rows, cols, mapped.asSlice(offset, size))
```

— no quantization, no copy, no RAM until a row is touched. `view` checks that `size == byteSize(type, rows,
cols)` so a stale or truncated cache fails loudly.

---

## 6.8 Dequantizing: never materialize the matrix

Reading values back is `x̂ = (q − 8) · d` per weight (`QTensor.dequantizeRow`), but the hot kernels never expand
a whole matrix to floats: that would multiply memory traffic by 6. Instead:

- **decode** (`VectorKernels.dotQ4`): load 16 bytes, unpack to two 16-lane vectors *in registers*, multiply with
  the activation, accumulate; apply the scale once per block. Weight bytes are read exactly once.
- **prefill** (`VectorKernels.dequantRow` + `dot4`): expand 4 rows into a thread-local float buffer, then reuse
  them for all 64 tokens of the batch, so the unpacking cost is amortized.
- **Metal** (`block_dot<QT>` in `kernels.metal`): same as decode, one `uchar4` at a time.

Details in [07](07-cpu-backend.md) and [08](08-metal-backend.md).

---

## 6.9 What the error does to the model

Each quantized weight is off by up to `amax/16` (Q4_0). An output element is `Σ_c W[r][c] · x[c]` over 1024
terms; the errors `ε_c` are roughly independent with zero mean, so the error of the sum grows like `sqrt(1024)
= 32` while the sum itself grows like 1024: the *relative* error of a dot product is ~30× smaller than that of
an individual weight. That is why a 15 % per-weight perturbation yields outputs that are still mostly correct.

The standard way to measure the damage is **perplexity**: run a text through the model and compute how
surprised it is by the actual next tokens (lower is better). For Q4_0 on small models the increase is typically
a few percent; for Q8_0 it is below measurement noise. In practice:

- greedy outputs of this project's Q4_0 Qwen3-0.6B agree between CPU and Metal exactly and give the same kind of
  answers as the full-precision model; individual token choices can differ from bf16 on close calls;
- `--quant q8` is the near-lossless reference when you want to check whether an odd answer is a Q4 artifact;
- the embedding/LM head stay Q8_0 because the logits see those errors with no averaging across layers.

Activations (`x`, `q`, `k`, `v`) stay f32: they are recomputed per token and never stored, so there is no
memory to save — and the attention scores are sensitive to their precision.

---

## 6.10 Other formats, for orientation

| format | idea | not in this project because |
|---|---|---|
| **Q2_K** (llama.cpp "K-quant") | super-blocks of 256 with 4-bit sub-block scales *and mins*, 2.625 bits/weight | implemented: `--quant q2k`, see [13](13-q2k-quantization.md) |
| **Q4_K / Q5_K / Q6_K** (llama.cpp "K-quants") | super-blocks of 256 with 6-bit sub-scales and minimums, ~0.56–0.81 B/weight with lower error than Q4_0 | more complex kernels; Q4_0 is the simplest format that works ([13 §13.8](13-q2k-quantization.md#138-what-the-other-k-quants-change) compares them) |
| **GPTQ / AWQ** | choose the quantized values by minimizing the *output* error on calibration data, not per-weight rounding | needs calibration runs and a solver at conversion time |
| **int8 activations** (Q8_0 for `x` too) | integer × integer dot products, 4× more lanes per SIMD register | the next CPU optimization ([07 §7.10](07-cpu-backend.md#710-what-to-optimize-next)) |
| **fp8 / NF4** | 8-bit floats; 4-bit non-uniform "normal float" levels | GPU-oriented; no hardware fp8 on Apple M1 |

---

## 6.11 Summary

- A weight is stored as `integer × block scale`; 32 weights share a 4-byte scale. Q8_0: 1.125 B/weight, error
  ≤ amax/254. Q4_0: 0.625 B/weight, error ≤ amax/16, with the extreme value pinned to −8 to use all 16 levels.
- Nibbles are packed as (j, j+16) pairs so SIMD code gets two vectors from one 16-byte load.
- A `QTensor` is `[blocks][64-aligned scales][page padding]`; offsets are pure arithmetic, so a row, a block, or
  an expert is addressed without any lookup.
- Quantization happens once into `jllm-q4_0.bin`; afterwards the file is memory-mapped and used in place by
  both the CPU (Vector API) and the GPU (zero-copy Metal buffers).
- Decode speed ∝ 1 / bytes, which is the whole reason to do this.
