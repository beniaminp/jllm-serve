# 13. Q2_K: 2-bit weights with sub-block scales and mins

Code: `model/Q2K.java`, `model/QType.java`, `model/QTensor.java`, `cpu/Q2KKernels.java`, the `Q2_K` section of
`metal/kernels.metal`, `metal/MetalBackend.java`

[06](06-quantization.md) stored every weight as `integer × scale` with 4 or 8 bits per integer. This chapter goes
down to **2 bits** — four levels per weight — which is where the simple "one scale per block" recipe stops
working and llama.cpp's *k-quant* machinery (super-blocks, mins, two-level scales) becomes necessary. The format
built here is `Q2_K`, byte-for-byte the same as llama.cpp's `block_q2_K`, at **2.625 bits per weight**. It is
selected with `--quant q2k`.

Two bits per weight is meant for large models, where the many thousands of terms in every dot product average
the error away. On Qwen3-0.6B, applied to every linear layer, it does not merely degrade the model — it breaks
it (13.7 has the measurements and the per-tensor analysis of why); the point of implementing it is the 30B
MoE, whose experts dominate the memory footprint ([12](12-mixture-of-experts.md)): 9.8 GB instead of 18.7 GB,
with the fragile tensors kept at 4 bits.

---

## 13.0 The toolbox

### Four levels are not enough for a symmetric grid

With `b` bits a weight can take 2^b values. Q4_0's recipe (`w ≈ q·d`, `q` signed) with `b = 2` would give
`q ∈ {−2, −1, 0, 1}`: one positive level, two negative, and the whole block's scale set by its largest
magnitude. Half of the weights of a typical block are positive and would all collapse onto one level (or
zero). The error of such a grid is about the same as replacing every weight by `sign(w) · amax/2`.

### Mins: an offset per group

The fix is an **affine** grid: `w ≈ scale · q − min`, `q ∈ {0, 1, 2, 3}`. The four levels are
`−min, −min + scale, −min + 2·scale, −min + 3·scale`: an evenly spaced ladder that starts at the group's
smallest value and ends at its largest, whatever the sign distribution. The sign convention is llama.cpp's:
`min ≥ 0` is *subtracted*, so the lowest level is `−min` (and the group minimum is clamped to ≤ 0 so that `0`
is always representable, which keeps sparse rows sparse).

### Why groups of 16 and a second level of scales

Four levels can only describe a group of weights well if the group is small enough to have a tight range. Q2_K
uses **sub-blocks of 16**. But a sub-block now needs *two* parameters (scale and min); storing them as floats
would cost 8 bytes per 16 weights = 4 bits per weight — more than the 2 bits of data. So they are themselves
quantized, to **4 bits each**, relative to two shared **super-block** scales `d` and `dmin` that cover 16
sub-blocks = 256 weights:

```
sub-block j:   scale_j = d · sc_j          sc_j ∈ 0 … 15    (4 bits)
               min_j   = dmin · m_j        m_j  ∈ 0 … 15    (4 bits)
weight:        w       = d·sc_j·q − dmin·m_j                 q ∈ 0 … 3  (2 bits)
```

`d` and `dmin` are stored as **f16** (half precision: 1 sign, 5 exponent, 10 mantissa bits, ~3 significant
decimal digits, the same `float16` as in [03](03-safetensors.md)). Java converts with
`Float.floatToFloat16` / `Float.float16ToFloat`.

### Bits per weight

```
per super-block of 256 weights:
   quants       256 × 2 bits         =  64 bytes      2.000 bits/weight
   sc_j, m_j    16 × (4 + 4) bits    =  16 bytes      0.500 bits/weight
   d, dmin      2 × 16 bits          =   4 bytes      0.125 bits/weight
                                       ----------
                                        84 bytes      2.625 bits/weight  =  0.328125 bytes/weight
```

Compared with Q4_0 (0.625 B/weight) a Q2_K matrix is 47.5 % of the size; compared with bf16 it is 16.4 %.
Because decode is memory-bound ([06 §6.0](06-quantization.md#memory-bandwidth-why-size-is-speed)) the speed
ceiling rises by the same ratio — in exchange for a much larger error per weight (13.7).

---

## 13.1 The super-block, byte by byte

`Q2K.BLOCK = 256` weights, `Q2K.BLOCK_BYTES = 84`:

```
offset  0 ┌──────────────────────────────────────────────────────────────────────────────┐
          │ scales[16]   one byte per sub-block j:   bits 0–3 = sc_j,  bits 4–7 = m_j     │
offset 16 ├──────────────────────────────────────────────────────────────────────────────┤
          │ qs[64]       2-bit quants, four per byte, interleaved (see below)             │
offset 80 ├──────────────────────────────────────────────────────────────────────────────┤
          │ d      f16 little-endian  (super-block scale for the sc_j)                     │
offset 82 ├──────────────────────────────────────────────────────────────────────────────┤
          │ dmin   f16 little-endian  (super-block scale for the m_j)                      │
offset 84 └──────────────────────────────────────────────────────────────────────────────┘
```

### The quant interleave

The 64 quant bytes are two **halves** of 32 bytes, one per 128 weights. Within a half, byte `l` (0 … 31) holds
four weights that are 32 apart, at bit positions 0, 2, 4, 6:

```
half h (weights 128h … 128h+127), byte l:

   bits 7 6 │ 5 4 │ 3 2 │ 1 0
        ────┼─────┼─────┼────
   weight 128h + l + 96 │ + 64 │ + 32 │ + 0          shift 6 │ 4 │ 2 │ 0
```

Seen from the sub-blocks: at a given shift, bytes 0–15 are one sub-block and bytes 16–31 the next one. So
sub-block `j` (weights `16j … 16j+15`) is found at

```
byte offset   16 + (j / 8) · 32 + (j % 2) · 16          (+ l for weight 16j + l)
bit shift     2 · ((j % 8) / 2)
```

| j | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | … | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| half | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | … | 1 |
| bytes | 0–15 | 16–31 | 0–15 | 16–31 | 0–15 | 16–31 | 0–15 | 16–31 | 32–47 | 48–63 | … | 48–63 |
| shift | 0 | 0 | 2 | 2 | 4 | 4 | 6 | 6 | 0 | 0 | … | 6 |

Why this odd order? It is what a SIMD loop wants: load the 32 bytes of a half **once** as two 16-byte vectors,
then `(v >> shift) & 3` with shift 0, 2, 4, 6 produces the eight sub-blocks of the half as eight vectors of
0…3 — no shuffles, no second load, exactly the same trick as Q4_0's nibble split
([06 §6.3](06-quantization.md#nibble-packing)) applied twice.

---

## 13.2 Dequantization

`Q2K.dequantizeBlock` is llama.cpp's `dequantize_row_q2_K` loop, written per sub-block:

```
d    = f16 at offset 80
dmin = f16 at offset 82
for j in 0 … 15:                                   (only the first n/16 for a partial super-block, 13.6)
    s   = scales[j]
    dl  = d · (s & 0xF)                            sub-block scale
    ml  = dmin · (s >> 4)                          sub-block min
    q   = qs + 16 + (j/8)·32 + (j%2)·16
    for l in 0 … 15:
        w[16j + l] = dl · ((q[l] >> shift_j) & 3) − ml
```

Every weight of a sub-block is one of just four numbers `−ml, dl − ml, 2dl − ml, 3dl − ml`.

---

## 13.3 Quantization: choosing scale and min per sub-block

Quantizing a 16-weight sub-block means choosing `(scale, min)` and the codes `q_l` to minimize the
reconstruction error. For a given `(scale, min)` the best code is `q = clamp(round((w + min)/scale), 0, 3)`;
the question is which `(scale, min)`.

**Closed form.** Map the group's range onto the four levels: `min = −min(w)` (clamped so the minimum is ≤ 0),
`scale = (max(w) − min(w)) / 3`. The extremes land exactly on levels 0 and 3; everything in between rounds to
the nearest of four evenly spaced values, error ≤ `scale/2`.

**Search.** That is not always optimal: letting one outlier be clipped can give the other 15 values a finer
grid. `Q2K.makeQkx2Quants` implements llama.cpp's `make_qkx2_quants` (with its Q2_K parameters
`rmin = −0.5, rdelta = 0.1, nstep = 15`): starting from the closed form it tries 16 candidate step sizes
`iscale = (2.5 + 0.1·is) / (max − min)`, `is = 0 … 15`, i.e. grids from `(max−min)/4` to `(max−min)/2.5`;
for each it rounds the weights to codes, then **re-fits** `(scale, min)` to those codes by weighted least
squares (the normal equations of `w ≈ scale·q − min`), clamps `min ≥ 0`, and keeps the candidate if it lowers
the weighted sum of absolute errors `Σ |w_l| · |ŵ_l − w_l|`. The weights `|w_l|` make large-magnitude weights
count more, since they contribute more to every dot product. This is the choice implemented and documented
here; the closed form alone is the `is = 5` candidate without the re-fit.

**Quantizing the 16 scales and mins.** With `scale_j, min_j` for the 16 sub-blocks (`Q2K.quantizeBlock`):

```
d    = f16( max_j scale_j / 15 )        sc_j = round( scale_j / d_exact )  ∈ 0 … 15
dmin = f16( max_j min_j   / 15 )        m_j  = round( min_j  / dmin_exact ) ∈ 0 … 15
```

so the largest scale and the largest min of the super-block get the full 4-bit code 15, and the others are
rounded relative to it (a sub-block whose scale is a tenth of the largest gets `sc = 2` — coarse; this is the
price of 0.5 bits/weight for the scales and the reason k-quants with more bits use 6-bit sub-scales).

**Recomputing the codes.** The stored `d·sc_j` differs slightly from the searched `scale_j`, so the final codes
are recomputed against the *dequantized* scale and min, `q_l = clamp(round((w_l + dmin·m_j) / (d·sc_j)), 0, 3)`.
Storage is therefore self-consistent: decoding uses exactly the numbers the codes were chosen for. A sub-block
whose `sc_j` rounds to 0 gets codes 0 (all its weights decode to `−dmin·m_j`).

Conversion speed: the search is ~20 passes over each sub-block; Qwen3-0.6B converts in a few seconds on all
cores (`QTensor.quantize` is parallel over rows, with a per-thread scratch so nothing is allocated per block).

---

## 13.4 Full worked example

The first 16 weights of row 0 of `model.layers.0.self_attn.k_proj.weight` of Qwen3-0.6B (sub-block 0 of
super-block 0; the numbers below are what `Q2K.quantizeBlock` computes on them):

```
l :    0        1        2        3        4        5        6        7
w : −0.01660 −0.07617 −0.03015  0.03345  0.05713  0.02039  0.02576  0.04321
l :    8        9       10       11       12       13       14       15
w : −0.05859 −0.01465  0.07764  0.01587  0.04102  0.00656 −0.02808  0.02527
```

**Step 1 — scale and min of the sub-block.** `min(w) = −0.07617`, `max(w) = 0.07764`, so the closed form gives
`scale = (0.07764 + 0.07617) / 3 = 0.051270`, `min = 0.076172`. The 16-candidate search finds nothing with a
lower weighted error on this sub-block, so these are kept. Codes `q = round((w + 0.076172) / 0.051270)`:

```
q : 1 0 1 2 3 2 2 2 0 1 3 2 2 2 1 2
```

**Step 2 — the other 15 sub-blocks** of the super-block go through the same search; their scales range from
0.0200 to 0.0513 (the largest is this sub-block's) and their mins from 0.0167 to 0.0861 (largest: sub-block 2).

**Step 3 — super-block scales.**

```
d    = max scale / 15 = 0.051270 / 15 = 0.0034180   → f16 0x1B00 = 0.0034180  (exactly representable here)
dmin = max min   / 15 = 0.086141 / 15 = 0.0057427   → f16 0x1DE1 = 0.0057411  (f16 rounding: −0.03 %)
sc_0 = round(0.051270 / 0.0034180) = 15
m_0  = round(0.076172 / 0.0057427) = round(13.26) = 13
scales[0] = sc_0 | m_0 << 4 = 15 | 13 << 4 = 0xDF
```

**Step 4 — recompute codes against the stored values** `dl = d·sc_0 = 0.0034180 · 15 = 0.051270`,
`ml = dmin·m_0 = 0.0057411 · 13 = 0.074635`. The four levels are now
`−0.07463, −0.02337, 0.02790, 0.07917`, and `q = clamp(round((w + 0.074635) / 0.051270), 0, 3)`:

```
 l       w      q      ŵ = dl·q − ml     error
 0   −0.01660   1      −0.02337        −0.00676
 1   −0.07617   0      −0.07463        +0.00154
 2   −0.03015   1      −0.02337        +0.00679
 3    0.03345   2       0.02790        −0.00554
 4    0.05713   3       0.07917        +0.02205   ← half-way between levels 2 and 3
 5    0.02039   2       0.02790        +0.00752
 6    0.02576   2       0.02790        +0.00215
 7    0.04321   2       0.02790        −0.01531
 8   −0.05859   0      −0.07463        −0.01604
 9   −0.01465   1      −0.02337        −0.00872
10    0.07764   3       0.07917        +0.00154
11    0.01587   2       0.02790        +0.01204
12    0.04102   2       0.02790        −0.01311
13    0.00656   2       0.02790        +0.02134
14   −0.02808   1      −0.02337        +0.00471
15    0.02527   2       0.02790        +0.00264
max |error| = 0.02205 ≈ dl / 2 = 0.02564 ✓   (compare Q4_0 on similar data: ≈ 0.004)
```

Over the whole super-block: max error 0.0273, RMS error 0.0073 against an RMS weight of 0.0275 — a **27 %
relative RMS error** per weight (30 % over whole tensors, 13.7), versus 8.7 % for Q4_0 on the same tensors.
Whether a model survives that depends on how much redundancy its layers have
([06 §6.9](06-quantization.md#69-what-the-error-does-to-the-model)); 13.7 shows that a 0.6B model does not.

**Step 5 — the 84 bytes.**

```
offset  0  scales:  df 57 fb 8a 88 47 97 98 88 79 38 56 66 77 88 77
offset 16  qs:      c9 f4 5d 52 db 6e ee 2e a4 e5 6f ee ee 66 69 a6     half 0, bytes 0–15
offset 32            d9 68 91 68 c7 96 26 82 dc 83 d5 b4 eb 43 22 b4     half 0, bytes 16–31
offset 48            52 b5 1e 60 66 f5 76 57 75 b6 e1 b3 c5 60 83 b5     half 1, bytes 0–15
offset 64            05 d4 53 61 b9 aa 7d a3 9c 7a ad 2f 15 a1 46 d4     half 1, bytes 16–31
offset 80  d, dmin: 00 1b e1 1d                                          0x1B00, 0x1DE1 little-endian
```

Reading byte 0 of `qs`, `0xC9 = 1100 1001₂`: bits 1–0 = `01` → weight 0 has code 1 ✓ (table above); bits 3–2 =
`10` → weight 32 has code 2; bits 5–4 = `00` → weight 64 has code 0; bits 7–6 = `11` → weight 96 has code 3.
Weight 1 is in byte 1 (`0xF4`, bits 1–0 = `00` → code 0 ✓), and weight 16 (sub-block 1) is in byte 16
(`0xD9`, shift 0). The `Q2KCheck` smoke test builds rows from randomly chosen `d, dmin, sc_j, m_j, q` and
checks they come back exactly, which pins this layout down.

---

## 13.5 The dot product

### The algebra

`Σ_c W[c] · x[c]` over a sub-block, with `W = d·sc·q − dmin·m`:

```
Σ_l (d·sc·q_l − dmin·m) · x_l  =  d·sc · Σ_l q_l·x_l   −   dmin·m · Σ_l x_l
```

The first term multiplies the raw codes with the activation — no dequantization per weight. The second term
only needs the **sum of the activation over the sub-block**, which does not depend on the weights at all. Over
a super-block:

```
dot = d · Σ_j sc_j · (q·x)_j   −   dmin · Σ_j m_j · (Σx)_j
```

so the per-weight work is "convert a 0…3 code to float, multiply-add with x" — the same as Q4_0 minus the
`−8`. (With int8 activations, as llama.cpp uses on ARM, this is a much bigger win: `q·x` becomes an integer
dot product and `Σx` per sub-block is precomputed once per activation vector as the "block sums". With f32
activations as in this project, the saving is the per-weight `fma(dl, q, −ml)` that direct dequantization
would need; the `Σx` adds cost about as much, so the two forms are close — the trick's real value here is that
the code lines up with the int8 path described in [07 §7.10](07-cpu-backend.md#710-int8-activations-integer-dot-products-int8kernels).)

### CPU: `Q2KKernels.dot`

Per super-block, with 4-lane NEON (`FL = 4`):

```
d, dmin ← two f16 reads
accQ = accM = 0                                         (lane vectors, like dotQ4's acc)
for half h in 0, 1:
    v0 = 16 bytes at qs + 32h,  v1 = 16 bytes at qs + 32h + 16     two loads for 128 weights
    for shift in 0, 2, 4, 6:                                       k = shift/2
        s0, s1 ← scales[8h + 2k], scales[8h + 2k + 1]
        q0 = (v0 >>> shift) & 3,  q1 = (v1 >>> shift) & 3          byte vectors of codes
        accQ += (s0 & 0xF) · subDot(q0, x[…])  + (s1 & 0xF) · subDot(q1, x[… + 16])
        accM += (s0 >>> 4) · subSum(x[…])      + (s1 >>> 4) · subSum(x[… + 16])
acc = accQ · d  −  accM · dmin        (fma into the row accumulator)
…
return acc.reduceLanes(ADD)           once per row
```

`subDot` is `halfDot` from [07 §7.3](07-cpu-backend.md#73-the-core-kernel-dotq4-block-by-block) (four
`castShape` conversions and four FMAs for 16 weights); `subSum` adds the four `x` vectors. Scales `sc_j`, `m_j`
and codes are small integers, so all these products are exact in f32; the rounding differences between this
kernel and the scalar `Q2K.dotRow` are at the 1e-7 level (the smoke test allows 1e-4 of `Σ|w·x|`).
`Q2KKernels.dequant` is the same unpacking with `q.fma(dl, −ml)` written to a float buffer for the prefill
path ([07 §7.4](07-cpu-backend.md#74-decode-vs-prefill-matmul-cpubackendmatmul)).

### Metal: `mm_row_q2k`, `block_dot<2>`

The GPU has no register-width reason to interleave, so `q2k_sub_block` simply dequantizes one 16-weight
sub-block into four `float4` (`(q >> shift) & 3) · dl − ml`) and the kernels `dot()` them with the activation,
as the Q4_0 path does with its `lo/hi`. Every quantized matmul on Metal — the plain `matmul`, the fused q/k/v
`matmul3`, the fused `matmul_swiglu`, with and without the folded rmsnorm — is one template on the
quantization type `QT` whose inner loop is `mm_row<QT>`; `mm_row` hands `QT == 2` to `mm_row_q2k`, so one
function covers all of them and the `_q2k` / `_q2k_norm` pipelines are plain instantiations
(`matmul<2, false>`, `matmul3<2, true>`, …). The one Q2_K-specific decision is the **work unit**: the Q4_0
loop gives each lane of a simdgroup one 32-weight block, and a 1024-wide row has 32 of them — a perfect fit. A
1024-wide Q2_K row has only **4 super-blocks**; one super-block per lane would leave 28 of 32 lanes idle.
`mm_row_q2k` therefore strides over 16-weight sub-blocks (`u = lane, lane + 32, …`, 64 per 1024-wide row),
computing `blk = wq + (row·nsb + u/16)·84` and `j = u % 16` for each. For the MoE kernels, `block_dot<2>`
keeps the 32-weight block interface (`moe_up`/`moe_down` are templates on it) and dequantizes sub-blocks `2b`
and `2b+1`. The `ws` scale buffer argument is unused — `MetalBackend.scaleOff` binds offset 0 for Q2_K tensors
because there is no scale region to point at, and `MetalBackend.byType` picks the `_q2k` pipeline from the
tensor's type (the Mamba backend reuses both).

---

## 13.6 A Q2_K `QTensor`

`QType.Q2_K` has `blockSize = 256`, `blockBytes = 84` and `scaleBytes = 0`: the generic arithmetic of
[06 §6.4](06-quantization.md#64-the-memory-layout-of-a-qtensor) becomes

```
nb             = ceil(cols / 256)                super-blocks per row
rowOffset(r)   = r × nb × 84
scalesOffset   = align(rows × nb × 84, 64)       = end of the data; the scale region has size 0
byteSize       = align(scalesOffset, 16384)
```

**Partial super-blocks.** Every linear layer of Qwen3-0.6B (cols 1024, 2048, 3072) and Qwen3-30B-A3B
(2048, 4096, 768) is a multiple of 256, and such rows are laid out exactly as llama.cpp would (llama.cpp
refuses other shapes). jllm additionally accepts any `cols` that is a multiple of 32: the last super-block of a
row then holds only `(cols mod 256) / 16` sub-blocks and the rest of its 84 bytes are zero. The SIMD and GPU
loops only touch whole super-blocks (or, on the GPU, only sub-blocks below `cols/16`); the tail goes through
the scalar helpers. This is what lets the 64-wide tiny MoE test model run with `--quant q2k` (at 10.5
bits/weight — a test, not a use case).

**Sizes.**

| model | linear weights | Q4_0 | Q2_K | embedding/LM head (Q8_0, unchanged) | cache file |
|---|---|---|---|---|---|
| Qwen3-0.6B | 440.4 M | 275.3 MB | 144.5 MB | 175.0 MB (tied) | q4: 452 MB → **q2k: 321 MB** (measured) |
| Qwen3-30B-A3B | 29.90 G | 18.69 GB | 9.81 GB | 2 × 350 MB | q4: ≈ 19.4 GB → **q2k: ≈ 10.5 GB** |

For the 0.6B the Q8_0 embedding is now the largest tensor (55 % of the file), so the overall saving is only
29 %. For the 30B MoE the experts are 97 % of the weights and the file nearly halves; at 10.5 GB the model
fits comfortably next to a 4096-token KV cache in 16 GB of unified memory.

---

## 13.7 Measurements

All numbers from a 10-core M1 Pro (32 GB) with Qwen3-0.6B, greedy decoding (`--temp 0`).

### Correctness

| check | result |
|---|---|
| on-grid rows (random `d, dmin, sc_j, m_j, q`) quantize → dequantize | exact (max error 0.0), for 1024-wide rows and 320-wide rows with a partial super-block |
| SIMD `Q2KKernels.dequant` vs scalar `Q2K.dequantizeBlock` | identical |
| SIMD `Q2KKernels.dot` vs scalar dequantize-then-dot | max difference 4e-8 of `Σ|w·x|` |
| Metal `matmul_q2k`, `matmul_q2k_norm` (folded rmsnorm), SwiGLU-fused `moe_up_q2k`, `moe_down_q2k` vs the CPU kernels (random expert tensors, 3 tokens, top-2) | max difference 7e-8 of `Σ|w·x|` |
| byte layout vs llama.cpp (16 rows of `k_proj` through `gguf-py 0.18.0`'s `dequantize(…, Q2_K)`) | bit-identical floats |
| 48 greedy tokens, CPU vs Metal | byte-identical output |
| tiny MoE test model (`hidden 64`, partial super-blocks everywhere), two CPU runs | byte-identical output; the Metal backend rejects this model's `headDim = 16` independently of the format |

### Size

```
jllm-q4_0.bin   452 132 864 B   (0.6B)
jllm-q2_k.bin   321 388 544 B   = 71 %; the linear layers alone shrink from 275 MB to 145 MB (52.5 %)
conversion      11 s on 10 cores (Q4_0: ~2 s — the make_qkx2 search is ~20 passes per sub-block)
```

### Speed

`jllm bench` (413-token prefill, 128-token decode, 3 rounds). **Caveat:** these were measured while two
unrelated conversions kept the machine at a load average of 67–143 on 10 cores, so absolute numbers are 2–10×
below the quiet-machine figures in [11](11-performance.md) and swing by 2× between rounds; `q4` and `q2k`
were interleaved so they saw the same conditions, and ranges over all rounds are given.

| backend | format | prefill tok/s | decode tok/s | reading |
|---|---|---|---|---|
| Metal | Q4_0 | 160 – 345 | 28 – 76 | |
| Metal | Q2_K | 168 – 337 | 32 – 57 | no faster than Q4_0 |
| CPU | Q4_0 | 36 – 41 | 2.8 – 3.1 | |
| CPU | Q2_K | 28 – 35 | 1.8 – 2.3 | ≈ 30 % slower than Q4_0 |

Reading fewer bytes did not buy speed on this model, and that is expected from [11](11-performance.md):

- On Metal the 0.6B decode is bound by the ≈ 340 kernel dispatches per token (12 per layer), not by the
  450 MB of weights (which at 200 GB/s would take 2.3 ms, i.e. > 400 tok/s); on a quiet machine Q4_0 reaches
  75 tok/s = 17 % of the bandwidth roof. Shrinking the weights to 321 MB cannot help a
  latency-bound loop; the Q2_K kernel also does a little more per lane (scale byte, `d`/`dmin`, shift and
  mask) and runs 2 sub-blocks per lane where `matmul_q4` runs 1 block. Where bandwidth *is* the limit — the
  30B MoE's 10–19 GB of expert weights — the byte count is what matters and Q2_K's 47 % should translate
  almost directly into decode speed.
- On the CPU the kernel is conversion-bound ([07 §7.9](07-cpu-backend.md#79-measured-numbers-and-the-bottleneck)):
  each weight still costs one byte→float `castShape` and one FMA regardless of its bit width, and Q2_K adds the
  four shift-and-mask steps per 32 bytes and the `Σx` accumulation. Fewer bytes to load, same amount of
  arithmetic — slightly slower. The int8-activation path in [07 §7.10](07-cpu-backend.md#710-int8-activations-integer-dot-products-int8kernels)
  is where 2-bit codes would pay off on the CPU, because the min term then folds into precomputed block sums
  and the codes feed integer dot products directly.

### Quality — the honest part

The same prompt as in the rest of these docs, *"What is the capital of France? Answer in one sentence."*,
greedy, pure Q2_K on every linear layer (embedding/LM head Q8_0 as always):

```
 이상;

 القر：

 tưởng：

 mơ：

：

：
… (punctuation and random CJK/Arabic/Vietnamese fragments for all 48 tokens; identical on CPU and Metal)
```

Not "degraded" — destroyed. To quantify it, teacher-forced perplexity on a 97-token English paragraph
(`PplProbe`, one forward per token):

| weights | perplexity | top-1 agreement with Q8_0 |
|---|---|---|
| Q8_0 | 7.16 | — |
| Q4_0 | 7.37 | 70 / 96 |
| **Q2_K** | **35 165** | 6 / 96 |

Is it the kernels or the numbers? Two experiments settle it. First, the per-tensor reconstruction error is a
uniform **30 % RMS** for every linear tensor of every layer (Q4_0: 8.7 %), with no sub-block whose 4-bit scale
collapsed to 0 or 1 — exactly the error the format predicts for Gaussian-like weights (four levels across a
~3.5σ span give a step of ~1.2σ and an RMS error of step/√12 ≈ 0.34σ). Second, re-encoding the *dequantized
Q2_K values* as Q8_0 and running them through the unchanged Q8_0 kernels gives perplexity 39 905: the Q2_K
values are what breaks the model, not the Q2_K code path.

Which tensors are the sensitive ones? Perplexity with Q2_K values in a subset of tensors only (the rest Q8_0):

| Q2_K values in … | perplexity |
|---|---|
| `q_proj`, `k_proj` only | 9.2 |
| `v_proj`, `o_proj` only | 23.9 |
| `gate_proj`/`up_proj` only | 25.4 |
| **`down_proj` only** | **205** |
| all attention | 77 |
| all MLP | 1 438 |
| only layers 0–3 (24 tensors) | 947 |
| everything except layers 0–3 | 736 |
| everything | 35 165 |

The damage multiplies across groups, `down_proj` and the first layers are by far the most fragile, and q/k
barely notice 2 bits. This is precisely why llama.cpp's "Q2_K" *model preset* is not pure Q2_K: it keeps
`attn_v` at Q4_K, `ffn_down` of the early layers at Q4_K/Q3_K, `output` at Q6_K, and strongly recommends an
importance matrix (calibration data that re-weights the `make_qkx2` search per column). Qwen3-0.6B is also the
worst case for 2 bits — 1024-wide dot products, no redundancy. The format in this project is the pure one; a
mixed recipe (`down_proj` and the first layers at Q4_0, the rest Q2_K) would be a 10-line change in
`ModelConverter` and is the obvious next step before using `q2k` on the 30B MoE, where the experts (97 % of
the bytes) are the ones to shrink and the attention/down projections can stay Q4_0 at little cost.

---

## 13.8 What the other k-quants change

Q2_K is the smallest member of llama.cpp's k-quant family. All of them share the 256-weight super-block with
f16 `d` (and `dmin` where there are mins) and differ in the bits per code and per sub-scale:

| format | code bits | sub-block | sub-scale bits | mins | bytes / 256 | bits/weight | what changes in the kernel |
|---|---|---|---|---|---|---|---|
| Q2_K | 2 | 16 | 4 + 4 | yes | 84 | 2.625 | this chapter |
| Q3_K | 3 | 16 | 6 | no | 110 | 3.4375 | codes are 2 low bits in `qs` plus a high bit in a separate 32-byte `hmask`; the 6-bit scales are packed across 12 bytes and are *signed* (levels −4 … 3, no min) |
| Q4_K | 4 | 32 | 6 + 6 | yes | 144 | 4.5 | the 6-bit scales/mins of 8 sub-blocks are packed into 12 bytes; nibbles are split like Q4_0 (`j`, `j+32`) |
| Q5_K | 5 | 32 | 6 + 6 | yes | 176 | 5.5 | Q4_K plus a 32-byte `qh` holding the fifth bit of every code |
| Q6_K | 6 | 16 | 8 | no | 210 | 6.5625 | 4 low bits in `ql`, 2 high bits in `qh`, signed 8-bit sub-scales, no `dmin` |

The pattern: fewer code bits need *mins* (an affine grid) and smaller sub-blocks; more code bits can afford a
symmetric grid and spend their sub-scale budget on precision (6 or 8 bits) instead. Q4_K at 4.5 bits/weight is
the usual sweet spot — Q4_0's size plus 0.125 bits, with roughly half its error because the per-32 scale no
longer has to be a power-of-two-free f32 but a 6-bit code under a shared `d`, and because the min absorbs the
asymmetry of each group. Implementing it in this project would reuse everything here (super-block
`QType`, partial-block handling, the `Σx` dot) with a different `quantizeBlock`/`dequantizeBlock` pair and a
different unpacking in the kernels.

---

## 13.9 Code map

| what | where |
|---|---|
| constants, byte offsets, `quantizeBlock`, `makeQkx2Quants`, scalar `dequantizeBlock`/`dotBlock` | `model/Q2K.java` |
| `QType.Q2_K` (`blockSize 256`, `blockBytes 84`, `scaleBytes 0`), `parse("q2k")` | `model/QType.java` |
| per-type block arithmetic, `quantize`/`dequantizeRow` dispatch | `model/QTensor.java` |
| SIMD `dot` and `dequant` | `cpu/Q2KKernels.java` (called from `VectorKernels.dotRow`/`dequantRow`) |
| `q2k_sub_block`, `mm_row_q2k` (hooked from `mm_row` for `QT == 2`), `block_dot<2>`, the `_q2k`/`_q2k_norm` instantiations of `matmul`, `matmul3`, `matmul_swiglu`, `moe_up`, `moe_down` | `metal/kernels.metal`, section `Q2_K` at the end |
| pipeline selection by `QTensor.type` (`byType`), scale binding (`scaleOff`) | `metal/MetalBackend.java`, `metal/MambaMetalBackend.java` (the MLA and shared-expert paths call `MetalBackend.matmul`/`swiglu`) |
| `jllm-q2_k.bin` / `.json` (type name per tensor; embeddings stay `Q8_0`) | `model/QuantFile.java`, `model/ModelConverter.java` (unchanged) |
| round trip, error bound, SIMD vs scalar | `src/test/java/org/robomiri/jllm/Q2KCheck.java` (from `SmokeTest`) |

---

## 13.10 Summary

- Two bits give four levels; they are only usable as an affine ladder `scale·q − min` over a small group, so
  Q2_K quantizes 16-weight sub-blocks with their own scale and min.
- Those 32 parameters per 256 weights are themselves 4-bit codes under two f16 super-block scales `d`, `dmin`:
  64 + 16 + 4 = 84 bytes per 256 weights = 2.625 bits/weight.
- The 2-bit codes are interleaved so that two 16-byte loads and four shift-and-mask operations unpack 128
  weights; the dot product uses `d·Σ sc·(q·x) − dmin·Σ m·(Σx)` so codes multiply activations directly.
- The quantizer is llama.cpp's `make_qkx2_quants` search (closed-form start, 16 candidate grids, weighted
  least-squares re-fit), and the stored codes are recomputed against the 4-bit-rounded scale and min.
- Per-weight error is 30 % RMS (vs 8.7 % for Q4_0). Pure Q2_K on every layer breaks Qwen3-0.6B (perplexity
  35 000 vs 7.4 for Q4_0); `down_proj` and the first layers are the fragile tensors, q/k are nearly free. The
  cache file is 321 MB for the 0.6B and ≈ 10.5 GB for the 30B MoE, where a mixed recipe is the way to use it.
