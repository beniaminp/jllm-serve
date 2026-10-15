# 6. Quantization: Q4_0 and Q8_0

Code: `model/QType.java`, `model/QTensor.java`

## Why

Decoding one token reads every weight once ([05](05-transformer-math.md#the-kv-cache-and-prefill-vs-decode)).
Speed is therefore roughly `memory bandwidth / model size`. Halving the bytes doubles the ceiling. Storing a
weight in 4 bits instead of 16 cuts the file 4×, and it turns out LLM weights tolerate this surprisingly well,
*as long as the scale is chosen locally*.

## The idea: a shared scale per small block

A 4-bit integer can hold 16 values. We cannot represent 1024 different floats with it directly, but we can
represent them as `integer × scale` if the scale fits the local range. The formats used here (same as
llama.cpp's `Q4_0`/`Q8_0`) split each matrix row into **blocks of 32 consecutive weights** and store one
`float32` scale per block.

### Q8_0 (8 bits, used for the embedding / LM head)

```
d = max |x_j| / 127                   the scale
q_j = round(x_j / d)                   an integer in [-127, 127]
x_j ≈ q_j · d
```

Storage per block: 32 bytes + 4 (scale) = 36 bytes for 32 weights → **1.125 bytes/weight** (vs 2 for bf16).

### Q4_0 (4 bits, used for all layer matrices)

```
m = the x_j with the largest magnitude (sign kept)
d = m / −8                             so that m itself maps to exactly −8
q_j = clamp( round(x_j / d) + 8, 0, 15 )   an unsigned nibble, 8 = zero
x_j ≈ (q_j − 8) · d
```

Storage: 16 bytes (two nibbles per byte) + 4 = 20 bytes per 32 weights → **0.625 bytes/weight**.
Mapping the extreme value to −8 (not ±7) uses the asymmetric range of 4-bit two's complement fully; this is the
llama.cpp trick and it measurably lowers error.

Nibble layout within a block (`QTensor.quantizeBlockQ4`): byte `j` holds weight `j` in its low nibble and
weight `j + 16` in its high nibble. This lets a SIMD kernel load 16 bytes and get two 16-wide vectors with a mask
and a shift ([07](07-cpu-backend.md)).

## Memory layout in `QTensor`

```
┌────────────── rows × blocksPerRow × blockBytes ──────────────┬─ 64-align ─┬── rows × blocksPerRow × 4 ──┐
│ quantized blocks, row-major                                  │            │ float32 scales, row-major   │
└──────────────────────────────────────────────────────────────┴────────────┴─────────────────────────────┘
```

Scales are kept in a separate region rather than interleaved so that both regions are naturally aligned and the
GPU can bind them as two buffers (`wq` and `ws` in `kernels.metal`). Memory is allocated off-heap with
`Arena.allocate(size, 16384)`: **16 KiB page alignment** matters because Metal's `newBufferWithBytesNoCopy`
only wraps page-aligned memory without copying ([08](08-metal-backend.md)).

Quantization happens once, in parallel over rows (`IntStream.range(0, rows).parallel()`), reading each row from
the memory-mapped safetensors (`RowSource`), and the result is written to `jllm-q4_0.bin`
([03 §3.5](03-safetensors.md#35-from-safetensors-to-jllms-quantized-cache)). Qwen3-0.6B converts in ~2 s; after
that the file is memory-mapped and start-up is instant.

## How much error?

Per block the maximum quantization error is `d / 2`, i.e. `max|x| / 16` for Q4_0 and `max|x| / 254` for Q8_0.
Because every block has its own scale, an outlier weight only hurts its 31 neighbours. In practice Q4_0 costs
Qwen3-0.6B a small but measurable amount of accuracy — this is why:

- the **embedding / LM head stays Q8_0** regardless of `--quant`: the final logits are directly exposed to the
  error and small models are sensitive there;
- **activations** (`x`, `q`, `k`, `v`, the KV cache) are kept in float32 — only weights are quantized;
- `--quant q8` is available for a near-lossless baseline.

Greedy outputs on both backends in this project agree with each other exactly, and match the kind of answers
the full-precision model gives.

## Sizes for Qwen3-0.6B

| Part | Params | bf16 | Q4_0 (+Q8 embed) |
|---|---|---|---|
| embedding (shared with LM head) | 155.6 M | 311 MB | 175 MB (Q8_0) |
| 28 layers | 440 M | 880 MB | 275 MB |
| **total** | 596 M | 1.19 GB | **≈ 450 MB** |

## The dequantization kernel

Reading a Q4_0 row back (`dequantizeRow`) is: for each byte, `lo = (b & 0xF) − 8`, `hi = (b >> 4) − 8`, multiply
by the block scale. The compute kernels never materialize a full dequantized matrix; they expand one block at a
time in registers and immediately multiply with the activation — which is where the next two documents start.
