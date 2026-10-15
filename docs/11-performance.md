# 11. Performance notes

## The roofline, in two lines

Decoding one token touches every weight once: ~450 MB for Q4_0 Qwen3-0.6B. The arithmetic is ~1.2 GFLOP. An
M1 Pro does several TFLOP/s on the GPU but has ~200 GB/s of memory bandwidth, so the *theoretical* ceiling is
`200 GB/s ÷ 450 MB ≈ 440 tokens/s`; the compute would allow thousands. Decode is **bandwidth-bound**, which is
why 4-bit weights matter and why bigger batches (prefill) are so much cheaper per token.

## Measured (Apple M1 Pro, 10 cores, 413-token prompt, 128 generated tokens)

| | Prefill | Decode | Weight traffic at decode rate |
|---|---|---|---|
| Metal Q4_0 | 344 tok/s | 75 tok/s | 34 GB/s |
| Metal Q8_0 | 338 tok/s | 77 tok/s | 54 GB/s |
| CPU Q4_0 (NEON, 10 threads) | 153 tok/s | 17 tok/s | 7.6 GB/s |

On a short prompt Metal decodes at ~110 tok/s; the attention cost grows with context (the 413-token context's
scores are recomputed per layer per token).

Q4 and Q8 decode at nearly the same speed on Metal, which tells us the GPU path is *not yet* bandwidth-bound: the
time goes to the ~340 kernel launches per token and to the small, latency-bound matrices (a 1024×1024 matmul is
only 256 threadgroups — a fraction of the GPU).

## Where the time goes

**Metal decode (~13 ms/token):**
- dispatch overhead: ~340 dispatches × a few µs;
- under-filled GPU on `k_proj`/`v_proj`/`o_proj`/`down_proj` (1024 rows → 256 threadgroups of 4 rows);
- the LM head `151936 × 1024` Q8 matmul is ~165 MB alone, a third of all traffic;
- attention reads the full KV prefix per head per layer.

**CPU decode (~59 ms/token):** instruction-bound in `dotQ4` — with only 4 float lanes, each 32-weight block costs
a load, two byte ops, eight `castShape` conversions and eight FMAs. 128-bit NEON is the worst case for this
design; on AVX-512 the same code runs 4× fewer instructions per block.

## Obvious next steps, in order of payoff

1. **Fuse kernels on Metal**: `rmsnorm` into the following matmul; q/k/v into one dispatch; `silu_mul` into the
   gate/up matmul epilogue. Cuts dispatches roughly in half.
2. **Int8 dot products on CPU**: quantize the activation vector to Q8 per block too, then use integer
   multiply-add (`ByteVector`/`ShortVector` lanes: 16 per 128-bit register instead of 4 floats). This is what
   llama.cpp does and it is the single biggest CPU win.
3. **Flash-style attention**: compute scores and the weighted sum in one pass with online softmax, never
   materializing `scores`; and store the KV cache in float16 to halve its traffic.
4. **Continuous batching** in the engine: several sequences per forward call so the weights read once serve
   many tokens — turns decode from bandwidth-bound into compute-bound.
5. **Speculative decoding** with a tiny draft model.

## Memory footprint

| Component | Size |
|---|---|
| Q4_0 weights + Q8_0 embedding (off-heap) | ~450 MB |
| KV cache, `--ctx 4096`, f32 | ~940 MB (CPU: Java heap; Metal: shared buffers) |
| activations, scratch, logits | < 50 MB |
| JVM | ~100 MB |

`--ctx 1024` brings the KV cache to 235 MB for short chats.

## JVM flags that matter

- `--add-modules jdk.incubator.vector`: without it the Vector API classes are missing and nothing compiles.
- `--enable-native-access=ALL-UNNAMED`: silences the FFM restricted-method warning.
- `-XX:+UseParallelGC -Xmx4g`: the CPU KV cache is on the heap; ParallelGC has the least overhead for this
  allocation pattern (almost nothing is allocated during generation).
- JIT warm-up: the first ~50 tokens run slower until the kernels are compiled to native code; `bench` does a
  warm-up round before measuring.
