# 11. Performance — the roofline, where the time goes, and what to do next

Measured on an Apple M1 Pro (10 CPU cores, 16-core GPU, 32 GB unified memory, ~200 GB/s memory bandwidth).

---

## 11.0 The toolbox: the roofline model

Any computation needs two things from the machine: **arithmetic** (multiply-adds) and **bytes moved** from
memory into the compute units. A chip has a ceiling for each:

| M1 Pro | ceiling |
|---|---|
| memory bandwidth | ≈ 200 GB/s (shared by CPU and GPU) |
| GPU compute (f32) | ≈ 5 TFLOP/s |
| CPU compute (10 cores, NEON) | ≈ 0.5 TFLOP/s |

A program's **arithmetic intensity** is `FLOPs performed / bytes moved`. The machine's **ridge point** is
`compute / bandwidth`: for the GPU `5 × 10¹² / 2 × 10¹¹ = 25 FLOP per byte`. Below the ridge, the program
cannot use all the compute because data does not arrive fast enough — it is **memory-bound** and its speed is
`bandwidth / bytes`. Above it, it is **compute-bound** and its speed is `compute / FLOPs`. The "roofline" is
the plot of achievable speed against intensity: a rising line that flattens at the ridge.

### Decoding Qwen3-0.6B

Per generated token ([05 §5.11](05-transformer-math.md#511-shapes-and-cost-of-one-layer)):

```
FLOPs   ≈ 2 × 596 M      = 1.2 GFLOP               (multiply + add per weight)
bytes   = every weight once = 452 MB (Q4_0 layers + Q8_0 embedding) + KV cache reads
intensity = 1.2e9 / 4.52e8 ≈ 2.7 FLOP/byte          ≪ 25  →  memory-bound
ceiling = 200 GB/s / 452 MB ≈ 440 tokens/s
```

So no amount of faster arithmetic helps decode; only fewer bytes (quantization, [06](06-quantization.md)) or
more tokens per pass (batching) do. The compute side would allow ~4 000 tok/s.

### Prefilling

A chunk of 64 tokens reads the weights once but does 64× the arithmetic: intensity ≈ 170 FLOP/byte, above the
ridge → compute-bound, ceiling ≈ `5e12 / 1.2e9 ≈ 4 000 tok/s` (theoretical; real kernels reach a fraction).

### Qwen3-30B-A3B (MoE, [12](12-mixture-of-experts.md))

```
active parameters per token ≈ 3.3 B  →  FLOPs ≈ 6.6 GFLOP
bytes per token: attention 48 × 18.9 M × 0.625 B ≈ 0.57 GB
                 8 experts × 48 layers × 3 × 768 × 2048 × 0.625 B ≈ 1.13 GB
                 LM head 151 936 × 2048 × 1.125 B ≈ 0.35 GB
                 total ≈ 2.05 GB      →  ceiling ≈ 200 / 2.05 ≈ 97 tokens/s
```

Still memory-bound (intensity ≈ 3.2). Prefill is worse than for a dense model: 64 tokens pick ~8 × 64 expert
slots that cover nearly all 128 experts, so the whole 19.7 GB is streamed per chunk. The CPU backend groups the
slots by expert and reuses each dequantized expert row for all its tokens; the Metal `moe_up` / `moe_down`
kernels still get no reuse across tokens.

---

## 11.1 Measured

### Current numbers

Apple M1 Pro, Q4_0 weights (Q8_0 embedding / LM head), 413-token prompt, 128 generated tokens, after JIT warm-up
(`./jllm bench --ctx 1024`, otherwise idle machine, 2026-10-16):

| model | backend | prefill | decode | weight traffic at that decode rate | % of bandwidth ceiling |
|---|---|---|---|---|---|
| Qwen3-0.6B | **Metal** (f32 or f16 KV cache) | **~1 360 tok/s** (~533 with `-Djllm.metal.mmaPrefill=false`) | **~111 tok/s** | 50 GB/s | 25 % |
| Qwen3-0.6B | **CPU, int8 activations** (default) | **~275 tok/s** | **~51 tok/s** | 23 GB/s | 11 % |
| Qwen3-0.6B | CPU, float activations | ~237 tok/s | ~13 tok/s | 6 GB/s | 3 % |
| Qwen3-30B-A3B | **Metal** | **~49 tok/s** | **~25 tok/s** | 51 GB/s (≈ 2.05 GB per token) | 26 % |
| Qwen3-30B-A3B | **CPU, int8 activations** | **~12 tok/s** | **~6 tok/s** | 12 GB/s | 6 % |

Short prompts decode faster still (less attention work). Metal and CPU-with-int8-off produce identical greedy
text; int8 activations change the last bits of the logits, so their greedy text can differ on close calls
([07 §7.14](07-cpu-backend.md#714-prefill-and-decode-are-not-bit-identical-results-depend-on-history)).

### History (earlier versions, for context)

| version | prefill | decode | what changed afterwards |
|---|---|---|---|
| Metal Q4_0, first version (one kernel per op, ~340 dispatches per token) | 344 tok/s | 75 tok/s | fused norm / q·k·v / SwiGLU kernels, 170 dispatches ([08 §8.4](08-metal-backend.md#84-recording-a-forward-pass-metalbackendforward)) |
| Metal Q8_0, first version | 338 tok/s | 77 tok/s | — |
| Metal Q4_0, fused kernels, before tiled prefill and pipelined command buffers | ~380 tok/s | ~105 tok/s | tiled prefill matmuls, commit every 8 layers ([08 §8.10](08-metal-backend.md#810-the-tiled-prefill-matmuls)) |
| Metal Qwen3-30B-A3B, same stage | ~44 tok/s | ~24 tok/s | the same |
| CPU Q4_0, float activations, first version | 153 tok/s | 17 tok/s | tiled prefill, parallel element-wise passes (prefill), int8 activations (decode) |

The striking row of the first Metal version was Q4 vs Q8: Q8 moves 1.6× the bytes yet decoded at the same speed.
That proved the GPU path was **not** at the memory roof: dispatch count and latency dominated, which is what the
fusion work attacked. Even now decode sits at about a quarter of the roof ([11.2](#112-where-the-time-goes)).

### CPU int8 activations (`backend.cpu.int8-activations`, [07 §7.10](07-cpu-backend.md#710-int8-activations-integer-dot-products-int8kernels))

The two settings differ only in decode (prefill keeps the float `dot4x4` tiles):

| | prefill | decode | decode speedup |
|---|---|---|---|
| float activations (`false`) | ~237 tok/s | ~13 tok/s | — |
| int8 activations (`true`, default) | ~275 tok/s | ~51 tok/s | **~3.9×** |

An earlier A/B run on a loaded machine (interleaved, best of two, load average 5–17) gave 13.4 and 53.7 tok/s
(4.0×) with the same prefill, so the ratio is stable. Per kernel, single-threaded and undisturbed, the Q4 row
dot product goes from 3.5 ms to 0.59 ms per 8 192 × 1 024 rows (6×) and the Q8 (LM head) row dot from 1.3–3.7 ms
to 0.92 ms; the end-to-end decode gain is smaller because attention, norms, SiLU and the thread-pool
synchronization are untouched (Amdahl). The accuracy cost is an RMS relative error of ~0.9 % on each dot product
(`SmokeTest.int8Kernels`, activations with outliers; ~0.6 % without). Greedy outputs with the switch on and
off: *"What is the capital of France? Answer in one sentence."* gives the identical sentence; the haiku
prompt gives a different but equally coherent continuation (the int8 run actually produces a haiku, the
float run a prose paragraph) — with a 0.6B model, greedy decoding amplifies any logit perturbation of this
size into a different continuation after the first near-tie, which is the expected behaviour of two
equally valid quantization noises, not a degradation.

### Qwen3-30B-A3B

Qwen3-30B-A3B-Thinking-2507 (Q4_0, 19.4 GB cache, same 413-token prompt): Metal **~49 tok/s prefill, ~25 tok/s
decode**; CPU (int8 activations) ≈ 12 tok/s prefill and ≈ 6 tok/s decode. Decode at 25 tok/s means ~51 GB/s of
expert traffic (8 of 128 experts per layer, ≈ 2 GB per token) — the same ~25 % of the bandwidth ceiling as the
dense model, so the MoE path is as efficient per byte as the dense one. Prefill is bandwidth-bound by streaming
nearly all experts per chunk, which is why it is only ~2× decode here versus ~4.8× for the dense model; on the
CPU the expert-grouped prefill reuses each dequantized expert row for all tokens routed to it
([07 §7.7](07-cpu-backend.md#77-mixture-of-experts-on-cpu-cpubackendmoe-route)), on Metal that is still open
([11.5](#115-optimization-roadmap)).

---

## 11.2 Where the time goes

### Metal decode (~9 ms per token at 413-token context)

1. **Dispatch count.** 6 kernels per layer × 28 + 2 = 170 dispatches per token (386 for the 30B MoE model; the
   first version had 338). Each costs a few microseconds of GPU front-end time even when its work is tiny, and a
   kernel cannot start before the previous one has finished (serial encoder).
2. **Under-filled GPU.** `o_proj` / `down_proj` have 1024 output rows → 256 threadgroups of 4
   simdgroups; the 16-core GPU wants thousands in flight to hide memory latency. Latency, not bandwidth, bounds
   these matmuls.
3. **The LM head.** `151 936 × 1024` in Q8_0 = 165 MB, over a third of all bytes, for one 1024-vector.
4. **Attention.** For each of 28 layers × 16 heads the kernel streams the whole 413-position KV prefix
   (413 × 128 × 4 B × 2 = 423 KB per head) through one threadgroup; writing scores to device memory and
   reading them back twice costs extra traffic. The opt-in flash decoding (`-Djllm.metal.flashDecode=true`,
   [08 §8.11](08-metal-backend.md#811-flash-decoding-opt-in)) spreads long contexts over more threadgroups.
5. **CPU-side work per token**: sampling over 151 936 logits (~0.3 ms), the embedding row copy, and the
   `commitAndWait` round trip. The command buffers are committed every 8 layers, so encoding overlaps GPU work and
   only the last part is exposed.

### CPU decode (~77 ms per token with float activations, ~20 ms with int8)

`dotQ4` with 4 float lanes: per 32-weight block, one 16-byte load, two byte-wise ops, eight `castShape`
conversions and eight FMAs — about 30 instructions for 32 multiply-adds. 596 M weights / 32 = 18.6 M blocks →
~560 M instructions per token. The float CPU path is **instruction-bound**, not memory-bound (6 GB/s used of
~200). AVX-512 would run the same source with 4× fewer instructions per block. The int8 activation
path ([07 §7.10](07-cpu-backend.md#710-int8-activations-integer-dot-products-int8kernels)) replaces the
conversions with shift-and-mask unpacking on 8 short lanes and cuts the kernel's time by 6×; what remains
of the decode step is attention, the element-wise passes and pool synchronization.

### CPU prefill (153 tok/s originally, ~275 tok/s now)

Rows are dequantized once per 4-row group and reused across the 64 tokens by `dot4x4` / `dot4`, which is why
prefill is ~5× faster per token than int8 decode and ~18× faster than float decode. Attention is tiled over
(kv head, token tile) so every K / V row is read once per tile ([07 §7.5](07-cpu-backend.md#75-attention-kvcacheattention)),
and the scalar `exp` / `silu` loops are still visible here (`-Djllm.cpu.fastmath` vectorizes the softmax's).

---

## 11.3 Memory footprint

| component | Qwen3-0.6B | Qwen3-30B-A3B |
|---|---|---|
| quantized weights (mmap'd `jllm-q4_0.bin`, off-heap) | 452 MB | ≈ 19.7 GB |
| KV cache, f32, per 1 024 tokens of `--ctx` | 28 layers × 1024 × 1024 × 8 B = 235 MB | 48 × 1024 × 512 × 8 B = 201 MB |
| KV cache at `--ctx 4096` | 940 MB | 805 MB |
| KV cache with `backend.kv-dtype=f16` (2 B per element) | 117 MB per 1 024 tokens; **470 MB** at `--ctx 4096` | 101 MB per 1 024 tokens; **403 MB** at `--ctx 4096` |
| KV cache, DeepSeek-V2-Lite (MLA latent cache, [14 §14.11](14-model-families.md#1411-deepseek-v2--v3-multi-head-latent-attention--shared-experts)) | 27 layers × 1024 × (512 + 64) × 4 B = 64 MB per 1 024 tokens; **255 MB** at `--ctx 4096`, 127 MB at f16 (the expanded per-head cache of the first version was 553 KB per token, 2.2 GB at `--ctx 4096`); plus 216 MB of dequantized `kv_b_proj` halves | — |
| attention score scratch (Metal) `min(32, batch) × heads × ctx × 4 B` | 32 × 16 × 4096 × 4 = 8.4 MB | 32 × 32 × 2048 × 4 = 8.4 MB |
| activations, logits, MoE slot buffers | < 50 MB | < 100 MB |
| JVM + classes | ≈ 100 MB | ≈ 100 MB |

The KV cache scales linearly with `--ctx`: `--ctx 1024` is enough for short chats and cuts it by 4×; `--ctx 16384`
on the 0.6B model needs 3.8 GB. On the CPU backend the cache lives on the Java heap (`float[][]`), so it must
fit under `-Xmx`; on Metal it is in shared buffers outside the heap. `backend.kv-dtype=f16` halves it on both
backends ([05 §5.9](05-transformer-math.md#59-prefill-decode-and-the-kv-cache)): measured on the 0.6B model at
`--ctx 4096`, the CPU backend fails to start with `-Xmx1g` at f32 (`OutOfMemoryError` allocating the cache) and
runs with `-Xmx1g` at f16 (`-Xmx640m` still starts but spends its time in GC; `-Xmx1280m` is the practical floor
for f32).

Speed with the f16 cache. Today the f16 cache decodes and prefills at about the f32 speed (~533 / ~111 tok/s on
Metal). The table below is the original measurement, taken before the fusion and tiling work, on a machine shared
with other jobs (treat differences under ~10 % as noise):

| | f32 | f16 | outputs |
|---|---|---|---|
| Metal `bench`, `--ctx 4096`, 413-token prompt | 342–345 tok/s prefill, 65–73 tok/s decode | 347–349 tok/s prefill, 64–74 tok/s decode | identical |
| Metal `run`, 623-token prompt, 119 generated | 55.6 tok/s decode | 63.0 tok/s decode | identical |
| CPU `bench`, `--ctx 4096` (heavily contended run) | 25 tok/s prefill, 2.2 tok/s decode | 25 tok/s prefill, 2.4 tok/s decode | identical |

At a few hundred tokens of context attention is a small share of a decode step on both backends
([8.7](08-metal-backend.md#87-measured-numbers-and-where-the-time-goes), [7.9](07-cpu-backend.md#79-measured-numbers-and-the-bottleneck)),
so halving its traffic is within noise there; the option is about memory and about long contexts (at 3 000
tokens Metal prefill is a third faster and decode ~10 % faster, [08 §8.9](08-metal-backend.md#89-the-f16-kv-cache)).
The f32 path is unchanged: same dispatches on Metal, one extra `arraycopy` per layer on the CPU, identical tokens.

---

## 11.4 JVM flags and tunables

Set by the `jllm` launcher:

| flag | why |
|---|---|
| `--add-modules jdk.incubator.vector` | the Vector API is an incubator module; without it `VectorKernels` does not load |
| `--enable-native-access=ALL-UNNAMED` | silences the warning for FFM restricted methods (`Linker`, `SymbolLookup`) |
| `-XX:+UseParallelGC` | almost nothing is allocated during generation; ParallelGC has the lowest overhead |
| `-Xmx4g` (`JLLM_XMX`) | room for the CPU KV cache at `--ctx 4096`; raise it for larger contexts on CPU, or halve the cache with `backend.kv-dtype=f16` |

Optional system properties, passed through `JAVA_OPTS` (for example `JAVA_OPTS="-Djllm.cpu.fastmath=true" ./jllm bench`):

| property | default | effect |
|---|---|---|
| `jllm.cpu.fastmath` | off | vectorized `exp` / `tanh` in the attention softmax and soft-caps, not bit-identical ([07 §7.15](07-cpu-backend.md#715-fast-math-jllmcpufastmath)) |
| `jllm.cpu.deterministic` | off | prefill through the single-token kernels, for exact comparisons; slower ([07 §7.14](07-cpu-backend.md#714-prefill-and-decode-are-not-bit-identical-results-depend-on-history)) |
| `jllm.metal.flashDecode` | off | flash decoding for decode steps over `jllm.metal.flashMinLen` (512) positions ([08 §8.11](08-metal-backend.md#811-flash-decoding-opt-in)) |
| `jllm.metal.commitLayers` | 8 | layers per Metal command buffer (0 = one per pass) |
| `jllm.metal.prefillMinTokens` | 8 | smallest chunk that uses the tiled prefill matmuls |
| `jllm.metal.mmaPrefill` | true | simdgroup-matrix prefill matmuls; `false` = the scalar tiled kernels (bit-identical to decode) |
| `jllm.metal.mmaMinTokens` | 32 | smallest chunk that uses the simdgroup-matrix matmuls ([08 §8.10](08-metal-backend.md#810-the-tiled-prefill-matmuls)) |
| `jllm.metal.attnGroup` | 1 | query heads per attention threadgroup (2, 4, 8 measured no faster) |

JIT warm-up matters: the first ~50 tokens run before C2 has compiled the kernels; `bench` discards a warm-up
round for that reason.

---

## 11.5 Optimization roadmap

Done:

| change | effect | where |
|---|---|---|
| **Kernel fusion on Metal**: `rmsnorm` folded into the following matmul at decode, q/k/v as one dispatch (with bias), `silu_mul` in the gate/up epilogue, f16 cache rows written by the projection and rope kernels, Gemma's final soft-cap in the LM-head epilogue | 338 → 170 dispatches per token; decode 75 → 111 tok/s on the 0.6B model | `kernels.metal`, `MetalKernels`, [08 §8.4](08-metal-backend.md#84-recording-a-forward-pass-metalbackendforward) |
| **Pipelined command buffers**: commit every 8 layers so encoding overlaps GPU work | part of the 380 → 533 tok/s prefill gain | `MetalBackend.forward` |
| **Simdgroup-matrix prefill matmuls on Metal** (64 rows × 32 tokens per threadgroup, f32 MMA, not bit-identical) | prefill 533 → ~1 360 tok/s; largest logit difference 8.6e-6 | `matmul_mma` family, [08 §8.10](08-metal-backend.md#810-the-tiled-prefill-matmuls) |
| **Tiled prefill matmuls on Metal** (2 rows × 8 tokens per simdgroup, bit-identical) | prefill 380 → 533 tok/s | `matmul_pf` family, [08 §8.10](08-metal-backend.md#810-the-tiled-prefill-matmuls) |
| **Int8 activations on CPU** ([07 §7.10](07-cpu-backend.md#710-int8-activations-integer-dot-products-int8kernels)) | Q4 row kernel 6× faster, decode 13 → 51 tok/s (~3.9×), ~0.9 % RMS dot-product error | `cpu/Int8Kernels`, `CpuMatmul` |
| **f16 KV cache**: `backend.kv-dtype=f16` ([07 §7.13](07-cpu-backend.md#713-the-f16-kv-cache-kvcache), [08 §8.9](08-metal-backend.md#89-the-f16-kv-cache)) | halves cache memory (940 → 470 MB at `--ctx 4096`) and attention read traffic; speed equal or better, outputs unchanged on the test prompts | `cpu/KvCache`, `_h` kernels |
| **Expert-grouped MoE prefill on CPU** ([07 §7.7](07-cpu-backend.md#77-mixture-of-experts-on-cpu-cpubackendmoe-route)) | each expert row dequantized once per chunk and reused for its tokens | `CpuBackend.moeBatched` |
| **Tiled CPU attention** over (kv head, token tile) | each K / V row read once per tile; bit-identical | `KvCache.attentionTiled` |
| **Flash decoding** on Metal, opt-in ([08 §8.11](08-metal-backend.md#811-flash-decoding-opt-in)) | +3–5 % decode at ~3 000 tokens of context, not bit-identical, so off by default | `attention_split`, `attention_combine` |
| **Vectorized `exp` / `tanh`** on CPU, opt-in (`jllm.cpu.fastmath`) | +5–9 % decode at ~3 000 tokens | `VectorKernels.expFast` |
| **MLA latent cache** (DeepSeek): absorbed `W_kb`, one 576-wide row per token | 2.2 GB → 255 MB at `--ctx 4096` ([14 §14.11](14-model-families.md#1411-deepseek-v2--v3-multi-head-latent-attention--shared-experts)) | `MlaCpu`, `MlaMetal` |

Remaining, by expected payoff:

| # | change | expected effect | where |
|---|---|---|---|
| 1 | **Continuous batching**: several sequences per forward pass so each weight read serves many tokens | turns decode from memory-bound toward compute-bound: near-linear throughput scaling with concurrent users | `Backend.forward` (batch of sequences), per-sequence KV cache slots, `Engine` scheduler, `ApiServer` lock |
| 2 | **Expert-grouped MoE prefill on Metal**: gather the tokens routed to each expert and run one matmul per expert over them | MoE prefill from "stream all experts, no reuse" to dense-like reuse; the CPU backend already does it | `moe_up` / `moe_down` kernels (+ gather / scatter) |
| 3 | **Speculative decoding** with a small draft model (e.g. 0.6B drafting for 30B-A3B), verified in one batched forward of the big model | 1.5–3× decode when the draft agrees often | new `Engine` loop, `Backend.forward` with verification of `k` tokens |
| 4 | **Flash-style prefill attention**: online softmax that never writes `scores` to device memory, a tile of query tokens per threadgroup (flash decoding covers decode only) | removes the score round trip at long context | `attention_t` |
| 5 | **`simdgroup_matrix`** (Apple's MMA) for prefill matmuls | different accumulation order, so opt-in; prefill gain | `kernels.metal` |
| 6 | **Mixed-precision low-bit formats** (a Q2_K / Q3_K mix with higher-precision sensitive tensors) | smaller files and fewer bytes per token ([13](13-q2k-quantization.md)) | quantizer, `QType`, kernels |
| 7 | Smaller items: fuse RMSNorm into the next matmul on CPU, wider threadgroups for small Metal matmuls, coalesced router weight reads, parallel shard downloads | each a few percent to 2× in its own corner | `CpuBackend`, `kernels.metal`, `HuggingFaceHub` |

A future direction, not started: a Windows backend (for example OpenCL or DirectML) so the GPU / NPU of a
Snapdragon X machine could be used the way Metal is used on Apple Silicon; the `Backend` interface already keeps
the model logic out of the kernel layer.

---

## 11.6 Summary

- Decode is bound by **bytes per token**, not arithmetic: 452 MB (0.6B) or ~2 GB (30B-A3B) per token against
  ~200 GB/s. Quantization and batching are the only levers that move that ceiling.
- Today's Metal decode reaches ~25 % of the bandwidth roof (~111 tok/s of a ~440 tok/s ceiling); the gap is
  dispatch overhead and latency-bound small matmuls. Prefill is ~1 360 tok/s.
- The CPU path was instruction-bound at 128-bit SIMD; int8 activations with integer dot products took the
  row kernel from ~30 conversion-heavy instructions per block to ~31 single-cycle integer ops doing twice
  the lanes, 6× per kernel and ~4× per decoded token (13 → 51 tok/s).
- Memory is dominated by the weights and the KV cache (`--ctx` and `backend.kv-dtype` are the knobs).
