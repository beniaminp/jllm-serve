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
slots that cover nearly all 128 experts, so the whole 19.7 GB is streamed per chunk and the `moe_up` kernel
gets no reuse across tokens.

---

## 11.1 Measured

Qwen3-0.6B, 413-token prompt, 128 generated tokens, after JIT warm-up (`./jllm bench`):

| backend | prefill | decode | weight traffic at that decode rate | % of bandwidth ceiling |
|---|---|---|---|---|
| Metal Q4_0 | 344 tok/s | 75 tok/s | 34 GB/s | 17 % |
| Metal Q8_0 | 338 tok/s | 77 tok/s | 54 GB/s | 27 % |
| CPU Q4_0 (NEON 128-bit, 10 threads), float activations | 153 tok/s | 17 tok/s | 7.6 GB/s | 4 % |

Short prompts decode at ~110 tok/s on Metal (less attention work). Both backends produce identical greedy text.

### CPU int8 activations (`backend.cpu.int8-activations`, [07 §7.10](07-cpu-backend.md#710-int8-activations-integer-dot-products-int8kernels))

Measured with `./jllm bench --backend cpu --ctx 1024` on the same M1 Pro while the machine was shared with
other heavy jobs (load average 50–100, so the absolute numbers are far below the idle-machine row above);
the two settings were run interleaved, best of two runs each:

| | prefill | decode | decode speedup |
|---|---|---|---|
| float activations (`false`) | 38.8 tok/s | 4.2 tok/s | — |
| int8 activations (`true`) | 36.8 tok/s | 16.0 tok/s | **3.8×** |

Prefill is unchanged (it keeps the float `dot4` path). Per kernel, single-threaded and undisturbed, the
Q4 row dot product goes from 3.5 ms to 0.59 ms per 8 192 × 1 024 rows (6×) and the Q8 (LM head) row dot
from 1.3–3.7 ms to 0.92 ms; the end-to-end decode gain is smaller because attention, norms, SiLU and the
thread-pool synchronization are untouched (Amdahl), and because under this load the 10 threads did not get
10 cores. The accuracy cost is an RMS relative error of ~0.9 % on each dot product
(`SmokeTest.int8Kernels`, activations with outliers; ~0.6 % without). Greedy outputs with the switch on and
off: *"What is the capital of France? Answer in one sentence."* gives the identical sentence; the haiku
prompt gives a different but equally coherent continuation (the int8 run actually produces a haiku, the
float run a prose paragraph) — with a 0.6B model, greedy decoding amplifies any logit perturbation of this
size into a different continuation after the first near-tie, which is the expected behaviour of two
equally valid quantization noises, not a degradation.

Qwen3-30B-A3B: *to be measured* (expected: decode a few tens of tok/s on Metal; prefill bandwidth-limited by
streaming all experts per chunk).

The striking row is Q4 vs Q8 on Metal: Q8 moves 1.6× the bytes yet decodes at the same speed. That proves the
GPU path is **not** at the memory roof yet — something else dominates.

---

## 11.2 Where the time goes

### Metal decode (~13 ms per token at 413-token context)

1. **Dispatch count.** 12 kernels per layer × 28 + 2 ≈ 340 dispatches per token. Each costs a few
   microseconds of GPU front-end time even when its work is tiny → ~1–2 ms of pure overhead.
2. **Under-filled GPU.** `k_proj`/`v_proj`/`o_proj`/`down_proj` have 1024 output rows → 256 threadgroups of 4
   simdgroups; the 16-core GPU wants thousands in flight to hide memory latency. Latency, not bandwidth, bounds
   these matmuls.
3. **The LM head.** `151 936 × 1024` in Q8_0 = 165 MB, over a third of all bytes, for one 1024-vector.
4. **Attention.** For each of 28 layers × 16 heads the kernel streams the whole 413-position KV prefix
   (413 × 128 × 4 B × 2 = 423 KB per head) through one threadgroup; writing scores to device memory and
   reading them back twice costs extra traffic.
5. **CPU-side work per token**: sampling over 151 936 logits (~0.3 ms), the embedding row copy, and the
   `commitAndWait` round trip.

### CPU decode (~59 ms per token with float activations)

`dotQ4` with 4 float lanes: per 32-weight block, one 16-byte load, two byte-wise ops, eight `castShape`
conversions and eight FMAs — ~20 instructions for 32 multiply-adds. 450 MB / 20 B per block = 22.6 M blocks →
~450 M instructions per token per core-equivalent. The CPU is **instruction-bound**, not memory-bound (7.6 GB/s
used of ~200). AVX-512 would run the same source with 4× fewer instructions per block. The int8 activation
path ([07 §7.10](07-cpu-backend.md#710-int8-activations-integer-dot-products-int8kernels)) replaces the
conversions with shift-and-mask unpacking on 8 short lanes and cuts the kernel's time by 6×; what remains
of the decode step is attention, the element-wise passes and pool synchronization.

### CPU prefill (153 tok/s)

Rows are dequantized once per 4-row group and reused across the 64 tokens by `dot4`, which is why prefill is 9×
faster per token than decode. Attention and the scalar `exp`/`silu` loops become visible here.

---

## 11.3 Memory footprint

| component | Qwen3-0.6B | Qwen3-30B-A3B |
|---|---|---|
| quantized weights (mmap'd `jllm-q4_0.bin`, off-heap) | 452 MB | ≈ 19.7 GB |
| KV cache, f32, per 1 024 tokens of `--ctx` | 28 layers × 1024 × 1024 × 8 B = 235 MB | 48 × 1024 × 512 × 8 B = 201 MB |
| KV cache at `--ctx 4096` | 940 MB | 805 MB |
| KV cache with `backend.kv-dtype=f16` (2 B per element) | 117 MB per 1 024 tokens; **470 MB** at `--ctx 4096` | 101 MB per 1 024 tokens; **403 MB** at `--ctx 4096` |
| attention score scratch (Metal) `batch × heads × ctx × 4 B` | 64 × 16 × 4096 × 4 = 17 MB | 64 × 32 × ctx × 4 B |
| activations, logits, MoE slot buffers | < 50 MB | < 100 MB |
| JVM + classes | ≈ 100 MB | ≈ 100 MB |

The KV cache scales linearly with `--ctx`: `--ctx 1024` is enough for short chats and cuts it by 4×; `--ctx 16384`
on the 0.6B model needs 3.8 GB. On the CPU backend the cache lives on the Java heap (`float[][]`), so it must
fit under `-Xmx`; on Metal it is in shared buffers outside the heap. `backend.kv-dtype=f16` halves it on both
backends ([05 §5.9](05-transformer-math.md#59-prefill-decode-and-the-kv-cache)): measured on the 0.6B model at
`--ctx 4096`, the CPU backend fails to start with `-Xmx1g` at f32 (`OutOfMemoryError` allocating the cache) and
runs with `-Xmx1g` at f16 (`-Xmx640m` still starts but spends its time in GC; `-Xmx1280m` is the practical floor
for f32).

Speed with the f16 cache (M1 Pro, Qwen3-0.6B Q4_0, greedy; the machine was shared with other jobs, so treat
differences under ~10 % as noise):

| | f32 | f16 | outputs |
|---|---|---|---|
| Metal `bench`, `--ctx 4096`, 413-token prompt | 342–345 tok/s prefill, 65–73 tok/s decode | 347–349 tok/s prefill, 64–74 tok/s decode | identical |
| Metal `run`, 623-token prompt, 119 generated | 55.6 tok/s decode | 63.0 tok/s decode | identical |
| CPU `bench`, `--ctx 4096` (heavily contended run) | 25 tok/s prefill, 2.2 tok/s decode | 25 tok/s prefill, 2.4 tok/s decode | identical |

At a few hundred tokens of context attention is a small share of a decode step on both backends
([8.7](08-metal-backend.md#87-measured-numbers-and-where-the-time-goes), [7.9](07-cpu-backend.md#79-measured-numbers-and-the-bottleneck)),
so halving its traffic is within noise there; the option is about memory and about long contexts. The f32 path
is unchanged: same dispatches on Metal, one extra `arraycopy` per layer on the CPU, identical tokens.

---

## 11.4 JVM flags (set by the `jllm` launcher)

| flag | why |
|---|---|
| `--add-modules jdk.incubator.vector` | the Vector API is an incubator module; without it `VectorKernels` does not load |
| `--enable-native-access=ALL-UNNAMED` | silences the warning for FFM restricted methods (`Linker`, `SymbolLookup`) |
| `-XX:+UseParallelGC` | almost nothing is allocated during generation; ParallelGC has the lowest overhead |
| `-Xmx4g` | room for the CPU KV cache at `--ctx 4096`; raise it for larger contexts on CPU, or halve the cache with `backend.kv-dtype=f16` |

JIT warm-up matters: the first ~50 tokens run before C2 has compiled the kernels; `bench` discards a warm-up
round for that reason.

---

## 11.5 Optimization roadmap, by expected payoff

| # | change | expected effect | where |
|---|---|---|---|
| 1 | **Kernel fusion on Metal**: fold `rmsnorm` into the following matmul, launch q/k/v as one dispatch, fuse `silu_mul` into the gate/up matmul epilogue | ~340 → ~150 dispatches per token; decode toward 120–150 tok/s on the 0.6B model | `kernels.metal`, `MetalBackend.forward` |
| 2 | **Int8 activations on CPU** — done: the activation vector is quantized to Q8 per block and multiplied on 8 short lanes per 128-bit register with no byte→float conversions ([07 §7.10](07-cpu-backend.md#710-int8-activations-integer-dot-products-int8kernels)) | measured: Q4 row kernel 6× faster, decode 4.2 → 16.0 tok/s (3.8×) under the same load, prefill unchanged, ~0.9 % RMS dot-product error | `cpu/Int8Kernels`, `CpuBackend.matmul` (`n == 1`) and `moe`, `backend.cpu.int8-activations` |
| 3 | **f16 KV cache** — done: `backend.kv-dtype=f16` ([07 §7.13](07-cpu-backend.md#713-the-f16-kv-cache-kvcache), [08 §8.9](08-metal-backend.md#89-the-f16-kv-cache)) | halves cache memory (940 → 470 MB at `--ctx 4096`) and attention read traffic; decode speed unchanged to slightly better at 400–600-token contexts, greedy outputs unchanged on the test prompts | `cpu/KvCache`, `kv_store_half`/`attention_h` in `kernels.metal` |
| 4 | **Flash-style attention**: compute scores and the weighted value sum in one pass with an online softmax, never writing `scores` to device memory; one threadgroup per head *and* query tile | removes the score round trip; large win at long context | `attention` kernel, `CpuBackend.attention` |
| 5 | **Continuous batching**: several sequences per forward pass so each weight read serves many tokens | turns decode from memory-bound toward compute-bound: near-linear throughput scaling with concurrent users | `Backend.forward` (batch of sequences), per-sequence KV cache slots, `Engine` scheduler, `ApiServer` lock |
| 6 | **Expert-grouped MoE prefill**: gather the tokens routed to each expert and run one matmul per expert over them | MoE prefill from "stream all experts, no reuse" to dense-like reuse; 5–10× prefill on 30B-A3B | `moe_up`/`moe_down` kernels (+ gather/scatter), `CpuBackend.moe` |
| 7 | **Speculative decoding** with a small draft model (e.g. 0.6B drafting for 30B-A3B), verified in one batched forward of the big model | 1.5–3× decode when the draft agrees often | new `Engine` loop, `Backend.forward` with verification of `k` tokens |
| 8 | Smaller items: a vectorized `exp` for softmax/SiLU on CPU; `simdgroup_matrix` (Apple's MMA) for prefill matmuls; coalesced router weight reads; parallel shard downloads | each a few percent to 2× in its own corner | `VectorKernels`, `kernels.metal`, `HuggingFaceHub` |

---

## 11.6 Summary

- Decode is bound by **bytes per token**, not arithmetic: 452 MB (0.6B) or ~2 GB (30B-A3B) per token against
  ~200 GB/s. Quantization and batching are the only levers that move that ceiling.
- Today's Metal decode reaches ~17 % of the bandwidth roof; the gap is dispatch overhead and latency-bound small
  matmuls, which kernel fusion attacks first.
- The CPU path was instruction-bound at 128-bit SIMD; int8 activations with integer dot products took the
  row kernel from ~30 conversion-heavy instructions per block to ~31 single-cycle integer ops doing twice
  the lanes, 6× per kernel and ~4× per decoded token.
- Memory is dominated by the weights and the KV cache (`--ctx` and `backend.kv-dtype` are the knobs).
