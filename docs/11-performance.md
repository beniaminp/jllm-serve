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
(`./jllm bench --ctx 1024`, otherwise idle machine, 2026-10-16; the Metal Qwen3-0.6B row re-measured by
`scripts/mac-validate.sh` on 2026-10-21, at the default 4096 context, where the model stops after 91 tokens):

| model | backend | prefill | decode | weight traffic at that decode rate | % of bandwidth ceiling |
|---|---|---|---|---|---|
| Qwen3-0.6B | **Metal** (f32, f16 or q8 KV cache) | **~2 100 tok/s** (1 426 with 64-token chunks, ~536 with `-Djllm.metal.mmaPrefill=false`) | **~143 tok/s** (speculation off; 118 before the multi-row matvec and decode attention) | 65 GB/s | 32 % |
| Qwen3-0.6B | **CPU, int8 activations** (default) | **~275 tok/s** | **~51 tok/s** | 23 GB/s | 11 % |
| Qwen3-0.6B | CPU, float activations | ~237 tok/s | ~13 tok/s | 6 GB/s | 3 % |
| Qwen3-30B-A3B | **Metal** | **~300 tok/s** (413-token prompt, 256-token chunks; 211 with 64) | **~27 tok/s** (25.0 before) | 55 GB/s (≈ 2.05 GB per token) | 28 % |
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
| int8 activations (`true`; what `auto` picks on NEON) | ~275 tok/s | ~51 tok/s | **~3.9×** |

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

### CPU native int8 kernels (`backend.cpu.native`, [07 §7.16](07-cpu-backend.md#716-native-int8-kernels-libjllm_cpu-nativecpu))

4-core Xeon with AVX512-VNNI, Qwen3-0.6B Q4_0, 4 threads. The machine was shared with three other agents' builds
(load average 35–50), so only the ratios mean something:

| | one token's matmuls (the `auto` probe) | decode |
|---|---|---|
| int8, native `avx512-vnni` (what `auto` picks with the library) | 76–79 ms | 4.2–5.2 tok/s |
| float (what `auto` picks without it) | 130–188 ms | 2.4–3.3 tok/s |
| int8, Java kernels | 268–337 ms | 0.5–4.7 tok/s (C2 trouble under load) |

Single-threaded row kernels: native is 2.8× (Q4_0) to 8× (Q6_K) faster than the Java int8 kernels. Prefill is
unchanged. Details and the per-type table are in 07 §7.16.

### Qwen3-30B-A3B

Qwen3-30B-A3B-Thinking-2507 (Q4_0, 19.4 GB cache, same 413-token prompt): Metal **~49 tok/s prefill, ~25 tok/s
decode**; CPU (int8 activations) ≈ 12 tok/s prefill and ≈ 6 tok/s decode. Decode at 25 tok/s means ~51 GB/s of
expert traffic (8 of 128 experts per layer, ≈ 2 GB per token) — the same ~25 % of the bandwidth ceiling as the
dense model, so the MoE path is as efficient per byte as the dense one. Prefill is bandwidth-bound by streaming
nearly all experts per chunk, which is why it is only ~2× decode here versus ~4.8× for the dense model; on the
CPU the expert-grouped prefill reuses each dequantized expert row for all tokens routed to it
([07 §7.7](07-cpu-backend.md#77-mixture-of-experts-on-cpu-cpubackendmoeffn-moeexperts)), on Metal that is still open
([11.5](#115-optimization-roadmap)).

---

### Against llama.cpp (`scripts/mac-validate.sh --vs-llama`)

🧸 To know who runs faster, both runners must run the same race: the same model file, the same length of question,
the same number of words to say back. The script makes jllm and llama.cpp do exactly that on a Mac and writes down
both times side by side.

📐 `scripts/mac-validate.sh --vs-llama` (add `--llama-only` to skip the rest, `--llama-moe unsloth/Qwen3-30B-A3B-GGUF`
for a mixture-of-experts row, `--llama-quant Q4_K_M` for another file of the repository) does, per GGUF repository:

1. `jllm bench` on the `.gguf` file as it is (default `unsloth/Qwen3-0.6B-GGUF`, `Q4_0`), with
   `--engine.speculative off`: llama-bench has no speculation, and the bench prompt repeats one sentence, which
   would flatter prompt-lookup drafting. Its output gives the prompt length P (≈ 413 tokens) and the generated
   count N (128).
2. `llama-bench -m <same file> -p P -n N -d P -r <rounds>` with flash attention off and on (the faster of each column
   is kept). `-d P` makes llama.cpp decode after P tokens of context, as jllm does after its prompt; a llama-bench
   without `-d` decodes from an empty context, which favours it slightly, and the table says so.
3. The ratio row `jllm / llama.cpp` (above 1 = jllm faster), and one more jllm row with its default speculation,
   labelled as favoured by the repetitive prompt.

It needs `llama-bench` (`brew install llama.cpp`, or `LLAMA_BENCH=/path`) and `python3` (to read llama-bench's
JSON).

Measured on the M1 Pro (plugged in, no thermal warning) on 2026-10-21 against llama.cpp 0.7.0 from Homebrew
(build 11669, commit 620d2225c, Metal), 3 rounds:

| model file | engine | prefill tok/s | decode tok/s | note |
|---|---|---|---|---|
| Qwen3-0.6B-Q4_0.gguf | jllm (Metal) | 1352.1 | 104.2 | prompt 413 tok, 128 generated |
| | llama.cpp | 3068.5 | 204.8 | decode at depth 413 |
| | jllm / llama.cpp | 0.44x | 0.51x | |
| | jllm, default speculation | | 207.4 | the bench prompt repeats itself, so this favours speculation |

After the prefill chunk (`backend.batch=auto` = 256), the multi-row decode matvec and the decode attention
([08 §8.29 – 8.31](08-metal-backend.md#829-the-prefill-chunk-backendbatchauto)), the same command on 2026-10-22 (llama.cpp
built against ggml 0.27.0 now, 3 repetitions):

| model file | engine | prefill tok/s | decode tok/s | note |
|---|---|---|---|---|
| Qwen3-0.6B-Q4_0.gguf | jllm (Metal) | 1903.4 | 137.5 | prompt 413 tok, 128 generated |
| | llama.cpp | 3096.8 | 216.9 | decode at depth 413 |
| | jllm / llama.cpp | 0.61x | 0.63x | was 0.44x and 0.51x |
| | jllm, default speculation | | 204.8 | the bench prompt repeats itself, so this favours speculation |

(jllm's GGUF prefill is 1 903 against 2 011 for its own Q4_0 cache.) The gap that remains is dispatch latency rather than
kernels: the matvecs of a token take 5.5 ms for 382 MB where llama.cpp's whole token takes 4.6 ms, and
[11.2](#112-where-the-time-goes) shows the layer matvecs at 50 – 70 GB/s.

Before those three changes (the first table): llama.cpp prefilled 2.3× and decoded 2× faster on the same file. The file is 382 MB, so jllm's decode reads ~40 GB/s
of weights and llama.cpp's ~78 GB/s, both well under the ~200 GB/s roof, so the gap is most likely the dispatch and
latency overhead of 11.2, not bandwidth. (jllm's decode here is lower than the ~117 tok/s of 11.1 partly because all 128
tokens are generated, where its own Q4_0 cache stops at 91.) The speculation row matches llama.cpp only because the
bench prompt is repetitive.

## 11.2 Where the time goes

### Metal decode (~9 ms per token at 413-token context)

**Measured split** (M1 Pro, Qwen3-0.6B Q4_0, 413-token context, 2026-10-22; `MvBench`-style micro-benchmarks over 28
distinct layers, so nothing is cached between layers, and `./jllm bench`): of the ~8.5 ms per token before the multi-row
matvec ([08 §8.30](08-metal-backend.md#830-several-output-rows-per-simdgroup-in-the-decode-matvec-mvnr_)), **6.8 ms were the
matvecs** (q/k/v 73 µs, o_proj 28, gate/up 83, down 35 µs per layer, the LM head 1.45 ms) and the rest rope, attention
(~1.5 ms at 413 positions, ~9.5 ms at 3 000), the norms and the CPU side. The matvecs ran at 40 – 70 GB/s for the layer
matrices and 115 GB/s for the LM head: the 1 – 3 MB matrices are latency-bound, not bandwidth-bound. Now: matvecs 5.5 ms
(4 rows per simdgroup, 18 % less), attention ~1.2 ms at 413 positions and ~5 ms at 3 000 ([08 §8.31](08-metal-backend.md#831-decode-attention-attention_dec)).
`./jllm bench` decode, speculation off: 117.9 → 143.3 tok/s at 413 tokens, 97.1 → 113.7 at ~1 000, 59.5 → 92.9 at ~3 000
(f16 and q8 caches at 3 000: 62 → 113).

1. **Dispatch count.** 6 kernels per layer × 28 + 2 = 170 dispatches per token (338 for the 30B MoE model since
   its norm moved into the router, 386 before; the first version had 338). Each costs a few microseconds of GPU
   front-end time even when its work is tiny, and a kernel cannot start before the previous one has finished (serial
   encoder). Encoding them cost one Java → native call each until decode replay
   ([08 §8.22](08-metal-backend.md#822-decode-replay-one-native-call-per-token)) recorded the step once and made it
   3 calls per token:

   | per decode token | dispatches | Java → native calls, encoded one by one | with decode replay |
   |---|---|---|---|
   | Qwen3-0.6B, `forward` of one token | 170 | 175 | 3 |
   | Qwen3-0.6B, pipelined greedy step | 172 | 174 | 3 |
   | Qwen3-0.6B, q8 KV cache | 198 → 170 ([08 §8.23](08-metal-backend.md#823-decode-fusions)) | 203 | 3 |
   | Qwen3-30B-A3B | 386 → 338 | 391 | 3 |

   Not yet measured on a Mac (the emulator's wall time means nothing); `scripts/mac-validate.sh` compares replay on
   and off.
2. **Under-filled GPU.** `o_proj` / `down_proj` have 1024 output rows → 256 threadgroups of 4
   simdgroups (128 with the multi-row kernels, whose lanes each keep 4 rows' loads in flight); the 16-core GPU wants thousands in flight to hide memory latency. Latency, not bandwidth, bounds
   these matmuls.
3. **The LM head.** `151 936 × 1024` in Q8_0 = 165 MB, over a third of all bytes, for one 1024-vector.
4. **Attention.** For each of 28 layers × 16 heads the kernel streams the whole 413-position KV prefix
   (413 × 128 × 4 B × 2 = 423 KB per head) through one threadgroup; writing scores to device memory and
   reading them back twice costs extra traffic. That is `attention_t`, which streamed the cache at ~70 GB/s; flash decoding
   ([08 §8.11](08-metal-backend.md#811-flash-decoding-chosen-at-start-up)) spreads long contexts over more threadgroups but kept the
   per-key loop, and `attention_dec` ([08 §8.31](08-metal-backend.md#831-decode-attention-attention_dec)) replaced both: simdgroups as independent
   workers, online softmax in registers, K/V rows shared by the two query heads of a kv head; 130 – 150 GB/s, ~2× at 3 000 tokens.
5. **CPU-side work per token**: sampling over 151 936 logits (~0.3 ms), the embedding row copy, and the
   `commitAndWait` round trip. The command buffers are committed every 8 layers, so encoding overlaps GPU work and
   only the last part is exposed; pipelined greedy decoding takes sampling off the CPU, and decode replay replaces
   the encoding of 170 dispatches from Java by one native call.

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
and the scalar `exp` / `silu` loops are still visible here (fast math, now the default, vectorizes the softmax's).

---

## 11.3 Memory footprint

| component | Qwen3-0.6B | Qwen3-30B-A3B |
|---|---|---|
| quantized weights (mmap'd `jllm-q4_0.bin`, off-heap) | 452 MB | ≈ 19.7 GB |
| KV cache, f32, per 1 024 tokens of `--ctx` | 28 layers × 1024 × 1024 × 8 B = 235 MB | 48 × 1024 × 512 × 8 B = 201 MB |
| KV cache at `--ctx 4096` | 940 MB | 805 MB |
| KV cache with `backend.kv-dtype=f16` (2 B per element) | 117 MB per 1 024 tokens; **470 MB** at `--ctx 4096` | 101 MB per 1 024 tokens; **403 MB** at `--ctx 4096` |
| KV cache with `backend.kv-dtype=q8` (1 B per element + 4 B per `headDim` row) | 61 MB per 1 024 tokens; **242 MB** at `--ctx 4096` | 52 MB per 1 024 tokens; **208 MB** at `--ctx 4096` |
| KV cache with `backend.kv-dtype=q4` (0.5625 B per element: nibbles + a half scale per 32 values; plus a ring of `backend.kv-q4-recent + batch` half rows per layer) | 33 MB per 1 024 tokens; **132 MB** at `--ctx 4096` (+ 22 MB ring at R = 128) | 28 MB per 1 024 tokens; **113 MB** at `--ctx 4096` (+ 19 MB ring) |
| KV cache, DeepSeek-V2-Lite (MLA latent cache, [14 §14.11](14-model-families.md#1411-deepseek-v2--v3-multi-head-latent-attention--shared-experts)) | 27 layers × 1024 × (512 + 64) × 4 B = 64 MB per 1 024 tokens; **255 MB** at `--ctx 4096`, 127 MB at f16 (the expanded per-head cache of the first version was 553 KB per token, 2.2 GB at `--ctx 4096`); plus 216 MB of dequantized `kv_b_proj` halves | — |
| KV cache + fixed state, Qwen3-Next-80B-A3B (12 attention + 36 Gated DeltaNet layers, [14 §14.14](14-model-families.md#1414-qwen3-next-hybrid-gated-deltanet--gated-attention--sparse-moe)) | — | 12 × 1024 × 512 × 8 B = **50 MB per 1 024 tokens** (a quarter of the 30B's 201 MB with the same attention shape), 201 MB at `--ctx 4096` (100 MB at f16), plus **79 MB** of DeltaNet state (36 × (2 MB recurrent + 96 KB conv)) whatever the context |
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

Optional system properties, passed through `JAVA_OPTS` (for example `JAVA_OPTS="-Djllm.cpu.fastmath=false" ./jllm bench`):

| property | default | effect |
|---|---|---|
| `jllm.cpu.fastmath` | on | vectorized `exp` / `tanh` in the attention softmax and soft-caps, not bit-identical ([07 §7.15](07-cpu-backend.md#715-fast-math-jllmcpufastmath)) |
| `jllm.cpu.native` | auto | the native int8 kernels when `backend.cpu.native` is not set (the Docker image's `JLLM_CPU_NATIVE`); `true` / `false` ([07 §7.16](07-cpu-backend.md#716-native-int8-kernels-libjllm_cpu-nativecpu)) |
| `jllm.cpu.nativeLib` | search | path of `libjllm_cpu.so` / `.dylib` (the `jllm` launcher passes `target/native/`'s) |
| `jllm.cpu.deterministic` | off | prefill through the single-token kernels, for exact comparisons; slower ([07 §7.14](07-cpu-backend.md#714-prefill-and-decode-are-not-bit-identical-results-depend-on-history)) |
| `jllm.metal.flashDecode` | auto | flash decoding (auto = on when the start-up self-check finds it correct and faster) for decode steps over `jllm.metal.flashMinLen` (512) positions ([08 §8.11](08-metal-backend.md#811-flash-decoding-chosen-at-start-up)) |
| `jllm.metal.commitLayers` | 8 | layers per Metal command buffer (0 = one per pass) |
| `jllm.metal.prefillMinTokens` | 8 | smallest chunk that uses the tiled prefill matmuls |
| `jllm.metal.mmaPrefill` | true | simdgroup-matrix prefill matmuls; `false` = the scalar tiled kernels (bit-identical to decode) |
| `jllm.metal.mmaMinTokens` | 32 | smallest chunk that uses the simdgroup-matrix matmuls ([08 §8.10](08-metal-backend.md#810-the-tiled-prefill-matmuls)) |
| `jllm.metal.hmma` | auto | which form of those matmuls: auto = per matrix shape, as the start-up check timed f32 `*_mma` against half-weight `*_hmma` on this chip; `true` / `false` = half-weight everywhere (exact formats) / nowhere ([08 §8.21](08-metal-backend.md#821-half-weight-mma-prefill-and-the-second-flash-prefill-form)) |
| `jllm.metal.hmmaKQuants` | false | also allow the lossy half-rounded `*_hmma` kernels for the K-quants (Q2_K, Q6_K, Q4_K, Q5_K); Q4_0 / Q8_0 (and GGUF Q4_0 / Q8_0) are exact and always allowed ([08 §8.21](08-metal-backend.md#821-half-weight-mma-prefill-and-the-second-flash-prefill-form)) |
| `jllm.metal.flashPrefillKernel` | auto | flash prefill form `fa` or `fb` (K and V staged together, half staging for f16 / q8 caches); auto = the faster one that agrees ([08 §8.21](08-metal-backend.md#821-half-weight-mma-prefill-and-the-second-flash-prefill-form)) |
| `jllm.metal.attnGroup` | 1 | query heads per attention threadgroup (2, 4, 8 measured no faster) |
| `jllm.metal.flashPrefill` | auto | flash-attention prefill kernel `attention_fa` instead of `attention_t` (auto = start-up self-check decides; [08 §8.13](08-metal-backend.md#813-flash-attention-prefill-attention_fa)) |
| `jllm.metal.moeMma` | true | expert-grouped simdgroup-matrix MoE prefill ([08 §8.15](08-metal-backend.md#815-expert-grouped-moe-prefill)); `false` = per-slot kernels |
| `jllm.metal.moeMmaMin` | 16 | smallest chunk that uses the expert-grouped MoE kernels |
| `jllm.metal.pipeline` | true | pipelined greedy decoding: the GPU picks the token and runs ahead ([08 §8.16](08-metal-backend.md#816-pipelined-greedy-decoding)) |
| `jllm.metal.replay` | true | decode replay: a decode step is recorded once per shape and replayed with its position patched in, one native call per token ([08 §8.22](08-metal-backend.md#822-decode-replay-one-native-call-per-token)); `false` = every dispatch from Java |
| `jllm.metal.replayVerify` | first | `all`: compare every replayed step with a fresh recording (tests); by default only each recording's first replay is compared |
| `jllm.metal.concurrent` | false | the compute encoder: `false` = serial (every dispatch waits for the one before), `true` = concurrent with barriers only between dependent dispatches, `batch` = concurrent in the passes of batched decoding only; bit-identical, no gain for single-stream decode or prefill (30B included), a few percent for three slots in shared passes, off by default ([08 §8.28](08-metal-backend.md#828-the-concurrent-compute-encoder-barriers-only-where-a-kernel-needs-the-one-before)) |
| `jllm.metal.fuse` | true | decode fusions (q8 cache store in `rope_store_q8`, the MoE norm in the router, MLA's norm in its projections; bit-identical, [08 §8.23](08-metal-backend.md#823-decode-fusions)); `false` = the separate dispatches |
| `jllm.metal.workingSetGb` | Metal's | the GPU memory budget `backend.metal.cpu-moe=auto` aims for |
| `jllm.metal.hotExperts` | true | place CPU experts by the saved usage statistics (hot ones stay on the GPU); `false` = whole layers |
| `jllm.metal.cpuMoeSyncUs` | 100 | the planner's estimate of one GPU → CPU → GPU round trip, µs (lower = more layers split) |
| `jllm.metal.expertStatsDir` | `~/.cache/jllm/expert-usage` | where the expert usage counts are kept |
| `jllm.metal.expertCountEvery` | 8 | layers whose experts are all on the GPU are counted on one forward pass in this many (1 = every pass, as `arch-smoke.sh` uses) |
| `jllm.metal.decodeThreads` | tuned | threads per threadgroup (64, 128, 256) of the plain single-token matmul; the start-up self-check times the three on the model's own output projection and keeps the fastest per chip (bit-identical results) |
| `jllm.engine.batchWindowMicros` | 2000 | how long a batched decode step waits for the other slots' steps ([10 §10.9b](10-engine-and-server.md#109b-batched-decoding-across-slots)) |
| `jllm.metal.emulator` | false | run the Metal backend on the CPU emulator (with `jllm.metal.lib` = the emulator library; [18](18-metal-emulator.md)) |

**Prefill chunk for MoE models.** With `backend.batch=auto` (the default), a mixture-of-experts model prefills in chunks
of 256 tokens instead of 64 (dense models on Metal do too, [08 §8.29](08-metal-backend.md#829-the-prefill-chunk-backendbatchauto)). With 128 experts and top-8, a 64-token chunk gives each expert only about
`64 · 8 / 128 = 4` tokens. An expert's weights are read once per chunk, and on the CPU each 4-row group is
dequantized once, both for all its tokens, so that fixed cost is shared by only ~4 tokens. At 256 tokens it is
shared by ~16. Measured on the CPU expert path (`cpu/MoeExperts`, one Qwen3-30B-A3B-shaped layer, Q4_0, 4 cores,
best of 3 runs each): 1 040–1 140 → 1 570–1 840 tokens/s, i.e. 1.5–1.6×. The Metal grouped kernels read each
expert's weights once per tile of up to `MMA_BT` slots, so they should gain the same way; that is not measured on a
Mac yet. Decode is unchanged (one token per step).

Experts on the CPU are chosen by a setting (`jllm.properties` key, or `--backend.metal.cpu-moe` on the command
line), not a system property: `backend.metal.cpu-moe` = `auto` (default: only when the device's GPU memory budget is
exceeded; the rarely used experts by saved usage, whole layers until statistics exist), `off`, `all` or a number of
whole layers ([08 §8.18](08-metal-backend.md#818-experts-on-the-cpu-backendmetalcpu-moe)). A backend that holds
its placement alone also re-plans it every 30 minutes of use from the counts so far (`MetalBackend.rebalanceExperts`).

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
| **Native int8 kernels** (`libjllm_cpu`, FFM, VNNI / AVX2 / `sdot`; [07 §7.16](07-cpu-backend.md#716-native-int8-kernels-libjllm_cpu-nativecpu)) | single-thread row kernels 2.8–8× faster than the Java int8 ones on AVX512-VNNI; one token of matmuls 1.7–2× faster than float, so `auto` now picks int8 on x86 ([11.1](#cpu-native-int8-kernels-backendcpunative-07-716)) | `src/main/native/jllm_cpu.c`, `cpu/NativeCpu` |
| **f16 KV cache**: `backend.kv-dtype=f16` ([07 §7.13](07-cpu-backend.md#713-the-f16-kv-cache-kvcache), [08 §8.9](08-metal-backend.md#89-the-f16-kv-cache)) | halves cache memory (940 → 470 MB at `--ctx 4096`) and attention read traffic; speed equal or better, outputs unchanged on the test prompts | `cpu/KvCache`, `_h` kernels |
| **Expert-grouped MoE prefill on CPU** ([07 §7.7](07-cpu-backend.md#77-mixture-of-experts-on-cpu-cpubackendmoeffn-moeexperts)) | each expert row dequantized once per chunk and reused for its tokens | `MoeExperts.batched` |
| **Tiled CPU attention** over (kv head, token tile) | each K / V row read once per tile; bit-identical | `KvCache.attentionTiled` |
| **Prefill chunk `backend.batch=auto`**: 256 tokens for dense decoders on Metal (64 filled 32 MMA threadgroups on 16 cores) | prefill 1 426 → 2 012 tok/s (Qwen3-0.6B, 413 tokens), logits bit-identical across chunk sizes | `ModelLoader.batch`, [08 §8.29](08-metal-backend.md#829-the-prefill-chunk-backendbatchauto) |
| **Multi-row decode matvec** on Metal (`mv<NR>_*`, 4 rows per simdgroup, activations reused across rows, rmsnorm prologue shared by more rows) | matvecs per token 6.76 → 5.52 ms; decode 118 → 132 tok/s (0.6B), GGUF Q4_0 105 → 133; `-Djllm.metal.mvRows=1` for the old kernels | `mvr_matmul*`, [08 §8.30](08-metal-backend.md#830-several-output-rows-per-simdgroup-in-the-decode-matvec-mvnr_) |
| **Decode attention** on Metal (`attention_dec`: independent simdgroups, online softmax in registers, shared K/V rows, reduce-scatter dot products), chosen by the self-check | decode at ~3 000 tokens 63 → 93 tok/s (f16/q8 caches: 62 → 113); +3 % at 413 tokens; agrees to rounding, not bit-identical | `attention_dec`, `MetalTuner.decodeAttention`, [08 §8.31](08-metal-backend.md#831-decode-attention-attention_dec) |
| **Flash decoding** on Metal, chosen by the start-up self-check ([08 §8.11](08-metal-backend.md#811-flash-decoding-chosen-at-start-up)) | +3–5 % decode at ~3 000 tokens of context, not bit-identical (agrees to rounding) | `attention_split`, `attention_combine` |
| **Vectorized `exp` / `tanh`** on CPU, default (`jllm.cpu.fastmath=false` to disable) | +5–9 % decode at ~3 000 tokens, −40 % long-prompt attention | `VectorKernels.expFast` |
| **MLA latent cache** (DeepSeek): absorbed `W_kb`, one 576-wide row per token | 2.2 GB → 255 MB at `--ctx 4096` ([14 §14.11](14-model-families.md#1411-deepseek-v2--v3-multi-head-latent-attention--shared-experts)) | `MlaCpu`, `MlaMetal` |

Remaining, by expected payoff:

| # | change | expected effect | where |
|---|---|---|---|
| 1 | **Continuous batching**: several sequences per forward pass so each weight read serves many tokens | turns decode from memory-bound toward compute-bound: near-linear throughput scaling with concurrent users | `Backend.forward` (batch of sequences), per-sequence KV cache slots, `Engine` scheduler, `ApiServer` lock |
| 2 | ~~Expert-grouped MoE prefill on Metal~~ (done: `moe_sort` + `moe_up_mma` / `moe_down_mma`, [08 §8.15](08-metal-backend.md#815-expert-grouped-moe-prefill)) | | |
| 3 | **Speculative decoding** with a small draft model (e.g. 0.6B drafting for 30B-A3B), verified in one batched forward of the big model | 1.5–3× decode when the draft agrees often | new `Engine` loop, `Backend.forward` with verification of `k` tokens |
| 4 | ~~Flash-style prefill attention~~ (done: `attention_fa`, and `attention_fb`, [08 §8.13](08-metal-backend.md#813-flash-attention-prefill-attention_fa), [§8.21](08-metal-backend.md#821-half-weight-mma-prefill-and-the-second-flash-prefill-form)) | | |
| 5 | ~~`simdgroup_matrix` prefill matmuls~~ (done: `*_mma`, and the half-weight `*_hmma` with the per-shape choice, [11.8](#118-the-prefill-kernels-against-llamacpps-design)); measured on an M1 Pro: `*_hmma` gives no gain there (11.8), the start-up check keeps the f32 form; next: larger prefill chunks (`--batch 256`: +46 %), larger tiles (64 × 64 with 8 simdgroups) or Metal 4 tensor ops on M5-class GPUs | see 11.8 | `kernels.metal`, `MetalTuner` |
| 6 | **Mixed-precision low-bit formats** (a Q2_K / Q3_K mix with higher-precision sensitive tensors) | smaller files and fewer bytes per token ([13](13-q2k-quantization.md)) | quantizer, `QType`, kernels |
| 6b | After the matvec and attention work the 0.6B decode token is ~7 ms; what is left is mostly dispatch latency (170 dispatches in a serial encoder): overlap independent dispatches ([08](08-metal-backend.md), concurrent encoder), fuse the attention combine into the o_proj prologue, prefetch a threadgroup's first weights before its norm prologue, aligned 16-byte loads of GGUF Q4_0 blocks (−2 %) | up to ~2 ms per token | `kernels.metal`, `jllm_metal.m` |
| 7 | Smaller items: fuse RMSNorm into the next matmul on CPU, wider threadgroups for small Metal matmuls, coalesced router weight reads, parallel shard downloads | each a few percent to 2× in its own corner | `CpuBackend`, `kernels.metal`, `HuggingFaceHub` |

A future direction, not started: a Windows backend (for example OpenCL or DirectML) so the GPU / NPU of a
Snapdragon X machine could be used the way Metal is used on Apple Silicon; the `Backend` interface already keeps
the model logic out of the kernel layer.

---

## 11.6 Speculative decoding

Setting `engine.speculative` ([10 §10.9](10-engine-and-server.md#109-speculative-decoding)). Apple M1 Pro (32 GB),
Metal, greedy, `--ctx 2048`. **The machine was heavily shared while this was measured** (load average 15 to 75 from
other jobs, the 30B cache competing for memory), so tok/s below are the best of 2 to 5 interleaved runs and carry
about ±15 % noise; the pass costs and acceptance counts are the reliable numbers.

**Cost of a verification pass** (minimum over 60 interleaved rounds, `forwardAll` including the LM head over n rows):

| tokens in the pass | 1 | 2 | 3 | 5 | 7 | 9 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B, ms (× a 1-token pass) | 8.0 (1.0) | 10.0 (1.25) | 13.1 (1.64) | 19.5 (2.44) | 26.5 (3.3) | 26.0 (3.3) |
| Qwen3-30B-A3B, ms (× a 1-token pass) | 39.2 (1.0) | 57.4 (1.47) | 80.4 (2.05) | 128 (3.3) | 175 (4.5) | 204 (5.2) |

A pass is not "free" for extra tokens on this backend: the multi-token kernels scale roughly linearly in n (the
dense model about 0.36, the MoE about 0.55 of a step per extra position), so drafting K tokens needs about
`1 + 0.36 K` (dense) or `1 + 0.55 K` (MoE) accepted-tokens-per-pass-equivalents to break even.

**Prompt lookup, 0.6B, 200 tokens max, best of 5 interleaved** (a: edit a 35-line Java class and print it again;
b: haiku + explanation; c: long-form explanation; accepted/drafted tokens of the best run):

| setting | a code edit | b haiku | c long-form |
|---|---|---|---|
| off | 102.5 tok/s | 98.5 | 115.8 |
| ngram min 1, k 4 | 110.8 (54/138) | 96.4 (0/20) | 108.6 (10/87) |
| ngram min 2, k 4 (default values) | 118.2 (62/123) | 115.8 | 113.1 (5/46) |
| ngram min 3, k 4 | 120.1 (62/110) | 84.4 | 122.9 (4/12) |
| ngram min 2, k 6 | 105.5 (64/161) | 113.6 | 112.5 (6/60) |

Only the code-editing prompt, whose answer repeats the input, drafts successfully (about 50 % of drafted tokens
accepted, +15 to +17 %); on b and c nothing is drafted or almost nothing is accepted and the differences are
noise (b with min 3 never drafted at all and still reads 84 vs 98).

**Qwen3-30B-A3B-Thinking-2507, 80 tokens, best of 2 interleaved, `engine.draft-model` = Qwen3-0.6B:**

| setting | a code edit | b haiku | c long-form |
|---|---|---|---|
| off | 23.0 tok/s | 23.8 | 20.4 |
| ngram min 2, k 4 | 20.2 (15/48) | 22.0 (5/24) | 20.6 (9/24) |
| ngram min 3, k 4 | 22.3 (9/27) | 24.1 (3/8) | 13.4 (7/12) |
| draft, k 2 | 14.4 (43/71) | 14.7 (38/80) | 18.3 (39/80) |
| draft, k 4 | 2.5 (53/104) | 1.5 (45/129) | 14.5 (47/125) |
| draft, k 6 | 7.1 (54/148) | 1.3 (45/189) | 11.8 (50/165) |

(The 30B shared the 32 GB of memory with other jobs for the whole run; the k 4 / k 6 draft rows on a and b
are far slower than their acceptance explains and are best read as "slow", not as a ratio.)
The draft model agrees with the 30B on only 50 to 60 % of positions (the 30B is a *Thinking* model with a
different output distribution), and every drafted token costs 0.55 of a 30B step, so the verification alone eats
the gain even at 100 % acceptance: with K = 2 and 60 % acceptance a pass yields 1.96 tokens for 2.1 steps.
With a perfectly agreeing draft (the 0.6B drafting for itself, only a correctness test) acceptance is 100 % and
throughput still drops (56 vs 95 tok/s) because the verification pass and the draft's own K steps cost more than
the K tokens saved.

**Since these measurements** the draft path changed on Metal
([08 §8.25](08-metal-backend.md#825-draft-model-speculation-on-the-gpu)): the draft's steps are pipelined on the GPU
(each feeds the argmax of the step before; before, every draft step copied a vocabulary of logits to the CPU and
waited for it), the draft is loaded before the target so the target's memory plan counts it (the k 4 / k 6 rows above
ran with the two models competing for memory), and the draft length comes from a cost model of measured times and
acceptance, which pauses speculation while no length beats plain decoding (with the costs of the table above, a
0.6B draft for the 30B pauses below about 75 % acceptance). `scripts/mac-validate.sh` measures the dense model
drafting for itself with adaptive length and draft pipelining on and off, and `--moe` the 30B with the 0.6B draft; those
rows are still to be taken on a quiet Mac. The emulator ([18](18-metal-emulator.md)) runs the same code and checks
exactness, but its timings say nothing about a GPU.

**Batched decoding across slots on Metal** ([08 §8.24](08-metal-backend.md#824-batched-decoding-across-slots-metalbackendforwardbatch)):
expected per round of n slots `W/BW + n·r` instead of `n·(W/BW + r)` (W/BW ≈ 8 ms for Qwen3-0.6B on the M1 Pro, r the
per-row attention and small dispatches), i.e. close to n times the total tokens per second at short contexts.
`scripts/mac-validate.sh` measures three slots together against taking turns (`-Djllm.metal.batchDecode=false`); not
measured on a Mac yet.

**Greedy exactness** (5 prompts, 100 to 150 tokens): Metal, `ngram` (min 1, k 4) and `draft` (self-draft, k 6): all
5 outputs byte-identical to the non-speculative output, accepted 100 % for the self-draft, so `forwardAll` is bit-identical to
the single-token pass on Metal. CPU, `-Djllm.cpu.deterministic=true --backend.cpu.int8-activations false`: 5 of 5
identical. CPU float activations without the deterministic flag: 5 of 5 identical on these prompts. CPU default
(int8 activations): 3 of 5 differ (the single-token decode path uses the int8 dot products, the multi-token
verification path the float kernels, [07 §7.14](07-cpu-backend.md#714-prefill-and-decode-are-not-bit-identical-results-depend-on-history));
both are valid greedy continuations of the same model. On the CPU, speculation also slows decode (int8 decode 49 tok/s vs
the float multi-token path).

**Default (then).** `engine.speculative=off`. Speculation never hurt by more than the noise on the dense 0.6B with the back-off, and helps
(about +15 %) for text that copies its input, but it is neutral to harmful on the 30B MoE (the headline model: every verified
position streams its own experts) and on the CPU, and no setting was a win everywhere.

**Default (now).** `engine.speculative=ngram` for dense models, after prompt lookup was reworked for coding agents
([17 §17.1](17-agent-speedups.md#171-prompt-lookup-tuned-for-agents)): a hash index over the whole context, matches of 3 to 8
tokens, drafts that grow to 16 tokens while they are accepted. On the CPU (4-core AVX-512, Qwen3-0.6B, greedy) rewriting
an 80-line file from the prompt went from 19.9 to 81.2 tok/s (98 % of drafts accepted), and a free-form story cost about 4 %
(23.3–23.4 vs 24.4 tok/s; 2-token matches cost 8 %, hence `engine.ngram-min=3`). MoE models keep it off unless it is set
explicitly, for the reason above. Enable `ngram` for editing, extraction
and quoting workloads on dense models; `draft` needs a draft model that agrees with the target on 80 % or more of the tokens and
a backend whose multi-token pass is much cheaper than K single steps, which the Metal kernels of this repository are not yet.
`engine.draft-tokens=4`, `engine.ngram-max=3`, `engine.ngram-min=2` were the best of the grid above, measured with the
earlier search that looked at the last 4 096 tokens only. The defaults now are `16`, `8` and `3`: the n-gram index and
the adaptive draft length of [17 §17.1](17-agent-speedups.md#171-prompt-lookup-tuned-for-agents) make long drafts
cheap when they fail, and 3-token minimum matches cost less on free text than 2.

---

## 11.7 Summary

- Decode is bound by **bytes per token**, not arithmetic: 452 MB (0.6B) or ~2 GB (30B-A3B) per token against
  ~200 GB/s. Quantization and batching are the only levers that move that ceiling.
- Today's Metal decode reaches ~25 % of the bandwidth roof (~111 tok/s of a ~440 tok/s ceiling); the gap is
  dispatch overhead and latency-bound small matmuls. Prefill is ~1 360 tok/s.
- The CPU path was instruction-bound at 128-bit SIMD; int8 activations with integer dot products took the
  row kernel from ~30 conversion-heavy instructions per block to ~31 single-cycle integer ops doing twice
  the lanes, 6× per kernel and ~4× per decoded token (13 → 51 tok/s).
- Speculative decoding ([11.6](#116-speculative-decoding)) is off by default: a verification pass of n tokens costs
  1 + 0.36 (n−1) single steps on the dense 0.6B and 1 + 0.55 (n−1) on the 30B MoE, so it gains only for text that
  copies its input (+15 % for code editing with prompt lookup) and loses on the MoE.
- Memory is dominated by the weights and the KV cache (`--ctx` and `backend.kv-dtype` are the knobs).

---

## 11.8 The prefill kernels against llama.cpp's design

🧸 Two bakeries use the same oven. One bakes 266 big loaves an hour, the other a few thousand small rolls; to know
which bakes better you count the flour that goes through the oven, not the loaves. Counted that way, our prefill
uses about a third of the oven time llama.cpp does, and the new kernels are built to close part of that gap.
Whether they do can only be weighed on a real Mac, so below is the arithmetic of what to expect, not a measurement.

📐 **Where llama.cpp is.** Its published Apple Silicon table gives an M1 Pro (16-core GPU) about 266 tok/s of
prompt processing (pp512) for LLaMA-7B Q4_0. A 7B model does `2 × 6.5 · 10⁹ ≈ 13 GFLOP` of matmul per token, so
`kernel_mul_mm` sustains `266 × 13 · 10⁹ ≈ 3.5 TFLOP/s`, about two thirds of the GPU's f32 peak
(`16 cores × 128 lanes × 2 FLOP × 1.3 GHz ≈ 5.3 TFLOP/s`) on large matrices. Its tile is the one our `*_mma`
kernels already used (64 rows × 32 tokens, 4 simdgroups, 8 accumulators each, 32-wide k steps, two barriers per
step); it differs in storing the dequantized weights as half, in 8 × 8-blocked threadgroup tiles, and in writing
full tiles straight to device memory.

**Where we are.** Qwen3-0.6B does `2 × 28 × (2.10 + 1.05 + 1.05 + 2.10 + 6.29 + 3.15) M ≈ 0.88 GFLOP` of matmul
per prefill token (q, k, v, o, gate+up, down; the LM head only for the last token). At ~1 360 tok/s that is at most
`1 360 × 0.88 · 10⁹ ≈ 1.2 TFLOP/s` even if prefill were all matmul: about a third of llama.cpp's rate. (The ~533
tok/s quoted for this model elsewhere is the scalar `matmul_pf` path, `-Djllm.metal.mmaPrefill=false`.)

**What each change of the `*_hmma` kernels buys** ([08 §8.21](08-metal-backend.md#821-half-weight-mma-prefill-and-the-second-flash-prefill-form)):

| change | bytes / work per 32-wide k step of one threadgroup | expected effect |
|---|---|---|
| half weight tile | threadgroup tile 8 KB → 4 KB; total 12 KB → 8 KB | resident threadgroups per core 2 → 4 (if a core holds 32 KB), twice the simdgroups to cover barrier and load stalls; the weight `simdgroup_load`s move half the bytes |
| 8 × 8-blocked tiles | each of the 24 loads per simdgroup reads one contiguous 128 / 256 B block instead of 8 rows 128 B apart | up to 8 bank passes → 1–2 per load; the loads are 6 per 8 multiply-accumulates, so this is the step most likely to matter on the dense shapes |
| next step's bytes in registers | Q4_0: 8 B + scale, Q8_0: 16 B + scale, 32 B of activations per thread | device latency (hundreds of cycles) hidden behind 32 multiply-accumulates per simdgroup |
| skip empty 8-token blocks | MoE tile with `c` slots: `⌈c / 8⌉` of 4 blocks multiplied | expert tiles average 16 slots at 256-token chunks (128 experts, top-8): ~⅓ fewer multiply-accumulates in `moe_up` / `moe_down` (count spread 8–24); the freed ALU time helps the other resident threadgroups, not the tile's own latency |
| direct stores | 2 048 floats per full tile not copied through threadgroup memory, one barrier fewer | a few % of a tile at 1 024-wide rows (32 k steps) |

**Expected prefill speed** (assumptions stated, to be replaced by `scripts/mac-validate.sh` numbers):

- *Qwen3-0.6B, 413-token prompt.* Today's ~0.74 ms per token splits into matmul time `T_mm` and the rest (norms,
  rope, attention, ~170 dispatches per chunk). If matmuls are ~70 % (`T_mm ≈ 0.51 ms`, i.e. ~1.7 TFLOP/s already) and
  the `*_hmma` kernels reach 2–3 TFLOP/s (60–85 % of llama.cpp's large-matrix rate; these matrices are small, 1 024–
  6 144 rows, so the GPU is less full than with 7B shapes), `T_mm` becomes 0.29–0.44 ms and prefill
  **~1 500–1 950 tok/s (1.1–1.45×)**. Larger dense models, where matmuls are a bigger share and fill the GPU better,
  should gain more (up to the ratio of matmul rates, ~1.5–2×).
- *Qwen3-30B-A3B, 256-token chunks.* Matmul work per token: experts `8 × 3 × 768 × 2 048 × 2 × 48 ≈ 3.6 GFLOP`,
  attention projections `2 × 48 × 18.9 M ≈ 1.8 GFLOP`. Reading every expert once per chunk is
  `≈ 19.5 GB / 200 GB/s ≈ 0.1 s` per 256 tokens (a ~2 600 tok/s bandwidth bound), so the grouped prefill is
  compute-bound: with half-empty tiles the `*_mma` expert kernels do ~7.2 GFLOP-equivalents per token, the
  `*_hmma` ones ~4.8 (blocks skipped) to 3.6 (perfectly packed). At 2–3 TFLOP/s for all matmuls that is a ceiling
  of roughly 300–450 tok/s before, 400–650 after, i.e. up to ~1.4× on the MoE share of the time. The measured
  ~49 tok/s predates the grouped MoE kernels; what they and `*_hmma` give together is for the Mac run to show.
- *Long prompts.* At 2 000 tokens attention is `2 × 2 × 128 × ~1 000 keys × 16 heads × 28 layers ≈ 0.23 GFLOP`
  per token, a quarter of the matmul work. `attention_fb` halves its barriers (2 instead of 4 per key block),
  removes the 8-row bank conflicts of its 2 × 64 K / V loads per block and simdgroup (at D = 128, BK = 32), and halves
  the staged bytes for f16 and q8 caches; if that makes the kernel 1.2–2× faster, prefill at 2 000 tokens gains
  ~5–15 % on top of the matmul gain.

**Accuracy cost.** Half weights add a relative error of `≈ u / √3 ≈ 2.8e-4` per dot product (`u = 2⁻¹¹`; measured
1.3e-4 to 4.5e-4 of `max |y|` in `kernel_tests.py`, against ~1e-7 to 1e-6 for the f32 MMA kernels), about 1 % of
the Q4_0 quantization error itself; the start-up check rejects the half-weight form for any shape where it
differs from the f32 one by more than `1e-3 · max |y| + 1e-5`. `attention_fb` rounds nothing `attention_fa` keeps.

**Measured on an Apple M1 Pro (prefill tok/s, 413-token prompt, Qwen3-0.6B q4, `./jllm bench --temp 0 --engine.speculative off`).**
The estimates above did not come true on this GPU: f32 `*_mma` 1 436 tok/s at the default 64-token chunk and 2 092 at
`--batch 256`; first-version `*_hmma` (weights rounded to half) 1 419 / 1 995 (no gain, -1 to -5 %); exact `*_hmma`
(scale applied in f32, which is what the kernels are now for Q4_0 / Q8_0) 873 / 1 105 (-40 %). GGUF Q4_0: 1 361 / 1 950
f32 against 1 368 / 1 954 first version and 853 / 1 106 exact; K-quants (Q4_K_M) with the lossy kernels 958 / 1 860 against
949 / 1 815 (+1 %, noise). The start-up check therefore keeps the f32 kernels for every shape ("half-weight MMA for 0 of 4
shapes"), and the speed ranking of the bigger lever is the chunk size: **`--batch 256` is +46 % prefill over the default
64 on this model** (1 436 -> 2 092 tok/s). Details and the accuracy story in
[08 §8.21](08-metal-backend.md#821-half-weight-mma-prefill-and-the-second-flash-prefill-form). The 30B MoE (grouped
expert kernels, 2 000-token prompts, `fa` / `fb`) was not measured for this change.

---

## 11.9 The q4 KV cache, measured

🧸 q4 keeps the memory shelf for the model's notes about 3.6 times smaller than f16 (about half of q8). That is what a
long conversation, or many conversations at once, needs. The price is paid in two places: the model's answers drift
away from the f16 ones more than they do with q8 (the accuracy tables are in [07 §7.13](07-cpu-backend.md#q4-rotated-keys-4-bit-groups-and-a-recent-window-kvcacheq4-kvq4)),
and the GPU has a little more to do for each page it reads, because the newest pages are kept twice, sharp and small.

📐 **Memory** (Qwen3-0.6B, 28 layers, `kvDim` 1 024; the ring of the recent window is extra and fixed):

| cache | per token | at `--ctx 4096` | at `--ctx 32768` | ring of the window (R = 128 / 512, batch 64) |
|---|---|---|---|---|
| f32 | 229 KB | 940 MB | 7.5 GB | – |
| f16 | 115 KB | 470 MB | 3.76 GB | – |
| q8 | 59 KB | 242 MB | 1.94 GB | – |
| q4 | **31.5 KB** | **132 MB** | **1.06 GB** | 22 MB / 66 MB |

**Speed** (M1 Pro, Qwen3-0.6B Q4_0, `./jllm bench --temp 0 --bench.rounds 3 --engine.speculative off`, median of 3
rounds, under the heavy lock of the validation run; the speculation is off because with it on a different greedy text
changes how often the n-gram drafter hits, which once faked a 2× "q8 speedup"). Window 128 unless stated. The prompt is
the 413-token bench prompt, or ~2 000 tokens (`--bench.prompt-repeat 200`); decode is 128 tokens after it.

| configuration | prompt tokens | prefill tok/s | decode tok/s |
|---|---|---|---|
| Metal, f32 | 413 | 1402.6 | 110.8 |
| Metal, f16 | 413 | 1448.6 | 114.4 |
| Metal, q8 | 413 | 1443.1 | 116.8 |
| Metal, **q4** (window 128) | 413 | 1217.5 | 104.2 |
| Metal, q4, no window | 413 | 1362.9 | 107.3 |
| Metal, q4, window 512 | 413 | 1112.3 | 102.4 |
| CPU, f32 | 413 | 302.5 | 89.3 |
| CPU, f16 | 413 | 291.1 | 69.2 |
| CPU, q8 | 413 | 256.4 | 83.6 |
| CPU, q4 | 413 | 214.3 | 49.2 |
| CPU, q4, no window | 413 | 251.4 | 73.4 |
| CPU, q4, window 512 | 413 | 198.5 | 37.1 |
| Metal, f32 (~2 000-token prompt) | 2013 | 1011.6 | 72.4 |
| Metal, f16 (~2 000-token prompt) | 2013 | 1026.9 | 76.9 |
| Metal, q8 (~2 000-token prompt) | 2013 | 1033.7 | 78.5 |
| Metal, q4 (~2 000-token prompt) | 2013 | 865.9 | 76.4 |
| Metal, q4, no window (~2 000-token prompt) | 2013 | 949.7 | 80.5 |
| Metal, q4, window 512 (~2 000-token prompt) | 2013 | 725.7 | 71.2 |
| Metal, f16, replay off | 413 | 1444.0 | 114.5 |
| Metal, q4, replay off | 413 | 1220.5 | 104.9 |


What it shows: on the GPU q4 decodes within 9 % of f16 at 413 tokens and equal to it at 2 000 (the cache is a small share
of a decode step at these lengths; it pays at tens of thousands of tokens), and prefill is 16 % slower at 2 000 tokens
with window 128 and 29 % slower with window 512, because every key block that overlaps a row's window is corrected from
the ring ([08 §8.27](08-metal-backend.md#827-the-q4-kv-cache-on-the-gpu-backendkv-dtypeq4)). The CPU numbers come from a
machine other (unlocked) jobs also used and move by ±20 % between runs; they say that q4 costs more than the byte
formats on the CPU (4-bit unpacking in the attention loop, the window's second pass), not by how much.
Decode replay (`-Djllm.metal.replay=false`) makes no measurable difference to either cache type here (f16 114.5 against
114.4 tok/s, q4 104.9 against 104.2): `ArchSmoke` checks that q4 steps are recorded and replayed and equal the unrecorded
ones bit for bit, the speed is not what it buys on this model.

**Choosing the window.** The data of [07 §7.13](07-cpu-backend.md#q4-rotated-keys-4-bit-groups-and-a-recent-window-kvcacheq4-kvq4):
the quality gain per position of window is largest at the start (0 → 32 positions: −0.54 nats of the −0.91, → 128:
−0.64, → 512: −0.76), the cost grows with it (prefill at 2 000 tokens: window 0 −7.5 %, 128 −16 %, 512 −29 % against f16). The default stays 128: it takes the larger part of the gain at a cost
of 16 % of prefill; raise `backend.kv-q4-recent` to 512 when answers matter more than prefill speed, set 0 to save the
ring and lose only 7.5 % of prefill speed against f16 (accepting about 3× the error).
