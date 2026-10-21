# 18. The Metal emulator: GPU kernels on a machine without a GPU

Code: `scripts/metal-emu/transpile.py` (Metal Shading Language → C++), `scripts/metal-emu/metal_emu.h` (the Metal
types and built-in functions in C++), `scripts/metal-emu/runtime.cpp` (the `jm_*` C API of `jllm_metal.m`, running
kernels on the CPU), `scripts/build-metal-emu.sh` (builds `target/native/libjllm_metal_emu.so`),
`scripts/metal-emu/emu.py` + `kernel_tests.py` (kernel tests from Python), `scripts/arch-smoke.sh` with `METAL_EMU=1`.

---

## 18.1 Why

🧸 You want to check that a recipe for a giant bakery works, but you only have a tiny home kitchen. So you follow
the same recipe, one tray at a time, very slowly. The cake is the same; it just takes longer.

📐 The Metal backend's kernels are about 2 700 lines of GPU code. Without the emulator, every change to them could
only be tested on a Mac, by hand. With it, the **unchanged** `MetalBackend` Java code runs on Linux (or a Mac)
against a library that has the same C functions as the real Objective-C shim, and the full cross-check of
`ArchSmoke` — every model family, CPU logits vs Metal logits — runs anywhere:

```
METAL_EMU=1 scripts/arch-smoke.sh                       # every family, CPU vs "Metal" on the emulator
python3 -I scripts/metal-emu/kernel_tests.py            # single kernels against numpy references
java ... -Djllm.metal.emulator=true -Djllm.metal.lib=target/native/libjllm_metal_emu.so ...
```

It is slow (every GPU thread becomes a CPU coroutine), so it is meant for the tiny test models, not real ones. It
checks **correctness**, not speed: timings on the emulator say nothing about an Apple GPU.

---

## 18.2 From Metal to C++ (`transpile.py`)

🧸 Metal and C++ are like British and American English: almost the same language, with a few words spelled
differently. The translator only has to fix those words.

📐 The Metal Shading Language is C++14 with extras. The transpiler:

1. runs the C preprocessor over `kernels.metal` (with `<metal_stdlib>` removed), so macro-generated kernel
   instantiations expand;
2. rewrites what C++ cannot parse: vector constructors `float4(x)` → `mk_float4(x)`, attributes on parameters
   (`[[buffer(3)]]`, `[[thread_position_in_grid]]`) are read and stripped, and `threadgroup` arrays declared inside a
   kernel become `static` (one copy shared by the threads of the running threadgroup — exactly what threadgroup
   memory is);
3. replaces each `template [[host_name("matmul_q4")]] kernel ...` instantiation by a **launcher**: a C++ function
   that takes the dispatch's buffers, threadgroup memory and thread position and calls the kernel with each
   parameter in its place, plus an `emu_register("matmul_q4", launcher)` call.

`metal_emu.h` supplies the rest. Metal's address spaces stay distinct types through clang's
`__attribute__((address_space(n)))` (`device` = 1, `constant` = 2, `threadgroup` = 3), so overloads that differ only
in address space still compile; `float4`, `half2`, `uchar4` … are clang `ext_vector_type`s, which support `.xyzw`,
swizzles and element-wise arithmetic like Metal's vectors; `half` is `_Float16`.

---

## 18.3 Running a grid of threads on one CPU thread (`runtime.cpp`)

🧸 A class plays "everybody stand up when all are ready". The emulator is one teacher who walks from child to
child: she lets a child work until it says "I have to wait for the others", then goes to the next child. When
every child is waiting, they may all go on. If she walks a whole round and nobody can move, someone is waiting for
a friend who never comes — she stops the game and names the kernel.

📐 A dispatch is a grid of threadgroups; the emulator runs them one after another (the real backend's serial
command encoder gives the same ordering between dispatches). Inside a threadgroup, every GPU thread is a **fiber**
(`ucontext`, its own 256 KB stack). A fiber runs until it finishes or reaches a collective:

- `threadgroup_barrier`: it waits until every live thread of the threadgroup has arrived;
- `simd_sum`, `simd_max`, `simd_shuffle`, `simd_broadcast`, …: it writes its value into the simdgroup's exchange
  buffer and waits for every live lane of its simdgroup (32 lanes); then each lane computes the result from the
  buffer.

Two details matter for correctness:

- **Double buffering by phase.** A fast lane may leave one collective and reach the next while slow lanes are still
  reading the first one's values. Each simdgroup has two buffers and alternates; a buffer is reused only when every
  lane has *released* it (a count of departures, not of the current width, which shrinks when lanes exit).
- **Participation mask.** When some lanes of a simdgroup have finished (an early `return` on the last, partial
  tile), a reduction must still cover exactly the lanes that took part. The exchange records a mask of the lanes
  present, and `simd_sum` sums over the mask — not over "the first `width` lanes", which dropped lane 31 in an early
  version.

A scheduler round in which no fiber gets past anything is a **deadlock** (a barrier or simd operation inside
divergent code, which real GPUs leave undefined) and aborts with the kernel's name — a class of bug that on a real
GPU shows up as a hang or garbage.

### Simdgroup matrices

`simdgroup_float8x8` is a cooperative 8 × 8 matrix on Apple GPUs; how its 64 numbers are spread over the 32 lanes
is not documented. The emulator gives **every lane the whole 8 × 8 matrix**. That is valid because loads, stores
and `simdgroup_multiply_accumulate` are collective — every lane executes them with the same arguments — so
computing the full product in each lane gives each lane the collective result:

```
D = A · B + C        D_ij = Σ_k A_ik B_kj + C_ij,   i, j, k ∈ 0..7
```

It also enforces the rule the real kernels follow: code must never assume which lane holds which element (the flash
attention kernel rescales rows with a diagonal matrix product for exactly this reason, [08 §8.13](08-metal-backend.md#813-flash-attention-prefill-attention_fa)).

---

## 18.4 What the C API looks like from Java

`Metal.java` binds the shim's functions (`jm_create`, `jm_buffer_wrap`, `jm_pipeline`, `jm_dispatch`, `jm_commit`,
`jm_commit_wait`, `jm_wait_pending`, …) with the FFM API. With `-Djllm.metal.emulator=true` it loads the library named by
`-Djllm.metal.lib` instead of the dylib and skips the macOS-only checks; nothing else in the backend changes.
Buffers are plain host memory (on Apple Silicon they are too — unified memory), so `newBufferWithBytesNoCopy`
becomes "remember this pointer". A dispatch runs at once; `jm_commit` and `jm_commit_wait` are no-ops, and `jm_wait_pending`
returns immediately, so pipelined decoding ([08 §8.16](08-metal-backend.md#816-pipelined-greedy-decoding)) runs
with its full logic. `jm_run` (a recorded decode step, [08 §8.21](08-metal-backend.md#821-decode-replay-one-native-call-per-token))
reads the same 384-byte command records as the Objective-C shim and runs them one after the other through
`jm_dispatch`, so the recording, the patching and the verification of replayed steps are exercised here exactly as on a
Mac; only the native encoding loop itself is the shim's own.

---

## 18.5 What it catches, and what it cannot

What it verified during this work:

- every family in `scripts/arch-smoke.sh` (Llama, Mistral, Qwen2, Qwen3-MoE, GLM / GLM-4, Phi-3, Gemma 2 / 3, Mamba,
  DeepSeek-V2 / V3 with MLA and MoE, in five formats each):
  - Metal logits equal CPU logits within tolerance (105 checks), and pipelined decoding equals unpipelined;
  - KV caches written by the CPU backend load into the Metal backend;
  - the MoE models also run with experts on the CPU in four placements (80 checks), with exact usage counts and
    two slots sharing a placement;
  - this end-to-end run is what exercises `moe_sort` and the grouped MoE kernels, the router's remap and counting (with emulated atomics), `embed_token`,
    `argmax` and the Q6_K matrix path;
- attention (`attention_t` variants), the Q4_0 prefill matmuls and `attention_fa` (windows, soft-cap, half cache)
  against float64 numpy references (`kernel_tests.py`, 18 tests).

What it cannot tell you:

- **speed** — the real GPU's memory bandwidth, occupancy and tile shapes decide that; measure with `bench` on a Mac;
- **hardware-specific behaviour** beyond the language: the real lane layout of simdgroup matrices, half-precision
  rounding modes in the GPU's ALUs, driver bugs. The start-up self-check ([08 §8.14](08-metal-backend.md#814-the-kernel-self-check-metaltuner))
  covers that on the user's own device, by comparing each new kernel with the established one before using it;
- **the Objective-C shim itself** (`jllm_metal.m`), which the emulator replaces.
