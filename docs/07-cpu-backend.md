# 7. The CPU backend: SIMD with the Vector API

Code: `cpu/VectorKernels.java`, `cpu/ThreadPool.java`, `cpu/CpuBackend.java`

## SIMD in one paragraph

A modern CPU core can execute one instruction on several numbers at once: NEON on Apple Silicon handles 4 floats
per 128-bit register, AVX2 8, AVX-512 16. The JDK's **Vector API** (`jdk.incubator.vector`) exposes this
portably: you write `a.fma(b, acc)` on `FloatVector` values and the JIT compiles it to the native instruction.
`FloatVector.SPECIES_PREFERRED` picks the widest shape the hardware supports; the kernels are written for 4, 8
and 16 lanes (the `FL == …` branches in `halfDot`) and fall back to scalar code otherwise.

The module is still incubating, hence `--add-modules jdk.incubator.vector` on both `javac` and `java`.

## The core kernel: dot product of a quantized row

`dotRow(w, r, x)` computes `Σ_c W[r][c] · x[c]` where `W[r]` is stored as Q4_0 blocks. Per block of 32 weights:

```java
ByteVector v  = ByteVector.fromMemorySegment(B16, seg, q + b*16, LE);  // 16 bytes = 32 nibbles
ByteVector lo = v.and((byte) 0x0F).sub((byte) 8);                     // weights 0..15
ByteVector hi = v.lanewise(LSHR, 4).sub((byte) 8);                     // weights 16..31
acc = halfDot(lo, x, xb).add(halfDot(hi, x, xb + 16)).fma(scale, acc); // acc += scale * (lo·x + hi·x)
```

`halfDot` converts byte lanes to float lanes (`castShape`) and multiplies with the matching slice of `x`. The
block scale is applied once per block instead of once per weight, and the lane accumulator `acc` is reduced to a
scalar only once at the end of the row (`reduceLanes(ADD)`). Weights are read from the off-heap `MemorySegment`
directly — no copies, no boxing.

Q8_0 is the same with two 16-byte loads and no nibble unpacking.

## Decode: one thread per group of rows

For `n = 1` (`matmul` in `CpuBackend`), each output element is independent, so the pool splits the row range
across threads; every thread streams its rows' bytes through `dotRow`. Memory traffic equals the matrix size,
which is optimal; the only cost is that each core must fetch from DRAM, so more cores ≈ more bandwidth until the
memory controller saturates.

## Prefill: dequantize once, reuse across tokens

For `n > 1` reading quantized weights per token would re-do the nibble unpacking `n` times. Instead the pool
hands each thread a group of 4 rows, which it dequantizes once into a thread-local buffer (`dequantRow`, SIMD),
and then `dot4` multiplies those 4 float rows against each of the `n` token vectors. `dot4` loads each `x`
vector chunk once and feeds it to 4 accumulators — 4 FMAs per load instead of 1.

## Attention

`attention()` parallelizes over `(token, head)` pairs. For each pair it does the score loop with the SIMD
`dot`, the softmax in scalar code (the `exp` calls dominate anyway), and the value accumulation with `axpy`
(`y += a·x`). Scores for up to `ctx` positions live in a thread-local scratch array.

## RMSNorm, RoPE, SiLU

These touch `dim` or `headDim` numbers per token — negligible next to the matmuls — and are written as plain
loops. `sumSquares` reuses the SIMD `dot`.

## The thread pool

`java.util.concurrent` pools are built for coarse tasks; a decode step issues ~340 tiny parallel loops (12 per
layer × 28 + head), each taking tens of microseconds. Parking and unparking OS threads at that rate would cost
more than the work. `ThreadPool` therefore:

- keeps `threads − 1` daemon workers that **spin** for up to 200 µs after finishing a job before parking;
- publishes a job as a single volatile `current` field; workers grab chunk indices with `getAndIncrement`;
- has the **calling thread participate** in every job (it is otherwise idle), and spin-wait for completion;
- propagates the first exception from a worker.

`forRange(n, grain, body)` splits `[0, n)` into at most `4 × threads` chunks so faster threads pick up slack.

## Memory

The KV cache is `float[layers][ctx × kvDim]` for keys and the same for values: 28 × 4096 × 1024 × 4 B × 2 =
**939 MB** at `--ctx 4096` (hence `-Xmx4g` in the launcher). Halve `--ctx` if memory is tight. The quantized
weights are off-heap and not counted against `-Xmx`.

## Measured

Apple M1 Pro, 10 threads, 128-bit NEON: ~153 tok/s prefill, ~17 tok/s decode. A 450 MB model at 17 tok/s is only
7.6 GB/s of weight traffic — far below the chip's bandwidth — so the CPU path is bound by instruction throughput
(nibble unpacking, byte→float conversion with only 4 lanes), not memory. See [11](11-performance.md).
