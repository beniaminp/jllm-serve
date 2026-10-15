# 7. The CPU backend: SIMD with the Vector API — in detail

Code: `cpu/VectorKernels.java`, `cpu/ThreadPool.java`, `cpu/CpuBackend.java`

This chapter explains how the 25 lines of math from [05](05-transformer-math.md#513-the-whole-thing-in-25-lines)
run fast on a CPU, using nothing but the JDK: the Vector API for SIMD arithmetic, the FFM API to read
quantized weights off-heap, and a hand-written thread pool. Everything is tied to the exact method in the code.

---

## 7.0 The toolbox

### What SIMD is

A normal (scalar) instruction does one operation on one value: `acc = acc + a × b`. A **SIMD** instruction
(Single Instruction, Multiple Data) does the same operation on a whole *register* of values at once:

```
                  lane 0   lane 1   lane 2   lane 3          one 128-bit register = 4 floats
   a   =        [  0.5  ,   2.0  ,  −1.0  ,   4.0 ]
   b   =        [  3.0  ,   0.5  ,   2.0  ,   0.25]
   acc =        [  1.0  ,   1.0  ,   1.0  ,   1.0 ]
   ─────────────────────────────────────────────────────  fma(a, b, acc) = a×b + acc, all lanes at once
   result       [  2.5  ,   2.0  ,  −1.0  ,   2.0 ]
```

One instruction, four multiply-adds. The register width is fixed by the hardware:

| hardware | register | float lanes | byte lanes |
|---|---|---|---|
| Apple M-series (NEON) | 128-bit | 4 | 16 |
| x86 AVX2 | 256-bit | 8 | 32 |
| x86 AVX-512 | 512-bit | 16 | 64 |

A **lane** is one slot of the register. **FMA** ("fused multiply-add", `a*b + c` in one rounding step) is the
instruction every dot product is built from. Since a matrix–vector product is nothing but dot products
([05 §5.0](05-transformer-math.md#50-the-math-toolbox)), SIMD FMAs are the whole game.

### The Vector API (`jdk.incubator.vector`)

Java's Vector API lets you write that register arithmetic explicitly; the JIT compiler turns each call into
the native instruction. The classes used here:

```java
VectorSpecies<Float> F = FloatVector.SPECIES_PREFERRED;   // "the widest float shape this CPU has": 4 lanes on M1
int FL = F.length();                                       // 4

FloatVector a = FloatVector.fromArray(F, x, i);            // load lanes x[i .. i+FL) from a float[]
FloatVector b = FloatVector.fromMemorySegment(F, seg, off, LITTLE_ENDIAN);   // same, from off-heap memory
acc = a.fma(b, acc);                                       // acc = a*b + acc, per lane
float s = acc.reduceLanes(VectorOperators.ADD);            // add the lanes together → one float
a.intoArray(y, i);                                         // store lanes back

ByteVector v = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, seg, off, LITTLE_ENDIAN);  // 16 bytes
ByteVector lo = v.and((byte) 0x0F);                        // bitwise ops per byte lane
ByteVector hi = v.lanewise(VectorOperators.LSHR, 4);       // logical shift right per byte lane
FloatVector f = (FloatVector) lo.castShape(F, part);       // convert byte lanes → float lanes (see 7.3)
```

`castShape(F, part)` converts a group of byte lanes into float lanes. A 16-byte vector holds 16 bytes but a
float vector holds only `FL` floats, so the conversion is done in *parts*: part `p` converts bytes
`[p·FL, (p+1)·FL)`. With 4 float lanes there are 4 parts; with 16 lanes, one.

The module is still incubating, hence `--add-modules jdk.incubator.vector` on both `javac` and `java`, and the
startup line `WARNING: Using incubator modules`.

### Arithmetic intensity

The ratio **FLOPs per byte loaded from memory** decides whether a loop is limited by the memory system or by
the arithmetic units. A loop that loads 1 byte per multiply-add is memory-bound on every CPU; a loop that does
64 multiply-adds per byte is compute-bound. We will compute this for decode and prefill in 7.4.

---

## 7.1 What the backend owns

`CpuBackend` implements `Backend.forward(tokens, off, n, pos, logits)` for `n` tokens at positions
`[pos, pos+n)`. Its state (constructor):

| field | shape | size at ctx = 4096, batch = 64 (Qwen3-0.6B) | purpose |
|---|---|---|---|
| `keyCache[layer]`, `valCache[layer]` | `float[ctx × kvDim]` each | 28 × 2 × 4096 × 1024 × 4 B = **940 MB** | the KV cache ([05 §5.9](05-transformer-math.md#59-prefill-decode-and-the-kv-cache)) |
| `x` | `[batch × dim]` | 256 KB | the residual stream of the batch |
| `xb` | `[batch × dim]` | 256 KB | normalized copy |
| `q`, `att` | `[batch × qDim]` | 512 KB each | queries; attention output |
| `gateUp` | `[batch × 2·ffn]` | 1.5 MB | gate and up activations |
| `hidden` | `[batch × ffn]` | 768 KB | `silu(gate) ⊙ up` |
| `scratch` | thread-local `float[max(ctx, 4·max(ffn, dim, qDim))]` | 48 KB per thread | attention scores / dequantized rows |
| MoE: `routerLogits`, `assign`, `routeWeight`, `moeGateUp`, `moeHidden` | per-token | tiny | [12](12-mixture-of-experts.md) |

All buffers are row-major `[token][feature]`, so token `t`'s vector starts at `t × width`. They are allocated
once; a forward pass allocates nothing (important for the thread pool and for avoiding GC pauses).

The KV cache lives on the Java heap, which is why the launcher passes `-Xmx4g`. Weights are off-heap
(`MemorySegment`) and do not count against it.

---

## 7.2 The forward pass (`CpuBackend.forward`)

```java
for t < n:  embed.dequantizeRow(tokens[off+t], x, t*dim)              // 5.2
for each layer L:
    rmsnorm(x, xb, L.attnNorm, n, dim)                                  // 5.3
    matmul(L.wq, xb, q, 0, n, false)                                    // 5.4
    matmul(L.wk, xb, keyCache[l], pos*kvDim, n, false)                  //     written straight into the cache
    matmul(L.wv, xb, valCache[l], pos*kvDim, n, false)
    normRope(q, ...);  normRope(keyCache[l], pos*kvDim, ...)            // 5.5
    attention(l, n, pos)                                                // 5.6
    matmul(L.wo, att, x, 0, n, true)                                    // 5.7  acc=true: x += ...
    rmsnorm(x, xb, L.ffnNorm, n, dim)
    if L.moe: moe(L, n)  else: matmul(wGateUp); siluMul(n); matmul(wDown, hidden, x, acc=true)   // 5.8 / 12
if logits != null: rmsnormRow(last token); matmul(lmHead, xb, logits, 0, 1, false)             // 5.10
```

Two things to notice. `k` and `v` are produced *directly into* the KV cache at `pos × kvDim` — no copy. And
the `acc` flag turns a matmul into "add to the destination", which implements the residual connections without
a separate addition pass.

---

## 7.3 The core kernel: `dotQ4`, block by block

Everything hinges on computing `Σ_c W[r][c] · x[c]` for one quantized row `r` as fast as possible.
`VectorKernels.dotRow(w, r, x, xo)` dispatches to `dotQ4` (or `dotQ8`); here is one 32-weight block of
`dotQ4` with the 4-lane NEON shapes:

```
step 1   ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + b*16, LE)
         ┌────┬────┬────┬────┬ ─ ─ ┬────┐   16 bytes of the block
         │0x47│0x39│0x67│0x77│ ... │0x69│   byte j = q_j | q_{j+16} << 4     (see 06 §6.3)
         └────┴────┴────┴────┴ ─ ─ ┴────┘

step 2   lo = v.and(0x0F).sub(8)          hi = v.lanewise(LSHR, 4).sub(8)
         lo = [ −1, 1, −1, −1, −2, −8, 3, 1, 3, 1, 1, 0, −4, 2, 1, 1 ]   = weights 0 … 15 as signed bytes
         hi = [ −4, −5, −2, −1, 1, 0, 1, −2, 1, 1, −2, 4, 1, 3, −2, −2 ] = weights 16 … 31

step 3   halfDot(lo, x, xb)      → one FloatVector holding partial sums for weights 0…15
         halfDot(hi, x, xb + 16) → same for weights 16…31

step 4   acc = (halfDot(lo) + halfDot(hi)).fma(scale, acc)       scale = broadcast(d_b): one multiply per block, not per weight
```

Inside `halfDot` with 4 float lanes (`FL == 4`), the 16 signed bytes are converted four at a time and
multiplied with the matching 4 floats of `x`:

```
part 0: castShape(lo, 0) = [ −1,  1, −1, −1 ] (as floats)  ×  x[xb+0 .. xb+4)   → a
part 1: castShape(lo, 1) = [ −2, −8,  3,  1 ]              ×  x[xb+4 .. xb+8)   → b
part 2: castShape(lo, 2) = [  3,  1,  1,  0 ]              fma into a
part 3: castShape(lo, 3) = [ −4,  2,  1,  1 ]              fma into b
return a + b                                               a 4-lane vector of partial sums
```

The `FL == 8` branch needs two parts, `FL == 16` one. Note what `halfDot` returns: **a vector, not a number**.
Reducing the 4 lanes to one float (`reduceLanes`) is a comparatively slow cross-lane operation, so `dotQ4`
keeps a vector accumulator `acc` across all 32 blocks of the row and reduces **once at the end**:

```
for b < nb:  acc = (lo·x + hi·x).fma(scale_b, acc)      vector ops only
return acc.reduceLanes(ADD)                              one horizontal add per row
```

Per block on 128-bit NEON that is: 1 byte load, 2 bitwise ops, 2 subtracts, 8 `castShape` conversions,
8 float loads, 8 FMAs/multiplies, 1 add, 1 broadcast, 1 FMA — about 30 instructions for 32 multiply-adds. The
conversions dominate; this is the CPU path's bottleneck (7.9).

`dotQ8` is the same with two 16-byte loads and no nibble unpacking. `dequantRow` reuses the unpacking to write
floats out (`mul(scale).intoArray`) for the prefill path.

---

## 7.4 Decode vs prefill matmul (`CpuBackend.matmul`)

### Decode: `n == 1`, parallel over rows

```java
pool.forRange(rows, 16, (r0, r1) -> {
    for (int r = r0; r < r1; r++) {
        float v = VectorKernels.dotRow(wt, r, in, 0);
        out[outOff + r] = acc ? out[outOff + r] + v : v;
    }
});
```

Each output element is one `dotRow`; the rows are split across threads in chunks of ≥ 16. Every weight byte is
loaded exactly once, which is optimal for traffic. **Arithmetic intensity:** per 32 weights we load 16 bytes of
weights + 128 bytes of `x` (but `x` is 4 KB and stays in L1 cache) and do 32 multiply-adds → 2 FMAs per
weight byte from DRAM. At that ratio the loop *would* be memory-bound if the instruction count per block were
small — on NEON it is not, so we are instruction-bound instead ([7.9](#79-measured-numbers-and-the-bottleneck)).

### Prefill: `n > 1`, dequantize four rows once, reuse `n` times

If we called `dotRow` for each of 64 tokens, the nibble unpacking (the expensive part) would run 64 times per
weight. Instead:

```java
pool.forRange(groups, 1, (g0, g1) -> {              // groups = rows / 4
    float[] tmp = scratch.get();
    for (int g = g0; g < g1; g++) {
        for k < 4:  VectorKernels.dequantRow(wt, r + k, tmp, k * cols);   // 4 rows → floats, once
        for t < n:  VectorKernels.dot4(tmp, 0, cols, in, t * cols, out, outOff + t * rows + r, acc);
    }
});
```

`dot4` computes four dot products at once, sharing each load of `x`:

```java
for (i = 0; i < n; i += FL) {
    FloatVector xv = FloatVector.fromArray(F, x, xo + i);          // 1 load of x
    s0 = row0[i].fma(xv, s0);  s1 = row1[i].fma(xv, s1);           // 4 FMAs
    s2 = row2[i].fma(xv, s2);  s3 = row3[i].fma(xv, s3);
}
```

Arithmetic intensity now: per 4 × 64 = 256 FMAs we loaded the 4 rows once (dequantized, from L2/L1) and `x`
once per 4 rows. The expensive unpack happens `1/n` as often — this is why prefill reaches ~150 tok/s where
decode reaches ~17 on the same hardware. Rows left over when `rows % 4 != 0` fall back to `dotRow`.

---

## 7.5 Attention (`CpuBackend.attention`)

Work is split over `(token, head)` pairs — `n × 16` of them — so even decode (`n = 1`) keeps all cores busy.
For each pair, with `len = pos + t + 1` cached positions and the thread-local scratch `sc`:

```java
for s < len:  sc[s] = VectorKernels.dot(q, qo, kc, s * kvDim + ko, hd) * scale;  max = max(max, sc[s])
for s < len:  sc[s] = exp(sc[s] - max);  sum += sc[s]
Arrays.fill(att, qo, qo + hd, 0)
for s < len:  VectorKernels.axpy(sc[s] / sum, vc, s * kvDim + ko, att, qo, hd)     // att += w_s · v_s
```

`dot` is a two-accumulator SIMD dot product (two independent FMA chains hide the FMA latency); `axpy` is
`y += a·x` over 128 floats (32 FMA instructions). `Math.exp` is scalar and ~20 ns each — at `len = 4096` and
16 heads that is 1.3 ms per layer per token, a visible share of the step; a vectorized exp is a known next step.
The scratch array holds `ctx` scores, so no allocation happens per call. `ko = (h / group) * hd` selects the
shared KV head (GQA).

---

## 7.6 RMSNorm, RoPE, SiLU

These touch `dim` or `headDim` values per token and are scalar loops:

- `rmsnormRow`: `sumSquares` (SIMD `dot(x, x)`), then `out[i] = in[i] * scale * weight[i]`.
- `normRope`: per head, optional `rmsnormRow` in place (QK-norm), then the 64 rotation pairs using the
  precomputed `rope.cos/sin` tables at row `(pos + t) * 64`.
- `siluMul`: `hidden[i] = g / (1 + exp(−g)) * up[i]` over `ffn` values.

Together they are < 3 % of a decode step; they could be vectorized but it would not show up.

---

## 7.7 Mixture of experts on CPU (`CpuBackend.moe`, `route`)

Per token: `route` computes 128 router logits with `VectorKernels.dot`, a softmax, then 8 passes of argmax
(128 entries — cheaper than sorting), and renormalizes. Then two parallel loops:

1. `forRange(topK * 2m, 32, …)` — all 8 experts' gate/up rows at once; index `i` → slot `k = i / 2m`, row
   `r = i % 2m`, weight row `assign[k] * 2m + r` in the stacked tensor.
2. SwiGLU per slot (scalar).
3. `forRange(dim, 8, …)` — each output row sums `routeWeight[k] · dotRow(expDown, assign[k]*dim + r, h, k*m)`
   over the 8 slots and adds to `x`. One writer per element, no synchronization.

Two pool jobs per token-layer instead of sixteen. Details and the math in [12](12-mixture-of-experts.md).

---

## 7.8 The thread pool (`ThreadPool`)

### Why not `ForkJoinPool` / `parallelStream`

A decode step issues roughly **340 parallel loops** (12 per layer × 28 + the LM head), each lasting 20–200 µs.
Standard executors hand off work through queues and park/unpark OS threads; the ~50 µs wake-up latency of a
parked thread would exceed the work itself. `ThreadPool` is built for exactly this pattern: many tiny jobs,
issued back to back from one thread.

### Design

- `threads − 1` daemon **workers**; the **caller participates** in every job (it would otherwise idle).
- A job is published by writing one `volatile Job current` field. Workers grab chunks with
  `next.getAndIncrement()` and count completions in `done`.
- After finishing a job a worker **spins** for up to `SPIN_NANOS = 200 µs` waiting for the next one
  (`Thread.onSpinWait()`), then **parks**. Between two matmuls of the same layer the gap is microseconds, so
  workers are almost always still spinning and pick up the next job with no OS involvement.

```
time ──────────────────────────────────────────────────────────────────────────▶
caller:   forRange(rmsnorm) ─ forRange(wq) ───── forRange(wk) ─── ... ─ sample token (~1 ms) ─ forRange(...)
worker 1:   work │spin│ work │spin│ work │spin...                      │spin 200µs│ park ──▶ unpark │ work
worker 2:   work │spin│ work │spin│ work │spin...                      │spin 200µs│ park ──▶ unpark │ work
                                                         `current` written ──┘   LockSupport.unpark(w) if w.parked
```

### `forRange(n, grain, body)` chunking

```
tasks = min(threads × 4, max(1, n / grain))          e.g. rows = 2048, grain = 16, threads = 10 → 40 tasks
chunk = ceil(n / tasks)                              → 52 rows per task
task t covers [t·chunk, min(n, (t+1)·chunk))
```

More tasks than threads (4×) lets a thread that finishes early take another chunk, evening out imbalance (e.g.
when one core is also running the JIT). `grain` prevents splitting trivially small loops; `tasks ≤ 1` runs
inline without touching the pool.

### The lost-wakeup argument

The dangerous interleaving is: a worker decides to park *just as* the caller publishes a job and sees
`parked == false`, so nobody unparks it and the job waits 200 µs+ for nothing. The code avoids it with a
re-check after publishing `parked`:

```
worker:   if (current == last)  { parked = true;  if (current == last) park(); parked = false; }
caller:   current = job;        if (w.parked) unpark(w);
```

Both `current` and `parked` are `volatile`, so the writes and reads are sequentially consistent: either the
worker's second read of `current` sees the new job (and skips the park), or its write `parked = true` is
visible to the caller's read (which then unparks). A spurious wake-up is harmless — the loop re-checks.
Exceptions thrown by a chunk are stored and rethrown on the caller after the job completes.

---

## 7.9 Measured numbers and the bottleneck

Apple M1 Pro, 10 threads, 128-bit NEON, Qwen3-0.6B Q4_0, 413-token prompt:

| | tok/s | ms/token | weight bytes per token | implied traffic |
|---|---|---|---|---|
| prefill | ~153 | 6.5 | 450 MB / 64 tokens | — |
| decode | ~17 | 59 | 450 MB | 7.6 GB/s |

7.6 GB/s is far below the ~200 GB/s the chip can deliver, so decode is **not** memory-bound here. Count the
work instead: 450 MB of Q4 weights = 14 million 32-weight blocks per token; at ~30 instructions per block
(7.3) that is ~420 M instructions, spread over 10 cores at ~3 GHz with maybe 2 vector instructions per cycle
→ ~7 ms if perfectly scheduled, ~59 ms observed — the gap is the `castShape` conversions (not single
instructions on NEON), the thread-pool synchronization of 340 jobs, and the scalar `exp` in attention at
this context length. On AVX-512 the same source code needs 4× fewer instructions per block.

---

## 7.10 What to optimize next

In order of expected payoff on this CPU:

1. **Int8 activations with integer dot products.** Quantize `x` to Q8 per 32-block too, then multiply bytes
   with bytes using 16-lane `ShortVector` multiplies and widen to int — 16 lanes instead of 4, and no
   byte→float conversions. This is what llama.cpp's `vec_dot_q4_0_q8_0` does and is worth 2–4×.
2. **Vectorized exp** in attention and SiLU (a polynomial approximation on `FloatVector`), removing the scalar
   `Math.exp` that dominates at long contexts.
3. **f16 KV cache**: halves the 940 MB and the attention read traffic; convert with `Float.float16ToFloat`
   lanes on load.
4. **Fuse RMSNorm into the following matmul** and SiLU into the gate/up matmul to cut passes over the
   activation and the number of pool jobs.
5. **Batch decode across requests** (continuous batching) so the weight read serves several tokens.

---

## 7.11 Summary

- The Vector API gives Java explicit SIMD: `FloatVector`/`ByteVector` lanes, `fma`, one `reduceLanes` per row.
- `dotQ4` loads 16 bytes, splits nibbles into two 16-lane vectors, converts to floats in parts, FMAs with `x`,
  applies the block scale once; weights are read exactly once per token.
- Decode parallelizes over output rows; prefill dequantizes four rows once and reuses them for all tokens
  (`dot4`), which is why it is ~9× faster per token.
- Attention runs per `(token, head)` with thread-local scratch; MoE adds a router and two loops per token.
- A spin-then-park pool with caller participation makes 340 micro-jobs per token cheap; the lost-wakeup race is
  closed by a volatile re-check.
- On 128-bit NEON the kernel is instruction-bound (byte→float conversions), not memory-bound; integer dot
  products are the fix.
