# 7. The CPU backend: SIMD with the Vector API — in detail

Code: `cpu/VectorKernels.java`, `cpu/ThreadPool.java`, `cpu/CpuMatmul.java`, `cpu/CpuBackend.java`, `cpu/KvCache.java`,
`cpu/Int8Kernels.java`, `cpu/MlaCpu.java`, `cpu/MambaCpuBackend.java`, `cpu/NativeCpu.java` and its C library
`src/main/native/jllm_cpu.c` ([7.16](#716-native-int8-kernels-libjllm_cpu-nativecpu))

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
| `kv.key[layer]`, `kv.val[layer]` (`KvCache`) | `float[ctx × kvDim]` each | 28 × 2 × 4096 × 1024 × 4 B = **940 MB** | the KV cache ([05 §5.9](05-transformer-math.md#59-prefill-decode-and-the-kv-cache)) |
| `x` | `[batch × dim]` | 256 KB | the residual stream of the batch |
| `xb` | `[batch × dim]` | 256 KB | normalized copy |
| `q`, `att` | `[batch × qDim]` | 512 KB each | queries; attention output |
| `gateUp` | `[batch × 2·ffn]` | 1.5 MB | gate and up activations |
| `hidden` | `[batch × ffn]` | 768 KB | `silu(gate) ⊙ up` |
| `scratch` (`CpuMatmul`) | thread-local `float[max(ctx, experts, 4·maxCols)]`, `maxCols` the widest tensor of the model | 48 KB per thread | router logits, four dequantized rows (`MlaCpu` has its own tile scratch, like `KvCache`) |
| `KvCache.scratch` | thread-local, `float[ctx + …]` for f32 decode, grown on demand to `tile·group·(pos+n) + 128·headDim + tile·group` for prefill | up to 8 MB per thread | attention scores of one tile and one block of K / V rows ([7.5](#75-attention-kvcacheattention)) |
| MoE: `routerLogits`, `assign`, `routeWeight`, `moeGateUp`, `moeHidden` | per-token | tiny | [12](12-mixture-of-experts.md) |

All buffers are row-major `[token][feature]`, so token `t`'s vector starts at `t × width`. They are allocated
once; a forward pass allocates nothing (important for the thread pool and for avoiding GC pauses).

The KV cache lives on the Java heap, which is why the launcher passes `-Xmx4g`. Weights are off-heap
(`MemorySegment`) and do not count against it. With `backend.kv-dtype=f16` the cache is `short[]` and half the
size, with `q8` int8 plus a scale per row, a little over a quarter ([7.13](#713-the-f16-kv-cache-kvcache)).

---

## 7.2 The forward pass (`CpuBackend.forward`)

```java
for t < n:  embed.dequantizeRow(tokens[off+t], x, t*dim)              // 5.2
for each layer l, with its LayerPlan p = (attention, residual, ffn):
    rmsnorm(x, xb, L.attnNorm, n, dim)                                  // 5.3
    p.attention.run(l, n, pos)                                          // 5.4-5.6: GqaStep (below) or MlaCpu.run
    p.residual.add(L.wo, att, L.postAttnNorm, n)                        // 5.7  x += o_proj(att)   (post-norm families: x += rmsnorm(o_proj(att)))
    rmsnorm(x, xb, L.ffnNorm, n, dim)
    p.ffn.run(L, n)                                                     // 5.8 / 12: DenseGlu, moe, or moe + shared experts
if logits != null: rmsnormRow(last token); mm.mul(lmHead, xb, logits, 0, 1, false)             // 5.10  (+ Gemma's final soft-cap)

GqaStep.run: matmul(w.wq, xb, q);  matmul(w.wk, xb, kStage);  matmul(w.wv, xb, vStage)         // k/v of the chunk, [n, kvDim] staging
           (+ bias)  normRope(q);  normRope(kStage)                                            // 5.5
           kv.store(l, pos, kStage, vStage, n)                           // into the KvCache: f32 copy or f16 rounding (7.13)
           kv.attention(pool, l, n, pos, q, att, window, scale, softcap)                        // 5.6, 7.5
DenseGlu.run:  matmul(w.gateUp, xb, gateUp);  siluMul | geluMul;  residual.add(w.down, hidden, postMlpNorm, n)
```

The three strategy objects of a layer (`LayerPlan`) are resolved once in the constructor from the layer's
`LayerSpec` and its weights - which attention (GQA or DeepSeek MLA), which residual rule (plain add, or the Glm4 / Gemma
post-norm add), which feed-forward (dense GLU with SiLU or GELU, routed experts, routed plus shared experts) -
so the per-layer loop has no conditionals on optional pieces; each strategy runs exactly the statements the
straight-line code ran, in the same order.

Two things to notice. `k` and `v` are produced into two small staging buffers (`[batch × kvDim]`), normalized
and rotated there, and only then stored into the `KvCache` at row `pos` — an `arraycopy` for f32, a rounding pass
for f16 ([7.13](#713-the-f16-kv-cache-kvcache)). And the `acc` flag turns a matmul into "add to the destination", which implements the residual connections without
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

`CpuBackend.matmul` is a one-line wrapper around `CpuMatmul.mul`. `CpuMatmul` is the one quantized matmul shared by
`CpuBackend`, `MlaCpu` and `MambaCpuBackend`, so everything below (and the int8 path of [7.10](#710-int8-activations-integer-dot-products-int8kernels))
applies to all three. It also owns the thread-local row scratch.

### Decode: `n == 1`, parallel over rows

```java
pool.forRange(rows, 16, (r0, r1) -> {
    for (int r = r0; r < r1; r++) {
        float v = VectorKernels.dotRow(wt, r, in, 0);
        out[outOff + r] = acc ? out[outOff + r] + v : v;
    }
});
```

Each output element is one `dotRow` (or `Int8Kernels.dotRow` when int8 activations apply, 7.10); the rows are split across threads in chunks of ≥ 16. Every weight byte is
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
        for t < n:  dot4x4 four tokens at a time, dot4 for the last ones      // out[t, r..r+4) (+)= ...
    }
});
```

`dot4` computes four dot products at once, sharing each load of `x`; `dot4x4` does the same for four rows against
four tokens (16 accumulators, each row load shared by four tokens), and the loop above uses it for every group of
four tokens and `dot4` for the remainder:

```java
for (i = 0; i < n; i += FL) {
    FloatVector xv = FloatVector.fromArray(F, x, xo + i);          // 1 load of x
    s0 = row0[i].fma(xv, s0);  s1 = row1[i].fma(xv, s1);           // 4 FMAs
    s2 = row2[i].fma(xv, s2);  s3 = row3[i].fma(xv, s3);
}
```

Arithmetic intensity now: per 4 × 64 = 256 FMAs we loaded the 4 rows once (dequantized, from L2/L1) and `x`
once per 4 rows. The expensive unpack happens `1/n` as often — this is why prefill reaches ~240 tok/s where
float decode reaches ~13 on the same hardware. Rows left over when `rows % 4 != 0` fall back to `dotRow`. With `-Djllm.cpu.deterministic` the prefill branch
is skipped and the tokens go one by one through the decode path ([7.14](#714-prefill-and-decode-are-not-bit-identical-results-depend-on-history)).

---

## 7.5 Attention (`KvCache.attention`)

The attention loop lives with the cache (`KvCache.F32` / `KvCache.F16` / `KvCache.Q8`, [7.13](#713-the-f16-kv-cache-kvcache));
`CpuBackend` only passes the layer's window, scale and soft-cap. For one query head with `len = pos + t + 1`
cached positions the math is:

```
for s < len:  sc[s] = (q · K[s]) * scale                 (softcap: tanh(s/cap)*cap)
p = softmax(sc[0..len))                                   max, exp(s - max), sum; p = e / sum
att = Σ_s p[s] · V[s]                                     ascending s
```

`ko = (h / group) * hd` selects the shared KV head (GQA).

**Layout.** A dot product per key, `q · K[s]`, ends in a horizontal reduction of a vector (`reduceLanes`: a
shuffle tree), and at `headDim = 128` with 16 float lanes that reduction costs as much as the eight FMAs before
it. The cache therefore stores K **transposed in chunks of 32 keys**: chunk `c` of kv head `g` is `[hd][32]`, so
one vector load holds element `i` of 16 (AVX-512) or 8 (AVX2) consecutive keys, and

```
score[s .. s+FL) = Σ_i  broadcast(q[i]) · Kchunk[i][s .. s+FL)        hd FMAs, no reduction
```

V stays row-major (`[ctx, kvDim]`): `att += p[s] · V[s]` is already a broadcast FMA per output vector.

**Tiles.** One task per `(kv head, tile of up to 64 consecutive tokens)`; the tile serves `tile × group` query
rows. Keys are walked in blocks of 128 (`KEY_BLOCK`):

```
for each block of K chunks any row of the tile sees:
    K chunks (f32: in place; f16: converted once into scratch)
    for each chunk, the rows that see one of its keys, four at a time (score4; then score2 / score1):
        8 accumulators = 4 rows × 2 key vectors; per head element 2 key loads + 4 broadcasts feed 8 FMAs
for each row: softmax over its keys in ascending s, then p = e · (1/sum) in place; att[row] = 0
for each block of V rows (in place / converted once):
    rows four at a time (accumulate4), two output vectors per row in registers:
        for s ascending: acc[r][c] = V[s][c] · p[r][s] + acc[r][c]
```

Whichever variant computes it, every score is the same sequential chain of FMAs over the head elements followed
by `* scale`, and every output element adds its keys in ascending order, so prefill (big tiles) and decode (a
one-token tile) are bit-identical. `SmokeTest.kvCacheAttention` checks prefill against decode and against a
double-precision reference, for f32 and f16, with and without window and soft-cap, at a head size that is whole
vectors and one that is not (that one falls back to `attentionTiledSimple`, one key at a time over expanded rows).
Scores of keys outside a row's window or past its position (the rest of a chunk) are computed and ignored.

**Decode with few kv heads.** A one-token tile gives `kvHeads` tasks. When that is fewer than the pool's threads
(4 kv heads on 10 cores, common in 7B coding models), `attentionDecode` runs three phases instead: scores by
(kv head, range of key chunks), softmax by query head, outputs by (kv head, slice of the head's elements). Same
operations per element, so still bit-identical. With 2 kv heads on 4 threads, 10 000 keys: 2.1 → 1.05 ms per
layer. With 8 kv heads on 4 threads the tiled path stays faster (a head's work stays on one core).

**Measured** with Qwen3-0.6B (28 layers, 16/8 heads, f16 cache, `ctx = 32768`, 4-core Xeon AVX-512):

| | before (per-key dots) | transposed K, 64-token tiles, fast exp |
|---|---|---|
| one layer, 64 tokens at position 6 000 | 44 ms | 21 ms |
| one layer, decode at position 10 000 | 3.2 ms (f32 2.8) | 2.35 ms (f32 1.6) |
| 6 033-token prompt (a coding agent's first request) | 115 s (313 s with ctx-sized tiles) | 60 s |
| decode after that prompt | ~2–3 tok/s | 7.6 tok/s |

About 40 % of the remaining "before" time was scalar `Math.exp` in the softmax, hence fast math by default
([7.15](#715-fast-math-jllmcpufastmath)). Tasks are handed out heaviest tile first (`forRangeFine`, chunk 1): later
tokens see more keys, and the fine grain keeps the tail short.

**MLA (DeepSeek).** `MlaCpu` replaces the q/k/v projections and this attention for latent-attention models. Its
cache holds one row per token and layer of `kvLora + rope` numbers (576 floats for DeepSeek-V2-Lite instead of the
20 KB of expanded per-head keys and values; half the bytes with `backend.kv-dtype=f16`). `W_kb_k` is absorbed into
the query, the softmax runs over the latent rows with the same tiling (all heads share the one cached row), and
`W_kb_v` is applied once to the weighted latent sum. Its matmuls are the shared `CpuMatmul` ([14](14-model-families.md)).

---

## 7.6 RMSNorm, RoPE, SiLU

These touch `dim` or `headDim` values per token:

- `rmsnormRow`: `sumSquares` (SIMD `dot(x, x)`), then `VectorKernels.scaleMul`, `out[i] = (in[i] * scale) * weight[i]`
  on float lanes with the same two roundings as the scalar expression.
- `normRope`: per head, optional `rmsnormRow` in place (QK-norm), then the 64 rotation pairs
  (`VectorKernels.rope`, lanes with separate multiplies and add/subtract so the bits match the scalar formula)
  using the precomputed `rope.cos/sin` tables at row `(pos + t) * 64`. GLM's interleaved pairs
  (`InterleavedRope.rotate`) stay scalar.
- `siluMul`: `hidden[i] = g / (1 + exp(−g)) * up[i]` over `ffn` values; the `Math.exp` is scalar.

Together they are < 3 % of a decode step. For one token (`n == 1`) the norms and RoPE run inline on the calling
thread; for a chunk they are pool jobs over tokens or (token, head) pairs.

---

## 7.7 Mixture of experts on CPU (`CpuBackend.MoeFfn`, `MoeExperts`)

Per token: `route` computes 128 router logits with `VectorKernels.dot` (parallel over the experts for a decode
token; in prefill every token is routed by its own thread, DeepSeek's router included, which keeps its logits,
working space and top-k buffers per thread), a softmax, then 8 passes of argmax (128 entries — cheaper than
sorting), and renormalizes. Then two parallel loops:

1. `forRange(topK * 2m, 32, …)` — all 8 experts' gate/up rows at once; index `i` → slot `k = i / 2m`, row
   `r = i % 2m`, weight row `assign[k] * 2m + r` in the stacked tensor. (Between the two loops: SwiGLU per slot, scalar.)
2. `forRange(dim, 8, …)` — each output row sums `routeWeight[k] · dotRow(expDown, assign[k]*dim + r, h, k*m)`
   over the 8 slots and adds to `x`. One writer per element, no synchronization.

Three pool jobs per token-layer (router, gate/up, down) instead of sixteen. The experts themselves live in
`MoeExperts` (`token`, `batched`), which the Metal backend also uses for the experts it leaves on the CPU
([08 §8.18](08-metal-backend.md#818-experts-on-the-cpu-backendmetalcpu-moe)). Prefill (`MoeExperts.batched`) routes the
tokens in parallel, groups the `n × topK` slots by expert with a counting sort and runs each used expert once over
all its slots (`expertMatmul`: four rows dequantized once, `dot4x4` over four slots at a time); the per-token combine
`x[t] += Σ_k w_k · down_k` is `VectorKernels.combine`, lanes that follow the scalar sum order exactly (product
rounded, added in `k` order from zero, no FMA, total added last) so it is bit-identical to the loop it replaced.
Details and the math in [12](12-mixture-of-experts.md).

---

## 7.8 The thread pool (`ThreadPool`)

### Why not `ForkJoinPool` / `parallelStream`

A decode step of Qwen3-0.6B issues **225 parallel loops** (8 per layer — the q, k, v and o projections,
attention, gate/up, SiLU, down; the norms and RoPE of a single token run inline — × 28 layers + the LM head),
each lasting 20–200 µs.
Standard executors hand off work through queues and park/unpark OS threads; the ~50 µs wake-up latency of a
parked thread would exceed the work itself. `ThreadPool` is built for exactly this pattern: many tiny jobs,
issued back to back from one thread.

### Design

- `threads − 1` daemon **workers**; the **caller participates** in every job (it would otherwise idle).
- There is no job object. The pool holds one set of job fields (body, size, chunk) that the caller writes and
  then publishes by setting one `AtomicLong next` to `epoch << 32 | tasks << 16`: the job's epoch, its task
  count and (low 16 bits) the next task index in one word (tasks per job are capped at `MAX_TASKS` so the index can
  never carry into the count). Workers claim tasks with `next.getAndIncrement()`;
  a claim is valid when its index is below the task count *of the same word*, and only then does the worker
  read the job fields — at that moment the job is provably still running (its claimed task is not done), so
  the fields are that job's. A worker that lagged behind the end of one job and claims a task of the next just
  reloads the fields and runs it. (The first version read the fields right after seeing a new epoch, which can
  race with the caller writing the *next* job's fields; `SmokeTest.threadPool` caught it.) Completions are
  counted in `done`. Nothing is allocated per job.
- The first exception a task throws is kept (`compareAndSet`) and rethrown on the caller once every task has been
  claimed; tasks claimed after it are counted as done without running, so a failing chunk ends the job quickly
  instead of after all its chunks. `SmokeTest.threadPool` hammers 20 000 back-to-back jobs and the failure path.
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

`forRangeFine(n, chunk, body)` is the variant with chunks of exactly `chunk` items and no cap on the task count, for
loops whose items cost very different amounts (causal attention, [7.5](#75-attention-kvcacheattention)).

More tasks than threads (4×) lets a thread that finishes early take another chunk, evening out imbalance (e.g.
when one core is also running the JIT). `grain` prevents splitting trivially small loops; `tasks ≤ 1` runs
inline without touching the pool.

### The lost-wakeup argument

The dangerous interleaving is: a worker decides to park *just as* the caller publishes a job and sees
`parked == false`, so nobody unparks it and the job waits 200 µs+ for nothing. The code avoids it with a
re-check after publishing `parked`:

```
worker:   if (epoch(next) == last)  { parked = true;  if (epoch(next) == last) park(); parked = false; }
caller:   next.set(epoch << 32 | tasks << 16);    if (w.parked) unpark(w);
```

`next` is atomic and `parked` is `volatile`, so the writes and reads are sequentially consistent: either the
worker's second read of the epoch sees the new job (and skips the park), or its write `parked = true` is
visible to the caller's read (which then unparks). A spurious wake-up is harmless — the loop re-checks.

---

## 7.9 Measured numbers and the bottleneck

Apple M1 Pro, 10 threads, 128-bit NEON, Qwen3-0.6B Q4_0, 413-token prompt (`./jllm bench`; the full table with
Metal and the MoE model is in [11.1](11-performance.md#111-measured)):

| | prefill tok/s | decode tok/s | decode ms/token | implied decode traffic (450 MB/token) |
|---|---|---|---|---|
| float activations (`--backend.cpu.int8-activations false`) | ~237 | ~13 | 77 | 5.9 GB/s |
| int8 activations (default, [7.10](#710-int8-activations-integer-dot-products-int8kernels)) | ~275 | ~51 | 20 | 23 GB/s |

For the float path, 5.9 GB/s is far below the ~200 GB/s the chip can deliver, so decode is **not** memory-bound here. Count the
work instead: 596 M weights = ~19 million 32-weight blocks per token; at ~30 instructions per block
(7.3) that is ~560 M instructions, spread over 10 cores at ~3 GHz with maybe 2 vector instructions per cycle
→ ~9 ms if perfectly scheduled, ~77 ms observed — the gap is the `castShape` conversions (not single
instructions on NEON), the thread-pool synchronization of 225 jobs, and the scalar `exp` in attention at
this context length. On AVX-512 the same source code needs 4× fewer instructions per block. The int8 path (7.10) removes the conversions and lifts decode to ~51 tok/s (23 GB/s), still well under the memory limit.

---

## 7.10 Int8 activations: integer dot products (`Int8Kernels`)

Code: `cpu/Int8Kernels.java`; switch: `backend.cpu.int8-activations` (default `true`); used by
`CpuBackend.matmul` for `n == 1` (decode) and by the MoE per-token loops. This is llama.cpp's
`q4_0 × q8_0` scheme, and it removes the bottleneck of 7.9.

### The idea

`dotQ4` spends most of its ~30 instructions per block converting bytes to floats so that 4-lane float FMAs
can consume them. Instead, quantize the *activation* vector `x` to 8-bit integers as well (once per matmul,
not per row — `x` is one vector of 1 024–3 072 floats, the weights are millions of rows), and multiply
integers with integers:

```
x block b (32 floats)  →  d_b = max|x| / 127,   q_i = rint(x_i / d_b)          (Int8Kernels.quantize)
row block b            →  s_b (the weight scale), w_i = nibble_i − 8
dot                    =  Σ_b  (s_b · d_b) · Σ_{i<32} w_i · q_i
```

The inner sum `Σ w_i q_i` is an exact integer: |w| ≤ 8 and |q| ≤ 127, so each product is at most 1 016
and 32 of them fit easily in a 16-bit lane sum. One 128-bit register holds **8 short lanes**, so one
multiply does 8 products where a float FMA does 4 — and, more importantly, the inputs need no conversion
to float at all: bytes are unpacked to 16-bit lanes with shifts and masks, which are single instructions.
Only the per-block integer total is converted to float, once per block instead of once per lane-group.

### One block, 128-bit lanes

```
ShortVector v = 16 packed Q4 bytes loaded as 8 shorts    lane k = byte 2k | byte 2k+1 << 8
w_even_lo = (v        & 0xF) − 8     weights  0, 2, …, 14     (low nibbles of the even bytes)
w_odd_lo  = (v >>> 8  & 0xF) − 8     weights  1, 3, …, 15
w_even_hi = (v >>> 4  & 0xF) − 8     weights 16, 18, …, 30    (high nibbles of the even bytes)
w_odd_hi  = (v >>> 12 & 0xF) − 8     weights 17, 19, …, 31
p  = w_even_lo·x[0..8) + w_odd_lo·x[8..16) + w_even_hi·x[16..24) + w_odd_hi·x[24..32)   (8 short lanes, exact)
i  = widen(p)                                     4 int lanes: lane j = p[2j] + p[2j+1]
acc = float(i) · (s_b · d_b) + acc                4 float lanes, reduced once per row
```

The quantized activations are stored in exactly the order the unpack produces — per block
`[x0 x2 … x14] [x1 x3 … x15] [x16 x18 … x30] [x17 x19 … x31]` — so the kernel loads them with four
plain 8-short loads and never shuffles (`Int8Kernels.permutedIndex`). Q8_0 rows use the same structure
with the even/odd bytes of each 16-byte half sign-extended by shifts; their products reach 127 × 127, so
only two are summed per short lane before widening.

Per block that is: 1 weight load, 4 activation loads, 11 shift/mask/subtract ops, 4 multiplies, 3 adds,
4 ops to widen, 3 to convert to float, 1 FMA — about 31 instructions for 32 multiply-adds, all of them
single-cycle-throughput integer or float ops with no cross-lane work. The float path's ~30 instructions
include 8 `castShape` conversions that are 2–3 instructions each on NEON.

### Error analysis

Rounding `x` to 8 bits adds an error of at most `d_b / 2` per element, uniformly distributed, i.e. an RMS
error of `d_b / √12`. With `d_b = max|x| / 127` and the typical block maximum of a Gaussian vector at
~2.5 σ, that is ≈ 0.6 % of σ per element — the same order as the Q4 weight error the model already
tolerates, and independent across elements, so it averages out over a 1 024-long dot product.
`SmokeTest.int8Kernels` measures it: on random rows and activations with outliers (one in 97 elements 8× larger,
which inflates that block's scale) the RMS relative error of the dot product is **0.9 %** for both Q4 and Q8
weights, the integer lanes match a scalar reference exactly, and the SIMD quantizer is bit-identical to its
scalar reference. Greedy outputs with the switch on and off are compared in [11](11-performance.md#111-measured).

### What the JIT needs from a Vector API kernel

Three things were learned the hard way while writing this kernel; they apply to `VectorKernels` too.

1. **Only C2 code is fast.** The interpreter and C1 execute every `lanewise`/`fromArray` call through the
   Java fallback (a boxed vector object and a lambda per lane): the same kernel runs **500–1 000× slower**
   until C2 has compiled it. A kernel that is "slow" is almost always one that has not been compiled yet, or
   whose C2 compilation bailed out or was deoptimized. `bench` discards a warm-up round for this reason, and
   the first second of a `run` is dominated by it.
2. **C2 compile time is the real cost, and it depends on the loop shape.** The inlined form of one block of
   this kernel is thousands of IR nodes, and C2 unrolls small loop bodies before range-check elimination on
   the `long` memory-segment offsets. `-XX:+LogCompilation` showed the Q4 kernel's C2 task taking **40 ms**,
   while the same Q8 arithmetic written as one iteration per block — i.e. *two* `fromMemorySegment` loads
   per iteration — had not finished compiling after 20 s and ran at C1 speed the whole time
   (`-XX:LoopUnrollLimit=0` brought it back to seconds). Rewritten as one 16-byte load per iteration, the
   Q8 kernel compiles in 26 ms. Keep one segment load per loop iteration, and call each helper from one
   site only: a helper used at two sites in the same loop body did not stay inlined and cost 8×.
3. **Lane conversions share one profiled template.** `convertShape`, `castShape`, `convert` and the
   reinterpret views all go through `AbstractVector.convert0`, which reads the operand's element type and
   the target species through virtual calls the JDK source marks `// (profile)`. A process that mixes many
   conversion kinds (the float kernels' byte→float plus byte→short, short→int, int→float, …) pollutes those
   shared profiles. To stay clear of it the kernel uses only loads, lanewise arithmetic, same-size
   reinterprets and one reduction per row: widening is done with shifts (`(i << 16) >> 16` and `i >> 16` on
   the int view of the short lanes), int→float with the 2²³ magic constant (`bits(i + 1.5·2²³) − bits(1.5·2²³)`
   is exact for |i| < 2²²), and the activation quantizer rounds the same way (`rint`, ties to even).

Measured in isolation on the M1 Pro (one thread, 8 192 rows × 1 024 columns, best of 30–40): Q4 row dot
**0.59 ms** int8 vs **3.5 ms** float (6×); Q8 row dot **0.92 ms** int8 (the half-block form; 0.60 ms for the
slow-to-compile full-block form) vs 1.3–3.7 ms float.

### Where it is used

- **Decode** (`n == 1`): every Q4_0 / Q8_0 linear layer and the LM head (`CpuMatmul.useInt8`: Q2_K rows and
  any width that is not a multiple of 32 keep the float `dotRow`). Quantize `in` once, then the parallel row
  loop calls `Int8Kernels.dotRow` instead of `VectorKernels.dotRow`. The MLA, shared-expert and Mamba code
  paths reach the kernel only through this same `CpuMatmul.mul` (one class shared by `CpuBackend`, `MlaCpu` and
  `MambaCpuBackend`), so the format check covers them. `MambaCpuBackend` takes the switch as a constructor
  argument like `CpuBackend`; its four-argument constructor is the exact float path the CPU-vs-Metal checks use.
- **MoE** (per token, any `n`): `xb` is quantized once per token for the gate/up rows, and the 8 expert
  hidden vectors once for the down rows (`k * m` is a multiple of 32, so a slot offset is a block offset).
- **Prefill** stays on the float path: measured per (row, token), the int8 kernel is slightly *slower* than
  the dequantize-4-rows-once + `dot4` path (~88 ns vs ~77 ns per 1 024-long dot, single thread), because
  `dot4` already amortizes the expensive unpack over the 64 tokens and shares each activation load across
  four rows; a second code path for a loss is not worth it (see 7.11 for the kernel that could change that).

Measured numbers are in [11.1](11-performance.md#111-measured).

### GGUF K-quants (`KQuantKernels`)

GGUF files are mostly Q4_K / Q5_K / Q6_K (256-weight super-blocks with 6- or 8-bit sub-block scales). These
used to go through the float `VectorKernels` path, which unpacks bytes with `castShape` widening casts. On a
512-bit machine that kernel triggered the C2 pathology from point 2 above: compilation took minutes, and
decode ran at C1 speed (~0.1 tok/s) until it finished. `KQuantKernels` brings the int8-activation scheme to
the K-quants, built from the same few operations (loads, shifts, masks, short multiplies, the magic-constant
int→float):

- **Decode** (`dotRow`). A 16-byte load of nibbles, seen as 8 short lanes, gives four 8-lane weight vectors
  whose activations are contiguous in the permuted Q8 layout. The mins of Q4_K / Q5_K are folded out as
  `dmin · m_j · xs_j · Σ xq` using per-block activation sums (`blockSums`, computed once per matmul). Q5_K's
  fifth bit and Q6_K's two high bits are OR-ed into both bytes of each short lane at once (`& 0x0F0F`,
  `& 0x0303`). Each row is one method: a scalar prologue per super-block computes the 8 (Q6_K: 16) combined
  scales `d · sc_j · xs_j` into a small scratch array, and the vector loop only broadcasts a float. Two earlier
  versions were 2–3× slower: one computed the scales inside the vector loop, the other used one method call
  per super-block (each with its own reduction).
- **Prefill** (`mulPermuted`). Every weight row is dequantized into the same permuted block order, the
  activations are permuted once (`permute`), and the float `dot4x4` / `dot4` kernels run unchanged; a dot
  product doesn't depend on the order of its terms.

One thread, 2 048 × 1 024, Xeon AVX-512: Q4_0 0.17–0.3 ns per weight, Q4_K 0.29, Q6_K 0.49. Qwen3-0.6B
Q4_K_M decode on 4 threads went from 6.9 to 11 tok/s (prefill ~110 tok/s). `GgufKernelsCheck` compares all
three formats against the `gguf.quants` reference.

**CPU vs Metal agreement.** The Metal backend computes with float activations, so the bit-for-bit
CPU == Metal checks (`scripts/arch-smoke.sh`, `ArchSmoke`, `DeepSeekRef`) construct the CPU backend with
int8 activations off; with the default `true`, CPU and Metal greedy outputs can diverge after the first
near-tie exactly as described in [11.1](11-performance.md#111-measured). Pass
`--backend.cpu.int8-activations false` whenever an exact CPU/Metal comparison is wanted.

### The other GGUF formats (`GgufInt8Kernels`, native AArch64 kernels)

Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4, IQ4_XS and Q3_K (layouts in [06 §6.12](06-quantization.md#612-the-other-ggml-block-formats-read-in-place-from-gguf-files)) have
int8-activation kernels in the same scheme (`cpu/GgufInt8Kernels.java`; F16 / BF16 and Q2_K stay on the float kernels).
Per 32 weights the exact integer product is scaled by `weight scale · activation scale` as everywhere else:

- **Q4_1 / Q5_1**: `Σ (d q + m) x = d Σ q x + m Σ x`; `Σ x` over the block is the integer sum `blockSums` already computes for the
  K-quants. **Q5_0**: `(q5 - 16)`. The fifth bit of weight `j` is bit `j` of `qh`: expanded in the byte domain (a constant
  `rearrange` puts byte `j/8` of `qh` in lane `j`, a compare against `1 << (j % 8)` and a blend give 16 or 0), then the 32 bytes are
  multiplied like Q8_0 bytes.
- **IQ4_NL, MXFP4, IQ4_XS** (and the nibble tables in general): `nibble.selectFrom(table)` is one byte shuffle (`tbl`); the looked-up
  signed bytes then go through the Q8_0 byte path. IQ4_XS multiplies each of its eight 32-weight sub-blocks by `d (ls - 32)`.
- **Q3_K**: sixteen 6-bit scales, so each 16-weight half of a 32-weight unit has its own integer sum and scale; the weight
  is `((qs >> 2j) & 3) - 4 + 4 hmask-bit`.

**What the JIT forbids (measured, `T3`-style one-kernel timings).** Three things made these kernels 5 – 300 times slower than
Q4_0's, all the cliff of 7.10 ("the loop must stay inside C2's inlining budget, else every vector becomes a heap object"):
(1) **two scalar `MemorySegment` reads per loop iteration** (the f16 `d` and `m` of a Q4_1 block): 40 – 60 ms per 2048 x 1024
matrix instead of 0.15 ms; reading both with one 32-bit (or 64-bit, for `d, m, qh` or IQ4_XS's `d, scales_h, scales_l`)
load and splitting it in registers is as fast as one read; (2) **masked byte arithmetic** (`sub(x, mask)`), for which C2 has no
intrinsic on every CPU; (3) **`ByteVector.toShuffle()`**, which validates the indexes with an exception path: use
`indexes.selectFrom(table)` (836 µs → 201 µs). The Java Q5_0 / Q5_1 kernels are still ~6x slower than Q4_0's (bit expansion with a
constant shuffle plus another vector operation tips the loop over the budget again); on AArch64 the native kernels below
make that irrelevant.

**Native** (`libjllm_cpu`, [7.16](#716-native-int8-kernels-libjllm_cpu-nativecpu)): scalar, NEON and `sdot` kernels for all seven
formats, on the natural-order activation: Q5's fifth bit comes from `vqtbl1q_u8(qh)` + `vtstq_u8` against `{1, 2, 4, ..., 128}`
+ a mask with 16, the IQ4 / MXFP4 tables from `vqtbl1q_s8`. There are no x86-64 versions (they could not be tested here); on
x86-64 `NativeCpu.supports` is false for these formats (`jllm_cpu_has_type`) and the Java kernels above are used.

One thread, 2048 x 1024 matrix, Apple M1 Pro (load average 60 – 70 from other jobs, `NativeCpuBench`, µs, best of 40):

| format | Java int8 | native `neon` | native `neon-dotprod` | float (Vector API) |
|---|---|---|---|---|
| Q4_0G (reference) | 142 | 121 | 98 | 964 |
| Q4_1 | 164 | 185 | 151 | 632 |
| Q5_0 | 839 | 154 | 130 | 1 111 |
| Q5_1 | 944 | 212 | 198 | 654 |
| IQ4_NL | 163 | 119 | 101 | 1 386 |
| IQ4_XS | 200 | 109 | 72 | 1 113 |
| MXFP4 | 157 | 81 | 67 | 1 473 |
| Q3_K | 1 648 | 279 | 246 | 1 534 |

`NativeCpuCheck` compares every level against the Java kernels (bit for bit with unit scales, 3e-5 of `Σ|w x|` with random scales).

### 7.10.1 `auto`: measure, don't guess

🧸 Two roads lead to the same town. On some machines the short road (int8) is faster, on others the wide road
(float) is. Instead of guessing, the program drives both once when a model loads and takes the faster one from then on.

📐 `backend.cpu.int8-activations=auto` (the default) times one token's single-token matmuls both ways: every
quantized tensor of the first decoder layer (weighted by the number of layers) plus the LM head, 16 alternating
runs each, keeping the best of the last 12. The int8 path wins where the CPU has cheap byte dot products (NEON
`sdot`: ~3.9× faster decode on Apple silicon, the table above). On x86 with AVX-512 but no VNNI, the 16-bit
widening multiplies cost more than streaming the float-dequantized weights. On a 4-core AVX-512 machine, Qwen3-0.6B
Q4_0 measured 292 ms (int8) vs 183 ms (float) per token of matmuls, so float was chosen, and decode ran at
2.6 tok/s against 0.5 tok/s with int8 forced (a heavily loaded machine: compare the ratio, not the numbers). The
result is printed (`int8 activations: auto -> off (...)`) and cached per model, so the slots of a server share one
measurement. `true` / `false` force a path. When the native int8 kernels of
[7.16](#716-native-int8-kernels-libjllm_cpu-nativecpu) are loaded, the int8 side of the probe runs them (the log
line says `int8 ... ms [native avx512-vnni]` or `[java]`), which turns the decision around on that machine: see the
measurements in 7.16.

**A JIT pitfall found on the way.** The first version timed only `o_proj` (Q4_0) and chose float, which made decode
far *slower*: the Q8_0 LM head (151936 × 1024) took 9–10 s per token in float instead of ~0.1 s. Inside the full
engine, C2 compiled `dotQ8`'s byte-to-float `castShape` as a non-intrinsic call (`VectorSupport::convert … failed to
inline (intrinsic)`, then `made not entrant: uncommon trap` in a loop), so every lane went through the generic
Java fallback; the same code in a small benchmark compiled fine. Single-token Q8_0 rows on the float path now
dequantize into the thread-local scratch and run a float dot (`CpuMatmul.mulOne`), which compiles reliably and
takes 107 ms for that head, and the probe times the whole token so one slow tensor can no longer mislead it.

## 7.11 What to optimize next

In order of expected payoff on this CPU:

1. **Vectorized exp** — done for attention, the soft-caps and the MLP's SiLU (`-Djllm.cpu.fastmath`, on by
   default; [7.15](#715-fast-math-jllmcpufastmath)).
2. **f16 KV cache** — done: `backend.kv-dtype=f16` ([7.13](#713-the-f16-kv-cache-kvcache)). What remains is
   a fused f16 dot product that converts inside the SIMD loop instead of expanding each row to scratch first.
3. **Fuse RMSNorm into the following matmul** and SiLU into the gate/up matmul to cut passes over the
   activation and the number of pool jobs.
4. **Batch decode across requests** (continuous batching) so the weight read serves several tokens.
5. **Int8 prefill with a shared-activation kernel**: a `dot4`-style int8 kernel that unpacks 4 rows once and
   streams the quantized tokens through them; on NEON the instruction counts are a tie with the float
   `dot4`, so this only pays on wider SIMD (AVX2/AVX-512, 16 or 32 short lanes).

---

## 7.12 Summary

- The Vector API gives Java explicit SIMD: `FloatVector`/`ByteVector` lanes, `fma`, one `reduceLanes` per row.
- `dotQ4` loads 16 bytes, splits nibbles into two 16-lane vectors, converts to floats in parts, FMAs with `x`,
  applies the block scale once; weights are read exactly once per token.
- Decode parallelizes over output rows; prefill dequantizes four rows once and reuses them for all tokens
  (`dot4`), which is why it is more than an order of magnitude faster per token than float decode.
- Attention runs per query head for decode and per (kv head, token tile) for prefill, with thread-local scratch;
  MoE adds a router and two loops per token.
- A spin-then-park pool with caller participation makes ~225 micro-jobs per token cheap; the lost-wakeup race is
  closed by a volatile re-check.
- On 128-bit NEON the float kernel is instruction-bound (byte→float conversions), not memory-bound; integer dot
  products (int8 activations, the default) are the fix: ~4× the decode speed.
- The byte dot-product instructions (VNNI, `sdot`) are out of the Vector API's reach; a small C library called through
  the FFM API, one call per chunk of rows, uses them for the int8 decode matmuls
  ([7.16](#716-native-int8-kernels-libjllm_cpu-nativecpu)).

---

## 7.13 The f16 KV cache (`KvCache`)

`backend.kv-dtype=f16` (default `f32`) halves the biggest heap allocation, `q8` (below) nearly quarters it. The
storage lives in `cpu/KvCache.java` with three implementations; `CpuBackend` only knows `kv.store(...)` and
`kv.attention(...)`:

| | `KvCache.F32` (default) | `KvCache.F16` |
|---|---|---|
| storage per layer | `float[]`: K in transposed 32-key chunks, V rows | `int[]`: the same shapes, two halves per `int` |
| 0.6B at `ctx = 4096` | 940 MB | **470 MB** |
| `store` | K scattered into its chunk, V `System.arraycopy` | `Float.floatToFloat16` per value (round-to-nearest-even) |
| attention | chunks and rows read in place | blocks converted into scratch with vector code |

**Why staging buffers.** The k/v projections used to write straight into the cache and `normRope` ran in place
on the cached row. With an f16 cache that would round k *before* RoPE and again after it. Instead both backends
now compute k and v for the chunk in f32 (`kStage`/`vStage`, `[batch × kvDim]`), apply QK-norm and RoPE there,
and store once — so a value is rounded exactly once, and the Metal backend rounds the same f32 numbers the same
way, which is why the two backends' f16 outputs are identical. The f32 path pays one extra `arraycopy` of
`n × kvDim` floats per layer, which does not show in the measurements.

**The f16 attention loop.** The Vector API has no half-float lane type, and widening `short` lanes to `int`
lanes is the shape-changing conversion that C2 handles badly ([7.10](#710-int8-activations-integer-dot-products-int8kernels)).
So the f16 cache packs two halves into each `int`: K pairs head elements `2p` and `2p+1` of a key, V pairs keys
`2j` and `2j+1` of an element. One `IntVector` load then yields two float vectors through same-shape operations
only (`KvCache.F16.lowHalf` / `highHalf`): shift the half's exponent and mantissa to f32 bits 13–27, add the
exponent rebias `112 << 23` as an integer, OR the sign back in, and for subnormal halves (exponent 0) take
`asFloat(bits + (113 << 23)) − 2⁻¹⁴` instead, an exact float subtraction that never touches a denormal operand.
That is bit-exact with `Float.float16ToFloat` for all 63 488 finite halves (`GgufKernelsCheck` checks every one;
infinities and NaNs, which a KV cache never holds, come out as large finite values). The tiled kernel converts
each block of K chunks and V rows once per tile into thread-local scratch, then runs the f32 kernels on it.

**Precision.** f16 keeps 11 significant bits (about 3 decimal digits) and a range of ±65 504; K and V values
after QK-norm and RoPE are O(1), so nothing overflows and the rounding error is ~5 × 10⁻⁴ relative per value,
averaged out by the 128-term dot products. Greedy outputs on a 25-token prompt, the 413-token `bench` prompt
and a 623-token reading-comprehension prompt were identical to f32 on both backends.

**Measured** (M1 Pro, 10 threads, Qwen3-0.6B Q4_0, ctx 4096, machine shared with other jobs so absolute
numbers are below [7.9](#79-measured-numbers-and-the-bottleneck)): see [11 §11.3](11-performance.md#113-memory-footprint).

### q8: int8 values and a scale per row (`KvCache.Q8`)

`backend.kv-dtype=q8` goes one step further and stores every value as one signed byte. **The simple picture:** take
the `headDim` numbers that one kv head produced for one position (a *row*), find the largest magnitude, and map
`[-max, +max]` evenly onto the integers `-127 … 127`; keep the integers and the one float that undoes the mapping. A
value then comes back as `integer × scale`, a little off, but never by more than half a step.

**The math.** For a row `x[0 .. headDim)`:

```
scale = max_i |x_i| / 127                    (0 for an all-zero row)
q_i   = rint(x_i / scale)  ∈ [-127, 127]     (KvDtype.q8: round half to even, clamped)
x̂_i   = q_i · scale
```

Because `|x_i / scale| ≤ 127`, rounding to the nearest integer is the only error, so

```
|x_i - x̂_i| ≤ scale / 2 = max_i |x_i| / 254          (0.39 % of the row's largest value)
```

The error is relative to the *row's* maximum, not to the value itself as with f16 (2⁻¹¹ ≈ 0.05 % of each value):
small values in a row with one large outlier lose the most. Symmetric quantization needs no zero point, so a dot
product with a row factors as `q · x̂ = scale · (q · x_int)` and a softmax weight times a value row as
`(p · scale) · v_int`. Memory per token, `L` layers, both K and V, one 4-byte scale per `headDim` values:

```
bytes per token = 2 · L · kvDim · (1 + 4 / headDim)
Qwen3-0.6B (L = 28, kvDim = 1024, headDim = 128):  2 · 28 · 1024 · 1.03125 = 59 136 B ≈ 59 KB
                                                   f16: 2 · 28 · 1024 · 2 = 114 688 B ≈ 115 KB;  f32: 229 KB
at --ctx 4096:                                     242 MB (q8)  vs  470 MB (f16)  vs  940 MB (f32)
```

**Layout.** K keeps the transposed 32-key chunks of `F32`, as bytes (`byte[layers][kvHeads · chunks · hd · 32]`), with
the scales of a chunk's 32 keys next to each other (`float[layers][kvHeads · chunks · 32]`); V rows are bytes
`[ctx][kvDim]` with scales `[ctx][kvHeads]`. `store` quantizes each (position, kv head) row of the f32 staging rows
(k after QK-norm and RoPE, as for f16), so a value is rounded exactly once.

**The attention loop.** The tiled kernel is unchanged; like `F16`, `Q8` expands a block of K chunks (then V rows) into
f32 scratch once per tile. The widening is vector code: a 16-byte `ByteVector` load, `castShape` to the float species
(`B2F`, lanes 0 … FL−1), multiplied by the scales. In a K chunk each lane is a different key, so the multiplier is a
*vector* of their 32 scales (loaded once per FL keys, reused for all `headDim` elements); in a V row it is one
broadcast scale. The expanded block holds exactly `q_i · scale` (the scalar fallback and save/load compute the same
float product), so prefill and decode stay bit-identical and `SmokeTest.kvCacheAttention` checks q8 against a
double-precision reference on the dequantized rows, as for f32 and f16.

**Saved caches** (`kv q8 layers=… kvDim=… headDim=…`): per position the K row as `kvDim` int8 values followed by
`kvHeads` f32 scales, then the V row the same way — the stored integers and scales themselves, so a cache saved and
loaded is bit-identical, and the layout is the Metal backend's row layout ([08 §8.9](08-metal-backend.md#89-the-f16-kv-cache)),
so a file saved by one backend loads into the other (`ArchSmoke.kvStateChecks`).

**DeepSeek** (`MlaCpu`): the latent row `[c | k_pe]` (576 values) gets one scale for the whole row: 0.6 KB instead of
1.2 KB (f16) per layer and token. `c` (RMS-normalized) and `k_pe` can differ in magnitude, and one shared scale costs
the smaller part precision: on the tiny random DeepSeek of `ArchSmoke` separate scales for the two parts halved the
logit drift. As the latent cache is already small, f16 is the safer choice for DeepSeek.

**Measured** on Qwen3-0.6B Q4_0, CPU backend (4-core VM shared with other jobs, int8 activations, 4 threads; the
speeds are indicative):

| | f32 | f16 | q8 |
|---|---|---|---|
| cache per token (28 layers) | 229 KB | 115 KB | **59 KB** |
| 300-token greedy answer (chat prompt, ctx 1024) vs f16 | 82 / 300 tokens equal, first difference at token 76 | — | 8 / 300 equal, first difference at token 1 (a near-tie: "A steam engine" vs "A **steam engine**"; the text stays fluent and on topic) |
| the f16 answer teacher-forced, max \|logit − logit_f16\| over the 300 steps (max \|logit\| 35.6) | 0.82 (mean of the per-step maxima 0.40), argmax equal 292 / 300 | — | 4.41 (mean 1.43), argmax equal 282 / 300 |
| prefill of a 2 000-token prompt, tok/s (two runs) | 17 / 38 | 25 / 38 | 32 / 40 |
| decode after it (64 tokens), tok/s (two runs) | 2.7 / 5.1 | 2.8 / 4.5 | 3.9 / 3.8 |

The speeds were taken with a load average of 15–20 on the 4 cores (other jobs), so they only say that q8 is not
slower to a degree that shows through that noise (the first runs include the JIT warming up). The accuracy numbers
are deterministic. q8 is clearly
less exact than f16: Qwen3's K rows have outlier channels (after QK-norm a row's largest value is on average 7.3×
its RMS, at worst 11×; V rows 3.6×), so the step `max / 127` is coarse for the typical entries — the RMS rounding
error is 1.6 % of a K row's RMS and 0.8 % of a V row's, against ~0.02 % for f16. Greedy text diverges early (as it
does for any change at a near-tie; f32 vs f16 diverges at token 76) but remains coherent. Use q8 when the cache
size is what limits the context or the number of slots, f16 otherwise.

---

## 7.14 Prefill and decode are not bit-identical; results depend on history

The two matmul paths of [7.4](#74-decode-vs-prefill-matmul-cpubackendmatmul) compute the same dot products in
different orders:

| | `n == 1` (decode) | `n > 1` (prefill) |
|---|---|---|
| weights | streamed as quantized blocks, `(lo·x + hi·x) · scale` accumulated per block | four rows dequantized to float once, then `dot4` / `dot4x4` FMAs over `cols` |
| int8 activations (default on) | integer products, `Σ_b (s_b · d_b) · Σ w q` | never (float path) |
| MoE | per token: `dotRow` per expert row | experts grouped over tokens, `dot4x4` |

Float addition is not associative, so the keys, values and logits of a token differ in the last bits depending
on whether its `forward` call carried one token or many (typically a relative 1e-6 to 1e-5 per element, more
with int8 activations). Attention, RMSNorm, RoPE and SiLU are the same code for both paths. The Metal backend has
the same property for its own reasons (different threadgroup shapes).

**Why this makes outputs depend on history.** The engine reuses the KV cache for the longest common prefix of
consecutive requests ([10](10-engine-and-server.md)). Cached positions were written either by a prefill (`n > 1`) or by
decode steps (`n == 1`) of an earlier request, so a new request that extends them attends to slightly different
K/V than it would if its prompt were prefilled from position 0 in one go. The same prompt can therefore yield
different logits in a second chat turn than in a fresh process, and greedy decoding can diverge after the first
near-tie. Exact reproducibility holds only for the same *sequence of `forward` calls* (`Backend.forward` states
this contract); the `run` command, `bench` (which evicts the prefix before each round) and `ArchSmoke` always
start from position 0, which is why their outputs are reproducible.

**`-Djllm.cpu.deterministic=true`** (system property, read once by `CpuBackend`; the name reports
`deterministic`) routes every `n > 1` matmul and MoE layer through the single-token kernels, one token at a
time, with the same int8 setting as decode. Then a token's K/V, hidden state and logits are bit-identical
however the prompt is chunked and whatever was in the cache before, which makes two runs comparable
element by element (for example `--batch 1` against `--batch 64`, or a chat turn against a fresh run). Prefill
loses the four-row reuse of [7.4](#74-decode-vs-prefill-matmul-cpubackendmatmul), so it is a comparison
tool, not a serving mode. It does not make CPU equal to Metal: that comparison needs int8 activations off
([7.10](#710-int8-activations-integer-dot-products-int8kernels)), and even then the two backends differ in the
last bits of the prefill path.

**One more source of last-bit noise: the JIT tier.** `reduceLanes(ADD)` on a `FloatVector` does not specify the
order in which lanes are combined; the Java fallback the interpreter and C1 run adds them sequentially, the C2
code on NEON adds them pairwise. The first few hundred calls of every kernel in a process therefore sum in a
different order than the rest, and two runs of the same command can differ in the 7th significant digit of a
logit depending on exactly when C2 took over (observed on Qwen3-0.6B: three identical commands, two distinct
results). Within one tier the results are exact: with `-Xint` (tiny models only; the Vector API fallback is
about 1 000× slower) the same prompt gives bit-identical logits run after run, and under
`jllm.cpu.deterministic` also for `--batch 1` versus `--batch 16`. For greedy text this noise only matters at a
near-tie; the `run` gates in [11](11-performance.md) have been stable across hundreds of runs.

---

## 7.15 Fast math (`jllm.cpu.fastmath`)

Fast math is **on by default** (the backend name reports `fastmath`); `-Djllm.cpu.fastmath=false` turns it off.
It replaces the scalar `Math.exp` of the attention softmax and the `Math.tanh` of the attention and final-logit
soft-caps with polynomial kernels on float lanes (`VectorKernels.expFast` / `tanhFast`, used through `softmaxExp`
and `softcapFast`), and the SwiGLU's `silu(g) · u = g / (1 + e^(−g)) · u` of the gated MLPs and the MoE experts
(`VectorKernels.siluMul`). Mamba's SiLU keeps `Math.exp`. The results are *not* bit-identical to the
scalar functions, so bit-exact comparisons with the Metal backend (`scripts/arch-smoke.sh`) pass
`-Djllm.cpu.fastmath=false`, but they are close:

- `expFast`: `x = n·ln 2 + r` with `|r| ≤ ln 2 / 2`, a degree-6 Taylor polynomial in `r` (Horner with FMAs) and
  `2^n` assembled from the exponent bits; `n` comes from the `1.5 · 2^23` rounding trick of
  [7.10](#710-int8-activations-integer-dot-products-int8kernels), so the kernel has no lane conversions. Input
  clamped to `[-87, 88]`: the softmax's `exp(s - max)` then bottoms out at `1.6e-38` instead of 0, which is
  invisible after the `1/sum` scaling. Relative error 2.2e-7 over the range (`SmokeTest.fastMath`).
- `tanhFast(x) = 1 - 2 / (expFast(2x) + 1)`, absolute error below 2e-6 (`SmokeTest.fastMath` checks it), saturating correctly at both ends.
- The softmax sum is accumulated lane-wise and reduced once (the default path sums in order).
- The lane kernels and their scalar tails use the same arithmetic, so a row's result does not depend on where
  the vector loop stops.

It became the default when the attention kernels stopped being reduction-bound: on a 6 000-key prefill chunk the
scalar exps were ~40 % of the attention time (44 → 26 ms per layer with the switch alone). In isolation one exp
costs 0.8 ns instead of 6.3 ns (8×). In a decode step the softmax is `heads × len` exps
per layer against `heads × len × headDim × 2` FMAs for the scores and values, so the gain is bounded: measured
on the M1 Pro (10 threads, Qwen3-0.6B Q4_0, int8 on, machine shared with other jobs), decode after a
3 023-token prompt went from 21.6 / 20.8 tok/s to 22.7 / 22.7 tok/s (two interleaved runs each, +5–9 %), the
`bench` decode at ctx 4096 from 57 to 60–62 tok/s, and prefill did not change beyond the noise (the long
prefill is dominated by the attention dot products, [7.5](#75-attention-kvcacheattention)). Greedy text was
identical with the switch on and off in these runs. Soft-capped models (Gemma) additionally save the per-score
`Math.tanh`, which costs about as much as the exp.

---

## 7.16 Native int8 kernels (`libjllm_cpu`, `NativeCpu`)

Code: `src/main/native/jllm_cpu.c` (+ `jllm_cpu_x86.inc`, `jllm_cpu_arm.inc`), `cpu/NativeCpu.java`, `cpu/CpuMatmul.java`;
build: `scripts/build-native-cpu.sh`; settings: `backend.cpu.native` (`auto` default, `true`, `false`) and
`backend.cpu.native-lib`.

🧸 The int8 path of [7.10](#710-int8-activations-integer-dot-products-int8kernels) multiplies small whole numbers
(bytes) and adds up the products. Modern CPUs have one instruction for exactly that: "multiply four pairs of bytes
and add the four products to a running total", 8 or 16 of them at once. Java's Vector API has no word for that
instruction, so the Java kernels take the long way: they stretch every byte to 16 bits, multiply, add neighbours, and
stretch again. The fix is a small C library that uses the instruction. Java hands it a whole stack of rows at a time,
like giving a fast clerk a pile of forms instead of walking over with each sheet.

### 📐 What the instructions do

| level | instruction | per instruction |
|---|---|---|
| `avx512-vnni` (x86 with AVX-512 VNNI, used on 256-bit registers) | `vpdpbusd ymm` | 8 lanes × (4 unsigned bytes · 4 signed bytes) added to an int32 |
| `avx-vnni` (x86, Alder Lake and later) | `vpdpbusd` (VEX) | the same |
| `avx2` | `vpmaddubsw` + `vpmaddwd` | the same in two instructions (pairs to int16, then pairs to int32) |
| `neon-dotprod` (ARMv8.2 `FEAT_DotProd`: Apple M1 and later, Graviton 2 and later) | `sdot` | 4 lanes × (4 signed · 4 signed bytes) added to an int32 |
| `neon` | `smull` + `sadalp` | 8 products to int16, pairwise added into int32 |
| `scalar` | C loops | fallback for other CPUs |

All levels of one architecture are in one binary. Every SIMD function carries a `__attribute__((target(...)))`, and
the file itself is compiled for the baseline (`-O2`, no `-march`). `jllm_cpu_level()` picks the best level from
`cpuid` / `xgetbv` (x86: AVX2 + FMA + F16C with OS-enabled YMM state; AVX512-VNNI also needs AVX512 F/BW/VL and ZMM
state) or from `getauxval(AT_HWCAP) & HWCAP_ASIMDDP` / `sysctl hw.optional.arm.FEAT_DotProd`. The backend name says
which one runs: `int8 activations (native avx512-vnni)`.

### 📐 The arithmetic, per format

The activation is the same int8 vector as in 7.10 (`d_b = max|x|/127`, `q = rint(x/d_b)`, ties to even), stored in
natural order with its 32-block scales `d_b` and block sums `s_b = Σ q_i`. `jllm_cpu_quantize` produces it, and
`-ffp-contract=off` keeps its float operations the ones the Java quantizer does, so both give the same bytes (this is
checked). Per 32 weights (a block or a K-quant sub-block) the kernels compute an exact int32, then do exactly what the
Java kernels do with it: multiply by the combined scale, rounded to float the same way, and add into float lanes.

- **Q4_0 / Q4_0G.** Byte `j` of a block holds weight `j` (low nibble) and weight `j+16` (high nibble). So
  `[low nibbles | high nibbles]` in one 256-bit register are the 32 weights in natural order, unsigned `n ∈ [0,15]`.
  `vpdpbusd` wants unsigned × signed, so instead of subtracting 8 from every weight it uses
  `Σ (n_i − 8) q_i = Σ n_i q_i − 8 s_b`. The correction `−8 s_b` is the start value of int32 lane 0 of the
  accumulator. The lane total is exact. The 8 lanes are converted to float and FMA-ed with `scale_b · d_b`: an f32
  scale for the engine's Q4_0, an f16 one (converted with `vcvtph2ps`) for ggml's.
- **Q8_0 / Q8_0G.** These are signed × signed, handled as `|w| · (q with w's sign)` (`vpsignb` twice), the standard
  trick. It is exact for `w = −128` too, since `|−128|` is 128 as an unsigned byte. With AVX2 the pair sums go through
  int16: `2 · 128 · 127 = 32512 < 32767`, so nothing saturates. The activations never reach −128.
- **Q4_K / Q5_K.** As in `KQuantKernels`: `dot = Σ_j (d·sc_j·d_j) · Σ_i n_i q_i − Σ_j (dmin·m_j·d_j) · s_j`. The low
  nibbles of a 32-byte group are sub-block `2g` and its high nibbles are `2g+1`, each multiplied (unsigned × signed)
  with its own contiguous activation block. Q5_K adds bit `2g` / `2g+1` of `qh[l]` as bit 4. The 16-bit shifts move
  bits across the two bytes of a lane, but only into bits the mask removes. The scalar prologue per super-block
  (`k4_prologue`: decode the 6-bit scales and mins, the 8 scales `d·sc_j·d_j`, the min term) is the same sequence of
  float operations as Java's.
- **Q6_K.** The codes `(nibble | 2 high bits << 4) − 32 ∈ [−32, 31]` are signed, so the sign trick is used again.
  Each 16-weight half of a 32-weight group has its own scale: int32 lanes 0–3 and 4–7 of the product are multiplied
  by `[s_k ×4 | s_{k+1} ×4]`.

Two float accumulators per row hide the FMA latency, and there is one horizontal sum per row. The only difference
from the Java kernels is the order of the float additions (8 lanes and two accumulators instead of 4 lanes). So the
results agree to float rounding. They agree **bit for bit** whenever all partial sums are integers below 2²⁴ (unit
scales), which the tests use to check the integer parts.

### 📐 Calling it: one downcall per chunk of rows

```java
nat.quantize(in, inOff, cols, act);                                     // calling thread, once per matmul
pool.forRange(rows, 16, (r0, r1) -> nat.dotRows(wt, r0, r1, act, 0, out, outOff + r0, acc));
```

```c
void jllm_cpu_dot_rows(int type, const uint8_t *w, int64_t scales_off, int r0, int r1, int cols,
                       const int8_t *xq, const float *xd, const int32_t *xs, float *out, int accumulate);
```

- **Granularity.** `forRange` makes about 4 tasks per thread, at least 16 rows each, and each task is one downcall.
  For Qwen3-0.6B's `o_proj` (1024 × 2048) on 4 threads that is 16 calls of 64 rows each, a few µs of C work per call.
  A critical downcall costs tens of nanoseconds.
- **Zero copy.** `w` is the tensor's own `MemorySegment`: the memory-mapped model file or the converted cache.
  `scales_off` locates the f32 scale region of the engine's layout. Split tensors (fused gate | up, per-expert
  interleave) are called part by part, with each run of consecutive rows of one part as one call.
- **Output.** The caller's `float[]` is passed as a heap segment. The downcall handles use
  `Linker.Option.critical(true)`, which allows heap segments and skips the thread-state transition. The library never
  calls back into Java. It returns in micro- to milliseconds (the longest call is one thread's share of the
  151 936-row LM head), so blocking the GC for that long is acceptable.
- **Activation.** Each `CpuMatmul` has one `NativeCpu.Activation`: a native allocation from `Arena.ofAuto()` (so any
  thread may read it), grown when needed. The calling thread writes it before the row loop is published, the same
  happens-before as for the Java path's arrays.
- **MoE** (`MoeExperts.token`): the gate/up rows are one call per run of rows of one slot inside a chunk. For the down
  projection, each chunk of output rows makes one call per slot into `down[k·dim + r]` (activation offset `k·m/32`
  blocks), then adds `Σ_k w_k · down` in slot order, the same order as the Java loop.

Prefill (`n > 1`) is unchanged: the float `dot4x4` path, with each row dequantized once for all tokens.

### 📐 Finding, checking and choosing the library

`NativeCpu.configure(backend.cpu.native, backend.cpu.native-lib)` runs at each backend creation
(`ModelLoader.newBackend`, before the int8 `auto` probe). The library is looked up the same way as the Metal library:
the setting, `-Djllm.cpu.nativeLib`, `$JLLM_CPU_NATIVE_LIB`, `target/native/`, the working directory, and `native/`
next to the jar or classes directory. It must report interface version 1 and pass a **self-check** before it is used.
The self-check compares against `Int8Kernels` / `KQuantKernels`: the quantizer bit for bit on 320 values that include
an all-zero block, and for every type 3 random rows with unit scales (exact equality) and with random scales (within
`3e-5 · Σ|w_i x_i|`).

- `auto`: a missing library or a failed check prints one line (`native CPU kernels: ... using the Java kernels`) and
  the Java kernels run.
- `true`: the backend refuses to start without the library.
- `false`: the library is never loaded.

Without a `configure` call (tests, or code that builds backends itself), `-Djllm.cpu.native` applies (default `auto`).

The int8 `auto` probe ([7.10.1](#7101-auto-measure-dont-guess)) builds its int8 `CpuMatmul` the normal way, so it
times the native kernels when they are active, and its log line says which kernels it timed (`int8 ... us [native
avx512-vnni]` or `[java]`).

### 📐 Tests

- **`NativeCpuCheck`** runs in `mvn test`, after `build-native-cpu.sh --optional` (without a C compiler it passes with
  a note). It runs at **every level the CPU can run** (here scalar, avx2 and avx512-vnni) and covers:
  - the quantizer, at 5 lengths;
  - every type with unit scales at 1/3/7 rows × 32/96/160/256 columns, bit for bit: once with full-range activations,
    once with small activations and random sub-block scales;
  - random scales at 1/5/13 rows × up to 3 072 columns (the 32-block formats also at 96, 1 056 and 3 072 columns,
    which are not multiples of 256);
  - activation offsets, accumulation, and split tensors across their part boundaries;
  - `CpuMatmul` on 3 threads, with and without the library.

  Output: `native cpu kernels (..., levels scalar, avx2, avx512-vnni; detected avx512-vnni): 678 checks passed, 0
  failed`. With two deliberate bugs planted (one wrong Q6_K scale index, one extra rounding in the K-quant min term),
  77 checks failed.
- **`src/test/native/jllm_cpu_levels.c`** checks every SIMD level against the scalar one in plain C, for
  architectures where a JVM would be too slow under emulation. Cross-compiled with `aarch64-linux-gnu-gcc` and run
  under `qemu-aarch64`, `neon` and `neon-dotprod` pass its 2 688 checks; with `-cpu cortex-a53` the library detects
  plain `neon`. The `avx-vnni` level is compiled but has not run, because the test machine has AVX512-VNNI without
  AVX-VNNI. Its code is the same as the AVX512-VNNI level's, with the VEX encoding of the same instruction.
- **`NativeRealCheck <model-dir>`** is run by hand. It checks the real tensors of a model's first layer and its LM
  head, native and Java int8, against a double-precision reference.
- **`NativeGreedyCheck <model-dir>`** is run by hand. It loads the model three times (native int8, Java int8, and float
  as the yardstick), teacher-forces the same tokens through all three one step at a time (the single-token steps are
  where the kernels differ) and compares the logits. Then it runs free greedy generation, native vs Java. The results
  are under "Outputs" below.

### 📐 Measured

On the 4-core Xeon with AVX512-VNNI (no AVX-VNNI), Linux, GCC 13. Three other agents ran builds and tests on the same
machine the whole time (load average 28–55 on 4 cores, sometimes too little free memory), so every number is noisy.
The ranges are over repeated runs; compare the ratios, not the absolute values.

**Row kernels** (`NativeCpuBench`, 2048 × 1024, one matrix times one activation, µs, best of 60 after ≥ 3 s of
warm-up, one type per JVM):

| type | Java int8, 1 thread | native avx2, 1 thread | native avx512-vnni, 1 thread | native scalar | float, 1 thread | `CpuMatmul`, 4 threads: native / Java int8 / float |
|---|---|---|---|---|---|---|
| Q4_0  | 255–279 | 109–115 | 92–99   | 381–915  | 242–299  | 88–102 / 143–896 / 116–247 |
| Q8_0  | 426–477 | 95–97   | 85–93   | 414–1197 | 36 414–73 548 (not C2-compiled, 7.10.1) | 97–108 / 769–913 / 28 058–44 923 |
| Q4_0G | 298–324 | 127     | 108–109 | 404–607  | 262–291  | 118 / 473 / 266 |
| Q8_0G | 619–666 | 107–109 | 105–108 | 479–1186 | 175–214  | 63 / 281 / 94 |
| Q4_K  | 660–745 | 190–202 | 181–188 | 786–2073 | 998–1009 | 112–209 / 331–463 / 2 379–3 507 |
| Q5_K  | 683–693 | 218–219 | 208     | 1321–1330 | 3 301–3 670 | 226 / 335 / 3 256 |
| Q6_K  | 1115    | 154–155 | 137–141 | 1697–2838 | 1597–1675 | 97–158 / 435–683 / 864–1 547 |

Single-threaded, the native kernels are about 2.8× faster than the Java int8 ones for Q4_0, 5× for Q8_0, 3.3–4×
for Q4_K / Q5_K and 8× for Q6_K. `vpdpbusd` gains only 2–16 % over the two-instruction AVX2 form at this size, so
the loop is close to load-bound. The float columns call `VectorKernels.dotRow` directly; in the engine,
`CpuMatmul.mulOne` now runs float Q8_0 rows as dequantize + dot instead (7.10.1). The scalar level is the C compiler's plain loops, a fallback only.

**Whole-token probe** (the `auto` probe's own line, Qwen3-0.6B Q4_0, the matmuls of one token on 4 threads):

```
int8 activations: auto -> on  (one token's matmuls: int8 78.6 ms [native avx512-vnni], float 154.1 ms)
int8 activations: auto -> on  (one token's matmuls: int8 75.6 ms [native avx512-vnni], float 129.9 ms)
int8 activations: auto -> off (one token's matmuls: int8 336.9 ms [java], float 188.3 ms)
int8 activations: auto -> off (one token's matmuls: int8 268.3 ms [java], float 175.9 ms)
```

With the library, `auto` picks int8 (1.7–2× less matmul time than float). Without it, `auto` picks float (the Java
int8 kernels are 1.5–1.8× slower than float here).

**Decode** (`./jllm bench`, Qwen3-0.6B Q4_0, 413-token prompt, ctx 1024, tok/s, two runs × two rounds each per
setting, interleaved):

| setting | 4 threads | 1 thread |
|---|---|---|
| int8, native avx512-vnni (`auto` with the library) | 4.2–5.2 (earlier, at load 30–45: 3.1–7.0) | 3.8–4.0 |
| `auto` without the library (= float) | 2.8–3.2 | — |
| float (`int8-activations=false`) | 2.4–3.3 | 3.2 |
| int8, Java kernels forced | 0.5 at load 40–49 (earlier, at load 30: 3.3–4.7) | 2.5–2.7 |

Native int8 decodes about 1.5× faster than the float path that `auto` chose before, at 4 threads. A decode step also
contains attention, norms and the thread pool's synchronization, and under this load those parts suffer the most from
preemption. On an idle machine the gap should be larger, but that was not measured. Prefill does not change (11–17
tok/s here, with every setting).

The Java int8 decode of 0.5 tok/s is the JIT trouble of [7.10](#710-int8-activations-integer-dot-products-int8kernels)
and 7.10.1 again: in the full engine, under load, C2 did not deliver compiled Java int8 kernels for minutes, while
the same kernels in a small benchmark compile fine. The native kernels cannot have this problem: there is nothing to
JIT-compile, and they run at full speed from the first token.

**Outputs.** Each matmul on its own agrees with a double-precision reference (the same quantized weights times the
same int8 activation) to float rounding. `NativeRealCheck` measures this on Qwen3-0.6B's first layer and LM head:
the largest `|dot − ref| / Σ|w_i x_i|` is 2–7e-8 native and 2–3e-8 Java. End to end it is different
(`NativeGreedyCheck`, 4 prompts × 48 teacher-forced decode steps, all three backends fed the same tokens):

| prompt | max \|Δlogit\| / logit range: native vs Java int8 (after 1 step) | Java int8 vs float | arg-max differs (native–Java / Java–float) | free greedy, native vs Java |
|---|---|---|---|---|
| "The capital of France is" | 0.025 (0.008) | 0.017–0.026 | 0 / 0 | identical (7 tokens) |
| "Write a haiku about autumn leaves." | 0.026 (0.010) | 0.015 | 1 / 0 | same up to token 14 |
| "def fibonacci(n):" | 0.015 (0.011) | 0.011 | 1 / 0 | same up to token 12 |
| "Explain why the sky is blue in two sentences." | 0.019 (0.008) | 0.015 | 1 / 2 | same up to token 38 |

So the outputs are near-identical, not identical, and native and Java int8 are about as far apart as Java int8 is
from float. The reason is that every layer quantizes its input to int8 again. A last-bit difference in one matmul
(the order of the float additions) occasionally moves a value across a rounding boundary of `rint`, which changes
that int8 value by a whole step (up to 1/254 of the block's largest value, and Qwen3's residual stream has large
outliers). Twenty-eight layers and the KV cache then carry the difference forward. The Java int8 path is not even
reproducible against itself: the Java-int8-vs-float column for the first prompt moved from 0.017 to 0.026 between
two runs of the same check. The likely cause is the Vector API: a kernel adds its lanes in another order before C2
compiles it than after.
Each arg-max disagreement happened at a near-tie, where the Java run's top two logits were within 0.04–0.8 % of the
range. The free greedy runs part exactly at those steps (12, 14, 38).
`NativeGreedyCheck` passes when native-vs-Java stays within twice Java-vs-float and every disagreement is a near-tie
(< 1 % of the range).
