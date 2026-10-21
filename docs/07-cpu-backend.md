# 7. The CPU backend: SIMD with the Vector API — in detail

Code: `cpu/VectorKernels.java`, `cpu/ThreadPool.java`, `cpu/CpuMatmul.java`, `cpu/CpuBackend.java`, `cpu/KvCache.java`,
`cpu/Int8Kernels.java`, `cpu/MlaCpu.java`, `cpu/MambaCpuBackend.java`

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
size ([7.13](#713-the-f16-kv-cache-kvcache)).

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

The attention loop lives with the cache (`KvCache.F32` / `KvCache.F16`, [7.13](#713-the-f16-kv-cache-kvcache));
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

### 7.10.1 `auto`: measure, don't guess

🧸 Two roads lead to the same town. On some machines the short road (int8) is faster, on others the wide road
(float) is. Instead of guessing, the program drives both once when a model loads and takes the faster one from then on.

📐 `backend.cpu.int8-activations=auto` (the default) times one representative single-token matmul of the loaded
model, the first layer's `o_proj` (else the LM head), 24 times each way, alternating, and compares the best of the
last 20. The int8 path wins where the CPU has cheap byte dot products (NEON `sdot`: ~3.9× faster decode on Apple
silicon, the table above). On x86 with AVX-512 but no VNNI, the 16-bit widening multiplies cost more than streaming
the float-dequantized weights; on a 4-core AVX-512 machine the probe measured int8 1619 µs vs float 402 µs for
Q4_0 1024×2048 and chose float. The result is printed (`int8 activations: auto -> off (...)`) and cached per model,
so the slots of a server share one measurement. `true` / `false` force a path.

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

---

## 7.13 The f16 KV cache (`KvCache`)

`backend.kv-dtype=f16` (default `f32`) halves the biggest heap allocation. The storage lives in `cpu/KvCache.java`
with two implementations; `CpuBackend` only knows `kv.store(...)` and `kv.attention(...)`:

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
