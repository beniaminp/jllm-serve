# 8. The Metal backend: GPU compute through Panama FFM — in detail

Code: `metal/MetalBackend.java` (records the forward pass), `metal/Metal.java` (FFM bindings),
`src/main/native/jllm_metal.m` (Objective-C shim with a C ABI), `src/main/resources/org/robomiri/jllm/metal/kernels.metal`
(the GPU programs), `scripts/build-native.sh` (one `clang` command).

This chapter assumes you have never written a line of GPU code and never called C from Java. It builds the
vocabulary first, then walks the actual code layer by layer, then every kernel with a worked example, and ends
with an honest look at where the time goes.

---

## 8.0 The toolbox: how a GPU thinks

### CPU vs GPU

A CPU core is a sprinter: a few very fast, very clever cores (10 on an M1 Pro) that each run one instruction
stream and are good at branchy, unpredictable code. A GPU is a marching band: thousands of simple cores (the M1
Pro has 16 GPU "cores", each containing 128 lanes = 2048 lanes) that all want to execute **the same instruction
on different data** at the same time. A GPU is slow at anything else, but a transformer is almost entirely
"the same dot product, a million times", which is exactly the shape it wants.

Two consequences shape everything below:

1. You never write "a loop over rows". You write *one function that handles one row* (a **kernel**) and ask the
   GPU to run it once per row. Loops become grids.
2. The GPU is on the other side of a fence. The CPU prepares work, hands it over in batches, and waits.
   Hand-overs are expensive (microseconds each) — so we hand over an entire forward pass at once.

### Threads, simdgroups, threadgroups, grid

```
grid  ───────────────────────────────────────────────────────────────────────────────
│ threadgroup (0,0)                 │ threadgroup (1,0)                 │ ...
│ ┌────────────────────────────┐    │ ┌────────────────────────────┐    │
│ │ simdgroup 0: 32 threads    │    │ │ simdgroup 0: 32 threads    │    │
│ │ simdgroup 1: 32 threads    │    │ │ simdgroup 1: 32 threads    │    │
│ │ simdgroup 2: 32 threads    │    │ │ simdgroup 2: 32 threads    │    │
│ │ simdgroup 3: 32 threads    │    │ │ simdgroup 3: 32 threads    │    │
│ │   threadgroup memory (fast)│    │ │   threadgroup memory       │    │
│ └────────────────────────────┘    │ └────────────────────────────┘    │
└───────────────────────────────────┴───────────────────────────────────┴────────────
                               device memory (the big, shared RAM)
```

| unit | size | what it is |
|---|---|---|
| **thread** | 1 | one execution of the kernel function; knows its own index |
| **simdgroup** | 32 threads | run in lock-step on one instruction; can exchange values *in registers* with `simd_sum`, `simd_max` — no memory involved |
| **threadgroup** | up to 1024 threads (we use 64, 128 or 256) | shares a small on-chip memory (`threadgroup float buf[32]`), can synchronize with `threadgroup_barrier` |
| **grid** | any number of threadgroups, 1-, 2- or 3-D | one kernel launch ("dispatch") |

Threads find their place through attributes on the kernel's parameters, e.g.
`uint2 tgp [[threadgroup_position_in_grid]]`, `ushort sg [[simdgroup_index_in_threadgroup]]`,
`ushort lane [[thread_index_in_simdgroup]]` (0–31). The kernel reads these and decides what piece of data it owns.

### Memory

- **device memory**: pointers marked `device const float *x`. The weights, the KV cache, activations. Large,
  shared by everything, comparatively slow (hundreds of cycles).
- **threadgroup memory**: `threadgroup float part[8][256]`, declared inside the kernel. Tiny (we use < 10 KB),
  very fast, visible only to the threads of one threadgroup. Used to combine partial results.
- **registers**: ordinary local variables (`float acc[8]`). Fastest; private to one thread.
- **constant memory**: `constant MatmulParams &p` — small read-only parameter structs.

### Barriers and reductions

Threads in a threadgroup run at their own pace. `threadgroup_barrier(mem_flags::mem_threadgroup)` makes
every thread wait until all have arrived *and* all their writes to threadgroup memory are visible — the
"everyone finished step 1 before anyone starts step 2" primitive. `mem_flags::mem_device` does the same for
writes to device memory.

`simd_sum(v)` adds the 32 lanes' values in a few hardware steps and returns the total to **every** lane.
Example with 4 lanes for brevity: lanes hold `[2, 5, 1, 4]`; after `simd_sum` every lane holds `12`. This is how
a dot product of length 1024 finishes: each lane sums its share, one `simd_sum` combines them.

A **two-level reduction** (`tg_sum` in `kernels.metal`) extends this across a whole threadgroup of `nsg`
simdgroups: each simdgroup `simd_sum`s, lane 0 writes the result to `buf[sg]`, barrier, then one simdgroup
reads `buf[0..nsg)` and `simd_sum`s again:

```
256 threads = 8 simdgroups, each lane with a partial sum
  simd_sum per simdgroup  → 8 numbers   buf = [s0 s1 s2 s3 s4 s5 s6 s7]
  barrier
  lane < 8 reads buf[lane], others 0, simd_sum → total, in every lane
```

### Apple unified memory and "zero copy"

On a PC, the GPU has its own RAM; using a weight means *copying* it over PCIe into GPU memory. Apple Silicon
has **one** pool of RAM shared by CPU and GPU. A Metal buffer in `MTLResourceStorageModeShared` is just a range
of that RAM with a GPU-visible handle; the CPU can read and write the same bytes through an ordinary pointer.
"Zero copy" means exactly that: our 450 MB (or 19 GB) of quantized weights sit in memory once, memory-mapped
from `jllm-q4_0.bin`, and the GPU reads them in place. The only requirement is that the range starts on a
16 KiB page boundary and has a page-multiple length — which is why `QTensor` and `QuantFile` align everything to
`PAGE = 16384`.

### Where MLX fits

MLX is Apple's machine-learning framework (arrays, autograd, a compiler) written in C++ on top of **this same
Metal API**. We skip the framework and use Metal directly: the project only needs ten kernels, and the whole
native layer is ~200 lines. The GPU work — writing kernels, managing buffers and command buffers — is the layer
MLX would otherwise do for us.

---

## 8.1 Four layers, one responsibility each

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ MetalBackend.java   knows the MODEL: which buffer is Wq of layer 3, grid sizes,      │  Java
│                     the order of operations in a forward pass, parameter structs     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ Metal.java          knows NOTHING about models: compile shader source, make/wrap     │  Java (FFM)
│                     buffers, begin / dispatch / commitAndWait; owns the handles      │
├───────────────────────────── C function calls (jm_*) ───────────────────────────────┤
│ jllm_metal.m        Objective-C: talks to the Metal framework; 13 C functions        │  native
├─────────────────────────────────────────────────────────────────────────────────────┤
│ kernels.metal       Metal Shading Language (C++ dialect): the code the GPU runs;     │  GPU
│                     compiled at start-up from source                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

Why not put the model logic in Objective-C? Because then the native file would grow with every feature (MoE,
new models) and need recompiling; keeping it generic means the dylib is built once and all intelligence stays
in Java and in the shader text, which is a resource file.

---

## 8.2 The native shim, function by function (`jllm_metal.m`)

Just enough Objective-C to read the file:

| thing | meaning |
|---|---|
| `id<MTLDevice>` | a pointer to an object implementing the `MTLDevice` protocol (the GPU). `id<...>` ≈ a Java interface-typed reference |
| `[obj method:arg]` | a method call: `[dev newCommandQueue]` |
| `@property`, `@interface`/`@implementation` | a class with fields (`JMContext` holds the device, queue, compiled library, pipelines, current command buffer) |
| `@autoreleasepool { … }` | Objective-C objects are reference counted; temporaries created inside the block are freed at its end |
| `(__bridge_retained void *) obj` | hand an object to C as a raw pointer **and** hold a reference so it stays alive; `(__bridge_transfer T) ptr` takes it back and releases it; `(__bridge T) ptr` just peeks |
| `MTLCommandQueue` | the GPU's work queue |
| `MTLCommandBuffer` | one batch of work, submitted atomically |
| `MTLComputeCommandEncoder` | the object you record dispatches into |
| `MTLComputePipelineState` | a compiled kernel ready to run (like a `MethodHandle` for GPU code) |

The 13 exported functions:

| C function | does | called from |
|---|---|---|
| `jm_create(source, err, len)` | `MTLCreateSystemDefaultDevice()`, a command queue, and `newLibraryWithSource:` — compiles the entire `kernels.metal` text at run time. Returns an opaque context pointer, or NULL with the compiler's message in `err` | `Metal.<init>` |
| `jm_destroy(ctx)` | releases everything | `Metal.close` |
| `jm_device_name(ctx, out, len)` | e.g. "Apple M1 Pro" | `MetalBackend.name` |
| `jm_recommended_working_set(ctx)` | how many bytes the GPU can keep resident without paging | start-up warning |
| `jm_pipeline(ctx, name, err, len)` | looks up kernel `name` in the compiled library, builds a `MTLComputePipelineState`, stores it in an array and returns its **index** (so Java never holds Objective-C objects) | once per kernel |
| `jm_pipeline_max_threads(ctx, idx)` | the pipeline's `maxTotalThreadsPerThreadgroup` | MoE sanity check (router needs `numExperts` threads) |
| `jm_buffer_new(ctx, len)` | `newBufferWithLength:options:Shared`, zeroed | KV cache, activations, norm weights |
| `jm_buffer_wrap(ctx, ptr, len)` | `newBufferWithBytesNoCopy:` — wraps memory **owned by Java** (the mmap'd cache file). Falls back to a copying `newBufferWithBytes:` if Metal refuses (misaligned) | every quantized weight tensor |
| `jm_buffer_contents(buf)` | the CPU pointer to a buffer's bytes | Java reads logits / writes embeddings |
| `jm_buffer_release(buf)` | drop the reference | `Metal.close` |
| `jm_begin(ctx)` | `commandBufferWithUnretainedReferences` + a compute encoder with `MTLDispatchTypeSerial` — dispatches execute **in order**, each seeing the previous one's writes, so Java needs no explicit barriers between kernels | start of `forward` |
| `jm_dispatch(ctx, pipe, bufs[], offsets[], nbuf, params, plen, gx,gy,gz, tx,ty,tz)` | `setComputePipelineState`, `setBuffer:offset:atIndex:` for `i < nbuf`, `setBytes:` for the params at index `nbuf`, then `dispatchThreadgroups:threadsPerThreadgroup:` | once per kernel launch |
| `jm_commit_wait(ctx, err, len)` | `endEncoding`, `commit`, `waitUntilCompleted`; returns −1 with the error text if the command buffer's status is `MTLCommandBufferStatusError` (a GPU fault) | end of `forward` |

"Unretained references" means the command buffer does not reference-count the buffers it uses — slightly
cheaper, and safe because `Metal.java` keeps every buffer alive in its `buffers` list until `close()`.

Building it: `scripts/build-native.sh` runs
`clang -O2 -fobjc-arc -dynamiclib -framework Metal -framework Foundation src/main/native/jllm_metal.m -o target/native/libjllm_metal.dylib`.
`-fobjc-arc` is automatic reference counting (hence the `__bridge` casts); the two frameworks are the Metal and
Foundation libraries. The `./jllm` launcher runs this on first use.

---

## 8.3 Calling C from Java: the FFM API (`Metal.java`)

Before JDK 22 calling C meant JNI: write a C glue function per call, generate headers, compile. The **Foreign
Function & Memory API** (`java.lang.foreign`, project "Panama") does it from pure Java:

| class | role | in `Metal.java` |
|---|---|---|
| `SymbolLookup` | finds a function's address in a loaded library | `SymbolLookup.libraryLookup(path, arena)` loads the dylib |
| `FunctionDescriptor` | describes a C signature: return layout + argument layouts | `FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT)` = `int f(void*, char*, int)` |
| `Linker` | turns address + descriptor into a callable | `Linker.nativeLinker().downcallHandle(addr, fd)` |
| `MethodHandle` | the callable; `invokeExact` must be called with *exactly* the descriptor's Java types | `(int) commitWait.invokeExact(ctx, err, ERR_LEN)` |
| `MemorySegment` | a pointer with bounds: `ADDRESS` arguments and results | `ctx`, buffer handles, the `err` text buffer |
| `Arena` | a lifetime: everything allocated from it is freed when the arena closes | `Arena.ofShared()` for the library and scratch |
| `arena.allocateFrom(String)` | a NUL-terminated C string in native memory | the shader source, kernel names |
| `seg.reinterpret(size)` | gives a raw pointer returned by C a size so Java may read it | `jm_buffer_contents` result |
| `seg.getString(0)` | reads a C string | error messages, device name |

**Worked example: the `jm_dispatch` binding.** The C prototype is

```c
void jm_dispatch(void *ctx, int pipeline, void **bufs, const uint64_t *offsets, int nbuf,
                 const void *params, int plen, int gx, int gy, int gz, int tx, int ty, int tz);
```

Java describes it once in the constructor:

```java
dispatch = LINKER.downcallHandle(lookup.findOrThrow("jm_dispatch"),
    FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, ADDRESS, ADDRESS, JAVA_INT,
                              ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));
```

and calls it per kernel launch:

```java
dispatch.invokeExact(ctx, pipe, bufs, offsets, nbuf, params, plen, gx, gy, gz, tx, ty, tz);
```

`bufs` is a native array of `MAX_BUFS = 8` pointers (`arena.allocate(ValueLayout.ADDRESS, 8)`), `offsets` an
array of 8 longs, `params` a 64-byte scratch block. `MetalBackend.bind(i, buffer, offset)` writes slot `i` of the
first two with `setAtIndex`; the parameter struct is written field by field (next section). Because the native
side copies the params (`setBytes`) and records the buffer bindings at dispatch time, the same three scratch
segments are reused for every dispatch.

**Getting a Java array into GPU memory.** `MetalBackend.upload(float[] data)`:

```java
Metal.Buffer b = metal.newBuffer((long) data.length * 4);                        // shared buffer
MemorySegment.copy(data, 0, b.contents(), ValueLayout.JAVA_FLOAT, 0, data.length); // one bulk copy
```

`b.contents()` is the `jm_buffer_contents` pointer reinterpreted to the buffer's size. The same path in reverse
reads the logits back (`MemorySegment.copy(logitsBuf.contents(), JAVA_FLOAT, 0, logits, 0, rows)`), and the
embedding rows are written straight into the `x` buffer before each forward. Quantized weights never go through
a copy at all: `metal.wrap(tensor.data)` passes the mmap'd segment's address and size to `jm_buffer_wrap`.

Every `invokeExact` is wrapped by `rethrow`, which converts the checked `Throwable` into a runtime exception;
a NULL result is turned into a descriptive exception using the `err` buffer. The JVM needs
`--enable-native-access=ALL-UNNAMED` to allow these "restricted" calls without a warning.

**Finding the library.** `Metal.locateLibrary` tries, in order: `backend.metal.lib` from the config,
`-Djllm.metal.lib`, `$JLLM_METAL_LIB`, `target/native/libjllm_metal.dylib`, `./libjllm_metal.dylib`, and a
`native/` directory next to the jar. `Metal.available()` (macOS + library found) decides whether `backend=auto`
picks Metal.

---

## 8.4 Recording a forward pass (`MetalBackend.forward`)

```
1. CPU: for each of the n tokens, dequantize its embedding row (Q8 → float) into buffer x      (≈ 4 KB each)
2. metal.begin()                                            ← one command buffer for everything below
3. for each layer: record 12 (dense) or 13 (MoE) dispatches
4. if logits wanted: final rmsnorm of the last token, LM-head matmul
5. metal.commitAndWait()                                    ← the ONLY CPU↔GPU synchronization
6. CPU: copy the logits out of logitsBuf
```

### Dispatch list per dense layer (Qwen3-0.6B: dim 1024, qDim 2048, kvDim 1024, ffn 3072, 16/8 heads)

Grid sizes are *threadgroups*; `×` = threads per threadgroup. `n` = tokens in this call.

| # | kernel | writes | grid, decode `n=1` | grid, prefill `n=64` | threads |
|---|---|---|---|---|---|
| 1 | `rmsnorm` | `xb` | 1 | 64 | 256 |
| 2 | `matmul_q4` (Wq 2048×1024) | `q` | (512, 1) | (512, 8) | 128 |
| 3 | `matmul_q4` (Wk 1024×1024) | `keyCache[l]` at offset `pos·kvDim·4` | (256, 1) | (256, 8) | 128 |
| 4 | `matmul_q4` (Wv) | `valCache[l]` at offset `pos·kvDim·4` | (256, 1) | (256, 8) | 128 |
| 5 | `norm_rope` (16 heads, q_norm) | `q` in place | (16, 1) | (16, 64) | 64 |
| 6 | `norm_rope` (8 heads, k_norm) | cache rows in place | (8, 1) | (8, 64) | 64 |
| 7 | `attention` | `att` | (16, 1) | (16, 64) | 256 |
| 8 | `matmul_q4` (Wo 1024×2048, accumulate) | `x +=` | (256, 1) | (256, 8) | 128 |
| 9 | `rmsnorm` | `xb` | 1 | 64 | 256 |
| 10 | `matmul_q4` (Wgate∥Wup 6144×1024) | `gateUp` | (1536, 1) | (1536, 8) | 128 |
| 11 | `silu_mul` | `hidden` | 12 | 768 | 256 |
| 12 | `matmul_q4` (Wdown 1024×3072, accumulate) | `x +=` | (256, 1) | (256, 8) | 128 |

28 layers × 12 = 336, plus the final `rmsnorm` (1 group) and the LM head `matmul_q8` (151 936 rows → grid
(37 984, 1)): **338 dispatches per decode step**.

Notice steps 3, 4 and 6: the K and V projections write *directly into the KV cache* at the byte offset of
position `pos` (`bind(3, keyCache[l], pos·kvDim·4)`), and `norm_rope` for K runs in place on those cache rows.
There is no separate "store into cache" step. (With `backend.kv-dtype=f16` steps 3, 4 and 6 target two f32
staging buffers instead and a thirteenth dispatch, `kv_store_half`, rounds the rows into the `half` caches —
[8.9](#89-the-f16-kv-cache).)

### Dispatch list per MoE layer (Qwen3-30B-A3B: dim 2048, qDim 4096, kvDim 512, 128 experts, top-8, moeDim 768)

| # | kernel | grid, decode `n=1` | grid, prefill `n=64` | threads |
|---|---|---|---|---|
| 1–9 | as above with the 30B shapes: Wq (1024, ·), Wk/Wv (128, ·), rope (32,·)/(4,·), attention (32,·), Wo (512, ·) | | | |
| 10 | `router` | 1 | 64 | 128 (= experts) |
| 11 | `moe_up_q4` (1536 rows per expert) | (384, 8) | (384, 512) | 128 |
| 12 | `silu_mul` over `n·8·768` | 24 | 1536 | 256 |
| 13 | `moe_down_q4` (2048 rows per expert) | (512, 1) | (512, 64) | 128 |

48 layers × 13 + 2 = **626 dispatches per decode step**.

### Parameter structs: why only 4-byte members

Each kernel takes a small `struct` by `constant` reference. Java has no struct type, so `MetalBackend` writes
the fields one after another into the 64-byte `params` segment and tells the shim how many bytes to copy:

```java
params.setAtIndex(ValueLayout.JAVA_INT,   0, t.rows);      // byte 0
params.setAtIndex(ValueLayout.JAVA_INT,   1, t.cols);      // byte 4
params.setAtIndex(ValueLayout.JAVA_INT,   2, n);           // byte 8
params.setAtIndex(ValueLayout.JAVA_INT,   3, acc ? 1 : 0); // byte 12
metal.dispatch(pipe, bufArgs, offArgs, 4, params, 16, ...); // plen = 16
```

This only works if the C++ struct has the same layout — which is guaranteed when every member is a 4-byte `int`
or `float` (no padding is ever inserted). `setAtIndex(JAVA_FLOAT, 1, eps)` writes a float at byte 4, matching
`struct NormParams { int n; float eps; }`. The structs and their sizes:

| struct | fields | bytes |
|---|---|---|
| `MatmulParams` | rows, cols, ntok, accumulate | 16 |
| `NormParams` | n, eps | 8 |
| `RopeParams` | headDim, rowStride, pos0, eps, useNorm | 20 |
| `AttnParams` | nHeads, nKvHeads, headDim, pos0, ctx, scale, qStride, kvStride | 32 |
| `SiluParams` | n, total | 8 |
| `RouterParams` | nExperts, dim, topK, normalize | 16 |
| `MoeParams` | rowsPerExpert, cols, topK | 12 |

Buffers are bound at indices `0 … nbuf−1` and the params at index `nbuf`, matching the `[[buffer(i)]]`
attributes in the kernel signature. Getting one index wrong is the classic bug: the kernel reads the wrong
buffer and silently produces garbage, which is why the CPU backend exists as a reference.

### Buffers allocated at start-up

| buffer | size (0.6B, ctx 4096, batch 64) | size (30B, ctx 2048) |
|---|---|---|
| weights (wrapped, zero copy) | 452 MB | ≈ 19.7 GB |
| `keyCache` + `valCache`, per layer `ctx·kvDim·4` (`·2` with `backend.kv-dtype=f16`) | 28 × 2 × 16 MB = 939 MB (470 MB f16) | 48 × 2 × 4 MB = 403 MB (201 MB f16) |
| `x`, `xb` (`batch·dim·4`) | 256 KB each | 512 KB |
| `q`, `att` (`batch·qDim·4`) | 512 KB each | 1 MB |
| `gateUp`, `hidden` | 1.5 MB, 768 KB | — (dense MLP unused) |
| `scores` (`batch·heads·ctx·4`) | 16.8 MB | 16.8 MB |
| `logitsBuf` | 608 KB | 608 KB |
| MoE: `assign`, `routeWeight`, `moeGateUp`, `moeHidden` | — | 2 KB, 2 KB, 3.1 MB, 1.6 MB |
| norm gains, RoPE tables (uploaded copies) | ~2 MB | ~3 MB |

`recommendedMaxWorkingSetSize` is compared with the weight size at start-up; exceeding it does not fail but
means the GPU will page weights in and out.

---

## 8.5 The kernels, one by one

All kernels follow the same recipe: *figure out which output element(s) this thread owns from its indices,
compute, write*. We go through them in the order they appear in `kernels.metal`.

### `tg_sum`, `tg_max` — threadgroup reductions

Shown in 8.0. Used by `rmsnorm` (sum of squares), `norm_rope` (per-head sum of squares), `attention` (max and
sum of the softmax). `buf` has 32 entries because a threadgroup has at most 1024 / 32 = 32 simdgroups.

### `block_dot<QT>` — the inner loop of every matmul

The same 32-weight block kernel as the CPU's `dotQ4`, written for one GPU thread:

```c++
template <int QT>                               // QT = 4 (Q4_0) or 8 (Q8_0), decided at compile time
float block_dot(wq, ws, grow, nb, b, x4):
    Q4: rq = 16 bytes of block b of global row grow
        for k in 0..3:  v = 4 bytes
            d += dot(float4(v & 15) - 8, x4[k]) + dot(float4(v >> 4) - 8, x4[k + 4])
    Q8: rq = 32 bytes
        for k in 0..7:  d += dot(float4(4 signed bytes), x4[k])
    return d * ws[grow * nb + b]                // apply the block's scale once
```

`uchar4`, `float4` are 4-wide vector types; `dot()` on `float4` is one instruction. The nibble layout matches
`QTensor` (byte `j` → weights `j` and `j+16`): the low nibbles of bytes 0–15 are weights 0–15, the high nibbles
weights 16–31, which is why `x4[k]` pairs with the low nibbles and `x4[k+4]` with the high ones. `grow` is a
`ulong` because stacked expert tensors have hundreds of thousands of rows and `rows × nb × 16` overflows 32 bits.

### `matmul_q4` / `matmul_q8` — `out[t, row] (+)= W[row] · x[t]`

Work assignment:

```
threadgroup = 128 threads = 4 simdgroups  →  4 output rows per threadgroup
grid.x = ceil(rows / 4)        which rows
grid.y = ceil(ntok / 8)        which group of up to TOK_TILE = 8 tokens
simdgroup sg handles row = tgp.x * 4 + sg
lane l handles blocks  l, l+32, l+64, ...  of that row
```

**Worked example**: Wq of 0.6B, `cols = 1024` → `nb = 32` blocks per row, `n = 1`. Lane 7 of the simdgroup
for row 100 loads block 7 (16 bytes at `100 × 32 × 16 + 7 × 16`), unpacks it into `lo[4]`, `hi[4]`, multiplies
with `x[224 … 256)` and scales; since `nb = 32` every lane handles exactly one block. Then `simd_sum(acc[0])`
adds the 32 lanes' partial sums and lane 0 writes `out[100]`. With `cols = 3072` (Wdown) each lane does 3
blocks.

**TOK_TILE**: during prefill the same unpacked block is multiplied with up to 8 token vectors
(`acc[t] += d * s` for `t < nt`), so the weight bytes are read once per 8 tokens instead of once per token. The
accumulator array lives in registers. `accumulate = 1` makes the write `out[o] += v` (residual connections).

### `rmsnorm` — one threadgroup per row

```
256 threads stride over the row: thread tid sums squares of elements tid, tid+256, tid+512, ...
tg_sum → total sum of squares, in every thread
scale = rsqrt(ss / n + eps)                     rsqrt = 1/sqrt, one instruction
thread tid writes out[i] = x[i] * scale * w[i] for its elements
```

For `n = 1024` each thread handles 4 elements. `MetalBackend.rmsnormAt` binds the input at a byte offset so
the final norm can read only the last token's row (`(n−1)·dim·4`) and write row 0 of `xb`.

### `norm_rope` — one thread per rotary pair

Grid `(heads, ntok)`; `headDim / 2 = 64` threads per threadgroup, so **thread `i` owns the pair `(i, i+64)`**
of one head of one token — exactly the "rotate-half" pairing from [05 §5.5](05-transformer-math.md#55-step-3--qk-norm-and-rotary-position-embedding-rope):

```
a = h[i];  b = h[i + 64]
if useNorm:  ss = tg_sum(a² + b²)              (64 threads cover all 128 numbers of the head exactly once)
             scale = rsqrt(ss / headDim + eps);  a *= scale * w[i];  b *= scale * w[i + 64]
pos = pos0 + token index
h[i]      = a·cos[pos][i] − b·sin[pos][i]
h[i + 64] = b·cos[pos][i] + a·sin[pos][i]
```

The `cos`/`sin` tables are the `Rope` arrays uploaded once (`[ctx × 64]` each). `rowStride` lets the same
kernel walk `q` (stride `qDim`) or the KV cache (stride `kvDim`, starting at position `pos`).

### `attention` — one threadgroup (256 threads = 8 simdgroups) per (head, token)

Three phases, following the formula `softmax(q·K / √hd) · V`:

```
per = headDim / 32 = 4     lane l owns head dimensions l, l+32, l+64, l+96  (held in registers qr[0..3])

phase 1  scores:    simdgroup sg takes cached positions sg, sg+8, sg+16, ...
                    for each: d = Σ_lane qr[i]·k[lane + 32i]  → simd_sum → × 1/√hd
                    lane 0 writes sc[s] (device scratch); every lane tracks the running max
                    tg_max → mx
phase 2  softmax:   all 256 threads stride over s: sc[s] = exp(sc[s] − mx); tg_sum → sum
phase 3  values:    simdgroup sg again takes positions sg, sg+8, ...: o[i] += sc[s] · v[lane + 32i]
                    each simdgroup writes its 128 partial outputs to part[sg][·]  (threadgroup memory)
                    barrier; thread d sums part[0..7][d] and writes out[d] / sum
```

**Worked example** with `pos0 = 412`, token `t = 0` of a decode step: `len = 413` cached positions; each of
the 8 simdgroups handles ~52 of them in phase 1 and 3; each phase-1 iteration is 4 multiplies per lane plus one
`simd_sum`. The `scores` buffer is indexed `[(t·heads + h)·ctx + s]` so every (token, head) has its own row of
`ctx` floats — putting scores in device memory rather than threadgroup memory is what allows `ctx = 4096`
(16 KB per row would not fit the 32 KB threadgroup limit for several rows). Causality is `len = pos0 + t + 1`.
GQA: `kvh = h / (nHeads / nKvHeads)` selects the shared KV head.

The two `threadgroup_barrier(mem_flags::mem_device)` calls make the scratch writes of phase 1 visible to
phase 2, and phase 2's to phase 3.

### `silu_mul` — one thread per element

`gid` indexes the flattened `[ntok, n]` output; `t = gid / n`, `i = gid % n`; reads `gate` at `[t, i]` and
`up` at `[t, n + i]` of the `[ntok, 2n]` input; writes `g / (1 + exp(−g)) · u`. Used for the dense FFN
(`n = 3072`) and for MoE slots (`n = 768`, `ntok = n·8`).

### `router` — one threadgroup per token, `nExperts` threads

```
thread e:   logits[e] = Wrouter[e] · x[t]          (scalar loop over dim; the router is 1 MB, not worth more)
barrier
thread 0:   mx = max; logits[e] = exp(logits[e] − mx); sum
            8 times: pick the largest remaining, assign[t·8+k] = e, rw[t·8+k] = p_e; set it to −∞
            if normalize: rw /= Σ selected p_e
```

Thread 0 doing 8 passes over 128 entries is ~1 000 operations — trivial next to a 1 MB read, and it keeps the
selection exactly identical to `CpuBackend.route` (softmax first, then top-k, then renormalize, like
transformers). `logits` lives in threadgroup memory (`[1024]`, hence the 1024-expert limit checked in the
constructor against `jm_pipeline_max_threads`).

### `moe_up<QT>` — one simdgroup per (slot, expert row)

A **slot** is `s = t·topK + k`: the `k`-th chosen expert of token `t`. Grid `(ceil(1536/4), n·8)`:

```
row  = tgp.x·4 + sg                      which of the expert's 1536 gate/up rows
s    = tgp.y;  t = s / 8;  e = assign[s]
grow = e·1536 + row                      the row inside the stacked [128·1536, 2048] tensor
acc  = Σ_blocks block_dot<QT>(…, grow, …, x[t])   over nb = 64 blocks, 2 per lane
simd_sum;  lane 0 writes moeGateUp[s·1536 + row]
```

Selecting an expert is purely arithmetic on the row index — the stacked layout from
[12 §12.3](12-mixture-of-experts.md#123-weight-layout-experts-stacked-into-two-matrices) means no per-expert
buffer and no indirection table on the GPU side beyond `assign`.

### `moe_down<QT>` — one simdgroup per (token, output row), loop over the 8 slots inside

```
row = tgp.x·4 + sg   (0 … 2047);   t = tgp.y
total = 0
for k in 0..7:   s = t·8 + k;  grow = assign[s]·2048 + row
                 acc = Σ_blocks block_dot<QT>(…, grow, …, moeHidden[s])      nb = 24 (768 / 32)
                 total += rw[s] · simd_sum(acc)
lane 0:  x[t·2048 + row] += total
```

Looping over the slots *inside* the kernel means each `x[t, row]` has exactly one writer, so the accumulate
into the residual stream needs no atomics. With `nb = 24` only 24 of the 32 lanes have a block — acceptable
for this first version.

### Template instantiation

`moe_up` and `moe_down` are C++ templates on `QT`; Metal needs concrete, named kernels, which is what this
syntax at the bottom of the file produces:

```c++
typedef decltype(moe_up<4>) moe_up_t;
template [[host_name("moe_up_q4")]] kernel moe_up_t moe_up<4>;
template [[host_name("moe_up_q8")]] kernel moe_up_t moe_up<8>;
```

`host_name` is the string `jm_pipeline` looks up. (`matmul_q4`/`matmul_q8` predate `block_dot` and are
written out twice; folding them into the template is a pending clean-up.)

---

## 8.6 Gotchas met along the way

- **Position attributes must agree in dimensionality.** A kernel cannot mix `uint2 tgp
  [[threadgroup_position_in_grid]]` with a scalar `ushort tid [[thread_position_in_threadgroup]]` — the
  compiler rejects it. Hence `ushort2 tid2` and `tid2.x` in `norm_rope` and `attention`. `simdgroup_index…`
  and `thread_index_in_simdgroup` are not position attributes and may stay scalar.
- **Struct layout**: only `int`/`float` members, in the order Java writes them; `plen` must match the struct size.
- **`newBufferWithBytesNoCopy` needs page alignment**; otherwise it returns nil and the shim silently copies —
  fine for small buffers, catastrophic for a 19 GB model (double memory). All `QTensor`/`QuantFile` blobs are
  16 KiB-aligned so the fast path is always taken. Read-only `mmap`'d pages work (this is what llama.cpp does).
- **Unretained command buffers**: every buffer used in a dispatch must outlive the command buffer. `Metal.java`
  keeps all buffers in a list until `close()`; `ModelWeights` (the mmap) must be closed *after* the backend.
- **Shader errors arrive at run time**, as text in `jm_create`'s `err` buffer, with line numbers relative to
  the source string — `kernels.metal` is compiled when the backend starts, not when the Java code is built.
- **GPU faults** (out-of-bounds reads, usually from a wrong `bind` index or grid size) show up as
  `MTLCommandBufferStatusError` from `jm_commit_wait`, often with an unhelpful message. The reliable debugging
  method is the CPU backend: run the same prompt with `--backend cpu --temp 0` and compare outputs token by
  token; they must be identical.
- **`exp` precision**: Metal's `exp` is not guaranteed bit-identical to Java's `Math.exp`. In practice greedy
  outputs match the CPU exactly, but a near-tie in the router or sampler could in principle flip.

---

## 8.7 Measured numbers and where the time goes

Apple M1 Pro, Qwen3-0.6B Q4_0, 413-token prompt: **~344 tok/s prefill, ~75 tok/s decode** (~110 tok/s on short
contexts); Q8_0 decodes at essentially the same speed (~77 tok/s). Greedy output is byte-identical to the CPU.

The Q4 ≈ Q8 result is the tell: if decode were limited by weight bandwidth, Q8 (1.8× the bytes) would be
clearly slower. So the ~13 ms per token go elsewhere:

| cost | evidence |
|---|---|
| **338 dispatches per token** | each costs a few µs of GPU front-end time plus the serial dependency between consecutive kernels; alone this is a sizeable fraction of 13 ms |
| **under-filled grids** | `Wk`, `Wv`, `Wo`, `Wdown` have 1024 rows → 256 threadgroups of 128 threads = 32 768 threads on a GPU that wants hundreds of thousands in flight; the M1 Pro's 16 cores are mostly idle during these kernels |
| **the LM head** | `151 936 × 1024` Q8 = 165 MB, a third of all bytes read per token, in one 37 984-threadgroup dispatch |
| **attention reads the whole prefix** | per layer, per head: `len × 128 × 4 B` of keys and the same of values; at 4 096 tokens that is 4 MB per head per layer |
| **a CPU round trip per token** | `commitAndWait` + logits copy + sampling + embedding write: the GPU idles while the CPU samples |

### The optimizations, by expected payoff

1. **Kernel fusion** — fold `rmsnorm` into the following matmul's prologue, compute q/k/v in one dispatch,
   apply SiLU in the gate/up matmul's epilogue, and apply RoPE inside the q/k projection. Roughly halves the
   dispatch count.
2. **Expert-grouped MoE prefill** — today `moe_up` handles one token per simdgroup, so a 64-token chunk reads
   every chosen expert's weights once *per token*. Sorting slots by expert and running a `TOK_TILE`-style loop
   over the tokens that share an expert gives the reuse the dense matmul already has.
3. **Flash-style attention** — compute scores, the running max/sum and the value accumulation in one pass with
   online softmax, never writing `scores` to device memory. (The `half` KV cache that halves the K/V traffic
   exists already: `backend.kv-dtype=f16`, [8.9](#89-the-f16-kv-cache).)
4. **Wider threadgroups / more rows per threadgroup for small matrices** so `Wk`-sized matmuls fill the GPU,
   and splitting the LM head over the token dimension in prefill.
5. **Simdgroup matrix instructions** (`simdgroup_multiply_accumulate` on 8×8 tiles) for prefill matmuls, which
   are compute-bound and currently use scalar `dot()` on `float4`.
6. **Overlap** — pipeline the CPU's sampling with the next step's GPU work, or sample on the GPU.

---

## 8.8 Summary

- The GPU runs *one function per output element*; we express every transformer operation as such a function
  and launch grids of threadgroups (`kernels.metal`).
- Dot products end with `simd_sum`; threadgroup-wide sums use a two-level reduction through threadgroup memory
  and barriers.
- Unified memory lets the memory-mapped quantized cache be used by the GPU in place (`jm_buffer_wrap`).
- A ~200-line Objective-C shim exposes 13 C functions; `Metal.java` binds them with the FFM API (`Linker`,
  `FunctionDescriptor`, `MethodHandle.invokeExact`, `MemorySegment`), no JNI, no generated code.
- `MetalBackend.forward` records the whole forward pass (338 dispatches for 0.6B, 626 for 30B-A3B) into one
  serial command buffer and waits once.
- Decode on the M1 Pro is currently limited by dispatch count and small grids rather than by bandwidth, which
  is exactly what the listed optimizations attack.

---

## 8.9 The f16 KV cache

`backend.kv-dtype=f16` allocates `keyCache`/`valCache` as `half` buffers (`ctx·kvDim·2` bytes per layer, 470 MB
instead of 939 MB for the 0.6B model at `ctx = 4096`) and changes three things in the dense layer; everything
else, including every f32 kernel, is untouched. The kernels live in the "f16 KV cache" section at the end of
`kernels.metal`.

| step | f32 (default) | f16 |
|---|---|---|
| 3, 4 `matmul` Wk, Wv | into `keyCache[l]`/`valCache[l]` at offset `pos·kvDim·4` | into f32 staging buffers `kStage`/`vStage` (`batch·kvDim·4` bytes) at offset 0 |
| 6 `norm_rope` (k) | in place on the cache rows | in place on `kStage` |
| new `kv_store_half` | — | one thread per element: `kc[pos·kvDim + i] = half(kStage[i])`, same for V; grid `ceil(n·kvDim / 256)` |
| 7 attention | `attention` | `attention_h` — the same kernel templated on the cache element type (`attention_kv<half>`) |

So a decode step records 13 dispatches per dense layer instead of 12 (366 instead of 338 for the 0.6B model);
`kv_store_half` moves 8 KB per layer and costs a few microseconds. The matmul and `norm_rope` kernels did not
need `half` variants because the staging buffers are f32: K is rounded exactly once, after RoPE, with Metal's
default round-to-nearest-even `half()` conversion — the same rounding as Java's `Float.floatToFloat16` on the
CPU path, which is why the two backends' f16 outputs are identical.

`attention_kv<KV>` differs from `attention` only in the pointer types of `kc`/`vc` and in the loads, which widen
each `half` to `float` before the multiply (`qr[i] * float(kv[lane + i * 32])`). Scores, softmax and the
value accumulation stay f32, so the f16 variant changes what is *read*, not what is computed: per cached
position and lane it loads 4 × 2 bytes of K and 4 × 2 bytes of V instead of 4 × 4 each. On the M1 Pro this is
worth a few percent of decode at a few hundred tokens of context (the attention kernel is not the dominant
cost there — [8.7](#87-measured-numbers-and-where-the-time-goes)); the point of the option is the memory and
the scaling at long contexts. Measurements are in [11 §11.3](11-performance.md#113-memory-footprint).
