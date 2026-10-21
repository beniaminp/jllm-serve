# 8. The Metal backend: GPU compute through Panama FFM — in detail

Code: `metal/MetalBackend.java` (records the forward pass; `MlaMetal.java` and `MambaMetalBackend.java` for the
other two families), `metal/MetalKernels.java` (one writer per kernel), `metal/KernelParams.java` / `Params.java` /
`MetalConstants.java` (parameter structs and constants), `metal/Metal.java` (FFM bindings),
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
- **threadgroup memory**: `threadgroup float part[8][256]`, declared inside the kernel. Tiny (32 KB per threadgroup at most; the kernels use a few KB, the fused-norm matmuls up to the whole 32 KB),
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
Metal API**. We skip the framework and use Metal directly: the project only needs a few dozen kernel templates, and the whole
native layer is ~270 lines. The GPU work — writing kernels, managing buffers and command buffers — is the layer
MLX would otherwise do for us.

---

## 8.1 Five layers, one responsibility each

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ MetalBackend.java   know the MODEL: which buffer is Wq of layer 3, the order of      │  Java
│ MlaMetal.java       operations in a forward pass (dense / DeepSeek MLA / Mamba);     │
│ MambaMetalBackend   own the weights, caches and activations                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ MetalKernels.java   knows the KERNELS: one writer per kernel (buffers, parameter     │  Java
│                     struct via Params/KernelParams, grid), the shared constants      │
│                     (MetalConstants) and tunables; owns the Metal context            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ Metal.java          knows NOTHING about models: compile shader source, make/wrap     │  Java (FFM)
│                     buffers, memoized pipelines, begin / dispatch / commit / wait    │
├───────────────────────────── C function calls (jm_*) ───────────────────────────────┤
│ jllm_metal.m        Objective-C: talks to the Metal framework; 16 C functions        │  native
├─────────────────────────────────────────────────────────────────────────────────────┤
│ kernels.metal       Metal Shading Language (C++ dialect): the code the GPU runs;     │  GPU
│                     compiled at start-up from source, in eight sections (0-7)             │
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

The 16 exported functions:

| C function | does | called from |
|---|---|---|
| `jm_create(source, err, len)` | `MTLCreateSystemDefaultDevice()`, a command queue, and `newLibraryWithSource:` — compiles the entire `kernels.metal` text at run time with `MTLMathModeRelaxed` (fast math that preserves the `-INFINITY` sentinels of the reductions; macOS 15+, older systems keep the default mode). Returns an opaque context pointer, or NULL with the compiler's message in `err` | `Metal.<init>` |
| `jm_destroy(ctx)` | releases everything | `Metal.close` |
| `jm_device_name(ctx, out, len)` | e.g. "Apple M1 Pro" | `MetalBackend.name` |
| `jm_recommended_working_set(ctx)` | how many bytes the GPU can keep resident without paging | start-up warning |
| `jm_device_max_tg_mem(ctx)` | `maxThreadgroupMemoryLength` (32 KB): the fused rmsnorm+matmul kernels keep `dim` floats in threadgroup memory next to 32 bytes of static scratch, so they are only used when `dim·4 + 32` fits | `MetalKernels.fusedNormFits`, called by `MetalBackend.init` |
| `jm_pipeline(ctx, name, err, len)` | looks up kernel `name` in the compiled library, builds a `MTLComputePipelineState`, stores it in an array and returns its **index** (so Java never holds Objective-C objects) | once per kernel |
| `jm_pipeline_max_threads(ctx, idx)` | the pipeline's `maxTotalThreadsPerThreadgroup` | available for sanity checks |
| `jm_buffer_new(ctx, len, zero)` | `newBufferWithLength:options:Shared`; `zero != 0` clears it with `memset`, otherwise the pages are left untouched and only committed when first written. Every scratch buffer the backends allocate is written before it is read (KV rows by the projections, activations per chunk), so they pass `zero = 0` and a 4 096-token context does not clear a gigabyte at start-up | KV cache, activations, norm weights |
| `jm_buffer_wrap(ctx, ptr, len)` | `newBufferWithBytesNoCopy:` — wraps memory **owned by Java** (the mmap'd cache file) without copying. If Metal refuses (`ptr` or `len` not page aligned) it prints the reason to stderr and returns NULL, and `Metal.wrap` throws: there is deliberately no copying fallback, which would silently double the memory of a multi-gigabyte model | every quantized weight tensor |
| `jm_buffer_contents(buf)` | the CPU pointer to a buffer's bytes | Java reads logits / writes embeddings |
| `jm_buffer_release(buf)` | drop the reference | `Metal.close` |
| `jm_begin(ctx)` | `commandBufferWithUnretainedReferences` + a compute encoder with `MTLDispatchTypeSerial` — dispatches execute **in order**, each seeing the previous one's writes, so Java needs no explicit barriers between kernels | start of `forward` |
| `jm_dispatch(ctx, pipe, bufs[], offsets[], nbuf, params, plen, tgmem, gx,gy,gz, tx,ty,tz)` | `setComputePipelineState`, `setBuffer:offset:atIndex:` for `i < nbuf`, `setBytes:` for the params at index `nbuf`, `setThreadgroupMemoryLength:` for the kernel's dynamic `[[threadgroup(0)]]` argument when `tgmem > 0`, then `dispatchThreadgroups:threadsPerThreadgroup:` | once per kernel launch |
| `jm_commit(ctx)` | `endEncoding`, `commit` **without waiting**, then opens a fresh command buffer and encoder: the GPU starts on the layers recorded so far while Java encodes the next ones. The queue runs command buffers in submission order, so the serial guarantee of `jm_begin` carries across the split | every 8 layers (`-Djllm.metal.commitLayers`) |
| `jm_abort(ctx)` | ends and drops the command buffer being recorded without running it (Metal aborts the process if an encoder is released un-ended), after waiting for the parts `jm_commit` already submitted | `forward` when recording throws |
| `jm_commit_wait(ctx, err, len)` | `endEncoding`, `commit`, `waitUntilCompleted`; returns −1 with the error text if any command buffer of the pass has status `MTLCommandBufferStatusError` (a GPU fault) | end of `forward` |

"Unretained references" means the command buffer does not reference-count the buffers it uses — slightly
cheaper, and safe because `Metal.java` keeps every buffer alive in its `buffers` list until `close()`
(which is idempotent).

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
                 const void *params, int plen, int tgmem, int gx, int gy, int gz, int tx, int ty, int tz);
```

Java describes it once in the constructor:

```java
dispatch = LINKER.downcallHandle(lookup.findOrThrow("jm_dispatch"),
    FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, ADDRESS, ADDRESS, JAVA_INT,
                              ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));
```

and calls it per kernel launch:

```java
dispatch.invokeExact(ctx, pipe, bufs, offsets, nbuf, params, plen, tgmem, gx, gy, gz, tx, ty, tz);
```

`bufs` is a native array of `MAX_BUFS = 16` pointers (`arena.allocate(ValueLayout.ADDRESS, 16)`; the fused
q/k/v projection binds 14), `offsets` an array of 16 longs, `params` a 64-byte scratch block, all owned by
`MetalKernels`. Its `bind(i, buffer, offset)` writes slot `i` of the first two with `setAtIndex`; the parameter struct is written field by field (next section). Because the native
side copies the params (`setBytes`) and records the buffer bindings at dispatch time, the same three scratch
segments are reused for every dispatch.

**Getting a Java array into GPU memory.** `MetalKernels.upload(float[] data)`:

```java
Metal.Buffer b = metal.newBuffer((long) data.length * 4, false);                 // shared buffer, not zeroed
MemorySegment.copy(data, 0, b.contents(), ValueLayout.JAVA_FLOAT, 0, data.length); // one bulk copy
```

`b.contents()` is the `jm_buffer_contents` pointer reinterpreted to the buffer's size. The same path in reverse
reads the logits back (`MemorySegment.copy(logitsBuf.contents(), JAVA_FLOAT, 0, logits, 0, rows)`), and the
embedding rows are written straight into the `x` buffer before each forward. Quantized weights never go through
a copy at all: `MetalKernels.wrap(tensor.data)` (which is `Metal.wrap`) passes the mmap'd segment's address and size to `jm_buffer_wrap`.

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
2. metal.begin()                                            ← a command buffer for the layers below
3. for each layer: record the layer's dispatches; after every 8 layers metal.commit() submits them
   (no wait) and opens the next command buffer, so the GPU runs them while the rest is encoded
4. if logits wanted: final rmsnorm of the last token, LM-head matmul
5. metal.commitAndWait()                                    ← the ONLY CPU↔GPU synchronization
6. CPU: copy the logits out of logitsBuf
```

### Dispatch list per dense layer (Qwen3-0.6B: dim 1024, qDim 2048, kvDim 1024, ffn 3072, 16/8 heads)

Grid sizes are *threadgroups*; `n` = tokens in this call. A decode step (`n = 1`) folds every rmsnorm into
the projection that follows it (the `*_norm` kernel variants normalise the input row on the fly, see
[8.5](#85-the-kernels-one-by-one)); a prefill chunk writes `xb` first and uses the tiled `*_pf` matmuls
([8.10](#810-the-tiled-prefill-matmuls)).

| # | decode `n = 1` | prefill `n = 64` | writes | grid (decode / prefill) | threads |
|---|---|---|---|---|---|
| 1 | — | `rmsnorm` | `xb` | — / 64 | 256 |
| 2 | `matmul3_q4_norm` | `matmul3_pf_q4` | `q`; k and v rows **straight into the caches** at `pos` (bias in the epilogue) | (1024, 1) / (512, 8) | 128 |
| 3 | `rope` (16 q heads + 8 k heads, qk-norm) | `rope` | `q` in place, the k cache rows | (24, 1) / (24, 64) | 64 |
| 4 | `attention_g1_d128` | `attention_g1_d128` (2 tiles of `ATT_TILE` = 32 tokens) | `att` | (16, 1) / (16, 32) × 2 | 256 |
| 5 | `matmul_q4` (Wo, accumulate) | `matmul_pf_q4` | `x +=` | (256, 1) / (128, 8) | 128 |
| 6 | — | `rmsnorm` | `xb` | — / 64 | 256 |
| 7 | `matmul_swiglu_q4_norm` (Wgate∥Wup) | `matmul_swiglu_pf_q4` | `hidden = silu(g)·u` | (768, 1) / (768, 8) | 128 |
| 8 | `matmul_q4` (Wdown, accumulate) | `matmul_pf_q4` | `x +=` | (256, 1) / (128, 8) | 128 |

28 layers × 6 = 168, plus the final `rmsnorm` and the LM head `matmul_q8` (151 936 rows, grid (37 984, 1),
Gemma's logit soft-cap in its epilogue): **170 dispatches per decode step**, in 4 command buffers
(`-Djllm.metal.commitLayers=8`). With `backend.kv-dtype=f16` the kernel names gain `_h` and k goes through
a staging row — [8.9](#89-the-f16-kv-cache); with `q8` they gain `_q8`, k and v go through staging rows and one
`kv_store_q8` per layer quantizes them. With `-Djllm.metal.flashDecode=true` step 4 of a decode step
over ≥ 512 positions becomes `attention_split` + `attention_combine` — [8.11](#811-flash-decoding-chosen-at-start-up).

### Dispatch list per MoE layer (Qwen3-30B-A3B: dim 2048, qDim 4096, kvDim 512, 128 experts, top-8, moeDim 768)

| # | kernel | grid, decode `n=1` | grid, prefill `n=64` | threads |
|---|---|---|---|---|
| 1–5 | as above with the 30B shapes | | | |
| 6 | `rmsnorm` | 1 | 64 | 256 |
| 7 | `router` (DeepSeek: `router_ds`) | 1 | 64 | 256 |
| 8 | `moe_up_q4` (1536 rows per expert, SwiGLU in the epilogue) | (192, 8) | (192, 512) | 128 |
| 9 | `moe_down_q4` (2048 rows per expert) | (512, 1) | (512, 64) | 128 |

48 layers × 8 + 2 = **386 dispatches per decode step**.

### Parameter structs: `Params`, `KernelParams` and `MetalParamsTest`

Each kernel takes a small `struct` by `constant` reference. Java has no struct type, so every writer in
`MetalKernels` fills the fields in declaration order through the typed builder and passes the returned size
as `plen`:

```java
p.reset(KernelParams.MATMUL).i(t.rows).i(t.cols).i(n).i(acc).f(eps).f(softcap).i(bias != null);   // MatmulParams
dispatch(kernel("matmul", t.type, ""), 6, tgmem, gx, gy, 1, MM_THREADS, 1, 1);   // params at 6, plen = p.done() = 28
```

This only works if the C++ struct has the same layout, which is guaranteed when every member is a 4-byte `int`
or `float` (no padding is ever inserted). Three checks keep the two sides together: `KernelParams` lists each
struct's name and field kinds (`MATMUL("MatmulParams", "iiiiffi")`), `Params.done()` fails the dispatch if a
writer wrote a different number or kind of fields, and `MetalParamsTest` parses the `struct` blocks of
`kernels.metal` and asserts that every layout has the same field count, kinds and byte size as its struct, that
no struct is without a writer, and that the constants `MetalConstants` mirrors equal the shader's `#define`s
(`java -cp target/classes:target/test-classes org.robomiri.jllm.MetalParamsTest`, no GPU needed).

| struct | fields | bytes |
|---|---|---|
| `MatmulParams` | rows, cols, ntok, accumulate, eps, softcap, hasBias (the `o_proj` bias, buffer 5) | 28 |
| `Matmul3Params` | rows0, rows1, rows2, cols, ntok, eps, hasBias | 28 |
| `SwigluParams` | hidden, cols, ntok, eps, gelu | 20 |
| `NormParams` | n, eps | 8 |
| `RopeParams` | headDim, qStride, kvStride, nHeads, pos0, eps, useNorm, rotOff, rotHalf | 36 |
| `AttnParams` | nHeads, nKvHeads, qkDim, vDim, pos0, ctx, scale, qStride, kStride, vStride, oStride, window, tokenOffset, softcap | 56 |
| `AttnSplitParams`, `AttnCombineParams` | flash decoding ([8.11](#811-flash-decoding-chosen-at-start-up)) | 48, 12 |
| `RouterParams` / `RouterDsParams` | nExperts, dim, topK, normalize (+ DeepSeek's groups, method, scale) | 16 / 40 |
| `MoeParams` | rowsPerExpert, cols, topK | 12 |
| `MlaParams`, `MambaConvParams`, `MambaScanParams` | MLA latent store / absorption (nope, rope, kvLora, vDim, heads), Mamba conv and scan | 20, 12, 16 |

Buffers are bound at indices `0 … nbuf−1` and the params at index `nbuf`, matching the `[[buffer(i)]]`
attributes in the kernel signature; a buffer a kernel declares but the dispatch does not use (norm weights,
biases) is bound to a small dummy buffer. Getting one index wrong is the classic bug: the kernel reads the
wrong buffer and silently produces garbage, which is why the CPU backend exists as a reference.

### Buffers allocated at start-up

| buffer | size (0.6B, ctx 4096, batch 64) | size (30B, ctx 2048) |
|---|---|---|
| weights (wrapped, zero copy) | 452 MB | ≈ 19.7 GB |
| `keyCache` + `valCache`, per layer `ctx·kvDim·4` (`·2` with `backend.kv-dtype=f16`) | 28 × 2 × 16 MB = 939 MB (470 MB f16) | 48 × 2 × 4 MB = 403 MB (201 MB f16) |
| `x`, `xb` (`batch·dim·4`) | 256 KB each | 512 KB |
| `q`, `att` (`batch·qDim·4`) | 512 KB each | 1 MB |
| `hidden` (`batch·ffn·4`) | 768 KB | — (dense MLP unused) |
| `scores` (`32·heads·ctx·4`: one row per tile token and head) | 8.4 MB | 8.4 MB |
| `logitsBuf` | 608 KB | 608 KB |
| MoE: `assign`, `routeWeight`, `moeHidden` | — | 2 KB, 2 KB, 1.6 MB |
| `kStage` (f16 cache only, `batch·kvDim·4`), `flashPart` (flash decoding only, `heads·32·(headDim+2)·4`) | 256 KB, 266 KB | 128 KB, 532 KB |
| norm gains, RoPE tables (uploaded copies) | ~2 MB | ~3 MB |

None of the scratch buffers is zeroed at allocation (`jm_buffer_new` with `zero = 0`): every element is written
before it is read, so the memset would only cost start-up time and commit untouched pages.

`recommendedMaxWorkingSetSize` is compared with the weight size at start-up; exceeding it does not fail but
means the GPU will page weights in and out.

---

## 8.5 The kernels, one by one

All kernels follow the same recipe: *figure out which output element(s) this thread owns from its indices,
compute, write*. We go through them in the order they appear in `kernels.metal`.

### `tg_sum`, `tg_max` — threadgroup reductions

Shown in 8.0. Used by `rmsnorm` / `rmsnorm_add` (sum of squares), `rope` (per-head sum of squares),
`attention_split` / `attention_combine` and the routers (max and sum of a softmax). `attention_t` does the same
two-level reduction inline, with one slot per grouped head (`red[G * 8]`). `buf` has 32 entries because a threadgroup has at most 1024 / 32 = 32 simdgroups.

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

(`block_dot<2>`, the Q2_K specialisation, dequantizes two 16-weight sub-blocks per 32-weight block and keeps its
scales inside the block, see [13](13-q2k-quantization.md); the tiled prefill kernels of
[8.10](#810-the-tiled-prefill-matmuls) use the equivalent `q4_block_dot` / `q8_block_dot`.)

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

For `n = 1024` each thread handles 4 elements. `MetalKernels.rmsnorm` binds the input at a byte offset so
the final norm can read only the last token's row (`(n−1)·dim·4`) and write row 0 of `xb`. At decode the
norm before a projection is not a dispatch at all: the `matmul_*_norm` variants recompute the row's scale in
every threadgroup with `norm_scale_128`, which reproduces the 256-thread reduction of `rmsnorm` lane for lane
on 128 threads, so the fused result is bit-identical to `rmsnorm` + `matmul` (used only while the normalised row fits the 32 KB of
threadgroup memory, `jm_device_max_tg_mem`). `rmsnorm_add` is the same kernel ending in `out[row] += …`: Glm4 and
Gemma normalise the output of a sub-layer before the residual add.

### `matmul3` and `matmul_swiglu` — the fused projections

`matmul3` runs the q, k and v projections in one dispatch: the row range `[0, rows0)` comes from Wq into `q`,
`[rows0, rows0+rows1)` from Wk into the key cache rows at `pos`, the rest from Wv into the value cache rows,
each row exactly as `matmul` would compute it, with the optional q/k/v bias added in the epilogue
(`out = v + b`, the same float as a separate pass). The v output is templated on the cache element type, so
the f16 cache receives `half(v + b)` directly ([8.9](#89-the-f16-kv-cache)). `matmul_swiglu` gives one
simdgroup the gate row `r` and the up row `hidden + r` of the stacked gate∥up matrix and writes
`silu(g)·u` (or Gemma's `gelu_tanh(g)·u`), so the FFN needs no gate/up scratch and no activation pass.

### `rope<KV, ADJACENT>` — one thread per rotary pair

One kernel for every rotary variant: grid `(nHeads + k heads, ntok)`; `rotHalf` threads per threadgroup,
thread `i` owning one pair of the rotated slice that starts at `rotOff` of each head. `ADJACENT = false`
pairs `(i, i + rotHalf)` — the "rotate-half" pairing from
[05 §5.5](05-transformer-math.md#55-step-3--qk-norm-and-rotary-position-embedding-rope) used by Llama, Qwen,
Gemma and Phi's partial rotation; `ADJACENT = true` pairs `(2i, 2i+1)` for GLM and DeepSeek. For the 0.6B
model (`rotHalf = 64`, `rotOff = 0`):

```
a = h[i];  b = h[i + 64]
if useNorm:  ss = tg_sum(a² + b²)              (64 threads cover all 128 numbers of the head exactly once)
             scale = rsqrt(ss / headDim + eps);  a *= scale * w[i];  b *= scale * w[i + 64]
pos = pos0 + token index
h[i]      = a·cos[pos][i] − b·sin[pos][i]
h[i + 64] = b·cos[pos][i] + a·sin[pos][i]
```

Threadgroups `x < nHeads` rotate q heads in place; the others read a k head from `k` (the f32 cache rows at
`pos`, or the f32 staging row of an f16 cache) and write it to `kOut` with the cache element type `KV`
(dims outside the rotated slice copied along), which is where the f16 cache rounds k, once, after the
rotation. The `cos`/`sin` tables are the `Rope` arrays uploaded once (`[ctx × rotHalf]` each); DeepSeek rotates
its q heads and the shared `k_pe` row through the q path (`rotOff = nope`, no k heads).

### `attention_t<KV, G, PERK, PERV, MAXP>` — one threadgroup (256 threads = 8 simdgroups) per (head group, token)

One template behind every attention variant: `KV` is the cache element type (`float`, or `half` converted on
load), `G` the query heads sharing a threadgroup (1 unless `-Djllm.metal.attnGroup` asks for 2/4/8, which are
then compiled on demand), `PERK` / `PERV` the key and value head widths in units of 32 lanes — compile-time for
head dims 64 / 128 / 256 (`attention_g1_d128`), or 0 for the widths in the params (`attention`: other head
dims; `attention_mla`: DeepSeek's absorbed queries, `kv_lora_rank + rope` = 576-wide keys and 512-wide values over
one shared kv head), and `MAXP` the bound of those runtime widths (`MAX_PER` = 8, or `MLA_MAX_PER` = 18 lanes of 32
for `attention_mla`). Per head the arithmetic is the same whatever the arguments. The tokens of a chunk are
dispatched in tiles of `ATT_TILE` = 32 (`MetalKernels.attention` loops over them), so the scores scratch needs
only 32 rows per head.

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
`simd_sum`. The `scores` buffer is indexed `[(tl·heads + h)·ctx + s]` with `tl` the token's index inside its tile, so
every (token, head) of the tile has its own row of `ctx` floats — putting scores in device memory rather than threadgroup memory is what allows `ctx = 4096`
(16 KB per row would not fit the 32 KB threadgroup limit for several rows). Causality is `len = pos0 + t + 1`.
GQA: `kvh = h / (nHeads / nKvHeads)` selects the shared KV head.

The two `threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)` calls make the scratch writes
of phase 1 visible to phase 2, and phase 2's to phase 3. With `softcap > 0` (Gemma 2) each score goes through
`tanh(s / cap) · cap` before the max.

### `router` / `router_ds` — one threadgroup (256 threads = 8 simdgroups) per token

```
simdgroup sg:  logits[e] for e = sg, sg+8, ...:  the 32 lanes read the router row in float4 steps
               (512 contiguous bytes per iteration) and finish with simd_sum
tg_max, exp, tg_sum:   the softmax max and sum as threadgroup reductions
thread 0:   8 times: pick the largest remaining, assign[t·8+k] = e, rw[t·8+k] = p_e; set it to −∞
            if normalize: rw /= Σ selected p_e
```

Thread 0 doing 8 passes over 128 entries is ~1 000 operations — trivial next to a 1 MB read, and it keeps the
selection exactly identical to the CPU backend's router (softmax first, then top-k, then renormalize, like
transformers). `router_ds` computes the logits the same way and applies DeepSeek's rule on thread 0 (sigmoid or
softmax scores, group-limited selection, bias for `noaux_tc`, scaling; see `DeepSeekRouter.java`). The logits
live in threadgroup memory (`MAX_EXPERTS = 1024`, checked in the constructors).

### `moe_up<QT>` — one simdgroup per (slot, hidden unit)

A **slot** is `s = t·topK + k`: the `k`-th chosen expert of token `t`. Grid `(ceil(768/4), n·8)`:

```
row   = tgp.x·4 + sg                     which of the expert's 768 hidden units
s     = tgp.y;  t = s / 8;  e = assign[s]
growG = e·1536 + row;  growU = growG + 768   gate row and up row inside the stacked [128·1536, 2048] tensor
g, u  = Σ_blocks block_dot<QT>(…, growG / growU, …, x[t])   over nb = 64 blocks, 2 per lane
simd_sum both;  lane 0 writes moeHidden[s·768 + row] = silu(g) · u
```

The SwiGLU is in the epilogue, so there is no gate/up scratch and no activation pass.

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

`host_name` is the string `jm_pipeline` looks up, and `Metal.pipeline(name)` looks it up lazily: a pipeline
is created the first time a dispatch names it and memoized, so only the variants a model runs are built, and
`MetalKernels.kernel(base, type, suffix)` assembles the name (`matmul3` + `q4` + `_norm_h`). Every quantized
kernel is a template on `QT` (4 = Q4_0, 8 = Q8_0, 2 = Q2_K), the matmul family additionally on the fused-norm
flag (and `matmul3` on the v output type), the attention, rope and `mla_lat_store` kernels on the cache element
type, `rope` also on the pairing (`ADJACENT`).
The grouped attention instantiations (`attention_g{2,4,8}_*`) sit behind `#ifdef JLLM_ATTN_GROUPS`, which
`MetalBackend` defines (through `Metal.create`'s preamble) only when `-Djllm.metal.attnGroup` asks for them.

---

## 8.6 Gotchas met along the way

- **Position attributes must agree in dimensionality.** A kernel cannot mix `uint2 tgp
  [[threadgroup_position_in_grid]]` with a scalar `ushort tid [[thread_position_in_threadgroup]]` — the
  compiler rejects it. Hence `ushort2 tid2` and `tid2.x` in `rope` and `attention_t`. `simdgroup_index…`
  and `thread_index_in_simdgroup` are not position attributes and may stay scalar.
- **Struct layout**: only `int`/`float` members, in the order Java writes them; `plen` must match the struct size.
- **`newBufferWithBytesNoCopy` needs page alignment**; otherwise it returns nil, the shim prints why to stderr
  and `Metal.wrap` throws — a silent copying fallback would be fine for small buffers and catastrophic for a
  19 GB model (double memory). All `QTensor`/`QuantFile` blobs are 16 KiB-aligned so the fast path is always
  taken. Read-only `mmap`'d pages work (this is what llama.cpp does).
- **Unretained command buffers**: every buffer used in a dispatch must outlive the command buffer. `Metal.java`
  keeps all buffers in a list until `close()`; `ModelWeights` (the mmap) must be closed *after* the backend.
- **Shader errors arrive at run time**, as text in `jm_create`'s `err` buffer, with line numbers relative to
  the source string — `kernels.metal` is compiled when the backend starts, not when the Java code is built.
- **Every declared `[[threadgroup(0)]]` argument needs a length.** The plain matmul instantiations share their
  signature with the fused-norm ones and so declare the dynamic threadgroup argument without using it; the
  dispatch still binds a 16-byte dummy length (`UNUSED_TG_BYTES`) for them.
- **GPU faults** (out-of-bounds reads, usually from a wrong `bind` index or grid size) show up as
  `MTLCommandBufferStatusError` from `jm_commit_wait`, often with an unhelpful message. The reliable debugging
  method is the CPU backend: run the same prompt with `--backend cpu --temp 0` and compare outputs token by
  token; they must be identical.
- **`exp` precision**: Metal's `exp` is not guaranteed bit-identical to Java's `Math.exp`. In practice greedy
  outputs match the CPU exactly, but a near-tie in the router or sampler could in principle flip.

---

## 8.7 Measured numbers and where the time goes

Apple M1 Pro, Qwen3-0.6B Q4_0, 413-token prompt (`bench`): **~1 360 tok/s prefill, ~111 tok/s decode** (the f16
cache is similar); Qwen3-30B-A3B Q4_0: **~49 tok/s prefill, ~25 tok/s decode**. The CPU backend on the same
machine does ~275 / ~51 (0.6B, int8 activations) and ~12 / ~6 (30B); the full table is in
[11.1](11-performance.md#111-measured). Greedy output is byte-identical to the CPU backend with float
activations (the int8 CPU path can diverge at a near-tie). Before the tiled prefill matmuls and the pipelined
command buffers the 0.6B model prefilled at ~380 tok/s and the 30B at ~44 tok/s; decode was within noise of
today's numbers.

Where a decode step of the 0.6B model (~9 ms) goes:

| cost | evidence |
|---|---|
| **170 dispatches per token** | each costs a few µs of GPU front-end time plus the serial dependency between consecutive kernels; the fusions ([8.4](#84-recording-a-forward-pass-metalbackendforward)) brought the count down from 338 |
| **under-filled grids** | `Wo`, `Wdown` have 1024 rows → 256 threadgroups of 128 threads = 32 768 threads on a GPU that wants hundreds of thousands in flight; the M1 Pro's 16 cores are mostly idle during these kernels |
| **the LM head** | `151 936 × 1024` Q8 = 165 MB, a third of all bytes read per token, in one 37 984-threadgroup dispatch |
| **attention reads the whole prefix** | per layer, per head: `len × 128 × 4 B` of keys and the same of values; at 4 096 tokens that is 4 MB per head per layer — the `half` cache halves it, flash decoding ([8.11](#811-flash-decoding-chosen-at-start-up)) spreads it over more threadgroups |
| **a CPU round trip per token** | `commitAndWait` + logits copy + sampling + embedding write: the GPU idles while the CPU samples |

A prefill chunk of 64 tokens is bound by the matmuls' reads of the activations (the single-row kernels gather x
at a 128-byte lane stride per row), which is what the tiled kernels of [8.10](#810-the-tiled-prefill-matmuls)
attack; at several thousand tokens of context the attention kernel (one threadgroup per head and token,
reading the whole prefix) dominates instead.

### What remains, by expected payoff

*Done since this list was written:* expert-grouped MoE prefill on simdgroup-matrix tiles
([8.15](#815-expert-grouped-moe-prefill)), with MoE models prefilling in 256-token chunks; flash-attention prefill
([8.13](#813-flash-attention-prefill-attention_fa)); and GPU sampling with pipelined decoding
([8.16](#816-pipelined-greedy-decoding)). Not yet measured on a Mac. What is left:

1. **Wider threadgroups / more rows per threadgroup for small matrices** so `Wo`-sized matmuls fill the GPU,
   and splitting the LM head over the token dimension in prefill.
2. *Done:* the remap and the usage counts of the experts on the CPU are part of the router kernels
   ([8.18](#818-experts-on-the-cpu-backendmetalcpu-moe)).

---

## 8.8 Summary of the core design

- The GPU runs *one function per output element*; we express every transformer operation as such a function
  and launch grids of threadgroups (`kernels.metal`).
- Dot products end with `simd_sum`; threadgroup-wide sums use a two-level reduction through threadgroup memory
  and barriers.
- Unified memory lets the memory-mapped quantized cache be used by the GPU in place (`jm_buffer_wrap`).
- A ~270-line Objective-C shim exposes 16 C functions; `Metal.java` binds them with the FFM API (`Linker`,
  `FunctionDescriptor`, `MethodHandle.invokeExact`, `MemorySegment`), no JNI, no generated code.
- `MetalBackend.forward` records the whole forward pass (170 dispatches for 0.6B, 386 for 30B-A3B) through
  the writers of `MetalKernels` (per-layer strategy objects picked once from the `LayerSpec`), submitting every 8 layers so the GPU runs them while the rest is encoded, and
  waits once at the end.
- Every parameter struct has one Java writer, checked against the shader source by `MetalParamsTest`.
- Decode on the M1 Pro is limited by dispatch count and small grids rather than by bandwidth; prefill by the
  matmuls' activation reads (tiled kernels) and, at long contexts, by attention.
- The sections that follow cover the pieces layered on top of this design: the f16 KV cache (8.9), the tiled
  prefill matmuls (8.10), flash decoding (8.11) and the DeepSeek MLA / Mamba kernels (8.12).

---

## 8.9 The f16 KV cache

`backend.kv-dtype=f16` allocates `keyCache`/`valCache` as `half` buffers (`ctx·kvDim·2` bytes per layer, 470 MB
instead of 939 MB for the 0.6B model at `ctx = 4096`) and selects the `_h` instantiations of three kernels;
the dispatch count does not change and everything else is untouched.

| step | f32 (default) | f16 |
|---|---|---|
| 2 `matmul3` | k and v rows into the caches at `pos·kvDim·4` | `matmul3_*_h`: v rows written as `half(v + bias)` into the cache at `pos·kvDim·2`; k rows into the f32 staging buffer `kStage` (`batch·kvDim·4` bytes) |
| 3 `rope` | k in place on the cache rows | `rope_h`: reads `kStage`, writes the rotated head as `half` into the cache rows |
| 4 attention | `attention_g1_d128` | `attention_g1_d128_h` — the same template with `KV = half` |

Each cache element is rounded exactly once — v after the projection, k after the norm and the rotation — with
Metal's default round-to-nearest-even `half()` conversion, the same rounding as Java's `Float.floatToFloat16`
on the CPU path, which is why the two backends' f16 outputs are identical.

The `half` attention instantiations differ from the `float` ones only in the pointer types of `kc`/`vc` and in
the loads, which widen each `half` to `float` before the multiply (`float(kv[lane + i * 32])`). Scores, softmax
and the value accumulation stay f32, so the f16 variant changes what is *read*, not what is computed: per
cached position and lane it loads 4 × 2 bytes of K and 4 × 2 bytes of V instead of 4 × 4 each. On the M1 Pro
this is worth a few percent of decode at a few hundred tokens of context (the attention kernel is not the
dominant cost there — [8.7](#87-measured-numbers-and-where-the-time-goes)); at 3 000 tokens the prefill is a
third faster and decode ~10 % faster. Measurements are in [11 §11.3](11-performance.md#113-memory-footprint).

### q8

`backend.kv-dtype=q8` stores int8 values with one f32 scale per (position, kv head) row (the math and the error
bound are in [07 §7.13](07-cpu-backend.md#713-the-f16-kv-cache-kvcache)). A position's row of `keyCache` /
`valCache` is its `kvDim` int8 values followed by `kvHeads` f32 scales — `kvDim + 4·kvHeads` bytes, 1 056 instead of
2 048 (f16) for the 0.6B model — which is also exactly the saved-cache row ([8.17](#817-saving-loading-and-copying-the-kv-cache)).
Quantizing needs a whole row's maximum, which no single thread of the projection epilogue or of `rope` sees, so both
k and v take the f32 path and a new kernel quantizes them:

| step | q8 |
|---|---|
| 2 `matmul3` | the plain (`float`) variant: k rows into `kStage`, v rows into a second staging buffer `vStage` |
| 3 `rope` | the `float` variant, k rotated in place in `kStage` |
| 3b `kv_store_q8` | new, grid `(2·kvHeads, n)`, one simdgroup per (token, kv head) row of k or v: `simd_max` of `|x|`, `scale = max / 127`, `q = rint(x / scale)` clamped to ±127 as `char`, the scale after the row's values |
| 4 attention | `attention_g1_d128_q8` (`KV = char`); flash decoding `attention_split_q8`, flash prefill `attention_fa_d128_q8` |

The attention kernels read `char` and multiply once per row: `kv_scale(row, stride, nKvHeads, h)` returns the scale of
a q8 row and the constant 1 for `float` / `half` rows (overloads, so the existing instantiations compile to the same
code), the score becomes `simd_sum(q · k_int) · (scale_attn · scale_k)` and the softmax weight `p · scale_v` before it
multiplies the integer values; `attention_fa` multiplies as it stages a block into threadgroup memory. Row strides
are passed in elements, which for `char` are bytes and include the scales. Every specialization the f16 cache has
exists for q8 (`attention[_q8]`, `attention_g{1,2,4,8}_d{64,128,256}_q8`), and `MetalTuner` checks the flash kernels
against `attention_t` on a q8 cache too (its own cache key `|q8`). DeepSeek's latent rows are stored by
`mla_lat_store_q8` (one scale per `[c | k_pe]` row, `aw + 4` bytes) and read by `attention_mla_q8`.

The cost is one extra dispatch per layer (`kv_store_q8`, `2·kvHeads·n` simdgroups) and the f32 round trip of v
through `vStage`; the reads in attention shrink from 2 to ~1 byte per value. CPU and Metal quantize the same f32
numbers with the same formula, but their f32 k / v differ in the last bits, so a value on a rounding edge can land one
step apart: `ArchSmoke` runs every family with q8 on both backends and requires the usual 10⁻³ relative agreement of
the logits (observed: at most 2.5 × 10⁻⁴ of the largest logit, on Gemma-2; ~10⁻⁶ for most families). It also
prints the distance from the f32 cache — 0.04 % to 1.2 % of the largest logit, up to 25 % on the tiny DeepSeeks,
whose routing amplifies any perturbation (their f16 caches already move the logits by 0.2–0.4 %) — and bounds it by
200 times the f16 cache's distance (observed 20–45× for the dense families, up to 115× for the DeepSeeks);
`kernel_tests.py` checks `kv_store_q8` bit for bit against numpy and the q8 attention, split and flash kernels
against a float64 reference.

---

## 8.10 The tiled prefill matmuls

The single-row matmuls give a simdgroup one output row and let each lane walk the row's quantization blocks
(`lane, lane + 32, ...`), re-reading the token's activations for every row: for a 64-token chunk the x
matrix is gathered once per row at a 128-byte lane stride, and that, not the weight reads, is what limits
prefill. `matmul_pf`, `matmul3_pf` and `matmul_swiglu_pf` give a simdgroup a tile of `PF_ROWS = 2` rows (one
gate+up unit for the SwiGLU kernel) × `PF_TOKS = 8` tokens: every x block is loaded once for the tile's rows
and every weight block unpacked once for its tokens. The lane walk over blocks, the block dot (the shared
`q4_block_dot` / `q8_block_dot` expression, so the compiler emits the same arithmetic for both families) and the
one `simd_sum` per output are the same as in the single-row kernels, so every output element is accumulated in
the same order and the results are **bit-identical** — checked element-wise for Q4_0, Q8_0 and Q2_K and on a
64-token chunk's logits for both cache types. Chunks of at least 8 tokens use them
(`-Djllm.metal.prefillMinTokens`).

Tile shape, measured per layer at the 0.6B shapes with 64 tokens: two adjacent rows per simdgroup coalesce
into 1 KB reads and halve the x gathers (4.0 ms per layer against 6.2–6.7 for the single-row kernels), four
rows cost too many registers, 16 tokens gain nothing more; the SwiGLU kernel is fastest with one unit
(1.5 ms against 2.2). End to end: 0.6B prefill 380 → 533 tok/s, 30B 44 → 49 tok/s (its prefill is dominated
by the MoE kernels, which still handle one token per simdgroup).

### Simdgroup-matrix kernels (chunks of 32+ tokens)

Chunks of at least `jllm.metal.mmaMinTokens` = 32 tokens use `matmul_mma`, `matmul3_mma` and
`matmul_swiglu_mma` instead (`-Djllm.metal.mmaPrefill=false` brings the scalar tiled kernels back). A threadgroup
of 4 simdgroups computes a `MMA_BR` = 64 rows × `MMA_BT` = 32 tokens output tile with `simdgroup_float8x8`
multiply-accumulates, walking the row in quantization blocks (32 weights = one k step):

1. every thread dequantizes 16 weights of one tile row (`value * scale` in f32; Q2_K through `q2k_sub_block`)
   into threadgroup memory `sa[row][k]` and copies 8 activations into `sb[token][k]`;
2. after a barrier each simdgroup owns a 32-row × 16-token quarter (4 × 2 accumulators) and runs four k8 steps,
   loading the weight operand transposed, so `C[token][row] += sb[token][k] · sa[row][k]`;
3. the tile is stored to threadgroup memory (aliasing `sa`) and written out by an epilogue that carries each
   shape's guards: accumulate / soft-cap (dense), bias and the three outputs with the f16 V cache (`matmul3`), or
   `silu(gate) * up` where the tile's first 32 rows are 32 gate units and the next 32 their up rows (SwiGLU / GeGLU).

The operands stay f32: `simdgroup_half8x8` inputs would round the dequantized weights (and activations) to 11
bits for no change in the memory traffic, and f32 MMA already moves the matmuls off the critical path. Because
the products are summed in a different order from the lane-walk kernels the results are **not bit-identical**:
on Qwen3-0.6B a 400-token prompt gives a largest logit difference of 8.6e-6 (max |logit| 10.2, same argmax), the
greedy text of seven diverse prompts (80 tokens each, 0.6B) and the 30B answer are unchanged, and the architecture
smoke test passes 60 / 60 with the MMA kernels forced on for its 12-token prompts (`JAVA_OPTS=-Djllm.metal.mmaMinTokens=2`).
A 64-token tile (16 accumulators per simdgroup) was ten times slower (register spills); 32 tokens it is.
Qwen3-0.6B prefill goes ~533 → ~1 360 tok/s (413-token prompt, best of several runs on a busy machine; the
scalar kernels measured 516 in the same session). The 30B went 18.7 → 34.0 tok/s in an A/B run under heavy load
(49 tok/s before on a quiet machine); its prefill is dominated by the MoE kernels. A 2 012-token prompt on the 0.6B
went from ~95 to ~300 tok/s (both loaded; attention over the growing prefix is a large share there).

---

## 8.11 Flash decoding (chosen at start-up)

`attention_split` + `attention_combine` replace the single-pass kernel for a decode step over at least 512 cached
positions (`-Djllm.metal.flashMinLen`): instead of one threadgroup per head walking the whole context,
`attention_split` runs the three phases of `attention_t` on a sub-range of `flashChunk = 128` keys per
threadgroup (grid `(heads, nSplit)`, `nSplit ≤ 32`), with the softmax relative to the sub-range's own maximum,
and writes the unnormalised partial `(o[headDim], max, sum)`; `attention_combine` rescales every partial to the
global maximum and divides by the combined sum:

```
m = max_j m_j      l = Σ_j l_j · e^(m_j − m)      o = Σ_j o_j · e^(m_j − m) / l
```

The combine adds the sub-ranges' sums in a different order from the single-pass kernel, so results agree to
rounding, not bit for bit. On Qwen3-0.6B with a 3 176-token prompt at `--ctx 4096` (M1 Pro) decode went
50–55 → 60–62 tok/s with the f32 cache (+3–5 % with f16), with the same greedy output and the largest logit
difference 4.4e-6. It used to be opt-in; now the start-up self-check ([8.14](#814-the-kernel-self-check-metaltuner))
turns it on when it agrees with `attention_t` and is faster on the chip. `-Djllm.metal.flashDecode=true|false`
forces the choice. Dense attention only; DeepSeek MLA keeps the single-pass kernel.

---

## 8.12 DeepSeek MLA and Mamba on the GPU

Both families reuse the generic kernels for their projections, norms and rotary embedding and add only the pieces
that carry their own state. Their parameter structs are `MlaParams`, `MambaConvParams` and `MambaScanParams` in
the table of [8.4](#84-recording-a-forward-pass-metalbackendforward).

**MLA** (`MlaMetal`, see `MlaCpu` and [14 §14.11](14-model-families.md#1411-deepseek-v2--v3-multi-head-latent-attention--shared-experts)
for the math). The cache holds the *latent* row `[c | k_pe]` (`kv_lora_rank + rope` = 576 numbers per token and
layer for DeepSeek-V2-Lite, f32 or half) instead of per-head keys and values, and `W_kb` is absorbed into the
query. Per layer after the projections and the two rotations (`rope_il` through the q path, `rotOff = nope`):

| kernel | grid | does |
|---|---|---|
| `mla_lat_store` (`_h`) | `(n)`, 64 threads | writes the cache row `[c, k_pe]` of every token |
| `mla_absorb_q` | `(heads, n)`, 256 threads | `qa[t, h] = [W_kb_k[h]ᵀ q_nope, q_pe]` |
| `attention_mla` (`_h`) | `(heads, tile)`, 256 threads | the generic `attention_t` with 576-wide keys, 512-wide values, one kv head shared by all heads |
| `mla_apply_v` | `(heads, n)`, 256 threads | `att[t, h] = W_kb_v[h] · o[t, h]` |

`W_kb_k` / `W_kb_v` live on the GPU dequantized to f32 (8 MB per layer for V2-Lite). The flash-decoding path is
dense attention only; MLA keeps the single-pass kernel.

**Mamba** (`MambaMetalBackend`): per layer RMSNorm, `in_proj`, `mamba_conv`, `x_proj`, `mamba_scan`, `out_proj`
(the projections are the ordinary matmul kernels). `mamba_conv` and `mamba_scan` run one thread per channel
(`d_inner` threads) and loop over the chunk's tokens inside the thread, because the recurrence is sequential over
tokens and independent over channels. The conv window and the channel's weights (`mamba_conv`) and the SSM state
row (`mamba_scan`, up to `MAMBA_MAX_STATE` = 64) stay in registers for the whole chunk and are written back once,
so a layer is one dispatch per stage regardless of the chunk size. The price is that a long prefill has only
`d_inner` threads of parallelism. See [15](15-mamba.md).

---

## 8.13 Flash-attention prefill (`attention_fa`)

🧸 Thirty-two children each need to compare their own question with every page of a big book. The old way: each
child walks to the shelf and reads every page alone. The new way: the teacher takes a few pages off the shelf, puts
them on the table, all 32 children read them together, and then the next few pages come out. The book leaves the
shelf only once for the whole group. And nobody writes down every comparison on paper — each child keeps just a
running score in their head.

📐 For a prefill chunk of `n` tokens, `attention_t` gives every (head, query token) its own pass over the keys and
keeps the scores in a scratch buffer. `attention_fa<KV, D>` is the flash-attention formulation: a threadgroup of
4 simdgroups (`FA_THREADS` = 128) takes `FA_ROWS` = 32 query rows of one head (8 rows per simdgroup) and walks the
keys they can see in blocks of `BK = 4096 / D` keys (32 for `D` = 128). Each block's K rows, then V rows, are staged
in threadgroup memory (16 KB) once and used by all 32 rows, so K and V are read from memory `n / 32` times fewer.

Per block, per row `i`, with running maximum `m_i`, running sum `l_i` and running output `o_i` (start
`m = −∞, l = 0, o = 0`):

```
s_ij  = scale · q_i · k_j                     (masked to −∞ when j > pos_i, or outside the sliding window;
                                               soft-cap: s = c · tanh(s / c))
m'_i  = max(m_i, max_j s_ij)
p_ij  = e^(s_ij − m'_i)
l_i   = l_i · e^(m_i − m'_i) + Σ_j p_ij
o_i   = o_i · e^(m_i − m'_i) + Σ_j p_ij v_j
m_i   = m'_i
                                    at the end:   out_i = o_i / l_i
```

This is exactly `softmax(s) · V`: whenever the maximum grows from `m` to `m'`, everything accumulated so far was
weighted by `e^(s − m)` and must become `e^(s − m')`, i.e. be multiplied by `e^(m − m')`. Why bother with the
maximum at all? `e^s` overflows f32 above `s ≈ 88`; `e^(s − m) ≤ 1` never does.

`S = Q Kᵀ` (8 × BK per simdgroup) and `O += P V` run on `simdgroup_float8x8` multiply-accumulates over the head in
8-wide slices. The row rescale `o_i · e^(m_i − m'_i)` is done as a product with a diagonal 8 × 8 matrix
`diag(e^(m − m'))`, because how a simdgroup matrix's 64 elements are spread over the 32 lanes is not specified, so
the kernel never addresses them one by one. Instances: `attention_fa_d64/d128/d256` and `_h` for a half KV cache
(widened to f32 as it is staged). The summation order differs from `attention_t`, so the two agree to rounding
(`kernel_tests.py` checks against a float64 reference, the self-check of [8.14](#814-the-kernel-self-check-metaltuner)
against `attention_t` on the device); `-Djllm.metal.flashPrefill=true|false` forces it.

---

## 8.14 The kernel self-check (`MetalTuner`)

🧸 Before using a new tool, a careful carpenter tries it on a scrap of wood next to the old tool. If the cut is the
same and quicker, the new tool goes in the box; otherwise it stays on the shelf. And he writes it down so he
doesn't have to test again tomorrow.

📐 The newer kernels (`attention_fa`, flash decoding) are not bit-identical to the ones they replace, and GPUs and
drivers differ. At load, `MetalTuner.decide` runs each candidate next to the established kernel on synthetic data
of the model's shapes (head size, group size, f32 or half cache) and accepts it only if

```
max |candidate − reference|  ≤  1e-3 · max |reference| + 1e-5
```

and, on a real GPU, if it is also faster (best of a few timed runs). The decision is cached in
`~/.cache/jllm/metal-tuning.properties` under the device name, a hash of the kernel source and the shape, so it runs
once (a few hundred ms) and again only after a driver or kernel change. On the emulator, timings mean nothing, so
correctness alone decides. It also chooses the threadgroup size of the plain single-token matmul (one
simdgroup per output row: 64, 128 or 256 threads, i.e. 2, 4 or 8 rows per threadgroup): it times the three on the
model's own output projection, checks that the results are bit-identical, and keeps the fastest. Small matrices fill
the GPU better with smaller threadgroups, and which size pays differs between chips. The log line `Metal kernels: flash prefill on|off, flash decode on|off (…)` says what was chosen and why.

---

## 8.15 Expert-grouped MoE prefill

🧸 A class of 64 children each asks two of eight teachers a question. The old way: every child walks to the
teacher's office alone, and the teacher opens her big book for each child. The new way: line the children up by
teacher first; then each teacher opens her book once and answers her whole line together.

📐 In an MoE layer every token picks `topK` experts. `moe_up` / `moe_down` give each slot (token, k) its own
simdgroups, so in a chunk an expert's weights are read once **per slot** that picked it. For a prefill chunk of at
least `jllm.metal.moeMmaMin` (16) tokens the grouped path runs instead:

1. `moe_sort` (one threadgroup): a stable counting sort of the `n · topK` slots by expert — count per expert,
   prefix sum, scatter — and the list of tiles `(expert, start, count)` of up to `MMA_BT` slots each. No atomics,
   so the order and the result are deterministic.
2. `moe_up_mma` / `moe_down_mma`: per tile, the simdgroup-matrix core of the dense prefill matmuls
   ([8.10](#810-the-tiled-prefill-matmuls)), `mma_core_rows`, with each slot's activation row **gathered** through
   `order[]`. The expert's weight rows are read once per tile, not once per slot.
3. `moe_down_mma` writes each slot's weighted output row to its own place; `moe_combine` sums a token's `topK`
   rows in k order, as `moe_down` does (no two tiles write the same element, no atomics).

Memory read for expert weights drops from `≈ n · topK` expert-matrix reads to `≈ tiles` (about
`experts_used + n·topK / MMA_BT`). Q6_K weights now also have a simdgroup-matrix path, so GGUF `Q6_K` layers use it
too. `-Djllm.metal.moeMma=false` keeps the per-slot kernels. Because the reuse grows with the tokens per expert
(`n · topK / E`: 4 at a 64-token chunk with 128 experts and top-8, 16 at 256), MoE models prefill in 256-token
chunks unless `backend.batch` is set ([12 §12.8](12-mixture-of-experts.md#128-why-decode-is-fast-and-prefill-is-not)).

---

## 8.16 Pipelined greedy decoding

🧸 See [17 §17.4](17-agent-speedups.md#174-pipelined-decoding-on-the-gpu) for the relay-race picture.

📐 For greedy sampling without penalties the GPU chooses the token itself: `argmax` (256 threads, ties go to the
lower index — the same first-maximum rule as `Sampler`) writes the id into a ring of 4 slots, and the next step
starts with `embed_token<QT>` reading that id and dequantizing its embedding row (`dequant_block`, one per format).
So step `j + 1` can be encoded and submitted before the CPU knows step `j`'s token; the CPU reads ids one step
behind through `jm_wait_pending(keep)`, which waits until at most `keep` command buffers are unfinished.
`MetalBackend.Pipe` implements `Backend.GreedyPipe` (`submit`, `next`, `pending`, `drain`); every other
backend call drains it first. `-Djllm.metal.pipeline=false` turns it off.

---

## 8.17 Saving, loading and copying the KV cache

`MetalBackend.saveState` / `loadState` / `copyStateFrom` implement the backend-neutral cache file and the slot copy
of [17 §17.2–17.3](17-agent-speedups.md#172-slots-that-share-a-prefix). The format string
`kv f16 layers=L kvDim=K headDim=D` (or `f32`, `q8`) is the same the CPU backend writes; rows are written as per
layer, per position, K then V, little-endian (q8: the row's `kvDim` int8 values, then its `kvHeads` f32 scales — the
cache's own row layout, so saving is a copy here too). The cache buffers live in unified memory, so loading reads from the file
straight into them, and a slot copy is a `memcpy` of `n` rows per layer between two engines' buffers.

---

## 8.18 Experts on the CPU (`backend.metal.cpu-moe`)

🧸 The kitchen has one very fast cook (the GPU) with a small worktable, and a slower cook (the CPU) next to a big
pantry. When the recipe book is too big for the worktable, the fast cook doesn't keep running to the cellar for
pages; he keeps most of the book on his table and hands a few chapters to the second cook. When the recipe reaches
one of those chapters, the fast cook waits a moment, the second cook does that step, and the fast cook goes on.

📐 **Why.** On Apple Silicon the CPU and GPU share one memory, but Metal keeps only about
`recommendedMaxWorkingSetSize` (about two thirds to three quarters of RAM) resident for the GPU. A model whose
GPU buffers exceed it pages: weights are evicted and read again from the SSD, every token. In an MoE model the
routed experts are nearly all the bytes (Qwen3-30B-A3B at Q4_0: about 18 of 19.4 GB), so leaving some layers'
experts out of the GPU's buffers is the cheapest way to fit. Those experts stay in the memory-mapped file and the
CPU reads them like any file data.

**How much must move.** The working set is the device's, so the budget covers everything on it: with `W` this
model's weights, `B` one backend's buffers (KV cache, activations), `s` the number of backends that will run on
these weights (a server's slots: `jllm-web` passes it at load; the later slots reuse the first one's placement), `O`
what the other open Metal backends hold (another model, a draft model, a Mamba model: `DeviceLedger`), and `b_l` the bytes of one expert of layer `l`,
leaving the set `C` of (layer, expert) pairs on the CPU keeps

```
G(C) = W + s · B + O − Σ_{(l,e) ∈ C} b_l          on the GPU, and C is chosen so that G(C) ≤ working set
```

`auto` (the default) moves nothing when everything fits. A draft model is loaded before its target, so the target's
plan already counts it. A backend whose load pushes the device past the working set prints a warning with the totals. `-Djllm.metal.workingSetGb=N` replaces Metal's figure, for
example to leave memory to other programs. A number (`backend.metal.cpu-moe=6`) puts every expert of that many MoE
layers on the CPU, `all` all of them, `off` none.

**Which experts: the hot ones stay on the GPU.** 🧸 In the kitchen, a few recipes are cooked all day and most
almost never. The fast cook keeps the popular pages on his table and gives the second cook the pages nobody orders.
The second cook is then rarely needed, although the same number of pages left the table.

📐 Routers are skewed: in a layer of 128 experts a minority takes most of the picks. If `p(l,e)` is how often a token
picks expert `e` of layer `l` (`Σ_e p(l,e) = topK`), moving it costs about `p(l,e) · b_l / BW_cpu` per token: a
never-picked expert costs nothing. But every layer with at least one CPU expert costs a GPU → CPU → GPU round trip
`S` per step, picked or not. `ExpertPlacement.plan` minimizes

```
cost(C) = S · |{ layers with a CPU expert }|  +  Σ_{(l,e) ∈ C} p(l,e) · b_l / BW_cpu
```

with `S` = 100 µs and `BW_cpu` = 60 GB/s (estimates; `-Djllm.metal.cpuMoeSyncUs` changes `S`). It works greedily,
taking the move with the lowest cost per byte freed: either the next-coldest expert of a layer that already has CPU
experts, or opening a new layer with its `k` coldest experts (the round trip shared by the `k`). The usage `p` comes
from earlier runs: counts plus one per expert. A layer without statistics is only ever moved whole (no expert of
it is known to be colder, so splitting it would only add a packed copy and a remap per step), so a first run
moves **whole layers**, freeing up to one layer more than needed; with real usage the plan moves the cold experts
of a few layers instead. The start-up log shows both the bytes and the expected share of
picks on the CPU:

```
Metal: 2.3 GB of experts run on the CPU (10 threads): 0 of 48 MoE layers entirely, 3 in part
(placed by the usage of earlier runs); expected share of expert picks on the CPU 4 %; GPU share …
```

**Where the usage comes from.** While any expert is on the CPU, every MoE layer's picks are counted. The router
kernel counts them itself (`remap_slot`): its thread 0, which picks each token's experts, adds one to `counts[e]`
per pick with a relaxed atomic add, so there is no extra dispatch and the counts are exact. A layer entirely on
the CPU is counted by the CPU from the router's choices it reads anyway. A layer entirely on the GPU is counted on
one forward pass in 8 (`-Djllm.metal.expertCountEvery`), because atomics in every router on every token would cost
more than they tell. A sample
is enough because the planner turns each layer's counts into shares of that layer's picks (`p(l,e)` sums to `topK`
in every layer), so how many passes a layer was counted on does not matter. The backend adds the counts to
`~/.cache/jllm/expert-usage/<model>.txt` (one line per MoE layer, headed by the model's description, size and a
hash of its first router, so fine-tunes of one base model keep separate files; halved once a layer passes 2³⁰
picks) every ten minutes, from a background thread, and when it closes. A lock file orders processes that save at
the same time; a damaged line loses only that layer's history. The first run of a model that does not fit therefore
moves whole layers; later runs place the experts from how you actually use it (code, chat, another language),
improving as the counts grow. `-Djllm.metal.hotExperts=false` ignores the statistics.

**Re-placement while running.** A server runs for days, and the usage it sees can change. Every 30 minutes of use
(or when `MetalBackend.rebalanceExperts()` is called), the backend saves its counts and plans again from them. If
the new plan's cost (the formula above) is at least 20 % lower, it moves the experts of the layers that change,
one layer at a time, between two forward passes: it packs the layer's new hot experts, switches the GPU to them and
frees the old copy, so at most one extra layer's copy exists at any moment. Only a placement used by one backend
is re-planned; slots sharing a placement keep the plan made at load.

**A layer whose experts are split.** Its hot experts are copied once, at load, into memory of their own (`QTensor.
gatherExperts`, in the tensor's own layout, in parallel) and only that copy is given to the GPU. The copy and
the CPU threads belong to the placement, which every backend on the same weights shares (a server's slots), so
they exist once per model, not once per slot. The mapped file stays untouched
for the CPU's cold ones. The router also writes, for every slot, the expert's index in that copy, or −1 for a
cold expert (`gassign[s] = map[e]`). `moe_up`, `moe_down`, `moe_sort` and `moe_combine` skip −1 slots, so the GPU kernels are the same ones
running on the packed tensors. Then the CPU and GPU work **at the same time**:

```
GPU:  … attention, norm, router (+remap) ┃ hot experts (+ shared experts) ┃              ┃ next layer …
CPU:                                     ┃ cold slots: Σ_k w_k·expert_k   ┃ wait, add    ┃
                           commit #1    commit #2, wait for #1 only
```

The CPU waits only for the router's command buffer (`jm_wait_pending(1)`), computes the slots that chose a cold
expert with `cpu/MoeExperts` (other slots marked −1 are skipped), then waits for the GPU's experts and adds its
part to `x`. **When no token chose a cold expert, the CPU adds nothing and does not wait for the GPU at all**, so
that layer costs only the short wait for the router. Layers whose experts are all on the CPU run as before (GPU
stops after the router; the CPU does all slots; DeepSeek's shared experts overlap on the GPU).

**Cost.** Per token, a cold expert that is picked costs the CPU `3 · moeDim · dim` weights:

```
Qwen3-30B-A3B, Q4_0 (16 B of nibbles + a 4 B scale per 32 weights = 0.625 B):  3 · 768 · 2048 · 0.625 B ≈ 2.9 MB
```

That is ≈ 50 µs at 60 GB/s, overlapped with the GPU's hot experts. A whole CPU layer is 8 picks, ≈ 24 MB, so
0.25–0.4 ms. Add the round trip (≈ 0.1 ms) per layer with CPU experts. Paging 24 MB back from an SSD at
~5 GB/s would cost about 5 ms. Prefill is a different story: the GPU's grouped matrix kernels are much faster
than the CPU on long chunks, so long prompts slow down more. All of these numbers are estimates, not yet measured
on a Mac. Pipelined decoding ([8.16](#816-pipelined-greedy-decoding)) is off while any expert runs on the CPU.

**Checked by** `scripts/arch-smoke.sh` on the emulator. Qwen3-MoE and the three DeepSeek models, in five formats
each, run with four placements: one whole layer, all layers, every other expert of every layer on the CPU, and all
but one expert of a layer. Each placement gives the CPU backend's tokens and final logits (within the usual CPU /
Metal tolerance), both for decode (per-slot kernels, single-token CPU path) and for a 20-token chunk (grouped
kernels, batched CPU path). The saved usage holds exactly `topK` picks per processed token in every layer.
`ExpertPlacementCheck` (run by `mvn test`) checks the planner: uniform usage fills whole layers, skewed usage
moves the coldest experts, at least the bytes asked for are freed, and within a layer a moved expert is never
more used than a kept one.

---

## 8.19 Thermal state

🧸 A runner who gets too hot slows down. A Mac does the same: when it is hot, macOS lowers the GPU's clock, and
tokens come out slower without any other sign of trouble.

📐 `jm_thermal_state` reads `NSProcessInfo.thermalState`: 0 nominal, 1 fair, 2 serious, 3 critical. The Metal
backends report it (`Backend.thermalState()`, -1 where nothing can tell, e.g. the CPU backend). At most every 10 s
during generation they log a warning when it reaches serious or critical, and a note when it cools down. The web
console's status panel shows it. Benchmarks taken while the state is serious measure the cooling, not the code;
`scripts/mac-validate.sh` notes it in its table.

## 8.20 Validating on a Mac

The emulator ([18](18-metal-emulator.md)) proves that the kernels compute the right numbers, not how fast they
are. `scripts/mac-validate.sh` does both on a real Mac:
- it runs `scripts/arch-smoke.sh` on the GPU;
- it benchmarks a dense model with each newer feature on (the default) and off: flash prefill, flash decode,
  pipelined decoding, simdgroup-matrix prefill and the f16 cache;
- draft-model speculation (§8.22) with the dense model drafting for itself: off, on, adaptive length off, draft
  pipelining off;
- batched decoding (§8.21): three slots generating together in shared passes against taking turns
  (`BatchDecodeTest bench --backend metal`, which also checks that each slot's answer equals its answer alone);
- optionally (`--moe <repo>`) it benchmarks an MoE model with 256- vs 64-token chunks, grouped prefill on and off,
  experts on the CPU with int8 on and off, all experts on the GPU, and the dense model as its draft.

It prints a markdown table of prefill and decode tokens per second (median of the rounds) for the docs.


---

## 8.21 Batched decoding across slots (`MetalBackend.forwardBatch`)

🧸 Three children each ask the teacher a question about the same big book. Reading the page aloud is the slow part;
writing each child's answer in that child's own notebook is quick. So she reads the page once, and while she reads
she writes three answers, each in its own notebook, each at the line where that child stopped last time.

📐 A server's slots on one model are `MetalBackend`s on the same weights, each with its own KV cache, command queue
and activation buffers ([10 §10.9b](10-engine-and-server.md#109b-batched-decoding-across-slots) for how
`BatchGroup` gathers their decode steps). `forwardBatch(members, tokens, positions, logits)` runs the steps of `n`
slots as one pass on the leader's queue (the member whose thread took the group's device lock first), with the
leader's activation buffers holding the `n` rows:

```
x[t] = embed(tokens[t])                                  t = 0 .. n-1, one row per slot
per layer:
  xb = rmsnorm(x)                                         n rows
  q, kRow, vRow = matmul3(xb)                             n rows: Wq, Wk, Wv read once
  for each row t (its slot s = members[t], p = positions[t]):
      rope_row(q[t], kRow[t] -> s.K[layer][p])            the slot's cache, the slot's position
      kv_put(vRow[t] -> s.V[layer][p])                    (q8: kv_store_q8 of kRow[t], vRow[t])
      att[t] = attention(q[t], s.K[layer][0..p], s.V[layer][0..p])
  x += o_proj(att)                                        n rows
  x += ffn(rmsnorm(x))                                    n rows: dense, or MoE (router + experts), CPU experts too
logits[t] = lm_head(rmsnorm(x))[t]                        n rows into an n-row buffer
```

Decoding is bound by reading the weights (W bytes per pass, [11](11-performance.md)), so with per-row work `r`
(RoPE, the cache store, attention over the slot's context, tiny next to W / bandwidth at moderate context):

```
taking turns:   n · (W / BW + r)
one pass:       W / BW + n · r          speed-up → n while n · r ≪ W / BW
```

**Same buffers, other caches.** The members' cache buffers are bound in the leader's command buffer. All of them were
created on the same `MTLDevice`, so any queue may use them; the group's lock guarantees that no member has work in
flight (`forwardBatch` also drains every member's pipelined steps first, §8.16, which the group only allows while a slot
is alone). The leader's `kStage` is not enough, since every row goes to another cache: the pass writes the rows' k
and v to two f32 buffers `rowK`, `rowV` (`[maxBatch, kvDim]`, made on the first batched pass), and each row's RoPE
reads its k row there and writes the rotated row into its slot's cache at its position (as half for f16; for q8 back
into `rowK`, then `kv_store_q8` quantizes k and v together, as in a single-sequence step). For the f32 and f16 caches v
needs a copy, `kv_put<KV>` (the one kernel added for this: one thread per value, rounding once to half for f16 —
the same `half(v + bias)` that `matmul3<.., half>` writes in a single-sequence step).

**Per-row dispatches, existing kernels.** RoPE, the cache store and attention are the single-sequence kernels,
dispatched once per row with the row's byte offsets (`MetalKernels.ropeRow`, `attentionRow`, `flashAttentionRow`,
`kvStoreQ8Row`, `mlaLatStoreRow`): the rope and attention kernels read one position `pos0` per dispatch and one cache
binding, so a single dispatch cannot cover rows at other positions in other buffers. That is 3 small dispatches per row
and layer (4 with flash decoding), negligible against the weight reads at a few slots. Sliding windows, soft-capping,
Gemma's local/global RoPE tables, partial and interleaved rotation (GLM, Phi) come with the layer's own parameters,
exactly as in a single step. DeepSeek's MLA does the same over its latent cache (`MlaMetal.runRows`: the projections,
the absorption of q and the value up-projection over the n rows; RoPE of q and k_pe, `mla_lat_store` and the attention
per row).

**MoE.** The router and the experts take the n rows like a prefill chunk of n tokens: the per-slot kernels below
`MOE_MMA_MIN` = 16 rows, the expert-grouped ones above. With experts on the CPU (§8.18) the leader's CPU side
computes the n rows' CPU experts at the layer's GPU → CPU → GPU round trip, so the round trip too is paid once per
pass instead of once per slot.

**Exactness.** For fewer than 8 rows every projection runs the `tokTiles` kernels, whose per-row arithmetic is the
single-row kernel's (§8.10), and the per-row kernels are the single-step ones, so a slot's logits in a batched pass are
normally bit-identical to its own step: `ArchSmoke` checks every attention family with three slots at other positions
(a rotating leader, one step where a slot runs on its own in between) for the f32, f16 and q8 caches, and found them
bit-identical on the emulator. With an MoE layer's experts on the CPU they agree to rounding (4e-6 on the tiny Qwen3-MoE):
the CPU computes several rows with its batched expert kernels, one row with its single-token ones. It requires equal tokens and logits
within 10⁻⁴ of the largest, then runs the same slots in a three-thread `BatchGroup` and requires the tokens they give
alone. From 8 rows on the tiled prefill kernels (still bit-identical) and from 32 the simdgroup-matrix ones (rounding
level, §8.10) take over.

**What stays one at a time.** Mamba backends (a recurrent state per slot), and a batch whose members differ in weights
or cache type (`forwardBatch` returns false and the group runs the steps one by one). `-Djllm.metal.batchDecode=false`
turns batched passes off.

---

## 8.22 Draft-model speculation on the GPU

🧸 A junior writer drafts the next few words, the senior editor reads the whole draft at once and keeps the words
she agrees with. The junior writes quickly when nobody makes him stop after every word to show it, and the editor
decides how long a draft to ask for from how often she agreed lately and how long each of them takes.

📐 `engine.speculative=draft` ([10 §10.9](10-engine-and-server.md#109-speculative-decoding)) loads
`engine.draft-model` on the same backend: on Metal a second `MetalBackend` on the same device, loaded before the
target so `DeviceLedger` counts its weights and cache in the target's memory plan (§8.18). Per pass:

```
draft:   feed the history after the prefix its cache already holds (chunks, no logits), then k pipelined steps:
         step 0 feeds the last history token, step j feeds the argmax of step j-1 (on the GPU), each step queued
         before the CPU reads the one before it; only k token ids come back
target:  one forwardAll([tok, d1..dk]) pass: k+1 rows through the multi-token kernels, k+1 rows of logits
accept:  the longest prefix the target agrees with, plus the target's own next token
```

**Pipelined draft.** The draft uses the greedy pipe of §8.16 (`DraftModel.propose`): on the GPU `argmax` writes each
step's token into the ring and the next step's `embed_token` reads it, so the draft's k steps run back to back without a
GPU → CPU → GPU round trip and without copying a vocabulary of logits per step (600 KB for a 150k vocabulary). A stop
token ends the draft early (the pipe is drained). Without a pipe (CPU backend, experts on the CPU) the draft returns
logits per step as before; `-Djllm.engine.draftPipeline=false` forces that.

**Roll-back costs nothing.** A KV cache is addressed by position: after a pass that accepted `a` of `k` drafts the
target continues at `pos + 1 + a`, and the rows after that are simply overwritten by the next pass before anything
attends to them (attention at position `p` reads rows `0..p` only). The draft keeps the list of tokens its cache
holds and on the next proposal re-feeds only what follows the longest common prefix with the new history (the accepted
drafts stay, a rejected one is overwritten by the correction). Every cache row is written whole for its position —
f32, half, or q8's int8 values with their per-row scales — so this holds for all three cache types; `ArchSmoke` checks
greedy output with and without speculation for each of them.

**Adaptive draft length.** With per-token acceptance probability α, a pass with k drafts yields
`E(k) = 1 + α + … + α^k` tokens and costs `k·d + V(k+1)`, where d is a draft step and `V(n) ≈ a + b(n−1)` a target pass
over n positions. `Speculator.CostModel` measures all of them while running (exponentially weighted: α from the
accepted and examined positions, d from the draft's time per step, a and b by least squares over the verification
passes, and a plain target step p from the steps the engine takes without a draft) and picks

```
k* = argmax_k  E(k) / (k·d + a + b·k)         if that beats 1 / p, else pause for 16 plain steps
```

(one in four pauses runs a pass anyway, so a rise in acceptance is noticed). Until two passes are measured the
doubling rule of prompt lookup decides; `-Djllm.engine.adaptiveDraft=false` keeps it. Worked example with the costs of
[11 §11.6](11-performance.md#116-speculative-decoding) for a 30B MoE target on an M1 Pro (a = 39 ms, b = 0.55 a,
a 0.6B draft step d ≈ 8 ms): at α = 0.6, k = 1 gives 1.6 / (8 + 39.2 + 21.6) ms = 23.3 tok/s against 25.5 tok/s plain
and longer drafts do worse, so speculation pauses; at α = 0.9, k = 1 gives 1.9 / 68.8 ms = 27.6 tok/s (k = 2: 27.6,
k = 3: 26.9) and wins by 8 %. The pass cost b is the lever: the multi-token kernels' cost per extra row decides whether
a draft can pay; at b = 0.1 a, α = 0.9 would choose k = 5 for 1.86x the plain rate.

Emulator timings are not representative of a GPU (the emulator runs every GPU thread of a kernel as a fiber on the
CPU), so the costs on a real Mac come from `scripts/mac-validate.sh` (§8.20).
