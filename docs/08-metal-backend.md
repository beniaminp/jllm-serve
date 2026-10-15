# 8. The Metal backend: GPU compute through Panama FFM

Code: `src/main/native/jllm_metal.m` (Objective-C), `src/main/resources/.../kernels.metal` (GPU code),
`metal/Metal.java` (bindings), `metal/MetalBackend.java` (the forward pass).

## GPU basics

A GPU runs thousands of tiny threads in lock-step groups. In Metal:

- a **kernel** is a function run once per thread; threads are organized in **threadgroups** (up to 1024) that can
  share fast on-chip `threadgroup` memory and synchronize with barriers;
- a **simdgroup** is 32 threads executing the same instruction; `simd_sum(v)` adds a value across the 32 lanes in
  one step — the natural way to finish a dot product;
- a **dispatch** launches a grid of threadgroups; dispatches recorded into one **command buffer** run back to back
  on the GPU with no CPU involvement in between.

Apple Silicon has **unified memory**: CPU and GPU address the same RAM. A `MTLBuffer` in shared mode is just a
pointer both can use, which is why our quantized weights are never copied to "the GPU".

MLX, Apple's ML framework, is a C++/Metal library sitting on exactly this API. We skip the framework and talk
to Metal directly; the shim is ~200 lines.

## Layering

```
MetalBackend.java    knows the model: binds weight buffers, computes grid sizes, records ~12 dispatches per layer
      │  java.lang.foreign downcalls
Metal.java           generic: compile source, make/wrap buffers, begin / dispatch / commit
      │  C ABI (jm_* functions)
jllm_metal.m         Objective-C: MTLDevice, MTLLibrary, MTLComputePipelineState, MTLCommandBuffer
      │
kernels.metal        GPU code, compiled at startup by newLibraryWithSource (no Xcode step for the shaders)
```

### The Objective-C shim (`jllm_metal.m`)

Metal is an Objective-C/Swift API, so *something* native must exist. We keep it minimal and model-agnostic:

| C function | Does |
|---|---|
| `jm_create(source)` | pick the default GPU, create a command queue, compile the shader source |
| `jm_pipeline(name)` | look up a kernel function and build a pipeline state; returns an index |
| `jm_buffer_new(len)` | allocate a zero-filled shared buffer |
| `jm_buffer_wrap(ptr, len)` | `newBufferWithBytesNoCopy`: wrap Java-owned page-aligned memory, zero copy |
| `jm_buffer_contents(buf)` | CPU pointer to the buffer so Java can read/write it through a `MemorySegment` |
| `jm_begin()` | start a command buffer with a *serial* compute encoder (each dispatch sees the previous one's writes) |
| `jm_dispatch(pipeline, bufs, offsets, params, grid, threadgroup)` | record one kernel launch |
| `jm_commit_wait()` | submit and block until done; report GPU errors |

Built with one `clang` command (`scripts/build-native.sh`) into `target/native/libjllm_metal.dylib`.

### The bindings (`Metal.java`)

The **Foreign Function & Memory API** (`java.lang.foreign`, "Panama") replaces JNI. For each C function we create
a `MethodHandle` once:

```java
SymbolLookup lib = SymbolLookup.libraryLookup(path, arena);
MethodHandle dispatch = Linker.nativeLinker().downcallHandle(lib.findOrThrow("jm_dispatch"),
    FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
                              JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));
...
dispatch.invokeExact(ctx, pipe, bufs, offsets, nbuf, params, plen, gx, gy, gz, tx, ty, tz);
```

`MemorySegment` is the pointer type; `arena.allocateFrom(String)` makes a C string; `reinterpret(size)` turns a
raw pointer returned by C into a bounded segment Java can read. No generated glue code, no header files. The JVM
needs `--enable-native-access=ALL-UNNAMED` (otherwise it warns).

### Zero-copy weights

`QTensor` allocates every matrix 16 KiB-aligned and 16 KiB-sized ([06](06-quantization.md)). `Metal.wrap(segment)`
passes that pointer to `newBufferWithBytesNoCopy`, so the GPU reads the same bytes the CPU backend would. The
quantized blocks and the scales are bound as two buffers at different offsets of the same `MTLBuffer`.

## The forward pass as dispatches (`MetalBackend.forward`)

1. Dequantize the embedding rows of the batch on the CPU (tiny) straight into the shared `x` buffer.
2. `metal.begin()`.
3. For each of the 28 layers record: `rmsnorm`, `matmul_q4` ×3 (q, k, v — k and v written directly into the KV
   cache buffers at offset `pos × kvDim × 4`), `norm_rope` ×2, `attention`, `matmul_q4` (o, accumulating into
   `x`), `rmsnorm`, `matmul_q4` (gate/up), `silu_mul`, `matmul_q4` (down, accumulating).
4. Final `rmsnorm` on the last token and `matmul_q8` with the LM head.
5. `metal.commitAndWait()`, then copy the 151 936 logits out of the shared buffer.

One command buffer, one CPU↔GPU round trip per forward call, ~340 dispatches. Small parameter structs
(`MatmulParams`, …) are passed with `setBytes`, so the Java side just writes ints/floats into a 64-byte scratch
segment before each dispatch.

## The kernels (`kernels.metal`)

### `matmul_q4` / `matmul_q8` — `out[t, row] = Σ_c W[row, c] · x[t, c]`

One **simdgroup per output row**. Lane `l` handles blocks `l, l+32, l+64, …` of the row: it loads 16 bytes,
unpacks nibbles to two `float4[4]`, and takes dot products with the matching 32 floats of `x` using `dot()` on
`float4`. The lane results are combined with `simd_sum`. Up to `TOK_TILE = 8` tokens share each unpacked block
(the `acc[8]` array), which is how prefill gets reuse. Grid: `(rows / 4 simdgroups, ntok / 8)`.

### `rmsnorm` — one threadgroup per row

Each thread sums squares of a strided subset; `tg_sum` combines via `simd_sum` + threadgroup memory; then every
thread scales its elements. Used for the two per-layer norms and the final norm.

### `norm_rope` — per-head QK-norm + rotation

Grid `(heads, ntok)`, 64 threads per group (one per rotary pair `(i, i+64)`). Each thread normalizes its pair
(threadgroup reduction for the RMS when `useNorm`) and applies the rotation with the precomputed `cos`/`sin`
tables.

### `attention` — one threadgroup (256 threads = 8 simdgroups) per `(head, token)`

1. Simdgroups split the cached positions; within a simdgroup each lane owns 4 of the 128 head dimensions, so a
   `q·k` dot product is 4 multiplies per lane plus one `simd_sum`. Scores go to a device scratch buffer
   (`scores`, sized `batch × heads × ctx`) so context length is not limited by threadgroup memory.
2. Threadgroup max, then `exp(score − max)` and the sum.
3. Each simdgroup accumulates `w_s · v_s` for its positions into registers, partials are reduced through
   threadgroup memory, and the result is written to `att`.

### `silu_mul` — element-wise `silu(gate) · up`

Plain one-thread-per-element kernel.

## Metal gotchas met along the way

- All "position" attributes of a kernel must have the same dimensionality (`uint2 tgp` with `ushort tid` is a
  compile error, hence `ushort2 tid2` and `.x`).
- Parameter structs must contain only 4-byte members in the order Java writes them.
- `newBufferWithBytesNoCopy` silently requires page alignment; otherwise the shim falls back to a copy.
- Each GPU dispatch costs a few microseconds of GPU front-end time; at ~340 dispatches per token that is a
  visible fraction of the 13 ms decode step. Fusing kernels is the main remaining optimization.

## Measured

Apple M1 Pro: ~344 tok/s prefill, ~75 tok/s decode with a 413-token context (~110 tok/s on short contexts).
Greedy output is byte-identical to the CPU backend.
