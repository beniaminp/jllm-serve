# 30. What it would take to rewrite jllm-serve in Rust

This page estimates the work needed to build a functionally identical jllm-serve in Rust instead of Java. It covers
what exists today, how each Java mechanism maps to Rust, which parts carry over unchanged, which parts are hard,
how to prove the port matches the original, and the effort involved.

> Method: six independent investigations, one per area (CPU backend, Metal backend, model loading / GGUF /
> tokenizer, engine / KV / server / sampling, vision / audio / diffusion, and web console / tests / build). Each
> one read the code and docs of its area, and the line counts and the specific claims about the code were then
> checked against the source. The effort numbers are estimates, not measurements; nothing was built or prototyped.
> File paths are relative to `src/main/java/org/robomiri/jllm/` unless shown otherwise.

---

## 30.1 Summary

| | |
|---|---|
| Java to port | ~73k lines of engine code, ~29k lines of test/bench programs (94 files, 81 with a `main`), ~9k lines of jllm-web Java plus ~3k of its tests; ~114k in all. Also ~1.3k lines of C and 575 of Objective-C |
| Reusable as is | ~7.9k lines of Metal shaders (`kernels.metal`, `diffusion.metal`, `vision.metal`, `audio.metal`), the Metal emulator (`scripts/metal-emu/`), all Python reference scripts and checked-in fixtures, the web console's static JS/CSS/HTML (~8.6k lines), the docs |
| Expected Rust size | ~0.8–1.0× the Java line count. Large Java-only fallbacks disappear, while unsafe and SIMD code gets longer |
| Effort | **~130–150 engineer-weeks** for full parity, or about 2.5–3 person-years (bottom-up; a line-rate cross-check in §30.9 gives ~90–120 as the optimistic end). A team of 4 would need **~9–12 months**, with Apple Silicon hardware available |
| Reduced scope | ~75–85 engineer-weeks for the LLM server only (CPU + Metal, GGUF + safetensors, OpenAI API, `./jllm serve`), without diffusion/video/audio/vision and without porting the Spring console |
| Hardest parts | (1) the unsafe boundary where kernels and the GPU write KV blocks while the pool's refcount protocol keeps them safe; (2) tokenizer regex/Unicode parity; (3) the volume of quant formats × kernels × model families; (4) scheduler timing semantics; (5) Metal host code that must match shader parameter layouts exactly; (6) a `Backend` interface whose batched passes hand one backend the caches of several others (§30.4) |
| What gets better | no JIT warm-up (Vector API runs 500–1000× slower before C2 compiles it), no FFM/JNI layer, a single static binary instead of a JDK, AVX-512 VNNI / dotprod usable directly in the language, predictable memory without `-Xmx` / `MaxDirectMemorySize` tuning, core pinning |
| What gets worse | stable Rust has no portable SIMD, so x86 and NEON kernels are written twice. Unsafe code appears in kernels, the thread pool, mmap→`MTLBuffer` and KV storage. The JDK's TLS, HTTP client, Unicode normalization, regex and ImageIO were free, and need crates or hand-written code in Rust |

"Identical" should mean the **same features, the same CLI and config keys, the same HTTP/JSON/SSE wire contract,
the same on-disk formats, and the same greedy token ids**. Bit-identical logits are not a realistic target, and
the Java engine does not guarantee them either: results already depend on prefill versus decode path, SIMD lane
width and reduction order (docs/07 §7.14).

---

## 30.2 The project as it is

| Area | Java packages | Lines | Notable |
|---|---|---|---|
| CPU backend | `cpu/` + `src/main/native/jllm_cpu*.c/.inc` | 11.7k + 1.3k C | Vector API kernels for 14 quant types, int8 activations, custom spin/park thread pool, f32/f16/q8/q4 KV caches, paged + sparse attention, dense / MoE / MLA / Mamba / Qwen3-Next / hybrid CPU+GPU backends |
| Metal backend | `metal/` + `jllm_metal.m` + `.metal` | 10.4k + 575 ObjC + 7.9k MSL | generic C ABI over Metal, decode replay (~170 dispatches recorded and patched), pipelined greedy decode, concurrent encoder with a reflection-based hazard tracker, `MetalTuner`, memory planner and expert placement |
| Models and formats | `model/`, `gguf/`, `safetensors/`, `hub/`, `ModelLoader`, `ContextBudget` | ~14k | 20+ HF `model_type` families as a sealed `Architecture` hierarchy, a quantized cache file (`QuantFile`), safetensors→Q4_0/Q8_0/Q2_K converter, GGUF v2/v3 read in place, a resumable SHA-256-verified HF downloader |
| Tokenizer | `tokenizer/` | 4.3k | byte-level BPE, SentencePiece (greedy + Unigram), normalizers / pre-tokenizers, GGUF vocabularies with llama.cpp pre-types, ~20 hand-written chat templates (no Jinja) |
| Engine | `engine/`, `kv/`, `sampling/` | ~9.2k | slots, `SlotPool`, mixed-pass scheduler (`Admission`, `BatchGroup`, `PassPlanner`), speculation (n-gram, draft model, MTP), paged KV with a radix prefix tree, refcounts, COW, tiered spill to RAM/SSD, JSON-schema and tool-call constrained decoding |
| Server, config | `server/`, `json/`, `config/`, `Main` | ~3.7k | hand-written HTTP/1.1 on virtual threads, OpenAI API + SSE, own JSON, layered `.properties` config with ~40 aliases |
| Multimodal | `vision/`, `audio/`, `diffusion/` | ~19.6k | CLIP/SigLIP projectors (SmolVLM, Gemma 3, Qwen2/2.5/3-VL), Whisper + word timestamps, OuteTTS + WavTokenizer, SD 1.x/2.x/SDXL, AnimateDiff, Wan 2.1 parts, hand-written VP8L WebP and GIF encoders, ffmpeg for MP4 and non-WAV audio |
| Web console | `jllm-web/` | ~9k Java + 8.6k JS/CSS | Spring Boot, **in-process** use of `SlotPool`, `ModelLoader`, `Engine`, etc.; 9 tabs, model management, KV save/restore, MCP client, integration configs |
| Tests | `src/test/` | ~29k | 94 files, 81 of them `main()` programs, run through 38 `exec-maven-plugin` executions (no JUnit in the engine); 11 MB of fixtures, parity scripts against llama.cpp / HF / whisper.cpp / diffusers, the Metal emulator |

The project's defining ethos is **zero external dependencies**: only the JDK. A Rust port has to decide early how
strictly to keep that (§30.5).

---

## 30.3 Java mechanism → Rust mapping

| Java | Used for | Rust |
|---|---|---|
| `jdk.incubator.vector` (`SPECIES_PREFERRED`, ~300 load sites, fma, `reduceLanes`) | all CPU kernels | `std::arch` intrinsics under `#[target_feature]`, one implementation per dispatch level (scalar / NEON / dotprod / AVX2 / AVX-VNNI / AVX-512 VNNI). `pulp` or `wide` can multiversion the float kernels. `std::simd` is nightly-only, so it is ruled out |
| JIT-stability workarounds (fixed-order reductions, magic-constant int→float, one call site per helper) | bit-stable results under C1/C2 | Mostly unnecessary. Keep the fixed reduction order where results must be stable, and use `f32::round_ties_even` for `rint` |
| `libjllm_cpu` via FFM `Linker` + self-check (`NativeCpu`, 610 lines) | int8 kernels | Removed. The C kernels become Rust intrinsics (~1.3k C → ~1.5–2k Rust) and the self-check becomes `cargo test`. CPU detection uses `is_x86_feature_detected!` / `is_aarch64_feature_detected!`. The AVX-512 and AVX-VNNI intrinsics are stable from Rust 1.89, which sets the minimum toolchain |
| `FileChannel.map` → `MemorySegment`, 16 KiB alignment | weights in place (GGUF, safetensors, cache) | `memmap2::Mmap` inside an `Arc`, with offsets rather than borrowed slices (avoids self-referential structs). `bytemuck`/`zerocopy` + `read_unaligned` for block structs |
| `Arena.ofAuto()` off-heap slabs | KV storage | a 64-byte-aligned allocation owned by the pool, freed on `Drop`. Unsafe views by block id |
| `ThreadPool` (packed `AtomicLong`, spin 200 µs then park) | hundreds of tiny parallel loops per token | 1:1 port with std atomics, `thread::park`, `hint::spin_loop`. **Not rayon**, whose per-job overhead is too high for this. Optionally add `core_affinity` |
| Virtual threads per connection | HTTP | Either OS threads capped by `server.max-connections` (faithful), or tokio + axum with the engine on dedicated threads and an mpsc channel for SSE (pragmatic). The engine core stays synchronous either way |
| `synchronized`, wait/notify, fair `ReentrantLock.tryLock(timeout)`, volatiles, `ThreadLocal` | KvPool, Admission, BatchGroup, Engine | `parking_lot::{Mutex, Condvar, FairMutex}` (std has no timed fair lock), atomics, `ArcSwap` for whole-object volatile swaps |
| GC-managed object graphs (trie with parent pointers, `BatchGroup.this`, callbacks) | KV prefix tree, scheduler | arena `Vec<Node>` + generational ids. `Arc`/`Weak` or id indirection for back-references. RAII guards for `Slot.close()` |
| Sealed interfaces + records | `Architecture`, `LayerSpec` | `enum Architecture { Decoder, Mamba, Qwen3Next, Hybrid }` + structs. `match model_type` in `Families` |
| `Float.float16ToFloat`, `Math.fma` | f16/bf16 | the `half` crate, or ~20 lines by hand. `f32::mul_add` only inside `fma`-enabled functions, otherwise it becomes a slow libm call |
| `java.util.regex` with `UNICODE_CHARACTER_CLASS` and lookahead (`\s+(?!\S)`) | pre-tokenizers | `regex` has no lookaround. Use either `fancy-regex` or hand-coded splitters for the GPT-2 / Qwen / Llama 3 patterns plus Unicode category tables |
| `java.text.Normalizer` | NFC/NFKC | `unicode-normalization` |
| `java.net.http.HttpClient` (manual redirects so `HF_TOKEN` never reaches the CDN, Range resume) | Hub downloader | `ureq` with redirects disabled, plus `sha2`. TLS via `rustls` (+ `rustls-native-certs` / `SSL_CERT_FILE` instead of keytool) |
| `javax.imageio` | PNG/JPEG/GIF/BMP decode, PNG/JPEG encode | `png` + `zune-jpeg` (or `image`). JPEG pixels will differ from ImageIO by a few LSB |
| `ProcessBuilder` | ffmpeg | `std::process::Command` (1:1) |
| Panama downcalls into `jllm_metal.m` | Metal | Phase 1: keep the ObjC shim, built by the `cc` crate in `build.rs`, with 22 `extern "C"` functions and a `#[repr(C)] JMCmd` (384 bytes, `_Static_assert` kept). Phase 2: replace it with `objc2-metal`. Avoid the deprecated `metal` crate. Reject wgpu (no simdgroup matrices, no zero-copy mmap buffers, every shader rewritten in WGSL) |
| `.metal` loaded and compiled at runtime with `#define` prefixes | shader variants | `include_str!` + `newLibraryWithSource`, unchanged. Offline metallibs are optional later |
| `-D` system properties / `.properties` / CLI | config | a hand-ported `Settings` (~500 lines: aliases, source tracking, typed validation, Java `.properties` escaping), with the bundled `jllm.properties` via `include_str!`. clap does not fit the arbitrary-dotted-key model |

---

## 30.4 The `Backend` interface and who owns what

`model/Backend.java` (373 lines, about 35 methods) is the seam between the engine and the two backends, so its Rust
shape has to be settled before anything else is ported. Most of it translates directly:

```rust
pub trait Backend: Send {
    fn name(&self) -> &str;
    fn context_length(&self) -> usize;
    fn max_batch(&self) -> usize;
    fn forward(&mut self, tokens: &[u32], pos: usize, logits: &mut [f32]) -> Result<()>;
    fn forward_all(&mut self, tokens: &[u32], pos: usize, logits: &mut [f32], stride: usize) -> Result<()>;
    fn forward_embeds(&mut self, embeds: &[f32], pos: usize, deepstack: Option<&[&[f32]]>, logits: &mut [f32]) -> Result<()>;
    // capabilities: supports_embeds, supports_prefix_reuse, supports_rollback, has_mtp, thermal_state ...
    // state: checkpoint_state / rollback_state, save_state / load_state (Write / Read), snapshot_state, copy_state_from
    // kv_sequence(), memory_report(), greedy_pipe()
}
```

Java's `(int[] tokens, int off, int n)` triples become slices, the optional methods with `supports…()` guards stay
as default methods, and `AutoCloseable` becomes `Drop`.

Three methods do not translate directly, because they pass one backend the others:

| Java | What it does | Problem in Rust |
|---|---|---|
| `forwardBatch(Backend[] members, …)` | one decode step for several slots; `this` lends its buffers and is also one of `members` | `&mut self` and `&mut members[i]` alias |
| `forwardMixed(Segment[] segments)` with `Segment(Backend owner, …)` | a prefill chunk plus other slots' decode tokens in one pass, each segment in its owner's cache | the same, across a list |
| `copyStateFrom(Backend other, int n)` | prefix copy between slots | needs two slots borrowed at once |

The clean fix is to split what Java keeps in one object:

- a shared, immutable **`Model`** (`Arc`): weights, architecture, the thread pool or GPU device, and the pass
  logic;
- a per-slot **`SlotState`**: the KV sequence or recurrent state, RoPE positions, scratch buffers.

The batched entry points then live on the model and take the slots they touch:
`model.forward_mixed(&mut [Segment<'_>])`, where each segment holds `&mut SlotState`. Distinct slots are distinct
borrows, and the "owners must be distinct, idle backends" rule in the Javadoc becomes something the compiler
checks. `BatchGroup` already serializes passes with its device lock, so no new locking is needed. This is a
restructuring, not a line-by-line translation, and both backends and the scheduler depend on it.

---

## 30.5 The dependency decision

The Java project depends only on the JDK, but the JDK supplies a lot: TLS, an HTTP client, regex, Unicode
normalization, image codecs and SHA-256. A Rust port has to settle this first. The recommended policy is
**"zero dependencies in the compute path, a small vetted set at the edges"**:

| Keep hand-written (as in Java) | Use a crate |
|---|---|
| all kernels, quant formats, GGUF/safetensors readers, the quant cache file, KV pool, scheduler, sampler, constrained decoding, tokenizers, chat templates, VP8L/GIF encoders, WAV codec, thread pool, config | `memmap2`, `half`, `parking_lot`, `rustls` + `ureq`, `sha2`, `unicode-normalization`, `fancy-regex` (or hand splitters), `png` + `zune-jpeg`, `objc2-metal` (phase 2). For the server, either hand-written HTTP or `axum`/`tokio`. `serde_json` with `preserve_order` or a ported `Json` |

Use the `tokenizers` crate **only as a dev-dependency test oracle**. It does not cover GGUF vocabularies,
llama.cpp pre-types, protected special tokens or the piece cache, so the from-scratch tokenizer has to be ported
anyway.

---

## 30.6 Java behaviour that has to be reproduced on purpose

A line-by-line translation compiles and then differs in small ways. These are the places found in the code where
Java's semantics leak into results, files or the wire format:

| Java behaviour | Where it matters | In Rust |
|---|---|---|
| `int`/`long` arithmetic wraps silently | hashes (the KV path hash seeded with `0x2545F4914F6CDD1D`, the `GOLDEN` multiplier in `diffusion/Rng.java`), bit packing in the quant formats | `wrapping_mul` / `wrapping_add` explicitly; a plain `*` panics in debug builds and wraps in release, so tests and production would differ |
| `byte` is signed, `>>>` vs `>>`, `char` is 16 bits | quant block decoding, tokenizer byte handling | `i8` vs `u8` chosen per field; most `& 0xFF` masks disappear, which is where translation slips happen |
| `String.hashCode()` in persisted names | `metal/ExpertPlacement.java` (routing-count file name), `metal/Metal.java` (`sourceHash` that keys the `MetalTuner` cache) | reimplement the 31-polynomial over UTF-16 units, or accept that these caches are rebuilt once |
| `SplittableRandom(seed).nextDouble()` | `sampling/Sampler.java`: a request's `seed` decides the sampled tokens | port SplitMix64 and Java's double conversion (about 20 lines) if the same seed must give the same text; no crate reproduces it |
| `StringBuilder.append(Number)` for JSON numbers | `json/Json.java` writes doubles with `Double.toString` (`1.0E-5`, `100.0`) | Rust prints `0.00001` and `100`. Clients parse both, but recorded-response tests need a number-aware comparison or a port of the Java formatting |
| Strings are UTF-16 | tokenizer offsets, stop strings, `StreamDecoder` | Rust is UTF-8, which suits the byte-level constraint code better; anything that counts or indexes `char`s must be re-derived, not copied |
| `HashMap` iteration order is stable for the same keys; `LinkedHashMap` keeps insertion order | JSON key order, and any place a map is iterated to pick a winner or build a list (to be audited per class during the port) | Rust's `HashMap` order is random per process. Use `IndexMap`/`BTreeMap` or an explicit sort wherever order reaches an output |
| `WeakHashMap` caches keyed by object identity | `ModelLoader`, `CpuMatmul`, `KvPools`, `TokenTrie`, `JsonConstraint` (per-tokenizer byte tables) | no equivalent. Move each cache into the object it is keyed on (the model or tokenizer owns its tables), which is simpler |
| Exceptions for control flow | `StaleCacheException` (delete and reconvert), `OutOfMemoryError` caught in `diffusion/MetalOps.java` to free scratch and retry, client-gone exceptions | `Result` with a typed error enum. Allocation failure cannot be caught as a panic, so the Metal retry must use a fallible allocation path |
| `Thread.interrupt()` (26 call sites) | waking waiters in `Admission`, `BatchGroup`, the HTTP probe | no interrupt in Rust. A shared cancel flag plus `Condvar::notify`, checked at the same points |
| `Math.exp` / `Math.tanh` / `Math.pow` | sampler softmax, schedulers, mel filters | Rust's libm results can differ in the last bit. Harmless for tokens in practice, but it rules out bit-identical float outputs outside the hand-written fast-math polynomials |
| `float` → `int` casts saturate, NaN → 0 | quantizers | Rust's `as` behaves the same since 1.45; no action, but do not replace with `to_int_unchecked` |
| Arrays are indexed by `int`; the paged cache was audited for 2³¹ (docs/24) | KV indices | `usize` everywhere removes the audit, but file formats keep their 32/64-bit field widths |
| Recursion on the JVM stack | `JsonConstraint` schema references (`$ref` recursion allowed) | Rust's default thread stack is smaller on spawned threads (2 MiB); set the stack size for request threads or keep the depth bounded |
| Heap `float[]` passed to native code with `Linker.Option.critical` | CPU kernels | disappears: everything is a slice in one address space |

---

## 30.7 Area by area

### 30.7.1 CPU backend (~18–22 weeks)

- Port order: thread pool, mmap and aligned allocation, CPU dispatch (1 w). Then the int8 kernels and activation
  quantizer, ported from `jllm_cpu.c` (3–4 w). Then the float/GGUF kernels with fast math (`expFast`/`tanhFast`
  polynomials verbatim), norms and RoPE (3 w). Then the KV caches and tiled attention (3 w), and paged and sparse
  attention (2 w). After that, the backend wiring for dense, MoE, MLA, Mamba, Qwen3-Next and hybrid (4–5 w).
  Last, parity and performance tuning (2–3 w).
- The Java side keeps two kernel sets: Vector API and native C. Rust needs only one. About 3.5k lines of
  Java-only fallback and the JIT workarounds disappear.
- The cost is volume, not depth: 14 quant types × {float dot, int8 dot, prefill tile} × {NEON, dotprod, AVX2,
  VNNI, AVX-512} plus scalar references.
- Rust never contracts FMAs, which matches `-ffp-contract=off` in `build-native-cpu.sh`. The int8 paths are exact
  integer arithmetic per sub-block, so only the final float summation order needs matching.

### 30.7.2 Metal backend (~20–23 weeks)

- **The shaders are reused verbatim at zero cost.** This alone saves 12–16 weeks against a shader rewrite.
- Port order: FFI/device layer and `build.rs` (1.5 w). Then `Metal`, `Params`, `KernelParams`, `DecodeProgram`
  and `MetalKernels` (4 w). Then `MetalBackend`: prefill, decode, pipelined decode, GPU sampling (4–5 w). Then the
  arch variants (3 w). Then the tuner, memory planner/governor/ledger, expert placement, `StreamedKv` and
  `MetalKvStorage` (3 w). Then emulator integration (2 w), and parity on real hardware (3–4 w). Rewriting the
  shim with `objc2-metal` adds ~2 w.
- Kernel parameter structs become `#[repr(C)]` types instead of hand-packed bytes. That is an improvement, but a
  grid, threadgroup or offset mismatch with the shader still fails silently. Generate the structs from one source,
  or test every kernel through the emulator.
- Hard spots: zero-copy mmap→`newBufferWithBytesNoCopy` (page alignment, offset arithmetic, the mapping must
  outlive every buffer). Also the concurrent-encoder hazard tracker; test it against mode 2 ("barrier before every
  dispatch") as the oracle.
- Keep a `trait GpuDevice` with two implementations, `MetalDevice` and `EmuDevice`. The emulator
  (`scripts/metal-emu/runtime.cpp`) implements the same `jm_*` ABI, so Linux CI keeps running the unchanged
  Metal host path.

### 30.7.3 Models, GGUF, safetensors, tokenizer, hub (~21–23 weeks)

| Piece | Weeks |
|---|---|
| safetensors, JSON, `QType`/`Ggml`/`Q2K`/`Q6K`, `QTensor`, `QuantFile` | 3 |
| `Architecture`, `Families`, hybrid / Qwen3-Next / Mamba parsers | 3 |
| `ModelConverter` (FP8, MLX, expert stacking, norm folds), `ModelWeights` | 3 |
| GGUF reader, families, quirks (q/k un-permute, tiled value heads, grouped SSM norm) | 3 |
| Tokenizer incl. regex/Unicode solution and differential tests | 5–6 |
| Chat templates (~20 formats) | 1.5–2 |
| Hub downloader | 1.5 |
| `ModelLoader`, `ContextBudget`, glue | 1.5 |

- **Keep the quant cache byte-compatible.** It is raw little-endian blobs at 16 KiB offsets plus a JSON index:
  `version=1`, `layout=2`, and the canonical `model` description string, which is compared on resume. The index
  is written last and atomically as the completion marker. If the quantizer's rounding matches, Java and Rust can
  then share one cache.
- Tokenizer parity is the largest correctness risk in this area. Check it by differential fuzzing on multilingual
  text against the Java build and HF `tokenizers`. Edge cases include added-token lstrip/rstrip, byte fallback,
  Metaspace schemes, Unigram tie-breaking and partial-UTF-8 stream decoding.
- Keep the chat templates hand-written. minijinja would only matter for a new feature: rendering arbitrary
  `tokenizer_config.json` templates.
- `ContextBudget` reads free JVM heap today. In Rust it reads `/proc/meminfo` and the cgroup v2 limits, `sysctl`
  on macOS, and Metal's `recommendedMaxWorkingSetSize`.

### 30.7.4 Engine, KV, scheduler, sampling, server, config (~24–28 weeks)

| Piece | Weeks |
|---|---|
| json, config, `Main` | 2 |
| HTTP server, `ApiServer`, `OpenAiFeatures`, `WebUi` | 3.5 (axum ~2.5) |
| Sampling incl. `JsonConstraint`, `ToolCallConstraint`, `ByteConstraint` | 4 |
| KV core: pool, sequence, trie arena, unsafe block storage | 3 |
| KV spill / tier / codec | 1.5 |
| Engine core: `Engine`, `SlotPool`, cache files, media prompts | 3.5 |
| Scheduler: `Admission`, `BatchGroup`, `PassPlanner` | 3 |
| Speculation (n-gram, draft, MTP with state checkpoint) | 2 |
| Media route glue | 1 |
| Integration, parity, concurrency stress | 3 |

Where Rust's ownership model pushes back:

1. **KV rows are written by kernels and the GPU outside the pool lock.** In Java this is safe through a runtime
   protocol: a block is referenced, so it is not evictable, and it is either private or published-immutable. In
   Rust it becomes an unsafe store addressed by block id that encodes the same invariant. Audit it, and test it
   with loom on the pool logic plus stress tests. This is the single most important piece to get right.
2. The prefix trie (parent pointers, child maps, LRU eviction toward the leaves) becomes an arena with
   generational ids.
3. Back-references and callbacks (`Reclaimer`, `TableSink`, `BatchGroup.Member`) become `Weak` references or ids.
4. Admission and `BatchGroup` (wait/notify, interrupt, aging, preemption, `parkNanos` spinning) port to
   `Condvar`, but they are timing-sensitive. Build a deterministic-clock harness and use
   `SchedulerPolicyTest` / `MixedSchedulerTest` as the specification.

Define the `Backend` trait and the model / slot-state split of §30.4 first. The speculator, scheduler and engine
all depend on `forwardAll`, `checkpointState`, `supportsPrefixReuse` and the mixed-pass entry points, and the CPU
and Metal ports need it as their target.

Wire fidelity covers SSE framing (`data: [DONE]`), error shapes and status codes, `usage.prompt_tokens_details`,
stop strings across partial UTF-8, JSON number formatting and seeded sampling (both in §30.6).

### 30.7.5 Vision, audio, diffusion, video (~24–30 weeks)

| Piece | Weeks |
|---|---|
| Vision (4 projector families, llama.cpp-exact preprocessing, CPU + Metal, M-RoPE prompt injection) | 5–6 |
| Audio (Whisper CPU + Metal, DTW word alignment, OuteTTS + WavTokenizer, WAV/AIFF/AU, ffmpeg) | 6–7 |
| Image diffusion (SD 1/2/XL, schedulers, pipelines, torch-exact `Rng`) | 7–8 |
| AnimateDiff and encoders (VP8L WebP, GIF, MP4 via ffmpeg) | 3 |
| Wan 2.1 (unverified end to end in Java: defer or put behind a flag) | 3–4 |
| Metal host glue for the three shader files | 2–3 |
| Parity harness | 2–3 |

- These modules should share the CPU kernel crate and the Metal device with the LLM path. `Ops` becomes a trait
  with CPU and Metal implementations. The new ops on top of the LLM set are conv2d (im2col + GEMM), GroupNorm,
  tanh-GELU and adaptive modulation.
- Port the hand-written VP8L and GIF encoders exactly, since they are pure bit manipulation. Crates would produce
  different bytes.
- ImageIO is the one JDK facility without a hand-written equivalent. Rust decoders will produce slightly
  different JPEG pixels, so vision golden tests must use PNG inputs or tolerances.
- `Multipart` is duplicated in `audio/` and `diffusion/`. Merge it into the server crate.

### 30.7.6 Web console, tests, build, packaging (~20–27 weeks)

- **The console cannot simply stay in Java.** It does not talk to the engine over HTTP. It calls `SlotPool`,
  `ModelLoader`, `Engine`, `Admission`, `ChatTemplate` and `KvStore` in-process. Keeping it would mean inventing
  a slot/lease/KV-save/load-unload RPC and shipping a JVM next to the Rust binary. That is acceptable only as a
  stop-gap.
- **Recommended:** port the Spring backend (~9k lines, `ModelManager` alone 1,061) to axum. Keep the static
  JS/CSS/HTML **byte-for-byte**, embedded with `rust-embed`, and preserve the `/api/*` JSON/SSE contract so the
  page and the node JS tests run unchanged. A Leptos/Yew rewrite of the UI is not justified. At the same time,
  merge the two server surfaces (`server/` and `jllm-web`) into one, behind a `web` feature.
- Tests: rewrite the 81 `main()` programs as `cargo test` (unit) and `cargo bench` / examples (benchmarks), about
  6–8 weeks. The Python references, fixtures, `make-tiny-model.py` and the llama.cpp shell checks are reused once
  small Rust CLI equivalents exist for `GgufGreedy`, `GgufTokenize`, `GgufTinyWrite` and `HfToGguf`.
- Build:

```
jllm/                       Cargo workspace
├─ crates/jllm-core         json, config, quant formats, GGUF/safetensors, Architecture
├─ crates/jllm-tokenizer
├─ crates/jllm-cpu          kernels (std::arch), thread pool, CPU backends
├─ crates/jllm-metal        build.rs: ObjC shim via cc (or objc2-metal), include_str!(".metal")
├─ crates/jllm-engine       engine, kv, sampling, scheduler, speculation
├─ crates/jllm-media        vision, audio, diffusion, video
├─ crates/jllm-server       HTTP + OpenAI API (+ console backend behind `web`)
├─ crates/jllm-cli          the `jllm` binary (replaces the ./jllm launcher)
└─ xtask                    golden-dump runner, fixture generation, emulator build
features: metal, web, emulator
```

- Docker: a cargo-chef builder, then a slim/distroless runtime with one binary (cross via buildx or
  cargo-zigbuild), about 50–100 MB instead of a JDK image. Keep `entrypoint.sh`'s API-key generation and env
  mapping, uid 10001, the `/data` volume and the healthcheck. Drop `maven-proxy.sh` and every JVM memory flag.
  Do not build with `target-cpu=native`, because runtime dispatch handles CPU features.
- There is no CI today: the six gates in `wip/README.md` are run by hand. A port should add CI on Linux (CPU,
  emulator, Python references) and an Apple Silicon runner for Metal.

---

## 30.8 Proving it is identical

1. **Freeze the Java build as the oracle.** Generate tiny random models of every family with
   `scripts/make-tiny-model.py`, as safetensors and as GGUF. Add a Java dump tool that writes, per fixture:
   tokenizer ids, rendered chat templates, dequantized rows of every quant type, first-position logits, greedy
   token ids, top-k logprobs and scheduler traces.
2. **Compare by category.** Tokenizer ids, template renders, dequantized blocks, quant cache bytes and greedy ids
   must match bit-exactly. Logits are compared with a tolerance plus top-1/top-5 agreement. The CPU and Metal
   backends must agree with each other through `scripts/arch-smoke.sh` (`METAL_EMU=1` on Linux).
3. **Independent referees.** The existing llama.cpp, HF transformers/diffusers and whisper.cpp scripts in
   `scripts/` and `src/test/python/` work against the Rust binary unchanged.
4. **Wire contract.** Record OpenAI-API and console `/api/*` request/response and SSE transcripts from the Java
   server, then replay them against the Rust server. Also add a tool-call suite (`scripts/tool-call-check.sh`).
5. **End to end.** The existing exactness md5 in `wip/` (`79472f690f73bd28598d5d7fb5dee1a4`) and real-weights
   checks on Qwen3-0.6B serve as the final gate.

---

## 30.9 Effort and plan

| Area | Engineer-weeks |
|---|---|
| CPU backend + native kernels | 18–22 |
| Metal backend (shaders reused) | 20–23 |
| Models / GGUF / safetensors / tokenizer / hub | 21–23 |
| Engine / KV / scheduler / sampling / server / config | 24–28 |
| Vision / audio / diffusion / video | 24–30 |
| Web console backend, tests, build, Docker | 20–27 |
| **Total** | **~127–153** |

These are single-engineer estimates per area by an experienced Rust/SIMD/Metal engineer, with tests included.
Integration across areas is partly double-counted, so ~130–150 is the realistic range.

A cross-check by volume: ~73k lines of engine code, ~9k of console backend and ~29k of tests, less the ~5k that
disappear (the Vector API fallback kernels and the FFM layer), is about 105k lines to carry over. A port against
an existing, tested implementation usually moves at 1,000–1,200 finished lines per engineer-week, which gives
**~90–105 weeks**, or up to ~120 with the restructuring of §30.4 and the parity harness. The bottom-up figure is
higher because each area's estimate carries its own debugging and tuning reserve. Treat ~90–120 as the optimistic
range and ~130–150 as the planning figure.

### Suggested phases (team of 4)

| Phase | Duration | Deliverable |
|---|---|---|
| 0. Foundations | 3–4 w | workspace, dependency policy, `trait Backend` / `trait GpuDevice`, json + config, Java golden-dump tool and tiny-model fixtures, CI |
| 1. CPU text MVP | 8–10 w | GGUF + safetensors loading, tokenizer, chat templates, Qwen3/Llama/Mistral on the CPU (Q4_0/Q8_0/K-quants), sampler, flat KV, single-slot `./jllm chat` and `serve` with the OpenAI API, matching greedy ids |
| 2. Metal + breadth | 10–12 w | Metal backend (shim, then decode replay, pipelining, concurrent encoder), all model families and quant formats, the converter and byte-compatible cache, hub downloader |
| 3. Serving features | 8–10 w | slots, batched decode, paged KV + prefix tree, mixed scheduler, speculation, constrained tools / jump-forward, tiered KV + memory planner, sparse attention, saved conversations |
| 4. Multimodal + console | 10–12 w | vision, audio, diffusion, AnimateDiff, console backend on axum with the unchanged static UI, Docker; Wan 2.1 last or behind a flag |
| 5. Hardening | 4–6 w | wire-replay suite, performance parity (tok/s against the numbers in the docs), stress and loom tests, documentation updates |

Total: about **10–12 months for four engineers**, or about 6–7 months to phase 3 (a feature-complete LLM server).

---

## 30.10 Risks

| Risk | Mitigation |
|---|---|
| Unsafe KV / GPU / mmap code introduces memory bugs that Java's bounds checks and GC prevented | Keep it in a small number of audited modules. Use `debug_assert!` bounds checks, Miri on the pure logic, loom on the pool, ASan builds in CI |
| Tokenizer regex/Unicode divergence | Differential fuzzing against Java and HF `tokenizers` before any model work depends on it |
| Silent shader/host parameter mismatch | `#[repr(C)]` params generated from one definition. Every kernel exercised through the emulator against numpy references (`kernel_tests.py`) |
| Scheduler behaviour drifts (priorities, aging, preemption) | Deterministic-clock tests, recorded traces from the Java scheduler |
| Performance regressions on hot paths (no JIT profile-guided inlining) | Benchmark each kernel against the Java numbers in docs/07, 08 and 11 as it is ported. Consider PGO builds |
| Scope: 20+ families × 15+ quant formats × 2 backends | Tiny-model fixtures per family from day one. Port families in the order of docs/28 |
| No CI and Mac-only GPU paths | Metal emulator on Linux, plus one Apple Silicon runner |
| Dependency creep against the zero-dependency ethos | Write the §30.5 policy down and review new crates |

---

## 30.11 Is it worth it?

Arguments for:

- one static binary with no JDK and no JVM memory flags
- no warm-up
- direct access to every SIMD extension
- easier embedding as a library in other programs (C ABI, Python via PyO3)
- a much smaller Docker image

Arguments against:

- 2.5–3 person-years to reach where the Java engine is today, while the Java version keeps moving
- the Java design already avoids most JVM costs: the hot paths are mmap'd memory, hand-written SIMD, a custom
  thread pool and GPU kernels that do not change
- a large share of the value (the shaders, the docs, the reference scripts and the console UI) carries over either
  way

A cheaper middle path is to port incrementally behind the existing C ABI style. Start with the CPU kernels as a
Rust `cdylib` replacing `libjllm_cpu`, which FFM already loads with a self-check. Then move the tokenizer and
format readers, and keep Java for the engine and server until the Rust core is proven.
