# 30. What it would take to rewrite jllm-serve in Rust

This page estimates the work needed to build a functionally identical jllm-serve in Rust instead of Java, and
gives an implementation plan for it. It covers what exists today, how each Java mechanism maps to Rust, which
parts carry over unchanged, which parts are hard, the plan itself (crates, work packages, milestones and their
acceptance tests), how to prove the port matches the original, and the effort involved.

> **How this page was made.** Six investigations, one per area, read the code and docs. Four more then checked
> every claim on this page against the source and against the Rust documentation (about 200 claims; the ones
> that were wrong are corrected here), and two derived the dependency graph and the work packages from the import
> lines and `wc -l`. Nothing was built or prototyped, so the effort numbers are estimates. Statements about Rust
> that could not be confirmed from the official documentation are marked *(unconfirmed)*. File paths are relative
> to `src/main/java/org/robomiri/jllm/` unless shown otherwise.

---

## 30.1 Summary

| | |
|---|---|
| Java to port | ~73k lines of engine code, ~29k lines of test/bench programs (94 files, 81 with a `main`), ~9k lines of jllm-web Java plus ~3k of its tests; ~114k in all. Also ~1.3k lines of C and 575 of Objective-C |
| Reusable as is | ~7.9k lines of Metal shaders (`metal/kernels.metal`, `diffusion.metal`, `vision.metal`: 7.5k; `audio/audio.metal`: 366), the Metal emulator (`scripts/metal-emu/`), all Python reference scripts and checked-in fixtures, the web console's static JS/CSS/HTML (~8.6k lines), the docs |
| Expected Rust size | ~0.8–1.0× the Java line count |
| Effort | **~125–150 engineer-weeks** for full parity (the work packages of §30.9 add up to 140), or about 2.5–3 person-years. A line-rate cross-check gives ~95–125 as the optimistic end (§30.10). A team of 4 would need **~10–12 months**, with Apple Silicon hardware available |
| Reduced scope | ~100 engineer-weeks for the LLM server only (CPU + Metal, GGUF + safetensors, OpenAI API, `./jllm serve`), without diffusion/video/audio/vision and without porting the Spring console |
| First results | first greedy token on the CPU after porting ~14k Java lines; first served chat completion after ~4.5k more; first Metal token after ~5.3k more (§30.9.3) |
| Hardest parts | (1) the unsafe boundary where kernels and the GPU write KV blocks while the pool's refcount protocol keeps them safe; (2) tokenizer regex/Unicode parity; (3) the volume of quant formats × kernels × model families; (4) scheduler timing semantics; (5) Metal host code that must match shader parameter layouts exactly; (6) a `Backend` interface whose batched passes hand one backend the caches of several others (§30.4) |
| What gets better | no JIT warm-up (docs/07 §7.10 reports the Vector API 500–1000× slower in the interpreter and C1), no FFM layer, a single static binary instead of a JDK, AVX-512 VNNI / dotprod usable directly in the language, predictable memory without `-Xmx` / `MaxDirectMemorySize` tuning, core pinning |
| Libraries | The Rust port may use any library for the math (§30.5): matrix multiply, FFT, generic SIMD, f16. That saves about 4 weeks, not more, because the hot path (int8 activations × quantized blocks read in place) has no library equivalent |
| What gets worse | stable Rust has no portable SIMD, so the int8 kernels are written once per instruction set (the float kernels can use a generic SIMD crate). Unsafe code appears in kernels, the thread pool, mmap→`MTLBuffer` and KV storage. The JDK's TLS, HTTP client, Unicode normalization, regex and ImageIO were free, and need crates in Rust |

"Identical" should mean the **same features, the same CLI and config keys, the same HTTP/JSON/SSE wire contract,
the same on-disk formats, and the same greedy token ids**. Bit-identical logits are not a realistic target, and
the Java engine does not guarantee them either: results already depend on prefill versus decode path and on the
reduction order (docs/07 §7.14).

---

## 30.2 The project as it is

| Area | Java packages | Lines | Notable |
|---|---|---|---|
| CPU backend | `cpu/` + `src/main/native/jllm_cpu*.c/.inc` | 11.7k + 1.3k C | Vector API kernels for 15 block quant types plus F16/BF16 (the C library has int8 dot products for 14 of them, and only 7 on x86), int8 activations, custom spin/park thread pool, f32/f16/q8/q4 KV caches, paged + sparse attention, dense / MoE / MLA / Mamba / Qwen3-Next / hybrid CPU+GPU backends |
| Metal backend | `metal/` + `jllm_metal.m` + shaders | 10.4k + 575 ObjC + 7.5k MSL | generic C ABI over Metal (22 functions), decode replay (~170 dispatches a step recorded and patched for Qwen3-0.6B, up to ~390 for the 30B MoE), pipelined greedy decode, concurrent encoder with a reflection-based hazard tracker, `MetalTuner`, memory planner and expert placement |
| Models and formats | `model/`, `gguf/`, `safetensors/`, `hub/`, `ModelLoader`, `ContextBudget` | ~14k | 22 HF `model_type` entries as a sealed `Architecture` hierarchy (4 permits), a quantized cache file (`QuantFile`), safetensors→Q4_0/Q8_0/Q2_K/Q6_K converter (reads FP8 and MLX checkpoints too), GGUF v2/v3 read in place, a resumable SHA-256-verified HF downloader |
| Tokenizer | `tokenizer/` | 4.3k | byte-level BPE, SentencePiece (greedy + Unigram), normalizers / pre-tokenizers, GGUF vocabularies with llama.cpp pre-types, 20 hand-written chat template formats (no Jinja) |
| Engine | `engine/`, `kv/`, `sampling/` | 9.2k | slots, `SlotPool`, mixed-pass scheduler (`Admission`, `BatchGroup`, `PassPlanner`), speculation (n-gram, draft model, MTP), paged KV with a radix prefix tree, refcounts, COW, tiered spill to RAM/SSD, JSON-schema and tool-call constrained decoding |
| Server, config | `server/`, `json/`, `config/`, `Main` | 3.7k | hand-written HTTP/1.1 on virtual threads, OpenAI API + SSE, own JSON, layered `.properties` config with 44 aliases, 10 subcommands |
| Multimodal | `vision/`, `audio/`, `diffusion/` | 19.6k | CLIP/SigLIP with 5 projector types (idefics3, gemma3, qwen2vl, qwen2.5vl, qwen3vl), Whisper + word timestamps, OuteTTS + WavTokenizer, SD 1.x/2.x/SDXL, AnimateDiff, Wan 2.1 parts, hand-written VP8L WebP and GIF encoders, ffmpeg for MP4 and for audio other than WAV/AIFF/AU |
| Web console | `jllm-web/` | ~9k Java + 8.6k JS/CSS/HTML | Spring Boot 4.1.1, **in-process** use of `SlotPool`, `ModelLoader`, `Engine`, `Admission`, `ChatTemplate` and the hub, image, video and audio APIs; 9 tabs, model management, KV save/restore (its own `web.models.KvStore`), MCP client, integration configs |
| Tests | `src/test/` | ~29k | 94 files, 81 of them `main()` programs, run through 38 `exec-maven-plugin` executions (no JUnit in the engine); 11 MB of fixtures; parity scripts against llama.cpp / HF / whisper.cpp / diffusers; the Metal emulator |

HTTP surface of `./jllm serve` (`server/`): `GET /`, `/index.html`, `/health`, `/v1/models`;
`POST /v1/chat/completions`, `/v1/completions`; `/v1/audio/transcriptions`, `/v1/audio/translations`,
`/v1/audio/speech`, `/api/audio/models`; `POST /v1/images/generations`, `/v1/images/edits`,
`GET /v1/images/files/{id}`; `POST /v1/videos/generations`, `/v1/videos/encode`, `GET /v1/videos/files/{id}`;
`OPTIONS` for CORS. Subcommands of `Main`: `config`, `download`, `chat`, `run`, `serve`, `image`, `video`,
`transcribe`, `speak`, `bench`.

The Java project's defining rule is **zero external dependencies**: only the JDK. The Rust port drops that rule
for the math and for what the JDK supplied (§30.5).

---

## 30.3 Java mechanism → Rust mapping

| Java | Used for | Rust |
|---|---|---|
| `jdk.incubator.vector` (`SPECIES_PREFERRED`, ~280 vector load sites in `cpu/`, fma, `reduceLanes`) | all CPU kernels | Float kernels: a generic SIMD crate with runtime dispatch (`macerator` or `fearless_simd`), and a library matrix multiply for the dense f32 paths (§30.5). Int8 kernels: `std::arch` intrinsics under `#[target_feature]`, one implementation per level (scalar / NEON / dotprod / AVX2 / AVX-VNNI / AVX-512 VNNI). `std::simd` is still nightly-only, so it is ruled out |
| JIT-stability workarounds (fixed-order reductions, magic-constant int→float, one call site per helper) | bit-stable results under C1/C2 | Mostly unnecessary. Keep the fixed reduction order where results must be stable, and use `f32::round_ties_even` (stable since 1.77) for `rint` |
| `libjllm_cpu` via FFM `Linker` with `Linker.Option.critical` + self-check (`NativeCpu`, 610 lines) | int8 dot products | Removed. The C kernels become Rust intrinsics (~1.3k C → ~1.5–2k Rust) and the self-check becomes `cargo test`. The AVX-512 VNNI and AVX-VNNI intrinsics are stable since Rust 1.89 and the NEON dot-product intrinsics (`vdotq_s32`) since **1.98**, which sets the minimum toolchain. `is_x86_feature_detected!` replaces cpuid. On Apple Silicon build with `+dotprod` (every M-series CPU has it), because the documentation of `is_aarch64_feature_detected!` only promises runtime detection on Linux |
| `FileChannel.map` → `MemorySegment` | weights in place (GGUF, safetensors, quant cache); also KV slabs (`CpuKvStorage`), `StreamedKv` and `FileSpill` | `memmap2::Mmap` inside an `Arc`, with offsets rather than borrowed slices (avoids self-referential structs). Only the quant cache is 16 KiB-aligned (`QTensor.PAGE`); GGUF and safetensors tensors sit at arbitrary offsets, and `Metal.java` wraps the pages a tensor touches and carries the offset. `bytemuck`/`zerocopy` + `read_unaligned` for block structs |
| `Arena.ofAuto()` and file-backed slabs | CPU KV storage: slabs are mappings of unlinked temp files, with a 64-byte-aligned arena allocation as the fallback | the same two paths, owned by the pool and freed on `Drop`. Unsafe views by block id |
| `ThreadPool` (packed `AtomicLong`, spins 200 µs then parks) | hundreds of tiny parallel loops per token | 1:1 port with std atomics, `thread::park`, `hint::spin_loop`. **Not rayon**, whose per-job overhead is too high for this. Optionally add `core_affinity` |
| Virtual threads per connection | HTTP | Either OS threads capped by `server.max-connections` (faithful), or tokio + axum 0.8 with the engine on dedicated threads and an mpsc channel for SSE (pragmatic). The engine core stays synchronous either way |
| `synchronized`, wait/notify, fair `ReentrantLock.tryLock(timeout)`, volatiles, `ThreadLocal` | KvPool, Admission, BatchGroup, Engine | `parking_lot::{Mutex, Condvar, FairMutex}` (its fair mutex supports locking with a timeout; std's has none), atomics, `ArcSwap` for whole-object volatile swaps |
| GC-managed object graphs (trie with parent pointers, `BatchGroup.this`, callbacks) | KV prefix tree, scheduler | arena `Vec<Node>` + generational ids. `Arc`/`Weak` or id indirection for back-references. RAII guards for `Slot.close()` |
| Sealed interfaces + records | `Architecture`, `LayerSpec` | `enum Architecture { Decoder, Mamba, Qwen3Next, Hybrid }` + structs. `match model_type` in `Families` |
| `Float.float16ToFloat`, `Math.fma` | f16/bf16 | the `half` crate, or ~20 lines by hand. Use `f32::mul_add` only inside `fma`-enabled functions; elsewhere it falls back to a slow library call *(unconfirmed)* |
| `java.util.regex` with `UNICODE_CHARACTER_CLASS` and lookahead (`\s+(?!\S)` in `Tokenizer`, `PreTokenizers`, `GgufPre`) | pre-tokenizers | `regex` has no lookaround. Use either `fancy-regex` (lookaround, Unicode classes) or hand-coded splitters for the GPT-2 / Qwen / Llama 3 patterns plus Unicode category tables |
| `java.text.Normalizer` | NFC / NFD / NFKC / NFKD in the tokenizer normalizers, NFC in `ClipTokenizer`, NFKD in `OuteText` | `unicode-normalization` (all four forms) |
| `java.net.http.HttpClient` (`Redirect.NEVER`, redirects followed by hand so `HF_TOKEN` never reaches the CDN, Range resume) | Hub downloader | `ureq` 3 with `max_redirects(0)`, a plain `Range` header, plus `sha2`. TLS via `rustls`, its default (+ `rustls-native-certs` / `SSL_CERT_FILE` instead of keytool) |
| `javax.imageio` | PNG/JPEG/GIF/BMP decode (`vision/ImageDecoder`), PNG/JPEG decode and encode (`diffusion/Imaging`) | `png` + `zune-jpeg` (or `image`). JPEG pixels will differ from ImageIO by a few LSB |
| `ProcessBuilder` | ffmpeg (`AudioDecoder`, `AudioEncoder`, `VideoEncoder`) | `std::process::Command` (1:1) |
| Panama downcalls into `jllm_metal.m` | Metal | Phase 1: keep the ObjC shim, built by the `cc` crate in `build.rs`, with 22 `extern "C"` functions and a `#[repr(C)] JMCmd` (384 bytes: 16 buffers, 16 offsets, 64 bytes of params; `_Static_assert` kept). Phase 2: replace it with `objc2-metal`, which exposes pipeline reflection and no-copy buffers. Avoid the `metal` crate, whose README marks it deprecated in favour of `objc2-metal`. Reject wgpu (no simdgroup matrices, no zero-copy mmap buffers, every shader rewritten in WGSL) |
| `.metal` loaded and compiled at runtime (`newLibraryWithSource`) with `#define` prefixes, no function constants | shader variants | `include_str!` + runtime compilation, unchanged. Offline metallibs are optional later |
| `-D` system properties / `.properties` / CLI | config | a hand-ported `Settings` (~500 lines: 44 aliases, source tracking, typed validation, Java `.properties` escaping), with the bundled `jllm.properties` via `include_str!`. clap does not fit the arbitrary-dotted-key model |

---

## 30.4 The `Backend` interface and who owns what

`model/Backend.java` (373 lines, 36 instance methods, 4 of them abstract) is the seam between the engine and the
two backends, so its Rust shape has to be settled before anything else is ported. Most of it translates directly:

```rust
pub trait Backend: Send {
    fn name(&self) -> &str;
    fn context_length(&self) -> usize;
    fn max_batch(&self) -> usize;
    /// `logits: None` = a prefill chunk whose logits nobody wants (Java passes null).
    fn forward(&mut self, tokens: &[u32], pos: usize, logits: Option<&mut [f32]>) -> Result<()>;
    fn forward_all(&mut self, tokens: &[u32], pos: usize, logits: &mut [f32], stride: usize) -> Result<()>;
    fn forward_embeds(&mut self, embeds: &[f32], pos: usize, deepstack: Option<&[&[f32]]>, logits: Option<&mut [f32]>) -> Result<()>;
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
| `forwardBatch(Backend[] members, …)` | one decode step for several slots; `this` lends its buffers and must be one of `members` | `&mut self` and `&mut members[i]` alias |
| `forwardMixed(Segment[] segments)` with `Segment(Backend owner, …)` | a prefill chunk plus other slots' decode tokens in one pass, each segment in its owner's cache; `this` lends its buffers and need not be an owner | a lender plus N owners, all mutably borrowed at once through trait objects |
| `copyStateFrom(Backend other, int n)` | prefix copy between slots | needs two slots borrowed at once |

How it works today: there is no model object. `ModelLoader.newBackend` builds a full `CpuBackend` or
`MetalBackend` per slot over the shared `ModelWeights` (the mmap). Per weights, the slots share the `BatchGroup`,
the paged `KvPool` and, on the CPU, the `ThreadPool`. On Metal each slot's constructor opens its own `Metal`
context (its own compiled library), and the batched passes accept only members with the same `ModelWeights`
object.

The clean Rust shape splits what Java keeps in one object:

- a shared, immutable **`Model`** (`Arc`): weights, architecture, the thread pool, and on Metal the one device
  context with its compiled pipelines (which today is per slot), plus the pass logic;
- a per-slot **`SlotState`**: the KV sequence or recurrent state, RoPE positions, scratch buffers.

The batched entry points then live on the model and take the slots they touch:
`model.forward_mixed(&mut [Segment<'_>])`, where each segment holds `&mut SlotState`. Distinct slots are distinct
borrows, and the "owners must be distinct, idle backends" rule in the Javadoc becomes something the compiler
checks. `BatchGroup` already serializes passes with its device lock, so no new locking is needed. This is a
restructuring, not a line-by-line translation, and both backends and the scheduler depend on it. It also compiles
the shaders once per model instead of once per slot.

---

## 30.5 Libraries: what to use and what to write

The Java project depends only on the JDK. **The Rust port does not keep that rule for the math:** any library may
be used there, and the edges (TLS, HTTP client, Unicode, image decoding) use crates too, because the JDK supplied
them. What stays hand-written is what defines the project's behaviour or has no library equivalent.

The crates below were checked on crates.io and docs.rs in November 2026 (versions in brackets).

**Math libraries to use**

| Need | Library | Replaces | Notes |
|---|---|---|---|
| f32 matrix multiply | `gemm` [0.19] or `matrixmultiply` [0.3.11]: pure Rust, runtime-detected AVX2 / AVX-512, NEON; optional threads. On macOS, Apple Accelerate through `blas-src` (links the system framework, nothing to install) | the dense f32 paths: `vision/VMath`, `audio/Dense`, `diffusion/CpuOps` (conv2d as im2col + sgemm, linear layers, attention products), and the F32 / F16 / BF16 weight paths of the LLM | `gemm` also multiplies f16 directly. Their threading must not fight the engine's own pool: run them single-threaded inside pool jobs, or give them the cores for the media models only |
| FFT | `rustfft` [6.4] + `realfft` [3.5]: AVX, SSE4.1 and NEON chosen at runtime on stable | `audio/Fft` and its uses in `LogMel` and `WavTokenizerDecoder` | |
| SIMD without writing each kernel twice | `macerator` [0.5] or `fearless_simd` [1.2]: one generic source, runtime dispatch, x86 (to AVX-512) and NEON. `wide` [1.7] has no runtime dispatch; `multiversion` [0.9] dispatches autovectorized scalar code | the float kernels of `VectorKernels`, `GgufKernels`, `KvCache` (dots, axpy, norms, RoPE, softmax) | None of them wraps the 4-way u8×i8→i32 dot product (VNNI `vpdpbusd`, NEON `sdot`), so the int8 kernels keep raw `std::arch`. `pulp` [0.22] is x86-centred: a NEON backend could not be confirmed from its documentation |
| f16 / bf16 | `half` [2.7], slice conversions | `Float.float16ToFloat` loops | |
| Tensor shapes in the media models | `ndarray` [0.17] (its `dot` goes through `matrixmultiply` or BLAS) | hand-indexed `float[]` in the UNet, VAE and Whisper code | optional; NHWC layout is kept so the Metal kernels see the same buffers |
| Log-mel, resampling | `mel_spec` [0.5] (aligned with whisper.cpp / librosa), `rubato` [5.1] | `audio/LogMel`, `audio/Resampler` | only if `scripts/audio-accuracy.py` still passes; otherwise port the ~300 lines |
| Scalar math | std / `libm` | `Math.exp` and friends | no stable crate vectorizes `exp` / `tanh` (`sleef` needs nightly), so the fast-math polynomials of `VectorKernels` are ported as they are |

**What no library replaces**

- **The quantized matrix-vector kernels.** The hot path multiplies int8-quantized activations with weight blocks
  read in place from the memory-mapped file, for 15 block formats, through VNNI or dotprod. General BLAS has
  nothing for this. `candle-core` [0.11] is the closest: it has block kernels for Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
  and Q2_K–Q6_K, quantizes activations to int8 and multiplies on the blocks directly, with AVX2 and NEON versions
  of most of them. But it copies the blocks out of the file (no in-place mmap, so no zero-copy Metal buffers and
  twice the memory), has no AVX-512 or VNNI path, and lacks IQ4_NL, IQ4_XS and MXFP4. Use it as a reference
  implementation and as a second oracle in tests, not as the kernel layer.
- **llama.cpp / ggml bindings** (`llama-cpp-2`) expose the whole llama.cpp API, not its vector kernels. Linking
  them would make the project a wrapper around llama.cpp and lose the paged KV cache, the scheduler and the Metal
  kernels.
- **Ready-made model crates** (`candle-transformers` has Whisper, Stable Diffusion, CLIP and SigLIP; `burn` is a
  tensor framework). They are different implementations with their own weights handling and GPU backends, so
  they cannot share the project's Metal kernels, KV cache or formats, and their outputs would not be the Java
  engine's. They serve as oracles, like diffusers and whisper.cpp today.
- Attention through the KV cache (flat, paged, sparse, quantized keys), the Metal backend, the sampler and the
  constrained decoding: project-specific, hand-written as in Java.
- Random numbers: no crate reproduces Java's `SplittableRandom` or `diffusion/Rng`; both are short hand ports.

**Other crates** (non-math): `memmap2`, `parking_lot`, `rustls` + `ureq`, `sha2`, `unicode-normalization`,
`fancy-regex` (or hand splitters), `png` + `zune-jpeg`, `objc2-metal` (phase 2), `serde_json` with
`preserve_order` or a ported `Json`, and `axum` / `tokio` if the server is not hand-written. The `tokenizers`
crate is a dev-dependency test oracle only: it does not cover GGUF vocabularies, llama.cpp pre-types, protected
special tokens or the piece cache.

**What the libraries change**

- *Effort:* about 4 weeks less (§30.9.4: C2, C3, V2, A1, A2, A4, D1). The saving is small because the
  expensive kernels are the quantized ones.
- *Results:* a library matrix multiply sums in its own order, so f32 results differ from Java's in the last
  bits. The media models are already compared with a tolerance against diffusers, whisper.cpp and llama.cpp, so
  their gates stay as they are. For the LLM, the exactness gate compares the generated text of a greedy run on
  CPU and Metal; the quantized kernels that decide it stay hand-written with a fixed summation order.
- *x86 coverage:* with a generic SIMD crate, the float kernels get AVX-512 and NEON from one source, which the
  Java build gets from the Vector API.

---

## 30.6 Java behaviour that has to be reproduced on purpose

A line-by-line translation compiles and then differs in small ways. These are the places found in the code where
Java's semantics leak into results, files or the wire format:

| Java behaviour | Where it matters | In Rust |
|---|---|---|
| `int`/`long` arithmetic wraps silently | hashes (the KV path hash seeded with `0x2545F4914F6CDD1D` in `kv/KvPool.java`, the `GOLDEN` multiplier in `diffusion/Rng.java`), bit packing in the quant formats | `wrapping_mul` / `wrapping_add` explicitly; a plain `*` panics in debug builds and wraps in release, so tests and production would differ |
| `byte` is signed, `>>>` vs `>>`, `char` is 16 bits | quant block decoding, tokenizer byte handling | `i8` vs `u8` chosen per field; most `& 0xFF` masks disappear, which is where translation slips happen |
| `String.hashCode()` in persisted names | `metal/ExpertPlacement.java:228` (routing-count file name), `metal/Metal.java:80` (`sourceHash` that keys the `MetalTuner` cache) | reimplement the 31-polynomial over UTF-16 units, or accept that these caches are rebuilt once |
| `SplittableRandom(seed).nextDouble()` | `sampling/Sampler.java`: a request's `seed` decides the sampled tokens | port SplitMix64 and Java's double conversion (about 20 lines) if the same seed must give the same text; no crate reproduces it |
| `StringBuilder.append(Number)` for JSON numbers | `json/Json.java` writes doubles with `Double.toString` and floats with `Float.toString` (`1.0E-5`, `100.0`) | Rust prints `0.00001` and `100`. Clients parse both, but recorded-response tests need a number-aware comparison or a port of the Java formatting |
| Strings are UTF-16 | tokenizer offsets, stop strings, `StreamDecoder` | Rust is UTF-8, which suits the byte-level constraint code better; anything that counts or indexes `char`s must be re-derived, not copied |
| Map iteration order | JSON objects are `LinkedHashMap` (insertion order), so JSON key order is deterministic. A search of engine, kv, sampling, server, json and config found **no** place where `HashMap` order reaches a token or the wire. The one order-dependent spot is `kv/FileSpill.trim()`, which sorts a `ConcurrentHashMap`'s entries by use and leaves ties in map order (it decides which spill file goes first) | `IndexMap` (or `serde_json` with `preserve_order`) for JSON objects; an explicit secondary sort key in `FileSpill` |
| `WeakHashMap` caches keyed weakly by an owning object (`ModelWeights`, `Tokenizer`, the vocabulary array) | `ModelLoader` (2), `CpuMatmul` (2), `KvPools`, `TokenTrie`, `JsonConstraint` | no equivalent. Move each cache into the object it is keyed on (the model or tokenizer owns its tables), which is simpler |
| Exceptions for control flow | `StaleCacheException` (`model/ModelWeights.java`: delete and reconvert once, a second failure is an error), `OutOfMemoryError` caught in `diffusion/MetalOps.java:149` to free scratch and retry, client-gone exceptions | `Result` with a typed error enum. Allocation failure cannot be caught as a panic, so the Metal retry must use a fallible allocation path |
| `InterruptedException` | 26 `interrupt()` sites, of which 24 only restore the flag after catching it. Two interrupt another thread: the `SlotPool` sweeper and a hub download thread. Waiters in `Admission` and `BatchGroup` wake by `notifyAll()` and `wait(50)` timeouts | no interrupt in Rust. A stop flag plus `Condvar::wait_timeout` for the two real cases; the 24 restore sites disappear |
| `Math.exp` | sampler softmax (`Sampler.java`), schedulers, mel filters | Rust's `exp` can differ in the last bit. Harmless for tokens in practice, but it rules out bit-identical float outputs outside the hand-written fast-math polynomials |
| `float` → `int` casts saturate, NaN → 0 | quantizers | Rust's `as` behaves the same; no action, but do not replace with `to_int_unchecked` |
| Arrays are indexed by `int`; the paged cache was audited for 2³¹ (docs/24) | KV indices | `usize` everywhere removes the audit, but file formats keep their 32/64-bit field widths |
| Recursion on the call stack | only at parse and compile time: `JsonConstraint.Schema.compile` (depth = the schema's nesting; a `$ref` cycle is cut by a placeholder) and the `Json` parser (`MAX_DEPTH` = 256). The constraint state during decoding is a heap `Frame[]` stack driven by a loop | keep the depth limits. No stack-size setting is needed for decoding |
| Heap `float[]` passed to native code | CPU kernels | disappears: everything is a slice in one address space |

---

## 30.7 Area by area

### 30.7.1 CPU backend (~20 weeks)

- The Java side has two kernel sets for the int8 dot products: the Vector API ones (`Int8Kernels`,
  `GgufInt8Kernels`, `KQuantKernels`, ~1k lines) and the C library. Rust needs one. Those ~1k lines, the 610-line
  FFM layer `NativeCpu` and the JIT workarounds disappear. The float and GGUF kernels (`VectorKernels`,
  `GgufKernels`, `Q2KKernels`, `Q6KKernels`, `PagedKernels`, ~1.9k lines) have no native twin and are ported.
- **New work that Java never did natively:** on x86 the C library covers only 7 of its 14 types (Q4_0, Q8_0,
  Q4_0G, Q8_0G, Q4_K, Q5_K, Q6_K). The other 7 (Q4_1, Q5_0, Q5_1, Q3_K, IQ4_NL, IQ4_XS, MXFP4) have NEON, dotprod
  and scalar versions only and run on the Vector API on x86 today. The Rust port has to write AVX2 / VNNI versions
  of them, or accept a scalar path there.
- The cost is volume, not depth: 15 block types × {float dot, int8 dot, prefill tile} × the dispatch levels, plus
  scalar references. A generic SIMD crate removes the per-architecture copies of the float kernels, and
  `candle-core`'s block kernels are a reference for the common formats (§30.5).
- Rust never contracts `a*b+c` into an FMA by itself, which matches `-ffp-contract=off` in
  `scripts/build-native-cpu.sh`. The int8 paths are exact integer arithmetic per sub-block, so only the final
  float summation order needs matching.
- Miri cannot run most SIMD intrinsics or foreign calls, so it only helps on the pool and KV logic; the kernels
  are tested natively against scalar references.

### 30.7.2 Metal backend (~21.5 weeks)

- **The shaders are reused verbatim at zero cost.** This alone saves 12–16 weeks against a shader rewrite.
- Kernel parameter structs become `#[repr(C)]` types instead of hand-packed bytes. That is an improvement, but a
  grid, threadgroup or offset mismatch with the shader still fails silently. Generate the structs from one source,
  or test every kernel through the emulator.
- Hard spots: zero-copy mmap→`newBufferWithBytesNoCopy` (wrapping the 16 KiB pages a tensor touches, offset
  arithmetic, the mapping must outlive every buffer). Also the concurrent-encoder hazard tracker; test it against
  mode 2 ("barrier before every dispatch") as the oracle.
- Keep a `trait GpuDevice` with two implementations, `MetalDevice` and `EmuDevice`. The emulator
  (`scripts/metal-emu/runtime.cpp`) implements the same 22 `jm_*` functions, so the unchanged Metal host path
  runs on Linux, by hand today and in CI once there is one.
- Pipelined decoding stays off while any expert runs on the CPU, as today.

### 30.7.3 Models, GGUF, safetensors, tokenizer, hub (~24.5 weeks)

- **Keep the quant cache byte-compatible.** It is raw little-endian blobs at 16 KiB offsets plus a JSON index:
  `version=1`, `layout=2`, and the canonical `model` description string, which is compared on resume. The index
  is written last with an atomic move as the completion marker. If the quantizer's rounding matches, Java and
  Rust can then share one cache.
- Tokenizer parity is the largest correctness risk in this area. Check it by differential fuzzing on multilingual
  text against the Java build and HF `tokenizers`. Edge cases include added-token lstrip/rstrip, byte fallback,
  Metaspace schemes, Unigram tie-breaking and partial-UTF-8 stream decoding.
- Keep the chat templates hand-written. minijinja would only matter for a new feature: rendering arbitrary
  `tokenizer_config.json` templates.
- `ContextBudget` reads the JVM's old-generation heap, the direct-memory limit and the physical memory through
  MXBeans. In Rust it reads `/proc/meminfo` and the cgroup v2 limits, `sysctl` on macOS, and Metal's
  `recommendedMaxWorkingSetSize`; the heap / direct split goes away.

### 30.7.4 Engine, KV, scheduler, sampling, server, config (~28.5 weeks)

Where Rust's ownership model pushes back:

1. **KV rows are written by kernels and the GPU outside the pool lock.** In Java this is safe through a runtime
   protocol: a block is referenced, so it is not evictable, and it is either private or published-immutable. In
   Rust it becomes an unsafe store addressed by block id that encodes the same invariant. Audit it, and test it
   with loom on the pool logic plus stress tests. This is the single most important piece to get right.
2. The prefix trie (parent pointers, a `HashMap` of child lists per node, LRU eviction toward the leaves) becomes
   an arena with generational ids.
3. Back-references and callbacks (`Reclaimer`, `TableSink`, `BatchGroup.Member`) become `Weak` references or ids.
4. Admission and `BatchGroup` (wait/notify with 50 ms timeouts, aging, preemption, `parkNanos` spinning) port to
   `Condvar`, but they are timing-sensitive. Build a deterministic-clock harness and use
   `SchedulerPolicyTest` / `MixedSchedulerTest` as the specification.

Wire fidelity covers SSE framing (`data: [DONE]`), error shapes and status codes, `usage.prompt_tokens_details`,
stop strings across partial UTF-8, JSON number formatting and seeded sampling (both in §30.6).

### 30.7.5 Vision, audio, diffusion, video (~26 weeks)

- These modules should share the CPU kernel crate and the Metal device with the LLM path. `Ops` becomes a trait
  with CPU and Metal implementations. The new ops on top of the LLM set are conv2d (im2col + GEMM), GroupNorm,
  tanh-GELU and adaptive modulation. On the CPU the matrix multiplies and the FFT come from libraries (§30.5),
  which is where most of the library saving is.
- Port the hand-written VP8L and GIF encoders exactly, since they are pure bit manipulation. Crates would produce
  different bytes, and `image-webp` cannot encode animations.
- ImageIO is the one JDK facility without a hand-written equivalent. Rust decoders will produce slightly
  different JPEG pixels, so vision golden tests must use PNG inputs or tolerances.
- `Multipart` is duplicated in `audio/` and `diffusion/`. Merge it into the server crate.
- Wan 2.1: every component is checked against diffusers, but it has never run end to end with the real UMT5
  weights (docs/23). Port it last, behind a feature.

### 30.7.6 Web console, tests, build, packaging (~19.5 weeks)

- **The console cannot simply stay in Java.** It does not talk to the engine over HTTP; it calls the engine
  classes in-process (§30.2). Keeping it would mean inventing a slot/lease/KV-save/load-unload RPC and shipping a
  JVM next to the Rust binary. That is acceptable only as a stop-gap.
- **Recommended:** port the Spring backend (~9k lines, `ModelManager` alone 1,061) to axum. Keep the static
  JS/CSS/HTML **byte-for-byte**, embedded with `rust-embed`, and preserve the `/api/*` JSON and streaming contract
  so the page and its three node JS tests run unchanged. The page streams with `fetch` and
  `res.body.getReader()`, not `EventSource`, so the chunk format of `/api/chat` is the contract. A Leptos/Yew
  rewrite of the UI is not justified. At the same time, merge the two server surfaces (`server/` and `jllm-web`)
  into one, behind a `web` feature.
- Docker: a cargo-chef builder, then a slim/distroless runtime with one binary (cross via buildx or
  cargo-zigbuild) instead of a JDK image. Keep `entrypoint.sh`'s API-key generation and env mapping, uid 10001,
  the `/data` volume and the healthcheck. Drop `maven-proxy.sh` and every JVM memory flag. Do not build with
  `target-cpu=native`, because runtime dispatch handles CPU features.
- There is no CI today: the six gates in `wip/README.md` (install, web, emulator, kernel tests, arch smoke,
  exactness) are run by hand on a Mac. A port should add CI on Linux (CPU, emulator, Python references) and an
  Apple Silicon runner for Metal.

---

## 30.8 Proving it is identical

1. **Freeze the Java build as the oracle.** Generate tiny random safetensors checkpoints with
   `scripts/make-tiny-model.py` (it covers Llama, Mistral, Qwen2, GLM, Phi-3, Gemma 2/3/4, Mamba, Qwen3-MoE,
   Qwen3-Next, DeepSeek V2/V3, gpt-oss, Llama 4, Granite, Nemotron-H and Jamba; it writes safetensors only and
   has no plain dense Qwen3). Make GGUF versions of them once with the Java test tools `HfToGguf` and
   `GgufTinyWrite`, and keep the files as fixtures, so those two tools need no Rust port before the first
   milestone. Add a Java dump tool that writes, per fixture: tokenizer ids, rendered chat templates, dequantized
   rows of every quant type, first-position logits, greedy token ids, top-k logprobs and scheduler traces.
2. **Compare by category.** Tokenizer ids, template renders, dequantized blocks, quant cache bytes and greedy ids
   must match bit-exactly. Logits are compared with a tolerance plus top-1/top-5 agreement. The CPU and Metal
   backends must agree with each other through `scripts/arch-smoke.sh` (`METAL_EMU=1` on Linux).
3. **Independent referees.** The existing llama.cpp, HF transformers/diffusers and whisper.cpp scripts in
   `scripts/` and `src/test/python/` work against the Rust binary once it offers the small commands the scripts
   call today as Java programs: `GgufGreedy` and `GgufTinyWrite` (`qwen35-llamacpp-check.sh`), `GgufTokenize`
   (`gguf-tokenizer-check.sh`) and `HfToGguf` (`arch-smoke.sh`, `gguf-llamacpp-check.sh`).
4. **Wire contract.** Record OpenAI-API and console `/api/*` request/response and stream transcripts from the
   Java server, then replay them against the Rust server, with a number-aware JSON comparison. Also run
   `scripts/tool-call-check.sh`.
5. **End to end.** The exactness gate of `wip/README.md` (the md5 `79472f690f73bd28598d5d7fb5dee1a4` of a fixed
   greedy generation, equal on CPU and Metal) and the real-weights checks on Qwen3-0.6B are the final gate.

---

## 30.9 Implementation plan

### 30.9.1 Decisions to take before writing code

| Decision | Recommendation |
|---|---|
| Libraries | §30.5: libraries for matrix multiply, FFT, generic SIMD and f16; hand-written quantized kernels, KV attention and Metal. Pick `gemm` or `matrixmultiply` (and Accelerate on macOS or not) by a benchmark in package C3 |
| Minimum Rust version | 1.98 (NEON dot-product intrinsics), edition 2024 |
| `Backend` shape | the `Model` / `SlotState` split of §30.4, agreed in a design note with both backend owners |
| Server | hand-written HTTP on OS threads for `jllm serve` first (it is 733 lines in Java and keeps the engine free of async); decide on axum when the console backend is ported |
| JSON | port `Json` (445 lines) with an ordered map; add Java's number formatting only if the replay tests need it |
| Metal binding | keep `jllm_metal.m` through `cc` until the Metal backend passes its gates, then move to `objc2-metal` |
| Cache compatibility | quant cache byte-compatible with Java (shared fixtures and a migration path); `MetalTuner` and expert-placement caches may be rebuilt |
| Seeded sampling | port `SplittableRandom`, so that `seed` gives the same text as the Java server |
| Scope order | text LLM first; media and console after the serving features |

### 30.9.2 Crates

The Java packages have two import cycles that a Cargo workspace cannot have, and both need a small move:

- **model ↔ kv.** `Backend.kvSequence()` returns `kv.KvSequence` and `Backend` calls `kv.KvReports`, while
  `kv/KvBlockCodec` imports `model.KvDtype` and `model.KvQ4`. Fix: `KvDtype`, `KvQ4` and `KvSparse` move into the
  kv crate, which then sits below the model crate.
- **config ↔ engine.** `Settings` imports `engine.Speculator` and `sampling.SamplingParams` (and refers to
  `KvBlockCodec.Format`, `KvTierConfig`, `MoeSplit` and `AudioSettings`), while `SlotPool` uses `Settings`. Fix:
  the base crate keeps only the raw key/value settings with aliases, sources and validation; each consumer crate
  reads its own typed configuration from it.

`ModelLoader` imports cpu, metal, engine and server, so it belongs at the top, in the engine crate, with its three
server references moved to the server. The metal package imports three cpu classes (`ThreadPool`, `MoeExperts`,
`Qwen3NextCpuBackend`), a one-way edge that stays.

```
jllm/                       Cargo workspace, bottom to top
├─ crates/jllm-base         json, settings (raw), errors
├─ crates/jllm-formats      safetensors, QType / Ggml / Q2K / Q6K / QTensor, GGUF file, quant cache file
├─ crates/jllm-tokenizer    tokenizers, GGUF vocabularies, chat templates, prompt cache
├─ crates/jllm-kv           KvDtype / KvQ4 / KvSparse, pool, sequence, prefix tree, spill and tiers
├─ crates/jllm-model        Architecture, Families, layer and weight descriptions, the Backend trait,
│                           ModelWeights, converter, GGUF mapping, hub downloader
├─ crates/jllm-cpu          thread pool, kernels (std::arch), KV caches, CPU backends
├─ crates/jllm-metal        build.rs: ObjC shim via cc (later objc2-metal), include_str!(".metal"),
│                           GpuDevice (Metal + emulator), Metal backends, planner
├─ crates/jllm-engine       sampling and constraints, Engine, SlotPool, scheduler, speculation, loader
├─ crates/jllm-media        vision, audio, diffusion, video (features per modality)
├─ crates/jllm-server       HTTP + OpenAI API, media routes, console backend behind `web`
├─ crates/jllm-cli          the `jllm` binary (replaces the ./jllm launcher)
└─ xtask                    golden-dump comparison, fixture tools (HfToGguf, GgufTinyWrite), emulator build
features: metal, emulator, web, vision, audio, diffusion
```

### 30.9.3 Three vertical slices first

Each slice is the smallest set of classes that produces an observable result, traced from the code. Porting them
in this order gives a running program early and exercises the design decisions on little code.

| Slice | What runs | Java classes needed | Java lines |
|---|---|---|---|
| A. First token on the CPU | greedy decoding of a Qwen3 GGUF (Q4_0 / Q8_0) with a flat f32 KV cache, from a ~100-line driver | `GgufFile`, `GgufModel`, `GgufWeights`, `GgufArchitecture`, `GgufFamilies` (Qwen3 part), `GgmlCodec`, `GgmlType`; `ModelWeights`, `QTensor`, `QType`, `Ggml`, `TensorSource`, `KvDtype`; `Architecture`, `LayerSpec`, `Decoder`, `DecoderLayer`, `Attention`, `AttnWeights`, `Mlp`, `MlpWeights`, `Rope`, `RopeTable`, `RopePositions`, `Scaling`, `Backend`; `Tokenizer`, `GgufTokenizer`, `GgufPre`, `AddedTokenTrie`, `LongLongMap`, `Normalizers`, `PreTokenizers`, `StreamDecoder`; `Json`; `CpuBackend`, `CpuMatmul`, `GgufKernels`, `VectorKernels`, `ThreadPool`, `KvCache` (base + F32) | ~13,900 |
| B. First served request | `jllm serve` with one slot: `/v1/chat/completions` with SSE, no speculation, no constraints | `Main`, `Settings`, `ModelLoader` (CPU path), `ContextBudget`, `HttpServer`, `ApiServer`, `OpenAiFeatures`, `WebUi`, `Engine`, `SlotPool`, `Admission`, `Priority`, `TokenLogprob`, `Sampler`, `SamplingParams`, `TokenConstraint`, `ChatTemplate` | ~7,200, about 4,500 once the speculation, cache-file and media branches are left out, plus 892 for `ChatTemplate` |
| C. First token on Metal | a dense Qwen3 decode step on the GPU, reusing slice A's loading | `jllm_metal.m`, `Metal`, `MetalKernels`, `DecodeProgram`, `KernelParams`, `Params`, `MetalConstants`, `MetalMemory`, and the dense path of `MetalBackend` (~1,500 of 3,436 lines) | ~5,300 |

Slice A does not need `ModelLoader`, the int8 kernels, the native library, the paged cache, MoE or MLA
(`CpuBackend` only references them, so they are stubbed), nor `Unigram` / `SpmGreedy` (Qwen3 uses BPE). Slice B
needs no kv class at all, because the flat cache lives in `cpu/`, and no `BatchGroup` with one slot. Slice C uses
about 12–14 of the kernel families in `kernels.metal` (token embedding, `rmsnorm`, the `matmul` / `matmul3` /
`matmul_swiglu` families for Q4 and Q8, `rope`, `kv_put`, `attention_dec`, `argmax`); the MoE, MLA, Mamba,
quantized-KV and sparse kernels wait for later packages.

### 30.9.4 Work packages

Each package is one to five engineer-weeks, lists the Java it replaces, and is done when its acceptance checks
pass. "Lines" are Java lines (`wc -l`). Test names are classes under `src/test/java/org/robomiri/jllm/`, to be
re-expressed as `cargo test` cases with the same inputs and the same assertions.

**Foundations**

| ID | Contents | Lines | Needs | Acceptance | Weeks |
|---|---|---|---|---|---|
| P0 | Workspace, CI (Linux + a Mac runner), the Java golden-dump tool, tiny-model fixtures as safetensors and GGUF, comparison tooling in `xtask` | new | none | dumps exist for every fixture; CI runs an empty test suite on both platforms | 3 |
| F1 | `Json`, `SafeTensors` | 842 | none | golden JSON round trips; safetensors tensors equal the Java dump | 1 |
| F2 | `QType`, `Ggml`, `Q2K`, `Q6K`, `QTensor` | 1,385 | F1 | `Q2KCheck`, `Q6KCheck`, `QuantFormatsTest`; dequantized rows bit-exact | 1.5 |
| F3 | `GgufFile`, `GgmlType`, `GgmlCodec`, `TensorSource` | 867 | F1, F2 | `GgufFileTest` | 1 |
| B1 | `Settings` (raw), the bundled `jllm.properties` (465 lines), `.properties` parsing | 466 | F1 | `jllm config` prints the same values and sources as Java | 1 |

**Tokenizer and models**

| ID | Contents | Lines | Needs | Acceptance | Weeks |
|---|---|---|---|---|---|
| T1 | `Tokenizer`, `AddedTokenTrie`, `LongLongMap`, `Normalizers`, `PreTokenizers`, `Unigram`, `SpmGreedy`, `StreamDecoder` | 2,530 | F1 | `TokenizerFamiliesTest`; differential fuzzing against the Java build | 3 |
| T2 | `GgufTokenizer`, `GgufPre`, `PromptCache` | 912 | T1, F3 | `GgufTokenizerTest`, `PromptCacheTest`, `scripts/gguf-tokenizer-check.sh` | 1.5 |
| T3 | `ChatTemplate` (20 formats) | 892 | T1 | `ChatRender` output equal to Java for every format | 1.5 |
| M1 | `Architecture`, `Families`, hybrid / Mamba / Qwen3-Next descriptions, `LayerSpec`, `Scaling`, the RoPE classes, `GemmaMath` | 2,975 | F1, F2 | `ArchSmoke`, `ArchRefCheck`, `GemmaChecks`, `scripts/hf-arch-check.sh` | 2.5 |
| M2 | `Decoder`, `DecoderLayer`, `Attention`, `AttnWeights`, `Mlp`, `MlpWeights`, `Router`, `DeepSeekRouter`, `MtpFeeder`, and the `Backend` trait with the `Model` / `SlotState` split | 1,488 | M1, K0 | design note accepted; a mock backend passes the engine's unit tests | 2 |
| M3 | `ModelWeights`, `ShardProvider`, `QuantFile` | 919 | F2, M1 | `RealWeightsCheck`; a Java-written cache opens, a Rust-written cache is byte-identical | 1.5 |
| M4 | `ModelConverter`, `Fp8Quant`, `MlxQuant` | 1,390 | M3 | `ConverterTest`, `GgufVsSafetensors` | 3 |
| G1 | `GgufArchitecture`, `GgufFamilies`, `GgufWeights`, `GgufQwenNext`, `GgufModel` | 2,109 | F3, M1, M3, T2 | `GgufModelTest`, `GgufFamiliesCheck`, `GgufArchSmoke`, `scripts/gguf-llamacpp-check.sh` | 3 |
| H1 | `HuggingFaceHub` | 843 | F1 | `GgufHubTest`; resume, SHA-256 mismatch and redirect cases against a local test server | 1.5 |
| L1 | `ModelLoader`, `ContextBudget` | 1,140 | G1, M4, T3, H1, C6 | `AgentDefaultsTest`, `SmokeTest` | 1.5 |

**CPU backend**

| ID | Contents | Lines | Needs | Acceptance | Weeks |
|---|---|---|---|---|---|
| C1 | `ThreadPool`; the int8 kernels and activation quantizer of `jllm_cpu.c` and its `.inc` files as Rust intrinsics, with runtime dispatch (replaces `NativeCpu`) | 2,085 | F2 | `NativeCpuCheck` against scalar references on every dispatch level | 3.5 |
| C2 | the remaining block kernels of `Int8Kernels`, `GgufInt8Kernels`, `KQuantKernels`, `Q2KKernels`, `Q6KKernels`, including x86 versions of the 7 types that have none; `candle-core` as a second oracle | 1,249 | C1 | `GgufKernelsCheck`, `JitStableKernelsTest` (its fixed-order sums) | 1.5 |
| C3 | `VectorKernels`, `GgufKernels`, `CpuMatmul` (float dots, fast math, norms, decode and prefill matmul) on a generic SIMD crate; a library matrix multiply for F32 / F16 / BF16 weights, chosen by benchmark | 1,838 | C1 | `GgufKernelsCheck`; results within tolerance of the Java dump | 2.5 |
| C4 | `KvCache` (f32, f16, q8, q4, rotated keys, tiled attention, save / restore), `KvQ4Sim`, `CpuKvStorage` | 2,824 | C3, K0 | `LongContextTest`, `KvStateTest` | 3 |
| C5 | `PagedKv`, `PagedKernels`, sparse block attention | 1,175 | C4, K1 | `PagedKvTest`, `PagedKvChecks`, `SparseKvTest`, `PagedEngineTest` | 2 |
| C6 | `CpuBackend`, `MoeExperts`, `MlaCpu` | 2,099 | C3, C4, M2 | `GgufBackendsTest`, `MixedPassChecks`, `DeepSeekRef` | 3 |
| C7 | `HybridCpuBackend`, `MambaCpuBackend`, `Qwen3NextCpuBackend` | 1,699 | C6 | `Qwen3NextChecks`, `GgufQwenNextTest` | 2.5 |
| C8 | Performance parity: tok/s against the numbers of docs/07 and docs/11, profile-guided builds, core pinning | — | C6 | decode and prefill tok/s not below Java on the reference machines | 2 |

**Metal backend**

| ID | Contents | Lines | Needs | Acceptance | Weeks |
|---|---|---|---|---|---|
| X1 | `jllm_metal.m` through `cc`, `Metal`, `MetalConstants`, `Params`, `KernelParams`, `MetalMemory`; `GpuDevice` with the Metal and emulator implementations | 1,464 | none | `MetalParamsTest`; `scripts/metal-emu/kernel_tests.py` | 1.5 |
| X2 | `MetalKernels`, `DecodeProgram` | 2,321 | X1 | `metal/GgufMetalCheck` on the emulator | 4 |
| X3 | `MetalBackend`: prefill, decode, replay, pipelined decode, GPU sampling, batched and mixed passes | 3,436 | X2, M2, K0 | `GgufBackendsTest`, `MixedPassChecks`, `KvStateTest`; CPU = Metal greedy ids | 5 |
| X4 | `MlaMetal`, `HybridMetalBackend`, `MambaMetalBackend`, `Qwen3NextMetalBackend`, `MoeSplit` | 1,892 | X3, C6, C7 | `scripts/arch-smoke.sh` with `METAL_EMU=1`, every family | 3 |
| X5 | `MetalTuner`, `MemoryPlanner`, `MemoryGovernor`, `DeviceLedger`, `ExpertPlacement` | 1,325 | X1 | `metal/MemoryPlannerTest`, `metal/ExpertPlacementCheck` | 2 |
| X6 | `StreamedKv`, `MetalKvStorage` | 519 | X3, K1 | `KvStateTest`, `TierChecks` | 1 |
| X7 | Emulator in CI: build, library selection, the kernel tests and arch smoke as CI jobs | — | X2 | the emulator gates of `wip/README.md` run on Linux without a JVM | 2 |
| X8 | Hardware parity and tuning on Apple Silicon; optional move to `objc2-metal` | — | X3 | `scripts/mac-validate.sh`, `metal/GgufMetalReal`, the exactness md5 on CPU and Metal, tok/s not below Java | 3 |

**KV, sampling, engine, server**

| ID | Contents | Lines | Needs | Acceptance | Weeks |
|---|---|---|---|---|---|
| K0 | `KvDtype`, `KvQ4`, `KvSparse` and the `KvSequence` interface the `Backend` trait refers to | ~300 | F2 | compiles; unblocks M2, C4, X3 | 0.5 |
| K1 | `KvPool`, `KvSequence`, `KvStorage`, `KvPools`, `KvSummary`: arena prefix tree, refcounts, copy on write, the unsafe block store | 1,159 | K0 | `KvPoolTest`, `PagedKvChecks`; loom on the pool; a million-position test | 2.5 |
| K2 | `KvBlockCodec`, `FileSpill`, `TieredSpill`, `CodecSpill`, `KvTierConfig` | 1,123 | K1 | `KvTierTest`, `TierChecks`, `TierEngineTest`; Java-written block files load | 1.5 |
| S1 | `Sampler`, `SamplingParams`, `TokenConstraint`, the `SplittableRandom` port | 715 | none | the same seed gives the same tokens as Java on dumped logits | 1 |
| S2 | `TokenTrie`, `ByteConstraint` | 620 | S1, T1 | `ConstraintBench` masks equal to Java | 1 |
| S3 | `JsonConstraint` | 1,070 | S2 | schema fixtures accept and reject the same byte sequences as Java | 1.5 |
| S4 | `ToolCallConstraint` | 432 | S3, T3 | `ToolCallTest`, `ToolCallCheck`, `scripts/tool-call-check.sh` | 1 |
| E1 | `Engine`, `TokenLogprob`, `Priority` | 1,398 | M2, S1, T1 | `EnginePipeTest`, `SmokeTest` | 3 |
| E2 | `SlotPool`, `Admission`, `PassPlanner` | 805 | E1 | `SchedulerPolicyTest`, `ServerSlotsTest`, on a deterministic clock | 2 |
| E3 | `BatchGroup` (batched and mixed passes, priorities, pausing) | 792 | E2, C6 | `BatchDecodeTest`, `MixedSchedulerTest`, `MixedPassChecks` | 2.5 |
| E4 | `CacheFile`, `MediaPrompt` | 260 | E1 | saved conversations written by Java restore | 0.5 |
| E5 | `Speculator`, `NgramIndex`, `DraftModel` | 805 | E1 | `SpeculationTest`, `DraftSpecTest`, `NgramIndexCheck` | 2 |
| SV1 | `HttpServer`, `WebUi` | 791 | B1 | `ServerTest` connection cases: keep-alive, timeouts, disconnect, HTTP/1.0 | 1.5 |
| SV2 | `ApiServer` | 741 | SV1, E2 | `ServerTest`; replay of recorded chat and completion transcripts | 1.5 |
| SV3 | `OpenAiFeatures` | 421 | SV2, S3 | replay of `response_format`, tools, logprobs, stop cases | 1 |
| SV4 | `ImageHttp`, `VideoHttp`, `AudioRoutes`, one `Multipart` | 482 | SV2, media | replay of the media endpoints | 1 |
| CL1 | `Main` and the launcher: 10 subcommands, help text, exit codes | 390 | L1, SV2 | the commands of docs/28 run unchanged | 1 |
| E9 | Integration: concurrency stress, cancellation, a soak test with several slots | — | E3, K2, SV3 | no deadlock or leak in a 24-hour mixed workload | 2.5 |

**Vision, audio, diffusion**

| ID | Contents | Lines | Needs | Acceptance | Weeks |
|---|---|---|---|---|---|
| V1 | Image decoding, `Preprocess`, `Resize`, `ImageParts`, `ImageFetcher` | 940 | F1 | preprocessing equal to the dumps (PNG inputs) | 1 |
| V2 | `ClipModel`, `Mmproj`, `ClipCpu`; `VMath` on the matrix-multiply library | 1,591 | V1, F3, C3 | `VisionTest`; `scripts/vision-llamacpp-check.cpp` | 1.5 |
| V3 | `VisionContext`, `MediaChat` (embeddings in the prompt, M-RoPE) | 399 | V2, E1 | `VisionTest` end to end | 1 |
| V4 | `ClipMetal` | 714 | V2, X1 | CPU = Metal embeddings within tolerance | 1.5 |
| A1 | `Wav`, ffmpeg decode / encode; `Fft` replaced by `rustfft`; `LogMel` and `Resampler` ported or from `mel_spec` / `rubato` | 1,079 | none | `AudioTest` front-end cases | 0.5 |
| A2 | Whisper on the CPU, on the matrix-multiply library | 1,242 | A1, C3 | `AudioTest`; `scripts/audio-accuracy.py` | 1 |
| A3 | `WhisperTranscriber`, `WordAligner`, `Transcript` | 1,342 | A2 | transcripts and timestamps equal to Java on the fixtures | 1.5 |
| A4 | OuteTTS, `WavTokenizerDecoder` (library FFT and matrix multiply), `TimeStretch` | 1,371 | A1, E1 | `AudioTest` speech cases | 1.5 |
| A5 | `AudioApi`, `AudioCli`, `AudioModels`, `AudioSettings` | 517 | A3, A4 | endpoint replay | 0.5 |
| A6 | Whisper on Metal | 731 | A2, X1 | CPU = Metal within tolerance | 1.5 |
| D1 | `Ops`, `CpuOps` (conv2d as im2col + library sgemm, GroupNorm, attention variants) | 1,599 | C3 | each op within tolerance of the Java dump on the tiny fixtures; `KernelBench` | 1 |
| D2 | `ClipText`, `ClipTokenizer`, `Umt5Encoder`, `UnigramTokenizer` | 1,058 | D1, T1 | embeddings equal to the `ref-*` fixtures | 1.5 |
| D3 | `Scheduler`, `UNet`, `Vae`, `Rng` | 1,699 | D1 | `DiffusionTest`, `DiffusionCompare` | 2 |
| D4 | `ImagePipeline`, `Imaging`, `ImageApi`, `ImageStore`, `Vp8l` | 2,005 | D2, D3 | `DiffusionTest`; WebP bytes identical to Java | 2.5 |
| D5 | Wan 2.1: `WanTransformer`, `WanVae`, `FlowScheduler`, `WanPipeline` | 884 | D2 | `WanTest`, `WanCompare` | 2 |
| D6 | `VideoPipeline`, `VideoEncoder` (GIF, WebP, MP4), `VideoApi`, `VideoCli` | 1,561 | D4 | `VideoTest`; GIF and WebP bytes identical to Java | 2 |
| D7 | `MetalOps` | 808 | D1, X1 | CPU = Metal images within tolerance | 1.5 |
| D8 | Media parity harness against diffusers, whisper.cpp and llama.cpp | — | D4, A3, V3 | the reference scripts pass against the Rust binary | 2 |

**Console, tooling, release**

| ID | Contents | Lines | Needs | Acceptance | Weeks |
|---|---|---|---|---|---|
| W1 | Console skeleton: static files, settings, logs, API-key filter | 1,298 | SV2 | the page loads; headers and compression as in docs/29 | 1.5 |
| W2 | `ModelManager`, `LoadedModel`, `KvStore`, download jobs, `/api/memory` | 1,800 | W1, L1, K2 | the web model and KV tests (MockMvc cases re-expressed) | 2.5 |
| W3 | Chat: `/api/chat` streaming with the tool loop and `<think>` splitting, the OpenAI controller | 1,491 | W2 | replay of recorded `/api/chat` streams | 2 |
| W4 | Built-in tools, integration configs (Continue, Aider, Cline, OpenCode) | 1,630 | W3 | tool tests | 1.5 |
| W5 | MCP client (stdio, HTTP, SSE) and media tools | 1,585 | W4 | MCP tests against a stub server | 2 |
| W6 | Image, video and audio controllers | 1,161 | W3, SV4 | endpoint replay | 1 |
| W7 | Front end unchanged: the node JS tests and a browser pass over the 9 tabs | — | W6 | JS tests pass; no change to any static file | 1.5 |
| R1 | The remaining benchmark and study programs as `cargo bench` / examples; `GgufGreedy`, `GgufTokenize`, `GgufTinyWrite`, `HfToGguf` as `xtask` commands | ~6k | G1 | the shell checks in `scripts/` run without a JVM | 3 |
| R2 | Docker image, entrypoint, release builds for Linux x86-64 / arm64 and macOS arm64, docs updated | — | CL1 | `scripts/docker-build.sh` equivalent; docs/19 and docs/28 commands verified | 1.5 |

Totals: foundations 7.5, tokenizer and models 21, CPU 20, Metal 21.5, KV / sampling / engine / server 27.5,
media 26, console and release 16.5 — **140 engineer-weeks**. The same total by the areas of §30.7: models,
formats, tokenizer and hub 24.5; CPU 20; Metal 21.5; engine, KV, sampling, server and config 28.5 (with B1 and
K0); media 26; console, tests, build and packaging 19.5 (with P0). Without the math libraries of §30.5 the CPU
and media figures would be 21 and 29, and the total 144.

### 30.9.5 Milestones

Each milestone ends with a gate that the Java engine passes today, so progress is measured against the original
and not against the plan.

| # | Milestone | Packages | Gate |
|---|---|---|---|
| 0 | Foundations | P0, F1, F2, F3, B1, K0 | golden dumps exist; dequantized rows and GGUF metadata equal Java bit for bit |
| 1 | First token on the CPU (slice A) | T1, T2, M1 and G1 (Qwen3 part), M2, M3, C1 (pool), C3, C4 (f32), C6 (dense) | greedy ids equal Java on the tiny fixtures and on Qwen3-0.6B; `scripts/gguf-llamacpp-check.sh` passes |
| 2 | First served request (slice B) | T3, S1, E1, E2, SV1, SV2, SV3, L1, CL1 | `ServerTest` and `SmokeTest` cases; recorded chat-completion transcripts replay |
| 3 | First token on Metal (slice C) | X1, X2, X3 (dense), X7 | emulator kernel tests pass on Linux; CPU = Metal greedy ids for Qwen3 |
| 4 | CPU breadth | C1 (kernels), C2, C4 (all dtypes), C6 (MoE, MLA), C7, M1 and G1 (all families), M4, H1 | `scripts/arch-smoke.sh` on the CPU for every family; converter caches byte-identical; `scripts/hf-arch-check.sh` |
| 5 | Metal complete | X3 (rest), X4, X5, X6, X8 | arch smoke with `METAL_EMU=1`; `scripts/mac-validate.sh`; the exactness md5 on CPU and Metal; tok/s not below Java |
| 6 | Serving features | K1, K2, C5, E3, E4, E5, S2, S3, S4, E9 | the KV, scheduler, speculation and tool-call tests; `scripts/tool-call-check.sh`; the multi-slot numbers of docs/10 and docs/24 reproduced |
| 7 | Media | V1–V4, A1–A6, D1–D8, SV4 | `VisionTest`, `AudioTest`, `DiffusionTest`, `VideoTest`; the reference scripts |
| 8 | Console | W1–W7 | the node JS tests; `/api/*` replay; the console's web gate of `wip/README.md` |
| 9 | Release | C8, R1, R2 | all six gates of `wip/README.md` without a JVM; Docker image; docs updated |

After milestone 6 the Rust build is a complete text LLM server and can replace `./jllm serve`; milestones 7
and 8 can be dropped or delayed without affecting it.

### 30.9.6 Who works on what, and when

With four engineers, four tracks run in parallel after milestone 0. The only package that has no predecessor
besides the foundations is X1, so the Metal track can start against the emulator on day one.

| Weeks (approx.) | Track 1: formats, models, tokenizer | Track 2: CPU | Track 3: Metal | Track 4: engine, server |
|---|---|---|---|---|
| 1–4 | P0, F1–F3 | C1 | X1, X2 | B1, K0, S1, SV1 |
| 5–12 | T1, T2, M1, M2, M3, G1 (**M1 at ~week 12**) | C3, C4, C6 | X2, X3 dense, X7 (**M3 at ~week 14**) | T3, E1, E2, SV2, SV3, CL1 (**M2 at ~week 14**) |
| 13–24 | M4, H1, L1, rest of M1 / G1 | C2, C5, C7 (**M4 at ~week 22**) | X3 rest, X4, X5, X6, X8 (**M5 at ~week 26**) | K1, K2, S2–S4, E3–E5, E9 (**M6 at ~week 28**) |
| 25–40 | V1–V3, D2, D5, R1 | D1, D3, D4, D6, C8 | V4, A6, D7, D8 | A1–A5, SV4, W1–W7 |
| 41–46 | hardening, R2, performance, documentation (**M9**) | | | |

Roughly 46 weeks, or 10–11 months, before holidays and the unplanned. The critical path runs through M2 (the
`Backend` design): the CPU backend, the Metal backend and the engine all wait on it, so it gets the first design
review.

### 30.9.7 Rules for every package

- A package is done when its Java checks, re-expressed as Rust tests on the same inputs, pass in CI, and its
  golden comparisons pass in the category of §30.8 (bit-exact or tolerance).
- `unsafe` is allowed only in the kernel modules, the thread pool, mmap wrappers, the KV block store and the
  Metal device layer. Each block states the invariant it relies on; everything else is `#![forbid(unsafe_code)]`.
- Every hand-written kernel has a scalar reference and is tested against it on each dispatch level the CI
  machine offers. A library call in the compute path is benchmarked against the Java figure before it is kept.
- A package that changes a file format or the wire contract fails review; such changes are separate, after parity.
- Performance is checked when a package lands, against the Java figure in the docs for the same model and
  machine, not at the end.
- The Java repository stays the specification until milestone 9: a behaviour question is settled by running the
  Java build, not by reading this page.

---

## 30.10 Effort

| Area | Engineer-weeks |
|---|---|
| Models / GGUF / safetensors / tokenizer / hub | 24.5 |
| CPU backend + native kernels | 20 |
| Metal backend (shaders reused) | 21.5 |
| Engine / KV / scheduler / sampling / server / config | 28.5 |
| Vision / audio / diffusion / video | 26 |
| Web console backend, test tooling, build, Docker | 19.5 |
| **Total** | **140** (range ~125–150) |

These are single-engineer estimates per package by an experienced Rust/SIMD/Metal engineer, with tests included.

A cross-check by volume: ~73k lines of engine code, ~9k of console backend and ~29k of tests, less the ~1.6k
that disappear (the duplicated int8 kernels and the FFM layer), is about 109k lines to carry over. A port against
an existing, tested implementation usually moves at 1,000–1,200 finished lines per engineer-week, which gives
**~90–110 weeks**, or up to ~125 with the restructuring of §30.4 and the parity harness. The bottom-up figure is
higher because each package carries its own debugging and tuning reserve. Treat ~95–125 as the optimistic range
and ~125–150 as the planning figure.

The text LLM server alone (milestones 0–6 and 9: everything except the media packages, SV4 and W1–W7) is 101
of those weeks.

---

## 30.11 Risks

| Risk | Mitigation |
|---|---|
| Unsafe KV / GPU / mmap code introduces memory bugs that Java's bounds checks and GC prevented | Keep it in a small number of audited modules. Use `debug_assert!` bounds checks, Miri on the pure logic, loom on the pool, ASan builds in CI |
| Tokenizer regex/Unicode divergence | Differential fuzzing against Java and HF `tokenizers` before any model work depends on it (package T1) |
| Silent shader/host parameter mismatch | `#[repr(C)]` params generated from one definition. Every kernel exercised through the emulator against numpy references (`kernel_tests.py`) |
| The `Backend` redesign turns out wrong after both backends depend on it | Slices A and C (§30.9.3) exercise it on ~19k lines before the remaining ~50k are ported |
| Scheduler behaviour drifts (priorities, aging, preemption) | Deterministic-clock tests, recorded traces from the Java scheduler |
| Performance regressions on hot paths (no JIT profile-guided inlining) | Benchmark each kernel against the Java numbers in docs/07, 08 and 11 as it is ported. Consider PGO builds |
| x86 kernels for the 7 types the C library never covered | Scalar path first (correct, slow), intrinsics in package C2; measure against the Vector API numbers |
| Scope: 22 families × 15 quant formats × 2 backends | Tiny-model fixtures per family from day one. Port families in the order of docs/28 |
| No CI and Mac-only GPU paths | Metal emulator on Linux, plus one Apple Silicon runner (package P0) |
| The Java engine keeps changing during the port | Pin a commit as the oracle; replay later Java changes as separate, small packages |
| A math library is slower than the hand-written Java kernel, or its threads fight the engine's pool | Benchmark before adopting (C3, D1); run libraries single-threaded inside pool jobs; keep the hand-written path as the fallback |
| A library changes results enough to flip a greedy token | Libraries only in f32 paths compared by tolerance; the quantized kernels behind the exactness gate stay hand-written |
| Dependency creep outside the math | Write the §30.5 list down and review new crates |

---

## 30.12 Is it worth it?

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

A cheaper middle path is to port incrementally behind the existing C ABI. Start with package C1 as a Rust
`cdylib` replacing `libjllm_cpu`, which FFM already loads with a self-check. Then move the tokenizer and format
readers, and keep Java for the engine and server until the Rust core is proven.
