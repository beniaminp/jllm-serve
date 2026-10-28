# 28 Best settings for each model

This chapter is the "how do I run model X fastest on my machine" page. It collects, for every model family the engine runs,
the command line to start from, the settings that change its speed, memory or answers, what each setting costs and buys (as
measured, with the command that measured it), and where the defaults are already the best choice. Everything here is a
summary of the measured sections of the other chapters, which are linked next to every number; nothing below is a guess
unless it says so. Section [28.10](#2810-not-measured-yet) lists what nobody has measured yet.

Machines, unless a row says otherwise:

| name in this chapter | machine | where its numbers come from |
|---|---|---|
| **M1 Pro** | Apple M1 Pro, 10 CPU cores (8 performance + 2 efficiency), 16-core GPU, 32 GB unified memory, ~200 GB/s, macOS; Metal working set 26.8 GB | almost every Metal and CPU number of chapters 07–27 |
| **x86 VM** | a 4-core cloud VM, Intel Xeon with AVX-512 and AVX-512 VNNI, Linux, JDK 26.0.2, 15 GB | the AVX-512 / AVX2 rows ([07 §7.14](07-cpu-backend.md#714-prefill-and-decode-are-not-bit-identical-results-depend-on-history), [07 §7.16](07-cpu-backend.md#716-native-int8-kernels-libjllm_cpu-nativecpu)), the agent runs on a CPU, the needle study of [27 §27.5](27-sparse-attention.md#275-measured) |

Contents:

- [28.1 In plain words](#281-in-plain-words)
- [28.2 How a setting reaches the engine](#282-how-a-setting-reaches-the-engine)
- [28.3 The short answer: one line per model](#283-the-short-answer-one-line-per-model)
- [28.4 The settings that decide speed and memory](#284-the-settings-that-decide-speed-and-memory)
- [28.5 Per model family](#285-per-model-family)
- [28.6 Coding agents and other long shared prompts](#286-coding-agents-and-other-long-shared-prompts)
- [28.7 CPU-only machines: Linux, x86, Docker](#287-cpu-only-machines-linux-x86-docker)
- [28.8 How much memory: the formulas and two worked examples](#288-how-much-memory-the-formulas-and-two-worked-examples)
- [28.9 When it is slow: a checklist](#289-when-it-is-slow-a-checklist)
- [28.10 Not measured yet](#2810-not-measured-yet)
- [28.11 Code and sources](#2811-code-and-sources)

---

## 28.1 In plain words

🧸 A model is a very big recipe book, and the engine is a cook who reads it once for every word it says. How fast the cook
talks depends on three things. **Where it cooks**: the graphics chip of a Mac (Metal) reads the book about five times faster
than the main chip. **How small the book is printed**: four-bit weights (`q4`) are half the paper of eight-bit ones (`q8`), so
the cook turns fewer pages, and the answers barely change. **How much it must remember**: every word already said is kept
on a shelf (the KV cache); a longer conversation is a longer shelf, which takes memory and a little time to look along.

Most of the defaults were chosen by measuring, so the best start for almost every model is simply:

```sh
./jllm chat --model <the model>
```

and the settings below are for the cases the defaults cannot guess: a machine without a GPU, a model that does not fit, an
agent that resends a 10 000-token prompt every turn, or a wish for exact, repeatable numbers.

📐 Decoding one token reads every weight once, so the decode speed is bounded by memory bandwidth / bytes of weights:

```
decode tok/s  ≤  bandwidth (bytes/s)  /  bytes read per token
             ≈  200 GB/s / 0.45 GB  ≈  440 tok/s   (Qwen3-0.6B Q4_0 on the M1 Pro, the ceiling)
measured:      ~143 tok/s on Metal (32 % of it), ~51 tok/s on the CPU with int8 activations (11 %)
```

([11 §11.1](11-performance.md#111-measured).) Prefill (reading the prompt) is bounded by arithmetic instead, which is why
it runs 10–20 times faster per token than decode and why it gains from bigger chunks of tokens per pass
(`backend.batch`). Every setting in this chapter acts on one of these two bounds, or on memory.

---

## 28.2 How a setting reaches the engine

Settings come from four places, the later one winning ([10 §10.10](10-engine-and-server.md#1010-configuration)):

```
 bundled src/main/resources/jllm.properties     (every key, its default and a comment)
        ▼  overridden by
 an external file: --config FILE, else $JLLM_CONFIG, else ./jllm.properties
        ▼  overridden by
 the command line: --key value or --key=value  (short aliases: --ctx, --batch, --quant, --model, --backend ...)
        ▼
 Settings.validate(): unknown keys and bad values are refused before anything loads (exit code 2)
```

`./jllm config [--key value ...]` prints the effective value of every key and where it came from. JVM switches
(`-Djllm.…`, documented as "JVM" below) go in `JAVA_OPTS`.

**The launchers** set the Java memory, which matters for the CPU backend:

| launcher | Java heap | direct-memory limit (paged CPU KV cache) | other |
|---|---|---|---|
| `./jllm` | `-Xmx${JLLM_XMX:-4g}` | the physical memory (`scripts/jvm-memory.sh`), or `JLLM_DIRECT_MEMORY` | `-XX:+UseParallelGC`, `JAVA_OPTS` appended |
| `jllm-web/run.sh` | `-XX:MaxRAMPercentage=60`, or `-Xmx$JLLM_XMX`; young generation `-Xmn${JLLM_XMN:-512m}` | the same | `JAVA_OPTS` appended |
| Docker (`docker/entrypoint.sh`) | `-XX:MaxRAMPercentage=${JLLM_HEAP_PERCENT:-60}` of the container limit | the container's memory limit (cgroup v2 `memory.max`, else v1), or `JLLM_DIRECT_MEMORY` | `JLLM_CPU_NATIVE` sets `-Djllm.cpu.native` |

🧸 The CPU keeps a one-conversation cache inside Java's own box (the heap) and a shared, paged cache outside it, in a second
box (direct memory) whose size Java also limits. Java's default for the second box is the size of the first (4 GB with
`./jllm`), far smaller than what `backend.ctx=auto` plans from the RAM, so the launchers now make the second box as big
as the RAM and let the context planner decide how much of it a cache really takes ([10 §10.13](10-engine-and-server.md#1013-the-default-context-length-backendctxauto)).

| environment variable | what it sets | when to change it |
|---|---|---|
| `JLLM_XMX` | the Java heap (flat CPU KV cache, vision / audio floats) | a long context on the CPU with one slot: the flat cache must fit in 60 % of the free heap |
| `JLLM_DIRECT_MEMORY` | `-XX:MaxDirectMemorySize` (`16g`, `8192m`, or `default` = the JVM's own limit) | to keep the paged CPU pool below a size (another program needs the RAM) |
| `JLLM_XMN` | young generation of `jllm-web/run.sh` | never needed so far |
| `JLLM_HEAP_PERCENT` | Docker heap share of the container limit | a container that runs only small models and wants more page cache |
| `JAVA_OPTS` | any JVM switch: `-Djllm.metal.autoBatch=512`, `-Djllm.cpu.fastmath=false`, `-XX:MaxVectorSize=32` ... | A/B tests, exact comparisons |

---

## 28.3 The short answer: one line per model

The best measured starting command for each family, and the settings to add for the case in the last column. "auto" means
leave the default; "—" means the setting does not apply. Speeds are tokens per second on the M1 Pro unless marked.

| model (family) | command | backend | weights | KV cache | context | prefill / decode (measured) | add when |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B … 32B, Qwen2 / 2.5, R1-Distill-Qwen (dense) | `./jllm chat --model Qwen/Qwen3-0.6B` | Metal (auto) | `q4` (default) | `f32` short, `--backend.kv-dtype q8` long | auto (40 960 for the 0.6B) | Metal ~2 100 / ~143; CPU ~275 / ~51 | `--quant q8` to check an odd answer; `--backend.kv-dtype q8` when the context is what runs out |
| Qwen3-30B-A3B (MoE) | `./jllm serve --model Qwen/Qwen3-30B-A3B-Thinking-2507 --low-disk` | Metal | `q4` (19.4 GB; `q8` does not fit 32 GB) | `f16` or `q8` for long contexts | auto (18 432 f32, 37 888 f16, 74 752 q8) | Metal ~300 / ~27; CPU ~12 / ~6 | `--batch 512` (+6 to +12 % prefill) when 2× scratch is fine |
| Qwen3.5 / "Qwen3.8-27B" hybrids (GGUF) | `./jllm chat --model unsloth/Qwen3.8-27B-GGUF --gguf-quant Q4_0` | Metal (CPU "not usable" for the 27B) | the file's | `f32` / `f16` (q8, q4 served as f16) | auto (44 032) | Metal 40.6 / 4.3; with its own MTP head 5.4 | `--speculative off` when fewer than ~61 % of its guesses are right |
| Llama 3.x, R1-Distill-Llama | `./jllm chat --model unsloth/Llama-3.2-1B-Instruct` | Metal | `q4` | as Qwen3 dense | auto | not measured (expect dense numbers scaled by size) | — |
| Mistral 7B v0.1–v0.3, Mistral Small 3 / Devstral (GGUF `mistral3`) | `./jllm chat --model <repo>` | Metal; `--backend cpu` when `head_dim` is not a multiple of 32 | `q4` | as dense | auto | not measured | — |
| Gemma 2 / Gemma 3 | `./jllm chat --model unsloth/gemma-3-1b-it` | Metal | `q4` | as dense (Metal q4 KV refused with a picture) | auto | Gemma 3 4B with a picture: decode 25.3 | — (Gemma 3's chunk is raised to ≥ 256 by itself) |
| Gemma 4 12B | `./jllm chat --model <Gemma 4 12B repo>` | Metal or CPU (512-wide heads on both) | `q4` | `f32` / `f16` / `q8` (checked on the 512-wide heads, [14 §14.10.7](14-model-families.md#14107-gemma-4-12b); q4 is served as q8) | auto | Metal 8 tok/s decode | — (no paged, batched or tiered KV) |
| Phi-3 / 3.5 / 4 | `./jllm chat --model microsoft/Phi-3.5-mini-instruct --ctx 4096` | Metal (`head_dim` 96 keeps the generic decode attention) | `q4` | as dense | **`--ctx 4096`** for short chats (the short LongRoPE factors) | not measured | `--ctx` larger only when the text is long |
| GLM-4 / GLM-Edge | `./jllm chat --model THUDM/GLM-4-9B-0414 --backend.kv-dtype f16` | Metal | `q4` | `f16` | auto | not measured | — |
| DeepSeek V2-Lite (MLA) | `./jllm chat --model deepseek-ai/DeepSeek-V2-Lite-Chat --backend.kv-dtype f16` | Metal | `q4` (≈ 10 GB) | **`f16`** (the latent cache is small; q8 has one scale per latent row) | auto | not measured | — |
| Mamba | `./jllm run --model state-spaces/mamba-130m-hf --raw --prompt "…"` | Metal | `q4` | — (state, no cache) | — | random-weight 130m: Metal ~1 060 / ~273 (loaded machine) | — |
| gpt-oss | `./jllm chat --model openai/gpt-oss-20b` (or a `gpt-oss` GGUF) | Metal | MXFP4 experts in place | `f32` / `f16` / `q8` (q4 served as q8) | auto | not measured (tiny models only) | `--think false` for low reasoning effort |
| Llama 4 (text) | `./jllm chat --model <Llama 4 repo or GGUF>` | Metal | `q4` | as gpt-oss | auto (full, not chunk-limited) | not measured | — |
| Granite 4 / Nemotron-H / Jamba | `./jllm chat --model <repo or GGUF>` (Granite has a template; the others: `/v1/completions`) | CPU or Metal | `q4` | `f32` / `f16` | auto | not measured | — (every request starts at position 0) |
| any GGUF file | `./jllm chat --model unsloth/Qwen3-0.6B-GGUF --gguf-quant Q4_0` | Metal | **Q4_0** files decode fastest; K-quants run in place | as the family | auto | Q4_0 GGUF Metal 1 903 / 134.8 | `-Djllm.gguf.requantize=none` on the CPU when only the in-place file fits |
| vision (SmolVLM, Qwen2.5-VL, Qwen3-VL, Gemma 3) | `./jllm chat --model ggml-org/Qwen3-VL-2B-Instruct-GGUF --gguf-quant Q8_0 --image a.png` | Metal (`vision.backend=auto`) | the file's; **F16 projector** | as the family | auto | Qwen3-VL-2B: encode 764 ms, decode 39.6 | `--vision.image-max-tokens N` to bound big pictures |
| Whisper | `./jllm transcribe --model openai/whisper-small --audio talk.wav` | Metal, GPU decoder for small and up (auto) | the checkpoint's | — | — | small: 61.7 s of speech in 4.71 s | `--audio.stt.fallback false` when a silent end is slow |
| OuteTTS 0.3 | `./jllm speak --model OuteAI/OuteTTS-0.3-500M --prompt "…"` | Metal LM + CPU codec | `audio.tts.quant=q8` | — | `audio.tts.ctx` 8192 | RTF 1.01 (q8), 0.81 (q4) | `--audio.tts.quant q4` for ~20 % less time |
| SD-Turbo / SDXL-Turbo | `./jllm image --image-model stabilityai/sd-turbo --prompt "…"` | Metal, f16 | — | — | — | 512²: 1.68 s (1 step) | `--image.precision f32` only to compare |
| SD 1.5 / SDXL | `./jllm image --image-model <repo> --prompt "…"` | Metal, f16 | — | — | — | UNet 0.42 s / step at 512² (SD 1.x) | `--steps 20` (the samplers are accurate at ~20) |
| AnimateDiff | `./jllm video --video-model guoyww/animatediff-motion-adapter-v1-5-2 --prompt "…" --width 256 --height 256` | Metal, f16 | — | — | — | 256², 16 frames: 3.88 s / step | `--video.base-model` = a fine-tuned SD 1.5 for good clips |

---

## 28.4 The settings that decide speed and memory

In order of how much they change, for a text model. Each table gives the default, what to change it to and when, and the
measured effect with its source. Settings without a measured effect are listed in [28.10](#2810-not-measured-yet).

### 28.4.1 `backend`: Metal or CPU

| value | what it does | measured (Qwen3-0.6B Q4_0, 413-token prompt, 128 tokens) |
|---|---|---|
| `auto` (default) | Metal on macOS when `libjllm_metal.dylib` is available, else the CPU | — |
| `metal` | every layer on the GPU | prefill ~2 100 tok/s, decode ~143 tok/s (speculation off) |
| `cpu` | Vector API kernels, int8 activations, native int8 dots | prefill ~275 tok/s, decode ~51 tok/s |

Qwen3-30B-A3B: Metal ~300 / ~27, CPU ~12 / ~6 ([11 §11.1](11-performance.md#111-measured)). **Use Metal on a Mac.** Use the
CPU to keep the GPU free for something else, on Linux, and in Docker (the image has no GPU). The other workloads choose
their device independently: `vision.backend`, `audio.backend`, `image.backend`, `video.backend` (all `auto`).

### 28.4.2 The weights: `model.quant`, `model.head-quant`, GGUF types

| key | default | choices | measured effect | best |
|---|---|---|---|---|
| `model.quant` (`--quant`) | `q4` | `q4`, `q8`, `q2k`, `q6k` | Qwen3-0.6B perplexity on a 97-token paragraph: Q8_0 7.16, Q4_0 7.37, Q2_K 35 165 ([13 §13.7](13-q2k-quantization.md#137-measurements)); Metal prefill q4 2 092 vs q8 2 105 tok/s at `--batch 256` ([08 §8.21](08-metal-backend.md)); size 0.6B: 452 MB (q4), ~670 MB (q8), 321 MB (q2k) | `q4`; `q8` to check whether an odd answer is a 4-bit artifact ([06 §6.9](06-quantization.md#69-what-the-error-does-to-the-model)); `q8` for MLX 4-bit checkpoints (keeps their values instead of re-quantizing them, [03 §3.7](03-safetensors.md)); **never `q2k` below ~7 B** ("destroyed": perplexity 35 165) |
| `model.head-quant` | `q8` | `q8`, `q6k` | 0.6B head 175 → 128 MB; Metal head matmul 1.2–1.3 → 1.8–2.0 ms; CPU head 3.4–4.0 → 18 ms (loses the int8 path); perplexity +0.28 % ([06 §6.5.1](06-quantization.md#651-the-head-format-and-q6_k)) | `q8`; `q6k` only saves memory |
| `model.low-disk` | `false` | bool | the 30B (61 GB of bf16) converts with ≈ 28 GB of disk; download-bound, ~30 minutes ([12 §12.6](12-mixture-of-experts.md#126-memory-and-disk-on-a-32-gb-laptop)) | `true` when the checkpoint is larger than the free disk |
| `-Djllm.gguf.requantize` (JVM) | Metal: nothing; CPU: Q5_1, Q3_K, F16, BF16 to Q8_0 | ggml type names, `all`, `none` | table below | the defaults; `=none` on the CPU when only the in-place file fits in memory |

**GGUF block types, in place against a Q8_0 copy** (Qwen3-0.6B requantized with `llama-quantize --pure`, M1 Pro;
tok/s in place / as Q8_0; [16 §16.9.1](16-gguf.md#1691-the-newer-block-types-in-place-apple-m1-pro-qwen3-06b-requantized-with-llama-quantize)):

| type | file / RAM as Q8_0 | Metal prefill | Metal decode | CPU decode | CPU default |
|---|---|---|---|---|---|
| Q4_1 | 379 / 670 MB | 1 426 / 1 489 | 101.0 / 99.9 | 83.7 / 92.2 (−9 %) | in place |
| Q5_0 | 416 / 670 | 1 558 / 1 466 | 103.3 / 103.4 | 85.6 / 91.8 (−7 %) | in place |
| Q5_1 | 453 / 670 | 1 375 / 1 469 | 105.1 / 102.8 | 66.2 / 91.9 (−28 %) | Q8_0 |
| Q3_K | 262 / 670 | 1 508 / 1 469 | 101.5 / 105.1 | 58.0 / 92.4 (−37 %) | Q8_0 |
| IQ4_NL | 341 / 670 | 1 356 / 1 492 | 92.0 / 107.6 | 100.1 / 95.1 (+5 %) | in place |
| IQ4_XS | 323 / 670 | 1 358 / 1 493 | 93.0 / 107.8 | 115.3 / 93.6 (+23 %) | in place |
| MXFP4 | 323 / 670 | 1 389 / 1 467 | 92.0 / 102.8 | 120.9 / 92.7 (+30 %) | in place |
| F16 | 1 198 / 670 | 1 428 / 1 468 | 81.2 / 102.9 | 22.4 / 91.7 (−76 %) | Q8_0 (lossy) |

A Q4_0 GGUF runs at the engine's own Q4_0 speed (prefill 1 903 vs 2 011 tok/s, matvecs 5.65 vs 5.52 ms, decode 134.8 tok/s;
[08 §8.30](08-metal-backend.md)). Q4_K_M decodes at about 90 tok/s on Metal ([16 §16.9.1](16-gguf.md#1691-the-newer-block-types-in-place-apple-m1-pro-qwen3-06b-requantized-with-llama-quantize))
and ~11 tok/s on 4 AVX-512 cores. **When you pick a file for speed, pick Q4_0**; pick a K-quant for its smaller error at a
size. In-place text equals llama.cpp's in 22 of 22 greedy runs; requantizing changes 3 of 22.

### 28.4.3 The context: `backend.ctx`, `backend.ctx-max`

🧸 The context is the length of the shelf that holds the conversation. A longer shelf costs memory, not speed: the engine
reads only the part that is filled.

| key | default | what it does |
|---|---|---|
| `backend.ctx` (`--ctx`) | `auto` | KV capacity per slot. `auto` sizes it from the free memory ([10 §10.13](10-engine-and-server.md#1013-the-default-context-length-backendctxauto)); a number fixes it |
| `backend.ctx-max` | `131072` | the most `auto` may pick (raise it for a 1M-trained model when memory allows) |
| `-Djllm.ctx.share` (JVM) | `0.6` | the share of the free memory `auto` gives to the caches |

📐 What `auto` computes (`ContextBudget.plan`):

```
ctx = min( backend.ctx-max,  max(max_position_embeddings, 4096),  floor(budget / (slots × bytes_per_token)) )
      rounded down to a multiple of 1024 (256 below 4096), at least 1024

budget = 0.6 × free, where free is
  Metal:            working set − weights − other loaded models − reserve (max(512 MB, working set / 32))
                    [+ 10 % of the working set for a MoE model whose experts may move to the CPU]
  CPU, one slot:    the free Java heap (the flat cache lives there)
  CPU, paged pool:  0.75 × RAM − weights, and at most 0.9 × (MaxDirectMemorySize − direct memory in use)
  tiered KV:        0.75 × RAM − weights
  every case:       at most 0.6 × (0.75 × RAM − weights)       (the weights are mapped: both must fit in RAM)
  a draft model:    budget / 2
```

Measured picks on the M1 Pro: 0.6B f32 40 960 (the trained length, 9.4 GB), 0.6B f16 40 960 (4.7 GB), 0.6B on the CPU with
`-Xmx4g` 7 168, Qwen3.8-27B GGUF 44 032 (5.8 GB), 30B f32 18 432 (3.6 GB), f16 37 888, q8 74 752. A large context costs
almost no speed: decode 133.7 tok/s at 40 960 vs 135.3 at 4 096, "about 1 %, the size of the run-to-run noise" ([10 §10.13](10-engine-and-server.md#1013-the-default-context-length-backendctxauto)).
**Leave it at `auto`**, except for Phi-3.x / Phi-4 (short chats: `--ctx 4096`, [28.5.6](#2856-phi-3--35--4)) and Qwen2.5-1M
(at most 262 144, [28.5.13](#28513-qwen25-1m)). The start-up line `Context: N tokens (backend.ctx=auto: KV … KB/token, <the budget>;
limited by …)` says which budget decided it.

### 28.4.4 The KV cache: `backend.kv-dtype` and its companions

| `backend.kv-dtype` | bytes per token (Qwen3-0.6B) | perplexity at 4 096 tokens, vs f16 21.2222 | needle (27 questions, 4k–16.8k) | Metal decode tok/s, 413 / 2 013 tokens (11.10) | Metal decode, 2 000 tokens (24.5) | CPU decode, 413 (11.9) / 2 000 tokens (24.5) |
|---|---|---|---|---|---|---|
| `f32` (default) | 229 KB | 21.2204 (exact) | — | ~143 at 413 (11.1) | 108.6 | 89.3 / **36.0** |
| `f16` | 115 KB | 21.2222 | 23 | 142.5 / 121.0 | 124.5 | 69.2 / **9.7** |
| `q8` rotated (default for q8) | 59 KB | 21.2804 (+0.0027 nats) | 23 | 139.1 / 122.9 | — | — |
| `q8` plain (`--backend.kv-q8-rotate false`) | 59 KB | 21.8289 (+0.028 nats) | 26 | 142.9 / 126.8 | 127.4 | 83.6 / **41.4** |
| `q4` (window 128) | 31.5 KB + ring | 32/128/512 window: +0.369 / +0.269 / +0.152 nats | 23 (w128), 24 (w512) | within 9 % of f16 at 413, equal at 2 000 (11.9) | — | 49.2 / — |

The two Metal columns are two measurement sessions (each compares its rows with each other, not with the other column):
docs/11 §11.10 alternated f16 / q8 / rotated q8 at 413 and 2 013 tokens; docs/24 §24.5 alternated the flat and the paged
cache at 2 000 tokens (the flat rows are quoted). The CPU's 413-token column moved by ±20 % between runs (a shared machine).

Sources: memory and quality [07 §7.13](07-cpu-backend.md#713-the-f16-kv-cache-kvcache); Metal speed
[11 §11.10](11-performance.md#1110-the-q8-cache-with-rotated-keys-measured); CPU speed [11 §11.9](11-performance.md#119-the-q4-kv-cache-measured)
(413 tokens, ±20 % noise) and [24 §24.5](24-paged-kv-cache.md#245-measured) (2 000 tokens, quiet machine). At ~3 000 tokens
on Metal: f32 92.9, f16 113.4, q8 114.4 tok/s ([08 §8.31](08-metal-backend.md#831-decode-attention-attention_dec)).

**What to choose:**

- **Metal, short conversations**: the default `f32`; every type decodes at about the same speed below ~1 000 tokens.
- **Metal, long conversations or several slots**: `q8` (rotated, the default for q8): a quarter of f32's memory, as fast as
  f16, and 0.003 nats from f16. `f16` if you want no quantization at all.
- **CPU: `q8`** (rotated, the default for q8) on both machines measured. On the M1 Pro's CPU the f16 cache decoded at
  **9.7 tok/s against 36.0 (f32) and 41.4 (q8)** at 2 000 tokens: the half-to-float conversion of the Vector API is the
  bottleneck there, so use `q8` or `f32`, not `f16`. On the x86 VM (AVX-512, F16C: the conversion is one instruction) f16 is
  faster than f32 and q8 faster again: 20.8 / 26.0 / **35.7** tok/s (f32 / f16 / rotated q8) at 2 048 tokens and
  3.59 / 4.93 / **6.80** at 16 384 ([27 §27.5.4](27-sparse-attention.md#2754-decode-speed-against-the-context-length-cpu-x86-2026-10-28)).
- **`q4`** only when memory decides; `backend.kv-q4-recent=512` "when answers matter more than prefill speed", `0` to save
  the ring and lose only 7.5 % of prefill speed against f16 ([11 §11.9](11-performance.md#119-the-q4-kv-cache-measured)).
- DeepSeek (MLA): `f16` ("the latent cache is already small, f16 is the safer choice"); q4 is served as q8.
- Families that serve another type than asked (the start-up note says so): Qwen3-Next / Qwen3.5 and Granite 4 / Nemotron-H / Jamba
  on Metal serve q8 and q4 as f16; gpt-oss and Llama 4 serve q4 as q8; Gemma 4 has no q4 or rotated q8.

`backend.kv-q8-rotate` stays `true` (perplexity cost 0.028 → 0.003 nats for 1–3 % of speed); `false` only reads q8 caches
saved by older versions ([11 §11.10](11-performance.md#1110-the-q8-cache-with-rotated-keys-measured)).

### 28.4.5 The prefill chunk: `backend.batch`

🧸 The prompt is read in spoonfuls. A bigger spoon keeps the GPU busier, but needs a bigger bowl (scratch memory) and makes
a "stop" wait for the spoon in progress.

| model and backend | `auto` gives | measured |
|---|---|---|
| dense decoder on Metal (Gemma 4 included) | 256 (`-Djllm.metal.autoBatch=N` to change) | 0.6B prefill 64 / 128 / 256 / 512: 1 426 / 1 612 / 2 012 / 2 058 tok/s, decode unchanged; 2 000-token prompt 999 / 1 155 / 1 172 (128/256/512) ([08 §8.29](08-metal-backend.md#829-the-prefill-chunk-backendbatchauto)) |
| any mixture-of-experts model (Metal and CPU) | 256 | 30B Metal 64 / 128 / 256 / 512: 158–160 / 206–208 / 262–267 / 293–306 tok/s; CPU experts 1.5–1.6× faster per token at 256 than 64 ([08 §8.33](08-metal-backend.md#833-should-the-prefill-chunk-come-from-the-tuner-backendbatchauto-stays-at-256), [11 §11.4](11-performance.md)) |
| Gemma 3 | at least 256 | a picture's 256 tokens must go through in one pass; a smaller explicit value is refused with a picture |
| dense on the CPU, Mamba, dense Qwen3-Next / Qwen3.5, the attention / state-space hybrids | 64 | — |

**Best**: `auto`. `--batch 512` on a Mac with 32+ GPU cores, or for MoE models when the extra scratch is acceptable
(+12 % on the 30B at 413 tokens, +6 % at 2 000). Smaller only when scratch memory is tight or cancel latency matters (127 ms
at 256 vs 45 ms at 64, [08 §8.29](08-metal-backend.md#829-the-prefill-chunk-backendbatchauto)).

### 28.4.6 Speculation: `engine.speculative`

🧸 Instead of saying one word at a time, the cook guesses the next few words (copied from the prompt, or proposed by a smaller
cook) and checks them all in one look. Right guesses are free words; wrong ones cost a little.

| value | what it proposes | best for | measured (agent-like prompts, [11 §11.11](11-performance.md#1111-agent-defaults-measured)) |
|---|---|---|---|
| `ngram` (default) | copies of the prompt or the output (prompt lookup), pausing when its drafts do not pay | every model with a KV cache | 0.6B Metal: code edit +106 %, tool call −1 %, prose 0 %; 0.6B CPU: edit +44 %, tool −5 %, prose −4 %; 30B Metal: edit +33…+68 %, tool −3…−10 %, prose ~−1 %; 4-core AVX-512 rewriting a file 19.9 → 81.2 tok/s ([17 §17.1](17-agent-speedups.md)) |
| `mtp` (the default when the model has its own next-token head, e.g. Qwen3.5 / Qwen3.8 GGUF) | the model's multi-token-prediction head | hybrids, whose state cannot be rewound for `ngram` | 27B: 4.3 → 5.4 tok/s (+26 %) on an 82-token function; break-even at 61 % accepted ([16 §16.14](16-gguf.md#1614-multi-token-prediction-the-model-as-its-own-draft-model)) |
| `draft` + `engine.draft-model` | a smaller model with the same tokenizer | only with ≥ 80 % agreement and a cheap verify pass | 0.6B drafting for the 30B: edit +17 %, tool −22 %, prose −42 % (not a default) |
| `off` | nothing | benchmarks against llama.cpp; Mamba (no rewind) | — |

Greedy output is identical with and without speculation; sampled output has the same distribution ([09](09-sampling.md)).
Leave `engine.draft-tokens=16` and `engine.ngram-min=3` (2-token matches cost 8 % on free text, [11 §11.6](11-performance.md#116-speculative-decoding)).

### 28.4.7 CPU kernels

| key | default | measured | best |
|---|---|---|---|
| `backend.cpu.int8-activations` | `auto` (times both on the loaded model at start and keeps the faster) | M1 Pro 0.6B: float ~237 / ~13 tok/s, int8 ~275 / ~51 (≈ 3.9× decode); ~0.9 % RMS error per dot product ([11 §11.1](11-performance.md#111-measured)) | `auto`; `false` for results identical to Metal |
| `backend.cpu.native` | `auto` (`libjllm_cpu`: NEON `sdot`, AVX2, AVX-VNNI, AVX-512 VNNI) | x86 VM, AVX-512 VNNI, 4 threads, loaded machine: one token's matmuls 76–79 ms native vs 130–188 float; decode 4.2–5.2 vs 2.4–3.3 tok/s; single-thread row kernels 2.8× (Q4_0) to 8× (Q6_K) the Java int8 ([07 §7.16](07-cpu-backend.md#716-native-int8-kernels-libjllm_cpu-nativecpu)) | `auto`; the launcher builds the library when a C compiler exists (`scripts/build-native-cpu.sh`) |
| `backend.cpu.batch-int8-rows` | `auto` | passes of 2..N rows (batched decode steps of several slots, short verification runs) row by row through the native int8 kernel, N measured once at load (3 on a 4-core x86 VM: a 2-row pass 46–49 ms instead of 65–68, a lone step 27–30; 16 on an M1 Pro: 13 ms instead of 57, 4 users on 4 slots 60 -> 130 tok/s), each row's logits its lone step's bit for bit ([07 §7.4.1](07-cpu-backend.md#741-a-few-tokens-the-single-token-int8-kernel-per-token-backendcpubatch-int8-rows)) | `auto`; a number 0..64 = fixed; `0` = the float tiles (the earlier behaviour) |
| `backend.threads` | `0` = every processor | not swept | `0`; fewer to leave cores to another program |
| `-Djllm.cpu.fastmath` (JVM) | on | decode after a 3 023-token prompt +5–9 %, bench decode 57 → 60–62 ([07 §7.15](07-cpu-backend.md#715-fast-math-jllmcpufastmath)) | on; `false` for bit-exact comparisons |
| `-Djllm.cpu.deterministic` (JVM) | off | prefill through the single-token kernels: slower | off; on only to compare prefill with decode bit for bit |

On x86 the vector width matters: the CPU kernels use every lane the JVM offers (16 floats on AVX-512, 8 on AVX2). The lane
sums of the tiled prefill kernels were rewritten this month as a butterfly of whole-vector permutes, which made the prefill
**about 8× faster on AVX2** and kept it at the old speed on AVX-512 with results that no longer depend on the JIT
([07 §7.14](07-cpu-backend.md#714-prefill-and-decode-are-not-bit-identical-results-depend-on-history); numbers in [28.7](#287-cpu-only-machines-linux-x86-docker)).

### 28.4.8 Several conversations: slots, the paged cache, batched decoding, the scheduler

| key | default | what it does | measured | best |
|---|---|---|---|---|
| `slots` (console load form, `/api/models/load`) | 1 | engines (KV caches) per model on shared weights | 2 slots: a title request no longer evicts the main conversation; 10 443 of 10 484 tokens cached, prefilled in 2.5 s (README) | **2 for agents**; more for several users |
| `server.slots` (`./jllm serve`) | 1 | the same for the plain API server: every request picks a slot, decode steps of the slots generating together share a pass | x86 VM, 0.6B, 4 users × 64 tokens: every first token within 0.8–1.1 s instead of up to 6.2–6.5 s on one slot, 38–41 tok/s together instead of ~31 ([10 §10.9c](10-engine-and-server.md#109c-several-users-at-once-serve---serverslots)) | **one per user expected at the same time**; 2 for one agent (a side request keeps the main conversation) |
| `backend.kv-paged` | `auto` = paged with ≥ 2 slots | one pool of 16-position blocks + a prefix tree; identical prefixes stored once | 8 conversations sharing 4 096 tokens: 7 663 MB flat vs 1 086 MB paged; the n-th conversation's prefill CPU 24 272 → 529 ms, Metal 6 640 → 196 ms; a resent 4 160-token conversation Metal 6 315 → 32 ms; decode paged / flat 0.96–1.02 ([24 §24.5](24-paged-kv-cache.md#245-measured)) | `auto` |
| `backend.kv-pool-tokens` | 0 (= every slot's ctx + spare) | most positions the pool keeps | not measured | raise it to keep more conversations cached than there are slots |
| `engine.batch-decode` | `true` | slots that generate together share one forward pass | 3 slots in shared passes 177 tok/s, 181–198 with `-Djllm.metal.concurrent=batch` ([08 §8.28](08-metal-backend.md#828-the-concurrent-compute-encoder-barriers-only-where-a-kernel-needs-the-one-before)) | `true` |
| `engine.scheduler` | `slots` | `mixed`: one queue per model and passes that mix a prefill chunk with every decode token and draft (needs `engine.batch-decode=true`, GQA attention, not q4) | not measured yet ([25 §25.5](25-scheduler.md#255-measured) is empty) | `slots` until the table exists |
| `engine.priority-order` | `true` | `"priority": "background"` requests give way | interactive first token 127 ms vs 937 ms behind a 1 500-token background job ([10 §10.14](10-engine-and-server.md#1014-request-priority)) | `true`; send `background` from subagents and batch jobs |

Paging does not apply to (the flat cache is used, with a note): q4 caches, Mamba, Qwen3-Next / Qwen3.5, DeepSeek MLA,
Gemma 4 (per-layer head shapes), and gpt-oss / Llama 4 (sinks, chunked attention, attention temperature) ([24 §24.4](24-paged-kv-cache.md#244-settings)).

### 28.4.9 Long contexts in little memory: sparse attention and tiers

| key | default | what it does | measured | best |
|---|---|---|---|---|
| `backend.kv-sparse` | `false` | Quest-style decode attention: each 16-position block keeps min / max key summaries; a step attends exactly over the best `kv-sparse-blocks` blocks + sink + recent window (≈ 1 300 positions) | perplexity at 8 192 / 16 384 / 32 768 tokens, 64 blocks: 29.58 vs 25.84, 8.35 vs 8.14, 3.025 vs 2.905; 128 blocks within 0.5–3 % of dense at 16 k and 32 k; needle in a haystack (CPU, 30 haystacks of 8 k – 32 k tokens): dense 26, 128 blocks 26, 64 blocks 25, 32 blocks 25, 16 blocks 19 found ([27 §27.5.3](27-sparse-attention.md#2753-needle-in-a-haystack-cpu-x86-2026-10-28)); CPU decode with 64 blocks 2.1× / 2.7× / 3.0× the dense speed at 8 k / 16 k / 32 k (7.85 against 2.62 tok/s at 32 k), 6 % slower than dense at 2 k with 128 blocks ([27 §27.5.4](27-sparse-attention.md#2754-decode-speed-against-the-context-length-cpu-x86-2026-10-28)) | off; on for contexts far beyond 1 300 tokens where a small accuracy loss is acceptable; `--backend.kv-sparse-blocks 128` costs little more time than 64 and keeps the perplexity closer to dense; never below 32 |
| `backend.kv-tier` | `false` | hot / host / SSD tiers for the paged pool; a slot larger than the hot cap streams its KV (Metal) | interim, 0.6B f16 Metal: streamed decode 58.7 tok/s (512) / 69.0 (1 024); pool tier with the planner 105.4 (1 024) → 49.3 (12 000) ([26 §26.5](26-tiered-kv-and-memory-planner.md#265-measured)) | off; on for a server that must keep many or very long conversations within a memory budget |
| `backend.memory-plan` | `false` | one GPU budget across experts, KV and scratch (needs `kv-tier` and `cpu-moe=auto`) | planner unit numbers for the 30B: 16 384 positions → 256 of 6 144 experts on the CPU, 1.0 ms / token | off; on for a big MoE model whose context grows |
| `backend.metal.cpu-moe` | `auto` | routed experts in CPU memory, run on the CPU, so a model larger than the working set fits | estimates only: a cold expert ≈ 50 µs, a whole CPU layer 0.25–0.4 ms ([08 §8.18](08-metal-backend.md#818-experts-on-the-cpu-backendmetalcpu-moe)) | `auto` (moves nothing when the model fits) |

---

## 28.5 Per model family

Each section: the command, then backend, weights, KV cache, context, chunk, speculation, memory, and what is known about
quality. "As dense" means the recommendations of Qwen3 dense (28.5.1) apply unchanged.

### 28.5.1 Qwen3 dense, Qwen2 / 2.5, DeepSeek-R1-Distill-Qwen

```sh
./jllm run  --prompt "Why is the sky blue?"                 # Qwen/Qwen3-0.6B, q4, Metal on macOS
./jllm chat --think                                         # Qwen3's thinking mode
./jllm chat --model Qwen/Qwen3-4B                           # any dense Qwen3
./jllm chat --model Qwen/Qwen2.5-1.5B-Instruct --quant q8
./jllm chat --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
./jllm serve --port 8080
```

- **Backend**: Metal. Qwen3-0.6B Q4_0: Metal ~2 100 prefill / ~143 decode tok/s (speculation off), CPU int8 ~275 / ~51,
  CPU float ~237 / ~13 ([11 §11.1](11-performance.md#111-measured)). Against llama.cpp on the same Q4_0 GGUF: 1 903 vs 3 097
  prefill, 138 vs 217 decode (0.61× / 0.63×).
- **Weights**: `q4`. `q8` (perplexity 7.16 vs 7.37) to check odd answers; never `q2k` on small models. Head `q8`.
- **KV**: `f32` (exact) for short chats; `q8` (rotated) for long ones on Metal — at ~3 000 tokens f32 decodes 92.9 tok/s,
  q8 114.4 ([08 §8.31](08-metal-backend.md#831-decode-attention-attention_dec)). On the CPU `q8` or `f32` (not `f16`, 28.4.4).
- **Context**: `auto` → 40 960 (the trained length) on the M1 Pro at f32 (9.4 GB) or f16 (4.7 GB); 7 168 on the CPU with
  the default 4 GB heap, more with a larger `JLLM_XMX` or two slots (paged, off-heap).
- **Chunk**: `auto` (256 Metal, 64 CPU); 512 gains 2–4 % for twice the scratch.
- **Speculation**: `ngram` (default); there is no smaller same-tokenizer draft for the 0.6B, and self-drafting is slower
  (56 vs 95 tok/s, [11 §11.6](11-performance.md#116-speculative-decoding)).
- **Sampling**: the bundled defaults are Qwen3's `generation_config.json` (temperature 0.6, top-k 20, top-p 0.95,
  [09 §9.10](09-sampling.md#910-where-each-setting-comes-from)); `--temp 0` for repeatable output.
- **Memory**: weights 452 MB mapped (q4) + KV (940 MB at 4 096 f32) + < 50 MB activations + ≈ 100 MB of JVM
  ([11 §11.3](11-performance.md#113-memory-footprint)).
- **Quality for agents**: "the 0.6B model itself is not a reliable editor: use 4B or larger" (README's aider check).

### 28.5.2 Qwen3-30B-A3B and other mixtures of experts

```sh
./jllm serve --model Qwen/Qwen3-30B-A3B-Thinking-2507 --low-disk        # 61 GB download, ~19.4 GB Q4_0 cache
./jllm serve --model Qwen/Qwen3-30B-A3B-Thinking-2507 --backend.kv-dtype q8   # 4x the context of f32
./jllm run   --model Qwen/Qwen3-30B-A3B-Thinking-2507 --batch 512 --prompt "…" # +6 to +12 % prefill
```

- **Backend**: Metal ~300 tok/s prefill (256-token chunks; 211 with 64) / ~27 decode; CPU ~12 / ~6.
- **Weights**: `q4` (19.4 GB, ≈ 18 GB of it experts). `q8` does not fit 32 GB. `q2k` would be ~10.5 GB but needs a mixed
  recipe that does not exist yet ([13 §13.7](13-q2k-quantization.md#137-measurements)).
- **Experts**: `backend.metal.cpu-moe=auto` moves nothing when the model fits (the 30B fits the 26.8 GB working set with room
  for an 18 432-token f32 cache). For a model larger than the working set it moves the coldest experts to the CPU, by usage
  counts saved in `~/.cache/jllm/expert-usage` (whole layers on a first run), re-planned every 30 minutes, moved only when
  ≥ 20 % cheaper; pipelined decoding and decode replay are off while any expert is on the CPU ([08 §8.18](08-metal-backend.md#818-experts-on-the-cpu-backendmetalcpu-moe)).
- **KV**: 197 KB / token at f32 (48 layers × 4 KV heads × 128 × 2 × 4 bytes) → `auto` 18 432 (f32) / 37 888 (f16) / 74 752 (q8).
  **For agents use `q8`**: four times the context of f32 for 0.003 nats.
- **Chunk**: 256 (auto); 512 gives +12 % at 413 tokens, +6 % at 2 000 ([08 §8.33](08-metal-backend.md#833-should-the-prefill-chunk-come-from-the-tuner-backendbatchauto-stays-at-256)).
- **Speculation**: `ngram` (edit +33…+68 %, tool −3…−10 %, prose ~−1 %). A verified extra position costs 0.55 of a step
  on the 30B (0.36 on a dense model), so wrong drafts cost more than on the 0.6B.
- **Quality**: the Thinking variant always reasons first: give it a generous `max_tokens`.
- **Memory**: 19.4 GB weights + KV + ~0.5 GB activations and JVM; Docker: about 24 GB ([19](19-docker.md)).

### 28.5.3 Llama 3.x and DeepSeek-R1-Distill-Llama

```sh
./jllm chat --model unsloth/Llama-3.2-1B-Instruct                    # an ungated mirror
./jllm chat --model ~/models/Llama-3.2-1B-Instruct-Q8_0.gguf
```

As dense. Llama 3's RoPE scaling is applied ([14 §14.4](14-model-families.md)); Llama-3.2-1B-Instruct was checked on real
weights at Q8_0 (66 of 80 greedy ids equal, the first difference at a 0.075 near-tie). Head sizes 64 (3.2) and 128 (8B) take
the fast decode attention on Metal. Tool calls use the Hermes prompt; Llama follows it less reliably than Qwen.

### 28.5.4 Mistral

As dense. `sliding_window` applies on every layer (null in v0.3); a `head_dim` that is not a multiple of 32 runs on the
CPU only. GGUF `mistral3` files (Mistral Small 3.x, Devstral) are read as Llama; Ministral 3 with
`attention.temperature_scale` is refused ([16 §16.15](16-gguf.md)). With `backend.kv-sparse` the sliding-window layers stay
dense. Only TinyMistral-248M was checked on real weights.

### 28.5.5 Gemma 2, Gemma 3, Gemma 4

- **Gemma 2** (`unsloth/gemma-2-2b-it`, 64 / 64 greedy ids equal): soft-caps 50 / 30, head 256 (fast on Metal). A system
  message is folded into the first user turn (the template refuses a system role).
- **Gemma 3** (`unsloth/gemma-3-1b-it`, 72 / 80): `backend.batch=auto` gives at least 256, because a picture's 256 tokens
  attend to each other and must go through in one pass. Pictures through a GGUF projector: Gemma 3 4B Q4_K_M with an F16
  `mmproj` encodes a picture in 2 970 ms (cold), prefills 278 tokens in 1 373 ms and decodes 25.3 tok/s (26.1 without the
  picture, [20 §20.13](20-vision.md#2013-gemma-3-attention-that-runs-both-ways-inside-a-picture)). Metal q4 KV is refused
  with a picture.
- **Gemma 4 (12B dense)**: CPU and Metal (512-wide attention kernels, per-layer head-shape dispatch). Each layer keeps its own
  flat cache, so the paged, batched, mixed and tiered passes, q4 KV and rotated q8 are not offered: use `f32`, `f16` or plain
  `q8` with one slot. E2B / E4B (per-layer embeddings) and 26B-A4B (MoE) are refused. Real weights on Metal (M1 Pro, Q4_K_M, f32 KV): decode about 8 tok/s, prefill 73-85 tok/s; the 512-wide flash kernels are off there ([14 §14.10.7](14-model-families.md#14107-gemma-4-12b)).

### 28.5.6 Phi-3 / 3.5 / 4

```sh
./jllm chat --model microsoft/Phi-3.5-mini-instruct --ctx 4096
```

LongRoPE picks its factors once, from `--ctx`: above the original length (4 096 for Phi-3.5 / Phi-4-mini) every position
uses the long factors, which are slightly worse on short text, so **`--ctx 4096` is the best setting for short chats**
([14 §14.9.2](14-model-families.md#1492-longrope-rope_scalingtype-longrope-legacy-su)); `auto` would pick a long context.
Phi-3-mini-4k has a real 2 047-token window. `head_dim` 96 runs on Metal with the generic attention kernel (the fast decode
attention covers 64 / 128 / 256). Not checked on real weights (they did not fit the machine used).

### 28.5.7 GLM-4 and GLM-Edge

```sh
./jllm chat --model THUDM/GLM-4-9B-0414 --backend.kv-dtype f16
```

As dense; GLM-Edge-1.5B was checked on real weights (80 / 80). The f16 cache halves the 9B's KV memory; not timed.

### 28.5.8 DeepSeek V2 / V3 (multi-head latent attention)

- The latent cache is small: V2-Lite 62 KB / token at f32 (255 MB at 4 096), 31 KB at f16, 16 KB at q8, instead of 553 KB
  expanded ([14 §14.11.6](14-model-families.md#14116-memory-on-this-machine)). **Use `f16`**: q8 keeps one scale per latent row;
  q4 is served as q8; rotated q8 does not apply.
- V2-Lite at Q4_0 ≈ 10 GB + 216 MB of dequantized `kv_b_proj`. V3 / R1 (≈ 420 GB at Q4_0) do not fit a 32 GB machine.
- One slot: the paged, sparse, tiered and mixed caches do not cover MLA; batched decoding works on Metal, not on the CPU.
- FP8 V3 / R1 checkpoints are dequantized while converting; `--quant q8` keeps their values within Q8_0 rounding.

### 28.5.9 Mamba

```sh
./jllm run --model state-spaces/mamba-130m-hf --raw --prompt "The capital of France is"     # a base model: --raw
```

No KV cache (`backend.kv-dtype` is ignored), no speculation, and a request resumes only when it extends what is cached;
batched decoding steps one sequence at a time. On a random-weight file with mamba-130m shapes, on a heavily loaded machine:
Metal ~1 060 / ~273 tok/s, CPU ~115 / ~15 ([15 §15.8](15-mamba.md#158-measured-numbers)). Chunk: 64 (auto).

### 28.5.10 Qwen3-Next and the Qwen3.5 / Qwen3.8 hybrids (Gated DeltaNet)

```sh
./jllm chat --model unsloth/Qwen3.8-27B-GGUF --gguf-quant Q4_0
./jllm run  --model unsloth/Qwen3.8-27B-GGUF --gguf-quant Q4_0 --temp 0 --speculative off --prompt "…"
```

- **Qwen3.8-27B Q4_0** (16 056 MB file, 14 893 MB used in place, loads in 1.4 s) on Metal: prefill 40.6 / 40.5 tok/s (128 /
  512 tokens) against llama.cpp's 77.8 / 78.1; decode 4.3 against 9.83 ([16 §16.13.5](16-gguf.md#16135-speed-apple-m1-pro-32-gb-under-the-heavy-lock)).
  "On the CPU the 27B is not usable." Context `auto`: 44 032 (5.8 GB f32).
- **Speculation**: `mtp` is the default when the file has the `nextn` head; `ngram` is impossible (the recurrent state cannot
  be rewound). +26 % on code it predicts well, break-even at 61 % accepted: `--speculative off` for free prose.
- **KV**: f32 or f16 for the attention layers (q8 / q4 are served as f16 on Metal); the DeltaNet state stays f32. One slot (no
  paging, sparse, tiers or mixed passes).
- Qwen3-Next-80B-A3B does not fit 32 GB (no `cpu-moe` for it on Metal yet).

### 28.5.11 gpt-oss

MXFP4 experts are read in place (the 20B's experts ≈ 10.1 GB, against 20.3 GB as Q8_0). Layers with attention sinks take
the generic attention kernel on Metal (no flash decoding: slower at long contexts, not measured). `q4` KV is served as `q8`;
no paged cache (sinks). `chat.thinking` / `enable_thinking` selects the harmony reasoning effort (true = medium, false =
low): `--think false` answers sooner. `backend.metal.cpu-moe` works. Only tiny random models were run: no speed is claimed
([14 §14.18](14-model-families.md#1418-gpt-oss-attention-sinks-sliding-windows-clamped-swiglu-experts-with-biases-mxfp4)).
**GGUF files** (`general.architecture` `gpt-oss`, e.g. `ggml-org/gpt-oss-20b-GGUF`) are read too, MXFP4 experts in place; the
SwiGLU clamp is llama.cpp's fixed 7.0 ([16 §16.16](16-gguf.md#1616-gpt-oss-llama-4-granite-4-nemotron-h-and-jamba-as-gguf-files)).

### 28.5.12 Llama 4 (text)

The text tower only. Chunked attention keeps a full (not chunk-limited) cache; q4 KV is served as q8; no Metal flash
attention for the chunked and temperature-tuned layers; no paged cache. Scout is 109 B: no real-weight run
([14 §14.19](14-model-families.md#1419-llama-4-text-chunked-attention-nope-layers-temperature-tuning-sigmoid-moe)). **GGUF
files** (`llama4`) are read with llama.cpp's fixed chunk (8192), temperature and NoPE layers; the vision projector is not
([16 §16.16](16-gguf.md#1616-gpt-oss-llama-4-granite-4-nemotron-h-and-jamba-as-gguf-files)).

### 28.5.13 Qwen2.5-1M

Dual Chunk Attention is not implemented: "run the 1M models with a context of at most 262 144 tokens (`backend.ctx`) to stay
in the region where plain attention is exact". The 7B at 262 144 tokens needs 15 GB of f16 or 7.5 GB of q8 KV
([14 §14.20](14-model-families.md#1420-dual-chunk-attention-for-qwen25-1m-the-design-not-implemented)). Raise
`backend.ctx-max` above 131 072 for `auto` to reach it.

### 28.5.14 Granite 4, Nemotron-H, Jamba (attention + state-space hybrids)

KV f32 or f16 (q8 / q4 served as f16 on Metal). No prefix reuse, no speculation as a target, no continuous batching, no
`cpu-moe`, no state save / restore: every request starts from position 0, so keep prompts short or accept the full prefill.
Nemotron-H and Jamba have no chat template: format the prompt yourself or use `/v1/completions`; Granite has one. Only tiny
models were run ([14 §14.22](14-model-families.md#1422-attention--state-space-hybrids-granite-4-nemotron-h-jamba)). **GGUF
files** (`granitehybrid`, `nemotron_h`, `jamba`) are read as well; for Granite the Mamba-2 norm is per group, as llama.cpp
computes it ([16 §16.16](16-gguf.md#1616-gpt-oss-llama-4-granite-4-nemotron-h-and-jamba-as-gguf-files)).

### 28.5.15 GGUF files (every family)

```sh
./jllm run   --model unsloth/Qwen3-0.6B-GGUF --gguf-quant Q4_0 --prompt "Why is the sky blue?"
./jllm serve --model unsloth/Qwen3-0.6B-GGUF --gguf-file Qwen3-0.6B-Q8_0.gguf
./jllm chat  --model ~/models/Llama-3.2-1B-Instruct-Q8_0.gguf
```

`model.gguf-quant` (or `--gguf-file`) picks one file of a repository, and only that file is downloaded. `model.quant` does not
apply: the file's own blocks are used. Read in place on both backends: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K,
Q6_K, IQ4_NL, IQ4_XS, MXFP4, F16, BF16; requantized to Q8_0 at load: F32 matrices always, and on the CPU by default Q5_1, Q3_K,
F16, BF16. Refused: IQ1_*, IQ2_*, IQ3_*, TQ1_0, TQ2_0, Q8_K, Q8_1 ([16 §16.4.1](16-gguf.md#1641-every-type-at-a-glance)).
Loading maps the file: a 397 MB Q4_K_M maps in 0.3–0.5 s with a peak RSS of ~195 MB.

### 28.5.16 Vision models

```sh
./jllm run  --model ggml-org/SmolVLM-256M-Instruct-GGUF --gguf-quant Q8_0 --image photo.png --prompt "What is in this picture?"
./jllm chat --model ggml-org/Qwen2.5-VL-3B-Instruct-GGUF --gguf-quant Q4_K_M --image a.png,b.jpg
./jllm serve --model ggml-org/Qwen3-VL-2B-Instruct-GGUF --gguf-quant Q8_0
```

Put the picture before the question. Use the **F16 projector** (`vision.auto-download` picks it; llama.cpp's own Q8_0
projector moves the embeddings by 6.6·10⁻²). Picture encode on Metal / CPU ([20 §20.11](20-vision.md#2011-measured)):
SmolVLM-256M 101 / 612 ms per tile, Qwen2.5-VL-3B 1 543 / 8 067 ms, Qwen3-VL-2B 764 / 4 807 ms, Gemma 3 4B 2 755 / 15 227 ms.
Whole 64-token answers on Metal: SmolVLM 819 ms (decode 251 tok/s), Qwen2.5-VL-3B Q4_K_M 8 344 ms (22.8 tok/s), Qwen3-VL-2B
Q8_0 4 293 ms (39.6 tok/s). For dynamic-resolution models, `vision.image-max-tokens` bounds the tokens (and the memory,
`n × ffn × 4` bytes of activations: 56 MB at 4 096 patches for Qwen2.5-VL) of a big picture; encoded pictures are cached
(`vision.cache-mb` 256), so later turns do not re-encode them.

### 28.5.17 Audio: Whisper and OuteTTS

```sh
./jllm transcribe --model openai/whisper-small --audio talk.wav --format srt
./jllm speak --model OuteAI/OuteTTS-0.3-500M --prompt "Hello from jllm." --voice en_female_1 --out hello.wav
./jllm serve --audio.stt-model openai/whisper-small --audio.tts-model OuteAI/OuteTTS-0.3-500M
```

| model | best settings | measured on a 61.7 s clip ([21 §21.13](21-audio.md#2113-measurements)) |
|---|---|---|
| Whisper tiny / base | `audio.backend=auto`, the CPU decoder (auto below `d_model` 768) | tiny 0.99 s on Metal (RTF 0.016), 1.56 s on the CPU; base 2.02 s |
| Whisper small | Metal with the GPU decoder (auto) | 4.71 s (RTF 0.076; whisper.cpp's GPU 2.75 s); CPU 14.2 s |
| Whisper large-v3-turbo | Metal, GPU decoder; `--audio.stt.fallback false` when a silent end retries | 26.6 s (GPU decoder) / 31.2 s (CPU decoder); the last window 41 s → 9.5 s without fallback |
| OuteTTS 0.3 500M | LM on Metal, codec on the CPU; `audio.tts.quant=q8` (q4 for ~20 % less time) | RTF 1.01 (q8), 0.81 (q4); round-trip WER 6.62 % vs 5.96 % on 151 words (within the noise) |

Word error rates on 73 LibriSpeech clips equal transformers' (tiny 12.61 %, base 9.22 %, small 7.39 %). `audio.threads` leaves
cores to the LLM; `audio.resident-mb` (1 500) is the heap budget for Whisper weights kept as floats (larger models stay
mapped).

### 28.5.18 Image generation

```sh
./jllm image --image-model stabilityai/sd-turbo --prompt "a red fox in a snowy forest, photo" --out fox.png
./jllm image --image-model stabilityai/sd-turbo --prompt "watercolor" --init fox.png --strength 0.5 --out fox2.png
```

| model | defaults (best) | Metal f16 | CPU f32 | memory |
|---|---|---|---|---|
| SD-Turbo | 1 step, guidance 0, 512², Euler (1–4 steps is the useful range; a negative prompt has no effect) | 1.68 s (1 step), 2.92 s (4), 9.63 s (20); 1024² 4 steps 16 s | 12.2 s (1), 24.3 s (4) | 2.6 GB weights, 3.2 GB GPU buffers |
| SDXL-Turbo | 1 step, 512² native | 2.70 s (1), 4.70 s (4) | — | 6.9 GB weights, peak 8.3 GB (VAE always f32) |
| SD 1.5 (and its inpainting model) | 25 steps, guidance 7.5, `dpmpp_2m_karras` ("accurate at about 20 steps") | UNet 0.42 s / step at 512² (guidance doubles the UNet work) | — | — |
| SDXL | 1024² native | UNet 0.72 s / step at 512² | — | — |

`image.precision=auto` is f16 (f32 costs 1.4× the time and 1.7× the memory); `image.vae-tile` 128 decodes 1024² in one piece
(~3 GB), 64 on a small machine ([22 §22.14](22-image-generation.md#2214-measured-speed-and-memory)).

### 28.5.19 Video generation

```sh
./jllm video --video-model guoyww/animatediff-motion-adapter-v1-5-2 --prompt "waves" --frames 16 --fps 8 --width 256 --height 256 --out waves.gif
```

- **AnimateDiff** (16 frames, guidance 7.5, DDIM): 256² f16 3.88 s / step, VAE 0.30 s / frame, 10 steps 43.7 s, 3.0 GB of
  buffers (peak 4.1 GB); 512² 17.6 s / step, peak 7.3 GB, a 20-step clip 371 s ([23 §23.11](23-video-generation.md#2311-measured-speed-and-memory)).
  **256² is 4.5× cheaper per step**; f32 costs 1.5× the time and twice the memory. The only quality lever is
  `video.base-model`: a fine-tuned SD 1.5 instead of the plain one. MP4 needs ffmpeg (0.83 MB against 5.9 MB WebP / 2.0 MB GIF).
- **Wan 2.1 1.3B** (CLI only): transformer evaluation of 1 200 tokens f32 2.05 s, f16 1.43 s, CPU 17.0 s; ≈ 90 s of GPU time
  per 17-frame 320×192 clip in f16 (an estimate; no full clip was made). Keep `video.text-encoder-cache=true` (the 5.2 GB int8
  UMT5 copy is reused instead of re-quantizing the 22.7 GB f32 encoder, [23 §23.12](23-video-generation.md#2312-wan-21-how-far-it-got-and-what-remains)).

---

## 28.6 Coding agents and other long shared prompts

🧸 An agent sends the whole conversation again every turn: thousands of tokens of instructions and tool descriptions, then
the files it read. The engine should read each of those tokens once, keep them, and only read what is new.

```sh
./jllm-web/run.sh                        # heap 60 % of RAM, direct memory = RAM
# Models page: load the model with the "Coding agent preset":
#   backend.ctx 32768, backend.kv-dtype f16, slots 2, server.queue-timeout-ms 1800000
./jllm-web/opencode.sh                   # or: ./jllm-web/agent-config.sh continue | aider | cline
```

| item | best value | why (measured) |
|---|---|---|
| backend | Metal on Apple silicon | on a 4-core VM the first opencode step (10 447 tokens) took 349 s (README) |
| model | Qwen3 4B or larger, or Qwen2.5-Coder | the 0.6B picks the right tool for simple requests but not for long chains |
| context | 32 768 (preset) | opencode's instructions and tools ≈ 11 k tokens, Cline ~12 k, Continue ~5 k, aider ~4 k |
| slots | 2 (preset) | a side request (a title) no longer evicts the main conversation: 10 443 of 10 484 tokens cached, prefilled in 2.5 s |
| KV paging | `auto` → paged with 2 slots | 8 conversations × 4 096 shared tokens: 1.1 GB instead of 7.7 GB; a resent conversation in 32 ms instead of 6.3 s (Metal) |
| KV type | `f16` (preset) on Metal; **`q8` on the CPU** | Metal: rotated q8 is as fast (122.9 vs 121.0 tok/s at 2 013 tokens) for half the memory again; CPU: f16 decoded 9.7 tok/s against 41.4 for q8 at 2 000 tokens on the M1 Pro's CPU, and on the x86 VM rotated q8 decodes 6.80 tok/s against 4.93 for f16 at 16 k |
| speculation | `ngram` (default) | code edits +106 % (0.6B Metal), +33…+68 % (30B), +44 % (CPU) |
| tool calls | `engine.constrained-tools`, `engine.jump-forward`, `engine.constraint-masks` all `true` | every call parses; a 32-token call 254 → 149 ms on Metal ([17 §17.7.2](17-agent-speedups.md#1772-measured-checks-limits-and-switches)) |
| tokenizing | `engine.prompt-cache-mb` 64 | a 100 k-character, 50-turn conversation tokenizes in 0.50 ms instead of 15.6 ms ([17 §17.5](17-agent-speedups.md#175-tokenizing-only-what-is-new)) |
| saved conversations | `jllm.web.kv-cache=true` | a 2 707-token prompt: 21.3 s of prefill → 0.9 s restore after a restart |
| priority | subagents send `"priority": "background"` | interactive first token 127 ms instead of 937 ms |
| thinking | off for speed (`"enable_thinking": false`), on for hard tasks; Cline needs `chat.thinking=false` on the server | — |
| scheduler | `slots` | `mixed` is implemented but not measured yet |

### 28.6.1 Several users on one server

🧸 One slot is one conversation being worked on. With one slot, users take turns: the fourth one waits for three whole answers
before seeing a word. Give the server as many slots as users who type at the same time.

```sh
./jllm serve --server.slots 4                                    # API server: 4 users at once, steps batched, paged KV
./jllm serve --server.slots 4 --backend.kv-dtype q8              # CPU: q8 halves the KV memory again and decodes fastest (28.4.4)
./jllm-web/run.sh                                                # console: "slots" in the Models page's load form
```

| item | best value | why (measured, [10 §10.9c](10-engine-and-server.md#109c-several-users-at-once-serve---serverslots)) |
|---|---|---|
| `server.slots` | the number of users generating at the same time (memory permitting) | 4 users × 64 tokens on the x86 VM: worst first token 0.8–1.1 s instead of 6.2–6.5 s, 38–41 tok/s together instead of ~31 |
| `backend.kv-paged` | `auto` (paged from 2 slots) | a shared system prompt is stored and prefilled once: the second user found 716 of its 731 prompt tokens cached |
| `backend.ctx` | `auto` | it divides the free memory by the number of slots; a fixed `--ctx` reserves `ctx` positions per slot with flat caches |
| `server.queue-timeout-ms` | `60000` (default) | more users than slots wait this long, then get a 503 `overloaded` (also under `engine.scheduler=mixed`) |
| `backend.cpu.batch-int8-rows` | `auto` (default; 3 on that x86 VM) | on the CPU, 2 users sharing a pass got 30–31 tok/s together where the float tiles gave 25 |
| `engine.scheduler` | `slots` | `mixed` gives every request one queue and mixes prefill into the decode passes; correct on CPU (`ServerSlotsTest`), its speed not measured yet |

---

## 28.7 CPU-only machines: Linux, x86, Docker

On a machine without a usable GPU everything runs on the CPU backend. What matters, in order:

1. **The native int8 library.** `./jllm` builds `target/native/libjllm_cpu.so` when a C compiler exists; the start-up line
   says `native CPU kernels: avx512-vnni` (or `avx-vnni`, `avx2`, `neon-dotprod`). Without it, AVX-512 machines often time the
   float path as faster (`auto` keeps it), and decode loses up to 2× ([07 §7.16](07-cpu-backend.md#716-native-int8-kernels-libjllm_cpu-nativecpu)).
   The Docker image ships it (`JLLM_CPU_NATIVE=auto`).
2. **The vector width.** The JVM uses AVX-512 when the CPU has it (16 float lanes). `-XX:MaxVectorSize=32` limits it to AVX2
   width (8 lanes), which is how the AVX2 rows below were measured.
3. **Memory**: one slot keeps its cache on the heap (`JLLM_XMX`); two or more slots use the off-heap paged pool, which the
   launchers now let grow to the RAM (`JLLM_DIRECT_MEMORY` to bound it).
4. **The KV type**: `q8` (rotated): the fastest on both CPUs measured (28.4.4).
5. **Sparse attention for long contexts**: `--backend.kv-sparse true` decodes 2 – 3× faster than dense attention at 8 k – 32 k
   tokens on the x86 VM for a retrieval accuracy within one haystack of 30 ([27 §27.5.3 – 27.5.4](27-sparse-attention.md#2753-needle-in-a-haystack-cpu-x86-2026-10-28)).
6. **Speculation**: `ngram` pays most on the CPU, where the verify pass is cheap: rewriting a file 19.9 → 81.2 tok/s on 4
   AVX-512 cores ([17 §17.1](17-agent-speedups.md)).

**Measured on the x86 VM** (4 cores, AVX-512 VNNI, JDK 26.0.2; Qwen3-0.6B Q4_0, float activations, f16 cache,
`./jllm bench --backend cpu --backend.cpu.int8-activations false --backend.kv-dtype f16 --ctx 4096 --bench.prompt-repeat 60
--bench.max-tokens 32 --bench.rounds 3 --engine.speculative off`, 613-token prompt, tok/s of two rounds;
[07 §7.14](07-cpu-backend.md#714-prefill-and-decode-are-not-bit-identical-results-depend-on-history)):

| lane sum in the tiled prefill kernels | prefill, 16 lanes (AVX-512) | decode, 16 lanes | prefill, 8 lanes (AVX2) | decode, 8 lanes |
|---|---|---|---|---|
| before (`reduceLanes` / `lane(0..7)`) | 228 – 258 | 28.0 – 30.8 | **23.1 – 23.2** | 17.5 – 17.9 |
| **now (a butterfly of whole-vector permutes)** | **219 – 254** | 27.8 – 35.2 | **186 – 199** | 19.2 – 21.6 |

So an AVX2 machine now prefills about 8× faster than before this change, and an AVX-512 one at the same speed, with results
that no longer depend on when the JIT compiled the kernel. Long prompts on the same VM: an 8 320-token haystack of the needle
study (`KvStudy needle`, f16 cache, 64-token chunks) prefilled in 128 s, 65 tok/s: each chunk also attends to everything
before it, so the rate falls as the prompt grows. A long agent prompt on a 4-core CPU therefore costs minutes, which is why saved conversations and the paged prefix cache matter most there.

**Docker** ([19 §19.1](19-docker.md#191-for-people-who-use-the-image)): give the container the memory of the model plus
its cache (`docker run -m 8g …`); the heap is `JLLM_HEAP_PERCENT` (60 %) of that limit and the direct memory the whole limit.
The 30B needs about 24 GB.

---

## 28.8 How much memory: the formulas and two worked examples

📐 **The KV cache per token** of an attention model:

```
bytes per token = 2 (key and value) × layers × kv_heads × head_dim × bytes per element
                  f32: 4    f16: 2    q8: 1 + 4/head_dim (one f32 scale per head row)
                  q4: 1/2 + 2/32 (one half scale per 32 values; + the ring of the recent rows as halves)
```

**Worked example 1, Qwen3-0.6B** (28 layers, 8 KV heads, head 128): `2 × 28 × 8 × 128 = 57 344` elements per token.

| type | per token | 4 096 tokens | 40 960 tokens |
|---|---|---|---|
| f32 | 57 344 × 4 = 229 376 B (224 KiB) | 0.94 GB | 9.4 GB |
| f16 | 114 688 B (112 KiB) | 0.47 GB | 4.7 GB |
| q8 | 57 344 × (1 + 4/128) = 59 136 B (57.75 KiB) | 0.24 GB | 2.4 GB |
| q4 | 57 344 × 0.5625 = 32 256 B (31.5 KiB) + ring | 0.13 GB + 44 MB | 1.3 GB + 44 MB |

Add the weights (452 MB for q4), < 50 MB of activations and ≈ 100 MB of JVM: at 4 096 tokens f32 the whole process is
≈ 1.5 GB ([11 §11.3](11-performance.md#113-memory-footprint)).

**Worked example 2, Qwen3-30B-A3B on the M1 Pro** (48 layers, 4 KV heads, head 128): `2 × 48 × 4 × 128 = 49 152` elements,
196 608 B (192 KiB) per token at f32, 98 304 B at f16, 49 152 × 1.03125 = 50 688 B at q8. `backend.ctx=auto` takes the smaller
of two budgets:

```
Metal:  0.6 × (working set 26.8 GB − weights 19.4 GB − reserve 0.84 GB (26.8 / 32) + experts that may move 2.68 GB (10 %))
        = 0.6 × 9.24 GB = 5.54 GB
RAM:    0.6 × (0.75 × 34.36 GB (32 GiB) − weights 19.4 GB) = 0.6 × 6.37 GB = 3.82 GB      ← the smaller: it decides

f32:  3.82 GB / 196 608 B = 19 430 positions → rounded down to a multiple of 1024:  18 432   (3.6 GB)
f16:  3.82 GB /  98 304 B = 38 860                                               →  37 888
q8:   3.82 GB /  50 688 B = 75 365                                               →  74 752
```

These are the three picks measured in [10 §10.13](10-engine-and-server.md#1013-the-default-context-length-backendctxauto). The
same computation explains why the cache type is the setting that buys context: halving the bytes per token doubles the
positions, at the costs of 28.4.4. The `Context:` start-up line prints the exact terms on your machine.

**Where each piece lives:**

```
 RAM ───────────────────────────────────────────────────────────────────────────────────
 │ weights: memory-mapped file (the q4 cache or the GGUF), shared by every slot          │
 │ Java heap (JLLM_XMX): flat CPU KV caches of single-slot engines, vision/audio floats   │
 │ direct memory (JLLM_DIRECT_MEMORY): the paged CPU KV pool's slabs                      │
 │ Metal buffers (unified memory, the GPU's working set): KV caches, scratch, experts     │
 │ page cache: what macOS / Linux keeps of the mapped weights                             │
 ──────────────────────────────────────────────────────────────────────────────────────────
```

---

## 28.9 When it is slow: a checklist

| symptom | look at | fix |
|---|---|---|
| decode far below the 28.3 numbers on a Mac | the start-up line `Backend: cpu` | the Metal library is missing (`scripts/build-native.sh` builds `target/native/libjllm_metal.dylib`; `./jllm` does it when Xcode's command-line tools exist), or `backend=cpu` is set |
| CPU decode ~2× slower than expected | no `native CPU kernels:` line | install a C compiler (the launcher builds `libjllm_cpu`) or set `backend.cpu.native-lib` |
| CPU decode slower at long contexts with f16 | `backend.kv-dtype` | `q8` (28.4.4) |
| `OutOfMemoryError: Cannot reserve … direct buffer memory` | an old launcher, or `JLLM_DIRECT_MEMORY` too small | the launchers now set the limit; `JLLM_DIRECT_MEMORY=16g` |
| `OutOfMemoryError: Java heap space` with one slot on the CPU | the heap holds the flat cache | `JLLM_XMX=8g`, a smaller `--ctx`, or `--backend.kv-dtype q8` |
| every turn of an agent re-reads the whole prompt | `slots` and `backend.kv-paged` | 2 slots (paged); keep the system prompt identical between turns |
| prose slower with speculation | the stats line of `./jllm run` (`speculation accepted A of D drafted`) | normal when drafts are rejected (−1 to −5 %); `--speculative off` for a pure-prose workload |
| a MoE prefill slow | `backend.batch` | leave `auto` (256); `--batch 512` if the scratch fits |
| a model larger than the GPU working set | the `cpu-moe` start-up note | `backend.metal.cpu-moe=auto` (default); a smaller `--ctx` |
| greedy text differs between CPU and Metal | int8 activations | `--backend.cpu.int8-activations false` for comparisons ([07 §7.14](07-cpu-backend.md#714-prefill-and-decode-are-not-bit-identical-results-depend-on-history)) |

---

## 28.10 Not measured yet

What this chapter cannot recommend from data yet, each with what would decide it:

1. **`engine.scheduler=mixed` against `slots`**: the tables of [25 §25.5](25-scheduler.md#255-measured) are still to be filled
   on the Mac (`SchedulerBench`); its knobs (`pass-budget`, `mixed-reserve`, `background-chunk-cap`, `max-running`) are unmeasured.
2. **Batched decoding against taking turns** on a quiet Mac ([10 §10.9b](10-engine-and-server.md#109b-batched-decoding-across-slots)), and
   `ServerSlotsTest bench --backend metal` for `serve --server.slots` (the CPU's table is in [10 §10.9c](10-engine-and-server.md#109c-several-users-at-once-serve---serverslots)).
3. **Sparse attention on Metal**: the needle table and the decode speed on the GPU (the CPU's are in [27 §27.5.3](27-sparse-attention.md#2753-needle-in-a-haystack-cpu-x86-2026-10-28)
   and 27.5.4); whether 64 or 128 blocks should be the default (128 is closer in perplexity and 9 – 25 % slower per step, both are within one haystack of dense).
4. **Tiered KV** beyond the interim 0.6B numbers, and the accuracy of `backend.kv-disk-format` q8 / q4 ("about 0.5 %" / "about 8 %").
5. Measured since: the f16 KV cache on an x86 CPU against f32 and q8 ([27 §27.5.4](27-sparse-attention.md#2754-decode-speed-against-the-context-length-cpu-x86-2026-10-28): q8 fastest, f32 slowest).
6. **q4 KV at long contexts on Metal** since the decode attention kernel of [08 §8.31](08-metal-backend.md#831-decode-attention-attention_dec).
7. **`backend.batch` on other GPUs** (M3 / M4, 32+ cores) and `hmma` there.
8. **`backend.metal.cpu-moe` placement costs** on a model larger than the working set.
9. **Draft-model speculation on Metal** since pipelined decoding.
10. **Q4_K_M decode on Metal** with the multi-row matvec (90 tok/s was measured before it).
11. **Q2_K** speed after the fused kernels, and a mixed recipe for the 30B.
12. **`model.quant=q6k`** for every layer.
13. **`backend.threads`**: no sweep.
14. **Real-weight speed and memory** of Gemma 4 12B, gpt-oss, Llama 4, Granite 4 / Nemotron-H / Jamba, Qwen3-Next-80B, DeepSeek V2-Lite,
    Phi-3 / 4, GLM-4 and Mistral 7B.
15. **Per-family sampling defaults**: only Qwen3's are bundled; other families use them too unless the request sets its own.
16. **Wan 2.1**: an end-to-end clip, and whether `video.precision=auto` should pick f16 (1.4× faster, 5.3·10⁻³ transformer error).
17. **Contexts beyond 32 k tokens** on any model.

---

## 28.11 Code and sources

The defaults live in `src/main/resources/jllm.properties` (every key with its comment) and are resolved in
`src/main/java/org/robomiri/jllm/ModelLoader.java`: `batch` (the prefill chunk of `backend.batch=auto`), `speculation`
(the proposer of `engine.speculative`), `kvPool` (whether the paged cache applies and why not), `mixedWanted` (the mixed
scheduler's conditions), `newBackend` (Metal or CPU, the int8 / native probes). `ContextBudget.forModel` computes
`backend.ctx=auto` and prints its basis. The launchers are `jllm`, `jllm-web/run.sh` and `docker/entrypoint.sh`, with the
direct-memory limit in `scripts/jvm-memory.sh`.

Every number in this chapter is copied from the measured section it links to, with that section's machine and command line;
when a number changes there, it should change here. `scripts/mac-validate.sh` re-measures the Metal rows on a Mac.
