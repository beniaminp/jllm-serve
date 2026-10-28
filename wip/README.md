# Work in progress that is not merged yet

Status on 2026-10-27 (UTC); items 1–6 as of 2026-10-23, item 7 added on 2026-10-27. Everything listed here is pushed to
GitHub on a `wip/*` branch (item 7: `ccr-be155af0-ycvdp8`), so another person or agent can pick it up. `main` holds only work that passed every gate below, except item 7, which the
owner had pushed to `main` on 2026-10-27 after the gates that run on Linux. Each `wip/*` branch has
`main` merged in at the commit given, and none of them is on `main` unless this file says so.

## How to continue a branch

1. `git fetch origin && git checkout -b <name> origin/wip/<name> && git merge origin/main`.
2. Read the branch's docs chapter (given per item) and its last commit messages; "WIP:" commits say what is left.
3. Finish the items under "Left", then run the gates. Merge to `main` only when all six pass on the merged tree.

### The gates (all must pass before anything goes to `main`)

On a Mac with JDK 26 (`export JAVA_HOME=/opt/homebrew/opt/openjdk@26`), from the repository root, with an empty
config so the user's own `./jllm.properties` is not read (`export JLLM_CONFIG=/path/to/empty.properties`), and one heavy
job at a time on the machine (they share the GPU and the `~/.m2` engine jar):

| gate | command | passes when |
|---|---|---|
| install | `mvn -o install` | `BUILD SUCCESS` (runs every engine test) |
| web | `cd jllm-web && mvn -o test` | `BUILD SUCCESS`; run it right after `install`, it tests against the jar in `~/.m2` |
| emulator | `scripts/build-metal-emu.sh` | exit 0 |
| kernel tests | `python3 scripts/metal-emu/kernel_tests.py` (needs numpy) | no `FAIL` line |
| arch smoke | `scripts/arch-smoke.sh` (real GPU, ~10 min) | last line `arch smoke passed` |
| exactness | `./jllm run --backend cpu --backend.cpu.int8-activations false --ctx 512 --prompt "Write a haiku about Java and explain it in two sentences." --temp 0 --max-tokens 40 2>&1 \| grep -v WARNING \| tail -2 \| head -1 \| md5`, then the same with `--backend metal` | both print `79472f690f73bd28598d5d7fb5dee1a4` (unless the change is documented as not bit-identical) |

A change to the Metal encoder or batched rows should also pass `JAVA_OPTS=-Djllm.metal.concurrent=true scripts/arch-smoke.sh`.

### House rules

- Every feature has a switch, keeps the old path when switched off, gets an on/off row in `scripts/mac-validate.sh`, and
  a **detailed** docs section: 🧸 plain words with an analogy; 📐 step by step with exact formulas, a worked example with
  real numbers, an ASCII diagram, the classes and kernels with paths; why this design; settings; measured tables with
  the command lines; checks and limits. A results table alone is not a docs section.
- Speed numbers come from a quiet machine, with the command line given.
- Commits stage explicit paths (never `git add -A`; never `models/` or `jllm.properties`), and carry **no AI
  references** (no `Co-Authored-By`, no session links).
- No model downloads for architecture work: tiny random checkpoints only (`scripts/make-tiny-model.py`).

## In progress

### 1. Mixed scheduler — `wip/scheduler` (docs/25)

- **Branch:** `wip/scheduler`, merged to `main` as `29ccca1` after all six gates passed (`mvn install`, the web tests,
  the Metal emulator, the kernel tests, the arch smoke and the exactness run). The merge resolved two conflicts:
  - `ModelLoader.kvPool` pages when `paged(...)` or `mixedWanted(...)`;
  - `Engine` keeps both `releaseKv` (idle slots, paged follow-up) and `releaseCache` (scheduler).
- **Two real fixes found during the merge:**
  - the Metal mixed pass (`forwardMixed`/`runSegments`) was missing the attention `AttnExtra` buffer, so it bound the
    paged block table at the wrong buffer index and produced garbage (NaN) logits; fixed by adding an `attentionAt`
    overload that takes the `AttnExtra`;
  - the concurrent-encoder check is now warmed up (the serial reference is computed on a second run), which stops the
    intermittent cold-JIT single-ULP flake.
- **Works (`engine.scheduler=mixed`; `slots` stays the default and is unchanged):**
  - `Backend.forwardMixed` on CPU and Metal: one prefill chunk plus every decode token plus every draft verification in
    one pass;
  - `BatchGroup` mixed mode with `PassPlanner` (token budget, aging);
  - a cache-aware global admission queue in `LoadedModel`, with the `priority` classes, session limits, and preemption
    by suspending a sequence so its blocks spill and reload;
  - client-gone wake in under 60 ms;
  - tests: `SchedulerPolicyTest`, `MixedSchedulerTest`, `MixedPassChecks` in arch smoke, jllm-web `MixedSlotsTest`, and
    the `SchedulerBench` agent workload.
- **Merged, follow-ups still open:**
  1. Second benchmark round (pass budget, background chunk cap, the 30B alone and with 2 slots), then fill the `TBD`
     markers in docs/25 §25.5 and the README numbers.
  2. Decide the default from the numbers. `engine.scheduler` is still `slots`, so mixed is off by default until those
     numbers decide.
- **Known:**
  - Round 1 on Metal, 0.6B agent workload: slots with the flat cache 65 s, slots with paged 36.5 s, mixed 36.0 s wall.
    Mixed raises the inter-token p95 (56–196 ms against 37 ms) because decode rows ride a 256-row chunk pass, and lowers
    the p99 stalls. The pass budget is the knob.
  - The 30B MoE ran at about 1 tok/s per stream in the 3-slot bench (expert placement under 3 slots); not understood
    yet.
  - Done (item 7): `ContextBudget` counts the off-heap paged pool for `auto` and when the mixed scheduler turns paging on, and
    bounds it by the JVM's direct-memory limit.

### 2. Long-context model families — `wip/model-families` (docs/14 §14.18–14.22)

- **Branch:** `wip/model-families`, merged to `main` as `830c59f` after all six gates passed (`mvn install`, the web
  tests, the Metal emulator, the kernel tests, the arch smoke and the exactness run).
- **Works:** gpt-oss (attention sinks, alternating sliding windows, MXFP4 experts read in place, harmony chat format),
  Llama 4 text (chunked attention, NoPE, temperature tuning, its chat format), Granite 4, Nemotron-H and Jamba
  (Mamba-2 SSD hybrids, Granite chat format).
  - All checked against transformers 5.20 on tiny random models: logits within 1e-5 of the largest, the same greedy
    ids, CPU and Metal.
  - `scripts/hf-arch-check.sh` reruns that comparison.
- **Merged:** the earlier `GptOssMxfp4` arch-smoke failure (batched decode, 3 slots, q8 cache) was fixed by `59d3809`.
  The `concurrent encoder, experts of 1 layer on the cpu` check proved intermittently flaky, not a regression: a
  single-ULP cold-JIT difference where a second serial run equals the concurrent run, and it passes on a re-run. The
  missing `b2bea69` `kv-paged=auto` fix was cherry-picked in.
- **Merged, follow-ups still open:**
  - Dual Chunk Attention (Qwen2.5-1M): only a design (docs 14.20), with no transformers reference to check against.
  - Hybrids: Nemotron-H MoE layers are refused, and state save, prefix reuse and batching are not supported.
  - No chat format for Nemotron-H and Jamba, and no GGUF for any of the new families.
  - gpt-oss and Llama 4: no q4 KV cache (served as q8), no Metal flash attention for sink, chunk or temperature layers,
    no paged cache (they fall back to flat), no chunk-limited cache for Llama 4.
  - No real weights have been run, so there are no speed numbers.

### 3. Sparse decode attention — `wip/sparse-attention` (docs/27)

- **Branch:** `wip/sparse-attention`, merged to `main` as `65f759a` after all six gates passed (`mvn install`, the web
  tests, the Metal emulator, the kernel tests, the arch smoke and the exactness run, whose md5 matched on both CPU and
  Metal). Merging `main` in resolved 11 conflicting files (`ContextBudget`, `ModelLoader`, `Settings`, `CpuBackend`,
  `PagedKv`, `MetalKvStorage`, `kernel_tests.py`, `mac-validate.sh`, `README`, `docs/24`, `docs/README`) to reconcile
  sparse attention with the paged-auto, tiered-KV and hybrid backends. One follow-up fix: `ModelLoader.sparse` reads
  `backend.kv-paged` as `auto`/`true`/`false` instead of `getBool`.
- **Works (`backend.kv-sparse`, off by default; it turns on the paged cache):**
  - per-block key summaries (half min/max per channel), the Quest upper bound, top-k blocks plus sinks plus the recent
    window, on the CPU paged cache and on Metal (`kv_summary`, `sparse_score_d*`, `sparse_select`, `attention_dec_sp`);
  - wired into decode, short verification chunks, batched rows, replay and saved caches;
  - checks: `cpu/SparseKvTest` (1405 checks, in `mvn install`), the kernel_tests sparse section, and `KvStudy
    sparsecheck` on the real GPU (replay on = off bit for bit);
  - perplexity at 8k/16k/32k is in docs/27 §27.5.2.
- **Merged, follow-ups still open:** the needle-in-a-haystack tables at 8k/16k/32k, decode tok/s against context
  length, and anything at 128k+ (the 30B supports 262k but did not fit beside other jobs).
- **Known:** CPU and Metal sparse logits differ by up to 0.4–2.8, because a rounding-order difference flips which block
  is selected. Argmax agrees 23–24 of 24.

### 4. Tiered KV storage and memory planner — `wip/tiered-kv-memory-planner` (docs/26)

- **Branch:** `wip/tiered-kv-memory-planner`, merged to `main` as `4643494` after all six gates passed (`mvn install`,
  the web tests, the Metal emulator, the kernel tests, the arch smoke and the exactness run, whose md5 was
  `79472f690f73bd28598d5d7fb5dee1a4` on both CPU and Metal). Merging `main` in resolved three conflicts
  (`ContextBudget.java`, `ModelLoader.java`, `KvPool.java`) to reconcile tiered KV with the paged-auto, idle-slot
  `Reclaimer` and hybrid-backend features that had landed on `main`.
- **Works (checked by quick runs):**
  - `backend.kv-tier`: a warm host-memory tier with background demotion and prefetch in front of the block files, an
    on-disk q8/q4 block codec, memory-mapped block reads, and a hot cap and persistence in `KvPool` (`KvTierTest`).
  - The Metal paged pool is counted in `DeviceLedger` (it registers `hotCap × blockBytes`), so the MoE expert placement
    sees it.
  - Streamed KV on Metal: rows on the host, two GPU windows, layer by layer. It is bit-identical to the flat cache
    (`TierChecks` inside ArchSmoke, LlamaPlain, CPU and Metal, f32/f16/q8/rotated q8).
  - CPU slabs are memory-mapped past the hot budget.
  - `MemoryPlanner` (experts against KV against streaming, with hysteresis and cap growth), with
    `metal/MemoryPlannerTest`. Live growth works on Qwen3-0.6B (cap 512 → 768 → 1026 blocks).
  - `GET /api/memory` and a Memory card on the console's Models page.
  - Saved sessions as block files (`KvStore.saveBlocks`).
  - `ContextBudget.tierManaged`: with `backend.kv-tier`, `backend.ctx=auto` is bounded by physical memory, not the GPU
    share.
- **Merged, follow-ups still open:**
  - the measurements: Qwen3-0.6B at 32k resident vs tiered, the 30B with a growing context, and the `TierDiskStudy`
    accuracy table, so docs/26 §26.5 has only interim numbers;
  - `TierEngineTest` (a real model through the engine) has not run to completion;
  - the live planner moving experts on the 30B is untested.
- **Checks:** `KvTierTest`, `metal.MemoryPlannerTest`, `TierEngineTest` (Qwen3-0.6B), `TierStudy <model> <lengths>
  --backend metal --backend.kv-tier true`, and ArchSmoke with its `TierChecks`.

### 5. Paged KV cache follow-ups — `wip/paged-kv` (docs/24)

- **Done:** merged to `main` as `3723c42` ("docs: measure the paged CPU decode cost (q8/f32 -3%, f16 free)"): the
  placeholders are filled with quiet-machine Qwen3-0.6B numbers, the §24.5 CPU rows and the §24.3 costs line are updated, and
  the 4× f16 figure is attributed to the discarded ternary build. Nothing is left on `wip/paged-kv`.

### 6. Gemma 4 (12B) — `wip/gemma4`

- **Branch:** `wip/gemma4`, merged to `main` as `d7b388e` ("Add Gemma 4 (12B): per-layer head shapes, K = V,
  proportional RoPE") and `bcec1eb` (the GGUF tokenizer fallback), after all six gates passed.
- **Gemma 4 is not a Gemma-3 config variant.** It needs:
  - per-layer head shapes: local 256-wide with 8 KV heads, full-attention 512-wide with 1 KV head;
  - `attention_k_eq_v` (V = K) plus a parameterless V-norm;
  - proportional RoPE (`partial_rotary_factor 0.25`, so 64 of 256 pairs are live, offset `i+256`);
  - plain norm gains (the stored `w`, not `1+w`), and unscaled attention (`queryScalar` 1.0);
  - a per-layer `layer_scalar` and a final soft-cap of 30;
  - `model_type gemma4_unified_text` / `Gemma4UnifiedForCausalLM` (wrapper `gemma4_unified`).
- **Metal is supported:** merged as `238941c` ("metal: V-norm the f32 cache rows at their own offset") with 512-wide
  attention kernels (the `d512` family) and per-layer head-shape dispatch, so Gemma 4 runs on the CPU and the GPU;
  `attention_k_eq_v` (V = the pre-RoPE key projection), the parameterless per-head V-norm and the per-layer
  `layer_scalar` are applied on the GPU too. The paged, batched, mixed and tiered KV passes, q4 KV and rotated q8 are
  still not offered (there is no shared row across per-layer shapes).
  Explicitly rejected: Gemma 4 E2B/E4B per-layer embeddings, the 26B-A4B MoE, `num_kv_shared_layers>0` and
  `use_double_wide_mlp`.
- **Tokenizer:** GGUF `tokenizer.ggml.model=gemma4` is an SPM-style BPE that falls back to the original repository's
  `tokenizer.json` (the same fallback serves sarvam-moe, mmbert and whitespace).
- **Verified:** the tiny-random arch smoke and `SmokeTest`, a pure-Python reference (max logit diff 2.4e-4), and the
  real 12B (`unsloth/gemma-4-12b-it-GGUF` Q4_K_M, 7.1 GB) loaded and generated coherent text end to end on the CPU
  backend.

### 7. Review of the video follow-ups, and fixes found on an x86 machine — `ccr-be155af0-ycvdp8`

- **Branch:** `ccr-be155af0-ycvdp8`, `4341f53` plus the commits of 2026-10-27, **pushed to `main` at the owner's request** (a
  fast-forward). It passed the gates that run on Linux; no Mac was available, so the Mac gates are still to be run on `main`
  (see "Gates" below).
- **Review of `5cae543` / `4341f53` (the video follow-ups), and what changed:**
  - *GIF delays:* allowing 1 cs made clips above 50 fps play 2.8× too slowly, because browsers play 0–1 cs as 10 cs.
    `VideoEncoder.gifTiming` now drops frames so that every delay is at least 2 cs and the clip keeps its length (docs/23 §23.8).
  - *Wan text encoder:* the in-memory cache kept 5.2 GB beside the transformer and saved nothing (`jllm video` makes one
    request per pipeline). The encoder is freed before the transformer again; its int8 copy is kept on disk instead
    (`text_encoder/jllm-q8.bin` + `jllm-q8.json`, `video.text-encoder-cache`, docs/23 §23.12.6), so later runs map 5.2 GB rather
    than read 18.5 GB of float32 weights and quantize them. Checked in `WanTest` on the tiny encoder (the same bits, stale and
    truncated copies refused); a `WanBench t5` off/on row is in `mac-validate.sh`.
  - *`image.turbo`:* documented (docs/22 §22.8, §22.12.1, README); in the console it is now a load-time setting.
  - *Wan's 13-frame default* and *`video.base-model` validation:* fine as they were.
- **Found on the way:**
  - `VectorKernels.laneSum` left 16-lane vectors (AVX-512) to `reduceLanes`, so `JitStableKernelsTest` and `PagedKvTest`
    failed on x86 (cold-JIT last-bit differences); fixed. The CPU exactness md5 is unchanged on x86. The GGUF block dots and
    the K-quant int8 dots had the same difference (12 of 13 fixture formats); they use the fixed order now, and
    `JitStableKernelsTest` checks them before anything makes `reduceLanes` hot (which hid it). The image, audio and vision
    CPU kernels (`diffusion/CpuOps`, `audio/Dense`, `vision/VMath`) use it too; no float `reduceLanes(ADD)` is left.
  - `LongContextTest` needs 4.3 GB of direct memory; below ~18 GB of RAM the default limit was too small (now set in the pom).
  - The emulator's `packed_float4` / `packed_half4` had vector alignment, so `kernel_tests.py` segfaulted on x86 in
    `wh_gemm` (rows of 33 floats); fixed in `metal_emu.h`.
  - **The CPU paged KV cache could run out of direct memory:** its slabs come from an automatic arena, which counts against
    `-XX:MaxDirectMemorySize` (by default the heap: 4 GB with `./jllm`), but `backend.ctx=auto` sized it from physical memory
    alone. Reproduced (`JLLM_XMX=1g`: auto planned 26 624 positions, a 5 000-token prompt died with `OutOfMemoryError: Cannot
    reserve ... direct buffer memory`). `ContextBudget` now bounds the budget by the limit and `CpuBackend.newKvPool` warns for
    an explicit context that cannot be held (docs/10 §10.13, docs/24).
  - **Gemma 4 never reached the GPU through `./jllm`:** `ModelLoader.newBackend` still switched a non-uniform decoder to the
    CPU ("the Metal attention kernels do not support" 512-wide heads), a note left from `d7b388e` that the Metal commits of
    item 6 did not remove (`ArchSmoke` builds `MetalBackend` directly, so it did not notice); `ContextBudget` sized its
    context from the heap for the same reason. Both removed (`bd83a8e`). The same line had bent `backend.batch=auto`
    (256-token chunks for Mamba, dense Qwen3-Next and the hybrids on Metal, documented as 64; 64 for Gemma 4); the rule of
    `a0ef980` is back: 256 for a dense decoder on Metal, Gemma 4 included.
- **Docs (at the owner's request: detailed, and one page with the parameters to run each model best):**
  - New chapter **docs/28 Best settings for each model**: a command line per family, the settings ranked by effect with
    their measured cost and source, the `backend.ctx=auto` budget worked out for the 0.6B and the 30B (it reproduces the
    measured 18 432 / 37 888 / 74 752), agents, CPU-only x86 / Docker machines and the launchers' Java memory, a checklist,
    and what is not measured. Linked from the README (configuration and measured sections) and the docs index.
  - Stale statements fixed: the README's M1 Pro table (now ~2 100 / ~143 and ~300 / ~27, as docs/11 §11.1); speculation
    "off by default" / "MoE keeps it off" (docs/11 summary, docs/17 §17.1) and the missing `mtp` value (docs/10 table);
    `backend.cpu.int8-activations` "default `true`" (docs/07, now `auto`); Llama 4 and the Mamba-2 hybrids listed as not
    supported (docs/14 §14.6, docs/15 §15.9).
- **GGUF for the remaining families (at the owner's request):** `gpt-oss`, `llama4`, `granitehybrid`, `nemotron_h` and `jamba`
  files are read (docs/16 §16.16). What llama.cpp's loader fixes and the file does not say is written into the description as
  llama.cpp's constant (gpt-oss's SwiGLU clamp 7.0; Llama 4's chunk 8192, temperature, NoPE every 4th layer, q/k norm unless 128
  experts; the Mamba-2 gated norm per group, `Architecture.GGUF_GROUPED_SSM_NORM`; no step clamp). `HfToGguf` ports the five
  converters (MXFP4 repacked raw, gate/up de-interleaved, per-layer `head_count_kv` / `feed_forward_length`, Jamba's merged
  experts) and refuses what llama.cpp would compute differently; `GgufFamiliesCheck` has seven new variants (F32 GGUF vs
  safetensors <= 4.9e-7; all 124 checks pass), and `scripts/arch-smoke.sh` generates them for `GgufArchSmoke` (CPU half run;
  the Metal half and `scripts/gguf-llamacpp-check.sh` are for a Mac: llama.cpp could not be built here).
- **Gates on this branch** (Linux x86-64, Xeon with AVX-512, 4 cores, 15 GB, Temurin JDK 26.0.2; no Mac):
  - `mvn -o install`: BUILD SUCCESS. The web tests: 62 run, 0 failures. `scripts/build-metal-emu.sh`: exit 0.
    `kernel_tests.py`: 1172 ok, 0 FAIL.
  - Arch smoke on the emulator (`METAL_EMU=1 scripts/arch-smoke.sh`, one emulated GPU thread per fiber, so it runs for
    hours on this machine): **not finished when this was written.** The run on `64f27d1` had passed the DeepSeek reference,
    the saved-cache, Qwen3-Next and GGUF-Qwen3-Next checks, `GgufArchSmoke` (20 models) and every `ArchSmoke` check through
    LlamaPlain, LlamaDraft and part of Mistral with no `FAIL` (672 ok lines); an earlier run on `ecf4aa4` had reached Qwen2
    with no `FAIL` before it was restarted on the newer commit. `f4e9739` changes only the image, audio and vision CPU
    kernels, which arch smoke does not run. Run it to the end (or `scripts/arch-smoke.sh` on a Mac) before relying on it.
  - Exactness, CPU: `79472f690f73bd28598d5d7fb5dee1a4`, as on the Mac. The Metal half needs a Mac.
- **Left:**
  - Run the six gates on a Mac against `main` (the arch smoke to its end, the Metal half of the exactness run).
  - Run the real Wan encoder once (22.7 GB download) to measure the read time with the copy off and on (mac-validate row).
  - Run the real Gemma 4 12B through `./jllm` on Metal (`unsloth/gemma-4-12b-it-GGUF` Q4_K_M) now that the loader lets it:
    the start-up line must say `Backend: metal`, and its text should match `--backend cpu --backend.cpu.int8-activations false`
    greedy; then put its speed into docs/28 §28.5.5.
  - docs/28 §28.10 lists what is not measured yet; the CPU f16 cache on x86 against q8 can be timed on Linux.
  - On a Mac: `scripts/gguf-llamacpp-check.sh gpt-oss gpt-oss-mxfp4 llama4 llama4-mix granite4 nemotron-h jamba` (llama.cpp's
    logits on the new GGUF families) and `scripts/arch-smoke.sh` (their CPU == Metal half); then fill the llama.cpp column of
    docs/16 §16.16.
  - Done (2026-10-27, at the owner's request): the launchers raise `-XX:MaxDirectMemorySize` to the physical memory or the
    container's limit (`scripts/jvm-memory.sh`, `JLLM_DIRECT_MEMORY`), and closing a pool that held slabs runs one GC so an
    unloaded model's cache returns at once (docs/10 §10.13).

## Not started (small follow-ups found along the way)

- **Docs:**
  - A final detailed pass on docs 10, 11, 14, 20 and 24 to the standard above.
  - Done: docs 11.9 / 07.13 give the 44 MB ring for `backend.batch=auto` and say the 11.9 speed table predates `auto`; the
    docs 11 anchor `#cpu-native-int8-kernels-backendcpunative-07-716` matches GitHub's slug of the 11.1 heading; the SD 1.5
    base is 2.1 GB (UNet, text encoder and VAE in fp16; the safety checker is not downloaded), now also in the README and the
    console.
- **Video:**
  - Done (item 7): the int8 UMT5 copy on disk, Wan's explicit 13-frame default, GIF timing above 50 fps, `image.turbo`.
  - AnimateDiff clips are flat or smeared with plain SD 1.5; a fine-tuned SD 1.5 base (`video.base-model`) is needed.
  - Wan has never run end to end: its UMT5 encoder is only published as 22.7 GB fp32, which the download cap excluded.
- **Audio:** OuteText's text front end already equals the Python `outetts` TextProcessor on the reference sentences (the closing quote + adjacent comma/period and the colon are dropped in both, via AudioTest's `outetext-ref.json`); checked, no change needed.
- **Plan (`wip/NEXT-PROMPT.md`):**
  - Stage 0 item 6: the arm64 Docker image was not built or run on a Mac (Docker Desktop was not running).
  - Stage 1 agent 2: tree speculation (optional).
  - Stage 2 item 3: sparse prefill (MInference-style).
- **History:** about 90 commits on `main` from 2026-10-21 (18:00–19:48 UTC), pushed from another machine, carry AI
  attribution lines. They were left as they are: removing them would mean rewriting public history, which is the
  owner's decision.
