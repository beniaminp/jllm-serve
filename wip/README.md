# Work in progress that is not merged yet

Status on 2026-10-23, evening (Europe/Amsterdam). Everything listed here is pushed to GitHub on a `wip/*` branch, so
another person or agent can pick it up. `main` holds only work that passed every gate below. Each `wip/*` branch has
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

- **Branch:** `wip/scheduler` at `cc23365`, with main `e951fc7` merged in. `cc23365` is a WIP docs commit on top of
  `2a32e70`, which passed all six gates.
- **On its way to `main`:** `2a32e70` merged onto main as `ba44b68`, gates running. The merge resolved two conflicts:
  - `ModelLoader.kvPool` pages when `paged(...)` or `mixedWanted(...)`;
  - `Engine` keeps both `releaseKv` (idle slots, paged follow-up) and `releaseCache` (scheduler).
- **Works (`engine.scheduler=mixed`; `slots` stays the default and is unchanged):**
  - `Backend.forwardMixed` on CPU and Metal: one prefill chunk plus every decode token plus every draft verification in
    one pass;
  - `BatchGroup` mixed mode with `PassPlanner` (token budget, aging);
  - a cache-aware global admission queue in `LoadedModel`, with the `priority` classes, session limits, and preemption
    by suspending a sequence so its blocks spill and reload;
  - client-gone wake in under 60 ms;
  - tests: `SchedulerPolicyTest`, `MixedSchedulerTest`, `MixedPassChecks` in arch smoke, jllm-web `MixedSlotsTest`, and
    the `SchedulerBench` agent workload.
- **Left:**
  1. Second benchmark round (pass budget, background chunk cap, the 30B alone and with 2 slots), then fill the `TBD`
     markers in docs/25 §25.5 and the README numbers.
  2. Arch smoke with the concurrent encoder at the final commit.
  3. Decide the default from the numbers.
- **Known:**
  - Round 1 on Metal, 0.6B agent workload: slots with the flat cache 65 s, slots with paged 36.5 s, mixed 36.0 s wall.
    Mixed raises the inter-token p95 (56–196 ms against 37 ms) because decode rows ride a 256-row chunk pass, and lowers
    the p99 stalls. The pass budget is the knob.
  - The 30B MoE ran at about 1 tok/s per stream in the 3-slot bench (expert placement under 3 slots); not understood
    yet.
  - `ContextBudget` counts the off-heap paged pool only for `backend.kv-paged=true`, not for `auto` or when the mixed
    scheduler turns paging on.

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

### 3. Sparse decode attention — `wip/sparse-attention` (docs/26)

- **Branch:** `wip/sparse-attention` at `9a41c17`, main `e951fc7` merged in.
- **Works (`backend.kv-sparse`, off by default; it turns on the paged cache):**
  - per-block key summaries (half min/max per channel), the Quest upper bound, top-k blocks plus sinks plus the recent
    window, on the CPU paged cache and on Metal (`kv_summary`, `sparse_score_d*`, `sparse_select`, `attention_dec_sp`);
  - wired into decode, short verification chunks, batched rows, replay and saved caches;
  - checks: `cpu/SparseKvTest` (1405 checks, in `mvn install`), the kernel_tests sparse section, and `KvStudy
    sparsecheck` on the real GPU (replay on = off bit for bit);
  - perplexity at 8k/16k/32k is in docs/26 §26.5.2.
- **Left:** the gates on this branch, the needle-in-a-haystack tables at 8k/16k/32k, decode tok/s against context
  length, and anything at 128k+ (the 30B supports 262k but did not fit beside other jobs).
- **Known:** CPU and Metal sparse logits differ by up to 0.4–2.8, because a rounding-order difference flips which block
  is selected. Argmax agrees 23–24 of 24.
- **Docs numbering:** both this branch and the tiered-KV branch wrote `docs/26`. The tiered-KV branch merged first with
  `docs/26-tiered-kv-and-memory-planner.md`, so this branch's `docs/26-sparse-attention.md` renumbers to `docs/27` when
  it merges.

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

- **Branch:** `wip/paged-kv` at `9a6a261`, main `92a57a1` plus one WIP docs commit. The code itself is on `main`.
- **Left:**
  - Re-measure CPU f16/q8/f32 decode, paged against flat, under a quiet machine.
  - Replace the placeholders `@@CPUCOST@@` in docs/24 and `@@CPUREADME@@` in README, and update the §24.5 CPU rows and
    the §24.3 costs line.
  - The §24.2.6 claim that the older (ternary) f16 version cost 4× must be confirmed or softened from those numbers.
  - **Do not merge this commit as it is: it has placeholders.**

## Not started (small follow-ups found along the way)

- **Docs:**
  - A final detailed pass on docs 10, 11, 14, 20 and 24 to the standard above.
  - docs 11.9 / 07.13: the q4 recent-window ring is `128 + backend.batch` rows, about 44 MB for Qwen3-0.6B with
    `backend.batch=auto` (256 on Metal), not the 22 MB quoted for batch 64; the 11.9 speed table predates `auto`.
  - docs 11, line about the native int8 kernels: check the anchor `#cpu-native-int8-kernels-backendcpunative-07-716` on
    GitHub.
  - docs 23.9 says SD 1.5 fp16 is 2.6 GB, while 22.2 says 2.1 GB (the 2.6 GB may include the safety checker).
- **Video:**
  - `Umt5Encoder` is rebuilt on every Wan request: the 22.7 GB f32 file is re-read and re-quantized each time; cache
    the int8 copy.
  - `jllm video`'s default `video.frames=16` becomes 13 frames for Wan (4k+1 rounding).
  - GIF frame delays are clamped to at least 2 cs, so above 50 fps a GIF plays slower than asked.
  - The Turbo defaults are chosen by the model folder name only.
  - AnimateDiff clips are flat or smeared with plain SD 1.5; a fine-tuned SD 1.5 base (`video.base-model`) is needed.
  - Wan has never run end to end: its UMT5 encoder is only published as 22.7 GB fp32, which the download cap excluded.
- **Audio:** OuteText on `Don't panic: it's "fine".` drops the colon, the closing quote and the period; check against the
  Python `outetts` package.
- **Plan (`wip/NEXT-PROMPT.md`):**
  - Stage 0 item 6: the arm64 Docker image was not built or run on a Mac (Docker Desktop was not running).
  - Stage 1 agent 2: tree speculation (optional).
  - Stage 2 item 3: sparse prefill (MInference-style).
- **History:** about 90 commits on `main` from 2026-10-21 (18:00–19:48 UTC), pushed from another machine, carry AI
  attribution lines. They were left as they are: removing them would mean rewriting public history, which is the
  owner's decision.
