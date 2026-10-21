# Prompt: finish the open work and the next phases of jllm-serve with parallel agents

Paste everything below the line into Claude Code, started at the root of your clone of `jllm-serve` on `main`.

---

You are the lead on **jllm-serve**, a pure-Java LLM inference engine:
- a CPU backend (Vector API, plus optional native int8 kernels through the FFM API);
- a Metal backend (our own kernels in `src/main/resources/org/robomiri/jllm/metal/kernels.metal`, a CPU emulator of
  them for Linux);
- an OpenAI-compatible server and the `jllm-web` console;
- a Docker image.

Read `README.md`, `docs/README.md`, `wip/README.md` and this whole prompt before starting. Then run the work below
with **parallel subagents, each in its own git worktree** (Agent tool with `isolation: "worktree"`). Merge their
branches into `main` yourself, one at a time, verifying after each merge.

## Ground rules (give these to every agent)

- **Build and test:**
  - JDK 26: `mvn -q -o install` builds the engine and runs every test; `cd jllm-web && mvn -o test` runs the console
    tests.
  - The Metal kernels are checked without a GPU by the emulator:
    - `scripts/build-metal-emu.sh`;
    - `python3 scripts/metal-emu/kernel_tests.py` (numpy references);
    - the full `METAL_EMU=1 ARCH_MODELS=<dir of tiny models> scripts/arch-smoke.sh`. It makes the tiny models
      itself with `scripts/make-tiny-model.py`, compares CPU against Metal for every family, and takes 1–2 h. Run it
      detached and watch its log.
  - **If this machine is an Apple-silicon Mac, use the real GPU:** `scripts/arch-smoke.sh` without `METAL_EMU`,
    and `scripts/mac-validate.sh`, which benchmarks every feature on and off and, with `--vs-llama`, against
    llama.cpp's `llama-bench` on the same GGUF. Emulator timings mean nothing; only Mac numbers count for speed.
- **Correctness bar:**
  - Every new kernel gets a numpy reference in `kernel_tests.py`, and ArchSmoke must end with `arch smoke passed`.
  - Greedy output must stay identical where the docs promise it: CPU float path vs Metal, speculation on vs off,
    batched vs alone, replay on vs off, saved/loaded caches.
  - A lossy feature (quantized caches, sparse attention) gets a measured accuracy table against f16 and a
    needle-in-a-haystack retrieval check before it can become a default.
  - Never weaken a test to make it pass; explain any tolerance in a comment.
- **Every new feature:**
  - has a switch (a validated setting in `Settings` + `jllm.properties`, or a `-Djllm...` flag);
  - keeps the old path when switched off;
  - gets an on/off row in `scripts/mac-validate.sh`.
- **Docs:** every feature gets a section in the matching `docs/NN-*.md`: first a 🧸 explanation a five-year-old
  could follow, then 📐 the exact math and design. Also update `README.md` (the "Beyond a plain port" table,
  settings) and `docs/README.md`. Section numbers collide between parallel branches: renumber when merging, and fix
  every `§` reference and `#anchor`.
- **Conflict hot spots:** `MetalBackend.java`, `MetalKernels.java`, `KernelParams.java`, `kernels.metal`,
  `MetalTuner.java`, `ArchSmoke.java`, `Engine.java`, `Speculator.java`, `CpuBackend.java`, `KvCache.java`,
  `pom.xml`, `scripts/arch-smoke.sh`, `scripts/mac-validate.sh`, `kernel_tests.py`, docs 08 / 10 / 11 / 17.
  - Add new kernels and methods rather than changing shared signatures.
  - Keep existing Metal buffer indices stable.
  - Give an agent the files it owns and tell it which ones others are changing.
- **Machine load:** don't run two heavy Java test runs at once, and use `nice`. Agents must not kill processes
  they did not start.
- **Commits:** commit often, with clear messages. Agents commit on their worktree branch only; you merge and push to
  `main` after `mvn install` and the web tests pass on the merge.

## Stage 0: stabilize what was just merged (do this first, mostly yourself)

`main` received six large merges at once, made by parallel agents that were stopped before final validation:
- Qwen3-Next;
- constrained tool calls;
- half-weight MMA prefill;
- decode replay and fusion;
- Metal batched decoding with draft models;
- the native int8 CPU kernels (validated).

Steps:
1. Run `mvn -o install`, the web tests, `kernel_tests.py` and the **full emulator arch smoke** on `main`, and on
   real Metal if you have a Mac. Fix whatever fails. Check these interactions in particular, which no test covered
   together:
   - decode replay × batched decoding;
   - replay × draft-model pipelining;
   - replay × the MoE CPU/GPU expert split and `rebalanceExperts`;
   - replay × saved caches;
   - Qwen3-Next × batching and draft;
   - constrained decoding × draft speculation.
2. **Half-weight MMA prefill** (`*_hmma`, `attention_fb`, chosen per shape in `MetalTuner.prefillMatmuls`):
   `scripts/arch-smoke.sh` currently passes `-Djllm.metal.hmma=false`. When forced on, Qwen3-MoE showed CPU-vs-Metal
   logit differences of ~4e-3 (baseline ~7e-6), and the re-placement check `rebalanceCheck` moved 0 layers
   (`REBALANCE FAILED`). Find the cause:
   - the precision loss of half weight tiles,
   - or a bug in the MoE `_hmma` kernels or their counting of expert usage.

   Fix it, or restrict hmma to the shapes where it is exact enough. Then remove the flag from arch-smoke so hmma is
   covered.
3. **Constrained tool calls** (`sampling/ToolCallConstraint`, `ByteConstraint`, `TokenTrie`, `JsonConstraint`,
   jump-forward in `Engine`, speculation under constraints in `Speculator`, `web/tools/ToolCallFormat`, docs 17 §17.7):
   its last commit was work in progress. Make `ToolCallTest` and `ConstraintBench` pass and add them to the pom.
   Measure µs per token of the mask before and after, check greedy constrained output with and without speculation
   on Qwen3-0.6B, and finish the docs.
4. **4-bit KV cache**: apply `wip/q4-kv-cache.patch` (`git apply --3way`) and resolve its conflicts as
   `wip/README.md` describes. Then:
   - finish the accuracy study (q8, q4 without rotation, q4 with Hadamard-rotated keys, with and without the recent
     window, against f16 on Qwen3-0.6B over 300 greedy tokens, plus a needle-in-a-haystack check at the longest
     context the machine allows);
   - choose the defaults from that data;
   - add q4 to replay, batched decoding and the saved-cache formats;
   - run arch smoke, write docs 05 / 07 / 08 / 11, then delete `wip/q4-kv-cache.patch`.
5. Re-run `scripts/mac-validate.sh --vs-llama` on a Mac (and `--moe unsloth/Qwen3-30B-A3B-GGUF` with
   `--llama-moe unsloth/Qwen3-30B-A3B-GGUF` if memory allows).
   - The features before this merge were measured on an M1 Pro (README, docs 08 §8.20, docs 11): llama.cpp is still
     about 2× faster.
   - Decode replay and fusion, half-weight MMA prefill, Metal batched decoding and draft speculation are not
     measured yet.

   Update the tables, and use them to decide what to tune first.
6. Build and run the arm64 Docker image on an ARM machine or a Mac (`scripts/docker-build.sh`; docs/19). The native
   CPU library's NEON/dotprod levels were only checked under qemu.

## Stage 1: agentic serving (parallel agents after stage 0)

Context for these agents:
- **How agent traffic looks:** coding agents resend the whole conversation every turn, with long shared prefixes
  (system prompt, tools, file contents), and the prompt is ~20× longer than the output. Several agents share
  prefixes.
- **Engine today** (`engine/Engine`, `engine/BatchGroup`, `jllm-web` `models/LoadedModel`, `models/KvStore`):
  - Each slot is an `Engine` with its own flat, full-size KV cache.
  - Prefix reuse is a linear compare against that slot's tokens, and slots share prefixes by copying.
  - The slot is chosen once, with no global queue.
  - Prefill chunks (64 tokens, 256 for MoE) and other slots' batched decode steps contend for a non-fair device
    lock with no policy.
  - Batched decode groups only single-token steps, and speculative verification passes run alone.

Agents:
1. **Paged KV cache with a prefix tree** (the foundation for most of the rest; one agent, CPU first, then Metal):
   - KV memory as a pool of fixed-size blocks (e.g. 16 tokens) per model, allocated as a conversation grows, outside
     the Java heap (`MemorySegment`) on the CPU and per-block buffers or one big buffer with a block table on Metal.
   - A radix tree over token ids maps prefixes to blocks with reference counts and copy-on-write: identical prefixes
     are stored once across slots and conversations, and branches survive (retries, subagents forking from one
     context).
   - LRU eviction of unreferenced blocks, then spilling to the saved-cache files (`KvStore` becomes a block store).
   - Attention kernels read through the block table (CPU tiled kernels; Metal split / flash / prefill kernels), with
     f32 / f16 / q8 (and q4 after stage 0) blocks.
   - Audit every `int` index for 1M-token contexts (positions × kvDim × layers overflows 2³¹), and add a 1M-token
     test on a tiny model.
   - Keep the flat cache selectable until the paged one matches it bit for bit.
2. **Scheduler with mixed batches** (after or with 1):
   - One global queue per model with cache-aware routing: the longest cached prefix wins, and work can move to any
     free capacity.
   - Each forward pass carries one prefill chunk plus the next decode token of every generating conversation, so
     decode rides along on compute-bound prefill (Sarathi-style chunked prefill).
   - A token budget per pass; fairness; priority classes (interactive vs background subagents); per-session limits;
     and pausing a long generation by spilling its blocks instead of making others wait.
   - Batched speculative verification: every slot's draft (k+1 tokens) verified in one pass. Optionally tree
     speculation, which verifies several candidate continuations from the n-gram index at once.
   - Wake queued requests on client disconnect (`web/chat/ClientGone` exists).
3. **Defaults and small items:**
   - choose the prefill chunk per chip with `MetalTuner` instead of a fixed 64;
   - re-measure whether n-gram speculation should stay off by default for MoE models;
   - size the default context from available memory (agents need 32k–128k);
   - expose priority in the API.

## Stage 2: long contexts (up to 1M tokens) with less RAM / VRAM

At 1M tokens an f16 KV cache is about 98 GB for Qwen3-30B-A3B and 115 GB for Qwen3-0.6B. Plain attention also
reads the whole cache for every token, so we need to both store less and read less.

1. **Sparse attention while decoding** (needs the paged cache):
   - Keep per-block key summaries (per-channel min/max, Quest-style).
   - Each new token scores the summaries and attends exactly over the top-k blocks, plus the first tokens (sinks) and
     the recent window.
   - Make k a setting. Measure retrieval accuracy (needle tests at 32k–1M on a long-context model) against dense
     attention. Off by default until measured.
2. **Tiered KV storage:**
   - Cold blocks move to host RAM or an SSD file (memory-mapped). On a Mac, Metal's working-set limit
     (`recommendedMaxWorkingSetSize`, `DeviceLedger`) is the "VRAM".
   - Summaries, recent blocks and hot blocks stay resident; the blocks sparse attention picks are paged in on
     demand, with prefetch.
   - Saved caches use the same block files, in the q4 format, so a 1M-token session reloads quickly and small.
3. **Sparse prefill** for very long prompts (block-sparse attention patterns, MInference-style), switchable, with
   accuracy checks.
4. **Memory planner:** one budget across weights (including the MoE expert split), KV blocks and activations. It
   trades GPU-resident experts against KV blocks as contexts grow, and reports it in the console.
5. **Long-context model families:**
   - gpt-oss: alternating 128-token sliding-window layers, attention sinks, MXFP4 MoE;
   - Llama 4: chunked attention;
   - Dual Chunk Attention, used by Qwen's 1M-context models;
   - other hybrids with linear-attention layers (Nemotron-H, Granite 4, Jamba), reusing the Mamba and Qwen3-Next
     state machinery.

   Verify each against a transformers or numpy reference, as `scripts/real-weights-check.sh` and
   `scripts/qwen3next-ref.py` do.

## How to run it

- **Stage 0:** sequential and mostly yours. Items 2, 3 and 4 can be three agents once item 1 is green.
- **Stage 1:** start agents 1 and 3 in parallel; start agent 2 when agent 1's block table API is stable.
- **Stage 2:** start items 1, 2 and 4 when stage 1's paged cache is merged. Item 5 (families) can start any time,
  in parallel with everything.
- **Each merge:** run `mvn -o install`, the web tests, `kernel_tests.py`, and arch smoke (emulator or Mac). Fix
  before the next merge. Then push to `main`.
- **At the end:** update README / docs / `docs/11-performance.md` with measured numbers from a real Mac, and list
  what is still unmeasured.
