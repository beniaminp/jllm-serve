# 26. Tiered KV storage and the memory planner

The paged cache of chapter 24 stores the keys and values of every conversation in one pool of small blocks. This chapter
makes that pool **a hierarchy**: the blocks that are in use right now live where the GPU can read them, the ones that are not
move to ordinary memory, and the ones nobody has asked for in a while move to files on the SSD; a block comes back when a prompt
needs it, usually before it is asked for. A conversation that is longer than the GPU's share of memory is kept in host memory
altogether and fed to the GPU one layer at a time. A **memory planner** decides, with one budget, how many bytes go to the
weights (and how many of a mixture-of-experts model's experts stay on the CPU), to the KV blocks and to everything else, and shows its
decisions in the console.

| | | where |
|---|---|---|
| [26.1](#261-in-plain-words) | the idea, told as a story | |
| [26.2](#262-how-it-works) | the tiers and the hot cap, streamed KV, the host-memory tier and the block files, the on-disk row formats, saved sessions as block files, the memory planner, the report | `kv/TieredSpill`, `kv/KvBlockCodec`, `kv/KvTier`, `metal/StreamedKv`, `metal/MemoryPlanner`, `metal/MemoryGovernor` |
| [26.3](#263-why-this-design) | what was rejected, and what the design costs | |
| [26.4](#264-settings) | `backend.kv-tier`, `backend.kv-hot-mb`, `backend.kv-host-mb`, `backend.kv-disk-format`, `backend.kv-stream`, `backend.memory-plan`, ... | |
| [26.5](#265-measured) | decode speed resident against tiered at 32 000 tokens, the 30B mixture-of-experts model with a growing context, the accuracy of the disk formats | `TierStudy`, `TierDiskStudy` |
| [26.6](#266-checks-and-limits) | the tests, and what is not done | |

---

## 26.1 In plain words

🧸 Imagine a cook who works at a small kitchen counter next to the stove (that is the **GPU's memory**), with a pantry down the
hall (the **computer's ordinary memory**) and a warehouse across the street (the **SSD**). The recipe books of all the dishes the
restaurant has ever cooked are the KV cache: to cook the next course the cook has to look at the pages of the dishes of *this*
evening's table.

* Everything the cook is using right now is on the counter. When the counter is full, the pages of dishes nobody is cooking go
  to the pantry, and the pages the pantry has no room for go to the warehouse. Nothing is thrown away; every page has a label
  (its path of words), so it can be found again.
* When a guest sits down who orders something cooked before, a helper starts carrying the pages back from the warehouse to the
  pantry **as soon as the order is taken**, so by the time the cook needs them they are one step away.
* Sometimes one single dish needs more pages than fit on the counter: a book of thirty thousand pages. Then the whole book stays
  in the pantry and the cook has it brought **one chapter at a time** (one layer of the model at a time): while the cook works
  on chapter 5 at the counter, the helper puts chapter 6 on the second half of the counter. It is slower than having the book on
  the counter, but the dish can be cooked, and it comes out exactly the same.
* A **planner** looks at the whole kitchen. The counter has to hold the cook's tools (the model's weights), the pages (the KV
  cache) and the plates (scratch space). For a model made of many specialist cooks (the experts of a mixture-of-experts model), a
  specialist who is rarely called can work in the pantry instead, and be fetched by phone: that is slower for the dishes that need
  them, but it frees room on the counter for pages. The planner compares that price with the price of fetching pages, picks the cheaper,
  and moves a specialist out only when the pages really need the room, and not back and forth every few minutes.

---

## 26.2 How it works

📐 The running example is Qwen3-0.6B (28 layers, 8 kv heads of 128, so a position row is `kvDim = 1024` values): one position
costs `28 · 2 · 1024 · 2 = 114 688` bytes at f16, a 16-position block `1 835 008` bytes (1.75 MiB), a 32 768-token context
3.76 GB. The second example is Qwen3-30B-A3B (48 layers, 4 kv heads of 128): 98 304 bytes per position at f16, 12.9 GB at 131 072
positions, next to 19.4 GB of Q4 weights.

### 26.2.1 The tiers

| tier | where | holds | read by | budget |
|---|---|---|---|---|
| **hot** | the block pool: Metal buffers (the GPU's share of memory) or, on the CPU, slabs in the process's memory | blocks of running sequences and the most recently used cached blocks | the attention kernels, through the block table (chapter 24) | `backend.kv-hot-mb` (the *hot cap*) |
| **warm** | off-heap host memory (`TieredSpill`) | payloads of blocks pushed out of the pool | adoption of a block into the pool (a `memcpy`), and prefetch | `backend.kv-host-mb` |
| **cold** | block files `<dir>/blocks/<fingerprint>/<path hash>.kvb`, read through a memory mapping (`FileSpill`) | what the warm tier overflowed, and blocks written for a saved session | adoption / prefetch | `backend.kv-tier-max-gb` (least recently used files go first) |

A block moves **down** when the pool needs room (`KvPool.alloc` finds no free block and the committed blocks have reached the hot
cap): the least recently used *cached* block with no resident child is chosen (chapter 24's leaf-first rule), its rows are written in the canonical
layout to the spill store, and its slot is reused. It moves **up** when a prompt's tokens find it: `KvPool.match` follows the
path hashes past the last resident node and counts the blocks the store holds, `adopt` reads them into fresh slots.

```
 hot cap = 5 blocks      TieredSpill (budget = 2 blocks)            files
 +--------------------+  put   +-------------------------+  overflow, in the background
 | b7 b2 b9 b4 b1     | -----> |  p3  p8                 | ----------------------------> p5 p1 p6 ... .kvb
 +--------------------+        +-------------------------+
        ^   adopt (memcpy)             ^   prefetch (read + decode, in the background)
        +------------------------------+--------------------------------------------------+
```

The **hot cap** is a ceiling on the blocks the pool ever commits (`KvPool.setHotCap`): the pool still has address space for
`maxBlocks`, but hands out at most `hotCap` ids; beyond it every allocation evicts. A block already committed cannot be given back
to the system (Metal buffer pages stay committed), so lowering the cap only stops growth; the cap is a high-water budget. If every
block is held by running sequences, `alloc` throws `KvPoolExhausted`, now naming `backend.kv-hot-mb`.

Metal's pool buffers belong to a Metal context of their own, so until this chapter no `DeviceLedger` entry counted them, and the
placement of a mixture-of-experts model's experts (`backend.metal.cpu-moe=auto`, chapter 12) budgeted weights and per-slot buffers
only. Every pool now registers itself in `DeviceLedger` with `hotCap · blockBytes` (what the GPU has to keep for it), so the experts
planned at load leave room for the KV the pool may grow to. That fix is independent of `backend.kv-tier`: any paged Metal pool is counted.

### 26.2.2 Streamed KV: a context larger than the hot tier

Dense attention reads every key and value of the conversation for each new token, so a *running* sequence needs all its rows
reachable at once. If a slot's whole context (`backend.ctx` positions) is more than the hot budget can hold, the slot cannot use the
pool. It keeps its rows in **host memory** instead (`metal/StreamedKv`) and gives the GPU **two windows**: a K buffer and a V buffer of `ctx`
rows each, used by even layers and by odd layers alternately. A pass over positions `[pos, pos+n)` is:

```
 CPU thread:   stage(0)      stage(1)            stage(2) ...            writeBack(L-2), writeBack(L-1)
 GPU:                 [ layer 0 ]        [ layer 1 ]        [ layer 2 ] ...      [ layer L-1 ]
 window A (even):  rows 0..pos of layer 0 | layer 0's new rows ... | rows 0..pos of layer 2 | ...
 window B (odd):                         rows 0..pos of layer 1 | ...
```

For layer `l`: **stage** copies rows `[0, pos)` of layer `l` from the host into window `l & 1` while the GPU is busy with layer `l-1` in the other
window; the GPU runs layer `l` as its own command buffer (the ordinary flat kernels: QKV projection, rope, store into the window at
`pos…`, attention over `[0, pos+n)`, output projection, MLP); **writeBack** copies the `n` new rows of layer `l-1` from its window to the
host once layer `l-1` is done (`waitPending(1)`: at most layer `l` still unfinished). Because the GPU runs the very kernels of the flat cache on
the very bytes of the flat cache, the logits are **bit-identical** to a flat Metal cache: nothing in the attention arithmetic knows
the rows were elsewhere a moment ago.

The cost is one copy of the whole cache through the CPU per forward pass, and one command buffer per layer:

```
bytes staged per pass  = pos · rowBytes · 2 (K and V) · layers
time per decode token ≈ staged bytes / copy bandwidth  +  layers · (round trip)
Qwen3-0.6B f16 at pos = 32 000:  32 000 · 2 048 · 2 · 28 = 3.67 GB   →  122 ms at 30 GB/s
```

(the copy runs on several threads; `jllm.metal.stageBytesPerUs` overrides the bandwidth the planner assumes, and the console shows the
measured one). The GPU memory such a slot needs is the two windows, `4 · ctx · rowBytes` = 262 MB for Qwen3-0.6B at `ctx` 32 768 and f16, instead of all
layers' rows (3.76 GB). The host side is `2 · layers · ctx · rowBytes` bytes; layers beyond `backend.kv-host-mb` are **memory-mapped files** (unlinked at once, so
they go with the process) which the operating system pages between RAM and the SSD, and the next layer's pages are announced with `madvise(WILLNEED)`.

A streaming slot has no block table, so it does not share prefixes with other slots; the engine's own in-slot prefix reuse, saved
conversation files (`saveState` reads the host rows), speculation and rewinds work unchanged. Decode replay, pipelined greedy
decoding and batched rows are off for it (a pass changes windows between layers, so it is not one recorded program).

### 26.2.3 The warm tier and the block files

`TieredSpill` implements the pool's `KvSpill` interface (`has / put / get / remove / prefetch`), so the pool did not change shape:

* `put` copies the payload (canonical layout, 1.75 MiB for the example) into off-heap memory and returns. When the tier is over
  its budget the least recently used payloads are written to the lower store by **one background thread** and dropped from memory
  *after* the write has finished; until then they are still readable, so a block is never in neither place.
* `get` is a memory hit, or reads through to the lower store. `FileSpill.get` maps the payload of the file read-only
  (`FileChannel.map` with a confined arena, copied once, unmapped at once) for payloads of 64 KB and more.
* `prefetch(hash, tokens, bytes)` reads a block from the files into the warm tier in the background. `KvPool.match` calls it for
  every block it finds beyond the resident part of a prompt's path, so the adoption that follows finds them in memory.
* On the **CPU** the pool's slabs (16 MB each) beyond the hot budget are memory-mapped files in `<dir>/slabs/` (`CpuKvStorage.setHotBudget`): the same code reads and
  writes them, and the operating system keeps the busy pages in RAM. The warm tier is not needed there (the pool already is
  host memory), so `backend.kv-host-mb` is best 0 on the CPU.

Worked example (the unit test `KvTierTest`, 16-token blocks): 20 conversations of 160 tokens (10 blocks each) go through a pool whose hot cap is 60 blocks with a warm tier of 40
blocks. After the last one 140 blocks have been evicted, 100 of them demoted to files; the pool never committed more than 60
blocks; all 20 conversations then adopt their whole prefix with identical rows, 180 blocks reloaded.

### 26.2.4 The on-disk row formats

Blocks on the SSD may be smaller than blocks in memory (`backend.kv-disk-format`). For an f32 or f16 pool, `KvBlockCodec` converts the canonical payload
(per layer, per position: the K row, then the V row, `kvDim` values each) as follows, per row and kv head `h` of `headDim` values `x`:

1. rotate `x ← H x` with the unnormalized Walsh–Hadamard transform of block `b = lowestOneBit(headDim)` (`KvQ4.rotate`, additions and subtractions only): a few large channels
   become many medium ones, which is what makes 8 or 4 bits enough (chapter 7 §7.13);
2. **q8**: `s = max|x| / 127`, `q = rint(x / s)` as int8, one f32 scale per head: `kvDim + 4 · kvHeads` bytes per row (the layout of `backend.kv-dtype=q8`);
   **q4**: groups of 32, `d = half(max|x| / 7.5)`, `code = clamp(rint(x / d + 7.5), 0, 15)`, `kvDim/2 + 2 · kvDim/32` bytes per row (the layout of q4, `KvQ4.quantize`).

Reading back dequantizes, applies the transform again and divides by `b` (`H·H = b·I`), then writes f16 or f32. The rotation is applied to the values too: a stored block is only ever
read back into the pool, so nothing has to be rotated at attention time, and a rotated, quantized value row has the same error as a key row.

| pool row | native | q8 file | q4 file |
|---|---|---|---|
| f32 | 4 B / value | 26.6 % (1.0625 B) | 14.1 % (0.5625 B) |
| f16 | 2 B / value | 53.1 % | 28.1 % |
| q8 (rotated keys) | stored as it is (re-quantizing a q8 row loses more for no saving) | | |

Native is the default and exact. The lossy formats are for a long session that is mostly stored (accuracy: §26.5). A file's fingerprint includes the
pool's row layout and the disk format, so a run with other settings never reads it; the disk format only applies below the warm tier (memory holds native payloads).

### 26.2.5 Saved sessions as block files

With `backend.kv-tier` the conversation files of chapter 17 §17.3 are not written for a paged model; a saved session *is* its blocks
(`KvStore.saveBlocks`): `KvPool.persist(tokens, n)` walks the tree along the conversation's tokens and writes the full blocks the store does not hold yet (nothing else), `flushSpill()`
moves the warm tier to the files, and reloading needs no file of its own, because the next prompt's tokens find the blocks by their path hashes (`match` + `adopt`, prefetched). Two things follow:

* a long session reloads **lazily and only as far as the new prompt matches**: a 100 000-token session whose next turn shares the first 90 000 reads 5 600 blocks, not the file;
* the same files serve every conversation that shares a prefix (a system prompt is stored once), and with `q8` / `q4` disk formats the session is a half to an eighth of the bytes of a native one.

### 26.2.6 The memory planner

One budget: Metal's recommended working set `W` (`recommendedMaxWorkingSetSize`, or `-Djllm.metal.workingSetGb`). What it must hold:

```
fixed  = non-expert weights + activations and scratch + other loaded models           (DeviceLedger)
E      = bytes of all experts of the MoE layers                                        (0 for a dense model)
x      = bytes of experts moved to the CPU          (chosen by ExpertPlacement.plan: coldest first)
K      = KV bytes on the GPU                        (resident: kvPerPosition · positions;  streamed: the windows)
GPU    = fixed + (E - x) + K     must be <= W
```

`MemoryPlanner.plan` compares two ways to make the GPU fit and takes the cheaper per generated token:

* **resident KV, experts to the CPU**: `x = fixed + E + K - W`, cost `C_e(x)` = the expert placer's own estimate: a round trip (100 us) for every layer that has an expert on the CPU,
  plus, for every expert on the CPU, the expected picks per token times its bytes divided by the CPU bandwidth (60 GB/s). Independent of the context.
* **streamed KV**: `x` for the (much smaller) window need, plus `C_s = layers · 100 us + typicalPositions · kvPerPosition / stageBandwidth`: it grows with the context.

Worked example, Qwen3-30B-A3B at f16 on a 22 GB working set, 2 GB fixed, 19.0 GB of experts in 6 144 experts (48 layers of 128), no usage statistics yet
(`MemoryPlannerTest` prints this table):

| positions of KV | KV bytes | experts to the CPU | planner's cost per token |
|---|---|---|---|
| 4 096 | 0.4 GB | 0 | 0 |
| 16 384 | 1.6 GB | 256 of 6 144 | 1.0 ms |
| 65 536 | 6.4 GB | 1 792 | 7.2 ms |
| 131 072 | 12.9 GB | 3 968 | 15.9 ms |

Streaming 131 072 positions would cost `131 072 · 98 304 / 30 GB/s` = 430 ms per token, so experts win at every length here; at 1 000 000 positions (98 GB) resident cannot fit even with all the
experts on the CPU and streaming is the only plan that fits. With usage statistics (a few experts take most picks) the same bytes cost a fraction (6.7 ms → 1.5 ms in the test).

**Live.** With `backend.memory-plan` the pool starts small (8 192 positions' worth, so the experts planned at load leave only that for KV) and `MetalBackend.memoryTick()`, between passes, watches its committed blocks:

1. at 85 % of the cap and at most every 2 s, `MemoryPlanner.nextCap` proposes `max(1.5 · cap, 1.5 · committed)` blocks, bounded by the pool (a geometric step with headroom: the cap does not creep);
2. the planner is run for the proposed KV; if even then the GPU does not fit (`fits == false`) the growth is **refused** (recorded, retried after 30 s) and sequences that need more fail with the
   `KvPoolExhausted` message naming the setting; if experts must move, `replaceExperts` re-places the layers whose plan changed, one layer at a time (the new hot experts are packed, the GPU switches, the old copy is freed);
3. a *voluntary* re-placement (the plan fits but a cheaper one exists, because usage statistics changed) happens only through `MemoryPlanner.Hysteresis`: at least 20 % cheaper and at
   least 30 s since the last change; a plan that no longer fits replaces the old one at once. Demand flapping by 10 % causes no change (test).

Growth is one-way because committed pages cannot be returned: after a long context the experts stay on the CPU until the model is reloaded (not done: shrinking, §26.6).
The streaming cost model is figured at half the context; the measured copy bandwidth replaces the estimate once a streaming slot has run. At load, with no
`backend.kv-hot-mb`, `MetalBackend.planStreams` plans for `slots · ctx` positions and chooses streamed slots only when that is the cheaper plan or the only one that fits.

### 26.2.7 The report

`Backend.memoryReport()` is a JSON-ready tree; `GET /api/memory` returns it for every loaded text model, and the Models page of `jllm-web` has a **Memory** card: a bar of the GPU
budget split into weights, scratch, KV blocks (or windows) and other models, the experts on the CPU, the KV tiers (blocks hot / warm / cold, evictions, reloads, prefetch hits), the streaming
counters and the planner's latest decisions. The plan is also printed at load (`Metal memory plan: ...`) and each decision as it happens.

```json
{ "model": "qwen3-30b", "slots": 1, "report": {
    "backend": "metal",
    "device": { "budgetBytes": 21800000000, "usedBytes": 21250000000, "fits": true,
                "components": [ {"name": "weights on the GPU", "bytes": 18100000000, "note": "..."},
                                {"name": "activations and scratch", ...}, {"name": "KV blocks (hot cap)", ...} ] },
    "experts": { "totalBytes": 19046400000, "onCpuBytes": 1300000000, "onCpuExperts": 380, "cpuLayers": 41, "mode": "auto" },
    "kv": { "mode": "paged", "blockTokens": 16, "blockBytes": 1572864,
            "hot": { "capBlocks": 768, "committedBlocks": 701, "usedBlocks": 640, "cachedBlocks": 61, ... },
            "warm": { "blocks": 120, "bytes": ..., "budgetBytes": ..., "prefetchHits": 9 },
            "cold": { "files": 4300, "bytes": ... }, "traffic": { "evictions": ..., "reloads": ..., "prefetchHints": ... } },
    "plan": { "enabled": true, "summary": ["..."], "decisions": [ {"atMillis": ..., "kind": "grow", "detail": "hot cap 512 -> 768 blocks ..."} ] } } }
```

---

## 26.3 Why this design

* **Tiers in the spill interface, not in the pool.** The pool already had the right seam: an evicted block is written through `KvSpill` and found again by its path hash. A warm tier is
  one more `KvSpill` in front of the files; the pool, the tree, the copy-on-write and the 2³¹ audit of chapter 24 did not change. The alternative, giving every block a tier flag and a
  second id space, means every kernel and every table entry must say where a block is.
* **Block ids stay physical slots.** Moving a block between tiers while a sequence holds it would need the GPU to find it at another address, or a copy to the GPU in the middle of a
  pass. Only blocks *nobody holds* move; a running sequence has all its blocks hot.
* **Streaming layer by layer through windows, not partial attention.** Dense attention for a context larger than the GPU share could be computed in chunks of keys with a
  flash-decoding merge. That changes the order of the floating point additions: the result is no longer bit-identical to the resident cache, and the merge needs new kernels in every
  attention variant (f32, f16, q8, windows, soft-caps). Windows reuse the flat kernels unchanged, at the price of copying the cache once per pass. A hybrid that keeps the recent
  blocks resident and streams only the old ones would copy less but needs a kernel that reads two buffers; it is the natural follow-up once sparse attention (which reads only the
  selected blocks) lands, because then the streamed set is small.
* **Host memory as plain memory, not a Metal buffer.** On unified memory a no-copy Metal buffer over the host rows would let the GPU read them without staging, but it would also be GPU
  memory by the working-set accounting and be wired while a command buffer uses it, which defeats the budget. Plain memory (pageable, file-backed for the rest) is what lets the budget mean something.
* **Experts before KV streaming, by cost.** A token reads all the KV but a handful of the experts, so per byte freed the experts are far cheaper to leave on the CPU until almost all of them are there.
  The planner derives that from the two cost models rather than hard-coding the order, so a model with few large experts or a fast SSD may choose otherwise.
* **Geometric growth, no shrinking.** A pool whose cap follows the live context would flap with every conversation that ends. Growth by half steps with a retry interval bounds the number of re-placements
  to a handful per session; shrinking would need the pool to give pages back, which a Metal buffer cannot.
* **Native disk format by default.** A lossy file makes a reloaded session differ from the one that was evicted (§26.5); exactness is the default, compactness opt-in.

**Costs.** A streaming slot copies its whole cache through the CPU every pass (§26.5 measures it) and loses replay, pipelining and batching; a pool with a hot cap fails requests whose running sequences
exceed it; the warm tier and prefetch use a background thread and host memory; committed pool pages are never returned.

---

## 26.4 Settings

| key | default | |
|---|---|---|
| `backend.kv-tier` | `false` | tiered KV storage; implies the paged pool for attention models (q4 caches, Mamba, Qwen3-Next and DeepSeek's latent attention keep their flat caches and say so). Off = everything resident as before |
| `backend.kv-hot-mb` | `0` | MB of KV the hot tier may hold (Metal: committed pool blocks; CPU: slabs in the process's memory, the rest mapped files). 0 = no limit of its own (the planner decides when `backend.memory-plan` is on) |
| `backend.kv-host-mb` | `2048` | MB of host memory for blocks pushed out of the pool before they go to the SSD. Set 0 on the CPU |
| `backend.kv-tier-max-gb` | `20` | SSD budget of the block files |
| `backend.kv-tier-dir` | empty (`~/.cache/jllm/kv-tier`) | where block files, memory-mapped slabs and streamed layers live |
| `backend.kv-disk-format` | `native` | `native` (exact), `q8` or `q4` for an f32 / f16 pool; a q8 pool is always stored as it is |
| `backend.kv-prefetch` | `true` | `match` asks the store to bring the blocks a prompt will adopt closer; false = read on adoption |
| `backend.kv-stream` | `auto` | `auto`: a slot whose whole context does not fit `backend.kv-hot-mb` streams (Metal); `off`: the pool fails when full; `force`: always (tests, measurements) |
| `backend.memory-plan` | `false` | the planner of §26.2.6: sizes the hot cap, moves experts as the KV grows, decides streaming at load, reports. Needs `backend.kv-tier` and `backend.metal.cpu-moe=auto` for the experts |
| `-Djllm.metal.stageBytesPerUs` | 30000 | the planner's assumption for the streaming copy speed (bytes per microsecond) until one has been measured |
| `-Djllm.metal.planCheckMs`, `-Djllm.metal.planRetryMs` | 2000, 30000 | how often the planner looks, and how long it waits after a refusal |
| `-Djllm.kv.mapReads=false` | on | plain reads instead of mapped reads in `FileSpill` |

When to change: leave everything off for a model and context that fit; set `backend.kv-tier=true` with `backend.kv-hot-mb` for a server that must keep many conversations or very long
ones within a memory budget; add `backend.memory-plan=true` for a big mixture-of-experts model whose context grows; use `q8` / `q4` disk formats only for sessions you are willing to
see differ in the last digits.

---

## 26.5 Measured

PLACEHOLDER-MEASURED

---

## 26.6 Checks and limits

Verified by (all in `mvn -o install`, `scripts/arch-smoke.sh`, `scripts/mac-validate.sh`):

* `KvTierTest` (no model): the warm tier (budget, background demotion, every payload readable from whichever tier holds it, wrong tokens refused, prefetch, survival of a restart), the
  codec sizes and errors, a pool with a hot cap of 60 blocks and 20 conversations (cap never exceeded, all conversations adopt back with identical rows), prefetch hits, `KvPoolExhausted`
  naming `backend.kv-hot-mb`, `persist` and a restarted pool adopting a saved session from the block files.
* `MemoryPlannerTest` (no model, no GPU): monotone expert movement as the KV grows, streaming only when resident cannot fit, usage statistics keeping hot experts, hysteresis (including 1 000 flapping checks), cap growth.
* `ArchSmoke` → `TierChecks`, every attention family, CPU and the real GPU, f32 / q8 (plain and rotated keys) and f16 on the first dense and MoE model: a pool with a hot cap of 5 blocks and a warm tier of 2
  blocks, prompts pushed through memory to the files and a backend adopting one back with the flat logits (bit for bit); CPU slabs all in memory-mapped files; on Metal a streaming slot against the flat cache for prefill,
  decode, a rewind into a block and a verification chunk (logits bit for bit), the saved-cache bytes equal, a flat cache loaded into a streaming slot and copied out of one, with and without layers in mapped files.
* `TierEngineTest` (Qwen3-0.6B through the engine, CPU and Metal): four 300-token conversations through a pool holding 1.6 of them answer like the flat cache, and the first, sent again, is adopted back
  from the tiers; a 2 MB budget makes the slot stream and its logits (700 tokens of prefill, 20 decode steps) and greedy answer equal the flat cache's.
* `WebApiTest`: `GET /api/memory` and its authentication.

**Not done / limits.**

* A *running* sequence must have all its blocks hot (pool mode) or be streamed whole (Metal): there is no partial residency within a pass. Sparse attention (reading only selected blocks, with
  per-block summaries kept hot) is what makes it possible; this chapter provides the tiers, `prefetch` and the hot cap it needs, but not the selection.
* Streaming is Metal only, one slot at a time (no batched rows), f32 / f16 / q8 caches (not q4), no sharing between streaming slots; each pass copies the whole cache.
* The hot cap only grows, and committed pages are not returned, so a long context leaves the experts on the CPU until the model is reloaded. Slots sharing one expert placement (a server with several slots)
  cannot move experts after load; the planner then refuses growth that does not fit.
* The planner's constants (CPU bandwidth 60 GB/s, round trip 100 us, copy 30 GB/s) are estimates; its decisions are only as good as their ratio.
* Lossy disk formats (`q8`, `q4`) change a reloaded block (§26.5); they apply to f32 / f16 pools only and below the warm tier.
* The block store is shared by the processes of one user in `backend.kv-tier-dir`; two processes with the same model write the same files (atomic renames, same content), but the size budget is per process.
* No cross-process sharing of the warm tier, no compression of the warm tier, no direct I/O.
