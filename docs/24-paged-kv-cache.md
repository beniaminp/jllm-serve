# 24. The paged KV cache and the prefix tree

Until now every slot of a server owned a flat, full-size KV cache: `backend.ctx` positions of keys and values, whether the
conversation used them or not, and two slots that started with the same 10 000-token system prompt each stored it.
This chapter replaces that, behind a switch (`backend.kv-paged`), by **one pool of small blocks per model**, a **block
table** per slot, and a **prefix tree** that finds the blocks a new prompt can reuse. It is the foundation for a scheduler
that treats the cache as shared memory and for contexts too long to keep a full array per conversation.

| | | where |
|---|---|---|
| [24.1](#241-the-idea) | blocks, block tables, the tree, sharing and copy on write | `kv/` |
| [24.2](#242-the-block-table-api-the-contract) | the API a backend and the engine program against | `kv/KvPool`, `KvSequence`, `KvStorage`, `Backend.kvSequence` |
| [24.3](#243-eviction-and-spilling) | LRU eviction, the disk block store | `kv/KvPool`, `kv/FileSpill`, `KvStore` |
| [24.4](#244-attention-through-the-block-table-cpu) | the CPU tile kernels over blocks, the layouts, bit-exactness | `cpu/CpuKvStorage`, `cpu/PagedKv` |
| [24.5](#245-the-metal-backend) | the same on the GPU, decode replay, batched rows | `metal/` |
| [24.6](#246-what-else-touches-the-cache) | speculation, hybrids, saved caches, pictures | |
| [24.7](#247-indexes-for-a-million-tokens) | the 2³¹ audit | |
| [24.8](#248-settings) | settings and defaults | |
| [24.9](#249-measured) | memory, time to first token, speed | |

---

## 24.1 The idea

🧸 Imagine a long picture book that a teacher reads to the class. Before, every child got their **own copy of the whole
book** (even the pages the teacher had not reached yet), and when two children both started with "Once upon a time…
the first ten pages", each of them still kept their own copy of those ten pages. Now the book is torn into **little
booklets of 16 pages**. Every child holds only a **list**: "first booklet is number 7, second is number 12, …". Children who
read the same story start with the same numbers in their lists, so the school keeps just one copy of each booklet. A
child who wants to **scribble in a shared booklet** first photocopies it and scribbles on the copy (so nobody else's story
changes). Booklets nobody holds any more stay on a shelf, in case someone asks for the same story again; when the shelf is
full, the booklet that was asked for longest ago goes to the cellar (a file on disk), and comes back when needed.

📐 **Blocks.** Let `B` = `backend.kv-block` (16) positions per block. A *block* holds the keys and values of `B`
consecutive positions of **every layer**. A *pool* has up to `maxBlocks` blocks, identified by an `int` id that never
changes while the block lives. A *sequence* (one per slot) is a **block table** `table[i]` = id of the block that holds
positions `[i·B, (i+1)·B)`. Position `p` lives at row `p mod B` of block `table[p / B]`; that one indirection is the only
thing attention needs to know.

**Prefix tree.** Keys and values at position `p` depend on all tokens `t₀ … t_p` (the causal mask and the residual
stream), so a block is identified by its *path*: the node reached by the tokens `t[0 .. (i+1)·B)` owns the block
whose rows were computed from exactly those. The tree has one node per full block, child lookup by the block's tokens.
This is a radix tree with page size `B` (what SGLang calls RadixAttention); a plain trie per block is enough because a
block is the unit of storage, and the match is still **to the token**: the last, partial match copies the first `l`
rows of a matching block (see copy on write below), so a prompt reuses exactly as many tokens as the flat cache's linear
compare did.

**References.** `refs[b]` counts the block tables that hold block `b`. A block with `refs = 0` that is in the tree is
*cached*: nobody uses it, but a prompt that matches it takes it back (`refs = 1`). A block that is in no table and not
in the tree is free. A published block is **immutable**.

**Copy on write.** Before a forward pass writes positions `[pos, pos + n)`, the sequence (`KvSequence.beginWrite`)
1. releases the blocks wholly after `pos` (positions past a rewind point are invalid: a retried prompt, a rejected
   speculative draft),
2. if `pos` is inside a block that someone else can see (`refs > 1`, or published), copies that block's first
   `pos mod B` rows into a fresh private block and puts it in the table,
3. allocates fresh blocks up to position `pos + n - 1`.

So kernels only ever write blocks that no one else reads, and branches (a retry that diverges at token 8 000, a
subagent forking from the parent context) share every block up to the divergence and cost one block copy.

**Publishing.** Tokens are final after a prefill and at the end of a request. `publish(tokens, n)` walks the tree:
a full block whose path exists already makes the sequence **switch to the existing block** (its own copy is freed: the
same prompt prefilled twice is stored once), otherwise the block becomes the node of its path.

---

## 24.2 The block-table API (the contract)

Everything is in `org.robomiri.jllm.kv` (no dependency on a backend):

```java
interface KvStorage {                 // the data: backend specific
    int  blockTokens();               // B
    int  spillBytes();                // bytes of one block in the canonical layout
    long residentBytes();             // device memory of one block
    void grow(int blocks);            // make ids [0, blocks) addressable (CPU: allocate slabs)
    void copy(int src, int dst, int rows);
    void save(int block, ByteBuffer dst);  // canonical rows: per layer, per position: K row, V row (= Backend.saveState)
    void load(int block, ByteBuffer src);
}

final class KvPool {                  // ids, references, tree, LRU, spill; thread safe
    KvPool(KvStorage storage, int maxBlocks, String format);
    KvSequence newSequence(int maxPositions);
    void setSpill(KvSpill s);  void setSharing(boolean on);
    Stats stats();  int residentBlocks();  int evictAllCached();
}

final class KvSequence {              // a slot's block table; used under the engine's lock
    int  length();  int blocks();  int block(int logical);  int[] table();
    void setSink(TableSink s);        // mirror every table change into a device buffer
    void beginWrite(int pos, int n);  // truncate to pos, copy on write, allocate to pos + n  (before every forward)
    void truncate(int n);  void clear();
    int  match(int[] tokens, int max);        // longest prefix the tree can supply, to the token
    int  adopt(int[] tokens, int n);          // table := that path (retains; copies a partial last block)
    void publish(int[] tokens, int n);        // make full blocks shareable
    void shareFrom(KvSequence o, int n);      // Backend.copyStateFrom
    KvSequence snapshot(int n);               // frozen view for a background save
    void setShareable(boolean on);            // off for pictures / custom rotary positions
}
```

**What a backend does.**
1. Create one `KvSequence` from the model's pool (`KvPools.get(weights, key, …)` gives every slot of a model the same
   pool) and expose it through `Backend.kvSequence()`.
2. Call `beginWrite(pos, n)` at the start of every forward pass (`forward`, `forwardAll`, embeddings, a batched step
   for each row), **before** any kernel runs. After it `table[0 .. ceil((pos+n)/B))` is valid and the blocks covering
   `[pos, pos+n)` are private. A speculative verification pass is just another `beginWrite`; the rollback is the next
   `beginWrite` at the accepted position.
3. Address row `p` as block `table[p / B]`, row `p mod B`. Use `long` for every byte offset.
4. Implement `copyStateFrom` as `shareFrom` (a memory copy becomes reference counting), `loadState` as
   `beginWrite(0, n)` and then writing rows, `saveState` by reading rows through the table, and `snapshotState(n)` so a
   background save of a paged cache needs no waiting.

**What the engine does** (`Engine.generateLocked`): after computing the slot's own reusable prefix it asks
`kvSequence().match(prompt)`; if the tree holds more, `adopt` replaces the table (no copying, no prefill for the shared
part). After the prefill, and at the end of a request, it `publish`es the final tokens. A request with pictures or custom
rotary positions sets `setShareable(false)`.

**Thread safety.** All pool and table-change operations take the pool's monitor, so slots on different threads share a
pool. Reading rows of blocks a sequence holds needs no lock: they are referenced (never evicted) and either private or
immutable.

---

## 24.3 Eviction and spilling

🧸 The shelf with the spare booklets is small. When it is full, the booklet that nobody has asked for the longest goes to the
cellar. A booklet is only taken from the shelf if no later booklet of the same story is still on it (you never throw out
page 3 while page 4 is still there, nobody could find page 4 any more). If somebody asks for a cellar booklet, it is carried
back up.

📐 `alloc()` takes a free id, else a never-used id below `maxBlocks` (the storage grows), else **evicts**: the cached node
with the smallest `(tick, −depth)` that has **no resident child** (the oldest touch; among equals the deepest), so the tree
stays connected: a prefix is only ever shortened from its end. `tick` is one counter value per `match`/`adopt`/`publish`
call, set on every node of the path it touched, so a whole path ages together and its leaf goes first.

If a `KvSpill` is configured, the evicted block's canonical rows are written to it first, under the **hash of its path**
(a 64-bit chain over the block tokens; the file also stores the tokens and a fingerprint of the model, cache type and
layout, and a lookup compares them, so a hash collision returns nothing instead of wrong rows). `match` continues down the
hashes past the last resident node, so a spilled prefix is found without keeping a node per spilled block, and a store
that survives a restart serves the next run. `adopt` reads the files back into fresh blocks (re-creating their nodes).
`FileSpill` keeps one file per block in `<dir>/<fingerprint>/<hash>.kvb`, trims the least recently used files beyond a
budget, and is what the web console's `KvStore` creates for the model (`jllm.web.kv-spill`). A pool whose every block
is held by running sequences raises `KvPoolExhausted` (an error to the request, not an out-of-memory).

---

## 24.4 Attention through the block table (CPU)

The CPU kernels of `KvCache` (docs/07) work on K "chunks" of 32 keys stored transposed (`[headDim][32]`, so a vector load
holds one head element of 8 or 16 consecutive keys), and on V rows. The paged cache keeps that transposed layout **per
block** with the block's `B` keys as the chunk, off the Java heap in `MemorySegment` slabs allocated as the pool grows:

```
slab    [block][layer][K region | V region]       one block = one contiguous range (copy on write = one memcpy)
f32  K  float [kvHead][headDim][B]                V float [B][kvDim]
f16  K  int   [kvHead][pairs][B] (2 halves/int)   V int   [B/2][kvDim]
q8   K  byte  [kvHead][headDim][B] + float scale [kvHead][B]     V byte [B][kvDim] + float scale [B][kvHead]
```

`PagedKv` implements the same `expandKeys` / `expandValues` hooks as the flat caches: it **gathers** a chunk from one or
two blocks, converting f16 and q8 on the way, into the kernels' float scratch with vector loads straight from the
segment; the existing tiled prefill kernel and the decode kernel then run unchanged. Every score is the same chain of
FMAs over the head elements and every output adds the keys in ascending order, so the result is **bit for bit the flat
cache's** for any block size (`cpu/PagedKvTest`: a few thousand comparisons over cache type, head shape, block size,
window and soft-cap, with rewinds, verification chunks and adoption). The price is the gather: f32 keys and values,
which the flat cache reads in place, are copied once per tile; a first measurement showed it: decode at 2 000 tokens of
context ran at 57 % of the flat cache (46.6 -> 26.8 tok/s), because a decode step is bound by reading the cache and the
gather doubles that traffic. Prefill did not notice (237 tok/s both ways: its tiles reuse each gathered block).

**Decode straight from the blocks (f32).** So a single-token step over f32 blocks skips the gather
(`PagedKv.attentionDecodeDirect`): the scores of a block's `B` keys are `headDim` FMAs over the block's transposed
`[headDim][B]` floats, loaded from the segment with `FloatVector.fromMemorySegment` (`scores4` / `scores2` / `scores1`,
the flat `score4` / `score2` / `score1` with the chunk width `B`); the softmax is the flat kernel's; the outputs add each
block's V rows in ascending key order into the head's accumulator (`accumulate`, the flat `accumulate1` over a segment).
Each score and each output element is the same chain of FMAs in the same order, whatever the width of the key group,
so this too is bit for bit the flat kernel (`PagedKvTest`'s n = 1 steps run it). It needs `B` to be a multiple of two
vector lengths (16 on NEON and AVX2; on AVX-512 with `B = 16` the gather stays), and `-Djllm.cpu.pagedGather=true` brings the
gather back for comparisons. The f16 and q8 caches expand into scratch in the flat kernels as well, so for them paged
and flat cost the same.

---

## 24.5 The Metal backend

🧸 On the graphics card the "booklets" are rows in one big shelf, and every worker that reads a page (an attention
kernel) first looks in the child's list: "page 37 is in booklet 12, line 5". A worker that writes a new page (the store
kernel) looks in the same list. Nothing else about the workers changes, so the fast ones that were checked against
the slow ones still are.

📐 **Layout.** Per layer one K buffer and one V buffer of `maxBlocks · B` rows, in the flat cache's row format (f32, half, or
q8: int8 values then one f32 scale per kv head; `KvDtype.rowBytes`), shared by every slot of the model
(`MetalKvStorage`, created uninitialised: pages are committed when a block is first written, which is the Metal
analogue of "allocated as the conversation grows"; the address range is reserved up front, as the flat cache's
per-slot arrays were, but once per model instead of once per slot). Block `b`, position `t` is row `b · B + t`. Each
slot has a small int buffer, its block table, kept equal to `KvSequence.table()` by the sequence's sink on every change.
I chose one big buffer per layer over per-block buffers: a kernel then needs one buffer binding and one table lookup per
key row, the `kv_row` function, instead of an argument-buffer indirection per block, and a block is contiguous for the
host-side copy on write and the spill.

**Reading.** Every attention kernel that walked `kb + s · stride` now walks `kb + kv_row(tbl, pageShift, s) · stride`,
where `kv_row` is `s` when `pageShift = 0` (the flat cache: the table is a dummy buffer and not read) and
`(tbl[s >> pageShift] << pageShift) | (s & (2^pageShift − 1))` otherwise, in `ulong`. The kernels are `attention_t`
(all head widths and cache types, prefill tiles and single tokens), `attention_split` (flash decoding), `attention_dec` (the
decode kernel), `attention_fa` and `attention_fb` (flash prefill). The score, softmax and accumulation arithmetic is untouched,
so the paged kernels compute the flat kernels' results bit for bit. The parameter structs got a trailing `pageShift`, the
table is the buffer before the struct (buffer indexes of the existing buffers unchanged).

**Writing.** The flat cache's projections write k and v straight into the cache rows `pos · stride` onward, which assumes the
rows of a chunk are contiguous. In a block table they are not (a 256-token chunk spans 17 blocks at random places), so a paged
step stages every cache type like the q8 cache does: the projection writes f32 rows, `rope` rotates k in place, and a store kernel
writes the chunk into the table's rows: `kv_store_q8` (now also paged: parameters `pos` and `pageShift`) or the new `kv_store_pg`
(f32 copy, or rounding to half, the same round-to-nearest-even as the `*_h` projection and `rope_h`). One extra dispatch per
layer. The fused `rope_store_q8` is left out of paged steps.

**Decode replay.** A recorded decode step patches the position into its parameters and the position-indexed byte offsets of the
cache buffers. A paged step has no position-dependent offsets at all: the pool buffers are bound at offset 0, the store takes
`pos` as a parameter (patched like the attention kernels' `pos0`), and the block table is read when the step runs, not when it is
recorded. So replay stays on for paged steps; the one requirement is that the host updates the table before a step is submitted,
which `beginWrite` does (in `run`, and for each step of a pipelined greedy run in `Pipe.submit`, before the step is encoded). A
table entry written for a later step is never read by an earlier one.

**Batched rows.** `runRows` issues per-row dispatches into each owner's cache; for a paged owner it sets the owner's table
(`MetalKernels.page`) before the row's store and attention dispatches, inside the same `beginIndependent` groups, so the
concurrent encoder's hazard detection sees the table as one more input of each dispatch.

**Not paged on Metal:** q4 (its recent-window ring is per slot and not a function of the tokens alone; `backend.kv-paged`
prints a note and the slot keeps its flat q4 cache), DeepSeek's latent cache, Mamba and Qwen3-Next. The flat attention kernels
keep a scores scratch of `tile · heads · ctx` floats, which bounds the context of a Metal run well below 1M; at long
contexts the flash and decode kernels, which have no scores scratch, are the ones to use.

---

## 24.6 What else touches the cache

- **Speculation (n-gram, draft model, MTP).** A verification pass writes `k + 1` positions and the next pass starts
  at the accepted one: `beginWrite` at the lower position releases the blocks past it and copies the boundary block if
  it is shared. Blocks are published only for final tokens, so a rejected draft is never visible to another slot.
- **Hybrids and recurrent states** (Mamba, Qwen3-Next / Qwen3.5), **DeepSeek's latent cache**: not paged. A recurrent
  state is a function of every earlier token and cannot be shared per block or rewound, and the latent cache has a
  different row shape. With `backend.kv-paged=true` these models print a note and keep their flat caches; prefix reuse for
  them is the old one (the slot's own cache, or a whole-state copy).
- **Saved caches** (`stateFormat` / `saveState` / `loadState` / `copyStateFrom`). The format string and the bytes are the
  flat cache's, so files are interchangeable between flat and paged and between CPU and Metal. `copyStateFrom` shares
  blocks, `loadState` writes into freshly allocated blocks, and a background save takes a `snapshotState` (the blocks are
  retained; whoever writes one afterwards copies it first), so the engine never waits for a save.
- **Pictures, custom rotary positions** (Qwen-VL M-RoPE): rows are not a function of the token ids, so the sequence is
  marked not shareable for that request.
- **Batched rows** (`forwardBatch`): each row's backend does `beginWrite(pos, 1)` first.

---

## 24.7 Indexes for a million tokens

At 1M positions, `position × kvDim × layers` exceeds 2³¹ for any real model (Qwen3-0.6B: 1M × 1024 × 28), so every
offset that multiplies a position by a width is a `long`: block ids are `int` (`maxBlocks < 2³¹`, 33 G positions at
`B = 16`), byte offsets inside the pool are `long`. The flat CPU cache refuses contexts whose per-layer array would need
more than 2³¹ elements; the paged one has no such limit. `cpu/LongContextTest` stores 1M positions of a small model and
attends over all of them, with byte offsets past 2³² (§24.9).

---

## 24.8 Settings

| key | default | |
|---|---|---|
| `backend.kv-paged` | `false` | the paged cache; the flat cache stays the default until measured (§24.9) |
| `backend.kv-block` | `16` | positions per block (power of two ≥ 16) |
| `backend.kv-pool-tokens` | `0` | capacity in positions over all slots (0 = every slot's `backend.ctx` plus a spare block) |
| `backend.kv-prefix-share` | `true` | adopt and publish blocks; `false` pages without sharing |
| `jllm.web.kv-spill` | `false` | console: evicted blocks go to `<kv-cache-dir>/blocks` |

## 24.9 Measured

*(filled in with the measurements)*
