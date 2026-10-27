# 24. The paged KV cache and the prefix tree

Until now every slot of a server owned a flat, full-size KV cache: `backend.ctx` positions of keys and values whether the
conversation used them or not, and two slots that started with the same 10 000-token system prompt each stored it (or
copied it from each other once, 34 ms). This chapter replaces that, behind a switch (`backend.kv-paged`), by **one pool of
small blocks per model**, a **block table** per slot, and a **prefix tree** that finds the blocks a new prompt can reuse.
It is the foundation for a scheduler that treats the cache as shared memory (several conversations, subagents forking from
one context, a conversation paused and resumed) and for contexts too long to keep a full array per conversation.

| | | where |
|---|---|---|
| [24.1](#241-in-plain-words) | the idea, told as a story | |
| [24.2](#242-how-it-works) | blocks and block tables, the prefix tree, references and copy on write, eviction and the disk store, the API contract, the CPU and Metal kernels, speculation / replay / batched rows / saved caches, the 2³¹ index audit | `kv/`, `cpu/PagedKv`, `cpu/CpuKvStorage`, `metal/MetalKvStorage`, `kernels.metal` |
| [24.3](#243-why-this-design) | the alternatives that were rejected, and what the design costs | |
| [24.4](#244-settings) | `backend.kv-paged`, `backend.kv-block`, `backend.kv-pool-tokens`, `backend.kv-prefix-share`, `jllm.web.kv-spill` | |
| [24.5](#245-measured) | memory for N conversations sharing a 4 096-token prompt, time to first token of a resent conversation, prefill and decode speed, CPU and Metal | `PagedKvBench` |
| [24.6](#246-checks-and-limits) | the tests, and what is not done | |

---

## 24.1 In plain words

🧸 Imagine a teacher who reads a long picture book to the class, and every child keeps a copy of the book so far. Until now each
child got a **whole empty book** to fill in (even the pages the story never reached), and when three children started with the
same first ten pages ("Once upon a time…"), each of them wrote those ten pages out again.

Now the book is torn into **little booklets of 16 pages**, and the school keeps all booklets in one cupboard. A child does not
hold a book any more, only a **list**: "my first booklet is number 7, my second is number 12, my third is number 3…". Children who
start with the same story start with the same numbers, so the school keeps **one** copy of each shared booklet, and a new child
whose story begins like an old one just copies the old child's list (a few numbers, not ten pages).

Rules of the cupboard:

* A child who wants to **scribble in a booklet someone else also uses** first photocopies it and scribbles on the copy, so
  nobody else's story changes. (That is *copy on write*, and it is why a child who retells the story from the middle of a
  booklet pays for one photocopy and not for the whole book.)
* A booklet nobody holds any more is not thrown away: it stays on a shelf with a little label saying where its story began, in
  case somebody asks for the same story again tomorrow.
* The shelf is small. When the cupboard is full, the booklet whose story was **asked for longest ago** goes down to the cellar
  (a file on disk) — but never a booklet from the *middle* of a story while a later booklet of it is still upstairs, because
  nobody could find the later one without it. When someone asks for a cellar story, the booklets are carried back up.
* The booklets are as big as one fits in a hand (16 pages) because that is the most that is ever wasted: a child's last booklet is
  half empty, and that is all the waste there is. The old books were as big as the longest story the school allows.

---

## 24.2 How it works

📐 Everything below is for a model with ordinary grouped-query attention (Qwen3, Llama, Mistral, Qwen2, GLM, Phi-3, Gemma 2 / 3,
Qwen3-MoE) and the f32, f16 and q8 caches. Qwen3-0.6B (28 layers, 8 kv heads of 128, so a position row is `kvDim = 1024`
values) is the running example: one position costs `28 · 2 · 1024 · 4 = 229 376` bytes in f32, `114 688` in f16, `59 136` in q8.

### 24.2.1 Blocks and the block table

Let `B` = `backend.kv-block` (a power of two, at least 16; default 16).

* A **block** holds the keys and values of `B` consecutive positions of **every layer**. A **pool** owns up to `maxBlocks` blocks;
  a block is identified by an `int` id that does not change while the block lives.
* A **sequence** (one per slot, `KvSequence`) is a **block table**: `table[i]` is the id of the block that holds positions
  `[i·B, (i+1)·B)`. Position `p` lives in **logical block** `p ÷ B` at **row** `p mod B`, which is row `p mod B` of **physical block**
  `table[p ÷ B]`. That one indirection is everything attention has to know.

Worked example with `B = 16`, a table `[7, 12, 3]` and `p = 37`:

```
p = 37 → logical block 37 ÷ 16 = 2, row 37 mod 16 = 5 → physical block table[2] = 3, row 5
          Metal pool row  = (3 << 4) | 5 = 53            (kv_row in kernels.metal)
          CPU byte offset = slab(3), base(3, layer) + …  (CpuKvStorage.slab / base, all long)

positions   0 ........ 15 | 16 ....... 31 | 32 ....... 47
table       [   7         |      12        |      3        ]      (three blocks hold 48 positions; 37 is in the last)
```

What a block weighs in Qwen3-0.6B: `16 · 229 376 = 3 670 016` bytes (3.5 MiB) in f32, 1.75 MiB in f16, 0.9 MiB in q8. A 4 160-token
conversation is `260` blocks in f32 (4 160 = 260 · 16, all full): about the 958 MB the flat cache stores for those positions (with a
few generated tokens) — but not the 1 879 MB a flat slot of `backend.ctx = 8192` reserves.

Two sequences that share a prefix point at the same physical blocks:

```
   sequence A (system prompt + question 1)          sequence B (system prompt + question 2)
   table: [ a0  a1  a2  …  a255 | a256 ]            table: [ a0  a1  a2  …  a255 | b256 ]
              \   |   |       |     |  (own tail)              /   /   /       /      (own tail)
               +--+---+-------+-----+----- shared, refs = 2 ---+---+---+-------+
```

### 24.2.2 The prefix tree

Keys and values at position `p` depend on every token `t₀ … t_p` (the causal mask and the residual stream), so a block is not
identified by its own 16 tokens but by its **path**: the tokens of every block from the start. The **prefix tree** (`KvPool`,
a radix tree with page size `B`, the idea behind SGLang's RadixAttention) has one node per **full** block; the node reached by the
tokens `t[0 .. (i+1)·B)` owns the physical block whose rows were computed from exactly those tokens. Children are looked up by
the first token of their block and compared on all `B`.

```
                          root
                           |
           [t0..t15]  ->  n0 (block 7)            <- the system prompt, 256 nodes deep
                           |
           [t16..t31] ->  n1 (block 12)
                           |
                          ...
                           |
                  n255 (block 91) ---------------+------------------+
                    |                            |                  |
         [question 1, 16 tokens]        [question 2, 16 tokens]   [retry of 1, differs at its 9th token]
              n256 (block 40)                n256' (block 55)      n256'' (block 62)
```

**To the token, not to the block.** `match(tokens, max)` walks the tree full block by full block; at the end it also looks
for a child that shares only the *start* of the next block and counts those `l < B` tokens too. `adopt` makes the table the
tree's path, and for that last part it allocates a fresh block and **copies** the first `l` rows of the matching one (copy on
write at adoption time). So a prompt reuses exactly the tokens the flat cache's linear compare reused, not just whole blocks.

Worked example: block 2 of a published conversation holds positions 32…47. A retry that agrees up to and including position
36 and then diverges has `match = 32 + 5 = 37` (tokens 32…36 agree); `adopt` shares blocks 0 and 1 (refs +1) and copies 5 rows of
block 2 into a private block; the prefill continues at position 37.

**Path hashes.** A node's `pathHash` is a 64-bit chain: `h₀ = ROOT`, and for a block with tokens `x₀ … x_{B-1}`,
`h ← (h ⊕ (xⱼ + φ)) · c; h ← h ⊕ (h ≫ 32)` for each token (φ = `0x9E3779B97F4A7C15`, c = `0xFF51AFD7ED558CCD`), then a final multiply-xor.
The hash is how a block is found on disk (24.2.4); a file also carries its tokens, which are compared on load, so a collision
returns nothing rather than wrong rows.

### 24.2.3 References, publishing and copy on write

Each physical block has a **reference count** `refs` (the number of block tables that hold it) and may be **published** (be the
owner of a tree node). Its state follows from the two:

```
                 alloc                       publish
   free  ───────────────►  used (refs>0)  ───────────────►  used + published
     ▲                       │   ▲                              │   ▲
     │ release (refs=0,      │   │ retain (a match takes it     │   │ retain
     │ not published)        │   │ back: refs 0→1)              ▼   │
     └───────────────────────┘   └──────── cached (refs=0, published) ◄── release (refs→0)
                                                   │
                                      evict (LRU, leaf first)
                                                   ▼
                                       free (+ a file if there is a store)
```

* A **published block is immutable.** Nobody may write rows into it while it is in the tree.
* **Copy on write** happens in `KvSequence.beginWrite(pos, n)`, which a backend calls before **every** forward pass that writes
  positions `[pos, pos+n)`: (1) release the blocks wholly after `pos` (positions past the new write are invalid anyway: a retried
  prompt, a rejected speculative draft); (2) if `pos` falls inside a block that someone else can see (`refs > 1`, or published) copy its
  first `pos mod B` rows into a fresh private block and put that in the table; (3) allocate blocks up to position `pos + n − 1`.
  After it, the blocks that cover `[pos, pos+n)` are private, so kernels write straight into them.
* **Publishing** (`publish(tokens, n)`, after a prefill and at the end of a request) walks the tree with the sequence's tokens. For a
  full block whose path exists already the sequence **switches to the existing block** (its own copy is freed: the same prefix
  prefilled twice is stored once); otherwise its block becomes the node.

Worked example, `B = 16`, a pool starting empty (`refs` in brackets):

```
 1. A prefills P = 40 tokens (3 blocks, the last holds 8 rows)       a0[1] a1[1] a2[1]
    A.publish(P, 40): n0←a0, n1←a1 (a2 is partial: not published)
 2. B: prompt = P[0..37) + new tokens.   match = 32 (a2 is not in the tree)
    B.adopt → table [a0, a1]                                        a0[2] a1[2]
    B prefills positions 32.. into a fresh b2                       b2[1]
 3. C: prompt = P[0..21) + others.       match = 16 + 5 = 21
    C.adopt → table [a0, c1]  with c1 = the first 5 rows of a1      a0[3] c1[1]      (one block copy)
 4. A rewinds to position 25 (a retry) and continues: it releases a2 (wholly past 25; unpublished → free), and
    block 1 is a1 (refs 2, published) → copy on write: A's table [a0, a1'], a1' = first 9 rows of a1
                                                                    a1[1] a1'[1]     (B still reads a1)
 5. A closes: a0[2], a1' is released → free (unpublished).   The tree still has n0 (a0) and n1 (a1);
    when B and C close too, a0 and a1 are cached (refs 0) and stay findable until evicted.
```

Nobody ever saw anybody else's rows change, and the system prompt exists once.

### 24.2.4 Eviction and the disk block store

`alloc()` hands out, in this order: a free id; a never-used id below `maxBlocks` (the storage grows); else it **evicts**: the
**cached** node (refs 0, published) with the oldest `tick` among those with **no resident child** (ties: the deepest). `tick` is a
counter taken once per `match` / `adopt` / `publish` call and set on every node of the path that call touched, so a conversation's
whole path ages together and its **leaf goes first**: a prefix is only ever shortened from its end, never from the middle (nobody
could find the later blocks otherwise). If every block is held by running sequences, `alloc` throws `KvPoolExhausted` (an error to
that request; the default capacity `slots · (ctx/B + 2)` blocks makes it impossible without `backend.kv-pool-tokens`).

With a `KvSpill` store (`FileSpill`) the evicted block's rows are written first, in the **canonical layout** (per layer, per
position: the K row then the V row, exactly what `Backend.saveState` writes), to
`<dir>/<fingerprint hash>/<pathHash>.kvb`:

```
 int  magic 'KVB1' | long fingerprint | long pathHash | int B | int tokens[B] | int payloadBytes | payload (spillBytes)
 Qwen3-0.6B f16: payload = 28 layers · 16 positions · 2 · 2 048 bytes = 1 835 008 bytes
```

`match` continues down the hash chain past the last resident node (`has(pathHash)`), counting spilled blocks, without keeping a node
per spilled block; `adopt` reads them back into fresh blocks and re-creates their nodes. The fingerprint (model, quantization, cache
type, layers, block size) keeps a file from ever being read into the wrong model; files survive a restart and are found again by
listing the directory; the budget (`jllm.web.kv-spill-max-gb`) deletes the least recently used files.

Worked example: a pool of 6 blocks, conversation A published as 3 blocks `A0←A1←A2` (one `publish` call: all three have tick 1),
then closed. Conversation D allocates 3 blocks (free ids), closes (tick 2 for `D0←D1←D2`); conversation E needs 3:
candidates are the leaves `A2` (tick 1, depth 3) and `D2` (tick 2) → `A2` goes (written to disk), then `A1` becomes a leaf with
tick 1 → `A1`, then `A0`. A is gone from memory, whole, and on disk; a request for A's prompt finds 48 tokens by hashes and
reloads 3 blocks (evicting D's leaves if the pool is full).

### 24.2.5 The API contract

Everything is in `org.robomiri.jllm.kv` (no dependency on a backend); backends implement `KvStorage`:

```java
interface KvStorage {                 // the data: backend specific
    int  blockTokens();               // B
    int  spillBytes();                // bytes of one block in the canonical layout
    long residentBytes();             // device memory of one block
    void grow(int blocks);            // make ids [0, blocks) addressable (CPU: allocate slabs; Metal: reserved up front)
    void copy(int src, int dst, int rows);
    void save(int block, ByteBuffer dst);  // canonical rows
    void load(int block, ByteBuffer src);
}
final class KvPool {                  // ids, references, tree, LRU, spill; thread safe (one monitor)
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
    int  match(int[] tokens, int max);        // longest prefix the tree (and the store) can supply, to the token
    int  adopt(int[] tokens, int n);          // table := that path (retains; copies a partial last block)
    void publish(int[] tokens, int n);        // make full blocks shareable
    void shareFrom(KvSequence o, int n);      // Backend.copyStateFrom: retain o's blocks
    KvSequence snapshot(int n);               // frozen view for a background save (Backend.snapshotState)
    void setShareable(boolean on);            // off for pictures / custom rotary positions
}
```

**What a backend does.** (1) One `KvSequence` per backend from the model's pool (`KvPools.get(weights, key, …)` gives every slot of a
model the same pool), exposed as `Backend.kvSequence()`. (2) `beginWrite(pos, n)` at the start of every forward pass (`forward`,
`forwardAll`, `forwardEmbeds`, one per row of a batched step) **before** any kernel runs. (3) Address row `p` as block
`table[p ÷ B]`, row `p mod B`, with `long` offsets. (4) `copyStateFrom` = `shareFrom`; `loadState` = `beginWrite(0, n)` then write
rows; `saveState` reads rows through the table; `snapshotState(n)` for background saves.

**What the engine does** (`Engine.generateLocked`): after the slot's own reusable prefix is computed it asks
`kvSequence().match(prompt)`; if the tree holds more, `adopt` replaces the table (no copying, no prefill for the shared part).
After the prefill, and at the end of a request (and after `loadCache`), it `publish`es the final tokens. A request with pictures
or custom rotary positions calls `setShareable(false)`. `Engine.cachedPrefix` also counts the tree, and a background save takes
a snapshot instead of pinning rows (`PendingSave`), so the next request never waits for it.

**Threads.** Pool and table-changing operations take the pool's monitor, so slots on different threads share a pool. Reading rows
of blocks a sequence holds needs no lock: they are referenced (never evicted) and either private or immutable.

### 24.2.6 The CPU backend

**Layout** (`cpu/CpuKvStorage`). Off the Java heap, in `MemorySegment` slabs allocated as the pool grows (about 16 MB each).
One block is one contiguous range of `blockBytes`, so copy on write is one `memcpy`:

```
slab:   [ block ][ block ] …            block  = [ layer 0 ][ layer 1 ] …       (one block = all layers)
layer:  [ K region | V region ]
 f32  K: float [kvHead][headDim][B]     V: float [B][kvDim]
 f16  K: int   [kvHead][pairs][B]       V: int   [B/2][kvDim]        two halves per int, as the flat F16 cache
 q8   K: byte  [kvHead][headDim][B]  +  float scale [kvHead][B]      V: byte [B][kvDim] + float scale [B][kvHead]
```

With `backend.kv-q8-rotate` (the shipped default for q8, docs 07 §7.13) the keys are Walsh–Hadamard rotated before they are
quantized and the queries are rotated at attention time with the score scale divided by the Hadamard block: `CpuKvStorage.storeRow`
rotates a key head before `q8` and `PagedKv.attention` rotates the queries, exactly as `KvCache.Q8` does, and the saved-cache format
string says `q8 rot` in both. The pool is created for one setting (it is part of its key and of its spill fingerprint).

The keys keep the flat cache's **transposed** layout (a vector load holds one head element of consecutive keys), with the block's
`B` keys as the "chunk" instead of the flat cache's fixed 32. Every byte offset is a `long`.

**Attention** (`cpu/PagedKv`, a `KvCache` subclass). The flat tile kernels ask for chunks of 32 keys and V rows through
`expandKeys` / `expandValues`; `PagedKv` **gathers** a chunk from one or two blocks into the kernels' float scratch with vector loads
straight from the segment, converting f16 and q8 on the way exactly as the flat cache does. The scratch is identical to what the
flat cache produces, and so are scores and outputs: **bit for bit the flat cache for any block size** (each score is the same chain
of FMAs over the head elements, each output adds the keys in ascending order). The gather costs memory traffic: with f32 the flat
cache reads K and V in place, and a first measurement showed the price: decode at 2 000 tokens of context ran at 57 % of the flat
speed (46.6 → 26.8 tok/s), prefill did not notice (237 tok/s both ways: its tiles reuse each gathered block).

**Decode straight from the blocks (f32, f16, q8).** So a single-token step skips the gather (`PagedKv.attentionDecodeDirect`): the
scores of a block's `B` keys are `headDim` FMAs over the block's keys, loaded from the segment with vector loads, the softmax is the
flat kernel's, and the outputs add each block's V rows in ascending key order into the head's accumulator. For f32 these are `scores4` /
`scores2` / `scores1` (the flat `score4/2/1` with the chunk width `B`) and `accumulate` (the flat `accumulate1` over a segment); for
f16 and q8 the same loops convert as they load (`cpu/PagedKernels`):

| | keys (`[kvHead][..][B]`) | values |
|---|---|---|
| f16 | ints `[pairs][B]`: element `2p` of key `t` in the low half of int `(g·pairs + p)·B + t`, `2p + 1` in the high half; `IntVector` load, `F16.lowHalf` / `highHalf`, FMA with the broadcast query element | ints `[B/2][kvDim]`: key `2j` low, `2j + 1` high; per key one `IntVector` load of the row `r >> 1`, the half picked by `r & 1` (outside the vector math), FMA with the broadcast probability |
| q8 | bytes `[hd][B]` and a float scale per key: `ByteVector` load widened to float, times the scale vector, FMA | bytes `[B][kvDim]` and a scale per key and head: widened, times the key's scale, FMA |

The converted operands, the order of the FMAs over the head elements (ascending) and the order of the keys are those of the flat
cache's expand-then-`score4`, so the results are **bit for bit the flat cache's** (`PagedKvTest` compares them for n = 1 steps over
f16, q8 and rotated q8). The half selection is written out straight-line (no `h == 0 ? lowHalf : highHalf` inside the loops): the
first version had the ternaries inside the loop and the JIT boxed the vectors, which made that build's decode about 4× slower than the
straight-line path; that is a development measurement of the discarded version, not what §24.5 measures, where f16 paged decode is
level with flat (9.7 → 9.9 tok/s). The direct path needs `B` to be a multiple of two vector lengths (16 on NEON and AVX2; on AVX-512
with `B = 16` the gather stays), and `-Djllm.cpu.pagedGather=true` brings the gather back for comparisons (prefill and multi-token
chunks always gather: their tiles reuse each gathered block).

### 24.2.7 The Metal backend

**Layout** (`metal/MetalKvStorage`). Per layer one K buffer and one V buffer of `maxBlocks · B` rows, in the flat cache's row format
(f32, half, or q8: int8 values then one f32 scale per kv head; `KvDtype.rowBytes`), shared by every slot of the model, created
uninitialised so pages are committed when a block is first written. Block `b`, position `t` is row `b·B + t`. Each slot has a small
`int` buffer, its block table, mirrored from `KvSequence.table()` by the sequence's sink on every change. They belong to a Metal
context of their own, so a slot closing does not free what the others use.

```
 K buffer, layer l:   | block 0 (16 rows) | block 1 | block 2 | block 3 | …        row = b·16 + t, rowBytes = KvDtype.rowBytes(kvDim, kvHeads)
 slot A table buffer: [ 3, 0, 5 ]    position 37 → (tbl[2] << 4) | 5 = 5·16+5 = 85
 slot B table buffer: [ 3, 0, 2 ]    positions 0..31 are the same rows as A's: blocks 3 and 0 are shared
```

**Reading.** Every attention kernel that walked `kb + s · stride` walks `kb + kv_row(tbl, pageShift, s) · stride`, where
`kv_row(tbl, shift, s) = s` when `shift = 0` (the flat cache; `tbl` is a dummy buffer and is not read) and
`(ulong(tbl[s >> shift]) << shift) | ulong(s & (2^shift − 1))` otherwise. The kernels are `attention_t` (every head width and cache type,
prefill tiles and single tokens), `attention_split` (flash decoding), `attention_dec` (the decode kernel), `attention_fa` and
`attention_fb` (flash prefill). The arithmetic is untouched, so the paged kernels give the flat kernels' results bit for bit. The
parameter structs got a trailing `pageShift`; the table is the buffer **before** the parameter struct (the parameters bind right
after the buffers, so the existing buffer indexes did not move).

**Writing.** The flat cache's projections write k and v straight into the cache rows `pos · stride` onward, which assumes the rows
of a chunk are contiguous. In a block table they are not (a 256-token chunk spans 17 blocks at arbitrary places), so a paged step
stages every cache type the way the q8 cache already did: the projection writes f32 rows, `rope` rotates k in place, and a store
kernel writes the chunk into the table's rows — `kv_store_q8` (now paged too: parameters `pos`, `pageShift`) or the new `kv_store_pg`
(f32 copy, or rounding to half with the same round-to-nearest-even as the `*_h` projection and `rope_h`); with rotated q8 keys
`kv_store_q8r` (rotates k and the chunk's queries in place, then quantizes; paged the same way: parameters `pos`, `pageShift`, the
table). That is one extra dispatch per layer; the fused `rope_store_q8` / `rope_store_q8r` are left out of paged steps.

**Decode replay: patchable, so it stays on.** A recorded decode step re-uses its dispatches and patches only the position into the
parameter structs and the position-indexed byte offsets of the cache buffers. A paged step has **no position-dependent buffer offset**:
the pool buffers are bound at offset 0, the store takes `pos` as a *parameter* (patched like the attention kernels' `pos0`), and the
block table is read by the GPU when the step *runs*, not when it was recorded. The one requirement is that the host updates the table
before a step is submitted, which `beginWrite` does (in `run`, and for each step of a pipelined greedy run in `Pipe.submit`, before the
step is encoded); an entry written for a later step is never read by an earlier one. Replay and pipelined decoding therefore work
unchanged (tested: `PagedKvChecks` runs decode steps through both and compares with the flat cache).

**Batched rows** (`MetalBackend.runRows`, three conversations in one pass): the per-row dispatches go into each owner's cache; for a
paged owner the row sets the owner's table (`MetalKernels.page`) before its store and attention dispatches, inside the same
`beginIndependent` groups, so the concurrent encoder's hazard detection sees the table as one more input of each dispatch.

**Not paged on Metal:** q4 (see 24.6), DeepSeek's latent cache, Mamba and Qwen3-Next. The flat attention kernels keep a scores
scratch of `tile · heads · ctx` floats, which bounds a Metal run's context well below a million; at long contexts the flash and decode
kernels, which have no scores scratch, are the ones to use.

### 24.2.8 What else touches the cache

* **Speculation (n-gram, draft model, MTP).** A verification pass writes `k + 1` positions and the next pass starts at the accepted
  one: `beginWrite` at the lower position releases the blocks past it and copies the boundary block if it is shared. Blocks are
  published only for final tokens, so a rejected draft is never visible to another slot.
* **Hybrids and recurrent states** (Mamba, Qwen3-Next / Qwen3.5), **DeepSeek's latent cache**: not paged. A recurrent state is a
  function of every earlier token and cannot be shared per block or rewound, and the latent cache has another row shape. With
  `backend.kv-paged=true` these models print a note and keep their flat caches (draft models always stay flat).
* **Saved caches** (`stateFormat` / `saveState` / `loadState` / `copyStateFrom`). The format string and the bytes are the flat
  cache's, so files are interchangeable between flat and paged and between CPU and Metal. `copyStateFrom` shares blocks, `loadState`
  writes into freshly allocated blocks, and a background save takes a `snapshotState` (the blocks are retained; whoever writes one
  afterwards copies it first).
* **Idle slots.** A slot's block table holds references to its conversation's blocks, so a slot nobody is talking to would pin the
  memory until it moved to another conversation. `Engine.releaseKv(minIdleMs)` publishes the slot's full blocks into the prefix tree and
  clears its table (the conversation stays findable: the next prompt adopts the blocks again, or the pool evicts them when it needs
  the memory); it only try-locks, so a busy engine is never touched, and it refuses before `minIdleMs` of idleness (`lastActive` is
  stamped by every unlock of the engine lock). `KvPool.setReclaimer` is the additive hook for pressure: when `alloc` finds no free
  block and nothing evictable it asks the reclaimer, and only then throws `KvPoolExhausted`. In the web console `LoadedModel.
  enablePagedHousekeeping` registers the reclaimer (release every slot no request holds, at once) and starts a virtual-thread sweeper
  that releases slots idle for `jllm.web.kv-idle-release-seconds` (default 120, `0` = pressure only); unloading stops it.
* **Pictures, custom rotary positions** (Qwen-VL M-RoPE): rows are not a function of the token ids, so the sequence is not shareable
  for that request.
* **The web console.** `LoadedModel` picks any idle slot (the engine pulls shared blocks from the pool whichever slot runs the prompt,
  so no prefix is copied between slots); `KvStore.attachSpill` gives the pool a `FileSpill` under `<kv-cache-dir>/blocks` (the
  conversation files of §17.3 are unchanged and still written for expensive prefills).

### 24.2.9 Indexes for a million tokens (the 2³¹ audit)

At 1M positions, `position × row width × layers` exceeds 2³¹ for any real model (Qwen3-0.6B f32: 1M · 1024 · 28 = 3·10¹⁰).

| place | type | at 1M positions |
|---|---|---|
| block id | `int` (`maxBlocks < 2³¹`) | 65 536 ids; 34 G positions at `B = 16` |
| byte offset inside a CPU slab / layer / block (`CpuKvStorage.base`, `keyIndex`, `valueIndex`) | `long` | 4.3 GB of f32 in the long-context test |
| Metal pool row `kv_row(...)`, `row · stride` in kernels | `ulong` | rows up to 2⁶⁴ |
| Metal buffer sizes, binding offsets (`MetalKvStorage`, `MetalBackend`) | `long` | |
| flat cache per-layer arrays (`KvCache`) | `int` index; **refused** when `ctx · kvDim ≥ 2³¹ − 16` | the paged cache has no such limit |
| positions, `ctx`, kernel strides (`AttnParams`, …) | `int` | < 2³¹ |
| per-call scratch (`tile · group · span` scores, `heads · span` in decode) | `int`, bounded by `SCORE_BUDGET` / `heads · ctx` | 16 M floats for 16 heads |

`cpu/LongContextTest` (in `mvn install`) stores 1 048 576 positions of a small model (4 layers, 4 kv heads of 32, f32: 65 536
blocks, 4.29 GB, **byte offsets past 2³²**), then runs attention of the last token and of a 64-token chunk over all of it and
compares with a float64 reference computed from the stored values.

---

## 24.3 Why this design

* **Fixed-size blocks, not growable arrays.** A per-slot array that grows by doubling copies everything and still reserves up to
  2× the conversation; a block table grows one 16-position block at a time and never moves a row, so a kernel (or a recorded replay) may
  hold block ids for as long as a sequence holds the block.
* **A block spans all layers.** One id, one table, one reference count per 16 positions instead of one per layer, and a copy on
  write or a spill is one contiguous range / one file. (vLLM keeps a table per layer group; for these models the layers always move
  together.)
* **A radix tree over blocks, hashes for the disk.** The alternative is vLLM's chained block hash (`hash(parent, tokens)` in a flat map):
  the same sharing, but no walk of children for the *partial* last block, which is how the engine keeps the old token-granular reuse
  (a retry that diverges at token 37 still reuses 37 tokens). The disk side does use the chain hash, because it lets a spilled prefix be
  found without keeping a node per spilled block and survives restarts.
* **Leaf-first LRU, with one tick per path.** Evicting the oldest block alone would cut a prefix in the middle and orphan its descendants;
  per-path ticks plus "no resident child" make every eviction shorten a prefix from its end.
* **Copy on write at the block, not copy-on-read of the prefix.** Slots used to copy a shared prefix (34 ms for 10 000 tokens, once per
  slot, and the copies then diverged in memory); sharing by reference costs a table write and the data exists once.
* **Gather on the CPU, not new tile kernels.** The flat tile kernels are the result of a lot of tuning and their bit-exactness is a
  property the test suite relies on; gathering a chunk into the scratch they already expect keeps the arithmetic identical for every
  cache type and block size. It cost 43 % of f32 decode at 2 000 tokens, which is why that one case reads the blocks in place now; for f16
  and q8 the flat kernels expand into scratch anyway.
* **One big buffer per layer on Metal.** Per-block buffers or argument buffers need a buffer binding or an indirection per block in every
  kernel; one buffer per layer needs one table lookup per key row (a cached `int` load) and keeps a block contiguous for host-side
  copy / spill. Not chosen: aliasing scattered pages into a contiguous virtual range with `mach_vm_remap` (no kernel changes, but a
  syscall per block per layer per growth, 16 KB page granularity for tiny blocks, and a native shim change).
* **Staging rows + a paged store kernel on Metal, not a paged `matmul3`.** `matmul3` and `rope` write the cache directly in
  more than twenty variants (f32 / half, plain / bias / norm); making each page-aware would touch the most tuned kernels, while the q8
  cache had already proven the staging path. The cost is one more dispatch per layer (2 % on decode at 0.6B).
* **No replay patching of tables.** The table is data, read when the kernel runs. Recording its contents would tie a recorded step to one
  sequence's blocks; reading it keeps recordings valid across copy on write, adoption and eviction.
* **Sharing is an engine feature, not a backend one.** Backends only keep a table valid; `match` / `adopt` / `publish` live in the
  engine, so a new backend gets sharing, eviction, spill and snapshots for free by implementing `KvStorage` and the `beginWrite` contract.

**Costs.** One table lookup per key row on the GPU (decode −2 … −3 %, prefill −2 … −7 % measured); CPU decode about −3 % at q8
(41.4 → 39.9 tok/s) and f32 (36.0 → 34.9), free at f16 (9.7 → 9.9, within noise), the blocks read in place (before that path the gather
cost f32 43 % and f16 / q8 13–17 %); a second `Metal` context and a reserved address range per model; q4 is not paged; idle slots keep
their blocks referenced until the caller or the console's sweeper releases them.

---

## 24.4 Settings

| key | default | |
|---|---|---|
| `backend.kv-paged` | `auto` | `auto`, `true` or `false`. `auto` = paged for a model loaded with **two or more slots** (the console's Models page, `slots`), flat for one slot (`run`, `chat`, `bench`, `serve` with a single slot), which has nothing to share. The decision is made once per model by `ModelLoader.paged` (the first `load` knows the slot count; `extraEngine` finds it again in a weak map keyed by the weights). Attention models on the CPU and Metal with f32 / f16 / q8; Mamba, Qwen3-Next / Qwen3.5, DeepSeek's latent attention and q4 keep their flat caches (a note is printed when `true` was asked for). Bit-identical to the flat cache, so switching changes nothing but memory and speed |
| `backend.kv-block` | `16` | positions per block, a power of two ≥ 16. Larger blocks mean fewer table entries and bigger spill files but more waste in a conversation's last block and coarser sharing granularity for the *table* (the engine still reuses to the token); change only to test (`64` measured the same) |
| `backend.kv-pool-tokens` | `0` | most positions all slots of a model keep in the pool (0 = every slot's `backend.ctx` plus two spare blocks each, i.e. never more memory than the flat caches). Raise it to cache more conversations than there are slots (they are evicted least recently used first); the pool allocates only what is used, on Metal as address range |
| `backend.kv-prefix-share` | `true` | adopt and publish blocks; `false` pages without sharing (each conversation keeps its own blocks, like the flat cache) |
| `jllm.web.kv-spill` | `true` | web console, paged models only: evicted blocks go to `<jllm.web.kv-cache-dir>/blocks/<fingerprint>/` |
| `jllm.web.kv-spill-max-gb` | `10` | disk budget of those block files (least recently used files go first) |
| `jllm.web.kv-idle-release-seconds` | `120` | web console, paged models only: a slot idle this long gives its blocks back to the pool (the conversation stays in the tree); `0` = only when the pool runs out |
| `-Djllm.cpu.pagedGather=true` | off | CPU decode through the gather instead of in place (for comparisons) |

`backend.ctx` is `auto` by default ([10 §10.13](10-engine-and-server.md#1013-the-default-context-length-backendctxauto)); `ModelLoader` decides it
once per model, before the pool is created, and the pool key and size use that number. With paging on, the CPU budget `auto` uses is the
physical memory (the blocks are off the Java heap) instead of the free heap of the flat cache, bounded by the JVM's direct-memory limit
(`-XX:MaxDirectMemorySize`, by default the heap size), which the slabs of `CpuKvStorage` count against ([10 §10.13](10-engine-and-server.md#1013-the-default-context-length-backendctxauto)).

---

## 24.5 Measured

Machine: Apple M1 Pro, 10 cores, 32 GB, macOS (Darwin 27.2), 2026-10-22. The memory and time-to-first-token CPU rows ran while other
agents' jobs were running at times (load average 5–11, so their absolute figures vary); the CPU speed rows were re-measured on a quiet
machine with JDK 26 and an empty `JLLM_CONFIG`. Model Qwen3-0.6B (Q4_0 weights), `backend.ctx 8192`, greedy, `engine.speculative off`,
under the heavy-job lock. **Compare the pairs, which run in the same process, alternating.** Tool:
`src/test/java/org/robomiri/jllm/PagedKvBench.java`.

```sh
JAVA="$JAVA_HOME/bin/java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Djllm.metal.lib=target/native/libjllm_metal.dylib"
$JAVA -cp target/classes:target/test-classes org.robomiri.jllm.PagedKvBench models/Qwen_Qwen3-0.6B --backend cpu     # memory, ttft, speed
$JAVA -cp target/classes:target/test-classes org.robomiri.jllm.PagedKvBench models/Qwen_Qwen3-0.6B --backend cpu --backend.kv-dtype <dt> --backend.kv-paged true  # CPU speed rows, <dt> = f16 / q8 / f32
$JAVA -cp target/classes:target/test-classes org.robomiri.jllm.PagedKvBench models/Qwen_Qwen3-0.6B --backend metal [--backend.kv-dtype f16]
$JAVA -Dbench.only=speed ...                                                                                        # one section
```

**Memory: N conversations sharing a 4 096-token system prompt** (each with 64 tokens of its own and 16 generated; `slots = N`). "Flat,
allocated" is what N flat slots reserve for `ctx 8192` (`KvCache.bytes`), "flat, in use" the same positions without the empty rest, "paged
pool" the pool's resident bytes after all conversations ran.

| conversations | flat slots, allocated (MB) | flat, positions in use (MB) | paged pool (MB) | paged / flat in use | prefill of the n-th conversation, CPU (ms) | Metal (ms) |
|---|---|---|---|---|---|---|
| 1 | 1 879 | 958 | 958 | 1.00 | 24 272 | 6 640 |
| 2 | 3 758 | 1 916 | 976 | 0.51 | 605 | 197 |
| 4 | 7 516 | 3 831 | 1 013 | 0.26 | 533 | 220 |
| 8 | 15 032 | 7 663 | 1 086 | 0.14 | 529 | 196 |

(f32 caches; with `--backend.kv-dtype f16` the pool is 479 / 488 / 506 / 543 MB.) Every extra conversation costs its own tail (about
18 MB at f32) instead of the 958 MB of the system prompt again, and its prefill is the tail only (0.2–0.6 s for 64 tokens, instead of 6.6 s on Metal /
24 s on the CPU). The flat engine could copy a prefix between slots (§17.2, 34 ms per 10 000 tokens) but stores it once per slot; here it exists once.

**Time to first token of a resent conversation** (a 4 160-token conversation A; "next turn" = A + 8 new tokens in the slot that holds A;
"resent after its slot was taken" = another 4 160-token conversation overwrote slot 0 and A arrives on slot 1, which has nothing):

| | prefill before the first token (ms) | tokens reused |
|---|---|---|
| CPU flat: next turn in the same slot | 118 | 4 160 |
| CPU flat: resent after its slot was taken | 23 734 | 0 |
| CPU paged: next turn in the same slot | 158 | 4 160 |
| CPU paged: resent after its slot was taken | **71** | 4 159 |
| Metal f32 flat: next turn / resent | 66 / 6 315 | 4 160 / 0 |
| Metal f32 paged: next turn / resent | 68 / **32** | 4 160 / 4 159 |
| Metal f16 flat: next turn / resent | 38 / 5 776 | |
| Metal f16 paged: next turn / resent | 44 / **22** | |

A resent conversation costs an adoption (a table write per block and one 16-row copy) instead of a prefill, whichever slot it lands on;
the same-slot case is unchanged within noise. (Evicted conversations come back from the block store at disk speed: `PagedEngineTest` reloads
10 blocks / 160 tokens of a 0.6B conversation from 77 MB of files.)

**Prefill and decode speed** (one conversation: 2 000-token prompt, 64 generated tokens, median of 5 alternating runs of the flat and the
paged engine in one process):

| backend, cache | prefill tok/s flat | paged | decode tok/s flat | paged | decode paged / flat |
|---|---|---|---|---|---|
| Metal f32 | 1 044 | 970 | 108.6 | 105.6 | 0.97 |
| Metal f16 | 1 136 | 1 091 | 124.5 | 121.5 | 0.98 |
| Metal q8 | 1 139 | 1 118 | 127.4 | 123.0 | 0.97 |
| Metal f32, decode replay off | 1 053 | 993 | 109.0 | 107.4 | 0.99 |
| CPU f32 | 202 | 202 | 36.0 | 34.9 | 0.97 |
| CPU f16 | 193 | 203 | 9.7 | 9.9 | 1.02 |
| CPU q8 | 188 | 188 | 41.4 | 39.9 | 0.96 |

(The q8 rows were measured with plain q8 keys, before `backend.kv-q8-rotate` became the default; the rotation adds the same Hadamard transform of each key row and query to the flat and the paged cache.) On Metal the page table costs 2–3 % of decode and 2–7 % of prefill (a lookup per key row, one more dispatch per layer for the store); replay on
or off makes no difference to the ratio. On the CPU the blocks are read in place: decode costs about 3 % at q8 (41.4 → 39.9 tok/s) and f32 (36.0 → 34.9) and is level at f16 (9.7 → 9.9, within noise); prefill is unchanged. Before the in-place path the gather cost f32 43 % in a quiet run (46.6 → 26.8 tok/s) and f16 / q8 13–17 % (a gathered chunk is assembled from two 16-key blocks).

**The default.** The paged cache gives the same logits and greedy tokens as the flat one on every family of `ArchSmoke` (CPU and
Metal; §24.6), so the choice is about speed and memory: with two or more slots, or with agents that resend long prefixes, paged wins by
an order of magnitude in memory and time to first token for a 2–3 % decode cost on Metal and about 3 % (q8, f32; f16 free) on the CPU, so
`backend.kv-paged=auto` (the default) pages a model loaded with two or more slots; a single-slot command line gains nothing from sharing
and stays flat. `true` / `false` force either.

---

## 24.6 Checks and limits

Verified by (all in `mvn -o install` / `scripts/arch-smoke.sh` / `kernel_tests.py`):

* `KvPoolTest` (no model): the pool, tree, copy on write, snapshots, LRU eviction leaf first, exhaustion, spill and reload also after a
  "restart", fingerprints, six threads on one pool.
* `cpu/PagedKvTest`: 1 968 comparisons of the paged CPU cache against the flat one, bit for bit, over f32 / f16 / q8, head widths 32 / 64 /
  128, group sizes 1…8, block sizes 16 / 32 / 64, windows and soft-caps, with rewinds into a block, verification chunks, adoption and a fork.
* `PagedEngineTest` (Qwen3-0.6B, CPU, deterministic): logits and greedy answers equal for f32 / f16 / q8; two slots share a prefix (11 blocks
  for the first, 4 more for the second), a fork at token 123, a retry, copy on write counted; eviction to a spill directory and reload; n-gram
  speculation; three slots in shared passes; flat ↔ paged saved caches and a background save from a snapshot while the engine serves.
* `ArchSmoke` → `PagedKvChecks`, for every attention family, CPU and the real GPU (or the emulator), f32 / q8 (plain and with rotated keys) and
  f16 on the first dense and MoE: scripted prefill, decode (Metal: recorded and replayed), rewind into a block, a verification chunk with every row's logits; adoption,
  fork, rewind in a shared block; eviction and reload; three rows in one batched pass; saved bytes both ways and a snapshot. All compare
  **bit for bit** with the flat cache. Each comparison runs once, from a cold JVM: the float `dot` and the softmax sum used `reduceLanes(ADD)`, which adds the lanes
  in another order in the interpreter and C1 than in C2, so the first calls of a fresh JVM differed in the last bit, even the flat cache from
  itself (an intermittent failure of the f16 check, about one run in thirty). They now add through `VectorKernels.laneSum`, one fixed
  order that equals C2's on NEON, so steady-state results did not change, and `cpu/JitStableKernelsTest` covers `dot`, `sumSquares`,
  `softmaxExp` and the int8-activation dot next to the quantized dots (it fails on the old `reduceLanes` code in a fresh JVM). The checks used to repeat until the flat run agreed with itself; that workaround is gone.
* `scripts/metal-emu/kernel_tests.py`: every changed kernel with a shuffled pool and a table (garbage in the unused blocks) against numpy:
  `attention_t` (f32, q8, all widths), `attention_split_q8`, `attention_dec` (f32 / f16 / q8, splits, window, soft-cap), `attention_fa` /
  `attention_fb` (f32 / f16 / q8, window, soft-cap, chunk starts inside a block), `kv_store_q8`, `kv_store_q8r` and `kv_store_pg` (f32 and
  half), block sizes 16 / 32 / 64.
* `jllm-web` `PagedSlotsTest`: two conversations with a shared 4 000-token prefix store it once (the pool's resident bytes are counted: both
  together take under 1.1× one conversation and about half of two flat slots), the second prefills only its tail, and a conversation
  pushed out of a small pool comes back from the block store.
* `cpu/LongContextTest`: a million positions (24.2.9).

**Not done / limits.**

* **q4 caches are not paged** (CPU and Metal): their recent-window ring is per slot and not a function of the tokens alone (rows of the last
  `backend.kv-q4-recent` positions are kept exact only in that slot), so a shared block cannot carry it. A q4 slot keeps its flat cache (a note
  is printed).
* DeepSeek's latent cache, Mamba and Qwen3-Next / Qwen3.5 keep flat caches (24.2.8).
* ~~The Metal pool is not counted in `DeviceLedger`~~: fixed in chapter 26 (§26.2.1): every Metal pool registers `hotCap · blockBytes`, so the MoE expert placement leaves room for it.
* On AVX-512 with `B = 16` the f32 decode gathers (no direct kernel for one-vector-wide blocks).
* An idle slot gives its blocks back only in the web console (`jllm.web.kv-idle-release-seconds`, or when the pool is full); a bare `Engine` on a
  paged backend keeps them until the slot moves on, unless the caller calls `releaseKv`. A pool holding more than its blocks of *running*
  sequences still ends in `KvPoolExhausted`.
* The GGUF K-quant kernels (`KQuantKernels`, `Q2KKernels`, `Q6KKernels`, `GgufKernels`) still sum lanes with `reduceLanes`; the paged-versus-flat
  checks do not use them (they run Q4_0 / Q8_0 weights), but a GGUF K-quant model can differ in the last bit between its first calls and the rest.
* The flat-cache attention on Metal needs a scores scratch of `tile · heads · ctx` floats, so the Metal paged cache reaches a million tokens
  in storage but not through `attention_t`; the decode kernels (`attention_dec`, `attention_split`) have no such limit.
* No scheduler; sparse decode attention on this cache is chapter 27, while tiered (host / SSD) storage, streamed KV and the memory planner are chapter 26. This chapter is the foundation they need.
