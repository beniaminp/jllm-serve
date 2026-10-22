# 26. Sparse attention while decoding

A decode step has one new token and a cache of everything before it, and plain attention reads **all** of that cache to produce
one token: at 32 000 tokens the keys and values of Qwen3-0.6B are 3.7 GB in f16, read again for every word the model says. This
chapter makes a decode step read **only the blocks of the paged cache that matter for this token** (chapter 24), behind a switch
(`backend.kv-sparse`, off by default): each block keeps a tiny *summary* of its keys, the new token's query is compared with the
summaries to rank the blocks, and the token attends, exactly, over the best few blocks plus the first block and the most recent
ones.

| | | where |
|---|---|---|
| [26.1](#261-in-plain-words) | the idea, told as a story | |
| [26.2](#262-how-it-works) | block summaries, the bound `q·k ≤ Σ max(q_i·lo_i, q_i·hi_i)` derived and worked with numbers, rotated keys, the selection, the exact attention over it, the CPU and Metal kernels, replay and batched rows | `kv/KvSummary`, `cpu/PagedKv`, `cpu/CpuKvStorage`, `metal/MetalKvStorage`, `kernels.metal` §4c |
| [26.3](#263-why-this-design) | StreamingLLM, H2O, Quest, DuoAttention, and why this one; what it costs | |
| [26.4](#264-settings) | `backend.kv-sparse`, `-blocks`, `-sink`, `-window`, `-dense-layers` | |
| [26.5](#265-measured) | needle in a haystack at 8k, 16k and 32k, perplexity, decode speed against context length | `KvStudy`, `./jllm bench` |
| [26.6](#266-checks-and-limits) | the tests, and what is not done | |

---

## 26.1 In plain words

🧸 You are looking for one sentence in a very long book, and you may glance at every page before you answer. That works, but a
glance at 2 000 pages takes a long time, and you do it again for every word you say.

Now the librarian has stuck a **tiny card on every 16 pages**. The card does not say what is on the pages. It says, for each
topic you could ask about, *the highest and the lowest* anything on those pages ever reaches ("somewhere here a page mentions
'harbors' at most this strongly, and at least this weakly"). When you ask a question, you do not read the book: you walk along
the cards, and for each card work out **the most that these 16 pages could possibly matter to your question**. A card cannot
promise that a page *does* matter, but if even its best case is small, nothing on those pages can matter, and you skip them.

You keep the best 64 sets of pages, and you always keep the very first pages (the title, the "once upon a time" that everything
refers back to) and the last few pages you just read (what you were just saying). Then you read those pages properly, every
word, exactly as before. The cards cost a little paper (about 3–12 % of the cache) and a quick walk along them (an eighth
to a quarter of what reading the pages would cost), and everything else is skipped.

It is **lossy**: if the cards rank a page below the cut and that page mattered, the answer misses it. That is why it is a switch,
why it is off, and why this chapter measures how often a hidden fact is still found.

---

## 26.2 How it works

📐 The running example is Qwen3-0.6B: 28 layers, 16 query heads over 8 kv heads of `headDim = 128`, blocks of `B = 16`
positions (chapter 24). Everything here is per layer and per kv head: the two query heads of a group share the kv head's blocks.

### 26.2.1 The block summary

For one layer, one kv head and one block, the summary is two vectors of `headDim` numbers, the **per-channel minimum** and
the **per-channel maximum** of the block's keys:

```
lo_i = min over the block's valid keys k of k_i          hi_i = max over the block's valid keys k of k_i          (i = 0 .. headDim-1)
```

It is stored as halves, **rounded outwards**: `lo` rounded down to the largest half that is `≤ lo`, `hi` rounded up. So the stored
box always contains the real keys (a round-to-nearest half could be a hair inside), which keeps the bound below an upper bound. A
value beyond ±65 504 (the largest half) is clamped; real keys are far below it.

Worked rounding (`KvSummary.down` / `up`): the channel minimum `-0.1` has the nearest half `-0.0999755859375`, which is *above* it,
so the summary stores the next half below, `-0.10003662109375`; a maximum of `0.3` has the nearest half `0.300048828125`, already
above it, and is stored as it is.

**Layout.** Per layer, `[block][kv head][lo[headDim] | hi[headDim]]` halves: 512 bytes per kv head of a block with `headDim = 128`,
`8 · 512 = 4 096` bytes per layer, **114 688 bytes per block of 28 layers (7 168 per position)**. On the CPU it is a third region
at the end of each layer of the block (after K and V), packed two channels to an `int` like the f16 keys, so a block copy (copy on
write) copies it with the rows; on Metal it is a buffer per layer next to the K and V buffers, indexed by the same block id.

```
Qwen3-0.6B, one position:   f32 229 376 B     f16 114 688 B     q8 59 136 B       + summaries 7 168 B  (3.1 %, 6.3 %, 12.1 %)
CPU block of one layer:     [ K region | V region | summary: int[kvHead][2][pairs] ]
Metal, summary buffer:      sums[layer] = half[maxBlocks][kvHeads][2][headDim]      (lo, then hi)
```

**What it is computed from.** The summary is of the keys **as stored**: the dequantized values for f16 and q8, and for the q8
cache with **rotated keys** (`backend.kv-q8-rotate`, the default for q8) the **rotated** and then quantized keys, in the
rotated basis. Section 26.2.3 explains the choice.

**When it is updated.** Every time keys are stored (a prefill chunk, a decode step, a batched row, a verification chunk) the summaries of the blocks
that were touched are **recomputed from their valid rows** (`KvCache.store` on the CPU, the `kv_summary` kernel on Metal),
not merged into. Recomputing is what keeps them exact when a block is written again after a rewind (speculation rejects a draft, a
conversation is retried in the middle of a block): a merge would keep the dead rows' extremes forever. A block copied in part
(copy on write, adopting a prefix's first rows) and a block read back from disk are recomputed by the host
(`CpuKvStorage.updateSummary`, `MetalKvStorage.updateSummary`); a saved cache that is loaded grows the summary row by row
(`mergeSummaryRow`) or recomputes it.

### 26.2.2 The bound: `q·k ≤ Σ max(q_i·lo_i, q_i·hi_i)`

**Claim.** If every key `k` of a block satisfies `lo_i ≤ k_i ≤ hi_i` in every channel, then for any query `q`

```
 q·k  =  Σ_i q_i k_i  ≤  Σ_i max(q_i lo_i , q_i hi_i)  =: U(q, block)
```

**Proof.** For one channel, `q_i k_i` is a linear function of `k_i`, and a linear function over the interval `[lo_i, hi_i]` is
largest at one of its ends: at `hi_i` if `q_i ≥ 0`, at `lo_i` if `q_i < 0`, which is exactly `max(q_i lo_i, q_i hi_i)`. Every term of
the sum is therefore at most its channel's maximum, and so is the sum. ∎  (The bound is the largest dot product with any point of
the *box* `[lo, hi]`; the block's real keys are some points of the box.)

**Worked example** (`headDim = 4`, a block of four keys):

```
 k1 = ( 0.5, -1.0,  2.0,  0.0)      lo = (-0.5, -1.0, -0.5, -1.0)        q = ( 1.0, -2.0, 0.5, 1.5)
 k2 = ( 1.5,  0.5, -0.5,  1.0)      hi = ( 1.5,  1.0,  2.0,  2.0)
 k3 = (-0.5,  1.0,  1.0, -1.0)
 k4 = ( 0.0, -0.5,  0.5,  2.0)
 scores   q·k1 = 0.5 + 2.0 + 1.0 + 0 = 3.5     q·k2 = 1.75     q·k3 = -3.5     q·k4 = 0 + 1.0 + 0.25 + 3.0 = 4.25   -> the best is 4.25
 per channel:  max(1·-0.5 , 1·1.5) = 1.5   max(-2·-1.0 , -2·1.0) = 2.0   max(0.5·-0.5 , 0.5·2.0) = 1.0   max(1.5·-1.0 , 1.5·2.0) = 3.0
 U = 1.5 + 2.0 + 1.0 + 3.0 = 7.5  >=  4.25
```

A block whose keys all sit in a small box — `lo = (-0.5, -0.5, -0.5, -0.5)`, `hi = (0.5, 0.5, 0.5, 0.5)` — gets `0.5 + 1.0 + 0.25 + 0.75 = 2.5`.
The first block ranks above it (7.5 against 2.5), and rightly: it holds a key that scores 4.25 and the other cannot score above 2.5.
The bound is loose by the distance between the box's corner and the real keys (7.5 against 4.25 here), so it can rank a block with a
lower true best score above one with a higher one; that is the approximation. What it guarantees is the one-sided statement: **a
block whose bound is below a threshold cannot contain a key above it**, so a block is never skipped *because the bound said it was
small* while a key in it was large.

**Groups of query heads.** The kv head's blocks are shared by its `G` query heads (2 for Qwen3-0.6B), so a block's score is
the **largest bound over the group's heads**, `max_h U(q_h, block)`: a block is wanted if any head wants it. The attention scale,
the soft-cap and the softmax are monotone in the score and need not be applied for ranking.

**Cost of the bound.** Per block and kv head it reads the 512-byte summary instead of the block's keys (2 048 B in q8, 4 096 B in f16,
8 192 B in f32): **a quarter, an eighth, a sixteenth** of the key traffic, with `2 · headDim` multiplications per query head. It
is still linear in the context (the cards must all be read); 26.3 says what a hierarchy of cards would do.

### 26.2.3 Rotated keys: summaries before or after the rotation?

With `backend.kv-q8-rotate` the cache stores `H k` (the Walsh–Hadamard transform of the key head, then quantized) and the attention
computes `(Hq)·(Hk) / B`, where `H` is the unnormalised transform and `B = headDim`: `(Hq)·(Hk) = B q·k`, because `H Hᵀ = B I`
(docs 07 §7.13). The bound holds in **any** basis as long as the query and the summary are in the **same** one, because it is a statement
about a dot product with a box. Two choices:

* **After the rotation (chosen).** The summary is the min / max of the stored rotated, quantized key channels, and the query that is
  scored is the rotated query that the attention itself uses (the store kernel rotates `q` in place, so it is simply the buffer the
  attention reads). `U` is then an upper bound of **the score the attention will compute**, quantization error included. Nothing extra is
  kept, nothing is rotated twice, and the kernels do not know that the cache is rotated: they see halves and a query.
* **Before the rotation.** The summary would be of the original keys, the query would have to be the original query, which is
  rotated in place and gone by the time the summary is scored (a copy per layer and head), and the keys would have to be kept
  unquantized for the summary pass. The bound would then be on the exact score, not on the one the model uses with the quantized cache.

The costs of the choice: the rotated box is a *rotated* box in the original space, so it can be looser or tighter than the
unrotated one. The rotation spreads a loud channel over all channels (which is why it helps quantization): the per-channel ranges
become more uniform, the original box's few wide channels (where Quest's bound is tight, because the query's weight on them is
small or large) disappear. The retrieval table in 26.5 measures both with the same selection settings.

### 26.2.4 The selection

For the token at position `p` (block `nb = p ÷ B + 1` counting the one that holds it):

```
 sink    = ceil(backend.kv-sparse-sink / B)       blocks 0 .. sink-1                       always attended   (attention sinks)
 window  = max(1, ceil(backend.kv-sparse-window / B))   blocks nb-window .. nb-1           always attended   (recent tokens, the token's own block)
 K       = backend.kv-sparse-blocks             the K best of the candidates sink .. nb-window-1 by U(q, block)
 budget  = sink + window + K
```

If `nb ≤ budget` the row is attended **densely**, exactly as with the switch off; only a longer context selects. The chosen blocks
are the sinks, the K candidates, the window, **in ascending block order** (the block with the token is the last, and the only
partly filled one). The ranking is a total order: higher bound first, **the lower block on a tie**
(`KvSummary.rank`), the same on the CPU (quickselect over `(score, block)` keys) and on the GPU (a radix select).

Worked example, `B = 16`, `sink = 1`, `window = 2` (32 tokens), `K = 3`, a token at `p = 700`: `nb = 44`, `budget = 6`, candidates are blocks 1 … 41.
Say the largest bounds are blocks 17 (91.2), 30 (88.0), 3 (87.5) and 22 (60.1). Selected: `[0, 3, 17, 30, 42, 43]` — the sink, the top three, the
window — **93 keys (five full blocks and 13 rows of the last), of the 701 the dense step reads**. The token attends exactly over those 93 keys (the last block holds `700 mod 16 + 1 = 13` rows).

### 26.2.5 Attention over the selection

Exactly dense attention, over fewer keys: scores `s_j = scale · q·k_j` for the keys of the selected blocks, the soft-cap if the model has
one, a softmax over **those** keys, and the weighted sum of their values. The CPU gathers each selected block's keys (and later
its values) into scratch (`PagedKv.gatherKeys` / `gatherValues`, any of f32 / f16 / q8) and uses the same vector arithmetic as the dense
kernels; the numbers equal a float64 attention over the selected keys to rounding (`SparseKvTest`).

On Metal the dense decode kernel `attention_dec` (chapter 08 §8.31) is reused as it is: a simdgroup takes a slice of the keys of a
group of query heads. Here the "cache" of a kv head is its own list of `budget` physical block ids, so key `s` of the *virtual* sequence is row
`(sel[kvh][s >> shift] << shift) | (s & mask)` of the pool, and the virtual length is `(budget-1)·B + (p mod B) + 1`. That is one more
argument of the kernel body (`dec_body<…, SP>` in `kernels.metal`, the dense kernel is the same body with `SP = false`) and a table
offset by the kv head; the work is split over `min(24, ceil(budget·B / 32))` simdgroups per head group, a number that does **not**
depend on the position.

### 26.2.6 The kernels, in the order a decode step runs them (per sparse layer)

```
 store k, v        (rope_store_q8r / kv_store_*)           the position's key and value rows go into the block table's rows
 kv_summary        grid (kvHeads, touched blocks), headDim threads: min / max of the block's valid rows -> sums[layer]
 sparse_score_d*   grid (ctxBlocks/8, kvHeads), 128 threads: four simdgroups, a block each, U for the group's heads -> scores[kvh][block]
 sparse_select     grid (kvHeads), 256 threads: radix select of the K-th best key (4 passes of 8 bits, a threadgroup histogram), then an
                   exclusive scan of the threads' counts writes sinks, K blocks ascending, window as PHYSICAL ids -> sel[kvh][budget]
 attention_dec_sp_*  attention_dec over sel; partials in `part`
 attention_combine   merges the partials (as the dense step)
```

Three dispatches more than a dense decode layer (`kv_summary`, `sparse_score`, `sparse_select`). `scores` is `kvHeads · ctxBlocks` floats and
`sel` `kvHeads · budget` ints, shared by the layers. The grid of `sparse_score` is fixed by the context length, not by the position:
threadgroups past the token's last block return at once, so a **recorded decode step stays valid** (decode replay, chapter 08 §8.22):
position-dependent values (`pos`, the number of blocks) are parameters patched like the other kernels' positions, and the one
structural change, "this layer is sparse from this position on", is part of the replay key (one more recording where the context
passes the budget). The block table is read by the GPU when the kernels run, as everywhere in the paged cache.

`sparse_select` details: the key of a score is `u ^ (sign ? 0xFFFFFFFF : 0x80000000)` of its float bits, which orders the floats as unsigned
integers; four passes find the largest key `T` with at least `K` candidates at or above it, `remaining` of the candidates equal to `T`
are taken lowest block first (thread 0 scans the 256 threads' counts, which also gives each thread its first output slot), so the
chosen set is the K first of the total order above and does not depend on the thread count.

CPU (`cpu/PagedKv`): the same steps per kv head as one task (`pool.forRangeFine(kvHeads)`): bounds with vector loads of the packed
halves (`F16.lowHalf` / `highHalf`, the same conversion as the f16 keys), `KvSummary.top` for the selection, gather, softmax, accumulate.

**Rows and chunks.** Sparse applies to a step of fewer than `KvSparse.MAX_ROWS = 16` tokens (the same threshold as the Metal decode attention): a decode
step, a speculative verification chunk (every row selects with its own position, so verification equals separate decode steps, bit for bit on the CPU),
the tail of a prompt. A prefill chunk of 16 or more tokens is **dense**, and so is a layer with a sliding window and the first
`backend.kv-sparse-dense-layers` layers (the first layers of a transformer attend broadly, so skipping their blocks costs the most; Quest also keeps two layers dense).

---

## 26.3 Why this design

The question is which keys a token may skip. The published answers:

| method | what it keeps | needs | why not (here) |
|---|---|---|---|
| **StreamingLLM** (Xiao et al. 2023) | the first few tokens (sinks) and a sliding window | nothing | cheap and simple, but it *forgets* the middle: a fact 20 000 tokens back is gone. We keep its two ideas (sinks, recent window) as the always-attended part. |
| **H2O** (Zhang et al. 2023) | the "heavy hitters": tokens with the largest accumulated attention so far, plus the recent window; the rest is *evicted* | attention scores of every step, per head | evicts permanently, so a token that is irrelevant now and essential later is lost; needs the attention matrix (or its row sums), which flash-style kernels never materialise. |
| **Quest** (Tang et al. 2024) | everything stays; each step picks pages by a query-aware bound from per-page key min / max | the summaries | nothing is lost, the choice is made again for every token and the query. **Chosen**, with the bound of 26.2.2. |
| **DuoAttention** (Xiao et al. 2024) | per head: *retrieval* heads keep the full cache, *streaming* heads keep sinks + window | an offline calibration that classifies heads | a per-model training step and a different cache size per head; it composes with the paged cache but needs data we do not ship. Not done. |

Choices inside Quest:

* **Per block, not per token.** The paged cache already groups 16 positions; a summary per token is the key itself. Blocks of 16 are
  also what Quest found a good page size.
* **Per kv head, shared by the query-head group,** so each K and V row is read once for the group (`attention_dec` does that). Per query head would double the
  summaries scored and the rows read for Qwen3-0.6B (`G = 2`) and 8× for Qwen3-30B-A3B (`G = 8`).
* **Min / max, not mean or a centroid.** A mean can rank a block low although one key in it is huge; the box bound cannot be below the best key.
* **Halves, outward.** 4× less summary memory and bandwidth than f32 on the q8 cache and exact containment; the cost is a bound up to one half-step looser per channel (relative 2⁻¹¹).
* **Recompute, not merge.** Exact after rewinds and copy on write; costs one pass over the block's ≤16 rows per layer per step (a vector reduction; the `kv_summary` dispatch).
* **Exactly the same attention over the selection,** no approximation inside the chosen blocks.
* **The order of selection is a total order,** so CPU runs are reproducible and speculation can be checked bit for bit.
* **Selection on the GPU,** not a readback of the scores: a readback would add a wait to every layer.
* **Dense for short contexts and prefill:** nothing is lost where nothing is saved.

**Costs.** Memory: 3.1 % (f32), 6.3 % (f16) or 12.1 % (q8) more per position. Time: three more dispatches per sparse layer, and the bound
is **linear** in the context (every block's summary is read each step), so at very long contexts the cards themselves dominate: at 1M tokens
65 536 blocks · 28 layers · 4 096 B = 7.5 GB of summaries read per token. A second level (a summary of 64 blocks, scored first, then the blocks of the best groups) would make it sub-linear; it is not built (26.6). Accuracy: lossy; the tables in 26.5.

---

## 26.4 Settings

| key | default | |
|---|---|---|
| `backend.kv-sparse` | `false` | sparse decode attention. Attention models (grouped-query, head size 64 / 128 / 256 on Metal) on the CPU and Metal with the f32, f16 and q8 (plain or rotated) paged cache; **turns on `backend.kv-paged`** (a note says so). With `false` nothing changes: no summaries are kept, the pool and the kernels are the paged cache's. Needs the paged cache: Mamba, Qwen3-Next / Qwen3.5, DeepSeek's latent attention and q4 keep dense flat attention (a note says so) |
| `backend.kv-sparse-blocks` | `64` | blocks chosen by their bound at every step (of `backend.kv-block` = 16 positions each). More is closer to dense and slower; 64 is 1 024 positions |
| `backend.kv-sparse-sink` | `16` | leading positions that are always attended, rounded up to whole blocks (16 = one block) |
| `backend.kv-sparse-window` | `256` | trailing positions that are always attended, rounded up to whole blocks (at least one) |
| `backend.kv-sparse-dense-layers` | `2` | the first layers keep dense attention |

With the defaults a context of 1 296 positions or less (81 blocks) is attended densely; beyond it a step reads 81 blocks = 1 296 positions of every sparse layer.
The values are the model loader's (`ModelLoader.sparse`); a pool is created for one sparse setting (it is part of its key), so one process serves one.

---

## 26.5 Measured

MEASURED_PLACEHOLDER

---

## 26.6 Checks and limits

LIMITS_PLACEHOLDER
