# 25. The scheduler: one queue, mixed passes, priorities

[Chapter 10](10-engine-and-server.md) gave every slot of a server its own engine and let the slots that happen to generate at the
same moment share a forward pass (§10.9b). Everything else was first come, first served at a lock: a slot was chosen when a
request arrived, a long prompt's prefill chunks and the other slots' decode steps took turns at the device, and nothing knew
that one request was a person waiting and another a subagent summarizing files. Chapter [24](24-paged-kv-cache.md) then made
the KV cache a pool of blocks that any slot can use. This chapter builds the scheduler on that, behind a switch
(`engine.scheduler=mixed`; `slots`, the behaviour of chapter 10, stays the default).

| | | where |
|---|---|---|
| [25.1](#251-in-plain-words) | the idea, told as a story | |
| [25.2](#252-how-it-works) | the global queue and who goes next, mixed passes and the token budget (a worked pass), the mixed forward on the CPU and Metal, priorities and aging, pausing a generation by spilling its blocks, batched verification of drafts, a client that leaves the queue | `engine/Admission`, `engine/PassPlanner`, `engine/BatchGroup`, `Backend.forwardMixed`, `kv/KvSequence.suspend`, `web/models/LoadedModel` |
| [25.3](#253-why-this-design) | the alternatives that were rejected, and what the design costs | |
| [25.4](#254-settings) | `engine.scheduler`, `engine.pass-budget`, `engine.mixed-reserve`, `engine.background-chunk-cap`, `engine.age-promote-ms`, `engine.max-running`, `engine.preempt`, `engine.session-max-running`, the request's `priority` and `user` | |
| [25.5](#255-measured) | a simulated agent workload on Qwen3-0.6B and Qwen3-30B-A3B: slots on the flat and the paged cache against mixed; throughput, time to first token, gaps between tokens | `SchedulerBench` |
| [25.6](#256-checks-and-limits) | the tests, and what is not done | |

---

## 25.1 In plain words

🧸 Picture a restaurant kitchen with one big stove. Some customers have **ordered a banquet** (a very long prompt: a coding agent
that sends the whole conversation again every turn, thousands of tokens), and others are **already eating and want the next
bite** (a conversation that is generating: one small token at a time). The old kitchen had a rule: the cook finishes a whole
tray of the banquet (a *chunk*), and only then looks at the little orders. A diner who wanted one more bite waited for the
whole tray, over and over, and the more banquets were ordered the longer every diner stared at an empty plate.

The new kitchen does something clever. A tray of banquet food is heavy work that the stove does *slowly* (the stove is the
bottleneck: **compute bound**), while one bite for one diner is light work that costs *almost nothing extra* once the stove is
hot (a small order is **memory bound**: the cook mostly walks to the pantry to fetch the ingredients, and a tray has already
fetched them). So **every time the cook puts a tray on the stove, the next bite of every diner goes on the same tray**. The
banquet loses a few percent of speed, and nobody ever waits for a whole tray any more.

Around the stove the kitchen also got a **waiting line with a manager**:

* **One line for the whole kitchen**, not one line per cook. When a cook is free, the manager looks at who waits and picks the
  order whose ingredients are **already prepared in the fridge** (the longest prefix of the prompt that is in the cache), because
  that order is the cheapest to serve and every cached minute is a minute for somebody else.
* **Regular diners go before catering orders** (interactive before background), but a catering order that has waited a long time
  is promoted a step, so nobody is forgotten.
* **One customer cannot take all the cooks** (a limit per session).
* If a regular diner arrives and the stove is crowded, the manager can ask a catering order to **step aside**: its food is put in
  the cold room (the blocks go back to the pool and, if space is needed, to disk) and it comes back and goes on from exactly
  where it stopped.
* If a customer **leaves the line**, the manager notices within a fraction of a second and forgets the order.

---

## 25.2 How it works

📐 Everything below is for a model with grouped-query attention on the CPU and Metal backends with an f32 / f16 / q8 cache
(Qwen3, Llama, Mistral, Qwen2, GLM, Phi-3, Gemma 2 / 3, Qwen3-MoE): the mixed scheduler needs the paged cache of chapter 24,
which it switches on by itself. A model it cannot serve (Mamba, Qwen3-Next, DeepSeek's latent attention, a q4 cache) prints no
error and keeps the `slots` scheduler. Qwen3-0.6B and Qwen3-30B-A3B are the running examples.

### 25.2.1 Where the time goes: the case for mixed passes

A forward pass over `n` rows costs, roughly, `T(n) = max( weight bytes / bandwidth , n · flops per token / compute )` plus the
attention of each row. For a single decode token (`n = 1`) the first term wins: the pass reads every weight once to produce one
row. For a prefill chunk (`n = 64 … 256`) the second term wins. Rows that join a pass whose second term is already the larger
one are almost free: their weights are read anyway.

```
decode token alone      n=1     | weights ################## (18 ms for the 0.6B model's 1/ms ... ) |  rows: 1
prefill chunk           n=256   | weights ###### compute ########################################    |  rows: 256
chunk + 7 decode tokens n=263   | weights ###### compute ##########################################  |  rows: 263  (+2.7 % rows, +~1 % time)
```

(the figures of the real passes are in [25.5](#255-measured)). Under the old scheduler a slot with a 4 000-token prompt held the
device for `4000 / 256 = 16` chunks in a row (each slot's thread took the device lock for its own chunk), and a generating
slot got a token only between two chunks, when it won the lock. Now:

```
slots:   device   [chunk A1][d B][d C][chunk A2][d B][d C][chunk A3] ...        (a lock decides the order)
         slot B   token ........ token ........ token                              (waits for A's chunk each time)

mixed:   device   [chunk A1 + d B + d C][chunk A2 + d B + d C][chunk A3 + d B + d C] ...
         slot B   token ................ token ................ token             (every pass gives it a token)
```

### 25.2.2 One queue per model, and who goes next

`engine/Admission` is the queue; `web/models/LoadedModel.pick` is its only user. A request arrives with its prompt ids, its
**class** (`interactive` = 0 or `background` = 1, from the request's `priority`) and its **session** (the OpenAI `user`, or none),
and waits in `Admission.acquire`. A slot is `IDLE`, `RUNNING` (takes part in passes) or `PAUSED` (still owned by its request, but its
blocks were given back, 25.2.6). At most `engine.max-running` slots run at once (default: all of them).

When a slot is released, a request arrives, or 50 ms pass (so aging is noticed), `dispatch` picks, among the waiting requests whose
session holds fewer than `engine.session-max-running` slots, and the paused slots:

1. the lowest **effective class**: the class minus one for every `engine.age-promote-ms` the request has waited, not below 0;
2. a paused slot before a new request of the same class (its work is half done and its blocks may still be in memory);
3. the highest **score** `s = h + 2000 · w`: `h` the tokens of the prompt already in the cache (`Engine.cachedPrefix`: the prefix
   tree plus the slot's own conversation), `w` the seconds waited;
4. the oldest.

The request then takes the idle slot with the longest cached prefix of its prompt. Because every slot reads and writes the same
pool through its block table, the slot is only a place to run: nothing is copied, a conversation's blocks follow it (a finished
request publishes its blocks and `Engine.releaseCache` lets go of them, so an idle slot pins nothing; the next turn finds its
conversation in the tree wherever it lands: in the paged benchmarks of chapter 24 that was 32 ms for a 4 160-token conversation).

Worked example, a model with 2 slots, both busy, three requests wait; the prefix tree holds a 3 000-token system prompt `S`:

```
 request   class  prompt                     cached h   waited   effective class   score s = h + 2000·w
 R1        1      S + 40 new tokens          3 000      4 s      1                 3 000 + 8 000 = 11 000
 R2        0      new 200-token question     0          0.2 s    0                 0 + 400 = 400
 R3        1      S + 90 new tokens          3 000      0.5 s    1                 3 000 + 1 000 = 4 000
```

The first slot that frees up goes to R2 (class 0 beats class 1 whatever the cache says), the next to R1 (same class as R3, and
11 000 > 4 000: the cache hit is the same, R1 has waited longer), then R3. Had R1 been waiting 12 s with `engine.age-promote-ms=10000` it
would have become class 0 too, and, with the higher score, gone first.

### 25.2.3 One pass, and its token budget

Every `forward` call of a slot that the engine makes now becomes a *step* in the group's queue instead of running at once
(`BatchGroup.Member.forward`):

| kind | what | rows |
|---|---|---|
| `DECODE` | the next token of a generating conversation (logits of its last row) | 1 |
| `VERIFY` | a speculative draft: `k + 1` tokens, the logits of every row (25.2.7) | `k + 1` |
| `CHUNK` | a slice of a prompt (up to `backend.batch` tokens); only the last chunk wants logits | up to 64 / 256 |

The thread that takes the device lock leads the pass (as in §10.9b it first waits up to 2 ms for the other generating slots) and asks
`PassPlanner.plan` what to run. The backend's buffers hold `backend.batch + engine.mixed-reserve` rows (default `+32`; the engine
still sees `backend.batch`, so prefill chunks keep their size). The **budget** `B` of a pass is `engine.pass-budget` rows or, at 0, all of
them. The plan:

1. The **riders** (`DECODE` and `VERIFY` steps) in order of effective class, then arrival, while they fit in `B`.
2. The **chunk**: the head of the `CHUNK` steps (by effective class, then arrival; since a chunk goes back to the end of the line
   when its next one is submitted, chunks of different prompts take turns). It gets the rest of the budget: all of it if it fits,
   else a piece of at least `min(16, its rows)` rows (the rest stays at the head of the queue); with fewer rows left the chunk
   waits for the next pass. With `engine.background-chunk-cap = c` a background chunk in a pass with an interactive rider is
   cut to `c` rows.
3. If nothing fits at all, the oldest step runs alone (the plan never stalls).

Worked example on Qwen3-30B-A3B (`backend.batch = 256`, reserve 32, buffers of 288 rows, `engine.pass-budget = 0`): a client sends a
2 400-token prompt (10 chunks of 256 rows, the first 2 already done), six conversations are generating, and one of them has
a prompt-lookup draft of 4 tokens to verify. The steps waiting, with their arrival times in ms:

```
 step           kind    rows  class  arrival
 D1             DECODE    1     0     1
 P-A (chunk 3)  CHUNK   256     0     2
 V1 (4 tokens)  VERIFY    5     0     3     (draft 4 + the token before it)
 D2..D6         DECODE    1     0     5..9
 P-B (chunk)    CHUNK   256     0    20      another prompt, arrived later

 pass:  P-A:256   D1:1  V1:5  D2:1 D3:1 D4:1 D5:1 D6:1        = 256 + 11 = 267 rows of 288
 next:  P-B:256   (the riders have been re-submitted and wait again)
```

The pass is one `forwardMixed` over 267 rows, and every conversation in it gets what it asked for: A another chunk done, six
tokens, five verified positions. With `engine.pass-budget = 100` the same plan gives the riders 11 rows and trims the chunk to
`100 − 11 = 89` rows (and P-B's chunk has to wait), a shorter pass, and so a smoother stream, for a slower prefill: the knob
for latency against prefill throughput (`SchedulerPolicyTest` checks these numbers).

### 25.2.4 The mixed forward

`Backend.forwardMixed(Segment[])` runs a list of `Segment(owner, tokens, off, n, pos, logits, stride)`: `n` tokens of one slot's
conversation at `pos`, with the logits of the last row, of every row (`stride > 0`: verification) or of none (a chunk that is
not the prompt's last). The rows of all segments are the rows of **one** set of activation buffers:

```
 rows of x:   | P-A chunk: 256 rows ............ | D1 | V1: 5 rows | D2 | D3 | D4 | D5 | D6 |   267 rows
 projections  q, k, v = one matmul over 267 rows      (each weight read once)
 per segment  RoPE (its positions), store into ITS blocks, attention over ITS cache
 MLP / MoE    one pass over 267 rows (a MoE layer's experts are read once for all of them)
 output       final norm only for the rows whose logits are wanted (1 + 1 + 5 + 5 = 12 rows), one LM-head matmul
```

On the CPU (`CpuBackend.forwardMixed`, `AttnStep.runSegments`) the segment's rows are copied to the lender's staging arrays and handed
to its owner's `KvCache.store` / `attention` exactly as `run(n)` would. On Metal (`MetalBackend.forwardMixed`,
`GqaStep.runSegments`) the projections write f32 rows into staging buffers (`rowK`, `rowV`, the batched-decode ones); then for
each segment the same dispatches a lone pass would issue are made, with the segment's rows at a **byte offset** of the shared buffers
(`MetalKernels.ropeAt`, `kvStoreQ8At`, `kvStoreQ8RotAt`, `attentionAt`, `flashPrefillAt`, `rmsnormTo`): a single-token segment takes the
batched-decode kernels (`ropeRow`, `attentionRow`, `decodeAttention`), a chunk the prefill ones (`attention_t` or the flash prefill
kernel, or `attention_dec` per row for a run shorter than 16 as in `run`). The block table of each owner is set before its stores
and attention (`MetalKernels.page`), inside the same independent groups the concurrent encoder uses for batched decoding. No kernel
changed: every new function is an existing dispatch with offsets.

Segments never see each other's rows: a segment's attention reads its owner's cache only, and a pass rejects two segments of the
same owner. That is why a mixed pass gives each conversation what it would get alone, 25.2.9.

### 25.2.5 Priorities

The request's `priority` (`"interactive"`, the default, or `"background"`, or the header `X-Jllm-Priority`; `engine/Priority`) travels
with the request thread (`Priority.current()`) and does four things under the mixed scheduler: it is the class in the queue (25.2.2),
the class of the steps in a pass (25.2.3: riders and chunks of class 0 go first when the budget does not hold all), the cap of
`engine.background-chunk-cap` and who may be paused (25.2.6).

### 25.2.6 Pausing a generation by spilling its blocks

With more slots than `engine.max-running`, an interactive request that finds no running place pauses the background generation that
started last (`Admission.pauseBackground`):

```
 background slot G generating:     table [b0 b1 ... b93 | t]       (93 full blocks, 1 partly filled tail t of 5 rows)
 pause requested ─▶ at G's next step boundary (Engine: checkpoint between two forwards)
     KvSequence.suspend(tokens, n):  publish the full blocks, release them (refs → 0, they are CACHED, evictable)
                                     keep only the tail t (its rows cannot be found through the tree)
 interactive request I runs:        needs 95 blocks; the pool has 46 free → evicts 49 of G's blocks, leaf first, to the block store (disk)
 I finishes ─▶ Admission resumes G: KvSequence.resume(tokens):  adopt b0..b93 from the tree, reload the evicted ones from disk,
                                     put the tail back → G goes on at position n with exactly the rows it had
```

`Backend.checkpoint` is called by the engine between prefill chunks and between decode steps (when no draft is queued): a plain
backend returns at once; a `BatchGroup.Member` with `pauseRequested` suspends its sequence, waits (polling the request's
cancellation every 50 ms), resumes it and returns the number of positions that are valid again. If a block was lost (evicted with
no block store, or the pool too full to bring it back) the engine computes the missing positions again from the tokens. A request
that asks to pause waits for the victim to reach its next step (`Member.awaitPaused`, at most 3 s), so the victim's blocks are in the
pool before the newcomer asks for its own.

### 25.2.7 Batched verification of drafts

With prompt-lookup or draft-model speculation (§10.9) a verification pass used to run alone: `forwardAll` took the device lock, the other
slots waited, and the weights were read for `k + 1` rows. Now `Member.forwardAll` queues a `VERIFY` step: the draft of every slot that has
one (up to 17 rows each, `engine.draft-tokens`) goes into the same pass as the other slots' tokens and chunks, with the logits of all its
rows. `Speculator` is unchanged: it sees the logits it asked for.

### 25.2.8 A client that leaves the queue

`Admission.acquire` waits in slices of 50 ms and polls the request's `gone` check (`web/chat/ClientGone`, the same check the generation uses).
When the client has closed its connection the request is removed from the queue and `Admission.Abandoned` ends the handler without a
response; before, the request kept its place, took a slot when its turn came and noticed only then. Unloading the model (`closing`)
wakes the queue the same way.

### 25.2.9 What "the same as alone" means

* A multi-token segment of a mixed pass is the dispatch sequence of `run(n)` for that many tokens: on Metal with the same kernels and
  chunking, so the logits agree with the lone run **to the rounding of the multi-row matmul kernels** (the kernel a matmul takes depends on the
  total rows of the pass: `n = 1` uses a vector kernel, `n ≥ 2 ` the tile kernels, a pass of 263 rows the MMA kernels).
* A decode row in a mixed pass takes the multi-row kernels where a lone step took the single-token ones, as in batched decoding (§10.9b). So
  logits are not bit-identical on the CPU float path and are within `1e-4` of the largest logit on Metal (`ArchSmoke` uses the bound of the
  batched-decode checks); the **arg max is equal** in every case tested, so greedy text is. With `-Djllm.cpu.deterministic=true` (every
  multi-token matmul runs token by token) the CPU is bit-identical: `MixedSchedulerTest` runs that way in `mvn test`.
* A chunk trimmed to the budget changes where chunks are cut (like `backend.batch`); not at the default settings (the chunk fits).
* A slot that is paused and resumed gets its full blocks back unchanged (the spill file is the canonical row layout); the tail keeps its rows.
  The last partly filled block of a finished request is not published, so the next turn of a conversation re-computes up to `B − 1 = 15` positions.

---

## 25.3 Why this design

* **Mixed passes, not a priority at the lock.** Ordering who gets the device next ([10 §10.14](10-engine-and-server.md), the
  `priority` request field) only chooses *which* pass runs next; a decode token still waits for a whole chunk. Putting the token
  into the chunk's pass removes the wait. The extra rows cost little because a chunk's pass is compute bound and the decode
  rows reuse the weights it has already read (25.5 measures it).
* **Riders first, the chunk takes the rest (Sarathi-style chunked prefill).** The alternative, decode tokens filling what the chunk
  leaves, would stall a conversation for as long as the prompt prefills. Riders cost one row each; the chunk is trimmed by those
  rows only when the budget is lowered by hand.
* **The engine keeps its thread per request.** Continuous batching engines run one loop that owns all sequences. Here every request's
  sampling, constraints, stop strings, jump-forward, speculation and streaming are a loop of `Engine.generateLocked`; turning it
  inside out would touch all of it and would put the risk of ten features into one change. So the *requests* still run their own
  loops, and meet only at the forward call, where a leader forms the pass. The price is thread hand-offs: a rider that just
  got its logits samples a token (microseconds) and comes back within the 2 ms window, a pass of 263 rows lasts 100 ms or more, so the
  cost is small against prefill passes and visible against decode-only passes (which therefore keep the slots path:
  `forwardBatch`, decode replay and the pipelined greedy step for a lone slot).
* **Chunks are not cut unless asked.** The engine's chunk is `backend.batch` rows; the backend's buffers hold `engine.mixed-reserve`
  more rows for the riders, so a full chunk plus the riders fit in one pass and the cut points of a prompt are the same as under
  `slots`. Cutting chunks to the leftover budget on every pass (Sarathi's dynamic chunk) would make a conversation's numbers depend on
  who else is running; it stays an opt-in (`engine.pass-budget`).
* **A slot is a place to run, not a cache.** With the paged cache any slot can run any conversation, so the old policy (pick the slot
  that holds the prefix, copy it from a donor slot, wait for the busy slot that has it) is replaced by *one queue*, and the cache is
  looked up in the prefix tree wherever the request lands. The finished conversation's blocks are given back
  (`Engine.releaseCache`) so idle slots do not eat the pool; its price is that the last, partly filled block is computed again on the
  next turn (at most 15 positions).
* **Cache hit plus waiting time as the score.** SGLang schedules by longest cached prefix; that starves a request whose prefix is not
  cached. Adding `2 000` tokens of score per second waited bounds the wait of any request to what a cache hit of that size is worth,
  and the priority classes on top (with aging) keep an interactive request ahead of a background one without ever starving it.
* **Pausing keeps the engine.** A paused conversation's generation loop is a Java stack; moving it to another slot would mean
  serializing sampler, constraint and speculator state. It keeps its slot and gives back what is expensive, the blocks (and its place
  in the passes). So preemption needs slots to spare (`engine.max-running` below `slots`), which is an explicit setting.
* **Two segments of one owner are refused, one pass is one set of owners.** Keeps each cache written by one segment per pass;
  the engine never has two requests on a slot anyway.

**Costs.** One more indirection per forward call (a queue and a thread hand-off, ~0.1 ms); a decode row in a pass of more than one row uses
the multi-row kernels (not bit-identical to a lone step, 25.2.9); with the CPU backend the activations of a mixed pass are copied
once per segment; the Metal pass is encoded without decode replay (a pass with a chunk is not a recorded program); finished
conversations are found again through the tree rather than in place.

---

## 25.4 Settings

| key | default | |
|---|---|---|
| `engine.scheduler` | `slots` | `slots`: chapter 10's scheduler (a slot chosen on arrival, batched decode steps). `mixed`: the one queue and mixed passes of this chapter; implies `backend.kv-paged=true`. Models it cannot serve keep `slots` |
| `engine.pass-budget` | `0` | most rows one pass carries; `0` = everything the buffers hold (`backend.batch + engine.mixed-reserve`). Lower = shorter passes and smoother streams, a prompt prefills in smaller parts |
| `engine.mixed-reserve` | `32` | rows the backend's buffers hold beyond `backend.batch`, for the riders of a full chunk (a verification run takes `k + 1`). Costs `32` rows of activations per slot |
| `engine.background-chunk-cap` | `0` | rows of a *background* chunk in a pass that also carries an interactive rider; `0` = no cap |
| `engine.age-promote-ms` | `10000` | waiting this long moves a request or a step one class up; `0` = no aging (a background request may wait for ever behind interactive ones) |
| `engine.max-running` | `0` | slots that take part in passes at once; `0` = all. Below `slots`: the others wait (queued with a slot, or paused) |
| `engine.preempt` | `true` | an interactive request that finds no running place pauses the background slot that started last; only with `engine.max-running < slots` |
| `engine.session-max-running` | `0` | most slots one session (the request's `user`) holds, running or paused; `0` = no limit |
| `slots` (console / `/api/models/load`) | `1` | 1 to 16 now (was 8); under `mixed` slots are cheap: a block table and some activation rows each, not a KV cache |
| request `priority`, header `X-Jllm-Priority` | `interactive` | `interactive` or `background` ([10 §10.14](10-engine-and-server.md)) |
| request `user` | none | the session for `engine.session-max-running` |

---
## 25.5 Measured

TBD-5

## 25.6 Checks and limits

Verified by:

* `SchedulerPolicyTest` (no model, in `mvn test`): the pass planner (the worked example of 25.2.3 with its 267 rows, a lowered budget
  trimming the chunk, a leftover too small for a chunk, classes, aging, the background cap, no stall), the queue (the slot with the
  prefix, the request with most cached tokens, an interactive request before an older background one, aging, a session limit,
  pausing the background slot and resuming it, no pausing for an interactive running slot, a request whose client left leaves within 55 ms)
  and `KvSequence.suspend` / `resume` (full blocks cached, tail kept, spilled by another conversation and reloaded; blocks lost without
  a store reported).
* `ArchSmoke` → `MixedPassChecks`, every attention family, CPU and the real GPU, f32 / f16 / q8 (plain and rotated keys) and, for MoE, with
  the experts of a layer on the CPU: three conversations' scripts (a 50-token prompt in chunks of 32 and 18, a 12-token prompt, a 9-token
  prompt followed by a 3-token verification run, then decode steps), alone and in mixed passes with the longest segment first and the
  leading backend rotating (once one that owns no segment): the arg max of every row must be equal and the logits within `1e-4` of the
  largest; the count of bit-identical segments is printed. Run with and without `-Djllm.metal.concurrent=true`.
* `MixedSchedulerTest` (Qwen3-0.6B, CPU, deterministic, in `mvn test`; on the GPU in `mac-validate.sh`): a 240-token prompt's chunks
  with two slots' decode tokens, every answer equal to the one alone; a generation paused in the middle and resumed gives the same
  text; prompt-lookup speculation with three slots, drafts verified in shared passes, the same text as alone.
* `jllm-web` `MixedSlotsTest` (fake backends whose rows are a hash of the whole prefix, so a wrong block changes the text): a pass
  carries a chunk and decode tokens and every conversation generates what it generates alone; an interactive request overtakes
  background ones; a background generation is paused for an interactive one, its blocks are spilled to the block store and
  reloaded, and the text is the same; a queued request whose client left is dropped.
* `SchedulerBench`: the workload of 25.5.

**Not done / limits.**

* Only grouped-query attention models with a paged f32 / f16 / q8 cache; Mamba, Qwen3-Next, DeepSeek's latent attention and the q4 cache
  keep `slots` (their steps cannot share one set of rows: a recurrent state, another cache layout).
* A pass is encoded without decode replay: a decode row in a pass with a chunk or a draft runs the batched-row dispatches, not a recorded
  step. Decode-only passes and a lone slot keep their old paths (replay, pipelining).
* One chunk per pass. Two prompts prefill alternately, not side by side.
* Tree speculation (verifying several candidate continuations from the n-gram index at once) is not done; each slot verifies one draft.
* Pausing needs a spare slot (`engine.max-running < slots`) and a slot whose cache is shareable (not a request with pictures or custom
  rotary positions); it takes effect at the victim's next step boundary (a step or a chunk), so an interactive request waits for that.
* Per-session limits count slots, not tokens; priority is two classes.
* A client that leaves is noticed when the queue polls it (50 ms); the generation itself stops as before, at its next token or chunk.
* The CPU backend's mixed pass copies each segment's rows between arrays; at Qwen3-0.6B sizes that is about a third of a percent.
