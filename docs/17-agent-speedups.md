# 17. Speedups for coding agents

A coding agent (opencode, Continue, aider, Cline) is a very particular customer. Its first request carries ~10 000
tokens of instructions and tool definitions; every later request repeats all of them plus the growing conversation;
it runs side requests (titles, summaries) next to the main one; and much of what it writes back is text that is
already in its context — lines of a file it read, the shape of earlier tool calls. It also gives up often: a user
presses Escape, a step times out and is retried. This chapter covers what jllm-serve does about that, each explained
simply first and then exactly:

| | what | where |
|---|---|---|
| [17.1](#171-prompt-lookup-tuned-for-agents) | guess the next tokens from the context and check them all at once | `engine/Speculator`, `engine/NgramIndex` |
| [17.2](#172-slots-that-share-a-prefix) | several conversations share an expensive prefix by copying it, not recomputing it | `web/models/LoadedModel`, `Engine.adoptPrefix`, `Backend.copyStateFrom` |
| [17.3](#173-saved-conversations) | conversations survive restarts on disk, saved in the background, restored in part | `web/models/KvStore`, `engine/CacheFile`, `Engine.beginSave`, `Backend.saveState` |
| [17.4](#174-pipelined-decoding-on-the-gpu) | the GPU runs the next step before the CPU has looked at the last one | `Backend.GreedyPipe`, `MetalBackend.Pipe` |
| [17.5](#175-tokenizing-only-what-is-new) | a resent conversation is tokenized only from where it is new | `tokenizer/PromptCache`, `Tokenizer.SequenceEncoder` |
| [17.6](#176-noticing-a-client-that-left) | a request nobody waits for stops within one prefill chunk or one token | `web/chat/ClientGone` |

---

## 17.1 Prompt lookup tuned for agents

🧸 When you copy a page from a book, you don't think about each word: you look at where you are, find the same words
on the page, and read off what comes next. Prompt lookup does that for the model: it finds the last few words it wrote
somewhere earlier in the conversation and proposes the words that followed them there. The model then checks the whole
guess in one go and keeps the part it agrees with.

📐 **Verification** is the speculative decoding of [10 §10.9](10-engine-and-server.md#109-speculative-decoding): feed the
current token plus a draft `d_1..d_m` in one pass, get the logits of all `m + 1` positions, accept the draft up to the
first disagreement. The output is exactly the non-speculative output (greedy) or has exactly its distribution
(sampling). With per-token acceptance probability `α`, a pass yields on average `(1 − α^(m+1)) / (1 − α)` tokens.

**The index.** Finding "the latest earlier occurrence of the text's last `n` tokens" by scanning is `O(context)` per
step, so the old implementation searched only the last 4 096 tokens; an agent's file contents are often further back.
`NgramIndex` is an open-addressing hash table from (`n`, the `n` tokens) to the position where that n-gram last ended,
for every `n` from `engine.ngram-min` (3) to `engine.ngram-max` (8):

```
key(n, t_{e-n+1} .. t_e) = mix(... mix(mix(φ·(n+1), t_{e-n+1}+1), ...), t_e+1)      mix(x, v) = ((x ⊕ v) · c₁) ⊕ ((x ⊕ v) · c₁ >>> 31)
```

with 64-bit multiplicative mixing (`φ` = 0x9E3779B97F4A7C15, the golden-ratio constant). A hit is verified against the
real tokens, so a hash collision costs a miss, never a wrong draft. Lookup tries `n = 8, 7, ..., 3` and takes the first
(longest) match: longer matches are more likely to continue the same way. Each token appended to the history adds the
n-grams that end **one token back**, so the suffix being looked up is never its own match; an agent's next request
repeats the previous one, so the index of the common prefix is kept, and rebuilt (milliseconds for tens of thousands of
tokens) only when the new request diverges inside what was indexed.

**Adaptive draft length.** A pass that verifies `m` drafted tokens costs more than a plain step, so `m` adapts:

```
whole draft accepted:   k ← min(K, max(k, 2m))          (grow while the text keeps copying)
otherwise:              k ← max(1, min(k, accepted + 1)) (fall back to what this text supports)
```

starting at 4 with `K = engine.draft-tokens = 16`, plus the existing back-off (after a pass that accepts nothing, skip
1, 3, 7, 15 steps). In copying stretches `k` reaches 16 within three passes; in free text drafts stay short or stop.

**Measured** (Qwen3-0.6B, CPU, 4-core AVX-512, greedy): rewriting an 80-line Java class from the prompt with one rename,
19.9 → 81.2 tok/s, 274 of 280 drafted tokens accepted; a free-form story, 24.4 → 23.3 tok/s (−4 %, few and short
drafts). Matches of 2 tokens cost 8 % on the story for no gain on the copy, hence the minimum of 3. MoE models keep it
off unless set ([11 §11.6](11-performance.md)): a verification pass reads the experts of every drafted position.

---

## 17.2 Slots that share a prefix

🧸 Two children ask the teacher for the same 100-page book, then each wants to write a different ending. The teacher
doesn't read the book twice: she photocopies her notes for the second child (fast) and each continues on their own copy.
And she gives the copy to the child who had nothing on their desk, not to the one in the middle of their own story.

📐 A model loaded with several **slots** has one engine (KV cache) per slot on the same weights. `LoadedModel.pick`
chooses where a request runs:

1. an idle slot whose **whole** cache the prompt continues (the next turn of its own conversation) runs it in place;
2. otherwise the request goes to the idle slot that matters least (empty first, then least recently used), and if
   another idle slot (the **donor**) holds a longer prefix of the prompt, at least 256 tokens longer, that prefix is
   **copied** into it first — unless the donor would lose less than 256 tokens by simply running the request itself;
3. when all slots are busy, the one holding the longest prefix is waited for.

The copy (`Engine.adoptPrefix` → `Backend.copyStateFrom`) moves the keys and values of positions `[0, n)` for every
layer: `n × 115 kB` for Qwen3-0.6B with an f16 cache, a memory copy at tens of GB/s. Both engines are locked in a fixed
order (no deadlock when two adopt from each other), and the copied rows are bit-identical, so the continuation is
exactly what the donor would have produced (`SmokeTest.persistCache` checks it).

**Measured** (CPU server, 2 slots): conversation A (1 666 tokens) prefills in 14.4 s; conversation B shares the first
1 256 tokens: copied in **34 ms**, only its own 558 tokens prefilled (3.9 s); A again: 1 665 of 1 666 tokens cached
(0.07 s), and B again likewise. Without the copy, B would have run in A's slot and A's next turn would have re-prefilled
everything after token 1 256.

*Why copy and not share memory pages?* Paging the KV cache (vLLM-style block tables) would let slots point at the same
physical rows, saving memory, but every attention kernel and every cache write on both backends would have to go
through a page table. On unified memory a copy of a long prefix costs milliseconds and leaves the hot kernels as they
are; the memory saving is the part this design gives up.

---

## 17.3 Saved conversations

🧸 Before going home, the teacher puts her notes in a drawer with a label saying which book they belong to. The next
morning, when a child asks about that book again, she takes the notes out instead of reading the book again.

📐 `Engine.saveCache(file, fingerprint)` writes `CacheFile`:

```
"JLLMKV01" | fingerprint (UTF) | state format (UTF) | n (int) | n token ids (int) | backend state
```

The **fingerprint** names the model and every load setting that changes its keys and values (source, quantization,
backend, cache type, int8 activations — not context size, batch or threads); the **state format** names the layout.
A file is loaded only under the same fingerprint and format. Both the CPU and the Metal backend write the same rows,
little-endian f32 or IEEE half (Metal's `half` has the same bits), so the layout is shared between backends:

| attention | state format | per layer, per position | bytes per token |
|---|---|---|---|
| standard (GQA / MHA) | `kv f16 layers=28 kvDim=1024 headDim=128` | the K row, then the V row | `2 × layers × kvDim × b` |
| DeepSeek latent (MLA) | `mla f16 layers=27 width=576` | the latent row `[c, k_pe]` (latent, rotary key) | `layers × (kv_lora_rank + rope) × b` |

(`b` = 2 for f16, 4 for f32.) The MLA row is exactly what the backends cache (`MlaCpu`, `MlaMetal`, see
[12](12-mixture-of-experts.md)): 62 kB per token for DeepSeek-V2-Lite in f32 (31 kB in f16) instead of the 553 kB of expanded keys
and values, so a saved DeepSeek conversation is small. Saved MLA caches, prefix copies between slots (17.2) and
partial loads work like the standard ones; `KvStateTest` checks on tiny standard and MLA checkpoints, f32 and f16,
that a saved + loaded or copied state continues with bit-identical logits, that a partial load holds the first rows
bit for bit, and (with `--metal`) that a CPU-written file loads into the Metal backend.

`jllm-web`'s `KvStore`:

- **saves** a slot after a request whose prefill took ≥ `jllm.web.kv-cache-save-after-ms` (2 000), before a restore
  overwrites it, and on unload/shutdown, when it holds ≥ `jllm.web.kv-cache-min-tokens` (512) tokens; a cache extending
  a saved one replaces it, one already covered is not written, and after a request a cache that a saved file covers
  up to `c` tokens is saved again only once it holds `≥ max(min-tokens, c / 4)` new ones;
- **restores** before a request the saved cache with the longest common prefix with the prompt, if that beats what the
  slot holds by ≥ 256 tokens, loading only that prefix;
- keeps the files under `jllm.web.kv-cache-max-gb` (10), deleting the least recently used.

### Saving in the background

🧸 The teacher no longer copies her notes into the drawer while the class waits. She writes on a sticky note "pages 1
to 40 are being copied, don't erase them" and goes on teaching: as long as the lesson only adds pages, nobody waits.
Only if she has to erase one of those pages does she wait for the copy to finish.

📐 A save used to hold the engine for the whole write (a GB at disk speed). Now `Engine.beginSave(file, fingerprint)`
holds the engine only to copy the cached token ids, sets a **pin** `[0, n)` and returns a `PendingSave` whose
`write()` (a virtual thread of `KvStore`) writes the header and the rows with `Backend.saveStateWhileServing` and then
drops the pin. Meanwhile requests run; the engine checks the pin where a request starts overwriting the cache:

```
reuse = common prefix of the cached ids and the prompt       (17.2, 10.2)
if pin > reuse: wait until the save has written its rows     (only then are rows < pin rewritten)
```

The next turn of an agent's conversation extends the cached tokens (`reuse ≥ n`) and never waits; a request that
diverges inside the saved part, a restore and a prefix copy into that slot wait (as every request waited before).
The rows of `[0, n)` are not written while pinned, so the file holds exactly the state of the moment the save began,
whatever ran in between (`KvStateTest` serves a request between `beginSave` and `write` and compares the file with
the rows taken before). Reading rows `< n` while a request writes rows `≥ n` is safe on both backends: the CPU cache
is Java arrays (a writer of an f16 pair rewrites the other half with the value it holds, a reader only reads), the
Metal cache is shared memory the GPU writes elsewhere; `saveStateWhileServing` touches neither the command queue nor
the greedy pipe. One save per engine at a time; `Engine.close` waits for a running one.

Because a save now costs the next request nothing, the thresholds were lowered: 2 s of prefill (10 s before) and 512
tokens (1 024): a 2 s prefill is ~250 tokens of Qwen3-0.6B on a 4-core CPU (2 707 tokens took 21 s, below);
anything that took that long is worth a file. The growth rule keeps a long conversation from being rewritten every
turn: with 20 000 tokens saved, the next file is written at 25 000.

### Restoring part of a file

🧸 If today's lesson starts like yesterday's but changes after page 30, she takes only the first 30 pages of
yesterday's notes out of the drawer.

📐 The best file is the one whose token list shares the longest prefix with the prompt. `KvStore` keeps each model's
lists in a `TreeMap` ordered lexicographically: for any sorted list `s_1 < ... < s_m` and a query `q` inserted between
`s_i` and `s_{i+1}`, the common prefix `lcp(q, s_j)` only shrinks as `j` moves away from the insertion point (two lists
that both agree with `q` on `k` tokens bracket everything in between, which then agrees on `k` too), so the best match is
`floor(q)` or `higher(q)`: two comparisons instead of one per file (`KvStoreTest` checks it against a linear scan).

It then loads only the shared prefix `p = min(lcp, prompt − 1)`: `Engine.loadCache(file, fp, p)` calls
`Backend.loadState(p, stored, channel)`, which reads `p` rows per layer and skips the other `stored − p` (a seek on a
file, since the rows are stored layer by layer). Rows beyond `p` would be overwritten by the prefill anyway; now they
are not read, and a file longer than the context can still serve a prompt that shares less than the context with it.

Size: `n × 2 × layers × kvDim × bytes` — 115 kB per token for Qwen3-0.6B f16, ~1.2 GB for opencode's 10k-token prompt.
**Measured:** a 2 707-token prompt prefilled in 21.3 s; after a restart the cache was restored in 0.9 s and the same
prompt was served with 2 706 tokens cached. On the Metal backend the file is read straight into the cache buffers
(unified memory: they are the GPU's memory), with no intermediate copy.

**Measured** (Qwen3-0.6B Q4_0, f16 cache, CPU, a 2 003-token conversation = 230 MB, on a 4-core machine shared with
other jobs): a blocking save held the engine 3.1 s; `beginSave` held it 0.3 ms, the rows were written in 7.3 s while
the next turn (20 new tokens, 2 003 reused) ran in 4.6 s and finished first; both files were byte-identical.
Restoring all 2 003 positions took 1.1 s, the first 500 (a prompt sharing a quarter) 0.22 s.

---

## 17.4 Pipelined decoding on the GPU

🧸 A relay race where the runner has to stop, show the baton to the coach, and wait for a nod before the next runner
may start, is slow. If the next runner can just take the baton and go, while the coach watches from the side, the race
never stops. Here the GPU is the runners and the CPU is the coach.

📐 A plain decode step on the Metal backend is: CPU writes the token's embedding → GPU runs ~170 kernels →
CPU waits, copies 600 KB of logits, picks the argmax, and only then encodes the next step. The GPU idles during the
CPU part. For greedy requests without penalties (`Sampler.greedyPlain`: the next token is exactly the argmax), the
engine uses a `Backend.GreedyPipe` instead:

```
step j:  embed_token(ring[j mod 4]) → 28 layers → final norm → LM head → argmax → ring[(j+1) mod 4]
```

The token of step `j + 1` is read **on the GPU** from the slot step `j` wrote, so the engine queues step `j + 1` right
after step `j`, before it has seen step `j`'s token; it reads the ids one step behind (for streaming and stop checks)
with `jm_wait_pending(keep)`, which waits only until at most `keep` submitted command buffers are unfinished. Timeline:

```
GPU:  [ step j ][ step j+1 ][ step j+2 ] ...            (no gaps)
CPU:     encode j+1 | read token j | encode j+2 | read token j+1 | ...
```

The argmax kernel reduces in a 256-thread threadgroup with the rule "larger value, or equal value and lower index",
which reproduces the CPU sampler's first-maximum argmax exactly, so pipelined output equals unpipelined output
(ArchSmoke checks every family; `EnginePipeTest` checks whole generations and follow-up requests against the CPU
backend). When prompt lookup has a draft, the pipe is drained first (the in-flight step's result is dropped; the
verification pass recomputes that position). Steps queued past the end of the answer write cache rows that the next
request overwrites. `-Djllm.metal.pipeline=false` turns it off; a native library built before `jm_wait_pending` existed
simply runs unpipelined.

*Expected gain:* the CPU share of a step — logits copy, argmax over 151 936 floats, encoding ~170 dispatches through
FFM — is on the order of a millisecond, out of ~9 ms per token for Qwen3-0.6B on an M1 Pro ([08 §8.7](08-metal-backend.md));
hiding it is worth roughly 10 %, more for small models and fast GPUs. Not measured on Apple hardware yet: run `bench`
with `-Djllm.metal.pipeline=false` and without to see it on yours.

---

## 17.5 Tokenizing only what is new

🧸 Every time the agent asks something, it hands over the whole notebook again, with one new page at the end. Reading
the notebook from the first page each time is silly: the teacher remembers where she stopped reading last time,
checks that the pages before it are the same ones, and reads only the new page.

📐 Every request re-renders the chat template over the whole conversation and turns ~100 000 characters into ids:
BPE on a CPU, tens of milliseconds per request, all of it work done the turn before. `PromptCache` (one per model,
`engine.prompt-cache-mb`, 64 MB, 0 = off) keeps the ids of recently rendered prompts and gives exactly the ids of a
full render.

**Where encoding can be resumed.** The template renders a prompt as a list of *pieces*: control strings (`<|im_start|>user\n`)
and message texts. The pieces are not tokenized one by one: the `SequenceEncoder` joins the ordinary text between two
added tokens into one run, so that pre-tokenization and BPE merges see the text as one string (a word split between a
control string and a message would otherwise tokenize differently). The places where the encoding of the rest does not
depend on the past are right **after an added token**: the run before it has been flushed, nothing is pending, and
what follows starts a new run. The encoder reports these *boundaries* `(offset in the piece, ids so far)`, except
after a token that swallows the whitespace after it (`rstrip`, Phi's `<|end|>`): its effect reaches into the next
piece. An `lstrip` token only trims the run before it, which starts after the boundary, so it is no exception.

**The key.** For a piece boundary `k` (after pieces `0..k−1`), the cache stores under a hash of those pieces the last
encoder boundary inside them:

```
key_k = H(scope, kind_0, len_0, text_0, ..., kind_{k−1}, len_{k−1}, text_{k−1})    two 64-bit multiply-rotate lanes
value = (entry, piece j < k, offset e in piece j, t = ids before it)
```

The scope is the template family and its protected tokens; a different tokenizer is a different cache. Hashing all of
pieces `0..k−1` is enough although the scan inside a piece looks ahead for the longest added token: a boundary inside
piece `j < k` was found by a scan that never left piece `j`, and piece `j` is hashed whole. Two different prompts
collide with probability about `2⁻¹²⁸` (two lanes, MurmurHash3's finalizer on each).

**A request** renders its pieces, computes `key_1..key_n` (one pass over the characters) and looks them up from `k = n`
down. On a hit it copies the entry's first `t` ids, resumes the encoder (`Tokenizer.resumeSequence`) and encodes
piece `j` from offset `e`, then the remaining pieces. The new prompt's own boundaries are registered for the next
request. With an agent, request `r + 1` repeats request `r` up to its generation prompt, so the hit is the end of the
last message of request `r`, and only the new assistant answer, tool results and user message are encoded.

**Memory.** An entry costs `4 bytes × ids + ~96 bytes × keys`; entries are evicted least recently used first; one
larger than a quarter of the budget is not kept. A key that a newer prompt takes over (same prefix) moves to it, and
an entry left without keys is dropped. Lookups and updates hold the cache's lock for a hash-map probe per piece;
the encoding runs outside it.

`PromptCacheTest` renders ~2 900 requests of growing, edited and interleaved conversations through the cache, for
every template family, with a cache small enough to evict and with `lstrip` / `rstrip` added tokens, and requires the
ids of a full render each time. **Measured** (Qwen3 tokenizer, a 50-turn conversation of 100 258 characters, 31 308
ids, the 50th request after the 49th; best of 15 on a shared 4-core machine): render + tokenize 15.6 ms in full,
0.50 ms with the 49th request cached (31×); under heavier load on the same machine, 60–90 ms against about 2 ms. The
render itself (string building) is ~0.3 ms; what is left is hashing the 100k characters and encoding the ~1 800
new ones.

---

## 17.6 Noticing a client that left

🧸 If the child who asked a question walks out of the room, the teacher stops answering — she doesn't finish a
ten-minute explanation to an empty chair while the next child waits.

📐 The engine polls a `cancelled` supplier between prefill chunks and before every token
([10 §10.4](10-engine-and-server.md#104-cancellation-the-lock-statistics)). The engine's own HTTP server knows when a
connection closes (a probe thread blocks reading the socket). In the web console (Spring on Tomcat) a servlet has no
such signal: a non-streaming request was only cancelled by an unload or the request timeout, so an agent that gave up
kept its slot for the whole answer, and a stream noticed only when a write failed, every 10–15 s during a prefill
(the keep-alive comments).

`ClientGone.of(request)` returns a check that asks Tomcat's connection for available input without blocking
(`ActionCode.AVAILABLE` with a read: the non-blocking read `ServletInputStream.isReady` does under a read listener).
After the request body has been read, the client sends nothing until its next request; a closed connection makes the
read hit the end of the stream, which Tomcat reports as available input (so that the next read throws). The
controllers OR the check into `cancelled` for `/v1/chat/completions`, `/v1/completions` (streaming or not) and the
console's `/api/chat`; it costs one system call and is made at most every 50 ms. After a disconnect nothing more is
written, the slot is released and the log line ends with `cancelled: the client went away`. A prefill stopped half-way
stays cached, so a retry of the same prompt continues where it stopped. Limits: a pipelined next request on the same
connection would look the same (no HTTP client pipelines POSTs); under another servlet container, HTTP/2 or MockMvc
the check is never true and only failed writes stop a stream.

`CancellationTest` runs Tomcat on a random port with a fake backend that sleeps per forward call, sends a request over
a raw socket, closes it during the prefill or during decoding, streaming or not, and requires the slot to be free
within about one forward call and no token to be fed afterwards. **Measured** (one forward call = 20 ms in decoding,
150 ms per 64-token prefill chunk): non-streaming, the slot was free 27 ms after the client left during
decoding (19 of 2 000 tokens generated) and 156 ms after it left half-way through a 3 001-token prefill; streaming,
25 ms and 166 ms. Before, the non-streaming request would have run to its 2 000 tokens (40 s) and the stream until
the next keep-alive comment failed.
