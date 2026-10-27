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
| [17.7](#177-constrained-tool-calls-and-jump-forward) | tool calls that are always valid, their fixed parts written without sampling | `sampling/ToolCallConstraint`, `ByteConstraint`, `TokenTrie`, `web/tools/ToolCallFormat` |

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
drafts). Matches of 2 tokens cost 8 % on the story for no gain on the copy, hence the minimum of 3. MoE models kept it
off at first ([11 §11.6](11-performance.md#116-speculative-decoding)), because a verification pass reads the experts of every
drafted position; since prompt lookup pauses when its drafts do not pay, it is on for them too ([11 §11.11](11-performance.md#1111-agent-defaults-measured):
the 30B gains 33 – 68 % on a code edit and loses 1 – 10 % on tool calls and prose).

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

**Update:** that is exactly what [chapter 24](24-paged-kv-cache.md) now does behind `backend.kv-paged=true`: slots share the
physical blocks of one pool through a prefix tree (copy on write), so a second conversation with the same system prompt costs a table
write, not a copy, and the prefix exists once (8 conversations sharing 4 096 tokens: 1.1 GB instead of 7.7 GB).

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
## 17.7 Constrained tool calls and jump-forward

🧸 Filling in a form is easier than writing a letter: the boxes and their labels are printed already, you only write
the answers, and the form will not let you write a phone number where the date goes. A tool call is such a form —
`<tool_call>`, `{"name": "`, one of the tool names, `", "arguments": {`, the fields — so jllm-serve prints the fixed
parts itself, all at once, and only lets the model write answers that fit their boxes. The model cannot produce a
broken call, and it spends no time writing the labels.

📐 Coding agents live on tool calls, and a small model's most common failure is a malformed one: a missing brace, a
misspelled tool, a string where the schema wants an integer, a key the tool does not have. Each costs a round trip
(the agent reports the error and the model tries again). Four mechanisms remove that failure and make the calls
cheaper to produce.

**The constraint** (`sampling/ToolCallConstraint`). A byte-level automaton over the whole output:

```
free text ── "<tool_call>" ──▶ "\n{\"name\": \"" NAME "\", \"arguments\": " ARGS "}\n</tool_call>" ──▶ free text …
    │                                   │                                      │
    └─ "<think>" … "</think>" (free)     one of the tools' names               JSON object matching that tool's
                                         (a trie over the names)                parameters (JsonConstraint, canonical)
```

Free text is not constrained (`TokenConstraint.relaxed()`: the sampler draws as if unconstrained and only falls back
to the masked distribution when the drawn token would start a malformed call, so free text is token for token the
unconstrained text). The constraint switches on when the bytes `<tool_call>` complete — whether the model wrote
Qwen's single added token or spelled it out — and off after `</tool_call>`; text inside a `<think>` block is free (a
call written while thinking is not a call, as `ToolCallFormat.parse` agrees). Inside a call, the envelope is fixed
text in exactly the layout the Qwen templates render calls in (`{"name": …, "arguments": …}`, Python `json.dumps`
spacing), the name is checked against a trie of the tools' (JSON-escaped) names, and the arguments run through the
`response_format` automaton ([10 §10.6](10-engine-and-server.md#more-than-one-answer-probabilities-and-structured-output))
in **canonical** whitespace mode: exactly one space after every `:` and `,` between tokens, none elsewhere. Earlier
calls in the conversation are rendered the same way (`ToolCallFormat.canonical`), so the model sees one layout.

The parameter schemas are compiled **leniently** (`Schema.compile(schema, true)`): agents send schemas they did not
write for us, full of `pattern`, `minimum`, `format`, `allOf`, `$schema`. What the automaton can enforce is enforced —
`type`, `properties`, `required`, `additionalProperties`, `enum`/`const`, `items`, `anyOf`/`oneOf` of different
JSON types, string/array lengths, local `$ref` — and any other keyword, a tuple `items`, a non-local `$ref` or an
unknown type allows anything; `anyOf` alternatives of the same JSON type (two object shapes, say) are merged into
"any value of that type". The constraint is then looser than the schema, never stricter, so a valid call is never
refused. Arguments are always an object.

| `tool_choice` | constraint |
|---|---|
| `"auto"` (default) | calls optional; free text free; the end of the turn allowed outside a call |
| `"required"` | the reply must start with `<tool_call>` (after a `<think>` block if Qwen3 thinks); the turn may end only after a call |
| `{"function": {"name": "f"}}` | as required, and only `f` (an unknown name is a 400) |
| `"none"` | no tools, no constraint |
| with `response_format` | the reply is calls, or the JSON answer matching the format (`"tools"` + `"response_format"` was a 400 before) |

**Jump-forward** (`TokenConstraint.jump`, `ByteConstraint.forcedBytes`). Whenever the automaton allows exactly one
next byte (and the output may not end), that byte is forced; the run of forced bytes is the *forced text*. After
`<tool_call>` it is `\n{"name": "`; once the name prefix is unambiguous (`get_w` → `eather`), the rest of the name
plus `", "arguments": {`; inside the arguments, `"` after `{` when a key is required, the rest of a key once only one
unused key of a closed object fits, `: ` after a key, the rest of an enum value, of `true`/`false`/`null`; after the
arguments, `}\n</tool_call>`. Computing it: try the 256 byte values on copies of the state and stop at the second
accepted one; repeat. The engine then appends the forced tokens without sampling and runs them, together with the
token just sampled, through the model in **one** `forward` call — a prefill chunk — instead of one pass per token:

```
sample tok → constraint.accept(tok) → forced = constraint.jump(tokenizer, room)
emit tok, forced[0..k)                     (stop strings, max_tokens, logprob bookkeeping as for sampled tokens)
backend.forward([tok, forced...], pos)     one pass, the logits after the last forced token
```

*Tokenization.* The forced text is encoded with the model's own tokenizer as ordinary text (`encode(text, false)`:
added tokens such as `</tool_call>` still match, special ones cannot appear), i.e. the way the model saw such text in
training, and each token is checked to spell the forced bytes and to be accepted by the constraint. The forced text
is first cut back to whole UTF-8 characters. The **last** token is held back unless no vocabulary token extends it
(`TokenTrie.extendable`): at the boundary between forced and free text the model's natural tokenization may merge
across it — after the forced `": "` the model would rather write ` "Paris` than `"` + `Paris` — so the model picks
that token itself, from the constrained choices (which include the held-back token, longer tokens that continue the
forced text into the free part, and shorter ones). An added token like `</tool_call>` is never extended, so the
closing of a call is committed whole. Example on the Qwen3 vocabulary: after `<tool_call>`, the forced head
`\n{"name": "` becomes `\n` `{"` `name` `":` (4 tokens in one pass) and ` "` is left to the model; after the
arguments, `}\n</tool_call>` is `}\n` `</tool_call>`, both committed. The output always parses: it is accepted by the
same automaton that defines a valid call. Jump-forward is off with `logprobs` (forced tokens have no distribution),
and `engine.jump-forward=false` (`Engine.setJumpForward`) turns it off.

**Speculation under the constraint** ([10 §10.9](10-engine-and-server.md#109-speculative-decoding)). Previously a JSON
constraint switched speculation off. Now prompt-lookup or draft-model drafts are checked against copies of the
constraint and cut at the first disallowed token (tool calls repeat the shapes of earlier calls, so prompt lookup
drafts them well), and each verification row is decided by the constrained sampling rule against the constraint as
it stands at that row. Greedy output is exactly the plain constrained greedy output: `ToolCallTest` checks it on a
scripted fake model (whose logits depend on the whole cached history) and on Qwen3-0.6B with deterministic CPU
kernels, with and without jump-forward.

**Cheaper masks** (`ByteConstraint`, `TokenTrie`). Per step the old path took the 256 most likely tokens (a heap
pass over 151 936 logits) and tested them; when none was allowed it tested the whole vocabulary token by token. Now:

1. *argmax first* (greedy): the model's own choice is checked first; usually it is allowed and nothing else runs;
2. *trie mask*: when a search goes beyond 300 candidates, the allowed set is computed once by walking a byte trie of
   the vocabulary alongside the automaton (shared prefixes tested once, a refused byte prunes its subtree; ~780
   nodes visited per structural state on the Qwen3 vocabulary, instead of 151 669 tokens fed one by one), and later
   checks are bit tests;
3. *memoized masks*: a mask is cached under the automaton state's key (the frame stack with schemas by identity), per
   schema / tool set, and the compiled tool sets are kept across requests (`ToolCallFormat`: agents resend the same
   tools every time), so a recurring state — the start of a tool's arguments, the position after a value — costs a
   lookup;
4. *small allowed lists*: where the allowed set is small (≤ 2 048 tokens: structural positions, keys of a closed
   object, enum values, the envelope) and the walk stays under 6 000 nodes, the constraint hands the sampler the list
   (`TokenConstraint.allowedTokens`), which picks among those tokens without any pass over the vocabulary. The walk is
   done at once when the most likely tokens contain nothing allowed, otherwise the second time a state is met (the
   memo marks it the first time), so one-off states never pay for it; string contents (most of the vocabulary
   allowed) are never walked.

### 17.7.1 A worked trace, and why it is built this way

**A tool call, pass by pass.** The model is asked for `get_weather` and writes the call below. This trace was replayed through the real
`ToolCallConstraint` with the Qwen3 tokenizer (the model's side is simulated: it "writes" the text, whatever jump-forward forces is not sampled);
`S` is a token the model chose, `F` a token the constraint forced and the engine appended without sampling. Each line is **one forward pass**
(the sampled token and the forced ones after it go through the model together):

```
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris", "days": 3, "unit": "celsius"}}
</tool_call>

pass  1: S[<tool_call>] F[\n] F[{"] F[name] F[":]                         the envelope head; ` "` is left to the model (4 forced tokens)
pass  2: S[ "]                                                            held-back token: the model's natural token, merged across the boundary
pass  3: S[get] F[_weather] F[",] F[ "] F[arguments] F[":] F[ {"]         the name is unambiguous after "get" (the other tool is read_file), then the envelope
pass  4: S[city] F[":]                                                    a key: only one unused key of a closed object fits after the first letters
pass  5: S[ "]    pass 6: S[Paris]    pass 7: S[",]    pass 8: S[ "]      values: the model's own choices; free text inside a string
pass  9: S[days] F[":]    pass 10: S[ ]    pass 11: S[3]                    an integer: digits only, no quote, no point
pass 12: S[,] F[ "] F[unit] F[":]                                         only `unit` is left, so the whole key is forced
pass 13: S[ "]
pass 14: S[c] F[elsius] F["}}\n] F[</tool_call>]                           an enum with one value starting with "c", then the closing: committed whole
tokens 32, sampled 14, forced 18, passes 14
```

That is the 32 tokens and 18 forced of the measurement below (decode 254 ms -> 149 ms on Metal): 18 of the 32 passes disappear, while the sampled
tokens are exactly the ones that carry information. Without jump-forward every one of the 32 tokens takes a pass, and a model that wrote
`"days":3` instead of `"days": 3`, or `Paris` for an integer, would have produced a call that does not parse.

**What is forced and what is not, around passes 11 and 12.** After `"days": 3` the next byte may be another digit (an integer continues; `maximum` is one of the
keywords the lenient compile ignores), `,` or `}` (`city` is present and `unit` is optional): three kinds of byte, nothing forced, so the model chooses (`,`). After the comma the object is
closed (`additionalProperties: false`) and `city`, `days` are used, so exactly one key fits and the whole ` "unit": "` follows without a sampling step. Greedy decoding checks the model's
top token first (usually allowed: 58 µs, the same as unconstrained greedy); only when the model's favourite is refused does the trie walk (about 780 nodes on the Qwen3 vocabulary) or a memoized mask run.

### Why this design

* **A byte automaton, not a token grammar.** Tokenizers cut text differently for every model; an automaton over bytes plus a byte trie of the vocabulary (`TokenTrie`) works
  for any tokenizer, and the same automaton decides what the forced text is, which tokens may be sampled, and whether a draft token is acceptable.
* **Fixed, canonical layout.** The envelope and the whitespace inside the arguments (`: ` and `, ` exactly) are fixed so that most positions have exactly one allowed byte: that is what jump-forward
  needs. Free whitespace would allow `,` followed by any number of spaces and force almost nothing. Earlier calls in the conversation are rendered in the same layout, so the model sees one format.
* **Free text stays free.** Outside a call the sampler draws as if unconstrained (`TokenConstraint.relaxed`), so a constrained request produces the same prose as an unconstrained one,
  token for token; the constraint switches on at `<tool_call>` and off at `</tool_call>`, and `<think>` blocks are free.
* **Lenient schemas.** Agents send schemas written for other servers; the constraint enforces what an automaton can (`type`, `required`, `additionalProperties`, `enum`, `items`, lengths, local `$ref`)
  and allows anything for the rest, so it is looser than the schema, never stricter: a valid call is never refused. The price is that a call can still violate a `pattern` or a `minimum`.
* **The last forced token is held back** when the vocabulary has a token that extends it, because at the boundary between forced and free text the model's natural tokenization merges across it
  (` "Paris` rather than `"` + `Paris`); forcing the short token would change the distribution of the free text.
* **Rejected: parse afterwards.** The old path (`engine.constrained-tools=false`) lets the model write free text and parses it: a missing brace or a wrong tool name costs a round trip, which for a small model is the most common failure.
  **Rejected: switch speculation off under a constraint** (what the first version did): tool calls repeat the shapes of earlier calls, so prompt lookup drafts them well; drafts are checked against copies of the constraint and cut at the first token it refuses.
* **Costs:** the device-side greedy pipeline ([17.4](#174-pipelined-decoding-on-the-gpu)) is off for constrained requests; `logprobs` turn jump-forward off; the layout is the Hermes / Qwen one (other families are served through the same prompt);
  the console's `/api/chat` is not constrained yet.


### 17.7.2 Measured, checks, limits and switches

**Measured.** Apple M1 Pro (10 cores, 32 GB), JDK 26, under the machine-wide benchmark lock (no other job ran), the
Qwen3-0.6B vocabulary (151 669 tokens).

*Per-token cost of the mask* (`ConstraintBench`: a teacher-forced 68-token tool-argument object with a closed schema
and synthetic Gaussian-noise logits; microseconds per sampling call, best of 20 rounds; "warm" = the rounds share the
mask memo, as the requests of an agent that sends the same tools every time do). Three code states, built from the
revision history and measured one after the other in one lock hold: *before* is `96a221a`, the sampler before any of
the constrained tool-call work (the 256 most likely tokens tested one by one, then the whole vocabulary); *trie masks*
is `aebeb6e`, the first tool-call commit (argmax first, the trie walk, memoized bit masks); *now* adds the small
allowed lists and the memo of walks that were not worth doing. *Masks off* is the current code with
`engine.constraint-masks=false`.

| logits | sampling | before | trie masks | now, cold | now, warm | masks off |
|---|---|---|---|---|---|---|
| model agrees (reference token on top) | greedy | 192 | 59 | 60 | 58 | 58 |
| | T = 0.7 | 227 | 257 | 241 | 176 | 237 |
| distracted (40 other tokens above it) | greedy | 190 | 149 | 152 | **85** | 156 |
| | T = 0.7 | 222 | 230 | 317 | 176 | 336 |
| disagrees (nothing allowed in the top 256 at structural positions) | greedy | 515 | 343 | **171** | **89** | 438 |
| | T = 0.7 | 702 | 545 | **268** | **178** | 1 818 |
| unconstrained (the floor: the same sampling without a constraint) | greedy | | | 57 | | |
| | T = 0.7 | | | 1 967 | | |

Commands (from the repository root with `J="$JAVA_HOME/bin/java --add-modules jdk.incubator.vector"`; `base-A` and
`base-B` are checkouts of `96a221a` and `aebeb6e` that got the same `ConstraintBench.java`, adapted to their API, and
were built with `mvn -q -o test-compile`):

```sh
$J -cp base-A/target/classes:base-A/target/test-classes org.robomiri.jllm.ConstraintBench models/Qwen_Qwen3-0.6B 20   # before
$J -Dwarm=true -cp base-B/target/classes:base-B/target/test-classes org.robomiri.jllm.ConstraintBench models/Qwen_Qwen3-0.6B 20
$J -cp target/classes:target/test-classes org.robomiri.jllm.ConstraintBench models/Qwen_Qwen3-0.6B 20                  # now, cold
$J -Dwarm=true -cp target/classes:target/test-classes org.robomiri.jllm.ConstraintBench models/Qwen_Qwen3-0.6B 20     # now, warm
$J -Djllm.constraint.masks=false -cp target/classes:target/test-classes org.robomiri.jllm.ConstraintBench models/Qwen_Qwen3-0.6B 20
```

(`scripts/mac-validate.sh` repeats the last three as its "token mask" rows.) What the table says. Greedy decoding with
a model that agrees with the constraint costs one pass over the logits (the argmax, 58 µs, the same as unconstrained
greedy) however the mask is computed; the masks matter when the model's favourite tokens are not allowed. There the
worst case (a full-vocabulary search at a structural position) went from 515 µs (greedy) and 702 µs (T = 0.7) to 171
and 268 µs cold and 89 and 178 µs warm; with the masks switched off the current code needs 438 and 1 818 µs (the
second is slower than the first implementation's 702 µs, because the automaton's state grew and a copy of it per
tested token costs more). Cold sampled decoding when the model's choice is allowed, or nearly so, got a little slower
than with the trie masks alone (317 against 230 µs for "distracted"): the second visit of a state pays for its walk;
warm, every row is faster than before. Sampled decoding under a constraint is far cheaper than the unconstrained draw
(1 967 µs: a softmax and a sort over the whole vocabulary) because only the 256 most likely tokens are looked at. All
of it is small next to a forward pass of Qwen3-0.6B (about 9 ms per token on this Mac's GPU at the default
quantization, 25–40 ms on the CPU).

`mvn install` runs the same class in a quick mode (`ConstraintBench <model> --quick`: two rounds, a few seconds) that
checks the answers instead of timing them: each sampled token must be allowed, each greedy token must be the most
likely allowed token as found by a brute-force pass over the whole vocabulary (feeding every token's bytes to the
automaton, independent of the masks), the reference text must complete the constraint and a mask must cost a few
hundred trie nodes, not the vocabulary.

*Jump-forward and speculation under the constraint* (`scripts/tool-call-check.sh`, i.e. `ToolCallCheck`: Qwen3-0.6B
at the default q4 quantization, a short prompt that describes two tools, greedy, `tool_choice` auto or `"required"`;
the same machine and lock as above; median decode time of 5 generations on Metal and 3 on the CPU after one untimed
warm-up). The first call is the 32-token
`<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "days": 3, "unit": "celsius"}}\n</tool_call>`, the
second the 43-token `read_file` call with a list of line numbers. "Drafts" are accepted / proposed.

| decode ms (tokens) | Metal, call 1 | Metal, call 2 | CPU, call 1 | CPU, call 2 |
|---|---|---|---|---|
| every token sampled (`engine.jump-forward=false`, speculation off) | 254 (32) | 349 (43) | 295 | 399 |
| + jump-forward | **149** (18 forced) | **288** (10 forced) | **222** | **358** |
| prompt-lookup speculation, no jump-forward | 228 (8/9 drafts) | 260 (25/44) | 280 | 455 |
| prompt lookup + jump-forward | 152 (14 forced, 4/4) | 260 (10 forced, 12/26) | 233 | 419 |
| the model drafting for itself, no jump-forward | 336 (28/28) | 483 (38/38) | 488 | 675 |
| the model drafting for itself + jump-forward | 301 (9 forced, 17/17) | 433 (6 forced, 31/31) | 437 | 632 |

(Metal: ~8 ms per plain constrained step, 113 tok/s unconstrained greedy with the pipelined step the constraint
turns off; CPU: native int8 kernels on 10 threads with `-Djllm.cpu.deterministic=true`, which is slower than the
default kernels.) Jump-forward removes 18 of the 32 passes of the first call (the head, the name once unambiguous,
the key and value openings, the closing): 1.7× on Metal, 1.3× on the CPU, where a pass of several tokens costs
more relative to a single token than on the GPU. Prompt lookup helps when the call repeats text from the prompt (the
second call, whose prompt contains a similar `read_file` call: 349 → 260 ms) and adds nothing on top of jump-forward
when jump-forward already wrote what it would have drafted (call 1: 152 against 149 ms). A *self*-draft, which costs
as much as the model it drafts for, is slower than plain steps here; it is in the table because every draft is
accepted, which is the case that exercises long accepted runs, not because it is a good draft (a real draft is
smaller: [10 §10.9](10-engine-and-server.md#109-speculative-decoding), [08 §8.25](08-metal-backend.md)).

**Speculation does not change the constrained output.** The greedy tokens are identical with speculation off, with
prompt lookup and with draft models, with and without jump-forward and for `auto` and `required`, on Metal and on the
CPU. Every row of `ToolCallCheck` compares the tokens, not just the text, with the speculation-off run, and the script
fails on any difference:

```sh
scripts/tool-call-check.sh            # CPU (deterministic kernels, int8 activations fixed) and Metal; add cpu or metal to pick one
scripts/tool-call-check.sh metal --check-rounds 5 --check-max 48       # the same flags as java ... ToolCallCheck, then jllm's
```

or by hand, for the CPU,
`java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Djllm.cpu.deterministic=true
-Djllm.engine.adaptiveDraft=false -cp target/classes:target/test-classes org.robomiri.jllm.ToolCallCheck --backend cpu
--ctx 1024 --backend.cpu.int8-activations true --check-draft Qwen/Qwen3-0.6B,hf-internal-testing/tiny-random-Qwen3MoeForCausalLM`.
Two drafts are used, since they exercise different parts of the interaction: the model drafting for itself (every
draft is right unless the constraint disagrees with it, so long runs are accepted, cut at the first token the
constraint refuses, and the draft model's cache is rolled back to what the target kept), and a random tiny model of
the same tokenizer (wrong almost every time: a rejection at the first position, then the target's constrained choice
is used). The check pins the draft length to the doubling rule (`-Djllm.engine.adaptiveDraft=false`; the cost model
would stop drafting when it measures speculation as not paying, and nothing would be checked) and, on the CPU, the
int8 activations (`auto` times both ways at every load and may choose differently in two loads). The tiny draft cannot
run on Metal (head size 16; the kernels need a multiple of 32), so Metal is checked with the self-draft only. Result
of the last run: 40 comparisons with the speculation-off run (8 requests, each against prompt lookup and the drafts: 24 on the
CPU, 16 on Metal), all identical.

Two things cost the check some debugging, and are worth knowing. (1) On the CPU the very first generation of a JVM
once wrote free text where every later one, and every other speculative setting, wrote the call: twice in five runs
(once with `int8-activations=auto`, which can pick differently in two loads, once with it fixed), always the first
generation of the process. The cause was not isolated; cold, interpreted code rounding a float reduction differently from
the optimized code at a near-tie of the first token fits, and a generation before anything is recorded, which
`ToolCallCheck` and `ToolCallTest` now do, removed it (three full runs since, no difference). (2) A GGUF Q4_K_M of the
same model, as the draft, is refused ("does not share the target's tokenizer": its tokenizer, read from the file,
encodes the probe text differently from `tokenizer.json`), so it is not among the drafts.

Without a model, `ToolCallTest` (in `mvn install`, with the Qwen3-0.6B tokenizer and a scripted fake model whose
logits depend on the whole cached history, and which is tempted to write tokens the constraint must mask) covers the
same interaction: prompt lookup and two draft models (one writing the same script, whose proposals include the masked
tokens and are cut; one writing another call, rejected at its first different token), with and without jump-forward,
all giving the plain constrained tokens, in fewer passes (78 plain; 25 and 69 with the two drafts, 63 with prompt lookup;
48 with jump-forward, 26 and 44 with jump-forward and the drafts); and the same tokens again with `engine.constraint-masks=false`. On Qwen3-0.6B with
deterministic CPU kernels it compares the speculative and plain constrained output as well.

*Reliability.* Every call the constrained engine writes parses and matches the schema subset: on the fake model that
is tempted by line breaks and tabs inside the call (`ToolCallTest`, `ToolCallApiTest`) and on Qwen3-0.6B. What the
constraint cannot do is choose *well*: a model that wants to answer in prose is forced, under `"required"`, into a
call whose argument values it then has to invent.

*Costs and limits.* The device-side greedy pipe ([17.4](#174-pipelined-decoding-on-the-gpu)) is off for constrained
requests (it picks tokens the CPU never checks); `logprobs` turn jump-forward off. The layout of a call is fixed to the
Hermes / Qwen one; families whose own tool format differs (Llama 3's bare JSON, Mistral's `[TOOL_CALLS]`) are served
through the same Hermes prompt, as before. The console's own chat (`/api/chat`) is not constrained yet; the `/v1`
endpoints are.

*Switches.* Three settings (bundled `jllm.properties`, any flag, the console's settings page) each keep the old path
when `false` (with the last two the output is the same tokens as with them on; the first changes what the model may write):

| setting | default | off |
|---|---|---|
| `engine.constrained-tools` | `true` | `tools` of `/v1/chat/completions` only go into the prompt; the reply is free text parsed afterwards (`ToolCallFormat.parse`), and `response_format` with `tools` is a 400 again. Read for every request |
| `engine.jump-forward` | `true` | every token is sampled and takes a pass of its own, for tool calls and `response_format` alike |
| `engine.constraint-masks` | `true` | no trie walk, no memo, no allowed lists: candidates are tested one by one, the 256 most likely first, then the whole vocabulary (applies to the process: it is set when a model loads) |

`scripts/mac-validate.sh` has rows for the last two on a Mac: the decode time of a constrained tool call with jump-forward
off and on and with each speculation mode, and the token mask in microseconds with the masks on, on with a shared memo,
and off.
