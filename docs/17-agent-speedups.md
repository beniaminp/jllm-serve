# 17. Speedups for coding agents

A coding agent (opencode, Continue, aider, Cline) is a very particular customer. Its first request carries ~10 000
tokens of instructions and tool definitions; every later request repeats all of them plus the growing conversation;
it runs side requests (titles, summaries) next to the main one; and much of what it writes back is text that is
already in its context — lines of a file it read, the shape of earlier tool calls. This chapter covers the four
things jllm-serve does about that, each explained simply first and then exactly:

| | what | where |
|---|---|---|
| [17.1](#171-prompt-lookup-tuned-for-agents) | guess the next tokens from the context and check them all at once | `engine/Speculator`, `engine/NgramIndex` |
| [17.2](#172-slots-that-share-a-prefix) | several conversations share an expensive prefix by copying it, not recomputing it | `web/models/LoadedModel`, `Engine.adoptPrefix`, `Backend.copyStateFrom` |
| [17.3](#173-saved-conversations) | conversations survive restarts on disk | `web/models/KvStore`, `engine/CacheFile`, `Backend.saveState` |
| [17.4](#174-pipelined-decoding-on-the-gpu) | the GPU runs the next step before the CPU has looked at the last one | `Backend.GreedyPipe`, `MetalBackend.Pipe` |

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
backend, cache type, int8 activations — not context size, batch or threads); the **state format** names the layout
(`kv f16 layers=28 kvDim=1024 headDim=128`). A file is loaded only under the same fingerprint and format. Both the CPU
and the Metal backend write the same rows (per layer, per position, the K row then the V row, little-endian f32 or IEEE
half — Metal's `half` has the same bits), so the layout is shared between backends (ArchSmoke loads a CPU-written cache
into the Metal backend). `jllm-web`'s `KvStore`:

- **saves** a slot after a request whose prefill took ≥ 10 s, before a restore overwrites it, and on unload/shutdown,
  when it holds ≥ 1 024 tokens; a cache extending a saved one replaces it, one already covered is not written;
- **restores** before a request the saved cache with the longest common prefix with the prompt, if that beats what the
  slot holds by ≥ 256 tokens;
- keeps the files under `jllm.web.kv-cache-max-gb` (10), deleting the least recently used.

Size: `n × 2 × layers × kvDim × bytes` — 115 kB per token for Qwen3-0.6B f16, ~1.2 GB for opencode's 10k-token prompt.
**Measured:** a 2 707-token prompt prefilled in 21.3 s; after a restart the cache was restored in 0.9 s and the same
prompt was served with 2 706 tokens cached. On the Metal backend the file is read straight into the cache buffers
(unified memory: they are the GPU's memory), with no intermediate copy.

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
