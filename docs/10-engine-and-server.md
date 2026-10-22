# 10. The engine and the server — generation loop, HTTP, OpenAI API, configuration

Code: `engine/Engine.java`, `engine/TokenLogprob.java`, `server/HttpServer.java`, `server/ApiServer.java`,
`server/OpenAiFeatures.java`, `sampling/JsonConstraint.java`, `server/WebUi.java`, `config/Settings.java`,
`src/main/resources/jllm.properties`, `Main.java`.

Everything before this chapter computes *one* next-token distribution. This chapter is about the loop around
it, and about getting text in and out over the network with nothing but `java.net`.

---

## 10.1 The generation loop

`Engine.generate(prompt, params, sink, cancelled)` is the whole decode algorithm (the prompt is either text,
which `generate` encodes, or ready-made ids such as `Engine.renderChat(messages, thinking)` returns). In
pseudo-code, with the matching line in `generateLocked` on the right:

```
ids = preparePrompt(ids)                                    # BOS if tokenizer.promptBos() ≥ 0; empty ⇒ [EOS]
acquire the fair lock, at most server.queue-timeout-ms      # one KV cache ⇒ one request at a time; else 503
if len(ids) >= ctx: error "prompt too long"
if cancelled(): return finish = "cancelled"                 # the client left while queued

reuse = length of common prefix(ids, cached), capped at len(ids) − 1      # 10.2
pos = reuse
while pos < len(ids):                                       # PREFILL in chunks of maxBatch (64)
    if pos > reuse and cancelled(): cached = ids[0..pos); return finish = "cancelled"   # kept for a retry
    n = min(maxBatch, len(ids) − pos)
    backend.forward(ids, pos, n, pos, logits if this is the last chunk else null)
    pos += n

sampler = Sampler(params, vocab, ids)                       # penalties know the prompt
for i in 0 .. min(max_tokens, ctx − len(ids)):              # DECODE
    if cancelled(): finish = "cancelled"; break
    tok = sampler.sample(logits)                            # (09)
    if tok in eosIds: finish = "stop"; break               # <|im_end|> / <|endoftext|>
    sampler.accept(tok); out.append(tok)
    piece = streamDecoder.push(bytes of tok)                # only complete UTF-8 characters (04 §4.8)
    pending += piece; handle stop strings; emit the safe part to the sink      # 10.3
    if more tokens may follow: backend.forward([tok], 0, 1, pos, logits); pos += 1
flush what is still pending
remember cached = ids + the generated tokens that were fed through forward()
return text, tokens, finish reason, statistics
```

Points worth noticing:

- **The BOS token** is decided by `Tokenizer.promptBos()` (the tokenizer's own `add_bos_token` rule, see
  [04 §4.11](04-tokenizer.md#411-other-tokenizer-families)), not guessed from `model_type`. `preparePrompt` is
  idempotent and public because the API server must measure the prompt *with* its BOS against the context.
  Chat prompts rendered by the template that already start with the BOS do not get a second one.
- **Logits only when needed.** During prefill only the last chunk asks for logits; the other chunks exist
  purely to fill the KV cache ([05 §5.9](05-transformer-math.md#59-prefill-decode-and-the-kv-cache)).
- **One forward per generated token.** The token sampled at step `i` is fed at position `pos` to obtain the
  logits for step `i + 1`. The very last sampled token is *not* fed (nobody needs its successor's logits).
- **`finish_reason`** is `"stop"` (EOS or a stop string), `"length"` (`max_tokens` or the context is full) or
  `"cancelled"`.
- **Timing.** `t0 → t1` is prefill, `t1 → t2` is decode; `Stats` reports both plus `tokPerSec()`.
- **Which ids end generation.** `eosIds` is the union of what `Main` collected (`config.json`, `generation_config.json`,
  well-known stop tokens present in the vocabulary, [04 §4.7](04-tokenizer.md#47-added-and-special-tokens)) and
  `template.stopIds(tokenizer)`; if the union is empty the engine prints a note and generation ends only at
  `max_tokens` or a stop string. The vocabulary given to the sampler is `min(core.vocabSize(), tokenizer.vocabSize())`
  ([09 §9.9](09-sampling.md#99-the-vocabulary-padding-issue)).

---

## 10.2 The prefix cache

A chat client re-sends the *whole* conversation every turn. Without care, turn 10 would re-prefill turns 1–9.
But the KV cache already contains the keys/values of the previous prompt, and K/V of a token depend only on the
tokens before it — so as long as the new prompt starts with the same ids, those entries are still valid.

Worked example (letters stand for token ids):

```
turn 1  prompt  = [A B C D]                  cached = []            reuse = 0 → prefill A B C D
        generated E, F, then EOS             fed through forward: E (pos 4), F (pos 5); EOS not fed
        cached = [A B C D E F]               (positions 0..5 hold valid K/V)

turn 2  prompt  = [A B C D E F G H I]        the template appended <|im_end|>, a new user turn, "assistant\n"
        common prefix with cached = 6        cap = min(cachedLen = 6, len − 1 = 8) = 6 → reuse = 6
        prefill only [G H I] at positions 6, 7, 8
```

Turn 2 costs three tokens of prefill instead of nine. `Stats.cachedTokens` reports `reuse`; the API exposes it as
`usage.cached_tokens`.

**Why at least the last token is always recomputed** (`cap = len − 1`): the loop needs the *logits* of the last
prompt token, and `forward()` only produces logits for tokens it computes. If the entire prompt were already
cached, reusing it all would leave no logits to sample from. Recomputing one token costs one decode step.

**Divergence is handled automatically.** If the user edits an earlier message, the common prefix is shorter and
everything after it is recomputed; positions beyond `pos` in the cache are simply overwritten by the next
`forward`. After a stop string, the tokens generated *past* the stop are still in `cached` (they were fed), so
a following prompt that does not contain them just diverges at that point.

The cache is a single `int[] cached` + `cachedLen` — no eviction policy, because there is one sequence.

**The same idea before the model: tokenizing.** 🧸 Re-sending the whole conversation also means re-*reading* it:
before the prefix cache can compare ids, the text has to become ids again. The prompt cache remembers how far it
read last time and reads only the new part. 📐 `Engine.renderChat` goes through a `PromptCache`
(`engine.prompt-cache-mb`) that keeps the ids of recent prompts under hashes of their template pieces and resumes
the tokenizer right after the last added token the new prompt shares with a cached one, where the encoding of the
rest cannot depend on what came before; the ids equal a full tokenization
([17 §17.5](17-agent-speedups.md#175-tokenizing-only-what-is-new)).

Two cases make the bookkeeping more careful:

- **Failure.** Prefill overwrites the KV cache, so `generateLocked` sets `cachedLen = 0` *before* the first
  `forward` and only restores it when the request completes. If `forward` (or the sink) throws, the next request
  starts from nothing instead of trusting a half-overwritten cache.
- **Recurrent models.** A Mamba backend keeps a fixed-size state, not one K/V entry per position, and a state
  cannot be rewound to an earlier token. Such backends answer `Backend.supportsPrefixReuse() == false`; the engine
  then reuses the cache only when the new prompt extends *everything* that was fed before (`reuse == cachedLen`,
  as in a chat that appends turns) and otherwise restarts at position 0.

**Saving the cache to disk.** `Engine.saveCache(file, fingerprint)` writes the cached token ids and their keys and
values; `loadCache(file, fingerprint)` puts them back, after which a prompt starting with those tokens reuses them
exactly as above. The file (`CacheFile`) is a small header (magic, the caller's model fingerprint, the backend's
`stateFormat()`, the token ids), readable on its own for indexing, followed by the backend's state. The CPU and
Metal backends write their cache per layer, row by row, in the stored type (f32 or f16, so nothing is rounded
twice): the K and V rows of standard attention, or the latent rows of DeepSeek's MLA; Mamba reports no state format
and is not saved. A file of another fingerprint or format is refused, and a load that fails half-way leaves the
cache empty rather than mixed. A loaded cache is bit-identical to the saved one (`SmokeTest.persistCache` saves it
back and compares the files) and continues a conversation exactly like the engine that saved it. jllm-web builds
its saved-conversation store on this (README, *Coding agents*; [17 §17.3](17-agent-speedups.md#173-saved-conversations)).

🧸 Saving is like photocopying the first 40 pages of a notebook that is still in use: you may keep writing on page
41 and beyond while the copier runs, but you must not erase page 12 until it is done.

📐 `beginSave(file, fingerprint)` takes the engine lock only to copy the cached ids and pins positions `[0, n)`;
`PendingSave.write()` writes the rows from any thread. A request computes `reuse` (above) and waits for the save only
if `reuse < n`, i.e. if it would rewrite a pinned row; a load or prefix copy into the engine always waits. So the
next turn of the saved conversation (which extends it, `reuse ≥ n`) runs while its predecessor is written to disk.
`loadCache(file, fingerprint, limit)` loads only the first `limit` positions of a longer file
(`Backend.loadState(n, stored, in)` reads `n` rows per layer and seeks past the rest).

---

## 10.3 Stop strings

`"stop": ["###"]` must end generation when `###` appears, *without* returning it, even if it arrives as
`"#"`, `"##"` across two tokens. The engine keeps a `pending` buffer and emits only text that can no longer be
the beginning of a stop string: with `maxStop = 3`, the last `maxStop − 1 = 2` characters are always held back.

```
token piece   pending before    pending after     action
"Hello"       ""                "Hello"           safe = 5 − 3 + 1 = 3 → emit "Hel", keep "lo"
" wor"        "lo"              "lo wor"          emit "lo w", keep "or"
"ld#"         "or"              "orld#"           no stop found; emit "orl", keep "d#"
"##"          "d#"              "d###"            "###" found at index 1 → emit "d", finish = "stop"
```

The client sees `Hello world` and never `###`. `findStop` picks the earliest match among several stop strings.
When there are no stop strings, `safe = pending.length()` and text is emitted immediately. At the end,
whatever is still pending is flushed (unless a stop string was hit).

---

## 10.4 Cancellation, the lock, statistics

- **Cancellation.** `cancelled` is a `BooleanSupplier` polled between prefill chunks and before every token. A
  prefill stopped half-way keeps its finished positions as the cached prefix, so a client that times out and
  retries the same prompt (coding agents do) continues where the first attempt stopped. `HttpServer` watches every
  connection while its request is handled (a virtual thread blocks reading the socket, which returns EOF the
  moment the client closes) and `ApiServer` ORs that flag into `cancelled` for both the streaming and the
  non-streaming path, so a client that closes its tab stops burning GPU time within one token (a request that is cancelled
  while still queued returns `"cancelled"` as soon as it gets the lock). `server.request-timeout-ms` adds a
  wall-clock limit per request, measured from the moment generation starts; when it fires the answer ends with
  `finish_reason: "length"`. A streaming
  client that stops *reading* is cut off after `server.write-timeout-ms` (10.5).

  🧸 Before saying each next word, the teacher glances at the door: if the child who asked has left, she stops.

  📐 In the web console the servlet container owns the socket, so there is no probe thread; `ClientGone` asks
  Tomcat's connection for available input with a non-blocking read instead (once the body is read, input can only
  be the end of the stream), at most every 50 ms from the `cancelled` supplier. An abandoned request, streaming or
  not, stops within one prefill chunk or one token and frees its slot ([17 §17.6](17-agent-speedups.md#176-noticing-a-client-that-left)).
- **The lock.** The backend owns exactly one KV cache, so generation runs under a **fair** `ReentrantLock`:
  concurrent HTTP requests are served one at a time in arrival order, and virtual threads make the waiting cheap.
  The wait is bounded by `server.queue-timeout-ms` (default 60 s, 0 = forever): after that `generate` throws
  `Engine.OverloadedException`, which the API turns into a 503 `overloaded` instead of letting requests pile up.
  Serving many users at once would require batching several sequences per forward pass ([11](11-performance.md)).
- **Statistics** (`Engine.Stats`): `promptTokens`, `cachedTokens`, `generatedTokens`, `prefillMs`, `decodeMs`,
  `draftedTokens`, `acceptedTokens` (speculation, 10.9), `tokPerSec()`. The CLI prints them after each answer; the API returns them inside `usage`.

---

## 10.5 HTTP/1.1 from zero

A browser or `curl` talks to the server in plain text over a TCP socket. A request:

```
POST /v1/chat/completions HTTP/1.1\r\n          request line: method, path, version
Host: 127.0.0.1:8080\r\n                        headers: "Name: value", one per line
Content-Type: application/json\r\n
Content-Length: 73\r\n                          how many body bytes follow
\r\n                                            empty line = end of headers
{"messages":[{"role":"user","content":"hi"}],"stream":false,"max_tokens":8}
```

A response has the same shape with a *status line* instead of a request line:

```
HTTP/1.1 200 OK\r\n
Content-Type: application/json\r\n
Content-Length: 412\r\n
Connection: keep-alive\r\n
\r\n
{"id":"chatcmpl-…", …}
```

`HttpServer` implements the subset that matters:

| concept | what it means | where |
|---|---|---|
| request line + headers | lines end in `\r\n`; header names are case-insensitive (we lower-case them) | `readLine`, `readRequest` |
| `Content-Length` body | read exactly that many bytes, at most `server.max-body-bytes` (4 MB by default, else 413) | `readRequest` → `readFully` |
| chunked request body | rare for clients, supported anyway; the running total is limited the same way | `readChunked` |
| `Expect: 100-continue` | answered with `100 Continue` before the body is read (curl sends it for bodies over 1 KB) | `readRequest` |
| malformed framing | both `Content-Length` and `Transfer-Encoding`, a duplicate of either, a bad length or chunk size → 400; a header line or block over 64 KB → 431; a body over the limit → 413 | `readRequest` |
| overall read deadline | `server.read-timeout-ms` bounds each read *and* the whole request (headers plus body) from its first byte; exceeded → 408, so a client trickling one byte at a time cannot hold a connection | `RequestClock` |
| keep-alive | after a response, loop and read the next request on the same socket unless `Connection: close` | `connection()` loop |
| HTTP/1.0 and `HEAD` | an HTTP/1.0 request gets no keep-alive and no chunked encoding; `HEAD` gets headers only | `Request.http10`, `Response` |
| fixed response | status + headers + `Content-Length` + body in one write | `Response.send` |
| streaming response | see below | `Response.startStream` / `write` / `end` |
| one thread per connection | `Executors.newVirtualThreadPerTaskExecutor()`: a blocked virtual thread costs ~1 KB, so thousands of idle keep-alive connections are fine; a semaphore caps them at `server.max-connections` (256), the rest wait in the kernel accept queue | `serve()` |
| disconnect probe | while a request is handled, a second virtual thread blocks reading the socket; EOF means the client left, and `Request.clientGone()` lets the handler cancel the generation | `Probe` |

### Chunked transfer encoding

For a streamed answer the total length is unknown when the headers are sent. HTTP/1.1 solves this with
`Transfer-Encoding: chunked`: the body is a sequence of chunks, each prefixed by its size in **hexadecimal**,
terminated by a zero-length chunk. Two chunks, `Hello` and ` world!`:

```
HTTP/1.1 200 OK\r\n
Content-Type: text/event-stream; charset=utf-8\r\n
Transfer-Encoding: chunked\r\n
\r\n
5\r\n                 ← 5 bytes follow
Hello\r\n
7\r\n                 ← 7 bytes follow
 world!\r\n
0\r\n                 ← last chunk
\r\n
```

`Response.write(chunk)` frames one such chunk and puts it on a **bounded queue** (256 chunks) drained by a writer
thread of the response's own, which flushes whenever the queue runs empty, so the client still receives each piece
immediately. The point of the queue: `write` is called by the engine thread while it holds the engine lock, and
a client that stops reading (its TCP window full) would otherwise block that thread, and with it every other
request. If the queue stays full for `server.write-timeout-ms` (default 10 s, 0 = forever) the socket is closed and
`write` throws, which stops the generation. `end()` queues `0\r\n\r\n` and waits for the writer to finish.

### Server-sent events (SSE)

Inside the chunked body the API uses the SSE text format: each event is `data: <payload>\n\n`. The OpenAI
convention is one JSON object per event and a final `data: [DONE]`. `ApiServer.sse()` is one line:
`res.write("data: " + Json.write(event) + "\n\n")`.

---

## 10.6 The OpenAI-compatible API

| route | method | purpose |
|---|---|---|
| `/v1/chat/completions` | POST | messages → chat template → generate |
| `/v1/completions` | POST | raw `prompt` → generate |
| `/v1/models` | GET | lists the loaded model id |
| `/health` | GET | `{"status":"ok","model":…}` |
| `/` (and `/index.html`) | GET | the chat page (`WebUi`) |
| any path | OPTIONS | CORS preflight: 204, with the `Access-Control-*` headers only when `server.cors-origin` is set |
| anything else | — | 404 JSON error; non-POST on a completion route → 405; a body that is not `application/json` → 415 |

### Chat, non-streaming

Request:

```json
{"messages": [{"role": "system", "content": "You are terse."},
              {"role": "user",   "content": "Name three planets."}],
 "max_tokens": 60, "temperature": 0.6, "enable_thinking": false}
```

`chatPrompt()` renders the messages to ids with the family template (`Engine.renderChat` →
`Format.renderTokens`; multi-part `content` arrays are flattened to their `text` parts and any other part type,
such as `image_url`, is a 400; `enable_thinking` may also be given inside `chat_template_kwargs`, and is ignored,
with a one-time note, for templates other than Qwen3's). Message content is tokenized like Hugging Face's
`encode_special_tokens`, so a user who types `<|im_end|>` cannot forge a control token
([04 §4.11](04-tokenizer.md#411-other-tokenizer-families)). `params()` merges the request's sampling fields over the
server's defaults ([09 §9.10](09-sampling.md#910-where-each-setting-comes-from)), and `engine.generate` runs. Response:

```json
{"id": "chatcmpl-1a14141287f-1", "object": "chat.completion", "created": 1792096151,
 "model": "Qwen/Qwen3-0.6B",
 "choices": [{"index": 0,
              "message": {"role": "assistant", "content": "Mercury, Venus, Earth."},
              "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 24, "completion_tokens": 7, "total_tokens": 31,
           "cached_tokens": 0, "prefill_ms": 96, "decode_ms": 65, "tokens_per_second": 107.7}}
```

`usage` carries the standard three counters plus jllm's extras (`cached_tokens`, `prefill_ms`, `decode_ms`,
`drafted_tokens`, `accepted_draft_tokens` (speculative decoding, [10.9](#109-speculative-decoding), 0 when off),
`forced_tokens` (tokens a JSON or tool-call constraint forced and jump-forward appended without sampling,
[17 §17.7](17-agent-speedups.md#177-constrained-tool-calls-and-jump-forward)), `tokens_per_second`), which OpenAI
clients ignore.

### Chat, streaming (`"stream": true`)

The response is `text/event-stream`, chunked. The headers are sent lazily, with the first token, so until then a
queue timeout can still be a proper 503. First an event with the role, then one event per emitted text piece, then
a closing event with `finish_reason` and `usage`, then `[DONE]`:

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk",…,"choices":[{"index":0,"delta":{"content":"Mercury"},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk",…,"choices":[{"index":0,"delta":{"content":", Venus"},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk",…,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{…}}

data: [DONE]
```

The engine's text sink queues each event for the connection's writer thread (10.5), so tokens appear in the
client as they are produced. If a write fails or times out, or the disconnect probe sees the client leave, the
`cancelled` supplier stops the engine.

`"stream_options": {"include_usage": true}` follows the OpenAI shape instead: `usage` is omitted from the
`finish_reason` event and sent in an extra final event with `"choices": []` (with `include_usage` false or absent
under `stream_options`, no usage at all). Without `stream_options` the legacy inline `usage` shown above is kept.

### Completions

`{"prompt": "The three primary colors are red,", "max_tokens": 40, "temperature": 0, "stop": ["."]}` →

```json
{"id": "cmpl-…", "object": "text_completion", "created": …, "model": "Qwen/Qwen3-0.6B",
 "choices": [{"index": 0, "text": " yellow, and blue", "finish_reason": "stop"}],
 "usage": {…}}
```

Streaming uses the same `text_completion` objects with a `text` field per event.

### More than one answer, probabilities and structured output

Beyond the sampling fields the server implements the rest of OpenAI's generation options. They are parsed and
validated in one place, `OpenAiFeatures`, which the web console's `/v1` endpoints use too, so both servers accept
the same requests. Each is explained in plain terms first, then exactly.

**`n`: several answers to one prompt.** With `"n": 3` the response has three choices, each an independent sample
(with `temperature: 0` they are all the same greedy answer, as with OpenAI). They are generated one after the other
by `Engine.generate(prompt, params, decoding, n, sink, cancelled)` under a single hold of the engine lock: choice 0
prefills the prompt, and each further choice finds the prompt in the prefix cache (10.2) and recomputes only its
last token, so a 2,000-token prompt costs one prefill, not three. Choice *i* of a seeded request samples with
`seed + i` (so choice 0 is exactly the single-answer response); without a seed each choice draws its own. `n` is
1 to 16. `usage.prompt_tokens` counts the prompt once, `completion_tokens` sums all choices. When streaming, the
choices are streamed in turn, every chunk carrying its choice's `index`: a role chunk, the deltas, and one
`finish_reason` chunk per index (the inline legacy `usage` rides on the last choice's, the `stream_options` one
follows in its own event as before). Because choice 1 onward recompute the last prompt token with a single-token
step rather than in the prefill batch, its logits can differ from choice 0's in the last bits (the
prefill-versus-decode caveat of `Backend.forward`).

**`logprobs`: how likely each token was.** The log-probability of a token is the natural log of the probability
the model gave it, a number at most 0 (−0.01 almost certain, −5 a long shot). For logits *z* over the vocabulary,

```
logprob(t) = z_t − log Σ_j exp(z_j)          (log-softmax, computed as  z_t − (m + log Σ_j exp(z_j − m)),  m = max z)
```

jllm reports the log-softmax of the **raw model logits**, before repetition and presence/frequency penalties,
`logit_bias`, temperature, top-k/top-p/min-p truncation and any `response_format` mask: the model's own
distribution, the same whatever the sampling settings, so values are comparable across requests and summing them
scores a text (OpenAI documents its values as the model's distribution too). The cost is one pass over the
vocabulary per token (`TokenLogprob.of`, a max, a sum of exponentials and a short sorted list for the
alternatives). A request with logprobs or `echo` scoring decodes every step on the CPU from its own logits:
speculative decoding (10.9), jump-forward and the pipelined greedy path are off for it. A JSON constraint alone
keeps speculation (drafts are checked against it, 10.9) and adds jump-forward; only the device-side greedy pipe,
which picks tokens the CPU never sees, is off.

- Chat: `"logprobs": true` and optionally `"top_logprobs": k` (0 to 20) return per choice
  `"logprobs": {"content": [{"token", "logprob", "bytes", "top_logprobs": [{"token", "logprob", "bytes"}, …]}, …],
  "refusal": null}`. `token` is the token's bytes as UTF-8 (a fragment of a multi-byte character shows as U+FFFD,
  which is why `bytes`, the raw byte values, is there). `top_logprobs` without `"logprobs": true` is a 400.
- Completions: `"logprobs": k` (0 to 20) returns parallel lists `tokens`, `token_logprobs`, `top_logprobs` (one
  object per position mapping the k most likely token texts, plus the chosen one as OpenAI does, to their
  logprobs) and `text_offset` (each token's character offset in the returned `text`).
- Streaming: each chunk carries the entries of the tokens generated since the previous chunk (a token held back
  by a possible stop string arrives with the text it belongs to; the tokens of a matched stop string are reported
  although their text is cut).

**`logit_bias`: nudging or banning tokens.** `{"logit_bias": {"15339": 5, "9906": -100}}` adds the given number
to those tokens' logits (keys are token ids as strings, values in [−100, 100]) before sampling, after the
penalties, exactly like OpenAI's: +1…+5 makes a token more likely, −100 removes it in practice (e^−100 next to
anything else), +100 all but forces it. The bias is a dense array in `Sampler`, so it applies to greedy and sampled
decoding and to speculative verification alike; only the device-side greedy pipe is skipped, since it computes
the argmax on the GPU without it. An id outside the vocabulary is a 400.

**`response_format`: answers that are valid JSON.** `{"type": "json_object"}` makes the output a JSON object;
`{"type": "json_schema", "json_schema": {"schema": {…}}}` makes it a JSON value matching the schema. This is
*constrained decoding*: at every step tokens that would break the JSON are masked out before sampling, so the
model can only write a valid prefix, and once the value is complete the generation ends (finish reason
`stop`). The model is not told about the format: as with OpenAI, put "answer in JSON" (and the fields you want)
in the prompt, or a model that wanted to write prose will produce `{}` after some whitespace. With Qwen3,
thinking is switched off for these requests (a `<think>` block is not JSON).

How the mask works (`JsonConstraint`). Every token's bytes are taken from the tokenizer once and cached
(`tokenTable`; special tokens never qualify, end-of-sequence tokens only when the value is complete). The output
so far is the state of a byte-level pushdown automaton: a stack of frames (object, array, string, number,
literal) that knows, for instance, "inside an object of schema S, keys `a` and `b` seen, expecting `,` or `}`".
A token is allowed when its bytes, fed one by one to a copy of that state, are all accepted. Testing all ~150,000
tokens at every step would cost more than the forward pass, so `Sampler.sample(logits, allowed)` checks lazily.
Greedy first tests the argmax: when the model's own choice is allowed (most steps) nothing else is looked at.
Otherwise it takes the 256 most likely tokens (after penalties and bias), tests them most likely first, and keeps
the allowed ones; only if none of them is allowed (typically the first token, when the model wanted to write
"Sure") does it search the whole vocabulary. That search no longer feeds 150,000 tokens one by one: after 300
checks in one state the constraint computes the whole allowed set by walking a **byte trie of the vocabulary**
(`TokenTrie`: one node per distinct token prefix) alongside the automaton, so a shared prefix is tested once and a
refused byte prunes its whole subtree; in a structural position only the subtrees under the few allowed first bytes
are visited. The resulting bit mask answers every later check, and is memoized by the automaton state (the whole
frame stack, schemas by identity) in a small cache shared by the constraints of one schema or tool set, so a
recurring state costs a lookup the next time (`ByteConstraint`). Greedy takes the first allowed token, i.e. the
argmax of the masked distribution. Per-token cost, measured in [17 §17.7](17-agent-speedups.md#177-constrained-tool-calls-and-jump-forward). Sampling renormalizes over the allowed candidates and applies temperature,
top-k (the k most likely *allowed* tokens), top-p and min-p to that distribution; allowed tokens outside the 256
candidates are left out, an approximation that only drops tail mass. Whitespace between JSON tokens is limited to
one newline and 64 bytes per gap, so a reluctant model cannot fill `max_tokens` with blank lines. If
`max_tokens` runs out first, the output is a valid but unfinished prefix with finish reason `length`, as with
OpenAI.

The schema subset that is enforced: `type` (`object`, `array`, `string`, `number`, `integer`, `boolean`, `null`,
or a list of them; without `type`, `properties` / `required` / `additionalProperties` imply an object and `items`
an array), `properties`, `required` (`}` is refused until every required key was written),
`additionalProperties` (`false`: only the listed keys, each at most once; a schema: the type of the other keys'
values), `items`, `minItems` / `maxItems`, `minLength` / `maxLength` (characters), `enum` and `const` (matched
against the values' compact serialization, e.g. `"red"` or `1.5`), `anyOf` / `oneOf` whose alternatives start with
different JSON types (the common `{"anyOf": [{…object…}, {"type": "null"}]}` of optional fields: the first byte of
the value picks the alternative), local `$ref` (`#`, `#/$defs/…`, `#/definitions/…`, recursion allowed) and
OpenAPI's `nullable`. Annotations (`title`, `description`, `default`, `examples`, `format`, `$schema`, …) are
ignored. Any other keyword (`pattern`, `minimum`, `allOf`, tuple `items`, …) is a 400 naming it, rather than a
schema that is silently enforced only in part. Keys are not reordered: the model writes them in the order it
likes. Duplicate keys are refused in any object.

**Jump-forward.** Where the automaton allows exactly one continuation (the rest of `true` after `t`, the rest of an
enum value or of the only key that fits, `"` after `{` in an object with required keys) the engine does not sample
it: the forced bytes are tokenized and appended in one pass (`TokenConstraint.jump`, details in
[17 §17.7](17-agent-speedups.md#177-constrained-tool-calls-and-jump-forward)). With free whitespace, as here,
little is forced; tool calls use a fixed layout and gain much more.

**Tool calls (web console `/v1`).** With `tools` the model's `<tool_call>` blocks are constrained too: the function
name must be one of the tools and the arguments must match its `parameters` schema, compiled *leniently* (keywords
the automaton cannot enforce, such as `pattern` or `minimum`, are ignored rather than refused, since the client did
not write the schema for us). `tool_choice: "required"` or a named function forces the call; `response_format`
together with `tools` is allowed and means "tool calls or that JSON answer". Three settings keep the old paths:
`engine.constrained-tools=false` (tool calls are free text parsed afterwards; `response_format` and `tools` exclude
each other again), `engine.jump-forward=false` (every token sampled) and `engine.constraint-masks=false` (candidates
tested one by one, no trie masks). See
[17 §17.7](17-agent-speedups.md#177-constrained-tool-calls-and-jump-forward).

**`echo` (completions): the prompt in front.** `"echo": true` returns the prompt followed by the completion in
`text` (streamed: the prompt is the first chunk of each choice). With `logprobs` the prompt's tokens come first in
the lists too, the first with `null` (nothing predicts it), the others with their log-probabilities: for this the
prompt is prefilled through `Backend.forwardAll` in chunks of 16 tokens, which returns every position's logits
(without the prefix cache, since every position is needed). `"max_tokens": 0` is then allowed and only scores the
prompt. A BOS token the engine adds in front is not echoed.

**`suffix` (completions): fill in the middle.** With `"prompt": "def area(r):\n    "` and
`"suffix": "\n    return a"`, a code model writes what goes between. This needs a model trained for it, whose
tokenizer has `<|fim_prefix|>`, `<|fim_suffix|>` and `<|fim_middle|>` (the Qwen2.5-Coder family): the prompt
becomes `<|fim_prefix|>` prompt `<|fim_suffix|>` suffix `<|fim_middle|>` (prefix-suffix-middle order) and the
completion is the middle. For any other tokenizer the request is a 400 that says so. `echo` and `suffix` together
are a 400.

**`best_of` (completions): pick the most likely.** `"best_of": 5, "n": 2` generates five candidates and returns
the two whose total log-probability (the sum of their tokens' logprobs, i.e. the log-likelihood of the sequence)
is highest, most likely first. `best_of` must be at least `n` (default `n`), at most 16, and cannot be combined
with `stream` (a 400, as with OpenAI): the choice is only known at the end. `usage.completion_tokens` counts all
candidates' tokens.

### Errors

Every error, from the HTTP layer or the API, is a JSON body of one shape:

```json
{"error": {"message": "invalid JSON: JSON expected object key at offset 1", "type": "invalid_request_error"}}
```

(`code` is added where there is one, e.g. `"context_length_exceeded"` or `"overloaded"`.)

| status | when |
|---|---|
| 400 | malformed JSON; missing `messages`; invalid sampling values, including non-finite ones ([09 §9.10](09-sampling.md#910-where-each-setting-comes-from)); a prompt that is too long (see below); malformed or ambiguous HTTP framing (duplicate or conflicting `Content-Length` / `Transfer-Encoding`, bad chunk size); a request feature this server does not implement |
| 404 | unknown route |
| 405 | wrong method on a completion route |
| 408 | the request did not arrive within `server.read-timeout-ms` (10.5) |
| 413 | body larger than `server.max-body-bytes` |
| 415 | `Content-Type` is not `application/json` |
| 431 | a request line or header block longer than 64 KB |
| 500 | an unexpected exception before anything was sent |
| 503 | `"code": "overloaded"`: the engine stayed busy longer than `server.queue-timeout-ms` |

**Context length.** The check measures the prompt *with* the BOS token that `Engine.preparePrompt` prepends, so
a request that would fit without it but not with it is rejected up front with 400
`"code": "context_length_exceeded"` instead of failing inside the engine (the engine's own check is the same
condition and is only a backstop). Prompts larger than `engine.max-prompt-bytes` (default 16 × `backend.ctx` bytes)
are refused with the same code *before* tokenizing, so a huge body never reaches the tokenizer. `max_tokens` is
clamped to the room left in the context.

**Unsupported OpenAI fields.** Rather than silently ignoring what it cannot honour, the server answers 400
(`"... is not supported"`) for non-empty `tools` and `tool_choice` other than `"none"` (function calling is the
web console's job: its `/v1` endpoints implement it), message content parts that are not `text`, and a completion
`prompt` that is not one string (batches and token-id arrays). Out-of-range or malformed values of the fields
above (`n` 17, `top_logprobs` 21, a `logit_bias` of 500, an unsupported schema keyword, `echo` on chat, …) are a
400 naming the field.

**After the stream has started** the status line is long gone. If an exception happens then, the server sends
one more event, `data: {"error": {"message": "internal server error", "type": "server_error"}}`, ends the
chunked body properly and closes the connection, with no `[DONE]`. The OpenAI SDKs raise an `APIError` when a
stream event carries an `error` key, so clients see a failure rather than a short answer. (A handler that dies
without reporting in-band is cut off without the chunked terminator, which also reads as an incomplete body.)

### Thinking mode

Qwen3 reasons inside `<think>…</think>` before answering. `enable_thinking` (request field, or
`chat_template_kwargs.enable_thinking`, default from `chat.thinking` in `jllm.properties`) controls whether
the template pre-fills an empty think block ([04 §4.9](04-tokenizer.md#49-the-chat-template)). The reasoning
is returned inline in `content`; set a generous `max_tokens`, as thinking can take hundreds of tokens.

### Using it from an SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
for chunk in client.chat.completions.create(model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": "Why is the sky blue?"}], stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

---

## 10.7 The JSON parser

`json/Json.java` (≈450 lines) is a **recursive-descent** parser: `value()` looks at the next character and
calls `object()`, `array()`, `string()`, `number()` or a literal; objects and arrays call `value()` for their
children. Types map to plain Java: object → `LinkedHashMap<String,Object>` (keeps key order), array →
`ArrayList<Object>`, integers → `Long`, other numbers → `Double`, plus `String`, `Boolean`, `null`. The writer
is a pattern-matching `switch` over those types with proper escaping (`"`, `\`, control characters, U+2028/9).
Typed helpers (`Json.str`, `Json.num`, `Json.map`, …) return defaults instead of throwing, which keeps the API
handlers short. Integers beyond `long` fall back to `Double`, nesting deeper than `Json.MAX_DEPTH` (256) is
rejected, and a parse error is an `IllegalArgumentException` the API reports as a 400 with its offset. It parses
the 11 MB `tokenizer.json` in well under a second.

---

## 10.8 The web page

`WebUi.HTML` is one string: a message log, an input box, a "enable thinking" checkbox and ~30 lines of
JavaScript. On submit it `fetch`es `/v1/chat/completions` with `stream: true`, reads the body with a
`ReadableStream` reader, splits on blank lines, parses each `data:` line and appends `delta.content` to the
current answer. The history lives in the page and is re-sent each turn, which the prefix cache (10.2) makes
cheap. After each answer it prints the token count and tok/s from the final `usage` event.

---

## 10.9 Speculative decoding

Decoding streams every weight once per token, so a token costs the same whether the pass computes one position or
several. Speculative decoding exploits that: guess the next few tokens, feed *all* of them in one pass, and keep
the guesses the model itself would have produced. Code: `engine/Speculator.java`, `engine/DraftModel.java`,
`Sampler.verify`, `Backend.forwardAll`. Prompt lookup is on by default (`engine.speculative=ngram`; `off` disables
it, measurements below and in [17](17-agent-speedups.md#171-prompt-lookup-tuned-for-agents)).

```
tok  = sample(logits)                       # a normal step: the next token is always the target's own choice
drafts d1..dm = propose(history + tok)      # prompt lookup or a draft model, m <= engine.draft-tokens
rows = backend.forwardAll([tok, d1..dm], pos)        # ONE pass, the logits of all m+1 positions
for j in 0..m:                              # rows[j] = the distribution after tok, d1..dj
    s = verify(rows[j], d[j+1]) if j < m else sample(rows[m])
    emit s;  if s != d[j+1]: break          # the first disagreement ends the chain, s is the correction
pos += 1 + (accepted drafts)                # K/V of rejected positions is simply overwritten by the next pass
```

**Exactness.** A draft token is *accepted* only if it is what the target would have sampled there.
Greedy: `verify` returns the argmax, so the draft is kept iff it is the argmax and the output is the
non-speculative output. Sampling: both proposers are deterministic (a prompt-lookup continuation, a greedy draft
model), i.e. a one-hot proposal distribution q, and the speculative-sampling rule `accept x with probability
min(1, p(x)/q(x))`, else draw from `norm(max(0, p − q))`, reduces to "accept the draft with probability p(x), else
draw from p with x's mass removed". `p` is the very distribution `sample` draws from (penalties, temperature,
top-k, top-p, min-p), so the result is distributed exactly like a plain sample; the tests check this on a toy
vocabulary with 200 000 seeded draws and on a fake-backend engine run. The penalties see accepted tokens in order:
`Speculator` calls `Sampler.accept` for every accepted draft before it samples the next row.

**Bookkeeping.** The engine's cached-token list contains only tokens that were really emitted
(`min(positions fed, prompt + output)`), so a rejected or end-of-sequence draft never enters the prefix cache.
Mamba backends (`supportsPrefixReuse() == false`) cannot rewind their state: speculation is refused for them
with a note.

**Prompt lookup (`ngram`).** The longest suffix of prompt + output (at most `engine.ngram-max` = 8, at least
`engine.ngram-min` = 3 tokens) is looked up in a hash index of every n-gram of the text (`NgramIndex`: constant time
however long the context, `engine.ngram-window` = 0 searches all of it); the tokens that followed its most recent
earlier occurrence are the draft, up to an adaptive length (doubling while whole drafts are accepted, up to
`engine.draft-tokens` = 16). It needs no second model and no memory, and works when the output copies the input:
editing code, quoting a document, extracting fields, structured output. Details and the math in
[17 §17.1](17-agent-speedups.md#171-prompt-lookup-tuned-for-agents).

**Draft model (`draft`).** `engine.draft-model` names a smaller model of the same tokenizer (checked at start-up
with a probe string), e.g. Qwen3-0.6B for Qwen3-30B-A3B. It gets its own backend (same `backend`, `backend.ctx`,
`model.quant`, `backend.kv-dtype`) and KV cache in the same process, on the same device; it is loaded before the target,
so on Metal the target's memory plan counts it ([08 §8.18](08-metal-backend.md#818-experts-on-the-cpu-backendmetalcpu-moe)).
It remembers which tokens its cache holds and re-feeds only the difference to the new history, then proposes up to
`engine.draft-tokens` tokens greedily. On Metal its decode steps are pipelined: each step takes its token from the
argmax the GPU wrote in the step before, so the draft runs its steps back to back and returns only token ids
([08 §8.25](08-metal-backend.md#825-draft-model-speculation-on-the-gpu)).

🧸 How long a draft to ask for is a bet: a long draft wins a lot when the draft model is usually right and wastes the
target's time when it is not. So the engine keeps score — how often the draft was right, how long a draft step and a
check take — and asks for the length that gives the most words per second, or for none at all while drafting does not
pay.

📐 With per-token acceptance α, k drafts yield `E(k) = 1 + α + … + α^k` tokens per pass at a cost of `k·d + a + b·k`
(d a draft step, `a + b(n−1)` a target pass over n positions); `Speculator.CostModel` measures α, d, a, b and a plain
step p while running and picks the k that maximizes `E(k) / (k·d + a + b·k)`, or pauses for 16 plain steps when no k
beats `1/p` (one pause in four tries a pass anyway). Until two passes are measured, and always for prompt lookup, the
doubling rule of prompt lookup (above) decides. `-Djllm.engine.adaptiveDraft=false` turns the cost model off.

**Back-off.** A pass whose drafts are all rejected cost more than a plain step, so after it the proposer
sits out 1, 3, 7, then 15 steps (reset by the next accepted draft).

**What a verification pass costs.** `forwardAll(n)` runs the n positions through the multi-token kernels and the
LM head over n rows (`CpuBackend`: one `mm.mul` with n rows; `MetalBackend`: `rmsnorm` and `matmul` over n rows
into an n-row buffer, copied back whole). On the Metal backend a pass of n tokens measured **1 + 0.36 (n−1)** single
steps for Qwen3-0.6B (n=5: 2.4×) and **1 + 0.55 (n−1)** for Qwen3-30B-A3B (n=5: 3.3×: each extra position
routes to its own experts, which are streamed again). Speculation pays off only if the accepted tokens per pass
exceed that factor, see [11 §11.6](11-performance.md#116-speculative-decoding). Greedy output with
speculation equals the plain greedy output bit for bit on Metal (the tiled kernels are bit-identical to the
single-token ones, [08 §8.10](08-metal-backend.md#810-the-tiled-prefill-matmuls)); on the CPU prefill and decode
kernels differ in the last bits, so it holds only with `-Djllm.cpu.deterministic=true
--backend.cpu.int8-activations false` ([07 §7.14](07-cpu-backend.md#714-prefill-and-decode-are-not-bit-identical-results-depend-on-history)).

**Under a constraint** (`response_format`, tool calls) speculation stays on. The draft is cut at its first token
the constraint does not allow, checked on copies of the constraint (`TokenConstraint.copy`), so only drafts that
could be emitted are verified; an end-of-sequence draft counts only where the output may end. Row *j* of the
verification pass is then decided with the constrained rule against the constraint as it stands after the first
*j* drafts (`Sampler.verify(row, draft, constraint, isStop)`): greedy takes the masked argmax, sampling accepts the
draft with its probability under the masked distribution. Greedy output is therefore exactly the non-speculative
constrained output (`ToolCallTest` checks it on a scripted fake model and on Qwen3-0.6B).
**Rolling back.** Nothing is copied or erased on a rejection, for either model and any cache type (f32, f16, q8: each
position's row is written whole): the target continues at `pos + 1 + accepted`, the positions after it are
overwritten by its next pass before anything attends to them, and the draft re-feeds what follows the longest prefix
its cache shares with the new history. `DraftSpecTest` (part of `mvn test`) checks it on the CPU with Qwen3-0.6B
drafting for itself, once as loaded from the settings and once with a draft that is wrong at every third position;
`ArchSmoke` on the Metal emulator with every attention family, f32 / f16 / q8 caches, drafting for itself, at Q2_K,
and (plain Llama) from another random model. Greedy output equals the output without speculation in all of them.

**The model's own draft head (`mtp`).** A Qwen3.5 GGUF carries a multi-token-prediction layer (`nextn`) that guesses the
token after the one just sampled from the main model's last hidden state ([16.14](16-gguf.md#1614-multi-token-prediction-the-model-as-its-own-draft-model)).
`engine.speculative=mtp` uses it as the proposer: one draft per step, verified in a two-position `forwardAll`.

🧸 The big model reads the conversation on a whiteboard that every word scribbles on. To check a guess it must read
the guessed word too, and if the guess was wrong, wipe the scribble. So before each check it takes a photo of the
board's state (on the GPU: the layers write the photo themselves while they work, a second set of buffers), and a wrong
guess just swaps the photo back in.

📐 Models whose recurrent state cannot be rewound by overwriting (the Gated DeltaNet hybrids, [14.14.5](14-model-families.md))
were refused by `Speculator.supported`; the MTP mode asks instead for `Backend.hasMtp() && supportsRollback()`. A step is
`checkpointState(); forwardAll([tok, d1], pos); if (d1 rejected) rollbackState(1)`; the rule of acceptance is the same
`Sampler.verify` as for the other proposers (the proposal is the head's argmax, one-hot), so greedy output is the plain
greedy output and sampled output has the target's distribution. With acceptance `α` a step yields `1 + α` tokens for one
two-position pass plus the head (about one layer and the output head) and one more head row for the KV of the position
just verified; `GgufQwenNextTest` checks it with an oracle head (right two of three times) and the real head on CPU and
Metal. On the Qwen3.8-27B (Q4_0, M1 Pro) the head guesses the next token 90 % of the time and the mode decodes a code answer
at 5.4 tok/s against 4.3 without (the output is the same text); a two-token pass costs 1.5 plain steps there, which bounds
the gain ([16.14](16-gguf.md#1614-multi-token-prediction-the-model-as-its-own-draft-model)). A model with a head uses the mode by default.

Statistics: `Engine.Stats.draftedTokens()` / `acceptedTokens()`, printed by `run`/`chat` and returned in the API
`usage` block as `drafted_tokens` and `accepted_draft_tokens` (0 when speculation is off).

---

## 10.9b Batched decoding across slots

🧸 A teacher reads a page of the big book aloud to answer one child's question. If three children each ask about
the same book, she can read the page once and answer all three, instead of reading it three times.

📐 Generating one token means reading every weight of the model once (decode is limited by memory bandwidth, see
[11](11-performance.md)). A server with several slots (`jllm-web`, one engine per slot) used to read them once
per conversation. With `engine.batch-decode=true` (the default) the slots of a model form an
`engine.BatchGroup`, and the next tokens of all conversations generating at the same time come out of one pass
(`Backend.forwardBatch`):

```
one at a time:   n conversations · (W / bandwidth)              per round of n tokens
batched:         W / bandwidth  +  n · (attention + small ops)  per round of n tokens
```

`W` is the weight bytes. The projections, the MLP and the LM head treat the n tokens as the rows of one
multi-token matmul, so each weight block is read (and dequantized) once for all of them. RoPE, the KV-cache store
and attention run per row, each in its own conversation's cache at its own position.

**How the steps meet.** Every slot's backend is wrapped in a `BatchGroup.Member`:
- A one-token `forward` (a decode step) joins a queue.
- Whichever thread takes the group's device lock first becomes the leader. It waits up to 2 ms
  (`-Djllm.engine.batchWindowMicros`) for the other slots that decoded within the last 200 ms, runs every queued
  step as one pass and hands each step its logits. The other threads just wait for their own result.
- A slot alone never waits.
- Prefill, verification passes and saved-cache operations take the same lock, so slots never compute at the same
  time. They also share one CPU thread pool.
- Pipelined GPU decoding ([08 §8.16](08-metal-backend.md#816-pipelined-greedy-decoding)) is used only while a slot
  is alone in its group.

**Results.** A batched step uses the multi-token kernels, which add up in a different order than the single-token
ones on the CPU, so its logits agree with a step taken alone to rounding (with int8 activations, to the int8 rounding). The
greedy answers were the same in `BatchDecodeTest` (three slots on Qwen3-0.6B, CPU, part of `mvn test`; 42 of the
steps ran in shared passes). Its throughput comparison runs with a second argument, `bench`, and takes settings after it
(`--backend metal` on a Mac, as `scripts/mac-validate.sh` runs it); it was not measured on a quiet machine yet.

**Backends.** The CPU backend implements `forwardBatch` for every attention family except DeepSeek's MLA. The Metal
backend implements it for every attention family including MLA, with f32, f16 and q8 caches, MoE layers and experts on
the CPU ([08 §8.24](08-metal-backend.md#824-batched-decoding-across-slots-metalbackendforwardbatch)): the pass runs on
the leader's command queue and activation buffers, and each row's RoPE, cache store and attention are dispatched
against its own slot's cache buffers. Below 8 slots those kernels compute each row exactly as the slot's own step, so on
Metal the batched logits were bit-identical to the alone ones in every family tested on the emulator (with experts on the
CPU: equal to rounding, the CPU's batched expert kernels add in another order). Mamba and Qwen3-Next (recurrent
states) fall back to running the queued steps one after another; they have no draft-model or prompt-lookup speculation
either (a recurrent state cannot be rolled back).

## 10.10 Configuration

Every tunable is a key in the bundled `src/main/resources/jllm.properties`, each with a comment. `Settings.load`
layers three sources, later ones winning:

```
1. bundled jllm.properties                           defaults; also the list of valid keys
2. one external file:   --config <path>
                   else $JLLM_CONFIG
                   else ./jllm.properties (if it exists)
3. command line flags
```

Flags come in two spellings: short aliases (`Settings.ALIASES`: `--model`, `--revision`, `--dir`, `--quant`,
`--low-disk`, `--backend`, `--ctx`, `--batch`, `--threads`, `--temp`/`--temperature`, `--top-k`, `--top-p`,
`--min-p`, `--rep-penalty`, `--seed`, `--max-tokens`, `--speculative`, `--draft-model`, `--draft-tokens`, `--think`, `--system`, `--host`, `--port`) or the
property key itself (`--server.port 9000`, `--bench.rounds=5`). `--key=value` also works and is the only way
to pass a value that itself starts with `--`. A boolean flag may stand alone (`--think` means `true`); it only
consumes the next argument when that is a boolean literal (`true`/`false`, `yes`/`no`, `on`/`off`, `1`/`0`), so
`--raw` never swallows what follows, and a stray positional argument (a typo such as `--think flase`) is an
error instead of being ignored. Three flags are per-invocation inputs rather than settings: `--prompt`, `--raw`,
`--config`. An unknown key is rejected (`unknown option --bogus`), also when it appears in the external file.

**Validation happens before the model loads.** `Settings.validate()` parses every typed value once (positive
`backend.ctx` / `backend.batch`, `server.port` in 0..65535, non-negative timeouts and byte limits, the
`sampling.*` ranges of [09 §9.10](09-sampling.md#910-where-each-setting-comes-from), the quantization, KV dtype and
backend names) and `Main` prints the message and exits with status 2, instead of failing minutes later when a
value is first used. Booleans are strict: `getBool` accepts `true/false`, `yes/no`, `on/off`, `1/0` and throws on
anything else, rather than treating a typo as `false`.

```sh
cp src/main/resources/jllm.properties jllm.properties      # then edit backend.ctx=2048 etc.
./jllm serve                                               # ./jllm.properties is picked up
./jllm serve --ctx 1024 --server.port 9000                 # flags override the file
./jllm config --ctx 1024                                   # show every value and where it came from:
#   backend.ctx   = 1024   (command line)
#   backend.batch = auto   (default)
```

The server uses `sampling.*` as the defaults for request fields that are absent (including `seed`) and
`chat.thinking` as the default for `enable_thinking`; `run`, `chat` and `bench` use them directly.

The `server.*` and `engine.*` keys, all of which this chapter has met:

| key | default | meaning |
|---|---|---|
| `server.host` / `server.port` | `127.0.0.1` / `8080` | listen address |
| `server.read-timeout-ms` | `30000` | per-read timeout and overall deadline for one request (408), 0 = none |
| `server.write-timeout-ms` | `10000` | how long a streamed answer may wait for a client that stopped reading before it is dropped and its generation cancelled, 0 = forever |
| `server.max-connections` | `256` | simultaneous connections; more wait in the accept queue |
| `server.max-body-bytes` | `4194304` | largest request body (413) |
| `server.queue-timeout-ms` | `60000` | how long a request waits for the engine before the 503, 0 = forever |
| `server.request-timeout-ms` | `0` | wall-clock limit for one generation, 0 = none; hit = `finish_reason: "length"` |
| `server.cors-origin` | empty | `Access-Control-Allow-Origin` value; empty = no CORS headers at all |
| `engine.max-prompt-bytes` | `0` | refuse longer prompts before tokenizing; 0 = 16 × `backend.ctx` |
| `engine.prompt-cache-mb` | `64` | memory for the ids of recently rendered chat prompts: a resent conversation is tokenized only from where it is new, with the ids of a full tokenization ([17 §17.5](17-agent-speedups.md#175-tokenizing-only-what-is-new)); 0 = off |
| `engine.speculative` | `ngram` | `off`, `ngram` (prompt lookup) or `draft` (draft model), [10.9](#109-speculative-decoding); flag `--speculative`. Left at the default, MoE models run without it |
| `engine.draft-model` | empty | Hub repo of the draft model for `draft` (same tokenizer); flag `--draft-model` |
| `engine.draft-tokens` | `16` | most tokens proposed per verification pass (prompt lookup adapts below it); flag `--draft-tokens` |
| `engine.ngram-max` / `engine.ngram-min` / `engine.ngram-window` | `8` / `3` / `0` | prompt lookup: longest / shortest suffix matched, how many recent tokens may be copied from (0 = all) |

---

## 10.11 `Main`: wiring it all together

| command | what it does |
|---|---|
| `download` | fetches the inference files of `model.repo` (refuses if they do not fit on disk; use `--low-disk` with `run`/`serve` instead) |
| `run --prompt "…"` | one chat-templated completion, streamed to stdout, statistics to stderr; `--raw` skips the template |
| `chat` | terminal REPL with history; `/clear`, `/quit`; `--think`, `--system` |
| `serve` | the HTTP API on `server.host:server.port` |
| `bench` | 413-token prompt, `bench.max-tokens` generated, `bench.rounds` rounds after a warm-up, prefix cache evicted between rounds |
| `config` | prints the effective settings |

`load()` is the assembly line: download metadata if needed → `Architecture.load` (a `Decoder` or a `MambaArch`,
[14 §14.12](14-model-families.md#1412-the-typed-model-architecture)) → `Tokenizer` → `ModelWeights.load`
(converting to the quantized cache on first use, with a `ShardProvider` that downloads shards on demand and
deletes them when `model.low-disk` is set) → the backend (`CpuBackend` / `MetalBackend`, or the Mamba variants
for a `MambaArch`; `backend=auto` picks Metal on macOS when the dylib is available) → `Engine`, given the EOS
ids from `Main.eosIds` and the queue timeout. A model family without a chat template fails right there, with the
backend and weights closed again.

---

## 10.12 Summary

- The engine is a loop: prefill the prompt in chunks, then sample → feed → sample, streaming complete UTF-8
  characters and holding back possible stop-string prefixes.
- The prefix cache turns multi-turn chat into incremental prefill by reusing the KV cache for the common prefix
  (always recomputing at least the last token).
- The HTTP server speaks just enough HTTP/1.1 — request parsing, keep-alive, `Content-Length`, chunked
  streaming — on virtual threads; SSE events ride inside chunked responses, written by a per-connection writer
  thread so that a stalled client costs a timeout, not the engine. Limits (body size, header size, read and
  write deadlines, queue wait) turn hostile or broken clients into plain JSON errors.
- The API mirrors OpenAI's chat and completion endpoints closely enough for the official SDKs and the bundled
  web page, including `n`, `logprobs` (log-softmax of the raw logits), `logit_bias`, `echo`, `suffix` (fill in the
  middle), `best_of` and `response_format` (constrained decoding: a byte-level JSON automaton masks the sampler's
  candidates, most likely first).
- Speculative decoding (off by default) proposes tokens by prompt lookup or with a small draft model, verifies them
  in one multi-token pass (`Backend.forwardAll`) and keeps what the target would have produced: identical greedy
  output, the target's distribution when sampling; it pays only when a pass of n tokens is much cheaper than n steps.
- Configuration is one properties file with three override layers and a `config` command to inspect them.
