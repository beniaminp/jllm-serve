# 10. The engine and the server — generation loop, HTTP, OpenAI API, configuration

Code: `engine/Engine.java`, `server/HttpServer.java`, `server/ApiServer.java`, `server/WebUi.java`,
`config/Settings.java`, `src/main/resources/jllm.properties`, `Main.java`.

Everything before this chapter computes *one* next-token distribution. This chapter is about the loop around
it, and about getting text in and out over the network with nothing but `java.net`.

---

## 10.1 The generation loop

`Engine.generate(prompt, params, sink, cancelled)` is the whole decode algorithm. In pseudo-code, with the
matching line in `generateLocked` on the right:

```
ids = tokenizer.encode(prompt)                              # text → token ids (04)
if ids is empty: ids = [EOS]                                # the model must see at least one token
acquire the lock                                            # one KV cache ⇒ one request at a time
if len(ids) >= ctx: error "prompt too long"

reuse = length of common prefix(ids, cached), capped at len(ids) − 1      # 10.2
pos = reuse
while pos < len(ids):                                       # PREFILL in chunks of maxBatch (64)
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

- **Logits only when needed.** During prefill only the last chunk asks for logits; the other chunks exist
  purely to fill the KV cache ([05 §5.9](05-transformer-math.md#59-prefill-decode-and-the-kv-cache)).
- **One forward per generated token.** The token sampled at step `i` is fed at position `pos` to obtain the
  logits for step `i + 1`. The very last sampled token is *not* fed (nobody needs its successor's logits).
- **`finish_reason`** is `"stop"` (EOS or a stop string), `"length"` (`max_tokens` or the context is full) or
  `"cancelled"`.
- **Timing.** `t0 → t1` is prefill, `t1 → t2` is decode; `Stats` reports both plus `tokPerSec()`.

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

- **Cancellation.** `cancelled` is a `BooleanSupplier` polled before every token. `HttpServer` watches every
  connection while its request is handled (a virtual thread blocks reading the socket, which returns EOF the
  moment the client closes) and `ApiServer` ORs that flag into `cancelled` for both the streaming and the
  non-streaming path, so a client that closes its tab stops burning GPU time within one token. A streaming
  client that stops *reading* is cut off after `server.write-timeout-ms` (10.5).
- **The lock.** The backend owns exactly one KV cache, so `generate` is wrapped in `synchronized (lock)`.
  Concurrent HTTP requests queue up and are served one at a time; virtual threads make the waiting cheap.
  Serving many users at once would require batching several sequences per forward pass ([11](11-performance.md)).
- **Statistics** (`Engine.Stats`): `promptTokens`, `cachedTokens`, `generatedTokens`, `prefillMs`, `decodeMs`,
  `tokPerSec()`. The CLI prints them after each answer; the API returns them inside `usage`.

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
| `Content-Length` body | read exactly that many bytes (≤ 32 MB) | `readRequest` → `in.readNBytes(len)` |
| chunked request body | rare for clients, supported anyway | `readChunked` |
| keep-alive | after a response, loop and read the next request on the same socket unless `Connection: close` | `connection()` loop |
| fixed response | status + headers + `Content-Length` + body in one write | `Response.send` |
| streaming response | see below | `Response.startStream` / `write` / `end` |
| one thread per connection | `Executors.newVirtualThreadPerTaskExecutor()`: a blocked virtual thread costs ~1 KB, so thousands of idle keep-alive connections are fine | `serve()` |

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

`Response.write(chunk)` emits one such chunk and flushes, so the client receives it immediately; `end()` sends
`0\r\n\r\n`.

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
| `/` | GET | the chat page (`WebUi`) |
| anything else | — | 404 JSON error; non-POST on a completion route → 405 |

### Chat, non-streaming

Request:

```json
{"messages": [{"role": "system", "content": "You are terse."},
              {"role": "user",   "content": "Name three planets."}],
 "max_tokens": 60, "temperature": 0.6, "enable_thinking": false}
```

`chatPrompt()` renders the messages through `ChatTemplate` (multi-part `content` arrays are flattened to their
`text` parts; `enable_thinking` may also be given inside `chat_template_kwargs`), `params()` merges the
request's sampling fields over the server's defaults, and `engine.generate` runs. Response:

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
`tokens_per_second`), which OpenAI clients ignore.

### Chat, streaming (`"stream": true`)

The response is `text/event-stream`, chunked. First an event with the role, then one event per emitted text
piece, then a closing event with `finish_reason` and `usage`, then `[DONE]`:

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk",…,"choices":[{"index":0,"delta":{"content":"Mercury"},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk",…,"choices":[{"index":0,"delta":{"content":", Venus"},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk",…,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{…}}

data: [DONE]
```

The engine's text sink writes each event directly, so tokens appear in the client as they are produced. If a
write fails (client gone) the sink records the error and the `cancelled` supplier stops the engine.

### Completions

`{"prompt": "The three primary colors are red,", "max_tokens": 40, "temperature": 0, "stop": ["."]}` →

```json
{"id": "cmpl-…", "object": "text_completion", "created": …, "model": "Qwen/Qwen3-0.6B",
 "choices": [{"index": 0, "text": " yellow, and blue", "finish_reason": "stop"}],
 "usage": {…}}
```

Streaming uses the same `text_completion` objects with a `text` field per event.

### Errors

```json
{"error": {"message": "invalid JSON: JSON expected object key at offset 1", "type": "invalid_request_error"}}
```

400 for malformed JSON, missing `messages`, or invalid sampling values; 404 unknown route; 405 wrong method.
An unexpected exception inside a handler produces a plain-text 500 from `HttpServer` (and, if the stream had
already started, the connection is dropped) — a prompt longer than the context currently ends up here rather
than as a 400, a known gap.

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

`json/Json.java` (≈300 lines) is a **recursive-descent** parser: `value()` looks at the next character and
calls `object()`, `array()`, `string()`, `number()` or a literal; objects and arrays call `value()` for their
children. Types map to plain Java: object → `LinkedHashMap<String,Object>` (keeps key order), array →
`ArrayList<Object>`, integers → `Long`, other numbers → `Double`, plus `String`, `Boolean`, `null`. The writer
is a pattern-matching `switch` over those types with proper escaping (`"`, `\`, control characters, U+2028/9).
Typed helpers (`Json.str`, `Json.num`, `Json.map`, …) return defaults instead of throwing, which keeps the API
handlers short. It parses the 11 MB `tokenizer.json` in well under a second.

---

## 10.8 The web page

`WebUi.HTML` is one string: a message log, an input box, a "enable thinking" checkbox and ~30 lines of
JavaScript. On submit it `fetch`es `/v1/chat/completions` with `stream: true`, reads the body with a
`ReadableStream` reader, splits on blank lines, parses each `data:` line and appends `delta.content` to the
current answer. The history lives in the page and is re-sent each turn, which the prefix cache (10.2) makes
cheap. After each answer it prints the token count and tok/s from the final `usage` event.

---

## 10.9 Configuration

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
`--min-p`, `--rep-penalty`, `--seed`, `--max-tokens`, `--think`, `--system`, `--host`, `--port`) or the
property key itself (`--server.port 9000`, `--bench.rounds=5`). A flag without a value means `true`
(`--think`). Three flags are per-invocation inputs rather than settings: `--prompt`, `--raw`, `--config`.
An unknown short flag is rejected (`unknown option --bogus`).

```sh
cp src/main/resources/jllm.properties jllm.properties      # then edit backend.ctx=2048 etc.
./jllm serve                                               # ./jllm.properties is picked up
./jllm serve --ctx 1024 --server.port 9000                 # flags override the file
./jllm config --ctx 1024                                   # show every value and where it came from:
#   backend.ctx   = 1024   (command line)
#   backend.batch = 64     (default)
```

The server uses `sampling.*` as the defaults for request fields that are absent and `chat.thinking` as the
default for `enable_thinking`; `run`, `chat` and `bench` use them directly.

---

## 10.10 `Main`: wiring it all together

| command | what it does |
|---|---|
| `download` | fetches the inference files of `model.repo` (refuses if they do not fit on disk; use `--low-disk` with `run`/`serve` instead) |
| `run --prompt "…"` | one chat-templated completion, streamed to stdout, statistics to stderr; `--raw` skips the template |
| `chat` | terminal REPL with history; `/clear`, `/quit`; `--think`, `--system` |
| `serve` | the HTTP API on `server.host:server.port` |
| `bench` | 413-token prompt, `bench.max-tokens` generated, `bench.rounds` rounds after a warm-up, prefix cache evicted between rounds |
| `config` | prints the effective settings |

`load()` is the assembly line: download metadata if needed → `ModelConfig` → `Tokenizer` → `ModelWeights.load`
(converting to the quantized cache on first use, with a `ShardProvider` that downloads shards on demand and
deletes them when `model.low-disk` is set) → `CpuBackend` or `MetalBackend` (`backend=auto` picks Metal on
macOS when the dylib is available) → `Engine` with the EOS ids from `generation_config.json` and the tokenizer.

---

## 10.11 Summary

- The engine is a loop: prefill the prompt in chunks, then sample → feed → sample, streaming complete UTF-8
  characters and holding back possible stop-string prefixes.
- The prefix cache turns multi-turn chat into incremental prefill by reusing the KV cache for the common prefix
  (always recomputing at least the last token).
- The HTTP server speaks just enough HTTP/1.1 — request parsing, keep-alive, `Content-Length`, chunked
  streaming — on virtual threads; SSE events ride inside chunked responses.
- The API mirrors OpenAI's chat and completion endpoints closely enough for the official SDKs and the bundled
  web page.
- Configuration is one properties file with three override layers and a `config` command to inspect them.
