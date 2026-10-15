# 10. The engine and the server

Code: `engine/Engine.java`, `server/HttpServer.java`, `server/ApiServer.java`, `server/WebUi.java`, `Main.java`

## The generation loop (`Engine.generate`)

```
ids = tokenizer.encode(prompt)
reuse = length of the common prefix between ids and what is already in the KV cache   (prefix cache)
for chunk in ids[reuse:] in slices of maxBatch:  backend.forward(chunk, pos, logits only for the last)
for i in 0..maxTokens:
    tok = sampler.sample(logits)
    if tok in eosIds: finish = "stop"; break
    text = streamDecoder.push(tokenizer.tokenBytes(tok))      only complete UTF-8 characters
    hold back any suffix that could be the start of a stop string; emit the rest to the sink
    backend.forward([tok], pos, logits); pos++
```

### Prefix cache

A chat conversation re-sends the whole history every turn. Since the KV cache already holds keys/values for the
previous prompt, the engine compares the new token ids with the cached ones and skips prefill for the common
prefix (always recomputing at least the last token, because its logits are needed). In a multi-turn chat the
cost of turn `n` is then proportional to the *new* text only. `Stats.cachedTokens` reports how many were reused.

The backend has one KV cache, so `generate` is serialized with a lock; concurrent HTTP requests queue. (Batching
multiple sequences is the natural next step.)

### Stop strings

The API accepts `stop: ["\n\n", "User:"]`. Because a stop string may arrive split across tokens, the engine keeps
the last `maxStopLength − 1` characters of output unemitted until it is sure they are not a prefix of a stop
string; when a stop string appears, the text before it is emitted and generation ends with `finish_reason:
"stop"` (as OpenAI does, the stop string itself is not returned).

### Cancellation

The streaming handler passes a `BooleanSupplier`; if writing to the socket fails (client disconnected) the
engine stops at the next token instead of burning GPU time on an answer nobody will read.

## The HTTP server (`HttpServer`)

Written on `java.net.ServerSocket` with **one virtual thread per connection** (`Executors.newVirtualThreadPerTaskExecutor`),
so thousands of idle keep-alive connections cost almost nothing. It implements the subset of HTTP/1.1 needed:

- request line, headers, `Content-Length` and `Transfer-Encoding: chunked` bodies (with size limits);
- keep-alive, `Connection: close`;
- fixed-length responses (`send`) and **chunked streaming** responses (`startStream` / `write` / `end`) — each
  `write` is one HTTP chunk: hex length, CRLF, bytes, CRLF; a final `0\r\n\r\n` terminates;
- `Access-Control-Allow-Origin: *` so the web page can be served from elsewhere during development.

## The OpenAI-compatible API (`ApiServer`)

| Route | Behaviour |
|---|---|
| `POST /v1/chat/completions` | renders `messages` through the chat template; `enable_thinking` (top level or inside `chat_template_kwargs`) toggles Qwen3's `<think>` mode |
| `POST /v1/completions` | raw `prompt` |
| `GET /v1/models` | lists the loaded model |
| `GET /health` | `{"status":"ok"}` |
| `GET /` | the chat page |

Non-streaming responses return the usual `choices[0].message.content` plus a `usage` object extended with
`cached_tokens`, `prefill_ms`, `decode_ms` and `tokens_per_second`.

### Server-sent events

With `"stream": true` the response is `Content-Type: text/event-stream` and each generated piece is one event:

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Mars"},"finish_reason":null}]}

data: {"…","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{…}}

data: [DONE]
```

The `Engine`'s text sink writes each event straight to the chunked response, so the browser shows tokens as they
are produced. The same format is what the OpenAI SDKs and most chat front-ends expect, so
`openai.OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="x")` works unchanged.

### JSON

`json/Json.java` is a 300-line recursive-descent parser (objects → `LinkedHashMap`, arrays → `ArrayList`,
integers → `Long`, others → `Double`) plus a writer with proper string escaping and pattern-matching `switch`
over the value types. It parses the 11 MB `tokenizer.json` in well under a second.

## The web page (`WebUi`)

A single HTML string: a `<div>` log, an input, and ~30 lines of JavaScript that `fetch`es the streaming endpoint,
splits the body on blank lines, parses each `data:` line and appends `delta.content`. It keeps the message
history client-side and re-sends it every turn (which the prefix cache then makes cheap).

## Configuration (`config/Settings`, `jllm.properties`)

Every tunable is a key in the bundled `src/main/resources/jllm.properties` (with a comment explaining it):
model repo/revision/dir/quant, `backend`, `backend.ctx`, `backend.batch`, `backend.threads`, all sampling
defaults, `chat.thinking`, `server.host/port`, bench and hub settings. `Settings.load` layers three sources:

1. the bundled resource (defaults; also the list of *valid* keys — unknown flags are rejected with a message);
2. an external file: `--config <path>`, else `$JLLM_CONFIG`, else `./jllm.properties` if it exists;
3. command line flags, either short aliases (`--ctx 2048`, `--temp 0`, `--think`) mapped through
   `Settings.ALIASES`, or any key written out (`--server.port 9000`, `--bench.rounds=5`).

`./jllm config` prints every effective value and which layer set it. The server uses the sampling keys as the
defaults for requests that omit a field, and `chat.thinking` as the default for `enable_thinking`.

## `Main`

```
config    → prints the effective settings
download  → HuggingFaceHub.download
run       → one prompt, prints tokens as they stream, then statistics
chat      → REPL with history, /clear, /quit; --think, --system
serve     → ApiServer on --host/--port
bench     → 413-token prompt, 128 generated tokens, 3 rounds, prefix cache evicted between rounds
```

`load()` wires everything: `ModelConfig` → `Tokenizer` → `ModelWeights` (quantized) → backend (`metal` when the
dylib exists on macOS, else `cpu`) → `Engine` with the EOS ids from `generation_config.json`.
