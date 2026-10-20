# jllm-serve

An LLM inference engine and OpenAI-compatible server written in Java from scratch with **zero external
dependencies** — only the JDK (Vector API, FFM/Panama) plus a ~200-line Objective-C shim for Metal.
An optional browser console with model management ([`jllm-web/`](#web-console-jllm-web)) is a separate Spring Boot
module; the engine itself does not depend on it.

Supported families (Hugging Face `model_type`): **Qwen3** dense and MoE (`qwen3`, `qwen3_moe`), **Qwen2 / 2.5**
and the DeepSeek-R1 distills (`qwen2`), **Llama 3.x** (`llama`), **Mistral** (`mistral`), **GLM / GLM-4**
(`glm`, `glm4`), **Phi-3 / 3.5 / 4** (`phi3`), **Gemma 2 / 3** text (`gemma2`, `gemma3_text`, `gemma3`),
**DeepSeek V2 / V3** (`deepseek_v2`, `deepseek_v3`; bf16 checkpoints only) and **Mamba** (`mamba`).
Weights are quantized once to **Q4_0, Q8_0 or Q2_K** and memory-mapped afterwards; the KV cache can be f32 or f16.
llama.cpp **GGUF** files (`Qwen/Qwen3-0.6B-GGUF`, `unsloth/...-GGUF`, a local `.gguf`) run as they are, without a
conversion step: Qwen3 / Qwen3-MoE / Qwen2 / Llama architectures, Q4_0, Q8_0, Q4_K, Q5_K and Q2_K blocks used in
place from the memory-mapped file on both backends ([docs/16](docs/16-gguf.md)).

| Piece | Where | Written from scratch |
|---|---|---|
| Hugging Face Hub downloader (resumable, SHA-256 verified, retries, stall watchdog) | `hub/HuggingFaceHub.java` | yes, on `java.net.http` |
| safetensors reader (memory-mapped via FFM) | `safetensors/SafeTensors.java` | yes |
| Byte-level and SentencePiece-style BPE tokenizers + per-family chat templates | `tokenizer/` | yes |
| Typed model description (`Architecture`, `LayerSpec`, `RopeTable`, one parser per family) | `model/Architecture.java`, `model/Families.java`, ... | yes |
| Q4_0 / Q8_0 / Q2_K block quantization + versioned, mmap'd quantized cache file, shard-by-shard resumable conversion | `model/QTensor.java`, `model/Q2K.java`, `model/QuantFile.java`, `model/ModelConverter.java` | yes |
| GGUF reader (memory-mapped, split files), ggml block formats read in place (Q4_0, Q8_0, Q4_K, Q5_K; CPU + Metal kernels), architecture / tokenizer / tensor mapping from the file's metadata | `gguf/`, `model/Ggml.java`, `cpu/GgufKernels.java`, `tokenizer/GgufTokenizer.java` | yes |
| Forward pass on CPU: dense, MoE, MLA, Mamba (Vector API SIMD, int8 activations, custom thread pool, f32/f16 KV cache) | `cpu/` | yes |
| Forward pass on GPU: the same on Metal (fused kernels, Panama FFM bindings) | `metal/`, `src/main/native/` | yes |
| Sampler (temperature, top-k, top-p, min-p, penalties) | `sampling/` | yes |
| JSON parser/writer | `json/Json.java` | yes |
| HTTP/1.1 server (virtual threads, chunked SSE) + OpenAI API + web chat UI | `server/` | yes |

**Read [`docs/`](docs/README.md)** for a beginner-friendly walkthrough of how each part works, including the math.

## Requirements

- JDK **26** (uses `java.lang.foreign` and `jdk.incubator.vector`). On this machine: `export JAVA_HOME=/opt/homebrew/opt/openjdk@26`
- Maven (only used to compile and to run the tests; there are no dependencies to download besides build plugins)
- macOS + Xcode command line tools for the Metal backend (optional; the CPU backend works everywhere)

## Quick start

```sh
export JAVA_HOME=/opt/homebrew/opt/openjdk@26   # or any JDK 26+
./jllm download                      # fetches Qwen/Qwen3-0.6B (~1.5 GB) into ./models
./jllm run --prompt "Why is the sky blue?"
./jllm chat --think                  # interactive terminal chat, Qwen3 thinking mode on
./jllm serve --port 8080             # OpenAI-compatible API + chat page at http://127.0.0.1:8080/
./jllm bench --backend cpu           # throughput numbers
```

The first run of a model quantizes it once into `models/<model>/jllm-q4_0.bin` (+ `.json` index); later starts
memory-map that file and are ready in milliseconds. The index records the converter version and the Hub
repository, revision and commit the cache was made from: a cache from an older converter or another revision is
reconverted automatically, and an interrupted conversion resumes where it stopped (`jllm-q4_0.partial.json`).
Only the shards listed in the model's `model.safetensors.index.json` (or `model.safetensors`) are downloaded.

### Other models

```sh
./jllm chat --model Qwen/Qwen3-4B                               # any dense Qwen3 checkpoint
./jllm serve --model Qwen/Qwen3-30B-A3B-Thinking-2507 --low-disk --ctx 2048   # 128-expert MoE, 61 GB download
./jllm chat --model unsloth/Llama-3.2-1B-Instruct               # Llama 3 (ungated mirror)
./jllm chat --model Qwen/Qwen2.5-1.5B-Instruct --quant q8        # Qwen2.5 at 8 bits
./jllm chat --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B   # R1 distill (Qwen2 architecture)
./jllm chat --model THUDM/GLM-4-9B-0414 --backend.kv-dtype f16  # GLM-4 with a half-precision KV cache
./jllm run  --model state-spaces/mamba-130m-hf --raw --prompt "The capital of France is"
```

### GGUF files

```sh
./jllm run  --model unsloth/Qwen3-0.6B-GGUF --gguf-quant Q4_K_M --prompt "Why is the sky blue?"   # downloads that one file
./jllm chat --model Qwen/Qwen3-0.6B-GGUF                         # one .gguf in the repo: no selection needed
./jllm serve --model unsloth/Qwen3-0.6B-GGUF --gguf-file Qwen3-0.6B-Q8_0.gguf
./jllm chat --model ~/models/Llama-3.2-1B-Instruct-Q8_0.gguf     # a local file
```

A repository with several `.gguf` files needs `--gguf-quant` (a label such as `Q4_K_M`) or `--gguf-file` (exact name);
without either the error lists the files. Only the chosen file (and the `-0000i-of-0000N` parts of a split model) is
downloaded, with the usual resume and SHA-256 check. Nothing is converted or cached: the weights are read from the
mapped file. Tensors in a format the kernels cannot read directly (Q6_K for now, F16/BF16) are requantized to Q8_0 in
memory at load. See [docs/16](docs/16-gguf.md) for the file format, the llama q/k permutation, the zero-copy mapping
and the verification.

`--low-disk` downloads, quantizes and deletes the safetensors shards one at a time, so the 30B MoE (61 GB in
bf16, ~20 GB quantized) converts on a machine with ~30 GB free. See [docs/12](docs/12-mixture-of-experts.md).
Family details (what differs between them, verified tensor names, limits) are in
[docs/14](docs/14-model-families.md) and [docs/15](docs/15-mamba.md). Note: the Llama/Mistral/Qwen2/GLM/Phi/
Gemma/DeepSeek/Mamba paths were verified against reference implementations on synthetic checkpoints and on
tokenizer fixtures, not yet on the real published weights — run a real checkpoint before relying on them.

The `./jllm` launcher compiles the Java code and the native Metal library on first use. Equivalent manual steps:

```sh
mvn -q compile && ./scripts/build-native.sh
$JAVA_HOME/bin/java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
     -cp target/classes org.robomiri.jllm.Main serve
```

### API

```sh
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is 7*8?"}],
  "stream": true, "enable_thinking": true, "max_tokens": 300 }'
```

Endpoints: `POST /v1/chat/completions`, `POST /v1/completions` (both support `"stream": true`), `GET /v1/models`,
`GET /health`, `GET /` (chat UI). Sampling fields: `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`,
`presence_penalty`, `frequency_penalty`, `seed`, `max_tokens`, `stop`, `enable_thinking`.

## Web console (`jllm-web/`)

A separate Spring Boot module on top of the engine: a browser console to chat, manage models and change settings,
plus the same OpenAI-compatible API. It lives in its own Maven project so the engine keeps its zero-dependency
build; it consumes the engine as an ordinary jar (`org.robomiri:jllm-serve`, installed into the local Maven
repository by its launcher).

```sh
./jllm-web/run.sh                                   # builds both, console at http://127.0.0.1:8090/
./jllm-web/run.sh --server.port=9000 --jllm.web.api-key=secret
./jllm-web/run.sh --jllm.web.autoload=unsloth/Qwen3-0.6B-GGUF:Q4_K_M   # load models at startup
```

- **Chat**: streaming chat with any loaded model; conversations kept in the browser (searchable, renamable, grouped
  by date, exportable as Markdown); edit a message and resend, regenerate, copy; Markdown with tables and code blocks;
  Qwen3 thinking toggle (the `<think>` part is collapsible); per-chat sampling parameters and system prompt; tokens/s
  and a context-usage meter; light / dark theme.
- **Tools**: pick per chat which functions the model may call. Built in: `web_search` (DuckDuckGo with no key,
  or a SearXNG instance, or the Brave Search API), `fetch_url` (a page as readable text; private addresses refused
  unless allowed), `calculator` and `current_time`. **MCP servers** add their tools: stdio commands
  (`npx -y @modelcontextprotocol/server-filesystem ~/notes`, `uvx mcp-server-fetch`, ...), Streamable HTTP and
  legacy SSE servers, with headers / environment for credentials; Claude Desktop / Cursor `mcpServers` JSON can be
  pasted as it is. The server runs the calls, feeds the results back and lets the model continue (up to a set number
  of rounds); each call shows as a card with its arguments and result. A *try a tool* panel runs any tool by hand.
  Settings and servers are stored in `jllm-web-tools.json`.
- **Models**: paste a Hugging Face link — the model page, `.../tree/<revision>`, `.../blob/main/<file>.gguf`,
  `owner/name`, `owner/name:Q4_K_M` — or a local `.gguf` path, or pick a popular one; *Inspect* lists the
  repository's files (and the GGUF files to pick from), then *Download only* or *Load* with per-model options: id,
  backend, context length, KV cache type, weight quantization, batch, threads, speculative decoding / draft model,
  low-disk conversion. Downloads and loads run as background jobs with progress and cancel. Several models can be
  loaded at once; the `model` field of an API request picks one (with a single model loaded any name works). Loaded
  models show their settings, requests and speed and can be unloaded (running requests are cancelled first). Models
  on disk can be loaded again or deleted.
- **Settings**: every key of `jllm.properties` with its documentation, its default and where its value comes
  from. Changes are validated and stored in `jllm-web.properties` (on top of `$JLLM_CONFIG` /
  `./jllm.properties`, as for the CLI). Sampling and chat defaults apply to the next request; backend, context and
  weight settings to the next load.
- **Logs**: the engine's output (download progress, conversion, MCP server stderr, errors) live, with a filter.
- **API**: `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions` (same fields and streaming
  behaviour as `jllm serve`). Chat completions also accept OpenAI **function calling**: `tools` and `tool_choice`
  (`auto`, `none`, `required` or a named function); the model's calls come back as `tool_calls` with finish reason
  `tool_calls`, and `tool` messages carry the results back. `POST /api/chat` is the console's chat with server-side
  tools (an event stream). The management API: `/api/models/load|download|loaded|local`, `/api/hub/inspect`,
  `/api/jobs`, `/api/tools` (`/settings`, `/call`, `/mcp`), `/api/settings`, `/api/logs`, `/api/status`.

Function calling uses the Hermes / Qwen format (`<tools>` in the system prompt, `<tool_call>` blocks in the answer,
`<tool_response>` for results), which Qwen2.5 and Qwen3 were trained on; other families follow it less reliably, and
a 0.6B model picks the right tool for simple requests but not for long chains. MCP stdio servers run as child
processes with your user's rights, and the tools of an MCP server can do whatever that server allows (write files,
call APIs), so only add servers you trust and enable their tools in chats where you want them.

The console binds to `127.0.0.1` by default: the management API can load models and delete files. Set
`jllm.web.api-key` before exposing it; `/v1/**` and `/api/**` then need `Authorization: Bearer <key>`.
Application settings are in `jllm-web/src/main/resources/application.properties`; `cd jllm-web && mvn test` runs
its tests (no model needed).

## Configuration

Every setting lives in [`src/main/resources/jllm.properties`](src/main/resources/jllm.properties) with its
documentation. Precedence: bundled defaults → external file (`--config path`, `$JLLM_CONFIG`, or
`./jllm.properties` if present) → command line. `./jllm config` prints the effective values and their origin.
A `./jllm.properties` in the directory you run from is git-ignored, so it is a good place for personal overrides.

```sh
cp src/main/resources/jllm.properties jllm.properties   # edit: backend.ctx=2048, server.port=9000, ...
./jllm serve                                            # picks up ./jllm.properties automatically
./jllm serve --ctx 1024 --server.port 9000              # short aliases or full keys on the command line
./jllm serve --server.cors-origin=https://app.example.com   # --key=value also works
```

Settings are validated strictly, before any weights are loaded: an unknown key, a non-numeric or out-of-range
number, a malformed boolean (booleans are `true|false|yes|no|on|off|1|0`; a stray `--think flase` is an error, not
`false`) or an unknown `model.quant` / `backend.kv-dtype` / `backend` value stops the program with a message and
exit status 2. `prompt`, `raw` and `config` are command line flags only and are rejected in a properties file.
Notable keys besides the usual ones: `backend.kv-dtype` (`f32` or `f16`), `backend.cpu.int8-activations`
(default `true`; the CPU quantizes activations to int8 for integer dot products, several times faster decode but
~1 % different numerics; `false` gives results identical to Metal), `engine.max-prompt-bytes`, `hub.timeout-ms`
and the `server.*` limits listed under Security below.

Short flags:

```
--model REPO|FILE    Hugging Face repository (default Qwen/Qwen3-0.6B) or a local .gguf file, --revision REV, --dir PATH (model directory)
--backend cpu|metal   default: metal on macOS when the native lib builds, else cpu
--quant q4|q8|q2k|q6k weight format (default q4; q2k = 2.625-bit Q2_K, q6k = 6.5625-bit Q6_K); cached per model
--head-quant q8|q6k   format of the embedding / LM head (default q8; q6k = Q6_K, a separate jllm-q4_0-h6k cache)
--gguf-quant LABEL    GGUF repository: the file whose name carries LABEL (Q4_K_M, Q8_0, ...); --gguf-file NAME: exact file name
--low-disk            fetch + quantize + delete safetensors shards one by one (big models)
--ctx N               context length / KV cache size (default 4096)
--batch N             prefill chunk size (default 64)
--backend.kv-dtype f32|f16   KV cache element type (no short alias)
--threads N           CPU threads
--temp, --top-k, --top-p, --min-p, --rep-penalty, --seed, --max-tokens, --think, --system "..."
--host, --port, --config <file>, --prompt "..." (run), --raw (run: no chat template)
```

## Security

The server is meant for your own machine. Keep these in mind before changing the defaults:

- It binds to `127.0.0.1`; `server.host=0.0.0.0` exposes it to everyone on the network.
- There is no authentication. Anyone who can reach the port can run completions and keep the single engine
  busy; `Authorization` headers are accepted and ignored.
- `server.cors-origin=*` lets any web page open in a browser on a machine that can reach the server call the API
  (with `server.host=0.0.0.0`, that is any page on any machine on the network). Set it to the exact origin of
  your own web app or leave it empty.
- Requests have limits (`server.max-body-bytes`, `engine.max-prompt-bytes`, `server.max-connections`,
  `server.queue-timeout-ms`, `server.request-timeout-ms`), a request must arrive within
  `server.read-timeout-ms` of its first byte (408 otherwise) and a streaming client that stops reading is
  dropped after `server.write-timeout-ms`. For anything beyond localhost put a reverse proxy in front for TLS,
  authentication and request buffering.

## Measured on an Apple M1 Pro (10 cores, 32 GB)

| Model | Backend | Prefill | Decode |
|---|---|---|---|
| Qwen3-0.6B Q4_0, 413-token prompt | Metal | ~533 tok/s | ~111 tok/s |
| Qwen3-0.6B Q4_0, 413-token prompt | CPU (NEON 128-bit, 10 threads), int8 activations (`backend.cpu.int8-activations`, the default) | ~275 tok/s | ~51 tok/s |
| Qwen3-30B-A3B Q4_0 (19.4 GB cache) | Metal | ~49 tok/s | ~25 tok/s |
| Qwen3-30B-A3B Q4_0 (19.4 GB cache) | CPU, int8 activations | ~12 tok/s | ~6 tok/s |

Float activations on the CPU (`--backend.cpu.int8-activations false`) decode about 4× slower; the full history of
measurements is in [docs/11](docs/11-performance.md#111-measured).

Both backends produce identical greedy output with float activations on the CPU (checked on every merge for
Qwen3 and the 30B MoE, and across 20 synthetic models of all families × 3 quantization formats by
`scripts/arch-smoke.sh`, which runs the CPU side with `backend.cpu.int8-activations` off); int8 activations
change the CPU decode numerics by ~1 % per dot product, so greedy output may differ from Metal's after a while.
The MLA latent KV cache of DeepSeek models is much smaller than a plain one (V2-Lite: 553 KB → 62 KB per token,
[docs/14](docs/14-model-families.md)).

## Tests

```sh
mvn test                   # SmokeTest + ServerTest + MetalParamsTest + the GGUF checks (plain main() classes, no test framework)
sh scripts/arch-smoke.sh   # 20 synthetic models x Q4/Q8/Q2_K, CPU == Metal; DeepSeek checked against a float64 reference
```

`mvn test` needs `models/Qwen_Qwen3-0.6B` (run `./jllm download` first); `ServerTest` checks the HTTP server
against a fake backend (settings validation, bad requests, keep-alive, slow, stalled and disconnected clients). Two more checks are run by hand:

```sh
CP=target/classes:target/test-classes
J="$JAVA_HOME/bin/java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -cp $CP org.robomiri.jllm"
$J.TokenizerFamiliesTest /tmp/jllm-tokenizers   # 3 083 checks, 14 tokenizer families
$J.ConverterTest --dir <scratch dir>            # converter, resume and cache-invalidation tests
```

`TokenizerFamiliesTest` compares the tokenizers and chat templates of all families against fixtures generated
with the reference `tokenizers`/`jinja2` libraries (`src/test/python/gen_tokenizer_fixtures.py`).
`ConverterTest` deletes and rewrites cache files, so give it private copies of the models in a scratch directory
(see the comment at the top of `src/test/java/org/robomiri/jllm/ConverterTest.java`), never `models/`.

Launcher environment: `JAVA_HOME`, `JLLM_XMX` (heap, default `4g`; the CPU KV cache lives on the heap),
`JAVA_OPTS` (extra JVM flags), `JLLM_NO_BUILD=1` to skip the incremental compile.

## Layout

```
src/main/java/org/robomiri/jllm/
  Main.java              CLI: download | run | chat | serve | bench | config
  ModelLoader.java       resolve / download / load a model as the settings describe (used by the CLI and jllm-web)
  config/                layered settings (jllm.properties → external file → flags)
  json/                  JSON parser & writer
  hub/                   Hugging Face downloader
  safetensors/           .safetensors reader (mmap + FFM)
  tokenizer/             BPE tokenizers, per-family chat templates, streaming UTF-8 decoder
  model/                 typed Architecture + per-family parsers, quantized tensors, cache file + converter, weights, RoPE, Backend interface
  cpu/                   Vector API kernels (float and int8 activations), thread pool, KV cache, CPU forward pass
  metal/                 FFM bindings + Metal forward pass
  sampling/              sampling parameters and sampler
  engine/                generation loop, prefix cache, stop strings
  server/                HTTP server, OpenAI-compatible API, web UI
src/main/native/jllm_metal.m                       Objective-C shim exposing a C ABI over Metal
src/main/resources/org/robomiri/jllm/metal/kernels.metal   GPU compute kernels
docs/                    how it all works
scripts/                 build-native.sh, arch-smoke.sh and the synthetic-model generators
jllm-web/                Spring Boot web console + OpenAI API (separate Maven project, depends on the engine jar)
  src/main/java/org/robomiri/jllm/web/
    models/              Hugging Face links, download / load jobs, loaded models, models on disk
    chat/                /api/chat (the agent loop) and what it shares with the OpenAI endpoints
    tools/               tool-call prompt format and parsing, built-in tools (search, fetch, calculator, time)
    mcp/                 MCP client: stdio, Streamable HTTP and SSE transports, server manager
    openai/              /v1 endpoints
    settings/            jllm.properties view and persisted overrides
    api/                 /api management and tools endpoints, API key filter
    logs/                captured engine output
  src/main/resources/static/   the console (plain HTML/CSS/JS modules, no build step)
```

## License

MIT, see [LICENSE](LICENSE).
