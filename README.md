# jllm-serve

An LLM inference engine and OpenAI-compatible server written in Java from scratch with **zero external
dependencies** — only the JDK (Vector API, FFM/Panama) plus a ~200-line Objective-C shim for Metal.

Supported families (Hugging Face `model_type`): **Qwen3** dense and MoE (`qwen3`, `qwen3_moe`), **Qwen2 / 2.5**
and the DeepSeek-R1 distills (`qwen2`), **Llama 3.x** (`llama`), **Mistral** (`mistral`), **GLM / GLM-4**
(`glm`, `glm4`), **Phi-3 / 3.5 / 4** (`phi3`), **Gemma 2 / 3** text (`gemma2`, `gemma3_text`, `gemma3`),
**DeepSeek V2 / V3** (`deepseek_v2`, `deepseek_v3`; bf16 checkpoints only) and **Mamba** (`mamba`).
Weights are quantized once to **Q4_0, Q8_0 or Q2_K** and memory-mapped afterwards; the KV cache can be f32 or f16.

| Piece | Where | Written from scratch |
|---|---|---|
| Hugging Face Hub downloader (resumable, SHA-256 verified) | `hub/HuggingFaceHub.java` | yes, on `java.net.http` |
| safetensors reader (memory-mapped via FFM) | `safetensors/SafeTensors.java` | yes |
| Byte-level and SentencePiece-style BPE tokenizers + per-family chat templates | `tokenizer/` | yes |
| Q4_0 / Q8_0 / Q2_K block quantization + mmap'd quantized cache file, shard-by-shard conversion | `model/QTensor.java`, `model/Q2K.java`, `model/QuantFile.java`, `model/ModelConverter.java` | yes |
| Forward pass on CPU: dense, MoE, MLA, Mamba (Vector API SIMD, custom thread pool, f32/f16 KV cache) | `cpu/` | yes |
| Forward pass on GPU: the same on Metal (fused kernels, Panama FFM bindings) | `metal/`, `src/main/native/` | yes |
| Sampler (temperature, top-k, top-p, min-p, penalties) | `sampling/` | yes |
| JSON parser/writer | `json/Json.java` | yes |
| HTTP/1.1 server (virtual threads, chunked SSE) + OpenAI API + web chat UI | `server/` | yes |

**Read [`docs/`](docs/README.md)** for a beginner-friendly walkthrough of how each part works, including the math.

## Requirements

- JDK **26** (uses `java.lang.foreign` and `jdk.incubator.vector`). On this machine: `export JAVA_HOME=/opt/homebrew/opt/openjdk@26`
- Maven (only used to compile; there are no dependencies to download besides compiler plugins)
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

The first run of a model quantizes it once into `models/<model>/jllm-q4_0.bin`; later starts memory-map that file
and are ready in milliseconds.

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

## Configuration

Every setting lives in [`src/main/resources/jllm.properties`](src/main/resources/jllm.properties) with its
documentation. Precedence: bundled defaults → external file (`--config path`, `$JLLM_CONFIG`, or
`./jllm.properties` if present) → command line. `./jllm config` prints the effective values and their origin.

```sh
cp src/main/resources/jllm.properties jllm.properties   # edit: backend.ctx=2048, server.port=9000, ...
./jllm serve                                            # picks up ./jllm.properties automatically
./jllm serve --ctx 1024 --server.port 9000              # short aliases or full keys on the command line
```

Short flags:

```
--backend cpu|metal   default: metal on macOS when the native lib builds, else cpu
--quant q4|q8|q2k     weight format (default q4; q2k = 2.625-bit Q2_K; embeddings/LM head are always q8); cached per model
--low-disk            fetch + quantize + delete safetensors shards one by one (big models)
--ctx N               context length / KV cache size (default 4096)
--batch N             prefill chunk size (default 64)
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
| Qwen3-0.6B Q4_0, 413-token prompt | Metal | ~530 tok/s | ~110 tok/s |
| Qwen3-0.6B Q4_0 | CPU (NEON 128-bit, 10 threads), float activations | ~153 tok/s | ~17 tok/s |
| Qwen3-0.6B Q4_0 | CPU, int8 activations (`backend.cpu.int8-activations`, the default) | same | ~4× the float row: 13.4 → 53.7 tok/s in the same session ([docs/11](docs/11-performance.md#111-measured)) |
| Qwen3-30B-A3B Q4_0 (19.4 GB cache) | Metal | ~49 tok/s | ~25 tok/s |

Both backends produce identical greedy output with float activations on the CPU (checked on every merge for
Qwen3 and the 30B MoE, and across 20 synthetic models of all families × 3 quantization formats by
`scripts/arch-smoke.sh`, which runs the CPU side with `backend.cpu.int8-activations` off); int8 activations
change the CPU decode numerics by ~1 % per dot product.

## Tests

```sh
mvn test                   # SmokeTest (plain main(), no test framework) against models/Qwen_Qwen3-0.6B
sh scripts/arch-smoke.sh   # every family on synthetic tiny checkpoints, Q4/Q8/Q2_K, CPU == Metal
```

`TokenizerFamiliesTest` compares the tokenizers and chat templates of all families against fixtures generated
with the reference `tokenizers`/`jinja2` libraries (`src/test/python/gen_tokenizer_fixtures.py`).

Launcher environment: `JAVA_HOME`, `JLLM_XMX` (heap, default `4g`; the CPU KV cache lives on the heap),
`JAVA_OPTS` (extra JVM flags), `JLLM_NO_BUILD=1` to skip the incremental compile.

## Layout

```
src/main/java/org/robomiri/jllm/
  Main.java              CLI: download | run | chat | serve | bench | config
  config/                layered settings (jllm.properties → external file → flags)
  json/                  JSON parser & writer
  hub/                   Hugging Face downloader
  safetensors/           .safetensors reader (mmap + FFM)
  tokenizer/             BPE tokenizer, chat template, streaming UTF-8 decoder
  model/                 config, quantized tensors, cache file + converter, weight loading, RoPE, Backend interface
  cpu/                   Vector API kernels, thread pool, CPU forward pass
  metal/                 FFM bindings + Metal forward pass
  sampling/              sampling parameters and sampler
  engine/                generation loop, prefix cache, stop strings
  server/                HTTP server, OpenAI-compatible API, web UI
src/main/native/jllm_metal.m                       Objective-C shim exposing a C ABI over Metal
src/main/resources/org/robomiri/jllm/metal/kernels.metal   GPU compute kernels
docs/                    how it all works
```
