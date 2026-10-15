# jllm-serve

An LLM inference engine and OpenAI-compatible server for **Qwen3** models (dense: `Qwen/Qwen3-0.6B` … `32B`;
Mixture-of-Experts: `Qwen/Qwen3-30B-A3B-*`), written in Java from scratch with **zero external dependencies** —
only the JDK (Vector API, FFM/Panama) plus a ~200-line Objective-C shim for Metal.

| Piece | Where | Written from scratch |
|---|---|---|
| Hugging Face Hub downloader (resumable, SHA-256 verified) | `hub/HuggingFaceHub.java` | yes, on `java.net.http` |
| safetensors reader (memory-mapped via FFM) | `safetensors/SafeTensors.java` | yes |
| Byte-level BPE tokenizer + Qwen chat template | `tokenizer/` | yes |
| Q4_0 / Q8_0 block quantization + mmap'd quantized cache file | `model/QTensor.java`, `model/QuantFile.java`, `model/ModelConverter.java` | yes |
| Transformer forward pass on CPU, dense + MoE (Vector API SIMD, custom thread pool) | `cpu/` | yes |
| Transformer forward pass on GPU, dense + MoE (Metal kernels, Panama FFM bindings) | `metal/`, `src/main/native/` | yes |
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
```

`--low-disk` downloads, quantizes and deletes the safetensors shards one at a time, so the 30B MoE (61 GB in
bf16, ~20 GB quantized) converts on a machine with ~30 GB free. See [docs/12](docs/12-mixture-of-experts.md).

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
--quant q4|q8         weight format (default q4; embeddings/LM head are always q8); cached per model
--low-disk            fetch + quantize + delete safetensors shards one by one (big models)
--ctx N               context length / KV cache size (default 4096)
--batch N             prefill chunk size (default 64)
--threads N           CPU threads
--temp, --top-k, --top-p, --min-p, --rep-penalty, --seed, --max-tokens, --think, --system "..."
--host, --port, --config <file>, --prompt "..." (run), --raw (run: no chat template)
```

## Measured on an Apple M1 Pro (10 cores), Qwen3-0.6B Q4_0, 413-token prompt

| Backend | Prefill | Decode |
|---|---|---|
| Metal | ~344 tok/s | ~75 tok/s (≈110 tok/s on short contexts) |
| CPU (NEON 128-bit, 10 threads) | ~153 tok/s | ~17 tok/s |

Both backends produce identical greedy output.

## Tests

```sh
mvn -q test-compile
$JAVA_HOME/bin/java --add-modules jdk.incubator.vector -cp target/classes:target/test-classes org.robomiri.jllm.SmokeTest
```

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
