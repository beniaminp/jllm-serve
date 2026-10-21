# 1. Overview: what we built, and how the pieces fit

This chapter is the map. It explains what a language model *does* (with numbers), how text flows through this
program from an HTTP request to the first streamed character, what each package is responsible for, and the
vocabulary the other chapters use. Nothing here needs prior machine-learning knowledge.

---

## 1.0 The toolbox: four ideas you need

### A model is a function

Strip away the mystique and a large language model (LLM) is one function:

```
float[] logits = model(int[] tokens)
```

Give it a list of integers (the text so far, encoded as **tokens**) and it returns one number per entry of its
**vocabulary** — 151 936 numbers for Qwen3 — each saying how plausible that entry is as the *next* token. The
numbers are called **logits** (raw scores; bigger = more likely). That is the entire interface. Chat, reasoning,
code generation: all of it is this function called in a loop.

### Tokens, not characters

The model never sees letters. A **tokenizer** cuts text into pieces from a fixed vocabulary and gives each piece
an id:

```
"Hello, world!"   →   ["Hello", ",", " world", "!"]   →   [9707, 11, 1879, 0]
```

Pieces are roughly words or word fragments; the leading space belongs to the word (`" world"` ≠ `"world"`).
Chapter [04](04-tokenizer.md) builds the tokenizer from scratch.

### Weights

`model()` is a fixed recipe of matrix multiplications ([05](05-transformer-math.md)) parameterized by ~600
million numbers, the **weights**, learned during training. We never train; we download the weights
([02](02-download.md)), read them ([03](03-safetensors.md)), shrink them ([06](06-quantization.md)) and run the
recipe as fast as possible ([07](07-cpu-backend.md), [08](08-metal-backend.md)). *Inference* = running the recipe.

### Sampling

Logits are scores, not a decision. A **sampler** ([09](09-sampling.md)) turns them into probabilities and draws
one token — deterministically (always the best) or with controlled randomness (so answers vary).

---

## 1.1 The next-token loop, by example

Everything the program does is this loop (`Engine.generate`):

```
tokens = tokenize(prompt)
repeat:
    logits = model(tokens)        # one forward pass
    next   = sample(logits)       # pick one id
    if next is an end-of-turn token: stop
    tokens.append(next)
    print(detokenize(next))
```

Watch it run for three steps on the prompt *"The capital of France is"*:

```
step 1   tokens = [785, 6722, 315, 9625, 374]                  "The" " capital" " of" " France" " is"
         logits = [-3.1, ..., 12.4 (" Paris"), ..., 7.9 (" a"), ..., 6.2 (" the"), ...]
         probabilities after softmax ≈  " Paris" 0.95   " a" 0.01   " the" 0.003  ...
         sampled: 12095 (" Paris")

step 2   tokens = [785, 6722, 315, 9625, 374, 12095]
         logits → "." 0.88   "," 0.09   " and" 0.01 ...
         sampled: 13 (".")

step 3   tokens = [..., 12095, 13]
         logits → <|im_end|> 0.97  ...
         sampled: 151645  = end-of-turn  → stop
```

Output: `" Paris."`. Three forward passes, three tokens. Two things to notice:

- the model only ever predicts **one token at a time**; "writing a paragraph" is hundreds of iterations;
- step 2 did not recompute everything about steps 1's tokens — the per-token state is kept in a **KV cache**, so
  each iteration costs about the same regardless of how long the text already is ([05 §5.9](05-transformer-math.md#59-prefill-decode-and-the-kv-cache)).

The real logits are produced by `Backend.forward`, the choice by `Sampler.sample`, the printing by
`StreamDecoder` + the text sink.

---

## 1.2 The pipeline and the data that flows through it

```
                         String                 int[]                   float[151936]          int
 HTTP request ──▶ ChatTemplate.render ──▶ Tokenizer.encode ──▶ Backend.forward ──▶ Sampler.sample ──┐
                                                                     ▲                               │
                                                                     └──────── feed the token back ──┘
                                                                                                     │
                            String ◀── StreamDecoder.push ◀── Tokenizer.tokenBytes ◀─── byte[] ◀─────┘
                               │
                               ▼
                    SSE "data: {...}" chunk on the socket
```

Reading left to right, with the Java type at each arrow:

| from → to | type | who | what happens |
|---|---|---|---|
| JSON body → prompt ids | `int[]` | `ApiServer.chatPrompt` → `Engine.renderChat` → `ChatTemplate.Format.renderTokens` | messages are laid out in the chat template of the model's family (ChatML for Qwen3) and tokenized; message text cannot inject control tokens |
| (plain text → ids) | `int[]` | `Tokenizer.encode` | `/v1/completions`, `run --raw`: regex split, BPE merges |
| ids → logits | `float[vocab]` | `Backend.forward` (`CpuBackend` or `MetalBackend`) | the transformer: 28 layers of attention + feed-forward, reading quantized weights |
| logits → id | `int` | `Sampler.sample` | penalties, temperature, top-k/top-p, random draw |
| id → bytes | `byte[]` | `Tokenizer.tokenBytes` | table lookup |
| bytes → text | `String` | `StreamDecoder.push` | only complete UTF-8 characters are released |
| text → wire | chunked HTTP | `Engine.TextSink` → `ApiServer.sse` → `HttpServer.Response.write` | one `data:` event per piece |

Before any of that can run, the **weights** must exist in memory. That is the loading path, executed once at
start-up (`Main.load`); the cache file is rebuilt automatically if it was made by an older converter or from
another revision ([03 §3.5](03-safetensors.md#35-from-safetensors-to-jllms-quantized-cache)):

```
Hub ──HuggingFaceHub──▶ models/<repo>/*.safetensors ──ModelConverter──▶ jllm-q4_0.bin ──QuantFile (mmap)──▶ ModelWeights ──▶ Backend
                       config.json ─────────────────▶ Architecture                                                  ▲
                       tokenizer.json ──────────────▶ Tokenizer ─────────────────────────────────────────────────────┘ (Engine)
```

---

## 1.3 Life of a request

What happens, in order, when a client sends `POST /v1/chat/completions` with `"stream": true`:

| # | class · method | step |
|---|---|---|
| 1 | `HttpServer.serve` → `connection` | waits for a free slot (`server.max-connections`), `ServerSocket.accept()` hands the socket to a **virtual thread** |
| 2 | `HttpServer.readRequest` | parses the request line, headers, reads `Content-Length` bytes of body (limits: `server.max-body-bytes`, `server.read-timeout-ms`) |
| 3 | `ApiServer.handle` → `completion` | routes on the path; `Json.parseObject` turns the body into a `Map` |
| 4 | `ApiServer.chatPrompt` → `Engine.renderChat` | `[{"role":"user","content":"hi"}]` becomes `<\|im_start\|>user\nhi<\|im_end\|>\n<\|im_start\|>assistant\n` for Qwen3 (plus the empty think block when thinking is off); other families use their own template; `Engine.preparePrompt` adds the BOS token where the family expects one |
| 5 | `ApiServer.params` | builds a `SamplingParams` from the request, falling back to `jllm.properties` defaults |
| 6 | `HttpServer.Response.startStream` | sends `200`, `Content-Type: text/event-stream`, `Transfer-Encoding: chunked` |
| 7 | `Engine.generate` | takes the engine lock (one KV cache = one request at a time; a request that waits longer than `server.queue-timeout-ms` gets a 503) |
| 8 | `Engine.generateLocked` | compares the ids with what is already in the KV cache (**prefix cache**), skips the shared prefix |
| 9 | `Backend.forward(prompt, pos, n, …)` in chunks of `backend.batch` | **prefill**: the new prompt tokens go through all 28 layers; their keys/values are written into the cache; logits only for the last token |
| 10 | `Sampler.sample(logits)` | first generated token |
| 11 | `Tokenizer.tokenBytes` → `StreamDecoder.push` | bytes → text, holding back incomplete UTF-8 |
| 12 | sink → `ApiServer.sse` → `Response.write` | `data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n` leaves the socket — **first byte to the client** |
| 13 | `Backend.forward(tok, pos, 1, …)` | **decode**: one token through the model, one more KV entry; back to 10 |
| 14 | end-of-turn id or `max_tokens` | final chunk with `finish_reason` and `usage`, then `data: [DONE]`; `Response.end` writes the terminating `0\r\n\r\n` |

Between 12 and 13 the socket is checked: if the client went away, or stopped reading for longer than
`server.write-timeout-ms`, the next write fails and the engine stops generating. Chapter [10](10-engine-and-server.md) covers the engine and server; [07](07-cpu-backend.md) and
[08](08-metal-backend.md) cover what happens inside step 9/13.

---

## 1.4 The models, in numbers

From `config.json` (`Architecture`):

| | Qwen3-0.6B (dense) | Qwen3-30B-A3B-Thinking-2507 (MoE) |
|---|---|---|
| `vocab_size` | 151 936 | 151 936 |
| `hidden_size` (`dim`) | 1024 | 2048 |
| `num_hidden_layers` | 28 | 48 |
| `num_attention_heads` / `num_key_value_heads` | 16 / 8 | 32 / 4 |
| `head_dim` | 128 | 128 |
| `intermediate_size` (FFN width) | 3072 | — (experts instead) |
| `num_experts` / `num_experts_per_tok` / `moe_intermediate_size` | — | 128 / 8 / 768 |
| `rope_theta` | 1 000 000 | 10 000 000 |
| `tie_word_embeddings` | true | false |
| parameters | ≈ 0.6 B | ≈ 30.5 B total, ≈ 3.3 B active per token |
| download (bf16) | 1.5 GB, 1 file | 61 GB, 16 shards |
| Q4_0 cache (`jllm-q4_0.bin`) | 452 MB | ≈ 20 GB |
| weights read per generated token | 452 MB | ≈ 2 GB (only the 8 chosen experts per layer) |

How the 0.6B parameter count arises: the embedding is 151 936 × 1024 ≈ 155.6 M; each of the 28 layers holds
≈ 15.7 M (attention 6.3 M + feed-forward 9.4 M) → 440 M; total ≈ 596 M. At 2 bytes each (bfloat16) that is
1.19 GB; the downloaded `model.safetensors` is 1.5 GB because it also stores `lm_head.weight`, a second copy of
the tied embedding that jllm never reads. After Q4_0 quantization the layers take 0.625 bytes per weight and the embedding 1.125 (Q8_0), giving
the 452 MB cache. The MoE model keeps 128 experts per layer in memory but runs only 8 of them per token
([12](12-mixture-of-experts.md)), which is why it is 40× larger on disk but only ~4× slower per token.

Qwen3 is the running example, but the same pipeline runs the other supported families: Qwen2 / 2.5 (and the
DeepSeek-R1 distills of Qwen2 and Llama), Llama 3.x, Mistral, GLM-4, Phi-3 / 3.5 / 4, Gemma 2 / 3 (text),
DeepSeek V2 / V3 (bf16, or FP8 dequantized while converting) and Mamba. What differs between them is parsed from
`config.json` once into a typed `Architecture` ([14](14-model-families.md), [15](15-mamba.md)); the backends never
look at the `model_type`. Weights can be stored as Q4_0 (default), Q8_0 or Q2_K (`model.quant`), and the KV cache
as f32 or f16 (`backend.kv-dtype`).

The "meaning" of the numbers: `dim` is how many features describe a token at every stage; `layers` is how many
times the token is refined; heads split attention into independent lookups; `vocab_size` sets the width of the
final score vector.

---

## 1.5 Why "from scratch"

The usual stack is PyTorch + a tokenizer library + a JSON library + a web framework + a CUDA/Metal runtime. Every
one of those is replaced here by a small Java class, so there is no layer you cannot read:

| normally | here | lines |
|---|---|---|
| `huggingface_hub` | `hub/HuggingFaceHub` on `java.net.http` | ~680 |
| `safetensors` | `safetensors/SafeTensors` on `FileChannel.map` + FFM | ~330 |
| `tokenizers` (Rust) + `jinja2` chat templates | `tokenizer/*` | ~2 300 |
| PyTorch / GGML kernels | `cpu/*` (`VectorKernels`, `Int8Kernels`, `Q2KKernels`, `CpuBackend`, ...) | ~2 750 |
| MLX / Metal Performance Shaders | `kernels.metal` + `jllm_metal.m` + `metal/*` | ~3 800 |
| Jackson / Gson | `json/Json` | ~450 |
| Spring / Netty / FastAPI | `server/HttpServer` + `ApiServer` | ~1 250 |

You can follow one request from the socket down to the individual multiply-add that computes an attention
score. Each piece is the simplest version that is still correct and reasonably fast; the docs point to the
non-obvious optimizations and to what a production system would add ([11](11-performance.md)).

---

## 1.6 Project layout and package responsibilities

```
org.robomiri.jllm
├── Main                 commands: download | run | chat | serve | bench | config
├── config.Settings      jllm.properties defaults → external file → command line flags
├── json.Json            recursive-descent JSON parser and writer
├── hub.HuggingFaceHub   lists a repo, downloads files (resumable, SHA-256 verified, retries), one shard on demand
├── safetensors.SafeTensors   memory-mapped tensor views over .safetensors files
├── tokenizer
│   ├── Tokenizer        BPE (byte-level and SentencePiece-style), Unigram, WordPiece: String ⇄ int[]
│   ├── Normalizers, PreTokenizers, Unigram, AddedTokenTrie, LongLongMap   the tokenizer.json pieces
│   ├── ChatTemplate     one prompt format per family (ChatML for Qwen3/Qwen2, Llama 3, Gemma, ...), thinking on/off
│   └── StreamDecoder    emits text only when a UTF-8 character is complete
├── model
│   ├── Architecture     sealed: Decoder | MambaArch, plus the shared Core (parsed from config.json)
│   ├── Families         one parser per model family, holding its defaults
│   ├── Decoder, DecoderLayer, LayerSpec   per-layer Attention (Gqa | Mla), Mlp (Dense | Moe), norms
│   ├── Attention, Mlp, Router, RopeTable, Scaling, MambaArch   the typed vocabulary
│   ├── LayerWeights, AttnWeights, MlpWeights   sealed weights mirroring LayerSpec (DecoderLayer | MambaLayer)
│   ├── QType, QTensor, Q2K   Q4_0 / Q8_0 / Q2_K block quantization, page-aligned off-heap layout
│   ├── KvDtype          f32 | f16 KV cache element type
│   ├── QuantFile        jllm-<quant>.bin/.json (+ .partial.json while converting): quantized weights on disk, memory-mapped
│   ├── ModelConverter   safetensors → QuantFile, one layer at a time, shards on demand, resumable
│   ├── ShardProvider    "give me shard N" / "I'm done with shard N" (used for --low-disk)
│   ├── ModelWeights     maps the cache file into QTensors (dense, MoE, MLA and Mamba layers)
│   ├── Rope, InterleavedRope, GemmaMath   position-embedding and family-specific math shared by the backends
│   └── Backend          interface: forward(tokens, off, n, pos, logits)
├── cpu
│   ├── VectorKernels, Int8Kernels, Q2KKernels   SIMD dot products on quantized rows (Vector API), int8 activations
│   ├── ThreadPool       spin-then-park fork/join for small parallel loops
│   ├── CpuBackend       the forward pass on CPU; KvCache holds the f32 or f16 KV cache on the heap
│   └── MlaCpu, MambaCpuBackend   DeepSeek latent attention and the Mamba recurrence
├── metal
│   ├── Metal            FFM bindings to libjllm_metal.dylib (compile shaders, buffers, dispatch)
│   ├── MetalBackend     the forward pass as GPU dispatches, zero-copy weights
│   └── MlaMetal, MambaMetalBackend   the same for latent attention and Mamba
├── sampling
│   ├── SamplingParams   temperature, top-k, top-p, min-p, penalties, seed, max tokens, stop
│   └── Sampler          float[] logits → int token
├── engine.Engine        the generation loop, prefix cache, stop strings, cancellation, statistics
└── server
    ├── HttpServer       HTTP/1.1 on ServerSocket + virtual threads, chunked streaming
    ├── ApiServer        OpenAI-compatible JSON endpoints, SSE streaming
    └── WebUi            a single-page chat client served at /
src/main/native/jllm_metal.m                               Objective-C shim: a C ABI over Metal
src/main/resources/org/robomiri/jllm/metal/kernels.metal   GPU compute kernels (compiled at start-up)
src/main/resources/jllm.properties                         all configurable settings, documented
```

Dependencies point downward only: `server` → `engine` → `sampling`/`tokenizer`/`model`; the two backends
depend on `model` and nothing depends on them except `Main`. `json` and `config` are leaves.

---

## 1.7 Where to go next

| you want to understand… | read |
|---|---|
| how the files get onto disk | [02](02-download.md) |
| what is *in* a weight file | [03](03-safetensors.md) |
| why `" world"` is id 1879 | [04](04-tokenizer.md) |
| the math inside `forward()` | [05](05-transformer-math.md) |
| why 4-bit weights work | [06](06-quantization.md) |
| how it runs fast on CPU / GPU | [07](07-cpu-backend.md), [08](08-metal-backend.md) |
| how a token is chosen | [09](09-sampling.md) |
| the loop and the HTTP layer | [10](10-engine-and-server.md) |
| what limits speed | [11](11-performance.md) |
| the 30B mixture-of-experts model | [12](12-mixture-of-experts.md) |
| 2-bit weights | [13](13-q2k-quantization.md) |
| Llama, Mistral, Qwen2, GLM, Phi, Gemma, DeepSeek: what differs between families | [14](14-model-families.md) |
| state-space models (Mamba) | [15](15-mamba.md) |

---

## Glossary

**Text and tokens**

- **Token** — a unit of text from the vocabulary (`"Hello"`, `" world"`, `"ing"`, `"!"`, or a single byte). Qwen3 has 151 936 slots, 151 669 of them in use.
- **Token id** — the integer index of a token in the vocabulary.
- **Vocabulary** — the fixed table of tokens the model can read and write.
- **BPE** (byte-pair encoding) — the algorithm that splits text into tokens by repeatedly merging frequent adjacent pairs.
- **Special token** — a control token such as `<|im_start|>`, `<|im_end|>` (end of turn), `<|endoftext|>`; never produced by splitting ordinary text.
- **Chat template** — the exact text layout (ChatML for Qwen3) in which conversations are presented to the model.
- **EOS** — end-of-sequence: a token id that tells the engine to stop generating.

**The model**

- **Weights / parameters** — the learned numbers; a model is "0.6B" when it has 0.6 billion of them.
- **Tensor** — a multi-dimensional array of numbers with a shape, e.g. `[2048, 1024]`.
- **Embedding** — the learned vector (1024 numbers) that represents a token at the model's input; also the matrix holding one such row per token.
- **Residual stream** — the vector `x` that flows through the layers; each layer *adds* its contribution to it.
- **Layer / block** — one repetition of attention + feed-forward; Qwen3-0.6B stacks 28.
- **RMSNorm** — rescales a vector to a standard size before each sub-block.
- **Attention** — the mechanism by which a token's vector reads from the vectors of earlier tokens, weighted by relevance.
- **Query / key / value** — the three vectors attention derives from each token: what I look for / what I offer / what I hand over.
- **Head** — attention is run several times in parallel on 128-number slices; each copy is a head.
- **GQA** (grouped-query attention) — several query heads share one key/value head (16 : 8 in Qwen3-0.6B), halving the KV cache.
- **RoPE** (rotary position embedding) — rotates query/key pairs by an angle proportional to position so attention can tell *where* tokens are.
- **Feed-forward network (FFN / MLP)** — the per-token two-matrix block after attention; holds most of the model's knowledge.
- **SwiGLU** — the FFN variant used here: `silu(gate) ⊙ up`, then a down projection.
- **Logits** — the raw, unnormalized score per vocabulary entry at the model's output.
- **Softmax** — turns scores into probabilities (positive, summing to 1).
- **MoE** (mixture of experts) — an FFN replaced by many small FFNs (**experts**) plus a **router** that picks a few per token (8 of 128 in Qwen3-30B-A3B).

**Running it**

- **Inference** — running the trained model (as opposed to training it).
- **Forward pass** — one evaluation of `model(tokens)`.
- **Prefill** — the forward pass over the whole prompt, many tokens per call.
- **Decode** — generating one token per forward call.
- **KV cache** — the keys and values of all previous tokens, stored per layer so they are not recomputed.
- **Context length (`ctx`)** — the maximum number of tokens the KV cache can hold. `backend.kv-dtype=f16` stores it in half precision (half the memory).
- **Prefix cache** — reusing KV entries across requests that share the same beginning (chat history).
- **Sampling** — choosing the next token from the probabilities; **temperature**, **top-k**, **top-p**, **min-p** shape the choice; **greedy** always takes the best.
- **Stop string** — text that ends generation when it appears in the output.

**Storage and numbers**

- **safetensors** — the file format of the downloaded weights: a JSON header plus raw tensor bytes.
- **bfloat16** — 16-bit float with float32's range and 3 significant digits; what the download uses.
- **Quantization** — storing weights with fewer bits (4 or 8 here) to cut memory and bandwidth.
- **Block** — 32 consecutive weights that share one **scale**; `value ≈ integer × scale`.
- **Q4_0 / Q8_0 / Q2_K** — the 4-bit, 8-bit and 2.625-bit block formats (same schemes as llama.cpp; Q2_K is for large models, [13](13-q2k-quantization.md)).
- **Quantized cache** — `jllm-<quant>.bin/.json` in the model directory: the weights quantized once, memory-mapped on every later start.
- **mmap** (memory mapping) — making a file appear as memory; the OS loads **pages** (16 KiB on Apple Silicon) on first access.
- **Arena** — the FFM object that owns native memory segments and frees them all when closed.
- **`MemorySegment`** — a bounds-checked view of native or mapped memory.

**Hardware**

- **SIMD** — one CPU instruction operating on several numbers at once; a **lane** is one of those numbers (4 lanes per 128-bit NEON register).
- **Vector API** — `jdk.incubator.vector`, Java's portable way to write SIMD code.
- **Memory bandwidth** — how many bytes per second RAM can deliver; the limit on decode speed.
- **FFM / Panama** — `java.lang.foreign`: calling C functions and reading native memory without JNI.
- **Metal** — Apple's GPU API; **MLX** is Apple's ML framework built on top of it. We use Metal directly.
- **Kernel** — a function run by thousands of GPU threads in parallel.
- **Threadgroup / simdgroup** — a block of GPU threads sharing fast memory / 32 threads executing in lock-step.
- **Dispatch** — one kernel launch with a grid of threadgroups.
- **Command buffer** — a recorded list of dispatches submitted to the GPU in one go (one per forward pass here).
- **Unified memory** — on Apple Silicon the CPU and GPU share RAM, so a weight buffer never needs copying.

**Serving**

- **HTTP/1.1 keep-alive / chunked encoding** — reusing a connection for several requests / sending a response whose length is not known in advance.
- **SSE** (server-sent events) — a streaming response of `data: …\n\n` lines; how tokens reach the client as they are produced.
- **Virtual thread** — a lightweight JDK thread; one per connection.
- **OpenAI-compatible API** — the `/v1/chat/completions` request/response shape that most clients speak.
