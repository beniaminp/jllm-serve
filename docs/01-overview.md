# 1. Overview: what we built, and how the pieces fit

This chapter is the map. It explains what a language model *does* (with numbers), how text flows through this
program from an HTTP request to the first streamed character, what each package is responsible for (including the
parts added later: GGUF models, pictures, speech, image and video generation, speculation, constrained tool calls,
quantized caches), and the vocabulary the other chapters use. Nothing here needs prior machine-learning knowledge.

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

### 1.2.1 The whole map: what sits around the text loop

The loop above is the heart; this is everything attached to it. Each box is a package, each arrow a Java call or a file, and the chapter that
explains it is in the table below the picture.

```
 clients                       server/                        engine/                          model/
 ┌──────────────────┐ HTTP  ┌────────────────────┐ tokens ┌───────────────────────────┐  ids  ┌─────────────────────────────┐
 │ OpenAI SDKs      │ ────▶ │ ApiServer          │ ─────▶ │ Engine                    │ ────▶ │ Backend                     │
 │ coding agents    │       │  chat, completions │        │  prefix cache, slots      │       │  CpuBackend │ MetalBackend  │
 │ curl             │ ◀──── │ AudioRoutes        │ ◀───── │  Speculator: n-gram,      │ ◀──── │  KV cache f32 f16 q8 q4     │
 │ jllm-web console │  SSE  │ ImageHttp          │  text  │   draft model, MTP        │logits │  weights: .bin cache / .gguf│
 └──────────────────┘       │ VideoHttp          │        │  Sampler + JsonConstraint │       └─────────────────────────────┘
                            └───┬────────────┬───┘        │   / ToolCallConstraint    │
                                │            │            └────────────▲──────────────┘
                     text prompt│            │ speech, images, video   │ picture embeddings replace placeholder tokens
                                ▼            ▼                         │
                  ChatTemplate · Tokenizer   audio/ · diffusion/       vision/ (VisionContext)
                  (BPE, SentencePiece,       Whisper, OuteTTS, UNet,   CLIP / ViT encoders
                   GGUF pre-types)           VAE, AnimateDiff, Wan     (CPU and Metal each)

 weights on disk (hub/ downloads them):   *.safetensors ──ModelConverter──▶ jllm-q4_0.bin (Q4_0 Q8_0 Q2_K Q6_K, memory-mapped)
                                          *.gguf ──GgufModel──▶ 15 block types used in place (Q4_0 .. MXFP4, F16, BF16)
                                          mmproj-*.gguf (vision) · Whisper · OuteTTS · diffusers folders · Wan 2.1
```

| subsystem | what it adds to the path | package · main classes | chapter |
|---|---|---|---|
| GGUF | llama.cpp's single-file models run as they are: the metadata becomes an `Architecture`, the tensors are views of the mapped file in 15 block types, the tokenizer comes from the file | `gguf/` (`GgufModel`, `GgufFile`, `GgufWeights`, `GgufFamilies`), `model/Ggml.java`, `tokenizer/GgufPre`, `GgufTokenizer`, `SpmGreedy` | [16](16-gguf.md), [06 §6.12](06-quantization.md#612-the-other-ggml-block-formats-read-in-place-from-gguf-files) |
| native block types | the CPU kernels for the formats above (Vector API float, Java int8, native NEON / dot-product C), the Metal `gg_unit` decoders | `cpu/GgufKernels`, `GgufInt8Kernels`, `NativeCpu`, `src/main/native/jllm_cpu*.c`; `kernels.metal` | [07 §7.10](07-cpu-backend.md#710-int8-activations-integer-dot-products-int8kernels), [08 §8.26](08-metal-backend.md#826-gguf-block-types-on-the-gpu-gg_unit-gg_new) |
| quantized KV caches | the cache holds int8 (rotated keys) or 4-bit rows (rotated keys, a window of exact recent positions) instead of f32 / f16 | `cpu/KvCache` (`Q8`, `Q4`), `model/KvQ4`, `KvDtype`; `kv_store_q4`, `rope_store_q8r` in `kernels.metal` | [07 §7.13](07-cpu-backend.md#713-the-f16-kv-cache-kvcache), [08 §8.27, §8.32](08-metal-backend.md#827-the-q4-kv-cache-on-the-gpu-backendkv-dtypeq4) |
| paged KV cache | (planned) the cache as blocks that slots and prefixes share | not in this tree yet | 24 (the paged KV cache; coming soon) |
| faster Metal decoding | one native call per recorded step, several conversations in one pass, rows in the decode matvec, a decode attention with many workers, barriers only where needed | `metal/` (`DecodeProgram`, `MetalKernels`, `MetalTuner`) | [08 §8.22 - 8.31](08-metal-backend.md#822-decode-replay-one-native-call-per-token) |
| speculation | a cheap guess of the next tokens, checked in one pass: prompt lookup (n-gram index), a small draft model, or the model's own multi-token-prediction head | `engine/Speculator`, `NgramIndex`, `DraftModel`; `model/MtpFeeder`, `Qwen3NextMtp` | [10 §10.9](10-engine-and-server.md#109-speculative-decoding), [17](17-agent-speedups.md), [16 §16.14](16-gguf.md#1614-multi-token-prediction-the-model-as-its-own-draft-model) |
| constrained output | `response_format` and tool calls cannot produce malformed JSON; the fixed parts of a call are written without sampling | `sampling/JsonConstraint`, `ToolCallConstraint`, `ByteConstraint`, `TokenTrie`; `web/tools/ToolCallFormat` | [10 §10.6](10-engine-and-server.md#more-than-one-answer-probabilities-and-structured-output), [17 §17.7](17-agent-speedups.md#177-constrained-tool-calls-and-jump-forward) |
| pictures as input | a vision encoder turns each picture into "picture tokens" whose embeddings replace placeholder tokens in the prompt | `vision/` (`VisionContext`, `Mmproj`, `ClipCpu`, `ClipMetal`), `engine/MediaPrompt`; `Backend.forwardEmbeds` | [20](20-vision.md) |
| speech | Whisper (speech to text, timestamps) and OuteTTS + WavTokenizer (text to speech) | `audio/`, `server/AudioRoutes` | [21](21-audio.md) |
| image generation | diffusion: a UNet denoises a latent guided by a CLIP text embedding, a VAE decodes it | `diffusion/` (`ImagePipeline`, `UNet`, `Vae`, `ClipText`, `CpuOps`, `MetalOps`), `server/ImageHttp` | [22](22-image-generation.md) |
| video generation | the same UNet with a motion adapter (AnimateDiff), frames written as GIF / WebP / MP4; Wan 2.1 from a local folder | `diffusion/` (`VideoPipeline`, `WanPipeline`, `VideoEncoder`), `server/VideoHttp` | [23](23-video-generation.md) |
| the console | model manager, saved conversations, tools, MCP servers, the Chat / Audio / Images / Video tabs | `jllm-web/` (Spring Boot) | [10](10-engine-and-server.md), [19](19-docker.md) |


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

### 1.3.1 The same request with the newer parts switched on

The fourteen steps above are the plain path. Most of the later chapters change one of them; this is where each lands (step numbers as in the
table above), followed by the picture of a decode step with everything on.

| step | what is different | classes | chapter |
|---|---|---|---|
| 3 route | `/v1/audio/*`, `/v1/images/*` and `/v1/videos/*` are separate routes into the speech and diffusion pipelines; they never touch the engine lock or a KV cache. A chat request may carry `image_url` parts, `tools`, `tool_choice`, `response_format` | `server/AudioRoutes`, `ImageHttp`, `VideoHttp`, `ApiServer` | [21](21-audio.md), [22](22-image-generation.md), [23](23-video-generation.md) |
| 4 prompt | a picture is fetched (data URL or `http(s)`, with private addresses refused), decoded and resized exactly as llama.cpp does, encoded by the vision transformer into "picture tokens" (embeddings), and the prompt carries a placeholder span for each; tools are rendered into the prompt in the Hermes / Qwen layout | `vision/` (`ImageParts`, `ImageFetcher`, `Preprocess`, `VisionContext`), `engine/MediaPrompt`, `web/tools/ToolCallFormat` | [20](20-vision.md), [17 §17.7](17-agent-speedups.md#177-constrained-tool-calls-and-jump-forward) |
| 5 parameters | `response_format` or `tool_choice` build a `TokenConstraint` (an automaton over bytes plus a trie of the vocabulary) that the sampler consults | `sampling/JsonConstraint`, `ToolCallConstraint` | [10 §10.6](10-engine-and-server.md#more-than-one-answer-probabilities-and-structured-output) |
| 7 lock | in the console one model has several slots; slots decoding at the same moment share one pass on Metal | `engine/BatchGroup`, `web/models/LoadedModel` | [10 §10.9b](10-engine-and-server.md), [08 §8.24](08-metal-backend.md#824-batched-decoding-across-slots-metalbackendforwardbatch) |
| 8 prefix cache | a picture's span is identified by a hash of the picture, so the same picture is a cache hit and another one is not; a conversation can be restored from a file or copied from a sibling slot | `engine/MediaPrompt`, `CacheFile`, `Backend.copyStateFrom` | [17 §17.2, §17.3](17-agent-speedups.md#172-slots-that-share-a-prefix) |
| 9 prefill | chunks of `backend.batch` tokens (`auto`: 256 on Metal); picture spans go in as embeddings (`forwardEmbeds`, with 2-D rotary positions for Qwen-VL, or attending both ways inside the picture for Gemma 3); a GGUF file's matrices are read in place | `Backend.forward`, `forwardEmbeds`; `cpu/GgufKernels`; `gg_unit` in `kernels.metal` | [08 §8.29](08-metal-backend.md#829-the-prefill-chunk-backendbatchauto), [16](16-gguf.md), [20](20-vision.md) |
| 10, 13 decode | a pass may check several guessed tokens at once (prompt lookup, a draft model, the model's own MTP head), and tokens the constraint forces are appended without sampling and go through in the same pass | `engine/Speculator`, `DraftModel`; `Backend.forwardAll`; `ByteConstraint.jump` | [10 §10.9](10-engine-and-server.md#109-speculative-decoding), [17](17-agent-speedups.md), [16 §16.14](16-gguf.md#1614-multi-token-prediction-the-model-as-its-own-draft-model) |
| 13 the step itself | on Metal the whole step is recorded once and replayed with one native call; the matvecs read several rows per simdgroup; the attention over a long cache is split among many workers; independent dispatches overlap | `metal/DecodeProgram`, `MetalKernels`; `mv<NR>_*`, `attention_dec` | [08 §8.22, §8.28 - 8.31](08-metal-backend.md#822-decode-replay-one-native-call-per-token) |
| 13 the cache | its element type is f32, f16, q8 (rotated keys, on by default when q8 is chosen) or q4 (rotated keys plus a window of exact recent positions) | `cpu/KvCache`, `model/KvQ4`, `KvDtype`; `kv_store_q4`, `rope_store_q8r` | [07 §7.13](07-cpu-backend.md#713-the-f16-kv-cache-kvcache), [08 §8.27, §8.32](08-metal-backend.md#827-the-q4-kv-cache-on-the-gpu-backendkv-dtypeq4) |

One decode step with everything on, for a chat request with tools to a GGUF model on Metal and speculation by prompt lookup:

```
sampled token t ─▶ ToolCallConstraint.accept(t) ─▶ jump(): forced text, e.g. `":` ─▶ forced ids f1 f2          (no sampling for f1 f2)
Speculator.propose(history) ─▶ drafts d1 d2 d3, cut at the first token the constraint refuses
        ┌──────────────────────────── one Backend.forwardAll([t, f1, f2, d1, d2], pos) ─────────────────────────────┐
        │ MetalBackend: embedding (gg_unit decodes the file's blocks) ─▶ per layer: matmul3 (mv4 rows) ─▶ rope_store (q8r or q4 cache)    │
        │               ─▶ attention (attention_dec over the rotated cache) ─▶ o_proj ─▶ swiglu ─▶ down ─▶ ... ─▶ logits for every row   │
        └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
for each row: Sampler.verify(row, draft)  (masked by the constraint at that row) ─▶ accepted drafts + one more token ─▶ StreamDecoder ─▶ SSE
```

The unconstrained, unspeculated step is the same picture with the first two lines removed and one row instead of five.


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
DeepSeek-R1 distills of Qwen2 and Llama), Llama 3.x, Mistral, GLM-4, Phi-3 / 3.5 / 4, Gemma 2 / 3 (text, and pictures),
DeepSeek V2 / V3 (bf16, or FP8 dequantized while converting), Mamba and the Qwen3-Next / Qwen3.5 hybrids. What differs between them is parsed from
`config.json` once into a typed `Architecture` ([14](14-model-families.md), [15](15-mamba.md)); the backends never
look at the `model_type`. The same families also load from llama.cpp's `.gguf` files, where the metadata of the file
is mapped onto the same `Architecture` ([16](16-gguf.md)). Weights can be stored as Q4_0 (default), Q8_0, Q2_K or Q6_K
(`model.quant`; [06](06-quantization.md), [13](13-q2k-quantization.md)) or used in place from a GGUF file in any of 15 block
types ([06 §6.12](06-quantization.md#612-the-other-ggml-block-formats-read-in-place-from-gguf-files)), and the KV cache
is f32, f16, q8 (int8 with rotated keys) or q4 (4-bit with rotated keys and a window of exact recent positions;
`backend.kv-dtype`, [07 §7.13](07-cpu-backend.md#713-the-f16-kv-cache-kvcache)).

The "meaning" of the numbers: `dim` is how many features describe a token at every stage; `layers` is how many
times the token is refined; heads split attention into independent lookups; `vocab_size` sets the width of the
final score vector.

---

## 1.5 Why "from scratch"

The usual stack is PyTorch + a tokenizer library + a JSON library + a web framework + a CUDA/Metal runtime. Every
one of those is replaced here by a small Java class, so there is no layer you cannot read:

| normally | here | lines |
|---|---|---|
| `huggingface_hub` | `hub/HuggingFaceHub` on `java.net.http` | ~840 |
| `safetensors` | `safetensors/SafeTensors` on `FileChannel.map` + FFM | ~400 |
| `tokenizers` (Rust) + `jinja2` chat templates + llama.cpp's tokenizer | `tokenizer/*` | ~4 200 |
| PyTorch / GGML kernels | `cpu/*` (`VectorKernels`, `Int8Kernels`, `GgufKernels`, `CpuBackend`, ...) + the C kernels in `src/main/native/jllm_cpu*` | ~8 300 + ~1 300 C |
| `gguf-py` / llama.cpp's loader | `gguf/*` | ~2 400 |
| MLX / Metal Performance Shaders | `kernels.metal` (5 243) + `jllm_metal.m` + `metal/*` | ~13 200 |
| Jackson / Gson | `json/Json` | ~450 |
| Spring / Netty / FastAPI | `server/HttpServer` + `ApiServer` | ~1 400 |
| `mtmd` / Pillow / ViT code | `vision/*` (+ `vision.metal`) | ~3 500 (+ 550) |
| whisper.cpp, an audio-codec library | `audio/*` (+ `audio.metal`) | ~6 300 (+ 370) |
| `diffusers` (UNet, VAE, schedulers, AnimateDiff, Wan) | `diffusion/*` (+ `diffusion.metal`) | ~9 300 (+ 1 060) |

(Counted with `wc -l` on this tree; they only say how big each replaced piece is. The engine around them, `engine/*`, `sampling/*` and `model/*`, is
another ~12 200 lines.)

You can follow one request from the socket down to the individual multiply-add that computes an attention
score. Each piece is the simplest version that is still correct and reasonably fast; the docs point to the
non-obvious optimizations and to what a production system would add ([11](11-performance.md)).

---

## 1.6 Project layout and package responsibilities

```
org.robomiri.jllm
├── Main, ModelLoader    commands: download | run | chat | serve | bench | image | video | transcribe | speak | config
├── config.Settings      jllm.properties defaults → external file → command line flags
├── json.Json            recursive-descent JSON parser and writer
├── hub.HuggingFaceHub   lists a repo, downloads files (resumable, SHA-256 verified, retries), one shard on demand
├── safetensors.SafeTensors   memory-mapped tensor views over .safetensors files
├── tokenizer
│   ├── Tokenizer        BPE (byte-level and SentencePiece-style), Unigram, WordPiece: String ⇄ int[]
│   ├── Normalizers, PreTokenizers, Unigram, AddedTokenTrie, LongLongMap   the tokenizer.json pieces
│   ├── GgufTokenizer, GgufPre, SpmGreedy   the tokenizer stored in a GGUF file: llama.cpp's pre-tokenizer regex lists, greedy SentencePiece
│   ├── ChatTemplate     one prompt format per family (ChatML for Qwen3/Qwen2, Llama 3, Gemma, Qwen3.5, ...), thinking on/off
│   ├── PromptCache      re-tokenizes only what is new in a resent conversation
│   └── StreamDecoder    emits text only when a UTF-8 character is complete
├── model
│   ├── Architecture     sealed: Decoder | MambaArch | Qwen3NextArch, plus the shared Core (parsed from config.json or GGUF metadata)
│   ├── Families         one parser per model family, holding its defaults
│   ├── Decoder, DecoderLayer, LayerSpec   per-layer Attention (Gqa | Mla), Mlp (Dense | Moe), norms
│   ├── Attention, Mlp, Router, RopeTable, Scaling, MambaArch   the typed vocabulary
│   ├── LayerWeights, AttnWeights, MlpWeights   sealed weights mirroring LayerSpec (DecoderLayer | MambaLayer | Qwen3NextLayer)
│   ├── QType, QTensor, Q2K, Q6K, Ggml   Q4_0 / Q8_0 / Q2_K / Q6_K block quantization and the 15 ggml block layouts, page-aligned off-heap
│   ├── KvDtype, KvQ4    f32 | f16 | q8 | q4 KV cache element type, and the rotation + 4-bit rows of q4 / rotated q8
│   ├── QuantFile        jllm-<quant>.bin/.json (+ .partial.json while converting): quantized weights on disk, memory-mapped
│   ├── ModelConverter   safetensors → QuantFile, one layer at a time, shards on demand, resumable
│   ├── ShardProvider    "give me shard N" / "I'm done with shard N" (used for --low-disk)
│   ├── ModelWeights     maps the cache file (or a GGUF file) into QTensors (dense, MoE, MLA, Mamba and DeltaNet layers)
│   ├── Rope, InterleavedRope, RopePositions, GemmaMath   position-embedding (incl. M-RoPE) and family-specific math shared by the backends
│   ├── Qwen3NextMtp, MtpFeeder   the multi-token-prediction head used as a draft model
│   └── Backend          interface: forward / forwardAll / forwardEmbeds / forwardBatch, checkpoint + rollback, saveState
├── gguf
│   ├── GgufFile, GgmlType    the container: header, metadata, tensor directory, mapped pages
│   ├── GgufModel, GgufArchitecture, GgufFamilies, GgufQwenNext   metadata → typed Architecture for each architecture name
│   └── GgufWeights, GgmlCodec   tensor names → engine tensors, fused qkv / gate-up, which types are used in place
├── cpu
│   ├── VectorKernels, Int8Kernels, Q2KKernels, Q6KKernels, KQuantKernels   SIMD dot products on quantized rows (Vector API), int8 activations
│   ├── GgufKernels, GgufInt8Kernels, NativeCpu   the newer ggml block types, in Java and through libjllm_cpu (C, NEON / dot-product / AVX)
│   ├── ThreadPool       spin-then-park fork/join for small parallel loops
│   ├── CpuBackend       the forward pass on CPU; KvCache holds the f32, f16, q8 or q4 KV cache on the heap
│   └── MlaCpu, MambaCpuBackend, Qwen3NextCpuBackend   DeepSeek latent attention, the Mamba recurrence, the Gated DeltaNet hybrids
├── metal
│   ├── Metal            FFM bindings to libjllm_metal.dylib (compile shaders, buffers, dispatch, concurrent encoder)
│   ├── MetalBackend, MetalKernels   the forward pass as GPU dispatches, zero-copy weights; one Java method per kernel family
│   ├── MetalTuner       start-up self-check: times and compares kernel variants on this GPU
│   ├── DecodeProgram    a decode step recorded once and replayed with one native call
│   ├── MoeSplit, ExpertPlacement, DeviceLedger   hot experts on the GPU, cold ones on the CPU
│   └── MlaMetal, MambaMetalBackend, Qwen3NextMetalBackend   the same for latent attention, Mamba and the hybrids
├── sampling
│   ├── SamplingParams, Sampler   temperature, top-k/top-p/min-p, penalties, seed; float[] logits → int token
│   └── TokenConstraint, ByteConstraint, JsonConstraint, ToolCallConstraint, TokenTrie   response_format and tool calls as automata over bytes, jump-forward
├── engine
│   ├── Engine           the generation loop, prefix cache, stop strings, cancellation, statistics
│   ├── Speculator, NgramIndex, DraftModel   prompt lookup, a small draft model, the model's own MTP head
│   ├── BatchGroup       several slots decode in one GPU pass
│   └── CacheFile, MediaPrompt   saved KV caches, picture-keyed prefixes
├── vision                pictures as input: Preprocess / Resize, VisionEncoder (ClipCpu, ClipMetal), Mmproj (the second GGUF), VisionContext
├── audio                 Whisper (LogMel, WhisperNet, Cpu / Metal) and OuteTTS + WavTokenizer; Wav, Resampler, AudioApi
├── diffusion             Stable Diffusion (UNet, Vae, ClipText, schedulers), AnimateDiff and Wan video, CpuOps / MetalOps, Vp8l and GIF / MP4 writers
└── server
    ├── HttpServer       HTTP/1.1 on ServerSocket + virtual threads, chunked streaming
    ├── ApiServer        OpenAI-compatible JSON endpoints, SSE streaming
    ├── AudioRoutes, ImageHttp, VideoHttp   /v1/audio/*, /v1/images/*, /v1/videos/*
    └── WebUi            a single-page chat client served at /
jllm-web/                                                  optional Spring Boot console: model manager, tools, MCP, saved conversations, tabs for chat / audio / images / video
src/main/native/jllm_metal.m                               Objective-C shim: a C ABI over Metal
src/main/native/jllm_cpu.c, jllm_cpu_arm.inc, jllm_cpu_x86.inc   native int8 dot-product kernels for the CPU backend
src/main/resources/org/robomiri/jllm/metal/kernels.metal   GPU compute kernels for the language model (compiled at start-up); vision.metal, diffusion.metal, audio/audio.metal beside it
src/main/resources/jllm.properties                         all configurable settings, documented
```

Dependencies point downward only: `server` → `engine` → `sampling`/`tokenizer`/`model`; the two backends
depend on `model` and nothing depends on them except `Main` (and the pipelines of `vision`, `audio` and `diffusion`, which
choose a CPU or Metal path for their own kernels). `json` and `config` are leaves; `jllm-web` depends on the engine, never the other way round.

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
| running llama.cpp's `.gguf` files, the newer block types, multi-token prediction | [16](16-gguf.md) |
| speedups for coding agents: prompt lookup, shared prefixes, constrained tool calls | [17](17-agent-speedups.md) |
| testing the GPU kernels without a GPU; running in Docker | [18](18-metal-emulator.md), [19](19-docker.md) |
| pictures as input, speech, image generation, video generation | [20](20-vision.md), [21](21-audio.md), [22](22-image-generation.md), [23](23-video-generation.md) |
| the paged KV cache | 24 (coming soon) |
| every idea told as a story first | [00](00-explained-simply.md), [00b](00b-math-on-paper.md) |

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
- **q8 / q4 KV cache** — the cache stored as int8 (a scale per row) or 4-bit groups; the keys are first rotated by a Walsh–Hadamard transform so that no entry sticks out ([07 §7.13](07-cpu-backend.md#713-the-f16-kv-cache-kvcache)).
- **Speculative decoding** — guessing the next tokens cheaply (**prompt lookup**, a **draft model**, or the model's own **multi-token-prediction** head) and checking them all in one pass; the output is the same as without ([10 §10.9](10-engine-and-server.md#109-speculative-decoding)).
- **Constrained decoding / jump-forward** — letting the sampler choose only tokens that keep the output valid (JSON, a tool call), and appending the text that is forced without sampling ([17 §17.7](17-agent-speedups.md#177-constrained-tool-calls-and-jump-forward)).
- **Sampling** — choosing the next token from the probabilities; **temperature**, **top-k**, **top-p**, **min-p** shape the choice; **greedy** always takes the best.
- **Stop string** — text that ends generation when it appears in the output.

**Storage and numbers**

- **safetensors** — the file format of the downloaded weights: a JSON header plus raw tensor bytes.
- **bfloat16** — 16-bit float with float32's range and 3 significant digits; what the download uses.
- **Quantization** — storing weights with fewer bits (4 or 8 here) to cut memory and bandwidth.
- **Block** — 32 consecutive weights that share one **scale**; `value ≈ integer × scale`.
- **Q4_0 / Q8_0 / Q2_K** — the 4-bit, 8-bit and 2.625-bit block formats (same schemes as llama.cpp; Q2_K is for large models, [13](13-q2k-quantization.md)).
- **GGUF** — llama.cpp's single-file model format (header, metadata, tensors in ggml block formats); jllm reads its blocks in place from the mapped file ([16](16-gguf.md)).
- **mmproj** — the second GGUF of a vision model, holding the picture encoder and its projector ([20](20-vision.md)).
- **Diffusion** — generating an image by starting from noise and removing a little of it at each of a few steps, guided by the text ([22](22-image-generation.md)).
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
