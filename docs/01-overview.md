# 1. Overview: what we built and why it works

## What a language model actually does

A large language model (LLM) is a function. Give it a list of *tokens* (small pieces of text, roughly words or
word fragments) and it returns, for every possible next token, a score saying how likely that token is to come
next. That is all. Everything else — chat, reasoning, code generation — is built by calling this function in a
loop:

```
tokens = tokenize("<|im_start|>user\nWhat is 7*8?<|im_end|>\n<|im_start|>assistant\n")
loop:
    scores = model(tokens)            # one score per vocabulary entry (151 936 of them for Qwen3)
    next   = pick_one(scores)         # "sampling": greedy argmax, or a random draw weighted by score
    if next is the end-of-turn token: stop
    tokens.append(next)
    print(detokenize(next))
```

The function `model(tokens)` is a *transformer*: a fixed sequence of matrix multiplications and a few simple
non-linear operations, parameterized by ~600 million numbers called *weights*. The weights were learned during
training (which we do not do here). Inference means: load the weights, run the function fast.

## The pipeline in this project

```
 Hugging Face Hub ──download──▶ models/Qwen_Qwen3-0.6B/
                                  ├─ config.json              hyper-parameters (how big is each matrix)
                                  ├─ tokenizer.json           vocabulary + merge rules
                                  ├─ model.safetensors        1.5 GB of bfloat16 weights
                                  └─ generation_config.json   default sampling settings

 text ──Tokenizer──▶ ids ──Backend.forward()──▶ logits ──Sampler──▶ id ──StreamDecoder──▶ text
                            │
                            ├─ CpuBackend   (Vector API SIMD + thread pool)
                            └─ MetalBackend (GPU kernels through Panama FFM)

 Engine wraps the loop; ApiServer exposes it as OpenAI-compatible HTTP (+ SSE streaming).
```

Each arrow is one package:

| Step | Package | Doc |
|---|---|---|
| download | `hub` | [02](02-download.md) |
| read weights | `safetensors` | [03](03-safetensors.md) |
| text → ids → text | `tokenizer` | [04](04-tokenizer.md) |
| the math of `forward()` | `model`, `cpu`, `metal` | [05](05-transformer-math.md), [07](07-cpu-backend.md), [08](08-metal-backend.md) |
| shrinking the weights | `model/QTensor` | [06](06-quantization.md) |
| logits → token | `sampling` | [09](09-sampling.md) |
| loop + HTTP | `engine`, `server` | [10](10-engine-and-server.md) |

## Qwen3-0.6B in numbers

From `config.json` (parsed by `ModelConfig`):

| Name | Value | Meaning |
|---|---|---|
| `vocab_size` | 151 936 | number of distinct tokens |
| `hidden_size` (we call it `dim`) | 1024 | width of the vector that represents each token inside the model |
| `num_hidden_layers` | 28 | how many identical transformer blocks are stacked |
| `num_attention_heads` | 16 | attention "heads" for queries |
| `num_key_value_heads` | 8 | attention heads for keys/values (grouped-query attention: 2 query heads share 1 KV head) |
| `head_dim` | 128 | size of each head; note 16 × 128 = 2048 ≠ 1024, Qwen3 projects *up* for attention |
| `intermediate_size` | 3072 | width of the feed-forward hidden layer |
| `rms_norm_eps` | 1e-6 | tiny constant to avoid division by zero in normalization |
| `rope_theta` | 1 000 000 | base frequency for positional encoding |
| `tie_word_embeddings` | true | the output matrix is the same as the input embedding matrix |

Parameter count: embedding 151 936 × 1024 ≈ 155.6 M, plus 28 layers × ≈ 15.7 M ≈ 440 M → **≈ 596 M** ("0.6B").
In bfloat16 (2 bytes each) that is the 1.5 GB file. After our Q4_0 quantization the layers take 0.56 bytes per
weight and the embedding 1.06 bytes, so the whole model is **≈ 450 MB** in RAM.

## Why "from scratch" matters here

Normally you would use PyTorch, a tokenizer library, a JSON library, an HTTP framework. Here every one of those
is replaced by a small, readable Java class, so there is no magic: you can follow a request from the socket
down to the individual multiply-add that computes an attention score. Each piece is deliberately the simplest
version that is still correct and reasonably fast.

## Project layout

```
org.robomiri.jllm
├── Main                 command line: download | run | chat | serve | bench | config
├── config.Settings      jllm.properties defaults → external file → command line flags
├── json.Json            recursive-descent JSON parser and writer
├── hub.HuggingFaceHub   downloads a repo's inference files
├── safetensors.SafeTensors   memory-mapped tensor views
├── tokenizer
│   ├── Tokenizer        byte-level BPE
│   ├── ChatTemplate     Qwen3 ChatML prompt format
│   └── StreamDecoder    emits text only when a UTF-8 character is complete
├── model
│   ├── ModelConfig      config.json
│   ├── QType, QTensor   Q4_0 / Q8_0 block quantization
│   ├── QuantFile        jllm-<quant>.bin/.json: quantized weights on disk, memory-mapped
│   ├── ModelConverter   safetensors → QuantFile, one layer at a time, shards on demand
│   ├── ModelWeights     maps the cache file into QTensors (dense + MoE layers)
│   ├── Rope             cos/sin tables for rotary position embedding
│   └── Backend          interface: forward(tokens, pos, logits)
├── cpu
│   ├── VectorKernels    SIMD dot products on quantized rows
│   ├── ThreadPool       spin-then-park fork/join for small parallel loops
│   └── CpuBackend       the forward pass on CPU
├── metal
│   ├── Metal            FFM bindings to libjllm_metal.dylib
│   └── MetalBackend     the forward pass as GPU dispatches
├── sampling
│   ├── SamplingParams   temperature, top-k, top-p, ...
│   └── Sampler          turns logits into one token id
├── engine.Engine        the generation loop, prefix cache, stop strings, statistics
└── server
    ├── HttpServer       HTTP/1.1 on sockets + virtual threads
    ├── ApiServer        OpenAI-compatible JSON endpoints, SSE streaming
    └── WebUi            a single HTML page chat client
```

## Glossary

- **Token**: a unit of text from the vocabulary (e.g. `Hello`, ` world`, `ing`, `!`). Qwen3 has 151 936.
- **Token id**: the integer index of a token in the vocabulary.
- **Embedding**: the vector (1024 floats) that represents a token at the model's input.
- **Logits**: the raw, unnormalized scores for every vocabulary entry at the model's output. Softmax turns them into probabilities.
- **Layer / block**: one repetition of attention + feed-forward; Qwen3-0.6B stacks 28.
- **Attention**: the mechanism by which a token's vector looks at (and mixes in) the vectors of earlier tokens.
- **Head**: attention is done several times in parallel with smaller vectors; each copy is a head.
- **KV cache**: keys and values computed for previous tokens, saved so they are not recomputed each step.
- **Prefill**: processing the whole prompt in one go (many tokens per forward call).
- **Decode**: generating one token per forward call.
- **Quantization**: storing weights with fewer bits (here 4 or 8 instead of 16).
- **Sampling**: picking the next token from the probabilities.
- **SIMD**: one CPU instruction operating on several numbers at once (the Vector API exposes this in Java).
- **FFM / Panama**: `java.lang.foreign`, the JDK API for calling C functions and reading native memory without JNI.
- **Metal**: Apple's GPU programming API. **MLX** is Apple's ML framework built on it; we talk to Metal directly, which is the layer MLX itself uses.
