# How jllm-serve works — a guided tour

These documents explain, from the ground up, how this project turns a 1.5 GB file downloaded from Hugging Face
into a chatbot that streams answers over HTTP. They assume you can read Java but know nothing about machine
learning. Math is introduced where it is needed, with concrete numbers from Qwen3-0.6B.

Suggested reading order:

| # | Document | What you will learn |
|---|---|---|
| 1 | [Overview](01-overview.md) | What an LLM is, what "inference" means, the pipeline from text to text, the supported model families, project layout |
| 2 | [Downloading from Hugging Face](02-download.md) | The Hub API, resumable HTTP downloads, SHA-256 verification, retries and the stall watchdog, downloading only the shards a model needs |
| 3 | [Tensors and safetensors](03-safetensors.md) | What a tensor is, how the file is laid out byte by byte, bfloat16, memory mapping, jllm's quantized cache file (versioned, reconverted when stale, resumable conversion) |
| 4 | [The tokenizer](04-tokenizer.md) | Byte-level BPE from scratch, regex pre-tokenization, the chat template, streaming UTF-8 |
| 5 | [The transformer, step by step](05-transformer-math.md) | Embeddings, RMSNorm, attention, RoPE, SwiGLU, the KV cache — every formula the code implements |
| 6 | [Quantization (Q4/Q8)](06-quantization.md) | Why 4-bit weights work, the Q4_0/Q8_0 block format, the error it introduces |
| 7 | [The CPU backend](07-cpu-backend.md) | SIMD with the Vector API, dot-product kernels, prefill vs decode, the custom thread pool |
| 8 | [The Metal backend](08-metal-backend.md) | GPU basics, Panama FFM bindings, the Objective-C shim, each compute kernel |
| 9 | [Sampling](09-sampling.md) | Softmax, temperature, top-k, top-p, min-p, repetition penalties |
| 10 | [Engine and server](10-engine-and-server.md) | The generation loop, prefix caching, stop strings, HTTP/1.1, SSE streaming, the OpenAI API |
| 11 | [Performance notes](11-performance.md) | Why decode is memory-bound, measured numbers (Metal, CPU with int8 activations, f16 KV cache), what to improve next |
| 12 | [Mixture of Experts](12-mixture-of-experts.md) | Routers and experts (Qwen3-30B-A3B), stacked expert weights, converting a 61 GB checkpoint on a 32 GB laptop |
| 13 | [Q2_K quantization](13-q2k-quantization.md) | 2-bit weights: sub-block scales and mins, the 84-byte super-block, the quantizer search, the Σx dot product, measured size/speed/quality |
| 14 | [Model families](14-model-families.md) | Llama 3.x, Mistral, Qwen2/2.5 and DeepSeek-R1 distills: bias, sliding window, llama3 RoPE scaling, tied embeddings, BOS, tensor names; DeepSeek-V2/V3: multi-head latent attention, YaRN, shared experts, routing rules, the FP8 limitation |
| 15 | [Mamba](15-mamba.md) | State-space models: the recurrence, discretization, the causal conv, state instead of a KV cache, why the prefix cache changes |

A glossary of terms is at the end of the [overview](01-overview.md#glossary).
