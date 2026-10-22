# jllm-serve

An LLM inference engine and OpenAI-compatible server written in Java from scratch with **zero external
dependencies** — only the JDK (Vector API, FFM/Panama) plus a ~200-line Objective-C shim for Metal.
An optional browser console with model management ([`jllm-web/`](#web-console-jllm-web)) is a separate Spring Boot
module; the engine itself does not depend on it.

Supported families (Hugging Face `model_type`): **Qwen3** dense and MoE (`qwen3`, `qwen3_moe`), **Qwen2 / 2.5**
and the DeepSeek-R1 distills (`qwen2`), **Llama 3.x** (`llama`), **Mistral** (`mistral`), **GLM / GLM-4**
(`glm`, `glm4`), **Phi-3 / 3.5 / 4** (`phi3`), **Gemma 2 / 3** text (`gemma2`, `gemma3_text`, `gemma3`),
**DeepSeek V2 / V3** (`deepseek_v2`, `deepseek_v3`; bf16, or FP8 dequantized while converting), **Mamba** (`mamba`) and
**Qwen3-Next** (`qwen3_next`; also **Qwen3.5** from GGUF: Gated DeltaNet linear attention in 3 of 4 layers, so a 4× smaller KV cache,
[docs/14 §14.14](docs/14-model-families.md#1414-qwen3-next-hybrid-gated-deltanet--gated-attention--sparse-moe)).
Weights are quantized once to **Q4_0, Q8_0 or Q2_K** and memory-mapped afterwards; the KV cache can be f32, f16 or
q8 (int8 with a scale per row, a quarter of f32).
llama.cpp **GGUF** files (`Qwen/Qwen3-0.6B-GGUF`, `unsloth/...-GGUF`, a local `.gguf`) run as they are, without a
conversion step: Qwen3 / Qwen3-MoE / Qwen2 / Llama / Mistral 3 architectures, **Gemma 2 / 3** (`gemma2`, `gemma3`), **Phi-3 / 4**
(`phi3`), **GLM** (`chatglm`, `glm4`), **Mamba** (`mamba`), **DeepSeek V2 / V3 / R1** (`deepseek2`) and the hybrid **Qwen3.5**
(`qwen35`) and Qwen3-Next (`qwen3next`) ones (the layouts of the new ones checked against llama.cpp's own loader on tiny files,
[docs/16 §16.15](docs/16-gguf.md#1615-gemma-23-phi-3-glm-mamba-deepseek-v2v3-and-mistral-3-as-gguf-files)), Q4_0, Q8_0, Q4_K, Q5_K and Q2_K blocks used in
place from the memory-mapped file on both backends ([docs/16](docs/16-gguf.md)); a Qwen3.5 file's own
multi-token-prediction layer can draft for speculative decoding (`--engine.speculative mtp`,
[docs/16 §16.14](docs/16-gguf.md#1614-multi-token-prediction-the-model-as-its-own-draft-model)). Apple **MLX** quantized checkpoints
(`mlx-community/*-4bit`, `-8bit`, and the `mxfp4` / `mxfp8` modes) and block-quantized **FP8** checkpoints are read too:
dequantized once while converting ([docs/03 §3.7, §3.8](docs/03-safetensors.md#37-mlx-quantized-checkpoints)).
It also draws: **Stable Diffusion** image generation (SD 1.x, SD 2.x / SD-Turbo, SDXL / SDXL-Turbo, from the Hugging Face
diffusers layout) as text to image, image to image, inpainting, outpainting and upscaling, on Metal or the CPU, with an
OpenAI-compatible `/v1/images` API and an Images tab in the web console ([docs/22](docs/22-image-generation.md)). And it makes
short **videos**: AnimateDiff (a motion adapter on Stable Diffusion 1.5) writes 8–32 frame clips as MP4 (with `ffmpeg`), animated
WebP or GIF, with a `/v1/videos/generations` API, `./jllm video`, a Video tab with a frame-strip editor (trim, reverse, speed, export)
and a `generate_video` chat tool ([docs/23](docs/23-video-generation.md)); the building blocks of Wan 2.1 (diffusion transformer,
causal 3-D VAE, UMT5 text encoder, flow sampler) are written and checked against diffusers too.

| Piece | Where | Written from scratch |
|---|---|---|
| Hugging Face Hub downloader (resumable, SHA-256 verified, retries, stall watchdog) | `hub/HuggingFaceHub.java` | yes, on `java.net.http` |
| safetensors reader (memory-mapped via FFM) | `safetensors/SafeTensors.java` | yes |
| Byte-level and SentencePiece-style BPE tokenizers + per-family chat templates | `tokenizer/` | yes |
| Typed model description (`Architecture`, `LayerSpec`, `RopeTable`, one parser per family) | `model/Architecture.java`, `model/Families.java`, ... | yes |
| Q4_0 / Q8_0 / Q2_K block quantization + versioned, mmap'd quantized cache file, shard-by-shard resumable conversion | `model/QTensor.java`, `model/Q2K.java`, `model/QuantFile.java`, `model/ModelConverter.java` | yes |
| GGUF reader (memory-mapped, split files), ggml block formats read in place (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, MXFP4, F16, BF16; CPU + Metal kernels), architecture / tokenizer / tensor mapping from the file's metadata | `gguf/`, `model/Ggml.java`, `cpu/GgufKernels.java`, `cpu/GgufInt8Kernels.java`, `tokenizer/GgufTokenizer.java` | yes |
| Forward pass on CPU: dense, MoE, MLA, Mamba (Vector API SIMD, int8 activations, custom thread pool, f32/f16/q8 KV cache) | `cpu/` | yes |
| Forward pass on GPU: the same on Metal (fused kernels, Panama FFM bindings) | `metal/`, `src/main/native/` | yes |
| Sampler (temperature, top-k, top-p, min-p, penalties) | `sampling/` | yes |
| JSON parser/writer | `json/Json.java` | yes |
| HTTP/1.1 server (virtual threads, chunked SSE) + OpenAI API + web chat UI | `server/` | yes |

**Read [`docs/`](docs/README.md)** for a beginner-friendly walkthrough of how each part works, including the math.
New to all of it? **[docs/00 Explained simply](docs/00-explained-simply.md)** tells every idea as a story first
(🧸, for a five-year-old) and then gives the exact math (📐).

### Beyond a plain port of llama.cpp

| Feature | What it does | Docs |
|---|---|---|
| Prompt lookup for agents | an n-gram index over the whole context drafts up to 16 tokens that the model verifies in one pass; 20 → 81 tok/s when rewriting code from the prompt, −4 % on free text | [17 §17.1](docs/17-agent-speedups.md#171-prompt-lookup-tuned-for-agents) |
| Constrained tool calls | inside `<tool_call>` the model can only write a known tool and arguments that match its schema, so a call always parses; the fixed parts of a call are appended in one pass without sampling (jump-forward), speculation checks its drafts against the constraint, and the token mask is computed with a byte trie and memoized (a 32-token call: 254 → 149 ms on an M1 Pro with jump-forward; the mask 60–170 µs per token where it was 190–510 µs; `engine.constrained-tools`, `engine.jump-forward`, `engine.constraint-masks`) | [17 §17.7](docs/17-agent-speedups.md#177-constrained-tool-calls-and-jump-forward) |
| Batched decoding across slots | conversations generating at the same time share one forward pass, so the weights are read once for all of them (CPU backend) | [10 §10.9b](docs/10-engine-and-server.md#109b-batched-decoding-across-slots) |
| Prefix sharing between slots | a second conversation with the same long system prompt copies the first one's cache (34 ms) instead of recomputing it (seconds) | [17 §17.2](docs/17-agent-speedups.md#172-slots-that-share-a-prefix) |
| Paged KV cache with a prefix tree | `backend.kv-paged=true`: KV memory as one pool of 16-position blocks per model with a block table per slot, instead of a full-size array per slot. Blocks are allocated as a conversation grows; a radix tree over the token ids gives identical prefixes (a shared system prompt, a retry, a subagent forked from one context) one copy across slots and conversations, with reference counts and copy on write; unreferenced blocks are evicted least recently used first and spilled to a disk block store. The CPU tile kernels and the Metal split / flash / decode / prefill kernels read through the block table, bit for bit as with the flat cache (f32, f16, q8); decode replay, pipelined decoding, batched rows, speculation rollback and saved caches work unchanged; every index is `long` (a million-position test). Qwen3-0.6B: 8 conversations sharing a 4 096-token prompt take 1.1 GB instead of 7.7 GB, a resent 4 160-token conversation 32 ms to first token on Metal instead of 6.3 s (CPU 71 ms instead of 24 s), decode −3 % on Metal; q4, DeepSeek and the recurrent hybrids stay flat. Off by default | [24](docs/24-paged-kv-cache.md) |
| Conversations saved to disk | long prompts survive restarts: 21 s prefill → 0.9 s restore | [17 §17.3](docs/17-agent-speedups.md#173-saved-conversations) |
| Pipelined GPU decoding | the GPU picks the greedy token itself and runs the next step before the CPU has looked | [08 §8.16](docs/08-metal-backend.md#816-pipelined-greedy-decoding) |
| Decode replay | a decode step's ~170 GPU dispatches are recorded once and replayed with one native call per token instead of one call per dispatch | [08 §8.22](docs/08-metal-backend.md#822-decode-replay-one-native-call-per-token) |
| Concurrent compute encoder | `-Djllm.metal.concurrent`: the Metal encoder lets independent dispatches overlap and the native layer inserts a barrier only where a kernel reads or overwrites what an earlier one wrote (found from the pipelines' reflection); bit-identical to the serial encoder, no gain for single-stream decode (every dispatch depends on the one before), no gain for single-stream decode or prefill (30B included), a few percent for three slots in shared passes, off by default | [08 §8.28](docs/08-metal-backend.md#828-the-concurrent-compute-encoder-barriers-only-where-a-kernel-needs-the-one-before) |
| Multi-row decode matvec on Metal | the single-token matmuls give a simdgroup 4 rows that share the activations and the rmsnorm prologue: matvecs per token 6.76 → 5.52 ms, 0.6B decode 118 → 132 tok/s, GGUF Q4_0 105 → 133 (`-Djllm.metal.mvRows=1` for the old kernels) | [08 §8.30](docs/08-metal-backend.md#830-several-output-rows-per-simdgroup-in-the-decode-matvec-mvnr_) |
| Decode attention on Metal | independent simdgroups with online softmax in registers and K/V rows shared by a kv head's query heads, picked by the start-up check: ~2× at 3 000 tokens of context (63 → 93 tok/s; f16 / q8 caches 62 → 113) | [08 §8.31](docs/08-metal-backend.md#831-decode-attention-attention_dec) |
| Prefill chunk `auto` | 256-token chunks for dense models on Metal (prefill 1 426 → 2 012 tok/s on Qwen3-0.6B), logits identical across chunk sizes | [08 §8.29](docs/08-metal-backend.md#829-the-prefill-chunk-backendbatchauto) |
| Flash-attention prefill on Metal | 32 query rows share each block of keys, online softmax, no score buffer | [08 §8.13](docs/08-metal-backend.md#813-flash-attention-prefill-attention_fa) |
| Kernel self-check at start-up | new kernels are compared with the established ones on your GPU and used only if correct and faster; cached per device | [08 §8.14](docs/08-metal-backend.md#814-the-kernel-self-check-metaltuner) |
| Expert-grouped MoE prefill | tokens are sorted by expert so each expert's weights are read once per tile, not once per token; MoE models prefill in 256-token chunks so each expert serves ~16 tokens per read (CPU experts measured 1.6× faster per token) | [08 §8.15](docs/08-metal-backend.md#815-expert-grouped-moe-prefill) |
| Hot experts on the GPU, cold on the CPU | an MoE model too big for the GPU's memory budget leaves its rarely used experts (learned from your own usage) on the CPU, which computes them alongside the GPU, instead of paging | [08 §8.18](docs/08-metal-backend.md#818-experts-on-the-cpu-backendmetalcpu-moe) |
| Tokenizing only what is new | a coding agent resends its whole conversation every turn; the rendered prompt's ids are cached, so only the new messages are tokenized (15.6 ms → 0.5 ms for a 100k-character, 50-turn conversation), with exactly the ids of a full tokenization | [17 §17.5](docs/17-agent-speedups.md#175-tokenizing-only-what-is-new) |
| Stopping for a client that left | an agent that cancels a request frees its slot within one prefill chunk or one token, streaming or not, instead of generating to the end | [17 §17.6](docs/17-agent-speedups.md#176-noticing-a-client-that-left) |
| Saving in the background, restoring part | a conversation is saved while the next turn already runs (the engine is held 0.3 ms instead of 3 s), and a saved file restores just the prefix a new prompt shares with it | [17 §17.3](docs/17-agent-speedups.md#173-saved-conversations) |
| q8 KV cache | `backend.kv-dtype=q8`: int8 keys and values with one scale per row of a kv head, half of f16 (59 KB vs 115 KB per token for Qwen3-0.6B), on both backends, files interchangeable | [07](docs/07-cpu-backend.md#q8-int8-values-and-a-scale-per-row-kvcacheq8), [08](docs/08-metal-backend.md#q8) |
| q8 KV cache with rotated keys | `backend.kv-q8-rotate` (default `true`): the q8 cache's keys are Walsh–Hadamard rotated before they are quantized (the query is rotated too, so scores are unchanged), which cuts q8's perplexity cost against f16 from 0.028 to 0.003 nats per token and its KL divergence 15-55x, for the same memory and 1-3 % of speed; `false` keeps the previous format. **Existing q8 users:** saved q8 caches from before this default are refused ("the saved cache has another layout"; the console deletes them and prefills afresh): set `backend.kv-q8-rotate=false` to keep reading them. Not for DeepSeek's latent cache | [07 §7.13](docs/07-cpu-backend.md#q8-with-rotated-keys-backendkv-q8-rotate), [08 §8.32](docs/08-metal-backend.md#832-the-q8-kv-cache-with-rotated-keys-on-the-gpu-backendkv-q8-rotate), [11 §11.10](docs/11-performance.md#1110-the-q8-cache-with-rotated-keys-measured) |
| q4 KV cache | `backend.kv-dtype=q4`: Hadamard-rotated keys (the query is rotated too, so scores are unchanged), 4-bit groups of 32 with a half scale, and the latest `backend.kv-q4-recent` positions also kept as halves; 31.5 KB per token for Qwen3-0.6B (f16 115 KB, q8 59 KB), on both backends, decode replay, batched decoding and saved files included. Measured against f16 on Qwen3-0.6B it is clearly less accurate than q8 (the keys need about 5 bits), so it is opt-in | [07](docs/07-cpu-backend.md#q4-rotated-keys-4-bit-groups-and-a-recent-window-kvcacheq4-kvq4), [08 §8.27](docs/08-metal-backend.md#827-the-q4-kv-cache-on-the-gpu-backendkv-dtypeq4), [11 §11.9](docs/11-performance.md#119-the-q4-kv-cache-measured) |
| int8 or float, measured | `backend.cpu.int8-activations=auto` times one token's matmuls both ways at load and keeps the faster (int8 on ARM, often float on AVX-512) | [07 §7.10.1](docs/07-cpu-backend.md#7101-auto-measure-dont-guess) |
| Checked against transformers on real weights | Qwen3, Qwen2.5, Llama 3.2, Mistral, GLM-Edge, Gemma 2/3 and Mamba give the same token ids, chat templates and (to quantization rounding) logits as Hugging Face transformers; `scripts/real-weights-check.sh` reruns it for any model | [14 §14.13](docs/14-model-families.md#1413-verified-on-real-weights) |
| Speech-to-text (Whisper) | `openai/whisper-tiny … large-v3-turbo` from the Hub: log-mel exactly as `transformers` computes it, encoder + decoder with cross-attention on the CPU (4x4 register tiles) and on Metal (own `audio.metal` kernels, encoder and decoder on the GPU), the OpenAI decoding protocol (language detection, timestamps, 30 s windows, temperature fallback), word timestamps by DTW over the alignment heads; identical token ids to `transformers`; `POST /v1/audio/transcriptions` and `/translations` | [21](docs/21-audio.md) |
| Text-to-speech (OuteTTS) | a Qwen2 language model that writes audio-codec tokens, run by the same engine as the chat models, plus the WavTokenizer decoder (convolutions, attention, ConvNeXt, inverse STFT) in Java; 10 voices, speed without changing the pitch; `POST /v1/audio/speech` | [21](docs/21-audio.md) |
| Audio tab and editor | record or upload, transcribe (SRT/VTT/JSON, word times), text to speech, a waveform editor (cut, fade, normalize, silence removal, mix, undo; transcribe or replace a selection, edit audio by editing its transcript), dictation and read-aloud in the chat, `transcribe_audio` / `speak` tools | [21](docs/21-audio.md) |
| Metal emulator | the GPU kernels and the unchanged Metal backend run on any CPU, so every change is checked against the CPU backend without a Mac | [18](docs/18-metal-emulator.md) |
| Video generation | AnimateDiff written from scratch on the image stack: the motion adapter's 21 temporal transformers (group norm over the whole clip, attention along the frame axis of every pixel, sinusoidal frame positions) inside the SD 1.5 UNet with the frames in the batch, DDIM and the linear schedule it was trained with, the VAE per frame; animated GIF (median-cut palette, dithering, LZW), lossless animated WebP (VP8L: predictors, back references, prefix codes) and MP4 through ffmpeg, all written in Java; `/v1/videos/generations` and `/encode` (streaming progress and previews), `./jllm video`, a Video tab with gallery and frame-strip editor, `generate_video` chat tool; new kernels (axis swap, row table add; RMS norm, rotary, modulation, frame shift for Wan). Equal to diffusers' `AnimateDiffPipeline` on the same noise: UNet within 8·10⁻⁶ relative, 16-frame clips at 78–87 dB PSNR (float16: 45 dB). Wan 2.1 (DiT, causal 3-D VAE, UMT5 with int8 weights, UniPC) checked against diffusers on tiny pipelines (130 dB) and on the real transformer and VAE | [23](docs/23-video-generation.md) |
| Image generation | Stable Diffusion 1.x / 2.x / Turbo and SDXL written from scratch (CLIP tokenizer and text encoders, UNet, VAE with tiled decoding, Euler / Euler-ancestral / DPM++ 2M, classifier-free guidance, image to image, inpainting with blending or a 9-channel UNet, outpainting, upscaling); on Metal in float16 with its own kernels (implicit-GEMM convolution, flash attention, chunked group norm) and on the CPU; `/v1/images/generations` and `/edits` (streaming progress and previews), `./jllm image`, an Images tab with gallery and canvas editor, `generate_image` / `edit_image` chat tools. Equal to Hugging Face diffusers stage by stage on the same noise: float32 UNet within 5·10⁻⁶ relative, final images at 74–137 dB PSNR (float16: 50–77 dB) | [22](docs/22-image-generation.md) |
| Pictures as input | vision-language GGUF models (SmolVLM, Qwen2.5-VL, Qwen3-VL; the Gemma 3 projector) see `image_url` parts of a chat request, `--image` files and the console's attached pictures: the vision transformer of the `mmproj-*.gguf` runs on the GPU (own Metal kernels) or the CPU, its embeddings replace the placeholder tokens, Qwen's two-coordinate M-RoPE and Qwen3-VL's deepstack work by rewriting rotary table rows and adding features, and the KV cache is keyed by a hash of the pixels; compared with llama.cpp's `libmtmd` (same prompt ids, embeddings to 2·10⁻³ … 6·10⁻³) | [20](docs/20-vision.md) |

The Metal items above (up to the M1 Pro measurement; decode replay and fusion, half-weight MMA prefill, Metal
batched decoding and draft speculation were merged after it and are verified on the emulator only) were verified for correctness on the emulator (every family, CPU == Metal) and on an M1 Pro
by `scripts/mac-validate.sh`, which also measures each one on and off and prints a table
([08 §8.20](docs/08-metal-backend.md#820-validating-on-a-mac)). On Qwen3-0.6B the simdgroup-matrix prefill is the big
one (536 → 1 430 tok/s), flash prefill adds ~6 %, while pipelined decoding, the f16 and the q8 KV cache leave decode
at ~113–118 tok/s; the kernel self-check keeps flash decode off on that GPU. Compare yourself with
`-Djllm.metal.flashPrefill=false`, `-Djllm.metal.moeMma=false`, `-Djllm.metal.pipeline=false`, `-Djllm.metal.replay=false`,
`-Djllm.metal.hmma=false` (passed through `JAVA_OPTS`) and `./jllm bench`. `scripts/mac-validate.sh --vs-llama` also runs llama.cpp's `llama-bench` on the same
GGUF file with the same prompt and output lengths: llama.cpp is still about 2× faster (see
[below](#measured-on-an-apple-m1-pro-10-cores-32-gb) and
[11 §11.1](docs/11-performance.md#against-llamacpp-scriptsmac-validatesh---vs-llama)).

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

### Audio

```sh
./jllm transcribe --model openai/whisper-small --audio talk.wav --format srt      # speech to text (Metal when available)
./jllm speak --model OuteAI/OuteTTS-0.3-500M --prompt "Hello from jllm." --voice en_female_1 --out hello.wav
./jllm serve --audio.stt-model openai/whisper-small --audio.tts-model OuteAI/OuteTTS-0.3-500M   # + /v1/audio/* endpoints
curl localhost:8080/v1/audio/transcriptions -F file=@talk.wav -F response_format=verbose_json -F 'timestamp_granularities[]=word'
curl localhost:8080/v1/audio/speech -H 'Content-Type: application/json' -d '{"input":"Hello.","voice":"en_male_1"}' -o hello.wav
```

Models load like text models (the console's Models page, `jllm.web.autoload`); WAV, AIFF and AU are read in pure Java, other formats through `ffmpeg` if installed
(the console's Audio tab converts to WAV in the browser). Details, settings (`audio.*`) and measurements: [docs/21](docs/21-audio.md).

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

### Images

```sh
./jllm image --image-model stabilityai/sd-turbo --prompt "a red fox in a snowy forest, photo" --out fox.png    # ~2.6 GB, once
./jllm image --image-model stabilityai/sd-turbo --prompt "watercolor" --init fox.png --strength 0.5 --out fox2.png   # image to image
./jllm serve --image-model stabilityai/sd-turbo --model none          # OpenAI-compatible POST /v1/images/generations and /edits
./jllm-web/run.sh                                                      # Models: load stabilityai/sd-turbo; then the Images tab
```

SD-Turbo makes a 512 × 512 picture in 1–4 steps without guidance; `stable-diffusion-v1-5/stable-diffusion-v1-5`,
`stabilityai/sdxl-turbo` and the inpainting model `stable-diffusion-v1-5/stable-diffusion-inpainting` work too. Settings:
`image.backend` (`auto` = Metal when there is a GPU), `image.precision` (`f16` / `f32`), `image.vae-tile`; all `image.*`
keys are in [`jllm.properties`](src/main/resources/jllm.properties). Details, the math of diffusion and the measured speed
against diffusers: [docs/22](docs/22-image-generation.md).

### Video

```sh
./jllm video --video-model guoyww/animatediff-motion-adapter-v1-5-2 --prompt "a red balloon floating over a green hill, clouds moving, photo" --out balloon.mp4
./jllm video --video-model guoyww/animatediff-motion-adapter-v1-5-2 --prompt "waves" --frames 16 --fps 8 --width 256 --height 256 --out waves.gif   # 4x faster at 256
./jllm serve --video-model guoyww/animatediff-motion-adapter-v1-5-2 --model none      # POST /v1/videos/generations and /v1/videos/encode
./jllm-web/run.sh                                                                      # Models: load the adapter; then the Video tab
```

The adapter (1.8 GB) and the Stable Diffusion 1.5 base it works with (2.6 GB, fp16 files) are downloaded once into `models/`. MP4 needs `ffmpeg`
on the PATH (`video.ffmpeg`); without it clips are animated WebP (lossless) or GIF. `video.enabled=false` switches the endpoints, the console tab
and the chat tool off. All `video.*` keys are in [`jllm.properties`](src/main/resources/jllm.properties); how it works, the API shape (OpenAI has no
video endpoint: it follows `/v1/images/generations`), the editor, accuracy against diffusers and the measured speed: [docs/23](docs/23-video-generation.md).
Quality note: the `v1-5-2` adapter on the plain Stable Diffusion 1.5 gives poor clips (the same ones diffusers makes: the tests compare the numbers, not
the taste); point `video.base-model` at a fine-tuned SD 1.5 checkpoint in diffusers format for good ones. Wan 2.1 is wired into
`./jllm video --video-model <local folder>` (tested on a tiny random pipeline), but its 22.7 GB float32 text encoder was beyond this work's download budget, so only the
transformer and the VAE were verified on the real weights.

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
mapped file. Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, the K-quants Q4_K / Q5_K / Q6_K (the usual `Q4_K_M` mix), the
non-linear 4-bit IQ4_NL / IQ4_XS, MXFP4 (gpt-oss) and F16 / BF16 matrices are computed in place, on both backends (a
mixed file such as the `Q4_0` of Qwen3.8 with `ffn_down` in Q4_1 stays at its 4.5-5 bits per weight instead of being
widened to 8.5); on the CPU most of them use int8-activation kernels like Q4_0 (Qwen3-0.6B `Q4_K_M` decodes at ~11
tok/s on 4 AVX-512 cores, [docs/07](docs/07-cpu-backend.md#gguf-k-quants-kquantkernels)). Only F32 matrices (and
whatever `-Djllm.gguf.requantize=<types>` lists, to compare with the old behaviour; the CPU backend requantizes
Q5_1, Q3_K, F16 and BF16 by default because its in-place decode of those is more than 25% slower, `=none` keeps them in place) are
requantized to Q8_0 in memory at load; IQ1-IQ3 and the ternary types are refused. See [docs/16](docs/16-gguf.md) for the file format, the type table
(16.4.1), the llama q/k permutation, the zero-copy mapping and the verification.

`--low-disk` downloads, quantizes and deletes the safetensors shards one at a time, so the 30B MoE (61 GB in
bf16, ~20 GB quantized) converts on a machine with ~30 GB free. See [docs/12](docs/12-mixture-of-experts.md).
Family details (what differs between them, verified tensor names, limits) are in
[docs/14](docs/14-model-families.md) and [docs/15](docs/15-mamba.md). Verification: every family runs on synthetic
checkpoints and tokenizer fixtures (CPU == Metal, reference implementations). On real published weights, the CPU
backend at Q8_0 was compared with Hugging Face transformers (float32) for Qwen3-0.6B, Qwen2.5-0.5B-Instruct,
Llama-3.2-1B-Instruct, GLM-Edge-1.5B-Chat, Gemma-2-2B-it, Gemma-3-1B-it, mamba-130m and the community
TinyMistral-248M: prompt ids and chat templates are identical, and the logits agree to a correlation ≥ 0.99998 with
the same model run on Q8_0-rounded weights (greedy differences come from the quantization, mostly at near ties), see
[docs/14.13](docs/14-model-families.md#1413-verified-on-real-weights) and `scripts/real-weights-check.sh`.
Phi-3/4, `glm4`, Mistral-7B, Qwen3-MoE, DeepSeek and Qwen3-Next are still verified on synthetic checkpoints and
configs only (Qwen3-Next against an independent float64 reference, `scripts/qwen3next-ref.py`).

The `./jllm` launcher compiles the Java code and the native Metal library on first use. Equivalent manual steps:

```sh
mvn -q compile && ./scripts/build-native.sh
$JAVA_HOME/bin/java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
     -cp target/classes org.robomiri.jllm.Main serve
```

### Pictures (vision-language models)

```sh
./jllm run  --model ggml-org/SmolVLM-256M-Instruct-GGUF --gguf-quant Q8_0 --image photo.png --prompt "What is in this picture?"
./jllm chat --model ggml-org/Qwen2.5-VL-3B-Instruct-GGUF --gguf-quant Q4_K_M --image a.png,b.jpg   # the pictures go with your first message
./jllm serve --model ggml-org/Qwen3-VL-2B-Instruct-GGUF --gguf-quant Q8_0                          # image_url parts in /v1/chat/completions
```

A vision-language GGUF model is two files: the text model and its multimodal projector `mmproj-*.gguf` (found next to
the model or in its Hub repository and downloaded with it, or named with `--mmproj`; `--vision.enabled false` loads
text only). A chat request carries a picture as an OpenAI `image_url` content part (a `data:` URL, or an http(s) URL
the server fetches, never from private addresses unless `vision.allow-private-urls`); the web console's chat box
attaches, pastes or drops pictures; a text-only model can use a loaded vision model through the `describe_image` tool.
Put the picture before the question. Settings: `vision.*` in [`jllm.properties`](src/main/resources/jllm.properties);
the chapter is [docs/20](docs/20-vision.md). `unsloth/Qwen3.8-27B-GGUF` (Qwen3.5, a Gated DeltaNet hybrid) sees pictures on CPU
and Metal and matches llama.cpp's prompt ids, first-position log-probs and greedy answer. Gemma 3: the projector and the
attention that runs both ways inside a picture are implemented and checked, and the `gemma3` GGUF text architecture is mapped
(docs/16 §16.15; its layout is checked against llama.cpp on tiny files, no real Gemma 3 file was loaded).

### API

```sh
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is 7*8?"}],
  "stream": true, "enable_thinking": true, "max_tokens": 300 }'
```

Endpoints: `POST /v1/chat/completions`, `POST /v1/completions` (both support `"stream": true`), `GET /v1/models`,
`GET /health`, `GET /` (chat UI), with an image model loaded `POST /v1/images/generations` and `/v1/images/edits` ([docs/22](docs/22-image-generation.md#2212-api-command-line-console-and-chat-tools)), with a video model loaded `POST /v1/videos/generations` and `/v1/videos/encode` ([docs/23](docs/23-video-generation.md#239-api-command-line-console-and-chat-tool)), and with audio models loaded `POST /v1/audio/transcriptions`, `/v1/audio/translations` and `/v1/audio/speech` ([docs/21](docs/21-audio.md)). Sampling fields: `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`,
`presence_penalty`, `frequency_penalty`, `seed`, `max_tokens`, `stop`, `enable_thinking`, `logit_bias` (token id →
−100…100, added to the logits). Also supported:

- `n` (up to 16 choices; the prompt is prefilled once and reused from the prefix cache, streamed chunks carry the
  choice `index`);
- `logprobs` / `top_logprobs` (chat) and `logprobs: k` (completions): the log-softmax of the model's raw logits,
  `z_t − log Σ exp(z_j)`, before penalties, bias, temperature and truncation; streamed too;
- `response_format` `json_object` or `json_schema`: constrained decoding, where tokens that would make the output an
  invalid JSON prefix are masked before sampling (a byte-level automaton checks the most likely candidates first),
  enforcing `type`, `properties`, `required`, `additionalProperties`, `items`, `enum`/`const`, `anyOf` of different
  types, local `$ref`, min/max items and length; other schema keywords are a 400. Text the schema leaves no choice
  about is appended without sampling (jump-forward), and speculative decoding stays on (drafts are checked against
  the constraint);
- completions only: `echo` (with prompt logprobs), `suffix` (fill-in-the-middle for tokenizers with
  `<|fim_prefix|>`/`<|fim_suffix|>`/`<|fim_middle|>`, a 400 otherwise) and `best_of` (the n most likely of best_of
  candidates; not with `stream`).

Details and the math: [docs/10 §10.6](docs/10-engine-and-server.md#106-the-openai-compatible-api).

## Docker

One image with the engine and the web console, for any machine with Docker (CPU backend: a container cannot use
the Metal GPU, also not on a Mac):

```sh
docker run -d --name jllm -p 8090:8090 -v jllm-data:/data <you>/jllm-serve:latest
docker logs jllm                      # the generated API key the console asks for
open http://localhost:8090/
```

Build it with `scripts/docker-build.sh` (add `-t <you>/jllm-serve:1.0 --push` for a linux/amd64 + linux/arm64
image on Docker Hub) or `docker compose up -d`. Data, settings and the key live in the `/data` volume. Details,
environment variables and memory sizing: [docs/19](docs/19-docker.md).

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
- **Pictures**: with a vision-language GGUF model (SmolVLM, Qwen2.5-VL, Qwen3-VL; the projector is loaded with it,
  the load form can name another or none) the chat box takes pictures: the attach button, paste and drag and drop;
  thumbnails stay in the browser, the model list shows a *vision* badge, and `image_url` parts work on `/v1/chat/completions`
  and `/api/chat`. The built-in `describe_image` tool lets a text-only model ask a loaded vision model about a picture
  (URL, `data:` URL or an id from `POST /api/images`). [docs/20](docs/20-vision.md)
- **Tools**: pick per chat which functions the model may call. Built in: `web_search` (DuckDuckGo with no key,
  or a SearXNG instance, or the Brave Search API), `fetch_url` (a page as readable text; private addresses refused
  unless allowed), `calculator`, `current_time` and `describe_image`. **MCP servers** add their tools: stdio commands
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
  on disk can be loaded again or deleted: *On disk* lists `model.dir` and the folders of `jllm.web.model-dirs`
  (default `../models`, so a console started from `jllm-web/` also shows the repository's `models/` folder that
  `./jllm` uses), grouped by folder. A model is loaded from the folder it is in, and a checkpoint whose quantized
  cache is there loads from the cache without contacting the Hub, even after its shards were deleted (`model.low-disk`).
- **Images**: pictures from an image model (Stable Diffusion, SDXL; load one on the Models page like a text model):
  prompt, negative prompt, size, steps, guidance, seed, scheduler, batch, a progress bar per step with live previews;
  a gallery kept in the browser (IndexedDB: download, use as input, re-run); and an editor on a canvas with crop,
  rotate / flip, resize, brightness / contrast / saturation, a brush and eraser that paint an inpainting mask, outpainting
  (extend the canvas and inpaint the border), undo / redo, PNG export and image-to-image / inpaint requests with a strength
  slider. The chat tools `generate_image` and `edit_image` let a model draw and change pictures, shown inline in the chat.
- **Video**: clips from a video model (AnimateDiff; load the motion adapter on the Models page, which fetches the Stable Diffusion base too):
  prompt, negative prompt, size, frames, fps, steps, guidance, scheduler, file format, seed; a progress bar per step with a live preview of the
  first frame; a clip editor (canvas player with scrubber, frame strip, trim start / end, delete or hold a frame, reverse, ping-pong, speed,
  loop / ping-pong / once playback, export as MP4 / WebP / GIF); a gallery kept in the browser (IndexedDB). The chat tool `generate_video`
  shows the clip inline in the conversation.
- **Settings**: every key of `jllm.properties` with its documentation, its default and where its value comes
  from. Changes are validated and stored in `jllm-web.properties` (on top of `$JLLM_CONFIG` /
  `./jllm.properties`, as for the CLI). Sampling and chat defaults apply to the next request; backend, context and
  weight settings to the next load.
- **Logs**: the engine's output (download progress, conversion, MCP server stderr, errors) live, with a filter.
- **API**: `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions` (same fields and streaming
  behaviour as `jllm serve`, including `n`, `logprobs`, `logit_bias`, `response_format`, `echo`, `suffix` and
  `best_of`). Chat completions also accept OpenAI **function calling**: `tools` and `tool_choice`
  (`auto`, `none`, `required` or a named function); the model's calls come back as `tool_calls` with finish reason
  `tool_calls`, and `tool` messages carry the results back. Calls are **constrained**: inside `<tool_call>` the model
  can only write a known function name and arguments matching its `parameters` schema, `required` / a named function
  force the call, the fixed parts (`{"name": "`, `", "arguments": `, `}</tool_call>`, …) are appended in one pass
  without sampling, and `response_format` may be combined with `tools` (calls or that JSON). `POST /api/chat` is the console's chat with server-side
  tools (an event stream). The management API: `/api/models/load|download|loaded|local`, `/api/hub/inspect`,
  `/api/jobs`, `/api/tools` (`/settings`, `/call`, `/mcp`), `/api/settings`, `/api/logs`, `/api/status`,
  `/api/kv-cache` (`GET`, `POST /save`, `DELETE`).

Function calling uses the Hermes / Qwen format (`<tools>` in the system prompt, `<tool_call>` blocks in the answer,
`<tool_response>` for results), which Qwen2.5 and Qwen3 were trained on; other families follow it less reliably, and
a 0.6B model picks the right tool for simple requests but not for long chains. MCP stdio servers run as child
processes with your user's rights, and the tools of an MCP server can do whatever that server allows (write files,
call APIs), so only add servers you trust and enable their tools in chats where you want them.

### Coding agents (opencode, Continue, aider, Cline)

The console serves [opencode](https://opencode.ai) (and other agents that speak the OpenAI API) directly:

1. `./jllm-web/run.sh`, then on the Models page load a model with **⚡ Coding agent preset** — context 32768, f16 KV
   cache, 2 slots, a long queue timeout. opencode's instructions and tool definitions alone are ~11k tokens.
2. In your project: `./jllm-web/opencode.sh` (or *Connect* page → *Download opencode.json*). It writes an
   `opencode.json` for the loaded models — provider `jllm` over `@ai-sdk/openai-compatible`, context and output
   limits, `tool_call`, Qwen3 thinking off for speed (`"enable_thinking": false`, switch it on for hard tasks), no
   overall request timeout — plus the warnings that apply (context too small, a single slot, a model family that
   follows the tool-call format less reliably).
3. Run `opencode`. Every request is one line in the console's Logs (slot, prompt and cached tokens, speed, finish).

**Continue, aider and Cline** get ready-made settings too, on the *Connect* page or from a terminal with
`./jllm-web/agent-config.sh continue|aider|cline` (`opencode.sh` is `agent-config.sh opencode`):

| agent | what is generated | where |
|---|---|---|
| [Continue](https://docs.continue.dev) | `config.yaml`: an `openai` provider per loaded model with `apiBase`, context length and max tokens, `tool_use` for agent mode, a one-hour request timeout, Qwen3 thinking off through `extraBodyProperties` | `~/.continue/config.yaml` (an existing file is kept; the new one goes next to it as `config.jllm.yaml`) |
| [aider](https://aider.chat) | `.aider.conf.yml` (`model: openai/<id>`, `openai-api-base`, timeout), `.aider.model.metadata.json` (context and output limits, zero cost) and `.aider.model.settings.yml` (`edit_format: whole` for models under 4 GB, `diff` above, repo map, thinking off) | the repository root |
| [Cline](https://cline.bot) | the values for its *OpenAI Compatible* provider: base URL, API key, model ID, context window, max output tokens (Cline keeps its settings in the editor, not in a file) | printed / shown on the page |

Each comes with the warnings that apply: a context too small for that agent's prompt (Continue's agent mode ~5k
tokens, aider ~4k with its repository map, Cline ~12k), a single slot, a family that follows tool and edit formats
less reliably, and for Cline that Qwen3 thinking can only be switched off server-side (`chat.thinking=false`).
Checked with aider 0.86 against Qwen3-0.6B: it connects, keeps the conversation's prefix cached between requests,
applies whole-file edits and commits them (the 0.6B model itself is not a reliable editor: use 4B or larger).

What makes it work rather than merely connect:

- **Slots.** A model can be loaded with several KV caches on the same memory-mapped weights; each request goes to
  the slot that already holds the longest prefix of its prompt. opencode sends a title request alongside every
  message: with one cache that evicted the 10k-token conversation and the next step prefilled it all again; with two
  slots the next step finds it cached (measured: 10 484-token prompt, 10 443 cached, prefilled in 2.5 s).
- **Long first prefill without timeouts.** Streaming responses start at once and send an SSE keep-alive comment
  every 15 s while the prompt is prefilled; a client that disconnects — streaming or not — stops its request within
  one prefill chunk or one token and frees the slot, and a retry of the same prompt continues where it stopped.
- **Tokenizing only what is new.** Each request resends the whole conversation; the ids of recent prompts are kept
  (`engine.prompt-cache-mb`, 64), so only the new messages are tokenized, with exactly the ids of a full
  tokenization (a 100k-character, 50-turn conversation: 15.6 ms → 0.5 ms per request, [docs/17](docs/17-agent-speedups.md#175-tokenizing-only-what-is-new)).
- **`reasoning_content`.** A `<think>` block is returned apart from the answer (DeepSeek style), as opencode, Open
  WebUI and Continue expect; `"reasoning_format": "none"` keeps it inline.
- **Saved conversations.** A slot's KV cache is written to disk (`kv-cache/`) after a prefill of 2 s or more and
  when a model is unloaded or the server stops, in the background: the next request of the conversation runs while
  the file is written. When a request's prompt shares a prefix with a saved conversation that covers more than its
  slot holds, that prefix is loaded back instead of prefilled (the rest of the file is skipped). DeepSeek's latent
  (MLA) caches are saved too. Every new opencode session starts with the
  same ~10k tokens of instructions, so after a restart the first step costs a disk read instead of minutes (measured:
  2 707-token prompt, 21.3 s prefill; after a restart restored in 0.9 s, 2 706 tokens reused). The Models page lists
  the saved caches (*save now*, *delete all*); `jllm.web.kv-cache=false` turns it off, `jllm.web.kv-cache-max-gb`
  (default 10) bounds the disk use (least recently used deleted first), `jllm.web.kv-cache-dir`,
  `-min-tokens` (512) and `-save-after-ms` (2000) tune it ([docs/17 §17.3](docs/17-agent-speedups.md#173-saved-conversations)). A cache is only reused by the same model with the
  same load settings (quantization, backend, KV type); f16 takes 2 bytes × 2 × layers × kv width per token (115 kB
  for Qwen3-0.6B, so ~1.2 GB for opencode's 10k-token prompt).
- **Long-prompt attention** was reworked for this: K stored transposed in 32-key chunks so scores need no per-key
  reduction, the f16 cache converted with vector code, a vectorized softmax exp, register-blocked kernels, and
  decode split across threads when a model has fewer kv heads than cores; prefill and decode still agree bit for
  bit. A 6 033-token prompt prefills in 60 s (313 s originally, 115 s after the first round) and decode after it
  runs at 7.6 tok/s instead of ~2–3, with Qwen3-0.6B on 4 CPU cores ([docs/07](docs/07-cpu-backend.md#75-attention-kvcacheattention)).

Measured end to end on a 4-core cloud VM with Qwen3-0.6B (`opencode run "Create a file named hello.txt containing
exactly the text: hi"`): the first step prefilled opencode's 10 447-token prompt in 349 s and called `write`, the
second step reused 10 443 cached tokens and answered in 2.5 s + generation. The first step of a session is the slow
one on a CPU; later steps only prefill what is new. A larger model (Qwen3 4B+, Qwen2.5-Coder) plans and edits far
better than 0.6B but prefills proportionally slower; on Apple Silicon the Metal backend is the one to use.

Memory: on the CPU backend the KV caches live on the Java heap (3.8 GB per slot for Qwen3-0.6B at 32768 with f16;
larger models need more). `run.sh` gives the JVM 60% of the RAM by default (`JLLM_XMX` to set it) with a small young
generation, so the caches can use almost all of it; a load whose caches do not fit fails at once with a message
saying how much is needed, instead of thrashing the garbage collector.

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
Notable keys besides the usual ones: `backend.kv-dtype` (`f32`, `f16`, `q8`: int8 with a scale per row of a
kv head, about a quarter of the f32 cache, or `q4`: Hadamard-rotated keys and 4-bit groups of 32 with a half scale,
about an eighth, for long contexts), `backend.kv-q4-recent` (q4: the latest positions also kept in half precision,
default 128, 0 = none), `backend.kv-q8-rotate` (q8: rotate the keys before the int8 quantization, default `true`,
`false` = the previous format; saved q8 caches of the old format are refused under the new default), `backend.kv-paged` (default false: one pool of `backend.kv-block`-position blocks per model with a
block table per slot and a prefix tree, docs/24; `backend.kv-pool-tokens`, `backend.kv-prefix-share`), `backend.cpu.int8-activations`
(default `auto`: times int8 and float activations on the loaded model once and keeps the faster; `true` quantizes
activations to int8 for integer dot products, several times faster decode on ARM but ~1 % different numerics;
`false` gives results identical to Metal), `backend.cpu.native` (default `auto`: the int8 decode dot products
run in the small C library `libjllm_cpu` when it is found and passes its self-check, using AVX512-VNNI / AVX-VNNI /
AVX2 or ARM dotprod / NEON; `true` requires it, `false` keeps the Java kernels; build it with
`scripts/build-native-cpu.sh`, which `./jllm` does when a C compiler is installed) and `backend.cpu.native-lib` (its
path, default: `target/native`, next to the jar, `-Djllm.cpu.nativeLib`), `engine.max-prompt-bytes`, `hub.timeout-ms`
`vision.*` (the multimodal projector, where the picture encoder runs, token budget, size limits: [docs/20](docs/20-vision.md#209-settings)),
`engine.constrained-tools`, `engine.jump-forward` and `engine.constraint-masks` (constrained tool calls, text a constraint forces appended without sampling, and the allowed-token masks: each `false` keeps the old path, [17 §17.7](docs/17-agent-speedups.md#177-constrained-tool-calls-and-jump-forward)),
`backend.metal.cpu-moe` (`auto`: when the model exceeds Metal's working set, its least used MoE experts run on
the CPU; `off`, `all` or a number of whole layers), `backend.batch` (prefill chunk: `auto` = 256 for MoE models and for dense models on Metal, 64 otherwise; a number overrides) and the `server.*` limits listed under Security below.

Short flags:

```
--model REPO|FILE    Hugging Face repository (default Qwen/Qwen3-0.6B) or a local .gguf file, --revision REV, --dir PATH (model directory)
--backend cpu|metal   default: metal on macOS when the native lib builds, else cpu
--quant q4|q8|q2k|q6k weight format (default q4; q2k = 2.625-bit Q2_K, q6k = 6.5625-bit Q6_K); cached per model
--head-quant q8|q6k   format of the embedding / LM head (default q8; q6k = Q6_K, a separate jllm-q4_0-h6k cache)
--gguf-quant LABEL    GGUF repository: the file whose name carries LABEL (Q4_K_M, Q8_0, ...); --gguf-file NAME: exact file name
--low-disk            fetch + quantize + delete safetensors shards one by one (big models)
--ctx N               context length / KV cache size (default 4096)
--batch N|auto       prefill chunk size (default auto: 256 on Metal and for MoE models, 64 on the CPU)
--backend.kv-dtype f32|f16|q8|q4   KV cache element type (no short alias); --backend.kv-q4-recent N; --backend.kv-q8-rotate true|false
--backend.kv-paged true|false     paged KV cache with a prefix tree (docs/24); --backend.kv-block N, --backend.kv-pool-tokens N
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
| Qwen3-0.6B Q4_0, 413-token prompt | Metal | ~1 430 tok/s | ~117 tok/s |
| Qwen3-0.6B Q4_0, 413-token prompt | CPU (NEON 128-bit, 10 threads), int8 activations (`backend.cpu.int8-activations`, the default) | ~275 tok/s | ~51 tok/s |
| Qwen3-30B-A3B Q4_0 (19.4 GB cache) | Metal | ~49 tok/s | ~25 tok/s |
| Qwen3-30B-A3B Q4_0 (19.4 GB cache) | CPU, int8 activations | ~12 tok/s | ~6 tok/s |

Float activations on the CPU (`--backend.cpu.int8-activations false`) decode about 4× slower; the full history of
measurements is in [docs/11](docs/11-performance.md#111-measured).

Against llama.cpp 0.7.0 (Homebrew, build 11669, Metal) on the same `unsloth/Qwen3-0.6B-GGUF` Q4_0 file, 413-token
prompt, 128 tokens decoded at depth 413, speculation off (`scripts/mac-validate.sh --vs-llama`, 2026-10-22, after the
prefill chunk, multi-row matvec and decode attention of [docs/08 §8.29 – 8.31](docs/08-metal-backend.md#829-the-prefill-chunk-backendbatchauto)):

| Engine | Prefill | Decode |
|---|---|---|
| jllm (Metal) | 1 903 tok/s | 138 tok/s |
| llama.cpp (`llama-bench`, best of flash attention off/on) | 3 097 tok/s | 217 tok/s |
| jllm / llama.cpp | 0.61× (was 0.44×) | 0.63× (was 0.51×) |

With its default n-gram speculation jllm decodes this benchmark at 205 tok/s, but the bench prompt repeats one
sentence, which flatters speculation; llama-bench has none.

Both backends produce identical greedy output with float activations on the CPU (checked on every merge for
Qwen3 and the 30B MoE, and across 20 synthetic models of all families × 3 quantization formats by
`scripts/arch-smoke.sh`, which runs the CPU side with `backend.cpu.int8-activations` off); int8 activations
change the CPU decode numerics by ~1 % per dot product, so greedy output may differ from Metal's after a while.
The MLA latent KV cache of DeepSeek models is much smaller than a plain one (V2-Lite: 553 KB → 62 KB per token,
[docs/14](docs/14-model-families.md)).

## Tests

```sh
mvn test                   # SmokeTest + ServerTest + MetalParamsTest + the GGUF checks (plain main() classes, no test framework)
sh scripts/arch-smoke.sh   # 21 synthetic models x 5 formats, CPU == Metal (MoE also with experts on the CPU); DeepSeek vs float64
METAL_EMU=1 sh scripts/arch-smoke.sh            # the same on any OS, Metal side on the CPU emulator (needs clang++, python3)
sh scripts/build-metal-emu.sh && python3 -I scripts/metal-emu/kernel_tests.py   # single Metal kernels on the emulator vs numpy
REAL_PYTHON=<venv>/bin/python scripts/real-weights-check.sh Qwen/Qwen2.5-0.5B-Instruct   # engine vs transformers on real weights (docs/14.13)
```

`mvn test` also runs `DiffusionTest`: tiny random Stable Diffusion pipelines (SD 1.x, SDXL, inpainting; a few hundred kB each,
`src/test/resources/diffusion`) against the outputs of Hugging Face diffusers for the same weights and noise, on the CPU and the
GPU (`scripts/diffusion/make-tiny.py`, `reference.py`; the real-model checks and the speed comparison are in
[docs/22](docs/22-image-generation.md#2213-verified-against-diffusers)).

`mvn test` also runs `VideoTest` (a tiny random AnimateDiff pipeline against diffusers' `AnimateDiffPipeline` for DDIM / Euler / DPM++ on the CPU and
the GPU, the GIF / animated-WebP / MP4 writers by round trips, the video API and the adapter download from a fake Hub; `scripts/diffusion/make-tiny-video.py`,
`reference_video.py`) and `WanTest` (a tiny random Wan 2.1 pipeline, its UMT5 encoder, tokenizer, transformer, VAE and UniPC sampler against diffusers and
transformers; `make-tiny-wan.py`, `reference_wan.py`); the real-model checks are in [docs/23](docs/23-video-generation.md#2310-verified-against-diffusers).

`mvn test` also runs `ToolCallTest` (constrained tool calls, jump-forward and speculation under the constraint, on a scripted model and on Qwen3-0.6B) and `ConstraintBench` in its quick mode (the token masks against a brute-force oracle); `scripts/tool-call-check.sh` checks on the CPU and on Metal that the constrained greedy output is the same with speculation off, prompt lookup and draft models ([docs/17 §17.7](docs/17-agent-speedups.md#177-constrained-tool-calls-and-jump-forward)). `mvn test` needs `models/Qwen_Qwen3-0.6B` (run `./jllm download` first); `ServerTest` checks the HTTP server
against a fake backend (settings validation, bad requests, keep-alive, slow, stalled and disconnected clients). Two more checks are run by hand:

```sh
CP=target/classes:target/test-classes
J="$JAVA_HOME/bin/java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -cp $CP org.robomiri.jllm"
$J.TokenizerFamiliesTest /tmp/jllm-tokenizers   # 4 239 checks: 14 tokenizer families + 5 Unigram/WordPiece tokenizers
$J.ConverterTest --dir <scratch dir>            # converter, resume and cache-invalidation tests
```

`VisionTest` (run by `mvn install`; plain `main`, no model download) checks pictures as input: resize against Pillow, request
parts, the rotary positions, and tiny random Qwen2-VL / Qwen3-VL style models written as GGUF files (prompt layout,
cache keyed by the picture, CPU == Metal). Comparing the encoders with llama.cpp needs a vision model and
`libmtmd`; the numbers are in [docs/20.10](docs/20-vision.md#2010-verification).

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
  cpu/                   Vector API kernels (float, int8 and K-quant int8), thread pool, KV cache, CPU forward pass
  metal/                 FFM bindings + Metal forward pass
  sampling/              sampling parameters and sampler
  engine/                generation loop, prefix cache, stop strings, KV cache files
  diffusion/             image and video generation: CLIP tokenizer + text encoders, UNet (with AnimateDiff motion modules), VAE, samplers, pipelines, Wan 2.1 (transformer, 3-D VAE, UMT5), file writers (GIF, WebP, MP4), Ops (CPU, Metal), image and video API
  vision/                pictures as input: preprocessing (Pillow-compatible resize), the vision transformer on CPU and Metal, chat-request parts, projector discovery
  server/                HTTP server, OpenAI-compatible API, web UI
src/main/native/jllm_metal.m                       Objective-C shim exposing a C ABI over Metal
src/main/resources/org/robomiri/jllm/metal/kernels.metal   GPU compute kernels
src/main/resources/org/robomiri/jllm/metal/vision.metal    GPU kernels of the vision encoder
src/main/resources/org/robomiri/jllm/metal/diffusion.metal  GPU kernels of the image-generation stack (own Metal library)
docs/                    how it all works
scripts/                 build-native.sh, arch-smoke.sh and the synthetic-model generators
jllm-web/                Spring Boot web console + OpenAI API (separate Maven project, depends on the engine jar)
  src/main/java/org/robomiri/jllm/web/
    models/              Hugging Face links, download / load jobs, loaded models and slots, models on disk, saved conversations
    chat/                /api/chat (the agent loop) and what it shares with the OpenAI endpoints
    tools/               tool-call prompt format and parsing, built-in tools (search, fetch, calculator, time)
    mcp/                 MCP client: stdio, Streamable HTTP and SSE transports, server manager
    openai/              /v1 endpoints
    images/              image models in memory, /v1/images/generations and /edits, the pictures served by link
    video/               video models in memory, /v1/videos/generations and /encode, the clips served by link
    settings/            jllm.properties view and persisted overrides
    api/                 /api management, tools and integrations (opencode, Continue, aider, Cline) endpoints, API key filter
    logs/                captured engine output
  src/main/resources/static/   the console (plain HTML/CSS/JS modules, no build step)
  run.sh                 launcher
  agent-config.sh        writes opencode / Continue / aider settings for the running console (prints Cline's); opencode.sh = agent-config.sh opencode
```

## License

MIT, see [LICENSE](LICENSE).
