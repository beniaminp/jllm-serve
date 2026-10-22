# 20. Pictures as input — vision-language models

Code: `vision/` (`Preprocess`, `Resize`, `ImageDecoder`, `ClipModel`, `ClipCpu`, `ClipMetal`, `VMath`, `VisionContext`,
`MediaChat`, `ImageParts`, `ImageFetcher`, `Mmproj`), `metal/vision.metal`, `engine/MediaPrompt`,
`engine/Engine.generate(MediaPrompt, …)`, `model/RopePositions`, `Rope.rows`, `Backend.forwardEmbeds`,
`Backend.setRopePositions`; in `jllm-web`: `WebImages`, `DescribeImageTool`, `ImagesController`.

Until now the model read words. This chapter gives it eyes: a chat request may contain pictures, and a model such as
SmolVLM, Qwen2.5-VL, Qwen3-VL or Gemma 3 answers questions about them. Nothing new is needed on the text side except
one thing: a way to put numbers into the prompt that did not come from the embedding table.

## 20.0 🧸 Explained simply

Imagine the model is a very clever friend who only understands *written words*. You want to show them a photo. You
cannot hand them the photo, so you hire a **translator** who is good at looking at pictures and describing them — but
not in English. The translator writes the description in the friend's own secret language: a row of little "picture
words", each one a bundle of numbers just like the bundle of numbers that every ordinary word turns into.

How does the translator look? Like you look at a jigsaw puzzle. First the photo is **cut into small squares**
(patches, 14 or 16 pixels wide). Each square alone says little ("some blue"). So the translator lets the squares
**talk to each other** many times over (that is the *vision transformer*, the same attention you met in chapter 5, but
every square may look at every other square, and nobody is blindfolded to the future). After a few dozen rounds a
square no longer means "some blue" but "the edge of a red circle on a blue background". Because 4 000 squares would
be far too many words, neighbouring squares are **merged** (four into one, or sixteen into one) before they are
handed over. The friend then reads the sentence

```
<start of picture> picture-word picture-word picture-word … <end of picture> What shapes do you see?
```

and answers. The friend never learned that the first words were not typed by a person: they have the right size and
the right "feel".

Two details make it work in practice:

* A picture is **not a word**, so we cannot look it up by its number to remember it. The engine remembers a picture
  by its *fingerprint* (a hash of the pixels): the same picture in the next turn of the conversation is not looked at
  again and its part of the memory (the KV cache) is reused.
* Some models (Qwen) also tell the friend **where in the picture** every picture-word sits: "row 3, column 5". Words in
  a sentence have one position, a picture-word has two. That is *M-RoPE*, 20.6.

## 20.1 The files

A vision-language GGUF model is **two files**: the text model (`Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf`, loaded like any
GGUF, chapter 16) and the **multimodal projector** (`mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf`, llama.cpp's `clip`
format) which holds the vision transformer and the small network that maps its output to the text model's width. The
projector is found, in this order:

1. `model.mmproj` (`--mmproj`): a path, or the name of a file next to the model or in its Hub repository;
2. an `mmproj*.gguf` next to the model file (several: the one whose name matches the model's, then F16 > BF16 > Q8_0);
3. the model's Hub repository (`vision.auto-download`, on by default): the best match is downloaded with the usual
   resume and SHA-256 check, next to the model.

`vision.enabled=false` (or no projector anywhere) gives a text-only model, and a request with a picture then fails
with HTTP 400 *"this model cannot see images: it was loaded without a vision projector"* instead of being silently answered blind.

Projector types implemented (`clip.projector_type`), each checked against llama.cpp (20.10):

| Type | Models | Vision transformer | Tokens per picture |
|---|---|---|---|
| `idefics3` | SmolVLM-256M / 500M / 2B | SigLIP, 512 px tiles, patch 16, learned positions | 64 per 512 px tile (pixel shuffle 4 × 4), plus an overview tile |
| `gemma3` | Gemma 3 4B / 12B / 27B | SigLIP, 896 px, patch 14 | 256 (average pooling 4 × 4) |
| `qwen2vl_merger`, `qwen2.5vl_merger` | Qwen2-VL, Qwen2.5-VL | any size, patch 14, 2-D rotary, windows (2.5) | one per 28 × 28 pixels (2 × 2 patches merged), 8 … 4 096 |
| `qwen3vl_merger` | Qwen3-VL (2B …), and the projector of `unsloth/Qwen3.8-27B-GGUF` | any size, patch 16, learned positions resized to the picture, 2-D rotary | one per 32 × 32 pixels; Qwen3-VL also adds *deepstack* features (20.4) |

## 20.2 Preparing the picture

`ImageDecoder` decodes PNG / JPEG / GIF / BMP with `javax.imageio` (the pixel count is checked from the header before
any pixel is decoded: a 100 000 × 100 000 PNG is refused, `vision.max-image-pixels`), drops the alpha channel
(as stb_image does for llama.cpp: a transparent pixel keeps its colour), and hands over 8-bit RGB. `Preprocess` then does **what llama.cpp's `mtmd` does**, because that is what the
published projectors were validated with and what this project compares against:

* **Resize.** `Resize` is a port of llama.cpp's `img_tool` (itself modelled on Pillow): separable resampling with a
  kernel of support *s* (bilinear *s* = 1, bicubic with *a* = −0.5 *s* = 2, Lanczos-3 *s* = 3), widened by the scale
  factor when shrinking, with Pillow's fixed-point coefficients (the weights are rounded to 22 bits, the sums to 8
  bit with `+ 128`) so the result is **bit-identical** to Pillow's `Image.resize` for the cases in `VisionTest`
  (one case where Pillow ≥ 13 does the vertical pass first differs by one grey level).
* **SmolVLM (idefics3).** The longest edge is scaled to a multiple of 512 (at most 2 048) with Lanczos, the picture is
  cut into 512 × 512 tiles in raster order, and a 512 × 512 *overview* of the whole picture is appended. Each tile
  becomes 64 tokens; the tokens are framed with `<fake_token_around_image>`, `<row_R_col_C>` and `<global-img>` (20.5).
* **Gemma 3.** Fixed 896 × 896: scale to fit keeping the aspect ratio (bilinear), pad with black.
* **Qwen-VL.** Dynamic size. The size is the closest multiple of 28 (Qwen2/2.5) or 32 (Qwen3) that keeps the aspect
  ratio inside the token budget (`vision.image-min-tokens` / `-max-tokens`, 0 = the model's own 8 … 4 096): a
  640 × 480 picture becomes 644 × 476 = 23 × 17 = 391 tokens. The resize is bicubic and the remainder is padded
  (llama.cpp pads where Hugging Face's processor *stretches* the picture a few pixels; the difference is small and we
  follow llama.cpp, since the projector files come from its converter).
* **Normalize.** `x = (pixel / 255 − mean) / std` per channel with the `clip.vision.image_mean / image_std` of the file
  (0.5 / 0.5 for SigLIP and Qwen3, CLIP's constants for Qwen2).

## 20.3 The vision transformer

📐 A picture of `W × H` pixels with patch size `p` becomes `n = (W/p)(H/p)` patches. Patch `i` is the `3 p²` numbers of
its square. The "convolution with stride `p`" that every ViT starts with is just a matrix product on those rows:

```
x_i = P · patch_i + b            P: E × 3p²     (E = 768 … 1280)
```

(Qwen's conv has a second kernel for the second video frame; for a still picture both kernels see the same pixels and
are **added** into one matrix when the projector is loaded.) Then `L` blocks, each exactly the transformer block of
chapter 5 — with three differences:

1. **LayerNorm with a bias** (SigLIP, Qwen2, Qwen3) or RMSNorm (Qwen2.5) instead of RMSNorm only;
2. **a GELU or SiLU MLP** (up → activation → down), or SwiGLU (Qwen2.5);
3. **attention without a mask**: every patch attends to every patch, `softmax(q kᵀ / √d) v` over all `n` rows.

Where does a patch learn *where it is*? Three ways:

* **SigLIP (SmolVLM, Gemma 3):** a learned table of `n` position vectors added to `x` (`v.position_embd.weight`).
* **Qwen2 / Qwen2.5: 2-D rotary.** The head's `d/2` rotary pairs are split into two halves; the first quarter of the
  pairs rotates with the patch's *row* `y`, the second with its *column* `x`:

  ```
  pair j < d/4 :       angle = y · θ^(−2j/(d/2))
  pair d/4 ≤ j < d/2 : angle = x · θ^(−2(j−d/4)/(d/2))        (θ = 10 000)
  ```

  so `q·k` depends on the *offset* in rows and in columns, as RoPE's `q·k` depends on the token offset in chapter 5.
* **Qwen3-VL: both.** A learned table of 48 × 48 vectors (`v.position_embd.weight`) is **resized to the picture's
  `gx × gy` grid** with bilinear interpolation with aligned corners (ggml's `BILINEAR | ALIGN_CORNERS`: grid
  coordinate `u` samples the table at `u (48−1)/(g−1)`), added to `x`, and the 2-D rotary above is used as well.

**Windowed attention (Qwen2.5-VL).** Full attention over 4 000 patches costs `n²`. Qwen2.5's ViT uses attention inside
**8 × 8-token windows** (112 px) for most layers and full attention only every `n_wa_pattern`-th layer. The patches are
reordered once so that each window is a contiguous range of rows (`blockOrder`), the kernel gets a `[lo, hi)` range per
row, and the output is put back in raster order at the end. Full-attention layers use the range `[0, n)`.

**Merge order.** Qwen merges 2 × 2 neighbouring patches into one token. The patches are therefore listed **block by
block** (the four patches of a block are consecutive rows), and the merger is a plain matrix product over *four
consecutive rows read as one row of 4E numbers*.

## 20.4 From patches to "picture words" (the projectors)

| Projector | Computation |
|---|---|
| `idefics3` | **pixel shuffle:** each 4 × 4 block of patches is concatenated into one row of `16 E` numbers, then one linear map to the text width (768·16 → 576 for the 256M model) |
| `gemma3` | **average pooling** of each 4 × 4 block, RMSNorm with `mm.soft_emb_norm`, then a linear map (`mm.input_projection`, stored transposed) |
| `qwen*` | post-LayerNorm/RMSNorm, then the merger `fc2(GELU(fc1(x₄ₑ)))` over the four merged rows (4E → 4E → text width) |

**Deepstack (Qwen3-VL).** The picture does not only enter at the embedding layer. After blocks 5, 11 and 17 of the ViT
(`clip.vision.is_deepstack_layers`) a second, smaller merger `LayerNorm → fc1 → GELU → fc2` turns the hidden state of
that block into an extra feature of the same shape as the picture's embeddings. The text model **adds** feature `k`
to its hidden state at the picture's positions after its decoder layer `k`:

```
h ← layer_0(h);  h[picture rows] += deepstack_0
h ← layer_1(h);  h[picture rows] += deepstack_1
h ← layer_2(h);  h[picture rows] += deepstack_2   (then the remaining layers as usual)
```

`VisionEncoder.Encoded` therefore carries `embd` and `deepstack[k]`; `MediaPrompt.Span` carries them to the
engine, `Backend.forwardEmbeds(embeds, …, deepstack, logits)` adds them after layers 0 … K−1 (on Metal the pass is
split in two command buffers around the add, since `x` is host-visible memory and the add is tiny).

## 20.5 Putting the pictures into the prompt

The prompt is rendered from the chat template with the picture as a **mark** (U+FFFC) in the user's text; the
tokenizer turns the mark into a special id (`Tokenizer.IMAGE_MARK`) which `VisionContext.build` replaces by the family's
**layout**:

```
SmolVLM     <fake_token_around_image><row_1_col_1> [64 × <image>] <row_1_col_2> [64 × <image>] \n … \n
            \n\n<fake_token_around_image><global-image> [64 × <image>] <fake_token_around_image>
Gemma 3     <start_of_image> [256 × <image_soft_token>] <end_of_image>        (llama.cpp's layout; Hugging Face's processor adds \n\n on both sides)
Qwen-VL     <|vision_start|> [n × <|image_pad|>] <|vision_end|>
```

A span of placeholder tokens is *not* looked up in the embedding table: `Engine` feeds the span's rows from the encoder
to `Backend.forwardEmbeds(embeds, off, n, pos, deepstack, logits)` (a prefill chunk where the input is embeddings
instead of ids; everything after the embedding lookup is unchanged), in the prefill chunk sizes the engine uses anyway.

**The cache is keyed by the picture, not by the placeholder.** The KV cache, the saved conversations and the prompt
cache work on token ids, and every picture token has the same id. So `MediaPrompt.identity()` gives each placeholder a
*virtual id* `2³⁰ | h₃₀(SHA-256(WxH:pixels), index)`; the cache compares those. The same picture in the next turn matches
(`cached_tokens` equals the whole prefix), another picture matches only the text before it, and the sampler's repetition
penalties see the real tokens. An image is encoded once per conversation: `VisionContext` keeps the encoded embeddings in
an LRU (`vision.cache-mb`).

**Order matters.** The model attends only to the past, so write the picture *before* the question (as every
published template does); a picture placed after the question works but the question was read blind.

## 20.6 M-RoPE: positions with two coordinates

A text token `i` has the position `i`. A picture token has *three* numbers: time `t`, row `h` and column `w`. In a
still picture `t` is the same for all of its tokens (the position of its first token `s`), `h = s + row`, `w = s +
column`. The text after the picture continues from **the largest position the picture used**, `s + max(nx, ny)`, not
from `s + nx·ny`, so a 16 × 16 picture advances the text position by 16, not 256.

The head's rotary pairs are shared out between the axes by the GGUF's `rope.dimension_sections` (`[16, 24, 24]` for
Qwen2.5-VL with `d = 128`): pair `j` turns by `position_axis(j) · θ^(−2j/d)`. Qwen3-VL **interleaves** the axes:
pair `j` uses axis `j mod 3` while that axis still has pairs left (`j < 3 · section`), the rest use the text
position (`[24, 20, 20]`, `d = 128`; in llama.cpp the rest uses a fourth coordinate that is 0, a difference that
turns pairs of wavelength > 10⁶ tokens by < 0.01 rad over 10⁴ positions).

**How the kernels stay unchanged.** The backends read their rotary angles from a table of `cos / sin` rows, one per
cache slot (`Rope.cos[slot * half + pair]`). Positions that are not "the slot number" would need new kernels — but the
table is just data: `Rope.rows(table, ctx, from, RopePositions, cos, sin)` **rewrites the rows** of the slots from
`from` on with the angle of the axis each pair uses, and the CPU and Metal kernels run as before. After a request
with a picture, the rows are put back (`ropeDirty`), so a text request equals a text request on a fresh engine
(`VisionTest` checks that bit for bit). `forwardBatch` refuses a backend whose table is custom, so batched decoding
never mixes the two.

## 20.7 The Metal encoder

`ClipMetal` runs the whole ViT in **one command buffer per picture**, with the kernels of `vision.metal` (each with a
numpy reference in `scripts/metal-emu/kernel_tests.py` and tested in the emulator, chapter 18):

| Kernel | What it does |
|---|---|
| `vit_gemm` | `out = act(x · Wᵀ + bias) (+ out)`: 64 × 64 output tiles of four simdgroups (each 32 × 32 with 16 `simdgroup_float8x8` accumulators), K in steps of 32, `x` converted to half in threadgroup memory, **weights in half** (an F16 tensor is used in place from the mapped file, others are converted once), rows padded by 8 halves against bank conflicts |
| `vit_norm` | LayerNorm / RMSNorm with optional weight and bias, one threadgroup per row |
| `vit_rope` | 2-D rotary from a per-row `cos/sin` table |
| `vit_attention_mma_d{64,72,80,128}` | flash-style attention with `simdgroup_half8x8`: 32 query rows per threadgroup, keys in blocks, online softmax, a `[lo, hi)` key range per row (windows); `vit_attention` is the generic version for other head sizes |
| `vit_gated`, `vit_act`, `vit_add`, `vit_gather`, `vit_gather_mean` | SwiGLU, activations, residual add, the pixel-shuffle and pooling reshapes |

The CPU encoder (`ClipCpu`) is the same computation with `VMath`'s register-tiled float GEMM on the Vector API and a
thread pool. `vision.backend=auto` picks Metal when the library is there; the text model's backend is independent
(the encoder may be on the GPU while the text runs on the CPU, or the reverse).

## 20.8 Using it

```sh
./jllm run  --model ggml-org/SmolVLM-256M-Instruct-GGUF --gguf-quant Q8_0 --image photo.jpg --prompt "What is in this picture?"
./jllm chat --model ggml-org/Qwen2.5-VL-3B-Instruct-GGUF --gguf-quant Q4_K_M --image a.png,b.png   # the pictures go with your first message
./jllm serve --model ggml-org/Qwen3-VL-2B-Instruct-GGUF --gguf-quant Q8_0
```

The OpenAI form, on the engine's server and on the console's `/v1/chat/completions` (and `/api/chat`):

```json
{"model": "m", "messages": [{"role": "user", "content": [
   {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR…"}},
   {"type": "text", "text": "What shapes do you see?"}]}]}
```

`image_url.url` may be a `data:` URL (always accepted) or an `http(s)` URL (the server fetches it: at most 6
redirects, `vision.max-image-bytes`, and **never from a loopback, private-network or link-local address** unless
`vision.allow-private-urls=true`, the same rule as the console's `fetch_url` tool). Errors are HTTP 400 with an OpenAI
error body: no projector, undecodable picture, too large, too many pictures (16), a picture in a system/assistant message,
`logprobs` of the prompt with a picture.

**Web console.** The chat box accepts pictures (attach button, paste, drag and drop); the thumbnails are kept in
the browser (IndexedDB), the model list shows a *vision* badge, and the load form chooses the projector
(`model.mmproj`, or "none" = `vision.enabled=false`).

**`describe_image` tool.** A text-only model can still look: when a vision model is loaded next to it, the built-in
`describe_image` tool takes `{"image": "<url | data: URL | img-… id>", "question": "…"}`, asks the vision model and returns
its answer as the tool result. `POST /api/images` uploads a picture and returns the `img-…` id.

## 20.9 Settings

| Key | Default | Meaning |
|---|---|---|
| `model.mmproj` (`--mmproj`) | empty | the projector (path or name); empty = look next to the model and in its repository |
| `vision.enabled` | true | false = never load a projector |
| `vision.auto-download` | true | download the projector of a Hub model when none is on disk |
| `vision.backend` | auto | `cpu`, `metal` or `auto` for the encoder |
| `vision.image-min-tokens`, `vision.image-max-tokens` | 0, 0 | token budget of dynamic-resolution pictures (0 = the model's) |
| `vision.cache-mb` | 256 | encoded pictures kept for later turns |
| `vision.max-image-bytes`, `vision.max-image-pixels` | 32 MiB, 10⁸ | refusal limits |
| `vision.allow-private-urls` | false | allow fetching pictures from private addresses |

(`vision.max-*` are live settings in the console.) `scripts/mac-validate.sh --vision mmproj.gguf picture.png` measures the
encoder on the GPU against the CPU.

## 20.10 Verification

Against **llama.cpp** (`libmtmd` built from the same source as `llama-mtmd-cli`, a small dump tool in the scratch area
writes its prompt token ids and the embeddings of each chunk; `VisionDump` writes ours):

| Model, projector | Picture | Prompt ids | Embeddings, relative error ‖ours − llama‖/‖llama‖ |
|---|---|---|---|
| SmolVLM-256M, F16 | 640 × 480 | identical | 9.5·10⁻⁴ (CPU), 9.0·10⁻⁴ (Metal), tile and overview |
| Qwen2.5-VL-3B, F16 | 640 × 480 | identical (391 tokens) | 6.4·10⁻³ (mean per token 4.2·10⁻³, cosine ≥ 0.995) |
| Qwen3-VL-2B, F16 | 448 × 448 | identical | CPU 2.0·10⁻³, Metal 2.4·10⁻³; deepstack 0.5·10⁻³ … 3.3·10⁻³ |
| Gemma 3 4B, F16 | 448 × 448 → 896 | 256 tokens | Metal 2.6·10⁻³ (cosine ≥ 0.9997) |

The relative error of a half-precision GPU encoder against a float one is a few 10⁻³; with a **Q8_0** projector llama.cpp's
own output moves by 6.6·10⁻² against its F16 run (it quantizes the activations to 8 bit), which is why the table uses F16
projectors. Answers: for the same picture and prompt at temperature 0 the engine and `llama-mtmd-cli` give the same
text for Qwen2.5-VL-3B and Qwen3-VL-2B (the first 80 tokens of the `shapes.png` answer are identical) and, for SmolVLM-256M,
for the first sentence (17 tokens), after which the two take different words at a near tie (the embeddings differ from
llama.cpp's by 10⁻³, as the half-precision encoder differs from its float one); the same happens between the CPU's
int8-activation decode and the float one. Do not expect token-for-token identical long answers from any two engines.

`VisionTest` (113 checks, in `mvn install`): the resize against Pillow's hashes, size rules, picture decoding and the
refusal paths, the request parts and the SSRF rules, projector choice, the rotary positions, and three **tiny random
vision-language models written as GGUF** (a Qwen2-VL text model with a Qwen2.5-VL style projector, a Qwen3-VL one with
deepstack and interleaved sections, and a Qwen3.5 hybrid with a Qwen3-VL style projector, 20.12): the prompt layout, one hash identity per picture, deterministic answers, the cache
reused for the same picture and not for another, the rotary rows restored for text, `forwardEmbeds` with the rows of
some ids equal to `forward(ids)`, and the Metal encoder and text backend against the CPU ones.

## 20.11 Measured

Apple M1 Pro (10 cores, 16 GPU cores), JDK 26, `llama-mtmd-cli` build 11669 (Homebrew, Metal), all F16 projectors, the
machine shared with other jobs (every timed run held the project's one-heavy-job lock, but other work was not stopped, so
read differences below 10 % as noise). Pictures: `shapes.png` 640 × 480, `scene.png` 800 × 600 (scripts in the scratch area;
the commands are `VisionDump <mmproj> <picture> <prefix>` with `-Djllm.vision.twice=true -Djllm.vision.reps=7` and
`llama-mtmd-cli -m <text.gguf> --mmproj <mmproj> --image <picture> -p … -n 1 --no-warmup -ngl 99 -v`).

**Encoding the picture on the GPU, weights resident** (jllm: the fastest of 7 runs; llama.cpp: the "mtmd batch encoding
done" line of three process runs, which include its graph build):

| Projector | Picture → tokens (patches) | jllm Metal | llama.cpp Metal | jllm CPU |
|---|---|---|---|---|
| SmolVLM-256M | 512 px tile + 512 px overview → 64 + 64 | 101 + 101 ms | 84–89 + 74–75 ms | 612 + 606 ms |
| Qwen2.5-VL-3B | 812 × 588 → 609 (2 436) | 1 543 ms | 2 003–2 012 ms | 8 067 ms |
| Qwen3-VL-2B | 800 × 608 → 475 (1 900) | 764 ms | 617–632 ms | 4 807 ms |
| Gemma 3 4B | 896 × 896 → 256 (4 096) | 2 755 ms | 3 625–3 686 ms | 15 227 ms |
| Qwen3.8-27B's projector | 800 × 608 → 475 (1 900) | 950 ms | (no matching text model for llama.cpp) | — |

(One llama.cpp Qwen3-VL run printed 183 ms; it is faster than the GPU's arithmetic allows for this ViT and I do not
understand it, so the table keeps the other two.) llama.cpp's own CPU path for the projector (`--no-mmproj-offload`) took
5.7 s per SmolVLM tile and 107 s for the Qwen2.5-VL picture on this machine; I did not look for why (default threads),
the jllm CPU encoder uses the Vector API on a thread pool and is 8–13 times faster in those runs.

**Where the GPU time goes.** `AttnBench` and `GemmBench` (manual micro-benchmarks, 20 dispatches in one command buffer):

| Kernel | Problem | Rate |
|---|---|---|
| `vit_gemm` | 1 900 × 1 024 → 4 096, 4 096 × 1 152 → 4 304 | 2 500 GFLOPS (half weights; the GPU's peak is ~5 300) |
| `vit_attention_mma_d64` | 1 900 patches, 16 heads | 8.5 ms = 1 730 GFLOPS |
| `vit_attention_mma_d72` | 4 096 patches, 16 heads (Gemma 3) | 45.5 ms = 1 700 GFLOPS |
| `vit_attention_mma_d80` | 2 436 patches, 16 heads (Qwen2.5-VL, full-attention layers) | 17.5 ms = 1 740 GFLOPS |

The first version of the attention kernel kept K, V and the probabilities in float, staged them with two barriers per
block and walked eight softmax rows per simdgroup with `simd_max` reductions: 430 GFLOPS, which was 61 % of the Qwen3-VL
encoding and 76 % of Gemma 3's (1 341 ms and 6 433 ms before, 764 ms and 2 755 ms now). The kernel now stages K and V as
half in one barrier pair, shares a softmax row between 4 lanes (two shuffles instead of a simdgroup reduction), writes
the probabilities as half and skips the diagonal rescale while no row's maximum moved. A variant with 16 query rows
per simdgroup (more reuse of each K slice) ran at 640 GFLOPS because of its register use; 8 rows per simdgroup is the
fast one. The price is the half-precision softmax input: the output differs from the float kernel by 5·10⁻⁴ … 1.3·10⁻³
(`kernel_tests.py`), and the embeddings stay at the same few 10⁻³ from llama.cpp's.

**A whole answer** (64 new tokens, temperature 0, one process each so the numbers include first use of the pipelines;
the text model's own decode speed is the engine's, not the vision code's):

| Model (text + projector), Metal | jllm: encode, prefill, decode, total | llama.cpp: encode, prefill, decode, total |
|---|---|---|
| SmolVLM-256M Q8_0, 149 tokens | 422 ms (cold), 119 ms, 251 tok/s, 819 ms | 161 ms, 198 ms (751 tok/s), 268 tok/s, 581 ms |
| Qwen2.5-VL-3B Q4_K_M, 646 / 625 tokens | 1 743 ms, 3 761 ms (172 tok/s), 22.8 tok/s, 8 344 ms | 2 010 ms, 3 109 ms (201 tok/s), 59.9 tok/s, 5 181 ms |
| Qwen3-VL-2B Q8_0, 491 tokens | 1 170 ms (cold), 1 481 ms, 39.6 tok/s, 4 293 ms | (not captured) |
| SmolVLM-256M, CPU | 1 884 ms, 851 ms, 215 tok/s, 3 050 ms | 13.0 s (two tiles), 13.2 s, 183 tok/s, 13.8 s |
| Qwen2.5-VL-3B, CPU | 9 328 ms, 14 494 ms, 20.4 tok/s, 26 987 ms | 107 s, 124 s, 18.9 tok/s, 130 s |

The picture costs about as much as llama.cpp's on the GPU and a small model answers a picture in under a second; the
remaining gap on Qwen2.5-VL is the text model: its Q4_K_M decode on Metal (22.8 against 59.9 tok/s) is the engine's K-quant
speed (chapter 11), not something this chapter changed. Re-running the table is `scripts/mac-validate.sh --vision
mmproj.gguf picture.png` (encoder, GPU and CPU) and the commands above.

## 20.12 Qwen3.5 / "Qwen3.8": pictures for a hybrid model

🧸 Most of this model's layers do not look back at all: three of every four keep a *running summary* (a Gated DeltaNet,
chapter 14) that they update word by word, like a person who reads and remembers the gist. Only every fourth layer is
real attention. A picture-word goes into the summary like any other word; only the attention layers care *where* in the
picture it was, so only they need the two-coordinate positions of 20.6.

📐 What had to change for `Qwen3NextCpuBackend` and `Qwen3NextMetalBackend`, the same three things as for the dense backends:

* **Embeddings as input.** `forwardEmbeds` copies the picture's rows into the residual stream instead of looking tokens up;
  the layers are unchanged. (The multi-token-prediction head, if switched on, sees token 0 for these rows: it drafts a little
  worse right after a picture and the verification keeps every answer exact.)
* **The rotary table.** The GGUF's `rope.dimension_sections = [11, 11, 10, 0]` over the 32 pairs of the 64 rotating
  dimensions (of 256 per head) is read into the `RopeTable` as interleaved M-RoPE (`Qwen3NextArch`, the same
  `mrope_section` / `mrope_interleaved` keys as Qwen3-VL): pair `j` turns by `position_axis(j mod 3)` while
  `j < 3 × section`, which for these sections uses every pair (11 + 11 + 10 = 32), so the fourth, "extra" axis is not needed.
  `setRopePositions` rewrites the table rows of the picture's slots and restores them afterwards, exactly as 20.6.
* **No deepstack.** Qwen3.5's projector has none; `forwardEmbeds` refuses deepstack features for this family.

A recurrent state cannot be rewound, so the engine reuses the previous request only when the new prompt extends it
exactly (the next turn of a conversation does: the picture's tokens carry the same hash). A request that changes anything
before the end of the cached tokens, or the picture, prefills from the start.

Checked against `llama-mtmd-cli`'s library (`libmtmd`, GPU) on `unsloth/Qwen3.8-27B-GGUF` (`Q4_0` text file, `mmproj-F16`),
the 640 × 480 `shapes.png` and "What shapes and colors do you see in this picture?" with thinking off:
the prompt is 25 text tokens around a 300-token picture (325 in all), **identical ids**; the first position's top-5 log-probabilities are

| token | jllm | llama.cpp |
|---|---|---|
| `In` | −0.05444 | −0.05453 |
| `I` | −3.15014 | −3.15053 |
| `Based` | −4.63837 | −4.62878 |
| `Here` | −8.23555 | −8.21890 |
| `Hello` | −9.37048 | −9.33001 |

and the 40 greedy tokens of the answer are **the same 40 ids** ("In this picture, I see three distinct geometric shapes on a
solid blue background: 1. **A red circle** – Located on the left side of the image. 2. **…"). On a tiny random Qwen3.5 GGUF
with a Qwen3-VL style projector (`VisionTest`), the CPU and Metal backends agree: embeddings equal ids bit for bit,
explicit identity positions equal the plain ones bit for bit, shifted positions change the logits and are restored,
the answers of the two backends are the same, and text after a picture request equals a fresh engine's.

## 20.13 Gemma 3: attention that runs both ways inside a picture

Code: `Backend.forwardEmbedsBidirectional` (`model/Backend`), its two implementations in `cpu/CpuBackend.GqaStep.run` and
`metal/MetalBackend.GqaStep.run`, the feeding in `engine/Engine.generateLocked`, the flag `MediaPrompt.nonCausal` set by
`vision/VisionContext` for the `gemma3` projector, the batch rule in `ModelLoader.batch`; tests in `VisionTest`
(`bidirectional`, and the engine-level checks of `tinyModel`).

### 🧸 In plain words

When you read a sentence you only know the words *before* the one you are on: a word cannot peek at the ones that come
after it. A picture is not a sentence. You see all of it at once, the sun in the corner and the house in the middle
together. Gemma 3 was trained this way: the 256 "picture words" of one picture may look at *each other*, forwards as well as
backwards, and at everything that came before them. The words after the picture still read in the normal order, and they
can see the whole picture. Think of a queue of people, each of whom can only see the people in front of them, except that
the people standing together for a group photo may all look at each other; the people behind the group see the whole group.

The engine's job is to let the picture words look at each other without breaking the rule for the words around them. The
catch is that a picture word can only be looked at once the thinking for it has been done *in that same layer*, so the
whole picture has to be worked on in one go, not in small pieces.

### 📐 How it works

**The mask.** Attention (chapter 5) scores a query row `i` against every key row `j` and turns the scores into weights:

```
a_ij = softmax_j( s_ij + M_ij ),      s_ij = q_i · k_j / sqrt(256)        (Gemma 3: query_pre_attn_scalar = 256)
out_i = Σ_j a_ij v_j                  M_ij = 0 if row i may see row j, −∞ if not
```

Ordinary (causal) attention: `M_ij = 0` for `j ≤ i`. Gemma 3 adds: `M_ij = 0` for all `i, j` inside the same picture `S`.
The tokens `<start_of_image>` and `<end_of_image>` that frame the picture are *not* part of `S`. For a prompt
`<bos> <start_of_turn> user ⏎ <start_of_image> [p0 p1 p2] <end_of_image> Q …` (three picture rows instead of 256) the grid
of "who sees whom" (1 = may see, `.` = may not; rows are queries, columns are keys) is:

```
             keys →   bos sot usr ⏎  soi | p0  p1  p2 | eoi  Q
query ↓
bos                    1   .   .   .   .  |  .   .   .  |  .   .
start_of_turn          1   1   .   .   .  |  .   .   .  |  .   .
user                   1   1   1   .   .  |  .   .   .  |  .   .
newline                1   1   1   1   .  |  .   .   .  |  .   .
start_of_image         1   1   1   1   1  |  .   .   .  |  .   .
                      ----------------------------------------------
p0                     1   1   1   1   1  |  1   1   1  |  .   .      <- causal would be:  1 . .
p1                     1   1   1   1   1  |  1   1   1  |  .   .      <-                   1 1 .
p2                     1   1   1   1   1  |  1   1   1  |  .   .      <-                   1 1 1
                      ----------------------------------------------
end_of_image           1   1   1   1   1  |  1   1   1  |  1   .
Q (text after)         1   1   1   1   1  |  1   1   1  |  1   1
```

Everything outside the two middle rows' picture block is the ordinary lower triangle; the picture block is a full square.

**A worked example with the real sizes.** `shapes.png` with the question "What shapes and colors do you see in this
picture?" is 278 tokens: 5 before the picture (`<bos> <start_of_turn> user ⏎ <start_of_image>`), the picture's 256, and 17 after.
Picture row `p0` sees `5 + 1 = 6` keys with causal attention and `5 + 256 = 261` with Gemma 3's; the last picture row sees 261 either
way. On average a picture row sees 133 keys causally and 261 both ways, so the picture's attention costs about twice as
much (about 256 × 261 score products per head per layer instead of 256 × 133).

**Why the picture goes through in one pass.** Row `p0` at layer `l` needs the key and value of row `p255` at layer `l`,
and those are computed from `p255`'s hidden state *after layer `l − 1`*, which itself looked at all the other rows.
So the layers must be run over all 256 rows together: the projections, QK-norm and RoPE of every row first, the
keys and values of all rows into the cache, and only then the attention of every row. The engine therefore feeds a
`nonCausal` picture as a single pass (`n = 256` rows), after the text before it and before the text after it.

**How it is done without a new kernel.** After the rows' keys and values are stored, row `t` of the pass (position
`pos + t`) attends *as if it were the last row of the pass*: the keys `[0, pos + n)`, which is exactly the causal rule for row
`pos + n − 1`.

* CPU (`CpuBackend.GqaStep.run`): one `kv.attention(pool, layer, 1, pos + n − 1, qRow, aRow, …)` call per row, with the row's
  query copied out and its output copied back; the KV cache classes (f32, f16, q8) are used as they are.
* Metal (`MetalBackend.GqaStep.run`): one single-row `attention_t` dispatch per row through `MetalKernels.attentionRow`
  (the kernel and the dispatch of the batched decode step), with the scores scratch sliced so that the rows of a layer run
  concurrently. The flash prefill kernels have the causal triangle built in, so they are bypassed for such a pass.

A Gemma layer with a sliding window (the "local" layers, window 1 024 keys) counts the window back from the last row of the
picture for every row. For a row it only differs from counting back from the row itself if more than ~770 tokens of text
come before the picture.

**Where the rest of Gemma 3's picture handling is.**

```
<bos> <start_of_turn> user ⏎ <start_of_image> [256 × placeholder] <end_of_image> Question <end_of_turn> ⏎ <start_of_turn> model ⏎
 ids:  2      105      2364 107     255999          rows from the projector     256000     3689 …
```

The placeholder is an id the model never looks up (the rows come from the projector, scaled by nothing: Gemma scales only
the embeddings of *tokens* by `sqrt(dim)`, not the picture rows, as llama.cpp's graph). The text-only vocabulary of a
Gemma 3 GGUF ends before `<image_soft_token>` (id 262 144), so `VisionContext` uses `<pad>` as the placeholder there.
The tokens' positions are the plain `0, 1, 2, …` (no M-RoPE); the global layers use RoPE with linear scaling ×8, the local
ones base 10 000 (chapter 14).

### Why this design

* **No new kernel, one more method.** A masked variant of the prefill attention kernels (`attention_t`, `attention_fa`,
  `attention_fb`, and the q4 forms) would be faster for 256 rows, but each carries its own tiling of the causal triangle;
  reusing the single-row decode kernels gives the exact kernels that decoding is checked with (chapter 8) at the price of 256 small
  dispatches per layer, a one-time cost per picture (prefill of the 278-token prompt took 1.4 s, of which the attention
  dispatches are a part). If pictures become the bottleneck, a masked `attention_t` is the next step.
* **One pass, not a layer-major loop.** Running the layers on chunks of the picture, layer by layer, would let `backend.batch`
  stay at 64 but needs the hidden state of all rows to be kept between the chunks and the loop order turned inside out
  in every backend. Raising the batch to 256 costs only larger activation buffers (256 × 10 240 floats = 10 MB per buffer
  for Gemma 3 4B).
* **Rows "as the last row" instead of a mask argument on the cache classes.** The three KV cache types and the q4 simulation
  all have `attention(…, n, pos, …)` with the causal rule inside; adding a parameter would touch all of them.

### Settings

| Switch | Default | Meaning |
|---|---|---|
| `backend.batch` | 64, **256 for a `gemma3` model while it is at its default** | tokens per pass. The picture's 256 tokens must fit in one pass; with a smaller explicit value the request is refused: "a picture of 256 tokens attends to itself both ways and must fit in one pass, but the batch is 64: set backend.batch to at least 256" |
| `-Djllm.vision.causal=true` (JVM flag) | off | the picture attends causally like text. Wrong for real Gemma 3 answers (see the table), kept as the comparison and as the old path |
| `vision.*` | see 20.9 | the projector, the encoder's backend, the token budget (fixed at 256 tokens for Gemma 3), the size limits |

The Metal q4 KV cache (`backend.kv-dtype=q4`) has its own attention path and does not do this; a request with a picture on
that combination is refused. `scripts/mac-validate.sh --vision mmproj.gguf picture.png --vision-model gemma.gguf` has the
on/off row (default against `-Djllm.vision.causal=true`).

### Measured

Machine: Apple M1 Pro, JDK 26, the project's heavy lock held, 2026-10-22. Model: `ggml-org/gemma-3-4b-it-GGUF`
(`gemma-3-4b-it-Q4_K_M.gguf`, `mmproj-model-f16.gguf`), pictures `shapes.png` (640 × 480) and `scene.png` (800 × 600), 32 new tokens,
temperature 0, thinking not applicable. Command lines:

```
JAVA_OPTS=-Djllm.vision.debug=true ./jllm run --model gemma-3-4b-it-Q4_K_M.gguf --mmproj mmproj-model-f16.gguf \
    --image shapes.png --prompt "What shapes and colors do you see in this picture?" --temp 0 --max-tokens 32 --backend metal --ctx 1024
GPU=1 vision-llamacpp-check gemma-3-4b-it-Q4_K_M.gguf mmproj-model-f16.gguf \
    "<start_of_turn>user\n<__media__>What shapes and colors do you see in this picture?<end_of_turn>\n<start_of_turn>model\n" shapes.png 32 99
```

(`scripts/vision-llamacpp-check.cpp` is the second program: libmtmd's tokenizer and `mtmd_helper_eval_chunks`, which sets
llama.cpp's non-causal attention for the picture, then the first position's top-5 and 32 greedy tokens; `--image -` of
`jllm run` and `-` for the picture are the text-only forms.)

*Prompt ids*: identical to llama.cpp's in all three cases (picture + `shapes.png` question: `2 105 2364 107 255999 [256] 256000 3689 15965
532 7913 776 611 1460 528 672 6083 236881 106 107 105 4368 107`; the `scene.png` prompt; a text-only prompt).
*Greedy tokens (32)*: identical to llama.cpp's in all three cases ("Here's a breakdown of the shapes and colors I see in the
image: * **Shapes:** * A red circle *"; "The image depicts a simple, stylized landscape featuring a green field, a gray pyramid-shaped
mountain, a red house, and a yellow sun in a blue sky"; "The sky appears blue due to a phenomenon called Rayleigh scattering, …").

*Log-probabilities at the first position* (natural log; llama.cpp against jllm):

| `shapes.png` | llama.cpp | jllm, both ways (default) | jllm with llama.cpp's own embeddings | jllm, CPU encoder | jllm, f16 KV | **jllm, causal picture** |
|---|---|---|---|---|---|---|
| `Here` | −0.32422 | −0.26553 | −0.26481 | −0.26364 | −0.26558 | −0.05192 |
| `Okay` | −1.28473 | −1.45667 | −1.45907 | −1.46294 | −1.45653 | −2.98662 |
| `Let` | −8.89864 | −8.86567 | −8.86310 | −8.86596 | −8.86628 | −9.34682 |
| `Certainly` | −10.24162 | −10.14024 | −10.14554 | −10.14461 | −10.13941 | −10.04553 |
| `Alright` | −11.47596 | −11.44506 | −11.44685 | −11.45199 | −11.44446 | −12.40469 |

| `scene.png` | llama.cpp | jllm, both ways | jllm, f16 KV | **jllm, causal picture** |
|---|---|---|---|---|
| first | `The` −0.42684 | `The` −0.46115 | `The` −0.45988 | **`This`** −0.16587 |
| second | `This` −1.07410 | `This` −1.00826 | `This` −1.01052 | `The` −2.27898 |
| third | `Here` −5.14519 | `Here` −5.38495 | `Here` −5.37954 | `Here` −2.98666 |

What the table says: with attention that runs both ways the log-probabilities are within 0.06 (0.24 for a −5 entry) of
llama.cpp's, with a causal picture they are 0.27 off for the best token of `shapes.png` and the order of the two best tokens of
`scene.png` is reversed. The remaining difference is not the picture's embeddings (llama.cpp's own embeddings give the same
numbers), not our encoder (the CPU encoder gives the same), not the KV precision (f16 gives the same); it is the size of the
differences between the two *text* paths on this Q4_K_M file. To see that, the same comparison without a picture
(`--image -`, `-` for llama.cpp):

| text only | llama.cpp | jllm |
|---|---|---|
| "What shapes and colors do you see in this picture?" | `Okay` −0.32389, `Please` −1.28789, `Unfortunately` −7.20351 | `Okay` −0.29323, `Please` −1.37449, `Unfortunately` −6.82108 |
| "Tell me about the shapes and colors of a rainbow." | `Okay` −0.00029, `Let` −8.27948 | `Okay` −0.00019, `Let` −8.68965 |

differences of the same size (0.03 … 0.4) with no picture at all, from the two engines' different Q4_K arithmetic (their
integer-dot-product kernels against ours, activations quantized differently), accumulated over 34 layers.

Time (cold process, so it includes building the Metal pipelines): `shapes.png`: encoding 2 970 ms, prefill of the 278 tokens
1 373 ms, 32 tokens in 1 263 ms (25.3 tok/s), 5 629 ms in all; llama.cpp's `mtmd_helper_eval_chunks` (encoding and
prefill of the same prompt) took 4 122 ms. Text only: 20 tokens prefilled in 173 ms, 26.1 tok/s.

### Checks and limits

* `VisionTest.bidirectional`, a *set oracle* on a tiny Qwen2 model that needs no reference implementation: give the n rows
  of a picture the same rotary position. Attention that runs both ways is then a weighted sum over a *set* of keys, so permuting the
  picture's rows must not change anything the text after it sees (max logit difference below 10⁻⁴ on the CPU and the GPU),
  while with causal attention it does (above 10⁻³). The CPU and Metal agree with each other (below 5·10⁻³) by hand and through the
  engine, with the picture's own positions and without; a batch smaller than the picture is refused; the cache of rows computed
  causally is never taken for rows computed both ways (a fresh engine is used; a model always runs in one mode).
* Against llama.cpp: the table above on a real Gemma 3 4B (ids and the 32 tokens identical, log-probabilities as discussed).
* A cancelled prefill that stopped inside a picture is restarted at the picture's first token.
* Not done or not exact: the sliding-window rule above (differs from llama.cpp only with > ~770 tokens before the picture); the Metal q4 KV cache;
  one picture is limited to what fits in `backend.batch` (256 tokens: Gemma 3's fixed size, so always); the cost of the 256
  single-row dispatches (above).

## 20.14 Limits, and what differs from Hugging Face

* **Gemma 3** works end to end (20.13) on CPU and Metal: the picture side is the 896 × 896 fixed size with 256 tokens, the
  text side the `gemma3` GGUF architecture; the Gemma 3 1B text-only model has no projector.
* **Qwen3.5 / "Qwen3.8"** works (20.12) on CPU and Metal; only the pictures' quality depends on the Q4_0 file.
* Pictures only: no video (a second frame), no audio input.
* One picture is encoded in one command buffer; a huge picture (`vision.image-max-tokens` large) needs
  `n × ffn × 4` bytes of activations (4 096 patches × 3 420 × 4 = 56 MB for Qwen2.5-VL), so the budget doubles as a memory limit.
* Hugging Face's `Qwen2VLImageProcessor` *stretches* the picture to the next multiple of 28, llama.cpp pads it; the
  engine follows llama.cpp (above). SmolVLM's template is llama.cpp's (a newline layout the Jinja template of the model
  does not have), which is what the projector's GGUF was prepared with.
