# 22. Image generation (diffusion)

jllm-serve can also draw. Given a sentence it paints a picture; given a picture and a sentence it repaints it; given a picture,
a mask and a sentence it fills in just the masked part. The models are Stable Diffusion 1.x / 2.x (including SD-Turbo) and SDXL
(including SDXL-Turbo), read from the files Hugging Face stores them in (the *diffusers* layout), and everything is written
from scratch like the rest of the project: the tokenizer, the three neural networks, the samplers, the Metal kernels, the HTTP
API and the browser editor. Nothing is borrowed from PyTorch or diffusers except, in the tests, their *answers* to compare with.

| | what | where |
|---|---|---|
| [22.1](#221-what-denoising-means) | the idea: remove noise a little at a time, guided by a sentence | `diffusion/Scheduler` |
| [22.2](#222-the-parts-of-a-pipeline) | text encoder, UNet, VAE, sampler; sizes of the three families | `diffusion/ImagePipeline` |
| [22.3](#223-the-latent-space-and-the-vae) | why the work happens on a 64 × 64 × 4 picture, not on 512 × 512 × 3 | `diffusion/Vae` |
| [22.4](#224-the-text-encoder) | a sentence becomes 77 vectors | `diffusion/ClipTokenizer`, `ClipText` |
| [22.5](#225-the-unet) | the network that predicts the noise: ResNet blocks, attention, cross-attention | `diffusion/UNet` |
| [22.6](#226-classifier-free-guidance) | pushing the picture towards the sentence | `ImagePipeline.one` |
| [22.7](#227-the-samplers) | Euler, Euler ancestral, DPM++ 2M, Karras sigmas | `diffusion/Scheduler` |
| [22.8](#228-turbo-models-one-to-four-steps) | SD-Turbo / SDXL-Turbo: distillation | |
| [22.9](#229-image-to-image-inpainting-outpainting-upscaling) | strength, masks, the nine-channel UNet, outpainting, upscaling | `ImagePipeline` |
| [22.10](#2210-random-numbers-and-reproducibility) | the seed, and exactly which numbers it makes | `diffusion/Rng` |
| [22.11](#2211-how-jllm-computes-it) | one set of operations on two devices; convolution as a matrix product; the GPU kernels | `Ops`, `CpuOps`, `MetalOps`, `diffusion.metal` |
| [22.12](#2212-api-command-line-console-and-chat-tools) | `/v1/images/*`, `jllm image`, the Images tab, `generate_image` / `edit_image` | `ImageApi`, `ImageHttp`, `web/images` |
| [22.13](#2213-verified-against-diffusers) | stage-by-stage comparison with Hugging Face diffusers | `DiffusionCompare`, `DiffusionTest` |
| [22.14](#2214-measured-speed-and-memory) | seconds per image on this Mac, against diffusers on the same GPU | `DiffusionBench` |
| [22.15](#2215-what-video-will-reuse-and-what-it-will-need) | the parts a video model can reuse, and what is missing | |

---

Each part of the pipeline is explained in the same order: 🧸 the idea in plain words, 📐 how it works (formulas with every symbol defined, a worked example with real numbers, the shapes of the tensors), why it was built this way, the
switches, the measurements and what they mean, and what is checked and what is not. The "why", the settings and the limits of the feature as a whole are gathered in
[22.2.1](#2221-why-this-design), [22.12.1](#22121-settings), [22.13.1](#22131-checks-and-limits) and [22.14.1](#22141-what-the-numbers-mean).

**The whole pipeline with its shapes** (SD 1.5 at 512 × 512, 20 steps, guidance 7.5; channels last, `[batch][height][width][channels]`):

```
 "a red fox in a snowy forest"
   │ ClipTokenizer: NFC, lower-case, regex pieces, byte-pair merges, pad to 77                       ids          [77]
   ▼ ClipText: 12 causal layers, width 768  (twice: [negative prompt, prompt] because of guidance)    text         [2][77][768]
   │
   │ seed ─► Rng (SplitMix64 + Box-Muller) ─► ε ~ N(0, 1) · σ₀                                         z            [1][64][64][4]  = 16 384 numbers
   │ for each step i = 0 … n−1:
   │     x  = z / √(σᵢ² + 1)                (twice, for the two guidance halves)                       x            [2][64][64][4]
   │     ε̂  = UNet(x, tᵢ, text)             860 M parameters, 0.80 TFLOP per call                      ε̂            [2][64][64][4]
   │     ε  = ε̂_neg + w · (ε̂_prompt − ε̂_neg)                                                          ε            [1][64][64][4]
   │     z  ← sampler(z, ε, σᵢ, σᵢ₊₁)       Euler, Euler-a or DPM++ 2M                                  z            [1][64][64][4]
   ▼
   z / 0.18215 ─► VAE decoder (84 M parameters, 2.5 TFLOP, once)                                        pixels       [1][512][512][3]  = 786 432 numbers ─► PNG
```

---

## 22.1 What denoising means

🧸 Take a photo and sprinkle it with static, like an old television with no signal. Sprinkle a little and you still see the
photo; sprinkle a lot and you see only snow. Now imagine a very patient helper who has looked at millions of photos, each
with a caption. Show the helper a snowy screen and tell it "a cat on a sofa": it cannot see the cat yet, but it can guess
*which snow flakes do not belong* and wipe away a few of them. Do that again and again, a little at a time, and a cat on a
sofa slowly appears out of pure snow. That is all an image generator does. "Drawing" is "removing noise, guided by words".
The first guesses are blurry; the last ones are sharp. A "Turbo" model is a helper who was taught to do it in 1 to 4 big
wipes instead of 20 small ones.

📐 **Adding noise (the forward process).** Let `x₀` be an image (in practice its latent, 22.3). Choose a noise schedule
`β_1..β_T` (T = 1000); for Stable Diffusion `β_t = (√β_start + (√β_end − √β_start)·(t−1)/(T−1))²` with `β_start = 0.00085` and
`β_end = 0.012` ("scaled linear"). With `α_t = 1 − β_t` and `ᾱ_t = ∏_{s≤t} α_s`, a noisy version of `x₀` is

```
x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε          ε ~ N(0, I)
```

so `ᾱ_t` is the fraction of the signal's *variance* that is left: `ᾱ_0 = 0.99915` (a clean picture), `ᾱ_249 = 0.675`,
`ᾱ_499 = 0.278`, `ᾱ_749 = 0.0566`, `ᾱ_999 = 0.00466` (almost pure noise: the signal's amplitude `√ᾱ_999 = 0.068`).

**Training.** A network `ε_θ(x_t, t, c)` is shown `x_t`, the timestep `t` and a caption `c` and must say which noise `ε` was
added: `L = E ‖ε − ε_θ(x_t, t, c)‖²`. Nothing else is learned; everything below is how to *use* such a network.

**Taking noise away (the reverse process).** From a prediction `ε̂ = ε_θ(x_t, t, c)` one can estimate the clean picture
(Tweedie's formula) `x̂₀ = (x_t − √(1 − ᾱ_t) ε̂) / √ᾱ_t`, and a *sampler* (22.7) uses `x̂₀` or `ε̂` to move from `x_t` to a
slightly less noisy `x_{t'}`, `t' < t`. Starting from pure noise `x_999 = ε` and repeating down to `t ≈ 0` gives a picture.

**The sigma view that the code uses.** Divide by `√ᾱ_t`: `x̃_t = x_t / √ᾱ_t = x₀ + σ_t ε`, with
`σ_t = √((1 − ᾱ_t)/ᾱ_t)`. The picture is the same, the noise is now simply *added* with a size `σ`:

| t | 0 | 249 | 499 | 749 | 999 |
|---|---|---|---|---|---|
| σ_t | 0.0292 | 0.693 | 1.613 | 4.082 | 14.615 |

Generation walks σ from `σ_max = 14.6146` down to 0. The network was trained on `x_t = x̃_t / √(σ²+1)` (unit-variance
inputs), so the sampler hands it `x̃/√(σ²+1)` and the timestep `t` (`Scheduler.scaleInput`).

*Worked example* (one latent number; the schedule above, computed with numpy). Let a clean latent value be `x₀ = 0.5` and the noise drawn for it `ε = 1.0`, at `t = 499` where `ᾱ = 0.27767`:

```
x_t   = √0.27767 · 0.5 + √(1 − 0.27767) · 1.0 = 0.5269 · 0.5 + 0.8499 = 1.1134          (what the network was trained to see)
σ_499 = √((1 − 0.27767) / 0.27767) = 1.6129
x̃     = x_t / √ᾱ = 1.1134 / 0.5269 = 2.1129  =  x₀ + σ ε = 0.5 + 1.6129 · 1.0            (the sigma view: the same picture, noise simply added)
what the sampler hands the network:  x̃ / √(σ² + 1) = 2.1129 / 1.8977 = 1.1134 = x_t       (so both views agree)
```

At `t = 999` the same recipe gives `σ = 14.6146`: the noise is 14.6 times the amplitude of the picture; the network's job at the first step is to find a picture inside that.

**Why this design (22.1).** Predicting the noise `ε` rather than the picture is what the Stable Diffusion checkpoints were trained to do, so the pipeline has to use it; the *sigma view* is used instead of diffusers' variance-preserving state because
Euler-type samplers are simplest there (a step is `x̃ += d · Δσ`) and because adding noise for image to image (22.9) is then one multiply-add. The cost is two conversions (`scaleInput` before the network, `predictOriginal` after it), both in `Scheduler`.
Working on latents rather than pixels (22.3) is a 48-fold saving in the UNet; its price is the VAE, which has to be run once at the end and is the slowest single stage for a Turbo model (22.14).

---

## 22.2 The parts of a pipeline

```
 "a cat on a sofa" ──▶ CLIP tokenizer ──▶ 77 token ids ──▶ text encoder ──▶ 77 × 768 (or 1024, 2048) numbers  ─┐
                                                                                                            │ cross-attention
 seed ──▶ noise  [4 × h/8 × w/8]  ──▶ ┌─────────────────────────────────────────────┐                         │
                                      │ for each step i:  ε̂ = UNet(x_i / √(σ_i²+1), t_i, text)  ◀───────────┘
                                      │                  x_{i+1} = sampler(x_i, ε̂, σ_i, σ_{i+1})
                                      └─────────────────────────────────────────────┘
                                                      │ x_final [4 × h/8 × w/8]
                                                      ▼
                                          VAE decoder ──▶ pixels [3 × h × w]  ──▶ PNG
```

| | SD 1.5 | SD 2.1 / SD-Turbo | SDXL / SDXL-Turbo |
|---|---|---|---|
| text encoder(s) | CLIP ViT-L/14 (12 layers, width 768) | OpenCLIP ViT-H/14 (23 layers, width 1024) | CLIP ViT-L/14 (the state before its last layer) + OpenCLIP ViT-bigG/14 (32 layers, width 1280, the state before its last layer, + a pooled 1280-vector); concatenated: 2048 |
| UNet parameters | 860 M | 865 M | 2 567 M |
| UNet channels per level | 320, 640, 1280, 1280 | 320, 640, 1280, 1280 | 320, 640, 1280 |
| attention heads per level | 8, 8, 8, 8 (head widths 40, 80, 160, 160) | 5, 10, 20, 20 (width 64) | 5, 10, 20 (width 64) |
| transformer blocks per attention layer | 1 | 1 | 1, 2, 10 |
| projections | 1 × 1 convolutions | linear | linear |
| extra conditioning | – | – | pooled text embedding + image size (6 numbers, 256-wide sinusoidal embeddings each) |
| VAE | 84 M, scaling 0.18215 | the same | the same architecture, scaling 0.13025, **float32 only** (its activations overflow in half precision) |
| native size | 512 | 512 (SD-Turbo) / 768 | 1024 (SDXL-Turbo: 512) |
| model files (fp16) | 2.1 GB | 2.6 GB | 6.9 GB |

jllm reads each component folder (`unet/`, `vae/`, `text_encoder/`, `tokenizer/`, `scheduler/`) and its `config.json`; the half-precision
variant (`*.fp16.safetensors`) is preferred when the repository has one (`DiffusersRepo`).

### 22.2.1 Why this design

* *Written from scratch against the diffusers file layout.* Everything but the answers used in tests is Java: the CLIP tokenizer, the three networks, the samplers, the random numbers, the Metal kernels, the PNG and JPEG handling. The reasons are the project's own (no
  Python or native library to ship, one engine, every kernel with a reference) and the price is speed: on the same GPU the code needs 1.9-2.1 times the time of PyTorch's MPS backend and 2.4-3.1 times PyTorch's CPU time (22.14).
  Reading the *diffusers* layout (`model_index.json`, `unet/`, `vae/`, `text_encoder/`, `tokenizer/`, `scheduler/`, safetensors files) means any fine-tune published in that layout loads; repositories that only have `.bin`, `.ckpt`, `.onnx` or `.msgpack`
  weights are skipped by the downloader (`DiffusersRepo`), and the half-precision files are taken when a repository has them.
* *One interface, two devices* (`Ops`, 22.11): the UNet, the VAE and the text encoders are each written once. The price is that a trick that belongs to one device only (a fused kernel, a different data layout) has to be an operation of the interface (`linear` with
  a fused bias, activation and residual; `conv2d` with a fused channel bias; `upsampleConv`).
* *Latent diffusion, classifier-free guidance, ancestral and multistep samplers, Turbo defaults* are what the published checkpoints need; things that need extra weights or code and are **not** implemented: LoRA, ControlNet, the SDXL refiner, textual inversion, schedulers other
  than the four in 22.7 (plus DDIM for video), `.ckpt` files, and images above 2048 × 2048 pixels.
* *float16 on the GPU by default*, float32 sums, norms and softmax, and a float32 VAE for SDXL-type models (22.3): half the memory and the speed of the half-precision matrix units, for a measured 50-55 dB image difference (22.13).

---

## 22.3 The latent space and the VAE

🧸 A 512 × 512 photo is a lot of tiny dots (786 432 numbers). Painting each of them step by step would be slow, and most of the
detail is redundant (a blue sky is a million nearly identical dots). So the model works on a small *sketch* of the picture:
64 × 64 squares, each described by 4 numbers (16 384 numbers, 48 times fewer). A separate pair of helpers, the **VAE**, turns a
picture into such a sketch (the *encoder*) and a sketch back into a picture (the *decoder*), trained so that the round trip loses
almost nothing a person would notice.

📐 **Shape.** The encoder has four stages of ResNet blocks with three strided 3 × 3 convolutions between them (channels
128 → 256 → 512 → 512), so an `H × W × 3` image becomes `H/8 × W/8 × 4`. It produces a mean and a log-variance per latent
number (a *posterior* `N(μ, σ²)`); during generation of *new* pictures only the decoder runs. Latents are multiplied by
`scaling_factor` (0.18215; 0.13025 for SDXL) so that they have about unit variance like the noise they are mixed with.

```
encode:  z = scaling · (μ + exp(½ logvar) · ε)               decode:  x ≈ D(z / scaling)
D:  post_quant_conv → conv_in (4 → 512) → ResNet · attention · ResNet → 4 up stages [3 ResNets, 2× upsample + 3×3 conv] → GroupNorm, SiLU, conv_out (→ 3)
```

The attention in the middle block has one head over all `h/8 · w/8` positions: 4 096 tokens for a 512 picture, 16 384 for 1024 (a
scores matrix of 1 GB in float32). Larger images are therefore decoded **in tiles** (`image.vae-tile`, default 128 latent pixels = 1024
image pixels): overlapping tiles are decoded separately and cross-faded with a linear ramp over the overlap
(`Vae.decodeTiled`; a unit test shows that the plan and the blend reproduce a per-pixel function exactly). A tiled decode differs
slightly from the whole one, because each tile's group norms see only the tile: measured on a 1024 × 1024 SD-Turbo picture with
tiles of 64 latent pixels, 29.7 dB PSNR against the whole decode, about 3 % brightness difference between tiles and no visible seam.

**Why the VAE of SDXL is float32.** Its decoder's activations exceed the float16 range on some images and become infinities
(black images); diffusers upcasts it, so does jllm (`Devices.open`: the GPU holds a float32 VAE next to the float16 UNet).

*Worked example, the tile plan of `Vae.decodeTiled`* (`image.vae-tile = 128`, `image.vae-tile-overlap = 8`, scale 8): a 1536 × 1536 picture has a 192 × 192 latent. The step between tiles is `128 − 8 = 120`, so the tile origins on each axis are 0 and 120
(the first tile covers latent pixels 0-127, the second 120-191, only 72 wide because the picture ends): 2 × 2 = 4 tiles. Neighbouring tiles share 8 latent pixels = 64 image pixels; there the weights ramp linearly,
`w = (k + ½)/64` for tile 2 and `(63 − k + ½)/64` for tile 1 at the shared pixel `k = 0 … 63`, which add up to exactly 1; the sum of the weights is kept per pixel and divides the result, so even corners (four tiles) are exact weighted means.
A picture up to 1024 × 1024 (a 128 × 128 latent) is a single tile and is decoded as a whole, which is the reason for the default of 128.

*Worked example, group normalisation* (the reason a tile differs a little from the whole). Four channels in two groups, two pixels; channel values `[1, 3, 10, 14]` at pixel 0 and `[3, 5, 12, 18]` at pixel 1. Group 0 holds channels 0-1 at both pixels: values 1, 3, 3, 5, mean 3, variance 2, normalised
`(x − 3)/√(2 + 10⁻⁵) = −1.414, 0, 0, +1.414`; group 1: values 10, 14, 12, 18, mean 13.5, variance 8.75, normalised `−1.183, +0.169, −0.507, +1.521`. The mean and the variance come from *every pixel the group sees*, so a tile that contains
only part of the picture has a different mean (about 3 % in brightness on a 1024 × 1024 SD-Turbo picture in 64-pixel tiles, 22.13).

**Why this design (22.3).** The decoder, not the UNet, is the memory problem at large sizes: its middle attention is one head over `h/8 · w/8` positions, a 1 GB scores matrix in float32 at 1024². The alternatives are a flash-attention kernel for the VAE's head width of 512 (not
written: widths other than 8-160 use `df_gemm`, `df_softmax`, `df_gemm` a few heads at a time) or tiling. Tiling is simple and bounds memory by the tile; its cost is the small brightness difference above, so the default tile was raised until everything up to 1024 × 1024 is one piece.
The encoder (used by image to image and inpainting) is tiled the same way (inside `Vae.encode`).

---

## 22.4 The text encoder

🧸 The computer cannot read. It gets the sentence cut into pieces of words ("a", "cat", "on", "a", "sofa") written as numbers,
and a small reader turns every piece into a list of numbers that says what it means *in this sentence*. The painter later looks
at those lists whenever it has to decide what a patch of the picture should show.

📐 **Tokenizer** (`ClipTokenizer`). Text is Unicode-normalised (NFC), white space collapsed, lower-cased and cut by the pattern
`<|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+`. Each piece's UTF-8 bytes are mapped to
printable characters (the GPT-2 byte table), the last one gets the end-of-word marker `</w>`, and byte-pair merges are applied by
rank (49 408 tokens). The sequence is `⟨bos⟩ w₁ … wₙ ⟨eos⟩` cut or padded to exactly **77** tokens: the padding is the end-of-text
token for CLIP ViT-L, `!` for OpenCLIP. OpenCLIP's tokenizer additionally lists `!` as a special token that matches in the raw
text before anything else, so a `!` in a prompt is token 0 there (`tokenizer_config.json`'s `added_tokens_decoder`; `make-tiny.py`
reproduces it so the tests catch it).

**Transformer.** Token embedding plus a learned position embedding, then `L` pre-norm layers with **causal** self-attention (a
token sees only the tokens before it):

```
h ← h + OutProj(Attention(LN₁(h)))          h ← h + FC₂(act(FC₁(LN₂(h))))
act = x·σ(1.702 x)  ("quick GELU", CLIP ViT-L)     or the exact GELU  x·Φ(x)  (OpenCLIP)
```

SD 1.x / 2.x condition the UNet on the final-norm output of the last layer, **all 77 positions** (the padding positions carry
information too). SDXL takes the *input of the last layer* of both encoders (`hidden_states[-2]`, no final norm), concatenates
them to 2048 channels, and feeds the second encoder's end-of-text vector through its text projection as the *pooled* embedding.
`clip_skip` n leaves out n further layers. The encoders run on the same device as the UNet (a few ms on the GPU).

*Worked example, "quick GELU" against the exact GELU* (both are `x` times a smooth step; CLIP ViT-L was trained with `x · σ(1.702 x)`, OpenCLIP with `x · Φ(x)`; `hidden_act` in the encoder's `config.json` says which; the two differ by less than 0.02 over this range):

```
x          −2       −1      0.5      1        2
x·σ(1.702x) −0.0643  −0.1542  0.3504   0.8458   1.9357
x·Φ(x)      −0.0455  −0.1587  0.3457   0.8413   1.9545
```

*Shapes and the causal mask.* The 77 token ids become `[77][768]` (token embedding + learned position embedding). Attention is causal: the row of token `i` only attends to tokens `0 … i`, so the row of `⟨eos⟩` has seen the whole sentence and padding rows (which repeat
the end-of-text token, or `!` for OpenCLIP) have seen it too. SD 1.x / 2.x hand all 77 rows to the UNet's cross-attention (`[2][77][768]` with guidance); SDXL concatenates two encoders' rows into `[2][77][2048]` and keeps the pooled vector (the projected `⟨eos⟩` row of the second encoder, 1280 numbers)
for the time-conditioning MLP. The SD 1.5 encoder has 123 M parameters (12 layers of 7.1 M and a 37.9 M token table, computed from its config) and runs in a few milliseconds on the GPU, 0.05 s in the measured image runs (22.14): it is never the bottleneck.

**Why this design (22.4).** The tokenizer is implemented rather than loaded from a library because byte-level BPE with the CLIP pattern is ~200 lines and its edge cases (Unicode normalisation, `</w>` word endings, OpenCLIP's special `!`) are exactly what the tests catch by comparison with `transformers`' tokenizer.
Text is limited to 77 tokens by the checkpoints (longer prompts are cut; there is no prompt-weighting syntax such as `(word:1.3)` and no chunking of long prompts).

---

## 22.5 The UNet

🧸 The painter has two skills. One looks at the whole picture at once at several zoom levels ("this is a sofa-shaped blob in
the lower middle"), the other looks closely at a small patch. The zoom levels are connected by shortcuts so fine detail is not
lost on the way down and back up. Whenever the painter is unsure what a patch should be, it looks at the sentence (that is
**cross-attention**: "which words matter for this patch?") and at how noisy the picture still is (that is the **timestep**).

📐 **Layout** (`UNet`). With `B` the batch (2 with guidance), the latent is `[B, h, w, 4]` (channels last):

```
emb  = MLP(sinusoid(t))   [+ MLP([pooled text, sinusoid(time ids)]) for SDXL]          (a vector per batch element)
x    = conv3×3(sample)                                              skips = [x]
down block i (i = 0..3):   2 × [ ResNet(x, emb) → (Transformer(x, text) if the block has attention) → push x ]
                           then a strided 3×3 convolution (push)                      (not on the last block)
mid:                       ResNet → Transformer → ResNet
up block i:                3 × [ pop skip, concat(x, skip) → ResNet → Transformer ]   then 2× upsample + 3×3 conv (not on the last)
out:                       conv3×3( SiLU( GroupNorm(x) ) )     → predicted noise [B, h, w, 4]
```

*Sinusoidal timestep embedding:* `[cos(t·f_j) | sin(t·f_j)]`, `f_j = 10000^(−j/(d/2))`, `j < d/2` (d = 320), then
`Linear → SiLU → Linear` to 1280 channels.

*ResNet block* (`resnet`): `h = conv(SiLU(GN(x)))`, add `Linear(SiLU(emb))` to every pixel of `h`, `h = conv(SiLU(GN(h)))`, output
`h + shortcut(x)` (a 1 × 1 convolution when the channel count changes). GN is group normalisation with 32 groups (ε = 1e-5):
statistics per group of `C/32` channels over all pixels,

```
GN(x)_c = γ_c · (x_c − μ_g) / √(σ_g² + ε) + β_c           μ_g, σ_g² over the group's channels and all h·w pixels
```

*Spatial transformer* (`transformer`): GN, project to tokens (a 1 × 1 convolution or a linear layer: on channels-last data they
are the same matrix product), then per block

```
x ← x + SelfAttn(LN(x))                           q, k, v from the pixels   (4096 tokens at the first level of a 512 image)
x ← x + CrossAttn(LN(x), text)                    q from the pixels, k, v from the 77 text vectors
x ← x + FF(LN(x)),   FF(y) = W₂ · ( (W₁ᵃ y) · GELU(W₁ᵇ y) )          "GEGLU": the projection is split in two halves, one gates the other
```

and a projection back plus the residual of the whole layer. Attention is `softmax(q kᵀ/√d_head) v` per head.
**Quirk of diffusers' configs:** the field `attention_head_dim` holds the *number of heads* of each level (SD 1.5: `8` →
8 heads of width `C/8` = 40, 80, 160; SD 2.1: `[5, 10, 20, 20]` → width 64); `UNet` follows that reading.

*Cost* (counted by PyTorch's FLOP counter on the model definitions, 2 FLOPs per multiply-add): one UNet evaluation at 512 × 512
is 0.80 TFLOP for SD 1.5 and SD 2.1 / SD-Turbo and 1.59 TFLOP for SDXL (6.76 at 1024 × 1024); the VAE decoder needs 2.51 TFLOP
at 512 × 512 (10.5 at 1024), three times a UNet evaluation, because it works at full pixel resolution. A 20-step image with
guidance is therefore 2 · 20 · 0.8 = 32 TFLOP in the UNet and 2.5 in the VAE; a 1-step Turbo image 0.8 + 2.5.

*Inpainting UNet.* A UNet with 9 input channels (`in_channels = 9`) is given, in addition to the 4 latent channels, the mask
(1 channel) and the latents of the masked picture (4 channels): 22.9.

*The shapes of one UNet call, SD 1.5 at 512 × 512 with guidance* (batch 2, channels last; 12 skip tensors are pushed on the way down and popped on the way up):

```
 conv_in                                          [2][64][64][4]    →  [2][64][64][320]                                       push 1
 down 0   2 × (ResNet, Transformer, 8 heads)         64 × 64 × 320   self-attention over 4 096 tokens          strided 3×3 conv → 32 × 32       push 2, 3, 4
 down 1   2 × (ResNet, Transformer)                  32 × 32 × 640   1 024 tokens                              strided conv → 16 × 16          push 5, 6, 7
 down 2   2 × (ResNet, Transformer)                  16 × 16 × 1280  256 tokens                                strided conv → 8 × 8            push 8, 9, 10
 down 3   2 × ResNet (no attention, no downsampling)  8 × 8 × 1280                                                                              push 11, 12
 mid      ResNet, Transformer (64 tokens), ResNet     8 × 8 × 1280
 up 0     3 × ResNet   (first input: 1280 + skip 12 = 2560 channels)   8 × 8 → upsample 2× + conv → 16 × 16
 up 1     3 × (ResNet, Transformer)                  16 × 16 × 1280                                            → 32 × 32
 up 2     3 × (ResNet, Transformer)                  32 × 32 × 640                                             → 64 × 64
 up 3     3 × (ResNet, Transformer)                  64 × 64 × 320
 out      GroupNorm(32), SiLU, conv 3×3 (320 → 4)    [2][64][64][4]  =  the predicted noise ε̂
```

The self-attention of the first level compares 4 096 tokens with 4 096: `4 096² · 8 heads · 2 images = 268 M` scores, 1.07 GB in float32; the flash kernel (22.11) never stores them. The time embedding is added to every ResNet block
(`Linear(SiLU(emb))` to each pixel); the time itself is `[cos(t f_j) | sin(t f_j)]` with `f_j = 10000^(−j/160)`, `j = 0 … 159`: `f₀ = 1`, `f₁ = 0.944`, `f₂ = 0.891`, …, `f₁₅₉ = 1.06·10⁻⁴`, so at `t = 999` the first three cosines are 0.9996, 0.8027, −0.2781, followed by `Linear → SiLU → Linear` to 1280 channels. The group normalisation of the ResNet blocks is the one worked in 22.3.

**Why this design (22.5).** The network is the checkpoint's: the layout, the head counts and the naming quirk described above are dictated by the weights, so `UNet` reads `config.json` rather than hard-coding SD 1.5, which is also how SD 2.1 and SDXL (three levels, ten transformer blocks in the deepest) run unchanged. What was free to choose is the data layout and the kernels:
channels last, so a convolution's gather reads contiguous channels and a 1×1 projection is a plain matrix product; the group norm and layer norm kernels with float32 statistics even when the data is half precision (22.11); attention by flash kernels with head widths 8-160. The cost of keeping the skips is memory: up to 12 skip tensors, the largest
`[2][64][64][320]` (5.2 MB in float16, 10.5 MB in float32), are alive at the same time.

---

## 22.6 Classifier-free guidance

🧸 To make the picture follow the sentence more closely, ask the painter twice: "what would you do *with* the sentence?" and "what
would you do *without* it?" Then go further in the direction that the sentence made a difference, like turning up the volume of
the sentence. Turning it up too much gives loud, over-saturated pictures; 5 to 8 is usual.

📐 The model is trained with the caption dropped at random, so it can predict noise both with and without it. With `w` the
guidance scale (`guidance`), at every step

```
ε̂ = ε_u + w · (ε_c − ε_u)          ε_c = UNet(x, t, text(prompt)),  ε_u = UNet(x, t, text(negative prompt or ""))
```

`ε_c − ε_u` is, up to a factor, the gradient of the log-probability of the caption given the noisy picture (an *implicit
classifier*), so `w > 1` pushes towards what the caption says. The two predictions are one UNet call with batch 2 (`[negative,
prompt]`); a **negative prompt** just replaces the empty text of the unconditional half, so "blurry, low quality" is something to
move away from. Guidance applies only when `w > 1` (as in diffusers); at `w ≤ 1` the unconditional half is not computed, which
halves the cost and is what the Turbo models use. SDXL's *empty* negative prompt is all zeros rather than the embedding of the
empty string (`force_zeros_for_empty_prompt`).

*Worked example.* One latent number, `ε_u = 0.20` (what the network predicts for the empty prompt) and `ε_c = 0.50` (for "a red fox"), `w = 7.5`: `ε = 0.20 + 7.5 · (0.50 − 0.20) = 2.45`. The guided value is *beyond* the conditional prediction (it extrapolates): that is why guidance sharpens the picture towards the caption,
and why very large `w` over-saturates it. With `w = 1` the formula gives `ε_c` (no guidance, and the unconditional half is not computed), with `w = 0` it gives `ε_u`.

**Why this design (22.6).** The alternatives are a separately trained classifier (the original guided diffusion; no checkpoint here has one) or no guidance (Turbo models, which were distilled to need none). Putting both halves in one UNet batch means the weights are read once and the convolution kernels see twice the rows;
the cost is that guidance doubles the UNet's activations and its work (2 · 20 · 0.8 = 32 TFLOP for a 20-step 512² image, 22.5). A Turbo model skips it, and `guidance ≤ 1` skips it for any model (`ImagePipeline.generate` uses `cfg = guidance > 1`).

---

## 22.7 The samplers

A sampler picks the `n` noise levels `σ_0 > σ_1 > … > σ_{n−1} > σ_n = 0` and the rule to go from one to the next. All three
samplers below follow diffusers' implementations number for number (22.13).

**Which timesteps.** `t_i` for `n` steps over the 1000 training steps, by the model's `timestep_spacing`:
*linspace* `t_i = (n−1−i)·999/(n−1)` (rounded for DPM++); *leading* `t_i = (n−1−i)·⌊1000/n⌋ + offset`; *trailing*
`t_i = round(1000 − i·1000/n) − 1` (SD-Turbo / SDXL-Turbo: 999, 749, 499, 249 for 4 steps, i.e. σ = 14.61, 4.08, 1.61, 0.69).
`σ_i` is the schedule's σ at `t_i`, linearly interpolated between the integer steps, and `σ_n = 0`.

**Euler** (`euler`): the reverse process is an ordinary differential equation `dx̃/dσ = ε_θ`; one Euler step is

```
x̂₀ = x̃ − σ_i ε̂          d = (x̃ − x̂₀)/σ_i = ε̂          x̃ ← x̃ + (σ_{i+1} − σ_i) · d
```

With one step (`n = 1`, σ_1 = 0) this is exactly `x̃ ← x̃ − σ_0 ε̂ = x̂₀`: the clean-picture estimate itself, which is what a Turbo
model is trained to make good.

**Euler ancestral** (`euler_a`): step *down* to a lower noise level and add *fresh* noise, so errors are forgotten:

```
σ_up² = σ_{i+1}² (σ_i² − σ_{i+1}²) / σ_i²          σ_down² = σ_{i+1}² − σ_up²
x̃ ← x̃ + (σ_down − σ_i) · ε̂  +  σ_up · z          z ~ N(0, I) (a new draw per step)
```

(it never converges to one picture as steps increase: the result keeps changing with the step count).

**DPM-Solver++ 2M** (`dpmpp_2m`, `dpmpp_2m_karras`): a second-order multistep solver of the same ODE in the variable
`λ = ln(α/σ_t)` with `α = 1/√(σ²+1)`, `σ_t = σ α` (the variance-preserving form of `x̃`), using the previous step's estimate to
correct the current one. With `x̂₀` the data prediction, `h = λ_{i+1} − λ_i` and `r = (λ_i − λ_{i−1})/h`:

```
first order (first and last step):   x ← (σ_t,i+1/σ_t,i) x − α_{i+1} (e^{−h} − 1) x̂₀
second order:   D₀ = x̂₀(i),  D₁ = (x̂₀(i) − x̂₀(i−1)) / r
                x ← (σ_t,i+1/σ_t,i) x − α_{i+1} (e^{−h} − 1) D₀ − ½ α_{i+1} (e^{−h} − 1) D₁
```

Its work is done in a state that the network sees unscaled (the Euler samplers scale by `1/√(σ²+1)`). It converges in ~20 steps
where Euler needs ~30. **Karras sigmas** spread the steps differently, more of them where the picture is decided (small σ):
`σ_i = (σ_max^{1/ρ} + (i/(n−1))(σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ`, `ρ = 7`, and the timestep given to the network is the (rounded)
inverse of the schedule at that σ. For `n = 5` this gives σ = 14.61, 4.80, 1.27, 0.248, 0.029, then 0 (`σ_min` and `σ_max` are the two ends of the 1000-entry table, 0.0292 and 14.6146; the example below has the numbers for `n = 4` too).

Prediction types: `epsilon` (all models here but SD 2.1-768) and `v_prediction`
(`x̂₀ = −σ/√(σ²+1)·v + x̃/(σ²+1)`) are implemented; the scheduler config is read from `scheduler/scheduler_config.json`.

*Worked example: Euler and DPM++ 2M side by side* (computed with numpy from the formulas above and the real schedule). One latent number whose clean value is 0.5, with the noise 0.7 drawn for it, and SD-Turbo's four steps `t = 999, 749, 499, 249`, i.e. `σ = 14.6146, 4.0817, 1.6129, 0.6932` and then 0:
the starting state is `x̃₀ = 0.5 + 14.6146 · 0.7 = 10.7302`. To show the update rules without a network, let the network's estimate `x̂₀` be the invented values 0.80, 0.62, 0.53, 0.50 (an estimate that improves as the noise level falls).

```
Euler      d = (x̃ − x̂₀)/σ ;  x̃' = x̃ + d (σ_next − σ)
 step   σ        x̃        x̂₀     d        σ_next − σ    x̃'
  0    14.6146   10.7302   0.80   0.6795   −10.5329       3.5734
  1     4.0817    3.5734   0.62   0.7236    −2.4688       1.7870
  2     1.6129    1.7870   0.53   0.7794    −0.9197       1.0703
  3     0.6932    1.0703   0.50   0.8226    −0.6932       0.5000     (σ_next = 0: the step lands exactly on x̂₀)

DPM++ 2M   λ = −ln σ ;  h = λ_next − λ ;  first order = Euler in x̃ ;  second order adds the trend of x̂₀ (D₁ = (x̂₀ − previous x̂₀)/r, r = h_prev / h)
 step   λ        h      order   x̃'(unscaled)     Euler's x̃'
  0    −2.682   1.276   1st     3.5734           3.5734         (the same number: DPM++(1) is Euler in the sigma view)
  1    −1.407   0.928   2nd     1.7474           1.7870         (x̂₀ fell from 0.80 to 0.62, so the solver bends the step)
  2    −0.478   0.844   2nd     1.0299           1.0703
  3     0.366   ∞       1st     0.5000           0.5000         (the last step is first order, as in diffusers)
```

*Euler ancestral, one step* (σ from 14.6146 to 4.0817): `σ_up² = 4.0817² (14.6146² − 4.0817²)/14.6146² = 15.362` so `σ_up = 3.919`, and `σ_down² = 4.0817² − 15.362 = 1.299`, `σ_down = 1.140`. The step moves *down* to noise level 1.140 and then adds fresh noise of standard
deviation 3.919; the total level `√(1.140² + 3.919²) = 4.0817` is where the schedule wants it, but 92 % of its variance (`3.919²/4.0817²`) is new, which is why the result keeps changing with the step count and never converges to one picture.

*Karras sigmas for n = 4 and n = 5* (`σ_min = 0.0292`, `σ_max = 14.6146`, `ρ = 7`): `n = 4`: 14.6146, 3.1686, 0.4469, 0.0292; `n = 5`: 14.6146, 4.7965, 1.2741, 0.2479, 0.0292 (then 0). Compare the four "trailing" steps above (14.61, 4.08, 1.61, 0.69): the Karras spacing puts its last step at
0.03, where the fine detail is decided, instead of stopping at 0.69.

**Why this design (22.7).** The four samplers are the ones people use with these checkpoints, and each follows its diffusers implementation line by line so that a run can be compared number for number (22.13). *Euler* is the sampler of Turbo models (one step is `x̂₀`). *DPM++ 2M with Karras sigmas* is the default for every other model because it gets a usable picture in about 20-25 steps, at the
price of keeping the previous step's `x̂₀` (one extra latent) and of amplifying rounding differences (the weakest row of 22.13). *Euler ancestral* is there for its different look; it is not deterministic in the step count. DDIM (chapter 23) is the sampler AnimateDiff was trained with.
Not implemented: UniPC for images (it exists for Wan, chapter 23), DPM++ SDE, Heun, LMS, PNDM, and the Karras option for Euler.

---

## 22.8 Turbo models: one to four steps

🧸 The slow painter takes 20 small steps. A clever teacher took the slow painter's work and trained a *student* to do the whole
job in one or two big strokes, and a second student (a "critic") to tell the student when the result does not look like a real
picture. SD-Turbo and SDXL-Turbo are such students.

📐 *Adversarial Diffusion Distillation* (Sauer et al., 2023): the student UNet, initialised from SD 2.1 / SDXL, is trained on
two losses at timesteps drawn from a *few-step* schedule (such as 999, 749, 499, 249): an **adversarial** loss against a
discriminator working on DINOv2 features, and a **distillation** loss that matches the output of a frozen teacher run on the
student's own noised output. The result predicts a sharp `x̂₀` even from nearly pure noise. Consequences for the pipeline:

- `n = 1..4` steps, `trailing` spacing (the training schedule), the Euler sampler (22.7: the single step is `x̂₀`);
- **no guidance** (`guidance = 0`, the CFG half is skipped), so a negative prompt has no effect;
- at 512 × 512 (SD-Turbo's training size); larger sizes work but repeat subjects.

`ImagePipeline` recognises these models by name (a folder or repository containing `turbo`) and takes those defaults
(`ImagePipeline.Defaults`: 1 step, guidance 0, 512, `euler`; other models: 25 steps, guidance 7.5, `dpmpp_2m_karras`).

*Worked example, what a Turbo step does.* With the single step of `n = 1` (`σ₀ = 14.6146`, `σ₁ = 0`) the Euler rule is `x̃ ← x̃ − σ₀ ε̂`, i.e. `x̂₀`. Start from the pure-noise latent `x̃ = σ₀ ε`; the picture is `σ₀ (ε − ε̂)`: a difference of two numbers that are both of order 1, multiplied by 14.6. An error of 0.01 in the predicted noise therefore moves the latent
by 0.146 (latents have a standard deviation of about 1 after the scaling factor), which is why a model must be *trained* for this regime: an ordinary model is not accurate enough at `σ = 14.6` for one step to give a sharp picture. With four steps the network is asked for `ε̂` at the noise levels 14.6, 4.08, 1.61 and 0.69 (the trailing schedule), each with less leverage on the result.

**Why this design (22.8).** The alternative for a fast picture is to run an ordinary model for fewer steps, which degrades (the samplers of 22.7 are accurate at about 20 steps, not at 1-4). SD-Turbo and SDXL-Turbo cost no guidance (half the UNet work) and one to four steps: measured 1.68 s for a 512² picture in one step, 9.63 s for 20 steps of the same model (22.14).
The price is picture quality at larger sizes (repeated subjects, above) and the loss of the negative prompt. **Settings**: a model is treated as Turbo when its folder or repository name contains `turbo` (or contains `lcm` with a `trailing` scheduler config), which selects its defaults (1 step, guidance 0, 512, Euler, trailing spacing); a renamed folder gets the ordinary defaults, so pass `steps`, `guidance` and `scheduler`
explicitly then. The per-request fields override the defaults; 1-4 steps is the useful range.

---

## 22.9 Image to image, inpainting, outpainting, upscaling

**Image to image** (`strength` s in (0, 1]). Instead of starting from pure noise, start from the *encoded input picture with
noise added to level `σ_{t_start}`*, and run only the remaining steps:

```
n_run = min(⌊n·s⌋, n)        t_start = n − n_run        x̃ = z₀ + σ_{t_start} · ε           (z₀ = encoded picture, a sampled latent)
```

`s = 1` redraws everything (a fully noised start), `s = 0.3` keeps the layout and colours and changes details; `s·n < 1`
returns the encoded and decoded input. (For the VP samplers the noising is `√ᾱ z₀ + √(1−ᾱ) ε`.)

**Inpainting** (a mask `m`, 1 = repaint). Two methods, both in `ImagePipeline`:

1. *Blending* (any model, the default for SD-Turbo): after every step the part to keep is put back, noised to the level the
   latents are now at, so the network always sees a consistent picture: `x ← (1 − m)·(z₀ + σ_{i+1} ε) + m·x`. The mask is
   area-averaged to latent resolution, so feathered masks blend softly. After decoding, pixels where `m = 0` are copied from
   the original, so the unmasked part is *exactly* the input (the VAE round trip would otherwise change it slightly).
2. *The inpainting UNet* (`in_channels = 9`, e.g. `stable-diffusion-v1-5/stable-diffusion-inpainting`): its input is
   `[x, m_latent, z(masked picture)]` where the masked picture is the input with the masked pixels set to 0; the network was
   trained to fill the hole, so no blending is needed (`strength` 1 starts from pure noise).

**Outpainting** (the browser editor): extend the canvas, fill the new border with the nearest edge pixels (blurred), mask the
border plus a soft overlap, and inpaint. **Upscaling** (`"upscale": 2` in the API): enlarge the picture by bicubic resampling and
redraw it at strength ~0.35, which adds detail without changing the composition (large pictures go through the tiled VAE).

*Worked example, strength.* With `steps = 20`: `strength 0.3` runs `⌊20 · 0.3⌋ = 6` steps and starts at step 14 (a start noise level close to the end of the schedule: the layout and colours of the input survive); `strength 0.6` runs 12 and starts at step 8; `strength 1` runs all 20 from a fully noised start; `strength 0.04` gives `⌊0.8⌋ = 0` steps and returns the
encoded-and-decoded input (`initSteps == 0` in `ImagePipeline.one`). The start state is `x̃ = z₀ + σ_{start} ε` where `z₀` is a *sample* from the encoder's posterior `N(μ, s²)` (not its mean), and `ε` the noise drawn after it (22.10).

*Worked example, one blended step of inpainting* (latent resolution, one number): the input's clean latent is `z₀ = 0.4`, the noise `ε = 0.7` (the same `ε` at every step), the next noise level `σ_{i+1} = 1.6129`, the network's new state `x = 1.1`, the mask `m = 0.25` (a soft edge):
the part to keep, noised to the level the latents are now at, is `z₀ + σ_{i+1} ε = 0.4 + 1.6129 · 0.7 = 1.529`, and the blended state is `(1 − m) · 1.529 + m · 1.1 = 0.75 · 1.529 + 0.25 · 1.1 = 1.422`. At `m = 0` the network never changes the pixel's latent; at `m = 1` it is free. After the VAE decode the pixels with `m < 1` are mixed with the original ones:
`m · generated + (1 − m) · original`, e.g. `0.25 · 200 + 0.75 · 100 = 125`, and pixels with `m = 0` are copied untouched.
*The mask* reaches the latent grid by area averaging (a 512² mask becomes 64²: each latent cell holds the mean of its 8 × 8 pixels), so a feathered mask stays feathered; for the 9-channel UNet the masked image is the input with the pixels where `m ≥ 0.5` set to 0 before encoding.

**Why this design (22.9).** *Blending* works with any model (SD-Turbo has no inpainting UNet), costs nothing extra per step, and keeps the unmasked region consistent with the noise level the network sees; its weakness is that the network was never trained to fill a hole, only to denoise a whole picture, so nothing forces the new content to match its surroundings except the noised keep-region it sees. The *9-channel UNet* was trained for exactly this task (it receives the mask and the masked picture) and needs no blending, but exists only for the models that were released with one, and needs the masked picture encoded as well (one more VAE encode). Both are implemented because neither covers every model. *Outpainting* is inpainting on an enlarged canvas, done in the browser (22.12). *Upscaling* is image to image at a low strength on a bicubic enlargement: it adds plausible detail, not
information that was not there; the API accepts `upscale` from 1 to 4 and the area of a picture is limited to 2048 × 2048 pixels. **Settings**: `strength` (default 0.8; 1.0 with a mask; 0.35 with `upscale`); `image.init`, `image.mask`, `image.strength` for the command line; the request's `mask` is white or transparent where the picture is to be repainted.

---

## 22.10 Random numbers and reproducibility

The same `seed` always gives the same picture (on one device and precision). The noise is *defined*, so that it can be reproduced
outside Java (the reference scripts do, in numpy), and a picture made by jllm can be compared with diffusers' on the same noise:

- a counter-based **SplitMix64** stream: output `i` (1-based) is `mix(seed + i · 0x9E3779B97F4A7C15)` with the usual finalizer
  `z = (z ⊕ z>>30)·0xBF58476D1CE4E5B9; z = (z ⊕ z>>27)·0x94D049BB133111EB; z ⊕ z>>31`;
- a uniform in (0, 1): `u = ((out >>> 11) + 0.5)·2⁻⁵³`;
- standard normals by **Box–Muller**, two per pair `(u₁, u₂)`: `r = √(−2 ln u₁)`, `z₀ = r cos 2πu₂`, `z₁ = r sin 2πu₂`, rounded to float32.

Image `k` of a batch uses `seed + k`. One stream per image supplies, in this order (the order of diffusers' pipelines, which is what
makes the comparison possible): the initial noise `[4, h, w]` in the channels-first order PyTorch uses, then for image to image
the posterior sample of the encoder and the noise added to it, then for inpainting the masked image's sample, then the fresh noise
of each ancestral step. `seed` omitted: random; the seeds used are returned (`data[i].seed`).

*Worked example, seed 42* (computed by hand-rolled numpy and by `Rng`, which agree): the first two stream outputs are `mix(42 + 1·0x9E3779B97F4A7C15) = 0xbdd732262feb6e95` and `mix(42 + 2·0x9E3779B97F4A7C15) = 0x28efe333b266f103`; their uniforms `((out >>> 11) + 0.5) · 2⁻⁵³` are `u₁ = 0.74156`, `u₂ = 0.15991`;
Box-Muller: `r = √(−2 ln 0.74156) = 0.77330`, `z₀ = r cos(2π · 0.15991) = 0.41472`, `z₁ = r sin(2π · 0.15991) = 0.65268`; the next pair gives `−0.89189` and `1.32683`. So the first noise numbers of `seed = 42` are 0.41472, 0.65268, −0.89189, 1.32683, … on every machine, in Java and in numpy; if the count of numbers asked
for is odd, the second number of the last pair is kept for the next call (`Rng.normal` keeps a spare). A counter-based stream (output `i` depends only on `seed` and `i`) was chosen over a stateful generator so the numbers can be reproduced anywhere without running the earlier ones, which is what lets the reference scripts feed diffusers the same noise (22.13).

---

## 22.11 How jllm computes it

**One set of operations, two devices** (`Ops`). The three networks are written once, against an interface of 14 operations on
channels-last tensors (`[B, H, W, C]` images, `[rows, C]` token matrices): `linear` (with bias, activation and residual add
fused), `conv2d` (with bias, a per-batch-element channel bias — the time embedding — and a residual add), `upsampleConv`,
`groupNorm`, `layerNorm`, `attention`, `geglu`, `add`, `silu`, `concat`, plus tensor creation, upload/download and weight loading
(which transposes a diffusers `[Cout, Cin, kh, kw]` convolution weight to `[Cout, kh, kw, Cin]`). `CpuOps` is the float32 reference
(Vector API); `MetalOps` runs the same calls on the GPU in float16 or float32. Because every operation of one network goes through
this interface, the UNet and the VAE are *tested once* against diffusers and *the GPU kernels are tested* against the CPU
(`DiffusionTest.opsParity`).

**A convolution is a matrix product.** With the input channels-last, output pixel `m = (b, oy, ox)` and patch index
`k = (ky, kx, ci)`:

```
Y[m, co] = Σ_k A[m, k] · W[co, k]          A[m, (ky,kx,ci)] = x[b, oy·s − pad + ky, ox·s − pad + kx, ci]   (0 outside the image)
```

so a 3 × 3 convolution from `Cin` to `Cout` channels is a `(B·Ho·Wo) × (9·Cin)` times `(9·Cin) × Cout` product, the same shape of
problem as a linear layer. The CPU gathers the patches of a 32-row tile into a scratch buffer and runs the register-tiled
micro-kernel (4 rows × 4 vectors of accumulators, weights packed in panels; each task copies the panel it needs from the
off-heap store into a small thread-local array first: with the weights read straight from the off-heap segment, this loop ran at 0.3
GFLOP/s on this JVM (JDK 26, Apple M1) instead of 66 per core with `fromArray`, although a plain sum over a segment is as fast as
over an array; the copy costs a few percent); the GPU **never builds the patch matrix**: its kernel
(`df_conv`) gathers the `A` slice it needs straight from the image while it stages it in threadgroup memory ("implicit GEMM").

**The GPU kernels** (`src/main/resources/org/robomiri/jllm/metal/diffusion.metal`, a Metal library of its own, about 1 060 lines with the video and Wan kernels; every
kernel has a numpy reference in `scripts/metal-emu/kernel_tests_diffusion.py`, run on the emulator of [chapter 18](18-metal-emulator.md)
in float32 and float16):

| kernel | what | design |
|---|---|---|
| `df_gemm` | `act(A·Wᵀ + bias) + residual`, also the batched products of attention | threadgroup = 4 simdgroups, 32-wide k slices staged in threadgroup memory; **half**: a 64 × 64 output tile, each simdgroup owns a 32 × 32 quarter = sixteen `simdgroup_float8x8` accumulators; **float**: a 64 × 32 tile, eight accumulators per simdgroup (with sixteen float accumulators the compiler's register allocation collapses and the kernel ran 6× slower: 0.3 instead of 1.8 TFLOP/s); inputs half or float, always float accumulation |
| `df_conv` | convolution as implicit GEMM; bias, channel bias, residual fused; optional nearest-neighbour 2× upsampling of the input (`ups`) | as `df_gemm`; the `A` slice is gathered per 8 channels (one vector load) from the input |
| `df_conv_direct` | the convolutions whose input or output channel count is not a multiple of 8 (4 latent channels in, 3 colours out, 9 for an inpainting UNet) | one thread per output element |
| `df_flash_d{8…160}` | attention `softmax(q kᵀ/√d) v` without a scores matrix | the flash attention of [chapter 8 §8.13](08-metal-backend.md#813-flash-attention-prefill-attention_fa): 32 query rows per threadgroup, 32 keys per block, online softmax, row rescaling by a diagonal `simdgroup_float8x8`; heads of width 8 to 160 (SD 1.5's 40, 80, 160 and the 64 of SD 2 / SDXL); other widths (the VAE's 512) use `df_gemm` for `QKᵀ`, `df_softmax`, `df_gemm` for `PV`, a few heads at a time |
| `df_groupnorm` / `df_gn_stats`, `df_gn_combine`, `df_gn_apply` | group norm: one threadgroup per group for small tensors; for large ones statistics per chunk of pixels (many threadgroups), merged with the parallel-variance formula `M₂ = Σ(M₂_c + n_c(μ_c − μ)²)`, then applied 4 channels per thread | |
| `df_layernorm` | one simdgroup per row (8 rows per threadgroup), no barriers | |
| `df_geglu`, `df_add`, `df_silu`, `df_upsample2x`, `df_concat` | elementwise and layout | |

*Memory.* Activations come from a pool of buffers keyed by size and are returned as soon as the code that made them is done with
them; the command stream is in order, so a returned buffer can be handed out again immediately. Dispatches are recorded and
submitted 192 at a time; a download waits for the GPU. The weights stay half precision on the GPU (a copy of the file's float16
values; the CPU path packs them to float32 in panels outside the Java heap).

*Precision.* Weights, activations and stored tensors are float16 on the GPU by default (`image.precision`), all sums are float32,
group and layer norm statistics are float32 two-pass, softmax is float32, the VAE of SDXL-type models is float32.

*Settings and switches* (`jllm.properties`, also in the console's Settings page): `image.backend` (`auto`/`metal`/`cpu`),
`image.precision` (`auto`/`f16`/`f32`), `image.vae-tile`, `image.vae-tile-overlap`; `-Djllm.diffusion.flash=false` (attention by
scores + softmax), `-Djllm.diffusion.simpleNorm=true` (group norm in one threadgroup per group), `-Djllm.diffusion.profile=true`
(time every kernel; slow). `scripts/mac-validate.sh` has a row for each.

*Worked example, the shape of a convolution as a matrix product* (the first level of the SD 1.5 UNet, batch 2): output pixels `M = 2 · 64 · 64 = 8 192`, `K = 9 · 320 = 2 880`, `N = 320`; the work is `2 · M · K · N = 15.1 GFLOP`. A row of `A` for the output pixel `(oy, ox)` is the nine neighbouring pixels' 320 channels side by side
(`ky, kx` outer, channels inner, zeros outside the picture), in the same `(ky, kx, ci)` order as the weight layout `[Cout, kh, kw, Cin]`:

```
 pixel (oy, ox) of the output                     A row (2 880 numbers)                                              weight row of output channel co (2 880)
 ┌────┬────┬────┐                                 [ x(oy-1,ox-1,0..319) | x(oy-1,ox,0..319) | x(oy-1,ox+1,..) | …    ·  W[co] = [ w(0,0,0..319) | w(0,1,..) | … | w(2,2,..) ]
 │ -1 │ -1 │ -1 │  neighbours read with zero          x(oy,ox-1,..) | x(oy,ox,..) | … | x(oy+1,ox+1,0..319) ]            y[pixel, co] = Σ_k A[pixel, k] · W[co, k]
 └────┴────┴────┘  padding at the picture's edge
 CPU: a tile of 32 rows of A is gathered into a scratch array, then multiplied;     GPU: no A is built (it would be 8 192 × 2 880 × 2 B = 47 MB per convolution);
 df_conv reads each 8-channel slice straight from the image into threadgroup memory while it stages the k slice (implicit GEMM)
```

*The tile of `df_gemm`* (`diffusion.metal`): a threadgroup of 4 simdgroups (128 threads) computes an output tile of `DF_BM = 64` rows of `A` by `BNT` columns of `B` (64 for half storage, 32 for float), stepping through `K` in slices of `DF_BK = 32`. Each slice is staged into threadgroup memory (`A`: 64 × 32, `B`: `BNT` × 32, two threads per `A` row, 16 values each, with 4-wide vector loads), then simdgroup `sg` multiplies its quarter: rows `32 · (sg & 1) …`, columns
`(sg >> 1) · BNT/2 …`, as `4 × SJ` blocks of 8 × 8 (`simdgroup_multiply_accumulate`, accumulators in float): 16 accumulators per simdgroup for half, 8 for float. The epilogue adds the bias, the per-batch-element channel bias (the time embedding of a ResNet), the activation and the residual before the single store. The float version has 8 accumulators per simdgroup because with 16 the compiler's register allocation collapsed (0.3 instead of 1.8 TFLOP/s, see the kernel table above).

### 22.11.1 The CPU matrix kernel and the off-heap loads

🧸 On the CPU the same matrix product is done by a few cooks who each handle a small block at a time with the kitchen's widest knife (a vector instruction). The kitchen turned out to have two shelves for ingredients: one inside the cook's reach (an ordinary Java array) and one in a storeroom next door (memory outside the Java heap, where
the 4.3 GB of float32 weights live). Taking an ingredient from the storeroom with the wide knife's own "reach" instruction worked, but the Java runtime did not turn it into one real wide grab: each grab was done piece by piece, 200 times slower. Carrying a panel of weights to the near shelf first, once, and then cooking from there fixed it.

📐 *The kernel.* The weights of a layer are packed into panels of `NR = 4 · L` output columns (`L` = lanes of a float vector, 4 on this machine, so `NR = 16`): `panel[p][k][j] = W[k][p · NR + j]`, in an off-heap `MemorySegment` (`Arena.ofShared`, 64-byte aligned). For one panel and four rows of `A` the
register tile (`CpuOps.micro`) keeps 16 vector accumulators; per `k` it loads four weight vectors, broadcasts four activations and does 16 fused multiply-adds (128 floating-point operations). A task is 32 rows of `A` (`MB`) times 4 panels; the work is spread over the cores with parallel streams.

*The problem.* The first version loaded the weight vectors with `FloatVector.fromMemorySegment(SP, seg, byteOffset, order)` straight from the packed panel. On this JVM (JDK 26, Apple M1) that loop ran at **0.3 GFLOP/s per core**; the same loop with `FloatVector.fromArray` on a heap array runs at **66 GFLOP/s per core**, about 220 times faster. The surprising part is that a plain sum over the
same `MemorySegment` is as fast as over an array, so the segment itself is fine; the slow part was the vector load from a segment *inside this loop*. The source comment of `micro` says it plainly: "`FloatVector.fromMemorySegment` is not compiled to vector loads by this JVM". Why the JIT gives up here was not investigated further; it is consistent with the cliff described in
[chapter 7](07-cpu-backend.md#what-the-jit-needs-from-a-vector-api-kernel) (a hot loop of the Vector API must stay inside C2's inlining and intrinsic budget, otherwise every vector becomes an object and the loop runs at fallback speed; there, two segment loads per iteration already pushed a compile past 20 s), and this loop has four segment loads per iteration with 16 live accumulators. That is an
explanation by analogy, not a measurement.

*The fix* (commit "CPU matrix kernel reads weight panels through a heap array"): each task copies the panel it is about to use, `k · NR` floats, from the segment into a thread-local `float[]` with one bulk `MemorySegment.copy`, and `micro` reads that array with `fromArray`. A panel of a 3 × 3 convolution with 1 280 input channels is `11 520 · 16 · 4 B = 737 KB`; it is copied once and used for 32 rows, 11.8 MFLOP of arithmetic, so the copy costs a few percent. After the change the CPU path reaches about
**390 GFLOP/s on ten cores** and a one-step SD-Turbo picture takes 12.2 s on the CPU (22.14; an end-to-end time before the fix is not recorded in the repository). The kernels of chapter 7 read their weights with `fromMemorySegment` and are fast because each of their loop iterations does one load; this kernel does four per iteration, which is where it differs.

**Why this design (22.11).** The weights are off-heap (4.3 GB for float32 SD-Turbo, next to 2-2.7 GB of Java heap in use, 22.14); copying a panel to a thread-local array gives the JIT the shape it handles well, at the price of an extra pass over the weights per task. The GPU side avoids the question: `df_conv` never builds a patch matrix.
**Measured**: 22.14. **Checks**: `DiffusionTest.opsParity` runs every operation on the CPU and the GPU and compares them; the CPU path is the float32 reference of 22.13.

---

## 22.12 API, command line, console and chat tools

```sh
./jllm image --image-model stabilityai/sd-turbo --prompt "a red fox in a snowy forest, photo" --out fox.png          # downloads ~2.6 GB once
./jllm image --image-model stabilityai/sd-turbo --prompt "an oil painting" --init fox.png --strength 0.5 --out fox2.png
./jllm serve --image-model stabilityai/sd-turbo --model none --port 8080                                             # images only
./jllm serve --image-model stabilityai/sd-turbo                                                                      # beside a text model
```

**`POST /v1/images/generations`** (OpenAI compatible; the engine's own server and the web console both serve it):

```sh
curl localhost:8080/v1/images/generations -H 'Content-Type: application/json' -d '{
  "prompt": "a red fox in a snowy forest, photo", "size": "512x512", "n": 2, "response_format": "b64_json",
  "steps": 4, "guidance": 0, "seed": 42, "scheduler": "euler" }'
```

`size` is `WxH`, both multiples of 64 (32 for SDXL-type models, whose UNet has three levels, and for the tiny test model); `response_format` is `b64_json` (default) or `url` (a path
`/v1/images/files/<id>.png`, served without the API key: the id is the capability); extras beyond OpenAI's: `negative_prompt`,
`steps`, `guidance` (or `guidance_scale`), `seed`, `scheduler` (`euler`, `euler_a`, `dpmpp_2m`, `dpmpp_2m_karras`), `clip_skip`.
The response lists `data[i] = {b64_json | url, seed, width, height, revised_prompt}`, `usage` and `timing` (text / unet / vae seconds).

**`POST /v1/images/edits`**: `image` and optional `mask` as multipart files (as the OpenAI SDKs send) or as base64 / data URLs in
a JSON body, plus `prompt`, `n`, `size` (`auto` = the picture's size, ≤ 1024), the extras above, and `strength` (default 0.8; 1
with a mask) or `upscale` (1 to 4, e.g. 2: enlarge ×2 and redraw at strength 0.35). Without a mask: image to image; with one: inpainting
(white or transparent pixels are repainted). With an inpainting model the 9-channel route is used.

**Streaming** (`"stream": true`): server-sent events `image_generation.progress` `{image, step, steps}`,
`image_generation.partial_image` (a small PNG preview of the current estimate, made by a 4 × 3 matrix from the latents — no VAE —
so it is free), `image_generation.completed` (the picture), `image_generation.done`, `[DONE]`. Closing the connection cancels
the generation at the next step.

**Console.** *Models* loads and downloads image models like text models (a diffusers repository's fp16 files only; the list shows
the *image* kind and the model's defaults). The **Images** tab: *Create* (prompt, negative prompt, size, steps, guidance, seed,
scheduler, batch, a progress bar per step with live previews), the *gallery* (kept in the browser's IndexedDB; download, use as
input, re-run, delete), and the *editor*: canvas with crop, rotate, flip, resize, brightness / contrast / saturation, a brush and
an eraser that paint the inpainting mask, outpainting (extend the canvas and mask the border), undo / redo, PNG export, and
image-to-image / inpaint / outpaint requests with a strength slider. **Chat tools** `generate_image` and `edit_image` (switch
them on in the chat's tools menu): the model writes the prompt, the picture appears inline in the conversation, and `edit_image`
takes its link back (`strength`, or a `region` rectangle to repaint).

**Limits of the API** (checked in `ImageApi` / `ImagePipeline.validate`): `n` 1 to 10 pictures per request (made one after the other: a pipeline serves one request at a time), `steps` 1 to 200, `guidance` 0 to 50, `size` a multiple of the model's step (`vae scale × 2^(levels − 1)`: 64 for the four-level UNets of SD 1.x / 2.x, 32 for SDXL's three levels and for the tiny test models), at most 2048 × 2048 pixels in area, `strength` 0 to 1, `upscale` 1 to 4 with an `image`;
a `mask` without an `image` is a 400, as is a bad number (each message names the field).

### 22.12.1 Settings

All keys are in `jllm.properties` (validated at start-up, edited in the console's Settings tab); the command line accepts them as `--image.steps 4` and so on. `0`, `-1` or empty mean "the model's own recommendation" (22.8: SD-Turbo 1 step, guidance 0, 512 × 512, Euler; any other model 25 steps, guidance 7.5, its native size, DPM++ 2M Karras).

| key / switch | default | what it does, and when to change it |
|---|---|---|
| `image.model` (`--image-model`) | empty | the model `./jllm serve` loads (a Hub repository, downloaded as fp16 files, or a local folder); `--model none` serves only images |
| `image.backend` | `auto` | `metal` when there is a GPU, else `cpu` (float32, 12.2 s for a one-step picture: only for machines without a GPU); `cpu` to keep the GPU free |
| `image.precision` | `auto` (= f16) | `f32` to compare with a reference or when a model overflows in half precision: 2.38 s instead of 1.68 s for a one-step picture, 4.3 GB of weights instead of 2.6; the VAE of an SDXL-type model is f32 in either case |
| `image.vae-tile`, `image.vae-tile-overlap` | `128`, `8` | latent pixels per side decoded in one piece (128 = 1024 × 1024 in about 3 GB), and the blended overlap of tiles; use `64` on a small machine (3 % brightness difference per tile, 22.13) |
| `image.steps`, `.guidance`, `.width`, `.height`, `.seed`, `.scheduler`, `.negative`, `.count`, `.out` | `0`, `-1`, `0`, `0`, `-1`, empty, empty, `1`, `image.png` | the `jllm image` command; the same names are fields of the API request (`guidance_scale`, `negative_prompt`, `size`, `n`) |
| `image.init`, `image.mask`, `image.strength` | empty, empty, `0.8` | the command's input picture, mask (white = repaint) and how much of the picture is redrawn |
| `-Djllm.diffusion.flash=false` | flash attention on | attention by scores + softmax: for comparison and for kernels the flash version lacks; it costs memory and time (Wan, float32, 1 200 tokens: 5.3 s instead of 2.05 s, chapter 23) |
| `-Djllm.diffusion.simpleNorm=true` | chunked group norm on | one threadgroup per group instead of the chunked three-kernel version (for comparison) |
| `-Djllm.diffusion.profile=true` | off | times every kernel (slow; the "where the time goes" numbers of 22.14) |

`scripts/mac-validate.sh` has a benchmark row for flash attention off and for the chunked group norm off, besides the plain rows for Metal f16 / f32 and the CPU.

### 22.12.2 The image editor

🧸 The editor is a small paint program in the browser with one extra brush. Whatever you paint with the magenta brush is the part of the picture you want the computer to draw again; everything else is kept as it is. Extend the canvas and the new empty border becomes "to be drawn" too, so the model continues the picture beyond its edge (outpainting). Crop, rotate, flip and
resize do what they do elsewhere; "Compare" slides the new picture over the old one.

📐 *Data model* (`images-editor.js`, `ImgCanvas`): the state is `{pic, mask}`, two canvases of the same size. An operation never changes a canvas that is already in the undo history; it makes new canvases (a brush stroke starts by cloning the mask), so the history of 30 states (`HIST`) is cheap to keep. The mask canvas holds one colour (magenta) and uses its alpha as the coverage. A picture over 2048 pixels on a side is scaled
when loaded (`LOAD_MAX`) and a canvas never grows past 4096 (`MAX_DIM`). Everything except the generation call runs in the browser.

*What is sent.* `sendSize()` scales the picture to the model's step (a multiple of 64 for a Stable Diffusion UNet) and, if the "max side" parameter is set, to that side. For image to image the picture goes as a PNG data URL with `strength` (default 0.8). For inpainting and outpainting the mask is built by `buildMask`:

```
mask canvas ─► resize to the sent size ─► grow by `grow` px (dilate)  ─► blur by `blur` px (softens the edge) ─► white scaled by the coverage (a blurred edge becomes grey), drawn on black   ─► PNG data URL
```

and `strength` defaults to 1 (the masked part is redrawn completely). On the server the picture becomes the mask `m` of 22.9 (`m ≥ 0.5` counts as repaint for the 9-channel UNet), area-averaged to the latent grid, so a blurred mask edge becomes a soft blend.

*Outpainting, with numbers.* Extending a 512 × 512 picture by 128 px on the right (a 640 × 512 canvas): (1) the picture is drawn at its place and the rightmost column of pixels is stretched over the new 128 px (edge replication; corners are the corner pixel); (2) that canvas is blurred with a radius of `max(8, min(64, 128/4)) = 32` px, so the border is a smooth smear of the
edge colours, and the sharp picture is drawn on top; (3) the mask is the whole new border (white) plus a **soft overlap** of `overlap` px (default 16, at most half the shorter side) inside the picture, a linear gradient from white at the border to transparent: the model may repaint 16 px of the old picture, fading out, so the seam is not a straight cut; (4) the mode switches to *Outpaint* with strength 1. The prompt should describe the
whole picture (the editor says so in a message), because the model has to continue one scene, not only paint the border.

**Why this design.** Editing is client-side because the pictures are small (a 512 × 512 canvas), the operations are cheap in a canvas, and the server then only needs the two generation endpoints it already has. Filling the border with a blurred copy of the edge gives the network colours to continue from instead of an arbitrary fill. The soft overlap lets the new and the old part be reconciled where they meet instead of along a straight cut (the intent of the design; no seam measurement was made).

**Settings.** None in `jllm.properties`: the editor needs an image model loaded (Models page or `image.model`). Its panel fields are the request fields of 22.12 (`steps`, `guidance`, `scheduler`, `strength`, batch `n`, seed) plus `grow` (0-64 px), `blur` (0-64 px) and the outpaint extents and overlap (0-128 px, default 16).
**Checks and limits.** The server side of the editor's requests is covered by the image routes tests (`org.robomiri.jllm.web.images` in the console, `DiffusionTest.engineServer` in the engine); the canvas code itself has no automated test. Edits are lossy only in the usual ways of a canvas (8 bits per channel, the resizes of the browser); the gallery is in the browser and not shared.

---

## 22.13 Verified against diffusers

The reference is Hugging Face diffusers 0.41 / PyTorch 2.14 on the CPU in float32, run on the *same weights* (the float16 files
widened to float32) and the *same random numbers* (22.10: the reference scripts monkey-patch diffusers' `randn_tensor` to draw
from jllm's stream, in the order the pipeline asks for them). `scripts/diffusion/reference.py` writes every stage's tensors,
`DiffusionCompare` compares them; the committed tests do it on tiny random pipelines
(`scripts/diffusion/make-tiny.py`: SD 1.x with 1 × 1 convolution projections and quick-GELU CLIP; SD 2-style with linear
projections and an OpenCLIP-style encoder; SDXL with two encoders, `text_time` conditioning and two transformer blocks per layer;
an inpainting UNet with 9 input channels).

Relative error is `‖a − b‖ / ‖b‖` over the whole tensor, `b` being diffusers'. PSNR is on the final 8-bit image. Reproduce a row with
`scripts/diffusion/reference.py <model> <out dir> …` and then `DiffusionCompare <model> <out dir> <cpu|metal> <f32|f16>`
(the docstrings at the top of both files list the options).

| Pipeline (weights) | Steps / sampler | jllm | First UNet call | Last latents | Final image |
|---|---|---|---|---|---|
| tiny SD 1.x, SD 2-style, SDXL, inpainting (random weights, committed fixtures) | 1 / 4 / 7, Euler, Euler-a, DPM++ 2M, DPM++ 2M Karras | CPU f32, Metal f32 | ~1·10⁻⁶ | ~1·10⁻⁶ | 130–137 dB |
| the same | the same | Metal f16 | ~1·10⁻³ | ~1·10⁻³ | 65–77 dB |
| `stabilityai/sd-turbo` 512 × 512 | 2, Euler, guidance 0 | Metal f32 | 5.1·10⁻⁶ (text encoder 1.7·10⁻⁶) | 4·10⁻⁵ | 108 dB (VAE decode alone 8·10⁻⁶) |
| the same | the same | Metal f16 | 2.6·10⁻³ | — | 52.8 dB |
| `stable-diffusion-v1-5` 512 × 512 | 6, DPM++ 2M Karras, guidance 7.5 | Metal f32 | 2.7·10⁻⁶ | — | 94 dB |
| `stable-diffusion-v1-5` 512 × 512 | 4, Euler, guidance 7.5 | Metal f32 | 2.7·10⁻⁶ | 8.9·10⁻⁵ | 94 dB |
| `stable-diffusion-v1-5` 512 × 512 | 5, DPM++ 2M, guidance 7.5 | Metal f32 | 2.3·10⁻⁶ | 1.8·10⁻³ | 73.9 dB |
| `stable-diffusion-inpainting` (9-channel UNet) | 2, Euler, guidance 7.5 | Metal f32 | — | 1.2·10⁻⁴ | 96 dB |
| `stabilityai/sdxl-turbo` 512 × 512 | 2, Euler, guidance 0, against diffusers on MPS in float16 | Metal f16 | 8.6·10⁻⁴ | — | 49.8 dB |

The DPM++ 2M row is the weakest: its second-order extrapolation makes the sampler amplify the per-step float32 rounding
differences (latent error 2·10⁻⁵ after the first step, 1.8·10⁻³ after the fifth, with guidance 7.5), where Euler's stay below 10⁻⁴.
The first UNet call is as exact as in every other row, and the sampler arithmetic itself is bit-compatible on the tiny pipelines
(130 dB for all four samplers), so this is conditioning, not a mistake; 74 dB is still far below what the eye can see.

Two things this table taught, both fixed:

- A scheduler config without `timestep_spacing` has the *class's* default in diffusers (`leading` for the PNDM / DDIM / DDPM
  schedulers the Stable Diffusion 1.x repositories ship, which Euler and DPM++ inherit when they are made `from_config`), not the
  Euler default. With the wrong default the inpainting run was off by a different set of sigmas; `Scheduler` now picks the default
  from `_class_name`.
- Decoding 1024 × 1024 in 64-pixel latent tiles differed from decoding in one piece by 29.7 dB PSNR: the VAE's group norms see
  statistics of the tile, not of the image, so every tile came out about 3 % brighter or darker. The default tile is now 128
  latent pixels (1024 image pixels), which means up to 1024 × 1024 is decoded in one piece; larger images are tiled with an 8-pixel
  blended overlap (`image.vae-tile`, `image.vae-tile-overlap`).

float16 is not "bit-identical": its error is the rounding of half-precision weights and activations (about 10⁻³ relative in a UNet,
which is 50–55 dB in the picture, indistinguishable by eye). Everything below that is checked in float32 where a mismatch cannot hide
behind rounding.

### 22.13.1 Checks and limits

**What is checked, and how.** (1) Stage by stage against diffusers 0.41 / PyTorch 2.14 on the same weights and the same noise (table above): text encoder 1.7·10⁻⁶, first UNet call 2-5·10⁻⁶, final images 74-137 dB in float32 on Metal and CPU (the low end is the DPM++ 2M row), 50-77 dB in float16. (2) The committed tests in `mvn test`
(`DiffusionTest`) on tiny random pipelines made by `scripts/diffusion/make-tiny.py`, with diffusers' reference answers stored in `src/test/resources`: SD 1.x, SD 2-style, SDXL, an inpainting UNet; CPU float32, Metal float32 and float16; every operation of `Ops` on the CPU against the GPU (`opsParity`); the random generator, the tokenizer, resampling, the VAE tiling plan, the diffusers file selection and the HTTP routes. `mvn test` runs it on the tiny pipelines (tolerance 10⁻⁵ relative in float32, looser for half precision). (3) Every Metal kernel against numpy on the emulator, in float32 and float16 (`scripts/metal-emu/kernel_tests_diffusion.py`, chapter 18).

**What is not done or not exact.**
* float16 is not bit-identical to float32 (about 10⁻³ relative in the UNet, 50-55 dB in the picture); DPM++ 2M amplifies rounding differences between implementations (74 dB against diffusers after five guided steps).
* A tiled VAE decode differs slightly from a whole one (3 % brightness per tile at 64-pixel tiles); the default keeps pictures up to 1024 × 1024 whole.
* Speed: about 2 times slower than PyTorch's MPS on the same GPU and 2.4-3.1 times slower than PyTorch on the CPU (22.14); no Winograd-style convolution.
* Not implemented: LoRA, ControlNet, the SDXL refiner, textual inversion, schedulers beyond 22.7, `.ckpt` / `.bin` weights, prompts over 77 tokens or weighted prompts, videos from this pipeline (chapter 23), and pictures above 2048 × 2048 pixels.
* Turbo defaults are chosen by the model's name; a renamed folder needs explicit `steps`, `guidance` and `scheduler`.
* The console's gallery lives in the browser (IndexedDB) and is not shared between browsers; the `url` response format serves a file by its id without the API key (the id is the capability).

---

## 22.14 Measured speed and memory

Measured on an Apple M1 Pro (16-core GPU, 32 GB), JDK 26, with `stabilityai/sd-turbo` (guidance 0, one image, prompt "a photo of an
astronaut riding a horse on the moon"), seconds per image *after* warm-up, the machine otherwise idle (the benchmarks ran inside the
project's heavy-job lock). Reproduce with

```
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -cp target/classes:target/test-classes \
     org.robomiri.jllm.diffusion.DiffusionBench MODEL_DIR metal|cpu f16|f32 STEPS WIDTH HEIGHT REPS
python -I scripts/diffusion/bench_diffusers.py MODEL_DIR --device mps|cpu --dtype float16|float32 --steps 1,4,20
```

| 512 × 512 | jllm 1 step | 4 steps | 20 steps | diffusers 1 step | 4 steps | 20 steps |
|---|---|---|---|---|---|---|
| Metal, float16 (sd-turbo) | 1.68 | 2.92 | 9.63 | 0.78 | 1.43 | 5.03 |
| Metal, float32 (sd-turbo) | 2.38 | 4.14 | — | 0.92 | 1.73 | — |
| CPU, float32 (sd-turbo) | 12.2 | 24.3 | — | 5.02 | 7.76 | — |
| Metal, float16 (sdxl-turbo) | 2.70 | 4.70 | — | 1.13 | 2.47 | — |

(diffusers: 0.41 / PyTorch 2.14 on the `mps` device or the CPU, same weights.) A 1024 × 1024 sd-turbo image in float16 takes
16 s for 4 steps (UNet 2.4 s per call, VAE 5.4–6.2 s). A step costs 0.42 s (UNet) on SD 1.x / 2.x at 512², 0.72 s on SDXL; the fixed cost
of an image is the VAE decode, 1.2 s at 512² (2.5 TFLOP), plus 0.05 s for the text encoder. So a 1-step Turbo image is dominated by the
decoder and a 20-step image by the UNet.

*Where the time goes* (`-Djllm.diffusion.profile=true`, one step, float16): convolutions 65 % (1.35 s of 2.1 s with every dispatch
waited for), linear layers 16 %, flash attention 5 %, group norms 10 %. The convolution and matrix kernels reach 2.3–2.7 TFLOP/s in
isolation (`KernelBench`), about half of the chip's 5.3 TFLOP/s; flash attention 1.5. Padding the threadgroup tiles and 16-byte
loads (`-Djllm.diffusion.defines`, `DF_PAD_BYTES` / `DF_LOAD16B`) were tried as four variants back to back, three rounds: the
differences are within noise (±3 %), so they stay off.

*Against diffusers.* On the GPU jllm needs about 2.1 × the time of PyTorch's MPS backend at one step and 1.9 × at twenty, in float16 as
in float32. MPS has the better convolution: a 3.3 TFLOP image in 0.78 s is 4.2 TFLOP/s "effective", which the chip's 5.3 peak makes
believable only if it does fewer multiplications than a plain implicit GEMM (Winograd-style; inferred, not inspected). That is the
obvious next step for the kernels and is not done. On the CPU jllm is 2.4–3.1 × slower than PyTorch's CPU path (a 4 × 4 register tile at about
390 GFLOP/s on the ten cores against PyTorch's own kernels).

*Memory* (512², peak, after the first image):

| | weights | GPU buffers (pool included) | Java heap in use |
|---|---|---|---|
| jllm Metal f16, sd-turbo | 2.6 GB (as in the file) | 3.2 GB | < 2 GB |
| jllm Metal f32, sd-turbo | 4.3 GB (widened) | 5.0 GB, peak 6.5 GB while loading | < 2 GB |
| jllm Metal f16, sdxl-turbo | 6.9 GB | 6.9 GB, peak 8.3 GB | ~1 GB |
| diffusers MPS fp16, sd-turbo | | 2.6 GB allocated, 4.7 GB driver | |
| diffusers MPS fp32, sd-turbo | | 5.2 GB allocated, 7.9 GB driver | |
| diffusers MPS fp16, sdxl-turbo | | 7.0 GB allocated, 10.0 GB driver | |
| jllm CPU f32, sd-turbo | 4.3 GB outside the heap | — | 2–2.7 GB |

The scratch-buffer pool is trimmed after every image (`Ops.trim`), so a server holding an image model idles at the weights' size.

### 22.14.1 What the numbers mean

* *Time adds up from the pieces.* A one-step Turbo picture is `0.42 s` (one UNet call, no guidance) `+ 1.2 s` (the VAE decode, 2.5 TFLOP) `+ 0.05 s` (text encoder) `= 1.67 s`, and the table says 1.68 s; a 20-step picture is `20 · 0.42 + 1.2 + 0.05 = 9.65 s` against the measured 9.63 s. So the cost of a picture is `steps · (UNet call) + a fixed 1.25 s`, the UNet call is 0.42 s on SD 1.x / 2.x and 0.72 s on SDXL at 512², and four steps of a Turbo model take
not much more than one because the VAE dominates the short runs.
* *Utilisation.* 0.80 TFLOP in 0.42 s is 1.9 TFLOP/s for the whole UNet call against 2.3-2.7 TFLOP/s for the convolution and matrix kernels alone and a chip peak of 5.3 TFLOP/s: about a third of the peak in the network as a whole, a half in the best kernels. PyTorch's 4.2 TFLOP/s "effective" is more than a plain implicit GEMM can reach, which is the argument for a Winograd-style convolution as the next step (inferred, not inspected).
* *Float32 costs 1.4 times the time and 1.7 times the memory of float16* (2.38 s against 1.68 s for one step; 4.3 GB against 2.6 GB of weights): the GPU stores and moves twice the bytes, while the arithmetic is float32 either way.
* *The CPU number is for machines without a GPU:* 12.2 s for a one-step picture is 390 GFLOP/s sustained by the ten cores after the fix of 22.11.1.
* *Memory.* The GPU rows are the weights plus the pooled scratch buffers; the pool is trimmed after every image, so an idle server holds the weights only. Peak figures above the steady ones (6.5 GB against 5.0 GB for f32) are the load, when the float16 file is widened.

---

## 22.15 What video will reuse and what it will need

*(Written before chapter 23; AnimateDiff and Wan 2.1 are now built on exactly these pieces, and chapter 23 says what was added and what is still missing.)*

A video model (Wan 2.x and similar) is a *diffusion transformer* over a video latent. The structure is the same: text encoder →
iterated denoiser → VAE decoder. What carries over unchanged:

- `Ops`, the tensor pool, the two devices and the way weights are loaded from safetensors (`Checkpoint`, `Loader`);
- `Scheduler`'s machinery (timesteps, sigmas, noising for video-to-video and masks); a flow-matching (rectified-flow) sampler is a
  new 20-line `step` — the models above use ε-prediction, Wan predicts the velocity `v = ε − x₀` along straight paths;
- the seeded `Rng`, the pipeline's cancellation / progress / preview hooks, `ImageApi`'s streaming events, the gallery and editor
  (frames are images), `DiffusersRepo` (Wan ships in the same layout), the tiled-VAE blending (in time as well as space);
- the GPU building blocks: `df_gemm` (every linear layer of a DiT), `df_flash_d*` (self-attention over *all* tokens of all frames
  is the whole point of a DiT: 30 000+ tokens, where the flash kernel's lack of a scores matrix matters), the norms, `df_geglu`.

What is missing: (1) a **3-D VAE** (causal 3-D convolutions with a temporal cache, 4 × 8 × 8 compression): `conv3d` as an
implicit GEMM is `df_conv` with a third gather dimension (the `A` slice gather already is the part that changes); (2) the DiT
blocks themselves (adaLN modulation from the timestep, 3-D rotary position embeddings, cross-attention to the text), all
expressible with the 14 existing operations plus a RoPE operation; (3) a **T5** text encoder (UMT5-XXL for Wan: relative position
buckets and gated-GELU feed-forward; the CLIP encoder here is a different, causal architecture); (4) memory: a 5-second video's
latent is hundreds of megabytes per activation, so the pool needs a size cap and the DiT needs sequence parallelism over frame
chunks or quantized (Q8 / Q4 `jllm` formats) weights, which the engine already has for language models.
