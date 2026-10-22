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
inverse of the schedule at that σ. For `n = 5` this gives σ = 14.61, 6.08, 2.23, 0.69, 0.17, then 0.

Prediction types: `epsilon` (all models here but SD 2.1-768) and `v_prediction`
(`x̂₀ = −σ/√(σ²+1)·v + x̃/(σ²+1)`) are implemented; the scheduler config is read from `scheduler/scheduler_config.json`.

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

**The GPU kernels** (`src/main/resources/org/robomiri/jllm/metal/diffusion.metal`, a Metal library of its own, 900 lines; every
kernel has a numpy reference in `scripts/metal-emu/kernel_tests_diffusion.py`, run on the emulator of [chapter 18](18-metal-emulator.md)
in float32 and float16):

| kernel | what | design |
|---|---|---|
| `df_gemm` | `act(A·Wᵀ + bias) + residual`, also the batched products of attention | threadgroup = 4 simdgroups, 32-wide k slices staged in threadgroup memory; **half**: a 64 × 64 output tile, each simdgroup owns a 32 × 32 quarter = sixteen `simdgroup_float8x8` accumulators; **float**: a 64 × 32 tile, eight accumulators per simdgroup (with sixteen float accumulators the compiler's register allocation collapses and the kernel ran 8× slower: 0.3 instead of 1.8 TFLOP/s); inputs half or float, always float accumulation |
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

`size` is `WxH`, both multiples of 64 (32 for the tiny test model); `response_format` is `b64_json` (default) or `url` (a path
`/v1/images/files/<id>.png`, served without the API key: the id is the capability); extras beyond OpenAI's: `negative_prompt`,
`steps`, `guidance` (or `guidance_scale`), `seed`, `scheduler` (`euler`, `euler_a`, `dpmpp_2m`, `dpmpp_2m_karras`), `clip_skip`.
The response lists `data[i] = {b64_json | url, seed, width, height, revised_prompt}`, `usage` and `timing` (text / unet / vae seconds).

**`POST /v1/images/edits`**: `image` and optional `mask` as multipart files (as the OpenAI SDKs send) or as base64 / data URLs in
a JSON body, plus `prompt`, `n`, `size` (`auto` = the picture's size, ≤ 1024), the extras above, and `strength` (default 0.8; 1
with a mask) or `upscale` (2: enlarge ×2 and redraw at strength 0.35). Without a mask: image to image; with one: inpainting
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

---

## 22.15 What video will reuse and what it will need

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
