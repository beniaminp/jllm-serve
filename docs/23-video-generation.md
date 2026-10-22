# 23. Video generation (AnimateDiff)

jllm-serve can also make short video clips. Given a sentence it paints a few seconds of moving picture, with no sound. The model is
**AnimateDiff**: the Stable Diffusion 1.5 image model of [chapter 22](22-image-generation.md), plus a small set of extra layers (the
*motion adapter*, `guoyww/animatediff-motion-adapter-v1-5-2`) that let the frames of a clip look at one another so that they agree on what
moves and how. Everything is written from scratch on the image stack: the extra layers, the sampler, the file writers (animated GIF,
animated lossless WebP, MP4 through `ffmpeg` when it is installed), the HTTP API, the command, the browser editor and the chat tool.
Nothing is borrowed from PyTorch or diffusers except, in the tests, their *answers* to compare with.

| | what | where |
|---|---|---|
| [23.1](#231-a-video-is-many-pictures-that-must-agree) | the idea: many pictures painted together so that they agree | `diffusion/VideoPipeline` |
| [23.2](#232-animatediff-a-motion-adapter-on-a-frozen-image-model) | the motion adapter: where its layers sit in the UNet | `UNet.attachMotion` |
| [23.3](#233-the-temporal-transformer) | one pixel looking at itself in every frame: norm over the clip, positions, attention along the frame axis | `UNet.motion` |
| [23.4](#234-frames-in-the-batch-guidance-text-noise) | frames in the batch, guidance on clips, the order of the noise numbers | `VideoPipeline.generate` |
| [23.5](#235-the-sampler-ddim-and-the-linear-schedule) | DDIM and the *linear* noise schedule AnimateDiff was trained with | `Scheduler` |
| [23.6](#236-decoding-the-frames) | the VAE, one frame at a time | `VideoPipeline.decode` |
| [23.7](#237-how-jllm-computes-it) | three new operations on the image stack; the new GPU kernels; memory | `Ops`, `diffusion.metal` |
| [23.8](#238-writing-the-file-gif-webp-mp4) | GIF (palette, dithering, LZW), lossless WebP (VP8L: predictors, back references, prefix codes), MP4 | `VideoEncoder`, `Vp8l` |
| [23.9](#239-api-command-line-console-and-chat-tool) | `/v1/videos/*`, `jllm video`, the Video tab and its editor, `generate_video` | `VideoApi`, `VideoHttp`, `web/video` |
| [23.10](#2310-verified-against-diffusers) | stage-by-stage comparison with Hugging Face diffusers | `VideoCompare`, `VideoTest` |
| [23.11](#2311-measured-speed-and-memory) | seconds per clip and per step on this Mac, against diffusers on the same GPU | `VideoBench` |
| [23.12](#2312-wan-21-how-far-it-got-and-what-remains) | Wan 2.1 (a diffusion transformer video model): what was checked and what is missing | |
| [23.13](#2313-why-the-clips-are-limited) | which limits of the clips are the code's and which the models' | |

---

Each part is explained in the same order: 🧸 the idea in plain words, 📐 how it works (formulas with every symbol defined, a worked example with real numbers, the shapes of the tensors), why it was built this way, the switches, the measurements and what they mean, and
what is checked and what is not. The gathered "settings" are in [23.9.1](#2391-settings), the editor in [23.9.2](#2392-the-clip-editor), what the measurements mean in [23.11.1](#23111-what-the-numbers-mean), and the second model family
(Wan 2.1) in [23.12](#2312-wan-21-how-far-it-got-and-what-remains) with its parts in 23.12.1-23.12.5; [23.13](#2313-why-the-clips-are-limited) says why the clips are not better than they are.

**The two pipelines with their shapes.**

```
AnimateDiff (SD 1.5 + motion adapter), 16 frames of 512 × 512, guidance 7.5, DDIM        VideoPipeline
 text ─► CLIP ─► [2][77][768] ─► repeated for every frame ─► [2·16][77][768]                  (negative, prompt)
 noise  [16][64][64][4]  (drawn as [4, F, h, w], stored frames-major)
   │ per step:  UNet on the batch [negative clip (16 frames) | prompt clip (16 frames)] = [32][64][64][4]
   │            ├ the image layers treat the 32 frames as 32 pictures                      (860 M parameters)
   │            └ 21 temporal transformers look across the 16 frames of a clip              (454 M parameters)
   │            ε = ε_neg + g (ε_prompt − ε_neg)  ─►  DDIM step on the whole clip           [16][64][64][4]
   ▼
 x / 0.18215 ─► VAE decoder, one frame at a time ─► [16][512][512][3] ─► GIF | WebP | MP4

Wan 2.1 T2V-1.3B, 17 frames of 320 × 192, guidance 5, UniPC, 30 steps                      WanPipeline
 prompt ─► UnigramTokenizer ─► ids [≤ 512] ─► UMT5 on the CPU, int8 ─► [L][4096] ─► zero-padded to [512][4096] ─► MLP ─► text [512][1536]
 noise  [5][24][40][16]      5 = (17 − 1)/4 + 1 latent frames, 24 = 192/8, 40 = 320/8
   │ patchify 1 × 2 × 2 ─► tokens [5 · 12 · 20 = 1 200][1536]
   │ per step: v = DiT(x, t, text) twice (positive and negative prompt), v = v_neg + g (v_pos − v_neg), x ← UniPC(x, v)     (30 blocks, 1.4 B parameters)
   ▼
 x · std + mean per channel ─► causal 3-D VAE decoder ─► 1 + 4 · (5 − 1) = 17 frames [17][192][320][3]
```

---

## 23.1 A video is many pictures that must agree

🧸 A cartoon is a pile of drawings, each a little different from the one before. If a clever helper drew every page separately from a
caption ("a balloon floating over a hill"), the balloon would jump around, change colour and turn into a bird on page 5. To make a
*movie* the pages must be drawn **together**: while the helper wipes the snow off page 7, it keeps glancing at pages 6 and 8 (and all
the others) to make sure the balloon is still the same balloon, only a bit higher. AnimateDiff teaches an image painter exactly that
glance, without teaching it to paint again: the painter stays as it was, and a few small "glance at the other pages" layers are added.

📐 A clip is a latent tensor `x ∈ ℝ^{F × h × w × 4}` (F frames of the 64 × 64 × 4 latents of chapter 22; 16 frames of 512 × 512 pictures
are `16 × 64 × 64 × 4` = 262 144 numbers). Denoising is the same as for one picture (22.1), applied to the whole tensor:
`x_{i+1} = sampler(x_i, ε̂, …)` with `ε̂ = ε_θ(x_i, t_i, text)`. The only change is the network: `ε_θ` sees all F frames at once and
mixes information **along the frame axis** in extra layers, while every layer of the image UNet keeps treating each frame on its
own. Each frame has its own noise, so the clip's frames start different; the temporal layers are what makes them converge to one scene.

*Worked example, what "many pictures" costs.* 16 frames of 512 × 512 are `16 × 64 × 64 × 4 = 262 144` latent numbers; with guidance every UNet evaluation processes `2 × 16 = 32` latents, so one step costs `32 × 0.80 = 25.6` TFLOP in the image layers alone (plus about 40 % for the temporal layers, 23.3), and a 20-step clip about 0.5 PFLOP before the 16 VAE decodes
(`16 × 2.5 = 40` TFLOP). That is why a clip takes minutes where a Turbo picture takes seconds (23.11).

**Why this design (23.1).** Generating the frames independently would cost the same and give a different scene in every frame (the "balloon turns into a bird" of the story); generating them as one tensor with a layer that mixes them is the cheapest change that fixes it. The alternatives are a video model trained as a whole (a video VAE and a diffusion transformer, which is what Wan 2.1 is, 23.12) or
image-to-image on the previous frame (each frame is conditioned only on the one before, so errors accumulate and the clip cannot be made in parallel). AnimateDiff was chosen first because it needs nothing that the image stack does not already have plus 21 temporal transformers.

---

## 23.2 AnimateDiff: a motion adapter on a frozen image model

🧸 Imagine the image painter as a building with many rooms in a row, each room doing one job (look at the words, smooth a patch, ...).
The *motion adapter* is a set of extra rooms, one built after every "look at the words" room, whose only job is to compare the same
patch across all pages. The old rooms are not touched, so any Stable Diffusion 1.5 painter (a cartoon style, a photo style) can be
used with the same adapter.

📐 The SD 1.5 UNet has four resolution levels (64², 32², 16², 8² latents; 320, 640, 1280, 1280 channels), each with two (down) or
three (up) ResNet + spatial-transformer pairs, and a middle block. The adapter holds one **temporal transformer** per pair and one in
the middle:

```
down level i, layer j:  x = ResNet(x);  x = SpatialTransformer(x, text);  x = Temporal(x)      j = 0, 1     (level 3 has no spatial transformer)
mid:                    x = ResNet(x);  x = SpatialTransformer(x, text);  x = Temporal(x);  x = ResNet(x)
up   level i, layer j:  x = ResNet(concat(x, skip));  x = SpatialTransformer(x, text);  x = Temporal(x)      j = 0, 1, 2
```

That is 4 · 2 + 1 + 4 · 3 = **21 temporal transformers**: 454 M parameters in 567 tensors (`guoyww/animatediff-motion-adapter-v1-5-2`
stores them as float32, 1.8 GB; jllm converts them to float16 on the GPU, 0.9 GB), against 860 M in the SD 1.5 UNet. The adapter's
`config.json` gives the layout (`block_out_channels`, `motion_layers_per_block` = 2, `motion_num_attention_heads` = 8,
`motion_norm_num_groups` = 32, `motion_max_seq_length` = 32 positions: no clip longer than 32 frames; the adapter was trained on 16).
`UNet.attachMotion` reads those, checks that the block channels match the UNet it is attached to, and loads
`down_blocks.i.motion_modules.j.*`, `mid_block.motion_modules.0.*`, `up_blocks.i.motion_modules.j.*`. The skip connections of the UNet
carry the output of the temporal layers too (the `skips` list of `UNet.forward` is filled after `Temporal`), as in diffusers'
`UNetMotionModel`.

**A video model is two folders.** The adapter (`video.model`) and the Stable Diffusion 1.x model it rides on (`video.base-model`,
default `stable-diffusion-v1-5/stable-diffusion-v1-5`; any SD 1.5 fine-tune in the diffusers layout works). `VideoModels` downloads both
once into `model.dir`: the adapter's `config.json` and fp16 weights (`diffusion_pytorch_model.fp16.safetensors`; the file is named fp16
but, in this repository, holds float32 values), the base's fp16 files as in 22.2.

*Worked example, the size of the adapter.* One temporal transformer on `C` channels has `proj_in` (`C²`), two attentions (`q, k, v, o`: `4C²` each), the GEGLU feed-forward (`C → 8C`: `8C²`; `4C → C`: `4C²`) and `proj_out` (`C²`): `22 C²` parameters. The 21 modules sit at these widths: down levels 0-3 (2 each) at 320, 640, 1280, 1280; the middle block (1) at 1280; up levels 0-3 (3 each) at 1280, 1280, 640, 320. Adding up:
`2 · 22 · (320² + 640² + 1280² + 1280²) = 166.7 M`, `22 · 1280² = 36.0 M`, `3 · 22 · (1280² + 1280² + 640² + 320²) = 250.1 M`: **452.8 M** of the adapter's 454 M parameters (the rest are the 32-row position tables, the norms and the biases). Most of the weights are in the 1280-wide modules, but the *work* of a module (its linear layers, `22 C²` multiply-adds per row, rows = batch · frames · pixels = `32 · H · W`) is the same at the three large levels: the parameters grow 4 times per level while the pixels shrink 4 times, so each of the 15 modules at 64², 32² and 16² costs `22 · 320² · 131 072 = 295 GMAC`
and each of the 6 modules at 8 × 8 a quarter of that, 74 GMAC. In total `15 · 295 + 6 · 74 = 4.9 TMAC = 9.7 TFLOP` for a 32-image batch, against `32 · 0.80 = 25.6 TFLOP` for the image layers: 38 % more arithmetic, and the measured cost of the modules is +39 to +45 % time (23.11).

**Why this design (23.2).** A motion adapter on a frozen image model reuses everything a Stable Diffusion 1.5 fine-tune already knows about pictures: any checkpoint in the diffusers layout can be the base (`video.base-model`), and only the 454 M adapter parameters are video-specific (1.8 GB as stored, float32; 0.9 GB in float16 on the GPU). The costs: the clip length is capped by the adapter's 32 position
rows (`motion_max_seq_length`; it was trained on 16 frames, so 16 is the safe length), the motion is only as good as the adapter and the base agree (23.13), and the image layers keep treating frames separately, so anything the temporal layers do not repair (the VAE decodes each frame alone, 23.6) stays as flicker. It supports SD 1.x bases only (an SDXL or an inpainting base is refused with a message).

---

## 23.3 The temporal transformer

🧸 Take one tiny square of the picture, say the square where the balloon is on page 5. The extra room gathers the *same square from
every page* into one little row (page 1's square, page 2's square, ...), lets the squares of that row talk to each other ("you moved up
a bit, I will too"), puts the row back. It also stamps every square with its page number, because "page 3 of 16" is not the same as
"page 11 of 16" and the talk needs to know who is before whom. Each of the 4 096 squares of the picture gets its own row.

📐 For hidden states `x ∈ ℝ^{B·F × H × W × C}` (frames of one clip adjacent), `Temporal(x)` is diffusers' `AnimateDiffTransformer3D`:

```
n   = GroupNorm_{32 groups, ε = 1e-6}(x)  over (F, H, W, channels of the group)   — the statistics are taken over the whole CLIP, not per frame
t   = reshape  [B·F, H, W, C] → [B·H·W, F, C]                                     — one sequence of F tokens per pixel
h   = t · W_in + b_in
repeat (1 transformer block):
    h = h + Attn₁(p₁),   p₁ = LayerNorm(h) + PE            — self-attention along the frames, 8 heads of C/8 (40, 80, 160, 160)
    h = h + Attn₂(p₂),   p₂ = LayerNorm(h) + PE            — a second self-attention (diffusers' "double_self_attention": no text here)
    h = h + W₂ · (GEGLU(W₁ · LayerNorm(h)))
y   = reshape back (h · W_out + b_out)  → [B·F, H, W, C];   output = x + y
PE[p, 2i] = sin(p / 10000^{2i/C}),  PE[p, 2i+1] = cos(p / 10000^{2i/C})   (a table of 32 positions, stored in the checkpoint)
```

Note the position embedding is added to the **input of the attention only** (queries, keys and values all come from `LayerNorm(h) + PE`),
not to the residual stream. The attention matrices are tiny (F × F = 16 × 16) but there are `B · H · W` of them: 2 · 4 096 sequences at the
top level. The cost is in the linear layers (`C → C` projections, a `C → 8C` GEGLU and `4C → C`), run on `B · F · H · W` rows, which is
why the temporal layers add roughly 40 % to the time of a UNet evaluation. The group norm over the clip is the one subtle
point: normalising each frame alone changes the result (a frame brighter than the others would be normalised back to the mean), and the
test of the tiny pipeline fails at 1 · 10⁻² if it is done per frame.

*Worked example 1: why the layer needs positions.* One pixel, four frames, one head of width 2, projections left out (`q = k = v = h`), the frame contents `a b a b` with `a = [1, 0]`, `b = [0, 1]`. Attention is `softmax(h hᵀ/√2) h`:

```
without positions   weights of frame 0: [0.335 0.165 0.335 0.165]  → output [0.67 0.33]      frame 2 → the same row → [0.67 0.33]
                    (frames 0 and 2 have the same content, so nothing tells them apart: attention cannot know the order of the frames)
with PE added       PE[p] = [sin p, cos p] · 0.5 (a scale chosen for the toy):  [0, 0.5], [0.42, 0.27], [0.45, −0.21], [0.07, −0.50]
                    weights of frame 0: [0.289 0.252 0.310 0.150] → output [0.86 0.48]      frame 2: [0.274 0.135 0.486 0.105] → output [1.05 0.26]
```

Self-attention is *permutation-equivariant*: shuffle the frames and the outputs shuffle the same way. The position embedding is what turns "a set of frames" into "a sequence", and it is added to the attention input only (queries, keys and values all come from `LayerNorm(h) + PE`). The real table has 32 rows of width `C`:
`PE[p, 2i] = sin(p/10000^{2i/C})`, `PE[p, 2i+1] = cos(p/10000^{2i/C})`; for a toy width `C = 4` the rows are `p=0: [0, 1, 0, 1]`, `p=1: [0.841, 0.540, 0.010, 1.000]`, `p=2: [0.909, −0.416, 0.020, 1.000]`, `p=3: [0.141, −0.990, 0.030, 1.000]`: the first pair turns fast (it tells neighbouring frames apart),
the last turns slowly (it tells early from late).

*Worked example 2: why the group norm is taken over the clip.* Four frames, one group of two channels each, the third and fourth frame brighter: values `[1, 3] [2, 4] [10, 12] [11, 13]`.

```
over the whole clip (mean 7, variance 21.5, ε 1e-6):   frame 0: [−1.29 −0.86]   frame 1: [−1.08 −0.65]   frame 2: [0.65 1.08]   frame 3: [0.86 1.29]      brightness differences between frames survive
frame by frame (each frame has its own mean and variance):   every frame becomes [−1, +1]                                                              the differences are gone
```

A frame that is brighter than the others *should* stay brighter after the norm (the temporal layer then decides whether that is motion or flicker); normalising each frame alone throws that information away, which is why the tiny-pipeline test fails at 1·10⁻² when it is done per frame. The implementation: `ops.groupNorm(x, γ, β, 32 groups, 1e-6, false, samples = bsz)` makes the statistics span the `F` frames of one clip (`UNet.motion`).

*The shapes.* `x [B·F, H, W, C]` → group norm → `swapAxes` → `[B·H·W, F, C]` (a sequence of `F` tokens per pixel) → `proj_in` → one transformer block (two attentions with `PE` added after each layer norm, then GEGLU) → `proj_out` → `swapAxes` back → `x + y`. For SD 1.5 at 512² with guidance and 16 frames that is `2 · 4 096 = 8 192` sequences of 16 tokens at the first level; their attention matrices are `16 × 16`,
one per head (8 heads of width `C/8` = 40, 80, 160, 160): tiny matrices, but thousands of them, which the batched flash kernels of chapter 22 run.

---

## 23.4 Frames in the batch, guidance, text, noise

🧸 Sixteen pages and two helpers (one who listens to the caption, one who ignores it), so thirty-two drawings are being improved at the
same time, in one go. The caption is written on a card and a copy of the card is put next to each page.

📐 With classifier-free guidance (22.6) the UNet batch is `[negative clip (F frames), prompt clip (F frames)]`, `2F` items; the text
embeddings `[2, 77, 768]` are repeated for each frame (diffusers' `repeat_interleave(F)`), the timestep embedding is computed per
batch item, and the two halves of the output combine as `ε = ε_neg + g (ε_prompt − ε_neg)` frame by frame. The temporal layers see
`B = 2` clips of `F` frames. In memory the clip is `[F, h, w, 4]` frames-major (NHWC per frame); PyTorch draws its noise as `[4, F, h, w]`
(channels first, then frames) from one `randn_tensor` call, and so does jllm (`Rng(seed).normal(4·F·h·w)`, then `toFrames`), so the same
seed gives the same noise as the reference. Clip k of a request uses seed `seed + k`.

*Worked example, the layout of the clip in memory and of the noise.* PyTorch draws the clip's noise as one tensor `[4, F, h, w]` (channels first, then frames); jllm stores the clip frames-major, `[F, h, w, 4]`. The noise number of channel `c`, frame `f`, pixel `p` (`0 ≤ p < h·w`) is entry `c·F·h·w + f·h·w + p` of the stream and is stored at `(f·h·w + p)·4 + c`; with `F = 16`, `h = w = 64`
the first number of channel 1 is stream entry `1 · 16 · 4 096 = 65 536` (and lands at index 1 of frame 0). Clip `k` of a request uses `Rng(seed + k)`, the same stream rule as images (22.10). The batch for one UNet call:

```
 batch index       0 … F−1                       F … 2F−1
 content           negative clip, frames 0…F−1    prompt clip, frames 0…F−1          text embedding repeated per frame (repeat_interleave): [neg ×F | prompt ×F]
 temporal layers   see the F frames of THEIR clip only (B = 2 clips of F frames); the two guidance halves never mix
 output            ε = ε_neg + g · (ε_prompt − ε_neg)   frame by frame  →  [F][h][w][4]
```

**Why this design (23.4).** One batch of `2F` instead of `2F` separate calls reads the weights once and gives the kernels `2F` times the rows; the price is memory: the activations of a 512², 16-frame clip with guidance are those of a batch of 32 images (23.7). Frames of one clip are adjacent in the batch (rather than interleaved with the other clip) because the temporal layers regroup them by `bsz` and `F`, and
the noise order mirrors diffusers' so the same seed gives the same clip as the reference (23.10).

---

## 23.5 The sampler: DDIM and the linear schedule

🧸 Removing snow can be done in big careful steps (DDIM goes straight to where a clean picture should be and comes back a bit) or in
other ways (the Euler and DPM++ samplers of chapter 22). AnimateDiff's authors trained with a slightly different amount of snow at each
step than Stable Diffusion's ("linear" instead of "scaled linear"), so that table has to be switched.

📐 `β_t` runs linearly from 0.00085 to 0.012 over 1000 steps (Stable Diffusion: linearly in `√β`); `VideoPipeline` copies the base
model's scheduler configuration and sets `beta_schedule = linear`, `timestep_spacing = linspace`, `steps_offset = 1`,
`clip_sample = false`, as diffusers' AnimateDiff example does. DDIM with η = 0 (`Scheduler.Kind.DDIM`, new in this chapter):

```
timesteps: round(linspace(0, 999, n)) reversed (999, 957, 916, ... 0 for 25 steps)
ε̂ = ε_θ(x_t),  x̂₀ = (x_t − √(1−ᾱ_t) ε̂) / √ᾱ_t,   x_{t'} = √ᾱ_{t'} x̂₀ + √(1−ᾱ_{t'}) ε̂,    t' = t − 1000/n  (t' < 0: ᾱ = ᾱ_0 when set_alpha_to_one is false)
```

The input is not rescaled (`scale_model_input` is the identity) and the initial noise has unit variance. Euler, Euler-ancestral and
DPM++ 2M are available too (`scheduler`); the request's `scheduler` picks one, `ddim` is the default. Everything is elementwise on the whole
clip, so the sampler code of chapter 22 works unchanged on `F` frames.

*Worked example, the schedule and one DDIM step* (computed with numpy from the linear `β`). For 25 steps `timesteps = round(linspace(0, 999, 25))` reversed: 999, 957, 916, 874, 832, 791, …, 83, 42, 0. The linear schedule keeps less signal at the top than the scaled-linear one of Stable Diffusion: `ᾱ_999 = 0.00158` (scaled linear: 0.00466), so `σ_999 = √((1 − ᾱ)/ᾱ) = 25.1` (14.6 for the
image models): the noise at the first step is 25 times the picture. A single step from `t = 999` with `x_t = 1.0` and the prediction `ε̂ = 0.8`:

```
x̂₀ = (x_t − √(1 − ᾱ_999) ε̂) / √ᾱ_999 = (1.0 − 0.99921 · 0.8) / 0.03974 = 5.049          (a huge estimate: the noise dominates)
t' = t − ⌊1000/25⌋ = 959                         ᾱ_959 = 0.002537
x_959 = √ᾱ_959 · x̂₀ + √(1 − ᾱ_959) · ε̂ = 0.05037 · 5.049 + 0.99873 · 0.8 = 1.0533
```

Two details follow diffusers exactly: the state moves to `t − 40 = 959`, while the next network call is made at the *next entry of the list*, 957 (the two differ by 2 timestep units because the `linspace` spacing is 41.6, not 40); and at the last step, `t = 0`, the "previous" timestep is negative and `ᾱ` is taken as `ᾱ_0` (`set_alpha_to_one` is false in this configuration), so the update leaves the state as it is (`x = 0.3` stays 0.3; the estimate `x̂₀` would be 0.2972): the state is then already at `ᾱ_0 = 0.99915`, the almost clean end of the schedule.
The step is elementwise on all `F · h · w · 4` numbers, so the code of chapter 22 is reused for `F` frames.

**Why this design (23.5).** DDIM is the sampler of diffusers' AnimateDiff example (and the default here, 25 steps), deterministic; the `linear` schedule has to be set because the adapter was trained with it (with the base's scaled-linear table the noise at the top would be 14.6 instead of 25.1 times the picture). Euler, Euler-ancestral and DPM++ 2M are selectable but not what the adapter was shown during training;
the Karras variant is accepted as well. The cost of fidelity to diffusers is the quirk above, which is harmless but visible if you compare the timestep list with the state.

---

## 23.6 Decoding the frames

🧸 The painter works on small sketches; each sketch must be turned into a real picture. That is the slow part of one picture and it is
done sixteen times, but only once at the end.

📐 `x_final / 0.18215` goes through the VAE decoder of 22.3 **one frame at a time** (`video.decode-chunk` frames at once, default 1):
a 512 × 512 frame needs 1.2 s and about 2.5 TFLOP, the clip's 16 frames 19 s, whatever the number of steps. Decoding frames
independently is exactly what diffusers' `decode_latents` does (it loops over chunks of the flattened batch). The VAE has no temporal
layers, so a flicker that the temporal transformers did not remove stays in the decoded frames. Tiling (22.3) applies per frame above
1024 × 1024.

*Worked example.* 16 frames of 512 × 512: one VAE decode is 2.5 TFLOP and 1.2 s on this GPU (22.14), so the clip's decode is `16 × 1.2 = 19.3 s` (measured 19.3 s in the whole-clip run of 23.11), independent of the number of steps; in the 4-step run of 23.11 (89.7 s) that is a fifth of the total, in the 20-step run (371 s) 5 %. `video.decode-chunk = 1` decodes one frame per call (the lowest memory peak); more frames per call are "not faster on the GPU" (comment in `jllm.properties`).

**Why this design (23.6).** The image VAE has no temporal layers, so decoding frames one by one is the only way to use it, and it is exactly what diffusers does (`decode_latents`). The alternative, a video VAE, is a different model (Wan's, 23.12.3). The cost is visible: any difference between neighbouring latent frames that the temporal transformers left behind is decoded independently, so small flicker (colour or texture shimmer) survives.
Above 1024 × 1024 each frame is tiled by the VAE's tiling (22.3); tiles blend inside a frame, never across frames.

---

## 23.7 How jllm computes it

Three operations were added to the image stack's `Ops` (CPU `CpuOps`, GPU `MetalOps`):

| operation | what | kernel |
|---|---|---|
| `groupNorm(x, γ, β, groups, ε, silu, samples)` | group norm whose statistics are taken over `rows / samples` positions per item: `[B·F, H, W, C]` with `samples = B` normalises over frames too | the existing `df_groupnorm` / `df_gn_stats` / `df_gn_combine` / `df_gn_apply` with `b = samples` |
| `swapAxes(x, outer, d1, d2, shape)` | `[outer, d1, d2, C] → [outer, d2, d1, C]`: "frames of one pixel" ↔ "pixels of one frame" | `df_swap_axes` (new) |
| `addRows(x, table, period)` | `x[r] += table[r mod period]`: the temporal position table | `df_add_rows` (new) |

Everything else of the temporal transformer is the existing operations: `linear` (with fused bias, residual add and GEGLU's
`geglu`), `layerNorm`, `attention` (the flash kernels of widths 40, 80 and 160 at 16 × 16, batched over `B·H·W` sequences). Both new
kernels are one thread per output element; each has a numpy reference in `scripts/metal-emu/kernel_tests_diffusion.py`, run on the
emulator of [chapter 18](18-metal-emulator.md) in float32 and float16, and a CPU-against-GPU test (`VideoTest.opsParity`).

*Memory.* The activations of a 512 × 512, 16-frame clip with guidance are those of a batch of 32 images: a `[32, 64, 64, 320]` tensor
is 84 MB in float16, the GEGLU intermediate of the first temporal layer `[131 072, 2560]` is 671 MB. Measured peaks of the GPU buffers: 4.1 GB at 256 × 256 and 7.3 GB at 512 × 512 in float16 (8.1 GB at 256 × 256 in float32). `video.decode-chunk` (frames per VAE call, default 1) trades the VAE's activation memory against speed.

*Worked example, `swapAxes` and `addRows`.* `swapAxes(x, outer, d1, d2, shape)` turns `[outer, d1, d2, C]` into `[outer, d2, d1, C]`. For the temporal layer `outer = B = 2`, `d1 = F = 16`, `d2 = H·W = 4 096`: element `(b, f, p, c)` of the "frames of one clip" layout moves to `(b, p, f, c)`, i.e. the `16` frames of pixel `p` become 16 consecutive rows, a sequence. Its inverse is the same call with `d1` and `d2` exchanged.
`addRows(x, table, period)` does `x[r] += table[r mod period]`: with the `[32][C]` position table and `period = F = 16`, row `r` of the `[B·H·W·F]` rows is frame `r mod 16` and receives `PE[r mod 16]`. Both kernels are one thread per output element (`df_swap_axes`, `df_add_rows` in `diffusion.metal`), memory-bound.

**Why this design (23.7).** Three new operations and no new matrix kernel: the temporal transformer is made of operations the image stack already had (`linear` with fused bias and residual, `layerNorm`, `attention`, `geglu`), so the new code is only the regrouping and the clip-wide statistics, each with a numpy reference and a CPU-against-GPU test. The price is that the regrouping copies the whole activation twice per
temporal layer (`swapAxes` there and back). The +39-45 % time of the temporal layers (23.11) was measured as a whole and the share of the copies not separately; the arithmetic alone is +38 % (23.2).
**Memory (checked by the arithmetic above).** `[32, 64, 64, 320]` in float16 is `32 · 4 096 · 320 · 2 B = 83.9 MB`; the GEGLU intermediate of the first temporal layer has `2 · 4 096 · 16 = 131 072` rows of `8C = 2 560`: `131 072 · 2 560 · 2 B = 671 MB` (the measured peak of all GPU buffers is 7.3 GB at 512², float16).

---

## 23.8 Writing the file: GIF, WebP, MP4

A clip leaves the engine as frames; `VideoEncoder` turns them into a file. There is no library behind the first two.

🧸 **GIF** is an old picture format that can only use 256 colours in a picture, so we first choose the 256 colours that suit the whole
clip best, then colour every dot with the nearest one, scattering the small mistakes to the neighbours so that smooth skies do not
turn into stripes. **WebP (lossless)** keeps every dot exactly: it first writes each dot as "how different it is from what its
neighbours predict" (mostly nearly nothing), then writes short codes for the common differences and long ones for the rare ones.
**MP4** is what phones and websites use; it is much smaller because it throws away detail the eye does not miss, and it is made by the
`ffmpeg` program when you have it.

📐 **GIF.** One global palette from a median-cut of a 120 000-pixel sample of *all* the frames (so colours do not flicker from frame to
frame): repeatedly split the box of samples with the largest `range × √count` along its widest channel at the median, the palette entry
is the mean of the box. Each frame is quantised with serpentine Floyd–Steinberg error diffusion (the nearest palette entry is found
through a 15-bit cache), then compressed with GIF's variable-width LZW (9 to 12 bit codes, clear code when the 4096-entry table is
full); a Netscape extension makes it loop forever; delays are in centiseconds, rounded so that the sum is exact (`⌊(i+1)·100/fps⌉ − ⌊i·100/fps⌉`).

**WebP** (`Vp8l`, from RFC 9649): an animated file is `RIFF/WEBP` with a `VP8X` header (animation flag), `ANIM` (loop forever) and one
`ANMF` chunk per frame holding a `VP8L` (lossless) bitstream; durations in milliseconds, with the same rounding rule. A frame is coded as

```
1. subtract green:   R ← R − G, B ← B − G                                     (decorrelates the colour channels)
2. predictor transform: the picture is cut into 16 × 16 blocks; per block the one of the 14 predictors
   (L, T, TR, TL, averages of two or three neighbours, "Select", clamp(L+T−TL), ...) with the smallest sum of |residual| is used;
   residual = pixel − prediction (per channel, mod 256); the first row uses L, the first column T, the rightmost column takes TR from the row's first pixel
3. LZ77: a run of ≥ 4 equal residuals is a back reference to the pixel on the left (distance code 2) or above (distance code 1)
4. five canonical prefix (Huffman) codes: green + length symbols (280), red, blue, alpha, distance (40); lengths limited to 15 by flattening the
   counts; the code lengths are themselves run-length coded (symbols 16, 17, 18) with a second prefix code of at most 7 bits
```

The decoded frames are bit-identical to the input (`VideoTest` decodes with an independently written decoder, and a one-off check with
Pillow's libwebp and `ffmpeg` agreed on every pixel of a 6-frame 200 × 150 clip). Lossless is big: a real 512 × 512, 16-frame clip is 5.9 MB as WebP against 12.6 MB raw (2.0 MB as GIF, 0.83 MB as MP4), because the detail of a diffusion model is noise-like and does not compress losslessly. **MP4**: raw RGB
frames are piped into `ffmpeg -f rawvideo -pix_fmt rgb24 -i - -c:v libx264 -crf 17 -pix_fmt yuv420p -movflags +faststart` (`mpeg4` when
libx264 is missing; dimensions are padded to even); `video.ffmpeg` is `auto` (look it up on the PATH), `off` or a path. Without ffmpeg the
format `auto` is WebP, and asking for `mp4` is a clear 400.

*Worked example, frame delays.* A GIF stores each frame's delay in whole hundredths of a second, but 8 frames per second is 12.5 cs per frame. `delay(i) = round((i+1)·100/fps) − round(i·100/fps)` (`VideoEncoder.delay`) gives 13, 12, 13, 12, … : the *running total* stays exact (16 frames: 200 cs = 2.00 s), where a plain rounding of 12.5 would give 13 for every frame and a 4 % slower clip.
The same rule with `scale = 1000` gives WebP's millisecond durations (125 ms each at 8 fps). Browsers play GIF delays below 2 cs as 10 cs, so the encoder writes at least 2: above 50 frames per second a GIF plays slower than requested (the API allows `fps` up to 60).

*Worked example, WebP's predictor.* Take the green channel of one row, `10 12 15 15 16`. With predictor "L" (left neighbour) the residuals are `pixel − left`, modulo 256: the first pixel is predicted as 0 in the first row, so `10`, then `2, 3, 0, 1`: small numbers, which the prefix code writes in a few bits. A drop from 250 to 5 would be `(5 − 250) mod 256 = 11`, also small. "Subtract green" first rewrites a pixel `(R, G, B) = (120, 100, 90)` as `(R − G, G, B − G) = (20, 100, 246)` (−10 mod 256), so the two colour channels become small wherever the picture is grey-ish. Per 16 × 16 block the encoder picks, among 14 predictors, the one with the smallest sum of absolute residuals.
That is *lossless*: a diffusion picture is full of noise-like texture, so its residuals are not small and a 512 × 512 frame stays large (hence 5.9 MB for 16 frames against 12.6 MB raw, 23.8 above).

**Why these three formats.** GIF because every browser, chat and document viewer plays it (256 colours, large files); lossless WebP because it needs no external program and keeps the model's output exactly (large files, plays in browsers); MP4 for everything else (0.83 MB for the same 16 frames) but only through `ffmpeg`, which jllm does not ship. Writing GIF and WebP by hand costs about 1 100 lines (`VideoEncoder`, `Vp8l`) and a test per format (round trip with an independent decoder).
**Settings**: `video.ffmpeg` (`auto`, `off` or a path), `video.format` (`auto` = MP4 with ffmpeg, else WebP; `mp4`, `webp`, `gif`), the request's `output_format`, and `fps` (default 8, up to 60). **Limits**: the mp4 encoder is `libx264` at `-crf 17`, falling back to `mpeg4` without it; frame sizes are padded to even numbers; WebP has no inter-frame prediction, so a still scene costs as much as a moving one.

---

## 23.9 API, command line, console and chat tool

```sh
./jllm video --video-model guoyww/animatediff-motion-adapter-v1-5-2 --prompt "a red balloon floating over a green hill, clouds moving, photo" --out balloon.mp4
./jllm video --video-model guoyww/animatediff-motion-adapter-v1-5-2 --prompt "waves" --frames 16 --fps 8 --width 256 --height 256 --steps 20 --out waves.gif
./jllm serve --video-model guoyww/animatediff-motion-adapter-v1-5-2 --model none --port 8080        # video only; or beside a text model
./jllm-web/run.sh                                                                                    # Models: load the adapter; then the Video tab
```

The first run downloads the adapter (1.8 GB) and the Stable Diffusion 1.5 base (2.6 GB, fp16 files) into `models/`. Flags: `--frames`,
`--fps`, `--steps`, `--guidance`, `--width`, `--height`, `--seed`, `--scheduler`, `--negative`, `--out` (the extension picks the format),
`--base-model`, `--video-format`; all `video.*` keys are in [`jllm.properties`](../src/main/resources/jllm.properties), including the
switch `video.enabled` (false hides the endpoints, the console tab and the chat tool, and refuses to load video models).

**`POST /v1/videos/generations`**: OpenAI has no video-generation endpoint that fits a synchronous call (its asynchronous video jobs are
a different shape), so the request and the response follow `/v1/images/generations` (22.12), which any OpenAI client can be pointed at:

```sh
curl localhost:8080/v1/videos/generations -H 'Content-Type: application/json' -d '{
  "prompt": "a red balloon floating over a green hill, clouds moving, photo", "size": "512x512", "num_frames": 16, "fps": 8,
  "steps": 25, "guidance": 7.5, "seed": 42, "scheduler": "ddim", "output_format": "mp4", "response_format": "b64_json" }'
```

`size` is `WxH`, both multiples of 64 (32 for the tiny test model); `num_frames` (alias `frames`) 1 to 32; `fps` up to 60; `n` 1 to 4 clips
(clip k uses seed + k); `output_format` is `mp4`, `webp`, `gif` or `auto` (MP4 with ffmpeg, else WebP; default `video.format`);
`response_format` is `b64_json` or `url` (`/v1/videos/files/<id>.<ext>`, served without the API key: the id is the capability);
extras `negative_prompt`, `steps`, `guidance` (or `guidance_scale`), `seed`, `scheduler` (`ddim`, `euler`, `euler_a`, `dpmpp_2m`,
`dpmpp_2m_karras`), `clip_skip`, and `include_frames` (`jpeg` or `png`: every frame as base64 in `frame_images`, for editors).
Response: `{created, model, data: [{b64_json | url, format, mime_type, width, height, frames, fps, duration, bytes, seed, revised_prompt,
frame_images?}], usage: {videos, frames, steps}, timing: {seconds, text_s, unet_s, vae_s, encode_s}}`. **Streaming** (`"stream": true`):
`video_generation.progress` `{video, step, steps}`, `video_generation.partial_image` (a small PNG preview of the first frame from the
latents, free), `video_generation.decoding` `{frame, frames}`, `video_generation.completed` (the clip), `video_generation.done`,
`[DONE]`; closing the connection cancels at the next step.

**`POST /v1/videos/encode`**: `{frames: [base64 or data-URL pictures of one size], fps, output_format, response_format}` writes
pictures as a video file (needs no model; the console's export of an edited clip). Both servers, the engine's own (`ApiServer` +
`VideoHttp`) and the web console (`VideoGenerationController`), serve the same routes through `VideoApi`.

**Console.** *Models* loads a motion adapter like any model (it downloads the base too; the load form has the video backend, precision and
base model; the list shows a *video* badge). The **Video** tab: *Create* (prompt, negative prompt, size, frames, fps, steps, guidance,
scheduler, file format, seed; a progress bar per step with a live preview of the first frame, then per decoded frame), the **clip editor**
(a canvas playing the frames at the chosen speed, a scrubber, the *frame strip*, playback loop / ping-pong / once, and the edits:
*trim start here*, *trim end here*, *delete frame*, *hold frame*, *reverse*, *ping-pong*, *reset*, a speed slider; *Export* as MP4 / WebP
/ GIF through `/v1/videos/encode`, *download original*, *save edits as a new clip*), and the *gallery* (IndexedDB in the browser: the
frames as JPEGs, the first frame as thumbnail, the settings, and the file as the server wrote it when it is under 12 MB). Edits work on the
frame list in the browser; only exporting asks the server. **Chat tool** `generate_video` (the chat's tools menu; it waits up to 30 minutes,
whatever the general tool timeout is): the model writes the prompt, the clip appears inline in the conversation (`<video>` for MP4, `<img>`
for GIF / WebP).

**Limits of the API** (`VideoApi`, `VideoPipeline.validate`): `n` 1 to 4 clips (one after the other), `size` a multiple of 64 (32 for the tiny test model) with an area of at most 1024 × 1024 pixels, `num_frames` 1 to the adapter's `motion_max_seq_length` (32), `steps` 1 to 200, `guidance` 0 to 50, `fps` above 0 and at most 60; `/v1/videos/encode` needs pictures of one size.
**Why this API shape.** OpenAI has no synchronous video endpoint, so the request and response copy `/v1/images/generations`, which clients already know; the streaming events (`video_generation.progress`, `.partial_image`, `.decoding`, `.completed`) mirror the image ones plus one for the decoding of frames, because decoding is the second long phase (23.6). Closing the connection cancels at the next step, which is the only way to stop a job that takes minutes.

### 23.9.1 Settings

All keys are in `jllm.properties` (validated at start-up, editable in the console's Settings tab, accepted on the command line as `--video.frames 8`). `0`, `-1` and empty mean "the model's own default" (AnimateDiff: 25 steps, guidance 7.5, 512 × 512, DDIM).

| key / switch | default | what it does, and when to change it |
|---|---|---|
| `video.enabled` | `true` | `false` hides the endpoints, the console tab and the chat tool and refuses to load video models |
| `video.model` | empty | the motion adapter `./jllm serve` loads at start (`guoyww/animatediff-motion-adapter-v1-5-2` or a folder); `--model none` serves only video |
| `video.base-model` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | the SD 1.x model (diffusers layout) the adapter rides on; any SD 1.5 fine-tune; change it to change the style (23.13) |
| `video.backend`, `video.precision` | `auto`, `auto` | `metal` / `cpu`; `auto` precision is float16 on the GPU for AnimateDiff (float32 for the Wan command, 23.12); `f32` gives the numbers of 23.10 for twice the memory and 1.5 times the time |
| `video.decode-chunk` | `1` | frames per VAE call; 1 keeps the memory peak lowest |
| `video.ffmpeg`, `video.format` | `auto`, `auto` | MP4 needs `ffmpeg` (`auto` looks on the PATH; `off`, or a path); `auto` format is MP4 with ffmpeg, else lossless WebP; `gif` for universal playback |
| `video.frames`, `video.fps` | `16`, `8` | clip length (at most 32 with the v1.5 adapters; the adapter was trained on 16) and playback speed; **for Wan the frame count is rounded down to 4k + 1, so 16 becomes 13**, and the default fps is 16 |
| `video.steps`, `.guidance`, `.width`, `.height`, `.seed`, `.scheduler`, `.negative`, `.out` | `0`, `-1`, `0`, `0`, `-1`, empty, empty, empty | the `jllm video` command; the `image.*` names of the same meaning (`image.steps` ...) count when given |
| `-Djllm.video.nomotion=true` | off | benchmark only: runs the frames as independent images to time the motion modules (the clip is garbage) |
| `-Djllm.diffusion.flash=false`, `-Djllm.diffusion.simpleNorm=true` | on, off | the image-stack switches of 22.12.1 (`mac-validate.sh` has a video row for each) |

### 23.9.2 The clip editor

🧸 The generated clip arrives in the browser as a strip of small pictures (the frames), like a film roll. The editor lets you cut the roll, flip it, copy a frame so it stays on screen longer, or run it forwards and then backwards so that it loops without a jump. Nothing is redrawn: you only change which frames play, and in which order; the computer is asked for a video file only when you press *Export*.

📐 *Data model* (`video.js`): the clip is the list of frames as pictures (`include_frames: jpeg`, quality 0.92 from the server) and an `order` array of indices into it. Every edit makes a new `order`:

```
frames 0 1 2 3 4        order = [0 1 2 3 4]      selected frame i = 2
 Trim start here        order.slice(i)                       → [2 3 4]
 Trim end here          order.slice(0, i + 1)                → [0 1 2]
 Delete frame           order without position i             → [0 1 3 4]      (never the last remaining frame)
 Hold frame             the frame is repeated after itself   → [0 1 2 2 3 4]    (shown twice as long)
 Reverse                order reversed                       → [4 3 2 1 0]
 Ping-pong              order + order[1 … n−2] reversed      → [0 1 2 3 4 3 2 1]  (2n − 2 frames: neither end frame is doubled, so looping 0…4,3,2,1,0… has no stall)
 Speed                  a slider of 1 to 30 frames per second (the clip's fps)
```

*Export* sends the frames in `order` as base64 pictures to `POST /v1/videos/encode` with the chosen `fps` and format; the server needs no model for it and returns the file. *Download original* gives the file as the generation wrote it (a gallery entry keeps it when it is under 12 MB), *save edits as a new clip* stores the new frame list in the browser's IndexedDB (JPEG frames, a JPEG thumbnail of the first frame at quality 0.85, the settings).

**Why this design.** Cutting is only reordering, so it is instant and lossless until export, and the server stays stateless. The cost is that an exported edit is the *JPEG* frames re-encoded (one more lossy generation than the server's own MP4; "download original" avoids it), that every frame lasts the same time (a hold is a duplicate frame, a speed change affects the whole clip) and that nothing is blended between frames.
**Settings**: none of its own (it needs a loaded video model for *Create*; editing a gallery clip works without one, but export needs a server). **Checks and limits**: the server side (`/v1/videos/*`) is covered by `VideoTest` and the console's video route tests; the editor's JavaScript has no automated test. No audio, no transitions, no per-frame timing.

---

## 23.10 Verified against diffusers

Two kinds of check, both against **Hugging Face diffusers** (`AnimateDiffPipeline`) given the same weights, the same noise (the
initial noise is made by torch and handed to jllm: the two do not share a random generator) and the same prompt.

**1. In `mvn test` (no download): a tiny random AnimateDiff.** `scripts/diffusion/make-tiny-video.py` builds a small random Stable Diffusion
pipeline and a motion adapter with the same module layout as the real one (`src/test/resources/diffusion/tiny-motion-adapter`);
`reference_video.py` runs diffusers on it and stores the text embeddings, one UNet output, the latents after every step and the frames
(`ref-video-ddim`, `-euler`, `-dpmpp_2m`). `VideoTest` compares the stages and the final frames; frames PSNR against diffusers (the limits
for the intermediate stages are in the test):

| scheduler | CPU f32 | Metal f32 | Metal f16 |
|---|---|---|---|
| DDIM | 131.2 dB | 130.7 dB | 65.8 dB |
| Euler | 125.6 dB | 125.9 dB | 60.4 dB |
| DPM++ 2M | 130.0 dB | 129.1 dB | 62.6 dB |

The file writers are checked by round trip: animated WebP with `Vp8lDecoder`, a decoder written separately from the encoder, bit-exact for
4-frame clips of several sizes down to 1 × 1, smooth and noisy (and once, by hand, against libwebp and `ffmpeg`); GIF by decoding it again (worst
frame 26.6 dB after the 256-colour palette, which is what a palette costs), including a noisy clip that forces LZW table resets; MP4 by the `ftyp`
box when `ffmpeg` is installed. The rest of `VideoTest`: DDIM timesteps against numpy's, `addNoise` then `predictOriginal` returns the clean latents,
the CPU and the GPU implementation of the new operations (`swapAxes`, `addRows`, the group norm over a whole clip) against each other, the
engine's HTTP server (`n=2`, streaming progress and previews, bad sizes, MP4 without ffmpeg), and the download of a motion adapter from a fake Hub
(only `config.json` and the fp16 weights, not downloaded twice).

**2. The real model, run by hand.** `guoyww/animatediff-motion-adapter-v1-5-2` on `stable-diffusion-v1-5` (fp16 files, widened to float32),
diffusers on the CPU in float32 as the reference (`--device cpu --dtype float32`: a float32 CPU run is the reference, not the MPS GPU), jllm on Metal. 16 frames, guidance 7.5, `--dump-steps` stores the latents after every step:

```sh
python scripts/diffusion/reference_video.py models/stable-diffusion-v1-5_stable-diffusion-v1-5 models/guoyww_animatediff-motion-adapter-v1-5-2 acc/r256 \
    --variant fp16 --adapter-variant fp16 --device cpu --dtype float32 --width 256 --height 256 --frames 16 --steps 4 --guidance 7.5 --seed 42 --scheduler ddim --dump-steps
java ... org.robomiri.jllm.diffusion.VideoCompare models/stable-diffusion-v1-5_stable-diffusion-v1-5 models/guoyww_animatediff-motion-adapter-v1-5-2 acc/r256 metal f32
```

| run | jllm | UNet, first step (relative max) | latents after the last step (relative max) | frames: PSNR (worst frame) |
|---|---|---|---|---|
| 64 × 64, 4 frames, DDIM 3 steps | CPU f32 | 1.8·10⁻⁶ | | 94.7 dB |
| 256 × 256, 16 frames, DDIM 4 steps | Metal f32 | 6.2·10⁻⁶ | 2.1·10⁻⁴ | 85.3 dB (81.8) |
| 256 × 256, 16 frames, DDIM 4 steps | Metal f16 | 5.4·10⁻⁴ | 2.5·10⁻² | 45.1 dB (41.3) |
| 256 × 256, 16 frames, Euler 3 steps, seed 7 | Metal f32 | 7.4·10⁻⁶ | 8.0·10⁻⁴ | 78.1 dB (69.7) |
| 256 × 256, 16 frames, Euler 3 steps, seed 7 | Metal f16 | 7.3·10⁻⁴ | 3.7·10⁻² | 39.4 dB (31.2) |
| 512 × 512, 16 frames, DDIM 2 steps | Metal f32 | 3.1·10⁻⁶ | 2.1·10⁻⁴ | 87.5 dB (86.9) |
| 512 × 512, 16 frames, DDIM 2 steps | Metal f16 | 6.4·10⁻⁴ | 5.7·10⁻² | 39.5 dB (39.0) |
| 192 × 192, 8 frames, DDIM **20** steps (the whole loop; the latents after each of the 20 steps are compared, the largest 1.2·10⁻⁴) | Metal f32 | 6.5·10⁻⁶ | 1.2·10⁻⁴ | 93.2 dB (88.4) |
| same | Metal f16 | 5.3·10⁻⁴ | 1.3·10⁻² | 49.4 dB (42.2) |

A float32 clip is the same picture as diffusers' to well under one grey level in the worst pixel (85 dB). Float16 clips agree to 39–45 dB:
the same clip to the eye, but the latents differ by a few per cent after the last step. The default precision on Metal is float16;
`video.precision=f32` gives the first rows for twice the memory and 1.5 times the time. Memory of the reference runs, for scale: diffusers on the
CPU in float32 needs 15–17 GB of resident memory for these 16-frame clips.


### 23.10.1 What the numbers mean, and the limits of the checks

* *PSNR* compares two 8-bit pictures: `20 · log10(255 / RMS difference)`. **85 dB** is an RMS difference of `255 · 10^(−85/20) = 0.014` grey levels, **45 dB** of 1.4 levels, **39 dB** of 2.8 levels. The float32 rows are therefore the same clip as diffusers', the float16 rows the same clip to the eye.
* *The relative max error of the latents* grows with the steps (2.1·10⁻⁴ after 4 float32 steps, 1.2·10⁻⁴ after 20 at a lower resolution): the sampler feeds the network's rounding back in. Float16 gives 5·10⁻⁴ in one UNet call and 2.5·10⁻² in the latents after 4 steps: the rounding of half-precision weights and activations accumulates over the steps (the clip is the same to the eye, 23.10).
* *What is compared, and what is not.* The tiny random model checks every layer type and the whole loop against diffusers at 10⁻⁵ (float32); the real model is checked by hand at 64 × 64, 192 × 192, 256 × 256 and 512 × 512, 2-20 steps, DDIM and Euler; **not** compared: DPM++ with the real model, clips longer than 16 frames, other base models than Stable Diffusion 1.5, and anything about the *look* of the clips beyond "equal to diffusers'" (23.13).
* *The writers* are checked by round trip, not by looking: lossless WebP bit-exact against an independent decoder, GIF to 26.6 dB (the cost of 256 colours), MP4 only by its header when ffmpeg is installed.

---

## 23.11 Measured speed and memory

Measured on the development machine (Apple M1 Pro, 32 GB shared with everything else), one benchmark at a time under the machine's
lock, `VideoBench` for jllm and `scripts/diffusion/bench_diffusers_video.py` for diffusers on the same GPU (MPS, `torch` as installed in
the reference environment). 16 frames with guidance, so every UNet evaluation is a batch of **32 images**; DDIM, guidance 7.5, the motion
adapter and base model as above. "Warm" is the second clip of the same process (shader compilation and weight upload are in the first).

```sh
java ... org.robomiri.jllm.diffusion.VideoBench BASE ADAPTER metal f16 10 256 256 16 2 7.5 ddim        # backend precision steps w h frames reps guidance scheduler
python scripts/diffusion/bench_diffusers_video.py BASE ADAPTER --device mps --dtype float16 --steps 10 --size 256 --frames 16 --reps 2
```

| 16 frames, guidance 7.5 | | UNet, s per step | VAE, s per frame | clip, s | GPU memory |
|---|---|---|---|---|---|
| 256 × 256, float16, 10 steps | **jllm Metal** | 3.88 | 0.30 | 43.7 | 3.0 GB buffers (peak 4.1 GB) |
| | diffusers MPS | 2.58 | 0.15 | 28.2 | 13.1 GB driver |
| 256 × 256, float32, 4 steps | **jllm Metal** | 5.66 | 0.42 | 29.5 | 5.9 GB buffers (peak 8.1 GB) |
| | diffusers MPS | 3.07 | 0.16 | 14.9 | 21.3 GB driver |
| 512 × 512, float16, 4 steps | **jllm Metal** | 17.6 | 1.21 | 89.7 | 3.0 GB buffers (peak 7.3 GB) |
| | diffusers MPS | 242 | 1.3 | 990 | 23.9 GB driver; the machine swapped (10 GB of swap in use) |

Read honestly: **at 256 × 256 diffusers on MPS is faster** (1.5 times per UNet step in float16, 1.8 in float32, 2 to 2.6 times in the VAE): PyTorch's
Metal kernels are better tuned than the ones written here for these shapes. **At 512 × 512 jllm is 14 times faster per step**, because it never
materialises the attention scores (flash attention, [chapter 22](22-image-generation.md)): diffusers' scores of the first level are
`32 images × 8 heads × 4096 × 4096` numbers (8.6 GB in float16), the driver asked for 23.9 GB and the Mac paged. Peak resident memory of the whole jllm process
(weights memory-mapped, widened to the activation type, plus buffers) was 8.3 GB in float16 at 512 × 512 and 11.5 GB in float32 at 256 × 256.

**What the motion modules cost** (`-Djllm.video.nomotion=true` runs the same batch of 32 frames as independent images; the clip is then
meaningless, only the time counts): 256 × 256 float16 2.66 s per step without, 3.87 s with (+45 %); 512 × 512 12.68 s without, 17.60 s with
(+39 %). The 21 temporal transformers are about 30 % of an AnimateDiff UNet evaluation.

**A whole real clip** (`./jllm video --video-model ... --width 512 --height 512 --frames 16 --fps 8 --steps 20 --seed 3 --video.precision f16`):
371 s = UNet 351 s (17.6 s per step) + VAE 19.3 s; the file as MP4 (libx264 crf 17) 0.83 MB, as animated lossless WebP 5.9 MB, as GIF 2.0 MB; the raw
frames are 12.6 MB. Time is linear in the number of frames and in steps, and roughly 4.5 times more for a doubled side.

**What the pictures look like.** Frame for frame these clips equal diffusers' (section above), and that includes how they look: with the plain
Stable Diffusion 1.5 as the base, the `v1-5-2` adapter gives poor clips (flat or smeared colours that change from frame to frame at 192 to 512
pixels), the same ones diffusers makes (a 20-step 192 × 192 clip from the same noise equals diffusers' on the CPU at 93.2 dB in float32, and shows the same colour smears). A single frame
without the adapter (`--frames 1`) is a normal Stable Diffusion picture. The adapter is meant for a *fine-tuned* Stable Diffusion 1.5
checkpoint (`video.base-model`, a diffusers folder or repository with `model_index.json`); none was available within the download limits of this
work, so no pretty clip was made.


### 23.11.1 What the numbers mean

* *A 32-image batch.* One UNet step of a 16-frame clip with guidance is the work of 32 pictures plus the temporal layers: at 512² that is `32 · 0.80 = 25.6 TFLOP + 9.7 TFLOP (23.2) = 35 TFLOP`; measured 17.6 s per step, i.e. 2.0 TFLOP/s, the same rate the image UNet reaches (22.14: 1.9 TFLOP/s). At 256² the work is a quarter, the time per step is 3.88 s, i.e. 2.3 TFLOP/s on `8.8 TFLOP`. So the kernels run at about their usual
speed in a clip (the 256² figure assumes a quarter of the image UNet's 0.80 TFLOP, approximately); what a clip costs is its size.
* *Time is linear in frames and steps, 4.5 times more for a doubled side:* 256² → 512² has 4 times the pixels, a little more for the attention: 3.88 → 17.6 s per step (4.5×).
* *Why jllm is 14 times faster than diffusers at 512² and 1.5 times slower at 256².* At 256² everything fits: PyTorch's tuned kernels win (its UNet step is 1.5 times faster in float16 and 1.8 times in float32, its VAE 2-2.6 times). At 512² diffusers materialises the attention scores of the first level: `32 images × 8 heads × 4 096 × 4 096` numbers is 8.6 GB in float16 (`32 · 8 · 16 777 216 · 2 B`), the driver asks for 23.9 GB on a 32 GB Mac that is also running the rest, and the machine swaps (10 GB of swap used); the flash kernel never builds that matrix. The comparison is therefore a statement about *memory* at 512², not about arithmetic.
* *Memory.* The weights are 0.9 GB (adapter, float16) + about 2.1 GB (the Stable Diffusion networks in float16, 22.2) = the "3.0 GB buffers" column; the rest of the peak (7.3 GB at 512²) is activations of the batch of 32: the 671 MB GEGLU buffer and its neighbours.
* *Where a real clip's 371 s go:* 20 steps × 17.6 s = 351 s in the UNet, 19.3 s in the VAE; the text encoder takes about 0.05 s (22.14).

---

## 23.12 Wan 2.1: how far it got and what remains

**Wan 2.1** (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`) is a different kind of video model: not a UNet with temporal layers but a *diffusion
transformer* over a 3-D latent, a *causal 3-D VAE* that squeezes 4 frames into 1 latent frame and 8 × 8 pixels into 1, and a large text
encoder (UMT5-XXL). Every component of it is written here and checked against diffusers; **what is missing is the text encoder's weights**, so no
real clip could be made end to end.

| component | what it does | notes |
|---|---|---|
| `WanTransformer` | 30 blocks of width 1536 (12 heads of 128): 2 × 2 patches of a frame as tokens, time embedding → 6 modulation vectors per block (shift, scale, gate for attention and for the feed-forward, plus a learned table), self-attention with RMS-normed queries and keys over all heads and 3-D rotary positions (44 channels for the frame axis, 42 + 42 for height and width), cross-attention to the text, tanh-GELU feed-forward | predicts the velocity `v = noise − clean` of a flow-matching model |
| `WanVae` | causal 3-D convolutions (a 3-tap kernel over time that only looks back, done as three shifted 2-D convolutions that are added), temporal up-sampling by a convolution that doubles the frames, the first frame decoded alone | latent `[F, h, w, 16]` → `1 + 4(F − 1)` frames of `8h × 8w` |
| `Umt5Encoder`, `UnigramTokenizer` | 24 layers of width 4096, relative position bias per layer, no attention scaling; SentencePiece **Unigram** tokenizer read from `tokenizer.json` (space collapsing, metaspace, Viterbi segmentation, fused unknowns) | on the CPU; the linear layers are quantized to int8 (Q8_0 layout, groups of 32) while they are read |
| `FlowScheduler` | flow-matching sigmas with shift 3, UniPC multistep predictor-corrector of order 2 (and plain Euler) | 30 steps by default, guidance 5 |
| `WanPipeline` | tokenize → UMT5 (read, used, freed per request) → transformer twice per step (guidance) → UniPC → VAE | `jllm video --video-model <local Wan folder>` |

New operations on the image stack (`Ops`, CPU and Metal): RMS norm, rotary embedding, adaptive modulation `x (1 + scale) + shift`, gated
residual, frame shift for the causal convolution, row gather and stack, the tanh GELU; each GPU kernel has a numpy reference in
`kernel_tests_diffusion.py` (float32 and float16, on the emulator) and a CPU-against-GPU test.

**Accuracy against diffusers / transformers**

| check | result |
|---|---|
| tokenizer, 8 prompts (accents, Japanese, emoji, runs of spaces, empty), tiny vocabulary and the real 256 000-piece `umt5-xxl` vocabulary | the same ids as Hugging Face `tokenizers` |
| UMT5 encoder, tiny random weights | float 3.7·10⁻⁷ relative; int8 weights 1.2·10⁻² (a random 2-layer model, the worst case for quantization) |
| transformer, tiny random weights | 3·10⁻⁷ (CPU and Metal float32), 7·10⁻⁴ (Metal float16) |
| **transformer, the real 1.3B weights**, latent 5 × 24 × 40, 64 random text vectors | CPU float32 **2.9·10⁻⁶**, Metal float32 **3.3·10⁻⁶**, Metal float16 5.3·10⁻³ (relative max error) |
| VAE decoder, tiny random weights | 3.5·10⁻⁶ (132 dB), float16 69 dB |
| **VAE decoder, the real weights**, latent 5 × 12 × 20 → 17 frames of 96 × 160 | Metal float32 **1.1·10⁻⁵ (128.3 dB)**, Metal float16 77.0 dB |
| flow sigmas and timesteps, 5 steps | 999, 923, 818, 667, 429 as diffusers |
| the whole tiny pipeline (UMT5 → transformer → UniPC → VAE) from the same noise, guidance 5 and 1 | 130 to 135 dB in float32 (CPU and Metal), 70 dB in float16 |

All of this is in `mvn test` (`WanTest`, tiny fixtures `src/test/resources/diffusion/tiny-wan`, `ref-wan-*`, made by `scripts/diffusion/make-tiny-wan.py`
and `reference_wan.py`) except the two rows on the real weights (`WanCompare dit|vae`, by hand). The real-weights run found a bug the tiny test could not:
the Metal tanh-GELU gave NaN for pre-activations in the hundreds (the real text projection has them; `tanh` of a huge argument is inf / inf
under fast math), fixed by clamping the argument, with a test of its own (`WanTest.geluTanhHuge` and a kernel test).

**Speed** (`WanBench`, random inputs; transformer evaluation = one of the two per step with guidance):

| latent (frames × h × w) = tokens, video | Metal float32 | Metal float16 | CPU float32 |
|---|---|---|---|
| 5 × 24 × 40 = 1 200 tokens, 17 frames of 320 × 192: transformer | 2.05 s | 1.43 s | 17.0 s |
| same: VAE decode | 6.4 s | 4.9 s | 39.5 s |
| 9 × 30 × 52 = 3 510 tokens, 33 frames of 416 × 240: transformer | | 6.3 s | |
| same: VAE decode | | 15 to 33 s | |

Flash attention matters here as well (float32, 1 200 tokens: 2.05 s with it, 5.3 s with `-Djllm.diffusion.flash=false`). Memory: the weights are 6.2 GB
on disk in float32 (transformer 5.7 GB, VAE 0.5 GB), 3.1 GB in float16; GPU buffers 10.0 GB (float32) and 5.0 GB (float16) for the 17-frame case.
A 17-frame 320 × 192 clip with 30 steps would be about 30 × 2 × 1.43 s + 5 s ≈ 90 s of GPU time in float16 plus the text encoder.

**What is missing, and why.** The text encoder of that repository is stored only as **float32 (22.7 GB, five shards)**; there is no float16 or
bfloat16 file. The download budget of this work was 20 GB in total for all models, so the encoder was not fetched, and a real prompt cannot be
run. Everything downstream was checked with random text vectors of the right shape (the transformer does not care where they come from), and the
encoder itself on the tiny model and by its tokenizer on the real vocabulary. The design for the real encoder: read tensor by tensor, never keep
the float32 file resident, quantize each linear layer to int8 as it is read (4.6 GB for the 24 layers; the 256 000 × 4096 embedding table is
read row by row from the memory-mapped file), run on the CPU for the few hundred tokens of a prompt, free it before the transformer is used
(`WanPipeline` does exactly this per request). Its numerical error on the real weights has not been measured: the tiny model's 1.2·10⁻² says
only that int8 costs about a percent. Not done either: the Wan pipeline is reachable from `jllm video` with a local folder only (not from the API
servers, the console or the chat tool, which stay AnimateDiff); image-to-video, the 14B models, the VAE encoder, float16 end-to-end checks on real
weights beyond the rows above, and a streaming preview of intermediate steps.

---

The five parts of Wan 2.1 are explained below in the order the data flows through them: the text (23.12.4), the diffusion transformer (23.12.1), flow matching and its sampler (23.12.2), the causal 3-D VAE (23.12.3) and the pipeline that joins them (23.12.5).

### 23.12.1 The diffusion transformer (`WanTransformer`)

🧸 The UNet of chapters 22 and 23 is a painter who looks at the picture through windows of different sizes. A *diffusion transformer* cuts the whole video into small bricks and lets **every brick look at every other brick, in every frame**, at every layer: a brick of the balloon in frame 3 can see the balloon's brick in frame 15 directly. Two dials on each layer tell it how noisy the video still is and
(through the text) what it should become; the layer turns the dials as "make this brick brighter, shift it, let only a fraction of the layer's answer through".

📐 *Tokens.* The latent `[F', h, w, 16]` is cut into patches of `1 × 2 × 2` (one frame, 2 × 2 latent pixels): each patch is `4 · 16 = 64` numbers, a linear layer makes it a token of width `dim = 12 heads · 128 = 1536`. For the 17-frame clip of 320 × 192: `F' = 5`, `h = 24`, `w = 40` → `5 · 12 · 20 = 1 200` tokens (33 frames of 416 × 240: `9 · 15 · 26 = 3 510`; 81 frames of 832 × 480: `21 · 30 · 52 = 32 760`).
*The dials.* The timestep `t` becomes a sinusoidal vector (256 wide), then an MLP gives `temb`; `Linear(SiLU(temb))` gives six vectors of width `dim`, to which each block adds its own learned table `scale_shift_table [6, dim]`: `(shift₁, scale₁, gate₁, shift₂, scale₂, gate₂)`. A block is

```
x ← x + gate₁ ⊙ SelfAttention( LN(x) ⊙ (1 + scale₁) + shift₁ )        3-D rotary positions, RMS-normed q and k over all 1 536 channels
x ← x +         CrossAttention( LN_affine(x), text )                    text [512][1536]; q and k RMS-normed
x ← x + gate₂ ⊙ FeedForward( LN(x) ⊙ (1 + scale₂) + shift₂ )            Linear, GELU(tanh), Linear
out = proj_out( LN(x) ⊙ (1 + scale) + shift ) → unpatchify → v  [F', h, w, 16]
```

Toy numbers for the dial: a normalised row `[−1, 0, 1]` with `scale = 0.5`, `shift = 0.1` becomes `[−1.4, 0.1, 1.6]`; with `gate = 0` the block's answer is not added at all.
*3-D rotary positions.* A head has 128 channels = 64 pairs, rotated like RoPE in the language models (chapter 5), but each pair belongs to one axis: 22 pairs (44 channels) rotate with the **frame** index, 21 pairs (42 channels) with the **row** and 21 (42) with the **column** of the token; for an axis of `d` channels pair `i` turns by `position / 10000^{2i/d}`.
Token (frame 2, row 5, column 7): the first pair of each axis turns by 2, 5 and 7 radians, the second by `2 / 10000^{2/44} = 1.316`, `5 / 10000^{2/42} = 3.225`, `7 / 10000^{2/42} = 4.515`. Two tokens attend to each other with a score that depends on their *offset* in each axis, as a language model's does on distance.
*Cost.* About `2 · 1.4 B parameters · tokens` for the linear layers plus `4 · 30 layers · tokens² · 1 536` for attention (the parameter count is 5.7 GB of float32 weights ÷ 4): at 1 200 tokens `3.42 + 0.27 = 3.7 TFLOP` per transformer call, measured 1.43 s in float16, i.e. 2.6 TFLOP/s, the rate of the image kernels (22.14); at 3 510 tokens `10.0 + 2.3 = 12.3 TFLOP`, measured 6.3 s (1.95 TFLOP/s). Attention is 7 % of the work at 1 200 tokens and 18 % at 3 510; at 32 760 tokens
(81 frames, 832 × 480) it would be twice the linear layers, and its scores would be 51.5 GB *per layer* in float32 if materialised: only the flash kernels (`df_flash_d128`) make that size thinkable.

**Why this design.** The checkpoint dictates the architecture; what the implementation chooses is to express it with the operations of the image stack (`linear`, `layerNorm`, `attention`, `geglu`) plus five new ones with their own numpy tests: RMS norm, rotary embedding, adaptive modulation `x (1 + scale) + shift`, gated residual and the tanh-GELU. A Wan guidance step is **two transformer calls** (positive and negative prompt), not one call with a batch of two. The price of global attention is the quadratic term above; its gain is that motion needs no special temporal layers.
**Settings.** The Wan pipeline has no switch of its own beyond the shared `video.*` and `image.*` names: precision `video.precision` (the Wan command uses float32 unless you say `f16`: 2.05 s against 1.43 s per call at 1 200 tokens, 10.0 GB of buffers against 5.0 GB), `video.steps` (default 30), `video.guidance` (5), `video.scheduler` (`unipc` or `euler`), size (default 480 × 272, multiples of 16), `video.frames` (rounded down to 4k + 1), `video.fps` (16).
**Checks and limits.** Tiny random weights 3·10⁻⁷ (CPU and Metal float32), 7·10⁻⁴ (float16); the real 1.3B weights 2.9·10⁻⁶ (CPU float32) and 3.3·10⁻⁶ (Metal float32), 5.3·10⁻³ (float16), against diffusers on a latent of 5 × 24 × 40 with 64 random text vectors (table above). Not done: image-to-video, the 14B models, and the I2V variant's extra image encoder (a configuration with `image_dim` is refused).

### 23.12.2 Flow matching and the UniPC sampler (`FlowScheduler`)

🧸 The earlier models learn to name the noise in a picture. A flow-matching model learns something simpler to picture: *the direction to walk*. If you draw a straight line from a heap of noise to the finished video, the model says at each point which way the line goes. A perfect model could jump to the end in one step, because the line is straight; a real model's direction changes a little along the way, so we take
several steps and correct as we go (UniPC does the correcting cleverly: it looks at the last direction and checks its own last step).

📐 *The path.* `x_σ = (1 − σ) x₀ + σ ε`, `σ` from 1 (pure noise `ε`) to 0 (the video `x₀`). The derivative along the path is constant: **`v = ε − x₀`**, which is what the transformer predicts. Given `x` and `v` at `σ`, the clean estimate is `x̂₀ = x − σ v` (check: `(1−σ)x₀ + σε − σ(ε − x₀) = x₀`). Euler: `x ← x + (σ_next − σ) v`.
*Worked example.* `x₀ = 0.5`, `ε = 0.7`, so `v = 0.2` exactly. At `σ = 1`, `x = 0.7`; with `σ_next = 0.9232`: `x = 0.7 + (0.9232 − 1) · 0.2 = 0.6846`, which is `(1 − 0.9232) · 0.5 + 0.9232 · 0.7`, on the path exactly. With a perfect `v` every sampler is exact in any number of steps.
*The schedule* (`setTimesteps`): `s_i = linspace(1, 1/1000, n + 1)[i]` for `i < n`, then the **shift** `σ_i = a s_i / (1 + (a − 1) s_i)` with `a = flow_shift = 3`; the network is given the integer `⌊1000 σ_i⌋`. For `n = 5`: `s = 1, 0.8002, 0.6004, 0.4006, 0.2008` becomes **`σ = 1 (0.999999), 0.9232, 0.8184, 0.6672, 0.4298`, then 0**, timesteps **999, 923, 818, 667, 429** (as diffusers); for `n = 30` the first are 999, 988, 976, 964, 951, 937. The shift moves the steps towards the noisy end: without it the fifth step would be at `σ = 0.20`, with it at 0.43;
the first steps decide the layout of the video.
*UniPC* (`bh2`, order 2, data prediction): with `α = 1 − σ`, `λ = ln α − ln σ`, `m_i = x_i − σ_i v_i` (the clean estimate), `h = λ_{i+1} − λ_i`:

```
predictor, first order:   x_{i+1} = (σ_{i+1}/σ_i) x_i − α_{i+1} (e^{−h} − 1) m_i                       the same step DDIM / DPM++(1) makes
predictor, second order:  … − ½ α_{i+1} (e^{−h} − 1) D₁ ,   D₁ = (m_{i−1} − m_i) / r ,   r = (λ_{i−1} − λ_i) / h        uses the previous estimate's trend
corrector (from step 1):  re-computes the sample the previous predictor made, with the new estimate m_i in place of its guess:
                          x_i^c = (σ_i/σ_{i−1}) x_{i−1} − α_i (e^{−h'} − 1) m_{i−1} − α_i (e^{−h'} − 1) (ρ₀ D₁' + ρ₁ (m_i − m_{i−1})) ,  h' = λ_i − λ_{i−1}
orders: 1 at the first and the last step (lower_order_final), 2 between; ρ₁ = ½ when the order is 1; for order 2 ρ₀, ρ₁ solve a 2 × 2 system
```

Everything is in double precision as in diffusers; the sigmas are float32 values as diffusers keeps them. The corrector spends no extra network call: it reuses `m_i`, which the next step needs anyway (that is the design of UniPC; this repository has no accuracy comparison between UniPC and Euler on a real model).
**Why this design.** `flow_shift` and `solver_order` are read from the checkpoint's `scheduler_config.json` (3.0 and 2 when absent); 30 steps and guidance 5 are the defaults of the command; Euler is there as the simple reference. Only `solver_order` 1 and 2 are implemented. **Checks**: the sigmas and timesteps equal diffusers' (5-step row of the table above), and the whole tiny pipeline from the same noise agrees to 130-135 dB (float32), with guidance 5 and 1.

### 23.12.3 The causal 3-D VAE (`WanVae`)

🧸 The image VAE (chapter 22) squeezes a picture 8 × in each direction. Wan's squeezes a *video* 8 × 8 in space **and 4 × in time**: four frames become one latent frame. And it is *causal*: a frame is built only from the present and the past, never from the future, so the first frame stands alone and can be a single picture.

📐 *Shapes.* The compression is `4 × 8 × 8 = 256` pixels (times three colours) per latent position of 16 numbers: the 17-frame 320 × 192 video has `17 · 192 · 320 · 3 = 3 133 440` numbers, its latent `5 · 24 · 40 · 16 = 76 800`, 41 times fewer. The latent frame count and the video frame count are related by `frames = 1 + 4 (F' − 1)`:

```
latent frames F'     1   2   5    9    21        video frames  1   5   17   33   81        (so `frames` must be 4k + 1; the Wan command rounds down)
```

*Causal convolution.* A `3 × 3 × 3` kernel looks at the present frame and the two before it (zeros before the first): `out[t] = W₀ ⋆ x[t−2] + W₁ ⋆ x[t−1] + W₂ ⋆ x[t]`. A scalar toy with `W = [0.5, 0.3, 0.2]` and `x = [1, 2, 3, 4]` gives `out = [0.2, 0.7, 1.7, 2.7]`: the first output sees only `x₀`, the third `x₀ … x₂`. jllm computes it as three 2-D convolutions of the clip shifted in time and added
(`Ops.frameShift` and the convolution's residual input); diffusers decodes frame by frame with a cache of the two previous frames, and the whole-clip form gives the same numbers.
*Temporal up-sampling* (`upsample3d`, in the first two of the four up blocks): the first frame passes unchanged; the others go through a causal `(3, 1, 1)` convolution to twice the channels whose two halves become two consecutive frames: `F' → 1 + 2 (F' − 1)`, twice, gives `1 + 4 (F' − 1)`. Spatial up-sampling is nearest-neighbour 2× plus a `3 × 3` convolution that halves the channels, in the first three blocks (the last block has no resampler: three doublings are the 8× of the compression). The decoder starts with `z · std + mean` per channel (the pipeline un-normalises the latent), `post_quant` and a 16 → 384 convolution, then ResNet, a spatial attention and a ResNet, the four up blocks (3 ResNets each), an RMS norm, SiLU and a 96 → 3 convolution clamped to [−1, 1].
**Why this design.** The checkpoint's decoder; implemented as a whole-clip computation, which uses more memory than the frame-by-frame cache but needs no state and maps onto the existing convolution kernels. The decoder is the slow stage for short clips: 17 frames of 320 × 192 take 4.9 s in float16 and 6.4 s in float32 on the GPU, 39.5 s on the CPU (table above).
**Checks and limits.** Real weights, latent 5 × 12 × 20 → 17 frames of 96 × 160: 1.1·10⁻⁵ relative (128.3 dB) in Metal float32, 77.0 dB in float16; tiny random weights 132 dB. Only the **decoder** exists: no VAE encoder, so no video-to-video or image-to-video.

### 23.12.4 The text encoder: UMT5 and the Unigram tokenizer

🧸 The words of the prompt must become numbers the transformer can read. A tokenizer cuts the sentence into pieces of words ("un", "believ", "able"); a very large reader (the *text encoder*, 24 layers, 5.7 billion numbers) turns each piece into a list of 4 096 numbers that says what it means in this sentence. The reader is bigger than the painter, so the code reads it from disk, uses it once, and lets it go before the painting starts.

📐 *Tokenizer* (`UnigramTokenizer`, from `tokenizer.json`): runs of spaces are collapsed, every word gets the marker `▁` in front, each word is cut into the vocabulary pieces whose scores (log-probabilities) add up highest (Viterbi), anything uncovered becomes the unknown token (consecutive ones fused), `</s>` ends the sequence. Toy scores for the word `▁unbelievable`: `▁un −4.0`, `believ −5.0`, `able −4.5` beats `▁u −5.5, n −6.0, believ, able` (−21.0) and `▁unbeliev −12.0, able` (−16.5):
the best cut is `▁un | believ | able` with −13.5. The prompt is cut to 512 tokens (the last replaced by `</s>`).
*Encoder* (`Umt5Encoder`): embedding `[L, 4096]`; 24 times `x += Attention(RMS(x))` and `x += W_o (GELU_tanh(W₀ RMS(x)) ⊙ W₁ RMS(x))` (gated feed-forward, `d_ff = 10 240`); the attention has **no `1/√d` scaling** and every layer adds a learned **relative-position bias** `bias[h, i, j] = table[bucket(j − i), h]`: 32 buckets, 16 per direction, exact for offsets below 8 and logarithmic up to 128 beyond:

```
offset j − i      0   +1   +7   +8   +12  +20  +50  +128  ≥ +200  |  −1   −7   −8   −12  −20  −50  −128  ≤ −200      (computed from `bucket()`)
bucket            0   17   23   24   25   26   29   31    31      |   1    7    8    9    10   13   15    15
```

Only the real tokens are encoded; the pipeline zero-pads the rows to 512 as diffusers does (`_get_t5_prompt_embeds`) and the transformer's cross-attention sees all 512 rows.
*int8.* 5.7 B parameters are 22.7 GB in float32 as shipped, 11 GB in bfloat16; with `q8` every linear weight is stored per row in groups of 32 as int8 plus a float scale (`w ≈ s · q`, the Q8_0 layout of chapter 6): **4.6 GB** for the 24 layers, read tensor by tensor so the float32 file is never resident, and the 256 000 × 4 096 embedding table read row by row from the mapped file. Activations stay float.
**Why this design.** The encoder runs on the CPU (a few hundred tokens), is read when a request arrives, used for the prompt and the negative prompt, and **freed before the transformer starts**, so its 4.6 GB and the transformer's weights are never needed at once. Because the quantization happens while the float32 file is read, **every request reads the whole 22.7 GB file again** and the int8 copy is not kept between requests; the time of that read was not measured, and caching the quantized form would be the obvious improvement.
**Checks and limits.** Tokenizer: eight prompts (accents, Japanese, emoji, runs of spaces, empty) give the same ids as Hugging Face `tokenizers` on the tiny and on the real 256 000-piece vocabulary. Encoder: tiny random weights 3.7·10⁻⁷ in float and 1.2·10⁻² with int8 (a random two-layer model is the worst case for quantization). **Not measured**: the encoder on its real weights (they were not downloaded, 23.12), and therefore the real error of int8.

### 23.12.5 The pipeline, and what a real run needs

🧸 To make one clip: read the words, ask the big reader, let it go; start from noise; thirty times ask the transformer "which way?" twice (once with the words, once without) and walk; then unpack the result into frames.

📐 `WanPipeline.generate`: (1) tokenize prompt and negative prompt, load the UMT5 encoder, encode both, free it; (2) draw the noise from `Rng(seed)` as one `[16, F', h, w]` tensor (PyTorch order) and store it frames-major; (3) for each of the 30 steps: `v = DiT(x, t, text)` and `v_u = DiT(x, t, negative)`, `v = v_u + g (v − v_u)` (`g = 5`), `x = FlowScheduler.step(v, i, x)`;
(4) `x · std + mean`, VAE decode, frames to pictures. Per request it needs: the text encoder 4.6 GB (briefly), the transformer and the VAE 6.2 GB in float32 (5.7 + 0.5) or 3.1 GB in float16, and activations (10.0 GB of GPU buffers in float32, 5.0 GB in float16, for 17 frames at 320 × 192). An estimate for 17 frames of 320 × 192 and 30 steps: `30 · 2 · 1.43 s + 5 s = 91 s` of GPU time in float16 plus the text encoder (23.12 above).
**Settings and entry point.** `./jllm video --video-model <local Wan folder> --prompt …` (the folder must hold `model_index.json` of class `WanPipeline`); not served by the API servers, the console or the chat tool, which stay AnimateDiff. **What would be needed for a first real clip**: the text encoder's weights. The repository stores them only as float32 (22.7 GB, five shards); the download budget of this work was 20 GB for all models. Nothing else is missing.

---

## 23.13 Why the clips are limited

🧸 The machinery of this chapter is exact: the numbers inside agree with the reference program to a fraction of a grey level. What a person sees depends on what the models were taught, and here the teaching is thin: the helper that makes motion was taught on a particular kind of painter, and we did not have that painter; and the better video model needs a reader we could not afford to download.

📐 The limits, and which of them are the code's and which the models':

* **AnimateDiff with plain Stable Diffusion 1.5 gives poor clips** (flat or smeared colours that change from frame to frame at 192-512 pixels). The clip equals diffusers' to 78-95 dB in float32 (23.10), so diffusers makes the same poor clip: the cause is the pairing. The `v1-5-2` adapter is meant for a *fine-tuned* Stable Diffusion 1.5 checkpoint (`video.base-model`); none was available within the download limits of this work, so **no pretty clip was made and none is claimed**. Changing `video.base-model` is the one setting that changes the look.
* **Short and small by construction:** at most 32 frames (the adapter's position table), 16 trained; 8 frames per second by default; the area is capped at 1024 × 1024 and a clip at 512 × 512 takes 6 minutes on this Mac (371 s for 20 steps, 23.11).
* **Flicker the temporal layers leave** is decoded frame by frame (23.6).
* **Wan 2.1** has every component written and checked against diffusers on real weights *except the text encoder's weights*, so no real prompt has been run through it: its transformer and VAE are verified with random text vectors of the right shape (23.12). Whether its clips look good is **not measured**.
* **Lossless output is large** (5.9 MB for 16 frames of 512 × 512 as WebP): MP4 needs `ffmpeg`.
* **Speed:** about 1.5 times slower than PyTorch on the same GPU at 256², and 14 times faster at 512² only because PyTorch runs out of memory there (23.11.1).

What would change the picture: a fine-tuned SD 1.5 base for the adapter (no code change), or the Wan text encoder (the 22.7 GB download, then `jllm video --video-model <folder>`), or a float16/bfloat16 copy of that encoder (about 11 GB).

