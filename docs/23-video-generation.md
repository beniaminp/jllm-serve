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
why the temporal layers add roughly @@MOTIONSHARE@@ % to the time of a UNet evaluation. The group norm over the clip is the one subtle
point: normalising each frame alone changes the result (a frame brighter than the others would be normalised back to the mean), and the
test of the tiny pipeline fails at 1 · 10⁻² if it is done per frame.

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

---

## 23.6 Decoding the frames

🧸 The painter works on small sketches; each sketch must be turned into a real picture. That is the slow part of one picture and it is
done sixteen times, but only once at the end.

📐 `x_final / 0.18215` goes through the VAE decoder of 22.3 **one frame at a time** (`video.decode-chunk` frames at once, default 1):
a 512 × 512 frame needs 1.2 s and about 2.5 TFLOP, the clip's 16 frames 19 s, whatever the number of steps. Decoding frames
independently is exactly what diffusers' `decode_latents` does (it loops over chunks of the flattened batch). The VAE has no temporal
layers, so a flicker that the temporal transformers did not remove stays in the decoded frames. Tiling (22.3) applies per frame above
1024 × 1024.

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
is 84 MB in float16, the GEGLU intermediate of the first temporal layer `[131 072, 2560]` is 671 MB. @@MEMORY@@

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
Pillow's libwebp and `ffmpeg` agreed on every pixel of a 6-frame 200 × 150 clip). Lossless is big: @@WEBPSIZE@@. **MP4**: raw RGB
frames are piped into `ffmpeg -f rawvideo -pix_fmt rgb24 -i - -c:v libx264 -crf 17 -pix_fmt yuv420p -movflags +faststart` (`mpeg4` when
libx264 is missing; dimensions are padded to even); `video.ffmpeg` is `auto` (look it up on the PATH), `off` or a path. Without ffmpeg the
format `auto` is WebP, and asking for `mp4` is a clear 400.

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

---

## 23.10 Verified against diffusers

@@VERIFIED@@

---

## 23.11 Measured speed and memory

@@SPEED@@

---

## 23.12 Wan 2.1: how far it got and what remains

@@WAN@@
