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
why the temporal layers add roughly 40 % to the time of a UNet evaluation. The group norm over the clip is the one subtle
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
is 84 MB in float16, the GEGLU intermediate of the first temporal layer `[131 072, 2560]` is 671 MB. Measured peaks of the GPU buffers: 4.1 GB at 256 × 256 and 7.3 GB at 512 × 512 in float16 (8.1 GB at 256 × 256 in float32). `video.decode-chunk` (frames per VAE call, default 1) trades the VAE's activation memory against speed.

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

