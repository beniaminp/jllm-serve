#!/usr/bin/env python3
"""Times Hugging Face diffusers' AnimateDiffPipeline on this machine for comparison with jllm: seconds per clip and per step.

    python -I scripts/diffusion/bench_diffusers_video.py BASE_DIR ADAPTER_DIR [--device mps|cpu] [--dtype float16|float32] \
        [--steps 10] [--size 512] [--frames 16] [--guidance 7.5] [--variant fp16] [--adapter-variant fp16] [--reps 1] [--decode-chunk 16]

Same prompt and scheduler setup as jllm's VideoBench (DDIM, linear betas, linspace timesteps, steps_offset 1), a fixed seed; the
device is synchronised after every clip; the first clip is a warm-up and reported apart. The time includes the VAE decoding of
every frame (decode_latents) and the conversion to PIL images, as `pipe(...)` does. The step time is measured with a callback
that synchronises the device at the end of each step.
"""
import argparse
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument('base')
ap.add_argument('adapter')
ap.add_argument('--device', default='mps')
ap.add_argument('--dtype', default='float16')
ap.add_argument('--steps', type=int, default=10)
ap.add_argument('--size', type=int, default=512)
ap.add_argument('--frames', type=int, default=16)
ap.add_argument('--guidance', type=float, default=7.5)
ap.add_argument('--variant', default='fp16')
ap.add_argument('--adapter-variant', default='fp16')
ap.add_argument('--reps', type=int, default=1)
ap.add_argument('--decode-chunk', type=int, default=16)
a = ap.parse_args()

from diffusers import AnimateDiffPipeline, DDIMScheduler, DiffusionPipeline, MotionAdapter

dtype = getattr(torch, a.dtype)
t0 = time.time()
adapter = MotionAdapter.from_pretrained(a.adapter, torch_dtype=dtype, variant=a.adapter_variant or None)
sd = DiffusionPipeline.from_pretrained(a.base, torch_dtype=dtype, variant=a.variant or None, safety_checker=None,
                                       feature_extractor=None, requires_safety_checker=False)
sched = DDIMScheduler.from_config(sd.scheduler.config, clip_sample=False, beta_schedule='linear', timestep_spacing='linspace', steps_offset=1)
pipe = AnimateDiffPipeline(vae=sd.vae, text_encoder=sd.text_encoder, tokenizer=sd.tokenizer, unet=sd.unet, motion_adapter=adapter,
                           scheduler=sched, feature_extractor=None, image_encoder=None).to(a.device)
pipe.set_progress_bar_config(disable=True)
print('loaded in %.1f s' % (time.time() - t0))
prompt = 'a red balloon floating over a green hill, white clouds moving, photo'


def sync():
    if a.device == 'mps':
        torch.mps.synchronize()


step_times = []


def on_step(p, i, t, kw):
    sync()
    step_times.append(time.time())
    return kw


times = []
for i in range(a.reps + 1):
    g = torch.Generator('cpu').manual_seed(42 + i)
    step_times.clear()
    t = time.time()
    step_times.append(t)
    pipe(prompt, negative_prompt='blurry, low quality', num_frames=a.frames, num_inference_steps=a.steps, guidance_scale=a.guidance,
         height=a.size, width=a.size, generator=g, decode_chunk_size=a.decode_chunk, callback_on_step_end=on_step)
    sync()
    total = time.time() - t
    per_step = [step_times[k + 1] - step_times[k] for k in range(len(step_times) - 1)]
    times.append((total, sum(per_step) / len(per_step), total - (step_times[-1] - step_times[0])))
    print('%s: %.2f s total, %.2f s/step (mean of %d), decode+postprocess %.2f s' % (
        'first ' if i == 0 else 'warm  ', times[-1][0], times[-1][1], len(per_step), times[-1][2]), flush=True)
warm = times[1:] or times
print('diffusers %s %s %dx%d, %d frames, %d step(s), guidance %.1f: warm mean %.2f s per clip, %.2f s/step, decode %.2f s' % (
    a.device, a.dtype, a.size, a.size, a.frames, a.steps, a.guidance, sum(w[0] for w in warm) / len(warm),
    sum(w[1] for w in warm) / len(warm), sum(w[2] for w in warm) / len(warm)))
if a.device == 'mps':
    print('mps memory: allocated %.0f MB, driver %.0f MB' % (torch.mps.current_allocated_memory() / 1e6, torch.mps.driver_allocated_memory() / 1e6))
