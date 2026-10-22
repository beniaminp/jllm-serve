#!/usr/bin/env python3
"""Times Hugging Face diffusers on this machine for comparison with jllm: seconds per image for a few step counts.

    python -I scripts/diffusion/bench_diffusers.py MODEL_DIR [--device mps|cpu] [--dtype float16|float32] [--steps 1,4,20] \
        [--size 512] [--guidance 0] [--variant fp16]

Uses the same prompt as jllm's DiffusionBench, a fixed seed, the pipeline's own scheduler, and waits for the device after every
image (torch.mps.synchronize). The first image of each step count is a warm-up and not reported.
"""
import argparse
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('--device', default='mps')
ap.add_argument('--dtype', default='float16')
ap.add_argument('--steps', default='1,4,20')
ap.add_argument('--size', type=int, default=512)
ap.add_argument('--guidance', type=float, default=0.0)
ap.add_argument('--variant', default='fp16')
ap.add_argument('--reps', type=int, default=3)
a = ap.parse_args()

from diffusers import DiffusionPipeline

dtype = getattr(torch, a.dtype)
t0 = time.time()
pipe = DiffusionPipeline.from_pretrained(a.model, torch_dtype=dtype, variant=a.variant or None, safety_checker=None,
                                         requires_safety_checker=False).to(a.device)
pipe.set_progress_bar_config(disable=True)
print('loaded in %.1f s' % (time.time() - t0))
prompt = 'a photo of an astronaut riding a horse on the moon'


def sync():
    if a.device == 'mps':
        torch.mps.synchronize()


for steps in [int(x) for x in a.steps.split(',')]:
    times = []
    for i in range(a.reps + 1):
        g = torch.Generator('cpu').manual_seed(42 + i)
        t = time.time()
        pipe(prompt, num_inference_steps=steps, guidance_scale=a.guidance, height=a.size, width=a.size, generator=g)
        sync()
        times.append(time.time() - t)
    warm = times[1:]
    print('diffusers %s %s %dx%d, %d step(s), guidance %.1f: first %.2f s, then %s (mean %.2f s)' % (
        a.device, a.dtype, a.size, a.size, steps, a.guidance, times[0], ', '.join('%.2f' % x for x in warm), sum(warm) / len(warm)))
if a.device == 'mps':
    print('mps memory: allocated %.0f MB, driver %.0f MB' % (torch.mps.current_allocated_memory() / 1e6, torch.mps.driver_allocated_memory() / 1e6))
