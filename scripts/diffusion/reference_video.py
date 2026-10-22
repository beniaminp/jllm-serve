#!/usr/bin/env python3
"""Reference outputs of Hugging Face diffusers' AnimateDiffPipeline for the video pipeline of jllm, stage by stage, with the
random numbers of jllm (org.robomiri.jllm.diffusion.Rng): the pipeline's one randn_tensor call (the initial noise
[1, 4, frames, h, w]) takes its numbers from the stream of the seed.

    python -I scripts/diffusion/reference_video.py BASE_DIR ADAPTER_DIR OUT_DIR [options]

BASE_DIR is a Stable Diffusion 1.x pipeline folder (diffusers layout), ADAPTER_DIR a MotionAdapter folder. Options:
    --prompt --negative --width --height --frames --steps --guidance --seed
    --scheduler ddim|euler|dpmpp_2m   (AnimateDiff's usual setup: linear betas, linspace timesteps, steps_offset 1)
    --device cpu|mps --dtype float32|float16 --variant fp16 --adapter-variant fp16 --dump-steps

Writes raw little-endian float32 / int32 files and ref.json into OUT_DIR:
    text.f32       text embeddings [B, 77, C]  (B = 2 with guidance: negative, prompt)
    noise.f32      the initial noise [4, F, h, w] (channels first, then frames: the order of the pipeline's randn_tensor call)
    unet_in.f32    the first UNet input [B, 4, F, h, w] (scaled noise, duplicated for guidance); unet_out.f32 its output
    latents.f32    final latents [4, F, h, w]  ;  frames.f32  the decoded frames [F, H, W, 3] (-1..1 range, not clamped)
    step_N.f32     with --dump-steps: the latents [4, F, h, w] after step N
    video.mp4-like checks need nothing from here: the frames are compared as pixels.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import Rng, save  # noqa: E402  (the numpy twin of jllm's Rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('adapter')
    ap.add_argument('out')
    ap.add_argument('--prompt', default='a red balloon floating over a green hill, clouds moving')
    ap.add_argument('--negative', default='blurry, low quality')
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--height', type=int, default=64)
    ap.add_argument('--frames', type=int, default=4)
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--guidance', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--scheduler', default='ddim')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--dtype', default='float32')
    ap.add_argument('--variant', default=None)
    ap.add_argument('--adapter-variant', default=None)
    ap.add_argument('--clip-skip', type=int, default=None)
    ap.add_argument('--dump-steps', action='store_true')
    ap.add_argument('--split-cfg', action='store_true', help='run the UNet on the negative and the prompt clip one after the other (same numbers, half the memory)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dtype = getattr(torch, a.dtype)

    from diffusers import (AnimateDiffPipeline, DDIMScheduler, DiffusionPipeline, DPMSolverMultistepScheduler,
                           EulerDiscreteScheduler, MotionAdapter)
    import diffusers.pipelines.animatediff.pipeline_animatediff as pm

    adapter = MotionAdapter.from_pretrained(a.adapter, torch_dtype=dtype, variant=a.adapter_variant)
    sd = DiffusionPipeline.from_pretrained(a.base, torch_dtype=dtype, variant=a.variant, safety_checker=None,
                                           feature_extractor=None, requires_safety_checker=False)
    # AnimateDiff's recommended scheduler setup: linear betas, linspace timesteps, steps_offset 1, no sample clipping
    kw = dict(beta_schedule='linear', timestep_spacing='linspace', steps_offset=1)
    if a.scheduler == 'ddim':
        sched = DDIMScheduler.from_config(sd.scheduler.config, clip_sample=False, **kw)
    elif a.scheduler == 'euler':
        sched = EulerDiscreteScheduler.from_config(sd.scheduler.config, **kw)
    elif a.scheduler == 'dpmpp_2m':
        sched = DPMSolverMultistepScheduler.from_config(sd.scheduler.config, **kw)
    else:
        raise SystemExit('unknown scheduler ' + a.scheduler)
    pipe = AnimateDiffPipeline(vae=sd.vae, text_encoder=sd.text_encoder, tokenizer=sd.tokenizer, unet=sd.unet,
                               motion_adapter=adapter, scheduler=sched, feature_extractor=None, image_encoder=None)
    pipe = pipe.to(a.device)
    lh, lw, nf = a.height // 8, a.width // 8, a.frames
    cfg = a.guidance > 1.0
    meta = dict(prompt=a.prompt, negative=a.negative, width=a.width, height=a.height, frames=nf, steps=a.steps,
                guidance=a.guidance, seed=a.seed, scheduler=a.scheduler, cfg=cfg)

    kwc = {} if a.clip_skip is None else {'clip_skip': a.clip_skip}
    with torch.no_grad():
        pe, npe = pipe.encode_prompt(a.prompt, a.device, 1, cfg, a.negative, **kwc)[:2]
        emb = torch.cat([npe, pe]) if cfg else pe
    save(a.out, 'text.f32', emb.float().cpu().numpy())

    if a.split_cfg:  # the two guidance clips are independent in the batch: evaluating them separately gives the same numbers
        orig_forward = pipe.unet.forward

        def split_forward(sample, timestep, encoder_hidden_states=None, **kw):
            if sample.shape[0] != 2:
                return orig_forward(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kw)
            n = encoder_hidden_states.shape[0] // 2
            outs = [orig_forward(sample[i:i + 1], timestep, encoder_hidden_states=encoder_hidden_states[i * n:(i + 1) * n], **kw).sample for i in range(2)]

            class Out:
                pass
            o = Out()
            o.sample = torch.cat(outs)
            return o

        pipe.unet.forward = split_forward

    # one UNet evaluation at the first timestep
    noise = Rng(a.seed).normal(4 * nf * lh * lw).reshape(1, 4, nf, lh, lw)
    save(a.out, 'noise.f32', noise)
    pipe.scheduler.set_timesteps(a.steps, device=a.device)
    t0 = pipe.scheduler.timesteps[0]
    x = torch.from_numpy(noise).to(a.device, dtype) * pipe.scheduler.init_noise_sigma
    xin = pipe.scheduler.scale_model_input(x, t0)
    xin2 = torch.cat([xin, xin]) if cfg else xin
    with torch.no_grad():
        out = pipe.unet(xin2, t0, encoder_hidden_states=emb.to(dtype).repeat_interleave(nf, dim=0)).sample
    meta['t0'] = float(t0)
    save(a.out, 'unet_in.f32', xin2.float().cpu().numpy())
    save(a.out, 'unet_out.f32', out.float().cpu().numpy())

    # the whole pipeline from the same noise
    rng = Rng(a.seed)

    def fake_randn(shape, generator=None, device=None, dtype=None, layout=None):
        n = int(np.prod(shape))
        return torch.from_numpy(rng.normal(n).reshape(tuple(shape))).to(device=device, dtype=dtype)

    pm.randn_tensor = fake_randn
    common = dict(prompt=a.prompt, negative_prompt=a.negative, num_inference_steps=a.steps, guidance_scale=a.guidance,
                  num_frames=nf, height=a.height, width=a.width, output_type='latent')
    if a.clip_skip is not None:
        common['clip_skip'] = a.clip_skip
    if a.dump_steps:
        def dump(pipeline, i, t, kw_):
            save(a.out, 'step_%d.f32' % i, kw_['latents'].float().cpu().numpy())
            return kw_
        common['callback_on_step_end'] = dump
        common['callback_on_step_end_tensor_inputs'] = ['latents']
    res = pipe(**common).frames  # output_type latent: [1, 4, F, h, w]
    save(a.out, 'latents.f32', res.float().cpu().numpy())
    vae = pipe.vae
    with torch.no_grad():
        z = res.to(vae.dtype) / vae.config.scaling_factor
        b, c, f, h, w = z.shape
        z = z.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        dec = torch.cat([vae.decode(z[i:i + 1]).sample for i in range(z.shape[0])])
    save(a.out, 'frames.f32', dec.float().cpu().numpy().transpose(0, 2, 3, 1))
    json.dump(meta, open(os.path.join(a.out, 'ref.json'), 'w'), indent=1)
    print('wrote', a.out, file=sys.stderr)


if __name__ == '__main__':
    main()
