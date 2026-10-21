#!/usr/bin/env python3
"""Reference outputs of Hugging Face diffusers for the diffusion stack of jllm, stage by stage, with the random numbers
of jllm (org.robomiri.jllm.diffusion.Rng) so both sides see the same noise.

    python -I scripts/diffusion/reference.py MODEL_DIR OUT_DIR [options]

Writes raw little-endian float32 / int32 files and ref.json into OUT_DIR:
    ids.i32        token ids [2, 77]            (negative prompt, prompt) -- [1, 77] without guidance
    text.f32       text embeddings [B, 77, C]   (what the UNet is conditioned on; B = 2 with guidance)
    pooled.f32     pooled text embeddings (SDXL)
    noise.f32      the initial noise [4, h, w] (NCHW) of Rng(seed)
    unet_in.f32    one UNet input [B, 4, h, w] (scaled noise duplicated) ; unet_out.f32 its output [B, 4, h, w]
    latents.f32    final latents [4, h, w] ; image.f32 the decoded image [H, W, 3] (-1..1 range, not clamped)
    vae_z.f32 / vae_dec.f32  a fixed latent [4, h, w] and its decoding [H, W, 3]
    vae_img.f32 / vae_mean.f32 / vae_std.f32  a fixed image [H, W, 3] and its posterior [4, h, w]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

GOLDEN = np.uint64(0x9E3779B97F4A7C15)


class Rng:
    """numpy twin of org.robomiri.jllm.diffusion.Rng (SplitMix64 counter stream + Box-Muller)."""

    def __init__(self, seed):
        self.seed = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
        self.counter = 0
        self.spare = None

    def _uniform(self, n):
        k = np.arange(self.counter + 1, self.counter + 1 + n, dtype=np.uint64)
        self.counter += n
        with np.errstate(over='ignore'):
            z = self.seed + k * GOLDEN
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z = z ^ (z >> np.uint64(31))
        return ((z >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0 ** -53

    def normal(self, n):
        out = []
        need = n
        if self.spare is not None and need > 0:
            out.append(np.array([self.spare], np.float32))
            self.spare = None
            need -= 1
        pairs = (need + 1) // 2
        if pairs:
            u = self._uniform(2 * pairs)
            u1, u2 = u[0::2], u[1::2]
            r = np.sqrt(-2.0 * np.log(u1))
            th = 2.0 * np.pi * u2
            z = np.empty(2 * pairs, np.float64)
            z[0::2] = r * np.cos(th)
            z[1::2] = r * np.sin(th)
            z = z.astype(np.float32)
            if need % 2 == 1:
                self.spare = z[-1]
                z = z[:-1]
            out.append(z)
        return np.concatenate(out) if out else np.zeros(0, np.float32)


def save(d, name, a):
    a = np.ascontiguousarray(a)
    if a.dtype == np.int64:
        a = a.astype(np.int32)
    elif a.dtype != np.int32:
        a = a.astype(np.float32)
    a.tofile(os.path.join(d, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('out')
    ap.add_argument('--prompt', default='a photo of a cat sitting on a red sofa')
    ap.add_argument('--negative', default='blurry, low quality')
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--height', type=int, default=64)
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--guidance', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--scheduler', default='euler')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--variant', default=None)
    ap.add_argument('--dtype', default='float32')
    ap.add_argument('--skip-pipeline', action='store_true')
    ap.add_argument('--clip-skip', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dtype = getattr(torch, a.dtype)

    from diffusers import (DiffusionPipeline, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                           DPMSolverMultistepScheduler)
    import diffusers.schedulers.scheduling_euler_ancestral_discrete as eam

    pipe = DiffusionPipeline.from_pretrained(a.model, torch_dtype=dtype, variant=a.variant, safety_checker=None,
                                             requires_safety_checker=False)
    pipe = pipe.to(a.device)
    cfgd = pipe.scheduler.config
    spacing = cfgd.get('timestep_spacing', 'linspace')
    if a.scheduler == 'euler':
        pipe.scheduler = EulerDiscreteScheduler.from_config(cfgd, timestep_spacing=spacing)
    elif a.scheduler == 'euler_a':
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(cfgd, timestep_spacing=spacing)
    elif a.scheduler == 'dpmpp_2m':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(cfgd, timestep_spacing=spacing)
    elif a.scheduler == 'dpmpp_2m_karras':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(cfgd, timestep_spacing=spacing, use_karras_sigmas=True)
    xl = hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None
    lh, lw = a.height // 8, a.width // 8
    cfg = a.guidance > 1.0
    meta = dict(prompt=a.prompt, negative=a.negative, width=a.width, height=a.height, steps=a.steps,
                guidance=a.guidance, seed=a.seed, scheduler=a.scheduler, xl=xl, cfg=cfg)

    # tokens
    tok = pipe.tokenizer
    def ids(p):
        return tok(p, padding='max_length', max_length=tok.model_max_length, truncation=True, return_tensors='pt').input_ids[0].numpy()
    rows = [ids(a.negative), ids(a.prompt)] if cfg else [ids(a.prompt)]
    save(a.out, 'ids.i32', np.stack(rows))
    # extra tokenizer cases
    cases = ['', 'A  Photo\tOF a Café — 日本語 😀 it\'s don\'t 123', 'x' * 400, '<|startoftext|>hello<|endoftext|> world!!']
    meta['tok_cases'] = [dict(text=c, ids=[int(v) for v in ids(c)]) for c in cases]

    # text embeddings
    kw = {}
    if a.clip_skip is not None:
        kw['clip_skip'] = a.clip_skip
    with torch.no_grad():
        if xl:
            pe, npe, pp, npp = pipe.encode_prompt(a.prompt, device=a.device, num_images_per_prompt=1,
                                                  do_classifier_free_guidance=cfg, negative_prompt=a.negative or None, **kw)
            emb = torch.cat([npe, pe]) if cfg else pe
            pooled = torch.cat([npp, pp]) if cfg else pp
            save(a.out, 'pooled.f32', pooled.float().cpu().numpy())
        else:
            pe, npe = pipe.encode_prompt(a.prompt, a.device, 1, cfg, a.negative, **kw)[:2]
            emb = torch.cat([npe, pe]) if cfg else pe
    save(a.out, 'text.f32', emb.float().cpu().numpy())

    # noise (the same stream jllm draws: [4, h, w] first)
    rng = Rng(a.seed)
    noise = rng.normal(4 * lh * lw).reshape(1, 4, lh, lw)
    save(a.out, 'noise.f32', noise)

    # one UNet evaluation at the first timestep
    sched = pipe.scheduler
    sched.set_timesteps(a.steps, device=a.device)
    t0 = sched.timesteps[0]
    x = torch.from_numpy(noise).to(a.device, dtype) * sched.init_noise_sigma
    xin = sched.scale_model_input(x, t0)
    xin2 = torch.cat([xin, xin]) if cfg else xin
    with torch.no_grad():
        if xl:
            tid = torch.tensor([[a.height, a.width, 0, 0, a.height, a.width]], dtype=dtype, device=a.device)
            tid = torch.cat([tid, tid]) if cfg else tid
            ca = {'text_embeds': pooled.to(dtype), 'time_ids': tid}
            out = pipe.unet(xin2, t0, encoder_hidden_states=emb.to(dtype), added_cond_kwargs=ca).sample
        else:
            out = pipe.unet(xin2, t0, encoder_hidden_states=emb.to(dtype)).sample
    meta['t0'] = float(t0)
    save(a.out, 'unet_in.f32', xin2.float().cpu().numpy())
    save(a.out, 'unet_out.f32', out.float().cpu().numpy())

    # VAE on fixed inputs
    vae = pipe.vae
    r2 = Rng(99)
    z = r2.normal(4 * lh * lw).reshape(1, 4, lh, lw) * 0.8
    with torch.no_grad():
        dec = vae.decode(torch.from_numpy(z).to(a.device, dtype) / vae.config.scaling_factor).sample
    save(a.out, 'vae_z.f32', z)
    save(a.out, 'vae_dec.f32', dec.float().cpu().numpy().transpose(0, 2, 3, 1))
    img = np.clip(r2.normal(3 * a.height * a.width).reshape(1, 3, a.height, a.width) * 0.5, -1, 1).astype(np.float32)
    with torch.no_grad():
        post = vae.encode(torch.from_numpy(img).to(a.device, dtype)).latent_dist
    save(a.out, 'vae_img.f32', img.transpose(0, 2, 3, 1))
    save(a.out, 'vae_mean.f32', post.mean.float().cpu().numpy())
    save(a.out, 'vae_std.f32', post.std.float().cpu().numpy())

    if not a.skip_pipeline:
        # the whole pipeline with our noise; ancestral noise comes from the stream after the initial noise
        extra = []
        if a.scheduler == 'euler_a':
            for _ in range(a.steps):
                extra.append(rng.normal(4 * lh * lw).reshape(1, 4, lh, lw))
        it = iter(extra)
        orig = eam.randn_tensor
        eam.randn_tensor = lambda shape, generator=None, device=None, dtype=None, layout=None: \
            torch.from_numpy(next(it)).to(device=device, dtype=dtype)
        try:
            kws = dict(prompt=a.prompt, negative_prompt=a.negative, num_inference_steps=a.steps, guidance_scale=a.guidance,
                       height=a.height, width=a.width, latents=torch.from_numpy(noise).to(a.device, dtype), output_type='latent')
            if a.clip_skip is not None:
                kws['clip_skip'] = a.clip_skip
            res = pipe(**kws).images
        finally:
            eam.randn_tensor = orig
        lat = res.float().cpu().numpy()
        save(a.out, 'latents.f32', lat)
        with torch.no_grad():
            dimg = vae.decode(res.to(dtype) / vae.config.scaling_factor).sample
        save(a.out, 'image.f32', dimg.float().cpu().numpy().transpose(0, 2, 3, 1))
    json.dump(meta, open(os.path.join(a.out, 'ref.json'), 'w'), indent=1)
    print('wrote', a.out, file=sys.stderr)


if __name__ == '__main__':
    main()
