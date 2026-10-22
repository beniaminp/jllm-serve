#!/usr/bin/env python3
"""Reference outputs of Hugging Face diffusers for the Wan 2.1 components of jllm, on given weights and seeded inputs.

    python -I scripts/diffusion/reference_wan.py MODEL_DIR OUT_DIR --stage dit|vae [options]
        --frames F --height H --width W      (latent frames and size of the DiT input; pixel frames / size follow from the VAE for --stage vae)
        --text-len L --text-dim D --seed S --t T --dtype float32|float16|bfloat16 --device cpu|mps

--stage dit: a seeded latent [1, C, F, H, W] and random text embeddings [1, L, D] go through the transformer at timestep T:
    x.f32 (NCFHW), text.f32 [1, L, D], out.f32 (NCFHW)
--stage vae: a seeded latent [1, Z, F, H, W] is decoded: z.f32 (NCFHW), video.f32 [1, 3, F', H', W'] (NCFHW)
Writes ref.json with the shapes. Inputs come from the numpy twin of jllm's Rng (reference.py), so the Java side regenerates them.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import Rng, save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('out')
    ap.add_argument('--stage', default='dit')
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--height', type=int, default=4)
    ap.add_argument('--width', type=int, default=6)
    ap.add_argument('--text-len', type=int, default=7)
    ap.add_argument('--text-dim', type=int, default=0)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--t', type=float, default=700.0)
    ap.add_argument('--dtype', default='float32')
    ap.add_argument('--prompt', default='a red balloon over the green hill')
    ap.add_argument('--negative', default='blurry')
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--guidance', type=float, default=5.0)
    ap.add_argument('--pixel-frames', type=int, default=5)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dtype = getattr(torch, a.dtype)
    meta = dict(stage=a.stage, frames=a.frames, height=a.height, width=a.width, seed=a.seed, t=a.t)
    if a.stage == 'dit':
        from diffusers import WanTransformer3DModel
        m = WanTransformer3DModel.from_pretrained(os.path.join(a.model, 'transformer'), torch_dtype=dtype).to(a.device).eval()
        c = m.config.in_channels
        td = a.text_dim or m.config.text_dim
        r = Rng(a.seed)
        x = r.normal(c * a.frames * a.height * a.width).reshape(1, c, a.frames, a.height, a.width)
        text = r.normal(a.text_len * td).reshape(1, a.text_len, td)
        with torch.no_grad():
            y = m(hidden_states=torch.from_numpy(x).to(a.device, dtype), timestep=torch.tensor([a.t]).to(a.device),
                  encoder_hidden_states=torch.from_numpy(text).to(a.device, dtype), return_dict=False)[0]
        save(a.out, 'x.f32', x)
        save(a.out, 'text.f32', text)
        save(a.out, 'out.f32', y.float().cpu().numpy())
        meta.update(channels=c, text_len=a.text_len, text_dim=td)
    elif a.stage == 'vae':
        from diffusers import AutoencoderKLWan
        v = AutoencoderKLWan.from_pretrained(os.path.join(a.model, 'vae'), torch_dtype=dtype).to(a.device).eval()
        zc = v.config.z_dim
        z = Rng(a.seed).normal(zc * a.frames * a.height * a.width).reshape(1, zc, a.frames, a.height, a.width) * 0.8
        with torch.no_grad():
            y = v.decode(torch.from_numpy(z).to(a.device, dtype), return_dict=False)[0]
        save(a.out, 'z.f32', z)
        save(a.out, 'video.f32', y.float().cpu().numpy())
        meta.update(z_dim=zc, out_shape=list(y.shape))
    elif a.stage == 'pipe':
        # the whole WanPipeline (UniPC, guidance, UMT5) with jllm's random numbers: the one randn_tensor call of the pipeline
        from diffusers import WanPipeline
        import diffusers.pipelines.wan.pipeline_wan as pm
        pipe = WanPipeline.from_pretrained(a.model, torch_dtype=dtype).to(a.device)
        rng = Rng(a.seed)

        def fake_randn(shape, generator=None, device=None, dtype=None, layout=None):
            n = int(np.prod(shape))
            return torch.from_numpy(rng.normal(n).reshape(tuple(shape))).to(device=device, dtype=dtype)

        pm.randn_tensor = fake_randn
        with torch.no_grad():
            pe, ne = pipe.encode_prompt(prompt=a.prompt, negative_prompt=a.negative, do_classifier_free_guidance=True, device=a.device, max_sequence_length=512)
        save(a.out, 'prompt_embeds.f32', pe.float().cpu().numpy())
        save(a.out, 'negative_embeds.f32', ne.float().cpu().numpy())

        def dump(pipeline, i, t, kw_):
            save(a.out, 'step_%d.f32' % i, kw_['latents'].float().cpu().numpy())
            return kw_

        lat = pipe(prompt=a.prompt, negative_prompt=a.negative, height=a.height, width=a.width, num_frames=a.pixel_frames,
                   num_inference_steps=a.steps, guidance_scale=a.guidance, output_type='latent', callback_on_step_end=dump,
                   callback_on_step_end_tensor_inputs=['latents']).frames
        v = pipe.vae
        mean = torch.tensor(v.config.latents_mean).view(1, v.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
        std = 1.0 / torch.tensor(v.config.latents_std).view(1, v.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
        with torch.no_grad():
            y = v.decode(lat / std + mean, return_dict=False)[0]
        save(a.out, 'latents.f32', lat.float().cpu().numpy())
        save(a.out, 'video.f32', y.float().cpu().numpy())
        meta.update(prompt=a.prompt, negative=a.negative, steps=a.steps, guidance=a.guidance, pixel_frames=a.pixel_frames, out_shape=list(y.shape),
                    lat_shape=list(lat.shape), width=a.width, height=a.height)
    elif a.stage in ('t5', 'tok'):
        from tokenizers import Tokenizer
        raw = Tokenizer.from_file(os.path.join(a.model, 'tokenizer', 'tokenizer.json'))

        class _Tok:   # the fast tokenizer of the pipeline (T5TokenizerFast) is this tokenizers.Tokenizer with </s> appended
            def __call__(self, text, add_special_tokens=True):
                class R:
                    input_ids = raw.encode(text, add_special_tokens=add_special_tokens).ids
                return R
        tok = _Tok()
        cases = ['a red balloon floating over a green hill', 'A  cat   sitting on the sofa, photo!', 'waves and the blue sky', 'xyz 123 \u00e9\u00e8 caf\u00e9 \u65e5\u672c\u8a9e',
                 'balloons running', '', 'a' * 40, 'the the the  cat']
        cases += ['A  cat \U0001F600 caf\u00e9, sitting', 'Hello   world.', 'x' * 50, 'a red balloon floating over a green hill, white clouds moving, photo',
                  'Ein Pferd l\u00e4uft \u00fcber die Wiese', ' leading and trailing  ']
        meta['cases'] = []
        for c in cases:
            ids = tok(c, add_special_tokens=True).input_ids
            meta['cases'].append({'text': c, 'ids': ids})
        if a.stage == 't5':
            from transformers import UMT5EncoderModel
            te = UMT5EncoderModel.from_pretrained(os.path.join(a.model, 'text_encoder'), torch_dtype=dtype).to(a.device).eval()
            prompt = cases[0]
            ids = torch.tensor([tok(prompt, add_special_tokens=True).input_ids])
            with torch.no_grad():
                h = te(ids.to(a.device)).last_hidden_state
            save(a.out, 'ids.i32', ids.numpy())
            save(a.out, 'emb.f32', h.float().cpu().numpy())
            meta['prompt'] = prompt
    else:
        raise SystemExit('unknown stage')
    json.dump(meta, open(os.path.join(a.out, 'ref.json'), 'w'), indent=1)
    print('wrote', a.out, file=sys.stderr)


if __name__ == '__main__':
    main()
