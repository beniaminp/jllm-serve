#!/usr/bin/env python3
"""Writes tiny random Stable Diffusion pipelines in the diffusers folder layout (a few hundred kB each), the fixtures of
the diffusion tests: random weights, the real architecture code paths of SD 1.x (conv 1x1 projections), SD 2.x /
Turbo (linear projections, OpenCLIP-style gelu text encoder), SDXL (two text encoders, text_time conditioning, two
transformer layers in a block) and an inpainting UNet (9 input channels).

    python -I scripts/diffusion/make-tiny.py OUT_DIR          (needs torch, diffusers, transformers)

Weights are rounded to float16 and saved as float16, so what the Java side reads is exactly what diffusers computes
with once the file is loaded back as float32.
"""
import json
import os
import sys

import torch
from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
from safetensors.torch import save_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

out = sys.argv[1]
torch.manual_seed(1234)


def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def make_tokenizer(path, pad_token):
    """A small CLIP BPE tokenizer: the 256 byte symbols, their word-final variants, a few merges, the two specials."""
    os.makedirs(path, exist_ok=True)
    b2u = bytes_to_unicode()
    syms = [b2u[b] for b in sorted(b2u)]
    vocab = {}
    for s in syms:
        vocab[s] = len(vocab)
    for s in syms:
        vocab[s + '</w>'] = len(vocab)
    # merges learned from a small corpus (greedy BPE)
    corpus = ("a photo of a cat sitting on a red sofa . an astronaut riding a horse on the moon . "
              "a painting of a mountain lake at sunrise , blue sky , green trees , high detail . "
              "a dog and a bird ; the quick brown fox jumps over the lazy dog ! café 日本語 emoji 😀 "
              "it's we're don't 123 45 ").split()
    words = {}
    for w in corpus:
        t = tuple(b2u[b] for b in w.encode('utf-8'))
        t = t[:-1] + (t[-1] + '</w>',)
        words[t] = words.get(t, 0) + 1
    merges = []
    for _ in range(120):
        pairs = {}
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + c
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))
        merges.append(best)
        new = {}
        for w, c in words.items():
            o, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    o.append(w[i] + w[i + 1])
                    i += 2
                else:
                    o.append(w[i])
                    i += 1
            new[tuple(o)] = new.get(tuple(o), 0) + c
        words = new
        vocab[best[0] + best[1]] = len(vocab)
    vocab['<|startoftext|>'] = len(vocab)
    vocab['<|endoftext|>'] = len(vocab)
    json.dump(vocab, open(os.path.join(path, 'vocab.json'), 'w'), ensure_ascii=False)
    with open(os.path.join(path, 'merges.txt'), 'w') as f:
        f.write('#version: 0.2\n')
        for a, b in merges:
            f.write(a + ' ' + b + '\n')
    json.dump({'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>',
               'pad_token': pad_token}, open(os.path.join(path, 'special_tokens_map.json'), 'w'))
    added = {str(vocab['<|startoftext|>']): {'content': '<|startoftext|>', 'normalized': True, 'special': True},
             str(vocab['<|endoftext|>']): {'content': '<|endoftext|>', 'normalized': True, 'special': True}}
    if pad_token == '!':  # OpenCLIP's tokenizer registers "!" as a (non-normalized) special token: it matches in raw text
        added[str(vocab['!'])] = {'content': '!', 'normalized': False, 'special': True}
    json.dump({'model_max_length': 77, 'pad_token': pad_token, 'bos_token': '<|startoftext|>',
               'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>', 'tokenizer_class': 'CLIPTokenizer',
               'added_tokens_decoder': added},
              open(os.path.join(path, 'tokenizer_config.json'), 'w'))
    return len(vocab)


def half_round(model):
    """Perturbs norm scales/biases (the defaults 1/0 would hide mistakes) and rounds every parameter to float16."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() == 1 and ('norm' in n or 'layer_norm' in n):
                p.add_(torch.randn_like(p) * 0.1)
            elif p.dim() == 1:
                p.add_(torch.randn_like(p) * 0.05)
            p.copy_(p.half().float())
    return model


def save(model, path, name):
    os.makedirs(path, exist_ok=True)
    if hasattr(model, 'save_config'):
        model.save_config(path)  # diffusers
    else:
        model.config.to_json_file(os.path.join(path, 'config.json'))  # transformers
    sd = {k: v.detach().half().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(path, name), metadata={'format': 'pt'})


def make(name, kind):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    sdxl = kind == 'sdxl'
    pad = '!' if kind in ('sd2', 'sdxl') else '<|endoftext|>'
    vocab = make_tokenizer(os.path.join(d, 'tokenizer'), pad)
    if sdxl:
        make_tokenizer(os.path.join(d, 'tokenizer_2'), '!')
    act = 'quick_gelu' if kind in ('sd1', 'inpaint') else 'gelu'
    te_cfg = CLIPTextConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                            num_attention_heads=4, max_position_embeddings=77, hidden_act=act, projection_dim=32,
                            bos_token_id=vocab - 2, eos_token_id=vocab - 1, pad_token_id=0 if pad == '!' else vocab - 1)
    te = half_round(CLIPTextModel(te_cfg).eval())
    save(te, os.path.join(d, 'text_encoder'), 'model.safetensors')
    if sdxl:
        te2_cfg = CLIPTextConfig(vocab_size=vocab, hidden_size=48, intermediate_size=96, num_hidden_layers=3,
                                 num_attention_heads=4, max_position_embeddings=77, hidden_act='gelu', projection_dim=40,
                                 bos_token_id=vocab - 2, eos_token_id=vocab - 1, pad_token_id=0)
        te2 = half_round(CLIPTextModelWithProjection(te2_cfg).eval())
        save(te2, os.path.join(d, 'text_encoder_2'), 'model.safetensors')
    cross = 32 + 48 if sdxl else 32
    common = dict(sample_size=8, out_channels=4, layers_per_block=1, norm_num_groups=8, cross_attention_dim=cross,
                  block_out_channels=(16, 32, 32), down_block_types=('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'),
                  up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'))
    if kind == 'sd1':
        unet = UNet2DConditionModel(in_channels=4, attention_head_dim=2, use_linear_projection=False, **common)
    elif kind == 'inpaint':
        unet = UNet2DConditionModel(in_channels=9, attention_head_dim=2, use_linear_projection=False, **common)
    elif kind == 'sd2':
        unet = UNet2DConditionModel(in_channels=4, attention_head_dim=(2, 4, 4), use_linear_projection=True, **common)
    else:
        unet = UNet2DConditionModel(in_channels=4, attention_head_dim=(2, 4, 4), use_linear_projection=True,
                                    addition_embed_type='text_time', addition_time_embed_dim=8,
                                    projection_class_embeddings_input_dim=40 + 6 * 8,
                                    transformer_layers_per_block=(1, 2, 2), **common)
    unet = half_round(unet.eval())
    save(unet, os.path.join(d, 'unet'), 'diffusion_pytorch_model.safetensors')
    vae = half_round(AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, norm_num_groups=8, layers_per_block=1,
                                   block_out_channels=(8, 16, 16, 16),
                                   down_block_types=('DownEncoderBlock2D',) * 4, up_block_types=('UpDecoderBlock2D',) * 4,
                                   sample_size=64, scaling_factor=0.18215).eval())
    save(vae, os.path.join(d, 'vae'), 'diffusion_pytorch_model.safetensors')
    os.makedirs(os.path.join(d, 'scheduler'), exist_ok=True)
    sch = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear',
                                 timestep_spacing='trailing' if kind != 'sd1' else 'leading', steps_offset=1)
    sch.save_config(os.path.join(d, 'scheduler'))
    cls = {'sd1': 'StableDiffusionPipeline', 'sd2': 'StableDiffusionPipeline', 'inpaint': 'StableDiffusionInpaintPipeline',
           'sdxl': 'StableDiffusionXLPipeline'}[kind]
    idx = {'_class_name': cls, '_diffusers_version': '0.41.0',
           'scheduler': ['diffusers', 'EulerDiscreteScheduler'],
           'text_encoder': ['transformers', 'CLIPTextModel'], 'tokenizer': ['transformers', 'CLIPTokenizer'],
           'unet': ['diffusers', 'UNet2DConditionModel'], 'vae': ['diffusers', 'AutoencoderKL']}
    if sdxl:
        idx['text_encoder_2'] = ['transformers', 'CLIPTextModelWithProjection']
        idx['tokenizer_2'] = ['transformers', 'CLIPTokenizer']
    json.dump(idx, open(os.path.join(d, 'model_index.json'), 'w'), indent=2)
    print(name, 'done')


for name, kind in (('tiny-sd', 'sd1'), ('tiny-sd2', 'sd2'), ('tiny-sdxl', 'sdxl'), ('tiny-inpaint', 'inpaint')):
    make(name, kind)
