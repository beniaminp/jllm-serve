#!/usr/bin/env python3
"""Writes a tiny random Wan 2.1 text-to-video pipeline (a few hundred kB) in the diffusers folder layout, the fixture of the
Wan tests: a diffusion transformer (2 heads of 24 channels, 2 layers), a causal 3-D VAE and (when asked) a UMT5 text encoder.
Random weights, float16-rounded and saved as float16, so what the Java side reads is exactly what diffusers computes with.

    python -I scripts/diffusion/make-tiny-wan.py OUT_DIR          (OUT_DIR/tiny-wan; needs torch, diffusers, transformers)
"""
import json
import os
import sys

import torch
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from transformers import UMT5Config, UMT5EncoderModel
from safetensors.torch import save_file

out = sys.argv[1]
torch.manual_seed(2468)
d = os.path.join(out, 'tiny-wan')


def half_round(model):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() == 1 and ('norm' in n or 'gamma' in n):
                p.add_(torch.randn_like(p) * 0.1)
            elif p.dim() == 1 or 'scale_shift_table' in n:
                p.add_(torch.randn_like(p) * 0.05)
            p.copy_(p.half().float())
    return model


def save(model, path, name):
    os.makedirs(path, exist_ok=True)
    model.save_config(path)
    sd = {k: v.detach().half().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(path, name), metadata={'format': 'pt'})


dit = half_round(WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=24, in_channels=4, out_channels=4,
                                       text_dim=32, freq_dim=16, ffn_dim=96, num_layers=2, cross_attn_norm=True, qk_norm='rms_norm_across_heads',
                                       eps=1e-6, rope_max_seq_len=64).eval())
save(dit, os.path.join(d, 'transformer'), 'diffusion_pytorch_model.safetensors')
print('tiny-wan transformer', sum(p.numel() for p in dit.parameters()), 'parameters')

if os.environ.get('WAN_VAE', '1') == '1':
    vae = half_round(AutoencoderKLWan(base_dim=8, z_dim=4, dim_mult=[1, 2, 2, 2], num_res_blocks=1, temperal_downsample=[False, True, True],
                                      latents_mean=[0.1, -0.2, 0.3, -0.1], latents_std=[1.1, 0.9, 1.2, 0.8]).eval())
    save(vae, os.path.join(d, 'vae'), 'diffusion_pytorch_model.safetensors')
    print('tiny-wan vae', sum(p.numel() for p in vae.parameters()), 'parameters')


def make_tokenizer(path):
    """A small T5-style Unigram tokenizer (tokenizer.json of the same structure as umt5-xxl's: Replace, Metaspace, TemplateProcessing)."""
    from tokenizers import Regex, Tokenizer, models, normalizers, pre_tokenizers, processors
    os.makedirs(path, exist_ok=True)
    words = ['a', 'the', 'red', 'cat', 'dog', 'on', 'sofa', 'balloon', 'over', 'green', 'hill', 'moving', 'clouds', 'photo', 'of', 'and',
             'sitting', 'running', 'blue', 'sky', 'waves', 'sea']
    vocab = [('<pad>', 0.0), ('</s>', 0.0), ('<s>', 0.0), ('<unk>', 0.0)]
    seen = set(p for p, _ in vocab)
    chars = sorted(set(''.join(words) + 'abcdefghijklmnopqrstuvwxyz0123456789.,!?\'-'))
    for c in chars:
        vocab.append((c, -8.0 - 0.01 * len(vocab)))
    vocab.append(('\u2581', -3.0))
    for w in words:
        vocab.append(('\u2581' + w, -2.0 - 0.02 * len(w)))
        if len(w) > 3:
            vocab.append((w, -3.5 - 0.02 * len(w)))
            vocab.append((w[:len(w) // 2], -5.0))
            vocab.append((w[len(w) // 2:], -5.2))
    vocab.append(('ing', -4.0))
    vocab.append(('s', -6.5))
    out, seen = [], set()
    for p, sc in vocab:
        if p not in seen:
            seen.add(p)
            out.append((p, sc))
    tok = Tokenizer(models.Unigram(out, unk_id=3, byte_fallback=False))
    tok.normalizer = normalizers.Sequence([normalizers.Replace(Regex(' {2,}'), ' ')])
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement='\u2581', prepend_scheme='always', split=True)
    tok.post_processor = processors.TemplateProcessing(single='$A </s>', special_tokens=[('</s>', 1)])
    tok.save(os.path.join(path, 'tokenizer.json'))
    json.dump({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>', 'unk_token': '<unk>'}, open(os.path.join(path, 'special_tokens_map.json'), 'w'))
    json.dump({'tokenizer_class': 'T5Tokenizer', 'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>', 'unk_token': '<unk>', 'extra_ids': 0,
               'model_max_length': 512, 'clean_up_tokenization_spaces': True}, open(os.path.join(path, 'tokenizer_config.json'), 'w'))
    return len(out)


if os.environ.get('WAN_T5', '1') == '1':
    vocab_size = make_tokenizer(os.path.join(d, 'tokenizer'))
    cfg = UMT5Config(vocab_size=vocab_size, d_model=32, d_ff=64, d_kv=8, num_heads=4, num_layers=2, num_decoder_layers=2,
                     feed_forward_proj='gated-gelu', dense_act_fn='gelu_new', is_gated_act=True, relative_attention_num_buckets=32,
                     relative_attention_max_distance=128, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0, tie_word_embeddings=False)
    te = half_round(UMT5EncoderModel(cfg).eval())
    te_dir = os.path.join(d, 'text_encoder')
    os.makedirs(te_dir, exist_ok=True)
    te.config.to_json_file(os.path.join(te_dir, 'config.json'))
    sd = {k: v.detach().half().contiguous() for k, v in te.state_dict().items() if k != 'encoder.embed_tokens.weight'}
    save_file(sd, os.path.join(te_dir, 'model.safetensors'), metadata={'format': 'pt'})
    print('tiny-wan text encoder', sum(p.numel() for p in te.parameters()), 'parameters, vocab', vocab_size)
    os.makedirs(os.path.join(d, 'scheduler'), exist_ok=True)
    json.dump({'_class_name': 'UniPCMultistepScheduler', 'beta_end': 0.02, 'beta_schedule': 'linear', 'beta_start': 0.0001, 'flow_shift': 3.0,
               'num_train_timesteps': 1000, 'prediction_type': 'flow_prediction', 'use_flow_sigmas': True, 'solver_order': 2, 'timestep_spacing': 'linspace',
               'final_sigmas_type': 'zero', 'lower_order_final': True, 'predict_x0': True, 'solver_type': 'bh2'}, open(os.path.join(d, 'scheduler', 'scheduler_config.json'), 'w'), indent=1)
    json.dump({'_class_name': 'WanPipeline', '_diffusers_version': '0.41.0', 'scheduler': ['diffusers', 'UniPCMultistepScheduler'],
               'text_encoder': ['transformers', 'UMT5EncoderModel'], 'tokenizer': ['transformers', 'T5TokenizerFast'],
               'transformer': ['diffusers', 'WanTransformer3DModel'], 'vae': ['diffusers', 'AutoencoderKLWan']}, open(os.path.join(d, 'model_index.json'), 'w'), indent=2)
