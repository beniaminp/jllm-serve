#!/usr/bin/env python3
"""Writes a tiny random AnimateDiff motion adapter (a few hundred kB) in the diffusers folder layout, the fixture of the
video tests. It fits the UNet of the tiny Stable Diffusion 1.x pipeline of make-tiny.py (`tiny-sd`: blocks of 16, 32, 32
channels, one layer per block, 8 norm groups); the video pipeline takes `tiny-sd` as its base model.

    python -I scripts/diffusion/make-tiny-video.py OUT_DIR          (OUT_DIR/tiny-motion-adapter; needs torch, diffusers)

Weights are rounded to float16 and saved as float16, so what the Java side reads is exactly what diffusers computes with
once the file is loaded back as float32. The adapter uses two attention heads (the real one eight), two transformer
layers in the first block (the real one 1) and a table of 8 temporal positions (the real one 32).
"""
import json
import os
import sys

import torch
from diffusers import MotionAdapter
from safetensors.torch import save_file

out = sys.argv[1]
torch.manual_seed(4321)
d = os.path.join(out, 'tiny-motion-adapter')
os.makedirs(d, exist_ok=True)
adapter = MotionAdapter(block_out_channels=(16, 32, 32), motion_layers_per_block=1, motion_transformer_layers_per_block=(2, 1, 1),
                        motion_mid_block_layers_per_block=1, motion_transformer_layers_per_mid_block=1,
                        motion_num_attention_heads=2, motion_norm_num_groups=8, motion_max_seq_length=8,
                        use_motion_mid_block=True).eval()
with torch.no_grad():
    for n, p in adapter.named_parameters():
        if p.dim() == 1 and 'norm' in n:
            p.add_(torch.randn_like(p) * 0.1)
        elif p.dim() == 1:
            p.add_(torch.randn_like(p) * 0.05)
        else:
            p.mul_(2.0)  # the default initialisation is tiny: make the temporal layers matter
        p.copy_(p.half().float())
sd = {k: v.detach().half().contiguous() for k, v in adapter.state_dict().items()}
save_file(sd, os.path.join(d, 'diffusion_pytorch_model.safetensors'), metadata={'format': 'pt'})
adapter.save_config(d)
print(d, len(sd), 'tensors', json.load(open(os.path.join(d, 'config.json')))['_class_name'])
