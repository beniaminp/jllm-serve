#!/usr/bin/env python3
"""Regenerates src/test/resources/audio/ (the fixtures of org.robomiri.jllm.AudioTest) with the reference implementations.

usage: python3 -I scripts/audio-fixtures.py [--text-ref TEXTPROC.json] [--numbers-ref NUMVEC.json]

Needs torch, transformers, tokenizers, numpy (a venv with `pip install torch transformers`). Always written:

  tiny-whisper/            a random Whisper (d_model 32, 2+2 layers, 2 heads, 400 tokens, 1 s windows) as transformers saves it:
                           config.json, generation_config.json, model.safetensors, and a byte-level tokenizer.json with
                           Whisper's special tokens
  tiny-whisper-ref.json    transformers' output for it: the input log-mel (seeded), the encoder output and the decoder logits
                           of a token sequence
  mel-ref.f32              WhisperFeatureExtractor's log-mel (80 x 100, little-endian float32) of a 1.1 s synthetic signal
                           (formula in AudioTest.signal), the numbers the Java log-mel must reproduce

The optional files are the OuteTTS text front end's references (`outetext-ref.json`, from the Python `TextProcessor` of the outetts
package run over a list of sentences; the script keeps the input and the prompt text) and the number names
(`numbers-ref.json`, from `inflect`): pass the JSON the reference scripts wrote. `codec-codes.json` and `codec-ref.wav` (75 WavTokenizer
codes and the original PyTorch decoder's output for them, with the GGUF's f16 weights) are made from the `outetts` package's
`WavDecoder` by hand and only used when the codec's GGUF is on disk.
"""
import json
import os
import struct
import sys

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "test", "resources", "audio")
os.makedirs(OUT + "/tiny-whisper", exist_ok=True)

# ---------------------------------------------------------------- the tiny Whisper
torch.manual_seed(5)
cfg = WhisperConfig(vocab_size=400, num_mel_bins=80, encoder_layers=2, encoder_attention_heads=2, decoder_layers=2,
                    decoder_attention_heads=2, decoder_ffn_dim=64, encoder_ffn_dim=64, d_model=32, max_source_positions=50,
                    max_target_positions=40, pad_token_id=256, bos_token_id=256, eos_token_id=256, decoder_start_token_id=257,
                    suppress_tokens=[], begin_suppress_tokens=[], activation_function="gelu")
model = WhisperForConditionalGeneration(cfg).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "layer_norm" in name or "layernorm" in name:
            p.copy_(1.0 + 0.1 * torch.randn_like(p) if name.endswith("weight") else 0.1 * torch.randn_like(p))
        elif "embed_positions" in name and "encoder" in name:
            continue                                   # the fixed sinusoids
        else:
            p.copy_(torch.randn_like(p) * (0.5 if "embed_tokens" in name else 0.2))
d = OUT + "/tiny-whisper"
model.save_pretrained(d, safe_serialization=True)
json.dump({"decoder_start_token_id": 257, "eos_token_id": 256, "no_timestamps_token_id": 262, "prev_sot_token_id": 260,
           "max_initial_timestamp_index": 10, "suppress_tokens": [], "alignment_heads": [[1, 0], [1, 1]],
           "pad_token_id": 256, "bos_token_id": 256, "max_length": 40}, open(d + "/generation_config.json", "w"), indent=1)

alphabet = pre_tokenizers.ByteLevel.alphabet()
vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
specials = ["<|endoftext|>", "<|startoftranscript|>", "<|translate|>", "<|transcribe|>", "<|startofprev|>", "<|nocaptions|>",
            "<|notimestamps|>"] + ["<|%.2f|>" % (i * 0.02) for i in range(137)]
tok.add_special_tokens(specials)
assert tok.get_vocab_size() == 400, tok.get_vocab_size()
tok.save(d + "/tokenizer.json")

rng = np.random.default_rng(5)
mel = (rng.standard_normal((1, 80, 100)) * 0.5).astype(np.float32)
tokens = [257, 262, 5, 77, 130]
with torch.no_grad():
    enc = model.model.encoder(torch.from_numpy(mel)).last_hidden_state
    logits = model(encoder_outputs=(enc,), decoder_input_ids=torch.tensor([tokens])).logits
json.dump({"mel": mel[0].tolist(), "encoder": enc[0].numpy().tolist(), "tokens": tokens, "logits": logits[0].numpy().tolist()},
          open(OUT + "/tiny-whisper-ref.json", "w"), separators=(",", ":"))

# ---------------------------------------------------------------- the log-mel reference
n = int(1.1 * 16000)
t = np.arange(n, dtype=np.float64) / 16000
sig = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * (300 + 1500 * t) * t) + 0.05 * np.sin(2 * np.pi * 3100 * t))
sig = sig.astype(np.float32)
fe = WhisperFeatureExtractor(feature_size=80)
feat = fe(sig, sampling_rate=16000, return_tensors="np").input_features[0]          # 80 x 3000
with open(OUT + "/mel-ref.f32", "wb") as f:
    f.write(feat[:, :100].astype("<f4").tobytes())

# ---------------------------------------------------------------- copied references
args = sys.argv[1:]
for flag, dest, convert in (("--text-ref", "outetext-ref.json", None), ("--numbers-ref", "numbers-ref.json", None)):
    if flag in args:
        src = args[args.index(flag) + 1]
        data = json.load(open(src))
        if flag == "--text-ref":
            data = [{"input": e["input"], "text": e["text"]} for e in data]
        else:
            keys = sorted(data, key=lambda k: float(k))
            data = {k: data[k] for k in keys[::max(1, len(keys) // 400)]}
        json.dump(data, open(OUT + "/" + dest, "w"), ensure_ascii=False, separators=(",", ":"))
print("wrote", OUT)
