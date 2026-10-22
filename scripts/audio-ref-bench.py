#!/usr/bin/env python3
"""Speed of the Python reference implementations on the same input as scripts/audio-bench.sh (docs/21-audio.md, "Measurements"):
transformers' Whisper (greedy, 30 s windows) on the CPU and on the Metal GPU through PyTorch (mps), and OuteTTS's language model
generating the same tokens through transformers.

usage: python3 -I scripts/audio-ref-bench.py stt MODEL_DIR AUDIO.wav [cpu|mps] [reps]
       python3 -I scripts/audio-ref-bench.py tts OUTETTS_DIR PROMPT_PREFIX [cpu|mps]     (PROMPT_PREFIX.prompt.json: see below)

Needs torch, transformers, soundfile. For tts, PROMPT_PREFIX.prompt.json is {"ids": [...]} (the engine writes one with
`AudioBench`/`TtsRef`-style tools; docs/21 describes the check) and the script times 250 greedy tokens.
"""
import json
import sys
import time

import soundfile as sf
import torch


def stt(model_dir, wav, device, reps):
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000, "use a 16 kHz file"
    model = WhisperForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()
    fe = WhisperFeatureExtractor(feature_size=model.config.num_mel_bins)
    seconds = len(audio) / 16000
    times = []
    for i in range(reps + 1):
        t0 = time.time()
        long_form = len(audio) > 480000
        if long_form:
            f = fe(audio, sampling_rate=16000, truncation=False, padding="longest", return_attention_mask=True, return_tensors="pt")
            kw = dict(attention_mask=f.attention_mask.to(device), return_timestamps=True, condition_on_prev_tokens=False)
        else:
            f = fe(audio, sampling_rate=16000, return_tensors="pt")
            kw = {}
        with torch.no_grad():
            model.generate(f.input_features.to(device), language="en", task="transcribe", do_sample=False, num_beams=1, **kw)
        if device == "mps":
            torch.mps.synchronize()
        if i > 0:
            times.append(time.time() - t0)
    times.sort()
    med = times[len(times) // 2]
    print("transformers %s on %s: %.2f s for %.1f s of audio, real-time factor %.3f (median of %d)" % (model_dir.split("/")[-1], device, med, seconds, med / seconds, reps))


def tts(model_dir, prefix, device):
    from transformers import AutoModelForCausalLM

    ids = json.load(open(prefix + ".prompt.json"))["ids"]
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()
    x = torch.tensor([ids]).to(device)
    for i in range(2):
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=250, do_sample=False, repetition_penalty=1.1, eos_token_id=151645, pad_token_id=151645)
        if device == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
    n = out.shape[1] - x.shape[1]
    print("transformers OuteTTS LM on %s: prompt %d tokens + %d generated in %.1f s (%.1f tok/s)" % (device, len(ids), n, dt, n / dt))


if __name__ == "__main__":
    if sys.argv[1] == "stt":
        stt(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "cpu", int(sys.argv[5]) if len(sys.argv) > 5 else 2)
    else:
        tts(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "cpu")
