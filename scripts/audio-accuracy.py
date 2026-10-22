#!/usr/bin/env python3
"""Speech-to-text accuracy against transformers on real recordings (docs/21-audio.md, 21.12): the 73 LibriSpeech utterances of
hf-internal-testing/librispeech_asr_dummy (481 s of read English), transcribed greedily by transformers and by jllm on the CPU
and on Metal; prints the word error rate of each against the reference text and how many transcripts are identical.

usage: python3 -I scripts/audio-accuracy.py <whisper dir, e.g. models/openai_whisper-base> [work dir]

Needs torch, transformers, soundfile, numpy, pyarrow (or polars) and librosa/scipy for resampling; a built engine
(`mvn -q -o test-compile`). The work directory (default /tmp/jllm-accuracy) keeps the downloaded parquet file and the clips.
"""
import io
import json
import os
import re
import subprocess
import sys
import urllib.request

import numpy as np
import soundfile as sf

URL = "https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/main/clean/validation-00000-of-00001.parquet"
model_dir = sys.argv[1]
work = sys.argv[2] if len(sys.argv) > 2 else "/tmp/jllm-accuracy"
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.makedirs(work + "/wav", exist_ok=True)
pq = work + "/validation.parquet"
if not os.path.exists(pq):
    urllib.request.urlretrieve(URL, pq)
import pyarrow.parquet as papq

rows = papq.read_table(pq).to_pylist()
refs = {}
for i, r in enumerate(rows):
    path = "%s/wav/u%02d.wav" % (work, i)
    refs["u%02d" % i] = r["text"]
    if not os.path.exists(path):
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:
            import scipy.signal
            x = scipy.signal.resample_poly(x, 16000, sr).astype(np.float32)
        sf.write(path, x, 16000, subtype="PCM_16")


def norm(s):
    return re.sub(r"[^a-z0-9' ]", "", s.lower().replace("-", " ")).split()


def edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def wer(hyp):
    errs = sum(edit(norm(refs[k]), norm(hyp[k])) for k in hyp)
    words = sum(len(norm(refs[k])) for k in hyp)
    return 100.0 * errs / words


# transformers
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizerFast

model = WhisperForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
tok = WhisperTokenizerFast.from_pretrained(model_dir)
fe = WhisperFeatureExtractor(feature_size=model.config.num_mel_bins)
hf = {}
for k in refs:
    x, _ = sf.read("%s/wav/%s.wav" % (work, k), dtype="float32")
    with torch.no_grad():
        g = model.generate(fe(x, sampling_rate=16000, return_tensors="pt").input_features, language="en", task="transcribe",
                           do_sample=False, num_beams=1)
    hf[k] = tok.decode(g[0], skip_special_tokens=True).strip()
print("transformers: WER %.2f %%" % wer(hf))

env = dict(os.environ, JAVA_HOME=os.environ.get("JAVA_HOME", "/opt/homebrew/opt/openjdk@26"))
for backend in ("cpu", "metal"):
    out = "%s/jllm-%s.json" % (work, backend)
    cmd = [env["JAVA_HOME"] + "/bin/java", "--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED",
           "-Djllm.metal.lib=target/native/libjllm_metal.dylib", "-cp", "target/classes:target/test-classes",
           "org.robomiri.jllm.AudioTranscribeDir", model_dir, work + "/wav", out, backend, "en"]
    r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        print(backend, "failed:", r.stderr[-300:])
        continue
    mine = json.load(open(out))
    same = sum(mine[k].strip() == hf[k].strip() for k in hf)
    print("jllm %-5s: WER %.2f %%, %d of %d transcripts identical to transformers  (%s)" % (
        backend, wer(mine), same, len(hf), r.stderr.strip().splitlines()[-1]))
