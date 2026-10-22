# 21. Audio: speech-to-text, text-to-speech and an audio editor

Code: `org.robomiri.jllm.audio` (`Wav`, `Resampler`, `LogMel`, `Fft`, `Dense`, `WhisperWeights`, `WhisperCpu`, `WhisperMetal`,
`WhisperTranscriber`, `WordAligner`, `OuteTts`, `OuteText`, `WavTokenizerDecoder`, `TimeStretch`, `AudioApi`), the Metal kernels
`src/main/resources/org/robomiri/jllm/audio/audio.metal`, the engine's routes `server/AudioRoutes.java`, the console's
`web/audio/*`, `web/tools/AudioTools.java` and `static/js/audio*.js`; settings `audio.*` in `jllm.properties`; tests `AudioTest`,
`kernel_tests.py` (the `wh_*` kernels), `jllm-web/src/test/java/org/robomiri/jllm/web/audio/AudioApiTest.java`; fixtures and how they were made in `scripts/audio-fixtures.py`;
speed in `scripts/audio-bench.sh`.

A language model reads and writes text. Sound is not text, but the same machinery reaches it from both sides:

```
speech-to-text   file.wav ─► samples ─► log-mel spectrogram ─► Whisper encoder ─► Whisper decoder ─► text (+ times)
text-to-speech   text ─► words + punctuation ─► OuteTTS language model ─► audio codes ─► WavTokenizer decoder ─► waveform
```

| | what | where |
|---|---|---|
| [21.1](#211-sound-as-numbers) | samples, the file formats, changing the sample rate | `Wav`, `Resampler`, `AudioDecoder` |
| [21.2](#212-the-log-mel-spectrogram) | the picture of a sound that Whisper looks at | `LogMel`, `Fft` |
| [21.3](#213-whisper-an-encoder-and-a-decoder) | the encoder, the decoder, cross-attention and two kinds of KV cache | `WhisperCpu`, `Dense` |
| [21.4](#214-how-whisper-is-told-what-to-do) | special tokens, language detection, timestamps, 30-second windows, fallback | `WhisperTranscriber` |
| [21.5](#215-word-timestamps) | which word was said when, from the decoder's attention | `WordAligner` |
| [21.6](#216-whisper-on-the-gpu) | the encoder and decoder as Metal kernels | `WhisperMetal`, `audio.metal` |
| [21.7](#217-text-to-speech-the-choice) | why OuteTTS + WavTokenizer | |
| [21.8](#218-from-text-to-a-prompt) | the text front end and the prompt | `OuteText`, `NumberWords`, `OuteTts` |
| [21.9](#219-from-codes-to-sound-the-wavtokenizer-decoder) | convolutions, a self-attention, ConvNeXt blocks and an inverse FFT | `WavTokenizerDecoder` |
| [21.10](#2110-speed-long-text-and-formats) | speed without changing the pitch, long texts, output formats | `TimeStretch`, `Speech`, `AudioEncoder` |
| [21.11](#2111-the-api-the-command-line-and-the-console) | endpoints, settings, commands, the Audio tab and the editor, chat tools | |
| [21.12](#2112-checked-against-the-references) | how every piece was checked | |
| [21.13](#2113-measurements) | speed, with the command lines | |
| [21.14](#2114-what-is-not-done) | limits | |

Each of the four features of this chapter (speech to text, word timestamps, text to speech, the audio editor with its API) is
explained in the same order: 🧸 the idea in plain words, 📐 how it works with the formulas and a worked example, why it was built
this way and not another way, the switches, the measurements and what they mean, and what is checked and what is not. The
sub-sections [21.6.1](#2161-speech-to-text-why-this-design-settings-and-limits),
[21.10.1](#21101-text-to-speech-why-this-design-settings-and-limits) and [21.11.1](#21111-the-audio-editor) gather the "why", the
settings and the limits of a feature in one place.

**The whole chapter in one picture** (shapes in brackets; `d` is Whisper's width, `Ld` its decoder depth):

```
SPEECH TO TEXT
 file (any rate, stereo)            Wav.read / ffmpeg ──► float[N] at the file's rate
   │  Resampler (polyphase sinc), average the channels
   ▼
 mono 16 kHz  float[M]              M = 16 000 · seconds
   │  + 30 s of zeros, STFT (400 / 160), 80 (or 128) mel filters, log10, floor, (x+4)/4               LogMel
   ▼
 log-mel  [80][3000 + …]            one column per 10 ms; a window is 3 000 columns = 30 s
   │  conv1 (k3, 80→d) GELU · conv2 (k3, stride 2, d→d) GELU · + sinusoidal positions
   ▼
 x  [1500][d]                       one row per 20 ms
   │  L × ( LN, self-attention, LN, MLP ) , LN                                      WhisperCpu | WhisperMetal
   ▼
 enc [1500][d] ──► cross-attention keys and values of every decoder layer, once per window: [Ld][2][1500][d]
                                               │
 prompt ids [n] ─► E[id] + P[pos]  [n][d] ─► Ld × ( self-attention, cross-attention, MLP ) ─► LN · Eᵀ ─► logits [51 865]
                                                                                                │  filters, arg-max or sample
                                                                                                ▼
                                       text + timestamp tokens ──► segments (and, with the alignment heads, words)

TEXT TO SPEECH
 text ─► OuteText: words + punctuation tokens ─► prompt (voice's reference words and codes + new words) [≈ 700 ids]
   │  OuteTTS = Qwen2 language model on the ordinary Engine (Q8_0, KV cache, Metal)
   ▼
 audio-code tokens  <|0|> … <|4095|>    75 per second, one per 13.3 ms
   │  WavTokenizerDecoder: codebook [4096][512] → convolutions + one self-attention + 12 ConvNeXt blocks → [T][1282]
   │  → 641 magnitudes and 641 phases per frame → inverse real FFT (1280) → Hann → overlap-add (hop 320)
   ▼
 waveform  float[320·T] at 24 kHz ─► trim silence ─► (WSOLA for speed) ─► wav / pcm (or ffmpeg)
```

---

## 21.1 Sound as numbers

🧸 A sound is air wiggling. A microphone turns the wiggle into a number: how far the air is pushed right now. Take 16 000
such numbers every second and you can draw the sound as a long line of dots, and play it back by wiggling a speaker the
same way. That is all a WAV file is: a title card ("16 000 numbers a second, 16 bits each, one microphone") and then the
numbers.

📐 **Samples.** A recording is `x[0], x[1], …` with `x[n]` the pressure at time `n / fs`, `fs` the **sample rate**. A 16-bit
PCM file stores `round(32767 · x[n])` as a little-endian integer; the readers in `Wav` give floats in `[-1, 1]`.

`Wav.read` understands RIFF/WAVE (PCM 8, 16, 24, 32 bit, IEEE float 32/64, A-law, µ-law, and `WAVE_FORMAT_EXTENSIBLE` around
any of these; a streamed file whose data size is `0` or `0xFFFFFFFF` is read to its end), AIFF / AIFF-C (PCM, `sowt`,
`fl32`/`fl64`; the sample rate is an 80-bit extended float) and Sun `.au`. Writing makes 16-bit PCM or 32-bit float WAV.
Anything else (MP3, FLAC, Ogg, M4A, WebM) is decoded by an `ffmpeg` executable when `audio.ffmpeg` finds one on the PATH; without it
the API says so (400) and the web console converts to WAV in the browser, which decodes whatever the browser plays.

**Resampling.** Whisper wants 16 kHz mono; a phone gives 48 kHz, a CD 44.1 kHz; the speech model produces 24 kHz. Changing the
rate is *band-limited interpolation*: the new sample at time `t` is the sum of the old samples weighted by a sinc, `sinc(u) =
sin(πu)/(πu)`, which is the ideal low-pass filter's impulse response. `Resampler` reduces the rate ratio by its gcd to
`L/M` (44 100 → 16 000 is `160/441`), builds one short filter per output phase `p/L`, so no sine is evaluated per sample, and
windows the sinc with a Kaiser window (β = 8.6, 24 zero crossings per side) so that the filter is finite. When the rate goes
down, the cut-off is lowered to 0.95 of the new Nyquist frequency `fs'/2` first: a 12 kHz tone sampled at 44.1 kHz must not
fold back down as a 4 kHz tone at 16 kHz (the test feeds one in and finds a residual of `4e-6`; a 1 kHz tone survives with an error of `2e-5`).
Stereo is averaged to mono.

*Worked example, 44 100 Hz to 16 000 Hz* (what `Resampler.resample` computes for one second of a phone-sized recording):

```
gcd(44100, 16000) = 100     L = 16000/100 = 160 output samples      M = 44100/100 = 441 input samples   (per period)
ratio = 160/441 = 0.3628    cutoff = min(1, ratio) · 0.95 = 0.3447   (in units of the input Nyquist 22 050 Hz  =  7 600 Hz)
half = ceil(24 zero crossings / 0.3447) = 70     taps = 2 · 70 = 140 per output sample     table = 160 phases × 140 weights
output j sits at input position j · 441/160 :   j = 1  →  2.75625 = 2 + 121/160    base = 2, phase = (1 · 441) mod 160 = 121
out[1] = Σ_{k<140} table[121][k] · in[2 − 70 + 1 + k]       (the weights of a phase add up to 1: unity gain at 0 Hz)
```

One second of audio costs `16 000 · 140 = 2.24 M` multiply-adds, which is why resampling is invisible in the timings of 21.13. The weights
are `sin(π c x)/(π x)` (the ideal low-pass with cut-off `c`) times a Kaiser window `I₀(β √(1 − (x/half)²)) / I₀(β)`, `β = 8.6`; normalising
each phase to sum 1 keeps a constant (DC) signal exactly constant. If the target is not a ratio with `L ≤ 4096` (44 100 → 16 001), the nearest of 4 096
phases is used instead.

---

## 21.2 The log-mel spectrogram

🧸 The ear does not hear single wiggles; it hears *which pitches are present*. A spectrogram is a picture of that: time goes to
the right, pitch goes up, and the colour says how loud that pitch is just then. The "mel" part squeezes the pitch axis the way
ears do: low sounds get many thin rows (we tell 100 Hz from 200 Hz easily), high sounds a few thick ones (7 000 Hz from
7 200 Hz is hard). The "log" part is the same idea for loudness: a sound ten times stronger does not sound ten times louder.
Whisper looks at 80 rows (128 for large-v3) and 3 000 columns (one every 10 ms) for every 30 seconds of speech.

📐 Exactly what `WhisperFeatureExtractor` (and OpenAI's `log_mel_spectrogram`) compute, which `LogMel` reproduces:

1. **STFT.** Cut the signal into frames of `n_fft = 400` samples (25 ms) every `hop = 160` samples (10 ms), after reflecting
   `n_fft/2 = 200` samples at both ends (so frame `t` is centred on sample `160 t`), multiply by the periodic Hann window
   `w[n] = ½ − ½ cos(2πn/400)` and take the discrete Fourier transform
   `X_t[k] = Σ_n w[n] x[160 t − 200 + n] e^{−2πi kn/400}`, `k = 0 … 200` (201 bins of 40 Hz). The last frame is dropped, so
   `N` samples give `⌊N/160⌋` frames: 480 000 samples (30 s) give 3 000.
2. **Power:** `P_t[k] = |X_t[k]|²`.
3. **Mel filter bank.** `M` triangular filters whose corners are equally spaced on the Slaney mel scale: linear below 1 kHz
   (`mel = 3 f / 200`), logarithmic above (`mel = 15 + ln(f/1000)·27/ln 6.4`). With 80 filters over 0 … 8 kHz the corners are
   0, 37.2, 74.5, 111.7, 149.0 … 7 408.5, 7 698.6, 8 000 Hz: filter 0 is centred at 37 Hz and 75 Hz wide, filter 40 at 1 722 Hz and 132 Hz
   wide, filter 79 at 7 699 Hz and 592 Hz wide. Each filter is scaled by `2 / (width in Hz)` ("Slaney normalisation") so that all of
   them have the same area. `S_t[m] = Σ_k F[m][k] P_t[k]`.
4. **Log and normalisation.** `L = log10(max(S, 1e-10))`, then `L = max(L, max(L) − 8)` (everything more than 80 dB below the
   loudest point of the recording is set to that floor) and `(L + 4) / 4` (roughly into `[-1, 1.5]`). The maximum is taken
   over the whole recording plus its 30 s of padding, which is why a long file is converted **once** and then cut into windows.

The transform is `n_fft = 400 = 2⁴ · 5²`, so `Fft` is a mixed-radix decimation-in-time FFT (radix 4, 2, 3, 5, and a plain DFT
for any larger prime factor), in double precision; 3 000 frames take ~100 ms. The test compares 80 × 100 values of a chirp with
`WhisperFeatureExtractor`'s: largest difference `3.6e-5` (the reference rounds to float32 in several places).

*Worked example: a 1 kHz tone of amplitude 0.5* (computed with numpy from the formulas above; the middle frame of one second of audio):

```
bin spacing = 16 000 / 400 = 40 Hz  →  1 000 Hz is exactly bin 25.   Σ w[n] over the Hann window = 200.
|X[25]| = A · Σw / 2 = 0.5 · 200 / 2 = 50          P[25] = 2 500          (the window leaks into the neighbours:  P[24] = P[26] = 625)
mel position of 1 kHz = 15  →  it falls between the corners of filters 25 (931.0, 968.2, 1005.6 Hz), 26 (968.2, 1005.6, 1045.0) and 27 (1005.6, 1045.0, 1085.9)
filter 26:  enorm = 2 / (1045.0 − 968.2) = 0.02604
            weight at 1000 Hz (bin 25) = (1000 − 968.2) / (1005.6 − 968.2) · enorm = 0.850 · 0.02604 = 0.02214
            weight at 1040 Hz (bin 26) = (1045.0 − 1040) / (1045.0 − 1005.6) · enorm = 0.127 · 0.02604 = 0.00330
            S[26] = 0.02214 · 2500 + 0.00330 · 625 = 57.36                  log10 S[26] = 1.759   ← the loudest value of the clip
filters 25 and 27:  S = 23.15 and 13.59  → log10 = 1.365 and 1.133;  filter 24: 3.70 → 0.569
every other filter: S = 0  →  log10(max(0, 1e-10)) = −10   →  the floor  max − 8 = 1.759 − 8 = −6.241  lifts it
normalised  (L + 4)/4:   filter 26 → 1.440     a silent filter → (−6.241 + 4)/4 = −0.560
```

So everything more than 80 dB below the loudest point (the 8 in `max − 8` is 8 decades of *power*) is flattened to the same value, here −0.56. A recording ten times
louder in amplitude (100 times in power) moves every `log10` value up by 2, but the floor moves with it, so the picture the encoder sees changes little with the
recording level except for what falls under the floor.

**Why this design.** The numbers must be the ones Whisper was trained on: a different window or mel scale gives the encoder a picture it was never trained on, and nothing
reports the difference. So `LogMel` follows `WhisperFeatureExtractor` line by line (periodic Hann, reflection padding, Slaney scale *and* Slaney
normalisation, the global floor) and is checked against it (21.12). Alternatives that were not taken: a power-of-two FFT (n_fft 512) would be faster
but is not what the model was trained on; the log-mel covers a range of 10⁸ in power, so the transform runs in double precision (a choice made for safety; float32 was not
tried) and costs 16-47 ms per 30 s window in the measured runs (21.13), a few per cent of a transcription at most. The maximum is taken over
the whole recording, not per window, so a quiet window of a long file is not boosted into noise.

---

## 21.3 Whisper: an encoder and a decoder

🧸 Whisper is two readers. The first one (the **encoder**) looks at the whole 30-second picture and writes notes about it, one
note for every 20 ms. The second one (the **decoder**) writes the words one at a time, like the chat models do, but before
each word it looks back at the first reader's notes ("what was being said around here?") as well as at the words it has
already written. Looking at the notes is **cross-attention**.

📐 Sizes (HF `config.json`):

| model | `d_model` | encoder layers | decoder layers | heads | mel bins | parameters | encoder, one 30 s window |
|---|---|---|---|---|---|---|---|
| tiny | 384 | 4 | 4 | 6 | 80 | 39 M | 18.5 GMAC |
| base | 512 | 6 | 6 | 8 | 80 | 74 M | 43.7 GMAC |
| small | 768 | 12 | 12 | 12 | 80 | 244 M | 172 GMAC |
| medium | 1024 | 24 | 24 | 16 | 80 | 769 M | 569 GMAC |
| large-v3 | 1280 | 32 | 32 | 20 | 128 | 1 550 M | 1 136 GMAC |
| large-v3-turbo | 1280 | 32 | 4 | 20 | 128 | 809 M | 1 136 GMAC |

(a GMAC is 10⁹ multiply-adds; per layer `12 d² T` for the projections and feed-forward plus `2 T² d` for attention, `T = 1500`,
plus the convolutions.)

**Encoder.** Input `mel` `[M][3000]`.

1. `conv1`: a 1-D convolution over time, kernel 3, padding 1, `M → d` channels, then GELU. `conv2`: kernel 3, **stride 2**, padding 1,
   `d → d`, GELU. The stride halves 3 000 frames to **1 500 positions** (20 ms each). A convolution is a matrix product over
   *patches*: row `t` of the patch matrix is `[mel[c][t+k−1]]` over channels `c` and taps `k`, so each convolution is one
   `T × (3M) · (3M) × d` product (`Dense`, and on the GPU `wh_im2col_*` + `wh_gemm`).
2. Add the **sinusoidal positions** (stored in the checkpoint): `PE[p][i] = sin(p · 10000^{−i/(d/2−1)})`, `PE[p][d/2+i] = cos(…)`.
3. `L` times, with pre-norm residuals: `x ← x + Wo · attention(LN(x))`, `x ← x + W₂ · GELU(W₁ · LN(x))`. Attention has `h` heads of
   width 64; **`k_proj` has no bias**, `q_proj`, `v_proj` and `out_proj` do; the query is scaled by `64^{−½}`. LN is
   `(x − μ)/√(σ² + 10⁻⁵) · γ + β`. GELU is the exact `x · ½(1 + erf(x/√2))`; the float `erf` is the rational approximation
   Eigen and XLA use (relative error ~1e-7), so the CPU and Metal kernels share it.
4. A final LN. The result is `1500 × d`.

**Decoder.** Tokens `t₀ t₁ …` (a GPT-2 style byte-level BPE vocabulary of 51 865 / 51 866 entries including ~1 500 special
tokens) become `E[tᵢ] + P[i]` (a *learned* position table of 448 rows). Each of `Ld` layers has three residual blocks:

```
x ← x + Wo · SelfAttention(LN₁(x))         causal: token i sees tokens 0..i, through the decoder's own KV cache
x ← x + Wo · CrossAttention(LN₂(x), enc)   q from x, k and v from the encoder output: 1500 positions, the same for every token
x ← x + W₂ · GELU(W₁ · LN₃(x))
```

and the logits are `LN(x) · Eᵀ` (the embedding matrix again: tied weights). Two caches therefore exist (a cross-attention one that is
computed **once per window** for all `Ld` layers right after the encoder: `Ld · 2 · 1500 · d` floats, 110 MB for small; and a self-attention
one that grows with the text), and the per-token work is small compared with the encoder (for small: 12 layers of
`4d² + 2d² + 2·d·3072·… ≈ 8 M` parameters each, ~100 M parameters read per token and 40 M for the logits).

**CPU implementation.** `Dense.mul` computes `out[t][r] = bias[r] + Σ_k x[t][k] W[r][k]` with a register tile of **4 weight
rows × 4 tokens** (16 vector accumulators; each weight vector is loaded once for four tokens and each activation vector once for
four rows), parallel over groups of weight rows (the engine's `ThreadPool`). Attention is two products per head and block of 32
queries: `S = Q Kᵀ` with the keys as the "weights", a vectorised softmax (`VectorKernels.softmaxExp`), then `O = P V` with **V stored
transposed** (`[head][d][position]`) so that this is again row-by-row dot products of the same kernel. A checkpoint that
is too big to keep as floats (`audio.resident-mb`; large-v3-turbo's 1.6 GB of f16) stays memory-mapped and is converted four rows at a
time into a thread-local scratch, which the 4 × 4 tile amortises over all tokens of an encoder call.

*Worked example of cross-attention* (one head of width 2 instead of 64, six encoder positions instead of 1 500, three decoder rows; the same numbers
are reused in 21.5). Row `i` asks, `q_i · k_p / √2` is the score for the encoder position `p` (20 ms each), the softmax turns the scores into weights, and the
row's context is the weighted sum of the values:

```
k_p (p = 0..5)  =  [1,0] [1,0.2] [0,1] [0.1,1] [-1,0] [-1,-0.2]          v_p  =  [1,0] [1,0] [0,1] [0,1] [-1,0] [-1,0]
q for the row that writes "Hello" = [2,0]   scores  1.414  1.414  0      0.141  -1.414 -1.414   weights 0.379 0.379 0.092 0.106 0.022 0.022   context [ 0.71, 0.20]
q for the row that writes " world" = [0,2]  scores  0      0.283  1.414  1.414  0      -0.283   weights 0.081 0.108 0.334 0.334 0.081 0.061   context [ 0.05, 0.67]
q for the row of <|endoftext|>     = [-2,0] scores -1.414 -1.414 0      -0.141 1.414  1.414    weights 0.023 0.023 0.095 0.082 0.389 0.389   context [-0.73, 0.18]
```

The first word listens to the first 40 ms, the second to the middle, the end-of-text row to the end: that is the diagonal that 21.5 turns into times. In
the real model 6 to 20 heads of width 64 do this in each of `Ld` layers, and the softmax runs over 1 500 positions: `Ld · 2 · 1 500 · d` floats of keys and
values per window (110 MB for small, as computed above) are made once, right after the encoder, and every decoded token only computes its query.

---

## 21.4 How Whisper is told what to do

🧸 The same decoder can transcribe English, translate German into English, guess which language it hears, and put times on
sentences. Which of these it does depends only on the first few words you give it, a little secret handshake made of special
tokens, like "now start", "this is French", "just copy it down".

📐 `WhisperTranscriber` follows OpenAI's `whisper.transcribe` and `DecodingTask`:

* **Prompt per window:** `<|startofprev|> previous text <|startoftranscript|> <|fr|> <|transcribe|> [<|notimestamps|>]`. English-only
  checkpoints (vocabulary 51 864) have neither language nor task tokens.
* **Language detection:** feed `<|startoftranscript|>` alone, take the logits of the ~99 `<|xx|>` tokens, softmax among them; the
  most likely one is used for the whole recording (the response's `language_probabilities` shows the top five).
* **Logit filters** at every step, in this order: at the first step suppress the blank `" "` and `<|endoftext|>`; always suppress the
  checkpoint's `suppress_tokens` (digits-in-brackets, music symbols ...) and the control tokens; and with timestamps:
  `<|notimestamps|>` is suppressed; timestamps appear **in pairs** (a timestamp after a text token must be followed by another one
  or by `<|endoftext|>`; after two timestamps comes text); timestamps never decrease and a segment is never empty; the first token
  must be a timestamp no later than `max_initial_timestamp_index` (1 s); and *if the probability mass of all timestamp tokens is above
  that of the best single text token, a timestamp is forced*. The log-probabilities used afterwards are those of the filtered logits.
* **Timestamps** are tokens `<|0.00|> … <|30.00|>` in steps of 20 ms (the encoder's resolution): `<|3.40|>` has id `50365 + 170`.
* **Windows.** Audio longer than 30 s is read in windows; after a window the read position moves to the end of the last
  *complete* segment (the last timestamp pair), or by the whole window when the text ended with a single timestamp (silence after
  it), so no word is cut in the middle. The previous windows' text, if any (up to 223 tokens), is the next window's prompt
  (`audio.stt.condition-on-previous`); a window decoded at a temperature above 0.5 resets it.
* **Fallback** (`audio.stt.fallback`): a window whose text compresses too well (zlib ratio above 2.4: a repeated phrase) or whose
  average token log-probability is below −1 is decoded again at temperature 0.2, 0.4 … 1.0 (the best of 5 samples by mean
  log-probability). A window whose `<|nocaptions|>` / `<|nospeech|>` probability after `<|startoftranscript|>` is above 0.6 and
  whose average log-probability is below −1 is skipped as silence.
* **Greedy** decoding takes the arg-max of the filtered logits; the tests compare the ids with `transformers`' `generate`
  token for token (21.12).

*Worked example: from tokens to segments and the next window* (the code is `WhisperTranscriber.transcribe`; `stride = 3000 mel frames / 1500 encoder positions = 2` mel
frames per timestamp step, one step = 2 · 10 ms = 20 ms). Suppose the decoder wrote, for the window that starts at `seek = 0`, with `T(x)` the timestamp token
`<|x|>` of id `timestampBegin + x/0.02` (`timestampBegin` is 50 365 for the multilingual checkpoints, `<|3.40|>` = 50 365 + 170):

```
tokens:   T(0.00)  " Hello" " world" "."  T(2.40) T(2.40)  " How" " are" " you" "?"  T(5.20) T(5.20)  " I" " am"      (cut: the 30 s limit or a token limit)
pairs of consecutive timestamps end a segment:      segment 1 = T(0.00) … T(2.40)       segment 2 = T(2.40) … T(5.20)       the rest is not a complete segment
output:   [0.00 → 2.40] "Hello world."    [2.40 → 5.20] "How are you?"          (a segment's start and end are its first and last timestamp + the window's offset)
seek   += tokens[last − 1] − timestampBegin  =  260 steps · stride 2  =  520 mel frames  = 5.20 s       → the next window starts at 5.20 s, so " I am" is read again with its audio
```

If the text had ended with a *single* timestamp (`… "?" T(5.20)` and then `<|endoftext|>`), the model says "nothing more is said in this window":
the last segment is closed and `seek` moves by the whole window (30 s, or what is left of the file). If no timestamp pair exists at all, the whole window is one
segment from `seek` to the last timestamp (or the window's length). A guard (`seek ≤ previousSeek` → move by the window) makes sure the loop always advances.

*Worked example: why a timestamp is forced.* After a text token Whisper's logits cover 51 865 tokens, 1 501 of which are timestamps. Toy numbers: the best single text
token has probability 0.20 (log-probability −1.61), the 1 501 timestamp tokens together 0.55 (log-probability −0.60). Since −0.60 > −1.61 the rule sets every text logit to −∞ and a timestamp is
written, although no single timestamp is as likely as the best word: the probability that "a time stamp comes here" is spread over many neighbouring times, each small, and a plain arg-max would keep choosing the best
word instead (this is OpenAI's rule, and the tests compare the resulting ids with `transformers`, 21.12).

*Worked example: the fallback test.* The compression ratio is `bytes(text) / bytes(zlib(text))` (Java's `Deflater`, the same algorithm as Python's `zlib`):
a normal sentence (`"Hello, world. This is a test of the speech synthesis system."`, 62 bytes) compresses to 66 bytes (ratio 0.94, the zlib header outweighs the gain on short text), but the loop
`"Thank you. "` repeated 20 times (220 bytes) to 22 bytes: ratio 10.0, far above the limit 2.4, so the window is decoded again at temperature 0.2 (five samples, the one with
the best mean log-probability wins), then 0.4 … up to 1.0 until neither test fires. The second test is the mean log-probability `Σ log p(token) / (n + 1)` over the written
tokens and the end-of-text: a text of 40 tokens whose log-probabilities add up to −52 has a mean of −52/41 = −1.27, below the limit −1.0, which usually means the model is guessing
(noise, music, a language it was not told). The third rule is the opposite: if `<|nocaptions|>` / `<|nospeech|>` has probability above 0.6 after `<|startoftranscript|>` *and* the mean
log-probability is below −1, the window is silence and is skipped, not retried.

---

## 21.5 Word timestamps

🧸 Whisper never says "this word starts here", but while writing a word its decoder *looks* at the part of the sound where that
word is: some of its attention heads point at the right place on the encoder's timeline. Collect where they look for every
word and you can read off the times, like a trail of footprints.

📐 `timestamp_granularities[]=word` runs the OpenAI algorithm (`whisper.timing`):

1. Feed `[<|startoftranscript|> lang task, <|notimestamps|>, text tokens…, <|endoftext|>]` once through the decoder and keep the
   cross-attention probabilities of the checkpoint's **alignment heads** (`alignment_heads` in `generation_config.json`: a few
   `(layer, head)` pairs found by the authors to be monotonic). Row `i` of a head is the attention of the position that *predicts*
   text token `i`.
2. Keep the first `⌈frames/2⌉` columns (the audio that exists; the rest of the window is padding) and renormalise each row
   (a softmax over fewer keys is the original row divided by its partial sum). Standardise every column across the token rows
   (`(a − μ)/σ`), median-filter each row over a window of 7 columns (reflect padding) and average the heads.
3. **Dynamic time warping** finds the cheapest monotone path through `−matrix` (steps: diagonal, down, right) from the top-left to
   the bottom-right; the column at which the path first enters row `i` is the start of token `i`, in units of 20 ms.
4. Tokens are grouped into words (a word starts at a token with a leading space or at punctuation, or at every character for Chinese,
   Japanese, Thai, Lao, Burmese, Cantonese); prepended and appended punctuation is merged into its neighbour; a word that lasts more
   than twice the median at a sentence end (or after one) is shortened; the first and last word of a segment are pulled
   to the segment's times when they are far off.

The test fixture checks that the path is monotone and that the rows are probabilities; the quality against `transformers`' own
word timestamps is in 21.12 (the same words, starts within a few 20 ms steps of each other).

*Worked example, the same toy attention as in 21.3* (the weights of one alignment head for the rows of " Hello", " world" and `<|endoftext|>` over six 20 ms
columns; no padding here, so the renormalisation of step 2 changes nothing, and the median filter of width 7 is left out because the toy has six columns):

```
attention weights p[row][column]                      standardised per column  z = (p − mean over rows) / std over rows
 Hello  0.379 0.379 0.092 0.106 0.022 0.022            1.398  1.377 −0.718 −0.599 −0.882 −0.822
 world  0.081 0.108 0.334 0.334 0.081 0.061           −0.512 −0.409  1.414  1.409 −0.516 −0.586
 eot    0.023 0.023 0.095 0.082 0.389 0.389           −0.886 −0.968 −0.696 −0.810  1.398  1.408

cost = −z; DTW accumulates  D[i][j] = cost[i][j] + min(D[i−1][j−1], D[i−1][j], D[i][j−1])  (D[−1][−1] = 0, the other borders +∞):
 Hello  −1.398 −2.774 −2.057 −1.458 −0.575  0.247
 world  −0.886 −2.366 −4.189 −5.598 −5.082 −4.496
 eot     0.000 −1.398 −3.492 −4.788 −6.996 −8.403         trace back from the bottom-right corner (diagonal / up / left):

 column (20 ms)         0     1     2     3     4     5
 " Hello"               ■     ■
 " world"                           ■     ■
 <|endoftext|>                                   ■     ■        the path first enters row i at column 0, 2, 4
```

The start of row `i` is the column where the path first enters it, `column × 20 ms`: " Hello" 0.00 s, " world" 0.04 s, end of text 0.08 s. A word's end is the start of the
next row, so " Hello" lasts from 0.00 to 0.04 s and the end-of-text row only provides the last word's end. (Standardising each column makes the columns comparable: the path follows where a
token is attended to *more than the other tokens are*, not where attention is large in absolute terms.) After DTW the clean-up of step 4 uses the median word duration (capped at 0.7 s) to
shorten a word that is longer than twice the median at a sentence end or after one, which is where Whisper's attention smears into pauses.

**Why this design.** The other ways to time words are a forced aligner (a second, CTC model per language, which jllm does not carry), or asking the decoder for timestamp tokens per word
(the timestamp tokens are only 20 ms-grained *sentence* marks and the model does not write them per word). The alignment heads are Whisper's own: no extra model, no extra download, and the
cost is one more pass of the decoder over the finished text (no encoder run, the window's cross-attention keys and values are still there). It costs a full-length pass with the attention
weights returned (`Session.crossAttention`), which is why it is only done on request (`timestamp_granularities[]=word`). On Metal the pass runs on the CPU decoder: `WhisperMetal`'s `crossAttention` copies the cross-attention
keys and values of the window back from the GPU (`Ld · 2 · 1 500 · d` floats) and hands them to a CPU session. (`WhisperNet.Session.crossAttention` is an optional method; a device
without it would return segments without words.) Accuracy is the accuracy of the heads: measured within 0.02-0.06 s of `transformers` (21.12), not better than what a person would
call exact. **Settings**: only the request field `timestamp_granularities[]=word` (CLI `--words true`); the alignment heads come from the checkpoint's `generation_config.json`.
**Limits**: not for streaming partial results; the times of a word spoken over music or in a crowd are as uncertain as the decoder's attention; Chinese, Japanese, Thai, Lao, Burmese and Cantonese are timed per character.

---

## 21.6 Whisper on the GPU

🧸 The encoder is a big pile of multiplication tables: 172 billion multiplications for 30 seconds with `small`. A graphics
chip is built to do exactly that pile in parallel, so the encoder (and the notes the decoder will look at) go to the GPU in one go,
and for the bigger models the decoder follows.

📐 `audio.metal` is a separate Metal library (`GpuCtx`, on its own `Metal` instance), transpiled together with `kernels.metal` by
the emulator, so each kernel has a numpy test (`scripts/metal-emu/kernel_tests.py`, section "audio.metal"):

| kernel | does |
|---|---|
| `wh_gemm_{hh,hf,fh,ff}` | `out[t][r] (+)= act(bias[r] + Σ_k x[t][k] w[r][k])`, 64 weight rows × 32 tokens per threadgroup of 4 simdgroups, `simdgroup_float8x8` accumulators; `w` half (`h`) or float (`f`), the weight tile in threadgroup memory half or float; flags bias, GELU, accumulate; batched over `z` with strides (all heads of the attention products); any `rows`, `cols`, `ntok` |
| `wh_layernorm` | one threadgroup per row, two passes with `simd_sum` |
| `wh_im2col_mel`, `wh_im2col_s2` | the patch matrices of the two convolutions |
| `wh_split_heads` | `q`,`k`,`v` `[n][d]` to `Q[h][n][64]` (scaled), `K[h][cap][64]` and `Vᵀ[h][64][cap]` at a position |
| `wh_softmax` | rows in place, optionally causal from a position; the tail is zeroed |
| `wh_embed_{h,f}` | `x[t] = E[token t] + P[pos0 + t]` |

An encoder layer is 12 dispatches (LN, three projections, split, `S = Q Kᵀ`, softmax, `O = P V`, output projection with
accumulate, LN, feed-forward ×2); the whole encoder, the cross-attention keys and values of every decoder layer and (for a GPU decoder)
nothing else form **one command buffer**. The weights are used where they are: an f16 checkpoint's tensors are wrapped as
Metal buffers without a copy (`Metal.wrapRange`), an f32 checkpoint's are converted once to half (the default) or wrapped as
they are (`-Djllm.audio.metalExact=true`: float weight tiles; the encoder output then agrees with the CPU's to `2e-3` on the tiny
model, `2.5e-3` with half tiles). Attention products always use float tiles: scores are not rounded to half.

The **decoder** on the GPU (`audio.metal.decoder`; `auto` = models of `d_model ≥ 768`) is one command buffer per call of up to 64 tokens:
embedding, then per layer 3 LNs, 5 projections (the cross-attention keys and values were made with the encoder), two batched attentions
against the session's own KV caches, the feed-forward block, then the last token's LN and the tied output projection (`wh_gemm`
with one valid token: 51 865 rows). Smaller models decode on the CPU from the keys and values the GPU wrote (copied out of the
shared buffers), because there a step costs more in dispatches than the CPU needs to compute it.

*The tile of `wh_gemm`* (`audio.metal`, `WH_BR = 64` weight rows by `WH_BT = 32` tokens per threadgroup; 128 threads = 4 simdgroups; each simdgroup keeps `acc[8]`, two token
blocks by four row blocks of 8 × 8 floats):

```
 output tile  out[token][row]                       one k step (32 of the `cols` inputs), repeated cols/32 times:
                  rows 0-31     rows 32-63            1. every thread stages 16 weights  (4-wide loads, half or float) of weight row  tid>>1
 tokens  0-15 │ simdgroup 0 │ simdgroup 1 │              and 8 activations (float)        of token       tid>>2   into threadgroup memory
 tokens 16-31 │ simdgroup 2 │ simdgroup 3 │           2. barrier
                                                      3. each simdgroup, 4 times: load 4 weight blocks and 2 activation blocks, 8 multiply-accumulates
 threadgroup memory: weight tile 64×32 (half: 4 KB, float: 8 KB) + activations 32×32 floats (4 KB)       4. barrier
 epilogue: tile to threadgroup memory, then out = (ACC ? out : 0) + gelu?(acc + bias?)  with the bounds checked for ragged shapes
```

An encoder layer is therefore six of these GEMMs (q, k, v, out, fc1, fc2) plus two batched ones for the attention (`z` = the head). The input positions are uploaded into `x`
first and the second convolution *accumulates* on top of them (`ACC`), so adding the sinusoidal positions costs no kernel. The encoder is cut into command buffers of
`COMMIT_LAYERS = 3` layers: macOS kills a command buffer that holds the GPU for seconds ("Impacting Interactivity"), and a pass that was killed is idempotent (it starts from the uploaded
mel), so it is simply tried again after 250 ms, 500 ms, … (`WhisperMetal.pause`).

### 21.6.1 Speech to text: why this design, settings and limits

**Why this design.**

* *A separate small network stack, not the chat engine.* The chat engine's `Architecture` is a causal decoder-only transformer; Whisper adds a convolutional front end, a bidirectional encoder, a decoder with
  *cross*-attention to a fixed memory, learned decoder positions, LayerNorm with biases and exact GELU. Bending all of that into `Architecture` would have put audio-only switches into every hot path of the LLM; instead
  `WhisperNet` (encoder + decoder sessions) has two implementations, `WhisperCpu` and `WhisperMetal`, and `WhisperTranscriber` (the protocol of 21.4) does not know which one it talks to.
* *The Metal kernels are a library of their own* (`audio.metal`, loaded by `GpuCtx` into its own `Metal` instance), so a Whisper pass never shares a command stream, buffer indices or start-up checks with the language model.
* *One tiled GEMM for everything* (`wh_gemm`) instead of a kernel per layer type: convolutions are turned into patch matrices (`wh_im2col_*`) and the attention products are the same kernel batched over heads. The cost is
  that it is not as tuned as whisper.cpp's kernels (21.13: 4.7 s against 2.75 s for `small` on the GPU); the gain is one kernel with a numpy test for every shape it meets.
* *Half-precision weight tiles by default, float tiles on request* (`-Djllm.audio.metalExact=true`): llama.cpp's matmul also rounds staged weights to half; the exact mode keeps float32 checkpoints exact at the price of
  memory and speed. Scores and attention products always use float tiles.
* *The decoder moves to the GPU only for `d_model ≥ 768`.* A decoding step of `tiny` or `base` needs more GPU dispatches than the CPU needs time to do the arithmetic; measured (21.13), `tiny` is 0.99 s with the decoder on the CPU
  and 1.34 s on the GPU, `small` is 7.61 s against 4.71 s. `auto` encodes that crossover.
* *Fallback and previous-text conditioning are on* because OpenAI's reference has them on: they are what stops Whisper from looping on silence or repeating one phrase (21.13 shows what the fallback costs when it fires: 41 s against 9.5 s
  for `large-v3-turbo`'s last window).
* *Weights stay as the checkpoint stores them.* An f16 checkpoint is wrapped as Metal buffers without a copy; on the CPU the decoder's weights are held as floats, the encoder's only when the CPU runs it, and a model whose floats would
  exceed `audio.resident-mb` stays memory-mapped and is converted four rows at a time (21.3).

**Settings** (all in `jllm.properties`, validated at start-up by `AudioSettings`; defaults in 21.11):

| switch | default | change it when |
|---|---|---|
| `audio.backend` | `auto` | `cpu` to keep Whisper off the GPU (a server started with `--backend cpu` already does); `metal` to fail instead of falling back |
| `audio.metal.decoder` | `auto` | `on` to force the decoder onto the GPU for a small model, `off` for a large one on a shared GPU; `mac-validate.sh` has a row for each |
| `audio.threads` | `0` | to leave cores for the language model (0 = `backend.threads`, then all) |
| `audio.resident-mb` | `1500` | lower on a small machine (the model is then converted per use, slower); raise to keep a larger model as floats |
| `audio.stt.fallback`, `audio.stt.condition-on-previous` | `true`, `true` | `fallback=false` when the retries are too slow and a looping window is acceptable (turbo: 9.5 s instead of 41 s); `condition-on-previous=false` when a wrong word early on keeps being repeated |
| request: `temperature` | 0 | above 0 samples at that temperature only, with no fallback |
| `audio.max-seconds`, `audio.max-upload-mb` | `3600`, `100` | to bound the work and memory of one request |
| `-Djllm.audio.metalExact=true` | off | to compare the GPU with the CPU float path to 2e-3 instead of 2.5e-3 on the tiny model (21.6) |

**Measured**: 21.13 (speed against whisper.cpp and transformers, word error rate on LibriSpeech).

**Checks and limits.** Log-mel, encoder, decoder logits, the token ids of `generate` and the word timestamps are compared with `transformers` (21.12); every `wh_*` kernel has a numpy test; word error rates on 73 real clips
equal the reference's. Not done: beam search, per-word log-probabilities, speaker diarisation, partial results while a window is still being decoded (the stream is per finished window), and the 30-second window is fixed (as in OpenAI's code).
`large-v3-turbo` takes the fallback path on a window of silence after speech (21.13); on `tiny` and `base` the GPU decoder is slower than the CPU one.

---

## 21.7 Text-to-speech: the choice

🧸 To make a computer speak you can teach a model "text in, sound out" directly, or you can notice that a chat model already knows how
to continue a story, and let it continue a story whose words are *sound pieces*. OuteTTS does the second thing: its sentences look
like `the<|t_0.12|><|1090|><|512|>...` where every `<|number|>` is one 13-millisecond piece of sound from a fixed catalogue of 4 096
pieces. A second model (the codec) turns the pieces back into a waveform.

📐 The candidates and why OuteTTS 0.3 (500M) + WavTokenizer was chosen:

| | what it needs | fit |
|---|---|---|
| **OuteTTS 0.3 500M** (CC-BY-SA 4.0) + **WavTokenizer large-75** (MIT) | a Qwen2 LM (the engine runs it unchanged: Q8_0 weights, KV cache, Metal) + a ~62 M-parameter convolutional decoder (convolutions, one self-attention, LayerNorm, GELU, an inverse FFT) | the whole language-model half is existing code; the codec is ~400 lines; 10 voices ship with the OuteTTS package; English, German, French (and Japanese, Korean, Chinese, see 21.14) |
| OuteTTS 1.0 0.6B / Llama-OuteTTS 1.0 1B | Qwen3 / Llama LM + DAC codec (24 kHz, two codebooks, much larger decoder: ~30-40 GMAC per second of audio) and a different prompt with per-word duration, energy and pitch features | better quality, but the codec costs ~8× more and the prompt builder is a project of its own |
| Kokoro-82M | StyleTTS2: text encoder, duration/prosody predictor with LSTMs, ISTFTNet decoder, and a phonemizer (espeak-ng / misaki dictionaries) | very good and tiny, but none of it is a transformer LM and the phoneme front end needs data files or native code |
| Pocket TTS, Qwen3-TTS | flow-matching / multi-codebook code predictors with a Mimi-style codec | too far from the engine for this round |

OuteTTS 0.2 uses the same prompt format family (v1); the 0.3 checkpoints add the punctuation tokens (`<|comma|>`, `<|period|>` ...).
The WavTokenizer used here is the 75-tokens-per-second large speech model (24 kHz output, hop 320), read from
llama.cpp's `ggml-org/WavTokenizer` GGUF (F16 weights, 130 MB).

---

## 21.8 From text to a prompt

🧸 Before the model can continue the story it needs the beginning in its own spelling: lower case, numbers spelled out
("21" becomes "twenty one"), no accents, and the commas and full stops turned into little tokens glued to their word. Then
a *reference voice* is shown: a few seconds of someone speaking, written as the same story pieces, so the model carries on in
that voice.

📐 `OuteText` is the Python package's `TextProcessor` without MeCab, `uroman` and `anyascii`:

1. delete `'` and `` ` `` ("don't" → "dont", the way the model was trained);
2. split into letter runs, digit runs and single symbols (that is what MeCab's `-Owakati` does to Latin text, including its
   dictionary entries that join a quotation mark and an adjacent `.,!?` into one token), join them again with two spaces between
   words and none before punctuation;
3. transliterate to ASCII (accents dropped: `ä → a`, `ß → ss`, `Æ → AE`, `ø → o`, `Ł → L`), dashes to spaces, lower case;
4. spell out numbers like `inflect` (`NumberWords`: "one hundred and one", "one thousand, two hundred and thirty-four", "one thousand and ten"; the
   hyphen is removed by step 5, so `twenty-four` is read `twentyfour`: the model's convention);
5. delete everything but letters and `. ! ? , " „ ¡ ¿ … 。 ！ ？ ， ؟`;
6. each word keeps the punctuation tokens that stood before and after it: `Hello,` → `hello<|comma|>`, `"Hi!"` →
   `<|double_quote|>hi<|exclamation_mark|><|double_quote|>`.

Checked on 20 sentences (digits, quotes, ellipses, accents, e-mail addresses ...) and 420 numbers against the Python code: identical.
Chinese, Japanese and Korean need MeCab and `uroman`; those characters are dropped here.

The prompt (OuteTTS interface v2):

```
<|im_start|>\n<|text_start|>{speaker words}<|space|>{new words}<|text_end|>\n<|audio_start|>\n
{speaker word 1}<|t_0.40|><|7|><|1090|>…<|space|>\n{speaker word 2}<|t_0.24|><|12|>…<|space|>\n
```

where the words are joined by `<|space|>`, each speaker word carries its duration `<|t_x.xx|>` and the codes of that stretch
of its recording, and the model continues `{new word 1}<|t_0.31|><|…|>…<|space|>\n{new word 2}…` until `<|audio_end|>`.
Every token of the form `<|0|> … <|4099|>` is a codec frame (the codebook has 4 096 entries); the time tokens and words are
bookkeeping. The 10 shipped voices (`en_male_1`, `en_female_1`, `de_male_1`, `fr_female_1` ...) are the package's speaker profiles
(`src/main/resources/org/robomiri/jllm/audio/outetts-speakers.json`, 26 kB). A voice is chosen with the `voice` field; a speaker built from your own recording would
need the WavTokenizer *encoder* and a forced aligner (not implemented).

The language model is loaded by `ModelLoader` like a chat model (`audio.tts.quant` = `q8` by default, `audio.tts.ctx` = 8192: a
voice costs ~700 prompt tokens and a second of speech ~90 generated ones) and sampled with `audio.tts.temperature` (0.1),
top-k 40, top-p 0.9 and the HF-style repetition penalty 1.1 over the prompt and the output, as the Python package does.

*Worked example of the text front end* (`OuteText.process` + `promptText`, run on these inputs): `Hello, world!` becomes `hello<|comma|><|space|>world<|exclamation_mark|>` and
`I have 21 cats.` becomes `i<|space|>have<|space|>twentyone<|space|>cats<|period|>` (digits spelled out, upper case dropped, the hyphen of `twenty-one` removed, the full stop
glued to its word as a token).

*Worked example of a voice.* A voice is a recording written as the same tokens. `en_male_1` (`outetts-speakers.json`) says "Their application for asylum has been accepted since they're victims of political
persecution." as 28 words with 515 codec codes in total, i.e. 515 / 75 = 6.87 s; the first entries are `their` (0.16 s, 12 codes: 12 / 75 = 0.16, so the duration token is just the code count over
75), `application` (0.72 s, 54 codes), `for` (0.19 s, 14 codes). The prompt for `Hello, world!` in this voice is therefore

```
<|im_start|>\n<|text_start|>their<|space|>application<|space|>for … victims…persecution<|period|><|space|>hello<|comma|><|space|>world<|exclamation_mark|><|text_end|>\n<|audio_start|>\n
their<|t_0.16|><|165|><|72|><|1785|><|1238|><|1055|><|1661|><|1746|><|873|>…<|space|>\n
application<|t_0.72|><|638|><|1086|><|1663|><|1490|><|860|><|1666|><|1318|><|1276|>…<|space|>\n   … (all 28 words) …<|space|>\n
```

and the model continues with `hello<|comma|><|t_0.31|><|…|>…<|space|>\nworld<|exclamation_mark|><|t_…|><|…|>…` until `<|audio_end|>`. A voice costs between 228 (`en_male_3`) and 515 (`en_male_1`) code tokens plus its words, duration tokens and separators; the prompt of the checked example in 21.12 was 729 tokens. With the default
`audio.tts.ctx = 8192` such a prompt leaves `8192 − 729 − 8 = 7 455` tokens, more than the 4 096 that one piece may write at most (`maxNew = min(4096, ctx − prompt − 8)`).

*The retry rule.* A sampled run (temperature 0.1 is not greedy) sometimes ends after a few frames, writes silence, or never stops. After each run the codes are decoded, trimmed, and
`spoken = trimmed length − 0.32 s` (the margins `trimSilence` keeps) must be at least `0.08 s · words` and the run no longer than `2.0 s · words + 3.0 s`; for the 12-word sentence of the benchmark
that window is 0.96 s to 27 s. Otherwise the piece is generated again with the seed `seed + 7919 · attempt`, up to 3 attempts (a greedy run, `temperature = 0`, is never repeated).

---

## 21.9 From codes to sound: the WavTokenizer decoder

🧸 Each of the 4 096 pieces is a point in a 512-dimensional space (a row of a table). Put the points for 75 pieces side by side and
you have a picture of "what the sound is like" for every 13 ms. A stack of small neural layers redraws that picture as a
*spectrogram with phases*, and an inverse Fourier transform turns the spectrogram back into the wiggles of the air.

📐 `WavTokenizerDecoder` (Vocos backbone, `ISTFTHead`), `T` frames in, `320 T` samples out at 24 kHz:

1. `x[t] = codebook[code_t]` (`4096 × 512`).
2. `embed`: Conv1d `512 → 768`, kernel 7, padding 3.
3. `pos_net`: ResBlock, ResBlock, **self-attention**, ResBlock, ResBlock, GroupNorm. A ResBlock is `x + conv₂(swish(GN(conv₁(swish(GN(x))))))` with
   kernel-3 convolutions; GroupNorm has 32 groups of 24 channels with statistics over the channels of a group *and all frames*
   (ε = 10⁻⁶); swish is `x · σ(x)`. The attention is one head of width 768 over all `T` frames, `softmax(Q Kᵀ / √768) V`, with `Q K V` and the
   output as 1×1 convolutions: the only place where the whole utterance mixes, `T² · 768` multiply-adds.
4. A LayerNorm whose scale and shift were picked from a table by the "bandwidth id" 0 (baked into the GGUF).
5. 12 **ConvNeXt blocks**: `x ← x + γ ⊙ (W₂ · GELU(W₁ · LN(depthwiseConv₇(x))))`, `W₁: 768 → 2304`, `γ` a learned per-channel scale; then a final LN.
6. `ISTFTHead`: a linear layer `768 → 1282` = 641 log-magnitudes `m` and 641 phases `φ`; the spectrum of frame `t` is
   `exp(min(m, ln 100)) · (cos φ + i sin φ)` (clipped at magnitude 100), its **inverse real FFT** (1280 points, the imaginary parts of DC and
   Nyquist ignored) is multiplied by the Hann window and added into the output at `320 t`; the sum is divided by the sum of the squared
   windows that overlap there (Vocos' "same" padding: `(1280 − 320)/2 = 480` samples at both ends are cut off, so exactly `320 T` remain).

Compared with the original PyTorch modules (`WavDecoder` of the `outetts` package) on the same codes with the same f16-rounded weights:
every stage agrees to 112-123 dB SNR (float rounding), the waveform to 107 dB; against the f32 original the GGUF's f16 weights
cost 60.6 dB (inaudible). The decoder runs on the CPU (21.13: 0.050 s per second of speech, an order of magnitude below the language model's cost).

*Worked example of the inverse transform* (`WavTokenizerDecoder.istft`, 75 frames = 1 s of codes): frame `t` has 641 complex bins `exp(min(m, ln 100)) · (cos φ + i sin φ)`; their
inverse real FFT gives 1 280 samples, multiplied by the periodic Hann window and added at offset `320 t`. The output buffer is `(75 − 1) · 320 + 1 280 = 24 960` samples long; Vocos'
"same" padding removes `(1 280 − 320)/2 = 480` samples at both ends: **24 000 samples = 1.000 s at 24 kHz**, i.e. exactly 320 samples per code. Four frames overlap at every interior sample (the
window is four hops long), and the sum of their squared windows is the constant `Σ_{k=0}^{3} sin⁴(θ + kπ/4) = 4 · 3/8 = 1.5` (checked numerically at two offsets), so the division by that envelope
makes the overlap-add an exact inverse of a windowed analysis. The model asks for 4 096 distinct codes = 12 bits per frame, 75 frames per second: the whole speech signal is described by 900 bits per second.

Codes the language model emits above 4 095 (the tokenizer has `<|4096|> … <|4099|>`, the codebook has 4 096 rows) are clamped to 4 095 (`Math.min(4095, …)` in `backboneRun`).

---

## 21.10 Speed, long text and formats

* **Speed** (`speed` 0.25-4 of `/v1/audio/speech`) changes the duration, not the pitch: `TimeStretch` is a waveform-similarity overlap-add
  (WSOLA). Frames of 20 ms (Hann window) are laid down every 10 ms; frame `k` is cut from the input around `k · 10 ms · speed`, at the
  offset within ±5 ms where its beginning best continues the samples that followed the previous frame in the input (normalised
  cross-correlation), so voiced speech stays in phase. A 440 Hz tone stretched by 0.5, 1.5 and 2 stays at 440 Hz.
* **Silence.** OuteTTS starts every utterance with about half a second of silence (the model learned to leave a lead-in) and the codec begins
  with a small transient (an offset of ~0.007 for 40 ms: a click when played from the first sample). `audio.tts.trim-silence` (on) cuts the windows of
  10 ms quieter than 8 % of the loudest one, keeping 0.12 s before the speech and 0.2 s after, with a 5 ms fade at the ends.
* **Long text.** `Speech.synthesizeLong` splits at sentence ends (then at commas and spaces) into pieces of at most ~220 characters,
  generates each separately with the same voice (each starts from the speaker prompt, so the voice does not drift), and joins them with 250 ms
  of silence; `audio.tts.max-chars` (4096, OpenAI's limit) bounds the request.
* **Formats.** `wav` (16-bit PCM, mono, 24 kHz) and `pcm` (raw 16-bit little endian) are written in Java; `mp3`, `opus`, `aac`, `flac` go through
  `ffmpeg` when present (400 otherwise, naming the setting).

*Worked example of WSOLA* (`TimeStretch.stretch`, 24 kHz, `speed = 1.5`): `hs = 240` samples (10 ms), frame `n = 480` (20 ms), search range `±120` samples (±5 ms). One second of speech (24 000 samples) becomes
`outLen = 16 000` samples, i.e. `frames = ⌈16 000/240⌉ = 67` frames. Frame `k` is cut from the input around `round(k · 240 · 1.5) = 360 k`, shifted by the `d ∈ [−120, 120]` that maximises
`Σ_i in[c + i] · in[natural + i] / √(Σ_i in[c + i]²)` over its first 240 samples, where `natural` is where the previous frame's *source* continued. The frames are windowed with
`w[i] = ½ − ½ cos(2π(i + ½)/480)`, whose two halves add up to exactly 1 at 50 % overlap (`w[100] + w[340] = 1.0000`), so no normalisation is needed. The choice of `d` is what keeps a voiced sound in phase; with `d = 0` always (plain
overlap-add) the overlapping frames of a voiced sound can be out of phase and partly cancel, heard as a warble. The browser editor has the same method with 25 ms frames and a ±6 ms search for edits of recordings (21.11.1).

*Worked example of the silence trim* (`Speech.trimSilence`): RMS of every 10 ms window; threshold `max(0.003, 0.08 · loudest)`; a clip whose loudest window is 0.2 has threshold 0.016. The first and
last windows above it bound the speech; the output starts 12 windows (0.12 s) before the first and ends 20 windows (0.2 s) after the last, with a 5 ms linear fade at both ends. That is why `spoken` in the retry rule subtracts 0.32 s.

### 21.10.1 Text to speech: why this design, settings and limits

**Why this design.**

* *Reuse the language-model engine for the hard half.* OuteTTS is a Qwen2 model, so the part that costs 75 tokens per second of speech runs on the same `Engine` as a chat model: Q8_0 or Q4_0 weights, the KV cache, the Metal kernels, the sampler.
  The alternatives in 21.7 (Kokoro, OuteTTS 1.0, Pocket/Qwen3-TTS) needed new network types or an 8 times bigger codec.
* *The codec stays on the CPU* (62 M parameters, 0.050 s per second of speech in 21.13): it is under a tenth of the language model's time, so a GPU port would not have helped the real-time factor.
* *Q8_0 for the speech language model by default* (`audio.tts.quant`): `jllm.properties` says 4 bits visibly hurt speech for a 0.5B model; the measured round trip (21.13, 12 sentences) found Q4_0 within noise (5.96 % against 6.62 % word error),
  so the safe default stayed and `q4` is a documented option for speed (RTF 0.81 against 1.01).
* *Reference voices as data, not code.* A voice is the package's speaker profile (words + codes) in a 26 kB JSON file, the prompt builder is ~40 lines, and every voice is verified at load (its words must equal what `OuteText` makes of its text, else it is skipped with a message).
* *Long text in pieces of ~220 characters*, each starting from the speaker prompt, joined with 250 ms of silence: the model is trained on utterances of a few seconds to a few tens of seconds, a piece never needs more context than `audio.tts.ctx`, and the first
  words can be played before the last are written. The cost is a pause of fixed length between sentences, whatever the punctuation.
* *Sampling at temperature 0.1 with top-k 40 and top-p 0.9 and a repetition penalty of 1.1 over prompt and output*: the Python package's defaults; greedy is available with `temperature=0` (the first 108 steps equal `transformers`, 21.12).

**Settings** (`jllm.properties`, 21.11 lists the defaults):

| switch | default | change it when |
|---|---|---|
| `audio.tts-model`, `audio.codec-repo`, `audio.codec-file` | empty, `ggml-org/WavTokenizer`, `WavTokenizer-Large-75-F16.gguf` | to load OuteTTS at start; the codec only to point at another copy of the same model |
| `audio.tts.quant` | `q8` | `q4` for ~20 % less time (RTF 0.81), `q2k`/`q6k` are accepted too (not measured) |
| `audio.tts.ctx` | `8192` | lower to save KV memory if texts are short; at least 1024; a voice costs ~700 tokens, a second of speech ~90 |
| `audio.tts.voice` | `en_male_1` | the voice when a request names none |
| `audio.tts.trim-silence` | `true` | `false` to keep the model's lead-in and the codec's start click exactly as generated |
| `audio.tts.temperature`, `.top-k`, `.top-p`, `.repetition-penalty` | `0.1`, `40`, `0.9`, `1.1` | `temperature=0` for repeatable output; raise for livelier but less stable speech |
| `audio.tts.max-chars` | `4096` | longer inputs are a 400 |
| request: `voice`, `speed` (0.25-4), `response_format` | the setting, 1, `wav` | `speed` is WSOLA, the pitch stays |

**Measured**: 21.13 (RTF 1.01 for Q8_0, 0.81 for Q4_0, codec 0.050; round-trip word error 6.62 % / 5.96 %).

**Checks and limits.** The prompt ids equal the Hugging Face tokenizer's, the text front end equals the package's on 20 sentences and 420 numbers, the codec equals the PyTorch modules at 107 dB, the first 108 greedy steps equal `transformers` (21.12).
Not done: speaker cloning, Chinese / Japanese / Korean text, other voices than the ten shipped (English, German, French), expressive control (emotion, speaking rate per word), and a GPU codec.
The pause between pieces is fixed; a name or an abbreviation the front end spells letter by letter is spoken as the model learned it.

---

## 21.11 The API, the command line and the console

**OpenAI endpoints** (the same in `./jllm serve` — `AudioRoutes`, with models loaded by `audio.stt-model` / `audio.tts-model` — and in the
web console — `AudioController`; both call `AudioApi`):

```sh
curl localhost:8090/v1/audio/transcriptions -F file=@talk.wav -F model=whisper-1 -F response_format=verbose_json \
     -F 'timestamp_granularities[]=word'                       # json | text | srt | vtt | verbose_json
curl localhost:8090/v1/audio/translations   -F file=@german.wav -F response_format=text      # into English
curl localhost:8090/v1/audio/speech -H 'Content-Type: application/json' \
     -d '{"input":"Hello there.","voice":"en_female_1","response_format":"wav","speed":1.1}' -o hello.wav
```

`model` may be any name when one model of that kind is loaded. Form fields: `file`, `model`, `language` (code or English name; default:
detect), `prompt` (names and jargon to prime the decoder), `response_format`, `temperature` (0 = greedy with fallback; above 0 that temperature only),
`timestamp_granularities[]` (`word` adds `words` to the verbose JSON, at the top level and in every segment), and `stream=true`: the answer is then
a server-sent-event stream, one `{"type":"transcript.text.segment","id","start","end","text"}` per segment as soon as its 30 s window is decoded, a
`{"type":"transcript.progress","progress"}` after each window, a last `{"type":"transcript.text.done","text","language","duration","transcript":{…verbose_json…}}`
and `data: [DONE]` (the console shows long recordings as they are read). Errors are OpenAI's
`{"error": {"message", "type", "code"}}` (`model_not_found` 404 when nothing of that kind is loaded, 400 for a bad format, voice, speed,
language, an unreadable file or audio longer than `audio.max-seconds`, 413 above `audio.max-upload-mb`). Additions of the console: `GET /api/audio/models`
(loaded audio models with their kind, languages / voices), `GET /api/audio/voices`, and a clip store (`POST /api/audio/files`, `GET`, `DELETE
/api/audio/files/{id}`) used by the chat tools and the player. `GET /v1/models` lists audio models with `kind` `speech-to-text` / `text-to-speech`.

**Loading** is the Models page's *Load* (or `jllm.web.autoload=openai/whisper-small,OuteAI/OuteTTS-0.3-500M`): after the download the
checkpoint is recognised (`config.json` with `model_type: whisper`, or a Qwen2 checkpoint whose tokenizer has `<|audio_start|>` and
`<|text_start|>`) and handed to `AudioService`; the page lists it with its kind. `./jllm transcribe --model openai/whisper-small --audio clip.wav
[--language de] [--translate true] [--format srt] [--words true]` and `./jllm speak --model OuteAI/OuteTTS-0.3-500M --prompt "Hello" --voice
en_male_1 --out hello.wav [--speed 1.2]` do the same from a terminal and print the real-time factor.

**Settings** (`jllm.properties`, validated at start-up, edited in the console's Settings tab):

| key | default | meaning |
|---|---|---|
| `audio.backend` | `auto` | `cpu`, `metal` or `auto` for Whisper |
| `audio.metal.decoder` | `auto` | decoder on the GPU too (`auto`: d_model ≥ 768) |
| `audio.threads` | `0` | CPU threads of the audio models (0 = `backend.threads`, then all) |
| `audio.resident-mb` | `1500` | heap for Whisper weights as floats; bigger models stay mapped and are converted per use |
| `audio.ffmpeg` | `auto` | ffmpeg for other audio formats and mp3/opus/aac/flac output (`off` or a path) |
| `audio.max-seconds`, `audio.max-upload-mb`, `audio.tts.max-chars` | `3600`, `100`, `4096` | limits |
| `audio.stt-model`, `audio.tts-model` | empty | models `./jllm serve` loads at start |
| `audio.codec-repo`, `audio.codec-file` | `ggml-org/WavTokenizer`, `WavTokenizer-Large-75-F16.gguf` | the codec |
| `audio.tts.quant`, `audio.tts.ctx` | `q8`, `8192` | the speech language model's weights and context |
| `audio.tts.voice` | `en_male_1` | the voice when a request names none |
| `audio.tts.trim-silence` | `true` | cut the silence around the generated speech |
| `audio.tts.temperature`, `.top-k`, `.top-p`, `.repetition-penalty` | `0.1`, `40`, `0.9`, `1.1` | its sampling |
| `audio.stt.fallback`, `audio.stt.condition-on-previous` | `true`, `true` | Whisper's decoding behaviour (21.4) |

**The Audio tab** (`audio.js`, `audio-editor.js`, `audio-edit.js`): *Speech to text* records from the microphone (an AudioWorklet writes PCM,
encoded as WAV in the browser) or takes a file of any format the browser decodes, shows the transcript with clickable timestamped segments,
and exports TXT, SRT, VTT or JSON; *Text to speech* has voice and speed controls, a player and a download; the **editor** shows the
waveform (canvas, zoom, selection, playhead), cuts, copies, pastes, deletes, trims, inserts silence, fades, changes the gain, normalises,
reverses, removes silence, changes speed by resampling (or by WSOLA), mixes a second clip in, has undo and redo and exports WAV, all in the browser
with the Web Audio API on `Float32Array`s (the pure functions are in `audio-edit.js`, tested by `node jllm-web/src/test/js/audio-edit-test.js`).
With models loaded it can also transcribe a selection, replace a selection with generated speech, and **edit by text**: the clip is transcribed with word
times, you edit the transcript, and the changed words are spoken by the TTS model and spliced in with short crossfades (deleted words are cut out; unchanged
audio is kept as it is).

**In the chat:** a microphone button dictates a message (records, transcribes, appends the text); a paperclip stores a clip and puts `[audio:ID]` in
the message; replies get a *read aloud* button; `[audio:ID]` in a message or in a tool result is shown with a player. The built-in tools
`transcribe_audio(audio_id, language?, translate?, timestamps?)` and `speak(text, voice?, speed?)` give a chat model ears and a voice (the `speak` result is
`[audio:ID]`, which the chat plays).

**Why this API and these defaults.**

* *OpenAI's shapes, so existing clients work.* `/v1/audio/transcriptions`, `/translations` and `/speech` take the same fields and return the same JSON (`json`, `text`, `srt`, `vtt`, `verbose_json`), the same SSE events for `stream=true` and
  the same error object, and `model` may be any name when one model of that kind is loaded. Both servers (the engine's `AudioRoutes` and the console's `AudioController`) call one `AudioApi`, so the two cannot drift apart.
* *ffmpeg is an external program, called with a fixed command line.* A file upload is untrusted input; jllm runs `ffmpeg -nostdin -protocol_whitelist file,pipe -format_whitelist <audio demuxers> -i <file> -vn -f f32le -ac 1 -ar 16000 …`
  (`AudioDecoder`), so a crafted "audio" file cannot make it open a network address or another file. Without ffmpeg only WAV, AIFF and AU are decoded in Java and everything else is a clear 400, never a guess.
* *Limits are settings, not constants* (`audio.max-seconds`, `audio.max-upload-mb`, `audio.tts.max-chars`): an hour-long recording is allowed by default (3 600 s), because a transcription is read in 30 s windows (the log-mel of the whole file is
  `80 × 100 × seconds` floats: 29 million floats, 115 MB, for an hour).
* *Console-only additions* (`/api/audio/models`, `/voices`, `/files`) keep the OpenAI endpoints clean: they exist for the Models page, the player and the chat tools.

### 21.11.1 The audio editor

🧸 The editor is a small sound-studio in the browser tab. You see the sound as a wavy ribbon, select a stretch with the mouse, and cut it, copy it, make it quieter, fade it, or play it back. Every action is
written on an "undo" list, so a mistake costs one key press. With a speech model and a speech-to-text model loaded it can also do things a plain studio cannot: write down what was said in a stretch, say a new sentence in its place, or let you
*edit the transcript like a document*: delete a word from the text and the word is cut out of the recording; retype it and the computer speaks the new word and splices it in.

📐 **Data model.** A clip is `{sampleRate, channels: [Float32Array, …]}`, all channels the same length, samples in `[-1, 1]`. Every function in `audio-edit.js` is pure: it returns a new clip and never changes its argument,
so the editor can keep the earlier clips as undo snapshots (at most 60 of them or 600 MB, whichever comes first: `UNDO_MAX`, `UNDO_BYTES` in `audio-editor.js`). Ranges are half open, `[a, b)` in frames. The waveform is drawn from a min/max pyramid with one entry per
256 frames (`buildPeaks`, `peaksFor`), so a long clip zooms smoothly. The formulas of the operations:

| operation | what it computes |
|---|---|
| gain `dB` | `g = 10^(dB/20)` (−6 dB gives 0.501), every sample of the range times `g` |
| fade in / out | `i = 0 … n−1`, `u = i/(n−1)` (`1 − u` for out); linear `g = u` or cosine `g = ½ − ½ cos(π u)`; for `n = 5` the cosine gains are 0, 0.146, 0.5, 0.854, 1 |
| normalize | `g = 10^(target/20) / level`, `level` the peak or the RMS of the range (`g = 1` when the range is silent); with RMS the peaks can pass 0 dBFS (the editor warns, the export clamps) |
| soft limiter | for `|v| > knee` (0.9): `sign(v) · (knee + (1 − knee) · tanh((|v| − knee)/(1 − knee)))`, smooth and bounded by 1 |
| mix | `out[at + i] += other[i] · 10^(dB/20)`; the clip grows if the second one ends later; the second one is first converted to the first one's rate and channel count |
| find / remove silence | level = the peak over windows of 10 ms; a run below `thresholdDb` (−40) lasting at least `minSec` (0.5 s) is a silence; removal keeps `keepSec` (0.1 s, half at each edge) and joins the edges with a 5 ms equal-power cross-fade |
| equal-power cross-fade over `x` frames | `t = (i+½)/x`, `out = a · cos(tπ/2) + b · sin(tπ/2)`; `cos² + sin² = 1`, so two uncorrelated signals keep their power (a linear fade would dip about 3 dB in the middle) |
| change speed by resampling | read the range as if recorded at `rate · factor`, resample back to `rate`: duration divided by `factor`, **pitch times factor** (like a tape) |
| time stretch (WSOLA) | frames of 25 ms (Hann, 50 % overlap), a search of ±6 ms for the offset that continues the previous frame (cross-correlation of the first half): duration divided by `factor`, pitch kept; a coarser search (stride 2 or 4) for audio longer than 60 s or 180 s |
| resample | windowed sinc (Lanczos, 8 lobes; a table of 2 048 entries per unit of the argument, linear interpolation), the kernel stretched by `rate_new/rate_old` when going down so that nothing above the new Nyquist frequency remains |
| speech range | 20 ms RMS every 10 ms; windows within 32 dB of the loudest are active; active runs closer than 0.12 s form a cluster; a cluster shorter than 0.08 s at either end is dropped when another cluster is next to it (the start click and burst of TTS output) |

**Edit by text** (`planTextEdit`, `applyTextEdit`, `diffWords` in `audio-edit.js`, `etPreview` in `audio-editor.js`). Worked example (the printed output of those functions) for the words `the` 0.00-0.20 s, `quick` 0.25-0.60, `brown` 0.65-1.00, `fox` 1.10-1.45, as
Whisper's word timestamps would give them:

```
1. diffWords: a longest-common-subsequence diff of the old and the new words, compared without case and punctuation ("Hello," = "hello"); the common prefix and suffix are cut off first.
2. planTextEdit: every change becomes a region [cutA, cutB) of the old clip and the text to say there:
   "the slow brown fox"         replace  quick → slow    cutA = 0.25 s (start of "quick"), cutB = 0.60 s (its end), say "slow"
   "the brown fox"              delete   quick           cutA = 0.25, cutB = max(0.60, min(start of "brown" = 0.65, 0.60 + 0.25)) = 0.65   (the pause after it goes too, at most 0.25 s, so no double pause stays)
   "the quick brown fox jumps"  insert   at the end      cutA = cutB = 1.45 (end of "fox"), say "jumps"
   "the quick red brown fox"    insert   in the pause    cutA = cutB = (0.60 + 0.65)/2 = 0.625, say "red"
3. every cut point is moved to the quietest spot within ±0.15 s (snapToValley: the 5 ms RMS there must be at least 12 dB under the neighbourhood's average, else the cut stays where the word times put it,
   because recognizer word boundaries are only good to about 50 ms), never crossing the previous region.
4. each piece of new text is spoken by the TTS model (cached per model, voice, speed and text), trimmed to the speech (trimToSpeech, 12 ms of margin); a replacement is stretched with WSOLA by the ratio
   length(piece) / length(cut), clamped to 0.6 … 1.6, so the new word takes about the old word's time; an insertion gets 50 ms of silence after it.
5. applyTextEdit: the kept audio between regions is untouched; every join is an equal-power cross-fade of 8 ms. The preview plays the result; Apply makes it one undo step.
```

*Replace a selection with speech* is the same idea for one region: the text is spoken, trimmed (`trimToSpeech`, 20 ms) and, with the fit mode `stretch` (WSOLA) or `speed` (resampling), fitted to the *spoken part* of the selection with a ratio clamped to 0.5-2 (a message appears when the clamp applied), then spliced in with 6 ms cross-fades.

**Why this design.**

* *All editing runs in the browser, on `Float32Array`s.* The server never sees an edit, only the stretch to transcribe (as a 16 kHz WAV) or the text to speak, so the server stays stateless and a clip is not uploaded for every cut. The price is browser memory:
  a mono hour at 44.1 kHz is 635 MB as floats, hence the 600 MB cap on the undo list, and decoding is `Web Audio`'s, which handles whatever the browser can play (the server need not have ffmpeg for that path).
* *Pure functions with a node test.* `audio-edit.js` has no DOM, so it runs in node: `node jllm-web/src/test/js/audio-edit-test.js` (25 tests: slicing, rate conversion, fades, cross-fade power, silence, WSOLA length and pitch, the diff and the plan).
* *Word times decide where to cut, the waveform decides exactly where.* Whisper's word times are good to a few 20 ms steps (21.12); cutting right there would clip phonemes, so each cut is snapped to a valley. When no valley is deep enough the cut is left alone.
* *The speech is the model's, not the speaker's.* A new word is spoken by one of the ten shipped voices, so a splice into a recording of someone else is audible; speaker cloning would need the WavTokenizer encoder (21.14).

**Settings.** The editor has no `jllm.properties` key of its own. It needs a loaded speech-to-text model (for *transcribe* and *edit by text*) and a text-to-speech model (for *replace* and *edit by text*), loaded from the Models page or by `audio.stt-model` / `audio.tts-model`.
The voice and speed controls in its panels default to `audio.tts.voice` and 1; the silence detector's threshold, minimum length and kept length, the normalize target and the fit mode are fields in the panels with the defaults of the table above.

**Measured.** Not timed as a feature: the costs are those of speech to text and text to speech (21.13) plus browser-side WSOLA, which is cheap next to the models. A preview of an edit by text costs one synthesis per changed phrase (cached for the next try).

**Checks and limits.** 25 node tests on the pure functions (`25 passed`), `AudioApiTest` on the endpoints; the edit-by-text parameters were tuned on real recordings (commit a89b039), but there is no automatic quality number for it.
Not done: a multi-track timeline, non-destructive effects (every edit makes a new clip), export formats other than WAV, and spectrum or pitch editing.

---

## 21.12 Checked against the references

All of this was checked against Python implementations, not just run: `refvenv` (torch 2.14, transformers 5.20), the `outetts` 0.3.3
package, `inflect`, MeCab, and `whisper-cpp` 1.9.5.

| piece | reference | result |
|---|---|---|
| log-mel (80 × 3 000, 5 s clip) | `WhisperFeatureExtractor` | largest difference 1.6e-5 (mean 2e-8); the 128-bin bank is the same code |
| tiny Whisper encoder, 5 s clip | transformers fp32 | largest difference 1.1e-3 on values up to 17 (mean 3.5e-6) |
| decoder logits (4 prompt tokens, in one call and token by token) | transformers | 7e-5 (CPU); argmax identical |
| greedy token ids, with and without timestamps, language auto-detected: tiny and base on an English and a German clip | `model.generate` | identical (20 and 18 tokens), also on large-v3-turbo (128 mels) |
| Metal encoder / Metal decoder | CPU | encoder 2.5e-3 (half weight tiles), logits 1e-4; ids identical |
| **word timestamps** | transformers pipeline `return_timestamps="word"` | same words, starts and ends within 0.02-0.06 s |
| text front end (20 sentences), number names (420 numbers) | `outetts` `TextProcessor`, `inflect` | identical |
| OuteTTS prompt token ids | HF tokenizer on the same prompt | identical (729 tokens) |
| OuteTTS greedy codes | HF generate (fp32 activations, Q8_0-rounded weights like the engine's) | the first 108 steps identical; teacher-forced over 250 steps the arg-max differs at 2 steps whose logit gaps are 2e-4 and 5e-4 (float rounding near ties) |
| WavTokenizer decoder | original PyTorch modules, same f16 weights | 107 dB SNR (every stage 112-123 dB) |
| the same against the f32 weights | | 60.6 dB (the GGUF's f16 weights) |
| `wh_*` kernels (`kernel_tests.py`) | numpy float64 | gemm 4e-7 relative (all four variants, ragged shapes, batches), layernorm 1e-6, softmax 4e-7, patches / split / embedding exact |
| synthesis, then transcription | Whisper small on the synthesized speech | "Hello, world. This is a test of the speech synthesis system." exactly |

(Word error rates on real recordings, the speech round trip and speed are in 21.13.)**What the numbers mean.** A "largest difference" is the worst single value of a tensor: `1.6e-5` on log-mel values that live in roughly `[-1, 1.5]` is a few float32 roundings. The strongest row is "greedy token ids identical": two networks that agree on every
arg-max of 18-20 steps (and on 73 of 73 real clips for `small`, 21.13) agree on every decision that matters, even if the logits differ in the fifth digit. Where they do not, the table says why (a logit gap of `2e-4` between two candidates is a tie that float
rounding breaks either way). A signal-to-noise ratio of `107 dB` means the error has an amplitude of `10^(-107/20) = 4.5e-6` of the signal's RMS; the `60.6 dB` against the f32 original (`9e-4`) is the cost of the GGUF's f16 weights, not of this code. Metal rows are looser than CPU rows because the
GPU path rounds weights to half precision when it stages them (21.6); the word error rates in 21.13 show that this does not change a transcript.



---

## 21.13 Measurements

Machine: Apple M1 Pro, 32 GB, JDK 26, the bundled defaults, run alone on the machine (every number below was taken while holding
the repository's heavy-job lock; a loaded machine makes them 2-3 times worse).

🧸 A real-time factor (RTF) is "seconds of work per second of sound": 0.1 means a minute of speech is written down in six seconds,
1.0 means the speaker and the computer take the same time. Lower is better for listening to a recording; for speaking, anything
near 1 is already comfortable (the first words arrive right away when the text is read sentence by sentence, 21.10).

📐 **Speech to text**, one 61.7 s clip (`long.wav`: a book passage read aloud, three 30 s windows), median of 2 runs after a warm-up,
whole transcription (log-mel, encoder, decoding with timestamps, fallback on):

```
scripts/audio-bench.sh .sp/audio/clips/long.wav --stt "tiny base small large-v3-turbo" --ggml-dir .sp/audio/ggml --reps 2
```

| model | jllm CPU | jllm Metal, decoder on CPU | jllm Metal, decoder on GPU | whisper.cpp 1.9.5 GPU | whisper.cpp CPU | transformers CPU | transformers `mps` |
|---|---|---|---|---|---|---|---|
| tiny (39 M) | 1.56 s, RTF 0.025 | 0.99 s, 0.016 | 1.34 s, 0.022 | 1.02 s, 0.016 | 6.0 s, 0.097 | 24.1 s, 0.391 | 1.96 s, 0.032 |
| base (74 M) | 2.59 s, 0.042 | 2.02 s, 0.033 | 2.84 s, 0.046 | 1.97 s, 0.031 | 9.8 s, 0.157 | 29.7 s, 0.481 | 3.95 s, 0.064 |
| small (244 M) | 14.2 s, 0.230 | 7.61 s, 0.123 | **4.71 s, 0.076** | 2.75 s, 0.042 | 30.0 s, 0.485 | 54.2 s, 0.877 | 6.68 s, 0.108 |
| large-v3-turbo (809 M) | not run (minutes) | 31.2 s, 0.506 | 26.6 s, 0.431 (see below) | not run | not run | not run | not run |

(`auto` picks the column that is fastest for the model: the decoder on the GPU for `d_model ≥ 768`, i.e. small and larger; the
whisper.cpp numbers are its own "total time" minus "load time", best of 2.) What the table says:

* On the CPU the engine is 2-2.5 times faster than whisper.cpp's CPU path on the same machine (small: 14.2 s against 30.0 s) and
  16-40 times faster than the transformers CPU path; the Vector API GEMM with a 4 × 4 register tile and 10 threads is the reason.
* On the GPU it is between the Python reference (`mps`, 6.7 s for small) and whisper.cpp (2.75 s for small). whisper.cpp's Metal
  kernels are better tuned than the one tiled simdgroup-matrix GEMM used here (`wh_gemm`), and it keeps the whole graph on the GPU;
  for tiny and base both finish a minute of speech in about 1-2 s and the differences are a few hundred milliseconds.
* Stages of one 30 s window (same run): log-mel 16-47 ms (CPU, all models); encoder plus the cross-attention keys and values: tiny 218 ms CPU
  / 26 ms Metal, base 480 / 54 ms, small 2243 / 194 ms (an 11.6 times speed-up), turbo 1.07-1.28 s Metal; one decoder step with a
  1500-frame cross-attention: tiny 2.0 ms, base 3.6 ms, small 12.2 ms CPU / 14.3 ms GPU (one isolated step; the GPU wins in a
  whole transcription, 4.7 s against 7.6 s, because the CPU decoder's attention over the 1500-frame cross-attention and the growing
  self-attention cache costs more than this isolated step shows).
* **Why turbo is slower than its size suggests**: its last window (4.8 s of speech then silence) makes greedy decoding loop
  (compression ratio 19.5, 220 tokens), and the fallback rule re-decodes that window at temperatures 0.2, 0.4 and 0.6 with five
  samples each, as OpenAI's reference does: 41 s with the fallback, 9.5 s with `audio.stt.fallback=false`. The other six windows
  take about 1.3 s each. The other models show no sign of a retry in their timings (about 255 decoder steps in 4.7 s for small).

📐 **Text to speech** (OuteTTS 0.3 500M, voice `en_male_1`, the script's default text (134 characters); LM on Metal, codec on the CPU):

(the same `audio-bench.sh` run: `--tts OuteAI/OuteTTS-0.3-500M` is its default, the q4 row is the second model line it prints.)

| | audio produced | time | RTF |
|---|---|---|---|
| Q8_0 LM (`audio.tts.quant=q8`, default) | 7.56 s | 7.67 s | 1.01 |
| Q4_0 LM (`q4`) | 8.65 s | 7.02 s | 0.81 |
| WavTokenizer decoder alone (CPU, 750 frames) | 10.0 s | 0.50 s | 0.050 |
| transformers on `mps` (LM only: 729 prompt tokens + 250 generated) | | 4.1 s, 61 tokens/s | |
| transformers on the CPU (same) | | 24.0 s, 10.4 tokens/s | |

The language model dominates (75 codec frames per second of sound, one token each, about 75 tokens/s on this GPU against 61 for the
Python reference on the same GPU), so speaking is real-time on this machine and a long text is spoken as it is generated (21.10).
Q4 is faster and, in the test below, not worse.

📐 **Accuracy on real recordings**: 73 clips of LibriSpeech (dev, 481 s, one speaker per clip; `WER` after lower-casing and removing
punctuation; transformers `generate` with its defaults as reference):

| model | transformers WER | jllm CPU | jllm Metal | identical transcripts jllm vs transformers |
|---|---|---|---|---|
| tiny | 12.78 % | 12.61 % | 12.61 % | 72 of 73 (one differs by a word) |
| base | 9.22 % | 9.22 % | 9.22 % | 73 of 73 (CPU and Metal) |
| small | 7.39 % | 7.39 % | 7.39 % | 73 of 73 (CPU and Metal) |

The 481 s are transcribed in 37 s on Metal with small (RTF 0.078). One tiny clip differs from transformers in a word because a
float rounding tie is broken the other way (the logits of the two candidates differ by 1e-4).

📐 **Speech round trip**: 12 sentences (151 words) spoken by OuteTTS and written down again by Whisper-small:

| LM quantization | word error rate | errors | speech generated | time |
|---|---|---|---|---|
| Q8_0 (`audio.tts.quant=q8`, default) | 6.62 % | 10 in 151 words | 43.4 s | 49.3 s |
| Q4_0 (`q4`) | 5.96 % | 9 in 151 words | 43.6 s | 47.0 s |

(Same sentences, voice `en_male_1` and seeds in both; the two are within the noise of a 151-word test. The tongue twister about sea shells
accounts for 6 errors in each. An earlier Q8 run had one silent output with 11 errors; that failure is what the engine now retries with another seed.)

The synthesized speech is therefore as intelligible to Whisper-small as it should be for a 500M model; the errors are single words
("frog" heard as "fraud").

**Summing up the measurements.** For listening to recordings the real-time factor is 0.08 with `small` on this Mac (the whole 61.7 s clip in 4.7 s), a speaker needs about 1.0 (Q8_0) or 0.8 (Q4_0) seconds of work per second of speech, the codec costs 0.05 of that, and the word error rate of
`jllm` equals `transformers`' on the 73 LibriSpeech clips (12.61 / 9.22 / 7.39 % for tiny / base / small, against 12.78 / 9.22 / 7.39 %). What is *not* measured: `large-v3` (non-turbo), CPU runs of `large-v3-turbo` (minutes), TTS on the CPU backend for the language model, and any
non-English accuracy beyond the one German clip used for the token-id comparison.

---

## 21.14 What is not done

* Speaker cloning (a voice from a user's recording) needs the WavTokenizer encoder, Whisper-based transcription of the reference and a CTC forced
  aligner. The 10 shipped voices cover English, German and French.
* Chinese, Japanese and Korean text for OuteTTS (MeCab / uroman); their characters are dropped.
* Beam search, `logprob`s per word and diarisation in Whisper; streaming of partial results.
* The codec (WavTokenizer decoder) is CPU-only: it is not the bottleneck (21.13).
* Output formats other than WAV and raw PCM need `ffmpeg`.
* Whisper's window is fixed at 30 s and the decoder is greedy with a temperature fallback (no beam search); `large-v3` (non-turbo) and CPU runs of `large-v3-turbo` were not measured.
* The word timestamps come from the cross-attention heads only; the result is as good as those heads (within 0.02-0.06 s of `transformers` on the checked clip), and no forced aligner backs it up.
* `audio.tts.quant=q8` is the default although one 12-sentence test found Q4_0 not worse (21.13): the test is small (151 words), so the conservative setting was kept.
* The audio editor has no automatic quality measure for *edit by text*; its parameters were tuned by ear on real recordings.
