# 21. Audio: speech-to-text, text-to-speech and an audio editor

Code: `org.robomiri.jllm.audio` (`Wav`, `Resampler`, `LogMel`, `Fft`, `Dense`, `WhisperWeights`, `WhisperCpu`, `WhisperMetal`,
`WhisperTranscriber`, `WordAligner`, `OuteTts`, `OuteText`, `WavTokenizerDecoder`, `TimeStretch`, `AudioApi`), the Metal kernels
`src/main/resources/org/robomiri/jllm/audio/audio.metal`, the engine's routes `server/AudioRoutes.java`, the console's
`web/audio/*`, `web/tools/AudioTools.java` and `static/js/audio*.js`; settings `audio.*` in `jllm.properties`; tests `AudioTest`,
`kernel_tests.py` (the `wh_*` kernels), `jllm-web/.../AudioApiTest`; fixtures and how they were made in `scripts/audio-fixtures.py`;
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
cost 60.6 dB (inaudible). The decoder runs on the CPU (21.13: ~0.08 s per second of speech, an order of magnitude below the language model's cost).

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

(Word error rates on real recordings, the speech round trip and speed are in 21.13.)

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

---

## 21.14 What is not done

* Speaker cloning (a voice from a user's recording) needs the WavTokenizer encoder, Whisper-based transcription of the reference and a CTC forced
  aligner. The 10 shipped voices cover English, German and French.
* Chinese, Japanese and Korean text for OuteTTS (MeCab / uroman); their characters are dropped.
* Beam search, `logprob`s per word and diarisation in Whisper; streaming of partial results.
* The codec (WavTokenizer decoder) is CPU-only: it is not the bottleneck (21.13).
* Output formats other than WAV and raw PCM need `ffmpeg`.
