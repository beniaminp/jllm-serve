# 0. jllm-serve explained simply — and the math behind it

Every idea in this project explained twice: first as if you were five (**🧸 the simple story**), then exactly, with
the formulas the code computes and small numbers you can check by hand (**📐 the math**). Read only the stories for
the big picture, or both for the real thing. The other chapters go deeper; each section links to its chapter.

---

## 1. What the whole thing does

🧸 Imagine a friend who has read every book in the world. You say the start of a sentence, and they guess the next
word. You write that word down, read the whole sentence to them again, and they guess the next one. Do that a few
hundred times and you have an answer to your question. That friend is the **model**. jllm-serve is the room you
put the friend in: it hands them your words, writes down their guesses, and passes the answer to your program over
the internet.

📐 A language model is one function from a list of integers to a list of scores:

```
logits = model(tokens)            tokens: int[n]      logits: float[V]   (V = 151 936 for Qwen3)
```

`logits[i]` says how plausible token `i` is as the *next* token. Generation is a loop:

```
repeat:  next = pick(model(tokens));  tokens = tokens + [next]     until next == end-of-text
```

Everything below makes `model(...)` fast and `pick(...)` sensible.

---

## 2. Tokens: cutting text into pieces

🧸 The friend doesn't read letters; they read **puzzle pieces** of words. "unbelievable" might be the pieces
"un", "believ", "able". Every piece has a number on its back. We give the friend numbers, not letters.

📐 The tokenizer (byte-pair encoding, [04](04-tokenizer.md)) starts from the bytes of the text and repeatedly
merges the pair of adjacent pieces that has the best (lowest) rank in a learned list of merges:

```
"lower" → l o w e r → (merge "l o") lo w e r → (merge "lo w") low e r → (merge "e r") low er → [ id(low), id(er) ]
```

A vocabulary of 151 936 pieces covers every byte sequence, so any text can be encoded, and decoding is just
gluing the pieces' bytes back together.

---

## 3. Embeddings: turning a number into a meaning

🧸 Each puzzle piece number opens a drawer with a long list of numbers inside — a "meaning card". Words with
similar meanings have similar cards.

📐 The embedding table `E` has one row of `d` floats per token (`d = 1024` for Qwen3-0.6B). Token `t` becomes the
vector `x = E[t]`. Nothing is computed; it is a lookup. The rest of the network transforms these vectors.

---

## 4. Attention: who should I listen to?

🧸 In a sentence, each word looks back at the earlier words and decides how much to listen to each one. In "The cat
sat on the mat because **it** was tired", the word "it" listens mostly to "cat". Every word asks a question
(**query**), every earlier word holds up a sign saying what it's about (**key**), and carries a message (**value**).
The better a sign matches the question, the louder that word's message is heard.

📐 For one head with head width `h` (128 in Qwen3), each position `s` has vectors `q_s, k_s, v_s` (made from `x` by
matrix multiplications). The output for position `t` is a weighted average of the values:

```
score(t, s) = (q_t · k_s) / sqrt(h)           for s <= t        (causal: never look at the future)
p(t, s)     = exp(score(t, s)) / Σ_s' exp(score(t, s'))       (softmax: positive, sums to 1)
out_t       = Σ_s p(t, s) · v_s
```

Worked example, `h = 2`, one query and two keys:

```
q = [1, 0],   k_0 = [1, 0], k_1 = [0, 1],   v_0 = [10, 0], v_1 = [0, 10]
scores = [1/√2, 0] = [0.707, 0]
p = [e^0.707, e^0] / (e^0.707 + 1) = [2.028, 1] / 3.028 = [0.670, 0.330]
out = 0.670·[10,0] + 0.330·[0,10] = [6.70, 3.30]
```

The query matched key 0 better, so value 0 dominates. **Why `/ sqrt(h)`**: if `q` and `k` have independent entries
of variance 1, their dot product has variance `h`; dividing by `sqrt(h)` keeps the scores around 1 so the softmax is
neither flat nor a hard maximum.

Chapter [05](05-transformer-math.md) has the full layer; [07 §7.5](07-cpu-backend.md#75-attention-kvcacheattention)
and [08](08-metal-backend.md) show how the CPU and GPU compute it.

### 4.1 The softmax that never overflows

📐 `exp(50)` is fine but `exp(500)` overflows a float. Since `exp(a - m) / Σ exp(b - m) = exp(a) / Σ exp(b)` for any
`m`, every softmax here first subtracts the maximum score `m`, so the largest exponent is `exp(0) = 1`.

### 4.2 Online softmax (the trick behind flash attention)

🧸 You are adding up how much you like each of 6 000 cookies, but you can only hold 64 in your hands at a time.
You keep a running total; whenever a new handful contains a cookie you like more than any before, you re-measure your
old total on the new scale and carry on. You never need all 6 000 on the table at once.

📐 Process the keys in blocks. Keep, per query row, the running maximum `m`, the running sum `l` and the running
output `o` (not yet divided by `l`). For a new block with scores `S_j`:

```
m'  = max(m, max_j S_j)
P_j = exp(S_j - m')
l'  = l · exp(m - m') + Σ_j P_j
o'  = o · exp(m - m') + Σ_j P_j · v_j
```

and at the end `out = o / l`. The factor `exp(m - m')` rescales everything accumulated so far to the new maximum,
so the result equals the ordinary softmax exactly (up to rounding), while the scores of all the keys never need to
exist at once. The Metal kernel `attention_fa` ([08 §8.13](08-metal-backend.md#813-flash-attention-prefill-attention_fa))
does this with 8×8 matrix tiles; the per-row rescale is multiplication by the diagonal matrix
`diag(exp(m - m'))`, so it is also a matrix product.

---

## 5. Positions: RoPE, the rotating clocks

🧸 Attention alone doesn't know word order — "dog bites man" and "man bites dog" would look the same. So each word's
question and sign are turned like clock hands by an amount that depends on its position. Two words that are 3
apart always end up with the same angle between them, wherever they are in the text.

📐 Pair up the coordinates of `q` (and `k`) and rotate pair `i` of position `p` by angle `p · θ_i` with
`θ_i = base^(-2i/h)`:

```
[q'_2i  ]   [cos(pθ_i)  -sin(pθ_i)] [q_2i  ]
[q'_2i+1] = [sin(pθ_i)   cos(pθ_i)] [q_2i+1]
```

Rotations preserve dot products up to the angle difference, so `q'_t · k'_s` depends only on `t - s`: relative
position, for free.

---

## 6. The feed-forward block and the residual stream

🧸 After listening to the others, each word thinks on its own for a moment (the **MLP**), then adds what it
concluded to its meaning card instead of replacing it. Stack 28 such rounds of "listen, then think" and the cards
end up describing what should come next.

📐 One layer (Qwen3 / Llama style):

```
x = x + Attention(RMSNorm(x))
x = x + W_down · ( silu(W_gate · RMSNorm(x)) ⊙ (W_up · RMSNorm(x)) )         silu(z) = z / (1 + e^-z)
RMSNorm(x) = x / sqrt(mean(x²) + ε) ⊙ g
```

The `x = x + ...` form (the **residual stream**) lets every layer make a small correction. After the last layer
`logits = W_head · RMSNorm(x)`.

---

## 7. Picking the next word: sampling

🧸 The friend gives every possible next word a score. You can always take the top one (**greedy**: predictable), or
roll a weighted die so that likely words come up often and odd ones sometimes (**sampling**: livelier). The
**temperature** knob makes the die fairer (hot) or more loaded (cold).

📐 With temperature `T`: `p_i = exp(logit_i / T) / Σ_j exp(logit_j / T)`. `T → 0` is the argmax; `T = 1` is the model's
own distribution; top-k keeps the `k` best, top-p keeps the smallest set whose probabilities add up to `p`. Worked:
logits `[2, 1, 0]` give `[0.665, 0.245, 0.090]` at `T = 1` and `[0.867, 0.117, 0.016]` at `T = 0.5`.
([09](09-sampling.md))

---

## 8. The KV cache: not re-reading the whole book

🧸 Each time the friend guesses a word, they would have to re-read the whole conversation. Instead they keep
**sticky notes** (the keys and values of every word so far) and only read the new word, looking at the notes.

📐 Keys and values of position `s` never change once computed (they depend only on tokens `0..s`). So we store
`k_s, v_s` for every layer and position. Memory per token:

```
2 (K and V) × layers × kvHeads × headDim × bytes  =  2 × 28 × 8 × 128 × 2 B  =  114 688 B ≈ 115 kB   (Qwen3-0.6B, f16)
```

A 10 000-token conversation keeps ~1.15 GB of notes. With the notes, a new token costs one pass over the weights
instead of a pass over the whole conversation. Saving those notes to disk ([17 §17.3](17-agent-speedups.md)) is why
a restarted server can skip a minutes-long first step.

---

## 9. Quantization: smaller numbers, almost the same answers

🧸 Instead of writing every weight with 16 digits, write it with a coarse ruler that has only 16 marks, plus one
"ruler size" for every group of 32 weights. The picture gets a little blurrier but you can still recognize it, and
it is four times smaller — and since the computer spends most of its time *reading* weights, smaller means faster.

📐 Q4_0 ([06](06-quantization.md)): a block of 32 weights `w_j` gets one scale `d = m / −8`, where `m` is the
weight of largest magnitude (with its sign), and 4-bit codes `q_j = round(w_j / d) + 8`, clamped to `[0, 15]`; the
weight is read back as `w_j ≈ d · (q_j − 8)`. The extreme weight lands exactly on code 0 (−8), and every other weight
is off by at most `|d| / 2 = max|w| / 16`. MLX's format ([03 §3.7](03-safetensors.md#37-mlx-quantized-checkpoints)) uses a scale **and** an offset per group:
`w ≈ s · q + b` with `b` the group's minimum, so the 16 levels cover `[min, max]` instead of `[-max, max]`.

**Why it is faster**: generating one token reads every weight once. Qwen3-0.6B in Q4_0 is ~0.4 GB; at 100 GB/s of
memory bandwidth that alone is 4 ms per token, a ceiling of ~250 tokens/s no matter how fast the arithmetic is
(**memory-bound**). Halving the bytes halves that time.

---

## 10. CPU and GPU: one fast chef or a thousand cooks

🧸 A CPU is a few very clever chefs who each can chop 16 carrots with one knife stroke (**SIMD**). A GPU is a
kitchen with thousands of slower cooks who all do the same step at the same time, in teams of 32 (**simdgroups**).
Big jobs with lots of identical small steps — multiplying matrices — go to the GPU kitchen.

📐 A matrix-vector product `y = W x` with `W` of `R × C` is `R` independent dot products of length `C`. The CPU runs
them on a few cores, 16 floats per instruction (AVX-512). The GPU gives each row to a simdgroup of 32 threads, each
thread summing every 32nd block, then adds the 32 partial sums with one `simd_sum` (a tree of 5 steps:
`32 → 16 → 8 → 4 → 2 → 1`). Prefill (many tokens at once) becomes a matrix-matrix product, which the GPU computes
with 8×8 matrix-multiply instructions.

---

## 11. Experts: asking the right specialist (MoE)

🧸 Instead of one huge brain, a Mixture-of-Experts model has 128 small specialists and a receptionist. For each
word the receptionist picks the 8 best-suited specialists, and only they work. The model knows as much as all 128
together but costs only 8 per word.

📐 Router logits `r = W_router · x`, top-k with softmax weights `g_i = exp(r_i) / Σ_{j in top-k} exp(r_j)`, output
`Σ_{i in top-k} g_i · Expert_i(x)`. Qwen3-30B-A3B has 30 B weights but reads ~3 B per token. On the GPU, a prompt's
tokens are first **sorted by expert** so each specialist's weights are read once for all the tokens that chose it
(and the prompt is read in pieces of 256 tokens, so each specialist has ~16 tokens to answer per read instead of ~4)
([08 §8.15](08-metal-backend.md#815-expert-grouped-moe-prefill)).

When the specialists don't all fit on the GPU's desk, some stay with the CPU. jllm-serve remembers which
specialists the receptionist calls most and keeps those on the GPU's desk; the ones almost nobody asks for go to the
CPU. Moving the same number of specialists then costs much less, because the CPU is rarely called. And when no word
needs a CPU specialist, the GPU doesn't wait at all ([08 §8.18](08-metal-backend.md#818-experts-on-the-cpu-backendmetalcpu-moe)).

---

## 12. Guess and check: speculative decoding

🧸 When the friend is copying a sentence you showed them earlier, you can guess the next ten words yourself. Ask the
friend to check all ten at once — checking is as quick as guessing one word. Keep the ones they agree with, throw away
the rest. If you guessed well, you got ten words for the price of one.

📐 With a draft of `k` tokens, the model computes the logits of all `k + 1` positions in one pass. The draft token
`d_j` is kept if it equals the model's choice (greedy) or with probability `min(1, p(d_j))` (sampling: proposal is
deterministic, so the accept rule is `p(d_j)`, and a rejection resamples from `p` with `d_j` removed — the output
distribution is exactly the model's). If each draft token is accepted independently with probability `α`, the expected
number of tokens produced per pass is

```
E[tokens per pass] = 1 + α + α² + ... + α^k = (1 - α^(k+1)) / (1 - α)
```

For `α = 0.98, k = 16`: `(1 - 0.98^17) / 0.02 ≈ 14.5` tokens per pass — which is why rewriting a file that is in the
prompt went from 19.9 to 81.2 tokens/s ([17 §17.1](17-agent-speedups.md#171-prompt-lookup-tuned-for-agents)).
Where the guesses come from (an index of every 2..8-token sequence of the text) is in the same chapter.

---

When several people talk to the same model at once, their next words are computed together: reading the model's
weights is the slow part, and one reading can serve every conversation
([10 §10.9b](10-engine-and-server.md#109b-batched-decoding-across-slots)). `./jllm serve --server.slots 4` keeps four
conversations going at the same time, so nobody waits for three other answers before seeing their first word
([10 §10.9c](10-engine-and-server.md#109c-several-users-at-once-serve---serverslots)).

## 13. Why the first answer of a coding agent is slow, and later ones are fast

🧸 A coding agent's first message comes with a 10 000-word instruction book. The friend must read it all once
(**prefill**). Afterwards, every new message starts with the same book, and the sticky notes for it are still there,
so only the new lines are read.

📐 Prefill costs `n` tokens of work but in big matrix-matrix products (efficient); decode costs one token at a time
(memory-bound). Attention adds `O(n²)` work: for `n = 10 000` each layer compares `~n²/2 = 5·10⁷` query-key pairs per
head. Reusing the cached prefix makes the cost of a follow-up request proportional to the *new* tokens only; sharing
that prefix between slots and saving it to disk ([17](17-agent-speedups.md)) extends this across conversations and
restarts.

---

## 14. Pictures in: giving the friend eyes

🧸 The friend only understands words, but you want to show them a photo. So you hire a **translator** who looks at the picture the way you look at a jigsaw puzzle: first the photo is cut into small squares, then the squares talk to each other until each one means more than "some blue" ("the edge of a red circle"), and neighbouring squares are merged so there are not thousands. The translator writes the result in the friend's own secret
language: a row of "picture words", each a bundle of numbers just like the bundle that every ordinary word turns into. The friend reads `<picture> picture-word picture-word … </picture> What shapes do you see?` and answers. And because a photo cannot be looked up by a number, the engine remembers it by its fingerprint: the same photo in the next turn is not looked at again.

📐 A picture of `W × H` pixels with patch size `p` has `(W/p)(H/p)` patches of `3p²` numbers; one matrix product turns each into a vector, `L` transformer blocks (attention without a mask: every patch sees every patch) refine them, and a merger shrinks them. Worked: a 640 × 480 picture for Qwen2.5-VL is resized to 644 × 476 (multiples of 28), cut into `46 × 34 = 1 564` patches of `14 × 14 × 3 = 588` numbers, and every 2 × 2 block of patches is merged:
`23 × 17 =` **391 picture words** replace the placeholder in the prompt, and the model's own KV cache rows for them are reused when the same picture comes back. Models that need to know *where* a picture word sits rotate it by its row and its column (M-RoPE). Details: [20](20-vision.md).

---

## 15. Speech in: giving the friend ears

🧸 You do not hear single wiggles of the air; you hear **which pitches are present**. So the recording is turned into a picture: time goes to the right, pitch goes up, brightness says how loud. The pitch axis is squeezed the way ears do it (many thin rows for low sounds, a few thick ones for high ones). A second pair of helpers, **Whisper**, reads the picture: one looks at the whole 30 seconds and writes notes, the other writes the words one at a time,
glancing back at the notes before each word ("what was said around here?"). Because it glances at the notes it can also tell you *when* each word was said.

📐 Sound is 16 000 numbers per second. Every 10 ms a window of 400 samples (25 ms) is multiplied by a Hann curve and Fourier-transformed (201 pitches, 40 Hz apart), the powers are summed into 80 mel bands, and `log10` is taken: 30 seconds become an `80 × 3 000` picture. The encoder turns it into 1 500 positions (20 ms each); the decoder attends to them (cross-attention, [21](21-audio.md)).
Worked: a 1 kHz tone of amplitude 0.5 lands exactly on pitch number `1000 / 40 = 25` with power `(0.5 · 200 / 2)² = 2 500`, between the mel bands 24-27 (mostly 25-27); the 80 dB floor sets every other band to `(−6.24 + 4)/4 = −0.56`. Measured: a 61.7 s recording is written down in 4.7 s with Whisper-small on the GPU (0.076 s of work per second of sound), with the same words as the Python reference on all 73 real clips used to check it.

---

## 16. Speech out: giving the friend a voice

🧸 Teaching a computer to speak can be done by building a machine "text in, sound out", or by noticing that the friend already knows how to continue a story, and letting them continue a story whose words are **sound pieces**. The text is `the<|t_0.12|><|1090|><|512|>…` where every `<|number|>` is a 13-millisecond piece of sound from a catalogue of 4 096. A second helper, the *codec*, turns the pieces back into air wiggles. To make it sound like a person, the story starts with a few seconds of
someone speaking, written in the same pieces; the friend carries on in that voice.

📐 The codec describes sound with 75 pieces per second, 4 096 possible pieces each, i.e. `75 · 12 bits = 900 bits` per second; each piece is 320 samples of 24 kHz sound (13.3 ms). The language model (Qwen2, run by the same engine as a chat model) writes about 90 tokens for each second of speech (75 sound pieces plus the words and their times), so speaking takes about as long as the speech lasts (measured 7.67 s of work for 7.56 s of speech with the 8-bit model,
0.81 s per second with the 4-bit one); the codec adds 0.05 s per second of speech. At the end, a Fourier transform in reverse (1 280 points, windows overlapped by hops of 320) puts the waves back together ([21](21-audio.md)).

---

## 17. Pictures out: removing the snow

🧸 Take a photo and sprinkle it with static. A little static and you still see the photo; a lot and you see only snow. A patient helper has seen millions of captioned photos: show them a snowy screen and tell them "a cat on a sofa" and they can guess *which flakes do not belong* and wipe a few away. Repeat many times and a cat on a sofa appears from pure snow. A "Turbo" helper was taught to do it in one or two big wipes. To keep the
work small, the helper does not wipe the 786 432 dots of the picture but a 16 384-number sketch of it, and a separate pair of helpers turns sketches into pictures and back.

📐 Noise is added as `x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε`: at step 499 of 1 000, `ᾱ = 0.278`, so a clean value 0.5 with noise 1.0 becomes `0.527 · 0.5 + 0.850 · 1.0 = 1.113`. The network predicts `ε`; a sampler walks the noise level down (Euler: `x̃ ← x̃ + (σ_next − σ) ε̂`), and "guidance" asks twice, with and without the caption, and goes further in the direction the caption made a difference:
`ε = ε_u + w (ε_c − ε_u)`, `0.20 + 7.5 · (0.50 − 0.20) = 2.45`. Measured: a 512 × 512 picture in one Turbo step takes 1.68 s on the M1 GPU (0.42 s for the network, 1.2 s to turn the sketch into dots); 20 steps of the same model take 9.6 s ([22](22-image-generation.md)).

---

## 18. Video out: many pictures that must agree

🧸 If a helper drew every page of a cartoon separately, the balloon would change colour on page 5. So the pages are drawn **together**: while wiping the snow off page 7 the helper keeps glancing at the same spot on all the other pages ("you moved up a bit, I will too"), and each page is stamped with its number, because "page 3 of 16" is not "page 11 of 16". AnimateDiff adds exactly those glances to an ordinary picture painter. Wan 2.1 is a different, bigger design: it cuts the whole video into bricks that all look at all
the others, and instead of naming the snow it learns the *direction to walk* from noise to video, which is a straight line.

📐 Sixteen 512 × 512 frames are `16 × 64 × 64 × 4 = 262 144` numbers; with guidance a step processes 32 pictures, so a 20-step clip takes 371 s on this Mac (351 s in the network, 19 s to turn the 16 sketches into pictures). For every pixel the temporal layer makes a little sequence of its 16 frames and lets them attend to each other; without the frame stamps frames with the same content would be indistinguishable (a toy: outputs `[0.67, 0.33]` for both; with stamps `[0.86, 0.48]` and `[1.05, 0.26]`).
In Wan, the path is `x_σ = (1 − σ) x₀ + σ ε`, so the direction `v = ε − x₀` is constant: with `x₀ = 0.5`, `ε = 0.7`, `v = 0.2` and one Euler step from `σ = 1` to 0.9232 lands exactly on the path (`0.7 + (−0.0768)(0.2) = 0.6846`). Honest status: both are verified against the reference programs to a fraction of a grey level, but the AnimateDiff clips look poor with the plain Stable Diffusion 1.5 it was paired with, and Wan's text reader (22.7 GB) was never downloaded, so no real Wan clip exists ([23](23-video-generation.md)).

---

## 19. The model guessing for itself: multi-token prediction

🧸 In section 12 someone guesses the next ten words and the friend checks them all at once. Usually the guesser is a small second model. Some models (Qwen3.5) come with their own tiny guesser: one extra layer, trained to look at what the big model was thinking and the word it just said, and guess the word *after* that. It costs about one layer per word and needs no second download. If the guess is right the friend gets two words for the price of one pass; if it is wrong, nothing is lost
except that the friend's whiteboards (the fixed-size memory of its Gated DeltaNet layers, [14 §14.14](14-model-families.md#1414-qwen3-next-hybrid-gated-deltanet--gated-attention--sparse-moe)) must be put back as they were, so jllm keeps a second copy of them.

📐 With one guess per pass and the guess right with probability `α`, a pass yields `1 + α` words: `1.9` for `α = 0.9`. Measured on the 27B Qwen3.5 file, the head guessed the next token 111 times out of 124 (`α = 0.90`) in greedy runs of four prompts. The output is exactly the output without the guess (greedy), because the friend verifies every guessed word against its own choice ([16 §16.14](16-gguf.md#1614-multi-token-prediction-the-model-as-its-own-draft-model); the sampling rule is [9 §9.7](09-sampling.md#97-the-full-order-of-operations)).

---

## 20. Smaller sticky notes: the KV cache in 16, 8 or 4 bits

🧸 The friend's sticky notes (section 8) grow with every word. You can write each note with fewer digits. **f16** keeps half the digits, **q8** writes every number as one whole-number step on a ruler (plus one note saying how long the ruler is), and **q4** keeps small thumbnails. Two tricks stop the thumbnails from being useless: the question and the notes are first *shaken* with a fixed recipe so that no single number is huge, and the newest few pages are kept sharp.
Some models need fewer notes by design: DeepSeek writes one short summary per word instead of notes per head, and models like Qwen3-Next keep a fixed-size whiteboard in most layers.

📐 Per token, Qwen3-0.6B (28 layers, K and V, 1 024 numbers each):

```
f32   2 · 28 · 1024 · 4 B                    = 229 376 B     940 MB at 4 096 tokens
f16   2 · 28 · 1024 · 2 B                    = 114 688 B     470 MB
q8    2 · 28 · 1024 · (1 + 4/128) B          =  59 136 B     242 MB      (one signed byte + one f32 scale per 128 numbers; error ≤ max|x|/254 of the row)
q4    2 · 28 · 1024 · 0.5625 B               =  32 256 B     132 MB      (4 bits + one f16 scale per 32 numbers, plus a ring of sharp recent rows)
```

The honest part: f16 and q8 are the choices to make for memory; q4 is clearly worse than q8 on this small model (the keys are too rough even after the shaking) and stays opt-in, `backend.kv-dtype=q4` ([7](07-cpu-backend.md), [11 §11.3](11-performance.md#113-memory-footprint)).

---

## Where to go next

- [01 Overview](01-overview.md) for the map of the code, then [05](05-transformer-math.md) for every formula of a layer.
- [17 Speedups for coding agents](17-agent-speedups.md) for prompt lookup, shared prefixes, saved caches and pipelining.
- [18 The Metal emulator](18-metal-emulator.md) for how the GPU kernels are tested on machines without a GPU.
- [20 Pictures as input](20-vision.md), [21 Audio](21-audio.md), [22 Image generation](22-image-generation.md) and [23 Video generation](23-video-generation.md) for sections 14-18 in full.
- [16 §16.14](16-gguf.md#1614-multi-token-prediction-the-model-as-its-own-draft-model) for the model's own guessing head (19) and [11 §11.3](11-performance.md#113-memory-footprint) for the KV cache sizes (20).
