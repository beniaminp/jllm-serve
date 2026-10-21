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
([10 §10.9b](10-engine-and-server.md#109b-batched-decoding-across-slots)).

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

## Where to go next

- [01 Overview](01-overview.md) for the map of the code, then [05](05-transformer-math.md) for every formula of a layer.
- [17 Speedups for coding agents](17-agent-speedups.md) for prompt lookup, shared prefixes, saved caches and pipelining.
- [18 The Metal emulator](18-metal-emulator.md) for how the GPU kernels are tested on machines without a GPU.
