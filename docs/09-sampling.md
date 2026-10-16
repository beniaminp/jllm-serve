# 9. Sampling: from logits to a token — with all the math

Code: `sampling/Sampler.java`, `sampling/SamplingParams.java`; request fields in `server/ApiServer.params`,
defaults in `src/main/resources/jllm.properties` (`sampling.*`).

The forward pass ends with 151 936 **logits** — one raw score per vocabulary entry. The sampler's job is to
turn that list into exactly one token id. This chapter builds the needed probability ideas from zero, then goes
through every knob (`temperature`, `top_k`, `top_p`, `min_p`, the three penalties, `seed`) with worked numbers,
in the order `Sampler.sample()` applies them.

---

## 9.0 The probability toolbox

### A distribution

A **probability distribution** over `V` tokens is a list of `V` numbers `p₀ … p_{V−1}` with every `pᵢ ≥ 0` and
`Σ pᵢ = 1`. `pᵢ` is "the chance that the next token is `i`". `[0.5, 0.3, 0.2]` is a distribution over three
tokens; `[0.5, 0.3, 0.3]` is not (sums to 1.1); `[0.9, −0.1, 0.2]` is not (negative entry).

### Sampling = drawing with weights

To *sample* from a distribution means to pick one index at random so that index `i` comes up with probability
`pᵢ`. The standard way needs one uniform random number `r` in `[0, 1)` and a walk over the cumulative sum:

```
p   = [0.5, 0.3, 0.2]
cum = [0.5, 0.8, 1.0]          running total

draw r = 0.65:   is 0.65 < 0.5 ?  no      is 0.65 < 0.8 ?  yes  → token 1
draw r = 0.12:   is 0.12 < 0.5 ?  yes                           → token 0
draw r = 0.97:   … < 0.5 no, < 0.8 no, < 1.0 yes                → token 2
```

Token 1 wins whenever `0.5 ≤ r < 0.8`, an interval of length 0.3 — exactly its probability. This is literally
the last loop of `Sampler.sample`:

```java
double r = rng.nextDouble() * total;        // total = sum of the kept weights (need not be 1)
double acc = 0;
for (int i = 0; i < keep; i++) {
    acc += val[i];
    if (r < acc) return idx[i];
}
```

Scaling `r` by `total` instead of normalizing every weight saves a division per candidate: the intervals keep
the same *proportions*.

### Logits vs probabilities

The model does not output probabilities. It outputs **logits**: unbounded real numbers, typically between −20
and +30, where only *differences* carry meaning. Softmax ([05 §5.0](05-transformer-math.md#softmax-scores--probabilities))
converts them:

```
pᵢ = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)
```

with `T` the **temperature** — the first knob.

---

## 9.1 Temperature

Dividing all logits by `T` before softmax rescales the *gaps* between them. Because `exp` turns additive gaps
into multiplicative ratios, the same four logits give very different distributions:

```
logits z = [3.0, 2.0, 1.0, 0.0]

T = 0.2   z/T = [15, 10, 5, 0]        p = [0.9933, 0.0067, 0.00005, 0.0000003]   almost certain
T = 0.6   z/T = [5, 3.33, 1.67, 0]    p = [0.812,  0.153,  0.029,   0.0055]      confident (Qwen3 default)
T = 1.0   z/T = [3, 2, 1, 0]          p = [0.644,  0.237,  0.087,   0.032]       the model's own belief
T = 1.5   z/T = [2, 1.33, 0.67, 0]    p = [0.523,  0.268,  0.138,   0.071]       adventurous
```

(For `T = 0.6`: `exp(5) = 148.4`, `exp(3.33) = 28.0`, `exp(1.67) = 5.29`, `exp(0) = 1`, sum `182.7`,
`148.4 / 182.7 = 0.812`.) Lower temperature → the top token dominates; higher → the tail gets a real chance.

**Greedy decoding** is the limit `T → 0`: always take the largest logit. `Sampler.sample` treats
`temperature == 0` (and `top_k == 1`) as exactly that, via `argmax`, with no randomness at all. Greedy output
is reproducible and good for tests — it is how the CPU and Metal backends were compared — but it tends to loop
on long outputs.

In the code `invT = 1 / T` is multiplied in during the softmax: `exp((val[i] − max) * invT)`.

---

## 9.2 Why truncate: the long tail

Even at `T = 0.6`, a vocabulary of 151 936 entries has a problem. Suppose the top 20 tokens carry 97 % of the
mass and the other ~151 900 share the remaining 3 %. Each of those is individually negligible
(`0.03 / 151 900 ≈ 2 × 10⁻⁷`), but **together they are drawn 3 % of the time** — one token in 33 would be a
random, usually nonsensical word. A single such token can derail a whole answer ("The capital of France is
Parisxenon…"). Truncation removes the tail before the draw. Three filters do it, in this order.

---

## 9.3 Top-k (`top_k`, default 20)

Keep only the `k` largest logits, discard everything else. Simple, but doing it by sorting 151 936 numbers
every token would cost more than the forward pass of a small model. `Sampler.topK()` instead keeps a
**min-heap of size k**: a tree in which the *smallest* kept value sits at the root, so "is this candidate better
than the worst thing I'm keeping?" is one comparison.

Worked example, `k = 3`, logits by token id `[1.2, 4.0, 0.5, 3.1, 2.2, 3.7]`:

```
id 0 (1.2): heap not full → insert                        heap {1.2}                root 1.2
id 1 (4.0): insert                                        heap {1.2, 4.0}           root 1.2
id 2 (0.5): insert, sift up (0.5 < 1.2)                   heap {0.5, 4.0, 1.2}      root 0.5
id 3 (3.1): 3.1 > root 0.5 → replace root, sift down      heap {1.2, 4.0, 3.1}      root 1.2
id 4 (2.2): 2.2 > root 1.2 → replace root, sift down      heap {2.2, 4.0, 3.1}      root 2.2
id 5 (3.7): 3.7 > root 2.2 → replace root, sift down      heap {3.1, 4.0, 3.7}      root 3.1
```

The heap now holds the three largest: ids 1, 5, 3. Each step touched at most `log₂ k` elements, so the pass is
`O(V log k)` — for `k = 20` about 4 comparisons per token instead of a full sort.

The later filters need the survivors **in descending order**, so `topK()` finishes with an in-place heap-sort:
repeatedly swap the root (the minimum) to the end and shrink the heap. Minimum-first into the last slot yields
ascending-from-the-back, i.e. descending from the front: `[4.0, 3.7, 3.1]` → ids `[1, 5, 3]`.

`top_k = 0` disables the filter; `top_k = 1` is greedy.

---

## 9.4 Top-p, "nucleus" sampling (`top_p`, default 0.95)

Top-k keeps a fixed *count*; top-p keeps a fixed *amount of probability*. After the softmax over the sorted
candidates, walk down accumulating probability and stop as soon as the running total reaches `p`:

```
sorted candidates (after softmax)   cum
  token a   0.45                     0.45
  token b   0.25                     0.70
  token c   0.12                     0.82
  token d   0.08                     0.90   ← cum >= 0.9: stop, keep = 4
  token e   0.05                     0.95
  token f   0.03
  token g   0.02
```

With `top_p = 0.9` the first four are kept. **The token that crosses the threshold is included** (`keep = i + 1`
after the `cum >= topP` test) — a common off-by-one in home-made samplers. Top-p adapts to confidence: when the
model is sure (`[0.97, 0.01, …]`) it keeps one token; when the sentence could go many ways it keeps dozens.

In the code: `cum += val[i] / sum` over the sorted candidates; the kept count `keep` bounds the final draw.

### Top-p without top-k: the candidate bound

When a request sets `top_k: 0`, there is no small sorted list to walk, and sorting all of `V` is exactly what we
want to avoid. `nucleusCandidates()` first discards tokens that *cannot* be in the nucleus. The argument: sort
descending; token `i` (not the top one) is inside the nucleus only if the mass *before* it is still below `p`,
i.e. the mass from `i` onward exceeds `1 − p`. Every token from `i` onward has probability ≤ `pᵢ` (sorted!) and
there are at most `V − 1` of them, so

```
mass from i onward  ≤  (V − 1) · pᵢ          hence   pᵢ < (1 − p) / (V − 1)   ⇒   not in the nucleus
```

For `p = 0.95`, `V = 151 936`: the cutoff is `3.3 × 10⁻⁷`. Typically only a few hundred tokens survive, and only
those are sorted (`sortDescending`).

One subtlety: the cumulative probabilities must be measured against the **full-vocabulary** softmax sum, not
the sum over the survivors — the pruned tail still carries up to `1 − p` of the mass. `nucleusCandidates()`
therefore returns the denominator computed over all `V` logits, and `sample()` uses it for `cum` on this path.
This is not a limitation of the `top_k: 0` path: the nucleus is the same set a full-vocabulary sort would give.
(Normalizing over the survivors only would cut it one token early; `SmokeTest` checks the sampled set against a
brute-force reference, including the boundary token.) With a non-zero `top_k` the survivors are the `k` best
tokens and top-p is computed over their softmax, as in Hugging Face, where the discarded logits are −∞.

---

## 9.5 Min-p (`min_p`, default 0 = off)

Drop any candidate whose probability is below `min_p × p_max`, where `p_max` is the most likely token's
probability. `min_p = 0.1` means "nothing less than a tenth as likely as the favourite". Like top-p it adapts
to confidence, but it never cuts a strong second choice the way a tight top-p can.

Implementation shortcut: after `exp((val − max) * invT)` the top token's unnormalized weight is exactly
`exp(0) = 1`, so the test `val[i] < minP` compares against `p_max` with no division. Filtered tokens get
weight 0 and are skipped by the draw.

---

## 9.6 Penalties: discouraging loops

Small models repeat themselves. Three standard corrections are applied **to the logits, before temperature**,
for tokens the sampler has *seen*: `seen[]` is initialized from the prompt and `accept(token)` marks each
generated token and counts it (`generatedCounts[]`).

| penalty | applies to | formula | typical value |
|---|---|---|---|
| `repetition_penalty` (Hugging Face style) | tokens in prompt **or** output | `z > 0 ? z / r : z × r` | 1.05–1.2 (1 = off) |
| `presence_penalty` (OpenAI style) | tokens generated so far | `z − a` | 0–1 (0 = off) |
| `frequency_penalty` (OpenAI style) | tokens generated so far | `z − b × count` | 0–1 (0 = off) |

Worked example, `r = 1.3`, `a = 0.5`, `b = 0.2`:

```
token " the"   logit  2.0, seen in prompt, generated 3 times
   repetition:  2.0 / 1.3        = 1.538       (positive → divide: less likely)
   presence:    1.538 − 0.5      = 1.038
   frequency:   1.038 − 0.2 × 3  = 0.438       final logit

token " zebra" logit −1.0, seen once in the prompt, never generated
   repetition: −1.0 × 1.3        = −1.3        (negative → multiply: pushed further down)
   presence/frequency: count = 0 → unchanged   final logit −1.3

token " cat"   logit  1.5, never seen → untouched
```

The sign rule in the repetition penalty (`divide if positive, multiply if negative`) always moves the logit
*down*; a naive "always divide" would make negative logits *more* likely. `applyPenalties` is skipped entirely
when all three are at their neutral values, so the default configuration pays nothing.

---

## 9.7 The full order of operations

```
logits  ──penalties──▶  ──[greedy? argmax]──▶  ──top-k (heap) or nucleus bound──▶
        ──softmax with 1/T──▶  ──top-p──▶  ──min-p──▶  ──weighted draw──▶  token
```

Hugging Face `generate()` runs penalties, then temperature, then top-k, then top-p. Our order differs only in
where temperature is applied, and that is harmless: top-k selects by logit, and dividing every logit by the same
`T` never changes which `k` are largest. The subsequent softmax over the kept set is then identical to HF's
(HF sets discarded logits to −∞, which also gives them zero weight). For the default `top_k = 20`, outputs are
distributed exactly as in `transformers`.

---

### Speculative verification

`Sampler.verify(logits, draft)` is the sampling step used by speculative decoding
([10 §10.9](10-engine-and-server.md#109-speculative-decoding)) for a draft token proposed deterministically (a
one-hot proposal). It builds the same distribution `sample` draws from (penalties, temperature, top-k, top-p, min-p),
accepts the draft with probability `p(draft)` and otherwise draws from that distribution with the draft's mass
removed, so the returned token is distributed exactly like a plain `sample` result and equals the draft iff the
draft was accepted. Greedy decoding returns the argmax.

---

## 9.8 Randomness and seeds

`Sampler` uses `java.util.SplittableRandom`: fast, good statistical quality, and seedable. With `seed >= 0`
(`"seed": 42` in a request, `--seed 42`, or `sampling.seed`) the same prompt and settings produce the same
output every time *on the same backend* — useful for tests and for reproducing a bad answer. `seed = −1`
(the default of `sampling.seed`; a request without `seed` inherits whatever the settings say) seeds from the clock. Note that CPU and Metal can differ in the last bits of a logit (different
summation order), so a seed guarantees reproducibility per backend, not across them.

---

## 9.9 The vocabulary padding issue

`config.json` declares `vocab_size = 151936`, but `tokenizer.json` only defines ids up to 151 668 (151 643
regular + 26 added tokens). The remaining 267 rows of the embedding / LM-head matrix are **padding** chosen so
the matrix dimension is a multiple of 128 for efficient GPU kernels; they were never trained and their logits
are arbitrary. `Engine` therefore constructs the sampler with
`vocab = min(core.vocabSize(), tokenizer.vocabSize()) = 151 669` (`core` is the model's `Architecture.Core`, whose
`vocabSize` is the row count of the embedding and LM head; the logits array keeps the padded length), so `argmax`, the heap and the candidate scan
never look at those rows — otherwise a padding row would occasionally win and decode to an empty byte string.

---

## 9.10 Where each setting comes from

| request field (`/v1/chat/completions`, `/v1/completions`) | `jllm.properties` default | CLI alias | `SamplingParams` field |
|---|---|---|---|
| `temperature` | `sampling.temperature = 0.6` | `--temp` | `temperature` |
| `top_k` | `sampling.top-k = 20` | `--top-k` | `topK` |
| `top_p` | `sampling.top-p = 0.95` | `--top-p` | `topP` |
| `min_p` | `sampling.min-p = 0` | `--min-p` | `minP` |
| `repetition_penalty` | `sampling.repetition-penalty = 1.0` | `--rep-penalty` | `repetitionPenalty` |
| `presence_penalty` | `sampling.presence-penalty = 0` | — | `presencePenalty` |
| `frequency_penalty` | `sampling.frequency-penalty = 0` | — | `frequencyPenalty` |
| `seed` | `sampling.seed = -1` | `--seed` | `seed` |
| `max_tokens` / `max_completion_tokens` | `sampling.max-tokens = 1024` | `--max-tokens` | `maxTokens` |
| `stop` (string or array) | — | — | `stop` |

The bundled defaults mirror Qwen3's own `generation_config.json` (`T = 0.6, top_k = 20, top_p = 0.95`).
`SamplingParams`' constructor validates the parameters: every float must be **finite** (NaN and ±∞ are rejected,
for instance a `1e999` that overflows to infinity as a float), and `T ≥ 0`, `top_k ≥ 0`, `0 < top_p ≤ 1`,
`0 ≤ min_p ≤ 1`, `r > 0`. Presence and frequency penalties may be negative. The server turns a violation into an
HTTP 400 (`invalid_request_error`); `top_k` must additionally be an integer (a fraction or a negative number is a
400). A request value overrides the setting only if present; `max_tokens` (or `max_completion_tokens`, which wins)
must be at least 1 and is clamped to the room left in the context. The defaults come from the `sampling.*`
settings via `Settings.sampling()`; `SamplingParams.defaults()` (2048 tokens) is only the programmatic fallback.

---

## 9.11 Summary

1. Logits → (penalties) → candidates → softmax with temperature → top-p / min-p → one weighted random draw.
2. Temperature reshapes the gaps; `0` means greedy.
3. Top-k keeps `k` tokens using a size-k min-heap (`O(V log k)`); top-p keeps enough tokens to reach
   probability `p` (boundary token included); min-p keeps tokens within a factor of the best.
4. Penalties push logits of already-used tokens down, each in its own way.
5. The draw walks a cumulative sum with one `SplittableRandom` number; a seed makes it reproducible.
6. Only the 151 669 real token ids are ever eligible.
