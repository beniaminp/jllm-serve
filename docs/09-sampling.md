# 9. Sampling: from logits to a token

Code: `sampling/Sampler.java`, `sampling/SamplingParams.java`

The model outputs 151 936 **logits** — raw scores, any real number. Sampling turns them into one token id.

## Softmax: scores → probabilities

```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

`exp` makes everything positive, dividing by the sum makes them add up to 1. **Temperature `T`** reshapes the
distribution: `T → 0` makes the largest logit dominate (deterministic, "greedy"), `T = 1` is the model's native
distribution, `T > 1` flattens it (more random). Qwen3 recommends `T = 0.6`. Implementation detail: we subtract
the max logit before `exp` to avoid overflow — mathematically a no-op.

With `temperature = 0` the sampler returns `argmax` directly.

## Truncation: don't sample the garbage tail

Even with a low temperature, thousands of tokens have tiny but non-zero probability; summed up they are a real
chance of producing a nonsense token. Three filters, applied in this order, remove the tail:

- **top-k** (`top_k = 20`): keep only the `k` highest logits. Implemented with a size-k **min-heap** over the
  vocabulary (`topK()`): O(V log k) instead of sorting all V entries. The heap is then heap-sorted into descending
  order so the next filters can walk it.
- **top-p / nucleus** (`top_p = 0.95`): walk the sorted candidates accumulating probability; stop once the
  cumulative mass reaches `p`. Adapts to confidence: a sure answer keeps 1–2 tokens, an open question keeps many.
- **min-p**: drop candidates whose probability is below `min_p × p_max`. (Off by default.)

Without top-k, `nucleusCandidates()` avoids sorting the full vocabulary with a bound: a token with probability
below `(1 − p) / (V − 1)` can never be inside the nucleus, so it is discarded before sorting.

Then one random draw: `r = rng.nextDouble() × total`, walk the kept candidates until the running sum exceeds `r`.
`seed` makes this reproducible (`SplittableRandom`).

## Penalties: discourage loops

Small models love to repeat themselves. Three standard mechanisms, all applied to the logits before softmax:

- **repetition penalty** (`repetition_penalty`, HF style): for every token that appeared in the prompt or the
  output so far, `z = z > 0 ? z / r : z × r`. Values 1.05–1.2 are typical; 1 disables.
- **presence penalty** (OpenAI style): subtract a constant from every token already generated.
- **frequency penalty** (OpenAI style): subtract `f × count` — the more often, the stronger.

`Sampler` keeps a `seen[]` bitmap and `generatedCounts[]` so these are O(V) per step at most, and skipped
entirely when all are at their neutral value.

## The vocabulary padding detail

`config.json` says `vocab_size = 151936` but the tokenizer only defines 151 669 ids; the extra embedding rows are
padding (for GPU-friendly sizes) and were never trained. `Engine` passes `min(cfg.vocabSize, tokenizer.vocabSize)`
to the sampler so those rows can never be chosen — otherwise a random untrained row could occasionally win.

## Defaults

`SamplingParams.defaults()` mirrors Qwen3's `generation_config.json`: `T = 0.6, top_k = 20, top_p = 0.95`. The
API and CLI override any field (`temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`,
`presence_penalty`, `frequency_penalty`, `seed`, `max_tokens`, `stop`).
