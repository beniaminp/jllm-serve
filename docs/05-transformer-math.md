# 5. The transformer, step by step — with all the math

Code: `cpu/CpuBackend.java` (the most readable implementation), `model/Rope.java`, `model/ModelWeights.java`.
The Metal version in `metal/MetalBackend.java` + `kernels.metal` does the identical math on the GPU.

This is the heart of the project. Everything the model "knows" is applied here, and it is only four kinds of
arithmetic: dot products, exponentials, square roots and additions. This chapter builds each piece from
scratch, with small worked examples you can check by hand, then assembles them into the full forward pass with
Qwen3-0.6B's real dimensions.

---

## 5.0 The math toolbox

### Vectors

A vector is a list of `n` numbers: `x = [x₀, x₁, …, x_{n−1}]`. In this project a vector is a `float[]` or a
slice of one. In the model, **a token is a vector of `dim = 1024` numbers** at every point of the computation —
think of it as 1024 "features" describing the token *in its context*. Nobody assigns meaning to feature 317;
the training process found whatever features were useful.

### Dot product

The dot product of two vectors of the same length multiplies them element-wise and sums:

```
a · b = Σᵢ aᵢ bᵢ

[1, 2, 3] · [4, −5, 6] = 1·4 + 2·(−5) + 3·6 = 4 − 10 + 18 = 12
```

Its meaning: **how much two vectors point in the same direction** (times their lengths). Equal vectors give a
large positive number, unrelated ("orthogonal") ones give ≈ 0, opposite ones give a negative number. Attention
uses exactly this to decide which earlier tokens are relevant. Code: `VectorKernels.dot`.

### Matrix–vector product

A matrix `W` with `rows × cols` numbers, applied to a vector `x` of length `cols`, gives a vector of length
`rows` where **entry `r` is the dot product of row `r` with `x`**:

```
(W · x)[r] = Σ_c W[r][c] · x[c]

      ┌ 1  0  2 ┐   ┌ 3 ┐     ┌ 1·3 + 0·1 + 2·2 ┐   ┌ 7 ┐
W·x = │ 0  1  0 │ · │ 1 │  =  │ 0·3 + 1·1 + 0·2 │ = │ 1 │
      └ −1 1  1 ┘   └ 2 ┘     └ −3 + 1 + 2      ┘   └ 0 ┘
```

That one formula — "for every row, a dot product" — is >95 % of all the arithmetic in an LLM. Each of the
model's big weight matrices is a learned **linear map**: it mixes the input features into new features. `W` with
shape `[rows, cols]` maps a `cols`-vector to a `rows`-vector; that is why `q_proj.weight` is `[2048, 1024]`:
it turns a 1024-feature token into a 2048-number query. Code: `CpuBackend.matmul` / `matmul_q4` in Metal.

### Applying a matrix to many vectors at once

During prefill we have `n` token vectors. Stack them as rows of an `[n, cols]` matrix `X`; then `Y = X · Wᵀ`
is `[n, rows]` and row `t` of `Y` is `W · x_t`. Nothing new happens mathematically — it is the same dot products
— but it lets a kernel load a weight row once and use it for all `n` tokens, which is what makes prefill
cheaper per token than decode ([5.9](#59-prefill-decode-and-the-kv-cache)).

### The exponential function, exp

`exp(z)` (also written `e^z`, with `e ≈ 2.718`) is the function that raises the constant `e` to the power `z`.
Three properties matter for us:

```
   z     │  −5     −2     −1      0      1      2      5     10
 exp(z)  │ 0.0067  0.135  0.368   1    2.718  7.39   148   22026
```

1. **Always positive.** Whatever `z` is — negative, zero, huge — `exp(z) > 0`. That is what lets us turn
   arbitrary scores (which can be negative) into weights or probabilities (which must not be).
2. **Adding to the input multiplies the output.** `exp(a + b) = exp(a) · exp(b)`. So adding 1 to a score
   multiplies its weight by 2.718; adding 2.3 multiplies by 10; subtracting 2.3 divides by 10. *Differences* of
   scores become *ratios* of weights. A score 4.6 above another gets 100× the weight.
3. **Monotonic.** Bigger input → bigger output, always. The ordering of scores is preserved.

Property 2 is the reason exp (rather than, say, squaring) is used: a model only has to move a score by a few
units to shift a weight by orders of magnitude, which is both expressive and easy to learn.

### Softmax: scores → probabilities

We constantly have a list of *scores* — one per earlier token (attention), or one per vocabulary entry
(the logits) — and need to turn it into a **probability distribution**: numbers that are all ≥ 0 and sum to 1,
so they can be used as mixing weights or sampled from. Softmax does exactly that in two steps:

```
softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)          step 1: exp each score      step 2: divide by the total
```

**Worked example** with three scores:

```
z        = [ 2.0,    1.0,    0.1 ]
exp(z)   = [ 7.389,  2.718,  1.105 ]        each one positive
sum      = 11.212
softmax  = [ 0.659,  0.242,  0.099 ]        7.389/11.212, 2.718/11.212, 1.105/11.212  → adds up to 1.000
```

Read the result: the top score gets 66 % of the mass, the second 24 %, the third 10 %. Compare with the ratios
promised by property 2: `0.659 / 0.242 = 2.72 = exp(2.0 − 1.0)` ✓, `0.242 / 0.099 = 2.46 = exp(1.0 − 0.1)` ✓.
Only the *differences* between scores matter; the absolute level does not:

```
z + 100  = [102.0, 101.0, 100.1]  →  softmax = [0.659, 0.242, 0.099]      identical
```

**Why "soft" max?** A hard `max` would return `[1, 0, 0]` — all weight on the winner. Softmax gives the winner
the most, but keeps some for the runners-up in proportion to how close they were. That matters twice: during
training it makes the function differentiable (a hard max has no useful gradient), and at inference it lets
attention *blend* several tokens and lets sampling produce varied text instead of always the same word.

**How the gaps control the shape.** Scale all scores by a factor and the distribution sharpens or flattens:

```
scores ×0.1 : [0.2, 0.1, 0.01]  → softmax = [0.36, 0.33, 0.31]      nearly uniform ("flat")
scores ×1   : [2.0, 1.0, 0.1 ]  → softmax = [0.66, 0.24, 0.10]
scores ×3   : [6.0, 3.0, 0.3 ]  → softmax = [0.95, 0.05, 0.003]     nearly one-hot ("peaked")
```

This is exactly what **temperature** does in sampling ([09](09-sampling.md)): dividing logits by `T < 1`
multiplies the gaps, making the model more decisive; `T > 1` flattens them, making it more random; `T → 0`
is the hard max (greedy). It is also why attention divides scores by `sqrt(head_dim)` ([5.6](#56-step-4--attention)):
to keep the gaps in a range where softmax is neither flat nor one-hot.

**Where it appears in this project**

| place | scores | what the probabilities mean |
|---|---|---|
| attention (`CpuBackend.attention`, `attention` kernel) | `q · k_j / sqrt(hd)` for each cached token `j` | how much of token `j`'s value to mix into the output |
| MoE router (`CpuBackend.route`, `router` kernel) | `Wrouter · x` for each expert | which experts to pick and how to weight them |
| sampling (`Sampler.sample`) | logits for each vocabulary token, divided by temperature | the chance of emitting each token |

**The numerical trick: subtract the maximum.** `exp(z)` for `z = 100` is `2.7 × 10⁴³`, beyond a `float`'s
`3.4 × 10³⁸` — the result becomes `Infinity`, and `Infinity / Infinity` is `NaN`. Real logits can reach such
values. The fix uses the "absolute level doesn't matter" property: subtract any constant `m` from every score
and the softmax is unchanged, because

```
exp(zᵢ − m) / Σⱼ exp(zⱼ − m)  =  exp(zᵢ)·exp(−m) / (exp(−m)·Σⱼ exp(zⱼ))  =  exp(zᵢ) / Σⱼ exp(zⱼ)
```

(`exp(−m)` is a common factor of the top and the bottom and cancels). Choosing `m = max z` makes the largest
exponent exactly `exp(0) = 1` and all the others ≤ 1: no overflow is possible, and very negative differences
simply underflow to 0, which is the right answer anyway. That is why every softmax in the code — attention,
router, sampler — first finds the maximum, then computes `exp(z - max)`, then divides by the sum.

### Square root and the RMS

The **root mean square** of a vector is `rms(x) = sqrt((1/n) Σ xᵢ²)` — a measure of its typical magnitude.
`rms([3, 4]) = sqrt((9 + 16)/2) = sqrt(12.5) ≈ 3.54`. Dividing a vector by its RMS rescales it so its typical
entry is about ±1 regardless of how big it was.

---

## 5.1 Overview of one forward pass

```
                   token id
                      │
          ┌───────────▼───────────┐
          │ 0. embedding lookup   │  x ← E[id]                          1024
          └───────────┬───────────┘
                      │   x  (the "residual stream")
     ╔════════════════▼═════════════════════════════════════╗
     ║  layer ℓ  (ℓ = 0 … 27, identical structure, own weights) ║
     ║                                                        ║
     ║   xb = RMSNorm(x) ⊙ g_attn                1. normalize ║
     ║   q, k, v = Wq·xb, Wk·xb, Wv·xb          2. project    ║
     ║   q, k = RoPE(QKNorm(q)), RoPE(QKNorm(k)) 3. position  ║
     ║   store k, v in the KV cache at position p              ║
     ║   a = Attention(q, K[0..p], V[0..p])      4. attend     ║
     ║   x = x + Wo·a                            5. residual   ║
     ║                                                        ║
     ║   xb = RMSNorm(x) ⊙ g_ffn                 6. normalize ║
     ║   h  = silu(Wgate·xb) ⊙ (Wup·xb)          7. FFN       ║
     ║   x  = x + Wdown·h                        8. residual   ║
     ╚════════════════╤═════════════════════════════════════╝
                      │   (× 28)
          ┌───────────▼───────────┐
          │ 9. final RMSNorm      │
          │ 10. logits = Wlm · x  │                           151 936
          └───────────┬───────────┘
                      ▼
                   sampler
```

The vector `x` running down the middle is called the **residual stream**. Layers do not replace it, they *add*
to it (steps 5 and 8). Keep this picture in mind; the rest of the chapter is each box in detail.

Qwen3-0.6B's dimensions, used throughout:

| symbol | value | config.json name |
|---|---|---|
| `dim` | 1024 | `hidden_size` |
| `L` | 28 | `num_hidden_layers` |
| `H` | 16 | `num_attention_heads` |
| `Hkv` | 8 | `num_key_value_heads` |
| `hd` | 128 | `head_dim` |
| `qDim = H·hd` | 2048 | — |
| `kvDim = Hkv·hd` | 1024 | — |
| `ffn` | 3072 | `intermediate_size` |
| `V` | 151 936 | `vocab_size` |
| `ε` | 1e-6 | `rms_norm_eps` |
| `θ` | 1 000 000 | `rope_theta` |

---

## 5.2 Step 0 — Embedding: token id → vector

The embedding matrix `E` has shape `[V, dim] = [151936, 1024]`: one row per token in the vocabulary. The
input vector for token `id` is simply **row `id`**:

```
x = E[id]            (1024 numbers)
```

No arithmetic. The rows were learned so that tokens used in similar ways end up with similar vectors
(`" cat"` and `" dog"` have a large dot product; `" cat"` and `" parliament"` a small one).

**Worked example** (a toy vocabulary of 4 tokens, `dim = 3`):

```
E = ┌  0.2  −1.0   0.5 ┐   ← token 0  "the"
    │  0.9   0.1  −0.3 │   ← token 1  " cat"
    │  0.8   0.2  −0.2 │   ← token 2  " dog"
    └ −0.7   0.6   0.9 ┘   ← token 3  " parliament"

prompt ids [0, 1]  →  x_0 = E[0] = [0.2, −1.0, 0.5]      x_1 = E[1] = [0.9, 0.1, −0.3]

similarity:  E[1]·E[2] = 0.72 + 0.02 + 0.06 = 0.80       " cat" vs " dog": close
             E[1]·E[3] = −0.63 + 0.06 − 0.27 = −0.84     " cat" vs " parliament": far
```

**Shapes in and out.** In: `n` token ids (`int[]`). Out: the activation matrix `x` of shape `[n, dim]` —
for prefill of a 64-token chunk that is `64 × 1024` floats = 256 KB; for decode `1 × 1024`.

**What the code does.** `CpuBackend.forward` loops `t = 0 … n−1` and calls
`w.embed.dequantizeRow(tokens[off + t], x, t * dim)`: the row is stored as Q8_0 ([06](06-quantization.md)) and is
expanded to 1024 floats directly into row `t` of `x`. The Metal backend does the same on the CPU (it is 2048
numbers per token — far cheaper than a GPU dispatch) and writes them straight into the shared `x` buffer.

**Common misconception.** The embedding is *not* a one-hot vector multiplied by a matrix — that would be the same
result at 151 936× the cost. It is an array index. (The one-hot view is how textbooks explain why it is still a
"linear layer".)

---

## 5.3 Step 1 — RMSNorm

**Formula** (for a vector `x` of length `n` and a learned gain vector `g` of the same length):

```
y_i = x_i / sqrt( (1/n) Σⱼ xⱼ² + ε ) · g_i
```

**Worked example** (`n = 4`, `ε = 0`, `g = [1, 1, 2, 1]`):

```
x = [2, −4, 4, 2]
mean of squares = (4 + 16 + 16 + 4) / 4 = 10         rms = sqrt(10) = 3.162
x / rms = [0.632, −1.265, 1.265, 0.632]
y = that ⊙ g = [0.632, −1.265, 2.530, 0.632]
```

**Why it exists.** Each layer adds to the residual stream, so without control `x` could grow by a factor per
layer and after 28 layers be astronomically large (or tiny). Normalizing before each sub-block makes every block
see inputs of a standard size, whatever happened before. The gain `g` lets the model re-scale individual
features (it was learned; `input_layernorm.weight` is `g`). Note that `x` itself is *not* modified — only the
copy `xb` that feeds the block. `ε` is there so that an all-zero vector does not divide by zero.

(The older LayerNorm also subtracts the mean; RMSNorm skips that, which is cheaper and works as well.)

**What happens without it — a 3-layer thought experiment.** Suppose each layer's update is, on average, as
large as its input. Then `|x|` doubles per layer: after 28 layers it is `2²⁸ ≈ 2.7 × 10⁸` times the embedding.
The dot products inside attention would be in the `10¹⁶` range, every softmax would be exactly one-hot, and
float arithmetic would lose all precision. Normalizing the *input to each block* (not the stream itself) keeps
every block operating in the regime it was trained in, while the residual stream is free to grow.

**Why `ε` inside the square root.** For a vector that is exactly zero (it can happen for padding) the RMS is
0 and `x / 0` would be `NaN`, which then poisons every later computation. `ε = 10⁻⁶` is small enough to be
invisible for normal vectors (whose mean square is around 1) and large enough to keep the division finite.

**Shapes.** In: `[n, width]` where `width` is `dim` (1024) for the two per-layer norms and the final norm, or
`head_dim` (128) for the QK-norms of 5.5. Out: same shape. The gain `g` has length `width` and is the only
learned parameter — 1024 numbers per norm, negligible next to the matrices.

**What the code does** (`CpuBackend.rmsnormRow(in, io, out, oo, weight, width)`):

```java
float ss = VectorKernels.sumSquares(in, io, width);                 // Σ x²   (SIMD dot of x with itself)
float scale = (float) (1.0 / Math.sqrt(ss / width + c.normEps()));  // 1 / rms
for (int i = 0; i < width; i++)
    out[oo + i] = in[io + i] * scale * weight[i];                   // y = x · (1/rms) · g
```

Computing `1/rms` once and multiplying is cheaper than dividing 1024 times. The Metal `rmsnorm` kernel does the
same with one threadgroup per row: each thread sums the squares of a strided subset, the partial sums are
combined with `simd_sum` and threadgroup memory, then every thread scales its elements.

**Count per token:** `2 × 28 + 1 = 57` RMSNorms of width 1024 plus `24 × 28` of width 128 — about 150 k
multiply-adds, under 0.03 % of the forward pass.

---

## 5.4 Step 2 — Projections: query, key, value

From the normalized vector `xb` (1024) three vectors are computed with three weight matrices:

```
q = Wq · xb        Wq: [2048, 1024]   → 2048 numbers = 16 heads × 128
k = Wk · xb        Wk: [1024, 1024]   → 1024 numbers =  8 heads × 128
v = Wv · xb        Wv: [1024, 1024]   → 1024 numbers =  8 heads × 128
```

Think of attention as a soft database lookup:

- **key** `k` — what this token *advertises* about itself ("I am a noun", "I am the subject", "I am a number").
- **query** `q` — what the *current* token is *looking for* ("I am a verb, where is my subject?").
- **value** `v` — the information this token *hands over* if it is selected.

`k` and `v` are computed once per token and then **stored in the KV cache** (`keyCache[layer][pos·kvDim …]`),
because every future token will need them. `q` is used only now and thrown away.

### Heads

Rather than one lookup with 2048-wide vectors, the vectors are cut into **heads** of `hd = 128`: `q` is 16
heads, `k` and `v` are 8 heads. Each head runs the attention procedure independently with its own 128-number
slices and the results are concatenated. Why? Several small, independent lookups can specialize: in trained
models one head tracks the previous token, another finds the matching open bracket, another the subject of the
sentence. The cost is the same as one wide lookup.

### Grouped-query attention (GQA)

16 query heads but only 8 key/value heads: **query heads `2j` and `2j+1` share KV head `j`**. This halves the
size of the KV cache and the bandwidth to read it, at a tiny quality cost. In the code the KV head of query
head `h` is `h / group` with `group = H / Hkv = 2`, i.e. `ko = (h / group) * hd` in `attention()`.

**Worked example of a projection** (toy sizes: `dim = 3`, one head of `hd = 2`):

```
xb = [1.0, −0.5, 2.0]                       the normalized token

Wq = ┌ 0.5  0.0  1.0 ┐      q = Wq·xb = [0.5·1 + 0·(−0.5) + 1·2,  −1·1 + 2·(−0.5) + 0·2] = [2.5, −2.0]
     └ −1.0 2.0  0.0 ┘

Wk = ┌ 1.0  1.0  0.0 ┐      k = Wk·xb = [1 − 0.5 + 0,  0 + 0 + 2]                        = [0.5, 2.0]
     └ 0.0  0.0  1.0 ┘

Wv = ┌ 0.0  1.0  0.0 ┐      v = Wv·xb = [−0.5,  1.0]
     └ 1.0  0.0  0.0 ┘
```

Each output number is one dot product of a weight row with `xb`. At real size that is 2048 + 1024 + 1024 =
4096 dot products of length 1024 per token — **4.2 M multiply-adds**, the first big cost of the layer.

**Where each head lives in memory.** `q` is one array of 2048 floats; head `h` is the slice
`[h·128, (h+1)·128)`. Nothing is physically split — "16 heads" is a way of *reading* the same array. For `k`
and `v` the slices are `[j·128, (j+1)·128)` for `j < 8`. A token's `k` (1024 floats) is written at
`keyCache[layer][pos · 1024 …]`, so the cache of one layer is a `[ctx, 1024]` matrix indexed by position.

```
q (2048):  | head 0 | head 1 | head 2 | … | head 15 |        each 128 wide
k (1024):  | kv 0   | kv 1   | …  | kv 7 |                  query heads 0,1 → kv 0;  2,3 → kv 1; …
```

**What the code does.** Three `matmul` calls per layer; the `k` and `v` ones write directly into the cache
(`matmul(L.wk, xb, keyCache[l], pos * kvDim, n, false)`) — no separate copy step. On Metal they are three
`matmul_q4` dispatches with the KV cache buffer bound at byte offset `pos × kvDim × 4`.

**Why Q is wider than the model (2048 > 1024).** Qwen3 chose `head_dim = 128` with 16 heads, giving
`16 × 128 = 2048`. Nothing requires `H × hd = dim`; `Wq` projects up and `Wo` (5.7) projects back down.

---

## 5.5 Step 3 — QK-norm and rotary position embedding (RoPE)

### QK-norm (Qwen3 specific)

Before adding position information, Qwen3 applies RMSNorm *per head* to each 128-slice of `q` and `k`, with
learned gains `q_norm.weight`, `k_norm.weight` (length 128, shared across heads and positions). This keeps the
attention scores in a sane range and stabilizes training; at inference it is just 24 more small RMSNorms per
layer. Code: the `norm != null` branch of `normRope`.

### The problem RoPE solves

Look at the attention formula coming up in 5.6: it only uses dot products `q · k`. Nothing in `q` or `k` says
*where* in the sentence the token is — shuffle the tokens and every dot product is unchanged. Yet "dog bites
man" ≠ "man bites dog". Positions must be injected.

### Rotation in 2-D

Take a pair of numbers `(a, b)` as a point in the plane and rotate it by angle `φ`:

```
a' = a·cos φ − b·sin φ
b' = b·cos φ + a·sin φ
```

Example: `(1, 0)` rotated by 90° → `(0, 1)`; by 180° → `(−1, 0)`. Rotation preserves length.

Now the key fact. Rotate `q` by angle `α` and `k` by angle `β`; their dot product becomes

```
rot(q, α) · rot(k, β) = |q| |k| cos(angle(q) − angle(k) + α − β)
```

It depends on `α − β` only. **If `α` is proportional to the query's position `p` and `β` to the key's
position `s`, the score depends on `p − s`, the relative distance** — exactly what a language model needs
("the token 3 back") without ever being told absolute positions.

### Many frequencies

One angle per head would be too coarse. RoPE splits the 128-number head into 64 pairs and gives each pair `i`
its own rotation speed:

```
θ_i   = θ^(−2i/hd) = 1 000 000^(−i/64)           i = 0 … 63
angle = p · θ_i                                  p = position of the token

pair 0:   θ₀ = 1            one radian per position — distinguishes neighbours
pair 16:  θ₁₆ ≈ 0.0316      a full turn every ~200 positions
pair 32:  θ₃₂ = 0.001       a full turn every ~6 300 positions
pair 63:  θ₆₃ ≈ 1.2e−6      barely moves inside a 4096-token context
```

Low pairs resolve fine, nearby structure; high pairs encode coarse, long-range position. The large base
`θ = 10⁶` (vs 10⁴ in the original paper) is what lets Qwen3 handle 40 960-token contexts: the slow pairs do not
wrap around within the context.

**Which numbers form a pair?** Two conventions exist. Hugging Face (and therefore the weights) uses
"rotate-half": pair `(i, i + 64)`, i.e. the first half of the head with the second half. The code follows it:

```java
a = v[o + i];  b = v[o + i + half];            // half = 64
v[o + i]        = a * cos − b * sin;
v[o + i + half] = b * cos + a * sin;
```

`Rope` precomputes `cos(p·θ_i)` and `sin(p·θ_i)` for every `p < ctx` and `i < 64` once at start-up
(2 × 4096 × 64 floats = 2 MB), so applying RoPE is 4 multiplies per pair.

Both `q` and `k` are rotated; `v` is not (values carry content, not position).

**Worked example: the relative-position property with numbers.** Take one pair with `θ = 1` (pair 0) and
the same key content `k = (1, 0)` and query content `q = (1, 0)`. Place the key at position 2 and the query at
position 5 — then move both by +10 and check the score does not change:

```
rotate (1, 0) by angle p:   (cos p, sin p)

key   at s = 2:   k' = (cos 2, sin 2) = (−0.416, 0.909)
query at p = 5:   q' = (cos 5, sin 5) = ( 0.284, −0.959)
score = q'·k' = −0.118 − 0.872 = −0.990          = cos(5 − 2) = cos 3 = −0.990  ✓

shift both by 10:
key   at s = 12:  k' = (cos 12, sin 12) = (0.844, −0.537)
query at p = 15:  q' = (cos 15, sin 15) = (−0.760, 0.650)
score = −0.641 − 0.349 = −0.990                   same: only p − s = 3 matters ✓
```

The query "sees" a key 3 positions back identically wherever the pair sits in the text — which is what lets a
model trained on short texts handle any position, and what makes the KV cache valid: a stored key never needs
to be re-rotated when more tokens arrive.

**Why pairs and not single numbers?** A rotation needs two coordinates. Rotating 64 pairs with 64 different
speeds gives each head a rich, multi-scale "clock": pair 0 ticks once per token, pair 63 once per million
tokens. Two tokens 1 apart differ on the fast clocks; two tokens 1000 apart differ on the slow ones.

**Shapes.** In place on `q` (`[n, 2048]`, 16 heads) and `k` (`[n, 1024]`, 8 heads, already in the cache).
For token `t` of a prefill chunk starting at `pos`, the angle uses position `pos + t`.

**What the code does** (`CpuBackend.normRope(v, base, stride, nHeads, norm, n, pos)`):

```java
for (int t = 0; t < n; t++)                                  // each token
    for (int h = 0; h < nHeads; h++) {                       // each head
        int o = base + t * stride + h * hd;                  // start of this head's 128 numbers
        if (norm != null) rmsnormRow(v, o, v, o, norm, hd);  // QK-norm, in place
        for (int i = 0; i < half; i++) {                     // 64 pairs
            float a = v[o + i], b = v[o + i + half];
            float cs = cos[(pos + t) * half + i], sn = sin[(pos + t) * half + i];
            v[o + i]        = a * cs - b * sn;
            v[o + i + half] = b * cs + a * sn;
        }
    }
```

The Metal `norm_rope` kernel launches one threadgroup per `(head, token)` with 64 threads, one per pair; the
QK-norm's sum of squares is a 64-thread reduction, then each thread rotates its own pair.

**Cost:** 16 + 8 heads × 64 pairs × 4 multiplies ≈ 6 k per layer — negligible. The precomputed tables are
the reason it stays negligible: computing `cos`/`sin` on the fly would be the most expensive instruction in
the layer.

---

## 5.6 Step 4 — Attention

For one query head `h` of the token at position `p`, with its 128-number query `q_h` and the cached keys
`k_0 … k_p` and values `v_0 … v_p` of its KV head:

```
1. scores:   s_j = (q_h · k_j) / sqrt(hd)              j = 0 … p        (causal: only earlier tokens + itself)
2. weights:  w_j = softmax(s)_j = exp(s_j − max s) / Σ exp(s_i − max s)
3. output:   a_h = Σ_j w_j · v_j                        a 128-number weighted average of the values
```

**Worked example**, shrunk to `hd = 2` and three cached tokens:

```
q = [1, 1]
k₀ = [1, 0]    k₁ = [0, 1]    k₂ = [2, 2]
v₀ = [1, 0]    v₁ = [0, 1]    v₂ = [5, 5]

scores (÷ sqrt 2 = 1.414):  s₀ = 1/1.414 = 0.707   s₁ = 0.707   s₂ = 4/1.414 = 2.828
softmax: exp = [2.03, 2.03, 16.9], sum = 20.96  →  w = [0.097, 0.097, 0.806]
output: 0.097·[1,0] + 0.097·[0,1] + 0.806·[5,5] = [4.13, 4.13]
```

Token 2's key matched the query best, so the output is mostly token 2's value, with a little of the others. That
"a little of the others" — soft rather than hard selection — is what makes the whole thing differentiable and
trainable, and what lets a head blend information from several tokens.

**Why divide by `sqrt(hd)`?** A dot product of two random 128-vectors with unit-size entries has a typical
magnitude of `sqrt(128) ≈ 11`. Feeding ±11 into softmax makes it nearly one-hot and kills the gradient during
training; dividing by `sqrt(hd)` brings scores back to order 1. At inference we just do what training did.

**Causality.** Token `p` may only look at positions `≤ p`. During decode this is automatic (only `p+1` entries
exist in the cache). During prefill, token `t` of the batch uses `len = pos + t + 1` cached entries — see
`attention()`.

All 16 heads produce 128 numbers each; concatenated they form `a` (2048).

**What a head actually learns — intuition.** Keys and queries are learned so that useful pairs score high.
A "previous-token head" learns `Wq`/`Wk` such that the query at position `p` matches the key of position
`p − 1` through the RoPE clocks (5.5). An "induction head" looks for an earlier occurrence of the current
token and copies what followed it — the mechanism behind in-context learning. Nobody programs these; they
emerge from training. At inference we only run the arithmetic.

**Shapes and the two loops.** For a prefill chunk of `n` tokens there are `n × 16` independent
`(token, head)` jobs; each job scores `pos + t + 1` cached keys, so the work is a triangle (later tokens
score more keys). Inputs: `q [n, 2048]`, the cache `K, V [ctx, 1024]`. Output: `att [n, 2048]`.

**What the code does** (`CpuBackend.attention(layer, n, pos)`), one `(t, h)` job:

```java
int ko = (h / group) * hd;                     // which KV head's slice this query head reads
int qo = t * qDim + h * hd;                    // this query head's 128 numbers
int len = pos + t + 1;                         // causal: keys 0 … pos+t

float max = -inf;
for (int s = 0; s < len; s++) {                // 1. scores
    sc[s] = VectorKernels.dot(q, qo, kc, s * kvDim + ko, hd) * scale;     // q·k_s / sqrt(128)
    max = Math.max(max, sc[s]);
}
float sum = 0;
for (int s = 0; s < len; s++) {                // 2. softmax numerators
    sc[s] = (float) Math.exp(sc[s] - max);
    sum += sc[s];
}
Arrays.fill(att, qo, qo + hd, 0f);
for (int s = 0; s < len; s++)                  // 3. weighted sum of values
    VectorKernels.axpy(sc[s] / sum, vc, s * kvDim + ko, att, qo, hd);     // att += w_s · v_s
```

`sc[]` is a per-thread scratch array of length `ctx`, so the three passes never allocate. The thread pool
hands out `(t, h)` jobs in chunks ([07](07-cpu-backend.md)). The Metal `attention` kernel runs the same three
phases with one 256-thread threadgroup per `(token, head)` and keeps the scores in a device scratch buffer
([08](08-metal-backend.md)).

**Cost per token:** `16 heads × (pos+1) keys × 128 × 2` multiply-adds (scores and values) — at `pos = 1000`
that is 4 M per layer, comparable to one of the projection matrices; at `pos = 4096` it exceeds all of them.
This growth is the "quadratic attention" everyone talks about: total work over a sequence of length `N` is
`O(N²)`.

**Why the KV cache holds K and V but not Q.** Each query is used exactly once (by its own token, now). Each key
and value is used by *every later* token. Caching Q would waste memory; not caching K/V would waste compute.

---

## 5.7 Step 5 — Output projection and the residual connection

```
x = x + Wo · a           Wo: [1024, 2048]
```

`Wo` mixes the 16 heads' outputs back into the 1024-feature space, and the result is **added** to the residual
stream rather than replacing it.

**Why add?** Two reasons. (1) *Trainability*: a 28-layer stack of pure replacements is extremely hard to train —
gradients vanish. With additions, every layer only has to learn a small *correction* on top of what is already
there, and information from the embedding can reach the last layer untouched. (2) *Interpretation*: each layer
reads the stream, computes something, and writes its contribution back. Layers communicate through this shared
scratchpad.

**Worked example** (toy `dim = 3`, two heads of `hd = 1`, so `a` has 2 numbers):

```
x  (residual stream before attention) = [0.2, −1.0, 0.5]
a  (concatenated head outputs)        = [4.0, −1.0]

Wo = ┌ 0.1  0.0 ┐      Wo·a = [0.4, −0.4 + 0.5·(−1.0)·0 …]   let's do it fully:
     │ 0.0  0.5 │             row 0: 0.1·4.0 + 0.0·(−1.0) = 0.40
     └ 0.2  0.2 ┘             row 1: 0.0·4.0 + 0.5·(−1.0) = −0.50
                              row 2: 0.2·4.0 + 0.2·(−1.0) = 0.60

x += Wo·a  →  x = [0.2 + 0.40, −1.0 − 0.50, 0.5 + 0.60] = [0.60, −1.50, 1.10]
```

The stream now carries both what it had (the token's own embedding and earlier layers' work) and the new
context information mixed in by attention.

**Shapes.** In: `att [n, 2048]`. `Wo [1024, 2048]`. Out: added to `x [n, 1024]`.

**What the code does.** `matmul(L.wo, att, x, 0, n, true)` — the last argument `acc = true` makes the kernel
compute `x[r] += dot(...)` instead of `x[r] = dot(...)`, so the residual addition costs nothing extra. On Metal
the `matmul_q4` kernel has the same flag in its parameter struct (`accumulate`).

**Why there is no normalization *after* the block.** Older designs ("post-norm") normalized the sum; modern
ones ("pre-norm", used here) normalize only the block's input and leave the stream raw. Pre-norm trains far
more reliably at depth, which is why every current LLM uses it.

---

## 5.8 Steps 6–8 — The feed-forward network (SwiGLU)

```
xb   = RMSNorm(x) ⊙ g_ffn                     1024
gate = Wgate · xb                              3072
up   = Wup · xb                                3072
h    = silu(gate) ⊙ up                         3072   (⊙ = element-wise product)
x    = x + Wdown · h                           1024
```

where **SiLU** (sigmoid linear unit, also called *swish*) is

```
silu(z) = z · σ(z) = z / (1 + e^(−z))

silu(−4) = −0.07    silu(−1) = −0.27    silu(0) = 0    silu(1) = 0.73    silu(4) = 3.93
```

A smooth version of "keep positive inputs, drop negative ones". Without such a *non-linearity* the whole network
would collapse into a single matrix (a product of linear maps is linear) and could not represent anything
interesting.

**Why the gate?** `silu(gate) ⊙ up` is a *gated linear unit*: `gate` decides, per hidden unit, how much of
`up` passes through. Experiments showed GLU variants beat plain `relu(W·x)` at equal parameter count, so
practically every modern LLM uses SwiGLU.

**What the FFN does.** Attention moved information *between* tokens; the FFN processes each token *on its own*
(the same for every position). The 3072 hidden units act like learned feature detectors ("this token is a city
name", "we are inside a code block") whose outputs `Wdown` turns back into residual-stream features. Most of a
model's factual knowledge is believed to live in these matrices — which is why the MoE models scale exactly this
block ([12](12-mixture-of-experts.md)).

**Worked example** (toy `dim = 2`, `ffn = 3` hidden units):

```
xb = [1.0, 2.0]

Wgate = ┌  1.0  0.0 ┐      gate = Wgate·xb = [ 1.0,  −0.5·1 + 0.5·2,  −1.0·1 − 1.0·2 ] = [1.0, 0.5, −3.0]
        │ −0.5  0.5 │
        └ −1.0 −1.0 ┘

Wup   = ┌  0.0  1.0 ┐      up   = Wup·xb   = [2.0,  3.0,  5.0]
        │  1.0  1.0 │
        └  2.0  1.5 ┘

silu(gate) = [1.0·σ(1.0), 0.5·σ(0.5), −3.0·σ(−3.0)] = [0.731, 0.311, −0.142]
h = silu(gate) ⊙ up = [0.731·2.0, 0.311·3.0, −0.142·5.0] = [1.462, 0.933, −0.711]

Wdown = ┌ 1.0  0.0  1.0 ┐   Wdown·h = [1.462 − 0.711,  0.933·0.5 + (−0.711)·(−1.0)] = [0.751, 1.178]
        └ 0.0  0.5 −1.0 ┘
x += Wdown·h
```

Read what the gate did: hidden unit 2 had a strongly negative gate (−3.0), so its `up` value of 5.0 was almost
entirely suppressed (−0.142 × 5.0). Unit 0 was "open" (gate 1.0) and passed 73 % of its `up` value. The gate
is a learned, input-dependent switch per hidden unit.

**Shapes.** In: `xb [n, 1024]`. `wGateUp [6144, 1024]` → `gateUp [n, 6144]`; `hidden [n, 3072]`;
`Wdown [1024, 3072]` → added to `x [n, 1024]`.

**What the code does** (`CpuBackend`):

```java
matmul(L.wGateUp, xb, gateUp, 0, n, false);   // one pass: rows 0..3071 = gate, 3072..6143 = up
siluMul(n);                                   // hidden[i] = gate[i] / (1 + exp(-gate[i])) * up[i]
matmul(L.wDown, hidden, x, 0, n, true);       // x += Wdown · hidden
```

**Implementation detail:** `ModelWeights` stacks `Wgate` and `Wup` into one `[6144, 1024]` matrix so a single
pass over `xb` produces both (`wGateUp`); `siluMul` then reads halves `[0, 3072)` and `[3072, 6144)`. The
stacking is free because both matrices have the same number of columns (1024) — rows are simply concatenated
([03 §3.1](03-safetensors.md#31-what-is-a-tensor)).

**Cost:** `6144 × 1024 + 1024 × 3072 = 9.4 M` multiply-adds per token per layer — **60 % of the layer**. The
FFN, not attention, is where the time goes at short context, and it is why MoE models scale this block.

---

## 5.9 Prefill, decode, and the KV cache

Generating text means calling the forward pass once per new token. Consider token number 500: its attention
needs `k_0 … k_499`. Those keys depend only on *their own* tokens (through steps 1–3), so they are exactly the
same values as when they were computed for tokens 1 … 499. Recomputing them would make every step cost
O(n) forward passes — O(n²) total for a conversation. The **KV cache** stores them instead:

```
keyCache[layer]  : [ctx, kvDim]  floats     = 28 × 4096 × 1024 × 4 B = 470 MB at ctx = 4096
valCache[layer]  : same                                                 (940 MB together)
```

`Backend.forward(tokens, off, n, pos, logits)` processes `n` tokens at positions `[pos, pos+n)`, *appends*
their K/V to the cache, and optionally returns the logits of the last one. Two regimes:

| | prefill | decode |
|---|---|---|
| tokens per call | up to `--batch` (64) | 1 |
| matmul shape | `[64, 1024] × W` | `[1, 1024] × W` |
| weight reuse | each weight row used 64× | each weight row used once |
| bound by | arithmetic | **memory bandwidth** (450 MB of weights per token) |
| measured (Metal) | ~344 tok/s | ~75–110 tok/s |

The engine runs prefill in chunks of 64 only because the activation buffers are sized for that; a longer prompt
is just more chunks. Logits are computed only for the final token of the prompt — earlier tokens are processed
solely to populate the cache.

**A timeline.** Prompt of 5 tokens, then 3 generated tokens, `--batch 64`:

```
call                      tokens fed          positions   cache afterwards        logits for
forward(prompt, 0, 5, 0)  t0 t1 t2 t3 t4       0..4        [k0 k1 k2 k3 k4]       t4  → sample t5
forward([t5], 0, 1, 5)    t5                   5           [k0 … k5]              t5  → sample t6
forward([t6], 0, 1, 6)    t6                   6           [k0 … k6]              t6  → sample t7
forward([t7], 0, 1, 7)    t7                   7           [k0 … k7]              t7  → sample t8 …
```

Each decode call attends to everything before it without recomputing anything: the cost of generating token
number 1000 is one 1-token forward pass plus 1000 dot products per head per layer — not 1000 forward passes.

**Where the cache lives.** CPU: `float[layers][ctx × kvDim]` on the Java heap (so `-Xmx` must cover it).
Metal: one shared `MTLBuffer` per layer for K and one for V, written by the `k`/`v` matmul dispatches and read
by the `attention` kernel. Both are plain f32 by default. `backend.kv-dtype=f16` stores the cache as IEEE
half floats instead (CPU: `short[]`, Metal: `half` buffers): K and V are still projected, normalized and rotated
in f32, rounded **once** (round-to-nearest-even) when they are stored, and expanded back to f32 inside
attention, so the arithmetic is unchanged and only the stored values lose precision (~3 significant decimal
digits instead of ~7). That halves the cache — 470 MB instead of 940 MB at `ctx = 4096` for the 0.6B model —
and the bytes attention reads per token; greedy outputs on the test prompts are unchanged
([07 §7.12](07-cpu-backend.md#712-the-f16-kv-cache-kvcache), [08 §8.9](08-metal-backend.md#89-the-f16-kv-cache),
[11 §11.3](11-performance.md#113-memory-footprint)).

**The prefix cache on top** ([10](10-engine-and-server.md)): the engine remembers which token ids currently
occupy the cache. When the next request shares a prefix (a chat history), only the new tokens are prefilled.

---

## 5.10 Steps 9–10 — Final norm and logits

```
xb     = RMSNorm(x) ⊙ g_final                  1024
logits = Wlm · xb                              151 936
```

`Wlm` has shape `[V, dim]`: row `i` is a learned vector for output token `i`, and `logits[i]` is its dot product
with the final state — "how compatible is the current state with emitting token `i`". With
`tie_word_embeddings = true`, `Wlm` **is the embedding matrix `E`**: the same vectors are used to read tokens in
and to score them on the way out, saving 155 M parameters. (Qwen3-30B-A3B has a separate `lm_head.weight`.)

**Worked example** (toy vocabulary of 4, `dim = 3`, reusing `E` from 5.2 as `Wlm`):

```
final state after norm:  xb = [0.8, 0.3, −0.4]

logits = E · xb:
   token 0 "the":          0.2·0.8 − 1.0·0.3 + 0.5·(−0.4) = 0.16 − 0.30 − 0.20 = −0.34
   token 1 " cat":         0.9·0.8 + 0.1·0.3 − 0.3·(−0.4) = 0.72 + 0.03 + 0.12 =  0.87
   token 2 " dog":         0.8·0.8 + 0.2·0.3 − 0.2·(−0.4) = 0.64 + 0.06 + 0.08 =  0.78
   token 3 " parliament": −0.7·0.8 + 0.6·0.3 + 0.9·(−0.4) = −0.56 + 0.18 − 0.36 = −0.74

softmax (T = 1):  [0.11, 0.37, 0.34, 0.07]     " cat" is most likely, " dog" close behind
```

The state `xb` "points toward" the `" cat"`/`" dog"` region of embedding space, so those rows score highest.
With tied embeddings this geometric picture is exact: the model predicts the next token by producing a vector
that resembles that token's embedding.

**Shapes.** In: `x [n, 1024]` — but only row `n − 1` is used. `Wlm [151936, 1024]`. Out: `logits [151936]`.

**What the code does.**

```java
if (logits != null) {
    rmsnormRow(x, (n - 1) * dim, xb, 0, w.finalNorm, dim);   // normalize the LAST token only, into xb row 0
    matmul(w.lmHead, xb, logits, 0, 1, false);               // 151 936 dot products of length 1024
}
```

**Cost:** 155.6 M multiply-adds — **a quarter of the whole forward pass** for this small model, and in bytes
(Q8_0: 165 MB) over a third of all weight traffic per token. Larger models are less dominated by it because
their layers grow while the vocabulary does not.

The logits go to the sampler ([09](09-sampling.md)), which applies softmax (with temperature) and draws a token.
Note that the forward pass itself never computes a softmax over the vocabulary — the sampler does, only once.

---

## 5.11 Shapes and cost of one layer

Every matrix in a layer, its shape, and the multiply-adds it costs per token:

| matrix | shape `[out, in]` | multiply-adds |
|---|---|---|
| `Wq` | [2048, 1024] | 2.10 M |
| `Wk` | [1024, 1024] | 1.05 M |
| `Wv` | [1024, 1024] | 1.05 M |
| `Wo` | [1024, 2048] | 2.10 M |
| `Wgate` + `Wup` | [6144, 1024] | 6.29 M |
| `Wdown` | [1024, 3072] | 3.15 M |
| attention (`p` cached tokens) | — | 16 heads × `p` × 128 × 2 ≈ 4 096·`p` |
| **total per layer** | 15.7 M parameters | ≈ 15.7 M + 4 096·`p` |

For 28 layers plus the LM head (`151 936 × 1024 = 155.6 M`): **≈ 596 M multiply-adds per token** at short
context (≈ 1.2 GFLOP counting multiply and add separately), growing by `28 × 4096 × p ≈ 115 k·p` with context
length — at `p = 4096` attention adds another 0.47 G, nearly doubling the work. That growth is why long contexts
get slower and why attention gets its own, specialized kernels.

---

## 5.12 Where each tensor comes from

| name in the checkpoint | field in `ModelWeights.Layer` | shape | step |
|---|---|---|---|
| `model.embed_tokens.weight` | `embed` (and `lmHead` when tied) | [151936, 1024] | 0, 10 |
| `model.layers.N.input_layernorm.weight` | `attnNorm` | [1024] | 1 |
| `model.layers.N.self_attn.q_proj.weight` | `wq` | [2048, 1024] | 2 |
| `model.layers.N.self_attn.k_proj.weight` | `wk` | [1024, 1024] | 2 |
| `model.layers.N.self_attn.v_proj.weight` | `wv` | [1024, 1024] | 2 |
| `model.layers.N.self_attn.q_norm.weight` | `qNorm` | [128] | 3 |
| `model.layers.N.self_attn.k_norm.weight` | `kNorm` | [128] | 3 |
| `model.layers.N.self_attn.o_proj.weight` | `wo` | [1024, 2048] | 5 |
| `model.layers.N.post_attention_layernorm.weight` | `ffnNorm` | [1024] | 6 |
| `model.layers.N.mlp.gate_proj.weight` + `up_proj.weight` | `wGateUp` (stacked) | [6144, 1024] | 7 |
| `model.layers.N.mlp.down_proj.weight` | `wDown` | [1024, 3072] | 8 |
| `model.norm.weight` | `finalNorm` | [1024] | 9 |

`ModelWeights.checkShapes` verifies all of these against `config.json` at load time, so a mismatched or stale
file fails immediately instead of producing noise.

---

## 5.13 The whole thing in 25 lines

This is `CpuBackend.forward` with the bookkeeping removed; every line maps to a section above.

```
x = E[token]                                                   # 5.2
for ℓ in 0..27:
    xb = rmsnorm(x) ⊙ g_attn[ℓ]                                # 5.3
    q = Wq[ℓ]·xb;  k = Wk[ℓ]·xb;  v = Wv[ℓ]·xb                 # 5.4   (k, v written into the cache at pos)
    for each head: q_h = rope(rmsnorm(q_h) ⊙ g_q, pos)         # 5.5
    for each kv head: k_h = rope(rmsnorm(k_h) ⊙ g_k, pos)
    for each head h:                                           # 5.6
        s_j = q_h · K[ℓ][j, kv(h)] / sqrt(128)       j ≤ pos
        w = softmax(s)
        a_h = Σ_j w_j · V[ℓ][j, kv(h)]
    x = x + Wo[ℓ]·a                                            # 5.7
    xb = rmsnorm(x) ⊙ g_ffn[ℓ]                                 # 5.8
    h = silu(Wgate[ℓ]·xb) ⊙ (Wup[ℓ]·xb)
    x = x + Wdown[ℓ]·h
logits = Wlm · (rmsnorm(x) ⊙ g_final)                          # 5.10
```

Everything else in the project — quantization, SIMD, GPU kernels, the server — exists to run these 25 lines
fast and to get text in and out of them.
