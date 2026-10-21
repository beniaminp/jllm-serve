# 00b. From prompt to answer, on paper

This chapter follows one question through a language model, from the text you type to the answer it prints, one
token at a time. Every number is written down, so you can redo the whole calculation with a pen, paper and a
pocket calculator.

The model is a toy. Every number in it is picked by hand, small and round, so that the sums stay short. It has
exactly the parts of the real models jllm runs (Qwen3, Llama, …), only smaller:

| | this toy | Qwen3-0.6B |
|---|---|---|
| vocabulary (tokens it knows) | 20 | 151 936 |
| width of a token's vector `d` | 4 | 1 024 |
| layers | 2 | 28 |
| attention heads × head width | 2 × 2 | 16 query heads × 128 (8 key/value heads) |
| feed-forward hidden width | 2 | 3 072 |
| numbers in the weights | 332 | ~596 million |

What the toy keeps from Qwen3: the RMSNorm before attention and before the feed-forward step (each with its own
weights), the q-norm and k-norm inside every head, RoPE, the residual adds, SwiGLU, and a stack of layers (two).

What the toy leaves out:
- the attention output matrix `Wo` is the identity, so it changes nothing;
- every head has its own keys and values (Qwen3 shares one key/value head between two query heads);
- the output matrix is separate from the embedding table (Qwen3-0.6B reuses the table).

None of these changes the recipe. The real model simply repeats each step with bigger vectors, 28 times. The
same steps with real sizes are in [05](05-transformer-math.md).

---

## 0. Rules for doing it by hand

- **Vectors are rows.** Multiplying a row vector `x` (length n) by a matrix `W` (n rows, m columns) gives a row of
  length m. Entry `j` is the dot product of `x` with column `j`:

  ```
  (x·W)_j = x_0·W[0][j] + x_1·W[1][j] + … + x_(n-1)·W[n-1][j]
  ```

  The weight matrices below hold only −1, 0 and 1, so every multiplication is "add it", "subtract it" or "skip it".
- **Round to 3 decimals after every step**, and use the rounded number in the next step. Inside one formula (a
  sum of products, or one `silu`), keep the in-between values exact (or at least 4 decimals) and round only the
  result. Doing this, you get exactly the numbers in this chapter.
- **The calculator does** `√`, `eˣ`, `cos` and `sin` (in radians).
- **Positions count from 0.**

---

## 1. The vocabulary

The model only knows 20 tokens. Each token has a number, its **id**:

| id | token | id | token | id | token | id | token |
|---|---|---|---|---|---|---|---|
| 0 | `<\|user\|>` | 5 | `sky` | 10 | `the` | 15 | `r` |
| 1 | `<\|assistant\|>` | 6 | `grass` | 11 | `color` | 16 | `co` |
| 2 | `<\|end\|>` | 7 | `is` | 12 | `c` | 17 | `col` |
| 3 | `?` | 8 | `blue` | 13 | `o` | 18 | `or` |
| 4 | `.` | 9 | `green` | 14 | `l` | 19 | `red` |

Tokens 0, 1 and 2 are **special tokens**. Nobody types them. They mark who is speaking (`<|user|>`,
`<|assistant|>`) and where an answer ends (`<|end|>`).

---

## 2. The chat template: from a question to one long text

You type:

```
sky color?
```

The model never sees a "conversation", only one sequence of tokens. The chat template wraps your message in the
special tokens, and ends with the token that means "now the assistant speaks":

```
<|user|> sky color? <|assistant|>
```

Whatever the model writes next is its answer.

(Real templates are longer, e.g. Qwen3's `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`; see
[04](04-tokenizer.md). The idea is the same.)

---

## 3. Tokenizing: from text to ids

**Step 3a, split into words.** The special tokens are cut out first. Then the text is split at spaces and in
front of punctuation:

```
<|user|>   sky   color   ?   <|assistant|>
```

**Step 3b, BPE on each word.** Byte-pair encoding starts from single letters and applies a fixed, ordered list of
merges. Each merge glues two neighbouring pieces into one, as long as the result is a vocabulary token.

This toy's merge list, in order:

```
merge 1:  o + r    → or
merge 2:  c + o    → co
merge 3:  co + l   → col
merge 4:  col + or → color
```

Applied to `color`:

```
start:     c  o  l  o  r
merge 1:   c  o  l  or        (the pair "o r" at the end; "o l" is not in the list)
merge 2:   co  l  or          (the pair "c o")
merge 3:   col  or            (the pair "co l")
merge 4:   color              (the pair "col or")
```

`sky` and `?` are tokens already. A word the merges cannot finish stays in pieces: `cool` would become
`co`, `o`, `l`.

**Step 3c, look up the ids:**

```
tokens:    <|user|>   sky   color   ?   <|assistant|>
ids:           0       5     11     3        1
position:      0       1      2     3        4
```

So the prompt is the id list **[0, 5, 11, 3, 1]** at positions **0, 1, 2, 3, 4**.

---


## 4. The weights (everything the model "knows")

### 4a. The embedding table `E` (one row of 4 numbers per token)

Only six rows are used in this chapter:

| token | id | E row |
|---|---|---|
| `<\|user\|>` | 0 | [1, 0, 0, −1] |
| `<\|assistant\|>` | 1 | [−1, 0, 1, 1] |
| `?` | 3 | [0, −1, 0, 1] |
| `sky` | 5 | [0, 1, 1, 0] |
| `blue` | 8 | [0, 1, 0, 1] |
| `color` | 11 | [1, 1, 0, 0] |

The other 14 rows exist but are never looked up here.

### 4b. Layer 1

**Norm weights.** RMSNorm before attention: `g_attn = [1, 2, 1, 0.5]`. RMSNorm before the feed-forward step: `g_ffn = [1, 1, 0.5, 0.5]`. q-norm: `g_q = [1, 0.5]`. k-norm: `g_k = [0.5, 1]` (one weight per number of a head, shared by both heads).

**Attention matrices** (4 × 4):

```
      Wq             Wk             Wv
  [1 0 0 0]      [1 0 0 0]      [1 0 0 0]
  [0 1 1 0]      [0 1 0 0]      [0 1 0 0]
  [0 0 1 0]      [0 1 1 0]      [0 0 1 0]
  [0 0 0 1]      [0 0 0 1]      [0 0 1 1]
```

Written out for an input row `n = [n0, n1, n2, n3]` (column j of the matrix says which n's to add):

```
q = n·Wq = [n0,  n1,  n1 + n2,  n3]
k = n·Wk = [n0,  n1 + n2,  n2,  n3]
v = n·Wv = [n0,  n1,  n2 + n3,  n3]
```

**Feed-forward matrices** (SwiGLU):

```
  Wgate (4×2)      Wup (4×2)       Wdown (2×4)
    [ 1  0]          [0 1]        [ 1  0  0  1]
    [ 0  1]          [1 0]        [ 0  1 −1  0]
    [ 1  0]          [0 1]
    [ 0 −1]          [1 0]
```

For an input row `n` and a hidden row `a = [a0, a1]`:

```
gate = [n0 + n2,  n1 − n3]
up   = [n1 + n3,  n0 + n2]
down = a·Wdown = [a0,  a1,  −a1,  a0]
```

`Wo` is the identity: the attention output goes through unchanged.

### 4c. Layer 2

**Norm weights.** RMSNorm before attention: `g_attn = [0.5, 1, 1, 2]`. RMSNorm before the feed-forward step: `g_ffn = [1, 0.5, 1, 1]`. q-norm: `g_q = [1, 1]`. k-norm: `g_k = [1, 2]` (one weight per number of a head, shared by both heads).

**Attention matrices** (4 × 4):

```
      Wq             Wk             Wv
  [0 1 0 0]      [1 0 0 0]      [1 0 0 1]
  [1 0 0 0]      [0 1 0 0]      [0 1 0 0]
  [0 0 0 1]      [0 0 1 0]      [0 0 1 0]
  [0 0 1 0]      [0 0 0 1]      [0 0 0 1]
```

Written out for an input row `n = [n0, n1, n2, n3]` (column j of the matrix says which n's to add):

```
q = n·Wq = [n1,  n0,  n3,  n2]
k = n·Wk = [n0,  n1,  n2,  n3]
v = n·Wv = [n0,  n1,  n2,  n0 + n3]
```

**Feed-forward matrices** (SwiGLU):

```
  Wgate (4×2)      Wup (4×2)      Wdown (2×4)
     [0 1]          [ 1  0]        [0 1 0 1]
     [1 0]          [ 0  1]        [1 0 1 0]
     [0 1]          [−1  0]
     [1 0]          [ 0  1]
```

For an input row `n` and a hidden row `a = [a0, a1]`:

```
gate = [n1 + n3,  n0 + n2]
up   = [n0 − n2,  n1 + n3]
down = a·Wdown = [a1,  a0,  a1,  a0]
```

`Wo` is the identity: the attention output goes through unchanged.

### 4d. After the last layer

Final RMSNorm weights: `g_final = [2, 1, 1, 0.5]`.

The output matrix `Wout` (20 × 4, one row per token):

| id | token | row | id | token | row |
|---|---|---|---|---|---|
| 0 | `<\|user\|>` | [0, 0, 0, 1] | 10 | `the` | [1, 0, 0, 0] |
| 1 | `<\|assistant\|>` | [0, −1, 0, 1] | 11 | `color` | [−1, 0, 1, 0] |
| 2 | `<\|end\|>` | [1, 1, 0, 0] | 12 | `c` | [0, 1, 0, 0] |
| 3 | `?` | [0, −1, 0, 0] | 13 | `o` | [0, 0, 0, −1] |
| 4 | `.` | [1, 0, 0, 1] | 14 | `l` | [−1, 0, 0, 0] |
| 5 | `sky` | [0, 0, 1, 1] | 15 | `r` | [0, 0, −1, 0] |
| 6 | `grass` | [1, 0, −1, 0] | 16 | `co` | [1, 0, 1, 0] |
| 7 | `is` | [0, 0, 1, 0] | 17 | `col` | [0, 1, 1, 0] |
| 8 | `blue` | [−1, −1, 1, 0] | 18 | `or` | [−1, −1, 0, 0] |
| 9 | `green` | [0, 1, −1, 0] | 19 | `red` | [−1, 0, 1, 1] |

### 4e. The two heads

The 4 numbers of `q`, `k` and `v` are split into two heads:
- **head 0** owns positions 0 and 1 of each vector;
- **head 1** owns positions 2 and 3.

Each head does its own attention, and their outputs are put back side by side.

---

## 5. Generating token 1 (the "prefill" of the prompt)

The whole prompt goes through the model at once, layer by layer. Only the **last** position (`<|assistant|>`,
position 4) predicts the next token, but the other positions are still needed:
- **in layer 1, every position needs its full output**, because that output is the input of layer 2;
- **in layer 2 (the last layer), the earlier positions need only their keys and values**, which position 4 looks at.
  Their queries, attention and feed-forward in layer 2 would only matter for a layer 3.

### 5.1 Embedding: look up each id's row

```
pos 0  <|user|>       x0 = [1, 0, 0, −1]
pos 1  sky            x1 = [0, 1, 1, 0]
pos 2  color          x2 = [1, 1, 0, 0]
pos 3  ?              x3 = [0, −1, 0, 1]
pos 4  <|assistant|>  x4 = [−1, 0, 1, 1]
```

### Layer 1

### 5.2 RMSNorm: make every vector the same size, then weight each number

```
RMSNorm(x) = x · (1 / √(mean of x²)) · g        (number by number with the weights g)
```

The first part scales the vector so that its numbers have an average square of 1. The weights `g` are learned:
they let the model make some of the 4 numbers louder (`g > 1`) or quieter (`g < 1`). (The real formula adds a
tiny ε ≈ 10⁻⁶ under the root so it never divides by zero; it is too small to matter here.)

Position 0 with `g_attn = [1, 2, 1, 0.5]`:

```
x            = [1, 0, 0, −1]
squares      = [1, 0, 0, 1]
mean         = (1 + 0 + 0 + 1) / 4 = 0.5
√ of mean    = √0.5 = 0.707
1 / √        = 1 / 0.707 = 1.414
n0 = x · 1.414 · g = [1·1.414·1,  0·1.414·2,  0·1.414·1,  −1·1.414·0.5]
                  = [1.414, 0, 0, −0.707]
```

All five:

| pos | input | mean of squares | √ | 1/√ | normalized (× 1/√, then × g_attn) |
|---|---|---|---|---|---|
| 0 `<\|user\|>` | [1, 0, 0, −1] | 0.5 | 0.707 | 1.414 | [1.414, 0, 0, −0.707] |
| 1 `sky` | [0, 1, 1, 0] | 0.5 | 0.707 | 1.414 | [0, 2.828, 1.414, 0] |
| 2 `color` | [1, 1, 0, 0] | 0.5 | 0.707 | 1.414 | [1.414, 2.828, 0, 0] |
| 3 `?` | [0, −1, 0, 1] | 0.5 | 0.707 | 1.414 | [0, −2.828, 0, 0.707] |
| 4 `<\|assistant\|>` | [−1, 0, 1, 1] | 0.75 | 0.866 | 1.155 | [−1.155, 0, 1.155, 0.578] |

### 5.3 Queries, keys and values for every position

Use the formulas of 4b. Position 1 (`sky`), `n = [0, 2.828, 1.414, 0]`:

```
q = [0, 2.828, 4.242, 0]
k = [0, 4.242, 1.414, 0]
v = [0, 2.828, 1.414, 0]
```

| pos | q | k | v |
|---|---|---|---|
| 0 | [1.414, 0, 0, −0.707] | [1.414, 0, 0, −0.707] | [1.414, 0, −0.707, −0.707] |
| 1 | [0, 2.828, 4.242, 0] | [0, 4.242, 1.414, 0] | [0, 2.828, 1.414, 0] |
| 2 | [1.414, 2.828, 2.828, 0] | [1.414, 2.828, 0, 0] | [1.414, 2.828, 0, 0] |
| 3 | [0, −2.828, −2.828, 0.707] | [0, −2.828, 0, 0.707] | [0, −2.828, 0.707, 0.707] |
| 4 | [−1.155, 0, 1.155, 0.578] | [−1.155, 1.155, 1.155, 0.578] | [−1.155, 0, 1.733, 0.578] |

### 5.4 q-norm and k-norm (Qwen3)

Qwen3 runs a small RMSNorm on every head's piece of `q` and of `k`, before RoPE. Each head's query and key then
have the same size, whatever came before, so no single key can dominate the attention just by being large. Here a
head has 2 numbers, so the mean is taken over those 2:

```
head piece (a, b) → (a, b) · (1 / √((a² + b²) / 2)) · (g0, g1)
```

The values `v` are not normalized.

Position 4 (`<|assistant|>`):

```
q-norm head 0: (−1.155, 0)  mean of squares (1.334 + 0) / 2 = 0.667, √ = 0.817, 1/√ = 1.224
               → (−1.155·1.224·1,  0·1.224·0.5) = (−1.414, 0)
q-norm head 1: (1.155, 0.578)  mean of squares (1.334 + 0.334) / 2 = 0.834, √ = 0.913, 1/√ = 1.095
               → (1.155·1.095·1,  0.578·1.095·0.5) = (1.265, 0.316)
k-norm head 0: (−1.155, 1.155)  mean of squares (1.334 + 1.334) / 2 = 1.334, √ = 1.155, 1/√ = 0.866
               → (−1.155·0.866·0.5,  1.155·0.866·1) = (−0.5, 1)
k-norm head 1: (1.155, 0.578)  mean of squares (1.334 + 0.334) / 2 = 0.834, √ = 0.913, 1/√ = 1.095
               → (1.155·1.095·0.5,  0.578·1.095·1) = (0.632, 0.633)
```

A head piece that is (0, 0) stays (0, 0): there is nothing to scale (the ε keeps the formula from dividing by zero).

| pos | q after q-norm | k after k-norm |
|---|---|---|
| 0 | [1.414, 0, 0, −0.707] | [0.707, 0, 0, −1.414] |
| 1 | [0, 0.707, 1.413, 0] | [0, 1.413, 0.707, 0] |
| 2 | [0.632, 0.632, 1.414, 0] | [0.316, 1.264, 0, 0] |
| 3 | [0, −0.707, −1.372, 0.171] | [0, −1.414, 0, 1.414] |
| 4 | [−1.414, 0, 1.265, 0.316] | [−0.5, 1, 0.632, 0.633] |

### 5.5 RoPE: tell each query and key where it stands

Attention by itself does not know the order of the words. RoPE (rotary position embedding) turns each head's pair
of numbers `(a, b)` by an angle that grows with the position `p`:

```
(a, b)  →  (a·cos p − b·sin p,   a·sin p + b·cos p)
```

In a real model each pair in a head turns at its own speed, `angle = p · base^(−2i/headDim)`. With a head width of
2 there is only one pair (i = 0), so the angle is just `p` radians. Both heads are turned by the same angle.

| p | cos p | sin p |
|---|---|---|
| 0 | 1 | 0 |
| 1 | 0.54 | 0.841 |
| 2 | −0.416 | 0.909 |
| 3 | −0.99 | 0.141 |
| 4 | −0.654 | −0.757 |
| 5 | 0.284 | −0.959 |

Position 1, the k-normed key [0, 1.413, 0.707, 0] (cos 1 = 0.54, sin 1 = 0.841):

```
RoPE k head 0: (0, 1.413) → (0·0.54 − 1.413·0.841,  0·0.841 + 1.413·0.54) = (−1.188, 0.763)
RoPE k head 1: (0.707, 0) → (0.707·0.54 − 0·0.841,  0.707·0.841 + 0·0.54) = (0.382, 0.595)
```

| pos | q after RoPE | k after RoPE |
|---|---|---|
| 0 | [1.414, 0, 0, −0.707] | [0.707, 0, 0, −1.414] |
| 1 | [−0.595, 0.382, 0.763, 1.188] | [−1.188, 0.763, 0.382, 0.595] |
| 2 | [−0.837, 0.312, −0.588, 1.285] | [−1.28, −0.239, 0, 0] |
| 3 | [0.1, 0.7, 1.334, −0.363] | [0.199, 1.4, −0.199, −1.4] |
| 4 | [0.925, 1.07, −0.588, −1.164] | [1.084, −0.276, 0.066, −0.892] |

The rotated keys and the values of every position are the layer's **KV cache**. They are stored, and reused for
every token generated later.

### 5.6 Attention: which earlier tokens matter?

For each position and each head:
1. **Score.** Take the dot product of the query with every key up to and including its own position (a token
   may not look at the future: this is the "causal mask"). Multiply by `1/√(head width) = 1/√2 = 0.707`.
2. **Softmax.** Turn the scores into weights that are positive and add up to 1:
   ```
   w_j = e^(s_j − max) / Σ e^(s_i − max)
   ```
   Subtracting the largest score first changes nothing in the result, but keeps `eˣ` small.
3. **Mix.** Add up the value vectors (head 0 uses numbers 0 and 1 of each v, head 1 numbers 2 and 3), each
   multiplied by its weight.
4. **Residual.** Add the result to the position's own input `x` (with a real `Wo`, the result would first be
   multiplied by `Wo`; ours is the identity).

Position 4 in full:

**Head 0**, query (0.925, 1.07):

| j | key (head 0) | dot | score = dot·0.707 | score − max | e^(…) | weight |
|---|---|---|---|---|---|---|
| 0 | (0.707, 0) | 0.654 | 0.462 | −0.727 | 0.483 | 0.206 |
| 1 | (−1.188, 0.763) | −0.282 | −0.199 | −1.388 | 0.25 | 0.107 |
| 2 | (−1.28, −0.239) | −1.44 | −1.018 | −2.207 | 0.11 | 0.047 |
| 3 | (0.199, 1.4) | 1.682 | 1.189 | 0 | 1 | 0.426 |
| 4 | (1.084, −0.276) | 0.707 | 0.5 | −0.689 | 0.502 | 0.214 |
| | | | max = 1.189 | | sum = 2.345 | |

```
out_0 = 0.206·1.414 + 0.107·0 + 0.047·1.414 + 0.426·0 + 0.214·(−1.155) = 0.111
out_1 = 0.206·0 + 0.107·2.828 + 0.047·2.828 + 0.426·(−2.828) + 0.214·0 = −0.769
```

**Head 1**, query (−0.588, −1.164):

| j | key (head 1) | dot | score = dot·0.707 | score − max | e^(…) | weight |
|---|---|---|---|---|---|---|
| 0 | (0, −1.414) | 1.646 | 1.164 | −0.071 | 0.931 | 0.314 |
| 1 | (0.382, 0.595) | −0.917 | −0.648 | −1.883 | 0.152 | 0.051 |
| 2 | (0, 0) | 0 | 0 | −1.235 | 0.291 | 0.098 |
| 3 | (−0.199, −1.4) | 1.747 | 1.235 | 0 | 1 | 0.337 |
| 4 | (0.066, −0.892) | 0.999 | 0.706 | −0.529 | 0.589 | 0.199 |
| | | | max = 1.235 | | sum = 2.963 | |

```
out_2 = 0.314·(−0.707) + 0.051·1.414 + 0.098·0 + 0.337·0.707 + 0.199·1.733 = 0.433
out_3 = 0.314·(−0.707) + 0.051·0 + 0.098·0 + 0.337·0.707 + 0.199·0.578 = 0.131
```

```
attention output = [0.111, −0.769, 0.433, 0.131]
h = x + attention = [−0.889, −0.769, 1.433, 1.131]
```

Positions 0 to 3 the same way. Position 0 can only look at itself, so both its weights are 1:

**Position 0** (`<\|user\|>`), queries: head 0 (1.414, 0), head 1 (0, −0.707)

| j | head 0: dot | score | e^(score−max) | weight | head 1: dot | score | e^(score−max) | weight |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 0.707 | 1 | 1 | 1 | 0.707 | 1 | 1 |
| | | max 0.707 | sum 1 | | | max 0.707 | sum 1 | |

Output: head 0 (1.414, 0), head 1 (−0.707, −0.707), so attention = [1.414, 0, −0.707, −0.707] and h = x + attention = [2.414, 0, −0.707, −1.707].

**Position 1** (`sky`), queries: head 0 (−0.595, 0.382), head 1 (0.763, 1.188)

| j | head 0: dot | score | e^(score−max) | weight | head 1: dot | score | e^(score−max) | weight |
|---|---|---|---|---|---|---|---|---|
| 0 | −0.421 | −0.298 | 0.366 | 0.268 | −1.68 | −1.188 | 0.15 | 0.13 |
| 1 | 0.998 | 0.706 | 1 | 0.732 | 0.998 | 0.706 | 1 | 0.87 |
| | | max 0.706 | sum 1.366 | | | max 0.706 | sum 1.15 | |

Output: head 0 (0.379, 2.07), head 1 (1.138, −0.092), so attention = [0.379, 2.07, 1.138, −0.092] and h = x + attention = [0.379, 3.07, 2.138, −0.092].

**Position 2** (`color`), queries: head 0 (−0.837, 0.312), head 1 (−0.588, 1.285)

| j | head 0: dot | score | e^(score−max) | weight | head 1: dot | score | e^(score−max) | weight |
|---|---|---|---|---|---|---|---|---|
| 0 | −0.592 | −0.419 | 0.275 | 0.13 | −1.817 | −1.285 | 0.189 | 0.101 |
| 1 | 1.232 | 0.871 | 1 | 0.471 | 0.54 | 0.382 | 1 | 0.534 |
| 2 | 0.997 | 0.705 | 0.847 | 0.399 | 0 | 0 | 0.682 | 0.365 |
| | | max 0.871 | sum 2.122 | | | max 0.382 | sum 1.871 | |

Output: head 0 (0.748, 2.46), head 1 (0.684, −0.071), so attention = [0.748, 2.46, 0.684, −0.071] and h = x + attention = [1.748, 3.46, 0.684, −0.071].

**Position 3** (`?`), queries: head 0 (0.1, 0.7), head 1 (1.334, −0.363)

| j | head 0: dot | score | e^(score−max) | weight | head 1: dot | score | e^(score−max) | weight |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.071 | 0.05 | 0.518 | 0.201 | 0.513 | 0.363 | 1 | 0.296 |
| 1 | 0.415 | 0.293 | 0.661 | 0.256 | 0.294 | 0.208 | 0.856 | 0.253 |
| 2 | −0.295 | −0.209 | 0.4 | 0.155 | 0 | 0 | 0.696 | 0.206 |
| 3 | 1 | 0.707 | 1 | 0.388 | 0.243 | 0.172 | 0.826 | 0.245 |
| | | max 0.707 | sum 2.579 | | | max 0.363 | sum 3.378 | |

Output: head 0 (0.503, 0.065), head 1 (0.322, −0.036), so attention = [0.503, 0.065, 0.322, −0.036] and h = x + attention = [0.503, −0.935, 0.322, 0.964].

### 5.7 Feed-forward (SwiGLU) for every position

Position 4 in full:

Normalize `h` with `g_ffn` (call the result `m`):

```
h            = [−0.889, −0.769, 1.433, 1.131]
squares      = [0.79, 0.591, 2.053, 1.279]
mean         = (0.79 + 0.591 + 2.053 + 1.279) / 4 = 1.179
√ of mean    = √1.179 = 1.086
1 / √        = 1 / 1.086 = 0.921
m = h · 0.921 · g = [−0.889·0.921·1,  −0.769·0.921·1,  1.433·0.921·0.5,  1.131·0.921·0.5]
                 = [−0.819, −0.708, 0.66, 0.521]
```

Gate and up from `m` (the formulas of 4b / 4c), then SiLU, `silu(z) = z / (1 + e^(−z))`:

```
gate = [−0.159, −1.229]
up   = [−0.187, −0.159]
silu(−0.159) = −0.159 / (1 + e^(0.159)) = −0.159 / (1 + 1.1723) = −0.073
silu(−1.229) = −1.229 / (1 + e^(1.229)) = −1.229 / (1 + 3.4178) = −0.278
a    = [−0.073·(−0.187),  −0.278·(−0.159)] = [0.014, 0.044]
down = [0.014, 0.044, −0.044, 0.014]
layer output = h + down = [−0.875, −0.725, 1.389, 1.145]
```

All five (the last column is what layer 2 receives):

| pos | h | mean of squares | 1/√ | m (× g_ffn) | gate | up | silu(gate) | a | down | layer output |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 `<\|user\|>` | [2.414, 0, −0.707, −1.707] | 2.31 | 0.658 | [1.588, 0, −0.233, −0.562] | (1.355, 0.562) | (−0.562, 1.355) | (1.077, 0.358) | (−0.605, 0.485) | [−0.605, 0.485, −0.485, −0.605] | [1.809, 0.485, −1.192, −2.312] |
| 1 `sky` | [0.379, 3.07, 2.138, −0.092] | 3.537 | 0.532 | [0.202, 1.633, 0.569, −0.024] | (0.771, 1.657) | (1.609, 0.771) | (0.527, 1.392) | (0.848, 1.073) | [0.848, 1.073, −1.073, 0.848] | [1.227, 4.143, 1.065, 0.756] |
| 2 `color` | [1.748, 3.46, 0.684, −0.071] | 3.875 | 0.508 | [0.888, 1.758, 0.174, −0.018] | (1.062, 1.776) | (1.74, 1.062) | (0.789, 1.519) | (1.373, 1.613) | [1.373, 1.613, −1.613, 1.373] | [3.121, 5.073, −0.929, 1.302] |
| 3 `?` | [0.503, −0.935, 0.322, 0.964] | 0.54 | 1.361 | [0.685, −1.273, 0.219, 0.656] | (0.904, −1.929) | (−0.617, 0.904) | (0.643, −0.245) | (−0.397, −0.221) | [−0.397, −0.221, 0.221, −0.397] | [0.106, −1.156, 0.543, 0.567] |
| 4 `<\|assistant\|>` | [−0.889, −0.769, 1.433, 1.131] | 1.179 | 0.921 | [−0.819, −0.708, 0.66, 0.521] | (−0.159, −1.229) | (−0.187, −0.159) | (−0.073, −0.278) | (0.014, 0.044) | [0.014, 0.044, −0.044, 0.014] | [−0.875, −0.725, 1.389, 1.145] |

### Layer 2

### 5.8 Layer 2: the same steps on layer 1's outputs

Layer 2 has its own weights (4c). Its input `x` is now layer 1's output, not the embedding. As said at the start
of section 5, positions 0 to 3 only need their keys and values here.

RMSNorm with layer 2's `g_attn = [0.5, 1, 1, 2]`:

| pos | input | mean of squares | √ | 1/√ | normalized (× 1/√, then × g_attn) |
|---|---|---|---|---|---|
| 0 `<\|user\|>` | [1.809, 0.485, −1.192, −2.312] | 2.568 | 1.602 | 0.624 | [0.564, 0.303, −0.744, −2.885] |
| 1 `sky` | [1.227, 4.143, 1.065, 0.756] | 5.094 | 2.257 | 0.443 | [0.272, 1.835, 0.472, 0.67] |
| 2 `color` | [3.121, 5.073, −0.929, 1.302] | 9.509 | 3.084 | 0.324 | [0.506, 1.644, −0.301, 0.844] |
| 3 `?` | [0.106, −1.156, 0.543, 0.567] | 0.491 | 0.701 | 1.427 | [0.076, −1.65, 0.775, 1.618] |
| 4 `<\|assistant\|>` | [−0.875, −0.725, 1.389, 1.145] | 1.133 | 1.064 | 0.94 | [−0.411, −0.681, 1.306, 2.153] |

Position 4 in full:

```
x            = [−0.875, −0.725, 1.389, 1.145]
squares      = [0.766, 0.526, 1.929, 1.311]
mean         = (0.766 + 0.526 + 1.929 + 1.311) / 4 = 1.133
√ of mean    = √1.133 = 1.064
1 / √        = 1 / 1.064 = 0.94
n4 = x · 0.94 · g = [−0.875·0.94·0.5,  −0.725·0.94·1,  1.389·0.94·1,  1.145·0.94·2]
                 = [−0.411, −0.681, 1.306, 2.153]
```

Keys and values (the formulas of 4c), k-norm and RoPE:

| pos | k | v | k after k-norm | k after RoPE |
|---|---|---|---|---|
| 0 | [0.564, 0.303, −0.744, −2.885] | [0.564, 0.303, −0.744, −2.321] | [1.245, 1.338, −0.353, −2.741] | [1.245, 1.338, −0.353, −2.741] |
| 1 | [0.272, 1.835, 0.472, 0.67] | [0.272, 1.835, 0.472, 0.942] | [0.207, 2.797, 0.814, 2.31] | [−2.24, 1.684, −1.503, 1.932] |
| 2 | [0.506, 1.644, −0.301, 0.844] | [0.506, 1.644, −0.301, 1.35] | [0.416, 2.703, −0.476, 2.667] | [−2.63, −0.746, −2.226, −1.542] |
| 3 | [0.076, −1.65, 0.775, 1.618] | [0.076, −1.65, 0.775, 1.694] | [0.065, −2.825, 0.611, 2.553] | [0.334, 2.806, −0.965, −2.441] |
| 4 | [−0.411, −0.681, 1.306, 2.153] | [−0.411, −0.681, 1.306, 1.742] | [−0.731, −2.423, 0.733, 2.416] | [−1.356, 2.138, 1.35, −2.135] |

The query is only needed at position 4: `q = [−0.681, −0.411, 2.153, 1.306]`.

```
q-norm head 0: (−0.681, −0.411)  mean of squares (0.464 + 0.169) / 2 = 0.316, √ = 0.562, 1/√ = 1.779
               → (−0.681·1.779·1,  −0.411·1.779·1) = (−1.211, −0.731)
q-norm head 1: (2.153, 1.306)  mean of squares (4.635 + 1.706) / 2 = 3.171, √ = 1.781, 1/√ = 0.561
               → (2.153·0.561·1,  1.306·0.561·1) = (1.208, 0.733)
RoPE q head 0: (−1.211, −0.731) → (−1.211·(−0.654) − (−0.731)·(−0.757),  −1.211·(−0.757) + (−0.731)·(−0.654)) = (0.239, 1.395)
RoPE q head 1: (1.208, 0.733) → (1.208·(−0.654) − 0.733·(−0.757),  1.208·(−0.757) + 0.733·(−0.654)) = (−0.235, −1.394)
```

### 5.9 Layer 2 attention and feed-forward at position 4

**Head 0**, query (0.239, 1.395):

| j | key (head 0) | dot | score = dot·0.707 | score − max | e^(…) | weight |
|---|---|---|---|---|---|---|
| 0 | (1.245, 1.338) | 2.164 | 1.53 | −1.294 | 0.274 | 0.145 |
| 1 | (−2.24, 1.684) | 1.814 | 1.282 | −1.542 | 0.214 | 0.113 |
| 2 | (−2.63, −0.746) | −1.669 | −1.18 | −4.004 | 0.018 | 0.009 |
| 3 | (0.334, 2.806) | 3.994 | 2.824 | 0 | 1 | 0.528 |
| 4 | (−1.356, 2.138) | 2.658 | 1.879 | −0.945 | 0.389 | 0.205 |
| | | | max = 2.824 | | sum = 1.895 | |

```
out_0 = 0.145·0.564 + 0.113·0.272 + 0.009·0.506 + 0.528·0.076 + 0.205·(−0.411) = 0.073
out_1 = 0.145·0.303 + 0.113·1.835 + 0.009·1.644 + 0.528·(−1.65) + 0.205·(−0.681) = −0.745
```

**Head 1**, query (−0.235, −1.394):

| j | key (head 1) | dot | score = dot·0.707 | score − max | e^(…) | weight |
|---|---|---|---|---|---|---|
| 0 | (−0.353, −2.741) | 3.904 | 2.76 | 0 | 1 | 0.375 |
| 1 | (−1.503, 1.932) | −2.34 | −1.654 | −4.414 | 0.012 | 0.004 |
| 2 | (−2.226, −1.542) | 2.673 | 1.89 | −0.87 | 0.419 | 0.157 |
| 3 | (−0.965, −2.441) | 3.63 | 2.566 | −0.194 | 0.824 | 0.309 |
| 4 | (1.35, −2.135) | 2.659 | 1.88 | −0.88 | 0.415 | 0.155 |
| | | | max = 2.76 | | sum = 2.67 | |

```
out_2 = 0.375·(−0.744) + 0.004·0.472 + 0.157·(−0.301) + 0.309·0.775 + 0.155·1.306 = 0.118
out_3 = 0.375·(−2.321) + 0.004·0.942 + 0.157·1.35 + 0.309·1.694 + 0.155·1.742 = 0.139
```

```
attention output = [0.073, −0.745, 0.118, 0.139]
h = x + attention = [−0.802, −1.47, 1.507, 1.284]
```

Normalize `h` with `g_ffn` (call the result `m`):

```
h            = [−0.802, −1.47, 1.507, 1.284]
squares      = [0.643, 2.161, 2.271, 1.649]
mean         = (0.643 + 2.161 + 2.271 + 1.649) / 4 = 1.681
√ of mean    = √1.681 = 1.297
1 / √        = 1 / 1.297 = 0.771
m = h · 0.771 · g = [−0.802·0.771·1,  −1.47·0.771·0.5,  1.507·0.771·1,  1.284·0.771·1]
                 = [−0.618, −0.567, 1.162, 0.99]
```

Gate and up from `m` (the formulas of 4b / 4c), then SiLU, `silu(z) = z / (1 + e^(−z))`:

```
gate = [0.423, 0.544]
up   = [−1.78, 0.423]
silu(0.423) = 0.423 / (1 + e^(−0.423)) = 0.423 / (1 + 0.6551) = 0.256
silu(0.544) = 0.544 / (1 + e^(−0.544)) = 0.544 / (1 + 0.5804) = 0.344
a    = [0.256·(−1.78),  0.344·0.423] = [−0.456, 0.146]
down = [0.146, −0.456, 0.146, −0.456]
layer output = h + down = [−0.656, −1.926, 1.653, 0.828]
```

### 5.10 Final norm and the logits

Normalize the last layer's output `y` with `g_final = [2, 1, 1, 0.5]`:

```
y            = [−0.656, −1.926, 1.653, 0.828]
squares      = [0.43, 3.709, 2.732, 0.686]
mean         = (0.43 + 3.709 + 2.732 + 0.686) / 4 = 1.889
√ of mean    = √1.889 = 1.374
1 / √        = 1 / 1.374 = 0.728
f = y · 0.728 · g = [−0.656·0.728·2,  −1.926·0.728·1,  1.653·0.728·1,  0.828·0.728·0.5]
                 = [−0.955, −1.402, 1.203, 0.301]
```

Each token's **logit** is the dot product of its `Wout` row with `f`. For example:

```
blue      [−1, −1, 1, 0]   → (−1)·(−0.955) + (−1)·(−1.402) + 1·1.203 = 3.56
<|end|>   [1, 1, 0, 0]     → 1·(−0.955) + 1·(−1.402) = −2.357
red       [−1, 0, 1, 1]    → (−1)·(−0.955) + 1·1.203 + 1·0.301 = 2.459
```

### 5.11 Softmax: logits to probabilities

Same softmax as in attention, over all 20 tokens:

| id | token | logit | logit − max | e^(…) | probability |
|---|---|---|---|---|---|
| 0 | `<\|user\|>` | 0.301 | −3.259 | 0.038 | 0.015 |
| 1 | `<\|assistant\|>` | 1.703 | −1.857 | 0.156 | 0.06 |
| 2 | `<\|end\|>` | −2.357 | −5.917 | 0.003 | 0.001 |
| 3 | `?` | 1.402 | −2.158 | 0.116 | 0.044 |
| 4 | `.` | −0.654 | −4.214 | 0.015 | 0.006 |
| 5 | `sky` | 1.504 | −2.056 | 0.128 | 0.049 |
| 6 | `grass` | −2.158 | −5.718 | 0.003 | 0.001 |
| 7 | `is` | 1.203 | −2.357 | 0.095 | 0.036 |
| **8** | **`blue`** | **3.56** | **0** | **1** | **0.382** |
| 9 | `green` | −2.605 | −6.165 | 0.002 | 0.001 |
| 10 | `the` | −0.955 | −4.515 | 0.011 | 0.004 |
| 11 | `color` | 2.158 | −1.402 | 0.246 | 0.094 |
| 12 | `c` | −1.402 | −4.962 | 0.007 | 0.003 |
| 13 | `o` | −0.301 | −3.861 | 0.021 | 0.008 |
| 14 | `l` | 0.955 | −2.605 | 0.074 | 0.028 |
| 15 | `r` | −1.203 | −4.763 | 0.009 | 0.003 |
| 16 | `co` | 0.248 | −3.312 | 0.036 | 0.014 |
| 17 | `col` | −0.199 | −3.759 | 0.023 | 0.009 |
| 18 | `or` | 2.357 | −1.203 | 0.3 | 0.115 |
| 19 | `red` | 2.459 | −1.101 | 0.333 | 0.127 |
| | | | | sum = 2.616 | |

### 5.12 Choosing the token

**Greedy** (temperature 0) takes the largest probability: **`blue`** (0.382).

**Sampling** draws at random, following the probabilities. A typical setting is temperature 0.7 with top-k = 3:
1. Keep the 3 largest logits: `blue` 3.56, `red` 2.459, `or` 2.357.
2. Divide by the temperature: 3.56 / 0.7 = 5.086, 2.459 / 0.7 = 3.513, 2.357 / 0.7 = 3.367. A temperature
   below 1 stretches the gaps, so the favourite gets more likely.
3. Softmax of these three (max 5.086): e^(0) = 1, e^(−1.573) = 0.207, e^(−1.719) = 0.179,
   sum 1.386, which gives probabilities **0.722, 0.149, 0.129**.
4. Draw a random number `u` between 0 and 1, and walk the running total 0.722, 0.871, 1.0:
   - `u = 0.42` falls in the first slot, so `blue`;
   - `u = 0.772` falls in the second slot, so `red`.

(Top-p keeps the smallest set of best tokens whose probabilities add up to at least `p`, instead of a fixed number
of them. See [09](09-sampling.md).)

We use greedy. **Generated token 1: `blue` (id 8).**

---

## 6. Generating token 2: only the new token is computed

The sequence is now `<|user|> sky color ? <|assistant|> blue`, and `blue` sits at position 5. Without a cache we
would redo everything from section 5 for six tokens. With the KV cache, the keys and values of positions 0 to 4 are
already stored, one set per layer (5.5 for layer 1, 5.8 for layer 2), so we run **only `blue`** through the two
layers.

### 6.1 Layer 1

```
x            = [0, 1, 0, 1]
squares      = [0, 1, 0, 1]
mean         = (0 + 1 + 0 + 1) / 4 = 0.5
√ of mean    = √0.5 = 0.707
1 / √        = 1 / 0.707 = 1.414
n = x · 1.414 · g = [0·1.414·1,  1·1.414·2,  0·1.414·1,  1·1.414·0.5]
                 = [0, 2.828, 0, 0.707]
```

```
q = [0, 2.828, 2.828, 0.707]
k = [0, 2.828, 0, 0.707]
v = [0, 2.828, 0.707, 0.707]
q-norm head 0: (0, 2.828)  mean of squares (0 + 7.998) / 2 = 3.999, √ = 2, 1/√ = 0.5
               → (0·0.5·1,  2.828·0.5·0.5) = (0, 0.707)
q-norm head 1: (2.828, 0.707)  mean of squares (7.998 + 0.5) / 2 = 4.249, √ = 2.061, 1/√ = 0.485
               → (2.828·0.485·1,  0.707·0.485·0.5) = (1.372, 0.171)
k-norm head 0: (0, 2.828)  mean of squares (0 + 7.998) / 2 = 3.999, √ = 2, 1/√ = 0.5
               → (0·0.5·0.5,  2.828·0.5·1) = (0, 1.414)
k-norm head 1: (0, 0.707)  mean of squares (0 + 0.5) / 2 = 0.25, √ = 0.5, 1/√ = 2
               → (0·2·0.5,  0.707·2·1) = (0, 1.414)
RoPE q head 0: (0, 0.707) → (0·0.284 − 0.707·(−0.959),  0·(−0.959) + 0.707·0.284) = (0.678, 0.201)
RoPE q head 1: (1.372, 0.171) → (1.372·0.284 − 0.171·(−0.959),  1.372·(−0.959) + 0.171·0.284) = (0.554, −1.267)
RoPE k head 0: (0, 1.414) → (0·0.284 − 1.414·(−0.959),  0·(−0.959) + 1.414·0.284) = (1.356, 0.402)
RoPE k head 1: (0, 1.414) → (0·0.284 − 1.414·(−0.959),  0·(−0.959) + 1.414·0.284) = (1.356, 0.402)
```

Append the new key and value to layer 1's cache (now 6 positions) and attend over all of them:

**Head 0**, query (0.678, 0.201):

| j | key (head 0) | dot | score = dot·0.707 | score − max | e^(…) | weight |
|---|---|---|---|---|---|---|
| 0 | (0.707, 0) | 0.479 | 0.339 | −0.368 | 0.692 | 0.186 |
| 1 | (−1.188, 0.763) | −0.652 | −0.461 | −1.168 | 0.311 | 0.084 |
| 2 | (−1.28, −0.239) | −0.916 | −0.648 | −1.355 | 0.258 | 0.069 |
| 3 | (0.199, 1.4) | 0.416 | 0.294 | −0.413 | 0.662 | 0.178 |
| 4 | (1.084, −0.276) | 0.679 | 0.48 | −0.227 | 0.797 | 0.214 |
| 5 | (1.356, 0.402) | 1 | 0.707 | 0 | 1 | 0.269 |
| | | | max = 0.707 | | sum = 3.72 | |

```
out_0 = 0.186·1.414 + 0.084·0 + 0.069·1.414 + 0.178·0 + 0.214·(−1.155) + 0.269·0 = 0.113
out_1 = 0.186·0 + 0.084·2.828 + 0.069·2.828 + 0.178·(−2.828) + 0.214·0 + 0.269·2.828 = 0.69
```

**Head 1**, query (0.554, −1.267):

| j | key (head 1) | dot | score = dot·0.707 | score − max | e^(…) | weight |
|---|---|---|---|---|---|---|
| 0 | (0, −1.414) | 1.792 | 1.267 | 0 | 1 | 0.297 |
| 1 | (0.382, 0.595) | −0.542 | −0.383 | −1.65 | 0.192 | 0.057 |
| 2 | (0, 0) | 0 | 0 | −1.267 | 0.282 | 0.084 |
| 3 | (−0.199, −1.4) | 1.664 | 1.176 | −0.091 | 0.913 | 0.271 |
| 4 | (0.066, −0.892) | 1.167 | 0.825 | −0.442 | 0.643 | 0.191 |
| 5 | (1.356, 0.402) | 0.242 | 0.171 | −1.096 | 0.334 | 0.099 |
| | | | max = 1.267 | | sum = 3.364 | |

```
out_2 = 0.297·(−0.707) + 0.057·1.414 + 0.084·0 + 0.271·0.707 + 0.191·1.733 + 0.099·0.707 = 0.463
out_3 = 0.297·(−0.707) + 0.057·0 + 0.084·0 + 0.271·0.707 + 0.191·0.578 + 0.099·0.707 = 0.162
```

```
attention output = [0.113, 0.69, 0.463, 0.162]
h = x + attention = [0.113, 1.69, 0.463, 1.162]
```

Normalize `h` with `g_ffn` (call the result `m`):

```
h            = [0.113, 1.69, 0.463, 1.162]
squares      = [0.013, 2.856, 0.214, 1.35]
mean         = (0.013 + 2.856 + 0.214 + 1.35) / 4 = 1.108
√ of mean    = √1.108 = 1.053
1 / √        = 1 / 1.053 = 0.95
m = h · 0.95 · g = [0.113·0.95·1,  1.69·0.95·1,  0.463·0.95·0.5,  1.162·0.95·0.5]
                = [0.107, 1.605, 0.22, 0.552]
```

Gate and up from `m` (the formulas of 4b / 4c), then SiLU, `silu(z) = z / (1 + e^(−z))`:

```
gate = [0.327, 1.053]
up   = [2.157, 0.327]
silu(0.327) = 0.327 / (1 + e^(−0.327)) = 0.327 / (1 + 0.7211) = 0.19
silu(1.053) = 1.053 / (1 + e^(−1.053)) = 1.053 / (1 + 0.3489) = 0.781
a    = [0.19·2.157,  0.781·0.327] = [0.41, 0.255]
down = [0.41, 0.255, −0.255, 0.41]
layer output = h + down = [0.523, 1.945, 0.208, 1.572]
```

### 6.2 Layer 2

The input is layer 1's output for `blue`, `x = [0.523, 1.945, 0.208, 1.572]`.

```
x            = [0.523, 1.945, 0.208, 1.572]
squares      = [0.274, 3.783, 0.043, 2.471]
mean         = (0.274 + 3.783 + 0.043 + 2.471) / 4 = 1.643
√ of mean    = √1.643 = 1.282
1 / √        = 1 / 1.282 = 0.78
n = x · 0.78 · g = [0.523·0.78·0.5,  1.945·0.78·1,  0.208·0.78·1,  1.572·0.78·2]
                = [0.204, 1.517, 0.162, 2.452]
```

```
q = [1.517, 0.204, 2.452, 0.162]
k = [0.204, 1.517, 0.162, 2.452]
v = [0.204, 1.517, 0.162, 2.656]
q-norm head 0: (1.517, 0.204)  mean of squares (2.301 + 0.042) / 2 = 1.171, √ = 1.082, 1/√ = 0.924
               → (1.517·0.924·1,  0.204·0.924·1) = (1.402, 0.188)
q-norm head 1: (2.452, 0.162)  mean of squares (6.012 + 0.026) / 2 = 3.019, √ = 1.738, 1/√ = 0.575
               → (2.452·0.575·1,  0.162·0.575·1) = (1.41, 0.093)
k-norm head 0: (0.204, 1.517)  mean of squares (0.042 + 2.301) / 2 = 1.171, √ = 1.082, 1/√ = 0.924
               → (0.204·0.924·1,  1.517·0.924·2) = (0.188, 2.803)
k-norm head 1: (0.162, 2.452)  mean of squares (0.026 + 6.012) / 2 = 3.019, √ = 1.738, 1/√ = 0.575
               → (0.162·0.575·1,  2.452·0.575·2) = (0.093, 2.82)
RoPE q head 0: (1.402, 0.188) → (1.402·0.284 − 0.188·(−0.959),  1.402·(−0.959) + 0.188·0.284) = (0.578, −1.291)
RoPE q head 1: (1.41, 0.093) → (1.41·0.284 − 0.093·(−0.959),  1.41·(−0.959) + 0.093·0.284) = (0.49, −1.326)
RoPE k head 0: (0.188, 2.803) → (0.188·0.284 − 2.803·(−0.959),  0.188·(−0.959) + 2.803·0.284) = (2.741, 0.616)
RoPE k head 1: (0.093, 2.82) → (0.093·0.284 − 2.82·(−0.959),  0.093·(−0.959) + 2.82·0.284) = (2.731, 0.712)
```

**Head 0**, query (0.578, −1.291):

| j | key (head 0) | dot | score = dot·0.707 | score − max | e^(…) | weight |
|---|---|---|---|---|---|---|
| 0 | (1.245, 1.338) | −1.008 | −0.713 | −1.271 | 0.281 | 0.155 |
| 1 | (−2.24, 1.684) | −3.469 | −2.453 | −3.011 | 0.049 | 0.027 |
| 2 | (−2.63, −0.746) | −0.557 | −0.394 | −0.952 | 0.386 | 0.213 |
| 3 | (0.334, 2.806) | −3.429 | −2.424 | −2.982 | 0.051 | 0.028 |
| 4 | (−1.356, 2.138) | −3.544 | −2.506 | −3.064 | 0.047 | 0.026 |
| 5 | (2.741, 0.616) | 0.789 | 0.558 | 0 | 1 | 0.551 |
| | | | max = 0.558 | | sum = 1.814 | |

```
out_0 = 0.155·0.564 + 0.027·0.272 + 0.213·0.506 + 0.028·0.076 + 0.026·(−0.411) + 0.551·0.204 = 0.306
out_1 = 0.155·0.303 + 0.027·1.835 + 0.213·1.644 + 0.028·(−1.65) + 0.026·(−0.681) + 0.551·1.517 = 1.219
```

**Head 1**, query (0.49, −1.326):

| j | key (head 1) | dot | score = dot·0.707 | score − max | e^(…) | weight |
|---|---|---|---|---|---|---|
| 0 | (−0.353, −2.741) | 3.462 | 2.448 | −0.022 | 0.978 | 0.342 |
| 1 | (−1.503, 1.932) | −3.298 | −2.332 | −4.802 | 0.008 | 0.003 |
| 2 | (−2.226, −1.542) | 0.954 | 0.674 | −1.796 | 0.166 | 0.058 |
| 3 | (−0.965, −2.441) | 2.764 | 1.954 | −0.516 | 0.597 | 0.209 |
| 4 | (1.35, −2.135) | 3.493 | 2.47 | 0 | 1 | 0.35 |
| 5 | (2.731, 0.712) | 0.394 | 0.279 | −2.191 | 0.112 | 0.039 |
| | | | max = 2.47 | | sum = 2.861 | |

```
out_2 = 0.342·(−0.744) + 0.003·0.472 + 0.058·(−0.301) + 0.209·0.775 + 0.35·1.306 + 0.039·0.162 = 0.355
out_3 = 0.342·(−2.321) + 0.003·0.942 + 0.058·1.35 + 0.209·1.694 + 0.35·1.742 + 0.039·2.656 = 0.355
```

```
attention output = [0.306, 1.219, 0.355, 0.355]
h = x + attention = [0.829, 3.164, 0.563, 1.927]
```

Normalize `h` with `g_ffn` (call the result `m`):

```
h            = [0.829, 3.164, 0.563, 1.927]
squares      = [0.687, 10.011, 0.317, 3.713]
mean         = (0.687 + 10.011 + 0.317 + 3.713) / 4 = 3.682
√ of mean    = √3.682 = 1.919
1 / √        = 1 / 1.919 = 0.521
m = h · 0.521 · g = [0.829·0.521·1,  3.164·0.521·0.5,  0.563·0.521·1,  1.927·0.521·1]
                 = [0.432, 0.824, 0.293, 1.004]
```

Gate and up from `m` (the formulas of 4b / 4c), then SiLU, `silu(z) = z / (1 + e^(−z))`:

```
gate = [1.828, 0.725]
up   = [0.139, 1.828]
silu(1.828) = 1.828 / (1 + e^(−1.828)) = 1.828 / (1 + 0.1607) = 1.575
silu(0.725) = 0.725 / (1 + e^(−0.725)) = 0.725 / (1 + 0.4843) = 0.488
a    = [1.575·0.139,  0.488·1.828] = [0.219, 0.892]
down = [0.892, 0.219, 0.892, 0.219]
layer output = h + down = [1.721, 3.383, 1.455, 2.146]
```

### 6.3 Final norm, logits, choice

```
y            = [1.721, 3.383, 1.455, 2.146]
squares      = [2.962, 11.445, 2.117, 4.605]
mean         = (2.962 + 11.445 + 2.117 + 4.605) / 4 = 5.282
√ of mean    = √5.282 = 2.298
1 / √        = 1 / 2.298 = 0.435
f = y · 0.435 · g = [1.721·0.435·2,  3.383·0.435·1,  1.455·0.435·1,  2.146·0.435·0.5]
                 = [1.497, 1.472, 0.633, 0.467]
```

```
<|end|>   [1, 1, 0, 0]     → 1·1.497 + 1·1.472 = 2.969
blue      [−1, −1, 1, 0]   → (−1)·1.497 + (−1)·1.472 + 1·0.633 = −2.336
```

| id | token | logit | logit − max | e^(…) | probability |
|---|---|---|---|---|---|
| 0 | `<\|user\|>` | 0.467 | −2.502 | 0.082 | 0.024 |
| 1 | `<\|assistant\|>` | −1.005 | −3.974 | 0.019 | 0.006 |
| **2** | **`<\|end\|>`** | **2.969** | **0** | **1** | **0.293** |
| 3 | `?` | −1.472 | −4.441 | 0.012 | 0.004 |
| 4 | `.` | 1.964 | −1.005 | 0.366 | 0.107 |
| 5 | `sky` | 1.1 | −1.869 | 0.154 | 0.045 |
| 6 | `grass` | 0.864 | −2.105 | 0.122 | 0.036 |
| 7 | `is` | 0.633 | −2.336 | 0.097 | 0.028 |
| 8 | `blue` | −2.336 | −5.305 | 0.005 | 0.001 |
| 9 | `green` | 0.839 | −2.13 | 0.119 | 0.035 |
| 10 | `the` | 1.497 | −1.472 | 0.229 | 0.067 |
| 11 | `color` | −0.864 | −3.833 | 0.022 | 0.006 |
| 12 | `c` | 1.472 | −1.497 | 0.224 | 0.066 |
| 13 | `o` | −0.467 | −3.436 | 0.032 | 0.009 |
| 14 | `l` | −1.497 | −4.466 | 0.011 | 0.003 |
| 15 | `r` | −0.633 | −3.602 | 0.027 | 0.008 |
| 16 | `co` | 2.13 | −0.839 | 0.432 | 0.127 |
| 17 | `col` | 2.105 | −0.864 | 0.421 | 0.123 |
| 18 | `or` | −2.969 | −5.938 | 0.003 | 0.001 |
| 19 | `red` | −0.397 | −3.366 | 0.035 | 0.01 |
| | | | | sum = 3.412 | |

Greedy picks **`<|end|>`** (0.293). **Generated token 2: `<|end|>` (id 2).**

---

## 7. Stopping and turning ids back into text

`<|end|>` is a **stop token**: the engine stops generating and does not print it. The answer is the ids generated
before it, `[8]`. Detokenizing looks each id up in the table of section 1 (and re-attaches spaces; real tokenizers
store the space inside the token, e.g. `Ġblue`):

```
[8] → "blue"
```

**The model answered: `blue`.**

---

## 8. What the cache saved, and how this scales

**Work per generated token.** In section 6 we computed, in each layer, one norm, one q, one k and one v (for
`blue`), plus six dot products per head. Without the cache we would also have redone, in both layers, everything
of the five prompt tokens. For the 100th token of an answer the saving is 99 positions' worth of work per layer.

**What the layers add.** Every layer repeats the same recipe with its own weights, and its output is the next
layer's input. That is why, during prefill, every layer but the last needs the full output of every position, and
why every layer keeps its own KV cache.

**What grows with the context.** The attention of one new token still reads every cached key and value of every
layer (6 rows per layer here). That is why long contexts get slower, and why the cache is worth storing compactly
(f16, q8: [07](07-cpu-backend.md), [08](08-metal-backend.md)).

**What the real model multiplies.** Qwen3-0.6B does each step above with 1 024 numbers instead of 4, 16 heads of
128 instead of 2 of 2, an FFN hidden width of 3 072 instead of 2, 151 936 logits instead of 20, and 28 layers instead
of 2. One token then costs about 1.2 billion multiply-adds, roughly 2 per weight ([11](11-performance.md)), and the
recipe stays the same.

---


## Appendix A. How a weight is stored: Q4_0 on paper

jllm usually keeps weights in **Q4_0**: blocks of 32 numbers, each stored as a 4-bit integer 0…15, plus one scale per
block. Here is the same thing with a block of 8 numbers.

```
weights w = [0.12, −0.50, 0.33, 0.07, −0.25, 0.44, −0.06, 0.20]
```

1. Find the number with the largest size, keeping its sign: −0.50.
2. Scale `d = max / −8 = −0.50 / −8 = 0.0625`, chosen so that this number becomes exactly −8·d.
3. For each weight, `q = floor(w/d + 8.5)`, i.e. round `w/d` and shift by 8 into 0…15, capped at 15:

| w | w / d | + 8.5 | q (stored) | back: (q − 8)·d | error |
|---|---|---|---|---|---|
| 0.12 | 1.92 | 10.42 | 10 | 0.125 | +0.005 |
| −0.50 | −8 | 0.5 | 0 | −0.5 | 0 |
| 0.33 | 5.28 | 13.78 | 13 | 0.3125 | −0.0175 |
| 0.07 | 1.12 | 9.62 | 9 | 0.0625 | −0.0075 |
| −0.25 | −4 | 4.5 | 4 | −0.25 | 0 |
| 0.44 | 7.04 | 15.54 | 15 | 0.4375 | −0.0025 |
| −0.06 | −0.96 | 7.54 | 7 | −0.0625 | −0.0025 |
| 0.20 | 3.2 | 11.7 | 11 | 0.1875 | −0.0125 |

Eight numbers of 16 or 32 bits become eight 4-bit numbers plus one scale. While multiplying, the kernels unpack
`(q − 8)·d` on the fly. The small errors in the last column are the "quantization error"
([06](06-quantization.md)).

## Appendix B. The whole recipe in one list

```
text ── chat template ──► "<|user|> sky color? <|assistant|>"
     ── tokenizer (split, BPE merges, id lookup) ──► [0, 5, 11, 3, 1]
for each new token (the prompt's tokens all at once the first time):
     x   = E[id]                                   embedding lookup
     for each layer (with that layer's weights):
         n     = RMSNorm(x) · g_attn
         q,k,v = n·Wq, n·Wk, n·Wv
         q,k   = per head: RMSNorm(piece) · g_q, · g_k     (Qwen3's q-norm and k-norm)
         q,k   = RoPE(q, k, position);  store k, v in this layer's KV cache
         per head: w = softmax(q·kⱼ / √headDim) over j ≤ now;  out = Σ wⱼ vⱼ
         h     = x + out·Wo                               residual
         m     = RMSNorm(h) · g_ffn
         x     = h + (silu(m·Wgate) ⊙ (m·Wup))·Wdown       residual; x feeds the next layer
     f   = RMSNorm(x) · g_final;  logits = f·Woutᵀ;  p = softmax(logits / T)
     id  = argmax p  (or a random draw from p)
     stop on <|end|>, else append id and repeat
ids ── detokenizer ──► "blue"
```
