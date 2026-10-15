# 15. Mamba — a model without attention

Code: `model/ModelConfig.MambaParams`, `model/MambaLayer`, `model/ModelConverter.planMamba`, `cpu/MambaCpuBackend`,
`metal/MambaMetalBackend`, `kernels.metal` (`mamba_conv`, `mamba_scan`), `Backend.supportsPrefixReuse`,
`engine/Engine.generateLocked`

Chapter 5 describes a transformer: every new token **looks back at all earlier tokens** through attention, which is
why the engine keeps a KV cache that grows with the conversation. **Mamba** (Gu & Dao, 2023, `MambaForCausalLM`;
checkpoints `state-spaces/mamba-130m-hf` … `mamba-2.8b-hf`) is built differently: it has no attention and no KV
cache. It carries a **fixed-size summary of everything seen so far** — a *state* — and updates it once per token.
This chapter explains the math with a worked example, how the code maps onto it, and what it changes for the
engine. Only Mamba v1 (`model_type: mamba`) is implemented; see [15.9](#159-not-supported).

## 15.0 The toolbox

### A recurrence

A **recurrence** computes each value from the previous one. A running total is the simplest:

```
s_t = s_{t-1} + x_t           x = 3, 1, 4, 1, 5   →   s = 3, 4, 8, 9, 14
```

After any number of inputs the memory is one number, `s`. A *state-space model* (SSM) is a richer version: the state
is a vector `s`, and each step first **shrinks** the old state and then **adds** something from the new input:

```
s_t = a · s_{t-1} + b · x_t          y_t = c · s_t
```

with `a` between 0 and 1. If `a = 0.5` the contribution of an input halves each step — the state is an
exponentially fading memory of the past. `a` close to 1 remembers for a long time, `a` close to 0 forgets almost
immediately. `y_t = c · s_t` reads the state out.

### Why this replaces attention (and what it costs)

Attention keeps *every* past key/value and lets the new token pick what it needs: precise, but memory and time per
token grow with the context. The recurrence compresses the whole past into `s` (here `d_inner × d_state` numbers per
layer), so each token costs the same however long the conversation is. The price is that the model must decide
*while reading* what is worth keeping — it cannot go back and re-read.

### What "selective" means

A classical SSM uses the same `a`, `b`, `c` for every token. Mamba computes the step size `Δ` and the vectors `B` and
`C` **from the current token**, so the model can choose per token to overwrite the state (large `Δ`: "this is
important, remember it") or leave it alone (`Δ ≈ 0`: "ignore this"). That input dependence is the selection; `A`
and `D` stay fixed learned parameters.

## 15.1 One Mamba block

For every layer and every token `x` (a vector of `d_model` numbers), in `MambaCpuBackend.forward`:

```
h        = RMSNorm(x)                                   # same norm as 5.3, weight  norm.weight
[xi | z] = in_proj · h                                  # 2·d_inner numbers: the conv/SSM branch xi and a gate z
u        = silu(causal_conv1d(xi) + conv_bias)          # 15.3: mixes the last d_conv inputs of each channel
[dt | B | C] = x_proj · u                               # dt_rank, d_state and d_state numbers
Δ        = softplus(dt_proj · dt + dt_bias)             # one positive step size per channel
A        = −exp(A_log)                                  # fixed, negative: [d_inner, d_state]
s[i,n]   = exp(Δ[i]·A[i,n]) · s[i,n] + Δ[i]·B[n]·u[i]  # 15.2: the recurrence, per channel i and state n
y[i]     = Σ_n s[i,n]·C[n]  +  D[i]·u[i]                # read-out plus a skip connection
y        = y · silu(z)                                  # the gate
x        = x + out_proj · y                             # residual, like 5.7
```

After the last layer: `norm_f` (RMSNorm) and the LM head, exactly like 5.10. There is no positional encoding of any
kind: order is carried by the recurrence itself. `silu(v) = v / (1 + e^{−v})` as in the SwiGLU of 5.8;
`softplus(v) = ln(1 + e^v)` is a smooth version of `max(v, 0)` that is always positive (the code returns `v` itself
above 20, like torch).

Notation: `d_model` = hidden size of the residual stream (768), `d_inner = expand · d_model` (1536) the width inside
the block, `d_state` (16) the size of the memory per channel, `d_conv` (4) the conv width, `dt_rank` (48,
`ceil(d_model/16)`) the width of the low-rank bottleneck for `Δ`.

## 15.2 The recurrence and its discretization

The model is derived from a continuous-time equation, `ds/dτ = A·s + B·x`, where `A` is negative so the state decays.
Advancing it by a time step `Δ` has the exact solution factor `exp(Δ·A)` for the old state, which is the
**discretization**: `Ā = exp(Δ·A)`. For the input term the reference implementation (and this one) uses the simple
Euler form `B̄ = Δ·B` instead of the exact integral. Hence

```
s ← exp(Δ·A) · s  +  Δ·B·u
```

Since `A < 0` and `Δ > 0`, `exp(Δ·A)` lies in (0, 1]: a small `Δ` gives a factor near 1 (keep the state, take in
almost nothing), a large `Δ` gives a factor near 0 (forget, replace by the new input). `A` differs per state index
(initialised to −1, −2, … −16), so some state entries fade fast and some slowly — a bank of memories with different
time scales.

**Worked example** (the numbers of the unit test in `SmokeTest.mamba`): one channel, `d_state = 2`.

```
Δ = 0.5     u = 2      A = (−1, −2)     B = (1, 2)     C = (3, 4)      previous state s = (1, 0.5)

s[0] = exp(0.5·−1)·1   + 0.5·1·2 = 0.6065 + 1       = 1.6065
s[1] = exp(0.5·−2)·0.5 + 0.5·2·2 = 0.3679·0.5 + 2   = 2.1839
y    = 1.6065·3 + 2.1839·4                           = 13.5554       (before D·u and the gate)
```

State entry 0 (`A = −1`) kept 61 % of its old value, entry 1 (`A = −2`) only 37 %. Feeding the same token with `Δ = 0`
changes nothing: the factors are `exp(0) = 1` and the added term is 0.

The recurrence runs **per channel** (`i` in `d_inner`) and the channels never interact inside the scan. Between
tokens it is strictly sequential, so token `t` needs the state after token `t−1`; across channels it is trivially
parallel. That is how both backends use it.

## 15.3 The causal convolution

Before the scan, each channel is mixed with its own recent past by a tiny filter of `d_conv = 4` weights:

```
c_t = silu( w[0]·x_{t−3} + w[1]·x_{t−2} + w[2]·x_{t−1} + w[3]·x_t + bias )
```

*Depthwise* means each of the `d_inner` channels has its own 4 weights (`conv1d.weight` is `[d_inner, 1, 4]`);
*causal* means it only looks backwards. To decode one token at a time the layer must remember the last `d_conv − 1`
inputs of every channel: that is the **conv state**, `[d_inner, d_conv − 1]` (oldest first). Example from the unit
test: state `(1, −1, 2)`, new input `3`, weights `(1, 2, 3, 4)`, bias `0.5`:

```
window = (1, −1, 2, 3)      sum = 1·1 + 2·(−1) + 3·2 + 4·3 + 0.5 = 17.5      silu(17.5) ≈ 17.5
new state = (−1, 2, 3)      # the oldest value dropped out, 3 came in
```

At the start of a sequence the conv state is zero, which is what padding the sequence with zeros on the left means in
the transformers implementation (`padding = d_conv − 1`). `convStep` in `MambaCpuBackend` is this function; the
Metal kernel `mamba_conv` is the same loop with one thread per channel.

## 15.4 Shapes for mamba-130m (verified on the Hub)

`config.json` (`state-spaces/mamba-130m-hf`): `hidden_size 768`, `num_hidden_layers 24`, `state_size 16`,
`conv_kernel 4`, `expand 2` (`intermediate_size 1536`), `time_step_rank 48`, `layer_norm_epsilon 1e-5`,
`vocab_size 50280`, `use_bias false`, `use_conv_bias true`, `eos_token_id 0`, `bos_token_id 0`, no
`tie_word_embeddings` key. `mamba-2.8b-hf`: 2560 / 64 layers / `intermediate_size 5120` / `time_step_rank 160`, the
rest identical. (The checkpoint's vocabulary is 50280, already padded, so the config and the embedding agree.)

Tensors (read from the safetensors header: 242 tensors for 130m; 2.8b is sharded in 3 files, 642 tensors):

| Tensor | Shape (130m) | Stored in jllm as |
|---|---|---|
| `backbone.embeddings.weight` | `[50280, 768]` | Q8_0 (also the LM head: the checkpoints have **no** `lm_head.weight`; transformers ties it) |
| `backbone.layers.N.norm.weight` | `[768]` | f32 |
| `…mixer.in_proj.weight` | `[3072, 768]` = `[2·d_inner, d_model]` | Q4_0 / Q8_0 |
| `…mixer.conv1d.weight` | `[1536, 1, 4]` | f32 |
| `…mixer.conv1d.bias` | `[1536]` | f32 |
| `…mixer.x_proj.weight` | `[80, 1536]` = `[dt_rank + 2·d_state, d_inner]` | Q4_0 / Q8_0 |
| `…mixer.dt_proj.weight` | `[1536, 48]` = `[d_inner, dt_rank]` | f32 |
| `…mixer.dt_proj.bias` | `[1536]` | f32 |
| `…mixer.A_log` | `[1536, 16]` | f32 as **`A = −exp(A_log)`**, name `…mixer.A` |
| `…mixer.D` | `[1536]` | f32 |
| `…mixer.out_proj.weight` | `[768, 1536]` | Q4_0 / Q8_0 |
| `backbone.norm_f.weight` | `[768]` | f32 |

Everything big goes through `QTensor`. `dt_proj` has `dt_rank` columns, which is 48 for 130m — not a multiple of the
32-wide quantization block — and it is tiny (0.07 M of the 129 M parameters), so it stays f32, as do the conv, `A`,
`D`, biases and norms (together under 4 % of the weights). The converter precomputes `A = −exp(A_log)` once, so the
scan never calls `exp` on a parameter.

## 15.5 State instead of a KV cache

Per layer the backend owns

```
conv state  [d_inner, d_conv − 1]   f32      1536·3·4  =  18 432 B
ssm  state  [d_inner, d_state]      f32      1536·16·4 =  98 304 B
```

For mamba-130m that is 116 736 B × 24 layers = **2.8 MB, whatever the context length**. Compare the KV cache of
Qwen3-0.6B (28 layers × 2 × 1024 floats per token): 229 kB *per token*, i.e. 117 MB at 512 tokens and 940 MB at 4096
([5.9](05-transformer-math.md#59-prefill-decode-and-the-kv-cache)). mamba-2.8b needs 25 MB of state in total. The
`ctx` setting no longer sizes any buffer; it is only the limit the engine enforces on prompt + generated tokens
(Mamba itself has no hard context limit, though quality drops far past the training length).

`Backend.forward(tokens, off, n, pos, logits)` keeps its contract, but positions are implicit: `pos == 0` **resets**
both states to zero, any other `pos` must equal the number of tokens fed since (otherwise
`IllegalStateException`). Splitting a prompt into chunks — 12 tokens at once, 5 + 7, or one by one — gives the same
logits (`ArchSmoke` checks this on both backends).

## 15.6 The prefix cache and Mamba

The engine's prefix cache ([10.2](10-engine-and-server.md#102-the-prefix-cache)) reuses the K/V rows of the common
prefix of consecutive prompts and recomputes only the rest. A transformer can do that because row `k` of the cache
depends only on tokens `0 … k`, so rows beyond the shared prefix can simply be overwritten. A Mamba state is a *single
summary of all the tokens so far*: it cannot be rewound to "the state after token 6" once it has absorbed token 9.

So `Backend` has a capability, `supportsPrefixReuse()` (default `true`, Mamba returns `false`), and the engine
treats it as follows:

| Situation | KV-cache backend | Mamba backend |
|---|---|---|
| new prompt extends all cached tokens (the usual chat turn) | resume after the shared prefix | resume at `cachedLen` — the current state is exactly right |
| new prompt shares only a shorter prefix (edited history) | resume after the shared prefix | restart at position 0 |
| identical prompt re-sent | recompute the last token | restart at position 0 (the last token is already in the state) |

The cheap case matters: in a chat the next prompt is the previous prompt + the model's reply + the new user turn, so
the state after the previous request is reused as-is and only the new tokens are fed. A snapshot of the state at
each turn (to rewind to a shared prefix) would cost only 2.8 MB for 130m and is a possible extension; it is not
implemented. Covered by `SmokeTest.prefixReuse` with a recording backend.

## 15.7 The implementations

**CPU (`MambaCpuBackend`)**. For a chunk of `n` tokens, per layer: RMSNorm of the `n` rows; `in_proj` as a quantized
matmul over the chunk (same scheme as `CpuBackend.matmul`: streamed rows for decode, four dequantized rows reused for
all tokens for prefill); `conv` — parallel over channels with `ThreadPool.forRange`, each worker loops over the `n`
tokens of its channels, so the state of a channel is touched by one thread only; `x_proj` matmul; `scan` — again
parallel over channels, sequential over tokens inside; `out_proj` matmul accumulating into the residual. The `dt_proj`
dot product (length `dt_rank`) uses `VectorKernels.dot`. The scan's `exp` is scalar (`d_state` values per channel and
token); `d_state` = 16 is too short to be worth vectorizing across, and the Vector API has no exp that is fast on every platform.

**Metal (`MambaMetalBackend`, kernels at the end of `kernels.metal`)**. The whole chunk is one command buffer.
Per layer: `rmsnorm`, `matmul_q4/q8` (`in_proj`), `mamba_conv`, `matmul` (`x_proj`), `mamba_scan`, `matmul` (`out_proj`,
accumulating) — six dispatches. The projections reuse the existing batched matmul kernels unchanged. `mamba_conv`
and `mamba_scan` use **one GPU thread per channel** that loops over the chunk's tokens itself; the SSM state of the
channel (≤ 64 floats) is held in registers for the whole chunk and written back once. The `dt_proj` product is done
inside the scan kernel so no `[n, d_inner]` intermediate is needed.

*Trade-off.* The alternative is one dispatch per token and layer, which would give each launch more independence but
costs `n` times more launches; or a parallel prefix scan over tokens (the recurrence `s ← a·s + b` composes
associatively), which exposes parallelism over tokens too but needs a combine step and extra buffers. With 1 536 –
5 120 channels per layer there is plenty of parallelism for decode (where the cost is reading the weights anyway), and
the per-thread sequential loop keeps one dispatch per stage regardless of the chunk size. For very long prompts the
prefill scan could become the bottleneck; the associative scan is the way to fix that.

Both backends compute `exp` / `softplus` the same way (Metal uses `precise::exp/log`), so CPU and Metal agree to
~1e-5 on the tiny test models.

## 15.8 Measured numbers

All on an M1 Pro **while the machine was heavily loaded by other jobs** (load average > 100), so treat them as
rough. The checkpoint is a random-weight file with the exact mamba-130m shapes
(`scripts/make-tiny-model.py <dir> mamba130m`; Q4_0 cache 110 MB), because downloading the real weights was not part
of this work; speed does not depend on the values. `jllm bench`, 401-token prompt, 64 generated tokens:

| Backend | prefill | decode |
|---|---|---|
| Metal | ~1 060 tok/s | ~273 tok/s |
| CPU (Vector API, 10 threads, loaded machine) | ~115 tok/s | ~15 tok/s |

Profiling decode on the CPU: the quantized matmuls (`in_proj`, `out_proj`) dominate; the conv + scan are ~10 % of the
time. Qwen3-0.6B measured in the same session on the same loaded machine ran at 4.7 tok/s on the CPU backend, so the
CPU number reflects the machine, not Mamba.

Correctness: a float64 pure-python implementation of the transformers `slow_forward` (full-sequence conv and scan) on
tiny random checkpoints agrees with the CPU and Metal backends to the Q8 quantization error (final logits within
1e-4 relative on d_model 64 / 3 layers and d_model 128 / 2 layers, with state 16 and 8, conv 4 and 3, tied and untied
head). Deep random networks amplify float32 rounding (at the 130m shape, 8 layers: 0.3 % vs float64, CPU vs Metal
0.01 %), which is why `arch-smoke` uses small models.

## 15.9 Not supported

- **Mamba-2 (`model_type: mamba2`)**, Falcon-Mamba (`falcon_mamba`), Jamba and other hybrids: different block
  structure (grouped heads, extra norm, scalar `A`), rejected at load time with the usual "unsupported model" message.
  Mamba v1 only.
- `use_bias: true` (biases on `in_proj`/`out_proj`): no released checkpoint uses it; rejected with a message.
  `use_conv_bias: false` is supported.
- `state_size` above 64 or `conv_kernel` above 8; `hidden_act` other than `silu`.
- `d_model` and `d_inner` must be multiples of 32 (quantization block): true for all `state-spaces/mamba-*-hf`.
- A snapshot/rewind of the recurrent state (15.6), and the associative scan for long prefills (15.7).
- `backend.kv-dtype` is ignored (there is no KV cache).

The plain-text chat format (no template; messages joined with newlines, `<|endoftext|>` stops) lives in
`ChatTemplate` for `mamba`; the model is a base model and simply continues the text. There is no BOS token
(`ModelConfig.promptBosId()` is −1 for Mamba even though the config has `bos_token_id: 0`).

## 15.10 Testing

```
sh scripts/arch-smoke.sh            # tiny Mamba (d_model 64 and 128): CPU == Metal ids and logits, chunking invariance
mvn -q test                         # SmokeTest.mamba: config, hand-computed scan step and causal conv; SmokeTest.prefixReuse
```
