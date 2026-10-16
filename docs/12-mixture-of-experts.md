# 12. Mixture of Experts (Qwen3-30B-A3B)

Code: `model/ModelConverter.convertMoe`, `cpu/CpuBackend.moe/route`, `kernels.metal` (`router`, `moe_up`,
`moe_down`), `metal/MetalBackend.moe`

## 12.1 The idea

In a dense transformer every token passes through the same feed-forward network (FFN), and the FFN is where
most parameters live. A **Mixture of Experts (MoE)** layer replaces that single FFN with many smaller ones — the
*experts* — plus a tiny **router** that picks, for each token, which few experts to use:

```
dense:   x += FFN(x)                                   one FFN of 3072 hidden units  (Qwen3-0.6B)

MoE:     p = softmax(Wrouter · x)                       128 scores, one per expert
         top = the 8 experts with the largest p
         x += Σ_{e ∈ top}  w_e · FFN_e(x)               8 small FFNs of 768 hidden units each
```

"30B-A3B" means **30 billion parameters in total, ~3 billion active per token**. Memory holds all 128 experts
of every layer (because any of them may be chosen), but each token only runs 8 of them. Compute per token is that
of a 3B dense model; knowledge capacity is that of a 30B one. That trade is exactly what a memory-rich,
compute-limited laptop wants.

Qwen3-30B-A3B-Thinking-2507 in numbers:

| | |
|---|---|
| `hidden_size` / heads / KV heads / head_dim | 2048 / 32 / 4 / 128 |
| layers | 48, all MoE |
| `num_experts` / `num_experts_per_tok` / `moe_intermediate_size` | 128 / 8 / 768 |
| parameters: experts | 48 × 128 × 3 × (768 × 2048) ≈ 29.0 B |
| parameters: attention + router + embeddings | ≈ 1.5 B |
| Q4_0 cache file | ≈ 19 GB (experts) + 0.7 GB (two Q8 embedding-sized matrices: `tie_word_embeddings` is false) |
| weights read per generated token | ≈ 2 GB (8 of 128 experts per layer + attention + LM head) |

## 12.2 The router, precisely

For one token vector `x` (after the pre-FFN RMSNorm) and router matrix `Wr [128 × 2048]`:

```
logits_e = Wr[e] · x                           e = 0..127
p_e      = exp(logits_e − max) / Σ exp(…)      softmax, as in sampling
top      = indices of the 8 largest p_e
w_e      = p_e / Σ_{e' ∈ top} p_e'             only if norm_topk_prob = true (it is): the 8 weights sum to 1
```

`CpuBackend.route()` and the `router` Metal kernel implement this; both follow transformers'
`Qwen3MoeSparseMoeBlock` to the letter (softmax *before* top-k, then renormalize). The router is tiny (1 MB per
layer) and is kept in float32, unquantized: a wrong expert choice costs more than a slightly wrong matmul.

Each expert is an ordinary SwiGLU FFN ([05 §3](05-transformer-math.md#3-feed-forward-network-with-swiglu)) with
`moe_intermediate_size = 768`:

```
h_e = silu(Wgate_e · x) ⊙ (Wup_e · x)        768
y_e = Wdown_e · h_e                           2048
x  += w_e · y_e
```

## 12.3 Weight layout: experts stacked into two matrices

Keeping 128 × 3 × 48 = 18 432 separate tensors (and 18 432 GPU buffers) would be clumsy. The converter stacks
all experts of a layer into two `QTensor`s — possible because row-major rows are contiguous
([03 §3.1](03-safetensors.md#31-what-is-a-tensor)):

```
mlp.experts.gate_up   [128 × 1536, 2048]     expert e: rows [e·1536, e·1536 + 768) = gate_e
                                                        rows [e·1536 + 768, (e+1)·1536) = up_e
mlp.experts.down      [128 × 2048, 768]      expert e: rows [e·2048, (e+1)·2048)
```

Selecting expert `e` is then just a row offset — no copying, no per-expert buffers. Two checkpoint layouts are
accepted: per-expert tensors `mlp.experts.N.{gate,up,down}_proj.weight` + router `mlp.gate.weight` (the 30B
repo, transformers ≤ 4.53) and fused 3-D tensors `mlp.experts.gate_up_proj [E, 2m, dim]`,
`mlp.experts.down_proj [E, dim, m]` + router `mlp.router.weight` (newer checkpoints). A `[E, 2m, dim]` tensor is
already the stacked matrix in memory, so the converter merely re-labels its shape (`flatten`).

Dense-MLP exceptions (`mlp_only_layers`, `decoder_sparse_step`) are honoured per layer via
`ModelConfig.isMoeLayer(l)`; Qwen3-30B has none.

## 12.4 CPU implementation

Per token (`CpuBackend.moe`):

1. `route()` — 128 dot products with the SIMD `dot`, softmax, 8 passes of argmax (128 entries: a sort would be
   slower), renormalize.
2. **gate/up for all 8 experts in one parallel loop** over `8 × 1536` output rows: index `i` → expert slot
   `k = i / 1536`, row `r = i % 1536`, weight row `assign[k]·1536 + r`. Threads stream quantized rows through
   `dotRow` exactly as in the dense matmul.
3. SwiGLU per slot: `h[k][i] = silu(g[k][i]) · u[k][i]`.
4. **down for all 8 experts in one parallel loop** over the 2048 output rows: each row sums
   `w_k · dotRow(down, assign[k]·2048 + r, h[k])` over the 8 slots and adds the result to `x[t][r]`. Looping
   over `k` inside means every output element has a single writer — no atomics, no reductions.

Two `forRange` calls per token per layer instead of 16 keeps thread-pool overhead low.

## 12.5 Metal implementation

Four dispatches per MoE layer (versus three for a dense layer), all recorded into the same command buffer:

| kernel | grid | what each threadgroup does |
|---|---|---|
| `router` | `ntok` groups × 128 threads | thread `e` computes logit `e`; thread 0 does softmax + top-8 in threadgroup memory, writes `assign[t·8+k]`, `rw[t·8+k]` |
| `moe_up_q4` | `(1536/4, ntok·8)` | one simdgroup per output row of slot `s`: expert `assign[s]`, token `s/8`; writes `gateUp[s]` |
| `silu_mul` | `ntok·8·768` threads | SwiGLU per slot |
| `moe_down_q4` | `(2048/4, ntok)` | one simdgroup per `(token, row)`: loops over the 8 slots, `x[t][row] += Σ rw·dot` — single writer |

`moe_up`/`moe_down` reuse `block_dot<QT>`, the same 32-weight block kernel the dense `matmul_q4` uses; the
only difference is the *global row* `assign[s] × rowsPerExpert + row` used to address the stacked tensor.
`Q4_0` and `Q8_0` variants are explicit template instantiations (`template [[host_name("moe_up_q4")]] …`).

The stacked expert tensors are bound as single buffers (252 MB and 126 MB per layer), wrapped zero-copy from
the memory-mapped cache file. Metal's `recommendedMaxWorkingSetSize` is checked at start-up and a warning is
printed when the weights exceed it.

## 12.6 Memory and disk on a 32 GB laptop

| | |
|---|---|
| Download | 61 GB of bf16 shards, 16 files |
| Disk needed during conversion with `--low-disk` | ≈ 2 shards (8 GB) + growing cache (19.7 GB) ≈ 28 GB |
| RAM at run time | 19.7 GB weights (mmap'd, paged in on first touch) + KV cache (48 layers × ctx × 512 × 8 B: 0.4 GB at `--ctx 2048`) + ~0.5 GB activations/JVM |

```sh
./jllm run --model Qwen/Qwen3-30B-A3B-Thinking-2507 --low-disk --ctx 2048 --prompt "..."
./jllm serve --model Qwen/Qwen3-30B-A3B-Thinking-2507 --ctx 2048
```

The first command downloads shard by shard (deleting each after it is quantized), writes `jllm-q4_0.bin`, and
runs. Every later start maps the cache in milliseconds. Note that the "Thinking" variant always reasons first:
its chat template has no thinking switch, so expect a `<think>` block before every answer and give it a generous
`max_tokens`.

## 12.7 Measured

Qwen3-30B-A3B-Thinking-2507 at Q4_0 (19.4 GB cache), Apple M1 Pro, idle machine, 413-token prompt
(`./jllm bench --model Qwen/Qwen3-30B-A3B-Thinking-2507 --ctx 1024`):

| backend | prefill | decode |
|---|---|---|
| Metal | 49.2 tok/s | 24.6 tok/s |
| CPU (10 threads, int8 activations) | ≈ 12 tok/s | ≈ 6 tok/s |

Conversion from the 61 GB bf16 shards took 30 minutes with `--low-disk` (download-bound); the cache then maps in
under 0.1 s. Greedy output is identical on both backends ("The capital of France is Paris." for the test prompt),
and thinking mode produces coherent step-by-step reasoning.

## 12.8 Why decode is fast and prefill is not

Decode reads ~2 GB of weights per token (8 active experts), so the bandwidth ceiling is ~10× lower than for the
0.6B model but the same as a dense 3B model. Prefill, however, cannot share weights between tokens the way a
dense matmul does: 64 tokens in a batch typically touch *all* 128 experts, so the whole 19 GB is streamed per
chunk, and the `moe_up` kernel processes one token per simdgroup with no reuse. Grouping tokens by expert
("expert-parallel batching") is the standard fix and the natural next optimization.
