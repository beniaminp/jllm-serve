# 12. Mixture of Experts (Qwen3-30B-A3B)

Code: `model/Mlp.Moe` + `model/Router` (the typed spec, built by `model/Families`), `model/MlpWeights.Moe`,
`model/ModelConverter.moeRules`, `cpu/CpuBackend.MoeFfn` (routing: `routeSoftmax`, `routeDeepseek`) and
`cpu/MoeExperts` (the experts: `token`, `batched`, `expertMatmul`; shared with the Metal backend's CPU experts),
`metal/MetalBackend.MoeStep`, `metal/ExpertPlacement`, `kernels.metal` section 5 (`router`, `moe_up`, `moe_down`,
the grouped `moe_sort` / `moe_up_mma` / `moe_down_mma` / `moe_combine`)

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
| Q4_0 cache file | 19.4 GB measured: ≈ 18 GB of experts, the rest attention + two Q8_0 embedding-sized matrices (`tie_word_embeddings` is false) |
| weights read per generated token | ≈ 2 GB (8 of 128 experts per layer + attention + LM head) |

## 12.2 The router, precisely

For one token vector `x` (after the pre-FFN RMSNorm) and router matrix `Wr [128 × 2048]`:

```
logits_e = Wr[e] · x                           e = 0..127
p_e      = exp(logits_e − max) / Σ exp(…)      softmax, as in sampling
top      = indices of the 8 largest p_e
w_e      = p_e / Σ_{e' ∈ top} p_e'             only if norm_topk_prob = true (it is): the 8 weights sum to 1
```

`CpuBackend.MoeFfn.routeSoftmax` and the `router` Metal kernel implement this (`Router.Softmax`, with
`Mlp.Moe.normTopK` as the `norm_topk_prob` flag); both follow transformers' `Qwen3MoeSparseMoeBlock` to the letter
(softmax *before* top-k, then renormalize). The router is tiny (1 MB per layer) and is stored in the cache as the f32
tensor `mlp.router`, unquantized: a wrong expert choice costs more than a slightly wrong matmul. DeepSeek-V2/V3
models use the other `Router` variant, `Router.DeepSeek` (sigmoid or softmax scores, group-limited selection, a
correction bias, a scaling factor): see [14.11.4](14-model-families.md#14114-the-three-routing-rules). It plugs into
the same stacked experts, the same `moe_up` / `moe_down` kernels and the same prefill grouping described below.

Each expert is an ordinary SwiGLU FFN ([05 §5.8](05-transformer-math.md#58-steps-68--the-feed-forward-network-swiglu)) with
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

Which layers are MoE is decided once, when `Families` builds the `LayerSpec` list: a layer is `Mlp.Moe` if the
config has `num_experts` > 0, the layer is not listed in `mlp_only_layers` and `(l + 1) % decoder_sparse_step == 0`,
otherwise it is an `Mlp.Dense`; the converter and both backends just switch on the variant. Qwen3-30B has no dense
layers (DeepSeek's rule, `first_k_dense_replace` and `moe_layer_freq`, is in [14.11.3](14-model-families.md#14113-mixture-of-experts-dense-prefix-routed--shared-experts)).
The cache tensors of a layer are `mlp.router` (f32), `mlp.experts.gate_up` and `mlp.experts.down`
(`MlpWeights.Moe`; `mlp.shared.*` and `mlp.router_bias` exist only for DeepSeek).

## 12.4 CPU implementation

Decode and prefill take different paths (`MoeFfn.run` routes, then calls `MoeExperts`).

**One token** (decode, `MoeExperts.token`):

1. route: 128 dot products with the SIMD `dot` (parallel over experts), softmax, 8 passes of argmax (128 entries: a
   sort would be slower), renormalize.
2. **gate/up for all 8 experts in one parallel loop** over `8 × 1536` output rows: index `i` → expert slot
   `k = i / 1536`, row `r = i % 1536`, weight row `assign[k]·1536 + r`. Threads stream quantized rows through
   `dotRow` exactly as in the dense matmul, or through the int8 `Int8Kernels.dotRow` when
   `backend.cpu.int8-activations` is on ([07](07-cpu-backend.md)).
3. SwiGLU per slot: `h[k][i] = silu(g[k][i]) · u[k][i]`.
4. **down for all 8 experts in one parallel loop** over the 2048 output rows: each row sums
   `w_k · dotRow(down, assign[k]·2048 + r, h[k])` over the 8 slots and adds the result to `x[t][r]`. Looping
   over `k` inside means every output element has a single writer — no atomics, no reductions.

Two `forRange` calls per token per layer instead of 16 keeps thread-pool overhead low.

**Many tokens** (prefill, `MoeExperts.batched`): doing the above per token would stream every selected expert's weights once
per token. Instead the whole chunk is grouped by expert:

1. route every token, one token per thread (the router logits are then computed serially inside the thread);
2. a counting sort turns the `ntok × 8` (token, expert) slots into per-expert lists (`expertStart`, `slotOrder`) and
   collects the experts that were hit at all (`usedExperts`);
3. `expertMatmul` for gate/up: parallel over (expert, group of 4 rows) pairs; the four rows are dequantized once and
   multiplied with the inputs of *all* slots routed to that expert, four slots at a time (`dot4x4`), so a weight row
   is read once per chunk instead of once per token;
4. SwiGLU per slot, then `expertMatmul` for down (input = that slot's hidden vector);
5. `VectorKernels.combine` adds `Σ_k w_k · y_k` to `x[t]` per token in slot order: one writer per element again.

This is the "expert-parallel batching" that the first version of this chapter listed as future work. With
`-Djllm.cpu.deterministic=true` prefill goes token by token through the single-token path, for bit-reproducible
output.

## 12.5 Metal implementation

Three dispatches per MoE layer after the pre-FFN RMSNorm, all recorded into the same command buffer:

| kernel | grid | what each threadgroup does |
|---|---|---|
| `router` (`router_ds` for DeepSeek) | `ntok` groups × 256 threads | the 8 simdgroups compute the logits (simdgroup `sg` takes experts `sg, sg+8, …`; its 32 lanes read the router row in coalesced `float4` steps and `simd_sum`); softmax max and sum are threadgroup reductions; thread 0 does the top-8 and writes `assign[t·8+k]`, `rw[t·8+k]` (at most 1024 experts) |
| `moe_up_q4` / `_q8` / `_q2k` | `(ceil(768/4), ntok·8)` | one simdgroup per output row of slot `s`: expert `assign[s]`, token `s/8`. It computes the gate row `row` **and** the up row `row + 768` of that expert and applies `silu(g)·u` in the epilogue, writing `h[s]` directly |
| `moe_down_q4` / `_q8` / `_q2k` | `(ceil(2048/4), ntok)` | one simdgroup per `(token, row)`: loops over the 8 slots, `x[t][row] += Σ rw·dot` — single writer |

Fusing the SwiGLU into `moe_up` removed the separate `silu_mul` pass and the gate/up scratch buffer; each of the two
dot products is accumulated in the same order as before, so the results did not change.

`moe_up`/`moe_down` reuse `block_dot<QT>`, the same block kernel the dense `matmul` uses; the only difference is the
*global row* `assign[s] × rowsPerExpert + row` used to address the stacked tensor. The `Q4_0`, `Q8_0` and `Q2_K`
variants are explicit template instantiations (`template [[host_name("moe_up_q4")]] …`).

The stacked expert tensors are bound as single buffers (252 MB and 126 MB per layer at Q4_0), wrapped zero-copy from
the memory-mapped cache file. Metal's `recommendedMaxWorkingSetSize` is checked at start-up: when the GPU's share
would exceed it, `backend.metal.cpu-moe=auto` leaves just enough of the least used experts on the CPU
([08 §8.18](08-metal-backend.md#818-experts-on-the-cpu-backendmetalcpu-moe)); otherwise a warning is printed. DeepSeek's shared experts (always-on, one dense SwiGLU) run after `moe_down`
through the ordinary fused SwiGLU and down matmul.

## 12.6 Memory and disk on a 32 GB laptop

| | |
|---|---|
| Download | 61 GB of bf16 shards, 16 files |
| Disk needed during conversion with `--low-disk` | ≈ 2 shards (8 GB) + growing cache (19.4 GB) ≈ 28 GB |
| RAM at run time | 19.4 GB weights (mmap'd, paged in on first touch) + KV cache (48 layers × ctx × 512 × 8 B: 0.4 GB at `--ctx 2048`) + ~0.5 GB activations/JVM |

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
0.6B model but the same as a dense 3B model. Prefill cannot share weights between tokens the way a dense matmul
does: 64 tokens in a batch typically touch *all* 128 experts, so the whole 19 GB is streamed per chunk.

📐 With `n` tokens per chunk, `E` experts and `topK` picks per token, an expert serves on average

```
tokens per expert = n · topK / E          64 · 8 / 128 = 4,     256 · 8 / 128 = 16
```

tokens of the chunk, and its weights are read (on the CPU: each group of 4 rows dequantized) once for all of them.
The fixed cost per expert is shared by 4 tokens at `n = 64` and by 16 at `n = 256`. Both backends group the slots
by expert: the CPU with a counting sort (12.4), Metal with `moe_sort` and the simdgroup-matrix tile kernels
([08 §8.15](08-metal-backend.md#815-expert-grouped-moe-prefill)). And MoE models prefill in **256-token chunks**
unless `backend.batch` is set to a number. Measured on the CPU expert path with one Qwen3-30B-A3B-shaped layer (Q4_0, 4
cores), that is 1.5–1.6× faster per token than 64-token chunks; on Metal it is not measured yet
([11](11-performance.md)).

The same code serves DeepSeek-V2/V3 ([14.11](14-model-families.md#1411-deepseek-v2--v3-multi-head-latent-attention--shared-experts)):
only the router (`Router.DeepSeek`) and the extra shared-expert matmul differ.

## 12.9 Testing

`ConverterTest` converts a tiny random `Qwen3MoeForCausalLM` checkpoint (both the per-expert and the fused 3-D expert
layouts are accepted) and checks the stacked tensors; the DeepSeek tiny models of `scripts/arch-smoke.sh` (8 routed
experts, shared experts, three routing rules; [14.11.8](14-model-families.md#14118-testing)) run the MoE kernels and the
router on CPU and Metal and require identical tokens and logits within 1e-3, and `DeepSeekRef` compares them with an
independent float64 reference. A tiny `Qwen3MoeForCausalLM` (8 experts, top 2, one dense layer) runs there too, and
every MoE model is also run with experts on the CPU: one whole layer, all layers, every other expert of every layer,
and all but one expert of a layer, each against the CPU backend. Those runs also check a 20-token chunk, the usage
counts, and two slots sharing one placement. `ExpertPlacementCheck` (in `mvn test`) checks the planner. The
Qwen3-30B numbers above come from the real model.

## 12.10 A model bigger than the GPU's memory budget

🧸 Some specialists are asked all day, some almost never. When there isn't room for all of them at the GPU's desk,
the rarely asked ones wait in the CPU's office. The GPU works through its part, and only when a word needs one of
the CPU's specialists does the CPU answer, at the same time as the GPU does its own share.

📐 Metal keeps only `recommendedMaxWorkingSetSize` (about two thirds to three quarters of RAM) resident for the GPU,
and an MoE model is almost all experts (Qwen3-30B-A3B Q4_0: ~18 of 19.4 GB). `backend.metal.cpu-moe=auto` (the
default) does nothing when everything fits. Otherwise it leaves just enough experts on the CPU, chosen to minimize

```
cost = 100 µs · (layers with a CPU expert)  +  Σ_{CPU experts} p(l,e) · expertBytes / CPU bandwidth
```

where `p(l,e)` is how often a token picks that expert, learned from earlier runs and saved in
`~/.cache/jllm/expert-usage`. A first run, with no statistics, moves whole layers. Later runs move the rarely
picked experts, so the CPU is called for a few percent of the picks instead of a whole layer's share. The hot
experts of a split layer are packed into their own GPU memory; the GPU skips the cold slots (the router marks them
−1) while the CPU computes them with the same `MoeExperts` code as the CPU backend. Every detail, the timeline
and the tests are in [08 §8.18](08-metal-backend.md#818-experts-on-the-cpu-backendmetalcpu-moe).
