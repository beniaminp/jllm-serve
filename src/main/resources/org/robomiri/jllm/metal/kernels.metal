// Compute kernels of the Metal backends (dense decoders, DeepSeek MLA, Mamba) for Apple GPUs.
// Compiled at runtime by libjllm_metal (newLibraryWithSource); every dispatch is written by MetalKernels.java.
// Parameter structs contain only 32-bit ints/floats so the Java side writes them field by field (KernelParams
// lists the layouts, MetalParamsTest checks them against the structs below). Sections:
//   0. constants   1. common (reductions, block formats)   2. matmul   3. norm + rope   4. attention
//   5. mixture of experts   6. state space (Mamba)   7. DeepSeek MLA (latent cache store, query absorption)
//   8. Q6_K (block format, row / tile walks and the _q6k instantiations of every matmul / moe kernel)
//   9. flash-attention prefill (attention_fa: simdgroup-matrix QK^T and PV, online softmax)
//  10. pipelined greedy decoding (embed_token: the embedding of a token id the GPU chose; argmax)
//  11. Qwen3-Next (Gated DeltaNet conv / delta rule / gated norm, the attention output gate, the shared-expert gate)
//  12. half-weight MMA prefill (*_hmma: the section 2b / 5 MMA kernels with half weight tiles, chosen per shape)
//  13. flash prefill, second form (attention_fb: K and V staged together, half staging for f16 / q8 caches)
//  14. q4 KV cache (kv_store_q4: Hadamard rotation + 4-bit groups; the *_q4 attention kernels, recent window)
// Model-specific pieces live in the generic kernels: Gemma's sandwich norms are rmsnorm_add, its GeGLU the
// `gelu` flag of matmul_swiglu, its attention soft-cap AttnParams.softcap and its final-logit soft-cap
// MatmulParams.softcap on the LM-head matmul; GLM / Phi partial rotation is rope_il / rope with rotHalf.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------------------------- 0. constants
// Tile sizes, thread counts and limits. MetalConstants.java mirrors every value below and MetalParamsTest
// checks the mirror against this block.

#define QK 32                   // weights per quantization block (Q4_0 / Q8_0)
#define QK_K 256                // weights per Q2_K super-block
#define Q2K_BYTES 84            // bytes per Q2_K super-block
#define Q6K_BYTES 210           // bytes per Q6_K super-block (QK_K weights)
#define TOK_TILE 8              // tokens per simdgroup of the single-row matmuls
#define PF_ROWS 2               // rows per simdgroup of matmul_pf / matmul3_pf
#define PF_UNITS 1              // gate + up units per simdgroup of matmul_swiglu_pf
#define PF_TOKS 8               // tokens per simdgroup of the prefill matmuls
#define MMA_BR 64               // output rows per threadgroup of the MMA prefill matmuls
#define MMA_BT 32               // tokens per threadgroup of the MMA prefill matmuls
#define MM_THREADS 128          // matmul threadgroup: 4 simdgroups = 4 rows (norm_scale_128 relies on it)
#define ROW_THREADS 256         // rmsnorm / router / attention_combine threadgroup (one row each)
#define ATT_THREADS 256         // attention threadgroup = ATT_SIMDGROUPS simdgroups
#define ATT_SIMDGROUPS 8
#define ATT_TILE 32             // tokens per attention dispatch (scores scratch rows per head)
#define MAX_PER 8               // widest head in units of 32 lanes (headDim <= 256)
#define MLA_MAX_PER 18          // widest absorbed MLA query / cache row in units of 32 lanes (kv_lora_rank + rope <= 576)
#define CHANNEL_THREADS 64      // mla_lat_store / mamba_conv / mamba_scan threadgroup
#define MAX_EXPERTS 1024        // router logits held in threadgroup memory
#define MAX_GROUPS 64           // DeepSeek expert groups
#define FLASH_MAX_SPLITS 64     // partials attention_combine can merge
#define MAMBA_MAX_STATE 64      // SSM state width held in registers
#define MAMBA_MAX_CONV 8        // conv kernel width held in registers
#define FA_ROWS 32              // query tokens per threadgroup of attention_fa (4 simdgroups x 8 rows)
#define FA_THREADS 128          // attention_fa threadgroup
#define TANH_CLAMP 20.0f        // |argument| bound of fast-math tanh (overflows to NaN beyond)
#define SOFTPLUS_LINEAR 20.0f   // softplus(x) = x beyond this

// ---------------------------------------------------------------------------------------------- 1. common: reductions

static inline float tg_sum(float v, threadgroup float *buf, ushort sg, ushort lane, ushort nsg) {
    v = simd_sum(v);
    if (lane == 0) buf[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? buf[lane] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

static inline float tg_max(float v, threadgroup float *buf, ushort sg, ushort lane, ushort nsg) {
    v = simd_max(v);
    if (lane == 0) buf[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? buf[lane] : -INFINITY;
    r = simd_max(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// ---------------------------------------------------------------------------------------------- 1. common: block formats
// scale * (32 quantized weights of global row `grow`, block `b`) . x4[0..8)   (QT = 4 or 8)

template <int QT>
static inline float block_dot(device const uchar *wq, device const float *ws, ulong grow, int nb, int b,
                              device const float4 *x4) {
    float d = 0.0f;
    if (QT == 4) {
        device const uchar4 *rq = (device const uchar4 *)(wq + grow * nb * 16) + b * 4;
        for (int k = 0; k < 4; k++) {
            uchar4 v = rq[k];
            d += dot(float4(v & uchar4(15)) - 8.0f, x4[k]) + dot(float4(v >> uchar4(4)) - 8.0f, x4[k + 4]);
        }
    } else {
        device const char4 *rq = (device const char4 *)(wq + grow * nb * 32) + b * 8;
        for (int k = 0; k < 8; k++) d += dot(float4(rq[k]), x4[k]);
    }
    return d * ws[grow * nb + b];
}

// llama.cpp's Q2_K: 256-weight super-blocks of 84 bytes = 16 scale/min bytes (low nibble = 4-bit scale sc,
// high nibble = 4-bit min m) + 64 bytes of 2-bit quants + f16 d + f16 dmin; weight = d*sc*q - dmin*m.
// Sub-block j (weights 16j .. 16j+15) is at qs + (j/8)*32 + (j%2)*16 with bit shift 2*((j%8)/2).
// Scales live inside the block, so buffer(1) (ws) is unused by these kernels. See docs/13-q2k-quantization.md.

// Dequantizes 16-weight sub-block u of global row `grow` (nsb super-blocks per row) into w[0..4).
static inline void q2k_sub_block(device const uchar *wq, ulong grow, int nsb, int u, thread float4 *w) {
    device const uchar *blk = wq + (grow * nsb + (u >> 4)) * Q2K_BYTES;
    const int j = u & 15;
    const uchar s = blk[j];
    const float2 dd = float2(as_type<half2>(*(device const uint *)(blk + 80)));   // d, dmin (4-byte aligned)
    const float dl = dd.x * float(s & 15);
    const float ml = dd.y * float(s >> 4);
    const uchar shift = uchar(((j & 7) >> 1) << 1);
    device const uchar4 *q = (device const uchar4 *)(blk + 16 + (j >> 3) * 32 + (j & 1) * 16);
    for (int k = 0; k < 4; k++) w[k] = float4((q[k] >> uchar4(shift)) & uchar4(3)) * dl - ml;
}

// block_dot for the MoE kernels: 32-weight block b of a Q2_K row = sub-blocks 2b and 2b+1 (nb = cols / 32).
template <>
inline float block_dot<2>(device const uchar *wq, device const float *ws, ulong grow, int nb, int b,
                          device const float4 *x4) {
    const int nsb = (nb + 7) >> 3;
    float4 w[4];
    float d = 0.0f;
    q2k_sub_block(wq, grow, nsb, 2 * b, w);
    for (int k = 0; k < 4; k++) d += dot(w[k], x4[k]);
    q2k_sub_block(wq, grow, nsb, 2 * b + 1, w);
    for (int k = 0; k < 4; k++) d += dot(w[k], x4[k + 4]);
    return d;
}

// ---------------------------------------------------------------------------------------------- 2. matmul
// out[t, row] (+)= sum_c W[row, c] * x[t, c]
// One simdgroup per output row; each lane walks blocks lane, lane+32, ...; up to TOK_TILE tokens share the
// dequantized block. Grid: (ceil(rows / simdgroups_per_tg), ceil(ntok / TOK_TILE)), 128 threads.
//
// Three shapes share the row loop (mm_row):
//   matmul         out[t, row] (+)= W[row] . x[t]
//   matmul3        W is three matrices (q, k, v) with their own outputs; the row range selects the one
//   matmul_swiglu  out[t, r] = silu(W[r] . x[t]) * (W[hidden + r] . x[t])   (gate rows, then up rows)
// Each has a NORM variant for a single token (ntok == 1) that folds the preceding rmsnorm in: every
// threadgroup recomputes the row's scale with norm_scale_128, which reproduces the rmsnorm kernel's
// 256-thread reduction lane for lane, writes x[i] * scale * w[i] (the product rmsnorm would have stored)
// into dynamic threadgroup memory xn (cols floats, sized by the host), and runs the row loop on xn.
// One dispatch therefore replaces rmsnorm + matmul at decode.

struct MatmulParams {
    int rows;
    int cols;
    int ntok;
    int accumulate;
    float eps;      // NORM variants only
    float softcap;  // > 0: out = tanh(out / softcap) * softcap (Gemma's final-logit cap on the LM head), 0 = off
    int hasBias;    // 1: out[t, r] (+)= W[r] . x[t] + bias[r] (o_proj bias), bias bound at 5; v + b is one float
};

// Gemma 2 / 3 soft-capping; the argument of tanh is clamped because fast-math tanh overflows for large inputs
static inline float softcap_f(float v, float cap) {
    return tanh(clamp(v / cap, -TANH_CLAMP, TANH_CLAMP)) * cap;
}

struct Matmul3Params {
    int rows0;
    int rows1;
    int rows2;
    int cols;
    int ntok;
    float eps;
    int hasBias;    // 1: out_i[t, r] = W_i[r] . x[t] + b_i[r] (Qwen2 / GLM q/k/v bias), the b_i bound at 11..13
};

struct SwigluParams {
    int hidden;     // W has 2 * hidden rows: gate rows [0, hidden), up rows [hidden, 2 * hidden)
    int cols;
    int ntok;
    float eps;
    int gelu;       // 1 = Gemma's GeGLU: gelu_tanh(gate) * up instead of silu(gate) * up
};

// gelu_pytorch_tanh; the argument of tanh is clamped because fast-math tanh overflows to NaN for large inputs
static inline float gelu_tanh_f(float g) {
    const float a = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    return 0.5f * g * (1.0f + tanh(clamp(a, -TANH_CLAMP, TANH_CLAMP)));
}

// rmsnorm's scale for the row x[0..n), reduced exactly as the 256-thread rmsnorm kernel reduces it, on a
// 128-thread threadgroup (4 simdgroups): thread (sg, lane) plays rmsnorm's threads v = sg*32 + lane and
// v + 128, so the two simd_sums equal the sums of rmsnorm's simdgroups sg and sg + 4, and the second stage
// adds the same eight partials from the same lanes. Must be called by all 128 threads (barriers).
static inline float norm_scale_128(device const float *x, int n, float eps, threadgroup float *buf,
                                   ushort sg, ushort lane) {
    const int v = sg * 32 + lane;
    float p1 = 0.0f;
    float p2 = 0.0f;
    for (int i = v; i < n; i += ROW_THREADS) {
        float a = x[i];
        p1 += a * a;
    }
    for (int i = v + MM_THREADS; i < n; i += ROW_THREADS) {
        float a = x[i];
        p2 += a * a;
    }
    p1 = simd_sum(p1);
    p2 = simd_sum(p2);
    if (lane == 0) {
        buf[sg] = p1;
        buf[sg + MM_THREADS / 32] = p2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < ROW_THREADS / 32 ? buf[lane] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return rsqrt(r / n + eps);
}

// Normalised copy of x[0..n) in threadgroup memory, all 128 threads; ends with a barrier.
static inline void norm_row_128(device const float *x, device const float *w, int n, float eps,
                                threadgroup float *xn, threadgroup float *buf, ushort sg, ushort lane) {
    const float scale = norm_scale_128(x, n, eps, buf, sg, lane);
    for (int i = sg * 32 + lane; i < n; i += MM_THREADS) xn[i] = x[i] * scale * w[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline device const float4 *as_f4(device const float *p) { return (device const float4 *)p; }
static inline threadgroup const float4 *as_f4(threadgroup const float *p) { return (threadgroup const float4 *)p; }

// mm_row for QT == 2 (called from the generic mm_row, so every matmul / matmul3 / matmul_swiglu variant gets it):
// a lane's unit of work is one 16-weight sub-block (cols / 16 units per row, nb = cols / 32) so all 32 lanes
// stay busy on 1024-wide rows, which hold only 4 super-blocks.
template <typename XP>
static inline void mm_row_q2k(device const uchar *wq, ulong grow, int nb, XP x, ulong xstride, int nt,
                              thread float *acc, ushort lane) {
    const int nsb = (nb + 7) >> 3;
    const int nsub = nb * 2;
    for (int u = lane; u < nsub; u += 32) {
        float4 w[4];
        q2k_sub_block(wq, grow, nsb, u, w);
        for (int t = 0; t < TOK_TILE; t++) {
            if (t < nt) {
                auto x4 = as_f4(x + (ulong)t * xstride + u * 16);
                float d = 0.0f;
                for (int k = 0; k < 4; k++) d += dot(w[k], x4[k]);
                acc[t] += d;
            }
        }
    }
}

// mm_tile for QT == 2: the sub-blocks of the NR rows are dequantized once per lane unit and used for TT tokens,
// each row's unit walk and dot being those of mm_row_q2k.
template <int NR, int TT>
static inline void mm_tile_q2k(thread device const uchar *const *wq, thread const ulong *grow, int nr, int nb,
                               device const float *x, ulong xstride, int nt, thread float (*acc)[TT], ushort lane) {
    const int nsb = (nb + 7) >> 3;
    const int nsub = nb * 2;
    for (int u = lane; u < nsub; u += 32) {
        float4 w[NR][4];
        for (int r = 0; r < NR; r++) {
            if (r < nr) q2k_sub_block(wq[r], grow[r], nsb, u, w[r]);
        }
        for (int t = 0; t < TT; t++) {
            if (t < nt) {
                auto x4 = as_f4(x + (ulong)t * xstride + u * 16);
                float4 xv[4];
                for (int k = 0; k < 4; k++) xv[k] = x4[k];
                for (int r = 0; r < NR; r++) {
                    if (r < nr) {
                        float d = 0.0f;
                        for (int k = 0; k < 4; k++) d += dot(w[r][k], xv[k]);
                        acc[r][t] += d;
                    }
                }
            }
        }
    }
}

// Q6_K walks (defined in section 8 at the end of this file), declared here for the generic mm_row / mm_tile.
template <typename XP>
static inline void mm_row_q6k(device const uchar *wq, ulong grow, int nb, XP x, ulong xstride, int nt,
                              thread float *acc, ushort lane);
template <int NR, int TT>
static inline void mm_tile_q6k(thread device const uchar *const *wq, thread const ulong *grow, int nr, int nb,
                               device const float *x, ulong xstride, int nt, thread float (*acc)[TT], ushort lane);

// mm_row: acc[t] += W[grow] . x[t * xstride ..] for t < nt; blocks walked in lane order, each dequantized
// once for all tokens. XP is a device or threadgroup float pointer. mm_row_q2k is the Q2_K walk (16-weight
// sub-blocks as the lane unit so all 32 lanes stay busy on 1024-wide rows); mm_tile / mm_tile_q2k are the
// prefill tiles of the next section.


// The dot of one packed 32-weight block with 32 activations: the one expression behind every matmul kernel
// (mm_row below and the prefill mm_tile), so the compiler emits the same arithmetic for both families and
// their results are bit-identical. pk is the block's 16 (Q4_0: low nibbles = weights 0..15, high = 16..31)
// or 32 (Q8_0) bytes, x4 the 32 activations.
static inline float q4_block_dot(thread const uchar4 *pk, thread const float4 *x4) {
    float d = 0.0f;
    for (int k = 0; k < 4; k++) {
        d += dot(float4(pk[k] & uchar4(15)) - 8.0f, x4[k]) + dot(float4(pk[k] >> uchar4(4)) - 8.0f, x4[k + 4]);
    }
    return d;
}

static inline float q8_block_dot(thread const char4 *pk, thread const float4 *x4) {
    float d = 0.0f;
    for (int k = 0; k < 8; k++) d += dot(float4(pk[k]), x4[k]);
    return d;
}

// GGUF block types (QT >= GG_FIRST, the GGUF section at the end of the file)
#define GG_FIRST 20
template <int QT, typename XP>
static inline void mm_row_gguf(device const uchar *wq, ulong grow, int nb, XP x, ulong xstride, int nt,
                               thread float *acc, ushort lane);
template <int QT>
static inline void gg_unit(device const uchar *wq, ulong grow, int nb, int u, thread float4 *w);
static inline void q6k_sub_block(device const uchar *wq, ulong grow, int nsb, int u, thread float4 *w);
template <int QT, int NR, int TT>
static inline void mm_tile_gguf(thread device const uchar *const *wq, thread const ulong *grow, int nr, int nb,
                                device const float *x, ulong xstride, int nt, thread float (*acc)[TT], ushort lane);

template <int QT, typename XP>
static inline void mm_row(device const uchar *wq, device const float *ws, ulong grow, int nb,
                          XP x, ulong xstride, int nt, thread float *acc, ushort lane) {
    if (QT == 2) {
        mm_row_q2k(wq, grow, nb, x, xstride, nt, acc, lane);
        return;
    }
    if (QT == 6) {
        mm_row_q6k(wq, grow, nb, x, xstride, nt, acc, lane);
        return;
    }
    if (QT >= GG_FIRST) {
        mm_row_gguf<QT>(wq, grow, nb, x, xstride, nt, acc, lane);
        return;
    }
    device const float *rs = ws + grow * nb;
    if (QT == 4) {
        device const uchar4 *rq = (device const uchar4 *)(wq + grow * nb * 16);
        for (int b = lane; b < nb; b += 32) {
            uchar4 pk[4];
            for (int k = 0; k < 4; k++) pk[k] = rq[b * 4 + k];
            const float s = rs[b];
            for (int t = 0; t < TOK_TILE; t++) {
                if (t < nt) {
                    auto x4 = as_f4(x + (ulong)t * xstride + b * QK);
                    float4 xv[8];
                    for (int k = 0; k < 8; k++) xv[k] = x4[k];
                    acc[t] += q4_block_dot(pk, xv) * s;
                }
            }
        }
    } else {
        device const char4 *rq = (device const char4 *)(wq + grow * nb * 32);
        for (int b = lane; b < nb; b += 32) {
            char4 pk[8];
            for (int k = 0; k < 8; k++) pk[k] = rq[b * 8 + k];
            const float s = rs[b];
            for (int t = 0; t < TOK_TILE; t++) {
                if (t < nt) {
                    auto x4 = as_f4(x + (ulong)t * xstride + b * QK);
                    float4 xv[8];
                    for (int k = 0; k < 8; k++) xv[k] = x4[k];
                    acc[t] += q8_block_dot(pk, xv) * s;
                }
            }
        }
    }
}

template <int QT, bool NORM>
kernel void matmul(device const uchar *wq [[buffer(0)]],
                   device const float *ws [[buffer(1)]],
                   device const float *x [[buffer(2)]],
                   device float *out [[buffer(3)]],
                   device const float *nw [[buffer(4)]],
                   device const float *bias [[buffer(5)]],
                   constant MatmulParams &p [[buffer(6)]],
                   threadgroup float *xn [[threadgroup(0)]],
                   uint2 tgp [[threadgroup_position_in_grid]],
                   ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]],
                   ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[8];
    const int row = int(tgp.x) * nsg + sg;
    const int t0 = int(tgp.y) * TOK_TILE;
    const int nt = min(TOK_TILE, p.ntok - t0);
    const int nb = p.cols / QK;
    if (NORM) norm_row_128(x, nw, p.cols, p.eps, xn, buf, sg, lane); // all threads: barriers inside
    if (row >= p.rows) return;

    float acc[TOK_TILE];
    for (int t = 0; t < TOK_TILE; t++) acc[t] = 0.0f;
    if (NORM) {
        mm_row<QT>(wq, ws, (ulong)row, nb, (threadgroup const float *)xn, 0, 1, acc, lane);
    } else {
        mm_row<QT>(wq, ws, (ulong)row, nb, x + (ulong)t0 * p.cols, (ulong)p.cols, nt, acc, lane);
    }
    for (int t = 0; t < TOK_TILE; t++) {
        if (t < nt) {
            float v = simd_sum(acc[t]);
            if (lane == 0) {
                ulong o = (ulong)(t0 + t) * p.rows + row;
                if (p.hasBias) v += bias[row];
                out[o] = p.accumulate ? out[o] + v : (p.softcap > 0.0f ? softcap_f(v, p.softcap) : v);
            }
        }
    }
}

// q, k and v projections of the same input in one dispatch: rows [0, rows0) come from W0 into out0,
// [rows0, rows0 + rows1) from W1 into out1, the rest from W2 into out2. All three share cols and QT. With
// hasBias the per-row bias is added in the epilogue (out = v + b, the same float as a separate v += b pass).
// out2 has the KV cache's element type VT: with VT = half the value rows are rounded once, here, as they are
// written into the f16 cache (the k rows are rounded by rope<half, ..> after the rotation instead).
template <int QT, bool NORM, typename VT>
kernel void matmul3(device const uchar *wq0 [[buffer(0)]],
                    device const float *ws0 [[buffer(1)]],
                    device const uchar *wq1 [[buffer(2)]],
                    device const float *ws1 [[buffer(3)]],
                    device const uchar *wq2 [[buffer(4)]],
                    device const float *ws2 [[buffer(5)]],
                    device const float *x [[buffer(6)]],
                    device float *out0 [[buffer(7)]],
                    device float *out1 [[buffer(8)]],
                    device VT *out2 [[buffer(9)]],
                    device const float *nw [[buffer(10)]],
                    device const float *b0 [[buffer(11)]],
                    device const float *b1 [[buffer(12)]],
                    device const float *b2 [[buffer(13)]],
                    constant Matmul3Params &p [[buffer(14)]],
                    threadgroup float *xn [[threadgroup(0)]],
                    uint2 tgp [[threadgroup_position_in_grid]],
                    ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]],
                    ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[8];
    const int row = int(tgp.x) * nsg + sg;
    const int t0 = int(tgp.y) * TOK_TILE;
    const int nt = min(TOK_TILE, p.ntok - t0);
    const int nb = p.cols / QK;
    if (NORM) norm_row_128(x, nw, p.cols, p.eps, xn, buf, sg, lane);
    if (row >= p.rows0 + p.rows1 + p.rows2) return;

    device const uchar *wq;
    device const float *ws;
    device float *out;
    device const float *bias;
    int rows;
    int r;
    const bool isV = row >= p.rows0 + p.rows1;
    if (row < p.rows0) {
        wq = wq0; ws = ws0; out = out0; bias = b0; rows = p.rows0; r = row;
    } else if (!isV) {
        wq = wq1; ws = ws1; out = out1; bias = b1; rows = p.rows1; r = row - p.rows0;
    } else {
        wq = wq2; ws = ws2; out = out0; bias = b2; rows = p.rows2; r = row - p.rows0 - p.rows1; // out: see below
    }
    float acc[TOK_TILE];
    for (int t = 0; t < TOK_TILE; t++) acc[t] = 0.0f;
    if (NORM) {
        mm_row<QT>(wq, ws, (ulong)r, nb, (threadgroup const float *)xn, 0, 1, acc, lane);
    } else {
        mm_row<QT>(wq, ws, (ulong)r, nb, x + (ulong)t0 * p.cols, (ulong)p.cols, nt, acc, lane);
    }
    const float b = p.hasBias ? bias[r] : 0.0f;
    for (int t = 0; t < TOK_TILE; t++) {
        if (t < nt) {
            float v = simd_sum(acc[t]);
            if (lane == 0) {
                const float o = p.hasBias ? v + b : v;
                const ulong idx = (ulong)(t0 + t) * rows + r;
                if (isV) out2[idx] = VT(o); else out[idx] = o;
            }
        }
    }
}

// Dense FFN gate/up with SwiGLU in the epilogue: one simdgroup per hidden unit r computes the gate row r
// and the up row hidden + r (each exactly as matmul would) and writes silu(g) * u.
template <int QT, bool NORM>
kernel void matmul_swiglu(device const uchar *wq [[buffer(0)]],
                          device const float *ws [[buffer(1)]],
                          device const float *x [[buffer(2)]],
                          device float *out [[buffer(3)]],
                          device const float *nw [[buffer(4)]],
                          constant SwigluParams &p [[buffer(5)]],
                          threadgroup float *xn [[threadgroup(0)]],
                          uint2 tgp [[threadgroup_position_in_grid]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]],
                          ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[8];
    const int row = int(tgp.x) * nsg + sg;
    const int t0 = int(tgp.y) * TOK_TILE;
    const int nt = min(TOK_TILE, p.ntok - t0);
    const int nb = p.cols / QK;
    if (NORM) norm_row_128(x, nw, p.cols, p.eps, xn, buf, sg, lane);
    if (row >= p.hidden) return;

    float accg[TOK_TILE];
    float accu[TOK_TILE];
    for (int t = 0; t < TOK_TILE; t++) {
        accg[t] = 0.0f;
        accu[t] = 0.0f;
    }
    // GGUF types: the up rows are a separate view of the file, bound as buffer(1) (ws is unused by these types)
    device const uchar *wqU = QT >= GG_FIRST ? (device const uchar *)ws : wq;
    const ulong rowU = QT >= GG_FIRST ? (ulong)row : (ulong)(p.hidden + row);
    if (NORM) {
        mm_row<QT>(wq, ws, (ulong)row, nb, (threadgroup const float *)xn, 0, 1, accg, lane);
        mm_row<QT>(wqU, ws, rowU, nb, (threadgroup const float *)xn, 0, 1, accu, lane);
    } else {
        mm_row<QT>(wq, ws, (ulong)row, nb, x + (ulong)t0 * p.cols, (ulong)p.cols, nt, accg, lane);
        mm_row<QT>(wqU, ws, rowU, nb, x + (ulong)t0 * p.cols, (ulong)p.cols, nt, accu, lane);
    }
    for (int t = 0; t < TOK_TILE; t++) {
        if (t < nt) {
            float g = simd_sum(accg[t]);
            float u = simd_sum(accu[t]);
            if (lane == 0) out[(ulong)(t0 + t) * p.hidden + row] = (p.gelu ? gelu_tanh_f(g) : g / (1.0f + exp(-g))) * u;
        }
    }
}

typedef decltype(matmul<4, false>) matmul_t;
template [[host_name("matmul_q4")]] kernel matmul_t matmul<4, false>;
template [[host_name("matmul_q8")]] kernel matmul_t matmul<8, false>;
template [[host_name("matmul_q4_norm")]] kernel matmul_t matmul<4, true>;
template [[host_name("matmul_q8_norm")]] kernel matmul_t matmul<8, true>;
typedef decltype(matmul3<4, false, float>) matmul3_t;
typedef decltype(matmul3<4, false, half>) matmul3_h_t;   // *_h: v rows written as half (f16 KV cache)
template [[host_name("matmul3_q4")]] kernel matmul3_t matmul3<4, false, float>;
template [[host_name("matmul3_q8")]] kernel matmul3_t matmul3<8, false, float>;
template [[host_name("matmul3_q4_norm")]] kernel matmul3_t matmul3<4, true, float>;
template [[host_name("matmul3_q8_norm")]] kernel matmul3_t matmul3<8, true, float>;
template [[host_name("matmul3_q4_h")]] kernel matmul3_h_t matmul3<4, false, half>;
template [[host_name("matmul3_q8_h")]] kernel matmul3_h_t matmul3<8, false, half>;
template [[host_name("matmul3_q4_norm_h")]] kernel matmul3_h_t matmul3<4, true, half>;
template [[host_name("matmul3_q8_norm_h")]] kernel matmul3_h_t matmul3<8, true, half>;
typedef decltype(matmul_swiglu<4, false>) matmul_swiglu_t;
template [[host_name("matmul_swiglu_q4")]] kernel matmul_swiglu_t matmul_swiglu<4, false>;
template [[host_name("matmul_swiglu_q8")]] kernel matmul_swiglu_t matmul_swiglu<8, false>;
template [[host_name("matmul_swiglu_q4_norm")]] kernel matmul_swiglu_t matmul_swiglu<4, true>;
template [[host_name("matmul_swiglu_q8_norm")]] kernel matmul_swiglu_t matmul_swiglu<8, true>;

// ---------------------------------------------------------------------------------------------- 2. matmul: prefill tiles
// The same three products for a chunk of tokens, with one simdgroup computing a tile of NR rows x TT tokens
// (mm_tile): every x block is loaded once and used for NR rows, every weight block is unpacked once and used
// for TT tokens, instead of one row per simdgroup re-reading x for each of its TOK_TILE tokens. The single-row
// kernels gather x from device memory at a 128-byte lane stride, which is what limits them; the tile divides
// those gathers by NR. Every output element is still accumulated exactly as the single-row kernels do: the
// same blocks in the same lane order, the same block dot expression, one simd_sum per (row, token), so the
// results are bit-identical and the host may pick either family per dispatch (MetalBackend uses these when the
// chunk has at least PREFILL_MIN_TOKENS tokens). Rows of a tile may belong to different matrices (matmul3) or
// be the gate and up rows of the same unit (matmul_swiglu), hence per-row pointers. nr <= NR rows are valid.
// Grid: (ceil(rows / (NR * simdgroups)), ceil(ntok / TT)), 128 threads.
//
// Tile shapes, measured on an M1 Pro at the Qwen3-0.6B shapes with 64 tokens (ms per layer, single-row kernels
// 6.2-6.7): 2 x 8 is the best for matmul / matmul3 (4.0: two adjacent rows per simdgroup coalesce into 1 KB
// reads, four rows cost too many registers, 16 tokens gain nothing more), and one unit (gate + up = 2 rows)
// x 8 tokens for matmul_swiglu (1.5 vs 2.2; two units are slower, 1.8).

template <int QT, int NR, int TT>
static inline void mm_tile(thread device const uchar *const *wq, thread device const float *const *ws,
                           thread const ulong *grow, int nr, int nb, device const float *x, ulong xstride, int nt,
                           thread float (*acc)[TT], ushort lane) {
    if (QT == 2) {
        mm_tile_q2k<NR, TT>(wq, grow, nr, nb, x, xstride, nt, acc, lane);
        return;
    }
    if (QT == 6) {
        mm_tile_q6k<NR, TT>(wq, grow, nr, nb, x, xstride, nt, acc, lane);
        return;
    }
    if (QT >= GG_FIRST) {
        mm_tile_gguf<QT, NR, TT>(wq, grow, nr, nb, x, xstride, nt, acc, lane);
        return;
    }
    if (QT == 4) {
        for (int b = lane; b < nb; b += 32) {
            uchar4 pk[NR][4];
            float s[NR];
            for (int r = 0; r < NR; r++) {
                if (r < nr) {
                    device const uchar4 *rq = (device const uchar4 *)(wq[r] + grow[r] * nb * 16) + b * 4;
                    for (int k = 0; k < 4; k++) pk[r][k] = rq[k];
                    s[r] = ws[r][grow[r] * nb + b];
                }
            }
            for (int t = 0; t < TT; t++) {
                if (t < nt) {
                    auto x4 = as_f4(x + (ulong)t * xstride + b * QK);
                    float4 xv[8];
                    for (int k = 0; k < 8; k++) xv[k] = x4[k];
                    for (int r = 0; r < NR; r++) {
                        if (r < nr) acc[r][t] += q4_block_dot(pk[r], xv) * s[r];
                    }
                }
            }
        }
    } else {
        for (int b = lane; b < nb; b += 32) {
            char4 pk[NR][8];
            float s[NR];
            for (int r = 0; r < NR; r++) {
                if (r < nr) {
                    device const char4 *rq = (device const char4 *)(wq[r] + grow[r] * nb * 32) + b * 8;
                    for (int k = 0; k < 8; k++) pk[r][k] = rq[k];
                    s[r] = ws[r][grow[r] * nb + b];
                }
            }
            for (int t = 0; t < TT; t++) {
                if (t < nt) {
                    auto x4 = as_f4(x + (ulong)t * xstride + b * QK);
                    float4 xv[8];
                    for (int k = 0; k < 8; k++) xv[k] = x4[k];
                    for (int r = 0; r < NR; r++) {
                        if (r < nr) acc[r][t] += q8_block_dot(pk[r], xv) * s[r];
                    }
                }
            }
        }
    }
}

template <int QT, int NR, int TT>
kernel void matmul_pf(device const uchar *wq [[buffer(0)]],
                      device const float *ws [[buffer(1)]],
                      device const float *x [[buffer(2)]],
                      device float *out [[buffer(3)]],
                      device const float *nw [[buffer(4)]],
                      device const float *bias [[buffer(5)]],
                      constant MatmulParams &p [[buffer(6)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]],
                      ushort nsg [[simdgroups_per_threadgroup]]) {
    const int row0 = (int(tgp.x) * nsg + sg) * NR;
    const int t0 = int(tgp.y) * TT;
    const int nt = min(TT, p.ntok - t0);
    const int nb = p.cols / QK;
    if (row0 >= p.rows) return;
    const int nr = min(NR, p.rows - row0);
    device const uchar *wqs[NR];
    device const float *wss[NR];
    ulong grow[NR];
    for (int r = 0; r < NR; r++) {
        wqs[r] = wq;
        wss[r] = ws;
        grow[r] = (ulong)(row0 + min(r, nr - 1));
    }
    float acc[NR][TT];
    for (int r = 0; r < NR; r++)
        for (int t = 0; t < TT; t++) acc[r][t] = 0.0f;
    mm_tile<QT, NR, TT>(wqs, wss, grow, nr, nb, x + (ulong)t0 * p.cols, (ulong)p.cols, nt, acc, lane);
    for (int r = 0; r < NR; r++) {
        if (r < nr) {
            for (int t = 0; t < TT; t++) {
                if (t < nt) {
                    float v = simd_sum(acc[r][t]);
                    if (lane == 0) {
                        ulong o = (ulong)(t0 + t) * p.rows + row0 + r;
                        if (p.hasBias) v += bias[row0 + r];
                        out[o] = p.accumulate ? out[o] + v : (p.softcap > 0.0f ? softcap_f(v, p.softcap) : v);
                    }
                }
            }
        }
    }
}

template <int QT, int NR, int TT, typename VT>
kernel void matmul3_pf(device const uchar *wq0 [[buffer(0)]],
                       device const float *ws0 [[buffer(1)]],
                       device const uchar *wq1 [[buffer(2)]],
                       device const float *ws1 [[buffer(3)]],
                       device const uchar *wq2 [[buffer(4)]],
                       device const float *ws2 [[buffer(5)]],
                       device const float *x [[buffer(6)]],
                       device float *out0 [[buffer(7)]],
                       device float *out1 [[buffer(8)]],
                       device VT *out2 [[buffer(9)]],
                       device const float *nw [[buffer(10)]],
                       device const float *b0 [[buffer(11)]],
                       device const float *b1 [[buffer(12)]],
                       device const float *b2 [[buffer(13)]],
                       constant Matmul3Params &p [[buffer(14)]],
                       uint2 tgp [[threadgroup_position_in_grid]],
                       ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]],
                       ushort nsg [[simdgroups_per_threadgroup]]) {
    const int total = p.rows0 + p.rows1 + p.rows2;
    const int row0 = (int(tgp.x) * nsg + sg) * NR;
    const int t0 = int(tgp.y) * TT;
    const int nt = min(TT, p.ntok - t0);
    const int nb = p.cols / QK;
    if (row0 >= total) return;
    const int nr = min(NR, total - row0);
    device const uchar *wqs[NR];
    device const float *wss[NR];
    device float *outs[NR];
    bool isV[NR];
    float bias[NR];
    int rows[NR];
    ulong grow[NR];
    for (int r = 0; r < NR; r++) {
        const int row = row0 + min(r, nr - 1);
        isV[r] = row >= p.rows0 + p.rows1;
        if (row < p.rows0) {
            wqs[r] = wq0; wss[r] = ws0; outs[r] = out0; rows[r] = p.rows0; grow[r] = (ulong)row;
            bias[r] = p.hasBias ? b0[grow[r]] : 0.0f;
        } else if (!isV[r]) {
            wqs[r] = wq1; wss[r] = ws1; outs[r] = out1; rows[r] = p.rows1; grow[r] = (ulong)(row - p.rows0);
            bias[r] = p.hasBias ? b1[grow[r]] : 0.0f;
        } else {
            wqs[r] = wq2; wss[r] = ws2; outs[r] = out0; rows[r] = p.rows2; grow[r] = (ulong)(row - p.rows0 - p.rows1);
            bias[r] = p.hasBias ? b2[grow[r]] : 0.0f; // written to out2 below
        }
    }
    float acc[NR][TT];
    for (int r = 0; r < NR; r++)
        for (int t = 0; t < TT; t++) acc[r][t] = 0.0f;
    mm_tile<QT, NR, TT>(wqs, wss, grow, nr, nb, x + (ulong)t0 * p.cols, (ulong)p.cols, nt, acc, lane);
    for (int r = 0; r < NR; r++) {
        if (r < nr) {
            for (int t = 0; t < TT; t++) {
                if (t < nt) {
                    float v = simd_sum(acc[r][t]);
                    if (lane == 0) {
                        const float o = p.hasBias ? v + bias[r] : v;
                        const ulong idx = (ulong)(t0 + t) * rows[r] + grow[r];
                        if (isV[r]) out2[idx] = VT(o); else outs[r][idx] = o;
                    }
                }
            }
        }
    }
}

// NR units per simdgroup: rows [unit0, unit0 + nr) are the gate rows, [hidden + unit0, ...) the up rows.
template <int QT, int NR, int TT>
kernel void matmul_swiglu_pf(device const uchar *wq [[buffer(0)]],
                             device const float *ws [[buffer(1)]],
                             device const float *x [[buffer(2)]],
                             device float *out [[buffer(3)]],
                             device const float *nw [[buffer(4)]],
                             constant SwigluParams &p [[buffer(5)]],
                             uint2 tgp [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]],
                             ushort nsg [[simdgroups_per_threadgroup]]) {
    const int unit0 = (int(tgp.x) * nsg + sg) * NR;
    const int t0 = int(tgp.y) * TT;
    const int nt = min(TT, p.ntok - t0);
    const int nb = p.cols / QK;
    if (unit0 >= p.hidden) return;
    const int nu = min(NR, p.hidden - unit0);
    device const uchar *wqs[2 * NR];
    device const float *wss[2 * NR];
    ulong grow[2 * NR];
    for (int r = 0; r < NR; r++) {
        const int unit = unit0 + min(r, nu - 1);
        wqs[r] = wq; wss[r] = ws; grow[r] = (ulong)unit;
        if (QT >= GG_FIRST) {   // GGUF types: the up rows are the separate view bound as buffer(1)
            wqs[NR + r] = (device const uchar *)ws; wss[NR + r] = ws; grow[NR + r] = (ulong)unit;
        } else {
            wqs[NR + r] = wq; wss[NR + r] = ws; grow[NR + r] = (ulong)(p.hidden + unit);
        }
    }
    float acc[2 * NR][TT];
    for (int r = 0; r < 2 * NR; r++)
        for (int t = 0; t < TT; t++) acc[r][t] = 0.0f;
    // nr = 2 * NR: the padded rows (unit clamped to the last valid one) are computed and discarded
    mm_tile<QT, 2 * NR, TT>(wqs, wss, grow, 2 * NR, nb, x + (ulong)t0 * p.cols, (ulong)p.cols, nt, acc, lane);
    for (int r = 0; r < NR; r++) {
        if (r < nu) {
            for (int t = 0; t < TT; t++) {
                if (t < nt) {
                    float g = simd_sum(acc[r][t]);
                    float u = simd_sum(acc[NR + r][t]);
                    if (lane == 0) {
                        out[(ulong)(t0 + t) * p.hidden + unit0 + r] = (p.gelu ? gelu_tanh_f(g) : g / (1.0f + exp(-g))) * u;
                    }
                }
            }
        }
    }
}

typedef decltype(matmul_pf<4, PF_ROWS, PF_TOKS>) matmul_pf_t;
template [[host_name("matmul_pf_q4")]] kernel matmul_pf_t matmul_pf<4, PF_ROWS, PF_TOKS>;
template [[host_name("matmul_pf_q8")]] kernel matmul_pf_t matmul_pf<8, PF_ROWS, PF_TOKS>;
typedef decltype(matmul3_pf<4, PF_ROWS, PF_TOKS, float>) matmul3_pf_t;
typedef decltype(matmul3_pf<4, PF_ROWS, PF_TOKS, half>) matmul3_pf_h_t;
template [[host_name("matmul3_pf_q4")]] kernel matmul3_pf_t matmul3_pf<4, PF_ROWS, PF_TOKS, float>;
template [[host_name("matmul3_pf_q8")]] kernel matmul3_pf_t matmul3_pf<8, PF_ROWS, PF_TOKS, float>;
template [[host_name("matmul3_pf_q4_h")]] kernel matmul3_pf_h_t matmul3_pf<4, PF_ROWS, PF_TOKS, half>;
template [[host_name("matmul3_pf_q8_h")]] kernel matmul3_pf_h_t matmul3_pf<8, PF_ROWS, PF_TOKS, half>;
typedef decltype(matmul_swiglu_pf<4, PF_UNITS, PF_TOKS>) matmul_swiglu_pf_t;
template [[host_name("matmul_swiglu_pf_q4")]] kernel matmul_swiglu_pf_t matmul_swiglu_pf<4, PF_UNITS, PF_TOKS>;
template [[host_name("matmul_swiglu_pf_q8")]] kernel matmul_swiglu_pf_t matmul_swiglu_pf<8, PF_UNITS, PF_TOKS>;

// ---------------------------------------------------------------------------------------------- 2b. matmul: MMA prefill
// Prefill chunks of at least MMA_MIN_TOKENS tokens use the matrix units: one threadgroup (4 simdgroups) computes a
// MMA_BR rows x MMA_BT tokens output tile with 8x8 simdgroup_float8x8 multiply-accumulates. Per 32-wide k step
// (one quantization block) the threads dequantize the tile's MMA_BR weight rows (value * scale, f32) into
// threadgroup memory sa[row][k] and stage MMA_BT activation rows into sb[token][k]; each simdgroup owns a
// 32 row x 16 token quarter (4 x 2 accumulators) and runs four k8 steps. C[token][row] = sum_k sb[token][k] *
// sa[row][k], so the weight operand is loaded transposed. The tile lands in threadgroup memory sc[token][row]
// (aliasing sa) and the epilogue of each shape writes it out with the guards. The summation order differs from
// the scalar kernels (results agree to f32 rounding, not bit for bit).

#define MMA_SA_FLOATS (MMA_BR * 32)
#define MMA_SB_FLOATS (MMA_BT * 32)

// mma_core_rows: the core with each thread's activation row given (xrow = the row of tile token tid >> 2), so a
// caller can gather the tile's tokens (the MoE kernels take the slots of one expert).
template <int QT>
static inline void mma_core_rows(device const uchar *rwq, device const float *rws, ulong rgrow, int nb,
                                 device const float *xrow,
                                 threadgroup float *sa, threadgroup float *sb, threadgroup float *sc,
                                 ushort tid, ushort sg) {
    simdgroup_float8x8 acc[8];
    for (int i = 0; i < 8; i++) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const int lr = tid >> 1;
    const int h = tid & 1;
    const int lt = tid >> 2;
    const int lp = tid & 3;
    device const float4 *xr = (device const float4 *)xrow + lp * 2;
    const int nsb = (nb + 7) >> 3;
    const ushort sr = sg & 1;
    const ushort st = sg >> 1;
    threadgroup float *wd = sa + lr * 32;
    for (int b = 0; b < nb; b++) {
        if (QT == 4) {
            device const uchar4 *q = (device const uchar4 *)(rwq + rgrow * nb * 16 + b * 16 + h * 8);
            const float s = rws[rgrow * nb + b];
            for (int k = 0; k < 2; k++) {
                uchar4 v = q[k];
                *(threadgroup float4 *)(wd + h * 8 + k * 4) = (float4(v & uchar4(15)) - 8.0f) * s;
                *(threadgroup float4 *)(wd + 16 + h * 8 + k * 4) = (float4(v >> uchar4(4)) - 8.0f) * s;
            }
        } else if (QT == 8) {
            device const char4 *q = (device const char4 *)(rwq + rgrow * nb * 32 + b * 32 + h * 16);
            const float s = rws[rgrow * nb + b];
            for (int k = 0; k < 4; k++) *(threadgroup float4 *)(wd + h * 16 + k * 4) = float4(q[k]) * s;
        } else if (QT >= GG_FIRST) {
            // GGUF types: the 32-weight unit b of the row, both threads of the row decode it and store their half
            float4 w[8];
            gg_unit<QT>(rwq, rgrow, nb, b, w);
            for (int k = 0; k < 4; k++) *(threadgroup float4 *)(wd + h * 16 + k * 4) = w[h * 4 + k];
        } else if (QT == 6) {
            float4 w[4];
            q6k_sub_block(rwq, rgrow, nsb, 2 * b + h, w);
            for (int k = 0; k < 4; k++) *(threadgroup float4 *)(wd + h * 16 + k * 4) = w[k];
        } else {
            float4 w[4];
            q2k_sub_block(rwq, rgrow, nsb, 2 * b + h, w);
            for (int k = 0; k < 4; k++) *(threadgroup float4 *)(wd + h * 16 + k * 4) = w[k];
        }
        threadgroup float4 *xd = (threadgroup float4 *)(sb + lt * 32 + lp * 8);
        xd[0] = xr[b * 8];
        xd[1] = xr[b * 8 + 1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; kk++) {
            simdgroup_float8x8 a[2];
            simdgroup_float8x8 w[4];
            for (int i = 0; i < 2; i++) simdgroup_load(a[i], sb + (st * 16 + i * 8) * 32 + kk * 8, 32);
            for (int j = 0; j < 4; j++) simdgroup_load(w[j], sa + (sr * 32 + j * 8) * 32 + kk * 8, 32, ulong2(0, 0), true);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i * 4 + j], a[i], w[j], acc[i * 4 + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
            simdgroup_store(acc[i * 4 + j], sc + (st * 16 + i * 8) * MMA_BR + sr * 32 + j * 8, MMA_BR);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}


template <int QT>
static inline void mma_core(device const uchar *rwq, device const float *rws, ulong rgrow, int nb,
                            device const float *x, int cols, int ntRem,
                            threadgroup float *sa, threadgroup float *sb, threadgroup float *sc,
                            ushort tid, ushort sg) {
    mma_core_rows<QT>(rwq, rws, rgrow, nb, x + (ulong)min(int(tid >> 2), ntRem - 1) * cols, sa, sb, sc, tid, sg);
}

template <int QT>
kernel void matmul_mma(device const uchar *wq [[buffer(0)]],
                       device const float *ws [[buffer(1)]],
                       device const float *x [[buffer(2)]],
                       device float *out [[buffer(3)]],
                       device const float *nw [[buffer(4)]],
                       device const float *bias [[buffer(5)]],
                       constant MatmulParams &p [[buffer(6)]],
                       uint2 tgp [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[MMA_SA_FLOATS];
    threadgroup float sb[MMA_SB_FLOATS];
    const int row0 = int(tgp.x) * MMA_BR;
    const int t0 = int(tgp.y) * MMA_BT;
    const int nb = p.cols / QK;
    const ulong grow = (ulong)min(row0 + (tid >> 1), p.rows - 1);
    mma_core<QT>(wq, ws, grow, nb, x + (ulong)t0 * p.cols, p.cols, p.ntok - t0, sa, sb, sa, tid, sg);
    for (int i = tid; i < MMA_BR * MMA_BT; i += MM_THREADS) {
        const int t = i / MMA_BR, r = i % MMA_BR;
        if (t0 + t < p.ntok && row0 + r < p.rows) {
            const float v = p.hasBias ? sa[t * MMA_BR + r] + bias[row0 + r] : sa[t * MMA_BR + r];
            const ulong o = (ulong)(t0 + t) * p.rows + row0 + r;
            out[o] = p.accumulate ? out[o] + v : (p.softcap > 0.0f ? softcap_f(v, p.softcap) : v);
        }
    }
}

template <int QT, typename VT>
kernel void matmul3_mma(device const uchar *wq0 [[buffer(0)]],
                        device const float *ws0 [[buffer(1)]],
                        device const uchar *wq1 [[buffer(2)]],
                        device const float *ws1 [[buffer(3)]],
                        device const uchar *wq2 [[buffer(4)]],
                        device const float *ws2 [[buffer(5)]],
                        device const float *x [[buffer(6)]],
                        device float *out0 [[buffer(7)]],
                        device float *out1 [[buffer(8)]],
                        device VT *out2 [[buffer(9)]],
                        device const float *nw [[buffer(10)]],
                        device const float *b0 [[buffer(11)]],
                        device const float *b1 [[buffer(12)]],
                        device const float *b2 [[buffer(13)]],
                        constant Matmul3Params &p [[buffer(14)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[MMA_SA_FLOATS];
    threadgroup float sb[MMA_SB_FLOATS];
    const int total = p.rows0 + p.rows1 + p.rows2;
    const int row0 = int(tgp.x) * MMA_BR;
    const int t0 = int(tgp.y) * MMA_BT;
    const int nb = p.cols / QK;
    const int lrow = min(row0 + (tid >> 1), total - 1);
    device const uchar *rwq;
    device const float *rws;
    ulong grow;
    if (lrow < p.rows0) { rwq = wq0; rws = ws0; grow = (ulong)lrow; }
    else if (lrow < p.rows0 + p.rows1) { rwq = wq1; rws = ws1; grow = (ulong)(lrow - p.rows0); }
    else { rwq = wq2; rws = ws2; grow = (ulong)(lrow - p.rows0 - p.rows1); }
    mma_core<QT>(rwq, rws, grow, nb, x + (ulong)t0 * p.cols, p.cols, p.ntok - t0, sa, sb, sa, tid, sg);
    for (int i = tid; i < MMA_BR * MMA_BT; i += MM_THREADS) {
        const int t = i / MMA_BR, r = i % MMA_BR;
        const int row = row0 + r;
        if (t0 + t < p.ntok && row < total) {
            float v = sa[t * MMA_BR + r];
            if (row < p.rows0) {
                if (p.hasBias) v += b0[row];
                out0[(ulong)(t0 + t) * p.rows0 + row] = v;
            } else if (row < p.rows0 + p.rows1) {
                const int g = row - p.rows0;
                if (p.hasBias) v += b1[g];
                out1[(ulong)(t0 + t) * p.rows1 + g] = v;
            } else {
                const int g = row - p.rows0 - p.rows1;
                if (p.hasBias) v += b2[g];
                out2[(ulong)(t0 + t) * p.rows2 + g] = VT(v);
            }
        }
    }
}

// Tile rows [0, 32) = gate units [unit0, unit0 + 32), rows [32, 64) = the matching up rows.
template <int QT>
kernel void matmul_swiglu_mma(device const uchar *wq [[buffer(0)]],
                              device const float *ws [[buffer(1)]],
                              device const float *x [[buffer(2)]],
                              device float *out [[buffer(3)]],
                              device const float *nw [[buffer(4)]],
                              constant SwigluParams &p [[buffer(5)]],
                              uint2 tgp [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[MMA_SA_FLOATS];
    threadgroup float sb[MMA_SB_FLOATS];
    const int unit0 = int(tgp.x) * (MMA_BR / 2);
    const int t0 = int(tgp.y) * MMA_BT;
    const int nb = p.cols / QK;
    const int lr = tid >> 1;
    const int unit = min(unit0 + (lr & 31), p.hidden - 1);
    // GGUF types: the up rows are the separate view bound as buffer(1), the same unit index in it
    device const uchar *rwq = (QT >= GG_FIRST && lr >= 32) ? (device const uchar *)ws : wq;
    const ulong grow = (ulong)((lr < 32 || QT >= GG_FIRST) ? unit : p.hidden + unit);
    mma_core<QT>(rwq, ws, grow, nb, x + (ulong)t0 * p.cols, p.cols, p.ntok - t0, sa, sb, sa, tid, sg);
    for (int i = tid; i < (MMA_BR / 2) * MMA_BT; i += MM_THREADS) {
        const int t = i / (MMA_BR / 2), j = i % (MMA_BR / 2);
        if (t0 + t < p.ntok && unit0 + j < p.hidden) {
            const float g = sa[t * MMA_BR + j];
            const float u = sa[t * MMA_BR + 32 + j];
            out[(ulong)(t0 + t) * p.hidden + unit0 + j] = (p.gelu ? gelu_tanh_f(g) : g / (1.0f + exp(-g))) * u;
        }
    }
}

typedef decltype(matmul_mma<4>) matmul_mma_t;
typedef decltype(matmul3_mma<4, float>) matmul3_mma_t;
typedef decltype(matmul3_mma<4, half>) matmul3_mma_h_t;
typedef decltype(matmul_swiglu_mma<4>) matmul_swiglu_mma_t;
template [[host_name("matmul_mma_q4")]] kernel matmul_mma_t matmul_mma<4>;
template [[host_name("matmul_mma_q8")]] kernel matmul_mma_t matmul_mma<8>;
template [[host_name("matmul_mma_q2k")]] kernel matmul_mma_t matmul_mma<2>;
template [[host_name("matmul3_mma_q4")]] kernel matmul3_mma_t matmul3_mma<4, float>;
template [[host_name("matmul3_mma_q8")]] kernel matmul3_mma_t matmul3_mma<8, float>;
template [[host_name("matmul3_mma_q2k")]] kernel matmul3_mma_t matmul3_mma<2, float>;
template [[host_name("matmul3_mma_q4_h")]] kernel matmul3_mma_h_t matmul3_mma<4, half>;
template [[host_name("matmul3_mma_q8_h")]] kernel matmul3_mma_h_t matmul3_mma<8, half>;
template [[host_name("matmul3_mma_q2k_h")]] kernel matmul3_mma_h_t matmul3_mma<2, half>;
template [[host_name("matmul_swiglu_mma_q4")]] kernel matmul_swiglu_mma_t matmul_swiglu_mma<4>;
template [[host_name("matmul_swiglu_mma_q8")]] kernel matmul_swiglu_mma_t matmul_swiglu_mma<8>;
template [[host_name("matmul_swiglu_mma_q2k")]] kernel matmul_swiglu_mma_t matmul_swiglu_mma<2>;

template [[host_name("matmul_q2k")]] kernel matmul_t matmul<2, false>;
template [[host_name("matmul_q2k_norm")]] kernel matmul_t matmul<2, true>;
template [[host_name("matmul_pf_q2k")]] kernel matmul_pf_t matmul_pf<2, PF_ROWS, PF_TOKS>;
template [[host_name("matmul3_pf_q2k")]] kernel matmul3_pf_t matmul3_pf<2, PF_ROWS, PF_TOKS, float>;
template [[host_name("matmul3_pf_q2k_h")]] kernel matmul3_pf_h_t matmul3_pf<2, PF_ROWS, PF_TOKS, half>;
template [[host_name("matmul_swiglu_pf_q2k")]] kernel matmul_swiglu_pf_t matmul_swiglu_pf<2, PF_UNITS, PF_TOKS>;
template [[host_name("matmul3_q2k")]] kernel matmul3_t matmul3<2, false, float>;
template [[host_name("matmul3_q2k_norm")]] kernel matmul3_t matmul3<2, true, float>;
template [[host_name("matmul3_q2k_h")]] kernel matmul3_h_t matmul3<2, false, half>;
template [[host_name("matmul3_q2k_norm_h")]] kernel matmul3_h_t matmul3<2, true, half>;
template [[host_name("matmul_swiglu_q2k")]] kernel matmul_swiglu_t matmul_swiglu<2, false>;
template [[host_name("matmul_swiglu_q2k_norm")]] kernel matmul_swiglu_t matmul_swiglu<2, true>;

// ---------------------------------------------------------------------------------------------- 3. norm + rope
// One threadgroup per row of length n.

struct NormParams {
    int n;
    float eps;
};

kernel void rmsnorm(device const float *x [[buffer(0)]],
                    device const float *w [[buffer(1)]],
                    device float *out [[buffer(2)]],
                    constant NormParams &p [[buffer(3)]],
                    uint tg [[threadgroup_position_in_grid]],
                    ushort tid [[thread_position_in_threadgroup]],
                    ushort tpg [[threads_per_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]],
                    ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[32];
    device const float *xv = x + (ulong)tg * p.n;
    device float *ov = out + (ulong)tg * p.n;
    float ss = 0.0f;
    for (int i = tid; i < p.n; i += tpg) {
        float v = xv[i];
        ss += v * v;
    }
    ss = tg_sum(ss, buf, sg, lane, nsg);
    const float scale = rsqrt(ss / p.n + p.eps);
    for (int i = tid; i < p.n; i += tpg) ov[i] = xv[i] * scale * w[i];
}

// rmsnorm_add: out[row] += rmsnorm(y[row]) * w, for Glm4's post-attention / post-MLP norms applied to the
// sub-layer output before the residual add. One threadgroup per row of length n.
kernel void rmsnorm_add(device const float *y [[buffer(0)]],
                        device const float *w [[buffer(1)]],
                        device float *out [[buffer(2)]],
                        constant NormParams &p [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]],
                        ushort tid [[thread_position_in_threadgroup]],
                        ushort tpg [[threads_per_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]],
                        ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[32];
    device const float *yv = y + (ulong)tg * p.n;
    device float *ov = out + (ulong)tg * p.n;
    float ss = 0.0f;
    for (int i = tid; i < p.n; i += tpg) {
        float v = yv[i];
        ss += v * v;
    }
    ss = tg_sum(ss, buf, sg, lane, nsg);
    const float scale = rsqrt(ss / p.n + p.eps);
    for (int i = tid; i < p.n; i += tpg) ov[i] += yv[i] * scale * w[i];
}


// rope<KV, ADJACENT>: rotary embedding (optionally preceded by the per-head qk-norm) for the q rows
// ([ntok, qStride], nHeads heads) and the k rows of the same tokens in one dispatch. Grid: (nHeads + k heads,
// ntok); threadgroup x < nHeads handles q head x in place, otherwise k head x - nHeads: read from `k` (the
// f32 cache rows at pos, or the f32 staging rows of an f16 cache) and written to the cache rows `kOut` with
// the cache element type KV, which for half is where the f16 cache rounds k, once, after the norm and the
// rotation (round-to-nearest-even, Java's Float.floatToFloat16); for float the host binds kOut = k.
// Threads per threadgroup = rotHalf = rotated dims / 2 = table entries per position; thread i owns one pair:
//   ADJACENT = false  (i, i + rotHalf) inside the rotated slice   (rotate-half: Llama, Qwen, Gemma, Phi partial)
//   ADJACENT = true   (2i, 2i + 1)                                 (interleaved: GLM, DeepSeek MLA)
// The rotated slice starts at rotOff within the head (0 except DeepSeek's q, where it follows the nope part);
// dims outside the slice are copied to kOut unchanged so a half cache receives the whole row. The qk-norm
// (useNorm) covers the whole head and so needs a full rotation (rotHalf = headDim / 2); the host rejects
// partial rotation with a norm. DeepSeek rotates q and its shared k_pe row with the q path (no k heads).

struct RopeParams {
    int headDim;     // head stride within a row (MLA q: nope + rope)
    int qStride;
    int kvStride;
    int nHeads;      // q heads; threadgroups beyond them are k heads
    int pos0;
    float eps;
    int useNorm;
    int rotOff;      // first rotated dim within the head
    int rotHalf;     // rotated dims / 2
};

template <typename KV, bool ADJACENT>
kernel void rope(device float *q [[buffer(0)]],
                 device const float *k [[buffer(1)]],
                 device const float *qw [[buffer(2)]],
                 device const float *kw [[buffer(3)]],
                 device const float *cosT [[buffer(4)]],
                 device const float *sinT [[buffer(5)]],
                 device KV *kOut [[buffer(6)]],
                 constant RopeParams &p [[buffer(7)]],
                 uint2 tgp [[threadgroup_position_in_grid]],
                 ushort2 tid2 [[thread_position_in_threadgroup]],
                 ushort sg [[simdgroup_index_in_threadgroup]],
                 ushort lane [[thread_index_in_simdgroup]],
                 ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[32];
    const ushort tid = tid2.x;
    const bool isQ = int(tgp.x) < p.nHeads;
    const int head = isQ ? int(tgp.x) : int(tgp.x) - p.nHeads;
    const ulong off = isQ ? (ulong)tgp.y * p.qStride + (ulong)head * p.headDim
                          : (ulong)tgp.y * p.kvStride + (ulong)head * p.headDim;
    device const float *h = (isQ ? q : k) + off;
    device const float *w = isQ ? qw : kw;
    const int i0 = p.rotOff + (ADJACENT ? 2 * tid : tid);
    const int i1 = p.rotOff + (ADJACENT ? 2 * tid + 1 : tid + p.rotHalf);
    float a = h[i0];
    float b = h[i1];
    if (p.useNorm) {
        float ss = tg_sum(a * a + b * b, buf, sg, lane, nsg);
        float scale = rsqrt(ss / p.headDim + p.eps);
        a *= scale * w[i0];
        b *= scale * w[i1];
    }
    const int pos = p.pos0 + int(tgp.y);
    const float c = cosT[pos * p.rotHalf + tid];
    const float s = sinT[pos * p.rotHalf + tid];
    const float ra = a * c - b * s;
    const float rb = b * c + a * s;
    if (isQ) {
        q[off + i0] = ra;
        q[off + i1] = rb;
    } else {
        kOut[off + i0] = KV(ra);
        kOut[off + i1] = KV(rb);
        for (int i = tid; i < p.headDim; i += p.rotHalf) {
            if (i < p.rotOff || i >= p.rotOff + 2 * p.rotHalf) kOut[off + i] = KV(h[i]);
        }
    }
}

template [[host_name("rope")]] kernel decltype(rope<float, false>) rope<float, false>;
template [[host_name("rope_h")]] kernel decltype(rope<half, false>) rope<half, false>;
template [[host_name("rope_il")]] kernel decltype(rope<float, true>) rope<float, true>;
template [[host_name("rope_il_h")]] kernel decltype(rope<half, true>) rope<half, true>;

// ---------------------------------------------------------------------------------------------- 4. attention
// Causal attention for the tokens t = tokenOffset + tgp.y of a tile at positions pos0 + t, 256 threads =
// 8 simdgroups per threadgroup. attention_t<KV, G, PERK, PERV> is the one kernel behind every variant:
//   KV          cache element type (float, half for backend.kv-dtype=f16, char for q8: int8 values with a scale per
//               (position, kv head) row, see kv_scale; converted on load, all arithmetic f32)
//   G           query heads of one KV head per threadgroup (grid (nHeads / G, tile)); G > 1 shares each K/V row
//               across the group and is opt-in (-Djllm.metal.attnGroup, JLLM_ATTN_GROUPS)
//   PERK, PERV  key / value head width in units of 32 (compile-time), or 0 for the widths in the params:
//               used for head dims other than 64 / 128 / 256 and for DeepSeek MLA (qkDim = kv_lora_rank + rope,
//               vDim = kv_lora_rank; the value is the leading part of the key row, see section 7)
//   MAXP        bound of the runtime widths in units of 32 (MAX_PER; MLA_MAX_PER for the attention_mla pair)
// Tokens are dispatched in tiles so the scores scratch only needs tile * nHeads * ctx floats (indexed by the
// tile-local token tgp.y), and the scores live in device memory so the context is not limited by threadgroup
// memory. Each lane owns dims lane, lane + 32, ... of q, k and v. Per head every value is produced the same
// way whatever the template arguments: scores by simdgroup-strided positions with one simd_sum each, max and
// sum by the two-level reductions, the value accumulation simdgroup-strided then reduced through `part`.

struct AttnParams {
    int nHeads;
    int nKvHeads;
    int qkDim;       // query / key head width (headDim; MLA: nope + rope)
    int vDim;        // value head width (headDim; MLA: vHeadDim)
    int pos0;
    int ctx;
    float scale;
    int qStride;     // q row stride (nHeads * qkDim)
    int kStride;     // key cache row stride in elements (nKvHeads * qkDim; q8: + 4 * nKvHeads scale bytes)
    int vStride;     // value cache row stride in elements (as kStride)
    int oStride;     // output row stride (nHeads * vDim)
    int window;      // sliding-window length in tokens, 0 = full causal attention
    int tokenOffset; // first token of this tile; the scores scratch is indexed by the tile-local token
    float softcap;   // Gemma 2: score = tanh(score / softcap) * softcap before the softmax, 0 = off
};

// The q8 cache (backend.kv-dtype=q8, KV = char): a position's row of the cache is its kvDim int8 values followed by
// one f32 scale per kv head (stride = kvDim + 4 * nKvHeads bytes; DeepSeek's latent row: aw values, one scale), so
// value i of kv head h at position s is row[h * headDim + i] * scale[h]. kv_scale(row, stride, nKvHeads, h) is that
// scale for a q8 row and 1 for float / half rows (the product then compiles to the plain f32 / f16 arithmetic): the
// kernels multiply a score (q . k_int) or a probability by it once per row instead of every element.
static inline float kv_scale(device const float *, int, int, int) { return 1.0f; }
static inline float kv_scale(device const half *, int, int, int) { return 1.0f; }
static inline float kv_scale(device const char *row, int stride, int nKvHeads, int kvh) {
    return ((device const float *)(row + stride - 4 * nKvHeads))[kvh];
}

template <typename KV, int G, int PERK, int PERV, int MAXP = MAX_PER>
kernel void attention_t(device const float *q [[buffer(0)]],
                        device const KV *kc [[buffer(1)]],
                        device const KV *vc [[buffer(2)]],
                        device float *out [[buffer(3)]],
                        device float *scores [[buffer(4)]],
                        constant AttnParams &p [[buffer(5)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        ushort2 tid2 [[thread_position_in_threadgroup]],
                        ushort2 tpg2 [[threads_per_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]],
                        ushort nsg [[simdgroups_per_threadgroup]]) {
    constexpr int MK = PERK > 0 ? PERK : MAXP;       // loop bounds
    constexpr int MV = PERV > 0 ? PERV : MAXP;
    threadgroup float red[G * ATT_SIMDGROUPS];  // per-head cross-simdgroup reduction slots
    threadgroup float part[ATT_SIMDGROUPS][MAXP * 32]; // = ATT_THREADS for MAX_PER
    const ushort tid = tid2.x;
    const ushort tpg = tpg2.x;
    const int perK = PERK > 0 ? PERK : p.qkDim / 32;
    const int perV = PERV > 0 ? PERV : p.vDim / 32;
    const int qkDim = PERK > 0 ? PERK * 32 : p.qkDim;
    const int vDim = PERV > 0 ? PERV * 32 : p.vDim;

    const int h0 = int(tgp.x) * G;
    const int tl = int(tgp.y);
    const int t = p.tokenOffset + tl;
    const int kvh = h0 / (p.nHeads / p.nKvHeads);
    const int len = p.pos0 + t + 1;
    const int first = p.window > 0 ? max(0, len - p.window) : 0;
    device const KV *kb = kc + kvh * qkDim;
    device const KV *vb = vc + kvh * vDim;
    device float *sc[G];
    for (int g = 0; g < G; g++) sc[g] = scores + ((ulong)tl * p.nHeads + h0 + g) * p.ctx;

    float qr[G][MK];
    for (int g = 0; g < G; g++) {
        device const float *qv = q + (ulong)t * p.qStride + (h0 + g) * qkDim;
        for (int i = 0; i < MK; i++) qr[g][i] = i < perK ? qv[lane + i * 32] : 0.0f;
    }

    // 1. scores: each K row is read once per threadgroup and dotted with the G query heads
    float mx[G];
    for (int g = 0; g < G; g++) mx[g] = -INFINITY;
    for (int s = first + sg; s < len; s += nsg) {
        device const KV *kv = kb + (ulong)s * p.kStride;
        const float ks = p.scale * kv_scale(kc + (ulong)s * p.kStride, p.kStride, p.nKvHeads, kvh);
        float kr[MK];
        for (int i = 0; i < MK; i++) kr[i] = i < perK ? float(kv[lane + i * 32]) : 0.0f;
        for (int g = 0; g < G; g++) {
            float d = 0.0f;
            for (int i = 0; i < MK; i++) if (i < perK) d += qr[g][i] * kr[i];
            d = simd_sum(d) * ks;
            if (p.softcap > 0.0f) d = softcap_f(d, p.softcap);
            if (lane == 0) sc[g][s] = d;
            mx[g] = max(mx[g], d);
        }
    }
    for (int g = 0; g < G; g++) {
        float v = simd_max(mx[g]);
        if (lane == 0) red[g * 8 + sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = 0; g < G; g++) {
        float r = lane < nsg ? red[g * 8 + lane] : -INFINITY;
        mx[g] = simd_max(r);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // 2. softmax numerators
    float sum[G];
    for (int g = 0; g < G; g++) sum[g] = 0.0f;
    for (int s = first + tid; s < len; s += tpg) {
        for (int g = 0; g < G; g++) {
            float e = exp(sc[g][s] - mx[g]);
            sc[g][s] = e;
            sum[g] += e;
        }
    }
    for (int g = 0; g < G; g++) {
        float v = simd_sum(sum[g]);
        if (lane == 0) red[g * 8 + sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = 0; g < G; g++) {
        float r = lane < nsg ? red[g * 8 + lane] : 0.0f;
        sum[g] = simd_sum(r);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // 3. weighted sum of values: each V row is read once per threadgroup, split over simdgroups then reduced
    float o[G][MV];
    for (int g = 0; g < G; g++)
        for (int i = 0; i < MV; i++) o[g][i] = 0.0f;
    for (int s = first + sg; s < len; s += nsg) {
        device const KV *vv = vb + (ulong)s * p.vStride;
        const float vs = kv_scale(vc + (ulong)s * p.vStride, p.vStride, p.nKvHeads, kvh);
        float vr[MV];
        for (int i = 0; i < MV; i++) vr[i] = i < perV ? float(vv[lane + i * 32]) : 0.0f;
        for (int g = 0; g < G; g++) {
            const float pr = sc[g][s] * vs;
            for (int i = 0; i < MV; i++) if (i < perV) o[g][i] += pr * vr[i];
        }
    }
    for (int g = 0; g < G; g++) {
        for (int i = 0; i < MV; i++) if (i < perV) part[sg][lane + i * 32] = o[g][i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = 1.0f / sum[g];
        for (int d = tid; d < vDim; d += tpg) {
            float acc = 0.0f;
            for (int k = 0; k < nsg; k++) acc += part[k][d];
            out[(ulong)t * p.oStride + (h0 + g) * vDim + d] = acc * inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// attention / attention_h / attention_q8: runtime widths (head dims other than 64 / 128 / 256);
// attention_mla[_h|_q8]: runtime widths up to MLA_MAX_PER * 32 (DeepSeek's absorbed queries);
// attention_g{G}_d{headDim}[_h|_q8]: compile-time widths, G query heads per threadgroup.
#define ATTN_INST(KV, G, PER, NAME) \
    template [[host_name(NAME)]] kernel decltype(attention_t<KV, G, PER, PER>) attention_t<KV, G, PER, PER>;
ATTN_INST(float, 1, 0, "attention")           ATTN_INST(half, 1, 0, "attention_h")          ATTN_INST(char, 1, 0, "attention_q8")
template [[host_name("attention_mla")]] kernel decltype(attention_t<float, 1, 0, 0, MLA_MAX_PER>) attention_t<float, 1, 0, 0, MLA_MAX_PER>;
template [[host_name("attention_mla_h")]] kernel decltype(attention_t<half, 1, 0, 0, MLA_MAX_PER>) attention_t<half, 1, 0, 0, MLA_MAX_PER>;
template [[host_name("attention_mla_q8")]] kernel decltype(attention_t<char, 1, 0, 0, MLA_MAX_PER>) attention_t<char, 1, 0, 0, MLA_MAX_PER>;
ATTN_INST(float, 1, 2, "attention_g1_d64")    ATTN_INST(half, 1, 2, "attention_g1_d64_h")   ATTN_INST(char, 1, 2, "attention_g1_d64_q8")
ATTN_INST(float, 1, 4, "attention_g1_d128")   ATTN_INST(half, 1, 4, "attention_g1_d128_h")  ATTN_INST(char, 1, 4, "attention_g1_d128_q8")
ATTN_INST(float, 1, 8, "attention_g1_d256")   ATTN_INST(half, 1, 8, "attention_g1_d256_h")  ATTN_INST(char, 1, 8, "attention_g1_d256_q8")
// The grouped variants are only instantiated when the host defines JLLM_ATTN_GROUPS (-Djllm.metal.attnGroup).
#ifdef JLLM_ATTN_GROUPS
ATTN_INST(float, 2, 2, "attention_g2_d64")    ATTN_INST(float, 4, 2, "attention_g4_d64")    ATTN_INST(float, 8, 2, "attention_g8_d64")
ATTN_INST(float, 2, 4, "attention_g2_d128")   ATTN_INST(float, 4, 4, "attention_g4_d128")   ATTN_INST(float, 8, 4, "attention_g8_d128")
ATTN_INST(float, 2, 8, "attention_g2_d256")   ATTN_INST(float, 4, 8, "attention_g4_d256")   ATTN_INST(float, 8, 8, "attention_g8_d256")
ATTN_INST(half, 2, 2, "attention_g2_d64_h")   ATTN_INST(half, 4, 2, "attention_g4_d64_h")   ATTN_INST(half, 8, 2, "attention_g8_d64_h")
ATTN_INST(half, 2, 4, "attention_g2_d128_h")  ATTN_INST(half, 4, 4, "attention_g4_d128_h")  ATTN_INST(half, 8, 4, "attention_g8_d128_h")
ATTN_INST(half, 2, 8, "attention_g2_d256_h")  ATTN_INST(half, 4, 8, "attention_g4_d256_h")  ATTN_INST(half, 8, 8, "attention_g8_d256_h")
ATTN_INST(char, 2, 2, "attention_g2_d64_q8")  ATTN_INST(char, 4, 2, "attention_g4_d64_q8")  ATTN_INST(char, 8, 2, "attention_g8_d64_q8")
ATTN_INST(char, 2, 4, "attention_g2_d128_q8") ATTN_INST(char, 4, 4, "attention_g4_d128_q8") ATTN_INST(char, 8, 4, "attention_g8_d128_q8")
ATTN_INST(char, 2, 8, "attention_g2_d256_q8") ATTN_INST(char, 4, 8, "attention_g4_d256_q8") ATTN_INST(char, 8, 8, "attention_g8_d256_q8")
#endif

// ---------------------------------------------------------------------------------------------- 4. attention: flash decoding
// Optional (-Djllm.metal.flashDecode=true): a decode step (one token) over a long context splits the key range of
// every head across nSplit threadgroups instead of one. attention_split<KV> runs the three phases of `attention`
// on its sub-range [first + split * chunk, ...) with the softmax taken relative to the sub-range's own maximum,
// and writes the unnormalised partial (o[headDim], max, sum) to `part`; attention_combine then rescales the
// partials to the global maximum and divides by the combined sum. The combine adds the sub-ranges' sums and
// value accumulations in a different order from the single-pass kernel, so the results are not bit-identical to
// it (max |delta logit| ~1e-6 on Qwen3-0.6B at 3 500 tokens, same greedy tokens), hence the opt-in.
// part layout: [(h * nSplit + split) * (headDim + 2)]: o[0..headDim), max, sum. Grid: (nHeads, nSplit), 256 threads.

struct AttnSplitParams {
    int nHeads;
    int nKvHeads;
    int headDim;
    int pos0;
    int ctx;
    float scale;
    int qStride;
    int kvStride;    // cache row stride in elements (q8: with the scales, see kv_scale)
    int window;
    float softcap;
    int nSplit;
    int chunk;       // keys per split; <= 0: ceil(keys / nSplit), computed by the kernel
};

template <typename KV>
kernel void attention_split(device const float *q [[buffer(0)]],
                            device const KV *kc [[buffer(1)]],
                            device const KV *vc [[buffer(2)]],
                            device float *part [[buffer(3)]],
                            device float *scores [[buffer(4)]],
                            constant AttnSplitParams &p [[buffer(5)]],
                            uint2 tgp [[threadgroup_position_in_grid]],
                            ushort2 tid2 [[thread_position_in_threadgroup]],
                            ushort2 tpg2 [[threads_per_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]],
                            ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[32];
    threadgroup float red[ATT_SIMDGROUPS][ATT_THREADS];
    const ushort tid = tid2.x;
    const ushort tpg = tpg2.x;

    const int h = int(tgp.x);
    const int split = int(tgp.y);
    const int kvh = h / (p.nHeads / p.nKvHeads);
    const int len = p.pos0 + 1;
    const int first = p.window > 0 ? max(0, len - p.window) : 0;
    // chunk <= 0: ceil(keys / nSplit), derived here so that a recorded decode step only patches the position
    const int chunk = p.chunk > 0 ? p.chunk : (len - first + p.nSplit - 1) / p.nSplit;
    const int s0 = first + split * chunk;
    const int s1 = min(len, s0 + chunk);
    const int per = p.headDim / 32;
    device const float *qv = q + h * p.headDim;
    device float *sc = scores + (ulong)h * p.ctx;
    device const KV *kb = kc + kvh * p.headDim;
    device const KV *vb = vc + kvh * p.headDim;
    device float *po = part + ((ulong)h * p.nSplit + split) * (p.headDim + 2);

    float qr[MAX_PER];
    for (int i = 0; i < MAX_PER; i++) qr[i] = i < per ? qv[lane + i * 32] : 0.0f;

    float mx = -INFINITY;
    for (int s = s0 + sg; s < s1; s += nsg) {
        device const KV *kv = kb + (ulong)s * p.kvStride;
        float d = 0.0f;
        for (int i = 0; i < MAX_PER; i++) if (i < per) d += qr[i] * float(kv[lane + i * 32]);
        d = simd_sum(d) * (p.scale * kv_scale(kc + (ulong)s * p.kvStride, p.kvStride, p.nKvHeads, kvh));
        if (p.softcap > 0.0f) d = softcap_f(d, p.softcap);
        if (lane == 0) sc[s] = d;
        mx = max(mx, d);
    }
    mx = tg_max(mx, buf, sg, lane, nsg);
    threadgroup_barrier(mem_flags::mem_device);
    if (s0 >= s1) { // empty split (fewer keys than nSplit * chunk): contributes nothing
        for (int d = tid; d < p.headDim; d += tpg) po[d] = 0.0f;
        if (tid == 0) { po[p.headDim] = -INFINITY; po[p.headDim + 1] = 0.0f; }
        return;
    }

    float sum = 0.0f;
    for (int s = s0 + tid; s < s1; s += tpg) {
        float e = exp(sc[s] - mx);
        sc[s] = e;
        sum += e;
    }
    sum = tg_sum(sum, buf, sg, lane, nsg);
    threadgroup_barrier(mem_flags::mem_device);

    float o[MAX_PER];
    for (int i = 0; i < MAX_PER; i++) o[i] = 0.0f;
    for (int s = s0 + sg; s < s1; s += nsg) {
        const float pr = sc[s] * kv_scale(vc + (ulong)s * p.kvStride, p.kvStride, p.nKvHeads, kvh);
        device const KV *vv = vb + (ulong)s * p.kvStride;
        for (int i = 0; i < MAX_PER; i++) if (i < per) o[i] += pr * float(vv[lane + i * 32]);
    }
    for (int i = 0; i < MAX_PER; i++) if (i < per) red[sg][lane + i * 32] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = tid; d < p.headDim; d += tpg) {
        float acc = 0.0f;
        for (int g = 0; g < nsg; g++) acc += red[g][d];
        po[d] = acc;
    }
    if (tid == 0) {
        po[p.headDim] = mx;
        po[p.headDim + 1] = sum;
    }
}

struct AttnCombineParams {
    int nHeads;
    int headDim;
    int nSplit;      // <= FLASH_MAX_SPLITS
};

// out[h, d] = sum_s o_s[d] * exp(m_s - M) / sum_s l_s * exp(m_s - M), M = max_s m_s. Grid: (nHeads), 256 threads.
kernel void attention_combine(device const float *part [[buffer(0)]],
                              device float *out [[buffer(1)]],
                              constant AttnCombineParams &p [[buffer(2)]],
                              uint tg [[threadgroup_position_in_grid]],
                              ushort tid [[thread_position_in_threadgroup]],
                              ushort tpg [[threads_per_threadgroup]]) {
    threadgroup float w[FLASH_MAX_SPLITS];
    threadgroup float inv[1];
    const int h = int(tg);
    const int stride = p.headDim + 2;
    device const float *ph = part + (ulong)h * p.nSplit * stride;
    if (tid == 0) {
        float m = -INFINITY;
        for (int s = 0; s < p.nSplit; s++) m = max(m, ph[s * stride + p.headDim]);
        float l = 0.0f;
        for (int s = 0; s < p.nSplit; s++) {
            const float ws = exp(ph[s * stride + p.headDim] - m);
            w[s] = ws;
            l += ws * ph[s * stride + p.headDim + 1];
        }
        inv[0] = 1.0f / l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = tid; d < p.headDim; d += tpg) {
        float acc = 0.0f;
        for (int s = 0; s < p.nSplit; s++) acc += w[s] * ph[s * stride + d];
        out[(ulong)h * p.headDim + d] = acc * inv[0];
    }
}

template [[host_name("attention_split")]] kernel decltype(attention_split<float>) attention_split<float>;
template [[host_name("attention_split_h")]] kernel decltype(attention_split<half>) attention_split<half>;
template [[host_name("attention_split_q8")]] kernel decltype(attention_split<char>) attention_split<char>;

// Optional half-precision cache (backend.kv-dtype=f16). v is rounded once by the projection's epilogue
// (matmul3<.., half> writes half(v + bias) straight into the cache row); k is projected into an f32 staging
// row, normalised and rotated there, and rounded once by rope<half, ..> when it writes the cache row
// (round-to-nearest-even, the same rounding as Java's Float.floatToFloat16). The *_h attention instantiations
// read half K/V with the same f32 arithmetic (attention_t<half, ...> above).
//
// Optional int8 cache (backend.kv-dtype=q8): the quantization needs the whole row of a kv head (its largest
// magnitude), so both k and v go through f32 staging rows (matmul3<.., float> writes them, rope<float, ..> rotates k
// in place) and kv_store_q8 then quantizes every (token, kv head) row into the cache: scale = max|x| / 127,
// q = rint(x / scale) clamped to [-127, 127] (Java: KvDtype.q8), the int8 values at row[h * headDim ..] and the
// scale after the row's kvDim values (see kv_scale). The *_q8 attention instantiations read char * scale.

struct KvStoreParams {
    int headDim;
    int kvDim;       // values per position (nKvHeads * headDim)
    int nKvHeads;
    int stride;      // bytes per cache position row: kvDim + 4 * nKvHeads
};

// Grid: (2 * nKvHeads, ntok), one simdgroup (32 threads) per row; x < nKvHeads quantizes k head x, the others v
// head x - nKvHeads. k / v are the f32 staging rows [ntok, kvDim]; kc / vc are bound at the row of the chunk's
// first position.
kernel void kv_store_q8(device const float *k [[buffer(0)]],
                        device const float *v [[buffer(1)]],
                        device char *kc [[buffer(2)]],
                        device char *vc [[buffer(3)]],
                        constant KvStoreParams &p [[buffer(4)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    const bool isV = int(tgp.x) >= p.nKvHeads;
    const int h = isV ? int(tgp.x) - p.nKvHeads : int(tgp.x);
    device const float *x = (isV ? v : k) + (ulong)tgp.y * p.kvDim + h * p.headDim;
    device char *row = (isV ? vc : kc) + (ulong)tgp.y * p.stride;
    float mx = 0.0f;
    for (int i = lane; i < p.headDim; i += 32) mx = max(mx, fabs(x[i]));
    const float s = simd_max(mx) / 127.0f;
    for (int i = lane; i < p.headDim; i += 32) {
        row[h * p.headDim + i] = s > 0.0f ? char(clamp(rint(x[i] / s), -127.0f, 127.0f)) : char(0);
    }
    if (lane == 0) ((device float *)(row + p.kvDim))[h] = s;
}

// rope_store_q8<ADJACENT>: rope<float, ADJACENT> and kv_store_q8 in one dispatch (the decode path of the q8 cache).
// Grid: (nHeads + 2 * nKvHeads, ntok), rotHalf threads per threadgroup. Threadgroups x < nHeads are rope's q heads,
// unchanged; x < nHeads + nKvHeads takes k head x - nHeads from the f32 staging rows `k`, applies rope's qk-norm and
// rotation, keeps the head's values in threadgroup memory and quantizes them straight into the cache rows `kc`; the
// remaining nKvHeads threadgroups quantize v head x - nHeads - nKvHeads of the staging rows `v` into `vc`. The
// quantization is kv_store_q8's (scale = max|x| / 127 over the head, q = rint(x / scale)); the maximum is taken by a
// threadgroup reduction instead of one simdgroup's, which gives the same number, so the cache bytes are the ones
// rope + kv_store_q8 write. kc / vc are bound at the row of the chunk's first position, like kv_store_q8's.

struct RopeQ8Params {
    int headDim;
    int qStride;
    int kvStride;    // floats per staging row (kvDim)
    int nHeads;
    int pos0;
    float eps;
    int useNorm;
    int rotOff;
    int rotHalf;
    int nKvHeads;
    int stride;      // bytes per cache position row: kvDim + 4 * nKvHeads
};

static inline char q8_value(float x, float s) {
    return s > 0.0f ? char(clamp(rint(x / s), -127.0f, 127.0f)) : char(0);
}

template <bool ADJACENT>
kernel void rope_store_q8(device float *q [[buffer(0)]],
                          device const float *k [[buffer(1)]],
                          device const float *qw [[buffer(2)]],
                          device const float *kw [[buffer(3)]],
                          device const float *cosT [[buffer(4)]],
                          device const float *sinT [[buffer(5)]],
                          device char *kc [[buffer(6)]],
                          device const float *v [[buffer(7)]],
                          device char *vc [[buffer(8)]],
                          constant RopeQ8Params &p [[buffer(9)]],
                          uint2 tgp [[threadgroup_position_in_grid]],
                          ushort2 tid2 [[thread_position_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]],
                          ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[32];
    threadgroup float hv[MAX_PER * 32];   // the k head after norm and rotation
    const ushort tid = tid2.x;
    const int kvDim = p.nKvHeads * p.headDim;
    if (int(tgp.x) >= p.nHeads + p.nKvHeads) { // a v head: quantized as it is
        const int h = int(tgp.x) - p.nHeads - p.nKvHeads;
        device const float *x = v + (ulong)tgp.y * p.kvStride + h * p.headDim;
        float mx = 0.0f;
        for (int i = tid; i < p.headDim; i += p.rotHalf) mx = max(mx, fabs(x[i]));
        const float s = tg_max(mx, buf, sg, lane, nsg) / 127.0f;
        device char *row = vc + (ulong)tgp.y * p.stride;
        for (int i = tid; i < p.headDim; i += p.rotHalf) row[h * p.headDim + i] = q8_value(x[i], s);
        if (tid == 0) ((device float *)(row + kvDim))[h] = s;
        return;
    }
    const bool isQ = int(tgp.x) < p.nHeads;
    const int head = isQ ? int(tgp.x) : int(tgp.x) - p.nHeads;
    const ulong off = isQ ? (ulong)tgp.y * p.qStride + (ulong)head * p.headDim
                          : (ulong)tgp.y * p.kvStride + (ulong)head * p.headDim;
    device const float *h = (isQ ? q : k) + off;
    device const float *w = isQ ? qw : kw;
    const int i0 = p.rotOff + (ADJACENT ? 2 * tid : tid);
    const int i1 = p.rotOff + (ADJACENT ? 2 * tid + 1 : tid + p.rotHalf);
    float a = h[i0];
    float b = h[i1];
    if (p.useNorm) {
        float ss = tg_sum(a * a + b * b, buf, sg, lane, nsg);
        float scale = rsqrt(ss / p.headDim + p.eps);
        a *= scale * w[i0];
        b *= scale * w[i1];
    }
    const int pos = p.pos0 + int(tgp.y);
    const float c = cosT[pos * p.rotHalf + tid];
    const float s = sinT[pos * p.rotHalf + tid];
    const float ra = a * c - b * s;
    const float rb = b * c + a * s;
    if (isQ) {
        q[off + i0] = ra;
        q[off + i1] = rb;
        return;
    }
    hv[i0] = ra;
    hv[i1] = rb;
    float mx = max(fabs(ra), fabs(rb));
    for (int i = tid; i < p.headDim; i += p.rotHalf) {
        if (i < p.rotOff || i >= p.rotOff + 2 * p.rotHalf) {
            hv[i] = h[i];
            mx = max(mx, fabs(h[i]));
        }
    }
    const float sc = tg_max(mx, buf, sg, lane, nsg) / 127.0f; // its barriers also publish hv
    device char *row = kc + (ulong)tgp.y * p.stride;
    for (int i = tid; i < p.headDim; i += p.rotHalf) row[head * p.headDim + i] = q8_value(hv[i], sc);
    if (tid == 0) ((device float *)(row + kvDim))[head] = sc;
}

template [[host_name("rope_store_q8")]] kernel decltype(rope_store_q8<false>) rope_store_q8<false>;
template [[host_name("rope_il_store_q8")]] kernel decltype(rope_store_q8<true>) rope_store_q8<true>;

// kv_put<KV>: batched decoding across slots (MetalBackend.forwardBatch). The rows of a batched step belong to different
// conversations, each with its own cache and position, so the projection writes every row's v to an f32 staging row
// and this kernel copies one row into its conversation's cache row (bound at that position): unchanged for the f32
// cache, rounded once to half for the f16 cache (the same round-to-nearest-even half(v + bias) as matmul3<.., half>
// applies in a single-sequence step). The q8 cache takes kv_store_q8 instead. Grid: (kvDim / CHANNEL_THREADS, ntok).

struct KvPutParams {
    int kvDim;       // values per row
    int stride;      // destination row stride in elements
};

template <typename KV>
kernel void kv_put(device const float *src [[buffer(0)]],
                   device KV *dst [[buffer(1)]],
                   constant KvPutParams &p [[buffer(2)]],
                   uint2 tgp [[threadgroup_position_in_grid]],
                   ushort2 tid2 [[thread_position_in_threadgroup]],
                   ushort2 tpg2 [[threads_per_threadgroup]]) {
    const int i = int(tgp.x) * int(tpg2.x) + int(tid2.x);
    if (i < p.kvDim) dst[(ulong)tgp.y * p.stride + i] = KV(src[(ulong)tgp.y * p.kvDim + i]);
}

template [[host_name("kv_put")]] kernel decltype(kv_put<float>) kv_put<float>;
template [[host_name("kv_put_h")]] kernel decltype(kv_put<half>) kv_put<half>;

// ---------------------------------------------------------------------------------------------- 5. mixture of experts
// router: one threadgroup (256 threads = 8 simdgroups) per token. Simdgroup sg computes the logits of experts
// sg, sg + 8, ...: the 32 lanes stride over dim in float4 steps (each iteration reads 512 contiguous bytes of
// the router row) and finish with simd_sum. The softmax max and sum are threadgroup reductions; the top-k
// selection is serial on thread 0 (nExperts <= 1024). Writes assign[t*topK+k] and routeWeight[t*topK+k].

struct RouterParams {
    int nExperts;
    int dim;
    int topK;
    int normalize;
    int remap;       // also write gassign and count the picks (experts on the CPU, see remap_slot)
};

// When some of a layer's experts run on the CPU (backend.metal.cpu-moe), the router also writes, for each slot, the
// expert's index among the layer's packed GPU experts (map[e], -1 for an expert on the CPU: the MoE kernels skip
// those slots) and counts the pick in counts[e], the usage the next placement is planned from. One relaxed atomic add
// per pick (thread 0 of each token's threadgroup), so the counts are exact.
static inline void remap_slot(device const int *map, device int *gassign, device atomic_uint *counts, int slot, int e) {
    gassign[slot] = map[e];
    atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
}

// the norm the MoE layer applies before its router, fused into the *_norm routers: xn row = rmsnorm(x row) * nw, the
// arithmetic of the rmsnorm kernel (the same threads, reduction and expression), then published to the threadgroup
static inline void norm_row_into(device const float *xv, device const float *nw, device float *ov, int n, float eps,
                                 ushort tid, ushort tpg, threadgroup float *buf, ushort sg, ushort lane, ushort nsg) {
    float ss = 0.0f;
    for (int i = tid; i < n; i += tpg) {
        float v = xv[i];
        ss += v * v;
    }
    ss = tg_sum(ss, buf, sg, lane, nsg);
    const float scale = rsqrt(ss / n + eps);
    for (int i = tid; i < n; i += tpg) ov[i] = xv[i] * scale * nw[i];
    threadgroup_barrier(mem_flags::mem_device);
}

// The router's work for token tg, whose (normalised) row is xv. P: RouterParams or RouterNormParams.
template <typename P>
static inline void router_row(device const float *xv, device const float *w, device int *assign, device float *rw,
                              device const int *map, device int *gassign, device atomic_uint *counts, constant P &p,
                              uint tg, ushort tid, ushort tpg, ushort sg, ushort lane, ushort nsg,
                              threadgroup float *logits, threadgroup float *buf) {
    device const float4 *x4 = (device const float4 *)xv;
    const int n4 = p.dim / 4;
    for (int e = sg; e < p.nExperts; e += nsg) {
        device const float *wv = w + (ulong)e * p.dim;
        device const float4 *w4 = (device const float4 *)wv;
        float d = 0.0f;
        for (int i = lane; i < n4; i += 32) d += dot(w4[i], x4[i]);
        for (int i = n4 * 4 + lane; i < p.dim; i += 32) d += wv[i] * xv[i]; // dim % 4 tail
        d = simd_sum(d);
        if (lane == 0) logits[e] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY;
    for (int e = tid; e < p.nExperts; e += tpg) mx = max(mx, logits[e]);
    mx = tg_max(mx, buf, sg, lane, nsg);
    float sum = 0.0f;
    for (int e = tid; e < p.nExperts; e += tpg) {
        const float v = exp(logits[e] - mx);
        logits[e] = v;
        sum += v;
    }
    sum = tg_sum(sum, buf, sg, lane, nsg); // its barriers also publish the exp'd logits to thread 0
    if (tid == 0) {
        float selected = 0.0f;
        for (int k = 0; k < p.topK; k++) {
            int best = -1;
            float bv = -INFINITY;
            for (int e = 0; e < p.nExperts; e++) {
                if (logits[e] > bv) { bv = logits[e]; best = e; }
            }
            assign[tg * p.topK + k] = best;
            if (p.remap) remap_slot(map, gassign, counts, tg * p.topK + k, best);
            rw[tg * p.topK + k] = bv / sum;
            selected += bv / sum;
            logits[best] = -INFINITY;
        }
        if (p.normalize && selected > 0.0f) {
            for (int k = 0; k < p.topK; k++) rw[tg * p.topK + k] /= selected;
        }
    }
}

kernel void router(device const float *x [[buffer(0)]],
                   device const float *w [[buffer(1)]],
                   device int *assign [[buffer(2)]],
                   device float *rw [[buffer(3)]],
                   device const int *map [[buffer(4)]],
                   device int *gassign [[buffer(5)]],
                   device atomic_uint *counts [[buffer(6)]],
                   constant RouterParams &p [[buffer(7)]],
                   uint tg [[threadgroup_position_in_grid]],
                   ushort tid [[thread_position_in_threadgroup]],
                   ushort tpg [[threads_per_threadgroup]],
                   ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]],
                   ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float logits[MAX_EXPERTS];
    threadgroup float buf[32];
    router_row(x + (ulong)tg * p.dim, w, assign, rw, map, gassign, counts, p, tg, tid, tpg, sg, lane, nsg, logits, buf);
}

// router_norm: the MoE layer's rmsnorm and its router in one dispatch (one threadgroup per token owns the token's row):
// xn = rmsnorm(x) * nw, written for the experts that follow, then the router on it. Bit-identical to rmsnorm + router.
struct RouterNormParams {
    int nExperts;
    int dim;
    int topK;
    int normalize;
    int remap;
    float eps;
};

kernel void router_norm(device const float *x [[buffer(0)]],
                        device const float *w [[buffer(1)]],
                        device int *assign [[buffer(2)]],
                        device float *rw [[buffer(3)]],
                        device const int *map [[buffer(4)]],
                        device int *gassign [[buffer(5)]],
                        device atomic_uint *counts [[buffer(6)]],
                        device const float *nw [[buffer(7)]],
                        device float *xn [[buffer(8)]],
                        constant RouterNormParams &p [[buffer(9)]],
                        uint tg [[threadgroup_position_in_grid]],
                        ushort tid [[thread_position_in_threadgroup]],
                        ushort tpg [[threads_per_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]],
                        ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float logits[MAX_EXPERTS];
    threadgroup float buf[32];
    device float *row = xn + (ulong)tg * p.dim;
    norm_row_into(x + (ulong)tg * p.dim, nw, row, p.dim, p.eps, tid, tpg, buf, sg, lane, nsg);
    router_row(row, w, assign, rw, map, gassign, counts, p, tg, tid, tpg, sg, lane, nsg, logits, buf);
}

// moe_up: h[s, row] = silu(Wgate_e[row] . x[t]) * (Wup_e[row] . x[t])   for slot s = token*topK + k,
// e = assign[s], t = s / topK. The stacked expert tensor holds gate rows [e*2m, e*2m + m) and up rows
// [e*2m + m, (e+1)*2m), so rowsPerExpert = 2m and one simdgroup computes rows `row` and `row + m` of the
// same expert and applies SwiGLU in the epilogue (no separate silu_mul pass, no gate/up scratch).
// Each row still accumulates its own blocks in lane order and is reduced with one simd_sum, so the
// per-row dot products are the ones the unfused kernel produced.
// Grid: (ceil(m / simdgroups), ntok * topK).

struct MoeParams {
    int rowsPerExpert;
    int cols;
    int topK;
};

template <int QT>
kernel void moe_up(device const uchar *wq [[buffer(0)]],
                   device const float *ws [[buffer(1)]],
                   device const float *x [[buffer(2)]],
                   device const int *assign [[buffer(3)]],
                   device float *out [[buffer(4)]],
                   constant MoeParams &p [[buffer(5)]],
                   uint2 tgp [[threadgroup_position_in_grid]],
                   ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]],
                   ushort nsg [[simdgroups_per_threadgroup]]) {
    const int m = p.rowsPerExpert / 2;
    const int row = int(tgp.x) * nsg + sg;
    if (row >= m) return;
    const int s = int(tgp.y);
    if (assign[s] < 0) return; // an expert left on the CPU (remap_slot)
    const int t = s / p.topK;
    // GGUF types: gate and up experts are two [experts * m, cols] views (wq, and buffer(1) = ws), same row in both
    const ulong growG = QT >= GG_FIRST ? (ulong)assign[s] * m + row : (ulong)assign[s] * p.rowsPerExpert + row;
    const ulong growU = QT >= GG_FIRST ? growG : growG + m;
    device const uchar *wqU = QT >= GG_FIRST ? (device const uchar *)ws : wq;
    const int nb = p.cols / QK;
    device const float4 *x4 = (device const float4 *)(x + (ulong)t * p.cols);
    float g = 0.0f;
    float u = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        g += block_dot<QT>(wq, ws, growG, nb, b, x4 + b * 8);
        u += block_dot<QT>(wqU, ws, growU, nb, b, x4 + b * 8);
    }
    g = simd_sum(g);
    u = simd_sum(u);
    if (lane == 0) out[(ulong)s * m + row] = g / (1.0f + exp(-g)) * u;
}

// moe_down: x[t, row] += sum_k rw[t,k] * (W_{assign[t,k]}[row] . h[t*topK + k]). rowsPerExpert = dim.
// A slot whose assign is -1 (its expert runs on the CPU, see remap_slot) is skipped here and in moe_up / moe_combine.
// Grid: (ceil(dim / simdgroups), ntok). Looping over k inside keeps a single writer per (t, row).

template <int QT>
kernel void moe_down(device const uchar *wq [[buffer(0)]],
                     device const float *ws [[buffer(1)]],
                     device const float *h [[buffer(2)]],
                     device const int *assign [[buffer(3)]],
                     device const float *rw [[buffer(4)]],
                     device float *x [[buffer(5)]],
                     constant MoeParams &p [[buffer(6)]],
                     uint2 tgp [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]],
                     ushort nsg [[simdgroups_per_threadgroup]]) {
    const int row = int(tgp.x) * nsg + sg;
    if (row >= p.rowsPerExpert) return;
    const int t = int(tgp.y);
    const int nb = p.cols / QK;
    float total = 0.0f;
    for (int k = 0; k < p.topK; k++) {
        const int s = t * p.topK + k;
        if (assign[s] < 0) continue; // computed by the CPU
        const ulong grow = (ulong)assign[s] * p.rowsPerExpert + row;
        device const float4 *h4 = (device const float4 *)(h + (ulong)s * p.cols);
        float acc = 0.0f;
        for (int b = lane; b < nb; b += 32) acc += block_dot<QT>(wq, ws, grow, nb, b, h4 + b * 8);
        total += rw[s] * simd_sum(acc);
    }
    if (lane == 0) x[(ulong)t * p.rowsPerExpert + row] += total;
}

typedef decltype(moe_up<4>) moe_up_t;
template [[host_name("moe_up_q4")]] kernel moe_up_t moe_up<4>;
template [[host_name("moe_up_q8")]] kernel moe_up_t moe_up<8>;
typedef decltype(moe_down<4>) moe_down_t;
template [[host_name("moe_down_q4")]] kernel moe_down_t moe_down<4>;
template [[host_name("moe_down_q8")]] kernel moe_down_t moe_down<8>;

template [[host_name("moe_up_q2k")]] kernel moe_up_t moe_up<2>;
template [[host_name("moe_down_q2k")]] kernel moe_down_t moe_down<2>;

// ---- expert-grouped prefill (chunks of many tokens): moe_sort, moe_up_mma, moe_down_mma, moe_combine
// moe_up / moe_down give every slot (token, k) its own simdgroups, so a chunk reads an expert's weights once per slot
// that picked it. Here the slots are first grouped by expert (moe_sort: a stable counting sort, deterministic, no
// atomics) into tiles of up to MMA_BT slots of one expert, and each tile runs the simdgroup-matrix core of the dense
// prefill matmuls with its slots' activation rows gathered: the expert's weights are read once per tile. moe_down_mma
// writes each slot's weighted output row apart (no two tiles write the same element) and moe_combine adds a token's
// topK rows in k order, as moe_down does.

struct MoeSortParams {
    int slots;       // ntok * topK
    int experts;
    int maxTiles;    // capacity of the tiles list
};

// One threadgroup. order[] = the slots grouped by expert (ascending slot within an expert); tiles[0] = number of
// tiles, then (expert, first index into order, count) per tile.
kernel void moe_sort(device const int *assign [[buffer(0)]],
                     device int *order [[buffer(1)]],
                     device int *tiles [[buffer(2)]],
                     constant MoeSortParams &p [[buffer(3)]],
                     ushort tid [[thread_index_in_threadgroup]],
                     ushort tpg [[threads_per_threadgroup]]) {
    threadgroup int cnt[MAX_EXPERTS];
    threadgroup int off[MAX_EXPERTS];
    for (int e = tid; e < p.experts; e += tpg) {
        int c = 0;
        for (int s = 0; s < p.slots; s++) c += assign[s] == e ? 1 : 0;
        cnt[e] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int o = 0;
        int nt = 0;
        for (int e = 0; e < p.experts; e++) {
            off[e] = o;
            for (int i = 0; i < cnt[e]; i += MMA_BT) {
                if (nt < p.maxTiles) {
                    tiles[1 + 3 * nt] = e;
                    tiles[2 + 3 * nt] = o + i;
                    tiles[3 + 3 * nt] = min(MMA_BT, cnt[e] - i);
                    nt++;
                }
            }
            o += cnt[e];
        }
        tiles[0] = nt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = tid; e < p.experts; e += tpg) {
        int k = off[e];
        for (int s = 0; s < p.slots; s++) {
            if (assign[s] == e) order[k++] = s;
        }
    }
}

// h[s, unit] = silu(Wgate_e[unit] . x[t]) * (Wup_e[unit] . x[t]) for the slots s of tile tgp.y; 32 units per
// threadgroup (rows 0..31 of the 64-row tile are the gate rows, 32..63 the up rows, as in matmul_swiglu_mma).
template <int QT>
kernel void moe_up_mma(device const uchar *wq [[buffer(0)]],
                       device const float *ws [[buffer(1)]],
                       device const float *x [[buffer(2)]],
                       device const int *order [[buffer(3)]],
                       device const int *tiles [[buffer(4)]],
                       device float *out [[buffer(5)]],
                       constant MoeParams &p [[buffer(6)]],
                       uint2 tgp [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[MMA_SA_FLOATS];
    threadgroup float sb[MMA_SB_FLOATS];
    const int ti = int(tgp.y);
    if (ti >= tiles[0]) return; // the grid is sized for the most tiles a chunk can have
    const int e = tiles[1 + 3 * ti];
    const int start = tiles[2 + 3 * ti];
    const int cnt = tiles[3 + 3 * ti];
    const int m = p.rowsPerExpert / 2;
    const int unit0 = int(tgp.x) * (MMA_BR / 2);
    const int nb = p.cols / QK;
    const int lr = tid >> 1;
    const int unit = min(unit0 + (lr & 31), m - 1);
    // GGUF types: gate and up experts are two [experts * m, cols] views (wq, and buffer(1) = ws)
    device const uchar *rwq = (QT >= GG_FIRST && lr >= 32) ? (device const uchar *)ws : wq;
    const ulong grow = QT >= GG_FIRST ? (ulong)e * m + unit : (ulong)e * p.rowsPerExpert + (lr < 32 ? unit : m + unit);
    const int slot = order[start + min(int(tid >> 2), cnt - 1)];
    mma_core_rows<QT>(rwq, ws, grow, nb, x + (ulong)(slot / p.topK) * p.cols, sa, sb, sa, tid, sg);
    for (int i = tid; i < (MMA_BR / 2) * MMA_BT; i += MM_THREADS) {
        const int t = i / (MMA_BR / 2), j = i % (MMA_BR / 2);
        if (t < cnt && unit0 + j < m) {
            const float g = sa[t * MMA_BR + j];
            const float u = sa[t * MMA_BR + 32 + j];
            out[(ulong)order[start + t] * m + unit0 + j] = g / (1.0f + exp(-g)) * u;
        }
    }
}

// ys[s, row] = rw[s] * (W_e[row] . h[s]) for the slots s of tile tgp.y; 64 rows per threadgroup.
template <int QT>
kernel void moe_down_mma(device const uchar *wq [[buffer(0)]],
                         device const float *ws [[buffer(1)]],
                         device const float *h [[buffer(2)]],
                         device const int *order [[buffer(3)]],
                         device const int *tiles [[buffer(4)]],
                         device const float *rw [[buffer(5)]],
                         device float *ys [[buffer(6)]],
                         constant MoeParams &p [[buffer(7)]],
                         uint2 tgp [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[MMA_SA_FLOATS];
    threadgroup float sb[MMA_SB_FLOATS];
    const int ti = int(tgp.y);
    if (ti >= tiles[0]) return;
    const int e = tiles[1 + 3 * ti];
    const int start = tiles[2 + 3 * ti];
    const int cnt = tiles[3 + 3 * ti];
    const int row0 = int(tgp.x) * MMA_BR;
    const int nb = p.cols / QK;
    const int row = min(row0 + int(tid >> 1), p.rowsPerExpert - 1);
    const int slot = order[start + min(int(tid >> 2), cnt - 1)];
    mma_core_rows<QT>(wq, ws, (ulong)e * p.rowsPerExpert + row, nb, h + (ulong)slot * p.cols, sa, sb, sa, tid, sg);
    for (int i = tid; i < MMA_BR * MMA_BT; i += MM_THREADS) {
        const int t = i / MMA_BR, j = i % MMA_BR;
        if (t < cnt && row0 + j < p.rowsPerExpert) {
            const int s = order[start + t];
            ys[(ulong)s * p.rowsPerExpert + row0 + j] = rw[s] * sa[t * MMA_BR + j];
        }
    }
}

// x[t, row] += sum_k ys[t * topK + k, row], k ascending, over the slots with an expert on the GPU (moe_sort leaves
// the others out, so their ys rows are not written). Grid: (ceil(dim / ROW_THREADS), ntok).
kernel void moe_combine(device const float *ys [[buffer(0)]],
                        device const int *assign [[buffer(1)]],
                        device float *x [[buffer(2)]],
                        constant MoeParams &p [[buffer(3)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]]) {
    const int row = int(tgp.x) * ROW_THREADS + tid;
    if (row >= p.rowsPerExpert) return;
    const int t = int(tgp.y);
    float total = 0.0f;
    for (int k = 0; k < p.topK; k++) {
        const ulong s = (ulong)t * p.topK + k;
        if (assign[s] >= 0) total += ys[s * p.rowsPerExpert + row];
    }
    x[(ulong)t * p.rowsPerExpert + row] += total;
}

typedef decltype(moe_up_mma<4>) moe_up_mma_t;
typedef decltype(moe_down_mma<4>) moe_down_mma_t;
template [[host_name("moe_up_mma_q4")]] kernel moe_up_mma_t moe_up_mma<4>;
template [[host_name("moe_up_mma_q8")]] kernel moe_up_mma_t moe_up_mma<8>;
template [[host_name("moe_up_mma_q2k")]] kernel moe_up_mma_t moe_up_mma<2>;
template [[host_name("moe_down_mma_q4")]] kernel moe_down_mma_t moe_down_mma<4>;
template [[host_name("moe_down_mma_q8")]] kernel moe_down_mma_t moe_down_mma<8>;
template [[host_name("moe_down_mma_q2k")]] kernel moe_down_mma_t moe_down_mma<2>;

// router_ds: one threadgroup (256 threads = 8 simdgroups) per token. The logits are computed as in `router`
// (simdgroup sg handles experts sg, sg + 8, ...; the 32 lanes read the router row in coalesced float4 steps and
// finish with simd_sum), the softmax max and sum are threadgroup reductions, and thread 0 then applies the
// DeepSeek rule serially (nExperts <= 1024; see DeepSeekRouter.java, which this mirrors): softmax or sigmoid
// scores; method 0 = greedy, 1 = group_limited_greedy (group score = max), 2 = noaux_tc (selection on scores +
// bias, group score = sum of the two best); the topkGroup best of nGroup (<= 64) groups are kept; top-k inside
// them; weights are the uncorrected scores, renormalized if requested, times routedScale (always if scaleAlways,
// otherwise only when not renormalized). Writes assign[t*topK+k] and rw[t*topK+k], consumed by moe_up / moe_down.

struct RouterDsParams {
    int nExperts;
    int dim;
    int topK;
    int normalize;
    int nGroup;
    int topkGroup;
    int sigmoidScore;
    int method;
    int scaleAlways;
    float routedScale;
    int remap;       // see remap_slot
};

// router_ds's work for token tg, whose (normalised) row is xv. P: RouterDsParams or RouterDsNormParams.
template <typename P>
static inline void router_ds_row(device const float *xv, device const float *w, device int *assign, device float *rw,
                                 device const float *bias, device const int *map, device int *gassign,
                                 device atomic_uint *counts, constant P &p, uint tg, ushort tid, ushort tpg, ushort sg,
                                 ushort lane, ushort nsg, threadgroup float *scores, threadgroup float *choice,
                                 threadgroup float *buf) {
    const int ne = p.nExperts;
    device const float4 *x4 = (device const float4 *)xv;
    const int n4 = p.dim / 4;
    for (int e = sg; e < ne; e += nsg) {
        device const float *wv = w + (ulong)e * p.dim;
        device const float4 *w4 = (device const float4 *)wv;
        float d = 0.0f;
        for (int i = lane; i < n4; i += 32) d += dot(w4[i], x4[i]);
        for (int i = n4 * 4 + lane; i < p.dim; i += 32) d += wv[i] * xv[i]; // dim % 4 tail
        d = simd_sum(d);
        if (lane == 0) scores[e] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p.sigmoidScore) {
        for (int e = tid; e < ne; e += tpg) scores[e] = 1.0f / (1.0f + exp(-scores[e]));
    } else {
        float mx = -INFINITY;
        for (int e = tid; e < ne; e += tpg) mx = max(mx, scores[e]);
        mx = tg_max(mx, buf, sg, lane, nsg);
        float sum = 0.0f;
        for (int e = tid; e < ne; e += tpg) {
            const float v = exp(scores[e] - mx);
            scores[e] = v;
            sum += v;
        }
        sum = tg_sum(sum, buf, sg, lane, nsg);
        for (int e = tid; e < ne; e += tpg) scores[e] /= sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = tid; e < ne; e += tpg) choice[e] = p.method == 2 ? scores[e] + bias[e] : scores[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) return;

    if (p.method != 0) {
        const int gs = ne / p.nGroup;
        float gsc[MAX_GROUPS];
        bool keep[MAX_GROUPS];
        for (int g = 0; g < p.nGroup; g++) {
            float b1 = -INFINITY;
            float b2 = -INFINITY;
            for (int i = 0; i < gs; i++) {
                float v = choice[g * gs + i];
                if (v > b1) { b2 = b1; b1 = v; }
                else if (v > b2) { b2 = v; }
            }
            gsc[g] = p.method == 2 && gs > 1 ? b1 + b2 : b1;
            keep[g] = false;
        }
        for (int k = 0; k < p.topkGroup; k++) {
            int best = -1;
            for (int g = 0; g < p.nGroup; g++) {
                if (!keep[g] && (best < 0 || gsc[g] > gsc[best])) best = g;
            }
            keep[best] = true;
        }
        for (int e = 0; e < ne; e++) {
            if (!keep[e / gs]) choice[e] = -INFINITY;
        }
    }

    float selected = 0.0f;
    for (int k = 0; k < p.topK; k++) {
        int best = -1;
        float bv = -INFINITY;
        for (int e = 0; e < ne; e++) {
            if (choice[e] > bv) { bv = choice[e]; best = e; }
        }
        assign[tg * p.topK + k] = best;
        if (p.remap) remap_slot(map, gassign, counts, tg * p.topK + k, best);
        rw[tg * p.topK + k] = scores[best];
        selected += scores[best];
        choice[best] = -INFINITY;
    }
    const bool renorm = p.topK > 1 && p.normalize != 0;
    const float denom = selected + 1e-20f;
    const bool scale = renorm ? p.scaleAlways != 0 : true;
    for (int k = 0; k < p.topK; k++) {
        float v = rw[tg * p.topK + k];
        if (renorm) v /= denom;
        if (scale) v *= p.routedScale;
        rw[tg * p.topK + k] = v;
    }
}

kernel void router_ds(device const float *x [[buffer(0)]],
                      device const float *w [[buffer(1)]],
                      device int *assign [[buffer(2)]],
                      device float *rw [[buffer(3)]],
                      device const float *bias [[buffer(4)]],
                      device const int *map [[buffer(5)]],
                      device int *gassign [[buffer(6)]],
                      device atomic_uint *counts [[buffer(7)]],
                      constant RouterDsParams &p [[buffer(8)]],
                      uint tg [[threadgroup_position_in_grid]],
                      ushort tid [[thread_position_in_threadgroup]],
                      ushort tpg [[threads_per_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]],
                      ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float scores[MAX_EXPERTS];
    threadgroup float choice[MAX_EXPERTS];
    threadgroup float buf[32];
    router_ds_row(x + (ulong)tg * p.dim, w, assign, rw, bias, map, gassign, counts, p, tg, tid, tpg, sg, lane, nsg, scores,
                  choice, buf);
}

// router_ds_norm: the MoE layer's rmsnorm and router_ds in one dispatch, as router_norm.
struct RouterDsNormParams {
    int nExperts;
    int dim;
    int topK;
    int normalize;
    int nGroup;
    int topkGroup;
    int sigmoidScore;
    int method;
    int scaleAlways;
    float routedScale;
    int remap;
    float eps;
};

kernel void router_ds_norm(device const float *x [[buffer(0)]],
                           device const float *w [[buffer(1)]],
                           device int *assign [[buffer(2)]],
                           device float *rw [[buffer(3)]],
                           device const float *bias [[buffer(4)]],
                           device const int *map [[buffer(5)]],
                           device int *gassign [[buffer(6)]],
                           device atomic_uint *counts [[buffer(7)]],
                           device const float *nw [[buffer(8)]],
                           device float *xn [[buffer(9)]],
                           constant RouterDsNormParams &p [[buffer(10)]],
                           uint tg [[threadgroup_position_in_grid]],
                           ushort tid [[thread_position_in_threadgroup]],
                           ushort tpg [[threads_per_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float scores[MAX_EXPERTS];
    threadgroup float choice[MAX_EXPERTS];
    threadgroup float buf[32];
    device float *row = xn + (ulong)tg * p.dim;
    norm_row_into(x + (ulong)tg * p.dim, nw, row, p.dim, p.eps, tid, tpg, buf, sg, lane, nsg);
    router_ds_row(row, w, assign, rw, bias, map, gassign, counts, p, tg, tid, tpg, sg, lane, nsg, scores, choice, buf);
}

// ---------------------------------------------------------------------------------------------- 6. state space (Mamba)
// Mamba (v1) mixer pieces, used by MambaMetalBackend. The projections (in_proj, x_proj, out_proj) reuse the
// matmul kernels above; the two kernels below are the parts that carry state across tokens. Both use one thread
// per channel and run the tokens of the chunk sequentially inside the thread (the recurrence is inherently
// sequential over tokens, independent over channels), so a layer costs one dispatch per kernel instead of one per
// token. The price: parallelism is d_inner threads (1536 .. 5120), no more during a long prefill.

struct MambaConvParams {
    int dInner;
    int dConv;      // <= MAMBA_MAX_CONV
    int ntok;
};

// xc[t, i] = silu(bias[i] + sum_k w[i, k] * window[k]) where window = (conv state of channel i, xz[t, i]);
// the state (d_conv - 1 previous inputs per channel, oldest first) is advanced past the chunk. The window and
// the channel's weights stay in registers for the whole chunk; the state is read once and written once.
// xz is [ntok, 2 * dInner] (x half first), xc is [ntok, dInner].
kernel void mamba_conv(device const float *xz [[buffer(0)]],
                       device const float *w [[buffer(1)]],
                       device const float *bias [[buffer(2)]],
                       device float *state [[buffer(3)]],
                       device float *xc [[buffer(4)]],
                       constant MambaConvParams &p [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.dInner) return;
    const int dc = p.dConv;
    device float *st = state + (ulong)i * (dc - 1);
    device const float *wi = w + (ulong)i * dc;
    float win[MAMBA_MAX_CONV - 1];
    float wr[MAMBA_MAX_CONV];
    for (int k = 0; k < dc - 1; k++) win[k] = st[k];
    for (int k = 0; k < dc; k++) wr[k] = wi[k];
    const float b = bias[i];
    for (int t = 0; t < p.ntok; t++) {
        const float u = xz[(ulong)t * 2 * p.dInner + i];
        float sum = b + wr[dc - 1] * u;
        for (int k = 0; k < dc - 1; k++) sum += wr[k] * win[k];
        for (int k = 0; k + 1 < dc - 1; k++) win[k] = win[k + 1];
        if (dc > 1) win[dc - 2] = u;
        xc[(ulong)t * p.dInner + i] = sum / (1.0f + precise::exp(-sum));
    }
    for (int k = 0; k < dc - 1; k++) st[k] = win[k];
}

struct MambaScanParams {
    int dInner;
    int dState;
    int dtRank;
    int ntok;
};

// Selective scan for one channel i over the tokens of the chunk:
//   dt = softplus(dot(dt_proj[i, :], dbc[t, 0:R]) + dt_bias[i])
//   s[n] = exp(dt * A[i, n]) * s[n] + dt * B[t, n] * u          (u = xc[t, i], B = dbc[t, R:R+N])
//   y = (sum_n s[n] * C[t, n] + D[i] * u) * silu(z)              (C = dbc[t, R+N:R+2N], z = xz[t, dInner + i])
// The SSM state [dInner, dState] lives in registers for the whole chunk and is written back at the end.
kernel void mamba_scan(device const float *xc [[buffer(0)]],
                       device const float *xz [[buffer(1)]],
                       device const float *dbc [[buffer(2)]],
                       device const float *dtw [[buffer(3)]],
                       device const float *dtb [[buffer(4)]],
                       device const float *A [[buffer(5)]],
                       device const float *D [[buffer(6)]],
                       device float *state [[buffer(7)]],
                       device float *y [[buffer(8)]],
                       constant MambaScanParams &p [[buffer(9)]],
                       uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.dInner) return;
    const int N = p.dState;
    const int R = p.dtRank;
    const int stride = R + 2 * N;
    device float *st = state + (ulong)i * N;
    device const float *a = A + (ulong)i * N;
    device const float *wi = dtw + (ulong)i * R;
    float s[MAMBA_MAX_STATE];
    for (int n = 0; n < N; n++) s[n] = st[n];
    for (int t = 0; t < p.ntok; t++) {
        device const float *row = dbc + (ulong)t * stride;
        float raw = dtb[i];
        for (int r = 0; r < R; r++) raw += wi[r] * row[r];
        const float dt = raw > SOFTPLUS_LINEAR ? raw : precise::log(1.0f + precise::exp(raw));
        const float u = xc[(ulong)t * p.dInner + i];
        const float du = dt * u;
        float acc = 0.0f;
        for (int n = 0; n < N; n++) {
            s[n] = precise::exp(dt * a[n]) * s[n] + du * row[R + n];
            acc += s[n] * row[R + N + n];
        }
        const float z = xz[(ulong)t * 2 * p.dInner + p.dInner + i];
        y[(ulong)t * p.dInner + i] = (acc + D[i] * u) * (z / (1.0f + precise::exp(-z)));
    }
    for (int n = 0; n < N; n++) st[n] = s[n];
}

// ---------------------------------------------------------------------------------------------- 7. DeepSeek MLA
// Multi-head latent attention (DeepSeek-V2/V3) in the absorbed form (see MlaCpu for the derivation): the cache
// holds one row [c | k_pe] per token (c = normalised kv_a latent, kvLora wide; k_pe = the rotated shared rotary
// key, rope wide), the two halves of kv_b_proj never expand it. Per head the query becomes
//   q_abs = [W_kb_k[h]^T q_nope | rope(q_pe)]           (kvLora + rope wide, mla_absorb_q)
// so the generic attention kernel (attention_mla[_h]: qkDim = kvLora + rope, vDim = kvLora, one kv head shared by
// all query heads, the value = the leading kvLora of the key row) produces o = sum_s w_s c_s per head, and
//   out = W_kb_v[h] o                                    (vDim wide, mla_apply_v)
// finishes it. Projections, norms and o_proj use the generic matmul / rmsnorm kernels. W_kb_k / W_kb_v are the
// per-head key / value rows of kv_b_proj, dequantised to f32 at load; both are read with the lane index on the
// contiguous axis: wk is [heads][nope][kvLora], wv is [heads][kvLora][vDim].

struct MlaParams {
    int nope;
    int rope;
    int kvLora;
    int vDim;
    int heads;
};

// mla_lat_store: cache row [c | k_pe] of every token of the chunk; `cache` is bound at the row of the first
// token (f32, or half for backend.kv-dtype=f16). Grid: (ntok), CHANNEL_THREADS.
template <typename KV>
kernel void mla_lat_store_t(device const float *lat [[buffer(0)]],
                            device const float *kpe [[buffer(1)]],
                            device KV *cache [[buffer(2)]],
                            constant MlaParams &p [[buffer(3)]],
                            uint2 tgp [[threadgroup_position_in_grid]],
                            ushort2 tid2 [[thread_position_in_threadgroup]],
                            ushort2 tpg2 [[threads_per_threadgroup]]) {
    const ushort tid = tid2.x;
    const ushort tpg = tpg2.x;
    const int t = int(tgp.x);
    device KV *row = cache + (ulong)t * (p.kvLora + p.rope);
    for (int i = tid; i < p.kvLora; i += tpg) row[i] = KV(lat[(ulong)t * p.kvLora + i]);
    for (int i = tid; i < p.rope; i += tpg) row[p.kvLora + i] = KV(kpe[(ulong)t * p.rope + i]);
}
template [[host_name("mla_lat_store")]] kernel decltype(mla_lat_store_t<float>) mla_lat_store_t<float>;
template [[host_name("mla_lat_store_h")]] kernel decltype(mla_lat_store_t<half>) mla_lat_store_t<half>;

// mla_lat_store_q8: the same row quantized for backend.kv-dtype=q8: one scale for the whole row [c | k_pe]
// (max|x| / 127 over both parts, as kv_store_q8 per kv head), stored after its kvLora + rope int8 values (row stride
// kvLora + rope + 4 bytes; attention_mla_q8 reads it through kv_scale with one kv head). Grid: (ntok), CHANNEL_THREADS.
kernel void mla_lat_store_q8(device const float *lat [[buffer(0)]],
                             device const float *kpe [[buffer(1)]],
                             device char *cache [[buffer(2)]],
                             constant MlaParams &p [[buffer(3)]],
                             uint2 tgp [[threadgroup_position_in_grid]],
                             ushort2 tid2 [[thread_position_in_threadgroup]],
                             ushort2 tpg2 [[threads_per_threadgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]],
                             ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[32];
    const ushort tid = tid2.x;
    const ushort tpg = tpg2.x;
    const int t = int(tgp.x);
    const int aw = p.kvLora + p.rope;
    device const float *c = lat + (ulong)t * p.kvLora;
    device const float *r = kpe + (ulong)t * p.rope;
    device char *row = cache + (ulong)t * (aw + 4);
    float mx = 0.0f;
    for (int i = tid; i < p.kvLora; i += tpg) mx = max(mx, fabs(c[i]));
    for (int i = tid; i < p.rope; i += tpg) mx = max(mx, fabs(r[i]));
    const float s = tg_max(mx, buf, sg, lane, nsg) / 127.0f;
    for (int i = tid; i < aw; i += tpg) {
        const float x = i < p.kvLora ? c[i] : r[i - p.kvLora];
        row[i] = s > 0.0f ? char(clamp(rint(x / s), -127.0f, 127.0f)) : char(0);
    }
    if (tid == 0) *((device float *)(row + aw)) = s;
}

// mla_absorb_q: qa[t, h] = [W_kb_k[h]^T q_nope[t, h] | q_pe[t, h]] from q ([ntok, heads * (nope + rope)], q_pe
// already rotated in place) and wk ([heads][nope][kvLora]). One thread per output j, the nope-long sum reads
// wk[h][i][j] coalesced across the threadgroup. Grid: (heads, ntok), ROW_THREADS.
kernel void mla_absorb_q(device const float *q [[buffer(0)]],
                         device const float *wk [[buffer(1)]],
                         device float *qa [[buffer(2)]],
                         constant MlaParams &p [[buffer(3)]],
                         uint2 tgp [[threadgroup_position_in_grid]],
                         ushort2 tid2 [[thread_position_in_threadgroup]],
                         ushort2 tpg2 [[threads_per_threadgroup]]) {
    const ushort tid = tid2.x;
    const ushort tpg = tpg2.x;
    const int h = int(tgp.x);
    const int t = int(tgp.y);
    const int qk = p.nope + p.rope;
    const int aw = p.kvLora + p.rope;
    device const float *qh = q + ((ulong)t * p.heads + h) * qk;
    device const float *w = wk + (ulong)h * p.nope * p.kvLora;
    device float *out = qa + ((ulong)t * p.heads + h) * aw;
    for (int j = tid; j < p.kvLora; j += tpg) {
        float acc = 0.0f;
        for (int i = 0; i < p.nope; i++) acc += qh[i] * w[(ulong)i * p.kvLora + j];
        out[j] = acc;
    }
    for (int i = tid; i < p.rope; i += tpg) out[p.kvLora + i] = qh[p.nope + i];
}

// mla_apply_v: att[t, h] = W_kb_v[h] o[t, h] from the attention output o ([ntok, heads * kvLora]) and wv
// ([heads][kvLora][vDim]); one thread per output i, wv[h][j][i] coalesced. Grid: (heads, ntok), ROW_THREADS.
kernel void mla_apply_v(device const float *o [[buffer(0)]],
                        device const float *wv [[buffer(1)]],
                        device float *att [[buffer(2)]],
                        constant MlaParams &p [[buffer(3)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        ushort2 tid2 [[thread_position_in_threadgroup]],
                        ushort2 tpg2 [[threads_per_threadgroup]]) {
    const ushort tid = tid2.x;
    const ushort tpg = tpg2.x;
    const int h = int(tgp.x);
    const int t = int(tgp.y);
    device const float *oh = o + ((ulong)t * p.heads + h) * p.kvLora;
    device const float *w = wv + (ulong)h * p.kvLora * p.vDim;
    device float *out = att + ((ulong)t * p.heads + h) * p.vDim;
    for (int i = tid; i < p.vDim; i += tpg) {
        float acc = 0.0f;
        for (int j = 0; j < p.kvLora; j++) acc += w[(ulong)j * p.vDim + i] * oh[j];
        out[i] = acc;
    }
}


// ---------------------------------------------------------------------------------------------- 8. Q6_K
// llama.cpp's Q6_K: 256-weight super-blocks of 210 bytes = ql[128] (low 4 bits, two per byte) + qh[64] (high
// 2 bits, four per byte) + int8 scales[16] + f16 d; weight = d * scales[j] * (q - 32), q in 0..63.
// Sub-block j (weights 16j .. 16j+15): half h = j/8, group g = (j%8)/2 (weights 32g.. of the half), lb = j%2;
//   ql byte  64h + 32*(g&1) + 16*lb + k, nibble (g < 2 ? low : high)
//   qh byte  32h + 16*lb + k, bits 2g..2g+1 -> bits 4..5 of q
// Blocks are 210 bytes, so only 2-byte aligned: the quants are read through packed_uchar4 (alignment 1) and d as
// a half. Scales live inside the block, so buffer(1) (ws) is unused. The matmul kernels walk the row in
// 16-weight units (cols / 16 per row, nb = cols / 32), as Q2_K does; the MoE block_dot uses sub-blocks.

// Dequantizes 16-weight sub-block u of global row `grow` (nsb super-blocks per row) into w[0..4).
static inline void q6k_sub_block(device const uchar *wq, ulong grow, int nsb, int u, thread float4 *w) {
    device const uchar *blk = wq + (grow * nsb + (u >> 4)) * Q6K_BYTES;
    const int j = u & 15;
    const int h = j >> 3;
    const int g = (j & 7) >> 1;
    const int lb = j & 1;
    const float dl = float(*(device const half *)(blk + 208)) * float(((device const char *)blk)[192 + j]);
    const uchar nshift = uchar((g >> 1) << 2);
    const uchar hshift = uchar(g << 1);
    device const packed_uchar4 *ql = (device const packed_uchar4 *)(blk + 64 * h + 32 * (g & 1) + 16 * lb);
    device const packed_uchar4 *qh = (device const packed_uchar4 *)(blk + 128 + 32 * h + 16 * lb);
    for (int k = 0; k < 4; k++) {
        const uchar4 lo = (uchar4(ql[k]) >> uchar4(nshift)) & uchar4(15);
        const uchar4 hi = (uchar4(qh[k]) >> uchar4(hshift)) & uchar4(3);
        w[k] = (float4(lo | (hi << uchar4(4))) - 32.0f) * dl;
    }
}

// block_dot for the MoE kernels: 32-weight block b of a Q6_K row = sub-blocks 2b and 2b+1 (nb = cols / 32).
template <>
inline float block_dot<6>(device const uchar *wq, device const float *ws, ulong grow, int nb, int b,
                          device const float4 *x4) {
    const int nsb = (nb + 7) >> 3;
    float4 w[4];
    float d = 0.0f;
    q6k_sub_block(wq, grow, nsb, 2 * b, w);
    for (int k = 0; k < 4; k++) d += dot(w[k], x4[k]);
    q6k_sub_block(wq, grow, nsb, 2 * b + 1, w);
    for (int k = 0; k < 4; k++) d += dot(w[k], x4[k + 4]);
    return d;
}

// The matmul lane unit of Q6_K is a "column quad", not a sub-block: unit u = (super-block u/16, half h = (u%16)/8,
// quad lq = u%8) covers weights 4lq .. 4lq+3 of the half's four 32-weight groups, i.e. the 16 weights
// 128h + 32g + 4lq + k (g = 0..3, k = 0..3). Those are exactly the weights that share three 4-byte pieces of the
// block (ql[4lq ..], ql[32 + 4lq ..], qh[4lq ..]), so a unit reads 12 quant bytes (as 2-byte loads: blocks are
// only 2-byte aligned) instead of the 32 a sub-block unit re-reads, and neighbouring lanes read neighbouring bytes
// and activations. Group g has scale scales[8h + 2g + lq/4]. Same unit count as the sub-block walk (16 per block).
// w[g] holds the four weights of group g; the activations are x4[(blk * 64 + h * 32 + g * 8 + lq)].
static inline void q6k_unit(device const uchar *wq, ulong grow, int nsb, int u, thread float4 *w, thread float *dl) {
    device const uchar *blk = wq + (grow * nsb + (u >> 4)) * Q6K_BYTES;
    const int h = (u >> 3) & 1;
    const int lq = u & 7;
    device const ushort *ql0 = (device const ushort *)(blk + 64 * h + 4 * lq);
    device const ushort *ql1 = (device const ushort *)(blk + 64 * h + 32 + 4 * lq);
    device const ushort *qh = (device const ushort *)(blk + 128 + 32 * h + 4 * lq);
    device const ushort *sc = (device const ushort *)(blk + 192 + 8 * h);
    const float d = float(*(device const half *)(blk + 208));
    const uchar4 qa = as_type<uchar4>(uint(ql0[0]) | (uint(ql0[1]) << 16));
    const uchar4 qb = as_type<uchar4>(uint(ql1[0]) | (uint(ql1[1]) << 16));
    const uchar4 qc = as_type<uchar4>(uint(qh[0]) | (uint(qh[1]) << 16));
    const uchar4 m15 = uchar4(15);
    const uchar4 m3 = uchar4(3);
    const uchar4 s4 = uchar4(4);
    const int shift = (lq >> 2) << 3;
    w[0] = float4((qa & m15) | ((qc & m3) << s4)) - 32.0f;
    w[1] = float4((qb & m15) | (((qc >> uchar4(2)) & m3) << s4)) - 32.0f;
    w[2] = float4((qa >> s4) | (((qc >> s4) & m3) << s4)) - 32.0f;
    w[3] = float4((qb >> s4) | ((qc >> uchar4(6)) << s4)) - 32.0f;
    dl[0] = d * float(char((sc[0] >> shift) & 0xFF));
    dl[1] = d * float(char((sc[1] >> shift) & 0xFF));
    dl[2] = d * float(char((sc[2] >> shift) & 0xFF));
    dl[3] = d * float(char((sc[3] >> shift) & 0xFF));
}

// Number of 32-weight groups of the unit's half that exist: a row whose width is not a multiple of 256 ends with a
// partial super-block holding cols % 256 weights (a multiple of 32), whose missing groups are skipped.
static inline int q6k_groups(int u, int cols) {
    const int n = min(256, cols - (u >> 4) * 256) - 128 * ((u >> 3) & 1);
    return clamp(n >> 5, 0, 4);
}

template <typename XP>
static inline void mm_row_q6k(device const uchar *wq, ulong grow, int nb, XP x, ulong xstride, int nt,
                              thread float *acc, ushort lane) {
    const int nsb = (nb + 7) >> 3;
    const int nunit = nsb * 16;
    for (int u = lane; u < nunit; u += 32) {
        const int ng = q6k_groups(u, nb * QK);
        if (ng == 0) continue;
        float4 w[4];
        float dl[4];
        q6k_unit(wq, grow, nsb, u, w, dl);
        const int xo = (u >> 4) * 64 + ((u >> 3) & 1) * 32 + (u & 7);
        for (int t = 0; t < TOK_TILE; t++) {
            if (t < nt) {
                auto x4 = as_f4(x + (ulong)t * xstride);
                float d = dot(w[0], x4[xo]) * dl[0];
                if (ng == 4) {
                    d += dot(w[1], x4[xo + 8]) * dl[1];
                    d += dot(w[2], x4[xo + 16]) * dl[2];
                    d += dot(w[3], x4[xo + 24]) * dl[3];
                } else if (ng == 2) {
                    d += dot(w[1], x4[xo + 8]) * dl[1];
                } else if (ng == 3) {
                    d += dot(w[1], x4[xo + 8]) * dl[1];
                    d += dot(w[2], x4[xo + 16]) * dl[2];
                }
                acc[t] += d;
            }
        }
    }
}

template <int NR, int TT>
static inline void mm_tile_q6k(thread device const uchar *const *wq, thread const ulong *grow, int nr, int nb,
                               device const float *x, ulong xstride, int nt, thread float (*acc)[TT], ushort lane) {
    const int nsb = (nb + 7) >> 3;
    const int nunit = nsb * 16;
    for (int u = lane; u < nunit; u += 32) {
        const int ng = q6k_groups(u, nb * QK);
        if (ng == 0) continue;
        float4 w[NR][4];
        float dl[NR][4];
        for (int r = 0; r < NR; r++) {
            if (r < nr) q6k_unit(wq[r], grow[r], nsb, u, w[r], dl[r]);
        }
        const int xo = (u >> 4) * 64 + ((u >> 3) & 1) * 32 + (u & 7);
        for (int t = 0; t < TT; t++) {
            if (t < nt) {
                auto x4 = as_f4(x + (ulong)t * xstride);
                float4 xv[4];
                xv[0] = x4[xo];
                if (ng > 1) xv[1] = x4[xo + 8];
                if (ng > 2) xv[2] = x4[xo + 16];
                if (ng > 3) xv[3] = x4[xo + 24];
                for (int r = 0; r < NR; r++) {
                    if (r < nr) {
                        float d = dot(w[r][0], xv[0]) * dl[r][0];
                        if (ng > 1) d += dot(w[r][1], xv[1]) * dl[r][1];
                        if (ng > 2) d += dot(w[r][2], xv[2]) * dl[r][2];
                        if (ng > 3) d += dot(w[r][3], xv[3]) * dl[r][3];
                        acc[r][t] += d;
                    }
                }
            }
        }
    }
}

template [[host_name("matmul_mma_q6k")]] kernel matmul_mma_t matmul_mma<6>;
template [[host_name("moe_up_mma_q6k")]] kernel moe_up_mma_t moe_up_mma<6>;
template [[host_name("moe_down_mma_q6k")]] kernel moe_down_mma_t moe_down_mma<6>;
template [[host_name("matmul3_mma_q6k")]] kernel matmul3_mma_t matmul3_mma<6, float>;
template [[host_name("matmul3_mma_q6k_h")]] kernel matmul3_mma_h_t matmul3_mma<6, half>;
template [[host_name("matmul_swiglu_mma_q6k")]] kernel matmul_swiglu_mma_t matmul_swiglu_mma<6>;
template [[host_name("matmul_q6k")]] kernel matmul_t matmul<6, false>;
template [[host_name("matmul_q6k_norm")]] kernel matmul_t matmul<6, true>;
template [[host_name("matmul_pf_q6k")]] kernel matmul_pf_t matmul_pf<6, PF_ROWS, PF_TOKS>;
template [[host_name("matmul3_pf_q6k")]] kernel matmul3_pf_t matmul3_pf<6, PF_ROWS, PF_TOKS, float>;
template [[host_name("matmul3_pf_q6k_h")]] kernel matmul3_pf_h_t matmul3_pf<6, PF_ROWS, PF_TOKS, half>;
template [[host_name("matmul_swiglu_pf_q6k")]] kernel matmul_swiglu_pf_t matmul_swiglu_pf<6, PF_UNITS, PF_TOKS>;
template [[host_name("matmul3_q6k")]] kernel matmul3_t matmul3<6, false, float>;
template [[host_name("matmul3_q6k_norm")]] kernel matmul3_t matmul3<6, true, float>;
template [[host_name("matmul3_q6k_h")]] kernel matmul3_h_t matmul3<6, false, half>;
template [[host_name("matmul3_q6k_norm_h")]] kernel matmul3_h_t matmul3<6, true, half>;
template [[host_name("matmul_swiglu_q6k")]] kernel matmul_swiglu_t matmul_swiglu<6, false>;
template [[host_name("matmul_swiglu_q6k_norm")]] kernel matmul_swiglu_t matmul_swiglu<6, true>;
template [[host_name("moe_up_q6k")]] kernel moe_up_t moe_up<6>;
template [[host_name("moe_down_q6k")]] kernel moe_down_t moe_down<6>;

// ============================================================================================== GGUF block types
// llama.cpp's own block layouts, read in place from a memory-mapped GGUF file (docs/16-gguf.md). The scales live
// inside the blocks, so buffer(1) (ws) is unused, exactly as for Q2_K. Blocks are only 2-byte aligned for the
// legacy formats (18 and 34 bytes), so their quants are read as ushorts; the K-quants' 144 / 176-byte blocks are
// 4-byte aligned. One lane unit is a 32-weight block (Q4_0 / Q8_0) or sub-block (Q4_K / Q5_K: eight per
// 256-weight super-block), so all four types walk a row with the same unit loop (cols / 32 units) and share
// gg_unit, which dequantizes a unit to 8 float4 (scale and min applied). The single-row kernels (mm_row_gguf),
// the prefill tiles (mm_tile_gguf) and the MoE block_dot all use it, so every kernel computes the same products.
//   QT  20 Q4_0 (18 B: half d, 16 nibble bytes)     21 Q8_0 (34 B: half d, 32 int8)
//       22 Q4_K (144 B: half d, dmin, 12 scale bytes, 128 nibble bytes)
//       23 Q5_K (176 B: half d, dmin, 12 scale bytes, 32 high-bit bytes, 128 nibble bytes)

#define GG_Q4_0 20
#define GG_Q8_0 21
#define GG_Q4_K 22
#define GG_Q5_K 23

// 32 weights of unit u of global row `grow` (nb units per row) into w[0..8) as dequantized floats.
template <int QT>
static inline void gg_unit(device const uchar *wq, ulong grow, int nb, int u, thread float4 *w) {
    if (QT == GG_Q4_0 || QT == GG_Q8_0) {
        constexpr int BYTES = QT == GG_Q4_0 ? 18 : 34;
        device const uchar *blk = wq + (grow * nb + u) * BYTES;
        const float d = float(*(device const half *)blk);
        device const ushort *p = (device const ushort *)(blk + 2);
        if (QT == GG_Q4_0) {
            for (int k = 0; k < 4; k++) {
                const uchar4 v = as_type<uchar4>(uint(p[2 * k]) | (uint(p[2 * k + 1]) << 16));
                w[k] = (float4(v & uchar4(15)) - 8.0f) * d;
                w[k + 4] = (float4(v >> uchar4(4)) - 8.0f) * d;
            }
        } else {
            for (int k = 0; k < 8; k++) {
                w[k] = float4(as_type<char4>(uint(p[2 * k]) | (uint(p[2 * k + 1]) << 16))) * d;
            }
        }
    } else {
        constexpr bool Q5 = QT == GG_Q5_K;
        constexpr int BYTES = Q5 ? 176 : 144;
        const int nsb = nb >> 3;
        device const uchar *blk = wq + (grow * nsb + (u >> 3)) * BYTES;
        const int j = u & 7;
        const float2 dd = float2(as_type<half2>(*(device const uint *)blk));   // d, dmin (4-byte aligned)
        device const uchar *sc = blk + 4;
        uint s;
        uint m;
        if (j < 4) {
            s = sc[j] & 63;
            m = sc[j + 4] & 63;
        } else {
            s = (sc[j + 4] & 15) | ((uint(sc[j - 4]) >> 6) << 4);
            m = (sc[j + 4] >> 4) | ((uint(sc[j]) >> 6) << 4);
        }
        const float dl = dd.x * float(s);
        const float ml = dd.y * float(m);
        const uchar sh = uchar((j & 1) << 2);
        device const uchar4 *q = (device const uchar4 *)(blk + (Q5 ? 48 : 16) + (j >> 1) * 32);
        device const uchar4 *h = (device const uchar4 *)(blk + 16);
        for (int k = 0; k < 8; k++) {
            uchar4 v = (q[k] >> uchar4(sh)) & uchar4(15);
            if (Q5) v |= ((h[k] >> uchar4(j)) & uchar4(1)) << uchar4(4);
            w[k] = float4(v) * dl - ml;
        }
    }
}

// mm_row for QT >= GG_FIRST (called from the generic mm_row): lane l walks units l, l + 32, ...
template <int QT, typename XP>
static inline void mm_row_gguf(device const uchar *wq, ulong grow, int nb, XP x, ulong xstride, int nt,
                               thread float *acc, ushort lane) {
    for (int u = lane; u < nb; u += 32) {
        float4 w[8];
        gg_unit<QT>(wq, grow, nb, u, w);
        for (int t = 0; t < TOK_TILE; t++) {
            if (t < nt) {
                auto x4 = as_f4(x + (ulong)t * xstride + u * QK);
                float d = 0.0f;
                for (int k = 0; k < 8; k++) d += dot(w[k], x4[k]);
                acc[t] += d;
            }
        }
    }
}

// mm_tile for QT >= GG_FIRST: the units of the NR rows are dequantized once and used for TT tokens, each
// row's unit walk and dot being those of mm_row_gguf.
template <int QT, int NR, int TT>
static inline void mm_tile_gguf(thread device const uchar *const *wq, thread const ulong *grow, int nr, int nb,
                                device const float *x, ulong xstride, int nt, thread float (*acc)[TT], ushort lane) {
    for (int u = lane; u < nb; u += 32) {
        float4 w[NR][8];
        for (int r = 0; r < NR; r++) {
            if (r < nr) gg_unit<QT>(wq[r], grow[r], nb, u, w[r]);
        }
        for (int t = 0; t < TT; t++) {
            if (t < nt) {
                auto x4 = as_f4(x + (ulong)t * xstride + u * QK);
                float4 xv[8];
                for (int k = 0; k < 8; k++) xv[k] = x4[k];
                for (int r = 0; r < NR; r++) {
                    if (r < nr) {
                        float d = 0.0f;
                        for (int k = 0; k < 8; k++) d += dot(w[r][k], xv[k]);
                        acc[r][t] += d;
                    }
                }
            }
        }
    }
}

// block_dot for the MoE kernels: 32-weight unit b of a row.
#define GG_BLOCK_DOT(QTV)                                                                                         \
    template <>                                                                                                   \
    inline float block_dot<QTV>(device const uchar *wq, device const float *ws, ulong grow, int nb, int b,       \
                                device const float4 *x4) {                                                        \
        float4 w[8];                                                                                              \
        gg_unit<QTV>(wq, grow, nb, b, w);                                                                         \
        float d = 0.0f;                                                                                           \
        for (int k = 0; k < 8; k++) d += dot(w[k], x4[k]);                                                        \
        return d;                                                                                                 \
    }
GG_BLOCK_DOT(GG_Q4_0)
GG_BLOCK_DOT(GG_Q8_0)
GG_BLOCK_DOT(GG_Q4_K)
GG_BLOCK_DOT(GG_Q5_K)

#define GG_INSTANTIATE(SUF, QTV)                                                                                  \
    template [[host_name("matmul_" SUF)]] kernel matmul_t matmul<QTV, false>;                                     \
    template [[host_name("matmul_" SUF "_norm")]] kernel matmul_t matmul<QTV, true>;                              \
    template [[host_name("matmul_pf_" SUF)]] kernel matmul_pf_t matmul_pf<QTV, PF_ROWS, PF_TOKS>;                 \
    template [[host_name("matmul3_" SUF)]] kernel matmul3_t matmul3<QTV, false, float>;                           \
    template [[host_name("matmul3_" SUF "_norm")]] kernel matmul3_t matmul3<QTV, true, float>;                    \
    template [[host_name("matmul3_" SUF "_h")]] kernel matmul3_h_t matmul3<QTV, false, half>;                     \
    template [[host_name("matmul3_" SUF "_norm_h")]] kernel matmul3_h_t matmul3<QTV, true, half>;                 \
    template [[host_name("matmul3_pf_" SUF)]] kernel matmul3_pf_t matmul3_pf<QTV, PF_ROWS, PF_TOKS, float>;       \
    template [[host_name("matmul3_pf_" SUF "_h")]] kernel matmul3_pf_h_t matmul3_pf<QTV, PF_ROWS, PF_TOKS, half>; \
    template [[host_name("matmul_swiglu_" SUF)]] kernel matmul_swiglu_t matmul_swiglu<QTV, false>;                \
    template [[host_name("matmul_swiglu_" SUF "_norm")]] kernel matmul_swiglu_t matmul_swiglu<QTV, true>;         \
    template [[host_name("matmul_swiglu_pf_" SUF)]] kernel matmul_swiglu_pf_t matmul_swiglu_pf<QTV, PF_UNITS, PF_TOKS>; \
    template [[host_name("matmul_mma_" SUF)]] kernel matmul_mma_t matmul_mma<QTV>;                                \
    template [[host_name("matmul3_mma_" SUF)]] kernel matmul3_mma_t matmul3_mma<QTV, float>;                      \
    template [[host_name("matmul3_mma_" SUF "_h")]] kernel matmul3_mma_h_t matmul3_mma<QTV, half>;                \
    template [[host_name("matmul_swiglu_mma_" SUF)]] kernel matmul_swiglu_mma_t matmul_swiglu_mma<QTV>;          \
    template [[host_name("moe_up_" SUF)]] kernel moe_up_t moe_up<QTV>;                                            \
    template [[host_name("moe_down_" SUF)]] kernel moe_down_t moe_down<QTV>;                                      \
    template [[host_name("moe_up_mma_" SUF)]] kernel moe_up_mma_t moe_up_mma<QTV>;                                \
    template [[host_name("moe_down_mma_" SUF)]] kernel moe_down_mma_t moe_down_mma<QTV>;
GG_INSTANTIATE("q4g", GG_Q4_0)
GG_INSTANTIATE("q8g", GG_Q8_0)
GG_INSTANTIATE("q4k", GG_Q4_K)
GG_INSTANTIATE("q5k", GG_Q5_K)

// ---------------------------------------------------------------------------------------------- 9. flash-attention prefill
// attention_fa<KV, D>: causal (optionally windowed, soft-capped) attention for a prefill chunk without a scores
// scratch. A threadgroup of 4 simdgroups takes FA_ROWS = 32 query tokens of one head (8 rows per simdgroup) and walks
// the keys they see in blocks of BK = 4096 / D keys, each block's K (then V) rows staged once in threadgroup memory
// (16 KB) and shared by the 32 rows:
//   S   = Q K^T          8 x BK per simdgroup, simdgroup_float8x8 multiply-accumulates over the head in 8-wide slices
//   m'  = max(m, rowmax(S)),  P = exp(S - m'),  l = l * exp(m - m') + rowsum(P)        (online softmax, per row)
//   O   = diag(exp(m - m')) O + P V
// and finally O = diag(1 / l) O. The row rescales are products with a diagonal matrix, so the kernel never
// addresses the elements of a simdgroup matrix directly (their distribution over the lanes is not specified); S and
// P go through 8 x BK threadgroup scratch per simdgroup for the masking and the exponentials. Everything is f32 (a
// half KV cache is widened as it is staged, a q8 one widened and multiplied by its row scales). Summation order differs from attention_t, so results agree to rounding
// (kernel_tests.py checks it against a float64 reference; the Metal tuner against attention_t on the device).

struct FlashParams {
    int nHeads;
    int nKvHeads;
    int pos0;        // position of the chunk's first token
    int ntok;        // tokens in the chunk
    float scale;
    int qStride;     // q row stride (nHeads * D)
    int kStride;     // key / value cache row stride (nKvHeads * D; q8: + 4 * nKvHeads bytes of scales)
    int oStride;     // output row stride (nHeads * D)
    int window;      // sliding-window length, 0 = full causal attention
    float softcap;   // > 0: score = tanh(score / softcap) * softcap
};

template <typename KV, int D>
kernel void attention_fa(device const float *q [[buffer(0)]],
                         device const KV *kc [[buffer(1)]],
                         device const KV *vc [[buffer(2)]],
                         device float *out [[buffer(3)]],
                         constant FlashParams &p [[buffer(4)]],
                         uint2 tgp [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int BK = 4096 / D;   // keys per block: one block of K or V rows is 16 KB
    constexpr int NI = D / 8;      // 8-wide slices of the head
    constexpr int NJ = BK / 8;     // 8-key slices of a block
    threadgroup float kvs[BK * D];
    threadgroup float sb[FA_THREADS / 32][8 * BK];
    threadgroup float st[FA_THREADS / 32][64];

    const int h = int(tgp.x);
    const int kvh = h / (p.nHeads / p.nKvHeads);
    const int t0 = int(tgp.y) * FA_ROWS;
    const int r0 = t0 + int(sg) * 8;                  // this simdgroup's first row (token of the chunk)
    const int tEnd = min(p.ntok, t0 + FA_ROWS);
    const int firstKey = p.window > 0 ? max(0, p.pos0 + t0 + 1 - p.window) : 0;
    const int lastKey = p.pos0 + tEnd;                // one past the last key any row sees
    device const KV *kb = kc + kvh * D;
    device const KV *vb = vc + kvh * D;

    // the simdgroup's 8 query rows as NI 8 x 8 slices; rows past the chunk are zeros
    simdgroup_float8x8 qm[NI];
    for (int i = 0; i < NI; i++) {
        for (int e = lane; e < 64; e += 32) {
            const int t = r0 + e / 8;
            st[sg][e] = t < p.ntok ? q[(ulong)t * p.qStride + h * D + i * 8 + e % 8] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(qm[i], &st[sg][0], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_float8x8 om[NI];
    for (int i = 0; i < NI; i++) om[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m[8];
    float l[8];
    for (int r = 0; r < 8; r++) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }

    for (int k0 = firstKey / BK * BK; k0 < lastKey; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup); // the previous block's V is no longer read
        for (int e = tid; e < BK * D; e += FA_THREADS) {
            const int key = k0 + e / D;
            kvs[e] = key < lastKey ? float(kb[(ulong)key * p.kStride + e % D])
                                     * kv_scale(kc + (ulong)key * p.kStride, p.kStride, p.nKvHeads, kvh) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T: K is loaded transposed, slice (dims i*8.., keys j*8..)
        for (int j = 0; j < NJ; j++) {
            simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int i = 0; i < NI; i++) {
                simdgroup_float8x8 kt;
                simdgroup_load(kt, &kvs[j * 8 * D + i * 8], D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qm[i], kt, s);
            }
            simdgroup_store(s, &sb[sg][j * 8], BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax, row by row; the row rescale factors go on the diagonal of st
        for (int r = 0; r < 8; r++) {
            const int t = r0 + r;
            const int len = p.pos0 + t + 1;
            const int first = p.window > 0 ? max(0, len - p.window) : 0;
            float mx = -INFINITY;
            for (int c = lane; c < BK; c += 32) {
                const int key = k0 + c;
                float v = sb[sg][r * BK + c] * p.scale;
                if (p.softcap > 0.0f) v = softcap_f(v, p.softcap);
                if (t >= p.ntok || key < first || key >= len) v = -INFINITY;
                sb[sg][r * BK + c] = v;
                mx = max(mx, v);
            }
            mx = simd_max(mx);
            const float mNew = max(m[r], mx);
            float sum = 0.0f;
            for (int c = lane; c < BK; c += 32) {
                const float e = mNew == -INFINITY ? 0.0f : exp(sb[sg][r * BK + c] - mNew);
                sb[sg][r * BK + c] = e;
                sum += e;
            }
            sum = simd_sum(sum);
            const float alpha = m[r] == -INFINITY ? 0.0f : exp(m[r] - mNew);
            l[r] = l[r] * alpha + sum;
            m[r] = mNew;
            if (lane < 8) st[sg][r * 8 + lane] = int(lane) == r ? alpha : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dg;
        simdgroup_load(dg, &st[sg][0], 8);
        for (int i = 0; i < NI; i++) simdgroup_multiply(om[i], dg, om[i]);
        // V block, then O += P V
        threadgroup_barrier(mem_flags::mem_threadgroup); // every simdgroup is done with the K block
        for (int e = tid; e < BK * D; e += FA_THREADS) {
            const int key = k0 + e / D;
            kvs[e] = key < lastKey ? float(vb[(ulong)key * p.kStride + e % D])
                                     * kv_scale(vc + (ulong)key * p.kStride, p.kStride, p.nKvHeads, kvh) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NJ; j++) {
            simdgroup_float8x8 pm;
            simdgroup_load(pm, &sb[sg][j * 8], BK);
            for (int i = 0; i < NI; i++) {
                simdgroup_float8x8 vm;
                simdgroup_load(vm, &kvs[j * 8 * D + i * 8], D);
                simdgroup_multiply_accumulate(om[i], pm, vm, om[i]);
            }
        }
    }
    // O / l, written row by row for the rows inside the chunk
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8) {
        for (int r = 0; r < 8; r++) st[sg][r * 8 + lane] = int(lane) == r && l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dg;
    simdgroup_load(dg, &st[sg][0], 8);
    for (int i = 0; i < NI; i++) {
        simdgroup_float8x8 o;
        simdgroup_multiply(o, dg, om[i]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(o, &st[sg][0], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = lane; e < 64; e += 32) {
            const int t = r0 + e / 8;
            if (t < p.ntok) out[(ulong)t * p.oStride + h * D + i * 8 + e % 8] = st[sg][e];
        }
    }
}

template [[host_name("attention_fa_d64")]] kernel decltype(attention_fa<float, 64>) attention_fa<float, 64>;
template [[host_name("attention_fa_d64_h")]] kernel decltype(attention_fa<half, 64>) attention_fa<half, 64>;
template [[host_name("attention_fa_d128")]] kernel decltype(attention_fa<float, 128>) attention_fa<float, 128>;
template [[host_name("attention_fa_d128_h")]] kernel decltype(attention_fa<half, 128>) attention_fa<half, 128>;
template [[host_name("attention_fa_d256")]] kernel decltype(attention_fa<float, 256>) attention_fa<float, 256>;
template [[host_name("attention_fa_d256_h")]] kernel decltype(attention_fa<half, 256>) attention_fa<half, 256>;
template [[host_name("attention_fa_d64_q8")]] kernel decltype(attention_fa<char, 64>) attention_fa<char, 64>;
template [[host_name("attention_fa_d128_q8")]] kernel decltype(attention_fa<char, 128>) attention_fa<char, 128>;
template [[host_name("attention_fa_d256_q8")]] kernel decltype(attention_fa<char, 256>) attention_fa<char, 256>;

// ---------------------------------------------------------------------------------------------- 10. pipelined greedy decoding
// A greedy decode step can be queued before the CPU has seen the previous token: argmax writes the chosen id into a
// small ring of token slots and the next step's embed_token reads it from there, so the GPU goes from step to step
// without waiting for the CPU (MetalBackend.GreedyPipe).

struct EmbedParams {
    int cols;        // model dim
    int slot;        // token ring slot to read
    float scale;     // the family's embedding multiplier (Gemma: sqrt(dim)), 1 otherwise
};

// 32 weights of quantization block b of row grow, as 8 float4 (any format of the embedding table)
template <int QT>
static inline void dequant_block(device const uchar *wq, device const float *ws, ulong grow, int nb, int b,
                                 thread float4 *w) {
    if (QT == 4) {
        device const uchar4 *q = (device const uchar4 *)(wq + grow * nb * 16 + b * 16);
        const float s = ws[grow * nb + b];
        for (int k = 0; k < 4; k++) {
            uchar4 v = q[k];
            w[k] = (float4(v & uchar4(15)) - 8.0f) * s;
            w[k + 4] = (float4(v >> uchar4(4)) - 8.0f) * s;
        }
    } else if (QT == 8) {
        device const char4 *q = (device const char4 *)(wq + grow * nb * 32 + b * 32);
        const float s = ws[grow * nb + b];
        for (int k = 0; k < 8; k++) w[k] = float4(q[k]) * s;
    } else if (QT >= GG_FIRST) {
        gg_unit<QT>(wq, grow, nb, b, w);
    } else if (QT == 6) {
        q6k_sub_block(wq, grow, (nb + 7) >> 3, 2 * b, w);
        q6k_sub_block(wq, grow, (nb + 7) >> 3, 2 * b + 1, w + 4);
    } else {
        q2k_sub_block(wq, grow, (nb + 7) >> 3, 2 * b, w);
        q2k_sub_block(wq, grow, (nb + 7) >> 3, 2 * b + 1, w + 4);
    }
}

// x[0, :] = scale * embedding row tok[slot]. One threadgroup; threads walk the blocks.
template <int QT>
kernel void embed_token(device const uchar *wq [[buffer(0)]],
                        device const float *ws [[buffer(1)]],
                        device const int *tok [[buffer(2)]],
                        device float *x [[buffer(3)]],
                        constant EmbedParams &p [[buffer(4)]],
                        ushort tid [[thread_index_in_threadgroup]],
                        ushort tpg [[threads_per_threadgroup]]) {
    const int nb = p.cols / QK;
    const ulong row = (ulong)tok[p.slot];
    for (int b = tid; b < nb; b += tpg) {
        float4 w[8];
        dequant_block<QT>(wq, ws, row, nb, b, w);
        device float4 *o = (device float4 *)(x + b * QK);
        for (int k = 0; k < 8; k++) o[k] = w[k] * p.scale;
    }
}

struct ArgmaxParams {
    int n;           // candidates: the valid token ids
    int slot;        // token ring slot to write
};

// tok[slot] = the index of the largest of v[0 .. n), the lowest index among equal maxima (as Sampler's argmax).
// One threadgroup of ROW_THREADS threads.
kernel void argmax(device const float *v [[buffer(0)]],
                   device int *tok [[buffer(1)]],
                   constant ArgmaxParams &p [[buffer(2)]],
                   ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float bv[ROW_THREADS];
    threadgroup int bi[ROW_THREADS];
    float best = -INFINITY;
    int idx = p.n;
    for (int i = tid; i < p.n; i += ROW_THREADS) {
        const float a = v[i];
        if (a > best) {   // strictly greater: within a thread the lower index is kept
            best = a;
            idx = i;
        }
    }
    bv[tid] = best;
    bi[tid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = ROW_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float o = bv[tid + s];
            const int oi = bi[tid + s];
            if (o > bv[tid] || (o == bv[tid] && oi < bi[tid])) {
                bv[tid] = o;
                bi[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) tok[p.slot] = bi[0] < p.n ? bi[0] : 0;
}

typedef decltype(embed_token<4>) embed_token_t;
template [[host_name("embed_token_q4")]] kernel embed_token_t embed_token<4>;
template [[host_name("embed_token_q8")]] kernel embed_token_t embed_token<8>;
template [[host_name("embed_token_q2k")]] kernel embed_token_t embed_token<2>;
template [[host_name("embed_token_q6k")]] kernel embed_token_t embed_token<6>;
template [[host_name("embed_token_q4g")]] kernel embed_token_t embed_token<GG_Q4_0>;
template [[host_name("embed_token_q8g")]] kernel embed_token_t embed_token<GG_Q8_0>;
template [[host_name("embed_token_q4k")]] kernel embed_token_t embed_token<GG_Q4_K>;
template [[host_name("embed_token_q5k")]] kernel embed_token_t embed_token<GG_Q5_K>;

// ---------------------------------------------------------------------------------------------- 11. Qwen3-Next
// The pieces of Qwen3NextMetalBackend that the generic kernels lack (docs/14.14). The projections, norms, RoPE,
// attention and experts reuse the kernels above; these add the Gated DeltaNet (gdn_conv, gdn_rule, gdn_norm_gate) and
// the two gates (sigmoid_mul: attention output * sigmoid(gate); shared_gate_add: x += sigmoid(w . xb) * shared expert).
// Like Mamba's scan, the delta rule runs the tokens of a chunk sequentially inside each thread: prefill is the token
// loop of decode, not the chunked (UT / WY) form of the CPU backend (the two agree to float rounding).

#define GDN_MAX_PER 8           // key head width held per lane, in units of 32 (linear_key_head_dim <= 256)
#define GDN_COLS 4              // state columns (value dims) per gdn_rule threadgroup: one simdgroup each
#define GDN_MAX_CONV 8          // conv taps held in registers

struct GdnConvParams {
    int channels;   // conv channels: q, k and v of all heads
    int taps;       // <= GDN_MAX_CONV
    int ntok;
    int inStride;   // floats per token of the input (the fused projection row)
    int outStride;  // floats per token of the output
};

// y[t, c] = silu(sum_k w[c, k] * window[k]) with window = (the conv state of channel c, x[t, c]): Qwen3-Next's
// depthwise causal conv (no bias). One thread per channel; the state (kernel - 1 previous inputs, oldest first) is
// read once, carried in registers over the chunk and written back.
kernel void gdn_conv(device const float *x [[buffer(0)]],
                     device const float *w [[buffer(1)]],
                     device float *state [[buffer(2)]],
                     device float *y [[buffer(3)]],
                     constant GdnConvParams &p [[buffer(4)]],
                     uint gid [[thread_position_in_grid]]) {
    const int c = int(gid);
    if (c >= p.channels) return;
    const int kc = p.taps;
    device float *st = state + (ulong)c * (kc - 1);
    device const float *wc = w + (ulong)c * kc;
    float win[GDN_MAX_CONV - 1];
    float wr[GDN_MAX_CONV];
    for (int k = 0; k < kc - 1; k++) win[k] = st[k];
    for (int k = 0; k < kc; k++) wr[k] = wc[k];
    for (int t = 0; t < p.ntok; t++) {
        const float u = x[(ulong)t * p.inStride + c];
        float sum = 0.0f;
        for (int k = 0; k < kc - 1; k++) sum += wr[k] * win[k];
        sum += wr[kc - 1] * u;
        for (int k = 0; k + 1 < kc - 1; k++) win[k] = win[k + 1];
        if (kc > 1) win[kc - 2] = u;
        y[(ulong)t * p.outStride + c] = sum / (1.0f + precise::exp(-sum));
    }
    for (int k = 0; k < kc - 1; k++) st[k] = win[k];
}

struct GdnRuleParams {
    int keyHeads;
    int valueHeads;
    int keyDim;     // multiple of 32, <= 32 * GDN_MAX_PER
    int valueDim;
    int ntok;
    int qkvStride;  // floats per token of the conv output [q | k | v]
    int projStride; // floats per token of the projection (b and a of the token are read from it)
    int bOff;       // offset of b in a projection row; a follows after valueHeads
    float eps;      // of the q / k L2 norm
};

// The gated delta rule of value head h = tgp.y, state column j (value dim) = tgp.x * GDN_COLS + simdgroup:
//   q, k = l2norm(q), l2norm(k) of the head's key head, q *= keyDim^-0.5;  beta = sigmoid(b);  g = A * softplus(a + dt_bias)
//   s *= exp(g);   d = (v_j - s . k) * beta;   s += d k;   o_j = s . q
// where s is column j of the head's [keyDim, valueDim] state, kept transposed in memory ([head][j][keyDim]) and in
// registers: lane l holds rows l, l + 32, ... (keyDim / 32 of them). The dot products are simd_sums; every simdgroup
// recomputes the two L2 norms of its token (two more simd_sums), which saves a pass over q and k. Tokens run in order.
kernel void gdn_rule(device const float *qkv [[buffer(0)]],
                     device const float *proj [[buffer(1)]],
                     device const float *A [[buffer(2)]],
                     device const float *dtBias [[buffer(3)]],
                     device float *state [[buffer(4)]],
                     device float *out [[buffer(5)]],
                     constant GdnRuleParams &p [[buffer(6)]],
                     uint2 tgp [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]]) {
    const int h = int(tgp.y);
    const int j = int(tgp.x) * GDN_COLS + sg;
    if (j >= p.valueDim) return;   // the whole simdgroup leaves together: no barrier below
    const int dk = p.keyDim;
    const int per = dk / 32;
    const int kh = h / (p.valueHeads / p.keyHeads);
    const int kw = p.keyHeads * dk;
    const float scale = rsqrt(float(dk));
    device float *col = state + ((ulong)h * p.valueDim + j) * dk;
    float s[GDN_MAX_PER];
    for (int r = 0; r < per; r++) s[r] = col[lane + 32 * r];
    const float a0 = A[h];
    const float db = dtBias[h];
    for (int t = 0; t < p.ntok; t++) {
        device const float *row = qkv + (ulong)t * p.qkvStride;
        device const float *pr = proj + (ulong)t * p.projStride;
        float q[GDN_MAX_PER];
        float k[GDN_MAX_PER];
        float qq = 0.0f;
        float kk = 0.0f;
        for (int r = 0; r < per; r++) {
            q[r] = row[kh * dk + lane + 32 * r];
            k[r] = row[kw + kh * dk + lane + 32 * r];
            qq += q[r] * q[r];
            kk += k[r] * k[r];
        }
        const float qn = scale * rsqrt(simd_sum(qq) + p.eps);
        const float kn = rsqrt(simd_sum(kk) + p.eps);
        const float b = pr[p.bOff + h];
        const float araw = pr[p.bOff + p.valueHeads + h] + db;
        const float beta = 1.0f / (1.0f + precise::exp(-b));
        const float sp = araw > SOFTPLUS_LINEAR ? araw : precise::log(1.0f + precise::exp(araw));
        const float gamma = precise::exp(a0 * sp);
        float kv = 0.0f;
        for (int r = 0; r < per; r++) {
            k[r] *= kn;
            s[r] *= gamma;
            kv += s[r] * k[r];
        }
        kv = simd_sum(kv);
        const float d = (row[2 * kw + h * p.valueDim + j] - kv) * beta;
        float o = 0.0f;
        for (int r = 0; r < per; r++) {
            s[r] += d * k[r];
            o += s[r] * (q[r] * qn);
        }
        o = simd_sum(o);
        if (lane == 0) out[(ulong)t * p.valueHeads * p.valueDim + h * p.valueDim + j] = o;
    }
    for (int r = 0; r < per; r++) col[lane + 32 * r] = s[r];
}

struct GdnNormParams {
    int valueDim;
    int valueHeads;
    int projStride;
    int zOff;       // offset of z in a projection row
    float eps;
};

// o[row] = rmsnorm(o[row]) * w * silu(z[row]) in place for the rows (token, value head) of valueDim floats: the
// DeltaNet's gated RMSNorm (its gain is applied as is, not 1 + w). One threadgroup per row.
kernel void gdn_norm_gate(device float *o [[buffer(0)]],
                          device const float *proj [[buffer(1)]],
                          device const float *w [[buffer(2)]],
                          constant GdnNormParams &p [[buffer(3)]],
                          uint tg [[threadgroup_position_in_grid]],
                          ushort tid [[thread_position_in_threadgroup]],
                          ushort tpg [[threads_per_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]],
                          ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[32];
    const int t = int(tg) / p.valueHeads;
    const int h = int(tg) - t * p.valueHeads;
    device float *ov = o + (ulong)tg * p.valueDim;
    device const float *z = proj + (ulong)t * p.projStride + p.zOff + h * p.valueDim;
    float ss = 0.0f;
    for (int i = tid; i < p.valueDim; i += tpg) {
        const float v = ov[i];
        ss += v * v;
    }
    ss = tg_sum(ss, buf, sg, lane, nsg);
    const float scale = rsqrt(ss / p.valueDim + p.eps);
    for (int i = tid; i < p.valueDim; i += tpg) {
        const float zi = z[i];
        ov[i] = ov[i] * scale * w[i] * (zi / (1.0f + precise::exp(-zi)));
    }
}

struct GateParams {
    int n;          // elements (sigmoid_mul) or the row width (shared_gate_add)
};

// a[i] *= sigmoid(g[i]) for i < n: the output gate of Qwen3-Next's full attention (before o_proj).
kernel void sigmoid_mul(device float *a [[buffer(0)]],
                        device const float *g [[buffer(1)]],
                        constant GateParams &p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.n) return;
    a[i] = a[i] * (1.0f / (1.0f + precise::exp(-g[i])));
}

// x[t] += sigmoid(dot(w, xb[t])) * y[t] for the token t = threadgroup: the shared expert's output y scaled by its gate
// (shared_expert_gate, one row of n = dim floats). One threadgroup per token.
kernel void shared_gate_add(device const float *xb [[buffer(0)]],
                            device const float *w [[buffer(1)]],
                            device const float *y [[buffer(2)]],
                            device float *x [[buffer(3)]],
                            constant GateParams &p [[buffer(4)]],
                            uint tg [[threadgroup_position_in_grid]],
                            ushort tid [[thread_position_in_threadgroup]],
                            ushort tpg [[threads_per_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]],
                            ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[32];
    device const float *xr = xb + (ulong)tg * p.n;
    float d = 0.0f;
    for (int i = tid; i < p.n; i += tpg) d += w[i] * xr[i];
    d = tg_sum(d, buf, sg, lane, nsg);
    const float s = 1.0f / (1.0f + precise::exp(-d));
    device const float *yr = y + (ulong)tg * p.n;
    device float *out = x + (ulong)tg * p.n;
    for (int i = tid; i < p.n; i += tpg) out[i] += s * yr[i];
}
// ---------------------------------------------------------------------------------------------- 12. half-weight MMA prefill
// The *_hmma kernels are a second form of the simdgroup-matrix prefill matmuls of section 2b (matmul_mma, matmul3_mma,
// matmul_swiglu_mma, moe_up_mma, moe_down_mma), with the same output tile (MMA_BR = 64 rows x MMA_BT = 32 tokens per
// threadgroup of 4 simdgroups, each owning a 32 x 16 quarter = 8 accumulators), the same buffers, parameters and grids,
// so the host picks either per matrix shape (MetalTuner times both on the device and keeps the faster if they agree).
// What differs:
//   - the dequantized weight tile is stored as half (4 KB instead of 8 KB): q * scale rounded once to 11 significant
//     bits, as llama.cpp's kernel_mul_mm does; activations stay f32 and the accumulators are f32 (the multiply is
//     simdgroup_float8x8 x simdgroup_half8x8 -> float), so the only new rounding is the weight's;
//   - both tiles are kept as contiguous 8 x 8 blocks (block (rb, kb) at (rb * 4 + kb) * 64): every simdgroup_load reads
//     128 or 256 contiguous bytes instead of 8 rows 128 bytes apart (rows at a 128-byte stride fall into the same
//     threadgroup-memory banks);
//   - the next block's weight bytes and activations are loaded into registers before the multiply-accumulates of the
//     current one (one tile in threadgroup memory, so still two barriers per 32-wide k step, but the device latency
//     overlaps the arithmetic);
//   - 8-token blocks past the tile's last valid token are not multiplied (MoE tiles hold ~n * topK / experts slots,
//     often 16 of 32; the last tile of a chunk is partial too);
//   - full tiles of matmul (no bias / soft-cap) and of matmul3's q and k ranges (no bias) are stored straight from the
//     accumulators to device memory (accumulating onto out by starting from its tile), skipping the staging pass.
// Total threadgroup memory 8 KB (the f32 result tile of the staging epilogue aliases both input tiles), so up to four
// threadgroups share a GPU core's 32 KB where the f32 kernels (12 KB) fit two.

#define HM_SMEM 2048        // floats of threadgroup memory per *_hmma threadgroup: half weights 64 x 32 + f32 x 32 x 32

// 16 weights (k = 16h .. 16h + 15 of the 32-weight block / unit u) of row grow of a GGUF tensor, dequantized as gg_unit
// does. The legacy Q4_0 keeps llama.cpp's nibble order (low nibbles = weights 0..15), so for it half h is the low and
// high nibbles of bytes 8h .. 8h + 7 = weights 8h .. 8h + 7 (w[0..1]) and 16 + 8h .. 16 + 8h + 7 (w[2..3]).
template <int QT>
static inline void gg_half(device const uchar *wq, ulong grow, int nb, int u, int h, thread float4 *w) {
    if (QT == GG_Q4_0 || QT == GG_Q8_0) {
        constexpr int BYTES = QT == GG_Q4_0 ? 18 : 34;
        device const uchar *blk = wq + (grow * nb + u) * BYTES;
        const float d = float(*(device const half *)blk);
        device const ushort *p = (device const ushort *)(blk + 2);
        if (QT == GG_Q4_0) {
            for (int k = 0; k < 2; k++) {
                const uchar4 v = as_type<uchar4>(uint(p[4 * h + 2 * k]) | (uint(p[4 * h + 2 * k + 1]) << 16));
                w[k] = (float4(v & uchar4(15)) - 8.0f) * d;
                w[2 + k] = (float4(v >> uchar4(4)) - 8.0f) * d;
            }
        } else {
            for (int k = 0; k < 4; k++) {
                w[k] = float4(as_type<char4>(uint(p[8 * h + 2 * k]) | (uint(p[8 * h + 2 * k + 1]) << 16))) * d;
            }
        }
    } else {
        constexpr bool Q5 = QT == GG_Q5_K;
        constexpr int BYTES = Q5 ? 176 : 144;
        const int nsb = nb >> 3;
        device const uchar *blk = wq + (grow * nsb + (u >> 3)) * BYTES;
        const int j = u & 7;
        const float2 dd = float2(as_type<half2>(*(device const uint *)blk));
        device const uchar *sc = blk + 4;
        uint s;
        uint m;
        if (j < 4) {
            s = sc[j] & 63;
            m = sc[j + 4] & 63;
        } else {
            s = (sc[j + 4] & 15) | ((uint(sc[j - 4]) >> 6) << 4);
            m = (sc[j + 4] >> 4) | ((uint(sc[j]) >> 6) << 4);
        }
        const float dl = dd.x * float(s);
        const float ml = dd.y * float(m);
        const uchar sh = uchar((j & 1) << 2);
        device const uchar4 *q = (device const uchar4 *)(blk + (Q5 ? 48 : 16) + (j >> 1) * 32);
        device const uchar4 *hb = (device const uchar4 *)(blk + 16);
        for (int k = 0; k < 4; k++) {
            uchar4 v = (q[4 * h + k] >> uchar4(sh)) & uchar4(15);
            if (Q5) v |= ((hb[4 * h + k] >> uchar4(j)) & uchar4(1)) << uchar4(4);
            w[k] = float4(v) * dl - ml;
        }
    }
}

// The raw bytes of this thread's share of block b for the engine's own formats (Q4_0: 8 quant bytes, Q8_0: 16, plus
// the f32 scale), loaded one k step ahead; the other formats are decoded straight from device memory by hm_weights.
template <int QT>
static inline void hm_fetch(device const uchar *rwq, device const float *rws, ulong grow, int nb, int b, int h,
                            thread uchar4 *rq, thread float &rs) {
    if (QT == 4) {
        device const uchar4 *q = (device const uchar4 *)(rwq + (grow * nb + b) * 16 + h * 8);
        rq[0] = q[0];
        rq[1] = q[1];
        rs = rws[grow * nb + b];
    } else if (QT == 8) {
        device const uchar4 *q = (device const uchar4 *)(rwq + (grow * nb + b) * 32 + h * 16);
        for (int k = 0; k < 4; k++) rq[k] = q[k];
        rs = rws[grow * nb + b];
    }
}

// Thread (row, h) writes its 16 weights of block b, as half, into the row's 8-wide k blocks of the weight tile;
// wd = sa + (row >> 3) * 256 + (row & 7) * 8 is the row inside its 8-row band (k block kb at + kb * 64).
// Q4_0 (and GGUF Q4_0): bytes 8h.. hold weights 8h.. (low nibbles, k block h) and 16 + 8h.. (high, k block 2 + h);
// every other format: weights 16h .. 16h + 15 = k blocks 2h and 2h + 1.
template <int QT>
static inline void hm_weights(device const uchar *rwq, ulong grow, int nb, int b, int h, thread const uchar4 *rq,
                              float rs, threadgroup half *wd) {
    if (QT == 4) {
        for (int k = 0; k < 2; k++) {
            *(threadgroup half4 *)(wd + h * 64 + k * 4) = half4((float4(rq[k] & uchar4(15)) - 8.0f) * rs);
            *(threadgroup half4 *)(wd + (2 + h) * 64 + k * 4) = half4((float4(rq[k] >> uchar4(4)) - 8.0f) * rs);
        }
    } else if (QT == 8) {
        for (int k = 0; k < 4; k++) {
            *(threadgroup half4 *)(wd + (2 * h + (k >> 1)) * 64 + (k & 1) * 4) = half4(float4(as_type<char4>(rq[k])) * rs);
        }
    } else {
        float4 w[4];
        if (QT == 2) {
            q2k_sub_block(rwq, grow, (nb + 7) >> 3, 2 * b + h, w);
        } else if (QT == 6) {
            q6k_sub_block(rwq, grow, (nb + 7) >> 3, 2 * b + h, w);
        } else {
            gg_half<QT>(rwq, grow, nb, b, h, w);
        }
        const int kb0 = QT == GG_Q4_0 ? h : 2 * h;
        const int kb1 = QT == GG_Q4_0 ? 2 + h : 2 * h + 1;
        *(threadgroup half4 *)(wd + kb0 * 64) = half4(w[0]);
        *(threadgroup half4 *)(wd + kb0 * 64 + 4) = half4(w[1]);
        *(threadgroup half4 *)(wd + kb1 * 64) = half4(w[2]);
        *(threadgroup half4 *)(wd + kb1 * 64 + 4) = half4(w[3]);
    }
}

// The k loop of every *_hmma kernel. Thread tid dequantizes 16 weights of tile row tid >> 1 (its global row rgrow of
// rwq / rws, clamped by the caller for a partial row tile) and stages 8 activations of tile token tid >> 2 from xrow
// (clamped to the last valid token; ntv tokens of the tile are valid). With `direct` the accumulators go straight to
// dout (row stride ldo; with accLoad they start from dout's tile, i.e. out += W x); otherwise the f32 result tile
// is left in smem as sc[t * MMA_BR + r] for the caller's epilogue. Must be called by all 128 threads (barriers).
template <int QT>
static inline void hm_core(device const uchar *rwq, device const float *rws, ulong rgrow, int nb,
                           device const float *xrow, int ntv, threadgroup float *smem,
                           bool direct, device float *dout, ulong ldo, bool accLoad, ushort tid, ushort sg) {
    threadgroup half *sa = (threadgroup half *)smem;
    threadgroup float *sb = smem + MMA_BR * QK / 2;
    const ushort sr = sg & 1;   // rows 32 sr .. 32 sr + 31 of the tile
    const ushort st = sg >> 1;  // tokens 16 st .. 16 st + 15
    const int ni = clamp((ntv - int(st) * 16 + 7) / 8, 0, 2);   // 8-token blocks of the quarter holding valid tokens
    simdgroup_float8x8 acc[8];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            if (direct && accLoad) {
                simdgroup_load(acc[i * 4 + j], dout + (ulong)(st * 16 + i * 8) * ldo + sr * 32 + j * 8, ldo);
            } else {
                acc[i * 4 + j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
    }
    const int lr = tid >> 1;
    const int h = tid & 1;
    const int lt = tid >> 2;
    const int lp = tid & 3;
    threadgroup half *wd = sa + (lr >> 3) * 256 + (lr & 7) * 8;
    threadgroup float4 *xd = (threadgroup float4 *)(sb + ((lt >> 3) * 4 + lp) * 64 + (lt & 7) * 8);
    device const float4 *xr = (device const float4 *)xrow + lp * 2;
    uchar4 rq[4];
    for (int k = 0; k < 4; k++) rq[k] = uchar4(0);
    float rs = 0.0f;
    hm_fetch<QT>(rwq, rws, rgrow, nb, 0, h, rq, rs);
    float4 x0 = xr[0];
    float4 x1 = xr[1];
    for (int b = 0; b < nb; b++) {
        hm_weights<QT>(rwq, rgrow, nb, b, h, rq, rs, wd);
        xd[0] = x0;
        xd[1] = x1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (b + 1 < nb) {
            hm_fetch<QT>(rwq, rws, rgrow, nb, b + 1, h, rq, rs);
            x0 = xr[(b + 1) * 8];
            x1 = xr[(b + 1) * 8 + 1];
        }
        if (ni > 0) {
            for (int kk = 0; kk < 4; kk++) {
                simdgroup_half8x8 w[4];
                for (int j = 0; j < 4; j++) simdgroup_load(w[j], sa + ((sr * 4 + j) * 4 + kk) * 64, 8, ulong2(0, 0), true);
                for (int i = 0; i < 2; i++) {
                    if (i < ni) {
                        simdgroup_float8x8 a;
                        simdgroup_load(a, sb + ((st * 2 + i) * 4 + kk) * 64, 8);
                        for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i * 4 + j], a, w[j], acc[i * 4 + j]);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            if (direct) {
                simdgroup_store(acc[i * 4 + j], dout + (ulong)(st * 16 + i * 8) * ldo + sr * 32 + j * 8, ldo);
            } else {
                simdgroup_store(acc[i * 4 + j], smem + (st * 16 + i * 8) * MMA_BR + sr * 32 + j * 8, MMA_BR);
            }
        }
    }
    if (!direct) threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <int QT>
kernel void matmul_hmma(device const uchar *wq [[buffer(0)]],
                        device const float *ws [[buffer(1)]],
                        device const float *x [[buffer(2)]],
                        device float *out [[buffer(3)]],
                        device const float *nw [[buffer(4)]],
                        device const float *bias [[buffer(5)]],
                        constant MatmulParams &p [[buffer(6)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[HM_SMEM];
    const int row0 = int(tgp.x) * MMA_BR;
    const int t0 = int(tgp.y) * MMA_BT;
    const int ntv = min(MMA_BT, p.ntok - t0);
    const int nb = p.cols / QK;
    const ulong grow = (ulong)min(row0 + int(tid >> 1), p.rows - 1);
    device const float *xrow = x + (ulong)(t0 + min(int(tid >> 2), ntv - 1)) * p.cols;
    const bool direct = ntv == MMA_BT && row0 + MMA_BR <= p.rows && !p.hasBias && !(p.softcap > 0.0f);
    hm_core<QT>(wq, ws, grow, nb, xrow, ntv, smem, direct, out + (ulong)t0 * p.rows + row0, (ulong)p.rows,
                p.accumulate != 0, tid, sg);
    if (direct) return;
    for (int i = tid; i < MMA_BR * MMA_BT; i += MM_THREADS) {
        const int t = i / MMA_BR, r = i % MMA_BR;
        if (t < ntv && row0 + r < p.rows) {
            const float v = p.hasBias ? smem[t * MMA_BR + r] + bias[row0 + r] : smem[t * MMA_BR + r];
            const ulong o = (ulong)(t0 + t) * p.rows + row0 + r;
            out[o] = p.accumulate ? out[o] + v : (p.softcap > 0.0f ? softcap_f(v, p.softcap) : v);
        }
    }
}

template <int QT, typename VT>
kernel void matmul3_hmma(device const uchar *wq0 [[buffer(0)]],
                         device const float *ws0 [[buffer(1)]],
                         device const uchar *wq1 [[buffer(2)]],
                         device const float *ws1 [[buffer(3)]],
                         device const uchar *wq2 [[buffer(4)]],
                         device const float *ws2 [[buffer(5)]],
                         device const float *x [[buffer(6)]],
                         device float *out0 [[buffer(7)]],
                         device float *out1 [[buffer(8)]],
                         device VT *out2 [[buffer(9)]],
                         device const float *nw [[buffer(10)]],
                         device const float *b0 [[buffer(11)]],
                         device const float *b1 [[buffer(12)]],
                         device const float *b2 [[buffer(13)]],
                         constant Matmul3Params &p [[buffer(14)]],
                         uint2 tgp [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[HM_SMEM];
    const int total = p.rows0 + p.rows1 + p.rows2;
    const int row0 = int(tgp.x) * MMA_BR;
    const int t0 = int(tgp.y) * MMA_BT;
    const int ntv = min(MMA_BT, p.ntok - t0);
    const int nb = p.cols / QK;
    const int lrow = min(row0 + int(tid >> 1), total - 1);
    device const uchar *rwq;
    device const float *rws;
    ulong grow;
    if (lrow < p.rows0) { rwq = wq0; rws = ws0; grow = (ulong)lrow; }
    else if (lrow < p.rows0 + p.rows1) { rwq = wq1; rws = ws1; grow = (ulong)(lrow - p.rows0); }
    else { rwq = wq2; rws = ws2; grow = (ulong)(lrow - p.rows0 - p.rows1); }
    device const float *xrow = x + (ulong)(t0 + min(int(tid >> 2), ntv - 1)) * p.cols;
    // a full tile inside the q or the k range (no bias) is stored straight from the accumulators
    const bool full = ntv == MMA_BT && !p.hasBias;
    const bool inQ = full && row0 + MMA_BR <= p.rows0;
    const bool inK = full && row0 >= p.rows0 && row0 + MMA_BR <= p.rows0 + p.rows1;
    device float *dst = inQ ? out0 + (ulong)t0 * p.rows0 + row0 : out1 + (ulong)t0 * p.rows1 + (row0 - p.rows0);
    hm_core<QT>(rwq, rws, grow, nb, xrow, ntv, smem, inQ || inK, dst, (ulong)(inQ ? p.rows0 : p.rows1), false, tid, sg);
    if (inQ || inK) return;
    for (int i = tid; i < MMA_BR * MMA_BT; i += MM_THREADS) {
        const int t = i / MMA_BR, r = i % MMA_BR;
        const int row = row0 + r;
        if (t < ntv && row < total) {
            float v = smem[t * MMA_BR + r];
            if (row < p.rows0) {
                if (p.hasBias) v += b0[row];
                out0[(ulong)(t0 + t) * p.rows0 + row] = v;
            } else if (row < p.rows0 + p.rows1) {
                const int g = row - p.rows0;
                if (p.hasBias) v += b1[g];
                out1[(ulong)(t0 + t) * p.rows1 + g] = v;
            } else {
                const int g = row - p.rows0 - p.rows1;
                if (p.hasBias) v += b2[g];
                out2[(ulong)(t0 + t) * p.rows2 + g] = VT(v);
            }
        }
    }
}

// Tile rows [0, 32) = gate units [unit0, unit0 + 32), rows [32, 64) = the matching up rows (as matmul_swiglu_mma).
template <int QT>
kernel void matmul_swiglu_hmma(device const uchar *wq [[buffer(0)]],
                               device const float *ws [[buffer(1)]],
                               device const float *x [[buffer(2)]],
                               device float *out [[buffer(3)]],
                               device const float *nw [[buffer(4)]],
                               constant SwigluParams &p [[buffer(5)]],
                               uint2 tgp [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[HM_SMEM];
    const int unit0 = int(tgp.x) * (MMA_BR / 2);
    const int t0 = int(tgp.y) * MMA_BT;
    const int ntv = min(MMA_BT, p.ntok - t0);
    const int nb = p.cols / QK;
    const int lr = tid >> 1;
    const int unit = min(unit0 + (lr & 31), p.hidden - 1);
    device const uchar *rwq = (QT >= GG_FIRST && lr >= 32) ? (device const uchar *)ws : wq;
    const ulong grow = (ulong)((lr < 32 || QT >= GG_FIRST) ? unit : p.hidden + unit);
    device const float *xrow = x + (ulong)(t0 + min(int(tid >> 2), ntv - 1)) * p.cols;
    hm_core<QT>(rwq, ws, grow, nb, xrow, ntv, smem, false, out, 0, false, tid, sg);
    for (int i = tid; i < (MMA_BR / 2) * MMA_BT; i += MM_THREADS) {
        const int t = i / (MMA_BR / 2), j = i % (MMA_BR / 2);
        if (t < ntv && unit0 + j < p.hidden) {
            const float g = smem[t * MMA_BR + j];
            const float u = smem[t * MMA_BR + 32 + j];
            out[(ulong)(t0 + t) * p.hidden + unit0 + j] = (p.gelu ? gelu_tanh_f(g) : g / (1.0f + exp(-g))) * u;
        }
    }
}

// moe_up_mma / moe_down_mma on the half-weight core: the same tiles of moe_sort, the same outputs.
template <int QT>
kernel void moe_up_hmma(device const uchar *wq [[buffer(0)]],
                        device const float *ws [[buffer(1)]],
                        device const float *x [[buffer(2)]],
                        device const int *order [[buffer(3)]],
                        device const int *tiles [[buffer(4)]],
                        device float *out [[buffer(5)]],
                        constant MoeParams &p [[buffer(6)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[HM_SMEM];
    const int ti = int(tgp.y);
    if (ti >= tiles[0]) return;
    const int e = tiles[1 + 3 * ti];
    const int start = tiles[2 + 3 * ti];
    const int cnt = tiles[3 + 3 * ti];
    const int m = p.rowsPerExpert / 2;
    const int unit0 = int(tgp.x) * (MMA_BR / 2);
    const int nb = p.cols / QK;
    const int lr = tid >> 1;
    const int unit = min(unit0 + (lr & 31), m - 1);
    device const uchar *rwq = (QT >= GG_FIRST && lr >= 32) ? (device const uchar *)ws : wq;
    const ulong grow = QT >= GG_FIRST ? (ulong)e * m + unit : (ulong)e * p.rowsPerExpert + (lr < 32 ? unit : m + unit);
    const int slot = order[start + min(int(tid >> 2), cnt - 1)];
    hm_core<QT>(rwq, ws, grow, nb, x + (ulong)(slot / p.topK) * p.cols, cnt, smem, false, out, 0, false, tid, sg);
    for (int i = tid; i < (MMA_BR / 2) * MMA_BT; i += MM_THREADS) {
        const int t = i / (MMA_BR / 2), j = i % (MMA_BR / 2);
        if (t < cnt && unit0 + j < m) {
            const float g = smem[t * MMA_BR + j];
            const float u = smem[t * MMA_BR + 32 + j];
            out[(ulong)order[start + t] * m + unit0 + j] = g / (1.0f + exp(-g)) * u;
        }
    }
}

template <int QT>
kernel void moe_down_hmma(device const uchar *wq [[buffer(0)]],
                          device const float *ws [[buffer(1)]],
                          device const float *h [[buffer(2)]],
                          device const int *order [[buffer(3)]],
                          device const int *tiles [[buffer(4)]],
                          device const float *rw [[buffer(5)]],
                          device float *ys [[buffer(6)]],
                          constant MoeParams &p [[buffer(7)]],
                          uint2 tgp [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[HM_SMEM];
    const int ti = int(tgp.y);
    if (ti >= tiles[0]) return;
    const int e = tiles[1 + 3 * ti];
    const int start = tiles[2 + 3 * ti];
    const int cnt = tiles[3 + 3 * ti];
    const int row0 = int(tgp.x) * MMA_BR;
    const int nb = p.cols / QK;
    const int row = min(row0 + int(tid >> 1), p.rowsPerExpert - 1);
    const int slot = order[start + min(int(tid >> 2), cnt - 1)];
    hm_core<QT>(wq, ws, (ulong)e * p.rowsPerExpert + row, nb, h + (ulong)slot * p.cols, cnt, smem, false, ys, 0, false,
                tid, sg);
    for (int i = tid; i < MMA_BR * MMA_BT; i += MM_THREADS) {
        const int t = i / MMA_BR, j = i % MMA_BR;
        if (t < cnt && row0 + j < p.rowsPerExpert) {
            const int s = order[start + t];
            ys[(ulong)s * p.rowsPerExpert + row0 + j] = rw[s] * smem[t * MMA_BR + j];
        }
    }
}

typedef decltype(matmul_hmma<4>) matmul_hmma_t;
typedef decltype(matmul3_hmma<4, float>) matmul3_hmma_t;
typedef decltype(matmul3_hmma<4, half>) matmul3_hmma_h_t;
typedef decltype(matmul_swiglu_hmma<4>) matmul_swiglu_hmma_t;
typedef decltype(moe_up_hmma<4>) moe_up_hmma_t;
typedef decltype(moe_down_hmma<4>) moe_down_hmma_t;
#define HM_INSTANTIATE(SUF, QTV)                                                                                  \
    template [[host_name("matmul_hmma_" SUF)]] kernel matmul_hmma_t matmul_hmma<QTV>;                             \
    template [[host_name("matmul3_hmma_" SUF)]] kernel matmul3_hmma_t matmul3_hmma<QTV, float>;                   \
    template [[host_name("matmul3_hmma_" SUF "_h")]] kernel matmul3_hmma_h_t matmul3_hmma<QTV, half>;             \
    template [[host_name("matmul_swiglu_hmma_" SUF)]] kernel matmul_swiglu_hmma_t matmul_swiglu_hmma<QTV>;       \
    template [[host_name("moe_up_hmma_" SUF)]] kernel moe_up_hmma_t moe_up_hmma<QTV>;                             \
    template [[host_name("moe_down_hmma_" SUF)]] kernel moe_down_hmma_t moe_down_hmma<QTV>;
HM_INSTANTIATE("q4", 4)
HM_INSTANTIATE("q8", 8)
HM_INSTANTIATE("q2k", 2)
HM_INSTANTIATE("q6k", 6)
HM_INSTANTIATE("q4g", GG_Q4_0)
HM_INSTANTIATE("q8g", GG_Q8_0)
HM_INSTANTIATE("q4k", GG_Q4_K)
HM_INSTANTIATE("q5k", GG_Q5_K)

// ---------------------------------------------------------------------------------------------- 13. flash prefill, second form
// attention_fb<KV, ST, D>: the online-softmax prefill attention of attention_fa (same grid, parameters, masking and
// math) with a different staging of the keys and values:
//   - a block's K and V rows are staged together: one load pass and two barriers per block instead of two and four;
//   - in ST = half for the f16 and q8 caches, which is exact (f16 values; int8 values, whose per-row scales are kept
//     apart in kq / vq and applied once per key to the block's score column, s_ij * scale_j, and probability column,
//     p_ij * scale_j, instead of to every staged element), and in f32 for the f32 cache: nothing is rounded that
//     attention_fa keeps;
//   - rows padded by 16 bytes (row stride LD = D + 16 / sizeof(ST)), so the 8 rows one simdgroup_load reads do not all
//     start in the same threadgroup-memory bank, as they do with a stride of D (a multiple of 128 bytes);
//   - BK = 8 KB / (D * sizeof(ST)) keys per block (32 at D = 128 with half staging, 16 with f32).
// S = Q K^T and O += P V are f32 x ST multiply-accumulates into f32: P and the accumulators stay f32 (a half P would
// round every probability to 11 bits, ~2.4e-4 relative, for no saving in the staged bytes).

template <typename KV, typename ST, int D>
kernel void attention_fb(device const float *q [[buffer(0)]],
                         device const KV *kc [[buffer(1)]],
                         device const KV *vc [[buffer(2)]],
                         device float *out [[buffer(3)]],
                         constant FlashParams &p [[buffer(4)]],
                         uint2 tgp [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int BK = 8192 / (D * int(sizeof(ST)));
    constexpr int LD = D + 16 / int(sizeof(ST));
    constexpr int NI = D / 8;
    constexpr int NJ = BK / 8;
    threadgroup ST ks[BK * LD];
    threadgroup ST vs[BK * LD];
    threadgroup float kq[BK];
    threadgroup float vq[BK];
    threadgroup float sb[FA_THREADS / 32][8 * BK];
    threadgroup float st[FA_THREADS / 32][64];

    const int h = int(tgp.x);
    const int kvh = h / (p.nHeads / p.nKvHeads);
    const int t0 = int(tgp.y) * FA_ROWS;
    const int r0 = t0 + int(sg) * 8;
    const int tEnd = min(p.ntok, t0 + FA_ROWS);
    const int firstKey = p.window > 0 ? max(0, p.pos0 + t0 + 1 - p.window) : 0;
    const int lastKey = p.pos0 + tEnd;
    device const KV *kb = kc + kvh * D;
    device const KV *vb = vc + kvh * D;

    simdgroup_float8x8 qm[NI];
    for (int i = 0; i < NI; i++) {
        for (int e = lane; e < 64; e += 32) {
            const int t = r0 + e / 8;
            st[sg][e] = t < p.ntok ? q[(ulong)t * p.qStride + h * D + i * 8 + e % 8] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(qm[i], &st[sg][0], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_float8x8 om[NI];
    for (int i = 0; i < NI; i++) om[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m[8];
    float l[8];
    for (int r = 0; r < 8; r++) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }

    for (int k0 = firstKey / BK * BK; k0 < lastKey; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup); // the previous block's K and V are no longer read
        for (int e = tid; e < BK * D; e += FA_THREADS) {
            const int r = e / D;
            const int c = e % D;
            const int key = k0 + r;
            const bool in = key < lastKey;
            ks[r * LD + c] = in ? ST(kb[(ulong)key * p.kStride + c]) : ST(0.0f);
            vs[r * LD + c] = in ? ST(vb[(ulong)key * p.kStride + c]) : ST(0.0f);
        }
        if (tid < BK) {
            const int key = k0 + int(tid);
            const bool in = key < lastKey;
            kq[tid] = in ? kv_scale(kc + (ulong)key * p.kStride, p.kStride, p.nKvHeads, kvh) : 0.0f;
            vq[tid] = in ? kv_scale(vc + (ulong)key * p.kStride, p.kStride, p.nKvHeads, kvh) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NJ; j++) {
            simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int i = 0; i < NI; i++) {
                simdgroup_matrix<ST, 8, 8> kt;
                simdgroup_load(kt, &ks[j * 8 * LD + i * 8], LD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qm[i], kt, s);
            }
            simdgroup_store(s, &sb[sg][j * 8], BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int r = 0; r < 8; r++) {
            const int t = r0 + r;
            const int len = p.pos0 + t + 1;
            const int first = p.window > 0 ? max(0, len - p.window) : 0;
            float mx = -INFINITY;
            for (int c = lane; c < BK; c += 32) {
                const int key = k0 + c;
                float v = sb[sg][r * BK + c] * (p.scale * kq[c]);
                if (p.softcap > 0.0f) v = softcap_f(v, p.softcap);
                if (t >= p.ntok || key < first || key >= len) v = -INFINITY;
                sb[sg][r * BK + c] = v;
                mx = max(mx, v);
            }
            mx = simd_max(mx);
            const float mNew = max(m[r], mx);
            float sum = 0.0f;
            for (int c = lane; c < BK; c += 32) {
                const float e = mNew == -INFINITY ? 0.0f : exp(sb[sg][r * BK + c] - mNew);
                sb[sg][r * BK + c] = e * vq[c];
                sum += e;
            }
            sum = simd_sum(sum);
            const float alpha = m[r] == -INFINITY ? 0.0f : exp(m[r] - mNew);
            l[r] = l[r] * alpha + sum;
            m[r] = mNew;
            if (lane < 8) st[sg][r * 8 + lane] = int(lane) == r ? alpha : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dg;
        simdgroup_load(dg, &st[sg][0], 8);
        for (int i = 0; i < NI; i++) simdgroup_multiply(om[i], dg, om[i]);
        for (int j = 0; j < NJ; j++) {
            simdgroup_float8x8 pm;
            simdgroup_load(pm, &sb[sg][j * 8], BK);
            for (int i = 0; i < NI; i++) {
                simdgroup_matrix<ST, 8, 8> vm;
                simdgroup_load(vm, &vs[j * 8 * LD + i * 8], LD);
                simdgroup_multiply_accumulate(om[i], pm, vm, om[i]);
            }
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8) {
        for (int r = 0; r < 8; r++) st[sg][r * 8 + lane] = int(lane) == r && l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dg;
    simdgroup_load(dg, &st[sg][0], 8);
    for (int i = 0; i < NI; i++) {
        simdgroup_float8x8 o;
        simdgroup_multiply(o, dg, om[i]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(o, &st[sg][0], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = lane; e < 64; e += 32) {
            const int t = r0 + e / 8;
            if (t < p.ntok) out[(ulong)t * p.oStride + h * D + i * 8 + e % 8] = st[sg][e];
        }
    }
}

#define FB_INSTANTIATE(D)                                                                                         \
    template [[host_name("attention_fb_d" #D)]] kernel decltype(attention_fb<float, float, D>) attention_fb<float, float, D>; \
    template [[host_name("attention_fb_d" #D "_h")]] kernel decltype(attention_fb<half, half, D>) attention_fb<half, half, D>; \
    template [[host_name("attention_fb_d" #D "_q8")]] kernel decltype(attention_fb<char, half, D>) attention_fb<char, half, D>;
FB_INSTANTIATE(64)
FB_INSTANTIATE(128)
FB_INSTANTIATE(256)


// ---------------------------------------------------------------------------------------------- 14. q4 KV cache
// backend.kv-dtype=q4 (Java: KvQ4, KvCache.Q4). A cache row (one position of K or V) is kvDim / 2 bytes of 4-bit
// codes followed by kvDim / 32 half scales: kv head h, group i (values 32 i .. 32 i + 31 of the head) has its 16 bytes
// at h * headDim / 2 + 16 i (byte j: value j in the low nibble, value j + 16 in the high one) and its scale at half
// index h * headDim / 32 + i; a value is (code - 7.5) * scale (levels at the odd multiples of half a step). K rows are stored Walsh-Hadamard rotated (unnormalized,
// blocks of the largest power of two dividing headDim); kv_store_q4 rotates the chunk's queries in place the same way
// and the host divides the score scale by the block (a power of two), so (Hq).(Hk) / B = q.k and no inverse is needed.
//
// Recent window (backend.kv-q4-recent = R > 0): per layer a ring of ringCap = R + batch half rows (kr, vr: rotated K,
// plain V, [ringCap][kvDim]) and a tag per slot (the position it holds, -1 none). A query at position P reads the keys
// [exactFrom(P), P] from the ring and the older ones from the q4 rows, exactFrom(P) = the first position from
// max(0, P - R + 1) on such that every slot up to P still holds its position (q4_held_from): the same rule as the CPU
// cache, so both backends read the same values for every (query, key) pair.

// Unnormalized Walsh-Hadamard transform of a head row held by one simdgroup (lane l owns values l + 32 i, i < per), in
// blocks of B values: the butterflies (a, b) -> (a + b, a - b) of stride 1, 2, 4, ... in KvQ4.rotate's order, strides
// below 32 between lanes, the others between registers. The same operations on the same operands as the Java loop.
static inline void q4_rotate(thread float *x, int per, int B, ushort lane) {
    for (int h = 1; h < 32; h <<= 1) {
        for (int i = 0; i < MAX_PER; i++) {
            if (i < per) {
                const float o = simd_shuffle_xor(x[i], ushort(h));
                x[i] = (lane & h) != 0 ? o - x[i] : x[i] + o;
            }
        }
    }
    for (int h = 1; h < B / 32; h <<= 1) {
        for (int i = 0; i < MAX_PER; i++) {
            if (i < per && (i & h) == 0) {
                const float a = x[i];
                const float b = x[i + h];
                x[i] = a + b;
                x[i + h] = a - b;
            }
        }
    }
}

// The 4-bit code of this lane's value of a group held one value per lane, and the group's scale (KvQ4.quantize):
// d = half(max|x| / 7.5), code = clamp(rint(x / d + 7.5), 0, 15) (8 for an all-zero group).
static inline uint q4_code(float x, thread half &dh) {
    const float amax = simd_max(fabs(x));
    dh = half(amax / 7.5f);
    const float d = float(dh);
    return d == 0.0f ? 8u : uint(clamp(rint(x / d + 7.5f), 0.0f, 15.0f));
}

// Value c (0 .. headDim) of kv head kvh in a q4 cache row.
static inline float q4_value(device const uchar *row, int kvDim, int kvh, int headDim, int c) {
    const int i = c / 32;
    const int j = c % 32;
    const uchar b = row[kvh * headDim / 2 + i * 16 + (j & 15)];
    const float d = float(((device const half *)(row + kvDim / 2))[kvh * (headDim / 32) + i]);
    return (float(j < 16 ? (b & 15) : (b >> 4)) - 7.5f) * d;
}

// The first position p >= max(0, top - span + 1) such that the ring holds every position of [p, top] (top + 1 when it
// does not hold top). One simdgroup, 32 positions per step, uniform control flow.
static inline int q4_held_from(device const int *tags, int ringCap, int top, int span, ushort lane) {
    const int lim = max(0, top - span + 1);
    for (int base = top; base >= lim; base -= 32) {
        const int s = base - int(lane);
        const bool missing = s >= lim && tags[s % ringCap] != s;
        const int last = simd_max(missing ? s : -1); // the highest position the ring lost in this step
        if (last >= 0) return last + 1;
    }
    return lim;
}

struct KvStoreQ4Params {
    int headDim;
    int kvDim;       // values per position (nKvHeads * headDim)
    int nKvHeads;
    int qStride;     // q row stride (nHeads * headDim)
    int stride;      // bytes per cache row: kvDim / 2 + kvDim / 16
    int pos0;        // position of the chunk's first token
    int ringCap;     // ring slots, 0 = no recent window
};

// kv_store_q4: grid (2 * nKvHeads + nHeads, ntok), one simdgroup (32 threads) per row. x < nKvHeads: k head x rotated
// and quantized into the cache; x < 2 * nKvHeads: v head x - nKvHeads quantized (not rotated); the others: query head
// x - 2 * nKvHeads rotated in place. k / v are the f32 staging rows [ntok, kvDim] (k after QK-norm and RoPE); kc / vc
// are bound at the row of the chunk's first position. With a ring the (rotated) rows also go to the ring as halves
// and the k-head-0 row writes the slot's tag.
kernel void kv_store_q4(device const float *k [[buffer(0)]],
                        device const float *v [[buffer(1)]],
                        device uchar *kc [[buffer(2)]],
                        device uchar *vc [[buffer(3)]],
                        device float *q [[buffer(4)]],
                        device half *kr [[buffer(5)]],
                        device half *vr [[buffer(6)]],
                        device int *tags [[buffer(7)]],
                        constant KvStoreQ4Params &p [[buffer(8)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    const int x = int(tgp.x);
    const int t = int(tgp.y);
    const int per = p.headDim / 32;
    const int B = p.headDim & -p.headDim;
    float r[MAX_PER];
    if (x >= 2 * p.nKvHeads) {
        device float *qr = q + (ulong)t * p.qStride + (x - 2 * p.nKvHeads) * p.headDim;
        for (int i = 0; i < MAX_PER; i++) r[i] = i < per ? qr[lane + i * 32] : 0.0f;
        q4_rotate(r, per, B, lane);
        for (int i = 0; i < MAX_PER; i++) if (i < per) qr[lane + i * 32] = r[i];
        return;
    }
    const bool isV = x >= p.nKvHeads;
    const int h = isV ? x - p.nKvHeads : x;
    device const float *src = (isV ? v : k) + (ulong)t * p.kvDim + h * p.headDim;
    for (int i = 0; i < MAX_PER; i++) r[i] = i < per ? src[lane + i * 32] : 0.0f;
    if (!isV) q4_rotate(r, per, B, lane);
    device uchar *row = (isV ? vc : kc) + (ulong)t * p.stride;
    device half *sc = (device half *)(row + p.kvDim / 2);
    for (int i = 0; i < MAX_PER; i++) {
        if (i < per) {
            half dh;
            const uint c = q4_code(r[i], dh);
            const uint hi = simd_shuffle_down(c, ushort(16));
            if (lane < 16) row[h * p.headDim / 2 + i * 16 + lane] = uchar(c | (hi << 4));
            if (lane == 0) sc[h * per + i] = dh;
        }
    }
    if (p.ringCap > 0) {
        const int slot = (p.pos0 + t) % p.ringCap;
        device half *rr = (isV ? vr : kr) + (ulong)slot * p.kvDim + h * p.headDim;
        for (int i = 0; i < MAX_PER; i++) if (i < per) rr[lane + i * 32] = half(r[i]);
        if (!isV && h == 0 && lane == 0) tags[slot] = p.pos0 + t;
    }
}

// attention_q4_t<PER>: attention_t<KV, 1, PER, PER> over a q4 cache (head width PER * 32, or p.qkDim for PER = 0):
// the queries were rotated by kv_store_q4 and p.scale divided by the Hadamard block; keys before exactFrom(P) are read
// from the q4 rows, the others from the ring. Same phases and reductions as attention_t.
struct AttnQ4Params {
    int nHeads;
    int nKvHeads;
    int qkDim;       // head width (headDim)
    int pos0;
    int ctx;
    float scale;     // the score scale divided by the Hadamard block
    int qStride;     // q row stride (nHeads * headDim)
    int stride;      // cache row bytes (kvDim / 2 + kvDim / 16)
    int oStride;     // output row stride (nHeads * headDim)
    int window;      // sliding-window length in tokens, 0 = full causal attention
    int tokenOffset; // first token of this tile; the scores scratch is indexed by the tile-local token
    float softcap;
    int recent;      // R, 0 = no ring
    int ringCap;
};

template <int PER>
kernel void attention_q4_t(device const float *q [[buffer(0)]],
                           device const uchar *kc [[buffer(1)]],
                           device const uchar *vc [[buffer(2)]],
                           device float *out [[buffer(3)]],
                           device float *scores [[buffer(4)]],
                           device const half *kr [[buffer(5)]],
                           device const half *vr [[buffer(6)]],
                           device const int *tags [[buffer(7)]],
                           constant AttnQ4Params &p [[buffer(8)]],
                           uint2 tgp [[threadgroup_position_in_grid]],
                           ushort2 tid2 [[thread_position_in_threadgroup]],
                           ushort2 tpg2 [[threads_per_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           ushort nsg [[simdgroups_per_threadgroup]]) {
    constexpr int M = PER > 0 ? PER : MAX_PER;
    threadgroup float red[ATT_SIMDGROUPS];
    threadgroup float part[ATT_SIMDGROUPS][MAX_PER * 32];
    const ushort tid = tid2.x;
    const ushort tpg = tpg2.x;
    const int per = PER > 0 ? PER : p.qkDim / 32;
    const int hd = per * 32;
    const int kvDim = p.nKvHeads * hd;

    const int h = int(tgp.x);
    const int tl = int(tgp.y);
    const int t = p.tokenOffset + tl;
    const int kvh = h / (p.nHeads / p.nKvHeads);
    const int len = p.pos0 + t + 1;
    const int first = p.window > 0 ? max(0, len - p.window) : 0;
    const int exact = p.recent > 0 ? max(first, q4_held_from(tags, p.ringCap, len - 1, p.recent, lane)) : len;
    device float *sc = scores + ((ulong)tl * p.nHeads + h) * p.ctx;

    float qr[M];
    device const float *qv = q + (ulong)t * p.qStride + h * hd;
    for (int i = 0; i < M; i++) qr[i] = i < per ? qv[lane + i * 32] : 0.0f;

    float mx = -INFINITY;
    for (int s = first + sg; s < len; s += nsg) {
        float kv[M];
        if (s >= exact) {
            device const half *row = kr + (ulong)(s % p.ringCap) * kvDim + kvh * hd;
            for (int i = 0; i < M; i++) kv[i] = i < per ? float(row[lane + i * 32]) : 0.0f;
        } else {
            device const uchar *row = kc + (ulong)s * p.stride;
            for (int i = 0; i < M; i++) kv[i] = i < per ? q4_value(row, kvDim, kvh, hd, lane + i * 32) : 0.0f;
        }
        float d = 0.0f;
        for (int i = 0; i < M; i++) if (i < per) d += qr[i] * kv[i];
        d = simd_sum(d) * p.scale;
        if (p.softcap > 0.0f) d = softcap_f(d, p.softcap);
        if (lane == 0) sc[s] = d;
        mx = max(mx, d);
    }
    {
        float v = simd_max(mx);
        if (lane == 0) red[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = simd_max(lane < nsg ? red[lane] : -INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    float sum = 0.0f;
    for (int s = first + tid; s < len; s += tpg) {
        float e = exp(sc[s] - mx);
        sc[s] = e;
        sum += e;
    }
    {
        float v = simd_sum(sum);
        if (lane == 0) red[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = simd_sum(lane < nsg ? red[lane] : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    float o[M];
    for (int i = 0; i < M; i++) o[i] = 0.0f;
    for (int s = first + sg; s < len; s += nsg) {
        const float pr = sc[s];
        if (s >= exact) {
            device const half *row = vr + (ulong)(s % p.ringCap) * kvDim + kvh * hd;
            for (int i = 0; i < M; i++) if (i < per) o[i] += pr * float(row[lane + i * 32]);
        } else {
            device const uchar *row = vc + (ulong)s * p.stride;
            for (int i = 0; i < M; i++) if (i < per) o[i] += pr * q4_value(row, kvDim, kvh, hd, lane + i * 32);
        }
    }
    for (int i = 0; i < M; i++) if (i < per) part[sg][lane + i * 32] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / sum;
    for (int d = tid; d < hd; d += tpg) {
        float acc = 0.0f;
        for (int k = 0; k < nsg; k++) acc += part[k][d];
        out[(ulong)t * p.oStride + h * hd + d] = acc * inv;
    }
}

template [[host_name("attention_q4")]] kernel decltype(attention_q4_t<0>) attention_q4_t<0>;
template [[host_name("attention_g1_d64_q4")]] kernel decltype(attention_q4_t<2>) attention_q4_t<2>;
template [[host_name("attention_g1_d128_q4")]] kernel decltype(attention_q4_t<4>) attention_q4_t<4>;
template [[host_name("attention_g1_d256_q4")]] kernel decltype(attention_q4_t<8>) attention_q4_t<8>;

// attention_split_q4: attention_split over a q4 cache (flash decoding of one token at pos0; the partials go to
// attention_combine). The extra fields: the q4 row bytes, R and the ring size.
struct AttnSplitQ4Params {
    int nHeads;
    int nKvHeads;
    int headDim;
    int pos0;
    int ctx;
    float scale;     // the score scale divided by the Hadamard block
    int qStride;
    int stride;      // cache row bytes
    int window;
    float softcap;
    int nSplit;
    int chunk;       // keys per split
    int recent;
    int ringCap;
};

kernel void attention_split_q4(device const float *q [[buffer(0)]],
                               device const uchar *kc [[buffer(1)]],
                               device const uchar *vc [[buffer(2)]],
                               device float *part [[buffer(3)]],
                               device float *scores [[buffer(4)]],
                               device const half *kr [[buffer(5)]],
                               device const half *vr [[buffer(6)]],
                               device const int *tags [[buffer(7)]],
                               constant AttnSplitQ4Params &p [[buffer(8)]],
                               uint2 tgp [[threadgroup_position_in_grid]],
                               ushort2 tid2 [[thread_position_in_threadgroup]],
                               ushort2 tpg2 [[threads_per_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]],
                               ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[32];
    threadgroup float red[ATT_SIMDGROUPS][ATT_THREADS];
    const ushort tid = tid2.x;
    const ushort tpg = tpg2.x;

    const int h = int(tgp.x);
    const int split = int(tgp.y);
    const int kvh = h / (p.nHeads / p.nKvHeads);
    const int len = p.pos0 + 1;
    const int first = p.window > 0 ? max(0, len - p.window) : 0;
    const int s0 = first + split * p.chunk;
    const int s1 = min(len, s0 + p.chunk);
    const int per = p.headDim / 32;
    const int kvDim = p.nKvHeads * p.headDim;
    const int exact = p.recent > 0 ? max(first, q4_held_from(tags, p.ringCap, p.pos0, p.recent, lane)) : len;
    device const float *qv = q + h * p.headDim;
    device float *sc = scores + (ulong)h * p.ctx;
    device float *po = part + ((ulong)h * p.nSplit + split) * (p.headDim + 2);

    float qr[MAX_PER];
    for (int i = 0; i < MAX_PER; i++) qr[i] = i < per ? qv[lane + i * 32] : 0.0f;

    float mx = -INFINITY;
    for (int s = s0 + sg; s < s1; s += nsg) {
        float d = 0.0f;
        if (s >= exact) {
            device const half *row = kr + (ulong)(s % p.ringCap) * kvDim + kvh * p.headDim;
            for (int i = 0; i < MAX_PER; i++) if (i < per) d += qr[i] * float(row[lane + i * 32]);
        } else {
            device const uchar *row = kc + (ulong)s * p.stride;
            for (int i = 0; i < MAX_PER; i++) if (i < per) d += qr[i] * q4_value(row, kvDim, kvh, p.headDim, lane + i * 32);
        }
        d = simd_sum(d) * p.scale;
        if (p.softcap > 0.0f) d = softcap_f(d, p.softcap);
        if (lane == 0) sc[s] = d;
        mx = max(mx, d);
    }
    mx = tg_max(mx, buf, sg, lane, nsg);
    threadgroup_barrier(mem_flags::mem_device);
    if (s0 >= s1) {
        for (int d = tid; d < p.headDim; d += tpg) po[d] = 0.0f;
        if (tid == 0) { po[p.headDim] = -INFINITY; po[p.headDim + 1] = 0.0f; }
        return;
    }

    float sum = 0.0f;
    for (int s = s0 + tid; s < s1; s += tpg) {
        float e = exp(sc[s] - mx);
        sc[s] = e;
        sum += e;
    }
    sum = tg_sum(sum, buf, sg, lane, nsg);
    threadgroup_barrier(mem_flags::mem_device);

    float o[MAX_PER];
    for (int i = 0; i < MAX_PER; i++) o[i] = 0.0f;
    for (int s = s0 + sg; s < s1; s += nsg) {
        const float pr = sc[s];
        if (s >= exact) {
            device const half *row = vr + (ulong)(s % p.ringCap) * kvDim + kvh * p.headDim;
            for (int i = 0; i < MAX_PER; i++) if (i < per) o[i] += pr * float(row[lane + i * 32]);
        } else {
            device const uchar *row = vc + (ulong)s * p.stride;
            for (int i = 0; i < MAX_PER; i++) if (i < per) o[i] += pr * q4_value(row, kvDim, kvh, p.headDim, lane + i * 32);
        }
    }
    for (int i = 0; i < MAX_PER; i++) if (i < per) red[sg][lane + i * 32] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = tid; d < p.headDim; d += tpg) {
        float acc = 0.0f;
        for (int g = 0; g < nsg; g++) acc += red[g][d];
        po[d] = acc;
    }
    if (tid == 0) {
        po[p.headDim] = mx;
        po[p.headDim + 1] = sum;
    }
}

// attention_fa_q4<D>: attention_fa over a q4 cache. The K and V blocks are staged dequantized from the q4 rows; where
// a block holds keys some row reads exactly (the band [exactFrom(P), P] of its rows), two corrections follow: the band
// scores are recomputed from the ring rows before the softmax, and after O += P V_q4 the band's share is added as
// O += P_band (V_ring - V_q4), P_band being P with every entry outside the row's band zeroed.
struct FlashQ4Params {
    int nHeads;
    int nKvHeads;
    int pos0;        // position of the chunk's first token
    int ntok;        // tokens in the chunk
    float scale;     // the score scale divided by the Hadamard block
    int qStride;     // q row stride (nHeads * D)
    int stride;      // cache row bytes
    int oStride;     // output row stride (nHeads * D)
    int window;
    float softcap;
    int recent;
    int ringCap;
};

template <int D>
kernel void attention_fa_q4(device const float *q [[buffer(0)]],
                            device const uchar *kc [[buffer(1)]],
                            device const uchar *vc [[buffer(2)]],
                            device float *out [[buffer(3)]],
                            device const half *kr [[buffer(4)]],
                            device const half *vr [[buffer(5)]],
                            device const int *tags [[buffer(6)]],
                            constant FlashQ4Params &p [[buffer(7)]],
                            uint2 tgp [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int BK = 4096 / D;
    constexpr int NI = D / 8;
    constexpr int NJ = BK / 8;
    threadgroup float kvs[BK * D];
    threadgroup float sb[FA_THREADS / 32][8 * BK];
    threadgroup float st[FA_THREADS / 32][64];

    const int h = int(tgp.x);
    const int kvh = h / (p.nHeads / p.nKvHeads);
    const int kvDim = p.nKvHeads * D;
    const int t0 = int(tgp.y) * FA_ROWS;
    const int r0 = t0 + int(sg) * 8;
    const int tEnd = min(p.ntok, t0 + FA_ROWS);
    const int firstKey = p.window > 0 ? max(0, p.pos0 + t0 + 1 - p.window) : 0;
    const int lastKey = p.pos0 + tEnd;
    // every row's band starts at max(held, P - R + 1); the threadgroup's band at bandLo. This equals exactFrom(P)
    // because the ring cannot have lost a position of the chunk itself (ringCap >= R + batch), so the highest lost
    // position below lastKey is below every row's own position
    const int held = p.recent > 0 ? q4_held_from(tags, p.ringCap, lastKey - 1, p.recent + FA_ROWS, lane) : lastKey;
    const int bandLo = p.recent > 0 ? max(held, p.pos0 + t0 - p.recent + 1) : lastKey;

    simdgroup_float8x8 qm[NI];
    for (int i = 0; i < NI; i++) {
        for (int e = lane; e < 64; e += 32) {
            const int t = r0 + e / 8;
            st[sg][e] = t < p.ntok ? q[(ulong)t * p.qStride + h * D + i * 8 + e % 8] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(qm[i], &st[sg][0], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_float8x8 om[NI];
    for (int i = 0; i < NI; i++) om[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m[8];
    float l[8];
    for (int r = 0; r < 8; r++) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }

    for (int k0 = firstKey / BK * BK; k0 < lastKey; k0 += BK) {
        const bool band = k0 + BK > bandLo;   // some row of the threadgroup reads keys of this block exactly
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = tid; e < BK * D; e += FA_THREADS) {
            const int key = k0 + e / D;
            kvs[e] = key < lastKey ? q4_value(kc + (ulong)key * p.stride, kvDim, kvh, D, e % D) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NJ; j++) {
            simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int i = 0; i < NI; i++) {
                simdgroup_float8x8 kt;
                simdgroup_load(kt, &kvs[j * 8 * D + i * 8], D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qm[i], kt, s);
            }
            simdgroup_store(s, &sb[sg][j * 8], BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (band) { // the band's scores from the ring rows
            for (int r = 0; r < 8; r++) {
                const int t = r0 + r;
                const int P = p.pos0 + t;
                const int ex = max(held, P - p.recent + 1);
                for (int c = lane; c < BK; c += 32) {
                    const int key = k0 + c;
                    if (t < p.ntok && key >= ex && key <= P) {
                        device const float *qv = q + (ulong)t * p.qStride + h * D;
                        device const half *kv = kr + (ulong)(key % p.ringCap) * kvDim + kvh * D;
                        float d = 0.0f;
                        for (int i = 0; i < D; i++) d += qv[i] * float(kv[i]);
                        sb[sg][r * BK + c] = d;
                    }
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (int r = 0; r < 8; r++) {
            const int t = r0 + r;
            const int len = p.pos0 + t + 1;
            const int first = p.window > 0 ? max(0, len - p.window) : 0;
            float mx = -INFINITY;
            for (int c = lane; c < BK; c += 32) {
                const int key = k0 + c;
                float v = sb[sg][r * BK + c] * p.scale;
                if (p.softcap > 0.0f) v = softcap_f(v, p.softcap);
                if (t >= p.ntok || key < first || key >= len) v = -INFINITY;
                sb[sg][r * BK + c] = v;
                mx = max(mx, v);
            }
            mx = simd_max(mx);
            const float mNew = max(m[r], mx);
            float sum = 0.0f;
            for (int c = lane; c < BK; c += 32) {
                const float e = mNew == -INFINITY ? 0.0f : exp(sb[sg][r * BK + c] - mNew);
                sb[sg][r * BK + c] = e;
                sum += e;
            }
            sum = simd_sum(sum);
            const float alpha = m[r] == -INFINITY ? 0.0f : exp(m[r] - mNew);
            l[r] = l[r] * alpha + sum;
            m[r] = mNew;
            if (lane < 8) st[sg][r * 8 + lane] = int(lane) == r ? alpha : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dg;
        simdgroup_load(dg, &st[sg][0], 8);
        for (int i = 0; i < NI; i++) simdgroup_multiply(om[i], dg, om[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = tid; e < BK * D; e += FA_THREADS) {
            const int key = k0 + e / D;
            kvs[e] = key < lastKey ? q4_value(vc + (ulong)key * p.stride, kvDim, kvh, D, e % D) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NJ; j++) {
            simdgroup_float8x8 pm;
            simdgroup_load(pm, &sb[sg][j * 8], BK);
            for (int i = 0; i < NI; i++) {
                simdgroup_float8x8 vm;
                simdgroup_load(vm, &kvs[j * 8 * D + i * 8], D);
                simdgroup_multiply_accumulate(om[i], pm, vm, om[i]);
            }
        }
        if (band) { // O += P_band (V_ring - V_q4)
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int r = 0; r < 8; r++) {
                const int t = r0 + r;
                const int P = p.pos0 + t;
                const int ex = max(held, P - p.recent + 1);
                for (int c = lane; c < BK; c += 32) {
                    const int key = k0 + c;
                    if (!(t < p.ntok && key >= ex && key <= P)) sb[sg][r * BK + c] = 0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup); // every simdgroup is done with the q4 V block
            for (int e = tid; e < BK * D; e += FA_THREADS) {
                const int key = k0 + e / D;
                if (key >= bandLo && key < lastKey) {
                    kvs[e] = float(vr[(ulong)(key % p.ringCap) * kvDim + kvh * D + e % D]) - kvs[e];
                } else {
                    kvs[e] = 0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int j = 0; j < NJ; j++) {
                simdgroup_float8x8 pm;
                simdgroup_load(pm, &sb[sg][j * 8], BK);
                for (int i = 0; i < NI; i++) {
                    simdgroup_float8x8 vm;
                    simdgroup_load(vm, &kvs[j * 8 * D + i * 8], D);
                    simdgroup_multiply_accumulate(om[i], pm, vm, om[i]);
                }
            }
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8) {
        for (int r = 0; r < 8; r++) st[sg][r * 8 + lane] = int(lane) == r && l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dg;
    simdgroup_load(dg, &st[sg][0], 8);
    for (int i = 0; i < NI; i++) {
        simdgroup_float8x8 o;
        simdgroup_multiply(o, dg, om[i]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(o, &st[sg][0], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = lane; e < 64; e += 32) {
            const int t = r0 + e / 8;
            if (t < p.ntok) out[(ulong)t * p.oStride + h * D + i * 8 + e % 8] = st[sg][e];
        }
    }
}

template [[host_name("attention_fa_d64_q4")]] kernel decltype(attention_fa_q4<64>) attention_fa_q4<64>;
template [[host_name("attention_fa_d128_q4")]] kernel decltype(attention_fa_q4<128>) attention_fa_q4<128>;
template [[host_name("attention_fa_d256_q4")]] kernel decltype(attention_fa_q4<256>) attention_fa_q4<256>;
