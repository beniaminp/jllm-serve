// Compute kernels of the Metal backends (dense decoders, DeepSeek MLA, Mamba) for Apple GPUs.
// Compiled at runtime by libjllm_metal (newLibraryWithSource); every dispatch is written by MetalKernels.java.
// Parameter structs contain only 32-bit ints/floats so the Java side writes them field by field (KernelParams
// lists the layouts, MetalParamsTest checks them against the structs below). Sections:
//   0. constants   1. common (reductions, block formats)   2. matmul   3. norm + rope   4. attention
//   5. mixture of experts   6. state space (Mamba)   7. DeepSeek MLA (latent cache store, query absorption)
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
                   constant MatmulParams &p [[buffer(5)]],
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
                      constant MatmulParams &p [[buffer(5)]],
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

template <int QT>
static inline void mma_core(device const uchar *rwq, device const float *rws, ulong rgrow, int nb,
                            device const float *x, int cols, int ntRem,
                            threadgroup float *sa, threadgroup float *sb, threadgroup float *sc,
                            ushort tid, ushort sg) {
    simdgroup_float8x8 acc[8];
    for (int i = 0; i < 8; i++) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const int lr = tid >> 1;
    const int h = tid & 1;
    const int lt = tid >> 2;
    const int lp = tid & 3;
    device const float4 *xr = (device const float4 *)(x + (ulong)min(lt, ntRem - 1) * cols) + lp * 2;
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
kernel void matmul_mma(device const uchar *wq [[buffer(0)]],
                       device const float *ws [[buffer(1)]],
                       device const float *x [[buffer(2)]],
                       device float *out [[buffer(3)]],
                       device const float *nw [[buffer(4)]],
                       constant MatmulParams &p [[buffer(5)]],
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
            const float v = sa[t * MMA_BR + r];
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
    const ulong grow = (ulong)(lr < 32 ? unit : p.hidden + unit);
    mma_core<QT>(wq, ws, grow, nb, x + (ulong)t0 * p.cols, p.cols, p.ntok - t0, sa, sb, sa, tid, sg);
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
//   KV          cache element type (float, or half for backend.kv-dtype=f16; converted on load, all arithmetic f32)
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
    int kStride;     // key cache row stride (nKvHeads * qkDim)
    int vStride;     // value cache row stride (nKvHeads * vDim)
    int oStride;     // output row stride (nHeads * vDim)
    int window;      // sliding-window length in tokens, 0 = full causal attention
    int tokenOffset; // first token of this tile; the scores scratch is indexed by the tile-local token
    float softcap;   // Gemma 2: score = tanh(score / softcap) * softcap before the softmax, 0 = off
};

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
        float kr[MK];
        for (int i = 0; i < MK; i++) kr[i] = i < perK ? float(kv[lane + i * 32]) : 0.0f;
        for (int g = 0; g < G; g++) {
            float d = 0.0f;
            for (int i = 0; i < MK; i++) if (i < perK) d += qr[g][i] * kr[i];
            d = simd_sum(d) * p.scale;
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
        float vr[MV];
        for (int i = 0; i < MV; i++) vr[i] = i < perV ? float(vv[lane + i * 32]) : 0.0f;
        for (int g = 0; g < G; g++) {
            const float pr = sc[g][s];
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

// attention / attention_h: runtime widths (head dims other than 64 / 128 / 256);
// attention_mla / attention_mla_h: runtime widths up to MLA_MAX_PER * 32 (DeepSeek's absorbed queries);
// attention_g{G}_d{headDim}[_h]: compile-time widths, G query heads per threadgroup.
#define ATTN_INST(KV, G, PER, NAME) \
    template [[host_name(NAME)]] kernel decltype(attention_t<KV, G, PER, PER>) attention_t<KV, G, PER, PER>;
ATTN_INST(float, 1, 0, "attention")           ATTN_INST(half, 1, 0, "attention_h")
template [[host_name("attention_mla")]] kernel decltype(attention_t<float, 1, 0, 0, MLA_MAX_PER>) attention_t<float, 1, 0, 0, MLA_MAX_PER>;
template [[host_name("attention_mla_h")]] kernel decltype(attention_t<half, 1, 0, 0, MLA_MAX_PER>) attention_t<half, 1, 0, 0, MLA_MAX_PER>;
ATTN_INST(float, 1, 2, "attention_g1_d64")    ATTN_INST(half, 1, 2, "attention_g1_d64_h")
ATTN_INST(float, 1, 4, "attention_g1_d128")   ATTN_INST(half, 1, 4, "attention_g1_d128_h")
ATTN_INST(float, 1, 8, "attention_g1_d256")   ATTN_INST(half, 1, 8, "attention_g1_d256_h")
// The grouped variants are only instantiated when the host defines JLLM_ATTN_GROUPS (-Djllm.metal.attnGroup).
#ifdef JLLM_ATTN_GROUPS
ATTN_INST(float, 2, 2, "attention_g2_d64")    ATTN_INST(float, 4, 2, "attention_g4_d64")    ATTN_INST(float, 8, 2, "attention_g8_d64")
ATTN_INST(float, 2, 4, "attention_g2_d128")   ATTN_INST(float, 4, 4, "attention_g4_d128")   ATTN_INST(float, 8, 4, "attention_g8_d128")
ATTN_INST(float, 2, 8, "attention_g2_d256")   ATTN_INST(float, 4, 8, "attention_g4_d256")   ATTN_INST(float, 8, 8, "attention_g8_d256")
ATTN_INST(half, 2, 2, "attention_g2_d64_h")   ATTN_INST(half, 4, 2, "attention_g4_d64_h")   ATTN_INST(half, 8, 2, "attention_g8_d64_h")
ATTN_INST(half, 2, 4, "attention_g2_d128_h")  ATTN_INST(half, 4, 4, "attention_g4_d128_h")  ATTN_INST(half, 8, 4, "attention_g8_d128_h")
ATTN_INST(half, 2, 8, "attention_g2_d256_h")  ATTN_INST(half, 4, 8, "attention_g4_d256_h")  ATTN_INST(half, 8, 8, "attention_g8_d256_h")
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
    int kvStride;
    int window;
    float softcap;
    int nSplit;
    int chunk;       // keys per split
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
    const int s0 = first + split * p.chunk;
    const int s1 = min(len, s0 + p.chunk);
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
        d = simd_sum(d) * p.scale;
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
        const float pr = sc[s];
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

// Optional half-precision cache (backend.kv-dtype=f16). v is rounded once by the projection's epilogue
// (matmul3<.., half> writes half(v + bias) straight into the cache row); k is projected into an f32 staging
// row, normalised and rotated there, and rounded once by rope<half, ..> when it writes the cache row
// (round-to-nearest-even, the same rounding as Java's Float.floatToFloat16). The *_h attention instantiations
// read half K/V with the same f32 arithmetic (attention_t<half, ...> above).

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
};

kernel void router(device const float *x [[buffer(0)]],
                   device const float *w [[buffer(1)]],
                   device int *assign [[buffer(2)]],
                   device float *rw [[buffer(3)]],
                   constant RouterParams &p [[buffer(4)]],
                   uint tg [[threadgroup_position_in_grid]],
                   ushort tid [[thread_position_in_threadgroup]],
                   ushort tpg [[threads_per_threadgroup]],
                   ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]],
                   ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float logits[MAX_EXPERTS];
    threadgroup float buf[32];
    device const float *xv = x + (ulong)tg * p.dim;
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
            rw[tg * p.topK + k] = bv / sum;
            selected += bv / sum;
            logits[best] = -INFINITY;
        }
        if (p.normalize && selected > 0.0f) {
            for (int k = 0; k < p.topK; k++) rw[tg * p.topK + k] /= selected;
        }
    }
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
};

kernel void router_ds(device const float *x [[buffer(0)]],
                      device const float *w [[buffer(1)]],
                      device int *assign [[buffer(2)]],
                      device float *rw [[buffer(3)]],
                      device const float *bias [[buffer(4)]],
                      constant RouterDsParams &p [[buffer(5)]],
                      uint tg [[threadgroup_position_in_grid]],
                      ushort tid [[thread_position_in_threadgroup]],
                      ushort tpg [[threads_per_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]],
                      ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float scores[MAX_EXPERTS];
    threadgroup float choice[MAX_EXPERTS];
    threadgroup float buf[32];
    const int ne = p.nExperts;
    device const float *xv = x + (ulong)tg * p.dim;
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
    template [[host_name("moe_up_" SUF)]] kernel moe_up_t moe_up<QTV>;                                            \
    template [[host_name("moe_down_" SUF)]] kernel moe_down_t moe_down<QTV>;
GG_INSTANTIATE("q4g", GG_Q4_0)
GG_INSTANTIATE("q8g", GG_Q8_0)
GG_INSTANTIATE("q4k", GG_Q4_K)
GG_INSTANTIATE("q5k", GG_Q5_K)
