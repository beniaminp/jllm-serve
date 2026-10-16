// Compute kernels for the Qwen3 forward pass on Apple GPUs.
// Compiled at runtime by libjllm_metal (newLibraryWithSource); dispatched from MetalBackend.java.
// Parameter structs contain only 32-bit ints/floats so the Java side can pack them sequentially.

#include <metal_stdlib>
using namespace metal;

#define QK 32          // weights per quantization block
#define TOK_TILE 8     // tokens processed per simdgroup in batched matmuls

// ---------------------------------------------------------------------------------------------- reductions

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

// ---------------------------------------------------------------------------------------------- block dot
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

// ---------------------------------------------------------------------------------------------- matmul
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
    return tanh(clamp(v / cap, -20.0f, 20.0f)) * cap;
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
    return 0.5f * g * (1.0f + tanh(clamp(a, -20.0f, 20.0f)));
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
    for (int i = v; i < n; i += 256) {
        float a = x[i];
        p1 += a * a;
    }
    for (int i = v + 128; i < n; i += 256) {
        float a = x[i];
        p2 += a * a;
    }
    p1 = simd_sum(p1);
    p2 = simd_sum(p2);
    if (lane == 0) {
        buf[sg] = p1;
        buf[sg + 4] = p2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < 8 ? buf[lane] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return rsqrt(r / n + eps);
}

// Normalised copy of x[0..n) in threadgroup memory, all 128 threads; ends with a barrier.
static inline void norm_row_128(device const float *x, device const float *w, int n, float eps,
                                threadgroup float *xn, threadgroup float *buf, ushort sg, ushort lane) {
    const float scale = norm_scale_128(x, n, eps, buf, sg, lane);
    for (int i = sg * 32 + lane; i < n; i += 128) xn[i] = x[i] * scale * w[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline device const float4 *as_f4(device const float *p) { return (device const float4 *)p; }
static inline threadgroup const float4 *as_f4(threadgroup const float *p) { return (threadgroup const float4 *)p; }

// acc[t] += W[grow] . x[t * xstride ..] for t < nt; blocks walked in lane order, each dequantized once for
// all tokens. XP is a device or threadgroup float pointer.
template <typename XP>
static inline void mm_row_q2k(device const uchar *wq, ulong grow, int nb, XP x, ulong xstride, int nt,
                              thread float *acc, ushort lane); // Q2_K section at the end of the file

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

template <int QT, typename XP>
static inline void mm_row(device const uchar *wq, device const float *ws, ulong grow, int nb,
                          XP x, ulong xstride, int nt, thread float *acc, ushort lane) {
    if (QT == 2) {
        mm_row_q2k(wq, grow, nb, x, xstride, nt, acc, lane);
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
// written into the f16 cache (the k rows are rounded by norm_rope<half> after the rotation instead).
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
    if (NORM) {
        mm_row<QT>(wq, ws, (ulong)row, nb, (threadgroup const float *)xn, 0, 1, accg, lane);
        mm_row<QT>(wq, ws, (ulong)(p.hidden + row), nb, (threadgroup const float *)xn, 0, 1, accu, lane);
    } else {
        mm_row<QT>(wq, ws, (ulong)row, nb, x + (ulong)t0 * p.cols, (ulong)p.cols, nt, accg, lane);
        mm_row<QT>(wq, ws, (ulong)(p.hidden + row), nb, x + (ulong)t0 * p.cols, (ulong)p.cols, nt, accu, lane);
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

// ---------------------------------------------------------------------------------------------- prefill matmul
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

#define PF_ROWS 2      // rows per simdgroup of matmul_pf / matmul3_pf
#define PF_UNITS 1     // units (gate + up rows) per simdgroup of matmul_swiglu_pf
#define PF_TOKS 8      // tokens per simdgroup of the prefill kernels

template <int NR, int TT>
static inline void mm_tile_q2k(thread device const uchar *const *wq, thread const ulong *grow, int nr, int nb,
                               device const float *x, ulong xstride, int nt, thread float (*acc)[TT], ushort lane);

template <int QT, int NR, int TT>
static inline void mm_tile(thread device const uchar *const *wq, thread device const float *const *ws,
                           thread const ulong *grow, int nr, int nb, device const float *x, ulong xstride, int nt,
                           thread float (*acc)[TT], ushort lane) {
    if (QT == 2) {
        mm_tile_q2k<NR, TT>(wq, grow, nr, nb, x, xstride, nt, acc, lane);
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
        wqs[NR + r] = wq; wss[NR + r] = ws; grow[NR + r] = (ulong)(p.hidden + unit);
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

// ---------------------------------------------------------------------------------------------- rmsnorm
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

// ---------------------------------------------------------------------------------------------- qk-norm + rope
// In place on the q rows ([ntok, qStride], nHeads heads) and the freshly written k rows of the KV cache
// ([ntok, kvStride], nKvHeads heads) in one dispatch. Grid: (nHeads + nKvHeads, ntok); threadgroup x < nHeads
// handles q head x, otherwise k head x - nHeads. Threads per threadgroup = headDim / 2, thread i owns the
// rotary pair (i, i + headDim/2).

struct RopeParams {
    int headDim;
    int qStride;
    int kvStride;
    int nHeads;
    int pos0;
    float eps;
    int useNorm;
};

// The k rows are read from `k` (f32: the cache rows themselves, or the f32 staging rows of an f16 cache) and
// written to `kOut` with the cache element type KV at the same row/head offsets: with KV = half this is where
// the f16 cache rounds k, once, after the norm and the rotation (round-to-nearest-even, Java's
// Float.floatToFloat16); with KV = float the host binds kOut = k and the update is in place.
template <typename KV>
kernel void norm_rope(device float *q [[buffer(0)]],
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
    const int half_dim = p.headDim / 2;
    const bool isQ = int(tgp.x) < p.nHeads;
    const int head = isQ ? int(tgp.x) : int(tgp.x) - p.nHeads;
    const ulong off = isQ ? (ulong)tgp.y * p.qStride + (ulong)head * p.headDim
                          : (ulong)tgp.y * p.kvStride + (ulong)head * p.headDim;
    device const float *h = (isQ ? q : k) + off;
    device const float *w = isQ ? qw : kw;
    float a = h[tid];
    float b = h[tid + half_dim];
    if (p.useNorm) {
        float ss = tg_sum(a * a + b * b, buf, sg, lane, nsg);
        float scale = rsqrt(ss / p.headDim + p.eps);
        a *= scale * w[tid];
        b *= scale * w[tid + half_dim];
    }
    const int pos = p.pos0 + int(tgp.y);
    const float c = cosT[pos * half_dim + tid];
    const float s = sinT[pos * half_dim + tid];
    const float ra = a * c - b * s;
    const float rb = b * c + a * s;
    if (isQ) {
        q[off + tid] = ra;
        q[off + tid + half_dim] = rb;
    } else {
        kOut[off + tid] = KV(ra);
        kOut[off + tid + half_dim] = KV(rb);
    }
}

template [[host_name("norm_rope")]] kernel decltype(norm_rope<float>) norm_rope<float>;
template [[host_name("norm_rope_h")]] kernel decltype(norm_rope<half>) norm_rope<half>;

// ---------------------------------------------------------------------------------------------- attention
// Causal GQA attention for token t = tokenOffset + tgp.y at position pos0 + t. Grid: (nHeads, tile),
// 256 threads = 8 simdgroups. Tokens are dispatched in tiles so the scores scratch only needs
// tile * nHeads * ctx floats: it is indexed by the tile-local token tgp.y, not by t.
// Scores go to a device scratch row (so context length is not limited by threadgroup memory).
// Each lane owns headDim/32 (<= 8) dimensions of q, k and v.

struct AttnParams {
    int nHeads;
    int nKvHeads;
    int headDim;
    int pos0;
    int ctx;
    float scale;
    int qStride;
    int kvStride;
    int window;      // sliding-window length in tokens, 0 = full causal attention
    int tokenOffset; // first token of this tile; the scores scratch is indexed by the tile-local token
    float softcap;   // Gemma 2: score = tanh(score / softcap) * softcap before the softmax, 0 = off
};

kernel void attention(device const float *q [[buffer(0)]],
                      device const float *kc [[buffer(1)]],
                      device const float *vc [[buffer(2)]],
                      device float *out [[buffer(3)]],
                      device float *scores [[buffer(4)]],
                      constant AttnParams &p [[buffer(5)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      ushort2 tid2 [[thread_position_in_threadgroup]],
                      ushort2 tpg2 [[threads_per_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]],
                      ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[32];
    threadgroup float part[8][256];
    const ushort tid = tid2.x;
    const ushort tpg = tpg2.x;

    const int h = int(tgp.x);
    const int t = p.tokenOffset + int(tgp.y);
    const int kvh = h / (p.nHeads / p.nKvHeads);
    const int len = p.pos0 + t + 1;
    const int first = p.window > 0 ? max(0, len - p.window) : 0;
    const int per = p.headDim / 32;
    device const float *qv = q + (ulong)t * p.qStride + h * p.headDim;
    device float *sc = scores + ((ulong)tgp.y * p.nHeads + h) * p.ctx;
    device const float *kb = kc + kvh * p.headDim;
    device const float *vb = vc + kvh * p.headDim;

    float qr[8];
    for (int i = 0; i < 8; i++) qr[i] = i < per ? qv[lane + i * 32] : 0.0f;

    // 1. scores
    float mx = -INFINITY;
    for (int s = first + sg; s < len; s += nsg) {
        device const float *kv = kb + (ulong)s * p.kvStride;
        float d = 0.0f;
        for (int i = 0; i < 8; i++) if (i < per) d += qr[i] * kv[lane + i * 32];
        d = simd_sum(d) * p.scale;
        if (p.softcap > 0.0f) d = tanh(clamp(d / p.softcap, -20.0f, 20.0f)) * p.softcap;
        if (lane == 0) sc[s] = d;
        mx = max(mx, d);
    }
    mx = tg_max(mx, buf, sg, lane, nsg);
    threadgroup_barrier(mem_flags::mem_device);

    // 2. softmax numerators
    float sum = 0.0f;
    for (int s = first + tid; s < len; s += tpg) {
        float e = exp(sc[s] - mx);
        sc[s] = e;
        sum += e;
    }
    sum = tg_sum(sum, buf, sg, lane, nsg);
    threadgroup_barrier(mem_flags::mem_device);

    // 3. weighted sum of values, split over simdgroups then reduced
    float o[8];
    for (int i = 0; i < 8; i++) o[i] = 0.0f;
    for (int s = first + sg; s < len; s += nsg) {
        const float pr = sc[s];
        device const float *vv = vb + (ulong)s * p.kvStride;
        for (int i = 0; i < 8; i++) if (i < per) o[i] += pr * vv[lane + i * 32];
    }
    for (int i = 0; i < 8; i++) if (i < per) part[sg][lane + i * 32] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / sum;
    for (int d = tid; d < p.headDim; d += tpg) {
        float acc = 0.0f;
        for (int g = 0; g < nsg; g++) acc += part[g][d];
        out[(ulong)t * p.qStride + h * p.headDim + d] = acc * inv;
    }
}

// attention_g<G, PER>: the same computation for G query heads of one KV head per threadgroup (G divides the
// GQA group nHeads / nKvHeads), so each K and V row is loaded once and used for G heads. Grid:
// (nHeads / G, tile), 256 threads. Per head every value is produced exactly as in `attention`: the same
// dot products, the same per-thread softmax partial sums (s = tid, tid + 256, ...), the same two-level
// simd reductions and the same simdgroup-strided value accumulation. PER = headDim / 32.

template <typename KV, int G, int PER>
kernel void attention_g(device const float *q [[buffer(0)]],
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
    threadgroup float red[G * 8];      // per-head cross-simdgroup reduction slots (nsg <= 8)
    threadgroup float part[8][256];
    const ushort tid = tid2.x;
    const ushort tpg = tpg2.x;
    const int headDim = PER * 32;

    const int h0 = int(tgp.x) * G;
    const int tl = int(tgp.y);
    const int t = p.tokenOffset + tl;
    const int kvh = h0 / (p.nHeads / p.nKvHeads);
    const int len = p.pos0 + t + 1;
    const int first = p.window > 0 ? max(0, len - p.window) : 0;
    device const KV *kb = kc + kvh * headDim;
    device const KV *vb = vc + kvh * headDim;
    device float *sc[G];
    for (int g = 0; g < G; g++) sc[g] = scores + ((ulong)tl * p.nHeads + h0 + g) * p.ctx;

    float qr[G][PER];
    for (int g = 0; g < G; g++) {
        device const float *qv = q + (ulong)t * p.qStride + (h0 + g) * headDim;
        for (int i = 0; i < PER; i++) qr[g][i] = qv[lane + i * 32];
    }

    // 1. scores: each K row is read once per threadgroup and dotted with the G query heads
    float mx[G];
    for (int g = 0; g < G; g++) mx[g] = -INFINITY;
    for (int s = first + sg; s < len; s += nsg) {
        device const KV *kv = kb + (ulong)s * p.kvStride;
        float kr[PER];
        for (int i = 0; i < PER; i++) kr[i] = float(kv[lane + i * 32]);
        for (int g = 0; g < G; g++) {
            float d = 0.0f;
            for (int i = 0; i < PER; i++) d += qr[g][i] * kr[i];
            d = simd_sum(d) * p.scale;
            if (p.softcap > 0.0f) d = tanh(clamp(d / p.softcap, -20.0f, 20.0f)) * p.softcap;
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

    // 3. weighted sum of values: each V row is read once per threadgroup
    float o[G][PER];
    for (int g = 0; g < G; g++)
        for (int i = 0; i < PER; i++) o[g][i] = 0.0f;
    for (int s = first + sg; s < len; s += nsg) {
        device const KV *vv = vb + (ulong)s * p.kvStride;
        float vr[PER];
        for (int i = 0; i < PER; i++) vr[i] = float(vv[lane + i * 32]);
        for (int g = 0; g < G; g++) {
            const float pr = sc[g][s];
            for (int i = 0; i < PER; i++) o[g][i] += pr * vr[i];
        }
    }
    for (int g = 0; g < G; g++) {
        for (int i = 0; i < PER; i++) part[sg][lane + i * 32] = o[g][i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = 1.0f / sum[g];
        for (int d = tid; d < headDim; d += tpg) {
            float acc = 0.0f;
            for (int k = 0; k < nsg; k++) acc += part[k][d];
            out[(ulong)t * p.qStride + (h0 + g) * headDim + d] = acc * inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// attention_g{G}_d{headDim} reads the f32 cache, attention_g{G}_d{headDim}_h the f16 cache (same arithmetic
// as attention_h: half values are converted on load, everything else is f32).
#define ATTN_INST(KV, G, PER, NAME) \
    template [[host_name(NAME)]] kernel decltype(attention_g<KV, G, PER>) attention_g<KV, G, PER>;
ATTN_INST(float, 1, 2, "attention_g1_d64")    ATTN_INST(float, 2, 2, "attention_g2_d64")
ATTN_INST(float, 4, 2, "attention_g4_d64")    ATTN_INST(float, 8, 2, "attention_g8_d64")
ATTN_INST(float, 1, 4, "attention_g1_d128")   ATTN_INST(float, 2, 4, "attention_g2_d128")
ATTN_INST(float, 4, 4, "attention_g4_d128")   ATTN_INST(float, 8, 4, "attention_g8_d128")
ATTN_INST(float, 1, 8, "attention_g1_d256")   ATTN_INST(float, 2, 8, "attention_g2_d256")
ATTN_INST(float, 4, 8, "attention_g4_d256")   ATTN_INST(float, 8, 8, "attention_g8_d256")
ATTN_INST(half, 1, 2, "attention_g1_d64_h")   ATTN_INST(half, 2, 2, "attention_g2_d64_h")
ATTN_INST(half, 4, 2, "attention_g4_d64_h")   ATTN_INST(half, 8, 2, "attention_g8_d64_h")
ATTN_INST(half, 1, 4, "attention_g1_d128_h")  ATTN_INST(half, 2, 4, "attention_g2_d128_h")
ATTN_INST(half, 4, 4, "attention_g4_d128_h")  ATTN_INST(half, 8, 4, "attention_g8_d128_h")
ATTN_INST(half, 1, 8, "attention_g1_d256_h")  ATTN_INST(half, 2, 8, "attention_g2_d256_h")
ATTN_INST(half, 4, 8, "attention_g4_d256_h")  ATTN_INST(half, 8, 8, "attention_g8_d256_h")

// ---------------------------------------------------------------------------------------------- flash decoding
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
    threadgroup float red[8][256];
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

    float qr[8];
    for (int i = 0; i < 8; i++) qr[i] = i < per ? qv[lane + i * 32] : 0.0f;

    float mx = -INFINITY;
    for (int s = s0 + sg; s < s1; s += nsg) {
        device const KV *kv = kb + (ulong)s * p.kvStride;
        float d = 0.0f;
        for (int i = 0; i < 8; i++) if (i < per) d += qr[i] * float(kv[lane + i * 32]);
        d = simd_sum(d) * p.scale;
        if (p.softcap > 0.0f) d = tanh(clamp(d / p.softcap, -20.0f, 20.0f)) * p.softcap;
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

    float o[8];
    for (int i = 0; i < 8; i++) o[i] = 0.0f;
    for (int s = s0 + sg; s < s1; s += nsg) {
        const float pr = sc[s];
        device const KV *vv = vb + (ulong)s * p.kvStride;
        for (int i = 0; i < 8; i++) if (i < per) o[i] += pr * float(vv[lane + i * 32]);
    }
    for (int i = 0; i < 8; i++) if (i < per) red[sg][lane + i * 32] = o[i];
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
    int nSplit;      // <= 64
};

// out[h, d] = sum_s o_s[d] * exp(m_s - M) / sum_s l_s * exp(m_s - M), M = max_s m_s. Grid: (nHeads), 256 threads.
kernel void attention_combine(device const float *part [[buffer(0)]],
                              device float *out [[buffer(1)]],
                              constant AttnCombineParams &p [[buffer(2)]],
                              uint tg [[threadgroup_position_in_grid]],
                              ushort tid [[thread_position_in_threadgroup]],
                              ushort tpg [[threads_per_threadgroup]]) {
    threadgroup float w[64];
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

// ---------------------------------------------------------------------------------------------- mixture of experts
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
    threadgroup float logits[1024];
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
    const ulong growG = (ulong)assign[s] * p.rowsPerExpert + row;
    const ulong growU = growG + m;
    const int nb = p.cols / QK;
    device const float4 *x4 = (device const float4 *)(x + (ulong)t * p.cols);
    float g = 0.0f;
    float u = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        g += block_dot<QT>(wq, ws, growG, nb, b, x4 + b * 8);
        u += block_dot<QT>(wq, ws, growU, nb, b, x4 + b * 8);
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

// ---------------------------------------------------------------------------------------------- f16 KV cache
// Optional half-precision cache (backend.kv-dtype=f16). v is rounded once by the projection's epilogue
// (matmul3<.., half> writes half(v + bias) straight into the cache row); k is projected into an f32 staging
// row, normalised and rotated there, and rounded once by norm_rope<half> when it writes the cache row
// (round-to-nearest-even, the same rounding as Java's Float.floatToFloat16). attention_h is the attention
// kernel reading half K/V with the same f32 arithmetic as `attention`.

// Same algorithm and parameters as `attention`, templated on the cache element type.
template <typename KV>
kernel void attention_kv(device const float *q [[buffer(0)]],
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
    threadgroup float buf[32];
    threadgroup float part[8][256];
    const ushort tid = tid2.x;
    const ushort tpg = tpg2.x;

    const int h = int(tgp.x);
    const int t = p.tokenOffset + int(tgp.y);
    const int kvh = h / (p.nHeads / p.nKvHeads);
    const int len = p.pos0 + t + 1;
    const int first = p.window > 0 ? max(0, len - p.window) : 0;
    const int per = p.headDim / 32;
    device const float *qv = q + (ulong)t * p.qStride + h * p.headDim;
    device float *sc = scores + ((ulong)tgp.y * p.nHeads + h) * p.ctx;
    device const KV *kb = kc + kvh * p.headDim;
    device const KV *vb = vc + kvh * p.headDim;

    float qr[8];
    for (int i = 0; i < 8; i++) qr[i] = i < per ? qv[lane + i * 32] : 0.0f;

    // 1. scores
    float mx = -INFINITY;
    for (int s = first + sg; s < len; s += nsg) {
        device const KV *kv = kb + (ulong)s * p.kvStride;
        float d = 0.0f;
        for (int i = 0; i < 8; i++) if (i < per) d += qr[i] * float(kv[lane + i * 32]);
        d = simd_sum(d) * p.scale;
        if (p.softcap > 0.0f) d = tanh(clamp(d / p.softcap, -20.0f, 20.0f)) * p.softcap;
        if (lane == 0) sc[s] = d;
        mx = max(mx, d);
    }
    mx = tg_max(mx, buf, sg, lane, nsg);
    threadgroup_barrier(mem_flags::mem_device);

    // 2. softmax numerators
    float sum = 0.0f;
    for (int s = first + tid; s < len; s += tpg) {
        float e = exp(sc[s] - mx);
        sc[s] = e;
        sum += e;
    }
    sum = tg_sum(sum, buf, sg, lane, nsg);
    threadgroup_barrier(mem_flags::mem_device);

    // 3. weighted sum of values, split over simdgroups then reduced
    float o[8];
    for (int i = 0; i < 8; i++) o[i] = 0.0f;
    for (int s = first + sg; s < len; s += nsg) {
        const float pr = sc[s];
        device const KV *vv = vb + (ulong)s * p.kvStride;
        for (int i = 0; i < 8; i++) if (i < per) o[i] += pr * float(vv[lane + i * 32]);
    }
    for (int i = 0; i < 8; i++) if (i < per) part[sg][lane + i * 32] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / sum;
    for (int d = tid; d < p.headDim; d += tpg) {
        float acc = 0.0f;
        for (int g = 0; g < nsg; g++) acc += part[g][d];
        out[(ulong)t * p.qStride + h * p.headDim + d] = acc * inv;
    }
}

typedef decltype(attention_kv<half>) attention_kv_t;
template [[host_name("attention_h")]] kernel attention_kv_t attention_kv<half>;

// ---------------------------------------------------------------------------------------------- GLM
// norm_rope_il: GLM (and rotate-half partial rotary, rotateHalf = 1) rotary embedding for the q rows and the k rows in one dispatch (like norm_rope, no qk-norm).
// Only the first 2*rotHalf dims of each head rotate, and the pairs are adjacent: (2i, 2i+1) is rotated by table
// entry i (rotate-half models pair i with i + headDim/2 instead). Grid: (nHeads + nKvHeads, ntok), rotHalf threads.

struct RopeIlParams {
    int headDim;
    int qStride;
    int kvStride;
    int nHeads;
    int pos0;
    int rotHalf; // rotated dims / 2 = entries per position in the cos/sin tables
    int rotateHalf; // 1: pair (i, i + rotHalf) inside the rotated prefix (Phi-4-mini); 0: adjacent pairs (GLM)
};

// k is read from `k` and written to `kOut` (cache element type KV) like norm_rope; the dims beyond the rotated
// prefix are copied over unchanged so a half cache receives the whole row.
template <typename KV>
kernel void norm_rope_il(device float *q [[buffer(0)]],
                         device const float *k [[buffer(1)]],
                         device const float *cosT [[buffer(2)]],
                         device const float *sinT [[buffer(3)]],
                         device KV *kOut [[buffer(4)]],
                         constant RopeIlParams &p [[buffer(5)]],
                         uint2 tgp [[threadgroup_position_in_grid]],
                         ushort2 tid2 [[thread_position_in_threadgroup]]) {
    const int tid = tid2.x;
    const bool isQ = int(tgp.x) < p.nHeads;
    const int head = isQ ? int(tgp.x) : int(tgp.x) - p.nHeads;
    const ulong off = isQ ? (ulong)tgp.y * p.qStride + (ulong)head * p.headDim
                          : (ulong)tgp.y * p.kvStride + (ulong)head * p.headDim;
    device const float *h = (isQ ? q : k) + off;
    const int pos = p.pos0 + int(tgp.y);
    const int i0 = p.rotateHalf ? tid : 2 * tid;
    const int i1 = p.rotateHalf ? tid + p.rotHalf : 2 * tid + 1;
    const float a = h[i0];
    const float b = h[i1];
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
        for (int i = 2 * p.rotHalf + tid; i < p.headDim; i += p.rotHalf) kOut[off + i] = KV(h[i]);
    }
}

template [[host_name("norm_rope_il")]] kernel decltype(norm_rope_il<float>) norm_rope_il<float>;
template [[host_name("norm_rope_il_h")]] kernel decltype(norm_rope_il<half>) norm_rope_il<half>;

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

// ---------------------------------------------------------------------------------------------- Gemma 2 / 3
// The sandwich norms reuse rmsnorm_add above, the GeGLU is the `gelu` flag of matmul_swiglu, the attention
// soft-cap is AttnParams.softcap and the final-logit soft-cap is MatmulParams.softcap on the LM-head matmul.

// ---------------------------------------------------------------------------------------------- Mamba
// Mamba (v1) mixer pieces, used by MambaMetalBackend. The projections (in_proj, x_proj, out_proj) reuse the
// matmul kernels above; the two kernels below are the parts that carry state across tokens. Both use one thread
// per channel and run the tokens of the chunk sequentially inside the thread (the recurrence is inherently
// sequential over tokens, independent over channels), so a layer costs one dispatch per kernel instead of one per
// token. The price: parallelism is d_inner threads (1536 .. 5120), no more during a long prefill.

#define MAMBA_MAX_STATE 64
#define MAMBA_MAX_CONV 8

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
        const float dt = raw > 20.0f ? raw : precise::log(1.0f + precise::exp(raw));
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

// ---------------------------------------------------------------------------------------------- DeepSeek MLA
// Multi-head latent attention (DeepSeek-V2/V3). Projections, norms and the output projection use the generic
// matmul / rmsnorm kernels; only what is specific to MLA is below. Keys and queries are qk = nope + rope wide, values
// are vDim wide (not equal), so the cache has separate row strides for K and V.

// mla_rope: rotates the interleaved pairs (2i, 2i+1) of the `rope` wide slice that starts `nope` into every head of
// each row (q: heads = nHeads, nope = qk - rope; the shared k_pe: heads = 1, nope = 0). In place.
// Grid: (heads, ntok); threads per threadgroup = rope / 2.

struct MlaRopeParams {
    int qk;         // head stride within a row
    int nope;
    int rope;
    int rowStride;
    int pos0;
};

kernel void mla_rope(device float *v [[buffer(0)]],
                     device const float *cosT [[buffer(1)]],
                     device const float *sinT [[buffer(2)]],
                     constant MlaRopeParams &p [[buffer(3)]],
                     uint2 tgp [[threadgroup_position_in_grid]],
                     ushort2 tid2 [[thread_position_in_threadgroup]]) {
    const ushort tid = tid2.x;
    const int half_dim = p.rope / 2;
    device float *h = v + (ulong)tgp.y * p.rowStride + (ulong)tgp.x * p.qk + p.nope;
    const int pos = p.pos0 + int(tgp.y);
    const float a = h[2 * tid];
    const float b = h[2 * tid + 1];
    const float c = cosT[pos * half_dim + tid];
    const float s = sinT[pos * half_dim + tid];
    h[2 * tid] = a * c - b * s;
    h[2 * tid + 1] = b * c + a * s;
}

// mla_kv_store: assembles key = [k_nope | k_pe] and value for every head of every token from the kv_b output
// ([ntok, heads * (nope + vDim)], per head nope key values then vDim values) and the rotated k_pe ([ntok, rope]).
// kc / vc are bound at the row of the first token. Grid: (heads, ntok), 64 threads.

struct MlaStoreParams {
    int nope;
    int rope;
    int vDim;
    int kvbStride;
    int kStride;
    int vStride;
};

kernel void mla_kv_store(device const float *kvb [[buffer(0)]],
                         device const float *kpe [[buffer(1)]],
                         device float *kc [[buffer(2)]],
                         device float *vc [[buffer(3)]],
                         constant MlaStoreParams &p [[buffer(4)]],
                         uint2 tgp [[threadgroup_position_in_grid]],
                         ushort2 tid2 [[thread_position_in_threadgroup]],
                         ushort2 tpg2 [[threads_per_threadgroup]]) {
    const ushort tid = tid2.x;
    const ushort tpg = tpg2.x;
    const int h = int(tgp.x);
    const int t = int(tgp.y);
    device const float *src = kvb + (ulong)t * p.kvbStride + (ulong)h * (p.nope + p.vDim);
    device float *kd = kc + (ulong)t * p.kStride + (ulong)h * (p.nope + p.rope);
    device float *vd = vc + (ulong)t * p.vStride + (ulong)h * p.vDim;
    for (int i = tid; i < p.nope; i += tpg) kd[i] = src[i];
    for (int i = tid; i < p.rope; i += tpg) kd[p.nope + i] = kpe[(ulong)t * p.rope + i];
    for (int i = tid; i < p.vDim; i += tpg) vd[i] = src[p.nope + i];
}

// attention_mla: same algorithm as `attention` (one head x token per threadgroup, 256 threads) with a q/k width
// `qkDim` and a different value width `vDim` (both multiples of 32, <= 256) and no grouped heads or window. Token tiles as
// in `attention`: t = tokenOffset + tgp.y, scores indexed by the tile-local token.

struct MlaAttnParams {
    int nHeads;
    int qkDim;
    int vDim;
    int pos0;
    int ctx;
    float scale;
    int qStride;
    int kStride;
    int vStride;
    int oStride;
    int tokenOffset; // first token of this tile; the scores scratch is indexed by the tile-local token
};

kernel void attention_mla(device const float *q [[buffer(0)]],
                          device const float *kc [[buffer(1)]],
                          device const float *vc [[buffer(2)]],
                          device float *out [[buffer(3)]],
                          device float *scores [[buffer(4)]],
                          constant MlaAttnParams &p [[buffer(5)]],
                          uint2 tgp [[threadgroup_position_in_grid]],
                          ushort2 tid2 [[thread_position_in_threadgroup]],
                          ushort2 tpg2 [[threads_per_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]],
                          ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[32];
    threadgroup float part[8][256];
    const ushort tid = tid2.x;
    const ushort tpg = tpg2.x;

    const int h = int(tgp.x);
    const int t = p.tokenOffset + int(tgp.y);
    const int len = p.pos0 + t + 1;
    const int perK = p.qkDim / 32;
    const int perV = p.vDim / 32;
    device const float *qv = q + (ulong)t * p.qStride + h * p.qkDim;
    device float *sc = scores + ((ulong)tgp.y * p.nHeads + h) * p.ctx;
    device const float *kb = kc + h * p.qkDim;
    device const float *vb = vc + h * p.vDim;

    float qr[8];
    for (int i = 0; i < 8; i++) qr[i] = i < perK ? qv[lane + i * 32] : 0.0f;

    float mx = -INFINITY;
    for (int s = sg; s < len; s += nsg) {
        device const float *kv = kb + (ulong)s * p.kStride;
        float d = 0.0f;
        for (int i = 0; i < 8; i++) if (i < perK) d += qr[i] * kv[lane + i * 32];
        d = simd_sum(d) * p.scale;
        if (lane == 0) sc[s] = d;
        mx = max(mx, d);
    }
    mx = tg_max(mx, buf, sg, lane, nsg);
    threadgroup_barrier(mem_flags::mem_device);

    float sum = 0.0f;
    for (int s = tid; s < len; s += tpg) {
        float e = exp(sc[s] - mx);
        sc[s] = e;
        sum += e;
    }
    sum = tg_sum(sum, buf, sg, lane, nsg);
    threadgroup_barrier(mem_flags::mem_device);

    float o[8];
    for (int i = 0; i < 8; i++) o[i] = 0.0f;
    for (int s = sg; s < len; s += nsg) {
        const float pr = sc[s];
        device const float *vv = vb + (ulong)s * p.vStride;
        for (int i = 0; i < 8; i++) if (i < perV) o[i] += pr * vv[lane + i * 32];
    }
    for (int i = 0; i < 8; i++) if (i < perV) part[sg][lane + i * 32] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / sum;
    for (int d = tid; d < p.vDim; d += tpg) {
        float acc = 0.0f;
        for (int g = 0; g < nsg; g++) acc += part[g][d];
        out[(ulong)t * p.oStride + h * p.vDim + d] = acc * inv;
    }
}

// ---------------------------------------------------------------------------------------------- DeepSeek router
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
    threadgroup float scores[1024];
    threadgroup float choice[1024];
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
        float gsc[64];
        bool keep[64];
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

// ============================================================================================== Q2_K
// llama.cpp's Q2_K: 256-weight super-blocks of 84 bytes = 16 scale/min bytes (low nibble = 4-bit scale sc,
// high nibble = 4-bit min m) + 64 bytes of 2-bit quants + f16 d + f16 dmin; weight = d*sc*q - dmin*m.
// Sub-block j (weights 16j .. 16j+15) is at qs + (j/8)*32 + (j%2)*16 with bit shift 2*((j%8)/2).
// Scales live inside the block, so buffer(1) (ws) is unused by these kernels. See docs/13-q2k-quantization.md.

#define QK_K 256
#define Q2K_BYTES 84

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
template [[host_name("moe_up_q2k")]] kernel moe_up_t moe_up<2>;
template [[host_name("moe_down_q2k")]] kernel moe_down_t moe_down<2>;
