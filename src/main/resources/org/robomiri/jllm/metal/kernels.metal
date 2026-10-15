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
};

struct Matmul3Params {
    int rows0;
    int rows1;
    int rows2;
    int cols;
    int ntok;
    float eps;
};

struct SwigluParams {
    int hidden;     // W has 2 * hidden rows: gate rows [0, hidden), up rows [hidden, 2 * hidden)
    int cols;
    int ntok;
    float eps;
};

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
template <int QT, typename XP>
static inline void mm_row(device const uchar *wq, device const float *ws, ulong grow, int nb,
                          XP x, ulong xstride, int nt, thread float *acc, ushort lane) {
    device const float *rs = ws + grow * nb;
    if (QT == 4) {
        device const uchar4 *rq = (device const uchar4 *)(wq + grow * nb * 16);
        for (int b = lane; b < nb; b += 32) {
            float4 lo[4], hi[4];
            for (int k = 0; k < 4; k++) {
                uchar4 v = rq[b * 4 + k];
                lo[k] = float4(v & uchar4(15)) - 8.0f;   // weights 4k .. 4k+3
                hi[k] = float4(v >> uchar4(4)) - 8.0f;   // weights 16+4k .. 16+4k+3
            }
            const float s = rs[b];
            for (int t = 0; t < TOK_TILE; t++) {
                if (t < nt) {
                    auto x4 = as_f4(x + (ulong)t * xstride + b * QK);
                    float d = 0.0f;
                    for (int k = 0; k < 4; k++) d += dot(lo[k], x4[k]) + dot(hi[k], x4[k + 4]);
                    acc[t] += d * s;
                }
            }
        }
    } else {
        device const char4 *rq = (device const char4 *)(wq + grow * nb * 32);
        for (int b = lane; b < nb; b += 32) {
            float4 w[8];
            for (int k = 0; k < 8; k++) w[k] = float4(rq[b * 8 + k]);
            const float s = rs[b];
            for (int t = 0; t < TOK_TILE; t++) {
                if (t < nt) {
                    auto x4 = as_f4(x + (ulong)t * xstride + b * QK);
                    float d = 0.0f;
                    for (int k = 0; k < 8; k++) d += dot(w[k], x4[k]);
                    acc[t] += d * s;
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
                out[o] = p.accumulate ? out[o] + v : v;
            }
        }
    }
}

// q, k and v projections of the same input in one dispatch: rows [0, rows0) come from W0 into out0,
// [rows0, rows0 + rows1) from W1 into out1, the rest from W2 into out2. All three share cols and QT.
template <int QT, bool NORM>
kernel void matmul3(device const uchar *wq0 [[buffer(0)]],
                    device const float *ws0 [[buffer(1)]],
                    device const uchar *wq1 [[buffer(2)]],
                    device const float *ws1 [[buffer(3)]],
                    device const uchar *wq2 [[buffer(4)]],
                    device const float *ws2 [[buffer(5)]],
                    device const float *x [[buffer(6)]],
                    device float *out0 [[buffer(7)]],
                    device float *out1 [[buffer(8)]],
                    device float *out2 [[buffer(9)]],
                    device const float *nw [[buffer(10)]],
                    constant Matmul3Params &p [[buffer(11)]],
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
    int rows;
    int r;
    if (row < p.rows0) {
        wq = wq0; ws = ws0; out = out0; rows = p.rows0; r = row;
    } else if (row < p.rows0 + p.rows1) {
        wq = wq1; ws = ws1; out = out1; rows = p.rows1; r = row - p.rows0;
    } else {
        wq = wq2; ws = ws2; out = out2; rows = p.rows2; r = row - p.rows0 - p.rows1;
    }
    float acc[TOK_TILE];
    for (int t = 0; t < TOK_TILE; t++) acc[t] = 0.0f;
    if (NORM) {
        mm_row<QT>(wq, ws, (ulong)r, nb, (threadgroup const float *)xn, 0, 1, acc, lane);
    } else {
        mm_row<QT>(wq, ws, (ulong)r, nb, x + (ulong)t0 * p.cols, (ulong)p.cols, nt, acc, lane);
    }
    for (int t = 0; t < TOK_TILE; t++) {
        if (t < nt) {
            float v = simd_sum(acc[t]);
            if (lane == 0) out[(ulong)(t0 + t) * rows + r] = v;
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
            if (lane == 0) out[(ulong)(t0 + t) * p.hidden + row] = g / (1.0f + exp(-g)) * u;
        }
    }
}

typedef decltype(matmul<4, false>) matmul_t;
template [[host_name("matmul_q4")]] kernel matmul_t matmul<4, false>;
template [[host_name("matmul_q8")]] kernel matmul_t matmul<8, false>;
template [[host_name("matmul_q4_norm")]] kernel matmul_t matmul<4, true>;
template [[host_name("matmul_q8_norm")]] kernel matmul_t matmul<8, true>;
typedef decltype(matmul3<4, false>) matmul3_t;
template [[host_name("matmul3_q4")]] kernel matmul3_t matmul3<4, false>;
template [[host_name("matmul3_q8")]] kernel matmul3_t matmul3<8, false>;
template [[host_name("matmul3_q4_norm")]] kernel matmul3_t matmul3<4, true>;
template [[host_name("matmul3_q8_norm")]] kernel matmul3_t matmul3<8, true>;
typedef decltype(matmul_swiglu<4, false>) matmul_swiglu_t;
template [[host_name("matmul_swiglu_q4")]] kernel matmul_swiglu_t matmul_swiglu<4, false>;
template [[host_name("matmul_swiglu_q8")]] kernel matmul_swiglu_t matmul_swiglu<8, false>;
template [[host_name("matmul_swiglu_q4_norm")]] kernel matmul_swiglu_t matmul_swiglu<4, true>;
template [[host_name("matmul_swiglu_q8_norm")]] kernel matmul_swiglu_t matmul_swiglu<8, true>;

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

kernel void norm_rope(device float *q [[buffer(0)]],
                      device float *k [[buffer(1)]],
                      device const float *qw [[buffer(2)]],
                      device const float *kw [[buffer(3)]],
                      device const float *cosT [[buffer(4)]],
                      device const float *sinT [[buffer(5)]],
                      constant RopeParams &p [[buffer(6)]],
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
    device float *h = isQ ? q + (ulong)tgp.y * p.qStride + (ulong)head * p.headDim
                          : k + (ulong)tgp.y * p.kvStride + (ulong)head * p.headDim;
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
    h[tid] = a * c - b * s;
    h[tid + half_dim] = b * c + a * s;
}

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
// Optional half-precision cache (backend.kv-dtype=f16). K and V are projected, normalized and rotated in f32
// staging buffers by the kernels above; kv_store_half then rounds them once (round-to-nearest-even, the same
// rounding as Java's Float.floatToFloat16) into the half caches, and attention_h is the attention kernel
// reading half K/V with the same f32 arithmetic as `attention`. The f32 kernels above are not involved.

struct KvStoreParams {
    int ntok;
    int pos0;
    int kvDim;
};

// kc/vc[pos0 + t, :] = half(k/v[t, :]) for t < ntok. One thread per element; grid: ceil(ntok * kvDim / 256).
kernel void kv_store_half(device const float *k [[buffer(0)]],
                          device const float *v [[buffer(1)]],
                          device half *kc [[buffer(2)]],
                          device half *vc [[buffer(3)]],
                          constant KvStoreParams &p [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
    if (int(gid) >= p.ntok * p.kvDim) return;
    const ulong o = (ulong)p.pos0 * p.kvDim + gid;
    kc[o] = half(k[gid]);
    vc[o] = half(v[gid]);
}

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

// ---------------------------------------------------------------------------------------------- bias
// v[i] += b[i % width] for i < total (Qwen2 q/k/v projection bias; v is [ntok, width] row-major).

struct BiasParams {
    int width;
    int total;
};

kernel void add_bias(device float *v [[buffer(0)]],
                     device const float *b [[buffer(1)]],
                     constant BiasParams &p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (int(gid) >= p.total) return;
    v[gid] += b[int(gid) % p.width];
}
