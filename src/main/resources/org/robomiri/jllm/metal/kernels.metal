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
// dequantized block. Grid: (ceil(rows / simdgroups_per_tg), ceil(ntok / TOK_TILE)).

struct MatmulParams {
    int rows;
    int cols;
    int ntok;
    int accumulate;
};

kernel void matmul_q4(device const uchar *wq [[buffer(0)]],
                      device const float *ws [[buffer(1)]],
                      device const float *x [[buffer(2)]],
                      device float *out [[buffer(3)]],
                      constant MatmulParams &p [[buffer(4)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]],
                      ushort nsg [[simdgroups_per_threadgroup]]) {
    const int row = int(tgp.x) * nsg + sg;
    if (row >= p.rows) return;
    const int t0 = int(tgp.y) * TOK_TILE;
    const int nt = min(TOK_TILE, p.ntok - t0);
    const int nb = p.cols / QK;
    device const uchar4 *rq = (device const uchar4 *)(wq + (ulong)row * nb * 16);
    device const float *rs = ws + (ulong)row * nb;

    float acc[TOK_TILE];
    for (int t = 0; t < TOK_TILE; t++) acc[t] = 0.0f;

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
                device const float4 *x4 = (device const float4 *)(x + (ulong)(t0 + t) * p.cols + b * QK);
                float d = 0.0f;
                for (int k = 0; k < 4; k++) d += dot(lo[k], x4[k]) + dot(hi[k], x4[k + 4]);
                acc[t] += d * s;
            }
        }
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

kernel void matmul_q8(device const char *wq [[buffer(0)]],
                      device const float *ws [[buffer(1)]],
                      device const float *x [[buffer(2)]],
                      device float *out [[buffer(3)]],
                      constant MatmulParams &p [[buffer(4)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]],
                      ushort nsg [[simdgroups_per_threadgroup]]) {
    const int row = int(tgp.x) * nsg + sg;
    if (row >= p.rows) return;
    const int t0 = int(tgp.y) * TOK_TILE;
    const int nt = min(TOK_TILE, p.ntok - t0);
    const int nb = p.cols / QK;
    device const char4 *rq = (device const char4 *)(wq + (ulong)row * nb * 32);
    device const float *rs = ws + (ulong)row * nb;

    float acc[TOK_TILE];
    for (int t = 0; t < TOK_TILE; t++) acc[t] = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        float4 w[8];
        for (int k = 0; k < 8; k++) w[k] = float4(rq[b * 8 + k]);
        const float s = rs[b];
        for (int t = 0; t < TOK_TILE; t++) {
            if (t < nt) {
                device const float4 *x4 = (device const float4 *)(x + (ulong)(t0 + t) * p.cols + b * QK);
                float d = 0.0f;
                for (int k = 0; k < 8; k++) d += dot(w[k], x4[k]);
                acc[t] += d * s;
            }
        }
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
// In place on [ntok, rowStride] rows holding nHeads heads of headDim. Grid: (nHeads, ntok);
// threads per threadgroup = headDim / 2, thread i owns the rotary pair (i, i + headDim/2).

struct RopeParams {
    int headDim;
    int rowStride;
    int pos0;
    float eps;
    int useNorm;
};

kernel void norm_rope(device float *v [[buffer(0)]],
                      device const float *w [[buffer(1)]],
                      device const float *cosT [[buffer(2)]],
                      device const float *sinT [[buffer(3)]],
                      constant RopeParams &p [[buffer(4)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      ushort2 tid2 [[thread_position_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]],
                      ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[32];
    const ushort tid = tid2.x;
    const int half_dim = p.headDim / 2;
    device float *h = v + (ulong)tgp.y * p.rowStride + (ulong)tgp.x * p.headDim;
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
// Causal GQA attention for token t at position pos0 + t. Grid: (nHeads, ntok), 256 threads = 8 simdgroups.
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
    const int t = int(tgp.y);
    const int kvh = h / (p.nHeads / p.nKvHeads);
    const int len = p.pos0 + t + 1;
    const int per = p.headDim / 32;
    device const float *qv = q + (ulong)t * p.qStride + h * p.headDim;
    device float *sc = scores + ((ulong)t * p.nHeads + h) * p.ctx;
    device const float *kb = kc + kvh * p.headDim;
    device const float *vb = vc + kvh * p.headDim;

    float qr[8];
    for (int i = 0; i < 8; i++) qr[i] = i < per ? qv[lane + i * 32] : 0.0f;

    // 1. scores
    float mx = -INFINITY;
    for (int s = sg; s < len; s += nsg) {
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
    for (int s = tid; s < len; s += tpg) {
        float e = exp(sc[s] - mx);
        sc[s] = e;
        sum += e;
    }
    sum = tg_sum(sum, buf, sg, lane, nsg);
    threadgroup_barrier(mem_flags::mem_device);

    // 3. weighted sum of values, split over simdgroups then reduced
    float o[8];
    for (int i = 0; i < 8; i++) o[i] = 0.0f;
    for (int s = sg; s < len; s += nsg) {
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

// ---------------------------------------------------------------------------------------------- mixture of experts
// router: one threadgroup per token with nExperts threads. Thread e computes logit e; thread 0 then does the
// softmax and the top-k selection (nExperts <= 1024). Writes assign[t*topK+k] and routeWeight[t*topK+k].

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
                   ushort tid [[thread_position_in_threadgroup]]) {
    threadgroup float logits[1024];
    device const float *xv = x + (ulong)tg * p.dim;
    device const float *wv = w + (ulong)tid * p.dim;
    float d = 0.0f;
    for (int i = 0; i < p.dim; i++) d += wv[i] * xv[i];
    logits[tid] = d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float mx = -INFINITY;
        for (int e = 0; e < p.nExperts; e++) mx = max(mx, logits[e]);
        float sum = 0.0f;
        for (int e = 0; e < p.nExperts; e++) {
            logits[e] = exp(logits[e] - mx);
            sum += logits[e];
        }
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

// moe_up: out[s, row] = W_{assign[s]}[row] . x[s / topK]   for slot s = token*topK + k.
// Grid: (ceil(rowsPerExpert / simdgroups), ntok * topK). One simdgroup per output row.

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
    const int row = int(tgp.x) * nsg + sg;
    if (row >= p.rowsPerExpert) return;
    const int s = int(tgp.y);
    const int t = s / p.topK;
    const ulong grow = (ulong)assign[s] * p.rowsPerExpert + row;
    const int nb = p.cols / QK;
    device const float4 *x4 = (device const float4 *)(x + (ulong)t * p.cols);
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) acc += block_dot<QT>(wq, ws, grow, nb, b, x4 + b * 8);
    acc = simd_sum(acc);
    if (lane == 0) out[(ulong)s * p.rowsPerExpert + row] = acc;
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

// ---------------------------------------------------------------------------------------------- swiglu
// gu is [ntok, 2n] (gate | up); out is [ntok, n].

struct SiluParams {
    int n;
    int total;
};

kernel void silu_mul(device const float *gu [[buffer(0)]],
                     device float *out [[buffer(1)]],
                     constant SiluParams &p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (int(gid) >= p.total) return;
    const int t = int(gid) / p.n;
    const int i = int(gid) - t * p.n;
    const float g = gu[(ulong)t * 2 * p.n + i];
    const float u = gu[(ulong)t * 2 * p.n + p.n + i];
    out[gid] = g / (1.0f + exp(-g)) * u;
}
