// Metal kernels of the audio models (Whisper's encoder and cross-attention projections, the speech codec). Compiled at run
// time by WhisperMetal / GpuCtx (its own library, apart from the text model kernels of metal/kernels.metal) and, for the
// numpy tests of scripts/metal-emu/kernel_tests.py, transpiled together with kernels.metal by the CPU emulator, so names
// carry a wh_ / WH_ prefix. Parameter structs hold 32-bit ints and floats only (GpuCtx writes them field by field).
//
//   1. constants and small helpers (erf, reductions)
//   2. wh_gemm        out[t][r] (+)= gelu?(bias[r] + sum_k x[t][k] * w[r][k]) with simdgroup-matrix tiles, batched over z
//   3. wh_layernorm   one threadgroup per row
//   4. wh_im2col_*    convolutions as products: the patches of conv1 (mel, channel-major) and conv2 (stride 2, time-major)
//   5. wh_split_heads q / k / v [n][d] to the per-head layouts attention uses (q scaled), k and v appended at a position
//   6. wh_softmax     rows of the score matrix, optionally causal
//   7. wh_embed / wh_add   x[t] = table[token t] + pos[pos0 + t] (the decoder's input), x += y

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------------------------- 1. constants

#define WH_BR 64                // weight rows (output features) per threadgroup of wh_gemm
#define WH_BT 32                // tokens per threadgroup of wh_gemm
#define WH_THREADS 128          // wh_gemm: 4 simdgroups, each owns a 32-row x 16-token quarter of the tile
#define WH_ROW_THREADS 256      // wh_layernorm / wh_softmax

#define WH_BIAS 1               // WhGemmParams.flags
#define WH_GELU 2
#define WH_ACC 4

// erf as Eigen / XLA evaluate it in float (relative error ~1e-7), the same polynomial as Dense.erf on the CPU
static inline float wh_erf(float x) {
    x = clamp(x, -4.0f, 4.0f);
    const float x2 = x * x;
    float p = -2.72614225801306e-10f;
    p = p * x2 + 2.77068142495902e-08f;
    p = p * x2 - 2.10102402082508e-06f;
    p = p * x2 - 5.69250639462346e-05f;
    p = p * x2 - 7.34990630326855e-04f;
    p = p * x2 - 2.95459980854025e-03f;
    p = (p * x2 - 1.60960333262415e-02f) * x;
    float q = -1.45660718464996e-05f;
    q = q * x2 - 2.13374055278905e-04f;
    q = q * x2 - 1.68282697438203e-03f;
    q = q * x2 - 7.37332916720468e-03f;
    q = q * x2 - 1.42647390514189e-02f;
    return p / q;
}

static inline float wh_gelu(float v) {
    return 0.5f * v * (1.0f + wh_erf(v * 0.70710678118654752f));
}

static inline float wh_tg_sum(float v, threadgroup float *buf, ushort sg, ushort lane, ushort nsg) {
    v = simd_sum(v);
    if (lane == 0) buf[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? buf[lane] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

static inline float wh_tg_max(float v, threadgroup float *buf, ushort sg, ushort lane, ushort nsg) {
    v = simd_max(v);
    if (lane == 0) buf[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? buf[lane] : -INFINITY;
    r = simd_max(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}


// 4-wide loads (any alignment of the element type) and stores of the staging loops of wh_gemm
static inline float4 wh_ld4(device const half *p) { return float4(*(device const packed_half4 *)p); }
static inline float4 wh_ld4(device const float *p) { return *(device const packed_float4 *)p; }
static inline void wh_st4(threadgroup half *d, float4 v) { *(threadgroup half4 *)d = half4(v); }
static inline void wh_st4(threadgroup float *d, float4 v) { *(threadgroup float4 *)d = v; }

// ---------------------------------------------------------------------------------------------- 2. wh_gemm
// out[t * ldo + r] (+)= act(bias[r] + sum_k x[t * ldx + k] * w[r * ldw + k]) for r < rows, t < ntok, k < cols, with
// WT the element type of the weights (half or float) and TT the type of the weight tile in threadgroup memory (half:
// the weight is rounded to 11 bits when staged, as llama.cpp's matmul does; float: exact). The activations are staged
// as float and the accumulators are float either way (the product is simdgroup_float8x8 x simdgroup_TT8x8). Grid:
// (ceil(rows / 64), ceil(ntok / 32), batch); z advances w, x and out by the *StrideZ fields (a batch of heads).
// One 32-wide k step per iteration: thread tid stages 16 weights of tile row tid >> 1 (halves of k) and 8 activations
// of tile token tid >> 2, then the four simdgroups multiply their 32 x 16 quarters block by block.

struct WhGemmParams {
    int rows;
    int cols;
    int ntok;
    int ldw;
    int ldx;
    int ldo;
    int flags;
    int wStrideZ;
    int xStrideZ;
    int oStrideZ;
};

template <typename WT, typename TT>
kernel void wh_gemm(device const WT *w [[buffer(0)]],
                    device const float *x [[buffer(1)]],
                    device float *out [[buffer(2)]],
                    device const float *bias [[buffer(3)]],
                    constant WhGemmParams &p [[buffer(4)]],
                    uint3 tgp [[threadgroup_position_in_grid]],
                    ushort tid [[thread_index_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[3072];                      // weight tile (half: 1024 floats, float: 2048) | activations 1024
    threadgroup TT *sa = (threadgroup TT *)smem;
    threadgroup float *sb = smem + 2048;
    const int row0 = int(tgp.x) * WH_BR;
    const int t0 = int(tgp.y) * WH_BT;
    device const WT *wz = w + (ulong)tgp.z * (ulong)p.wStrideZ;
    device const float *xz = x + (ulong)tgp.z * (ulong)p.xStrideZ;
    device float *oz = out + (ulong)tgp.z * (ulong)p.oStrideZ;
    const int lr = tid >> 1;                           // weight row of the tile this thread stages
    const int lh = tid & 1;                            // which 16 of the 32 k
    const int lt = tid >> 2;                           // token of the tile
    const int lp = tid & 3;                            // which 8 of the 32 k
    const int grow = min(row0 + lr, p.rows - 1);
    const int gtok = min(t0 + lt, p.ntok - 1);
    device const WT *wrow = wz + (ulong)grow * (ulong)p.ldw;
    device const float *xrow = xz + (ulong)gtok * (ulong)p.ldx;
    const ushort sr = sg & 1;                          // rows 32 sr .. 32 sr + 31 of the tile
    const ushort st = sg >> 1;                         // tokens 16 st .. 16 st + 15
    simdgroup_float8x8 acc[8];
    for (int i = 0; i < 8; i++) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    threadgroup TT *wd = sa + (lr >> 3) * 256 + (lr & 7) * 8;           // block row band, row inside the band
    threadgroup float *xd = sb + ((lt >> 3) * 4 + lp) * 64 + (lt & 7) * 8;
    for (int k0 = 0; k0 < p.cols; k0 += 32) {
        // stage 16 weights: k = k0 + 16 lh .. +15 into blocks 2 lh and 2 lh + 1 of this row, and 8 activations
        const int kw = k0 + 16 * lh;
        const int kx = k0 + 8 * lp;
        threadgroup TT *dst0 = wd + (2 * lh) * 64;
        threadgroup TT *dst1 = dst0 + 64;
        if (k0 + 32 <= p.cols) {                          // a whole k step: 4-wide loads
            device const WT *src = wrow + kw;
            wh_st4(dst0, wh_ld4(src));
            wh_st4(dst0 + 4, wh_ld4(src + 4));
            wh_st4(dst1, wh_ld4(src + 8));
            wh_st4(dst1 + 4, wh_ld4(src + 12));
            device const float *xs = xrow + kx;
            *(threadgroup float4 *)xd = *(device const packed_float4 *)xs;
            *(threadgroup float4 *)(xd + 4) = *(device const packed_float4 *)(xs + 4);
        } else {                                          // the last, partial step: zero past the end
            for (int j = 0; j < 8; j++) {
                dst0[j] = kw + j < p.cols ? TT(wrow[kw + j]) : TT(0);
                dst1[j] = kw + 8 + j < p.cols ? TT(wrow[kw + 8 + j]) : TT(0);
                xd[j] = kx + j < p.cols ? xrow[kx + j] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; kk++) {
            simdgroup_matrix<TT, 8, 8> wb[4];
            for (int j = 0; j < 4; j++) simdgroup_load(wb[j], sa + ((sr * 4 + j) * 4 + kk) * 64, 8, ulong2(0, 0), true);
            for (int i = 0; i < 2; i++) {
                simdgroup_float8x8 a;
                simdgroup_load(a, sb + ((st * 2 + i) * 4 + kk) * 64, 8);
                for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i * 4 + j], a, wb[j], acc[i * 4 + j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // the f32 result tile [token][row] (ld 64) in smem, then the guarded epilogue
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
            simdgroup_store(acc[i * 4 + j], smem + (st * 16 + i * 8) * WH_BR + sr * 32 + j * 8, WH_BR);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < WH_BR * WH_BT; i += WH_THREADS) {
        const int t = i / WH_BR, r = i % WH_BR;
        if (t0 + t < p.ntok && row0 + r < p.rows) {
            float v = smem[t * WH_BR + r];
            if (p.flags & WH_BIAS) v += bias[row0 + r];
            if (p.flags & WH_GELU) v = wh_gelu(v);
            const ulong o = (ulong)(t0 + t) * (ulong)p.ldo + (ulong)(row0 + r);
            oz[o] = (p.flags & WH_ACC) ? oz[o] + v : v;
        }
    }
}

template [[host_name("wh_gemm_hh")]] kernel decltype(wh_gemm<half, half>) wh_gemm<half, half>;
template [[host_name("wh_gemm_fh")]] kernel decltype(wh_gemm<float, half>) wh_gemm<float, half>;
template [[host_name("wh_gemm_ff")]] kernel decltype(wh_gemm<float, float>) wh_gemm<float, float>;
template [[host_name("wh_gemm_hf")]] kernel decltype(wh_gemm<half, float>) wh_gemm<half, float>;

// ---------------------------------------------------------------------------------------------- 3. wh_layernorm
// out[row] = (x[row] - mean) / sqrt(var + eps) * gamma + beta, one threadgroup per row of n values. out may be x.

struct WhLnParams {
    int n;
    float eps;
    int hasBeta;
};

kernel void wh_layernorm(device const float *x [[buffer(0)]],
                         device float *out [[buffer(1)]],
                         device const float *gamma [[buffer(2)]],
                         device const float *beta [[buffer(3)]],
                         constant WhLnParams &p [[buffer(4)]],
                         uint row [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float buf[32];
    device const float *xr = x + (ulong)row * (ulong)p.n;
    device float *orow = out + (ulong)row * (ulong)p.n;
    const ushort nsg = WH_ROW_THREADS / 32;
    float s = 0.0f;
    for (int i = tid; i < p.n; i += WH_ROW_THREADS) s += xr[i];
    const float mean = wh_tg_sum(s, buf, sg, lane, nsg) / float(p.n);
    float v = 0.0f;
    for (int i = tid; i < p.n; i += WH_ROW_THREADS) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    const float inv = 1.0f / sqrt(wh_tg_sum(v, buf, sg, lane, nsg) / float(p.n) + p.eps);
    for (int i = tid; i < p.n; i += WH_ROW_THREADS) {
        orow[i] = (xr[i] - mean) * inv * gamma[i] + (p.hasBeta ? beta[i] : 0.0f);
    }
}

// ---------------------------------------------------------------------------------------------- 4. im2col

struct WhConvParams {
    int t;          // output positions
    int c;          // input channels
    int tin;        // input positions
};

// conv1: patches X[t][ch * 3 + k] = mel[ch][t + k - 1] (zero outside), mel channel-major [c][tin]; one thread per element
kernel void wh_im2col_mel(device const float *mel [[buffer(0)]],
                          device float *col [[buffer(1)]],
                          constant WhConvParams &p [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    const int total = p.t * p.c * 3;
    if (int(gid) >= total) return;
    const int k = int(gid) % 3;
    const int ch = (int(gid) / 3) % p.c;
    const int t = int(gid) / (3 * p.c);
    const int i = t + k - 1;
    col[gid] = (i >= 0 && i < p.tin) ? mel[(ulong)ch * (ulong)p.tin + (ulong)i] : 0.0f;
}

// conv2 (stride 2): patches X[t][ch * 3 + k] = h[2 t + k - 1][ch], h time-major [tin][c]
kernel void wh_im2col_s2(device const float *h [[buffer(0)]],
                         device float *col [[buffer(1)]],
                         constant WhConvParams &p [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    const int total = p.t * p.c * 3;
    if (int(gid) >= total) return;
    const int k = int(gid) % 3;
    const int ch = (int(gid) / 3) % p.c;
    const int t = int(gid) / (3 * p.c);
    const int i = 2 * t + k - 1;
    col[gid] = (i >= 0 && i < p.tin) ? h[(ulong)i * (ulong)p.c + (ulong)ch] : 0.0f;
}

// ---------------------------------------------------------------------------------------------- 5. wh_split_heads
// q, k, v [n][d] (token-major) into  qh[h][n][hd] (times scale),  kh[h][cap][hd] rows pos0 .. pos0 + n,  vt[h][hd][cap]
// columns pos0 .. pos0 + n. One thread per (token, feature).

struct WhSplitParams {
    int n;
    int heads;
    int hd;
    int cap;        // positions per head in kh / vt
    int pos0;
    float scale;    // applied to q
    int doQ;
    int doKV;
};

kernel void wh_split_heads(device const float *q [[buffer(0)]],
                           device const float *k [[buffer(1)]],
                           device const float *v [[buffer(2)]],
                           device float *qh [[buffer(3)]],
                           device float *kh [[buffer(4)]],
                           device float *vt [[buffer(5)]],
                           constant WhSplitParams &p [[buffer(6)]],
                           uint gid [[thread_position_in_grid]]) {
    const int d = p.heads * p.hd;
    if (int(gid) >= p.n * d) return;
    const int f = int(gid) % d;
    const int t = int(gid) / d;
    const int h = f / p.hd;
    const int e = f % p.hd;
    if (p.doQ) qh[((ulong)h * p.n + t) * p.hd + e] = q[gid] * p.scale;
    if (p.doKV) {
        kh[((ulong)h * p.cap + p.pos0 + t) * p.hd + e] = k[gid];
        vt[((ulong)h * p.hd + e) * p.cap + p.pos0 + t] = v[gid];
    }
}

// ---------------------------------------------------------------------------------------------- 6. wh_softmax
// In place over the rows of S[h][n][len]: row (h, t) is softmax'ed over its first vis = causal ? min(len, pos0 + t + 1) : len
// entries; the rest are set to 0. One threadgroup per row; grid (n, heads).

struct WhSoftmaxParams {
    int n;
    int len;
    int causal;
    int pos0;
};

kernel void wh_softmax(device float *s [[buffer(0)]],
                       constant WhSoftmaxParams &p [[buffer(1)]],
                       uint2 tgp [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float buf[32];
    const int t = int(tgp.x);
    const int h = int(tgp.y);
    device float *row = s + ((ulong)h * p.n + t) * (ulong)p.len;
    const int vis = p.causal ? min(p.len, p.pos0 + t + 1) : p.len;
    const ushort nsg = WH_ROW_THREADS / 32;
    float m = -INFINITY;
    for (int i = tid; i < vis; i += WH_ROW_THREADS) m = max(m, row[i]);
    m = wh_tg_max(m, buf, sg, lane, nsg);
    float sum = 0.0f;
    for (int i = tid; i < vis; i += WH_ROW_THREADS) {
        const float e = precise::exp(row[i] - m);
        row[i] = e;
        sum += e;
    }
    const float inv = 1.0f / wh_tg_sum(sum, buf, sg, lane, nsg);
    for (int i = tid; i < p.len; i += WH_ROW_THREADS) row[i] = i < vis ? row[i] * inv : 0.0f;
}

// ---------------------------------------------------------------------------------------------- 7. copies

struct WhCopyParams {
    int n;          // elements
};

// out[i] = a[i] + b[i]  (adding a position table to a conv output, say)
kernel void wh_add(device float *a [[buffer(0)]],
                   device const float *b [[buffer(1)]],
                   constant WhCopyParams &p [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (int(gid) < p.n) a[gid] += b[gid];
}

struct WhEmbedParams {
    int n;          // tokens
    int d;
    int pos0;       // position of the first token
};

// x[t][i] = table[tok[t]][i] + pos[pos0 + t][i]: the decoder's token embedding plus its learned position, WT = half or float
template <typename WT>
kernel void wh_embed(device const WT *table [[buffer(0)]],
                     device const float *pos [[buffer(1)]],
                     device const int *tok [[buffer(2)]],
                     device float *x [[buffer(3)]],
                     constant WhEmbedParams &p [[buffer(4)]],
                     uint gid [[thread_position_in_grid]]) {
    if (int(gid) >= p.n * p.d) return;
    const int t = int(gid) / p.d;
    const int i = int(gid) % p.d;
    x[gid] = float(table[(ulong)tok[t] * (ulong)p.d + (ulong)i]) + pos[(ulong)(p.pos0 + t) * (ulong)p.d + (ulong)i];
}

template [[host_name("wh_embed_h")]] kernel decltype(wh_embed<half>) wh_embed<half>;
template [[host_name("wh_embed_f")]] kernel decltype(wh_embed<float>) wh_embed<float>;
