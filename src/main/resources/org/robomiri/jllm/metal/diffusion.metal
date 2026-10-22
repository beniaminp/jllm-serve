// Compute kernels of the image-generation stack (org.robomiri.jllm.diffusion.MetalOps) for Apple GPUs: the UNet,
// the VAE and the CLIP text encoders of Stable Diffusion. A separate Metal library from kernels.metal (the language-model
// kernels), compiled when an image model is loaded. Every dispatch is written by MetalOps.java; the parameter structs hold
// only 32-bit ints / floats, written field by field (DiffusionParams lists the layouts, DiffusionParamsTest checks them
// against the structs below). The CPU emulator (scripts/metal-emu) transpiles this file together with kernels.metal.
//
// Layout: activations are channels-last. [B, H, W, C] for images, [B, L, C] / [rows, C] for token sequences; a linear weight
// is [N, K], a convolution weight [Cout, kh, kw, Cin] (K = kh * kw * Cin). Activations and weights are `half` or `float`
// (the kernels are templates over T, instantiated as *_h and *_f); all arithmetic and accumulation is float.
// Sections:  1. common   2. matrix product (gemm), convolution as implicit gemm   3. direct convolution (few channels)
//            4. normalisation   5. attention (flash, and scores + softmax for wide heads)   6. elementwise and layout

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------------------------- 1. common

#define DF_THREADS 128          // gemm / conv / flash threadgroup: 4 simdgroups
#define DF_BM 64                // output rows of a gemm tile
#define DF_BN 64                // output columns of a gemm tile
#define DF_BK 32                // k slice per step
#define DF_ROW_THREADS 256      // normalisation / softmax threadgroup (one row or group each)
#define DF_FA_ROWS 32           // query rows per flash-attention threadgroup (4 simdgroups x 8 rows)
#define DF_FA_BK 32             // keys per flash-attention block
#define DF_EW_THREADS 256       // elementwise threadgroup

#define ACT_NONE 0
#define ACT_SILU 1
#define ACT_GELU 2
#define ACT_QUICK_GELU 3

static inline float df_tg_sum(float v, threadgroup float *buf, ushort sg, ushort lane, ushort nsg) {
    v = simd_sum(v);
    if (lane == 0) buf[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? buf[lane] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

static inline float df_tg_max(float v, threadgroup float *buf, ushort sg, ushort lane, ushort nsg) {
    v = simd_max(v);
    if (lane == 0) buf[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? buf[lane] : -INFINITY;
    r = simd_max(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// erf with an absolute error below 7e-8: a rational approximation (the CPU path uses the same coefficients)
static inline float df_erf(float x) {
    x = clamp(x, -4.0f, 4.0f);
    const float x2 = x * x;
    float p = -2.72614225801306e-10f;
    p = p * x2 + 2.77068142495902e-08f;
    p = p * x2 + -2.10102402082508e-06f;
    p = p * x2 + -5.69250639462346e-05f;
    p = p * x2 + -7.34990630326855e-04f;
    p = p * x2 + -2.95459980854025e-03f;
    p = p * x2 + -1.60960333262415e-02f;
    float q = -1.45660718464996e-05f;
    q = q * x2 + -2.13374055278905e-04f;
    q = q * x2 + -1.68282697438203e-03f;
    q = q * x2 + -7.37332916720468e-03f;
    q = q * x2 + -1.42647390514189e-02f;
    return x * p / q;
}

static inline float df_gelu(float x) {
    return 0.5f * x * (1.0f + df_erf(x * 0.70710678118654752f));
}

static inline float df_silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float df_act(float v, int act) {
    if (act == ACT_SILU) return df_silu(v);
    if (act == ACT_GELU) return df_gelu(v);
    if (act == ACT_QUICK_GELU) return v / (1.0f + exp(-1.702f * v));
    return v;
}

// ---------------------------------------------------------------------------------------------- 2. matrix product
// df_gemm<TA, TB, TC, TM, BT>: C[z][m][n] = act(alpha * sum_k A[z][m][k] * B[z][n][k] + bias[n]) + res[m][n]
//   BT = true : B is [N, K] rows (a weight, or the keys of attention); BT = false: B is [K, N] rows (the values).
// A threadgroup (4 simdgroups) computes a 64 x BNT tile (BNT = 64 for half, 32 for float: see DF_BNT) in 32-wide k slices
// staged in threadgroup memory as TM (half or float); each simdgroup owns a quarter of it (4 x SJ simdgroup 8 x 8
// accumulators, float). z = batch * heads + head
// indexes the independent products of attention (strides in elements); a plain product has heads = 1, z = 0. Rows /
// columns / k past m / n / k read as zero and are not written. With vec = 1 (lda, ldb and the strides are multiples of 4
// elements and the pointers aligned) full 16-element runs are loaded as 4-vectors.

struct DfGemmParams {
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    int heads;
    int aSB;        // element offset of A per batch index / per head index (same for B: bSB bSH, C: cSB cSH)
    int aSH;
    int bSB;
    int bSH;
    int cSB;
    int cSH;
    float alpha;
    int hasBias;
    int hasRes;     // res is [m, ldc] like C (read through the batch offsets of C)
    int act;
    int vec;
};

// 4 elements converted to another element type (Metal's float4(half4))
template <typename TD, typename TS>
static inline vec<TD, 4> df_cvt4(vec<TS, 4> v) {
    return vec<TD, 4>{TD(v.x), TD(v.y), TD(v.z), TD(v.w)};
}

// Tile shapes. Four simdgroups share a threadgroup and each owns 4 x SJ blocks of 8 x 8 (float) accumulators:
//   half:  SJ = 4, a 64 x 64 output tile (every simdgroup a 32 x 32 quarter, 16 accumulators)
//   float: SJ = 2, a 64 x 32 output tile (a 32 x 16 quarter, 8 accumulators): with 16 float accumulators and float fragments
//          the compiler's register allocation falls over and the kernel runs 8 times slower, measured
// BNT is the tile's width in n. The staged slices (TM) and the float tile the accumulators go through before the epilogue are
// optionally padded (DF_PAD_BYTES per row) against threadgroup-memory bank conflicts; DF_LOAD16B copies half rows with two
// 16-byte loads instead of four 8-byte ones (vec = 1 then means: all strides and offsets are multiples of 8 elements).
//   LDK: row stride (TM elements) of a slice that is 32 wide in k     LDN: of a slice that is BNT wide in n
//   LDS: row stride (floats) of the accumulator tile
#ifndef DF_PAD_BYTES
#define DF_PAD_BYTES 0
#endif
#ifndef DF_LOAD16B
#define DF_LOAD16B 0
#endif
#define DF_BNT(TM) (sizeof(TM) == 4 ? 32 : 64)
#define DF_SJ(TM) (DF_BNT(TM) / 16)
#define DF_LDK(TM) (DF_BK + DF_PAD_BYTES / int(sizeof(TM)))
#define DF_LDN(TM) (DF_BNT(TM) + DF_PAD_BYTES / int(sizeof(TM)))
#define DF_LDS(TM) (DF_BNT(TM) + DF_PAD_BYTES / 4)
#define DF_SMEM_FLOATS 5120     // 20 KB: the two slices (at most 2 x 64 x 36 x 4 B) or the accumulator tile

// 8 contiguous elements (16-byte aligned for half) to threadgroup memory, converted when the types differ
template <typename TS, typename TM>
static inline void df_load8(device const TS *src, threadgroup TM *dst) {
    device const vec<TS, 4> *s4 = (device const vec<TS, 4> *)src;
    threadgroup vec<TM, 4> *d4 = (threadgroup vec<TM, 4> *)dst;
    d4[0] = df_cvt4<TM, TS>(s4[0]);
    d4[1] = df_cvt4<TM, TS>(s4[1]);
}

template <typename TS, typename TM>
static inline void df_load16(device const TS *src, threadgroup TM *dst, int avail, bool useVec) {
    if (useVec && avail >= 16) {
        device const vec<TS, 4> *s4 = (device const vec<TS, 4> *)src;
        threadgroup vec<TM, 4> *d4 = (threadgroup vec<TM, 4> *)dst;
        for (int i = 0; i < 4; i++) d4[i] = df_cvt4<TM, TS>(s4[i]);
    } else {
        for (int i = 0; i < 16; i++) dst[i] = i < avail ? TM(src[i]) : TM(0);
    }
}

#if DF_LOAD16B
static inline void df_load8(device const half *src, threadgroup half *dst) {
    *(threadgroup float4 *)dst = *(device const float4 *)src;
}

static inline void df_load16(device const half *src, threadgroup half *dst, int avail, bool useVec) {
    if (useVec && avail >= 16) {
        device const float4 *s4 = (device const float4 *)src;
        threadgroup float4 *d4 = (threadgroup float4 *)dst;
        d4[0] = s4[0];
        d4[1] = s4[1];
    } else {
        for (int i = 0; i < 16; i++) dst[i] = i < avail ? src[i] : half(0);
    }
}
#endif

// the accumulators of one simdgroup (4 x SJ blocks) over the staged 64 x 32 A and BNT x 32 B slices
template <typename TM, bool BT>
static inline void df_mma(threadgroup TM *sa, threadgroup TM *sb, thread simdgroup_float8x8 *acc, ushort sr, ushort sc) {
    constexpr int LDK = DF_LDK(TM);
    constexpr int LDN = DF_LDN(TM);
    constexpr int SJ = DF_SJ(TM);
    for (int kk = 0; kk < 4; kk++) {
        simdgroup_matrix<TM, 8, 8> bf[SJ];
        for (int j = 0; j < SJ; j++) {
            if (BT) simdgroup_load(bf[j], sb + (sc * 8 * SJ + j * 8) * LDK + kk * 8, LDK, ulong2(0, 0), true);
            else simdgroup_load(bf[j], sb + (kk * 8) * LDN + sc * 8 * SJ + j * 8, LDN);
        }
        for (int i = 0; i < 4; i++) {  // one A fragment at a time: fewer live registers
            simdgroup_matrix<TM, 8, 8> af;
            simdgroup_load(af, sa + (sr * 32 + i * 8) * LDK + kk * 8, LDK);
            for (int j = 0; j < SJ; j++) simdgroup_multiply_accumulate(acc[i * SJ + j], af, bf[j], acc[i * SJ + j]);
        }
    }
}

template <typename TA, typename TB, typename TC, typename TM, bool BT>
kernel void df_gemm(device const TA *a [[buffer(0)]],
                    device const TB *b [[buffer(1)]],
                    device TC *c [[buffer(2)]],
                    device const float *bias [[buffer(3)]],
                    device const TC *res [[buffer(4)]],
                    constant DfGemmParams &p [[buffer(5)]],
                    uint3 tgp [[threadgroup_position_in_grid]],
                    ushort tid [[thread_index_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int LDK = DF_LDK(TM);
    constexpr int LDN = DF_LDN(TM);
    constexpr int LDS = DF_LDS(TM);
    constexpr int BNT = DF_BNT(TM);
    constexpr int SJ = DF_SJ(TM);
    threadgroup float smem[DF_SMEM_FLOATS];
    threadgroup TM *sa = (threadgroup TM *)smem;
    threadgroup TM *sb = sa + DF_BM * LDK;
    const int n0 = int(tgp.x) * BNT;
    const int m0 = int(tgp.y) * DF_BM;
    const int z = int(tgp.z);
    const int zb = z / p.heads;
    const int zh = z - zb * p.heads;
    device const TA *ab = a + (long)zb * p.aSB + (long)zh * p.aSH;
    device const TB *bb = b + (long)zb * p.bSB + (long)zh * p.bSH;
    device TC *cb = c + (long)zb * p.cSB + (long)zh * p.cSH;
    const ushort sr = sg & 1;
    const ushort sc = sg >> 1;
    const bool useVec = p.vec != 0;

    simdgroup_float8x8 acc[4 * SJ];
    for (int i = 0; i < 4 * SJ; i++) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const int ar = tid >> 1;            // A rows / BT B rows: 64 rows, two threads (16 k each) per row
    const int ak = (tid & 1) * 16;
    const int bk = tid >> 2;            // !BT: B rows are k (32), four threads (16 n each) per row
    const int bn = (tid & 3) * 16;
    for (int k0 = 0; k0 < p.k; k0 += DF_BK) {
        {
            const int m = m0 + ar;
            threadgroup TM *dst = sa + ar * LDK + ak;
            if (m < p.m) df_load16<TA, TM>(ab + (long)m * p.lda + k0 + ak, dst, p.k - k0 - ak, useVec);
            else for (int i = 0; i < 16; i++) dst[i] = TM(0);
        }
        if (BT) {
            if (ar < BNT) {
                const int n = n0 + ar;
                threadgroup TM *dst = sb + ar * LDK + ak;
                if (n < p.n) df_load16<TB, TM>(bb + (long)n * p.ldb + k0 + ak, dst, p.k - k0 - ak, useVec);
                else for (int i = 0; i < 16; i++) dst[i] = TM(0);
            }
        } else if (bn < BNT) {
            const int kr = k0 + bk;
            threadgroup TM *dst = sb + bk * LDN + bn;
            if (kr < p.k) df_load16<TB, TM>(bb + (long)kr * p.ldb + n0 + bn, dst, p.n - n0 - bn, useVec);
            else for (int i = 0; i < 16; i++) dst[i] = TM(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        df_mma<TM, BT>(sa, sb, acc, sr, sc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // tile -> threadgroup memory (aliasing the slices) -> epilogue
    threadgroup float *stg = smem;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < SJ; j++) simdgroup_store(acc[i * SJ + j], stg + (sr * 32 + i * 8) * LDS + sc * 8 * SJ + j * 8, LDS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const TC *rb = res + (long)zb * p.cSB + (long)zh * p.cSH;
    for (int e = tid; e < DF_BM * BNT; e += DF_THREADS) {
        const int r = e / BNT;
        const int cc = e - r * BNT;
        const int m = m0 + r;
        const int n = n0 + cc;
        if (m < p.m && n < p.n) {
            float v = stg[r * LDS + cc] * p.alpha;
            if (p.hasBias != 0) v += bias[n];
            v = df_act(v, p.act);
            if (p.hasRes != 0) v += float(rb[(long)m * p.ldc + n]);
            cb[(long)m * p.ldc + n] = TC(v);
        }
    }
}

// df_conv<T, TM>: out[b][oy][ox][co] = act(sum_{ky,kx,ci} x[b][oy*stride - padT + ky][ox*stride - padL + kx][ci] *
// w[co][ky][kx][ci] + bias[co] + chan[b][co]) + res: a convolution as an implicit gemm. Row m of the gemm is an output
// pixel, k a (tap, input channel) pair; the A slices are gathered from the input while they are staged (zeros outside the
// image), so no im2col buffer exists. cin is a multiple of 8, so 8 consecutive k stay inside one tap.

struct DfConvParams {
    int batch;
    int h;
    int w;
    int cin;
    int ho;
    int wo;
    int cout;
    int kh;
    int kw;
    int stride;
    int padT;
    int padL;
    int hasBias;
    int hasChan;    // chan is [batch, cout]
    int hasRes;     // res is [batch, ho, wo, cout]
    int ups;        // 1: the input is read through a nearest-neighbour 2x upsampling: h and w are the upsampled size
};

template <typename T, typename TM>
kernel void df_conv(device const T *x [[buffer(0)]],
                    device const T *w [[buffer(1)]],
                    device T *out [[buffer(2)]],
                    device const float *bias [[buffer(3)]],
                    device const T *chan [[buffer(4)]],
                    device const T *res [[buffer(5)]],
                    constant DfConvParams &p [[buffer(6)]],
                    uint3 tgp [[threadgroup_position_in_grid]],
                    ushort tid [[thread_index_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int LDK = DF_LDK(TM);
    constexpr int LDS = DF_LDS(TM);
    constexpr int BNT = DF_BNT(TM);
    constexpr int SJ = DF_SJ(TM);
    threadgroup float smem[DF_SMEM_FLOATS];
    threadgroup TM *sa = (threadgroup TM *)smem;
    threadgroup TM *sb = sa + DF_BM * LDK;
    const int n0 = int(tgp.x) * BNT;
    const int m0 = int(tgp.y) * DF_BM;
    const int M = p.batch * p.ho * p.wo;
    const int K = p.kh * p.kw * p.cin;
    const ushort sr = sg & 1;
    const ushort sc = sg >> 1;

    simdgroup_float8x8 acc[4 * SJ];
    for (int i = 0; i < 4 * SJ; i++) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const int ar = tid >> 1;
    const int ak = (tid & 1) * 16;
    // the output pixel of this thread's A row
    const int m = m0 + ar;
    int bi = 0, oy = 0, ox = 0;
    if (m < M) {
        bi = m / (p.ho * p.wo);
        const int rem = m - bi * (p.ho * p.wo);
        oy = rem / p.wo;
        ox = rem - oy * p.wo;
    }
    const int n = n0 + ar;
    for (int k0 = 0; k0 < K; k0 += DF_BK) {
        threadgroup TM *da = sa + ar * LDK + ak;
        for (int g = 0; g < 2; g++) {
            const int kb = k0 + ak + g * 8;
            if (m < M && kb < K) {
                const int tap = kb / p.cin;
                const int ci = kb - tap * p.cin;
                const int ky = tap / p.kw;
                const int kx = tap - ky * p.kw;
                const int iy = oy * p.stride - p.padT + ky;
                const int ix = ox * p.stride - p.padL + kx;
                if (iy >= 0 && iy < p.h && ix >= 0 && ix < p.w) {
                    df_load8(x + ((long)(bi * (p.h >> p.ups) + (iy >> p.ups)) * (p.w >> p.ups) + (ix >> p.ups)) * p.cin + ci,
                             da + g * 8);
                } else {
                    for (int i = 0; i < 8; i++) da[g * 8 + i] = TM(0);
                }
            } else {
                for (int i = 0; i < 8; i++) da[g * 8 + i] = TM(0);
            }
        }
        if (ar < BNT) {
            threadgroup TM *db = sb + ar * LDK + ak;
            if (n < p.cout) {
                device const T *wr = w + (long)n * K + k0 + ak;
                const int avail = K - k0 - ak;
                if (avail >= 16) {
                    df_load8(wr, db);
                    df_load8(wr + 8, db + 8);
                } else {
                    for (int i = 0; i < 16; i++) db[i] = i < avail ? TM(wr[i]) : TM(0);
                }
            } else {
                for (int i = 0; i < 16; i++) db[i] = TM(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        df_mma<TM, true>(sa, sb, acc, sr, sc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float *stg = smem;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < SJ; j++) simdgroup_store(acc[i * SJ + j], stg + (sr * 32 + i * 8) * LDS + sc * 8 * SJ + j * 8, LDS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = tid; e < DF_BM * BNT; e += DF_THREADS) {
        const int r = e / BNT;
        const int cc = e - r * BNT;
        const int mm = m0 + r;
        const int nn = n0 + cc;
        if (mm < M && nn < p.cout) {
            float v = stg[r * LDS + cc];
            if (p.hasBias != 0) v += bias[nn];
            if (p.hasChan != 0) v += float(chan[(long)(mm / (p.ho * p.wo)) * p.cout + nn]);
            if (p.hasRes != 0) v += float(res[(long)mm * p.cout + nn]);
            out[(long)mm * p.cout + nn] = T(v);
        }
    }
}

#define DF_INST_GEMM(NAME, TA, TB, TC, TM, BT) \
    template [[host_name(NAME)]] kernel decltype(df_gemm<TA, TB, TC, TM, BT>) df_gemm<TA, TB, TC, TM, BT>;

// weights x activations (linear layers, projections, convolutions as 1 x 1)
DF_INST_GEMM("df_gemm_nt_f", float, float, float, float, true)
DF_INST_GEMM("df_gemm_nt_h", half, half, half, half, true)
// attention scores Q K^T (float scores out) and P V (float probabilities in, values T)
// (in float32 the scores product is df_gemm_nt_f itself)
DF_INST_GEMM("df_scores_h", half, half, float, half, true)
DF_INST_GEMM("df_pv_f", float, float, float, float, false)
DF_INST_GEMM("df_pv_h", float, half, half, half, false)

template [[host_name("df_conv_f")]] kernel decltype(df_conv<float, float>) df_conv<float, float>;
template [[host_name("df_conv_h")]] kernel decltype(df_conv<half, half>) df_conv<half, half>;

// ---------------------------------------------------------------------------------------------- 3. direct convolution
// df_conv_direct<T>: the same convolution, one thread per output element, for the first and last layers whose input or
// output channel count is not a multiple of 8 (4 latent channels, 3 colour channels, 9 for an inpainting UNet).

template <typename T>
kernel void df_conv_direct(device const T *x [[buffer(0)]],
                           device const T *w [[buffer(1)]],
                           device T *out [[buffer(2)]],
                           device const float *bias [[buffer(3)]],
                           device const T *chan [[buffer(4)]],
                           device const T *res [[buffer(5)]],
                           constant DfConvParams &p [[buffer(6)]],
                           uint gid [[thread_position_in_grid]]) {
    const long total = (long)p.batch * p.ho * p.wo * p.cout;
    if ((long)gid >= total) return;
    const int co = int(gid % p.cout);
    const long pix = gid / p.cout;
    const int bi = int(pix / (p.ho * p.wo));
    const int rem = int(pix - (long)bi * (p.ho * p.wo));
    const int oy = rem / p.wo;
    const int ox = rem - oy * p.wo;
    float acc = 0.0f;
    device const T *wr = w + (long)co * p.kh * p.kw * p.cin;
    for (int ky = 0; ky < p.kh; ky++) {
        const int iy = oy * p.stride - p.padT + ky;
        if (iy < 0 || iy >= p.h) continue;
        for (int kx = 0; kx < p.kw; kx++) {
            const int ix = ox * p.stride - p.padL + kx;
            if (ix < 0 || ix >= p.w) continue;
            device const T *xp = x + ((long)(bi * (p.h >> p.ups) + (iy >> p.ups)) * (p.w >> p.ups) + (ix >> p.ups)) * p.cin;
            device const T *wp = wr + (ky * p.kw + kx) * p.cin;
            for (int ci = 0; ci < p.cin; ci++) acc += float(xp[ci]) * float(wp[ci]);
        }
    }
    if (p.hasBias != 0) acc += bias[co];
    if (p.hasChan != 0) acc += float(chan[(long)bi * p.cout + co]);
    if (p.hasRes != 0) acc += float(res[pix * p.cout + co]);
    out[pix * p.cout + co] = T(acc);
}

template [[host_name("df_conv_direct_f")]] kernel decltype(df_conv_direct<float>) df_conv_direct<float>;
template [[host_name("df_conv_direct_h")]] kernel decltype(df_conv_direct<half>) df_conv_direct<half>;

// ---------------------------------------------------------------------------------------------- 4. normalisation
// df_groupnorm<T>: one threadgroup per (group, batch element): mean and variance over the group's channels at every pixel
// (two passes over the data for the variance), then (x - mean) * rstd * gamma + beta, optionally followed by SiLU.
// df_layernorm<T>: one threadgroup per row.

struct DfNormParams {
    int c;          // channels (the last dimension)
    int hw;         // rows per batch element (pixels or tokens)
    int groups;     // group norm only
    float eps;
    int silu;
    int hasAffine;  // layer norm only: gamma / beta given
};

template <typename T>
kernel void df_groupnorm(device const T *x [[buffer(0)]],
                         device const float *gamma [[buffer(1)]],
                         device const float *beta [[buffer(2)]],
                         device T *out [[buffer(3)]],
                         constant DfNormParams &p [[buffer(4)]],
                         uint2 tgp [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    const int g = int(tgp.x);
    const int bi = int(tgp.y);
    const int gs = p.c / p.groups;
    const long base = (long)bi * p.hw * p.c + (long)g * gs;
    const int n = p.hw * gs;
    float s = 0.0f;
    for (int i = tid; i < n; i += DF_ROW_THREADS) {
        const int px = i / gs;
        s += float(x[base + (long)px * p.c + (i - px * gs)]);
    }
    const float mean = df_tg_sum(s, red, sg, lane, DF_ROW_THREADS / 32) / float(n);
    float q = 0.0f;
    for (int i = tid; i < n; i += DF_ROW_THREADS) {
        const int px = i / gs;
        const float d = float(x[base + (long)px * p.c + (i - px * gs)]) - mean;
        q += d * d;
    }
    const float rstd = rsqrt(df_tg_sum(q, red, sg, lane, DF_ROW_THREADS / 32) / float(n) + p.eps);
    for (int i = tid; i < n; i += DF_ROW_THREADS) {
        const int px = i / gs;
        const int j = i - px * gs;
        const long o = base + (long)px * p.c + j;
        float v = (float(x[o]) - mean) * rstd * gamma[g * gs + j] + beta[g * gs + j];
        if (p.silu != 0) v = df_silu(v);
        out[o] = T(v);
    }
}

template <typename T>
kernel void df_layernorm(device const T *x [[buffer(0)]],
                         device const float *gamma [[buffer(1)]],
                         device const float *beta [[buffer(2)]],
                         device T *out [[buffer(3)]],
                         constant DfNormParams &p [[buffer(4)]],
                         uint tg [[threadgroup_position_in_grid]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    // one simdgroup per row (hw = number of rows), 8 rows per threadgroup: no barriers, reductions with simd_sum
    const long row = (long)tg * (DF_ROW_THREADS / 32) + sg;
    if (row >= p.hw) return;
    device const T *xr = x + row * p.c;
    float s = 0.0f;
    for (int i = lane; i < p.c; i += 32) s += float(xr[i]);
    const float mean = simd_sum(s) / float(p.c);
    float q = 0.0f;
    for (int i = lane; i < p.c; i += 32) {
        const float d = float(xr[i]) - mean;
        q += d * d;
    }
    const float rstd = rsqrt(simd_sum(q) / float(p.c) + p.eps);
    for (int i = lane; i < p.c; i += 32) {
        float v = (float(xr[i]) - mean) * rstd;
        if (p.hasAffine != 0) v = v * gamma[i] + beta[i];
        out[row * p.c + i] = T(v);
    }
}

// Group norm of a large tensor in three steps, so that the whole GPU works on it instead of one threadgroup per group:
//   df_gn_stats    (chunks, groups, batch): the mean and the sum of squared deviations of one chunk of pixels of one group
//   df_gn_combine  (groups, batch): the chunks' statistics merged (Chan et al.: M2 = sum(M2_c + n_c (mean_c - mean)^2)) into
//                  (mean, 1 / sqrt(var + eps))
//   df_gn_apply    one thread per 4 channels: (x - mean) * rstd * gamma + beta, optionally followed by SiLU

struct DfGnParams {
    int c;          // channels
    int hw;         // pixels per batch element
    int groups;
    float eps;
    int silu;
    int chunkPix;   // pixels per chunk
    int chunks;
    int batch;
};

template <typename T>
kernel void df_gn_stats(device const T *x [[buffer(0)]],
                        device float2 *part [[buffer(1)]],
                        constant DfGnParams &p [[buffer(2)]],
                        uint3 tgp [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    const int ch = int(tgp.x);
    const int g = int(tgp.y);
    const int bi = int(tgp.z);
    const int gs = p.c / p.groups;
    const int p0 = ch * p.chunkPix;
    const int p1 = min(p.hw, p0 + p.chunkPix);
    const int n = (p1 - p0) * gs;
    const long base = ((long)bi * p.hw + p0) * p.c + (long)g * gs;
    float s = 0.0f;
    for (int i = tid; i < n; i += DF_ROW_THREADS) {
        const int px = i / gs;
        s += float(x[base + (long)px * p.c + (i - px * gs)]);
    }
    const float mean = df_tg_sum(s, red, sg, lane, DF_ROW_THREADS / 32) / float(n);
    float q = 0.0f;
    for (int i = tid; i < n; i += DF_ROW_THREADS) {
        const int px = i / gs;
        const float d = float(x[base + (long)px * p.c + (i - px * gs)]) - mean;
        q += d * d;
    }
    q = df_tg_sum(q, red, sg, lane, DF_ROW_THREADS / 32);
    if (tid == 0) part[((long)bi * p.groups + g) * p.chunks + ch] = float2(mean, q);
}

kernel void df_gn_combine(device const float2 *part [[buffer(0)]],
                          device float2 *stats [[buffer(1)]],
                          constant DfGnParams &p [[buffer(2)]],
                          uint2 tgp [[threadgroup_position_in_grid]]) {
    const int g = int(tgp.x);
    const int bi = int(tgp.y);
    const int gs = p.c / p.groups;
    device const float2 *pc = part + ((long)bi * p.groups + g) * p.chunks;
    float total = 0.0f;
    float mean = 0.0f;
    for (int ch = 0; ch < p.chunks; ch++) {
        const int p0 = ch * p.chunkPix;
        const float n = float((min(p.hw, p0 + p.chunkPix) - p0) * gs);
        mean += n * pc[ch].x;
        total += n;
    }
    mean /= total;
    float m2 = 0.0f;
    for (int ch = 0; ch < p.chunks; ch++) {
        const int p0 = ch * p.chunkPix;
        const float n = float((min(p.hw, p0 + p.chunkPix) - p0) * gs);
        const float d = pc[ch].x - mean;
        m2 += pc[ch].y + n * d * d;
    }
    stats[(long)bi * p.groups + g] = float2(mean, rsqrt(m2 / total + p.eps));
}

template <typename T>
kernel void df_gn_apply(device const T *x [[buffer(0)]],
                        device const float2 *stats [[buffer(1)]],
                        device const float *gamma [[buffer(2)]],
                        device const float *beta [[buffer(3)]],
                        device T *out [[buffer(4)]],
                        constant DfGnParams &p [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
    // c is a multiple of 4: 4 consecutive elements of a thread are the channels c0..c0+3 of one pixel
    const long e = (long)gid * 4;
    if (e >= (long)p.batch * p.hw * p.c) return;
    const long row = e / p.c;                 // pixel index over the whole batch
    const int c0 = int(e - row * p.c);
    const long bi = row / p.hw;
    const int gs = p.c / p.groups;
    for (int k = 0; k < 4; k++) {
        const int c = c0 + k;
        const float2 st = stats[bi * p.groups + c / gs];
        float v = (float(x[e + k]) - st.x) * st.y * gamma[c] + beta[c];
        if (p.silu != 0) v = df_silu(v);
        out[e + k] = T(v);
    }
}

template [[host_name("df_groupnorm_f")]] kernel decltype(df_groupnorm<float>) df_groupnorm<float>;
template [[host_name("df_groupnorm_h")]] kernel decltype(df_groupnorm<half>) df_groupnorm<half>;
template [[host_name("df_gn_stats_f")]] kernel decltype(df_gn_stats<float>) df_gn_stats<float>;
template [[host_name("df_gn_stats_h")]] kernel decltype(df_gn_stats<half>) df_gn_stats<half>;
template [[host_name("df_gn_apply_f")]] kernel decltype(df_gn_apply<float>) df_gn_apply<float>;
template [[host_name("df_gn_apply_h")]] kernel decltype(df_gn_apply<half>) df_gn_apply<half>;
template [[host_name("df_layernorm_f")]] kernel decltype(df_layernorm<float>) df_layernorm<float>;
template [[host_name("df_layernorm_h")]] kernel decltype(df_layernorm<half>) df_layernorm<half>;

// ---------------------------------------------------------------------------------------------- 5. attention
// df_flash<T, D>: softmax(q k^T * scale) v for heads of width D without a scores buffer (flash attention). A threadgroup of
// 4 simdgroups takes DF_FA_ROWS = 32 query rows of one head (8 rows per simdgroup) and walks the keys in blocks of
// DF_FA_BK = 32, each block's K (then V) rows staged once in threadgroup memory and shared by the 32 rows:
//   S = Q K^T (8 x 32 per simdgroup, simdgroup 8 x 8 multiply-accumulates over the head in 8-wide slices, float)
//   m' = max(m, rowmax S), P = exp(S - m'), l = l exp(m - m') + rowsum P          (online softmax, per row)
//   O = diag(exp(m - m')) O + P V            and finally O = diag(1 / l) O
// The row rescales are products with a diagonal matrix, so the kernel never addresses the elements of a simdgroup matrix
// (their distribution over the lanes is not specified); S and P go through an 8 x 32 scratch per simdgroup. Keys past lk
// (and, with causal, past the row) are masked. q, k, v, out are [batch][length][heads * D] rows with the given strides.

struct DfFlashParams {
    int heads;
    int lq;
    int lk;
    int qs;         // row strides of q, k, v, out (elements)
    int ks;
    int vs;
    int os;
    float scale;
    int causal;
};

template <typename TT, int D>
kernel void df_flash(device const TT *q [[buffer(0)]],
                     device const TT *k [[buffer(1)]],
                     device const TT *v [[buffer(2)]],
                     device TT *out [[buffer(3)]],
                     constant DfFlashParams &p [[buffer(4)]],
                     uint3 tgp [[threadgroup_position_in_grid]],
                     ushort tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int NI = D / 8;                  // 8-wide slices of the head
    constexpr int NJ = DF_FA_BK / 8;           // 8-key slices of a block
    threadgroup TT kvs[DF_FA_BK * D];
    threadgroup float sb[4][8 * DF_FA_BK];
    threadgroup float st[4][64];
    threadgroup TT sq[4][64];

    const int h = int(tgp.x);
    const int bi = int(tgp.z);
    const int t0 = int(tgp.y) * DF_FA_ROWS;
    const int r0 = t0 + int(sg) * 8;
    device const TT *qb = q + (long)bi * p.lq * p.qs + h * D;
    device const TT *kb = k + (long)bi * p.lk * p.ks + h * D;
    device const TT *vb = v + (long)bi * p.lk * p.vs + h * D;
    const int tEnd = min(p.lq, t0 + DF_FA_ROWS);
    const int lastKey = p.causal != 0 ? min(p.lk, tEnd) : p.lk;

    simdgroup_matrix<TT, 8, 8> qm[NI];
    for (int i = 0; i < NI; i++) {
        for (int e = lane; e < 64; e += 32) {
            const int t = r0 + e / 8;
            sq[sg][e] = t < p.lq ? qb[(long)t * p.qs + i * 8 + e % 8] : TT(0);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(qm[i], &sq[sg][0], 8);
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

    for (int k0 = 0; k0 < lastKey; k0 += DF_FA_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = tid; e < DF_FA_BK * D / 4; e += DF_THREADS) {
            const int key = k0 + (e * 4) / D;
            const int d = (e * 4) % D;
            vec<TT, 4> val = vec<TT, 4>{TT(0), TT(0), TT(0), TT(0)};
            if (key < p.lk) val = *(device const vec<TT, 4> *)(kb + (long)key * p.ks + d);
            *(threadgroup vec<TT, 4> *)(kvs + e * 4) = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NJ; j++) {
            simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int i = 0; i < NI; i++) {
                simdgroup_matrix<TT, 8, 8> kt;
                simdgroup_load(kt, &kvs[j * 8 * D + i * 8], D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qm[i], kt, s);
            }
            simdgroup_store(s, &sb[sg][j * 8], DF_FA_BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int r = 0; r < 8; r++) {
            const int t = r0 + r;
            const int limit = p.causal != 0 ? min(p.lk, t + 1) : p.lk;
            float mx = -INFINITY;
            {
                const int key = k0 + lane;
                float sv = sb[sg][r * DF_FA_BK + lane] * p.scale;
                if (t >= p.lq || key >= limit) sv = -INFINITY;
                sb[sg][r * DF_FA_BK + lane] = sv;
                mx = sv;
            }
            mx = simd_max(mx);
            const float mNew = max(m[r], mx);
            const float e = mNew == -INFINITY ? 0.0f : exp(sb[sg][r * DF_FA_BK + lane] - mNew);
            sb[sg][r * DF_FA_BK + lane] = e;
            const float sum = simd_sum(e);
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
        for (int e = tid; e < DF_FA_BK * D / 4; e += DF_THREADS) {
            const int key = k0 + (e * 4) / D;
            const int d = (e * 4) % D;
            vec<TT, 4> val = vec<TT, 4>{TT(0), TT(0), TT(0), TT(0)};
            if (key < p.lk) val = *(device const vec<TT, 4> *)(vb + (long)key * p.vs + d);
            *(threadgroup vec<TT, 4> *)(kvs + e * 4) = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NJ; j++) {
            simdgroup_float8x8 pm;
            simdgroup_load(pm, &sb[sg][j * 8], DF_FA_BK);
            for (int i = 0; i < NI; i++) {
                simdgroup_matrix<TT, 8, 8> vm;
                simdgroup_load(vm, &kvs[j * 8 * D + i * 8], D);
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
            if (t < p.lq) out[(long)bi * p.lq * p.os + (long)t * p.os + h * D + i * 8 + e % 8] = TT(st[sg][e]);
        }
    }
}

#define DF_INST_FLASH(D) \
    template [[host_name("df_flash_d" #D "_f")]] kernel decltype(df_flash<float, D>) df_flash<float, D>; \
    template [[host_name("df_flash_d" #D "_h")]] kernel decltype(df_flash<half, D>) df_flash<half, D>;
DF_INST_FLASH(8)
DF_INST_FLASH(16)
DF_INST_FLASH(24)
DF_INST_FLASH(32)
DF_INST_FLASH(40)
DF_INST_FLASH(48)
DF_INST_FLASH(64)
DF_INST_FLASH(80)
DF_INST_FLASH(96)
DF_INST_FLASH(128)
DF_INST_FLASH(160)

// df_softmax: in place on rows of float scores [rows][ld]: the first n entries of a row are softmax-ed (causal: only the
// first row%lq + 1), the rest of the padded row is zeroed, so the P V product can read the padding.

struct DfSoftmaxParams {
    int n;          // valid scores per row
    int ld;         // row stride
    int lq;         // rows per (batch, head): the causal limit of row r is r % lq + 1
    int causal;
};

kernel void df_softmax(device float *s [[buffer(0)]],
                       constant DfSoftmaxParams &p [[buffer(1)]],
                       uint row [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    device float *r = s + (long)row * p.ld;
    const int n = p.causal != 0 ? min(p.n, int(row % uint(p.lq)) + 1) : p.n;
    float mx = -INFINITY;
    for (int i = tid; i < n; i += DF_ROW_THREADS) mx = max(mx, r[i]);
    mx = df_tg_max(mx, red, sg, lane, DF_ROW_THREADS / 32);
    float sum = 0.0f;
    for (int i = tid; i < n; i += DF_ROW_THREADS) sum += exp(r[i] - mx);
    const float inv = 1.0f / df_tg_sum(sum, red, sg, lane, DF_ROW_THREADS / 32);
    for (int i = tid; i < p.ld; i += DF_ROW_THREADS) r[i] = i < n ? exp(r[i] - mx) * inv : 0.0f;
}

// ---------------------------------------------------------------------------------------------- 6. elementwise and layout

struct DfEwParams {
    long n;         // output elements
    int a;          // kernel specific (channels, ...)
    int b;
    int c;
    int d;
};

template <typename T>
kernel void df_add(device const T *x [[buffer(0)]], device const T *y [[buffer(1)]], device T *out [[buffer(2)]],
                   constant DfEwParams &p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if ((long)gid < p.n) out[gid] = T(float(x[gid]) + float(y[gid]));
}

template <typename T>
kernel void df_silu_k(device const T *x [[buffer(0)]], device T *out [[buffer(1)]],
                      constant DfEwParams &p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if ((long)gid < p.n) out[gid] = T(df_silu(float(x[gid])));
}

// x[row][0..n) * gelu(x[row][n..2n)): p.a = n
template <typename T>
kernel void df_geglu(device const T *x [[buffer(0)]], device T *out [[buffer(1)]],
                     constant DfEwParams &p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if ((long)gid >= p.n) return;
    const long row = gid / p.a;
    const int j = int(gid - row * p.a);
    out[gid] = T(float(x[row * 2 * p.a + j]) * df_gelu(float(x[row * 2 * p.a + p.a + j])));
}

// nearest-neighbour 2x: out [B, 2h, 2w, C]; p.a = h, p.b = w, p.c = C
template <typename T>
kernel void df_upsample2x(device const T *x [[buffer(0)]], device T *out [[buffer(1)]],
                          constant DfEwParams &p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if ((long)gid >= p.n) return;
    const int c = int(gid % p.c);
    const long px = gid / p.c;
    const int ox = int(px % (2 * p.b));
    const long r = px / (2 * p.b);
    const int oy = int(r % (2 * p.a));
    const long bi = r / (2 * p.a);
    out[gid] = x[((bi * p.a + oy / 2) * p.b + ox / 2) * p.c + c];
}

// channel concatenation: p.a = ca, p.b = cb
template <typename T>
kernel void df_concat(device const T *x [[buffer(0)]], device const T *y [[buffer(1)]], device T *out [[buffer(2)]],
                      constant DfEwParams &p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if ((long)gid >= p.n) return;
    const int ct = p.a + p.b;
    const long row = gid / ct;
    const int c = int(gid - row * ct);
    out[gid] = c < p.a ? x[row * p.a + c] : y[row * p.b + (c - p.a)];
}

#define DF_INST_EW(SUF, T) \
    template [[host_name("df_add" SUF)]] kernel decltype(df_add<T>) df_add<T>; \
    template [[host_name("df_silu" SUF)]] kernel decltype(df_silu_k<T>) df_silu_k<T>; \
    template [[host_name("df_geglu" SUF)]] kernel decltype(df_geglu<T>) df_geglu<T>; \
    template [[host_name("df_upsample2x" SUF)]] kernel decltype(df_upsample2x<T>) df_upsample2x<T>; \
    template [[host_name("df_concat" SUF)]] kernel decltype(df_concat<T>) df_concat<T>;
DF_INST_EW("_f", float)
DF_INST_EW("_h", half)
