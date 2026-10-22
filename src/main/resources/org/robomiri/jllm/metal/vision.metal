// Compute kernels of the vision encoder on Metal (vision/ClipMetal.java): a vision transformer over an image's patches,
// as float activations and half weights. Compiled by libjllm_metal in a context of its own (the text model's
// kernels.metal is separate), and by scripts/build-metal-emu.sh into the CPU emulator next to it. Every kernel has a
// numpy reference in scripts/metal-emu/kernel_tests.py. Parameter structs hold only 32-bit ints and floats.
//
//   vit_gemm        out = act(x . W^T + bias) (+ out): half weights, simdgroup-matrix tiles, float accumulation
//   vit_norm        LayerNorm / RMSNorm of rows
//   vit_attention   bidirectional attention inside segments (windows), online softmax, one simdgroup per query row
//   vit_rope        2-D rotary embedding of q / k from a cos/sin table, in place
//   vit_gated       a = act(a) * b   (SwiGLU)
//   vit_act, vit_add, vit_gather, vit_gather_mean   elementwise helpers and row gathers (pixel shuffle, windows, pooling)

#include <metal_stdlib>
using namespace metal;

#define VG_BM 64                // tokens per vit_gemm threadgroup
#define VG_BN 64                // output columns per threadgroup
#define VG_BK 32                // k per step
#define VG_THREADS 128
#define VG_LDS 40               // row stride of the half tiles in threadgroup memory (32 + 8: avoids bank conflicts)
#define VA_QPS 4                // query rows per simdgroup of vit_attention
#define VA_SGS 4                // simdgroups per vit_attention threadgroup (VA_QPS * VA_SGS rows per threadgroup)
#define VA_MAX_PER 4            // dims per lane: head dims up to 128
#define VN_THREADS 256          // vit_norm threadgroup (one row)
// (vit_attention_mma: 4 simdgroups, each 8 * RS query rows: 64 rows per threadgroup, 32 for head size 128)
#define VA_MMA_THREADS 128

constant float VIT_TANH_CLAMP = 20.0f;

static inline float vit_gelu_tanh(float x) {
    const float u = clamp(0.7978845608028654f * x * (1.0f + 0.044715f * x * x), -VIT_TANH_CLAMP, VIT_TANH_CLAMP);
    const float e = exp(2.0f * u);
    return 0.5f * x * (1.0f + (e - 1.0f) / (e + 1.0f));
}

static inline float vit_silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float vit_gelu_quick(float x) {
    return x / (1.0f + exp(-1.702f * x));
}

// act: 0 none, 1 gelu (tanh form), 2 silu, 3 quick gelu
static inline float vit_apply(float x, int act) {
    if (act == 1) return vit_gelu_tanh(x);
    if (act == 2) return vit_silu(x);
    if (act == 3) return vit_gelu_quick(x);
    return x;
}

// ------------------------------------------------------------------------------------------------ matrix product

struct VitGemmParams {
    int m;          // rows of x (tokens)
    int n;          // rows of W = columns of the output
    int k;          // inner dimension (a multiple of 4)
    int ldx;        // row stride of x, in floats
    int ldo;        // row stride of out, in floats
    int act;        // activation applied to x . W^T + bias
    int accumulate; // 1: out += result (the residual add)
    int hasBias;
};

// out[t][c] = act(sum_k x[t][k] W[c][k] + bias[c]) (+ out[t][c]). A 64 x 64 tile per threadgroup of four simdgroups,
// each owning 32 x 32 of it; x tiles are converted to half, W is half, products are accumulated in float.
kernel void vit_gemm(device const float *x [[buffer(0)]],
                     device const half *w [[buffer(1)]],
                     device const float *bias [[buffer(2)]],
                     device float *out [[buffer(3)]],
                     constant VitGemmParams &p [[buffer(4)]],
                     uint2 tgp [[threadgroup_position_in_grid]],
                     ushort tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[VG_BM * VG_BN];   // the half tiles during the k loop, the float result tile after it
    threadgroup half *sa = (threadgroup half *)smem;            // [64 tokens][32 k], rows VG_LDS apart
    threadgroup half *sb = sa + VG_BM * VG_LDS;                 // [64 columns][32 k], rows VG_LDS apart
    const int n0 = int(tgp.x) * VG_BN;
    const int m0 = int(tgp.y) * VG_BM;
    const ushort sr = sg & 1;    // columns 32 sr .. of the tile
    const ushort st = sg >> 1;   // tokens 32 st .. of the tile
    simdgroup_float8x8 acc[16];
    for (int i = 0; i < 16; i++) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const int lr = tid >> 1;     // the tile row this thread loads (tokens for sa, columns for sb)
    const int lh = tid & 1;      // which 16 of the 32 k
    const int tokRow = m0 + lr;
    const int colRow = n0 + lr;
    for (int k0 = 0; k0 < p.k; k0 += VG_BK) {
        for (int i = 0; i < 4; i++) {
            const int kk = k0 + lh * 16 + i * 4;
            float4 xv = float4(0.0f);
            half4 wv = half4(0.0f);
            if (kk < p.k) {
                if (tokRow < p.m) xv = *(device const float4 *)(x + (ulong)tokRow * p.ldx + kk);
                if (colRow < p.n) wv = *(device const half4 *)(w + (ulong)colRow * p.k + kk);
            }
            *(threadgroup half4 *)(sa + lr * VG_LDS + lh * 16 + i * 4) = half4(xv);
            *(threadgroup half4 *)(sb + lr * VG_LDS + lh * 16 + i * 4) = wv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; kk++) {
            simdgroup_half8x8 a[4];
            simdgroup_half8x8 b[4];
            for (int i = 0; i < 4; i++) simdgroup_load(a[i], sa + (st * 32 + i * 8) * VG_LDS + kk * 8, VG_LDS);
            for (int j = 0; j < 4; j++) simdgroup_load(b[j], sb + (sr * 32 + j * 8) * VG_LDS + kk * 8, VG_LDS, ulong2(0, 0), true);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i * 4 + j], a[i], b[j], acc[i * 4 + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            simdgroup_store(acc[i * 4 + j], smem + (st * 32 + i * 8) * VG_BN + sr * 32 + j * 8, VG_BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < VG_BM * VG_BN; i += VG_THREADS) {
        const int t = i / VG_BN;
        const int c = i % VG_BN;
        if (m0 + t < p.m && n0 + c < p.n) {
            float v = smem[i];
            if (p.hasBias) v += bias[n0 + c];
            v = vit_apply(v, p.act);
            const ulong o = (ulong)(m0 + t) * p.ldo + n0 + c;
            out[o] = p.accumulate ? out[o] + v : v;
        }
    }
}

// ------------------------------------------------------------------------------------------------ norms

struct VitNormParams {
    int cols;
    int rms;        // 1: x / sqrt(mean(x^2) + eps) * w; 0: LayerNorm (x - mean) / sqrt(var + eps) * w + b
    int hasW;
    int hasB;
    float eps;
    int ldx;        // row strides
    int ldo;
};

kernel void vit_norm(device const float *x [[buffer(0)]],
                     device float *out [[buffer(1)]],
                     device const float *w [[buffer(2)]],
                     device const float *b [[buffer(3)]],
                     constant VitNormParams &p [[buffer(4)]],
                     uint row [[threadgroup_position_in_grid]],
                     ushort tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    device const float *xr = x + (ulong)row * p.ldx;
    float s = 0.0f;
    for (int i = tid; i < p.cols; i += VN_THREADS) s += xr[i];
    float mean = 0.0f;
    if (p.rms == 0) {
        s = simd_sum(s);
        if (lane == 0) red[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float t = lane < VN_THREADS / 32 ? red[lane] : 0.0f;
        mean = simd_sum(t) / float(p.cols);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float q = 0.0f;
    for (int i = tid; i < p.cols; i += VN_THREADS) {
        const float d = xr[i] - mean;
        q += d * d;
    }
    q = simd_sum(q);
    if (lane == 0) red[sg] = q;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t2 = lane < VN_THREADS / 32 ? red[lane] : 0.0f;
    const float scale = 1.0f / sqrt(simd_sum(t2) / float(p.cols) + p.eps);
    device float *orow = out + (ulong)row * p.ldo;
    for (int i = tid; i < p.cols; i += VN_THREADS) {
        float v = (xr[i] - mean) * scale;
        if (p.hasW) v *= w[i];
        if (p.hasB) v += b[i];
        orow[i] = v;
    }
}

// ------------------------------------------------------------------------------------------------ attention

struct VitAttnParams {
    int t;          // tokens
    int heads;
    int dh;         // head size (<= 128)
    float scale;
};

// Bidirectional attention. Query row i attends to the keys [lo[i], hi[i]) (its window; the whole sequence for full
// attention). One simdgroup per (VA_QPS consecutive query rows, head): lane l owns dims l, l + 32, ... of the head, the
// score of a key is a simd_sum over lanes, the softmax is online (running max and sum), so no score buffer is needed.
// The key and value rows are read once for all VA_QPS query rows (the rows of a window share them), which is what
// makes it fast: with one row per simdgroup the kernel is bound by reading K and V from memory.
kernel void vit_attention(device const float *q [[buffer(0)]],
                          device const float *k [[buffer(1)]],
                          device const float *v [[buffer(2)]],
                          device float *out [[buffer(3)]],
                          device const int *lo [[buffer(4)]],
                          device const int *hi [[buffer(5)]],
                          constant VitAttnParams &p [[buffer(6)]],
                          uint2 tgp [[threadgroup_position_in_grid]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    const int row0 = (int(tgp.x) * VA_SGS + sg) * VA_QPS;
    if (row0 >= p.t) return;
    const int h = int(tgp.y);
    const int dim = p.heads * p.dh;
    const ulong base = (ulong)h * p.dh;
    float qv[VA_QPS][VA_MAX_PER];
    float acc[VA_QPS][VA_MAX_PER];
    float m[VA_QPS];
    float l[VA_QPS];
    int rlo[VA_QPS];
    int rhi[VA_QPS];
    int jlo = p.t;
    int jhi = 0;
    for (int r = 0; r < VA_QPS; r++) {
        const int row = row0 + r;
        const bool ok = row < p.t;
        rlo[r] = ok ? lo[row] : 0;
        rhi[r] = ok ? hi[row] : 0;
        if (ok) {
            jlo = min(jlo, rlo[r]);
            jhi = max(jhi, rhi[r]);
        }
        m[r] = -INFINITY;
        l[r] = 0.0f;
        for (int i = 0; i < VA_MAX_PER; i++) {
            const int d = lane + 32 * i;
            qv[r][i] = ok && d < p.dh ? q[(ulong)row * dim + base + d] * p.scale : 0.0f;
            acc[r][i] = 0.0f;
        }
    }
    for (int j = jlo; j < jhi; j++) {
        float kd[VA_MAX_PER];
        float vd[VA_MAX_PER];
        device const float *kr = k + (ulong)j * dim + base;
        device const float *vr = v + (ulong)j * dim + base;
        for (int i = 0; i < VA_MAX_PER; i++) {
            const int d = lane + 32 * i;
            kd[i] = d < p.dh ? kr[d] : 0.0f;
            vd[i] = d < p.dh ? vr[d] : 0.0f;
        }
        for (int r = 0; r < VA_QPS; r++) {
            if (j >= rlo[r] && j < rhi[r]) {
                float s = 0.0f;
                for (int i = 0; i < VA_MAX_PER; i++) s += qv[r][i] * kd[i];
                s = simd_sum(s);
                const float mn = max(m[r], s);
                const float corr = exp(m[r] - mn);
                const float pj = exp(s - mn);
                l[r] = l[r] * corr + pj;
                for (int i = 0; i < VA_MAX_PER; i++) acc[r][i] = acc[r][i] * corr + pj * vd[i];
                m[r] = mn;
            }
        }
    }
    for (int r = 0; r < VA_QPS; r++) {
        const int row = row0 + r;
        if (row < p.t) {
            const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
            for (int i = 0; i < VA_MAX_PER; i++) {
                const int d = lane + 32 * i;
                if (d < p.dh) out[(ulong)row * dim + base + d] = acc[r][i] * inv;
            }
        }
    }
}

// The same attention with simdgroup matrices (the structure of attention_fa in kernels.metal, section 9), in half
// precision like the projection kernel: a threadgroup of four simdgroups takes 4 * 8 * RS query rows of one head (RS
// sets of 8 rows per simdgroup: 64 rows for RS = 2, 32 for the largest head size) and walks the keys any of them sees in
// blocks of BK = 32 keys. Per block K and V are converted to half once into threadgroup memory (one barrier pair for
// both), then each simdgroup
//   S = (Q * scale) K^T          half x half products accumulated in float, the K^T slice shared by the RS row sets,
//   online softmax per row       LPR = 32 / (8 RS) lanes share a row: max / sum by shuffles, P = exp(S - m) written as half,
//   O = diag(exp(m - m')) O + P V   (the diagonal rescale is skipped while no row's maximum moved),
// with the keys outside [lo[row], hi[row]) masked; finally O = diag(1 / l) O. Row rescales are products with a diagonal
// matrix, so the kernel never addresses a simdgroup matrix's elements. Threadgroup tiles have a row stride of 8 mod 16
// halves, so the eight rows of an 8 x 8 load fall in different banks.
template <int D, int RS>
kernel void vit_attention_mma(device const float *q [[buffer(0)]],
                              device const float *k [[buffer(1)]],
                              device const float *v [[buffer(2)]],
                              device float *out [[buffer(3)]],
                              device const int *lo [[buffer(4)]],
                              device const int *hi [[buffer(5)]],
                              constant VitAttnParams &p [[buffer(6)]],
                              uint2 tgp [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int BK = 32;                   // keys per block
    constexpr int NI = D / 8;                // 8-wide slices of the head
    constexpr int NJ = BK / 8;               // 8-key slices of a block
    constexpr int ROWS = 8 * RS;             // query rows per simdgroup
    constexpr int LPR = 32 / ROWS;           // lanes per row in the softmax
    constexpr int CPL = BK / LPR;            // scores per lane
    constexpr int LDH = D + 8;               // half row stride of the K and V tiles
    constexpr int LDS = BK + 4;              // float row stride of the scores
    constexpr int LDP = BK + 8;              // half row stride of the probabilities
    threadgroup float smem[BK * LDH + 4 * ROWS * LDS + 2 * ROWS * LDP + 4 * RS * 64];
    threadgroup half *kh = (threadgroup half *)smem;                       // K tile [BK][LDH]
    threadgroup half *vh = kh + BK * LDH;                                  // V tile [BK][LDH]
    threadgroup float *sb = smem + BK * LDH + sg * ROWS * LDS;             // scores [ROWS][LDS]
    threadgroup half *pb = (threadgroup half *)(smem + BK * LDH + 4 * ROWS * LDS) + sg * ROWS * LDP;  // probabilities [ROWS][LDP]
    threadgroup float *dg = smem + BK * LDH + 4 * ROWS * LDS + 2 * ROWS * LDP + sg * RS * 64;  // diagonal matrices
    threadgroup int range[8];

    const int h = int(tgp.x);
    const int dim = p.heads * D;
    const int t0 = int(tgp.y) * (4 * ROWS);
    const int r0 = t0 + int(sg) * ROWS;
    // the keys any row of the threadgroup sees
    {
        int fk = p.t;
        int lk = 0;
        if (tid < 4 * ROWS && t0 + tid < p.t) {
            fk = lo[t0 + tid];
            lk = hi[t0 + tid];
        }
        fk = simd_min(fk);
        lk = simd_max(lk);
        if (lane == 0) {
            range[sg] = fk;
            range[4 + sg] = lk;
        }
    }
    for (int i = tid; i < 4 * RS * 64; i += VA_MMA_THREADS) {
        smem[BK * LDH + 4 * ROWS * LDS + 2 * ROWS * LDP + i] = 0.0f;   // the diagonal matrices: only their diagonals are written
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int firstKey = min(min(range[0], range[1]), min(range[2], range[3]));
    const int lastKey = max(max(range[4], range[5]), max(range[6], range[7]));
    device const float *kb = k + h * D;
    device const float *vb = v + h * D;

    // this simdgroup's queries, scaled, as half matrices
    simdgroup_half8x8 qm[RS * NI];
    for (int s = 0; s < RS; s++) {
        for (int i = 0; i < NI; i++) {
            for (int e = lane; e < 64; e += 32) {
                const int t = r0 + s * 8 + e / 8;
                pb[(s * 8 + e / 8) * LDP + e % 8] = t < p.t ? half(q[(ulong)t * dim + h * D + i * 8 + e % 8] * p.scale) : half(0.0f);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(qm[s * NI + i], pb + s * 8 * LDP, LDP);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    simdgroup_float8x8 om[RS * NI];
    for (int i = 0; i < RS * NI; i++) om[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    // the row this lane's softmax works on
    const int row = lane / LPR;
    const int c0 = (lane % LPR) * CPL;
    const int myT = r0 + row;
    const int rlo = myT < p.t ? lo[myT] : 0;
    const int rhi = myT < p.t ? hi[myT] : 0;
    float mrow = -INFINITY;
    float lrow = 0.0f;

    for (int k0 = firstKey / BK * BK; k0 < lastKey; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = tid; e < BK * D / 4; e += VA_MMA_THREADS) {
            const int kl = (e * 4) / D;
            const int col = (e * 4) % D;
            const int key = k0 + kl;
            float4 kx = float4(0.0f);
            float4 vx = float4(0.0f);
            if (key < p.t) {
                kx = *(device const float4 *)(kb + (ulong)key * dim + col);
                vx = *(device const float4 *)(vb + (ulong)key * dim + col);
            }
            *(threadgroup half4 *)(kh + kl * LDH + col) = half4(kx);
            *(threadgroup half4 *)(vh + kl * LDH + col) = half4(vx);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NJ; j++) {
            simdgroup_float8x8 sm0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 sm1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int i = 0; i < NI; i++) {
                simdgroup_half8x8 kt;
                simdgroup_load(kt, kh + j * 8 * LDH + i * 8, LDH, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sm0, qm[i], kt, sm0);
                if (RS > 1) simdgroup_multiply_accumulate(sm1, qm[NI + i], kt, sm1);
            }
            simdgroup_store(sm0, sb + j * 8, LDS);
            if (RS > 1) simdgroup_store(sm1, sb + 8 * LDS + j * 8, LDS);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float sv[CPL];
        float mx = -INFINITY;
        for (int c = 0; c < CPL; c++) {
            const int key = k0 + c0 + c;
            float x = sb[row * LDS + c0 + c];
            if (key < rlo || key >= rhi) x = -INFINITY;
            sv[c] = x;
            mx = max(mx, x);
        }
        for (int m = 1; m < LPR; m <<= 1) mx = max(mx, simd_shuffle_xor(mx, ushort(m)));
        const float mNew = max(mrow, mx);
        float sum = 0.0f;
        for (int c = 0; c < CPL; c++) {
            const float e = mNew == -INFINITY ? 0.0f : exp(sv[c] - mNew);
            sum += e;
            pb[row * LDP + c0 + c] = half(e);
        }
        for (int m = 1; m < LPR; m <<= 1) sum += simd_shuffle_xor(sum, ushort(m));
        const float alpha = mrow == -INFINITY ? 0.0f : exp(mrow - mNew);
        lrow = lrow * alpha + sum;
        mrow = mNew;
        if (lane % LPR == 0) dg[(row / 8) * 64 + (row % 8) * 9] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_min(alpha == 1.0f ? 1 : 0) == 0) {
            for (int s = 0; s < RS; s++) {
                simdgroup_float8x8 dm;
                simdgroup_load(dm, dg + s * 64, 8);
                for (int i = 0; i < NI; i++) simdgroup_multiply(om[s * NI + i], dm, om[s * NI + i]);
            }
        }
        for (int j = 0; j < NJ; j++) {
            simdgroup_half8x8 pm0;
            simdgroup_half8x8 pm1;
            simdgroup_load(pm0, pb + j * 8, LDP);
            if (RS > 1) simdgroup_load(pm1, pb + 8 * LDP + j * 8, LDP);
            for (int i = 0; i < NI; i++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, vh + j * 8 * LDH + i * 8, LDH);
                simdgroup_multiply_accumulate(om[i], pm0, vm, om[i]);
                if (RS > 1) simdgroup_multiply_accumulate(om[NI + i], pm1, vm, om[NI + i]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // O = diag(1 / l) O
    if (lane % LPR == 0) dg[(row / 8) * 64 + (row % 8) * 9] = lrow > 0.0f ? 1.0f / lrow : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 0; s < RS; s++) {
        simdgroup_float8x8 dm;
        simdgroup_load(dm, dg + s * 64, 8);
        for (int i = 0; i < NI; i++) {
            simdgroup_float8x8 o;
            simdgroup_multiply(o, dm, om[s * NI + i]);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(o, sb, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int e = lane; e < 64; e += 32) {
                const int t = r0 + s * 8 + e / 8;
                if (t < p.t) out[(ulong)t * dim + h * D + i * 8 + e % 8] = sb[e];
            }
        }
    }
}

template [[host_name("vit_attention_mma_d64")]] kernel decltype(vit_attention_mma<64, 1>) vit_attention_mma<64, 1>;
template [[host_name("vit_attention_mma_d72")]] kernel decltype(vit_attention_mma<72, 1>) vit_attention_mma<72, 1>;
template [[host_name("vit_attention_mma_d80")]] kernel decltype(vit_attention_mma<80, 1>) vit_attention_mma<80, 1>;
template [[host_name("vit_attention_mma_d128")]] kernel decltype(vit_attention_mma<128, 1>) vit_attention_mma<128, 1>;

// ------------------------------------------------------------------------------------------------ rotary embedding

struct VitRopeParams {
    int t;
    int heads;
    int dh;         // head size; the pairs are (j, j + dh / 2)
};

// x[t][h][j], x[t][h][j + dh/2] rotated by cs[t][j] = (cos, sin) (interleaved pairs of floats), in place.
kernel void vit_rope(device float *x [[buffer(0)]],
                     device const float *cs [[buffer(1)]],
                     constant VitRopeParams &p [[buffer(2)]],
                     uint3 gid [[thread_position_in_grid]]) {
    const int half_dh = p.dh / 2;
    const int j = int(gid.x);
    const int h = int(gid.y);
    const int t = int(gid.z);
    if (j >= half_dh || h >= p.heads || t >= p.t) return;
    const ulong o = ((ulong)t * p.heads + h) * p.dh + j;
    const float c = cs[((ulong)t * half_dh + j) * 2];
    const float s = cs[((ulong)t * half_dh + j) * 2 + 1];
    const float x0 = x[o];
    const float x1 = x[o + half_dh];
    x[o] = x0 * c - x1 * s;
    x[o + half_dh] = x0 * s + x1 * c;
}

// ------------------------------------------------------------------------------------------------ elementwise

struct VitElemParams {
    int n;
    int act;
};

// a[i] = act(a[i]) * b[i]
kernel void vit_gated(device float *a [[buffer(0)]],
                      device const float *b [[buffer(1)]],
                      constant VitElemParams &p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (int(gid) < p.n) a[gid] = vit_apply(a[gid], p.act) * b[gid];
}

// a[i] = act(a[i])
kernel void vit_act(device float *a [[buffer(0)]],
                    constant VitElemParams &p [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    if (int(gid) < p.n) a[gid] = vit_apply(a[gid], p.act);
}

// a[i] += b[i]
kernel void vit_add(device float *a [[buffer(0)]],
                    device const float *b [[buffer(1)]],
                    constant VitElemParams &p [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (int(gid) < p.n) a[gid] += b[gid];
}

struct VitGatherParams {
    int rows;       // output rows of `width` floats
    int width;
    int group;      // vit_gather_mean: source rows averaged per output row
};

// out[r][c] = in[idx[r]][c]
kernel void vit_gather(device const float *in [[buffer(0)]],
                       device float *out [[buffer(1)]],
                       device const int *idx [[buffer(2)]],
                       constant VitGatherParams &p [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    const int c = int(gid.x);
    const int r = int(gid.y);
    if (c < p.width && r < p.rows) out[(ulong)r * p.width + c] = in[(ulong)idx[r] * p.width + c];
}

// out[r][c] = mean over g < group of in[idx[r * group + g]][c]
kernel void vit_gather_mean(device const float *in [[buffer(0)]],
                            device float *out [[buffer(1)]],
                            device const int *idx [[buffer(2)]],
                            constant VitGatherParams &p [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const int c = int(gid.x);
    const int r = int(gid.y);
    if (c >= p.width || r >= p.rows) return;
    float s = 0.0f;
    for (int g = 0; g < p.group; g++) s += in[(ulong)idx[r * p.group + g] * p.width + c];
    out[(ulong)r * p.width + c] = s / float(p.group);
}
