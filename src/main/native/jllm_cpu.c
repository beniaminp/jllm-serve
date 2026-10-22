/*
 * libjllm_cpu: the CPU backend's int8 decode dot products in C, called from Java through the Foreign Function &
 * Memory API (org.robomiri.jllm.cpu.NativeCpu). See docs/07-cpu-backend.md, section 7.16.
 *
 * The Java Vector API has no byte dot-product instruction, so the Java kernels (Int8Kernels, KQuantKernels) multiply
 * on 16-bit lanes. This library uses what the CPU has:
 *   x86-64  AVX512-VNNI (vpdpbusd on 256-bit registers), AVX-VNNI (the VEX form), AVX2 (vpmaddubsw + vpmaddwd)
 *   AArch64 dotprod (sdot), plain NEON (smull + sadalp)
 * with a portable scalar fallback. Every SIMD level is compiled into the same binary with per-function target
 * attributes; jllm_cpu_level() reports which one the running CPU supports and the dispatch table uses it.
 *
 * Data formats (identical to the Java side):
 *   activations  int8 q[n] in natural order, float d[n/32] (one scale per 32-element block), int32 s[n/32] (the
 *                block sums of q, for the K-quants' min terms and Q4_0's -8 offset), from jllm_cpu_quantize
 *   Q4_0 / Q8_0  the engine's layout: rows of 16 / 32 byte blocks, then (at scales_off) rows of f32 block scales
 *   Q4_0G, Q8_0G ggml blocks (f16 scale + 16 / 32 bytes), Q4_K, Q5_K, Q6_K ggml super-blocks of 256 weights
 *   Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4 ggml 32-weight blocks and IQ4_XS super-blocks (AArch64 and the scalar level only:
 *   jllm_cpu_has_type tells; the x86-64 levels leave them to the Java kernels)
 *
 * Exactness: every 32 (Q6_K: 16) weight sub-block's integer dot product is computed exactly, then multiplied by the
 * combined scale (weight scale * activation scale, rounded to float exactly as the Java code does) and summed in
 * float. Only that final float summation order differs from the Java kernels.
 *
 * Build: scripts/build-native-cpu.sh (no -march flags: the baseline code is generic, the SIMD functions carry their
 * own target attributes). Must be compiled without -ffast-math and with -ffp-contract=off, so the activation
 * quantizer rounds exactly like Int8Kernels.quantize.
 */
#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(_WIN32)
#define JLLM_API __declspec(dllexport)
#else
#define JLLM_API __attribute__((visibility("default")))
#endif

#define JCAT2(a, b) a##_##b
#define JCAT(a, b) JCAT2(a, b)

enum {
    T_Q4_0 = 0, T_Q8_0 = 1, T_Q4_0G = 2, T_Q8_0G = 3, T_Q4_K = 4, T_Q5_K = 5, T_Q6_K = 6,
    T_Q4_1 = 7, T_Q5_0 = 8, T_Q5_1 = 9, T_IQ4_NL = 10, T_MXFP4 = 11, T_IQ4_XS = 12, T_Q3_K = 13, T_COUNT = 14
};

enum {
    L_SCALAR = 0, L_NEON = 1, L_DOTPROD = 2, L_AVX2 = 3, L_AVXVNNI = 4, L_AVX512VNNI = 5, L_COUNT = 6
};

static const char *LEVEL_NAMES[L_COUNT] = {"scalar", "neon", "neon-dotprod", "avx2", "avx-vnni", "avx512-vnni"};

/* block geometry */
#define QK 32
#define QK_K 256
#define Q4_0G_BYTES 18
#define Q8_0G_BYTES 34
#define Q4_K_BYTES 144
#define Q5_K_BYTES 176
#define Q6_K_BYTES 210
#define Q4_1_BYTES 20
#define Q5_0_BYTES 22
#define Q5_1_BYTES 24
#define IQ4_NL_BYTES 18
#define MXFP4_BYTES 17
#define IQ4_XS_BYTES 136
#define Q3_K_BYTES 110

/* ggml's kvalues_iq4nl and kvalues_mxfp4 (the latter doubled: the block scale carries the factor 1/2) */
static const int8_t KV_IQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static const int8_t KV_MXFP4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

typedef void (*rows_fn)(const uint8_t *w, int64_t soff, int32_t r0, int32_t r1, int32_t cols,
                        const int8_t *xq, const float *xd, const int32_t *xs, float *out, int32_t acc);

/* ------------------------------------------------------------------------------------------------ helpers */

static inline float f16_to_f32(const uint8_t *p) {
    uint16_t h = (uint16_t) (p[0] | (p[1] << 8));
    uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else { /* subnormal: mant * 2^-24, exact in float */
            float f = (float) mant * 5.9604644775390625e-8f;
            memcpy(&bits, &f, 4);
            bits |= sign;
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

static inline float ld_f32(const uint8_t *p) {
    float f;
    memcpy(&f, p, 4);
    return f;
}

static inline uint32_t ld_u32(const uint8_t *p) { /* little-endian, like every target this library supports */
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

/* ggml_e8m0_to_fp32_half: 2^(e - 128), a subnormal for e < 2 */
static inline float e8m0_half(uint32_t e) {
    uint32_t bits = e < 2 ? 0x00200000u << e : (e - 1) << 23;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* The 6-bit scale and min of sub-block j of a Q4_K / Q5_K super-block (ggml's get_scale_min_k4). */
static inline void scale_min_k4(const uint8_t *q, uint8_t *sc, uint8_t *mn) {
    for (int j = 0; j < 4; j++) {
        sc[j] = q[j] & 63;
        mn[j] = q[j + 4] & 63;
    }
    for (int j = 4; j < 8; j++) {
        sc[j] = (uint8_t) ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        mn[j] = (uint8_t) ((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

/*
 * The per-sub-block float factors of a K-quant super-block, computed exactly as KQuantKernels does (same operations,
 * same order, so the same roundings): ws[j] = d * sc_j * xd_j and the min term
 * minus += dmin * m_j * xd_j * xs_j accumulated in sub-block order.
 */
static inline float k4_prologue(const uint8_t *q, const float *xd, const int32_t *xs, float *ws, float minus) {
    float d = f16_to_f32(q);
    float dmin = f16_to_f32(q + 2);
    uint8_t sc[8], mn[8];
    scale_min_k4(q + 4, sc, mn);
    for (int j = 0; j < 8; j++) {
        float x = xd[j];
        minus += dmin * (float) mn[j] * x * (float) xs[j];
        ws[j] = d * (float) sc[j] * x;
    }
    return minus;
}

static inline void q6k_prologue(const uint8_t *q, const float *xd, float *ws) {
    float d = f16_to_f32(q + 208);
    const int8_t *sc = (const int8_t *) (q + 192);
    for (int k = 0; k < 16; k++) {
        ws[k] = d * (float) sc[k] * xd[k >> 1];
    }
}

static inline void store(float *out, int32_t i, float v, int32_t acc) {
    out[i] = acc ? out[i] + v : v;
}

/* ------------------------------------------------------------------------------------------------ scalar */

static float dot_q4_0_scalar(const uint8_t *blk0, size_t stride, size_t qoff, const uint8_t *sc, int nb,
                             const int8_t *x, const float *xd, const int32_t *xs) {
    (void) xs;
    float sum = 0;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = blk0 + b * stride;
        const uint8_t *qs = blk + qoff;
        const int8_t *xb = x + b * QK;
        int32_t isum = 0;
        for (int j = 0; j < 16; j++) {
            isum += ((qs[j] & 0xF) - 8) * xb[j] + ((qs[j] >> 4) - 8) * xb[j + 16];
        }
        float s = (sc ? ld_f32(sc + 4 * b) : f16_to_f32(blk)) * xd[b];
        sum += (float) isum * s;
    }
    return sum;
}

static float dot_q8_0_scalar(const uint8_t *blk0, size_t stride, size_t qoff, const uint8_t *sc, int nb,
                             const int8_t *x, const float *xd) {
    float sum = 0;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = blk0 + b * stride;
        const int8_t *qs = (const int8_t *) (blk + qoff);
        const int8_t *xb = x + b * QK;
        int32_t isum = 0;
        for (int j = 0; j < QK; j++) {
            isum += qs[j] * xb[j];
        }
        float s = (sc ? ld_f32(sc + 4 * b) : f16_to_f32(blk)) * xd[b];
        sum += (float) isum * s;
    }
    return sum;
}

static float dot_q4_k_scalar(const uint8_t *q, int nsb, const int8_t *x, const float *xd, const int32_t *xs) {
    float sum = 0, minus = 0, ws[8];
    for (int sb = 0; sb < nsb; sb++, q += Q4_K_BYTES) {
        int blk = sb * 8;
        minus = k4_prologue(q, xd + blk, xs + blk, ws, minus);
        const uint8_t *qs = q + 16;
        for (int g = 0; g < 4; g++) {
            const int8_t *xa = x + (blk + 2 * g) * QK;
            int32_t lo = 0, hi = 0;
            for (int l = 0; l < 32; l++) {
                lo += (qs[32 * g + l] & 0xF) * xa[l];
                hi += (qs[32 * g + l] >> 4) * xa[l + QK];
            }
            sum += (float) lo * ws[2 * g];
            sum += (float) hi * ws[2 * g + 1];
        }
    }
    return sum - minus;
}

static float dot_q5_k_scalar(const uint8_t *q, int nsb, const int8_t *x, const float *xd, const int32_t *xs) {
    float sum = 0, minus = 0, ws[8];
    for (int sb = 0; sb < nsb; sb++, q += Q5_K_BYTES) {
        int blk = sb * 8;
        minus = k4_prologue(q, xd + blk, xs + blk, ws, minus);
        const uint8_t *qh = q + 16;
        const uint8_t *qs = q + 48;
        for (int g = 0; g < 4; g++) {
            const int8_t *xa = x + (blk + 2 * g) * QK;
            int32_t lo = 0, hi = 0;
            for (int l = 0; l < 32; l++) {
                int a = (qs[32 * g + l] & 0xF) | (((qh[l] >> (2 * g)) & 1) << 4);
                int b = (qs[32 * g + l] >> 4) | (((qh[l] >> (2 * g + 1)) & 1) << 4);
                lo += a * xa[l];
                hi += b * xa[l + QK];
            }
            sum += (float) lo * ws[2 * g];
            sum += (float) hi * ws[2 * g + 1];
        }
    }
    return sum - minus;
}

static float dot_q6_k_scalar(const uint8_t *q, int nsb, const int8_t *x, const float *xd) {
    float sum = 0, ws[16];
    for (int sb = 0; sb < nsb; sb++, q += Q6_K_BYTES) {
        int blk0 = sb * 8;
        q6k_prologue(q, xd + blk0, ws);
        for (int half = 0; half < 2; half++) {
            const uint8_t *ql = q + 64 * half;
            const uint8_t *qh = q + 128 + 32 * half;
            for (int g = 0; g < 4; g++) {
                const int8_t *xa = x + (blk0 + 4 * half + g) * QK;
                for (int hh = 0; hh < 2; hh++) {
                    int32_t isum = 0;
                    for (int l = 16 * hh; l < 16 * hh + 16; l++) {
                        int nib = (ql[32 * (g & 1) + l] >> (4 * (g >> 1))) & 0xF;
                        int v = (nib | (((qh[l] >> (2 * g)) & 3) << 4)) - 32;
                        isum += v * xa[l];
                    }
                    sum += (float) isum * ws[8 * half + 2 * g + hh];
                }
            }
        }
    }
    return sum;
}

/* Q4_1 / Q5_1: w = d q + m, so a block's product is d xd (sum q x) + m xd (sum x), the latter from the activation's block sums. */
static float dot_q41_scalar(const uint8_t *q, int nb, const int8_t *x, const float *xd, const int32_t *xs, int q5) {
    float sum = 0, minus = 0;
    for (int b = 0; b < nb; b++) {
        const uint8_t *k = q + b * (q5 ? Q5_1_BYTES : Q4_1_BYTES);
        const uint8_t *qs = k + (q5 ? 8 : 4);
        uint32_t qh = q5 ? ld_u32(k + 4) : 0;
        const int8_t *xb = x + b * QK;
        int32_t isum = 0;
        for (int j = 0; j < 16; j++) {
            int lo = (qs[j] & 0xF) | (q5 ? (int) (((qh >> j) & 1) << 4) : 0);
            int hi = (qs[j] >> 4) | (q5 ? (int) (((qh >> (j + 16)) & 1) << 4) : 0);
            isum += lo * xb[j] + hi * xb[j + 16];
        }
        float f = f16_to_f32(k) * xd[b];
        sum += (float) isum * f;
        minus += f16_to_f32(k + 2) * xd[b] * (float) xs[b];
    }
    return sum + minus;
}

/* Q5_0: w = (q5 - 16) d */
static float dot_q50_scalar(const uint8_t *q, int nb, const int8_t *x, const float *xd) {
    float sum = 0;
    for (int b = 0; b < nb; b++) {
        const uint8_t *k = q + b * Q5_0_BYTES;
        const uint8_t *qs = k + 6;
        uint32_t qh = ld_u32(k + 2);
        const int8_t *xb = x + b * QK;
        int32_t isum = 0;
        for (int j = 0; j < 16; j++) {
            int lo = ((qs[j] & 0xF) | (int) (((qh >> j) & 1) << 4)) - 16;
            int hi = ((qs[j] >> 4) | (int) (((qh >> (j + 16)) & 1) << 4)) - 16;
            isum += lo * xb[j] + hi * xb[j + 16];
        }
        sum += (float) isum * (f16_to_f32(k) * xd[b]);
    }
    return sum;
}

/* IQ4_NL (f16 scale, nibbles at byte 2) and MXFP4 (e8m0 byte, nibbles at byte 1): w = d table[nibble] */
static float dot_tab_scalar(const uint8_t *q, int nb, const int8_t *x, const float *xd, int fp4) {
    const int8_t *tab = fp4 ? KV_MXFP4 : KV_IQ4NL;
    float sum = 0;
    for (int b = 0; b < nb; b++) {
        const uint8_t *k = q + b * (fp4 ? MXFP4_BYTES : IQ4_NL_BYTES);
        const uint8_t *qs = k + (fp4 ? 1 : 2);
        const int8_t *xb = x + b * QK;
        int32_t isum = 0;
        for (int j = 0; j < 16; j++) {
            isum += tab[qs[j] & 0xF] * xb[j] + tab[qs[j] >> 4] * xb[j + 16];
        }
        float d = fp4 ? e8m0_half(k[0]) : f16_to_f32(k);
        sum += (float) isum * (d * xd[b]);
    }
    return sum;
}

/* The sixteen 6-bit scales (minus 32) of a Q3_K block: scale s takes its low 4 bits from byte 4 (g & 1) + i of the 12 scale
 * bytes (low nibble for g < 2, high for g >= 2) and its top 2 bits from bits 2g.. of byte 8 + i, with g = s >> 2, i = s & 3. */
static inline void q3k_scales(const uint8_t *q, int *sc) {
    const uint8_t *s = q + 96;
    for (int k = 0; k < 16; k++) {
        int g = k >> 2, i = k & 3;
        int lowb = s[4 * (g & 1) + i];
        int lo = g < 2 ? (lowb & 15) : (lowb >> 4);
        sc[k] = (lo | (((s[8 + i] >> (2 * g)) & 3) << 4)) - 32;
    }
}

/* Q3_K: w = d (sc - 32) (((qs[32 n + l] >> 2j) & 3) - 4 + 4 hmask[l] bit u), unit u = 4 n + j, two 16-weight runs with their own scale */
static float dot_q3_k_scalar(const uint8_t *q, int nsb, const int8_t *x, const float *xd) {
    float sum = 0;
    int sc[16];
    for (int sb = 0; sb < nsb; sb++, q += Q3_K_BYTES) {
        float d = f16_to_f32(q + 108);
        q3k_scales(q, sc);
        for (int u = 0; u < 8; u++) {
            int n = u >> 2, j = u & 3, blk = sb * 8 + u;
            const uint8_t *qs = q + 32 + 32 * n;
            const int8_t *xb = x + blk * QK;
            int32_t i0 = 0, i1 = 0;
            for (int l = 0; l < 16; l++) {
                i0 += (((qs[l] >> (2 * j)) & 3) - 4 + 4 * ((q[l] >> u) & 1)) * xb[l];
                i1 += (((qs[16 + l] >> (2 * j)) & 3) - 4 + 4 * ((q[16 + l] >> u) & 1)) * xb[16 + l];
            }
            sum += (float) i0 * (d * (float) sc[2 * u] * xd[blk]);
            sum += (float) i1 * (d * (float) sc[2 * u + 1] * xd[blk]);
        }
    }
    return sum;
}

/* IQ4_XS: eight IQ4_NL-style 32-weight sub-blocks per 256 with their own 6-bit scales */
static float dot_iq4_xs_scalar(const uint8_t *q, int nsb, const int8_t *x, const float *xd) {
    float sum = 0;
    for (int sb = 0; sb < nsb; sb++, q += IQ4_XS_BYTES) {
        float d = f16_to_f32(q);
        uint32_t sh = (uint32_t) (q[2] | (q[3] << 8));
        for (int ib = 0; ib < 8; ib++) {
            const uint8_t *qs = q + 8 + 16 * ib;
            const int8_t *xb = x + (sb * 8 + ib) * QK;
            int32_t isum = 0;
            for (int j = 0; j < 16; j++) {
                isum += KV_IQ4NL[qs[j] & 0xF] * xb[j] + KV_IQ4NL[qs[j] >> 4] * xb[j + 16];
            }
            int ls = ((q[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) | (int) (((sh >> (2 * ib)) & 3) << 4);
            sum += (float) isum * (d * (float) (ls - 32) * xd[sb * 8 + ib]);
        }
    }
    return sum;
}

/* Row loops shared by every level: ROWS_BODY(dot expression using row pointer `rw` and scale pointer `rs`). */
#define NB32 (cols / QK)
#define NBK (cols / QK_K)

#define DEFINE_ROWS(SFX, ATTR)                                                                                          \
    ATTR static void JCAT(rows_q4_0, SFX)(const uint8_t *w, int64_t soff, int32_t r0, int32_t r1, int32_t cols,        \
                                         const int8_t *xq, const float *xd, const int32_t *xs, float *out,             \
                                         int32_t acc) {                                                                \
        size_t rb = (size_t) NB32 * 16, sb = (size_t) NB32 * 4;                                                        \
        for (int32_t r = r0; r < r1; r++)                                                                              \
            store(out, r - r0, JCAT(dot_q4_0, SFX)(w + r * rb, 16, 0, w + soff + r * sb, NB32, xq, xd, xs), acc);    \
    }                                                                                                                  \
    ATTR static void JCAT(rows_q8_0, SFX)(const uint8_t *w, int64_t soff, int32_t r0, int32_t r1, int32_t cols,        \
                                         const int8_t *xq, const float *xd, const int32_t *xs, float *out,             \
                                         int32_t acc) {                                                                \
        size_t rb = (size_t) NB32 * 32, sb = (size_t) NB32 * 4;                                                        \
        (void) xs;                                                                                                     \
        for (int32_t r = r0; r < r1; r++)                                                                              \
            store(out, r - r0, JCAT(dot_q8_0, SFX)(w + r * rb, 32, 0, w + soff + r * sb, NB32, xq, xd), acc);         \
    }                                                                                                                  \
    ATTR static void JCAT(rows_q4_0g, SFX)(const uint8_t *w, int64_t soff, int32_t r0, int32_t r1, int32_t cols,       \
                                          const int8_t *xq, const float *xd, const int32_t *xs, float *out,            \
                                          int32_t acc) {                                                               \
        size_t rb = (size_t) NB32 * Q4_0G_BYTES;                                                                       \
        (void) soff;                                                                                                   \
        for (int32_t r = r0; r < r1; r++)                                                                              \
            store(out, r - r0, JCAT(dot_q4_0, SFX)(w + r * rb, Q4_0G_BYTES, 2, NULL, NB32, xq, xd, xs), acc);        \
    }                                                                                                                  \
    ATTR static void JCAT(rows_q8_0g, SFX)(const uint8_t *w, int64_t soff, int32_t r0, int32_t r1, int32_t cols,       \
                                          const int8_t *xq, const float *xd, const int32_t *xs, float *out,            \
                                          int32_t acc) {                                                               \
        size_t rb = (size_t) NB32 * Q8_0G_BYTES;                                                                       \
        (void) soff;                                                                                                   \
        (void) xs;                                                                                                     \
        for (int32_t r = r0; r < r1; r++)                                                                              \
            store(out, r - r0, JCAT(dot_q8_0, SFX)(w + r * rb, Q8_0G_BYTES, 2, NULL, NB32, xq, xd), acc);            \
    }                                                                                                                  \
    ATTR static void JCAT(rows_q4_k, SFX)(const uint8_t *w, int64_t soff, int32_t r0, int32_t r1, int32_t cols,        \
                                         const int8_t *xq, const float *xd, const int32_t *xs, float *out,             \
                                         int32_t acc) {                                                                \
        size_t rb = (size_t) NBK * Q4_K_BYTES;                                                                         \
        (void) soff;                                                                                                   \
        for (int32_t r = r0; r < r1; r++)                                                                              \
            store(out, r - r0, JCAT(dot_q4_k, SFX)(w + r * rb, NBK, xq, xd, xs), acc);                                \
    }                                                                                                                  \
    ATTR static void JCAT(rows_q5_k, SFX)(const uint8_t *w, int64_t soff, int32_t r0, int32_t r1, int32_t cols,        \
                                         const int8_t *xq, const float *xd, const int32_t *xs, float *out,             \
                                         int32_t acc) {                                                                \
        size_t rb = (size_t) NBK * Q5_K_BYTES;                                                                         \
        (void) soff;                                                                                                   \
        for (int32_t r = r0; r < r1; r++)                                                                              \
            store(out, r - r0, JCAT(dot_q5_k, SFX)(w + r * rb, NBK, xq, xd, xs), acc);                                \
    }                                                                                                                  \
    ATTR static void JCAT(rows_q6_k, SFX)(const uint8_t *w, int64_t soff, int32_t r0, int32_t r1, int32_t cols,        \
                                         const int8_t *xq, const float *xd, const int32_t *xs, float *out,             \
                                         int32_t acc) {                                                                \
        size_t rb = (size_t) NBK * Q6_K_BYTES;                                                                         \
        (void) soff;                                                                                                   \
        (void) xs;                                                                                                     \
        for (int32_t r = r0; r < r1; r++)                                                                              \
            store(out, r - r0, JCAT(dot_q6_k, SFX)(w + r * rb, NBK, xq, xd), acc);                                    \
    }

DEFINE_ROWS(scalar, )

/* Row loops of the formats added later: the scalar and AArch64 levels only (the x86-64 levels do not define them). */
#define DEFINE_ROWS_NEW(SFX, ATTR)                                                                                      \
    ATTR static void JCAT(rows_q4_1, SFX)(const uint8_t *w, int64_t soff, int32_t r0, int32_t r1, int32_t cols,        \
                                         const int8_t *xq, const float *xd, const int32_t *xs, float *out,             \
                                         int32_t acc) {                                                                \
        size_t rb = (size_t) NB32 * Q4_1_BYTES;                                                                        \
        (void) soff;                                                                                                   \
        for (int32_t r = r0; r < r1; r++)                                                                              \
            store(out, r - r0, JCAT(dot_q41, SFX)(w + r * rb, NB32, xq, xd, xs, 0), acc);                             \
    }                                                                                                                  \
    ATTR static void JCAT(rows_q5_1, SFX)(const uint8_t *w, int64_t soff, int32_t r0, int32_t r1, int32_t cols,        \
                                         const int8_t *xq, const float *xd, const int32_t *xs, float *out,             \
                                         int32_t acc) {                                                                \
        size_t rb = (size_t) NB32 * Q5_1_BYTES;                                                                        \
        (void) soff;                                                                                                   \
        for (int32_t r = r0; r < r1; r++)                                                                              \
            store(out, r - r0, JCAT(dot_q41, SFX)(w + r * rb, NB32, xq, xd, xs, 1), acc);                             \
    }                                                                                                                  \
    ATTR static void JCAT(rows_q5_0, SFX)(const uint8_t *w, int64_t soff, int32_t r0, int32_t r1, int32_t cols,        \
                                         const int8_t *xq, const float *xd, const int32_t *xs, float *out,             \
                                         int32_t acc) {                                                                \
        size_t rb = (size_t) NB32 * Q5_0_BYTES;                                                                        \
        (void) soff;                                                                                                   \
        (void) xs;                                                                                                     \
        for (int32_t r = r0; r < r1; r++)                                                                              \
            store(out, r - r0, JCAT(dot_q50, SFX)(w + r * rb, NB32, xq, xd), acc);                                    \
    }                                                                                                                  \
    ATTR static void JCAT(rows_iq4_nl, SFX)(const uint8_t *w, int64_t soff, int32_t r0, int32_t r1, int32_t cols,      \
                                           const int8_t *xq, const float *xd, const int32_t *xs, float *out,           \
                                           int32_t acc) {                                                              \
        size_t rb = (size_t) NB32 * IQ4_NL_BYTES;                                                                      \
        (void) soff;                                                                                                   \
        (void) xs;                                                                                                     \
        for (int32_t r = r0; r < r1; r++)                                                                              \
            store(out, r - r0, JCAT(dot_tab, SFX)(w + r * rb, NB32, xq, xd, 0), acc);                                 \
    }                                                                                                                  \
    ATTR static void JCAT(rows_mxfp4, SFX)(const uint8_t *w, int64_t soff, int32_t r0, int32_t r1, int32_t cols,       \
                                          const int8_t *xq, const float *xd, const int32_t *xs, float *out,            \
                                          int32_t acc) {                                                               \
        size_t rb = (size_t) NB32 * MXFP4_BYTES;                                                                       \
        (void) soff;                                                                                                   \
        (void) xs;                                                                                                     \
        for (int32_t r = r0; r < r1; r++)                                                                              \
            store(out, r - r0, JCAT(dot_tab, SFX)(w + r * rb, NB32, xq, xd, 1), acc);                                 \
    }                                                                                                                  \
    ATTR static void JCAT(rows_q3_k, SFX)(const uint8_t *w, int64_t soff, int32_t r0, int32_t r1, int32_t cols,        \
                                         const int8_t *xq, const float *xd, const int32_t *xs, float *out,             \
                                         int32_t acc) {                                                                \
        size_t rb = (size_t) NBK * Q3_K_BYTES;                                                                         \
        (void) soff;                                                                                                   \
        (void) xs;                                                                                                     \
        for (int32_t r = r0; r < r1; r++)                                                                              \
            store(out, r - r0, JCAT(dot_q3_k, SFX)(w + r * rb, NBK, xq, xd), acc);                                    \
    }                                                                                                                  \
    ATTR static void JCAT(rows_iq4_xs, SFX)(const uint8_t *w, int64_t soff, int32_t r0, int32_t r1, int32_t cols,      \
                                           const int8_t *xq, const float *xd, const int32_t *xs, float *out,           \
                                           int32_t acc) {                                                              \
        size_t rb = (size_t) NBK * IQ4_XS_BYTES;                                                                       \
        (void) soff;                                                                                                   \
        (void) xs;                                                                                                     \
        for (int32_t r = r0; r < r1; r++)                                                                              \
            store(out, r - r0, JCAT(dot_iq4_xs, SFX)(w + r * rb, NBK, xq, xd), acc);                                  \
    }

DEFINE_ROWS_NEW(scalar, )

/* ------------------------------------------------------------------------------------------------ SIMD levels */

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <cpuid.h>

#define SFX avx2
#define TGT __attribute__((target("avx2,fma,f16c")))
#define KIND 0
#include "jllm_cpu_x86.inc"
DEFINE_ROWS(avx2, TGT)
#undef SFX
#undef TGT
#undef KIND

#define SFX avxvnni
#define TGT __attribute__((target("avx2,fma,f16c,avxvnni")))
#define KIND 1
#include "jllm_cpu_x86.inc"
DEFINE_ROWS(avxvnni, TGT)
#undef SFX
#undef TGT
#undef KIND

#define SFX avx512vnni
#define TGT __attribute__((target("avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")))
#define KIND 2
#include "jllm_cpu_x86.inc"
DEFINE_ROWS(avx512vnni, TGT)
#undef SFX
#undef TGT
#undef KIND

/* bit L set: the CPU can run level L (the levels are not supersets of each other: AVX512-VNNI does not imply AVX-VNNI) */
static unsigned supported = 1u << L_SCALAR;

static uint64_t xgetbv0(void) {
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t) hi << 32) | lo;
}

static int detect_level(void) {
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) {
        return L_SCALAR;
    }
    int osxsave = (c >> 27) & 1, fma = (c >> 12) & 1, f16c = (c >> 29) & 1, avx = (c >> 28) & 1;
    if (!osxsave || !avx || !fma || !f16c) {
        return L_SCALAR;
    }
    uint64_t xcr0 = xgetbv0();
    if ((xcr0 & 6) != 6) { /* XMM and YMM state */
        return L_SCALAR;
    }
    if (__get_cpuid_max(0, NULL) < 7) {
        return L_SCALAR;
    }
    __cpuid_count(7, 0, a, b, c, d);
    int avx2 = (b >> 5) & 1, avx512f = (b >> 16) & 1, avx512bw = (b >> 30) & 1, avx512vl = (b >> 31) & 1;
    int avx512vnni = (c >> 11) & 1;
    unsigned a1, b1, c1, d1;
    __cpuid_count(7, 1, a1, b1, c1, d1);
    int avxvnni = (a1 >> 4) & 1;
    if (!avx2) {
        return L_SCALAR;
    }
    int zmm = (xcr0 & 0xE6) == 0xE6; /* opmask, ZMM_Hi256, Hi16_ZMM state */
    supported |= 1u << L_AVX2;
    if (avxvnni) {
        supported |= 1u << L_AVXVNNI;
    }
    if (zmm && avx512f && avx512bw && avx512vl && avx512vnni) {
        supported |= 1u << L_AVX512VNNI;
        return L_AVX512VNNI;
    }
    return avxvnni ? L_AVXVNNI : L_AVX2;
}

#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
static unsigned supported = (1u << L_SCALAR) | (1u << L_NEON);
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#endif

#define SFX neon
#define TGT
#define KIND 0
#include "jllm_cpu_arm.inc"
DEFINE_ROWS(neon, TGT)
DEFINE_ROWS_NEW(neon, TGT)
#undef SFX
#undef TGT
#undef KIND

#define SFX dotprod
#define TGT __attribute__((target("arch=armv8.2-a+dotprod")))
#define KIND 1
#include "jllm_cpu_arm.inc"
DEFINE_ROWS(dotprod, TGT)
DEFINE_ROWS_NEW(dotprod, TGT)
#undef SFX
#undef TGT
#undef KIND

static int detect_level(void) {
#if defined(__APPLE__)
    int v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &len, NULL, 0) == 0 && v) {
        supported |= 1u << L_DOTPROD;
        return L_DOTPROD;
    }
    return L_NEON;
#elif defined(__linux__)
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
    if (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) {
        supported |= 1u << L_DOTPROD;
        return L_DOTPROD;
    }
    return L_NEON;
#else
    return L_NEON;
#endif
}

#else
static unsigned supported = 1u << L_SCALAR;

static int detect_level(void) {
    return L_SCALAR;
}
#endif

/* ------------------------------------------------------------------------------------------------ dispatch */

static rows_fn TABLE[L_COUNT][T_COUNT];
static int detected = -1;
static int active = L_SCALAR;

#define FILL(L, SFX)                                                                                                   \
    TABLE[L][T_Q4_0] = JCAT(rows_q4_0, SFX);                                                                           \
    TABLE[L][T_Q8_0] = JCAT(rows_q8_0, SFX);                                                                           \
    TABLE[L][T_Q4_0G] = JCAT(rows_q4_0g, SFX);                                                                         \
    TABLE[L][T_Q8_0G] = JCAT(rows_q8_0g, SFX);                                                                         \
    TABLE[L][T_Q4_K] = JCAT(rows_q4_k, SFX);                                                                           \
    TABLE[L][T_Q5_K] = JCAT(rows_q5_k, SFX);                                                                           \
    TABLE[L][T_Q6_K] = JCAT(rows_q6_k, SFX);

#define FILL_NEW(L, SFX)                                                                                               \
    TABLE[L][T_Q4_1] = JCAT(rows_q4_1, SFX);                                                                           \
    TABLE[L][T_Q5_0] = JCAT(rows_q5_0, SFX);                                                                           \
    TABLE[L][T_Q5_1] = JCAT(rows_q5_1, SFX);                                                                           \
    TABLE[L][T_IQ4_NL] = JCAT(rows_iq4_nl, SFX);                                                                       \
    TABLE[L][T_MXFP4] = JCAT(rows_mxfp4, SFX);                                                                         \
    TABLE[L][T_IQ4_XS] = JCAT(rows_iq4_xs, SFX);                                                                       \
    TABLE[L][T_Q3_K] = JCAT(rows_q3_k, SFX);

static void init(void) {
    if (detected >= 0) {
        return;
    }
    FILL(L_SCALAR, scalar)
    FILL_NEW(L_SCALAR, scalar)
#if defined(__x86_64__) || defined(_M_X64)
    FILL(L_AVX2, avx2)
    FILL(L_AVXVNNI, avxvnni)
    FILL(L_AVX512VNNI, avx512vnni)
#elif defined(__aarch64__) || defined(_M_ARM64)
    FILL(L_NEON, neon)
    FILL(L_DOTPROD, dotprod)
    FILL_NEW(L_NEON, neon)
    FILL_NEW(L_DOTPROD, dotprod)
#endif
    detected = detect_level();
    active = detected;
}

/* Version of this C interface; NativeCpu refuses a library with another one. */
JLLM_API int32_t jllm_cpu_abi(void) {
    return 1;
}

/* The best level this CPU supports (the active one unless jllm_cpu_set_level lowered it). */
JLLM_API int32_t jllm_cpu_level(void) {
    init();
    return active;
}

JLLM_API int32_t jllm_cpu_detected_level(void) {
    init();
    return detected;
}

/* Whether the active level has a kernel for the type (the formats added later are not compiled for x86-64). */
JLLM_API int32_t jllm_cpu_has_type(int32_t type) {
    init();
    return type >= 0 && type < T_COUNT && TABLE[active][type] != NULL;
}

JLLM_API const char *jllm_cpu_level_name(int32_t level) {
    return level >= 0 && level < L_COUNT ? LEVEL_NAMES[level] : "?";
}

/*
 * Selects a level for the following calls (tests: every level the CPU supports can be checked against the Java
 * kernels). A level the CPU does not support, or not compiled for this architecture, is refused; returns the active
 * level.
 */
JLLM_API int32_t jllm_cpu_set_level(int32_t level) {
    init();
    if (level >= 0 && level < L_COUNT && TABLE[level][0] != NULL && (supported & (1u << level))) {
        active = level;
    }
    return active;
}

/*
 * Quantizes x[0 .. n) (n a multiple of 32) to int8 per 32-block, as Int8Kernels.quantize does but in natural order:
 * d = max|x| / 127, q = rint(x / d) (round half to even), s = sum of the block's q.
 */
JLLM_API void jllm_cpu_quantize(const float *x, int32_t n, int8_t *q, float *d, int32_t *s) {
    for (int b = 0; b < n / QK; b++) {
        const float *xb = x + b * QK;
        float amax = 0;
        for (int j = 0; j < QK; j++) {
            float a = fabsf(xb[j]);
            amax = a > amax ? a : amax;
        }
        float dd = amax / 127.0f;
        float id = dd != 0 ? 1.0f / dd : 0.0f;
        d[b] = dd;
        int32_t sum = 0;
        for (int j = 0; j < QK; j++) {
            int v = (int) rintf(xb[j] * id);
            q[b * QK + j] = (int8_t) v;
            sum += v;
        }
        s[b] = sum;
    }
}

/*
 * out[i] (+)= dot(row r0 + i, x) for i < r1 - r0. w is the tensor's first byte; scales_off the offset of the f32
 * scale region (Q4_0, Q8_0 only); cols the row length (a multiple of 32, of 256 for the K-quants). xq / xd / xs are
 * the activation from jllm_cpu_quantize, starting at the block the rows multiply (a MoE slot's offset).
 */
JLLM_API void jllm_cpu_dot_rows(int32_t type, const uint8_t *w, int64_t scales_off, int32_t r0, int32_t r1,
                                int32_t cols, const int8_t *xq, const float *xd, const int32_t *xs, float *out,
                                int32_t accumulate) {
    init();
    if (type < 0 || type >= T_COUNT || TABLE[active][type] == NULL) {
        return;
    }
    TABLE[active][type](w, scales_off, r0, r1, cols, xq, xd, xs, out, accumulate);
}
