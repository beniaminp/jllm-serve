/*
 * AArch64 dot products, included once per level by jllm_cpu.c with SFX (name suffix), TGT (target attribute) and
 * KIND 0 = plain NEON (smull + sadalp), 1 = dotprod (sdot, ARMv8.2 FEAT_DotProd: Apple M1 and later, Cortex-A76+,
 * Neoverse N1+, Graviton 2+).
 *
 * The one primitive is dot16(acc, a, b): acc + the products of 16 signed bytes, summed into 4 int32 lanes (which
 * products land in which lane differs between the two forms; only the lanes' total is used, per 16 or 32 weights).
 */

#define F(name) JCAT(name, SFX)

TGT static inline int32x4_t F(dot16)(int32x4_t acc, int8x16_t a, int8x16_t b) {
#if KIND == 1
    return vdotq_s32(acc, a, b);
#else
    int16x8_t lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vpadalq_s16(vpadalq_s16(acc, lo), hi);
#endif
}

/* Q4_0 / Q4_0G: nibbles minus 8, low nibbles = weights 0..15, high = 16..31. */
TGT static inline float F(dot_q4_0)(const uint8_t *blk0, size_t stride, size_t qoff, const uint8_t *sc, int nb,
                                      const int8_t *x, const float *xd, const int32_t *xs) {
    (void) xs;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8x16_t s8 = vdupq_n_s8(8);
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    int b = 0;
    for (; b + 2 <= nb; b += 2) {
        const uint8_t *k0 = blk0 + b * stride;
        const uint8_t *k1 = k0 + stride;
        uint8x16_t r0 = vld1q_u8(k0 + qoff), r1 = vld1q_u8(k1 + qoff);
        const int8_t *x0 = x + b * QK, *x1 = x0 + QK;
        int32x4_t p0 = F(dot16)(vdupq_n_s32(0), vsubq_s8(vreinterpretq_s8_u8(vandq_u8(r0, m4)), s8), vld1q_s8(x0));
        p0 = F(dot16)(p0, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(r0, 4)), s8), vld1q_s8(x0 + 16));
        int32x4_t p1 = F(dot16)(vdupq_n_s32(0), vsubq_s8(vreinterpretq_s8_u8(vandq_u8(r1, m4)), s8), vld1q_s8(x1));
        p1 = F(dot16)(p1, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(r1, 4)), s8), vld1q_s8(x1 + 16));
        float f0 = (sc ? ld_f32(sc + 4 * b) : f16_to_f32(k0)) * xd[b];
        float f1 = (sc ? ld_f32(sc + 4 * (b + 1)) : f16_to_f32(k1)) * xd[b + 1];
        acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(p0), f0);
        acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(p1), f1);
    }
    for (; b < nb; b++) {
        const uint8_t *k0 = blk0 + b * stride;
        uint8x16_t r0 = vld1q_u8(k0 + qoff);
        const int8_t *x0 = x + b * QK;
        int32x4_t p0 = F(dot16)(vdupq_n_s32(0), vsubq_s8(vreinterpretq_s8_u8(vandq_u8(r0, m4)), s8), vld1q_s8(x0));
        p0 = F(dot16)(p0, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(r0, 4)), s8), vld1q_s8(x0 + 16));
        float f0 = (sc ? ld_f32(sc + 4 * b) : f16_to_f32(k0)) * xd[b];
        acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(p0), f0);
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1));
}

TGT static inline float F(dot_q8_0)(const uint8_t *blk0, size_t stride, size_t qoff, const uint8_t *sc, int nb,
                                      const int8_t *x, const float *xd) {
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    int b = 0;
    for (; b + 2 <= nb; b += 2) {
        const uint8_t *k0 = blk0 + b * stride;
        const uint8_t *k1 = k0 + stride;
        const int8_t *w0 = (const int8_t *) (k0 + qoff), *w1 = (const int8_t *) (k1 + qoff);
        const int8_t *x0 = x + b * QK, *x1 = x0 + QK;
        int32x4_t p0 = F(dot16)(F(dot16)(vdupq_n_s32(0), vld1q_s8(w0), vld1q_s8(x0)), vld1q_s8(w0 + 16), vld1q_s8(x0 + 16));
        int32x4_t p1 = F(dot16)(F(dot16)(vdupq_n_s32(0), vld1q_s8(w1), vld1q_s8(x1)), vld1q_s8(w1 + 16), vld1q_s8(x1 + 16));
        float f0 = (sc ? ld_f32(sc + 4 * b) : f16_to_f32(k0)) * xd[b];
        float f1 = (sc ? ld_f32(sc + 4 * (b + 1)) : f16_to_f32(k1)) * xd[b + 1];
        acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(p0), f0);
        acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(p1), f1);
    }
    for (; b < nb; b++) {
        const uint8_t *k0 = blk0 + b * stride;
        const int8_t *w0 = (const int8_t *) (k0 + qoff);
        const int8_t *x0 = x + b * QK;
        int32x4_t p0 = F(dot16)(F(dot16)(vdupq_n_s32(0), vld1q_s8(w0), vld1q_s8(x0)), vld1q_s8(w0 + 16), vld1q_s8(x0 + 16));
        float f0 = (sc ? ld_f32(sc + 4 * b) : f16_to_f32(k0)) * xd[b];
        acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(p0), f0);
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1));
}

TGT static inline float F(dot_q4_k)(const uint8_t *q, int nsb, const int8_t *x, const float *xd, const int32_t *xs) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    float minus = 0, ws[8];
    for (int sb = 0; sb < nsb; sb++, q += Q4_K_BYTES) {
        int blk = sb * 8;
        minus = k4_prologue(q, xd + blk, xs + blk, ws, minus);
        const uint8_t *qs = q + 16;
        const int8_t *xb = x + blk * QK;
        for (int g = 0; g < 4; g++) {
            uint8x16_t ra = vld1q_u8(qs + 32 * g), rb = vld1q_u8(qs + 32 * g + 16);
            const int8_t *xa = xb + 64 * g, *xh = xa + 32;
            int32x4_t p0 = F(dot16)(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(ra, m4)), vld1q_s8(xa));
            p0 = F(dot16)(p0, vreinterpretq_s8_u8(vandq_u8(rb, m4)), vld1q_s8(xa + 16));
            int32x4_t p1 = F(dot16)(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8(ra, 4)), vld1q_s8(xh));
            p1 = F(dot16)(p1, vreinterpretq_s8_u8(vshrq_n_u8(rb, 4)), vld1q_s8(xh + 16));
            acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(p0), ws[2 * g]);
            acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(p1), ws[2 * g + 1]);
        }
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1)) - minus;
}

TGT static inline float F(dot_q5_k)(const uint8_t *q, int nsb, const int8_t *x, const float *xd, const int32_t *xs) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const uint8x16_t b16 = vdupq_n_u8(0x10);
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    float minus = 0, ws[8];
    for (int sb = 0; sb < nsb; sb++, q += Q5_K_BYTES) {
        int blk = sb * 8;
        minus = k4_prologue(q, xd + blk, xs + blk, ws, minus);
        const uint8_t *qs = q + 48;
        const int8_t *xb = x + blk * QK;
        uint8x16_t ha = vld1q_u8(q + 16), hb = vld1q_u8(q + 32);
        for (int g = 0; g < 4; g++) {
            uint8x16_t ra = vld1q_u8(qs + 32 * g), rb = vld1q_u8(qs + 32 * g + 16);
            /* bit 0 of the (shifted) qh bytes -> bit 4 */
            uint8x16_t la = vorrq_u8(vandq_u8(ra, m4), vandq_u8(vshlq_n_u8(ha, 4), b16));
            uint8x16_t lb = vorrq_u8(vandq_u8(rb, m4), vandq_u8(vshlq_n_u8(hb, 4), b16));
            uint8x16_t ua = vorrq_u8(vshrq_n_u8(ra, 4), vandq_u8(vshlq_n_u8(ha, 3), b16));
            uint8x16_t ub = vorrq_u8(vshrq_n_u8(rb, 4), vandq_u8(vshlq_n_u8(hb, 3), b16));
            ha = vshrq_n_u8(ha, 2);
            hb = vshrq_n_u8(hb, 2);
            const int8_t *xa = xb + 64 * g, *xh = xa + 32;
            int32x4_t p0 = F(dot16)(F(dot16)(vdupq_n_s32(0), vreinterpretq_s8_u8(la), vld1q_s8(xa)),
                                    vreinterpretq_s8_u8(lb), vld1q_s8(xa + 16));
            int32x4_t p1 = F(dot16)(F(dot16)(vdupq_n_s32(0), vreinterpretq_s8_u8(ua), vld1q_s8(xh)),
                                    vreinterpretq_s8_u8(ub), vld1q_s8(xh + 16));
            acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(p0), ws[2 * g]);
            acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(p1), ws[2 * g + 1]);
        }
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1)) - minus;
}

TGT static inline float F(dot_q6_k)(const uint8_t *q, int nsb, const int8_t *x, const float *xd) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const uint8x16_t m3 = vdupq_n_u8(0x30);
    const int8x16_t c32 = vdupq_n_s8(32);
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    float ws[16];
    for (int sb = 0; sb < nsb; sb++, q += Q6_K_BYTES) {
        int blk0 = sb * 8;
        q6k_prologue(q, xd + blk0, ws);
        for (int half = 0; half < 2; half++) {
            const uint8_t *ql = q + 64 * half;
            const uint8_t *qh = q + 128 + 32 * half;
            const int8_t *xb = x + (blk0 + 4 * half) * QK;
            const float *wh = ws + 8 * half;
            for (int hh = 0; hh < 2; hh++) { /* 16-weight halves: one scale each */
                uint8x16_t l0 = vld1q_u8(ql + 16 * hh), l1 = vld1q_u8(ql + 32 + 16 * hh);
                uint8x16_t h = vld1q_u8(qh + 16 * hh);
                int8x16_t w0 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l0, m4), vandq_u8(vshlq_n_u8(h, 4), m3))), c32);
                int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l1, m4), vandq_u8(vshlq_n_u8(h, 2), m3))), c32);
                int8x16_t w2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l0, 4), vandq_u8(h, m3))), c32);
                int8x16_t w3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l1, 4), vandq_u8(vshrq_n_u8(h, 2), m3))), c32);
                int32x4_t p0 = F(dot16)(vdupq_n_s32(0), w0, vld1q_s8(xb + 16 * hh));
                int32x4_t p1 = F(dot16)(vdupq_n_s32(0), w1, vld1q_s8(xb + 32 + 16 * hh));
                int32x4_t p2 = F(dot16)(vdupq_n_s32(0), w2, vld1q_s8(xb + 64 + 16 * hh));
                int32x4_t p3 = F(dot16)(vdupq_n_s32(0), w3, vld1q_s8(xb + 96 + 16 * hh));
                acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(p0), wh[0 + hh]);
                acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(p1), wh[2 + hh]);
                acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(p2), wh[4 + hh]);
                acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(p3), wh[6 + hh]);
            }
        }
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1));
}

#undef F
