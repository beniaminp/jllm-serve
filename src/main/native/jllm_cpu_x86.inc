/*
 * x86-64 dot products on 256-bit registers, included once per level by jllm_cpu.c with
 *   SFX  the level's name suffix, TGT its target attribute,
 *   KIND 0 = AVX2 (vpmaddubsw + vpmaddwd), 1 = AVX-VNNI (VEX vpdpbusd), 2 = AVX512-VNNI (EVEX vpdpbusd, ymm form).
 *
 * The one primitive is dp(acc, u, s): acc + the sums of four adjacent products u_i * s_i (u unsigned bytes, s
 * signed bytes) in each of the 8 int32 lanes. Signed x signed products go through it as |a| * (b with a's sign)
 * (vpsignb), which is exact for a = -128 too. All products here stay within the exact range of the AVX2 form
 * (vpmaddubsw saturates int16 pair sums: at most 2 * 128 * 127 = 32512 here).
 */

#define F(name) JCAT(name, SFX)

TGT static inline __m256i F(dp)(__m256i acc, __m256i u, __m256i s) {
#if KIND == 0
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)));
#elif KIND == 1
    return _mm256_dpbusd_avx_epi32(acc, u, s);
#else
    return _mm256_dpbusd_epi32(acc, u, s);
#endif
}

/* signed a x signed b, four products per int32 lane */
TGT static inline __m256i F(dp_ss)(__m256i acc, __m256i a, __m256i b) {
    return F(dp)(acc, _mm256_sign_epi8(a, a), _mm256_sign_epi8(b, a));
}

TGT static inline float F(hsum)(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

TGT static inline float F(f16)(const uint8_t *p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return _cvtsh_ss(h);
}

/*
 * Q4_0 (engine layout: sc != NULL, f32 scales) and Q4_0G (ggml: f16 scale at the block start). Byte j of a block
 * holds weight j (low nibble) and j + 16 (high nibble): low nibbles in the lower 128 bits, high nibbles in the upper
 * 128 bits give the 32 weights in natural order. The nibbles enter vpdpbusd unsigned (0..15) and the -8 offset is
 * folded into the start value of the accumulator: -8 * (sum of the block's activations) in lane 0, exact.
 */
TGT static inline float F(dot_q4_0)(const uint8_t *blk0, size_t stride, size_t qoff, const uint8_t *sc, int nb,
                                      const int8_t *x, const float *xd, const int32_t *xs) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int b = 0;
    for (; b + 2 <= nb; b += 2) {
        const uint8_t *k0 = blk0 + b * stride;
        const uint8_t *k1 = k0 + stride;
        __m128i r0 = _mm_loadu_si128((const __m128i *) (k0 + qoff));
        __m128i r1 = _mm_loadu_si128((const __m128i *) (k1 + qoff));
        __m256i n0 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(r0, 4), m4), _mm_and_si128(r0, m4));
        __m256i n1 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(r1, 4), m4), _mm_and_si128(r1, m4));
        __m256i p0 = F(dp)(_mm256_setr_epi32(-8 * xs[b], 0, 0, 0, 0, 0, 0, 0), n0,
                           _mm256_loadu_si256((const __m256i *) (x + b * QK)));
        __m256i p1 = F(dp)(_mm256_setr_epi32(-8 * xs[b + 1], 0, 0, 0, 0, 0, 0, 0), n1,
                           _mm256_loadu_si256((const __m256i *) (x + (b + 1) * QK)));
        float s0 = (sc ? ld_f32(sc + 4 * b) : F(f16)(k0)) * xd[b];
        float s1 = (sc ? ld_f32(sc + 4 * (b + 1)) : F(f16)(k1)) * xd[b + 1];
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(s0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(s1), acc1);
    }
    for (; b < nb; b++) {
        const uint8_t *k0 = blk0 + b * stride;
        __m128i r0 = _mm_loadu_si128((const __m128i *) (k0 + qoff));
        __m256i n0 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(r0, 4), m4), _mm_and_si128(r0, m4));
        __m256i p0 = F(dp)(_mm256_setr_epi32(-8 * xs[b], 0, 0, 0, 0, 0, 0, 0), n0,
                           _mm256_loadu_si256((const __m256i *) (x + b * QK)));
        float s0 = (sc ? ld_f32(sc + 4 * b) : F(f16)(k0)) * xd[b];
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(s0), acc0);
    }
    return F(hsum)(_mm256_add_ps(acc0, acc1));
}

/* Q8_0 / Q8_0G: 32 signed weights per block against 32 signed activations. */
TGT static inline float F(dot_q8_0)(const uint8_t *blk0, size_t stride, size_t qoff, const uint8_t *sc, int nb,
                                      const int8_t *x, const float *xd) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    const __m256i z = _mm256_setzero_si256();
    int b = 0;
    for (; b + 2 <= nb; b += 2) {
        const uint8_t *k0 = blk0 + b * stride;
        const uint8_t *k1 = k0 + stride;
        __m256i p0 = F(dp_ss)(z, _mm256_loadu_si256((const __m256i *) (k0 + qoff)),
                              _mm256_loadu_si256((const __m256i *) (x + b * QK)));
        __m256i p1 = F(dp_ss)(z, _mm256_loadu_si256((const __m256i *) (k1 + qoff)),
                              _mm256_loadu_si256((const __m256i *) (x + (b + 1) * QK)));
        float s0 = (sc ? ld_f32(sc + 4 * b) : F(f16)(k0)) * xd[b];
        float s1 = (sc ? ld_f32(sc + 4 * (b + 1)) : F(f16)(k1)) * xd[b + 1];
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(s0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(s1), acc1);
    }
    for (; b < nb; b++) {
        const uint8_t *k0 = blk0 + b * stride;
        __m256i p0 = F(dp_ss)(z, _mm256_loadu_si256((const __m256i *) (k0 + qoff)),
                              _mm256_loadu_si256((const __m256i *) (x + b * QK)));
        float s0 = (sc ? ld_f32(sc + 4 * b) : F(f16)(k0)) * xd[b];
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(s0), acc0);
    }
    return F(hsum)(_mm256_add_ps(acc0, acc1));
}

/*
 * Q4_K: per super-block a scalar prologue (k4_prologue: the 8 combined sub-block scales and the min term), then 4
 * groups of 32 bytes; the low nibbles are sub-block 2g, the high nibbles sub-block 2g + 1, both unsigned 0..15 and
 * in natural order against activation blocks 2g and 2g + 1.
 */
TGT static inline float F(dot_q4_k)(const uint8_t *q, int nsb, const int8_t *x, const float *xd, const int32_t *xs) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i z = _mm256_setzero_si256();
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    float minus = 0, ws[8];
    for (int sb = 0; sb < nsb; sb++, q += Q4_K_BYTES) {
        int blk = sb * 8;
        minus = k4_prologue(q, xd + blk, xs + blk, ws, minus);
        const uint8_t *qs = q + 16;
        const int8_t *xb = x + blk * QK;
        for (int g = 0; g < 4; g++) {
            __m256i raw = _mm256_loadu_si256((const __m256i *) (qs + 32 * g));
            __m256i lo = _mm256_and_si256(raw, m4);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), m4);
            __m256i p0 = F(dp)(z, lo, _mm256_loadu_si256((const __m256i *) (xb + 64 * g)));
            __m256i p1 = F(dp)(z, hi, _mm256_loadu_si256((const __m256i *) (xb + 64 * g + 32)));
            acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(ws[2 * g]), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(ws[2 * g + 1]), acc1);
        }
    }
    return F(hsum)(_mm256_add_ps(acc0, acc1)) - minus;
}

/* Q5_K: Q4_K plus a fifth bit per weight, bit 2g (low sub-block) / 2g + 1 (high) of qh[l], moved to bit 4. */
TGT static inline float F(dot_q5_k)(const uint8_t *q, int nsb, const int8_t *x, const float *xd, const int32_t *xs) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i one = _mm256_set1_epi8(1);
    const __m256i z = _mm256_setzero_si256();
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    float minus = 0, ws[8];
    for (int sb = 0; sb < nsb; sb++, q += Q5_K_BYTES) {
        int blk = sb * 8;
        minus = k4_prologue(q, xd + blk, xs + blk, ws, minus);
        const uint8_t *qs = q + 48;
        const int8_t *xb = x + blk * QK;
        /* 16-bit shifts move bits across the two bytes of a lane, but only into bits the mask with 1 drops */
        __m256i hb = _mm256_loadu_si256((const __m256i *) (q + 16));
        for (int g = 0; g < 4; g++) {
            __m256i raw = _mm256_loadu_si256((const __m256i *) (qs + 32 * g));
            __m256i lo = _mm256_or_si256(_mm256_and_si256(raw, m4),
                                         _mm256_slli_epi16(_mm256_and_si256(hb, one), 4));
            __m256i hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(raw, 4), m4),
                                         _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 1), one), 4));
            hb = _mm256_srli_epi16(hb, 2);
            __m256i p0 = F(dp)(z, lo, _mm256_loadu_si256((const __m256i *) (xb + 64 * g)));
            __m256i p1 = F(dp)(z, hi, _mm256_loadu_si256((const __m256i *) (xb + 64 * g + 32)));
            acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(ws[2 * g]), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(ws[2 * g + 1]), acc1);
        }
    }
    return F(hsum)(_mm256_add_ps(acc0, acc1)) - minus;
}

/*
 * Q6_K: per 128-weight half, group g (32 weights, activation block 4 * half + g) is the low (g < 2) or high nibbles
 * of ql[32 * (g & 1) ..] with bits 2g, 2g + 1 of qh on top, minus 32: signed codes -32..31. Each 16-weight half of a
 * group has its own scale, i.e. int32 lanes 0..3 and 4..7 of the product.
 */
TGT static inline float F(dot_q6_k)(const uint8_t *q, int nsb, const int8_t *x, const float *xd) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i c32 = _mm256_set1_epi8(32);
    const __m256i z = _mm256_setzero_si256();
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    float ws[16];
    for (int sb = 0; sb < nsb; sb++, q += Q6_K_BYTES) {
        int blk0 = sb * 8;
        q6k_prologue(q, xd + blk0, ws);
        for (int half = 0; half < 2; half++) {
            const uint8_t *ql = q + 64 * half;
            __m256i l0 = _mm256_loadu_si256((const __m256i *) ql);
            __m256i l1 = _mm256_loadu_si256((const __m256i *) (ql + 32));
            __m256i hb = _mm256_loadu_si256((const __m256i *) (q + 128 + 32 * half));
            const int8_t *xb = x + (blk0 + 4 * half) * QK;
            const float *wh = ws + 8 * half;
            __m256i w0 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l0, m4),
                                         _mm256_slli_epi16(_mm256_and_si256(hb, m3), 4)), c32);
            __m256i w1 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l1, m4),
                                         _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 2), m3), 4)), c32);
            __m256i w2 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4),
                                         _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 4), m3), 4)), c32);
            __m256i w3 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4),
                                         _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 6), m3), 4)), c32);
            __m256i p0 = F(dp_ss)(z, w0, _mm256_loadu_si256((const __m256i *) xb));
            __m256i p1 = F(dp_ss)(z, w1, _mm256_loadu_si256((const __m256i *) (xb + 32)));
            __m256i p2 = F(dp_ss)(z, w2, _mm256_loadu_si256((const __m256i *) (xb + 64)));
            __m256i p3 = F(dp_ss)(z, w3, _mm256_loadu_si256((const __m256i *) (xb + 96)));
            acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0),
                                   _mm256_set_m128(_mm_set1_ps(wh[1]), _mm_set1_ps(wh[0])), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1),
                                   _mm256_set_m128(_mm_set1_ps(wh[3]), _mm_set1_ps(wh[2])), acc1);
            acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p2),
                                   _mm256_set_m128(_mm_set1_ps(wh[5]), _mm_set1_ps(wh[4])), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p3),
                                   _mm256_set_m128(_mm_set1_ps(wh[7]), _mm_set1_ps(wh[6])), acc1);
        }
    }
    return F(hsum)(_mm256_add_ps(acc0, acc1));
}

#undef F
