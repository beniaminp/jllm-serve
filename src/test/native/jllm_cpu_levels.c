/*
 * Every SIMD level of libjllm_cpu against its scalar level, in C (no JVM): for checking the kernels of another
 * architecture under emulation, where NativeCpuCheck's JVM would be too slow. The scalar level itself is checked
 * against the Java kernels by NativeCpuCheck on any machine, so passing here carries that over.
 *
 *   cc -O2 -ffp-contract=off -Isrc/main/native -o /tmp/levels src/test/native/jllm_cpu_levels.c -lm && /tmp/levels
 *   aarch64-linux-gnu-gcc -O2 -ffp-contract=off -static -Isrc/main/native -o /tmp/levels-arm64 \
 *       src/test/native/jllm_cpu_levels.c -lm && qemu-aarch64 /tmp/levels-arm64
 *
 * Unit scales (every intermediate value an integer below 2^24): results must be equal bit for bit. Random scales:
 * within 1e-5 of the sum of the absolute block contributions, computed in double here.
 */
#include "jllm_cpu.c"

#include <stdio.h>
#include <stdlib.h>

static uint32_t rng = 12345;

static uint32_t next(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static uint16_t to_f16(float f) { /* positive normal values only */
    uint32_t b;
    memcpy(&b, &f, 4);
    return (uint16_t) ((((b >> 23) & 0xff) - 112) << 10 | ((b >> 13) & 0x3ff));
}

static const int BLOCK_BYTES[T_COUNT] = {16, 32, Q4_0G_BYTES, Q8_0G_BYTES, Q4_K_BYTES, Q5_K_BYTES, Q6_K_BYTES};
static const char *TYPE_NAMES[T_COUNT] = {"Q4_0", "Q8_0", "Q4_0G", "Q8_0G", "Q4_K", "Q5_K", "Q6_K"};

static void put16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t) v;
    p[1] = (uint8_t) (v >> 8);
}

/* rows x cols of type t: random quants; unit: all scales 1 (sub-block scales 1 unless subs) */
static uint8_t *make(int t, int rows, int cols, int unit, int subs, int64_t *soff) {
    int k = t >= T_Q4_K;
    int nb = cols / (k ? QK_K : QK);
    size_t qbytes = (size_t) rows * nb * BLOCK_BYTES[t];
    size_t total = qbytes + (t <= T_Q8_0 ? (size_t) rows * nb * 4 : 0);
    uint8_t *w = malloc(total + 64);
    for (size_t i = 0; i < qbytes; i++) {
        w[i] = (uint8_t) next();
    }
    *soff = (int64_t) qbytes;
    for (size_t blk = 0; blk < (size_t) rows * nb; blk++) {
        uint8_t *b = w + blk * BLOCK_BYTES[t];
        float d = unit ? 1.0f : 0.001f + 0.05f * (float) (next() % 1000) / 1000.0f;
        float dmin = unit ? 1.0f : 0.001f + 0.02f * (float) (next() % 1000) / 1000.0f;
        switch (t) {
            case T_Q4_0:
            case T_Q8_0:
                memcpy(w + qbytes + 4 * blk, &d, 4);
                break;
            case T_Q4_0G:
            case T_Q8_0G:
                put16(b, to_f16(d));
                break;
            case T_Q4_K:
            case T_Q5_K:
                put16(b, to_f16(d));
                put16(b + 2, to_f16(dmin));
                if (unit && !subs) {
                    for (int j = 0; j < 8; j++) b[4 + j] = 1;
                    for (int j = 8; j < 12; j++) b[4 + j] = 0x11;
                }
                break;
            default:
                put16(b + 208, to_f16(d));
                if (unit && !subs) {
                    for (int j = 0; j < 16; j++) b[192 + j] = 1;
                }
        }
    }
    return w;
}

static int failures, checks;

static void run(int t, int rows, int cols, int unit, int small) {
    int64_t soff;
    uint8_t *w = make(t, rows, cols, unit, small, &soff);
    int nb = cols / QK;
    int8_t *xq = malloc(cols);
    float *xd = malloc(nb * 4);
    int32_t *xs = malloc(nb * 4);
    if (unit) {
        int lim = small ? 7 : 127;
        for (int i = 0; i < cols; i++) xq[i] = (int8_t) ((int) (next() % (2 * lim + 1)) - lim);
        for (int b = 0; b < nb; b++) {
            xd[b] = 1;
            xs[b] = 0;
            for (int j = 0; j < QK; j++) xs[b] += xq[b * QK + j];
        }
    } else {
        float *x = malloc(cols * 4);
        for (int i = 0; i < cols; i++) x[i] = (float) ((int) (next() % 2001) - 1000) / 300.0f;
        jllm_cpu_quantize(x, cols, xq, xd, xs);
        free(x);
    }
    float ref[64], got[64];
    jllm_cpu_set_level(L_SCALAR);
    jllm_cpu_dot_rows(t, w, soff, 0, rows, cols, xq, xd, xs, ref, 0);
    for (int level = 1; level < L_COUNT; level++) {
        if (jllm_cpu_set_level(level) != level) continue;
        jllm_cpu_dot_rows(t, w, soff, 0, rows, cols, xq, xd, xs, got, 0);
        for (int r = 0; r < rows; r++) {
            checks++;
            int ok = unit ? memcmp(&ref[r], &got[r], 4) == 0
                          : fabs((double) ref[r] - got[r]) <= 1e-5 * (fabs(ref[r]) + 1.0) * sqrt((double) cols);
            if (!ok) {
                failures++;
                printf("FAIL %s %s %dx%d %s row %d: %.9g vs scalar %.9g\n", LEVEL_NAMES[level], TYPE_NAMES[t], rows,
                       cols, unit ? (small ? "unit-small" : "unit") : "random", r, got[r], ref[r]);
            }
        }
    }
    free(w);
    free(xq);
    free(xd);
    free(xs);
}

int main(void) {
    init();
    printf("detected level: %s\n", LEVEL_NAMES[detected]);
    for (int t = 0; t < T_COUNT; t++) {
        int k = t >= T_Q4_K;
        int cs[] = {k ? 256 : 32, k ? 256 : 96, k ? 512 : 256, k ? 2048 : 1056};
        for (int c = 0; c < 4; c++) {
            for (int rows = 1; rows <= 13; rows += 6) {
                if (cs[c] <= 256) {
                    run(t, rows, cs[c], 1, 0);
                    run(t, rows, cs[c], 1, 1);
                }
                run(t, rows, cs[c], 0, 0);
            }
        }
    }
    int levels = 0;
    for (int level = 1; level < L_COUNT; level++) {
        if (supported & (1u << level)) {
            printf("tested level %s\n", LEVEL_NAMES[level]);
            levels++;
        }
    }
    printf("jllm_cpu levels vs scalar: %d checks, %d failed (%d SIMD levels)\n", checks, failures, levels);
    return failures != 0 || levels == 0;
}
