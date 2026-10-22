package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.Attention;

import java.util.Arrays;

/**
 * Accuracy-study simulation of q4 KV variants (docs/07 §7.13, "where the error comes from"; never used unless
 * {@code -Dq4.sim=true}): K and V are rounded through the chosen quantizer and stored dequantized in an f32 cache, so
 * one run can vary what the shipped {@link Q4} fixes. Switches: {@code q4.rotk} / {@code q4.rotv} (normalized
 * Hadamard rotation of K / V), {@code q4.qk} / {@code q4.qv} (quantize K / V at all), {@code q4.bits},
 * {@code q4.group}, {@code q4.symh} (the shipped symmetric mid-rise levels; off: ggml Q4_0's {@code d = m / -8}),
 * {@code q4.asymk} / {@code q4.asymv} (min/max scale and offset), {@code q4.sink} (keep the first positions exact),
 * {@code q4.recent} (a window of exact positions).
 */
public final class KvQ4Sim extends KvCache {
    static final boolean ROT_K = Boolean.parseBoolean(System.getProperty("q4.rotk", "true"));
    static final boolean ROT_V = Boolean.parseBoolean(System.getProperty("q4.rotv", "false"));
    static final boolean ASYM_K = Boolean.parseBoolean(System.getProperty("q4.asymk", "false"));
    static final boolean ASYM_V = Boolean.parseBoolean(System.getProperty("q4.asymv", "false"));
    static final int RECENT = Integer.getInteger("q4.recent", 0);
    static final int GROUP = Integer.getInteger("q4.group", 32);
    static final int BITS = Integer.getInteger("q4.bits", 4);
    static final boolean QK = Boolean.parseBoolean(System.getProperty("q4.qk", "true"));
    static final boolean QV = Boolean.parseBoolean(System.getProperty("q4.qv", "true"));
    static final int SINK = Integer.getInteger("q4.sink", 0);
    static final boolean SYMH = Boolean.getBoolean("q4.symh");
    /** 8 = q8-style per-row (headDim) symmetric int8, for cross-checking. */

    private final KvCache deq;
    private final float[][] exK;
    private final float[][] exV;
    private final float[] rowBuf;

    KvQ4Sim(int layers, Attention.Gqa a, int ctx) {
        super(layers, a, ctx);
        deq = KvCache.create(org.robomiri.jllm.model.KvDtype.F32, layers, a, ctx);
        exK = RECENT > 0 ? new float[layers][ctx * kvDim] : null;
        exV = RECENT > 0 ? new float[layers][ctx * kvDim] : null;
        rowBuf = new float[headDim];
        System.err.printf("KvQ4Sim rotk=%s rotv=%s asymk=%s asymv=%s recent=%d group=%d bits=%d%n", ROT_K, ROT_V, ASYM_K,
                ASYM_V, RECENT, GROUP, BITS);
    }

    static void fwht(float[] x, int o, int n) {
        int b = Integer.lowestOneBit(n);
        for (int bo = o; bo < o + n; bo += b) {
            for (int h = 1; h < b; h <<= 1) {
                for (int i = 0; i < b; i += 2 * h) {
                    for (int j = i; j < i + h; j++) {
                        float u = x[bo + j];
                        float w = x[bo + j + h];
                        x[bo + j] = u + w;
                        x[bo + j + h] = u - w;
                    }
                }
            }
            float s = (float) (1 / Math.sqrt(b));
            for (int i = 0; i < b; i++) {
                x[bo + i] *= s;
            }
        }
    }

    static float h(float x) {
        return Float.float16ToFloat(Float.floatToFloat16(x));
    }

    /** quantize-dequantize groups in place */
    static void qdq(float[] x, int o, int n, boolean asym) {
        int levels = (1 << BITS) - 1;
        for (int g = o; g < o + n; g += GROUP) {
            if (asym) {
                float lo = Float.POSITIVE_INFINITY, hi = Float.NEGATIVE_INFINITY;
                for (int i = g; i < g + GROUP; i++) {
                    lo = Math.min(lo, x[i]);
                    hi = Math.max(hi, x[i]);
                }
                float d = h((hi - lo) / levels);
                float m = h(lo);
                for (int i = g; i < g + GROUP; i++) {
                    float q = d == 0 ? 0 : Math.max(0, Math.min(levels, (float) Math.rint((x[i] - m) / d)));
                    x[i] = Math.fma(q, d, m);
                }
            } else if (SYMH) {
                float a = 0;
                for (int i = g; i < g + GROUP; i++) {
                    a = Math.max(a, Math.abs(x[i]));
                }
                float h = (levels) / 2f; // 7.5
                float d = h(a / h);
                for (int i = g; i < g + GROUP; i++) {
                    float q = d == 0 ? 0 : Math.max(0, Math.min(levels, (float) Math.rint(x[i] / d + h)));
                    x[i] = (q - h) * d;
                }
            } else {
                float m = 0;
                for (int i = g; i < g + GROUP; i++) {
                    if (Math.abs(x[i]) > Math.abs(m)) {
                        m = x[i];
                    }
                }
                int half = 1 << (BITS - 1);
                float d = h(m / -half);
                for (int i = g; i < g + GROUP; i++) {
                    float q = d == 0 ? 0 : Math.max(-half, Math.min(half - 1, (float) Math.rint(x[i] / d)));
                    x[i] = q * d;
                }
            }
        }
    }

    /** stats: [k|v][0 rows, 1 sum max/rms, 2 max max/rms, 3 sum sq err, 4 sum sq] */
    public static final double[][] STATS = new double[2][5];

    static synchronized void stat(int kind, float[] before, float[] after, int o, int n) {
        double ss = 0, mx = 0, err = 0;
        for (int i = o; i < o + n; i++) {
            ss += (double) before[i] * before[i];
            mx = Math.max(mx, Math.abs(before[i]));
            err += (double) (before[i] - after[i]) * (before[i] - after[i]);
        }
        double rms = Math.sqrt(ss / n);
        if (rms == 0) {
            return;
        }
        double[] st = STATS[kind];
        st[0]++;
        st[1] += mx / rms;
        st[2] = Math.max(st[2], mx / rms);
        st[3] += err;
        st[4] += ss;
    }

    public static String stats() {
        StringBuilder b = new StringBuilder();
        for (int k = 0; k < 2; k++) {
            double[] st = STATS[k];
            b.append(String.format("%s: rows %.0f, max/rms mean %.2f max %.2f, rel rms err %.4f%n", k == 0 ? "K" : "V",
                    st[0], st[1] / st[0], st[2], Math.sqrt(st[3] / st[4])));
        }
        return b.toString();
    }

    @Override
    public void store(int layer, int pos, float[] k, float[] v, int n) {
        float[] kq = new float[n * kvDim];
        float[] vq = new float[n * kvDim];
        System.arraycopy(k, 0, kq, 0, n * kvDim);
        System.arraycopy(v, 0, vq, 0, n * kvDim);
        for (int r = 0; r < n * kvHeads; r++) {
            int o = r * headDim;
            if (ROT_K) {
                fwht(kq, o, headDim);
            }
            if (ROT_V) {
                fwht(vq, o, headDim);
            }
        }
        if (RECENT > 0) {
            for (int i = 0; i < n * kvDim; i++) {
                exK[layer][pos * kvDim + i] = h(kq[i]);
                exV[layer][pos * kvDim + i] = h(vq[i]);
            }
        }
        float[] k0 = kq.clone();
        float[] v0 = vq.clone();
        for (int r = 0; r < n * kvHeads; r++) {
            if (pos + r / kvHeads < SINK) {
                continue; // an attention sink position stays exact
            }
            if (QK) {
                qdq(kq, r * headDim, headDim, ASYM_K);
            }
            if (QV) {
                qdq(vq, r * headDim, headDim, ASYM_V);
            }
            stat(0, k0, kq, r * headDim, headDim);
            stat(1, v0, vq, r * headDim, headDim);
        }
        deq.store(layer, pos, kq, vq, n);
    }

    @Override
    public void attention(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window, float scale,
                          float softcap) {
        float[] qr = Arrays.copyOf(q, n * qDim);
        if (ROT_K) {
            for (int r = 0; r < n * qDim / headDim; r++) {
                fwht(qr, r * headDim, headDim);
            }
        }
        if (RECENT == 0) {
            deq.attention(pool, layer, n, pos, qr, att, window, scale, softcap);
        } else {
            simple(pool, layer, n, pos, qr, att, window, scale, softcap);
        }
        if (ROT_V) {
            for (int r = 0; r < n * qDim / headDim; r++) {
                fwht(att, r * headDim, headDim);
            }
        }
    }

    private void simple(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window, float scale,
                        float softcap) {
        int heads = qDim / headDim;
        pool.forRange(n * heads, 1, (i0, i1) -> {
            float[] kr = new float[headDim];
            float[] sc = new float[pos + n];
            for (int i = i0; i < i1; i++) {
                int t = i / heads;
                int hh = i % heads;
                int g = hh / group;
                int P = pos + t;
                int len = P + 1;
                int first = window > 0 ? Math.max(0, len - window) : 0;
                int qo = t * qDim + hh * headDim;
                float max = Float.NEGATIVE_INFINITY;
                for (int s = first; s < len; s++) {
                    float d;
                    if (P - s < RECENT) {
                        d = VectorKernels.dot(q, qo, exK[layer], s * kvDim + g * headDim, headDim);
                    } else {
                        deq.expand(layer, false, s, g * headDim, kr, 0);
                        d = VectorKernels.dot(q, qo, kr, 0, headDim);
                    }
                    d *= scale;
                    if (softcap > 0) {
                        d = (float) Math.tanh(d / softcap) * softcap;
                    }
                    sc[s] = d;
                    max = Math.max(max, d);
                }
                double sum = 0;
                for (int s = first; s < len; s++) {
                    sc[s] = (float) Math.exp(sc[s] - max);
                    sum += sc[s];
                }
                Arrays.fill(att, qo, qo + headDim, 0f);
                for (int s = first; s < len; s++) {
                    float p = (float) (sc[s] / sum);
                    if (P - s < RECENT) {
                        VectorKernels.axpy(p, exV[layer], s * kvDim + g * headDim, att, qo, headDim);
                    } else {
                        deq.expand(layer, true, s, g * headDim, kr, 0);
                        VectorKernels.axpy(p, kr, 0, att, qo, headDim);
                    }
                }
            }
        });
    }

    @Override
    float[] keysInPlace(int layer) {
        return null;
    }

    @Override
    float[] valuesInPlace(int layer) {
        return null;
    }

    @Override
    void expandKeys(int layer, int g, int c0, int c1, float[] dst, int d) {
        throw new UnsupportedOperationException();
    }

    @Override
    void expandValues(int layer, int g, int s0, int s1, int i0, int i1, float[] dst, int d) {
        throw new UnsupportedOperationException();
    }

    @Override
    void expand(int layer, boolean values, int s, int ko, float[] dst, int d) {
        deq.expand(layer, values, s, ko, dst, d);
    }

    @Override
    void copyLayer(KvCache o, int layer, int n, int chunksUsed) {
        throw new UnsupportedOperationException();
    }
}
