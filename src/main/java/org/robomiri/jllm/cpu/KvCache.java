package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelConfig;

import java.util.Arrays;

/**
 * The CPU backend's KV cache: per layer one K and one V matrix of {@code [ctx, kvDim]}, row-major, on the Java
 * heap, and the causal grouped-query attention over it. {@link F32} holds plain floats that the attention loop
 * reads in place. {@link F16} holds IEEE half floats in {@code short[]} (converted with
 * {@link Float#floatToFloat16} / {@link Float#float16ToFloat}, both JIT intrinsics) and expands each cached row
 * into thread-local scratch before the f32 SIMD kernels read it. The math is f32 in both cases; f16 only changes
 * what is stored between forward calls.
 *
 * <p><b>Tiling.</b> For {@code n > 1} the work is split into tiles of one kv head and up to {@link #tile}
 * consecutive tokens, which serve {@code tile × group} query heads. Each cached K row is read (f32) or expanded
 * (f16) once per tile and dotted with all of the tile's query heads; the scores of every query head are kept for
 * all its keys, the softmax runs per head, and each V row is read / expanded once per tile and added to every
 * head's output. Per output element nothing changes against the per-(token, head) loop: the same {@code dot}
 * over {@code headDim}, max and sum over the keys in ascending order, and the same ascending {@code axpy}
 * accumulation, so the results are bit-identical; only the memory traffic drops by {@code tile × group}.
 * Decode ({@code n == 1}) on f32 keeps one task per query head (more tasks than kv heads); f16 decode is the
 * one-token tile, which converts each row once per kv head instead of once per query head.
 */
public abstract class KvCache {

    final int layers;
    final int ctx;
    final int kvDim;
    final int kvHeads;
    final int headDim;
    final int qDim;
    /** Query heads per kv head. */
    final int group;
    /** Tokens per prefill tile: as many as keep the tile's {@code group × ctx} score rows within 32 K floats, 1..8. */
    final int tile;
    /** Per thread: [tile × group × ctx scores][headDim K row][headDim V row][tile × group 1/sum]. */
    final ThreadLocal<float[]> scratch;
    private final int kOff;
    private final int vOff;
    private final int invOff;

    KvCache(ModelConfig c, int ctx) {
        this.layers = c.layers();
        this.ctx = ctx;
        this.kvDim = c.kvDim();
        this.kvHeads = c.kvHeads();
        this.headDim = c.headDim();
        this.qDim = c.qDim();
        this.group = c.heads() / c.kvHeads();
        this.tile = Math.max(1, Math.min(8, 32768 / (group * ctx)));
        kOff = tile * group * ctx;
        vOff = kOff + headDim;
        invOff = vOff + headDim;
        int len = invOff + tile * group;
        scratch = ThreadLocal.withInitial(() -> new float[len]);
    }

    public static KvCache create(KvDtype dtype, ModelConfig c, int ctx) {
        long cacheLen = (long) ctx * c.kvDim();
        if (cacheLen > Integer.MAX_VALUE - 16) {
            throw new IllegalArgumentException("context " + ctx + " is too large for the CPU KV cache");
        }
        return switch (dtype) {
            case F32 -> new F32(c, ctx);
            case F16 -> new F16(c, ctx);
        };
    }

    /**
     * Stores the {@code n} rows of {@code k} and {@code v} (each {@code [n, kvDim]} row-major, f32, already
     * normalized/rotated) at positions {@code [pos, pos + n)} of layer {@code layer}.
     */
    public abstract void store(int layer, int pos, float[] k, float[] v, int n);

    /**
     * Causal grouped-query attention for the tokens at positions {@code [pos, pos + n)}: reads the queries
     * {@code q[t * qDim + h * headDim ..]}, writes {@code att} in the same layout.
     *
     * @param window  sliding-window length in tokens, 0 = full causal attention
     * @param scale   multiplier of q.k
     * @param softcap {@code tanh(s / cap) * cap} on the scaled score, 0 = off
     */
    public abstract void attention(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window,
                                   float scale, float softcap);

    /** The f32 K (or V) rows of {@code layer} when they can be read in place, null when they must be expanded. */
    abstract float[] rowsInPlace(int layer, boolean values);

    /** Expands the {@code headDim} values of kv head slice {@code ko} of row {@code s} into {@code dst[d ..]}. */
    abstract void expand(int layer, boolean values, int s, int ko, float[] dst, int d);

    private static final boolean FASTMATH = VectorKernels.FASTMATH;

    /**
     * Softmax numerator of the {@code n} scaled scores at {@code sc[o ..]} in place, returning their sum: the
     * max in ascending order, {@code exp(s - max)} and the sum in ascending order. Fast math
     * ({@link VectorKernels#FASTMATH}) applies the soft-cap here on lanes (the score pass left it out) and uses the
     * polynomial exp; otherwise the scalar {@code Math.exp} of the original loop.
     */
    static float softmax(float[] sc, int o, int n, float softcap) {
        if (softcap > 0 && FASTMATH) {
            VectorKernels.softcapFast(sc, o, n, softcap);
        }
        float max = Float.NEGATIVE_INFINITY;
        for (int i = o; i < o + n; i++) {
            max = Math.max(max, sc[i]);
        }
        return VectorKernels.softmaxExp(sc, o, n, max);
    }

    /** The tiled kernel described in the class comment, for every (kv head, token tile) pair of the chunk. */
    final void attentionTiled(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window,
                              float scale, float softcap) {
        int hd = headDim;
        int tiles = (n + tile - 1) / tile;
        float[] kIn = rowsInPlace(layer, false);
        float[] vIn = rowsInPlace(layer, true);
        // later tiles see more keys: hand them out first so the tail of the job is made of the light ones
        pool.forRangeFine(kvHeads * tiles, 1, (i0, i1) -> {
            float[] sc = scratch.get();
            float[] kArr = kIn != null ? kIn : sc;
            float[] vArr = vIn != null ? vIn : sc;
            for (int i = i0; i < i1; i++) {
                int g = i % kvHeads;
                int t0 = (tiles - 1 - i / kvHeads) * tile;
                int t1 = Math.min(n, t0 + tile);
                int ko = g * hd;
                int sMin = window > 0 ? Math.max(0, pos + t0 + 1 - window) : 0;
                int sMax = pos + t1;
                // scores: row (t - t0) * group + h holds the keys of query head g * group + h of token t
                for (int s = sMin; s < sMax; s++) {
                    int koff;
                    if (kIn != null) {
                        koff = s * kvDim + ko;
                    } else {
                        expand(layer, false, s, ko, sc, kOff);
                        koff = kOff;
                    }
                    for (int t = t0; t < t1; t++) {
                        int len = pos + t + 1;
                        int first = window > 0 ? Math.max(0, len - window) : 0;
                        if (s < first || s >= len) {
                            continue;
                        }
                        int qo = t * qDim + g * group * hd;
                        int row = (t - t0) * group;
                        for (int h = 0; h < group; h++) {
                            float d = VectorKernels.dot(q, qo + h * hd, kArr, koff, hd) * scale;
                            sc[(row + h) * ctx + s] = softcap > 0 && !FASTMATH ? (float) Math.tanh(d / softcap) * softcap : d;
                        }
                    }
                }
                for (int t = t0; t < t1; t++) {
                    int len = pos + t + 1;
                    int first = window > 0 ? Math.max(0, len - window) : 0;
                    int qo = t * qDim + g * group * hd;
                    for (int h = 0; h < group; h++) {
                        int row = (t - t0) * group + h;
                        int so = row * ctx;
                        sc[invOff + row] = 1f / softmax(sc, so + first, len - first, softcap);
                        Arrays.fill(att, qo + h * hd, qo + (h + 1) * hd, 0f);
                    }
                }
                for (int s = sMin; s < sMax; s++) {
                    int voff;
                    if (vIn != null) {
                        voff = s * kvDim + ko;
                    } else {
                        expand(layer, true, s, ko, sc, vOff);
                        voff = vOff;
                    }
                    for (int t = t0; t < t1; t++) {
                        int len = pos + t + 1;
                        int first = window > 0 ? Math.max(0, len - window) : 0;
                        if (s < first || s >= len) {
                            continue;
                        }
                        int qo = t * qDim + g * group * hd;
                        int row = (t - t0) * group;
                        for (int h = 0; h < group; h++) {
                            VectorKernels.axpy(sc[(row + h) * ctx + s] * sc[invOff + row + h], vArr, voff, att, qo + h * hd, hd);
                        }
                    }
                }
            }
        });
    }

    // ------------------------------------------------------------------ f32

    static final class F32 extends KvCache {
        final float[][] key;
        final float[][] val;

        F32(ModelConfig c, int ctx) {
            super(c, ctx);
            key = new float[layers][ctx * kvDim];
            val = new float[layers][ctx * kvDim];
        }

        @Override
        public void store(int layer, int pos, float[] k, float[] v, int n) {
            System.arraycopy(k, 0, key[layer], pos * kvDim, n * kvDim);
            System.arraycopy(v, 0, val[layer], pos * kvDim, n * kvDim);
        }

        @Override
        float[] rowsInPlace(int layer, boolean values) {
            return values ? val[layer] : key[layer];
        }

        @Override
        void expand(int layer, boolean values, int s, int ko, float[] dst, int d) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void attention(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window, float scale,
                       float softcap) {
            if (n > 1) {
                attentionTiled(pool, layer, n, pos, q, att, window, scale, softcap);
                return;
            }
            // decode: one task per query head, the finest split there is for a single token
            float[] kc = key[layer];
            float[] vc = val[layer];
            int hd = headDim;
            int nh = kvHeads * group;
            int len = pos + 1;
            int first = window > 0 ? Math.max(0, len - window) : 0;
            pool.forRangeFine(nh, 1, (h0, h1) -> {
                float[] sc = scratch.get();
                for (int h = h0; h < h1; h++) {
                    int ko = (h / group) * hd;
                    int qo = h * hd;
                    for (int s = first; s < len; s++) {
                        float d = VectorKernels.dot(q, qo, kc, s * kvDim + ko, hd) * scale;
                        sc[s] = softcap > 0 && !FASTMATH ? (float) Math.tanh(d / softcap) * softcap : d;
                    }
                    float inv = 1f / softmax(sc, first, len - first, softcap);
                    Arrays.fill(att, qo, qo + hd, 0f);
                    for (int s = first; s < len; s++) {
                        VectorKernels.axpy(sc[s] * inv, vc, s * kvDim + ko, att, qo, hd);
                    }
                }
            });
        }
    }

    // ------------------------------------------------------------------ f16

    static final class F16 extends KvCache {
        final short[][] key;
        final short[][] val;

        F16(ModelConfig c, int ctx) {
            super(c, ctx);
            key = new short[layers][ctx * kvDim];
            val = new short[layers][ctx * kvDim];
        }

        @Override
        public void store(int layer, int pos, float[] k, float[] v, int n) {
            toHalf(k, 0, key[layer], pos * kvDim, n * kvDim);
            toHalf(v, 0, val[layer], pos * kvDim, n * kvDim);
        }

        @Override
        float[] rowsInPlace(int layer, boolean values) {
            return null;
        }

        @Override
        void expand(int layer, boolean values, int s, int ko, float[] dst, int d) {
            toFloat(values ? val[layer] : key[layer], s * kvDim + ko, dst, d, headDim);
        }

        @Override
        public void attention(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window, float scale,
                       float softcap) {
            attentionTiled(pool, layer, n, pos, q, att, window, scale, softcap);
        }

        static void toHalf(float[] src, int so, short[] dst, int d, int n) {
            for (int i = 0; i < n; i++) {
                dst[d + i] = Float.floatToFloat16(src[so + i]);
            }
        }

        static void toFloat(short[] src, int so, float[] dst, int d, int n) {
            for (int i = 0; i < n; i++) {
                dst[d + i] = Float.float16ToFloat(src[so + i]);
            }
        }
    }
}
