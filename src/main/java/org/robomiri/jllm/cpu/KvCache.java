package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelConfig;

import java.util.Arrays;

/**
 * The CPU backend's KV cache: per layer one K and one V matrix of {@code [ctx, kvDim]}, row-major, on the Java
 * heap. {@link F32} holds plain floats that {@code CpuBackend.attention} reads directly. {@link F16} holds IEEE
 * half floats in {@code short[]} (converted with {@link Float#floatToFloat16} / {@link Float#float16ToFloat},
 * both JIT intrinsics) and brings its own attention loop, which expands each cached row slice once per
 * (token, kv-head) into thread-local scratch and then reuses the f32 SIMD kernels for every query head of the
 * group. The math is f32 in both cases; f16 only changes what is stored between forward calls.
 */
abstract class KvCache {

    final KvDtype dtype;
    final int layers;
    final int ctx;
    final int kvDim;

    KvCache(KvDtype dtype, int layers, int ctx, int kvDim) {
        this.dtype = dtype;
        this.layers = layers;
        this.ctx = ctx;
        this.kvDim = kvDim;
    }

    static KvCache create(KvDtype dtype, ModelConfig c, int ctx) {
        long cacheLen = (long) ctx * c.kvDim();
        if (cacheLen > Integer.MAX_VALUE - 16) {
            throw new IllegalArgumentException("context " + ctx + " is too large for the CPU KV cache");
        }
        return switch (dtype) {
            case F32 -> new F32(c.layers(), ctx, c.kvDim());
            case F16 -> new F16(c, ctx);
        };
    }

    /** Bytes of cache storage (K and V, all layers). */
    long bytes() {
        return 2L * layers * ctx * kvDim * dtype.bytes;
    }

    /**
     * Stores the {@code n} rows of {@code k} and {@code v} (each {@code [n, kvDim]} row-major, f32, already
     * normalized/rotated) at positions {@code [pos, pos + n)} of layer {@code layer}.
     */
    abstract void store(int layer, int pos, float[] k, float[] v, int n);

    // ------------------------------------------------------------------ f32

    static final class F32 extends KvCache {
        final float[][] key;
        final float[][] val;

        F32(int layers, int ctx, int kvDim) {
            super(KvDtype.F32, layers, ctx, kvDim);
            key = new float[layers][ctx * kvDim];
            val = new float[layers][ctx * kvDim];
        }

        @Override
        void store(int layer, int pos, float[] k, float[] v, int n) {
            System.arraycopy(k, 0, key[layer], pos * kvDim, n * kvDim);
            System.arraycopy(v, 0, val[layer], pos * kvDim, n * kvDim);
        }
    }

    // ------------------------------------------------------------------ f16

    static final class F16 extends KvCache {
        final short[][] key;
        final short[][] val;
        private final int kvHeads;
        private final int headDim;
        private final int qDim;
        private final int group;
        /** Per thread: [group × ctx scores][headDim expanded K row][headDim expanded V row][group 1/sum]. */
        private final ThreadLocal<float[]> scratch;

        F16(ModelConfig c, int ctx) {
            super(KvDtype.F16, c.layers(), ctx, c.kvDim());
            key = new short[layers][ctx * kvDim];
            val = new short[layers][ctx * kvDim];
            kvHeads = c.kvHeads();
            headDim = c.headDim();
            qDim = c.qDim();
            group = c.heads() / c.kvHeads();
            int len = group * ctx + 2 * headDim + group;
            scratch = ThreadLocal.withInitial(() -> new float[len]);
        }

        @Override
        void store(int layer, int pos, float[] k, float[] v, int n) {
            toHalf(k, 0, key[layer], pos * kvDim, n * kvDim);
            toHalf(v, 0, val[layer], pos * kvDim, n * kvDim);
        }

        /** Causal grouped-query attention over this cache for tokens at positions [pos, pos+n); same math as f32. */
        void attention(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att) {
            attention(pool, layer, n, pos, q, att, 0);
        }

        /** @param window sliding-window length in tokens, 0 = full causal attention */
        void attention(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window) {
            attention(pool, layer, n, pos, q, att, window, (float) (1.0 / Math.sqrt(headDim)), 0f);
        }

        /** @param scale multiplier of q.k; @param softcap {@code tanh(s/cap)*cap} on the scaled score, 0 = off */
        void attention(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window,
                       float scale, float softcap) {
            short[] kc = key[layer];
            short[] vc = val[layer];
            int hd = headDim;
            int kOff = group * ctx;
            int vOff = kOff + hd;
            int invOff = vOff + hd;
            pool.forRange(n * kvHeads, 1, (i0, i1) -> {
                float[] sc = scratch.get();
                for (int i = i0; i < i1; i++) {
                    int t = i / kvHeads;
                    int g = i % kvHeads;
                    int ko = g * hd;
                    int qo0 = t * qDim + g * group * hd; // first query head of this kv head
                    int len = pos + t + 1;
                    int first = window > 0 ? Math.max(0, len - window) : 0;
                    for (int s = first; s < len; s++) {
                        toFloat(kc, s * kvDim + ko, sc, kOff, hd);
                        for (int h = 0; h < group; h++) {
                            float d = VectorKernels.dot(q, qo0 + h * hd, sc, kOff, hd) * scale;
                            sc[h * ctx + s] = softcap > 0 ? (float) Math.tanh(d / softcap) * softcap : d;
                        }
                    }
                    for (int h = 0; h < group; h++) {
                        int so = h * ctx;
                        float max = Float.NEGATIVE_INFINITY;
                        for (int s = first; s < len; s++) {
                            max = Math.max(max, sc[so + s]);
                        }
                        float sum = 0;
                        for (int s = first; s < len; s++) {
                            float e = (float) Math.exp(sc[so + s] - max);
                            sc[so + s] = e;
                            sum += e;
                        }
                        sc[invOff + h] = 1f / sum;
                        Arrays.fill(att, qo0 + h * hd, qo0 + (h + 1) * hd, 0f);
                    }
                    for (int s = first; s < len; s++) {
                        toFloat(vc, s * kvDim + ko, sc, vOff, hd);
                        for (int h = 0; h < group; h++) {
                            VectorKernels.axpy(sc[h * ctx + s] * sc[invOff + h], sc, vOff, att, qo0 + h * hd, hd);
                        }
                    }
                }
            });
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
