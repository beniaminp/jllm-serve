package org.robomiri.jllm.model;

/**
 * The scalar math that is specific to Gemma 2 / 3, in one place so the CPU backend, the Metal kernels' reference
 * and the unit tests agree on the formulas (see transformers' modeling_gemma2.py / modeling_gemma3.py).
 */
public final class GemmaMath {

    private GemmaMath() {
    }

    /** {@code gelu_pytorch_tanh}: {@code 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))}. */
    public static float geluTanh(float x) {
        float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1f + (float) Math.tanh(u));
    }

    /** Logit soft-capping {@code tanh(x / cap) * cap}; {@code cap <= 0} means off. */
    public static float softcap(float x, float cap) {
        return cap > 0 ? (float) Math.tanh(x / cap) * cap : x;
    }

    /**
     * The gain Gemma's RMSNorm multiplies with: the checkpoint stores {@code w} and the layer computes
     * {@code x * rsqrt(mean(x^2) + eps) * (1 + w)} in float32. The converter stores {@code 1 + w} so the shared
     * norm kernels ({@code x * rsqrt(..) * gain}) apply unchanged.
     */
    public static float[] foldNormGain(float[] w) {
        float[] g = new float[w.length];
        for (int i = 0; i < w.length; i++) {
            g[i] = 1.0f + w[i];
        }
        return g;
    }

    /** Reference RMSNorm with gain: {@code x * rsqrt(mean(x^2) + eps) * gain}. */
    public static float[] rmsnorm(float[] x, float[] gain, float eps) {
        double ss = 0;
        for (float v : x) {
            ss += (double) v * v;
        }
        float scale = (float) (1.0 / Math.sqrt(ss / x.length + eps));
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i] * scale * gain[i];
        }
        return out;
    }
}
