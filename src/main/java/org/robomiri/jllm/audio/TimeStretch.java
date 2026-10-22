package org.robomiri.jllm.audio;

/**
 * Pitch-preserving time stretching by waveform-similarity overlap-add (WSOLA). The output is built from 20 ms
 * Hann-windowed frames placed every 10 ms; frame k is cut from the input near position {@code k * 10 ms * speed}, at
 * the offset (within +-5 ms) whose first half best continues the samples that would naturally have followed the
 * previous frame (normalised cross-correlation), so periodic structure such as voiced speech stays in phase.
 */
public final class TimeStretch {

    private TimeStretch() {
    }

    /** @param speed 0.25 .. 4: 2 plays twice as fast (half as long) */
    public static float[] stretch(float[] in, int rate, double speed) {
        if (Math.abs(speed - 1.0) < 1e-3 || in.length < 4) {
            return in;
        }
        int hs = Math.max(8, rate / 100);          // synthesis hop (10 ms)
        int n = 2 * hs;                            // frame length (20 ms)
        int tol = hs / 2;                          // search range
        int outLen = (int) Math.round(in.length / speed);
        float[] out = new float[outLen + n];
        float[] win = new float[n];
        for (int i = 0; i < n; i++) {
            win[i] = (float) (0.5 - 0.5 * Math.cos(2 * Math.PI * (i + 0.5) / n));
        }
        int frames = (outLen + hs - 1) / hs;
        int prev = 0;                              // input position of the previous frame
        for (int k = 0; k < frames; k++) {
            int target = (int) Math.round(k * hs * speed);
            int pos;
            if (k == 0) {
                pos = 0;
            } else {
                int natural = prev + hs;           // what followed the previous frame in the input
                int best = Math.max(0, Math.min(in.length - 1, target));
                double bestScore = Double.NEGATIVE_INFINITY;
                for (int d = -tol; d <= tol; d++) {
                    int c = target + d;
                    if (c < 0 || c + hs > in.length || natural + hs > in.length) {
                        continue;
                    }
                    double dot = 0;
                    double e = 1e-9;
                    for (int i = 0; i < hs; i++) {
                        float a = in[c + i];
                        float b = in[natural + i];
                        dot += a * b;
                        e += a * a;
                    }
                    double score = dot / Math.sqrt(e);
                    if (score > bestScore) {
                        bestScore = score;
                        best = c;
                    }
                }
                pos = best;
            }
            int o = k * hs;
            for (int i = 0; i < n; i++) {
                int src = pos + i;
                if (src < in.length && o + i < out.length) {
                    out[o + i] += win[i] * in[src];
                }
            }
            prev = pos;
        }
        float[] res = new float[outLen];
        System.arraycopy(out, 0, res, 0, outLen);
        return res;
    }
}
