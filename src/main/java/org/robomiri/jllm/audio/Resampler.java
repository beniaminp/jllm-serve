package org.robomiri.jllm.audio;

/**
 * Sample-rate conversion by a polyphase windowed-sinc filter (a Kaiser window, beta 8.6, 24 zero crossings per side at
 * the lower of the two rates). The rates are reduced by their gcd to L/M; output sample j is the input convolved with
 * the filter's phase {@code (j * M) mod L} around input position {@code j * M / L}, so each phase is a short
 * pre-computed FIR (no sine evaluated per sample). Downsampling low-passes at 0.95 of the output Nyquist frequency
 * first (the filter's cutoff scales with the ratio), so nothing above it folds back.
 */
public final class Resampler {

    private Resampler() {
    }

    private static final int ZERO_CROSSINGS = 24;
    private static final double BETA = 8.6;
    private static final double CUTOFF = 0.95;

    public static float[] resample(float[] in, int fromRate, int toRate) {
        if (fromRate == toRate || in.length == 0) {
            return in;
        }
        int g = gcd(fromRate, toRate);
        long l = toRate / g;
        long m = fromRate / g;
        boolean exact = l <= 4096;   // an odd ratio such as 44100 -> 16001 uses the nearest of 4096 phases instead
        int phases = exact ? (int) l : 4096;
        int step = (int) m;
        double ratio = (double) toRate / fromRate;
        double cutoff = Math.min(1.0, ratio) * CUTOFF;     // in cycles per input sample * 2 (relative to Nyquist)
        int half = (int) Math.ceil(ZERO_CROSSINGS / cutoff);
        int taps = 2 * half;
        float[][] table = new float[phases][taps];
        double i0 = bessel0(BETA);
        for (int p = 0; p < phases; p++) {
            double frac = (double) p / phases;        // sub-sample position of the output inside the input grid
            double sum = 0;
            double[] t = new double[taps];
            for (int k = 0; k < taps; k++) {
                double x = (k - half + 1) - frac;      // distance from the output instant, in input samples
                double w = Math.abs(x) >= half ? 0 : bessel0(BETA * Math.sqrt(1 - (x / half) * (x / half))) / i0;
                double s = x == 0 ? cutoff : Math.sin(Math.PI * cutoff * x) / (Math.PI * x);
                t[k] = s * w;
                sum += t[k];
            }
            for (int k = 0; k < taps; k++) {
                table[p][k] = (float) (t[k] / sum);    // unity gain at DC
            }
        }
        long outLen = (long) in.length * l / m;
        float[] out = new float[(int) outLen];
        for (int j = 0; j < out.length; j++) {
            int base;
            int ph;
            if (exact) {
                long num = (long) j * step;
                base = (int) (num / phases);
                ph = (int) (num % phases);
            } else {
                double pos = (double) j * m / l;
                base = (int) Math.floor(pos);
                ph = (int) Math.round((pos - base) * phases);
                if (ph == phases) {
                    ph = 0;
                    base++;
                }
            }
            float[] f = table[ph];
            int start = base - half + 1;
            float acc = 0;
            if (start >= 0 && start + taps <= in.length) {
                for (int k = 0; k < taps; k++) {
                    acc += f[k] * in[start + k];
                }
            } else {
                for (int k = 0; k < taps; k++) {
                    int idx = start + k;
                    if (idx >= 0 && idx < in.length) {
                        acc += f[k] * in[idx];
                    }
                }
            }
            out[j] = acc;
        }
        return out;
    }

    private static int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    private static double bessel0(double x) {
        double sum = 1;
        double term = 1;
        double q = x * x / 4;
        for (int k = 1; k < 60; k++) {
            term *= q / ((double) k * k);
            sum += term;
            if (term < 1e-14 * sum) {
                break;
            }
        }
        return sum;
    }
}
