package org.robomiri.jllm.audio;

/**
 * A complex FFT of any length whose prime factors are small (Whisper's n_fft is 400 = 2^4 * 5^2; the WavTokenizer
 * codec uses 1280 = 2^8 * 5; the generic fallback is a plain DFT for a large prime factor). Decimation in time, mixed
 * radix, out of place, in double precision. Thread-safe once built: {@link #forward} only reads the plan.
 */
public final class Fft {

    private final int n;
    private final int[] factors;
    private final double[] cos;
    private final double[] sin;

    public Fft(int n) {
        if (n < 1) {
            throw new IllegalArgumentException("fft size " + n);
        }
        this.n = n;
        int[] f = new int[32];
        int nf = 0;
        int rest = n;
        for (int p : new int[]{4, 2, 3, 5}) { // radix 4 first: fewer passes for powers of two
            while (rest % p == 0) {
                f[nf++] = p;
                rest /= p;
            }
        }
        for (int p = 7; rest > 1; p += 2) {
            while (rest % p == 0) {
                f[nf++] = p;
                rest /= p;
            }
            if ((long) p * p > rest && rest > 1) {
                f[nf++] = rest;
                rest = 1;
            }
        }
        this.factors = java.util.Arrays.copyOf(f, nf);
        cos = new double[n];
        sin = new double[n];
        for (int i = 0; i < n; i++) {
            double a = -2 * Math.PI * i / n;
            cos[i] = Math.cos(a);
            sin[i] = Math.sin(a);
        }
    }

    public int size() {
        return n;
    }

    /** The scratch space {@link #forward} needs: two arrays of n doubles, per calling thread. */
    public double[][] scratch() {
        return new double[][]{new double[n], new double[n]};
    }

    /** Forward transform of (xr, xi) into (yr, yi), all of length n; {@code x} is left untouched. */
    public void forward(double[] xr, double[] xi, double[] yr, double[] yi) {
        rec(xr, xi, 0, 1, yr, yi, 0, n, 0);
    }

    private void rec(double[] xr, double[] xi, int xo, int stride, double[] yr, double[] yi, int yo, int len, int fi) {
        if (len == 1) {
            yr[yo] = xr[xo];
            yi[yo] = xi[xo];
            return;
        }
        int p = factors[fi];
        int m = len / p;
        for (int r = 0; r < p; r++) {
            rec(xr, xi, xo + r * stride, stride * p, yr, yi, yo + r * m, m, fi + 1);
        }
        int tstep = n / len;           // W_len^j = W_n^(j * tstep)
        double[] tr = new double[p];
        double[] ti = new double[p];
        for (int k = 0; k < m; k++) {
            for (int r = 0; r < p; r++) {      // twiddle the r-th sub-result: W_len^(r k)
                int idx = (r * k * tstep) % n;
                double a = yr[yo + r * m + k];
                double b = yi[yo + r * m + k];
                tr[r] = a * cos[idx] - b * sin[idx];
                ti[r] = a * sin[idx] + b * cos[idx];
            }
            for (int q = 0; q < p; q++) {      // p-point DFT: Y[k + q m] = sum_r W_p^(r q) t[r]
                double sr = tr[0];
                double si = ti[0];
                int pstep = (n / p);
                for (int r = 1; r < p; r++) {
                    int idx = (int) (((long) r * q % p) * pstep);
                    sr += tr[r] * cos[idx] - ti[r] * sin[idx];
                    si += tr[r] * sin[idx] + ti[r] * cos[idx];
                }
                yr[yo + k + q * m] = sr;      // safe: the sub-results were copied to tr/ti before any of them is overwritten
                yi[yo + k + q * m] = si;
            }
        }
    }
}
