package org.robomiri.jllm.audio;

import java.util.stream.IntStream;

/**
 * Whisper's log-mel spectrogram, as {@code WhisperFeatureExtractor} (and OpenAI's {@code log_mel_spectrogram}) compute
 * it: a Hann-windowed STFT (n_fft 400, hop 160, reflect padding of n_fft/2 on both sides, last frame dropped), power
 * spectrum, a Slaney-scale mel filter bank with Slaney area normalisation ({@code mel_filter_bank(norm="slaney",
 * mel_scale="slaney")}, 80 bins, or 128 for large-v3), {@code log10(max(x, 1e-10))}, a floor at {@code max - 8} and
 * {@code (x + 4) / 4}. The FFT and the filters are evaluated in double precision.
 */
public final class LogMel {

    public static final int SAMPLE_RATE = 16000;
    public static final int N_FFT = 400;
    public static final int HOP = 160;
    public static final int CHUNK_SECONDS = 30;
    public static final int N_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS;      // 480000
    public static final int N_FRAMES = N_SAMPLES / HOP;                   // 3000
    private static final int BINS = N_FFT / 2 + 1;                        // 201

    private final int nMels;
    private final double[][] filters;   // [nMels][BINS]
    private final double[] window;
    private final Fft fft = new Fft(N_FFT);

    public LogMel(int nMels) {
        this.nMels = nMels;
        this.filters = melFilters(nMels, BINS, SAMPLE_RATE, 0.0, SAMPLE_RATE / 2.0);
        this.window = new double[N_FFT];
        for (int i = 0; i < N_FFT; i++) {
            window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N_FFT);    // periodic Hann
        }
    }

    public int mels() {
        return nMels;
    }

    private static double hzToMel(double f) {
        double minLogHz = 1000.0;
        double minLogMel = 15.0;
        double logstep = 27.0 / Math.log(6.4);
        return f >= minLogHz ? minLogMel + Math.log(f / minLogHz) * logstep : 3.0 * f / 200.0;
    }

    private static double melToHz(double m) {
        double minLogHz = 1000.0;
        double minLogMel = 15.0;
        double logstep = Math.log(6.4) / 27.0;
        return m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : 200.0 * m / 3.0;
    }

    /** The Slaney mel filter bank, as {@code transformers.audio_utils.mel_filter_bank} builds it; [nMels][bins]. */
    public static double[][] melFilters(int nMels, int bins, int sampleRate, double fMin, double fMax) {
        double melMin = hzToMel(fMin);
        double melMax = hzToMel(fMax);
        double[] filterFreqs = new double[nMels + 2];
        for (int i = 0; i < nMels + 2; i++) {
            filterFreqs[i] = melToHz(melMin + (melMax - melMin) * i / (nMels + 1));
        }
        double[] fftFreqs = new double[bins];
        for (int i = 0; i < bins; i++) {
            fftFreqs[i] = (sampleRate / 2.0) * i / (bins - 1);
        }
        double[][] f = new double[nMels][bins];
        for (int m = 0; m < nMels; m++) {
            double lo = filterFreqs[m];
            double mid = filterFreqs[m + 1];
            double hi = filterFreqs[m + 2];
            double enorm = 2.0 / (hi - lo);
            for (int b = 0; b < bins; b++) {
                double down = -(lo - fftFreqs[b]) / (mid - lo);
                double up = (hi - fftFreqs[b]) / (hi - mid);
                f[m][b] = Math.max(0, Math.min(down, up)) * enorm;
            }
        }
        return f;
    }

    /** The mel filter bank, [nMels][201] (exposed for tests and docs). */
    public double[][] filters() {
        return filters;
    }

    /**
     * Log-mel of {@code audio} (16 kHz mono) followed by {@code padSamples} zeros: [nMels][frames] with
     * {@code frames = (len + pad) / 160}.
     */
    public float[][] compute(float[] audio, int padSamples) {
        int total = audio.length + padSamples;
        int frames = total / HOP;       // frames of the centred STFT (1 + total/hop) minus the dropped last one
        int half = N_FFT / 2;
        float[][] mel = new float[nMels][frames];
        // reflect-padded view of the signal: sample(i) for i in [-half, total + half)
        IntStream.range(0, frames).parallel().forEach(t -> {
            double[] xr = new double[N_FFT];
            double[] xi = new double[N_FFT];
            double[] yr = new double[N_FFT];
            double[] yi = new double[N_FFT];
            double[] power = new double[BINS];
            int start = t * HOP - half;
            for (int i = 0; i < N_FFT; i++) {
                xr[i] = sample(audio, total, start + i) * window[i];
            }
            fft.forward(xr, xi, yr, yi);
            for (int b = 0; b < BINS; b++) {
                power[b] = yr[b] * yr[b] + yi[b] * yi[b];
            }
            for (int m = 0; m < nMels; m++) {
                double[] fm = filters[m];
                double s = 0;
                for (int b = 0; b < BINS; b++) {
                    s += fm[b] * power[b];
                }
                mel[m][t] = (float) Math.log10(Math.max(s, 1e-10));
            }
        });
        float max = Float.NEGATIVE_INFINITY;
        for (float[] row : mel) {
            for (float v : row) {
                max = Math.max(max, v);
            }
        }
        float floor = max - 8f;
        for (float[] row : mel) {
            for (int i = 0; i < frames; i++) {
                row[i] = (Math.max(row[i], floor) + 4f) / 4f;
            }
        }
        return mel;
    }

    /** Sample {@code i} of {@code audio} extended with zeros to {@code total} samples and reflected around both ends. */
    private static float sample(float[] a, int total, int i) {
        if (i < 0) {
            i = -i;
        } else if (i >= total) {
            i = 2 * (total - 1) - i;
        }
        if (i < 0 || i >= total) {
            return 0f;     // only for degenerate (shorter than n_fft/2) inputs
        }
        return i < a.length ? a[i] : 0f;
    }
}
