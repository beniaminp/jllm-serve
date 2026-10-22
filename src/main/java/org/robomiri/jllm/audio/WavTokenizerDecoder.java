package org.robomiri.jllm.audio;

import jdk.incubator.vector.FloatVector;
import org.robomiri.jllm.audio.Dense.Mat;
import org.robomiri.jllm.cpu.ThreadPool;
import org.robomiri.jllm.cpu.VectorKernels;
import org.robomiri.jllm.gguf.GgufFile;
import org.robomiri.jllm.gguf.GgmlType;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Path;

/**
 * The WavTokenizer (large, 75 tokens per second, 24 kHz) decoder, the vocoder of OuteTTS: a codebook of 4096 vectors of
 * 512 numbers, a Vocos backbone and an inverse-STFT head.
 * <ol>
 *   <li>codes to features: one 512-vector per 13.3 ms frame;</li>
 *   <li>{@code embed}: Conv1d 512 to 768, kernel 7;</li>
 *   <li>{@code pos_net}: two residual blocks (GroupNorm 32, swish, Conv1d k3, twice), one single-head self-attention
 *       over all frames with 1x1 convolutions for q, k, v and the output, two more residual blocks, a GroupNorm;</li>
 *   <li>a layer norm, then 12 ConvNeXt blocks (depthwise Conv1d k7, layer norm, 768 to 2304, GELU, 2304 to 768, layer
 *       scale gamma, residual), a final layer norm (the adaptive norms are fixed at bandwidth id 0, as in the GGUF);</li>
 *   <li>the head: Linear 768 to 1282 = 641 log-magnitudes and 641 phases per frame; the spectrum
 *       {@code exp(m) (cos p + i sin p)} goes through an inverse real FFT of 1280 points, a Hann window, overlap-add
 *       with hop 320 and a division by the window's squared envelope (Vocos' "same" padding: 480 samples trimmed at
 *       both ends), giving 320 samples per frame.</li>
 * </ol>
 * Weights are read from llama.cpp's {@code WavTokenizer-Large-75-F16.gguf} (the F16 weights are expanded to floats).
 */
public final class WavTokenizerDecoder implements AutoCloseable {

    public static final int SAMPLE_RATE = 24000;
    public static final int HOP = 320;
    private static final int N_FFT = 1280;
    private static final int BINS = N_FFT / 2 + 1;
    private static final int DIM = 768;
    private static final int FEATURES = 512;
    private static final int FFN = 2304;
    private static final float EPS = 1e-6f;
    private static final int GROUPS = 32;

    private final GgufFile gguf;
    private final ThreadPool pool;
    private final float[] codebook;            // [4096][512]
    private final Mat embed;                   // [768][512*7]
    private final PosRes[] posRes = new PosRes[4];
    private final Mat attQ, attK, attV, attOut;
    private final float[] attNormG, attNormB, posNormG, posNormB;
    private final float[] tokNormG, tokNormB;
    private final ConvNext[] blocks = new ConvNext[12];
    private final float[] outNormG, outNormB;
    private final Mat head;                    // [1282][768]
    private final float[] window = new float[N_FFT];
    private final Fft fft = new Fft(N_FFT);
    private final java.util.concurrent.locks.ReentrantLock lock = new java.util.concurrent.locks.ReentrantLock();

    private record PosRes(float[] n1g, float[] n1b, Mat c1, float[] n2g, float[] n2b, Mat c2) {
    }

    private record ConvNext(float[] dw, float[] dwBias, float[] normG, float[] normB, Mat pw1, Mat pw2, float[] gamma) {
    }

    public WavTokenizerDecoder(Path gguf, int threads) throws IOException {
        this.gguf = GgufFile.open(gguf);
        this.pool = new ThreadPool(threads);
        try {
            String arch = this.gguf.str("general.architecture", "");
            if (!arch.equals("wavtokenizer-dec")) {
                throw new IOException(gguf.getFileName() + " is not a WavTokenizer decoder (architecture '" + arch + "')");
            }
            codebook = floats("token_embd.weight");
            embed = Mat.of(floats("conv1d.weight"), DIM, FEATURES * 7, floats("conv1d.bias"));
            for (int i = 0; i < 4; i++) {
                int b = i < 2 ? i : i + 1;          // posnet blocks 0, 1, 3, 4 are residual; 2 is attention; 5 the norm
                String p = "posnet." + b + ".";
                posRes[i] = new PosRes(floats(p + "norm1.weight"), floats(p + "norm1.bias"),
                        Mat.of(floats(p + "conv1.weight"), DIM, DIM * 3, floats(p + "conv1.bias")),
                        floats(p + "norm2.weight"), floats(p + "norm2.bias"),
                        Mat.of(floats(p + "conv2.weight"), DIM, DIM * 3, floats(p + "conv2.bias")));
            }
            attNormG = floats("posnet.2.attn_norm.weight");
            attNormB = floats("posnet.2.attn_norm.bias");
            attQ = Mat.of(floats("posnet.2.attn_q.weight"), DIM, DIM, floats("posnet.2.attn_q.bias"));
            attK = Mat.of(floats("posnet.2.attn_k.weight"), DIM, DIM, floats("posnet.2.attn_k.bias"));
            attV = Mat.of(floats("posnet.2.attn_v.weight"), DIM, DIM, floats("posnet.2.attn_v.bias"));
            attOut = Mat.of(floats("posnet.2.attn_output.weight"), DIM, DIM, floats("posnet.2.attn_output.bias"));
            posNormG = floats("posnet.5.attn_norm.weight");
            posNormB = floats("posnet.5.attn_norm.bias");
            tokNormG = floats("token_embd_norm.weight");
            tokNormB = floats("token_embd_norm.bias");
            for (int i = 0; i < 12; i++) {
                String p = "convnext." + i + ".";
                blocks[i] = new ConvNext(floats(p + "dw.weight"), floats(p + "dw.bias"), floats(p + "norm.weight"),
                        floats(p + "norm.bias"), Mat.of(floats(p + "pw1.weight"), FFN, DIM, floats(p + "pw1.bias")),
                        Mat.of(floats(p + "pw2.weight"), DIM, FFN, floats(p + "pw2.bias")), floats(p + "gamma.weight"));
            }
            outNormG = floats("output_norm.weight");
            outNormB = floats("output_norm.bias");
            head = Mat.of(floats("output.weight"), N_FFT + 2, DIM, floats("output.bias"));
        } catch (IOException | RuntimeException e) {
            this.gguf.close();
            pool.close();
            throw e;
        }
        for (int i = 0; i < N_FFT; i++) {
            window[i] = (float) (0.5 - 0.5 * Math.cos(2 * Math.PI * i / N_FFT));
        }
    }

    private float[] floats(String name) {
        GgufFile.Tensor t = gguf.tensor(name);
        MemorySegment seg = gguf.data(t);
        int n = Math.toIntExact(t.elements());
        float[] out = new float[n];
        ValueLayout.OfShort h = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
        ValueLayout.OfFloat f = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
        if (t.type() == GgmlType.F16) {
            for (int i = 0; i < n; i++) {
                out[i] = Float.float16ToFloat(seg.getAtIndex(h, i));
            }
        } else if (t.type() == GgmlType.F32) {
            for (int i = 0; i < n; i++) {
                out[i] = seg.getAtIndex(f, i);
            }
        } else {
            throw new IllegalArgumentException(name + " has type " + t.type() + ", expected F16 or F32");
        }
        return out;
    }

    /** The number of audio samples {@code frames} codec frames decode to. */
    public static int samplesFor(int frames) {
        return frames * HOP;
    }

    /** Decodes codes (0..4095 each; larger ones are clamped) into 24 kHz mono samples. */
    public float[] decode(int[] codes) {
        lock.lock();
        try {
            return run(codes);
        } finally {
            lock.unlock();
        }
    }

    /** The backbone's output [frames][768] for testing (before the head). */
    public float[] backbone(int[] codes) {
        lock.lock();
        try {
            return backboneRun(codes);
        } finally {
            lock.unlock();
        }
    }

    /** When non-null, {@link #backboneRun} appends a copy of the activations [frames][768] after each stage (tests). */
    public java.util.List<float[]> trace;

    private void trace(float[] h) {
        if (trace != null) {
            trace.add(h.clone());
        }
    }

    private float[] backboneRun(int[] codes) {
        int t = codes.length;
        float[] x = new float[t * FEATURES];
        for (int i = 0; i < t; i++) {
            int c = Math.max(0, Math.min(4095, codes[i]));
            System.arraycopy(codebook, c * FEATURES, x, i * FEATURES, FEATURES);
        }
        // embed: conv k7 pad 3
        float[] col = im2col(x, t, FEATURES, 7, 3);
        float[] h = new float[t * DIM];
        Dense.mul(pool, embed, col, 0, FEATURES * 7, t, h, 0, DIM, false);
        trace(h);
        // pos_net
        for (int i = 0; i < 2; i++) {
            h = resBlock(h, t, posRes[i]);
            trace(h);
        }
        h = attention(h, t);
        trace(h);
        for (int i = 2; i < 4; i++) {
            h = resBlock(h, t, posRes[i]);
            trace(h);
        }
        groupNorm(h, t, posNormG, posNormB);
        trace(h);
        // adaptive layer norm (bandwidth id 0) in place
        Dense.layerNorm(h, 0, t, DIM, tokNormG, tokNormB, EPS, h, 0);
        trace(h);
        float[] ffn = new float[t * FFN];
        float[] y = new float[t * DIM];
        final float[] hh = h;
        for (ConvNext b : blocks) {
            depthwise(h, t, b, y);
            Dense.layerNorm(y, 0, t, DIM, b.normG(), b.normB(), EPS, y, 0);
            Dense.mul(pool, b.pw1(), y, 0, DIM, t, ffn, 0, FFN, false);
            pool.forRange(ffn.length, 1 << 14, (s, e) -> Dense.gelu(ffn, s, e - s));
            Dense.mul(pool, b.pw2(), ffn, 0, FFN, t, y, 0, DIM, false);
            float[] g = b.gamma();
            pool.forRange(t, 32, (s, e) -> {
                for (int i = s; i < e; i++) {
                    for (int c = 0; c < DIM; c++) {
                        hh[i * DIM + c] += g[c] * y[i * DIM + c];
                    }
                }
            });
            trace(hh);
        }
        Dense.layerNorm(h, 0, t, DIM, outNormG, outNormB, EPS, h, 0);
        return h;
    }

    private float[] run(int[] codes) {
        int t = codes.length;
        if (t == 0) {
            return new float[0];
        }
        float[] h = backboneRun(codes);
        float[] spec = new float[t * (N_FFT + 2)];
        Dense.mul(pool, head, h, 0, DIM, t, spec, 0, N_FFT + 2, false);
        return istft(spec, t);
    }

    // ------------------------------------------------------------------ pieces

    /** {@code col[t][c * k + j] = x[t + j - pad][c]} (zero outside), for a convolution as a matrix product. */
    private float[] im2col(float[] x, int t, int c, int k, int pad) {
        float[] col = new float[t * c * k];
        pool.forRange(t, 16, (s, e) -> {
            for (int i = s; i < e; i++) {
                int base = i * c * k;
                for (int j = 0; j < k; j++) {
                    int src = i + j - pad;
                    if (src < 0 || src >= t) {
                        continue;
                    }
                    int so = src * c;
                    for (int ch = 0; ch < c; ch++) {
                        col[base + ch * k + j] = x[so + ch];
                    }
                }
            }
        });
        return col;
    }

    /** {@code x + conv2(swish(gn2(conv1(swish(gn1(x))))))}. */
    private float[] resBlock(float[] x, int t, PosRes r) {
        float[] a = x.clone();
        groupNorm(a, t, r.n1g(), r.n1b());
        swish(a);
        float[] col = im2col(a, t, DIM, 3, 1);
        float[] b = new float[t * DIM];
        Dense.mul(pool, r.c1(), col, 0, DIM * 3, t, b, 0, DIM, false);
        groupNorm(b, t, r.n2g(), r.n2b());
        swish(b);
        col = im2col(b, t, DIM, 3, 1);
        float[] out = x.clone();
        Dense.mul(pool, r.c2(), col, 0, DIM * 3, t, out, 0, DIM, true);
        return out;
    }

    /** GroupNorm with 32 groups over [t][768] in place: statistics per group over its 24 channels and all frames. */
    private void groupNorm(float[] x, int t, float[] gamma, float[] beta) {
        int per = DIM / GROUPS;
        for (int g = 0; g < GROUPS; g++) {
            double sum = 0;
            for (int i = 0; i < t; i++) {
                int o = i * DIM + g * per;
                for (int c = 0; c < per; c++) {
                    sum += x[o + c];
                }
            }
            double n = (double) t * per;
            double mean = sum / n;
            double var = 0;
            for (int i = 0; i < t; i++) {
                int o = i * DIM + g * per;
                for (int c = 0; c < per; c++) {
                    double d = x[o + c] - mean;
                    var += d * d;
                }
            }
            float inv = (float) (1.0 / Math.sqrt(var / n + EPS));
            float m = (float) mean;
            for (int i = 0; i < t; i++) {
                int o = i * DIM + g * per;
                for (int c = 0; c < per; c++) {
                    int ch = g * per + c;
                    x[o + c] = (x[o + c] - m) * inv * gamma[ch] + beta[ch];
                }
            }
        }
    }

    private void swish(float[] a) {
        pool.forRange(a.length, 1 << 14, (s, e) -> {
            int bound = s + Dense.F.loopBound(e - s);
            int i = s;
            for (; i < bound; i += Dense.F.length()) {
                FloatVector v = FloatVector.fromArray(Dense.F, a, i);
                // x * sigmoid(x) = x / (1 + e^-x)
                v.div(VectorKernels.expFast(v.neg()).add(1f)).intoArray(a, i);
            }
            for (; i < e; i++) {
                a[i] = a[i] / (1f + VectorKernels.expFast(-a[i]));
            }
        });
    }

    /** The posnet's self-attention: one head of width 768 over all frames, q k v and the output as 1x1 convolutions. */
    private float[] attention(float[] x, int t) {
        float[] n = x.clone();
        groupNorm(n, t, attNormG, attNormB);
        float[] q = new float[t * DIM];
        float[] k = new float[t * DIM];
        float[] v = new float[t * DIM];
        Dense.mul(pool, attQ, n, 0, DIM, t, q, 0, DIM, false);
        Dense.mul(pool, attK, n, 0, DIM, t, k, 0, DIM, false);
        Dense.mul(pool, attV, n, 0, DIM, t, v, 0, DIM, false);
        float[] vt = new float[t * DIM];                  // [768][t]
        for (int i = 0; i < t; i++) {
            for (int c = 0; c < DIM; c++) {
                vt[c * t + i] = v[i * DIM + c];
            }
        }
        float scale = (float) (1.0 / Math.sqrt(DIM));
        for (int i = 0; i < q.length; i++) {
            q[i] *= scale;
        }
        float[] ctx = new float[t * DIM];
        int blocks = (t + 31) / 32;
        pool.forRangeFine(blocks, 1, (s, e) -> {
            float[] sc = new float[32 * t];
            for (int b = s; b < e; b++) {
                int t0 = b * 32;
                int nq = Math.min(32, t - t0);
                Dense.block(k, 0, t, DIM, q, t0 * DIM, DIM, nq, sc, 0, t, false);
                for (int r = 0; r < nq; r++) {
                    float max = Float.NEGATIVE_INFINITY;
                    for (int j = 0; j < t; j++) {
                        max = Math.max(max, sc[r * t + j]);
                    }
                    float sum = VectorKernels.softmaxExp(sc, r * t, t, max);
                    float inv = 1f / sum;
                    for (int j = 0; j < t; j++) {
                        sc[r * t + j] *= inv;
                    }
                }
                Dense.block(vt, 0, DIM, t, sc, 0, t, nq, ctx, t0 * DIM, DIM, false);
            }
        });
        float[] out = x.clone();
        Dense.mul(pool, attOut, ctx, 0, DIM, t, out, 0, DIM, true);
        return out;
    }

    /** Depthwise Conv1d (kernel 7, pad 3) of [t][768] into {@code y}. The weights are [768][7]. */
    private void depthwise(float[] h, int t, ConvNext b, float[] y) {
        float[] w = b.dw();
        float[] bias = b.dwBias();
        pool.forRange(t, 16, (s, e) -> {
            for (int i = s; i < e; i++) {
                for (int c = 0; c < DIM; c++) {
                    float acc = bias[c];
                    for (int j = 0; j < 7; j++) {
                        int src = i + j - 3;
                        if (src >= 0 && src < t) {
                            acc += w[c * 7 + j] * h[src * DIM + c];
                        }
                    }
                    y[i * DIM + c] = acc;
                }
            }
        });
    }

    /** The head's [t][1282] (641 log-magnitudes, then 641 phases) to audio. */
    private float[] istft(float[] spec, int t) {
        int pad = (N_FFT - HOP) / 2;
        int outLen = (t - 1) * HOP + N_FFT;
        float[] y = new float[outLen];
        float[] env = new float[outLen];
        // frames are independent; the overlap-add is done after (sequentially) from per-frame results
        float[][] frames = new float[t][];
        pool.forRange(t, 4, (s, e) -> {
            double[] xr = new double[N_FFT];
            double[] xi = new double[N_FFT];
            double[] yr = new double[N_FFT];
            double[] yi = new double[N_FFT];
            for (int f = s; f < e; f++) {
                int base = f * (N_FFT + 2);
                for (int k = 0; k < BINS; k++) {
                    double mag = Math.min(Math.exp(spec[base + k]), 100.0);
                    double p = spec[base + BINS + k];
                    double re = mag * Math.cos(p);
                    double im = mag * Math.sin(p);
                    if (k == 0 || k == BINS - 1) {
                        im = 0;            // an inverse real FFT ignores the imaginary part of DC and Nyquist
                    }
                    xr[k] = re;
                    xi[k] = -im;           // forward FFT of the conjugate = conjugate of the inverse FFT
                    if (k > 0 && k < BINS - 1) {
                        xr[N_FFT - k] = re;
                        xi[N_FFT - k] = im;
                    }
                }
                fft.forward(xr, xi, yr, yi);
                float[] fr = new float[N_FFT];
                for (int n = 0; n < N_FFT; n++) {
                    fr[n] = (float) (yr[n] / N_FFT) * window[n];
                }
                frames[f] = fr;
            }
        });
        for (int f = 0; f < t; f++) {
            float[] fr = frames[f];
            int o = f * HOP;
            for (int n = 0; n < N_FFT; n++) {
                y[o + n] += fr[n];
                env[o + n] += window[n] * window[n];
            }
        }
        float[] out = new float[outLen - 2 * pad];
        for (int i = 0; i < out.length; i++) {
            out[i] = y[pad + i] / env[pad + i];
        }
        return out;
    }

    @Override
    public void close() {
        pool.close();
        gguf.close();
    }
}
