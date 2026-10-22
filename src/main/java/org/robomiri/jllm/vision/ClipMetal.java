package org.robomiri.jllm.vision;

import org.robomiri.jllm.gguf.GgmlType;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.vision.ClipModel.Layer;
import org.robomiri.jllm.vision.Preprocess.Pixels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * The vision encoder on the GPU: the same computation as {@link ClipCpu} (and llama.cpp's clip graphs), with the
 * kernels of {@code vision.metal} in a Metal context of their own. Weights are half (F16 tensors of the projector file
 * are used in place, other types are converted once at load), activations float; the patches are unfolded on the CPU
 * (unified memory: the GPU reads the same bytes), everything else is one command buffer per image.
 */
public final class ClipMetal implements VisionEncoder {

    private record W(Metal.Buffer buf, int rows, int cols) {
    }

    private final ClipModel m;
    private final Metal metal;
    private final Arena arena = Arena.ofShared();
    private final MemorySegment bufArgs = arena.allocate(ValueLayout.ADDRESS, 8);
    private final MemorySegment offArgs = arena.allocate(ValueLayout.JAVA_LONG, 8);
    private final MemorySegment params = arena.allocate(64);
    private final Metal.Buffer dummy;
    private final Map<String, Metal.Buffer> scratch = new HashMap<>();
    private final Map<Mat, W> weights = new IdentityHashMap<>();
    private final Map<float[], Metal.Buffer> vectors = new IdentityHashMap<>();
    private final String device;
    private int pcount;

    public ClipMetal(ClipModel m, String explicitLib) {
        this.m = m;
        if (m.nEmbd % 4 != 0) {
            throw new IllegalArgumentException("the encoder width " + m.nEmbd + " is not a multiple of 4");
        }
        this.metal = Metal.create(explicitLib, "", "vision.metal");
        this.device = metal.deviceName();
        this.dummy = metal.newBuffer(64, true);
    }

    /** True when a GPU can run the encoder here. */
    public static boolean available(String explicitLib) {
        return Metal.available(explicitLib);
    }

    @Override
    public String name() {
        return "metal (" + device + ")";
    }

    @Override
    public void close() {
        profile.forEach((k, a) -> System.err.printf("vision profile: %-16s %8.1f ms in %d dispatches%n", k, a[0], (long) a[1]));
        metal.close();
        arena.close();
    }

    // ------------------------------------------------------------------ buffers and dispatch plumbing

    private Metal.Buffer scratch(String name, long bytes) {
        Metal.Buffer b = scratch.get(name);
        if (b == null || b.size() < bytes) {
            if (b != null) {
                metal.release(b);
            }
            b = metal.newBuffer(Math.max(16, bytes), false);
            scratch.put(name, b);
        }
        return b;
    }

    private static void putFloats(Metal.Buffer b, float[] a, int n) {
        MemorySegment.copy(a, 0, b.contents(), ValueLayout.JAVA_FLOAT, 0, n);
    }

    private static void putInts(Metal.Buffer b, int[] a, int n) {
        MemorySegment.copy(a, 0, b.contents(), ValueLayout.JAVA_INT, 0, n);
    }

    /** A matrix as half on the GPU: F16 tensors in place, anything else converted (once). */
    private W weight(Mat mat) {
        W w = weights.get(mat);
        if (w != null) {
            return w;
        }
        if (mat.cols % 4 != 0) {
            throw new IllegalArgumentException(mat + ": the row length must be a multiple of 4");
        }
        Metal.Buffer buf;
        if (mat.type() == GgmlType.F16) {
            buf = metal.wrapRange(mat.segment());
        } else {
            buf = convert(mat.rows, mat.cols, (r, dst) -> mat.rows(r, 1, dst, mat.cols));
        }
        w = new W(buf, mat.rows, mat.cols);
        weights.put(mat, w);
        return w;
    }

    private interface RowSource {
        void row(int r, float[] dst);
    }

    private Metal.Buffer convert(int rows, int cols, RowSource src) {
        Metal.Buffer buf = metal.newBuffer((long) rows * cols * 2, false);
        MemorySegment seg = buf.contents();
        ValueLayout.OfShort h = ValueLayout.JAVA_SHORT;
        IntStream.range(0, rows).parallel().forEach(r -> {
            float[] row = new float[cols];
            src.row(r, row);
            long o = (long) r * cols * 2;
            for (int c = 0; c < cols; c++) {
                seg.set(h, o + 2L * c, Float.floatToFloat16(row[c]));
            }
        });
        return buf;
    }

    /** A half weight made of a float matrix in memory ({@code rows x cols}). */
    private W weight(float[] a, int rows, int cols) {
        return new W(convert(rows, cols, (r, dst) -> System.arraycopy(a, r * cols, dst, 0, cols)), rows, cols);
    }

    private Metal.Buffer vec(float[] v) {
        if (v == null) {
            return null;
        }
        return vectors.computeIfAbsent(v, x -> {
            Metal.Buffer b = metal.newBuffer(4L * x.length, false);
            putFloats(b, x, x.length);
            return b;
        });
    }

    private void bind(int i, Metal.Buffer b) {
        Metal.Buffer x = b != null ? b : dummy;
        bufArgs.setAtIndex(ValueLayout.ADDRESS, i, x.handle());
        offArgs.setAtIndex(ValueLayout.JAVA_LONG, i, x.offset());
    }

    private void p(Object... v) {
        pcount = 0;
        for (Object o : v) {
            if (o instanceof Float f) {
                params.setAtIndex(ValueLayout.JAVA_FLOAT, pcount++, f);
            } else {
                params.setAtIndex(ValueLayout.JAVA_INT, pcount++, ((Number) o).intValue());
            }
        }
    }

    /** {@code -Djllm.vision.profile=true}: waits after every dispatch and prints the time per kernel (slow; for tuning). */
    private static final boolean PROFILE = Boolean.getBoolean("jllm.vision.profile");
    private final Map<String, double[]> profile = new java.util.TreeMap<>();

    private void run(String kernel, int nbuf, int gx, int gy, int gz, int tx, int ty, int tz) {
        int pipe = metal.pipeline(kernel);
        if (!PROFILE) {
            metal.dispatch(pipe, bufArgs, offArgs, nbuf, params, 4 * pcount, 0, gx, gy, gz, tx, ty, tz);
            return;
        }
        long t0 = System.nanoTime();
        metal.dispatch(pipe, bufArgs, offArgs, nbuf, params, 4 * pcount, 0, gx, gy, gz, tx, ty, tz);
        metal.commitAndWait();
        metal.begin();
        double[] a = profile.computeIfAbsent(kernel, k -> new double[2]);
        a[0] += (System.nanoTime() - t0) / 1e6;
        a[1]++;
    }

    private static int cdiv(int a, int b) {
        return (a + b - 1) / b;
    }

    // ------------------------------------------------------------------ the operations

    private int actCode() {
        return switch (m.act) {
            case GELU -> 1;
            case SILU -> 2;
            case GELU_QUICK -> 3;
        };
    }

    /** out = act(x . W^T + bias) (+ out): {@code mTok} rows of x with stride ldx. */
    private void gemm(Metal.Buffer x, W w, Metal.Buffer bias, Metal.Buffer out, int mTok, int ldx, int ldo, int act, boolean acc) {
        bind(0, x);
        bind(1, w.buf());
        bind(2, bias);
        bind(3, out);
        p(mTok, w.rows(), w.cols(), ldx, ldo, act, acc ? 1 : 0, bias != null ? 1 : 0);
        run("vit_gemm", 4, cdiv(w.rows(), 64), cdiv(mTok, 64), 1, 128, 1, 1);
    }

    private void norm(boolean rms, Metal.Buffer x, Metal.Buffer out, float[] w, float[] b, int rows, int cols) {
        bind(0, x);
        bind(1, out);
        bind(2, vec(w));
        bind(3, vec(b));
        p(cols, rms ? 1 : 0, w != null ? 1 : 0, b != null ? 1 : 0, m.eps, cols, cols);
        run("vit_norm", 4, rows, 1, 1, 256, 1, 1);
    }

    private void elementwise(String kernel, Metal.Buffer a, Metal.Buffer b, int n, int act) {
        bind(0, a);
        int nb = 1;
        if (b != null) {
            bind(1, b);
            nb = 2;
        }
        p(n, act);
        run(kernel, nb, cdiv(n, 256), 1, 1, 256, 1, 1);
    }

    private void add(Metal.Buffer a, Metal.Buffer b, int n) {
        elementwise("vit_add", a, b, n, 0);
    }

    /** The q, k or v part ({@code which} = 0, 1, 2) of a fused attention weight, as its own half matrix (made once). */
    private W qkvPart(int il, Layer l, int which) {
        int e = m.nEmbd;
        return derived("qkv." + il + "." + which, () -> {
            float[] part = new float[e * l.qkv.cols];
            l.qkv.rows(which * e, e, part, l.qkv.cols);
            return weight(part, e, l.qkv.cols);
        });
    }

    private Metal.Buffer qkvBias(int il, Layer l, int which) {
        int e = m.nEmbd;
        return vec(cached.computeIfAbsent("qkvB." + il + "." + which,
                k -> l.qkvB == null ? null : java.util.Arrays.copyOfRange(l.qkvB, which * e, (which + 1) * e)));
    }

    private void blocks(Metal.Buffer x, int t, boolean rms, Metal.Buffer ropeCs, Metal.Buffer lo, Metal.Buffer hi,
                        Metal.Buffer loW, Metal.Buffer hiW, java.util.function.IntConsumer afterLayer) {
        int e = m.nEmbd;
        int heads = m.nHead;
        int dh = e / heads;
        int ldF = m.nFf;
        float scale = 1.0f / (float) Math.sqrt(dh);
        Metal.Buffer h = scratch("h", 4L * t * e);
        Metal.Buffer q = scratch("q", 4L * t * e);
        Metal.Buffer k = scratch("k", 4L * t * e);
        Metal.Buffer v = scratch("v", 4L * t * e);
        Metal.Buffer att = scratch("att", 4L * t * e);
        Metal.Buffer f1 = scratch("f1", 4L * t * ldF);
        Metal.Buffer f2 = scratch("f2", 4L * t * ldF);
        for (int il = 0; il < m.nLayer; il++) {
            Layer l = m.layers.get(il);
            norm(rms, x, h, l.ln1W, l.ln1B, t, e);
            if (l.qkv != null) {
                gemm(h, qkvPart(il, l, 0), qkvBias(il, l, 0), q, t, e, e, 0, false);
                gemm(h, qkvPart(il, l, 1), qkvBias(il, l, 1), k, t, e, e, 0, false);
                gemm(h, qkvPart(il, l, 2), qkvBias(il, l, 2), v, t, e, e, 0, false);
            } else {
                gemm(h, weight(l.q), vec(l.qB), q, t, e, e, 0, false);
                gemm(h, weight(l.k), vec(l.kB), k, t, e, e, 0, false);
                gemm(h, weight(l.v), vec(l.vB), v, t, e, e, 0, false);
            }
            if (ropeCs != null) {
                for (Metal.Buffer qk : new Metal.Buffer[]{q, k}) {
                    bind(0, qk);
                    bind(1, ropeCs);
                    p(t, heads, dh);
                    run("vit_rope", 2, cdiv(dh / 2, 8), heads, t, 8, 1, 1);
                }
            }
            boolean fullAttn = loW == null || m.nWaPattern <= 0 || (il + 1) % m.nWaPattern == 0;
            bind(0, q);
            bind(1, k);
            bind(2, v);
            bind(3, att);
            bind(4, fullAttn ? lo : loW);
            bind(5, fullAttn ? hi : hiW);
            p(t, heads, dh, scale);
            if (dh == 64 || dh == 72 || dh == 80 || dh == 128) { // simdgroup-matrix kernel
                run("vit_attention_mma_d" + dh, 6, heads, cdiv(t, 32), 1, 128, 1, 1);
            } else {
                run("vit_attention", 6, cdiv(t, 16), heads, 1, 128, 1, 1);
            }
            gemm(att, weight(l.o), vec(l.oB), x, t, e, e, 0, true);
            norm(rms, x, h, l.ln2W, l.ln2B, t, e);
            gemm(h, weight(l.up), vec(l.upB), f1, t, e, ldF, l.gate != null ? 0 : actCode(), false);
            if (l.gate != null) {
                gemm(h, weight(l.gate), vec(l.gateB), f2, t, e, ldF, 0, false);
                elementwise("vit_gated", f2, f1, t * ldF, actCode());
                gemm(f2, weight(l.down), vec(l.downB), x, t, ldF, e, 0, true);
            } else {
                gemm(f1, weight(l.down), vec(l.downB), x, t, ldF, e, 0, true);
            }
            if (afterLayer != null) {
                afterLayer.accept(il);
            }
        }
    }

    // ------------------------------------------------------------------ patches

    private float[] unfold(Pixels px, int[] orderY, int[] orderX, int kPad) {
        int p = m.patchSize;
        int w = px.width();
        float[] src = px.hwc();
        int t = orderY.length;
        float[] out = new float[t * kPad];
        IntStream.range(0, t).parallel().forEach(i -> {
            int o = i * kPad;
            int y0 = orderY[i] * p;
            int x0 = orderX[i] * p;
            for (int c = 0; c < 3; c++) {
                for (int ky = 0; ky < p; ky++) {
                    int s = 3 * ((y0 + ky) * w + x0) + c;
                    int d = o + c * p * p + ky * p;
                    for (int kx = 0; kx < p; kx++) {
                        out[d + kx] = src[s + 3 * kx];
                    }
                }
            }
        });
        return out;
    }

    private W patchWeights(int kPad, String... names) {
        float[] w = null;
        int rows = 0;
        for (String name : names) {
            if (!m.has(name)) {
                continue;
            }
            Mat c = m.convMat(name);
            rows = c.rows;
            float[] part = new float[c.rows * kPad];
            c.rows(0, c.rows, part, kPad);
            if (w == null) {
                w = part;
            } else {
                for (int i = 0; i < w.length; i++) {
                    w[i] += part[i];
                }
            }
        }
        if (w == null) {
            throw new IllegalArgumentException(m.path.getFileName() + ": no patch embedding tensor");
        }
        return weight(w, rows, kPad);
    }

    // ------------------------------------------------------------------ encode

    private final Map<String, W> derived = new HashMap<>();

    @Override
    public synchronized Encoded encode(Pixels px) {
        profile.clear(); // (the profile of the last picture only: the first one includes compiling the pipelines)
        return switch (m.projector) {
            case IDEFICS3, GEMMA3 -> siglip(px);
            case QWEN2VL, QWEN25VL, QWEN3VL -> qwen(px);
        };
    }

    private W derived(String key, java.util.function.Supplier<W> make) {
        return derived.computeIfAbsent(key, k -> make.get());
    }

    private W weightByName(String name) {
        return derived(name, () -> weight(m.mat(name)));
    }

    private Encoded siglip(Pixels px) {
        int p = m.patchSize;
        if (px.width() % p != 0 || px.height() % p != 0) {
            throw new IllegalArgumentException("image " + px.width() + "x" + px.height() + " is not a multiple of the patch size " + p);
        }
        int gx = px.width() / p;
        int gy = px.height() / p;
        int t = gx * gy;
        int e = m.nEmbd;
        int kPad = (3 * p * p + 3) / 4 * 4;
        int[] oy = new int[t];
        int[] ox = new int[t];
        for (int i = 0; i < t; i++) {
            oy[i] = i / gx;
            ox[i] = i % gx;
        }
        W pw = derived("patch", () -> patchWeights(kPad, "v.patch_embd.weight"));
        Mat posM = m.mat("v.position_embd.weight");
        if (posM.rows != t || posM.cols != e) {
            throw new IllegalArgumentException("position embeddings " + posM + " do not fit " + gx + "x" + gy + " patches of width " + e);
        }
        Metal.Buffer pos = vec(derivedFloats("pos", posM));
        Metal.Buffer patches = scratch("patches", 4L * t * kPad);
        putFloats(patches, unfold(px, oy, ox, kPad), t * kPad);
        Metal.Buffer x = scratch("x", 4L * t * e);
        Metal.Buffer[] range = ranges(t);
        metal.begin();
        boolean ok = false;
        try {
            gemm(patches, pw, vec(m.vecOpt("v.patch_embd.bias")), x, t, kPad, e, 0, false);
            add(x, pos, t * e);
            float[] preW = cachedVec("v.pre_ln.weight");
            if (preW != null) {
                norm(false, x, x, preW, cachedVec("v.pre_ln.bias"), t, e);
            }
            blocks(x, t, false, null, range[0], range[1], null, null, null);
            float[] pw2 = cachedVec("v.post_ln.weight");
            if (pw2 != null) {
                norm(false, x, x, pw2, cachedVec("v.post_ln.bias"), t, e);
            }
            Encoded out = switch (m.projector) {
                case IDEFICS3 -> idefics3Projector(x, gx, gy);
                case GEMMA3 -> gemma3Projector(x, gx, gy);
                default -> throw new IllegalStateException();
            };
            ok = true;
            return out;
        } finally {
            if (!ok) {
                metal.abort();
            }
        }
    }

    private final Map<String, float[]> cached = new HashMap<>();

    private float[] cachedVec(String name) {
        return cached.computeIfAbsent(name, k -> m.vecOpt(k));
    }

    private float[] derivedFloats(String key, Mat mat) {
        return cached.computeIfAbsent(key, k -> mat.toFloats());
    }

    /** lo / hi buffers for full attention: every row attends to {@code [0, t)}. */
    private Metal.Buffer[] ranges(int t) {
        Metal.Buffer lo = scratch("lo", 4L * t);
        Metal.Buffer hi = scratch("hi", 4L * t);
        int[] a = new int[t];
        int[] b = new int[t];
        java.util.Arrays.fill(b, t);
        putInts(lo, a, t);
        putInts(hi, b, t);
        return new Metal.Buffer[]{lo, hi};
    }

    private Encoded idefics3Projector(Metal.Buffer x, int gx, int gy) {
        int s = m.nMerge;
        int e = m.nEmbd;
        int ox = gx / s;
        int oy = gy / s;
        int tokens = ox * oy;
        int g = s * s;
        int[] idx = new int[tokens * g];
        for (int by = 0; by < oy; by++) {
            for (int bx = 0; bx < ox; bx++) {
                for (int ys = 0; ys < s; ys++) {
                    for (int xs = 0; xs < s; xs++) {
                        idx[(by * ox + bx) * g + ys * s + xs] = (by * s + ys) * gx + bx * s + xs;
                    }
                }
            }
        }
        Metal.Buffer ib = scratch("idx", 4L * idx.length);
        putInts(ib, idx, idx.length);
        Metal.Buffer shuffled = scratch("shuf", 4L * tokens * g * e);
        bind(0, x);
        bind(1, shuffled);
        bind(2, ib);
        p(tokens * g, e, 1);
        run("vit_gather", 3, cdiv(e, 16), cdiv(tokens * g, 4), 1, 16, 4, 1);
        W fc = weightByName("mm.model.fc.weight");
        Metal.Buffer out = scratch("out", 4L * tokens * fc.rows());
        gemm(shuffled, fc, null, out, tokens, g * e, fc.rows(), 0, false);
        return finish(out, tokens, fc.rows(), ox, oy);
    }

    private Encoded gemma3Projector(Metal.Buffer x, int gx, int gy) {
        int s = m.nMerge;
        int e = m.nEmbd;
        int ox = gx / s;
        int oy = gy / s;
        int tokens = ox * oy;
        int g = s * s;
        int[] idx = new int[tokens * g];
        for (int by = 0; by < oy; by++) {
            for (int bx = 0; bx < ox; bx++) {
                for (int ys = 0; ys < s; ys++) {
                    for (int xs = 0; xs < s; xs++) {
                        idx[(by * ox + bx) * g + ys * s + xs] = (by * s + ys) * gx + bx * s + xs;
                    }
                }
            }
        }
        Metal.Buffer ib = scratch("idx", 4L * idx.length);
        putInts(ib, idx, idx.length);
        Metal.Buffer pooled = scratch("pool", 4L * tokens * e);
        bind(0, x);
        bind(1, pooled);
        bind(2, ib);
        p(tokens, e, g);
        run("vit_gather_mean", 3, cdiv(e, 16), cdiv(tokens, 4), 1, 16, 4, 1);
        norm(true, pooled, pooled, cachedVec("mm.soft_emb_norm.weight"), null, tokens, e);
        Mat pj = m.mat("mm.input_projection.weight");
        W wt = derived("gemma-proj", () -> {
            float[] w = pj.toFloats(); // [e][nOut]
            float[] tr = new float[pj.cols * pj.rows];
            for (int i = 0; i < pj.rows; i++) {
                for (int o = 0; o < pj.cols; o++) {
                    tr[o * pj.rows + i] = w[i * pj.cols + o];
                }
            }
            return weight(tr, pj.cols, pj.rows);
        });
        Metal.Buffer out = scratch("out", 4L * tokens * wt.rows());
        gemm(pooled, wt, null, out, tokens, e, wt.rows(), 0, false);
        return finish(out, tokens, wt.rows(), ox, oy);
    }

    private Encoded qwen(Pixels px) {
        int p = m.patchSize;
        int mg = m.nMerge;
        if (px.width() % (p * mg) != 0 || px.height() % (p * mg) != 0) {
            throw new IllegalArgumentException("image " + px.width() + "x" + px.height() + " is not a multiple of " + p * mg);
        }
        int gx = px.width() / p;
        int gy = px.height() / p;
        int t = gx * gy;
        int e = m.nEmbd;
        boolean rms = m.projector == ClipModel.Projector.QWEN25VL;
        int[] oy = new int[t];
        int[] ox = new int[t];
        int n = 0;
        for (int y = 0; y < gy; y += mg) {
            for (int x = 0; x < gx; x += mg) {
                for (int dy = 0; dy < mg; dy++) {
                    for (int dx = 0; dx < mg; dx++) {
                        oy[n] = y + dy;
                        ox[n] = x + dx;
                        n++;
                    }
                }
            }
        }
        int mm = mg * mg;
        int bw = gx / mg;
        int bh = gy / mg;
        int[] blockOrder = new int[bw * bh];
        int[] lo = new int[t];
        int[] hi = new int[t];
        java.util.Arrays.fill(hi, t);
        int[] loW = null;
        int[] hiW = null;
        if (m.nWaPattern > 0) {
            int ws = m.windowSize > 0 ? m.windowSize : 112;
            int gw = Math.max(1, ws / p / mg);
            loW = new int[t];
            hiW = new int[t];
            int dst = 0;
            for (int y = 0; y < bh; y += gw) {
                for (int x = 0; x < bw; x += gw) {
                    int wh = Math.min(gw, bh - y);
                    int wwid = Math.min(gw, bw - x);
                    int start = dst * mm;
                    for (int dy = 0; dy < wh; dy++) {
                        for (int dx = 0; dx < wwid; dx++) {
                            blockOrder[dst++] = (y + dy) * bw + (x + dx);
                        }
                    }
                    for (int i = start; i < dst * mm; i++) {
                        loW[i] = start;
                        hiW[i] = dst * mm;
                    }
                }
            }
            int[] ny = new int[t];
            int[] nx = new int[t];
            for (int d = 0; d < bw * bh; d++) {
                System.arraycopy(oy, blockOrder[d] * mm, ny, d * mm, mm);
                System.arraycopy(ox, blockOrder[d] * mm, nx, d * mm, mm);
            }
            oy = ny;
            ox = nx;
        } else {
            for (int i = 0; i < blockOrder.length; i++) {
                blockOrder[i] = i;
            }
        }
        int kPad = (3 * p * p + 3) / 4 * 4;
        W pw = derived("patch", () -> patchWeights(kPad, "v.patch_embd.weight", "v.patch_embd.weight.1"));
        Metal.Buffer patches = scratch("patches", 4L * t * kPad);
        putFloats(patches, unfold(px, oy, ox, kPad), t * kPad);
        Metal.Buffer x = scratch("x", 4L * t * e);
        // rotary table: per row the cos / sin of its (y, x) position
        int dh = e / m.nHead;
        int half = dh / 2;
        int quarter = dh / 4;
        float[] cs = new float[t * half * 2];
        double[] inv = new double[quarter];
        for (int j = 0; j < quarter; j++) {
            inv[j] = 1.0 / Math.pow(10000.0, 2.0 * j / half);
        }
        for (int r = 0; r < t; r++) {
            for (int j = 0; j < quarter; j++) {
                double a = oy[r] * inv[j];
                double b = ox[r] * inv[j];
                cs[(r * half + j) * 2] = (float) Math.cos(a);
                cs[(r * half + j) * 2 + 1] = (float) Math.sin(a);
                cs[(r * half + quarter + j) * 2] = (float) Math.cos(b);
                cs[(r * half + quarter + j) * 2 + 1] = (float) Math.sin(b);
            }
        }
        Metal.Buffer csb = scratch("cs", 4L * cs.length);
        putFloats(csb, cs, cs.length);
        Metal.Buffer lob = scratch("lo", 4L * t);
        Metal.Buffer hib = scratch("hi", 4L * t);
        putInts(lob, lo, t);
        putInts(hib, hi, t);
        Metal.Buffer lowb = null;
        Metal.Buffer hiwb = null;
        if (loW != null) {
            lowb = scratch("loW", 4L * t);
            hiwb = scratch("hiW", 4L * t);
            putInts(lowb, loW, t);
            putInts(hiwb, hiW, t);
        }
        int tokens = t / mm;
        int feat = e * mm;
        W w0 = weightByName("mm.0.weight");
        W w2 = weightByName("mm.2.weight");
        Metal.Buffer hid = scratch("hid", 4L * tokens * w0.rows());
        Metal.Buffer merged = scratch("out", 4L * tokens * w2.rows());
        metal.begin();
        boolean ok = false;
        int nDeep = 0;
        try {
            gemm(patches, pw, vec(m.vecOpt("v.patch_embd.bias")), x, t, kPad, e, 0, false);
            if (m.projector == ClipModel.Projector.QWEN3VL) { // learned positions, interpolated to this grid
                Metal.Buffer posb = scratch("pos", 4L * t * e);
                putFloats(posb, ClipCpu.interpolatedPositions(m.mat("v.position_embd.weight"), gx, gy, oy, ox), t * e);
                add(x, posb, t * e);
            }
            float[] preW = cachedVec("v.pre_ln.weight");
            if (preW != null) {
                norm(rms, x, x, preW, cachedVec("v.pre_ln.bias"), t, e);
            }
            Metal.Buffer dsNorm = scratch("dsNorm", 4L * t * e);
            int[] dsCount = {0};
            java.util.function.IntConsumer hook = m.projector != ClipModel.Projector.QWEN3VL ? null : il -> {
                Layer l = m.layers.get(il);
                if (!l.hasDeepstack()) {
                    return;
                }
                // 2 x 2 neighbouring tokens are consecutive rows of x: the same memory read as rows of 4 E features
                norm(false, x, dsNorm, l.dsNormW, l.dsNormB, tokens, feat);
                W f1 = derived("ds1." + il, () -> weight(l.dsFc1));
                W f2 = derived("ds2." + il, () -> weight(l.dsFc2));
                Metal.Buffer dh1 = scratch("dsHid", 4L * tokens * f1.rows());
                Metal.Buffer dout = scratch("ds" + dsCount[0]++, 4L * tokens * f2.rows());
                gemm(dsNorm, f1, vec(l.dsFc1B), dh1, tokens, feat, f1.rows(), 1, false);
                gemm(dh1, f2, vec(l.dsFc2B), dout, tokens, f1.rows(), f2.rows(), 0, false);
            };
            blocks(x, t, rms, csb, lob, hib, lowb, hiwb, hook);
            nDeep = dsCount[0];
            float[] pw2 = cachedVec("v.post_ln.weight");
            if (pw2 != null) {
                norm(rms, x, x, pw2, cachedVec("v.post_ln.bias"), t, e);
            }
            gemm(x, w0, vec(cachedVec("mm.0.bias")), hid, tokens, feat, w0.rows(), 1, false);
            gemm(hid, w2, vec(cachedVec("mm.2.bias")), merged, tokens, w0.rows(), w2.rows(), 0, false);
            ok = true;
        } finally {
            if (!ok) {
                metal.abort();
            }
        }
        metal.commitAndWait();
        float[] outAll = new float[tokens * w2.rows()];
        MemorySegment.copy(merged.contents(), ValueLayout.JAVA_FLOAT, 0, outAll, 0, outAll.length);
        float[] out = outAll;
        if (m.nWaPattern > 0) { // back to raster order of the merged grid
            out = new float[outAll.length];
            int dim = w2.rows();
            for (int d = 0; d < tokens; d++) {
                System.arraycopy(outAll, d * dim, out, blockOrder[d] * dim, dim);
            }
        }
        float[][] deep = null;
        if (nDeep > 0) {
            deep = new float[nDeep][];
            for (int j = 0; j < nDeep; j++) {
                Metal.Buffer b = scratch("ds" + j, 0);
                deep[j] = new float[tokens * w2.rows()];
                MemorySegment.copy(b.contents(), ValueLayout.JAVA_FLOAT, 0, deep[j], 0, deep[j].length);
            }
        }
        return new Encoded(out, tokens, bw, bh, deep);
    }

    /** Waits for the command buffer and reads the result. */
    private Encoded finish(Metal.Buffer out, int tokens, int dim, int nx, int ny) {
        metal.commitAndWait();
        float[] res = new float[tokens * dim];
        MemorySegment.copy(out.contents(), ValueLayout.JAVA_FLOAT, 0, res, 0, res.length);
        return new Encoded(res, tokens, nx, ny);
    }
}
