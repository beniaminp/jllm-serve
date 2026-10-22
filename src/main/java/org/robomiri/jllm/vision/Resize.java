package org.robomiri.jllm.vision;

/**
 * Picture resizing for the image preprocessing, a port of llama.cpp's {@code img_tool} (tools/mtmd/mtmd-image.cpp):
 * Pillow-compatible separable resampling (bilinear, bicubic with a = -0.5, Lanczos 3) in 22-bit fixed point, so the
 * pixels a model sees are the ones Hugging Face's processors (which call Pillow) produce, and the ones llama.cpp
 * feeds its own copy of the encoder. Also the aspect-preserving size rules ("smart resize"), padding and cropping.
 */
public final class Resize {

    public enum Algo { BILINEAR, BICUBIC, LANCZOS }

    /**
     * How a picture is fitted into a target size. NONE stretches to the target; CEIL keeps the aspect ratio
     * (rounding the scaled size up) and centres the result on a canvas of the pad colour; NEAREST does the same
     * rounding to the nearest integer.
     */
    public enum Pad { NONE, CEIL, NEAREST }

    private static final int PRECISION_BITS = 32 - 8 - 2;

    private Resize() {
    }

    /** A target size in pixels. */
    public record Size(int width, int height) {
    }

    /**
     * @param alignSize   sizes are multiples of this
     * @param minPixels   0 = no minimum
     * @param maxPixels   0 = no maximum
     * @param longestEdge 0 = off; else the longer side is scaled to this first (before min / max pixels)
     */
    public record SizeRule(int alignSize, int minPixels, int maxPixels, int longestEdge) {
    }

    /**
     * The size of the resized picture when the aspect ratio is kept and both sides are multiples of
     * {@code alignSize} ("smart_resize" of the Qwen-VL processors), within the pixel budget.
     */
    public static Size calcSizePreservedRatio(Size in, SizeRule opts) {
        if (opts.alignSize() <= 0) {
            throw new IllegalArgumentException("align size must be positive");
        }
        int width = in.width();
        int height = in.height();
        if (width <= 0 || height <= 0) {
            return new Size(0, 0);
        }
        int f = opts.alignSize();
        int wBar;
        int hBar;
        if (opts.longestEdge() > 0) {
            float scale = Math.min((float) opts.longestEdge() / width, (float) opts.longestEdge() / height);
            wBar = ceilBy((float) width * scale, f);
            hBar = ceilBy((float) height * scale, f);
        } else {
            wBar = Math.max(f, roundBy((float) width, f));
            hBar = Math.max(f, roundBy((float) height, f));
        }
        if (opts.maxPixels() > 0 && (long) hBar * wBar > opts.maxPixels()) {
            float beta = (float) Math.sqrt((float) height * width / opts.maxPixels());
            hBar = Math.max(f, floorBy(height / beta, f));
            wBar = Math.max(f, floorBy(width / beta, f));
        } else if (opts.minPixels() > 0 && (long) hBar * wBar < opts.minPixels()) {
            float beta = (float) Math.sqrt((float) opts.minPixels() / ((float) height * width));
            hBar = ceilBy(height * beta, f);
            wBar = ceilBy(width * beta, f);
        }
        return new Size(wBar, hBar);
    }

    private static int roundBy(float x, int f) {
        return (int) Math.round((double) (x / (float) f)) * f; // std::round: half away from zero (x >= 0 here)
    }

    private static int ceilBy(float x, int f) {
        return (int) Math.ceil(x / (float) f) * f;
    }

    private static int floorBy(float x, int f) {
        return (int) Math.floor(x / (float) f) * f;
    }

    /** Resizes {@code src} to {@code target} the way {@code img_tool::resize} does. */
    public static Rgb resize(Rgb src, Size target, Algo algo, Pad pad, int[] padColor) {
        if (target.width() == src.width() && target.height() == src.height()) {
            return src;
        }
        if (pad == Pad.NONE) {
            return resizePillow(src, target.width(), target.height(), algo);
        }
        float scaleW = (float) target.width() / src.width();
        float scaleH = (float) target.height() / src.height();
        float scale = Math.min(scaleW, scaleH);
        int newW;
        int newH;
        if (pad == Pad.NEAREST) {
            newW = Math.min(Math.round((float) src.width() * scale), target.width());
            newH = Math.min(Math.round((float) src.height() * scale), target.height());
        } else {
            newW = Math.min((int) Math.ceil((float) src.width() * scale), target.width());
            newH = Math.min((int) Math.ceil((float) src.height() * scale), target.height());
        }
        Rgb resized = resizePillow(src, newW, newH, algo);
        byte[] dst = new byte[3 * target.width() * target.height()];
        if (padColor != null) {
            for (int i = 0; i < dst.length; i += 3) {
                dst[i] = (byte) padColor[0];
                dst[i + 1] = (byte) padColor[1];
                dst[i + 2] = (byte) padColor[2];
            }
        }
        int ox;
        int oy;
        if (pad == Pad.NEAREST) {
            ox = Math.round((target.width() - newW) / 2.0f);
            oy = Math.round((target.height() - newH) / 2.0f);
        } else {
            ox = (target.width() - newW) / 2;
            oy = (target.height() - newH) / 2;
        }
        for (int y = 0; y < newH; y++) {
            int dy = y + oy;
            if (dy < 0 || dy >= target.height()) {
                continue;
            }
            for (int x = 0; x < newW; x++) {
                int dx = x + ox;
                if (dx < 0 || dx >= target.width()) {
                    continue;
                }
                System.arraycopy(resized.data(), 3 * (y * newW + x), dst, 3 * (dy * target.width() + dx), 3);
            }
        }
        return new Rgb(target.width(), target.height(), dst);
    }

    /** The {@code w x h} part of {@code src} at {@code (x, y)}. */
    public static Rgb crop(Rgb src, int x, int y, int w, int h) {
        if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > src.width() || y + h > src.height()) {
            throw new IllegalArgumentException("crop " + w + "x" + h + " at " + x + "," + y + " of " + src.width() + "x" + src.height());
        }
        byte[] d = new byte[3 * w * h];
        for (int i = 0; i < h; i++) {
            System.arraycopy(src.data(), 3 * ((y + i) * src.width() + x), d, 3 * i * w, 3 * w);
        }
        return new Rgb(w, h, d);
    }

    // ------------------------------------------------------------------ Pillow resampling

    private static double filter(Algo algo, double x) {
        if (algo == Algo.LANCZOS) {
            if (-3.0 <= x && x < 3.0) {
                return sinc(x) * sinc(x / 3.0);
            }
            return 0.0;
        }
        if (x < 0.0) {
            x = -x;
        }
        if (algo == Algo.BILINEAR) {
            return x < 1.0 ? 1.0 - x : 0.0;
        }
        final double a = -0.5;
        if (x < 1.0) {
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        }
        if (x < 2.0) {
            return (((x - 5) * x + 8) * x - 4) * a;
        }
        return 0.0;
    }

    private static double sinc(double v) {
        if (v == 0.0) {
            return 1.0;
        }
        double piV = v * Math.PI;
        return Math.sin(piV) / piV;
    }

    /** Filter weights of one dimension in fixed point; {@code bounds[2*xx]} = first input pixel, {@code [2*xx+1]} = count. */
    private record Weights(int ksize, int[] bounds, int[] weights) {
    }

    private static Weights precompute(Algo algo, int inSize, int outSize) {
        double filterSupport = switch (algo) {
            case BILINEAR -> 1.0;
            case BICUBIC -> 2.0;
            case LANCZOS -> 3.0;
        };
        double scale = (double) inSize / outSize;
        double filterscale = Math.max(scale, 1.0);
        double support = filterSupport * filterscale;
        int ksize = (int) Math.ceil(support) * 2 + 1;
        double[] pre = new double[outSize * ksize];
        int[] bounds = new int[outSize * 2];
        for (int xx = 0; xx < outSize; xx++) {
            double center = (xx + 0.5) * scale;
            double ww = 0.0;
            double ss = 1.0 / filterscale;
            int xmin = (int) (center - support + 0.5);
            if (xmin < 0) {
                xmin = 0;
            }
            int xmax = (int) (center + support + 0.5);
            if (xmax > inSize) {
                xmax = inSize;
            }
            xmax -= xmin;
            int x;
            for (x = 0; x < xmax; x++) {
                double w = filter(algo, (x + xmin - center + 0.5) * ss);
                pre[xx * ksize + x] = w;
                ww += w;
            }
            for (x = 0; x < xmax; x++) {
                if (ww != 0.0) {
                    pre[xx * ksize + x] /= ww;
                }
            }
            bounds[xx * 2] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }
        int[] weights = new int[outSize * ksize];
        double fxp = Math.scalb(1.0, PRECISION_BITS);
        for (int i = 0; i < weights.length; i++) {
            double rounded = pre[i] * fxp + (pre[i] < 0 ? -0.5 : 0.5);
            weights[i] = (int) rounded; // truncation toward zero, like Pillow
        }
        return new Weights(ksize, bounds, weights);
    }

    private static byte clip8(int v) {
        return (byte) (v < 0 ? 0 : Math.min(v, 255));
    }

    /** Pillow's two-pass resampling: horizontal, then vertical, each a pass over the pixels with the weights above. */
    static Rgb resizePillow(Rgb img, int tw, int th, Algo algo) {
        if (tw <= 0 || tw > 65536 || th <= 0 || th > 65536) {
            throw new IllegalArgumentException("resize target " + tw + "x" + th + " is out of range (max 65536)");
        }
        int sw = img.width();
        int sh = img.height();
        boolean needH = tw != sw;
        boolean needV = th != sh;
        byte[] cur = img.data();
        int curW = sw;
        if (needH) {
            Weights wh = precompute(algo, sw, tw);
            byte[] out = new byte[tw * sh * 3];
            for (int yy = 0; yy < sh; yy++) {
                int srcRow = yy * sw * 3;
                int dstRow = yy * tw * 3;
                for (int xx = 0; xx < tw; xx++) {
                    int xmin = wh.bounds[xx * 2];
                    int xcnt = wh.bounds[xx * 2 + 1];
                    int kb = xx * wh.ksize;
                    int ss0 = 1 << (PRECISION_BITS - 1);
                    int ss1 = ss0;
                    int ss2 = ss0;
                    int p = srcRow + xmin * 3;
                    for (int x = 0; x < xcnt; x++) {
                        int k = wh.weights[kb + x];
                        ss0 += (cur[p] & 0xFF) * k;
                        ss1 += (cur[p + 1] & 0xFF) * k;
                        ss2 += (cur[p + 2] & 0xFF) * k;
                        p += 3;
                    }
                    out[dstRow + xx * 3] = clip8(ss0 >> PRECISION_BITS);
                    out[dstRow + xx * 3 + 1] = clip8(ss1 >> PRECISION_BITS);
                    out[dstRow + xx * 3 + 2] = clip8(ss2 >> PRECISION_BITS);
                }
            }
            cur = out;
            curW = tw;
        }
        if (needV) {
            Weights wv = precompute(algo, sh, th);
            int rowElems = curW * 3;
            byte[] out = new byte[rowElems * th];
            int[] acc = new int[rowElems];
            for (int yy = 0; yy < th; yy++) {
                int ymin = wv.bounds[yy * 2];
                int ycnt = wv.bounds[yy * 2 + 1];
                java.util.Arrays.fill(acc, 1 << (PRECISION_BITS - 1));
                for (int y = 0; y < ycnt; y++) {
                    int w = wv.weights[yy * wv.ksize + y];
                    int base = (ymin + y) * rowElems;
                    for (int i = 0; i < rowElems; i++) {
                        acc[i] += (cur[base + i] & 0xFF) * w;
                    }
                }
                int dst = yy * rowElems;
                for (int i = 0; i < rowElems; i++) {
                    out[dst + i] = clip8(acc[i] >> PRECISION_BITS);
                }
            }
            cur = out;
        }
        return new Rgb(tw, th, needH || needV ? cur : img.data().clone());
    }
}
