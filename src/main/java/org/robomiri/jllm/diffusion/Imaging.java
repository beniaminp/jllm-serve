package org.robomiri.jllm.diffusion;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

/** Pixels in and out: PNG/JPEG decoding and PNG encoding (the JDK's ImageIO) and resampling. */
public final class Imaging {

    static {
        System.setProperty("java.awt.headless", "true");
    }

    private Imaging() {
    }

    /** An 8-bit RGB image, {@code rgb[(y·width + x)·3 + c]}. */
    public record Image(int width, int height, byte[] rgb) {
        public Image {
            if (rgb.length != width * height * 3) {
                throw new IllegalArgumentException("rgb has " + rgb.length + " bytes for " + width + "x" + height);
            }
        }
    }

    /** Decodes a PNG / JPEG / GIF / BMP; transparent pixels are composited over white. */
    public static Image decode(byte[] data) throws IOException {
        BufferedImage bi = read(data);
        int w = bi.getWidth();
        int h = bi.getHeight();
        byte[] rgb = new byte[w * h * 3];
        int[] px = bi.getRGB(0, 0, w, h, null, 0, w);
        for (int i = 0; i < px.length; i++) {
            int p = px[i];
            int a = p >>> 24;
            int r = (p >> 16) & 255;
            int g = (p >> 8) & 255;
            int b = p & 255;
            if (a < 255) {
                r = (r * a + 255 * (255 - a) + 127) / 255;
                g = (g * a + 255 * (255 - a) + 127) / 255;
                b = (b * a + 255 * (255 - a) + 127) / 255;
            }
            rgb[i * 3] = (byte) r;
            rgb[i * 3 + 1] = (byte) g;
            rgb[i * 3 + 2] = (byte) b;
        }
        return new Image(w, h, rgb);
    }

    private static BufferedImage read(byte[] data) throws IOException {
        BufferedImage bi = ImageIO.read(new ByteArrayInputStream(data));
        if (bi == null) {
            throw new IOException("not a supported image (PNG, JPEG, GIF or BMP expected)");
        }
        return bi;
    }

    /**
     * Decodes an inpainting mask to values in 0..1 where 1 means "repaint here". A picture with transparent pixels is
     * read as OpenAI's edit masks are (transparent = repaint); otherwise white = repaint, black = keep.
     */
    public static float[] decodeMask(byte[] data, int[] sizeOut) throws IOException {
        BufferedImage bi = read(data);
        int w = bi.getWidth();
        int h = bi.getHeight();
        int[] px = bi.getRGB(0, 0, w, h, null, 0, w);
        boolean hasAlpha = false;
        for (int p : px) {
            if ((p >>> 24) < 255) {
                hasAlpha = true;
                break;
            }
        }
        float[] m = new float[w * h];
        for (int i = 0; i < m.length; i++) {
            int p = px[i];
            if (hasAlpha) {
                m[i] = 1f - (p >>> 24) / 255f;
            } else {
                m[i] = (((p >> 16) & 255) * 0.299f + ((p >> 8) & 255) * 0.587f + (p & 255) * 0.114f) / 255f;
            }
        }
        sizeOut[0] = w;
        sizeOut[1] = h;
        return m;
    }

    public static byte[] png(Image img) {
        BufferedImage bi = new BufferedImage(img.width(), img.height(), BufferedImage.TYPE_INT_RGB);
        int[] px = new int[img.width() * img.height()];
        for (int i = 0; i < px.length; i++) {
            px[i] = ((img.rgb()[i * 3] & 255) << 16) | ((img.rgb()[i * 3 + 1] & 255) << 8) | (img.rgb()[i * 3 + 2] & 255);
        }
        bi.setRGB(0, 0, img.width(), img.height(), px, 0, img.width());
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try {
            ImageIO.write(bi, "png", out);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        return out.toByteArray();
    }

    // ------------------------------------------------------------------ resampling

    /**
     * Resamples {@code src} (interleaved, {@code ch} values per pixel) to {@code dw × dh} with a bicubic (Catmull–Rom)
     * kernel; when shrinking, the kernel is widened by the scale so that every source pixel counts (no aliasing).
     */
    public static float[] resize(float[] src, int sw, int sh, int ch, int dw, int dh) {
        if (sw == dw && sh == dh) {
            return src.clone();
        }
        float[] tmp = resizeAxis(src, sw, sh, ch, dw, true);
        return resizeAxis(tmp, dw, sh, ch, dh, false);
    }

    private static float[] resizeAxis(float[] src, int w, int h, int ch, int newSize, boolean horizontal) {
        int oldSize = horizontal ? w : h;
        double scale = (double) oldSize / newSize;
        double filterScale = Math.max(scale, 1.0);
        double support = 2.0 * filterScale;
        int outW = horizontal ? newSize : w;
        int outH = horizontal ? h : newSize;
        float[] out = new float[outW * outH * ch];
        for (int o = 0; o < newSize; o++) {
            double center = (o + 0.5) * scale;
            int lo = (int) Math.max(0, Math.floor(center - support));
            int hi = (int) Math.min(oldSize, Math.ceil(center + support));
            double[] wt = new double[hi - lo];
            double sum = 0;
            for (int i = lo; i < hi; i++) {
                double wv = cubic((i + 0.5 - center) / filterScale);
                wt[i - lo] = wv;
                sum += wv;
            }
            for (int i = 0; i < wt.length; i++) {
                wt[i] /= sum;
            }
            if (horizontal) {
                for (int y = 0; y < h; y++) {
                    for (int c = 0; c < ch; c++) {
                        double acc = 0;
                        for (int i = lo; i < hi; i++) {
                            acc += wt[i - lo] * src[(y * w + i) * ch + c];
                        }
                        out[(y * outW + o) * ch + c] = (float) acc;
                    }
                }
            } else {
                for (int x = 0; x < w; x++) {
                    for (int c = 0; c < ch; c++) {
                        double acc = 0;
                        for (int i = lo; i < hi; i++) {
                            acc += wt[i - lo] * src[(i * w + x) * ch + c];
                        }
                        out[(o * outW + x) * ch + c] = (float) acc;
                    }
                }
            }
        }
        return out;
    }

    /** Catmull–Rom kernel (a = −0.5). */
    private static double cubic(double x) {
        x = Math.abs(x);
        if (x < 1) {
            return 1.5 * x * x * x - 2.5 * x * x + 1;
        }
        if (x < 2) {
            return -0.5 * x * x * x + 2.5 * x * x - 4 * x + 2;
        }
        return 0;
    }

    /** Area-average downsampling of a one-channel map by integer-ish factors (used for latent-size masks). */
    public static float[] areaResize(float[] src, int sw, int sh, int dw, int dh) {
        float[] out = new float[dw * dh];
        for (int y = 0; y < dh; y++) {
            int y0 = (int) ((long) y * sh / dh);
            int y1 = Math.max(y0 + 1, (int) ((long) (y + 1) * sh / dh));
            for (int x = 0; x < dw; x++) {
                int x0 = (int) ((long) x * sw / dw);
                int x1 = Math.max(x0 + 1, (int) ((long) (x + 1) * sw / dw));
                double acc = 0;
                for (int yy = y0; yy < y1; yy++) {
                    for (int xx = x0; xx < x1; xx++) {
                        acc += src[yy * sw + xx];
                    }
                }
                out[y * dw + x] = (float) (acc / ((y1 - y0) * (x1 - x0)));
            }
        }
        return out;
    }

    /** RGB bytes to floats in -1..1 (the VAE's input range). */
    public static float[] toUnit(Image img) {
        float[] f = new float[img.rgb().length];
        for (int i = 0; i < f.length; i++) {
            f[i] = (img.rgb()[i] & 255) / 127.5f - 1f;
        }
        return f;
    }

    /** Floats in -1..1 (roughly) to RGB bytes, clamped. */
    public static Image fromUnit(float[] f, int w, int h) {
        byte[] rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.length; i++) {
            float v = f[i] / 2f + 0.5f;
            rgb[i] = (byte) Math.max(0, Math.min(255, Math.round(v * 255f)));
        }
        return new Image(w, h, rgb);
    }

    /** Resizes an RGB image (bicubic, antialiased). */
    public static Image resize(Image img, int w, int h) {
        if (img.width() == w && img.height() == h) {
            return img;
        }
        float[] f = new float[img.rgb().length];
        for (int i = 0; i < f.length; i++) {
            f[i] = img.rgb()[i] & 255;
        }
        float[] r = resize(f, img.width(), img.height(), 3, w, h);
        byte[] out = new byte[w * h * 3];
        for (int i = 0; i < out.length; i++) {
            out[i] = (byte) Math.max(0, Math.min(255, Math.round(r[i])));
        }
        return new Image(w, h, out);
    }
}
