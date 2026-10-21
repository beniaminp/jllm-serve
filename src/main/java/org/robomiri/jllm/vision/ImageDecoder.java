package org.robomiri.jllm.vision;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Base64;
import java.util.Iterator;

/**
 * Turns the bytes of a picture file (PNG, JPEG, GIF, BMP: what the JDK's ImageIO reads) into {@link Rgb}. The
 * decoder reads the header first and refuses pictures over {@code maxPixels} before allocating anything, so a
 * small file that claims a huge canvas cannot exhaust memory. Alpha is dropped (like stb_image, which llama.cpp
 * uses), a gray picture is replicated to three channels, a 16-bit one is cut to its high byte, EXIF orientation
 * is not applied.
 */
public final class ImageDecoder {

    /** Default limit on the decoded size of a source picture: 100 megapixels. */
    public static final long MAX_PIXELS = 100_000_000L;

    static {
        if (System.getProperty("java.awt.headless") == null) {
            System.setProperty("java.awt.headless", "true");
        }
        ImageIO.setUseCache(false);
    }

    private ImageDecoder() {
    }

    public static Rgb decode(byte[] bytes) throws IOException {
        return decode(bytes, MAX_PIXELS);
    }

    /** @throws IOException if the bytes are not a picture ImageIO can read, or it has more than {@code maxPixels} pixels */
    public static Rgb decode(byte[] bytes, long maxPixels) throws IOException {
        if (bytes.length == 0) {
            throw new IOException("the image is empty");
        }
        BufferedImage img;
        try (ImageInputStream in = ImageIO.createImageInputStream(new ByteArrayInputStream(bytes))) {
            Iterator<ImageReader> readers = ImageIO.getImageReaders(in);
            if (!readers.hasNext()) {
                throw new IOException("unrecognised image format (supported: PNG, JPEG, GIF, BMP)");
            }
            ImageReader reader = readers.next();
            try {
                reader.setInput(in, true, true);
                long w = reader.getWidth(0);
                long h = reader.getHeight(0);
                if (w <= 0 || h <= 0 || w * h > maxPixels) {
                    throw new IOException("the image is " + w + "x" + h + " pixels, more than the limit of " + maxPixels);
                }
                img = reader.read(0);
            } finally {
                reader.dispose();
            }
        } catch (IOException e) {
            throw e;
        } catch (RuntimeException e) { // ImageIO reports corrupt data in many ways
            throw new IOException("cannot decode the image: " + e.getMessage(), e);
        }
        if (img == null) {
            throw new IOException("cannot decode the image");
        }
        return toRgb(img);
    }

    /** {@code data:image/png;base64,...}: the decoded bytes of a data URL, or throws. */
    public static byte[] dataUrlBytes(String url) throws IOException {
        int comma = url.indexOf(',');
        if (!url.startsWith("data:") || comma < 0) {
            throw new IOException("not a data URL");
        }
        String meta = url.substring(5, comma);
        String payload = url.substring(comma + 1);
        try {
            if (meta.endsWith(";base64")) {
                return Base64.getMimeDecoder().decode(payload);
            }
            return java.net.URLDecoder.decode(payload, java.nio.charset.StandardCharsets.ISO_8859_1).getBytes(
                    java.nio.charset.StandardCharsets.ISO_8859_1);
        } catch (IllegalArgumentException e) {
            throw new IOException("invalid base64 image data: " + e.getMessage(), e);
        }
    }

    static Rgb toRgb(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        byte[] out = new byte[3 * w * h];
        int type = img.getType();
        if (type == BufferedImage.TYPE_BYTE_GRAY || type == BufferedImage.TYPE_USHORT_GRAY) {
            // getRGB would run gray through a color space conversion that changes the values
            Raster r = img.getRaster();
            int shift = type == BufferedImage.TYPE_USHORT_GRAY ? 8 : 0;
            int[] row = new int[w];
            for (int y = 0; y < h; y++) {
                r.getSamples(0, y, w, 1, 0, row);
                for (int x = 0; x < w; x++) {
                    byte v = (byte) (row[x] >> shift);
                    int o = 3 * (y * w + x);
                    out[o] = v;
                    out[o + 1] = v;
                    out[o + 2] = v;
                }
            }
            return new Rgb(w, h, out);
        }
        int[] row = new int[w];
        for (int y = 0; y < h; y++) {
            img.getRGB(0, y, w, 1, row, 0, w);
            for (int x = 0; x < w; x++) {
                int p = row[x];
                int o = 3 * (y * w + x);
                out[o] = (byte) (p >> 16);
                out[o + 1] = (byte) (p >> 8);
                out[o + 2] = (byte) p;
            }
        }
        return new Rgb(w, h, out);
    }
}
