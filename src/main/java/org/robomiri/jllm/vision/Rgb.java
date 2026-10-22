package org.robomiri.jllm.vision;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/**
 * A decoded picture: {@code width x height} pixels, 3 bytes (R, G, B) each, row by row. This is what the encoders'
 * preprocessing starts from, whatever file format the picture came in.
 */
public record Rgb(int width, int height, byte[] data) {

    public Rgb {
        if (width <= 0 || height <= 0 || data.length != 3 * width * height) {
            throw new IllegalArgumentException("a " + width + "x" + height + " RGB image needs " + 3L * width * height
                    + " bytes, got " + data.length);
        }
    }

    /** SHA-256 of the size and the pixels: identifies the picture for caches (two equal pictures share it). */
    public String hash() {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            md.update((width + "x" + height + ":").getBytes(java.nio.charset.StandardCharsets.US_ASCII));
            md.update(data);
            return HexFormat.of().formatHex(md.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    public int r(int x, int y) {
        return data[3 * (y * width + x)] & 0xFF;
    }

    public int g(int x, int y) {
        return data[3 * (y * width + x) + 1] & 0xFF;
    }

    public int b(int x, int y) {
        return data[3 * (y * width + x) + 2] & 0xFF;
    }
}
