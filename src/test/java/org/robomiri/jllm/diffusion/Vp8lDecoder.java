package org.robomiri.jllm.diffusion;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A decoder for what {@link Vp8l} writes (and the parts of the VP8L format around it: any predictor, subtract green, plain
 * and run-length coded prefix code descriptions, LZ77 with the distance codes 1 and 2), written independently from the encoder
 * from the format specification, so a test can check the encoder by a round trip. Also reads the animated WebP container.
 * Not a general WebP decoder: no colour cache, colour indexing, cross-colour transform or meta prefix images.
 */
final class Vp8lDecoder {

    private final byte[] d;
    private int pos;
    private long acc;
    private int nacc;

    private Vp8lDecoder(byte[] d, int pos) {
        this.d = d;
        this.pos = pos;
    }

    private int bits(int n) {
        while (nacc < n) {
            if (pos >= d.length) {
                throw new IllegalStateException("VP8L data ends early");
            }
            acc |= (long) (d[pos++] & 255) << nacc;
            nacc += 8;
        }
        int v = (int) (acc & ((1L << n) - 1));
        acc >>>= n;
        nacc -= n;
        return v;
    }

    /** A canonical prefix code read from the stream. */
    private static final class Huff {
        final int[] count = new int[16];
        final int[] sorted;
        final int only;

        Huff(int[] lens) {
            int used = 0;
            int last = -1;
            for (int s = 0; s < lens.length; s++) {
                if (lens[s] > 0) {
                    count[lens[s]]++;
                    used++;
                    last = s;
                }
            }
            only = used == 1 ? last : -1;
            sorted = new int[used];
            int[] offs = new int[17];
            for (int l = 1; l < 16; l++) {
                offs[l + 1] = offs[l] + count[l];
            }
            for (int s = 0; s < lens.length; s++) {
                if (lens[s] > 0) {
                    sorted[offs[lens[s]]++] = s;
                }
            }
        }

        int read(Vp8lDecoder in) {
            if (only >= 0) {
                return only;
            }
            int code = 0;
            int first = 0;
            int index = 0;
            for (int len = 1; len <= 15; len++) {
                code |= in.bits(1);
                int c = count[len];
                if (code - c < first) {
                    return sorted[index + (code - first)];
                }
                index += c;
                first += c;
                first <<= 1;
                code <<= 1;
            }
            throw new IllegalStateException("bad prefix code");
        }
    }

    private static final int[] ORDER = {17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    private Huff readCode(int alphabet) {
        int[] lens = new int[alphabet];
        if (bits(1) == 1) {
            int n = bits(1) + 1;
            int first8 = bits(1);
            lens[bits(first8 == 1 ? 8 : 1)] = 1;
            if (n == 2) {
                lens[bits(8)] = 1;
            }
            return new Huff(lens);
        }
        int[] cl = new int[19];
        int num = 4 + bits(4);
        for (int i = 0; i < num; i++) {
            cl[ORDER[i]] = bits(3);
        }
        Huff clh = new Huff(cl);
        int maxSymbol = alphabet;
        if (bits(1) == 1) {
            int nbits = 2 + 2 * bits(3);
            maxSymbol = 2 + bits(nbits);
        }
        int sym = 0;
        int prev = 8;
        while (sym < alphabet) {
            if (maxSymbol-- == 0) {
                break;
            }
            int c = clh.read(this);
            if (c < 16) {
                lens[sym++] = c;
                if (c != 0) {
                    prev = c;
                }
            } else {
                int extra = c == 16 ? 2 : c == 17 ? 3 : 7;
                int add = c == 16 ? 3 : c == 17 ? 3 : 11;
                int rep = bits(extra) + add;
                int value = c == 16 ? prev : 0;
                if (sym + rep > alphabet) {
                    throw new IllegalStateException("code length run past the alphabet");
                }
                while (rep-- > 0) {
                    lens[sym++] = value;
                }
            }
        }
        return new Huff(lens);
    }

    private int value(int symbol) {
        if (symbol < 4) {
            return symbol + 1;
        }
        int extra = (symbol - 2) >> 1;
        int offset = (2 + (symbol & 1)) << extra;
        return offset + bits(extra) + 1;
    }

    private int[] image(int w, int h, boolean main, List<int[]> transforms) {
        if (main) {
            while (bits(1) == 1) {
                int type = bits(2);
                switch (type) {
                    case 2 -> transforms.add(new int[]{2});
                    case 0 -> {
                        int sb = bits(3) + 2;
                        int bw = (w + (1 << sb) - 1) >> sb;
                        int bh = (h + (1 << sb) - 1) >> sb;
                        int[] modes = image(bw, bh, false, null);
                        int[] t = new int[3 + modes.length];
                        t[0] = 0;
                        t[1] = sb;
                        t[2] = bw;
                        System.arraycopy(modes, 0, t, 3, modes.length);
                        transforms.add(t);
                    }
                    default -> throw new IllegalStateException("transform " + type + " not supported by this test decoder");
                }
            }
        }
        if (bits(1) == 1) {
            throw new IllegalStateException("colour cache not supported by this test decoder");
        }
        if (main && bits(1) == 1) {
            throw new IllegalStateException("meta prefix image not supported by this test decoder");
        }
        Huff g = readCode(256 + 24);
        Huff r = readCode(256);
        Huff b = readCode(256);
        Huff a = readCode(256);
        Huff dist = readCode(40);
        int[] px = new int[w * h];
        int i = 0;
        while (i < px.length) {
            int s = g.read(this);
            if (s < 256) {
                int rr = r.read(this);
                int bb = b.read(this);
                int aa = a.read(this);
                px[i++] = (aa << 24) | (rr << 16) | (s << 8) | bb;
            } else {
                int len = value(s - 256);
                int code = value(dist.read(this));
                int d;
                if (code == 1) {
                    d = w;
                } else if (code == 2) {
                    d = 1;
                } else if (code > 120) {
                    d = code - 120;
                } else {
                    throw new IllegalStateException("distance code " + code + " not supported by this test decoder");
                }
                if (d > i || i + len > px.length) {
                    throw new IllegalStateException("bad back reference");
                }
                for (int k = 0; k < len; k++) {
                    px[i] = px[i - d];
                    i++;
                }
            }
        }
        return px;
    }

    private static int add(int a, int b) {
        int out = 0;
        for (int s = 0; s < 32; s += 8) {
            out |= ((((a >>> s) & 255) + ((b >>> s) & 255)) & 255) << s;
        }
        return out;
    }

    /** Decodes a {@code VP8L} chunk payload to RGB bytes; {@code size} receives width and height. */
    static byte[] decode(byte[] chunk, int[] size) {
        Vp8lDecoder in = new Vp8lDecoder(chunk, 0);
        if (in.bits(8) != 0x2f) {
            throw new IllegalStateException("not a VP8L stream");
        }
        int w = in.bits(14) + 1;
        int h = in.bits(14) + 1;
        in.bits(1);
        if (in.bits(3) != 0) {
            throw new IllegalStateException("unknown VP8L version");
        }
        List<int[]> transforms = new ArrayList<>();
        int[] px = in.image(w, h, true, transforms);
        for (int k = transforms.size() - 1; k >= 0; k--) {
            int[] t = transforms.get(k);
            if (t[0] == 2) {
                for (int i = 0; i < px.length; i++) {
                    int p = px[i];
                    int gr = (p >> 8) & 255;
                    int rr = (((p >> 16) & 255) + gr) & 255;
                    int bb = ((p & 255) + gr) & 255;
                    px[i] = (p & 0xff00ff00) | (rr << 16) | bb;
                }
            } else {
                int sb = t[1];
                int bw = t[2];
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int pred;
                        if (x == 0 && y == 0) {
                            pred = 0xff000000;
                        } else if (y == 0) {
                            pred = px[x - 1];
                        } else if (x == 0) {
                            pred = px[(y - 1) * w];
                        } else {
                            int mode = (t[3 + (y >> sb) * bw + (x >> sb)] >> 8) & 15;
                            int tr = x == w - 1 ? px[y * w] : px[(y - 1) * w + x + 1];
                            pred = Vp8l.predict(mode, px[y * w + x - 1], px[(y - 1) * w + x], tr, px[(y - 1) * w + x - 1]);
                        }
                        px[y * w + x] = add(px[y * w + x], pred);
                    }
                }
            }
        }
        byte[] rgb = new byte[w * h * 3];
        for (int i = 0; i < px.length; i++) {
            rgb[i * 3] = (byte) (px[i] >> 16);
            rgb[i * 3 + 1] = (byte) (px[i] >> 8);
            rgb[i * 3 + 2] = (byte) px[i];
        }
        size[0] = w;
        size[1] = h;
        return rgb;
    }

    /** One frame of an animated WebP: its pixels and its duration in milliseconds. */
    record Frame(int width, int height, byte[] rgb, int durationMs) {
    }

    /** Parses an animated (or still) lossless WebP file. */
    static List<Frame> parseWebp(byte[] f) throws IOException {
        if (f.length < 12 || f[0] != 'R' || f[1] != 'I' || f[2] != 'F' || f[3] != 'F' || f[8] != 'W' || f[9] != 'E' || f[10] != 'B' || f[11] != 'P') {
            throw new IOException("not a RIFF WEBP file");
        }
        long riff = le(f, 4, 4);
        if (riff != f.length - 8) {
            throw new IOException("RIFF size " + riff + " does not match the file (" + (f.length - 8) + ")");
        }
        List<Frame> out = new ArrayList<>();
        int p = 12;
        boolean animated = false;
        int cw = 0;
        int ch = 0;
        while (p < f.length) {
            String id = new String(f, p, 4, java.nio.charset.StandardCharsets.US_ASCII);
            int len = (int) le(f, p + 4, 4);
            int body = p + 8;
            switch (id) {
                case "VP8X" -> {
                    animated = (f[body] & 2) != 0;
                    cw = (int) le(f, body + 4, 3) + 1;
                    ch = (int) le(f, body + 7, 3) + 1;
                }
                case "VP8L" -> {
                    int[] sz = new int[2];
                    byte[] rgb = decode(java.util.Arrays.copyOfRange(f, body, body + len), sz);
                    out.add(new Frame(sz[0], sz[1], rgb, 0));
                }
                case "ANMF" -> {
                    int w = (int) le(f, body + 6, 3) + 1;
                    int h = (int) le(f, body + 9, 3) + 1;
                    int dur = (int) le(f, body + 12, 3);
                    if (le(f, body, 3) != 0 || le(f, body + 3, 3) != 0 || w != cw || h != ch) {
                        throw new IOException("frame is not the whole canvas");
                    }
                    String sub = new String(f, body + 16, 4, java.nio.charset.StandardCharsets.US_ASCII);
                    int sublen = (int) le(f, body + 20, 4);
                    if (!sub.equals("VP8L")) {
                        throw new IOException("frame chunk " + sub);
                    }
                    int[] sz = new int[2];
                    byte[] rgb = decode(java.util.Arrays.copyOfRange(f, body + 24, body + 24 + sublen), sz);
                    if (sz[0] != w || sz[1] != h) {
                        throw new IOException("frame size mismatch");
                    }
                    out.add(new Frame(w, h, rgb, dur));
                }
                default -> {
                }
            }
            p = body + len + (len & 1);
        }
        if (animated && out.isEmpty()) {
            throw new IOException("no frames");
        }
        return out;
    }

    private static long le(byte[] f, int off, int n) {
        long v = 0;
        for (int i = n - 1; i >= 0; i--) {
            v = (v << 8) | (f[off + i] & 255);
        }
        return v;
    }
}
