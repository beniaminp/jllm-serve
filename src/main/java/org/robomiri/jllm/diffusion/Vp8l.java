package org.robomiri.jllm.diffusion;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;

/**
 * A lossless WebP (VP8L) encoder, written from the format specification (RFC 9649) so that animated WebP files can be made
 * without any library. The bitstream is coded with the transforms and tools that matter most for pictures:
 *
 * <ul>
 *   <li><b>subtract green</b>: red and blue are replaced by their difference to green (decorrelates the channels);</li>
 *   <li><b>predictor transform</b>: every pixel is replaced by its difference to a prediction from its already coded
 *       neighbours (left, top, top-right, top-left); the image is cut into 16 × 16 blocks and each block uses the one of
 *       the 14 predictors of the format that leaves the smallest residual;</li>
 *   <li><b>LZ77</b> back references to the pixel on the left or above (long runs of equal residuals cost a few bits);</li>
 *   <li>one <b>canonical prefix (Huffman) code</b> per channel: green + length, red, blue, alpha, distance, with code
 *       lengths limited to 15 and written with the format's run-length coding of code lengths.</li>
 * </ul>
 *
 * Not used: colour cache, colour indexing, cross-colour transform, several prefix-code groups.
 * {@link #encode} returns the payload of a {@code VP8L} chunk.
 */
final class Vp8l {

    private Vp8l() {
    }

    private static final int BLOCK_BITS = 4;
    private static final int MAX_CODE_LENGTH = 15;
    private static final int[] CODE_LENGTH_ORDER = {17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    // ------------------------------------------------------------------ bit writer (least significant bit first)

    private static final class Bits {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();
        private long acc;
        private int n;

        void put(int value, int bits) {
            if (bits == 0) {
                return;
            }
            acc |= ((long) value & ((1L << bits) - 1)) << n;
            n += bits;
            while (n >= 8) {
                out.write((int) (acc & 255));
                acc >>>= 8;
                n -= 8;
            }
        }

        byte[] finish() {
            if (n > 0) {
                out.write((int) (acc & 255));
                acc = 0;
                n = 0;
            }
            return out.toByteArray();
        }
    }

    // ------------------------------------------------------------------ prefix codes

    /** A canonical prefix code: lengths per symbol and the codes, bit-reversed for the LSB-first stream. */
    private static final class Code {
        final int[] lens;
        final int[] codes;
        /** exactly one symbol in use: it costs zero bits */
        final boolean single;

        Code(int[] lens, int maxLen) {
            this.lens = lens;
            int used = 0;
            for (int l : lens) {
                if (l > 0) {
                    used++;
                }
            }
            single = used == 1;
            codes = new int[lens.length];
            int[] count = new int[maxLen + 2];
            for (int l : lens) {
                count[l]++;
            }
            count[0] = 0;
            int[] next = new int[maxLen + 2];
            int code = 0;
            for (int l = 1; l <= maxLen; l++) {
                code = (code + count[l - 1]) << 1;
                next[l] = code;
            }
            for (int s = 0; s < lens.length; s++) {
                int l = lens[s];
                if (l > 0) {
                    int c = next[l]++;
                    int r = 0;
                    for (int i = 0; i < l; i++) { // reverse: the format reads code bits most significant first from an LSB-first stream
                        r = (r << 1) | ((c >> i) & 1);
                    }
                    codes[s] = r;
                }
            }
        }

        void write(Bits b, int sym) {
            if (single) {
                return;
            }
            b.put(codes[sym], lens[sym]);
        }
    }

    /** Code lengths for the given symbol counts, none longer than {@code limit}. */
    static int[] lengths(int[] counts, int limit) {
        int n = counts.length;
        int[] lens = new int[n];
        int used = 0;
        for (int c : counts) {
            if (c > 0) {
                used++;
            }
        }
        if (used == 0) {
            return lens;
        }
        if (used == 1) {
            for (int i = 0; i < n; i++) {
                if (counts[i] > 0) {
                    lens[i] = 1;
                }
            }
            return lens;
        }
        for (long floor = 1; ; floor *= 2) { // flatten the distribution until the longest code fits
            long[] w = new long[n];
            for (int i = 0; i < n; i++) {
                w[i] = counts[i] > 0 ? Math.max(counts[i], floor) : 0;
            }
            int max = huffman(w, lens);
            if (max <= limit) {
                return lens;
            }
        }
    }

    private static int huffman(long[] w, int[] lens) {
        int n = w.length;
        Arrays.fill(lens, 0);
        // nodes: leaves 0..n-1, internal n.. ; parent[] gives the tree
        PriorityQueue<long[]> pq = new PriorityQueue<>((a, b) -> a[0] != b[0] ? Long.compare(a[0], b[0]) : Long.compare(a[1], b[1]));
        int[] parent = new int[2 * n];
        Arrays.fill(parent, -1);
        for (int i = 0; i < n; i++) {
            if (w[i] > 0) {
                pq.add(new long[]{w[i], i});
            }
        }
        int next = n;
        while (pq.size() > 1) {
            long[] a = pq.poll();
            long[] b = pq.poll();
            parent[(int) a[1]] = next;
            parent[(int) b[1]] = next;
            pq.add(new long[]{a[0] + b[0], next});
            next++;
        }
        int max = 0;
        for (int i = 0; i < n; i++) {
            if (w[i] > 0) {
                int d = 0;
                for (int p = parent[i]; p >= 0; p = parent[p]) {
                    d++;
                }
                lens[i] = Math.max(d, 1);
                max = Math.max(max, lens[i]);
            }
        }
        return max;
    }

    /** Writes the description of a prefix code over {@code lens.length} symbols. */
    private static void writeCode(Bits b, int[] lens) {
        List<Integer> used = new ArrayList<>();
        for (int i = 0; i < lens.length; i++) {
            if (lens[i] > 0) {
                used.add(i);
            }
        }
        if (used.isEmpty() || (used.size() <= 2 && used.get(used.size() - 1) < 256)) { // "simple" code: one or two symbols below 256
            b.put(1, 1);
            int count = Math.max(1, used.size());
            b.put(count - 1, 1);
            int s0 = used.isEmpty() ? 0 : used.get(0);
            if (s0 <= 1) {
                b.put(0, 1);
                b.put(s0, 1);
            } else {
                b.put(1, 1);
                b.put(s0, 8);
            }
            if (count == 2) {
                b.put(used.get(1), 8);
            }
            return;
        }
        b.put(0, 1);
        // run-length tokens of the code lengths: (symbol, extra bits value, extra bit count)
        List<int[]> tokens = new ArrayList<>();
        int n = lens.length;
        int i = 0;
        int prev = 8; // the initial "previous non-zero length" of the format
        while (i < n) {
            int l = lens[i];
            int run = 1;
            while (i + run < n && lens[i + run] == l) {
                run++;
            }
            if (l == 0) {
                int left = run;
                while (left >= 11) {
                    int r = Math.min(left, 138);
                    tokens.add(new int[]{18, r - 11, 7});
                    left -= r;
                }
                if (left >= 3) {
                    tokens.add(new int[]{17, left - 3, 3});
                    left = 0;
                }
                for (; left > 0; left--) {
                    tokens.add(new int[]{0, 0, 0});
                }
            } else {
                int left = run;
                if (l != prev) {
                    tokens.add(new int[]{l, 0, 0});
                    prev = l;
                    left--;
                }
                while (left >= 3) {
                    int r = Math.min(left, 6);
                    tokens.add(new int[]{16, r - 3, 2});
                    left -= r;
                }
                for (; left > 0; left--) {
                    tokens.add(new int[]{l, 0, 0});
                }
            }
            i += run;
        }
        int[] clCounts = new int[19];
        for (int[] t : tokens) {
            clCounts[t[0]]++;
        }
        int[] clLens = lengths(clCounts, 7);
        Code cl = new Code(clLens, 7);
        int last = 3;
        for (int k = 0; k < 19; k++) {
            if (clLens[CODE_LENGTH_ORDER[k]] > 0) {
                last = Math.max(last, k);
            }
        }
        b.put(last + 1 - 4, 4);
        for (int k = 0; k <= last; k++) {
            b.put(clLens[CODE_LENGTH_ORDER[k]], 3);
        }
        b.put(0, 1); // max_symbol = the alphabet size
        for (int[] t : tokens) {
            cl.write(b, t[0]);
            b.put(t[1], t[2]);
        }
    }

    // ------------------------------------------------------------------ LZ77 value coding

    /** Prefix symbol and extra bits of a length or distance value (the format's {@code PrefixEncodeBits}). */
    private static int[] prefix(int value) {
        if (value <= 4) {
            return new int[]{value - 1, 0, 0};
        }
        int v = value - 1;
        int highest = 31 - Integer.numberOfLeadingZeros(v);
        int second = (v >> (highest - 1)) & 1;
        int extraBits = highest - 1;
        return new int[]{2 * highest + second, extraBits, v & ((1 << extraBits) - 1)};
    }

    // ------------------------------------------------------------------ pixel coding

    /** A coded image: for each position either a literal (back = 0) or a copy of {@code len} pixels from {@code dist} pixels back. */
    private record Tok(int pixel, int len, int distCode) {
    }

    private static List<Tok> tokenize(int[] px, int w, int h) {
        List<Tok> out = new ArrayList<>();
        int n = w * h;
        int i = 0;
        while (i < n) {
            int bestLen = 0;
            int bestDist = 0;
            int bestCode = 0;
            if (i >= 1) {
                int l = 0;
                while (i + l < n && l < 4096 && px[i + l] == px[i + l - 1]) {
                    l++;
                }
                if (l > bestLen) {
                    bestLen = l;
                    bestDist = 1;
                    bestCode = 2; // plane code 2 = (1, 0): the pixel on the left
                }
            }
            if (i >= w && w > 1) {
                int l = 0;
                while (i + l < n && l < 4096 && px[i + l] == px[i + l - w]) {
                    l++;
                }
                if (l > bestLen) {
                    bestLen = l;
                    bestDist = w;
                    bestCode = 1; // plane code 1 = (0, 1): the pixel above
                }
            }
            if (bestLen >= 4) {
                out.add(new Tok(0, bestLen, bestCode));
                i += bestLen;
            } else {
                out.add(new Tok(px[i], 0, 0));
                i++;
            }
        }
        return out;
    }

    /**
     * Writes an ARGB image with the format's entropy coding: no colour cache, [no] meta prefix image, five prefix codes, then the
     * pixels.
     */
    private static void writeImage(Bits b, int[] px, int w, int h, boolean main) {
        b.put(0, 1); // no colour cache
        if (main) {
            b.put(0, 1); // one prefix-code group for the whole image
        }
        List<Tok> toks = tokenize(px, w, h);
        int[] cg = new int[256 + 24];
        int[] cr = new int[256];
        int[] cb = new int[256];
        int[] ca = new int[256];
        int[] cd = new int[40];
        for (Tok t : toks) {
            if (t.len == 0) {
                cg[(t.pixel >> 8) & 255]++;
                cr[(t.pixel >> 16) & 255]++;
                cb[t.pixel & 255]++;
                ca[(t.pixel >>> 24) & 255]++;
            } else {
                cg[256 + prefix(t.len)[0]]++;
                cd[prefix(t.distCode)[0]]++;
            }
        }
        int[] lg = lengths(cg, MAX_CODE_LENGTH);
        int[] lr = lengths(cr, MAX_CODE_LENGTH);
        int[] lb = lengths(cb, MAX_CODE_LENGTH);
        int[] la = lengths(ca, MAX_CODE_LENGTH);
        int[] ld = lengths(cd, MAX_CODE_LENGTH);
        writeCode(b, lg);
        writeCode(b, lr);
        writeCode(b, lb);
        writeCode(b, la);
        writeCode(b, ld);
        Code g = new Code(lg, MAX_CODE_LENGTH);
        Code r = new Code(lr, MAX_CODE_LENGTH);
        Code bl = new Code(lb, MAX_CODE_LENGTH);
        Code a = new Code(la, MAX_CODE_LENGTH);
        Code d = new Code(ld, MAX_CODE_LENGTH);
        for (Tok t : toks) {
            if (t.len == 0) {
                g.write(b, (t.pixel >> 8) & 255);
                r.write(b, (t.pixel >> 16) & 255);
                bl.write(b, t.pixel & 255);
                a.write(b, (t.pixel >>> 24) & 255);
            } else {
                int[] p = prefix(t.len);
                g.write(b, 256 + p[0]);
                b.put(p[2], p[1]);
                int[] q = prefix(t.distCode);
                d.write(b, q[0]);
                b.put(q[2], q[1]);
            }
        }
    }

    // ------------------------------------------------------------------ predictors

    private static int avg2(int a, int b) {
        return (((a ^ b) & 0xfefefefe) >>> 1) + (a & b);
    }

    private static int clamp(int v) {
        return v < 0 ? 0 : Math.min(v, 255);
    }

    private static int select(int l, int t, int tl) {
        int pl = 0;
        int pt = 0;
        for (int s = 0; s < 32; s += 8) {
            int cl = (l >>> s) & 255;
            int ct = (t >>> s) & 255;
            int ctl = (tl >>> s) & 255;
            int p = cl + ct - ctl;
            pl += Math.abs(p - cl);
            pt += Math.abs(p - ct);
        }
        return pl < pt ? l : t;
    }

    private static int addSubFull(int a, int b, int c) {
        int out = 0;
        for (int s = 0; s < 32; s += 8) {
            out |= clamp(((a >>> s) & 255) + ((b >>> s) & 255) - ((c >>> s) & 255)) << s;
        }
        return out;
    }

    private static int addSubHalf(int a, int b) {
        int out = 0;
        for (int s = 0; s < 32; s += 8) {
            int ca = (a >>> s) & 255;
            int cb = (b >>> s) & 255;
            out |= clamp(ca + (ca - cb) / 2) << s;
        }
        return out;
    }

    static int predict(int mode, int l, int t, int tr, int tl) {
        return switch (mode) {
            case 0 -> 0xff000000;
            case 1 -> l;
            case 2 -> t;
            case 3 -> tr;
            case 4 -> tl;
            case 5 -> avg2(avg2(l, tr), t);
            case 6 -> avg2(l, tl);
            case 7 -> avg2(l, t);
            case 8 -> avg2(tl, t);
            case 9 -> avg2(t, tr);
            case 10 -> avg2(avg2(l, tl), avg2(t, tr));
            case 11 -> select(l, t, tl);
            case 12 -> addSubFull(l, t, tl);
            case 13 -> addSubHalf(avg2(l, t), tl);
            default -> throw new IllegalArgumentException("predictor " + mode);
        };
    }

    private static int sub(int a, int b) {
        int out = 0;
        for (int s = 0; s < 32; s += 8) {
            out |= (((a >>> s) - (b >>> s)) & 255) << s;
        }
        return out;
    }

    private static int cost(int residual) {
        int c = 0;
        for (int s = 0; s < 32; s += 8) {
            int v = (residual >>> s) & 255;
            c += Math.min(v, 256 - v);
        }
        return c;
    }

    /** The mode (0..13) of every 16 × 16 block, and the residual image. */
    private static int[] predictorModes(int[] px, int w, int h, int bw, int bh) {
        int bs = 1 << BLOCK_BITS;
        int[] modes = new int[bw * bh];
        for (int by = 0; by < bh; by++) {
            for (int bx = 0; bx < bw; bx++) {
                long[] total = new long[14];
                for (int y = by * bs; y < Math.min(h, (by + 1) * bs); y++) {
                    for (int x = bx * bs; x < Math.min(w, (bx + 1) * bs); x++) {
                        if (x == 0 || y == 0) {
                            continue; // the first row and column have fixed predictors
                        }
                        int l = px[y * w + x - 1];
                        int t = px[(y - 1) * w + x];
                        int tl = px[(y - 1) * w + x - 1];
                        int tr = x == w - 1 ? px[y * w] : px[(y - 1) * w + x + 1]; // rightmost column: the leftmost pixel of this row
                        int p = px[y * w + x];
                        for (int m = 0; m < 14; m++) {
                            total[m] += cost(sub(p, predict(m, l, t, tr, tl)));
                        }
                    }
                }
                int best = 1;
                for (int m = 0; m < 14; m++) {
                    if (total[m] < total[best]) {
                        best = m;
                    }
                }
                modes[by * bw + bx] = best;
            }
        }
        return modes;
    }

    static int[] residuals(int[] px, int w, int h, int[] modes, int bw) {
        int[] out = new int[px.length];
        int bs = 1 << BLOCK_BITS;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int pred;
                if (y == 0 && x == 0) {
                    pred = 0xff000000;
                } else if (y == 0) {
                    pred = px[x - 1];
                } else if (x == 0) {
                    pred = px[(y - 1) * w];
                } else {
                    int mode = modes[(y / bs) * bw + x / bs];
                    int tr = x == w - 1 ? px[y * w] : px[(y - 1) * w + x + 1];
                    pred = predict(mode, px[y * w + x - 1], px[(y - 1) * w + x], tr, px[(y - 1) * w + x - 1]);
                }
                out[y * w + x] = sub(px[y * w + x], pred);
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ entry point

    /**
     * Encodes an opaque RGB picture as the payload of a {@code VP8L} chunk.
     *
     * @param rgb {@code rgb[(y·w + x)·3 + c]}
     */
    static byte[] encode(int w, int h, byte[] rgb) {
        if (w < 1 || h < 1 || w > 16384 || h > 16384) {
            throw new IllegalArgumentException("WebP pictures are 1..16384 pixels on a side, got " + w + "x" + h);
        }
        int[] px = new int[w * h];
        for (int i = 0; i < px.length; i++) {
            int r = rgb[i * 3] & 255;
            int g = rgb[i * 3 + 1] & 255;
            int bl = rgb[i * 3 + 2] & 255;
            // subtract green: red and blue relative to green
            px[i] = 0xff000000 | (((r - g) & 255) << 16) | (g << 8) | ((bl - g) & 255);
        }
        int bs = 1 << BLOCK_BITS;
        int bw = (w + bs - 1) / bs;
        int bh = (h + bs - 1) / bs;
        int[] modes = predictorModes(px, w, h, bw, bh);
        int[] res = residuals(px, w, h, modes, bw);
        int[] modeImage = new int[bw * bh];
        for (int i = 0; i < modeImage.length; i++) {
            modeImage[i] = 0xff000000 | (modes[i] << 8);
        }
        Bits b = new Bits();
        b.put(0x2f, 8);
        b.put(w - 1, 14);
        b.put(h - 1, 14);
        b.put(0, 1); // alpha_is_used
        b.put(0, 3); // version
        b.put(1, 1); // a transform follows
        b.put(2, 2); //   SUBTRACT_GREEN
        b.put(1, 1);
        b.put(0, 2); //   PREDICTOR
        b.put(BLOCK_BITS - 2, 3);
        writeImage(b, modeImage, bw, bh, false);
        b.put(0, 1); // no more transforms
        writeImage(b, res, w, h, true);
        return b.finish();
    }
}
