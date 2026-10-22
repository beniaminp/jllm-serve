package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.diffusion.Imaging.Image;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

/**
 * Turns frames into a video file: an animated GIF or an animated lossless WebP, both written here without any library, or an MP4
 * (H.264) made by an external {@code ffmpeg} when there is one.
 *
 * <ul>
 *   <li><b>GIF</b>: one 256-colour palette for the whole clip (median cut on a sample of all frames, so colours do not flicker from
 *       frame to frame), Floyd–Steinberg dithering, LZW; loops forever; delays are in centiseconds, rounded so that the total is right.</li>
 *   <li><b>WebP</b>: the {@code VP8X} / {@code ANIM} / {@code ANMF} container with one {@link Vp8l lossless} frame each; loops forever;
 *       delays in milliseconds. Lossless means big (hundreds of kB per 512 × 512 frame): prefer MP4 for sharing.</li>
 *   <li><b>MP4</b>: raw RGB frames piped into {@code ffmpeg -c:v libx264 -pix_fmt yuv420p} (or {@code mpeg4} where libx264 is missing).</li>
 * </ul>
 */
public final class VideoEncoder {

    /** The file formats. */
    public enum Format {
        MP4("mp4", "video/mp4"), WEBP("webp", "image/webp"), GIF("gif", "image/gif");

        public final String ext;
        public final String mime;

        Format(String ext, String mime) {
            this.ext = ext;
            this.mime = mime;
        }

        /** {@code mp4}, {@code webp}, {@code gif}; {@code auto} (or empty) is MP4 when ffmpeg is usable, else WebP. */
        public static Format parse(String s, boolean ffmpegAvailable) {
            String n = s == null ? "" : s.trim().toLowerCase(Locale.ROOT);
            return switch (n) {
                case "", "auto" -> ffmpegAvailable ? MP4 : WEBP;
                case "mp4", "h264" -> MP4;
                case "webp", "awebp" -> WEBP;
                case "gif" -> GIF;
                default -> throw new IllegalArgumentException("unknown video format '" + s + "' (mp4, webp, gif, auto)");
            };
        }
    }

    private VideoEncoder() {
    }

    /** Encodes the frames (all the same size). {@code ffmpeg} is the executable for MP4 (from {@link #resolveFfmpeg}), may be null otherwise. */
    public static byte[] encode(List<Image> frames, double fps, Format format, String ffmpeg) throws IOException {
        check(frames, fps);
        return switch (format) {
            case GIF -> gif(frames, fps);
            case WEBP -> webp(frames, fps);
            case MP4 -> mp4(frames, fps, ffmpeg);
        };
    }

    private static void check(List<Image> frames, double fps) {
        if (frames.isEmpty()) {
            throw new IllegalArgumentException("no frames");
        }
        if (!(fps > 0 && fps <= 240)) {
            throw new IllegalArgumentException("fps must be above 0 and at most 240");
        }
        Image f0 = frames.get(0);
        for (Image f : frames) {
            if (f.width() != f0.width() || f.height() != f0.height()) {
                throw new IllegalArgumentException("all frames must have the same size");
            }
        }
    }

    /** The duration of frame {@code i} in units of {@code 1/scale} s, rounded so that the running total stays exact. */
    static int delay(int i, double fps, int scale) {
        return (int) (Math.round((i + 1) * scale / fps) - Math.round(i * scale / fps));
    }

    // ------------------------------------------------------------------ ffmpeg

    /**
     * The ffmpeg executable for a setting: {@code off} = null, {@code auto} (or empty) = {@code ffmpeg} on the PATH if it runs,
     * anything else = that path if it runs; null when it does not.
     */
    public static String resolveFfmpeg(String setting) {
        String s = setting == null ? "auto" : setting.trim();
        if (s.equalsIgnoreCase("off")) {
            return null;
        }
        String exe = s.isEmpty() || s.equalsIgnoreCase("auto") ? "ffmpeg" : s;
        try {
            Process p = new ProcessBuilder(exe, "-hide_banner", "-version").redirectErrorStream(true).start();
            p.getOutputStream().close();
            p.getInputStream().readAllBytes();
            if (!p.waitFor(10, TimeUnit.SECONDS)) {
                p.destroyForcibly();
                return null;
            }
            return p.exitValue() == 0 ? exe : null;
        } catch (IOException e) {
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

    public static byte[] mp4(List<Image> frames, double fps, String ffmpeg) throws IOException {
        if (ffmpeg == null) {
            throw new IOException("MP4 output needs ffmpeg: install it (video.ffmpeg=auto finds it on the PATH) or use webp / gif");
        }
        check(frames, fps);
        Path out = Files.createTempFile("jllm-video-", ".mp4");
        try {
            IOException first = null;
            for (String codec : new String[]{"libx264", "mpeg4"}) {
                try {
                    runFfmpeg(frames, fps, ffmpeg, codec, out);
                    return Files.readAllBytes(out);
                } catch (IOException e) {
                    if (first == null) {
                        first = e;
                    }
                }
            }
            throw first;
        } finally {
            Files.deleteIfExists(out);
        }
    }

    private static void runFfmpeg(List<Image> frames, double fps, String ffmpeg, String codec, Path out) throws IOException {
        Image f0 = frames.get(0);
        String[] codecArgs = codec.equals("libx264")
                ? new String[]{"-c:v", "libx264", "-preset", "medium", "-crf", "17"}
                : new String[]{"-c:v", "mpeg4", "-q:v", "2"};
        java.util.ArrayList<String> cmd = new java.util.ArrayList<>(List.of(ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f0.width() + "x" + f0.height(), "-framerate", String.valueOf(fps), "-i", "-",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"));
        cmd.addAll(Arrays.asList(codecArgs));
        cmd.addAll(List.of("-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", "-f", "mp4", out.toString()));
        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        ByteArrayOutputStream log = new ByteArrayOutputStream();
        Thread drain = Thread.ofVirtual().start(() -> {
            try (InputStream in = p.getInputStream()) {
                in.transferTo(log);
            } catch (IOException ignored) {
                // the process ended
            }
        });
        try (OutputStream in = p.getOutputStream()) {
            for (Image f : frames) {
                in.write(f.rgb());
            }
        } catch (IOException e) {
            // ffmpeg quit early: its message below explains
        }
        try {
            if (!p.waitFor(300, TimeUnit.SECONDS)) {
                p.destroyForcibly();
                throw new IOException("ffmpeg did not finish within 300 s");
            }
            drain.join(2000);
        } catch (InterruptedException e) {
            p.destroyForcibly();
            Thread.currentThread().interrupt();
            throw new IOException("interrupted while encoding");
        }
        if (p.exitValue() != 0) {
            throw new IOException("ffmpeg (" + codec + ") failed: " + log.toString(StandardCharsets.UTF_8).strip());
        }
    }

    // ------------------------------------------------------------------ animated WebP

    private static void le16(ByteArrayOutputStream o, int v) {
        o.write(v & 255);
        o.write((v >> 8) & 255);
    }

    private static void le24(ByteArrayOutputStream o, int v) {
        le16(o, v);
        o.write((v >> 16) & 255);
    }

    private static void le32(ByteArrayOutputStream o, long v) {
        le16(o, (int) (v & 0xffff));
        le16(o, (int) ((v >> 16) & 0xffff));
    }

    private static void chunk(ByteArrayOutputStream o, String fourcc, byte[] data) {
        o.writeBytes(fourcc.getBytes(StandardCharsets.US_ASCII));
        le32(o, data.length);
        o.writeBytes(data);
        if ((data.length & 1) == 1) {
            o.write(0);
        }
    }

    /** An animated lossless WebP (loops forever). A single frame gives a plain still WebP. */
    public static byte[] webp(List<Image> frames, double fps) {
        check(frames, fps);
        Image f0 = frames.get(0);
        int w = f0.width();
        int h = f0.height();
        ByteArrayOutputStream body = new ByteArrayOutputStream();
        body.writeBytes("WEBP".getBytes(StandardCharsets.US_ASCII));
        if (frames.size() == 1) {
            chunk(body, "VP8L", Vp8l.encode(w, h, f0.rgb()));
        } else {
            ByteArrayOutputStream vp8x = new ByteArrayOutputStream();
            vp8x.write(0x02); // animation
            vp8x.write(0);
            vp8x.write(0);
            vp8x.write(0);
            le24(vp8x, w - 1);
            le24(vp8x, h - 1);
            chunk(body, "VP8X", vp8x.toByteArray());
            ByteArrayOutputStream anim = new ByteArrayOutputStream();
            le32(anim, 0xffffffffL); // background colour (ignored by viewers that blend over the page)
            le16(anim, 0);           // loop forever
            chunk(body, "ANIM", anim.toByteArray());
            for (int i = 0; i < frames.size(); i++) {
                ByteArrayOutputStream fr = new ByteArrayOutputStream();
                le24(fr, 0);
                le24(fr, 0);
                le24(fr, w - 1);
                le24(fr, h - 1);
                le24(fr, Math.max(1, delay(i, fps, 1000)));
                fr.write(0x02); // do not blend, do not dispose
                chunk(fr, "VP8L", Vp8l.encode(w, h, frames.get(i).rgb()));
                chunk(body, "ANMF", fr.toByteArray());
            }
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.writeBytes("RIFF".getBytes(StandardCharsets.US_ASCII));
        le32(out, body.size());
        out.writeBytes(body.toByteArray());
        return out.toByteArray();
    }

    // ------------------------------------------------------------------ animated GIF

    /** An animated GIF with one global palette (loops forever). */
    public static byte[] gif(List<Image> frames, double fps) {
        check(frames, fps);
        Image f0 = frames.get(0);
        int w = f0.width();
        int h = f0.height();
        int[] palette = medianCut(frames, 256);
        int[] cache = new int[32768];
        Arrays.fill(cache, -1);
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        o.writeBytes("GIF89a".getBytes(StandardCharsets.US_ASCII));
        le16(o, w);
        le16(o, h);
        o.write(0xF7); // global colour table of 256 entries, 8 bits per primary
        o.write(0);
        o.write(0);
        for (int c : palette) {
            o.write((c >> 16) & 255);
            o.write((c >> 8) & 255);
            o.write(c & 255);
        }
        if (frames.size() > 1) { // NETSCAPE2.0: loop forever
            o.writeBytes(new byte[]{0x21, (byte) 0xFF, 0x0B});
            o.writeBytes("NETSCAPE2.0".getBytes(StandardCharsets.US_ASCII));
            o.writeBytes(new byte[]{0x03, 0x01, 0, 0, 0});
        }
        for (int i = 0; i < frames.size(); i++) {
            o.writeBytes(new byte[]{0x21, (byte) 0xF9, 0x04, 0x04});
            le16(o, Math.max(2, delay(i, fps, 100))); // browsers treat delays below 2 cs as 10 cs
            o.write(0);
            o.write(0);
            o.write(0x2C);
            le16(o, 0);
            le16(o, 0);
            le16(o, w);
            le16(o, h);
            o.write(0);
            byte[] idx = quantize(frames.get(i), palette, cache);
            o.write(8);
            lzw(o, idx);
            o.write(0);
        }
        o.write(0x3B);
        return o.toByteArray();
    }

    /** Median-cut palette of up to {@code colors} colours from a sample of all the frames. */
    static int[] medianCut(List<Image> frames, int colors) {
        int total = 0;
        for (Image f : frames) {
            total += f.width() * f.height();
        }
        int step = Math.max(1, total / 120_000);
        int[] sample = new int[total / step + 1];
        int n = 0;
        int k = 0;
        for (Image f : frames) {
            byte[] rgb = f.rgb();
            for (int p = 0; p < rgb.length / 3; p++, k++) {
                if (k % step == 0) {
                    sample[n++] = ((rgb[p * 3] & 255) << 16) | ((rgb[p * 3 + 1] & 255) << 8) | (rgb[p * 3 + 2] & 255);
                }
            }
        }
        java.util.ArrayList<int[]> boxes = new java.util.ArrayList<>(); // {lo, hi) ranges into the sorted-in-place sample
        boxes.add(new int[]{0, n});
        while (boxes.size() < colors) {
            int best = -1;
            double bestScore = 0;
            int bestAxis = 0;
            for (int bi = 0; bi < boxes.size(); bi++) {
                int[] bx = boxes.get(bi);
                if (bx[1] - bx[0] < 2) {
                    continue;
                }
                int[] lo = {255, 255, 255};
                int[] hi = {0, 0, 0};
                for (int i = bx[0]; i < bx[1]; i++) {
                    for (int c = 0; c < 3; c++) {
                        int v = (sample[i] >> (16 - 8 * c)) & 255;
                        lo[c] = Math.min(lo[c], v);
                        hi[c] = Math.max(hi[c], v);
                    }
                }
                for (int c = 0; c < 3; c++) {
                    double score = (double) (hi[c] - lo[c]) * Math.sqrt(bx[1] - bx[0]);
                    if (score > bestScore) {
                        bestScore = score;
                        best = bi;
                        bestAxis = c;
                    }
                }
            }
            if (best < 0) {
                break;
            }
            int[] bx = boxes.get(best);
            final int shift = 16 - 8 * bestAxis;
            long[] keyed = new long[bx[1] - bx[0]];
            for (int i = 0; i < keyed.length; i++) {
                int v = sample[bx[0] + i];
                keyed[i] = ((long) ((v >> shift) & 255) << 32) | (v & 0xffffffffL);
            }
            Arrays.sort(keyed);
            for (int i = 0; i < keyed.length; i++) {
                sample[bx[0] + i] = (int) keyed[i];
            }
            int mid = bx[0] + (bx[1] - bx[0]) / 2;
            boxes.set(best, new int[]{bx[0], mid});
            boxes.add(new int[]{mid, bx[1]});
        }
        int[] pal = new int[256];
        for (int bi = 0; bi < boxes.size() && bi < 256; bi++) {
            int[] bx = boxes.get(bi);
            long r = 0;
            long g = 0;
            long b = 0;
            int cnt = Math.max(1, bx[1] - bx[0]);
            for (int i = bx[0]; i < bx[1]; i++) {
                r += (sample[i] >> 16) & 255;
                g += (sample[i] >> 8) & 255;
                b += sample[i] & 255;
            }
            pal[bi] = (int) (r / cnt) << 16 | (int) (g / cnt) << 8 | (int) (b / cnt);
        }
        return pal;
    }

    private static int nearest(int[] pal, int r, int g, int b, int[] cache) {
        int key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
        if (cache[key] >= 0) {
            return cache[key];
        }
        int rr = (r & 0xf8) | 4;
        int gg = (g & 0xf8) | 4;
        int bb = (b & 0xf8) | 4;
        int best = 0;
        int bd = Integer.MAX_VALUE;
        for (int i = 0; i < pal.length; i++) {
            int dr = rr - ((pal[i] >> 16) & 255);
            int dg = gg - ((pal[i] >> 8) & 255);
            int db = bb - (pal[i] & 255);
            int d = 3 * dr * dr + 4 * dg * dg + 2 * db * db;
            if (d < bd) {
                bd = d;
                best = i;
            }
        }
        cache[key] = best;
        return best;
    }

    /** Palette indices of a frame with serpentine Floyd–Steinberg error diffusion. */
    private static byte[] quantize(Image img, int[] pal, int[] cache) {
        int w = img.width();
        int h = img.height();
        byte[] rgb = img.rgb();
        byte[] out = new byte[w * h];
        float[][] err = new float[2][(w + 2) * 3];
        for (int y = 0; y < h; y++) {
            float[] cur = err[y & 1];
            float[] nxt = err[(y + 1) & 1];
            Arrays.fill(nxt, 0f);
            boolean ltr = (y & 1) == 0;
            for (int i = 0; i < w; i++) {
                int x = ltr ? i : w - 1 - i;
                int dir = ltr ? 1 : -1;
                int[] want = new int[3];
                for (int c = 0; c < 3; c++) {
                    float v = (rgb[(y * w + x) * 3 + c] & 255) + cur[(x + 1) * 3 + c];
                    want[c] = Math.max(0, Math.min(255, Math.round(v)));
                }
                int idx = nearest(pal, want[0], want[1], want[2], cache);
                out[y * w + x] = (byte) idx;
                int[] got = {(pal[idx] >> 16) & 255, (pal[idx] >> 8) & 255, pal[idx] & 255};
                for (int c = 0; c < 3; c++) {
                    float e = (rgb[(y * w + x) * 3 + c] & 255) + cur[(x + 1) * 3 + c] - got[c];
                    e = Math.max(-64f, Math.min(64f, e));
                    cur[(x + dir + 1) * 3 + c] += e * 7f / 16f;
                    nxt[(x - dir + 1) * 3 + c] += e * 3f / 16f;
                    nxt[(x + 1) * 3 + c] += e * 5f / 16f;
                    nxt[(x + dir + 1) * 3 + c] += e * 1f / 16f;
                }
            }
        }
        return out;
    }

    /** GIF's variable-width LZW with 8-bit symbols; the output is cut into sub-blocks of at most 255 bytes. */
    private static void lzw(ByteArrayOutputStream o, byte[] data) {
        final int clear = 256;
        final int eoi = 257;
        ByteArrayOutputStream blocks = new ByteArrayOutputStream();
        int[] table = new int[8209 * 2]; // open addressing: key (prefix << 8 | byte) + 1 and code
        int next = 258;
        int bits = 9;
        // bit writer, least significant bit first
        class W {
            long a;
            int n;

            void put(int code, int width) {
                a |= (long) code << n;
                n += width;
                while (n >= 8) {
                    blocks.write((int) (a & 255));
                    a >>>= 8;
                    n -= 8;
                }
            }
        }
        W w = new W();
        w.put(clear, bits);
        int prefix = data[0] & 255;
        for (int i = 1; i < data.length; i++) {
            int c = data[i] & 255;
            int key = (prefix << 8) | c;
            int slot = (key * 40503) & 0x1fff;
            int found = -1;
            while (table[slot * 2] != 0) {
                if (table[slot * 2] == key + 1) {
                    found = table[slot * 2 + 1];
                    break;
                }
                slot = (slot + 1) % 8209;
            }
            if (found >= 0) {
                prefix = found;
                continue;
            }
            w.put(prefix, bits);
            if (next < 4096) {
                table[slot * 2] = key + 1;
                table[slot * 2 + 1] = next;
                if (next == (1 << bits) && bits < 12) {
                    bits++;
                }
                next++;
            } else {
                w.put(clear, bits);
                Arrays.fill(table, 0);
                next = 258;
                bits = 9;
            }
            prefix = c;
        }
        w.put(prefix, bits);
        w.put(eoi, bits);
        if (w.n > 0) {
            blocks.write((int) (w.a & 255));
        }
        byte[] all = blocks.toByteArray();
        for (int p = 0; p < all.length; p += 255) {
            int n = Math.min(255, all.length - p);
            o.write(n);
            o.write(all, p, n);
        }
    }
}
