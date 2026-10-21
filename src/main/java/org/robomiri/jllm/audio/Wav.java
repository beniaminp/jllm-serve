package org.robomiri.jllm.audio;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Audio files in pure Java: reads RIFF/WAVE (PCM 8/16/24/32-bit, IEEE float 32/64, A-law and mu-law, WAVE_FORMAT_EXTENSIBLE
 * wrapping any of these), AIFF / AIFF-C (PCM 8..32-bit, "sowt" little-endian and "fl32"/"fl64" float) and Sun/NeXT .au
 * (PCM, float, mu-law), and writes 16-bit or 32-bit float PCM WAV. Other containers (MP3, FLAC, Ogg, M4A, WebM) are not
 * decoded here: {@link AudioDecoder} hands them to an installed {@code ffmpeg} when there is one, and the web console
 * converts recordings and uploads to WAV in the browser, which decodes every format the browser itself plays.
 */
public final class Wav {

    private Wav() {
    }

    /**
     * Decoded audio.
     *
     * @param channels  one array of samples in [-1, 1] per channel (equal lengths)
     * @param sampleRate samples per second of each channel
     */
    public record Audio(float[][] channels, int sampleRate) {
        public int length() {
            return channels.length == 0 ? 0 : channels[0].length;
        }

        public double seconds() {
            return sampleRate == 0 ? 0 : (double) length() / sampleRate;
        }

        /** The mean of the channels. */
        public float[] mono() {
            if (channels.length == 1) {
                return channels[0];
            }
            int n = length();
            float[] out = new float[n];
            for (float[] c : channels) {
                for (int i = 0; i < n; i++) {
                    out[i] += c[i];
                }
            }
            float k = 1f / channels.length;
            for (int i = 0; i < n; i++) {
                out[i] *= k;
            }
            return out;
        }
    }

    /** True when {@code b} starts like a file this class reads (RIFF/WAVE, FORM/AIFF, .snd). */
    public static boolean recognizes(byte[] b) {
        if (b.length < 12) {
            return false;
        }
        String a = new String(b, 0, 4, java.nio.charset.StandardCharsets.ISO_8859_1);
        String c = new String(b, 8, 4, java.nio.charset.StandardCharsets.ISO_8859_1);
        return (a.equals("RIFF") && c.equals("WAVE")) || (a.equals("FORM") && (c.equals("AIFF") || c.equals("AIFC")))
                || a.equals(".snd");
    }

    /** Reads a WAV, AIFF or AU file. */
    public static Audio read(byte[] file) throws IOException {
        if (file.length < 12) {
            throw new IOException("not an audio file: only " + file.length + " bytes");
        }
        String magic = new String(file, 0, 4, java.nio.charset.StandardCharsets.ISO_8859_1);
        return switch (magic) {
            case "RIFF", "RF64" -> readWave(file);
            case "FORM" -> readAiff(file);
            case ".snd" -> readAu(file);
            default -> throw new IOException("unsupported audio container (expected WAV, AIFF or AU)");
        };
    }

    // ------------------------------------------------------------------ WAVE

    private static Audio readWave(byte[] file) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(file).order(ByteOrder.LITTLE_ENDIAN);
        if (!new String(file, 8, 4, java.nio.charset.StandardCharsets.ISO_8859_1).equals("WAVE")) {
            throw new IOException("not a WAVE file");
        }
        int pos = 12;
        int format = 0;
        int channels = 0;
        int rate = 0;
        int bits = 0;
        boolean haveFmt = false;
        while (pos + 8 <= file.length) {
            String id = new String(file, pos, 4, java.nio.charset.StandardCharsets.ISO_8859_1);
            long size = b.getInt(pos + 4) & 0xFFFFFFFFL;
            int body = pos + 8;
            if (id.equals("fmt ")) {
                if (size < 16 || body + 16 > file.length) {
                    throw new IOException("truncated WAVE fmt chunk");
                }
                format = b.getShort(body) & 0xFFFF;
                channels = b.getShort(body + 2) & 0xFFFF;
                rate = b.getInt(body + 4);
                bits = b.getShort(body + 14) & 0xFFFF;
                if (format == 0xFFFE && size >= 26) { // WAVE_FORMAT_EXTENSIBLE: the real format is the sub-format GUID's first 2 bytes
                    format = b.getShort(body + 24) & 0xFFFF;
                }
                haveFmt = true;
            } else if (id.equals("data")) {
                if (!haveFmt) {
                    throw new IOException("WAVE data chunk before its fmt chunk");
                }
                long avail = file.length - body;
                // a streamed file (size 0 or 0xFFFFFFFF) or a truncated one: use what is there
                long len = (size == 0 || size == 0xFFFFFFFFL || size > avail) ? avail : size;
                return decodePcm(file, body, (int) len, format, channels, rate, bits, ByteOrder.LITTLE_ENDIAN);
            }
            long next = body + size + (size & 1);
            if (next > file.length) {
                break;
            }
            pos = (int) next;
        }
        throw new IOException("WAVE file without a data chunk");
    }

    private static Audio decodePcm(byte[] f, int off, int len, int format, int channels, int rate, int bits,
                                   ByteOrder order) throws IOException {
        if (channels < 1 || rate < 1) {
            throw new IOException("invalid audio format: " + channels + " channels at " + rate + " Hz");
        }
        int bytes = format == 6 || format == 7 ? 1 : (bits + 7) / 8;
        if (bytes < 1 || bytes > 8) {
            throw new IOException("unsupported sample size: " + bits + " bits");
        }
        int frames = len / (bytes * channels);
        float[][] out = new float[channels][frames];
        ByteBuffer b = ByteBuffer.wrap(f).order(order);
        int p = off;
        for (int i = 0; i < frames; i++) {
            for (int c = 0; c < channels; c++) {
                out[c][i] = sample(b, p, format, bytes);
                p += bytes;
            }
        }
        return new Audio(out, rate);
    }

    private static float sample(ByteBuffer b, int p, int format, int bytes) throws IOException {
        switch (format) {
            case 1: // PCM
                return switch (bytes) {
                    case 1 -> ((b.get(p) & 0xFF) - 128) / 128f;
                    case 2 -> b.getShort(p) / 32768f;
                    case 3 -> {
                        int lo = b.order() == ByteOrder.LITTLE_ENDIAN ? (b.get(p) & 0xFF) | ((b.get(p + 1) & 0xFF) << 8)
                                | (b.get(p + 2) << 16)
                                : (b.get(p + 2) & 0xFF) | ((b.get(p + 1) & 0xFF) << 8) | (b.get(p) << 16);
                        yield lo / 8388608f;
                    }
                    case 4 -> b.getInt(p) / 2147483648f;
                    default -> throw new IOException("unsupported PCM sample size: " + bytes * 8 + " bits");
                };
            case 3: // IEEE float
                if (bytes == 4) {
                    return b.getFloat(p);
                } else if (bytes == 8) {
                    return (float) b.getDouble(p);
                }
                throw new IOException("unsupported float sample size: " + bytes * 8 + " bits");
            case 6: // A-law
                return alaw(b.get(p) & 0xFF) / 32768f;
            case 7: // mu-law
                return mulaw(b.get(p) & 0xFF) / 32768f;
            default:
                throw new IOException("unsupported WAVE format tag " + format
                        + " (PCM, float, A-law and mu-law are read; convert other encodings to WAV)");
        }
    }

    private static int mulaw(int u) {
        u = ~u & 0xFF;
        int t = ((u & 0x0F) << 3) + 0x84;
        t <<= (u & 0x70) >> 4;
        return (u & 0x80) != 0 ? 0x84 - t : t - 0x84;
    }

    private static int alaw(int a) {
        a ^= 0x55;
        int t = (a & 0x0F) << 4;
        int seg = (a & 0x70) >> 4;
        t = switch (seg) {
            case 0 -> t + 8;
            case 1 -> t + 0x108;
            default -> (t + 0x108) << (seg - 1);
        };
        return (a & 0x80) != 0 ? t : -t;
    }

    // ------------------------------------------------------------------ AIFF

    private static Audio readAiff(byte[] file) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(file).order(ByteOrder.BIG_ENDIAN);
        boolean aifc = new String(file, 8, 4, java.nio.charset.StandardCharsets.ISO_8859_1).equals("AIFC");
        int pos = 12;
        int channels = 0;
        int bits = 0;
        int rate = 0;
        String comp = "NONE";
        while (pos + 8 <= file.length) {
            String id = new String(file, pos, 4, java.nio.charset.StandardCharsets.ISO_8859_1);
            long size = b.getInt(pos + 4) & 0xFFFFFFFFL;
            int body = pos + 8;
            if (id.equals("COMM")) {
                channels = b.getShort(body) & 0xFFFF;
                bits = b.getShort(body + 6) & 0xFFFF;
                rate = (int) Math.round(extended(file, body + 8));
                if (aifc && size >= 22) {
                    comp = new String(file, body + 18, 4, java.nio.charset.StandardCharsets.ISO_8859_1);
                }
            } else if (id.equals("SSND")) {
                int offset = b.getInt(body);
                int start = body + 8 + offset;
                long avail = file.length - start;
                int len = (int) Math.min(Math.max(0, size - 8 - offset), avail);
                boolean little = comp.equals("sowt");
                boolean flt = comp.equals("fl32") || comp.equals("FL32") || comp.equals("fl64") || comp.equals("FL64");
                if (!comp.equals("NONE") && !comp.equals("twos") && !little && !flt && !comp.equals("in24")
                        && !comp.equals("in32")) {
                    throw new IOException("unsupported AIFF-C compression '" + comp + "'");
                }
                ByteOrder order = little ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN;
                int bytes = (bits + 7) / 8;
                if (!flt && bytes == 1) { // AIFF 8-bit PCM is signed, unlike WAVE's unsigned
                    int frames = len / channels;
                    float[][] out = new float[channels][frames];
                    for (int i = 0; i < frames; i++) {
                        for (int c = 0; c < channels; c++) {
                            out[c][i] = file[start + i * channels + c] / 128f;
                        }
                    }
                    return new Audio(out, rate);
                }
                return decodePcm(file, start, len, flt ? 3 : 1, channels, rate, bits, order);
            }
            long next = body + size + (size & 1);
            if (next > file.length) {
                break;
            }
            pos = (int) next;
        }
        throw new IOException("AIFF file without a sound chunk");
    }

    /** An 80-bit IEEE extended float (the AIFF sample rate). */
    private static double extended(byte[] f, int p) {
        int se = ((f[p] & 0xFF) << 8) | (f[p + 1] & 0xFF);
        long mant = 0;
        for (int i = 0; i < 8; i++) {
            mant = (mant << 8) | (f[p + 2 + i] & 0xFF);
        }
        int exp = se & 0x7FFF;
        double v = (mant >>> 11 == 0 && exp == 0) ? 0 : Math.scalb((double) (mant >>> 11), exp - 16383 - 52);
        return (se & 0x8000) != 0 ? -v : v;
    }

    // ------------------------------------------------------------------ AU

    private static Audio readAu(byte[] file) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(file).order(ByteOrder.BIG_ENDIAN);
        int off = b.getInt(4);
        long size = b.getInt(8) & 0xFFFFFFFFL;
        int enc = b.getInt(12);
        int rate = b.getInt(16);
        int channels = b.getInt(20);
        long avail = file.length - off;
        int len = (int) (size == 0xFFFFFFFFL || size > avail ? avail : size);
        return switch (enc) {
            case 1 -> decodePcm(file, off, len, 7, channels, rate, 8, ByteOrder.BIG_ENDIAN);
            case 2 -> signed8(file, off, len, channels, rate);
            case 3 -> decodePcm(file, off, len, 1, channels, rate, 16, ByteOrder.BIG_ENDIAN);
            case 4 -> decodePcm(file, off, len, 1, channels, rate, 24, ByteOrder.BIG_ENDIAN);
            case 5 -> decodePcm(file, off, len, 1, channels, rate, 32, ByteOrder.BIG_ENDIAN);
            case 6 -> decodePcm(file, off, len, 3, channels, rate, 32, ByteOrder.BIG_ENDIAN);
            case 7 -> decodePcm(file, off, len, 3, channels, rate, 64, ByteOrder.BIG_ENDIAN);
            case 27 -> decodePcm(file, off, len, 6, channels, rate, 8, ByteOrder.BIG_ENDIAN);
            default -> throw new IOException("unsupported AU encoding " + enc);
        };
    }

    private static Audio signed8(byte[] f, int off, int len, int channels, int rate) {
        int frames = len / channels;
        float[][] out = new float[channels][frames];
        for (int i = 0; i < frames; i++) {
            for (int c = 0; c < channels; c++) {
                out[c][i] = f[off + i * channels + c] / 128f;
            }
        }
        return new Audio(out, rate);
    }

    // ------------------------------------------------------------------ writing

    /** 16-bit PCM WAV of {@code mono} samples in [-1, 1] (clipped, rounded to nearest). */
    public static byte[] write16(float[] mono, int sampleRate) {
        return write16(new float[][]{mono}, sampleRate);
    }

    public static byte[] write16(float[][] channels, int sampleRate) {
        int n = channels[0].length;
        int ch = channels.length;
        ByteBuffer b = header(n * ch * 2, ch, sampleRate, 1, 16);
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < ch; c++) {
                float v = channels[c][i];
                int s = Math.round(Math.max(-1f, Math.min(1f, v)) * 32767f);
                b.putShort((short) s);
            }
        }
        return b.array();
    }

    /** 32-bit float WAV (lossless for float audio). */
    public static byte[] writeFloat(float[][] channels, int sampleRate) {
        int n = channels[0].length;
        int ch = channels.length;
        ByteBuffer b = header(n * ch * 4, ch, sampleRate, 3, 32);
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < ch; c++) {
                b.putFloat(channels[c][i]);
            }
        }
        return b.array();
    }

    private static ByteBuffer header(int dataBytes, int channels, int rate, int format, int bits) {
        ByteBuffer b = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN);
        b.put("RIFF".getBytes(java.nio.charset.StandardCharsets.ISO_8859_1)).putInt(36 + dataBytes)
                .put("WAVE".getBytes(java.nio.charset.StandardCharsets.ISO_8859_1));
        b.put("fmt ".getBytes(java.nio.charset.StandardCharsets.ISO_8859_1)).putInt(16).putShort((short) format)
                .putShort((short) channels).putInt(rate).putInt(rate * channels * bits / 8)
                .putShort((short) (channels * bits / 8)).putShort((short) bits);
        b.put("data".getBytes(java.nio.charset.StandardCharsets.ISO_8859_1)).putInt(dataBytes);
        return b;
    }

    /** Concatenates mono pieces. */
    public static float[] concat(java.util.List<float[]> pieces) {
        int n = 0;
        for (float[] p : pieces) {
            n += p.length;
        }
        float[] out = new float[n];
        int o = 0;
        for (float[] p : pieces) {
            System.arraycopy(p, 0, out, o, p.length);
            o += p.length;
        }
        return out;
    }

    static byte[] drain(java.io.InputStream in) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        in.transferTo(bos);
        return bos.toByteArray();
    }

    static float[] copy(float[] a) {
        return Arrays.copyOf(a, a.length);
    }
}
