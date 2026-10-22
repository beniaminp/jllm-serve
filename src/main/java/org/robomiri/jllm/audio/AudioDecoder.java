package org.robomiri.jllm.audio;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

/**
 * Turns an uploaded audio file into 16 kHz mono floats. WAV, AIFF and AU are decoded in Java ({@link Wav}); anything
 * else (MP3, FLAC, Ogg, M4A, WebM, video files) goes through an {@code ffmpeg} executable when one is installed
 * ({@code audio.ffmpeg}: {@code auto} looks it up on the PATH, {@code off} never runs it, or a path). Without ffmpeg
 * those formats are refused with a message saying so; the web console avoids the problem by decoding recordings and
 * uploads in the browser and sending WAV.
 */
public final class AudioDecoder {

    /** The demuxers an upload may use (ffmpeg's names). */
    static final String DEMUXERS = "wav,w64,aiff,au,caf,mp3,flac,ogg,aac,ac3,eac3,amr,asf,mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,"
            + "mpegts,mpeg,avi,wv,ape,tta,s16le,mp2";

    private final String ffmpeg;      // resolved executable, or null
    private final long maxSamples;    // samples at 16 kHz

    /**
     * @param ffmpeg     "auto", "off" or a path to the executable
     * @param maxSeconds longest accepted recording (after decoding)
     */
    public AudioDecoder(String ffmpeg, double maxSeconds) {
        this.ffmpeg = resolve(ffmpeg);
        this.maxSamples = (long) (maxSeconds * LogMel.SAMPLE_RATE);
    }

    public static AudioDecoder defaults() {
        return new AudioDecoder("auto", 4 * 3600);
    }

    private static String resolve(String spec) {
        if (spec == null || spec.equalsIgnoreCase("off") || spec.equalsIgnoreCase("false")) {
            return null;
        }
        if (!spec.equalsIgnoreCase("auto")) {
            return Files.isExecutable(Path.of(spec)) ? spec : null;
        }
        String path = System.getenv("PATH");
        if (path != null) {
            for (String d : path.split(java.io.File.pathSeparator)) {
                Path p = Path.of(d, "ffmpeg");
                if (Files.isExecutable(p)) {
                    return p.toString();
                }
            }
        }
        for (String d : new String[]{"/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"}) {
            if (Files.isExecutable(Path.of(d))) {
                return d;
            }
        }
        return null;
    }

    public boolean hasFfmpeg() {
        return ffmpeg != null;
    }

    /** The ffmpeg executable in use, or null. */
    public String ffmpegPath() {
        return ffmpeg;
    }

    /** Longest accepted recording in seconds. */
    public double maxSeconds() {
        return (double) maxSamples / LogMel.SAMPLE_RATE;
    }

    /** The file as 16 kHz mono samples in [-1, 1]. */
    public float[] decode16k(byte[] file) throws IOException {
        float[] out;
        if (Wav.recognizes(file)) {
            Wav.Audio a = Wav.read(file);
            if (a.length() > maxSamples * a.sampleRate() / LogMel.SAMPLE_RATE) {
                throw new IOException("audio longer than " + maxSamples / LogMel.SAMPLE_RATE + " s");
            }
            out = Resampler.resample(a.mono(), a.sampleRate(), LogMel.SAMPLE_RATE);
        } else if (ffmpeg != null) {
            out = viaFfmpeg(file);
        } else {
            throw new IOException("unsupported audio format: only WAV, AIFF and AU are decoded without ffmpeg"
                    + " (install ffmpeg, or send WAV; the web console converts in the browser)");
        }
        if (out.length > maxSamples) {
            throw new IOException("audio longer than " + maxSamples / LogMel.SAMPLE_RATE + " s");
        }
        return out;
    }

    private float[] viaFfmpeg(byte[] file) throws IOException {
        Path in = Files.createTempFile("jllm-audio-", ".bin");
        try {
            Files.write(in, file);
            // an upload is untrusted: only plain files, and only the audio / media containers, never a playlist or a
            // network-reaching demuxer (concat, hls, sdp ...) that could read other files or addresses
            Process p = new ProcessBuilder(ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist",
                    "file,pipe", "-format_whitelist", DEMUXERS, "-i", in.toString(), "-vn", "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", String.valueOf(LogMel.SAMPLE_RATE),
                    "pipe:1").redirectError(ProcessBuilder.Redirect.DISCARD).start();
            byte[] raw;
            try (var s = p.getInputStream()) {
                raw = Wav.drain(s);
            }
            try {
                if (!p.waitFor(60, TimeUnit.SECONDS)) {
                    p.destroyForcibly();
                    throw new IOException("ffmpeg timed out");
                }
            } catch (InterruptedException e) {
                p.destroyForcibly();
                Thread.currentThread().interrupt();
                throw new IOException("interrupted");
            }
            if (p.exitValue() != 0 || raw.length == 0) {
                throw new IOException("ffmpeg could not decode the file (exit " + p.exitValue() + ")");
            }
            float[] out = new float[raw.length / 4];
            ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out);
            return out;
        } finally {
            Files.deleteIfExists(in);
        }
    }
}
