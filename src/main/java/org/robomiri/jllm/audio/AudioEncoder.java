package org.robomiri.jllm.audio;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Encodes generated speech for {@code POST /v1/audio/speech}. {@code wav} (16-bit PCM) and {@code pcm} (raw 16-bit
 * little-endian, the OpenAI name) are written in Java; {@code mp3}, {@code opus}, {@code aac} and {@code flac} need an
 * {@code ffmpeg} executable ({@code audio.ffmpeg}).
 */
public final class AudioEncoder {

    private AudioEncoder() {
    }

    public static final List<String> FORMATS = List.of("wav", "pcm", "mp3", "opus", "aac", "flac");

    public static String contentType(String format) {
        return switch (format) {
            case "wav" -> "audio/wav";
            case "pcm" -> "audio/L16; rate=24000; channels=1";
            case "mp3" -> "audio/mpeg";
            case "opus" -> "audio/ogg";
            case "aac" -> "audio/aac";
            case "flac" -> "audio/flac";
            default -> "application/octet-stream";
        };
    }

    /** @throws IOException when the format needs ffmpeg and there is none (or it fails) */
    public static byte[] encode(String format, float[] samples, int rate, AudioDecoder tools) throws IOException {
        switch (format) {
            case "wav":
                return Wav.write16(samples, rate);
            case "pcm": {
                ByteBuffer b = ByteBuffer.allocate(samples.length * 2).order(ByteOrder.LITTLE_ENDIAN);
                for (float v : samples) {
                    b.putShort((short) Math.round(Math.max(-1f, Math.min(1f, v)) * 32767f));
                }
                return b.array();
            }
            default:
        }
        String ffmpeg = tools.ffmpegPath();
        if (ffmpeg == null) {
            throw new IOException("response_format " + format + " needs ffmpeg (audio.ffmpeg); use wav or pcm");
        }
        String[] codec = switch (format) {
            case "mp3" -> new String[]{"-f", "mp3", "-c:a", "libmp3lame"};
            case "opus" -> new String[]{"-f", "ogg", "-c:a", "libopus"};
            case "aac" -> new String[]{"-f", "adts", "-c:a", "aac"};
            case "flac" -> new String[]{"-f", "flac"};
            default -> throw new IOException("unsupported response_format " + format);
        };
        Path in = Files.createTempFile("jllm-speech-", ".wav");
        try {
            Files.write(in, Wav.write16(samples, rate));
            List<String> cmd = new java.util.ArrayList<>(List.of(ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-i", in.toString()));
            cmd.addAll(List.of(codec));
            cmd.add("pipe:1");
            Process p = new ProcessBuilder(cmd).redirectError(ProcessBuilder.Redirect.DISCARD).start();
            byte[] out;
            try (var s = p.getInputStream()) {
                out = Wav.drain(s);
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
            if (p.exitValue() != 0 || out.length == 0) {
                throw new IOException("ffmpeg could not encode " + format + " (exit " + p.exitValue() + ")");
            }
            return out;
        } finally {
            Files.deleteIfExists(in);
        }
    }
}
