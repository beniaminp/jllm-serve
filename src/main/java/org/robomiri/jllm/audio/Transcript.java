package org.robomiri.jllm.audio;

import org.robomiri.jllm.json.Json;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * A transcription: the text, its language and the timed segments, with the renderings of the OpenAI audio API
 * ({@code json}, {@code text}, {@code srt}, {@code vtt}, {@code verbose_json}).
 *
 * @param language    the language code ("en"); the detected one when none was requested
 * @param task        "transcribe" or "translate"
 * @param duration    seconds of audio
 * @param languageProbs the detection's probabilities of the most likely languages (empty when the language was given)
 */
public record Transcript(String language, String task, double duration, String text, List<Segment> segments,
                         Map<String, Double> languageProbs) {

    /**
     * @param seek   start of the 30-second window the segment came from, in 10 ms frames
     * @param tokens text token ids (without timestamps)
     */
    public record Segment(int id, int seek, double start, double end, String text, int[] tokens, double temperature,
                          double avgLogprob, double compressionRatio, double noSpeechProb) {
    }

    public static final List<String> FORMATS = List.of("json", "text", "srt", "vtt", "verbose_json");

    /** The body for {@code response_format}; throws IllegalArgumentException for an unknown format. */
    public String render(String format) {
        return switch (format == null ? "json" : format) {
            case "json" -> Json.write(Json.obj("text", text));
            case "text" -> text + "\n";
            case "srt" -> srt();
            case "vtt" -> vtt();
            case "verbose_json" -> Json.write(verbose());
            default -> throw new IllegalArgumentException("response_format must be one of " + FORMATS);
        };
    }

    public static String contentType(String format) {
        return switch (format == null ? "json" : format) {
            case "text", "srt", "vtt" -> "text/plain; charset=utf-8";
            default -> "application/json";
        };
    }

    public Map<String, Object> verbose() {
        List<Object> segs = new ArrayList<>();
        for (Segment s : segments) {
            List<Object> toks = new ArrayList<>();
            for (int t : s.tokens()) {
                toks.add((long) t);
            }
            segs.add(Json.obj("id", (long) s.id(), "seek", (long) s.seek(), "start", round(s.start()), "end", round(s.end()),
                    "text", s.text(), "tokens", toks, "temperature", s.temperature(), "avg_logprob", round(s.avgLogprob()),
                    "compression_ratio", round(s.compressionRatio()), "no_speech_prob", round(s.noSpeechProb())));
        }
        Map<String, Object> m = Json.obj("task", task, "language", Languages.NAMES.getOrDefault(language, language),
                "duration", round(duration), "text", text, "segments", segs);
        if (!languageProbs.isEmpty()) {
            m.put("language_probabilities", new java.util.LinkedHashMap<String, Object>(languageProbs));
        }
        return m;
    }

    private static double round(double v) {
        return Math.round(v * 100000.0) / 100000.0;
    }

    public String srt() {
        StringBuilder sb = new StringBuilder();
        int i = 1;
        for (Segment s : segments) {
            sb.append(i++).append('\n').append(stamp(s.start(), ',')).append(" --> ").append(stamp(s.end(), ','))
                    .append('\n').append(s.text().strip()).append("\n\n");
        }
        return sb.toString();
    }

    public String vtt() {
        StringBuilder sb = new StringBuilder("WEBVTT\n\n");
        for (Segment s : segments) {
            sb.append(stamp(s.start(), '.')).append(" --> ").append(stamp(s.end(), '.')).append('\n')
                    .append(s.text().strip()).append("\n\n");
        }
        return sb.toString();
    }

    static String stamp(double seconds, char sep) {
        long ms = Math.round(seconds * 1000);
        long h = ms / 3_600_000;
        ms %= 3_600_000;
        long m = ms / 60_000;
        ms %= 60_000;
        return String.format(Locale.ROOT, "%02d:%02d:%02d%c%03d", h, m, ms / 1000, sep, ms % 1000);
    }
}
