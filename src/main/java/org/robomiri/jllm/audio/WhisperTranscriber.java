package org.robomiri.jllm.audio;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import java.util.zip.Deflater;

/**
 * Whisper's decoding protocol over a {@link WhisperNet}, following OpenAI's {@code whisper.transcribe} /
 * {@code DecodingTask}:
 * <ol>
 *   <li>the log-mel of the whole recording (plus 30 s of silence) is computed once; windows of
 *       {@code windowFrames} frames (30 s) are cut from it, the window moving by the last complete timestamp the
 *       decoder wrote (or by a whole window when the text ends without one);</li>
 *   <li>per window the decoder is primed with {@code <|startofprev|> previous text <|startoftranscript|> <|lang|>
 *       <|transcribe|>|<|translate|> [<|notimestamps|>]}; with no language given, the first window's logits after
 *       {@code <|startoftranscript|>}, restricted to the language tokens, pick it for the whole recording;</li>
 *   <li>token by token the logits are filtered (blank and end-of-text suppressed at the first step, the checkpoint's
 *       {@code suppress_tokens} always, timestamps in pairs and non-decreasing, a timestamp forced when the
 *       probability mass of all timestamps beats the best text token), then the argmax is taken (or a sample at a
 *       temperature); a window that decodes badly (compression ratio above 2.4, average log-probability below -1) is
 *       decoded again at the next temperature of 0.2, 0.4 ... 1.0; a window whose no-speech probability is above 0.6
 *       (with a poor average log-probability) is skipped.</li>
 * </ol>
 */
public final class WhisperTranscriber {

    /**
     * @param language     code or English name; null to detect
     * @param translate    translate into English instead of transcribing
     * @param prompt       text the decoder is conditioned on (names, jargon, style), or null
     * @param timestamps   decode timestamp tokens (segments, and the window follows the speech); false: 30 s windows
     * @param temperature  0 = greedy, with the temperature fallback when {@code fallback}; above 0 = that temperature only
     * @param fallback     retry bad windows at higher temperatures (only with {@code temperature == 0})
     * @param conditionOnPrevious feed the previous windows' text as the prompt of the next one
     * @param maxTokens    most tokens decoded per window (at most 224)
     * @param seed         for the sampling at temperatures above 0
     * @param cancelled    polled between tokens
     * @param progress     fraction of the audio done, after each window
     * @param onSegment    each segment as soon as its window is decoded
     */
    public record Options(String language, boolean translate, String prompt, boolean timestamps, double temperature,
                          boolean fallback, boolean conditionOnPrevious, int maxTokens, long seed,
                          BooleanSupplier cancelled, DoubleConsumer progress, Consumer<Transcript.Segment> onSegment,
                          boolean wordTimestamps) {
        public static Options defaults() {
            return new Options(null, false, null, true, 0.0, true, true, 224, 0L, () -> false, f -> { }, s -> { }, false);
        }

        public Options withLanguage(String l) {
            return new Options(l, translate, prompt, timestamps, temperature, fallback, conditionOnPrevious, maxTokens, seed,
                    cancelled, progress, onSegment, wordTimestamps);
        }

        public Options withTranslate(boolean t) {
            return new Options(language, t, prompt, timestamps, temperature, fallback, conditionOnPrevious, maxTokens, seed,
                    cancelled, progress, onSegment, wordTimestamps);
        }

        public Options withPrompt(String p) {
            return new Options(language, translate, p, timestamps, temperature, fallback, conditionOnPrevious, maxTokens, seed,
                    cancelled, progress, onSegment, wordTimestamps);
        }

        public Options withTimestamps(boolean t) {
            return new Options(language, translate, prompt, t, temperature, fallback, conditionOnPrevious, maxTokens, seed,
                    cancelled, progress, onSegment, wordTimestamps);
        }

        public Options withTemperature(double t) {
            return new Options(language, translate, prompt, timestamps, t, fallback, conditionOnPrevious, maxTokens, seed,
                    cancelled, progress, onSegment, wordTimestamps);
        }

        public Options withFallback(boolean fb, boolean cond) {
            return new Options(language, translate, prompt, timestamps, temperature, fb, cond, maxTokens, seed,
                    cancelled, progress, onSegment, wordTimestamps);
        }

        public Options withWords(boolean w) {
            return new Options(language, translate, prompt, timestamps, temperature, fallback, conditionOnPrevious, maxTokens,
                    seed, cancelled, progress, onSegment, w);
        }

        public Options withCallbacks(BooleanSupplier c, DoubleConsumer p, Consumer<Transcript.Segment> s) {
            return new Options(language, translate, prompt, timestamps, temperature, fallback, conditionOnPrevious, maxTokens,
                    seed, c, p, s, wordTimestamps);
        }
    }

    /** Thrown when {@link Options#cancelled} turned true. */
    public static final class Cancelled extends RuntimeException {
        public Cancelled() {
            super("cancelled", null, false, false);
        }
    }

    static final double COMPRESSION_LIMIT = 2.4;
    static final double LOGPROB_LIMIT = -1.0;
    static final double NO_SPEECH_LIMIT = 0.6;
    static final double[] FALLBACK = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
    static final int BEST_OF = 5;

    private final WhisperNet net;
    private final Tokenizer tok;
    private final WhisperConfig cfg;
    private final LogMel mel;
    private final int sot;
    private final int eot;
    private final int notimestamps;
    private final int timestampBegin;
    private final int transcribeId;
    private final int translateId;
    private final int prevId;
    private final int noSpeechId;
    private final int blankId;
    private final int maxInitialTimestamp;
    private final Map<String, Integer> langIds = new LinkedHashMap<>();
    private final boolean[] suppress;
    private final int[][] alignmentHeads;

    public WhisperTranscriber(WhisperNet net, Tokenizer tok, Path modelDir) throws IOException {
        this.net = net;
        this.tok = tok;
        this.cfg = net.config();
        this.mel = new LogMel(cfg.nMels());
        Map<String, Object> gen = Map.of();
        Path gp = modelDir == null ? null : modelDir.resolve("generation_config.json");
        if (gp != null && Files.isRegularFile(gp)) {
            gen = Json.parseObject(Files.readString(gp));
        }
        sot = special("<|startoftranscript|>", (int) Json.num(gen, "decoder_start_token_id", -1));
        eot = special("<|endoftext|>", (int) Json.num(gen, "eos_token_id", -1));
        notimestamps = special("<|notimestamps|>", (int) Json.num(gen, "no_timestamps_token_id", -1));
        timestampBegin = notimestamps + 1;
        transcribeId = tok.tokenId("<|transcribe|>");
        translateId = tok.tokenId("<|translate|>");
        prevId = special("<|startofprev|>", (int) Json.num(gen, "prev_sot_token_id", -1));
        noSpeechId = firstId("<|nocaptions|>", "<|nospeech|>");
        int[] blank = tok.encode(" ", false);
        blankId = blank.length == 1 ? blank[0] : -1;
        maxInitialTimestamp = (int) Json.num(gen, "max_initial_timestamp_index", 50);
        Object l2i = gen.get("lang_to_id");
        if (l2i instanceof Map<?, ?> m) {
            for (Map.Entry<?, ?> e : m.entrySet()) {
                String k = String.valueOf(e.getKey());
                langIds.put(k.substring(2, k.length() - 2), ((Number) e.getValue()).intValue());
            }
        } else if (cfg.multilingual()) {
            for (String code : Languages.NAMES.keySet()) {
                int id = tok.tokenId("<|" + code + "|>");
                if (id >= 0) {
                    langIds.put(code, id);
                }
            }
        }
        Object ah = gen.get("alignment_heads");
        if (ah instanceof List<?> hl && !hl.isEmpty()) {
            alignmentHeads = new int[hl.size()][];
            for (int i = 0; i < hl.size(); i++) {
                List<?> lh = (List<?>) hl.get(i);
                alignmentHeads[i] = new int[]{((Number) lh.get(0)).intValue(), ((Number) lh.get(1)).intValue()};
            }
        } else { // OpenAI's default when a checkpoint names none: every head of the last half of the decoder
            List<int[]> hs = new ArrayList<>();
            for (int l = cfg.decLayers() / 2; l < cfg.decLayers(); l++) {
                for (int h = 0; h < cfg.decHeads(); h++) {
                    hs.add(new int[]{l, h});
                }
            }
            alignmentHeads = hs.toArray(new int[0][]);
        }
        suppress = new boolean[cfg.vocab()];
        Object st = gen.get("suppress_tokens");
        if (st instanceof List<?> list) {
            for (Object o : list) {
                int id = ((Number) o).intValue();
                if (id >= 0 && id < suppress.length) {
                    suppress[id] = true;
                }
            }
        }
        for (int id : new int[]{transcribeId, translateId, sot, prevId, tok.tokenId("<|startoflm|>"), noSpeechId}) {
            if (id >= 0 && id < suppress.length) {
                suppress[id] = true;
            }
        }
    }

    private int special(String name, int fallback) {
        int id = tok.tokenId(name);
        if (id < 0) {
            id = fallback;
        }
        if (id < 0) {
            throw new IllegalArgumentException("the tokenizer has no " + name + " token: not a Whisper tokenizer");
        }
        return id;
    }

    private int firstId(String... names) {
        for (String n : names) {
            int id = tok.tokenId(n);
            if (id >= 0) {
                return id;
            }
        }
        return -1;
    }

    public WhisperConfig config() {
        return cfg;
    }

    public WhisperNet net() {
        return net;
    }

    public boolean multilingual() {
        return !langIds.isEmpty();
    }

    // ------------------------------------------------------------------ transcribe

    private record DecodeResult(int[] tokens, double sumLogprob, double avgLogprob, double noSpeech, double temperature,
                                double compression, String text) {
    }

    /** Transcribes (or translates) 16 kHz mono {@code audio}. */
    public Transcript transcribe(float[] audio, Options o) {
        int windowFrames = cfg.windowFrames();
        int windowSamples = windowFrames * LogMel.HOP;
        float[][] m = mel.compute(audio, windowSamples);
        int contentFrames = m[0].length - windowFrames;
        int stride = windowFrames / cfg.sourcePos();
        double timePrecision = stride * (double) LogMel.HOP / LogMel.SAMPLE_RATE;
        double duration = (double) audio.length / LogMel.SAMPLE_RATE;
        String language = o.language() == null ? null : Languages.code(o.language());
        if (o.language() != null && (language == null || (multilingual() && !langIds.containsKey(language)))) {
            throw new IllegalArgumentException("unsupported language '" + o.language() + "'");
        }
        if (!multilingual()) {
            if (language != null && !language.equals("en")) {
                throw new IllegalArgumentException("this is an English-only model: language must be 'en'");
            }
            if (o.translate()) {
                throw new IllegalArgumentException("this is an English-only model: it cannot translate");
            }
            language = "en";
        }
        int[] promptTokens = new int[0];
        if (o.prompt() != null && !o.prompt().isBlank()) {
            int[] t = tok.encode(" " + o.prompt().strip(), false);
            promptTokens = Arrays.copyOfRange(t, Math.max(0, t.length - (cfg.targetPos() / 2 - 1)), t.length);
        }
        List<Transcript.Segment> segments = new ArrayList<>();
        List<Integer> all = new ArrayList<>();      // text tokens decoded so far (the next window's prompt)
        int promptResetSince = 0;
        double lastSpeech = 0.0;
        Map<String, Double> langProbs = Map.of();
        int seek = 0;
        Random rnd = new Random(o.seed());
        while (seek < contentFrames) {
            if (o.cancelled().getAsBoolean()) {
                throw new Cancelled();
            }
            double timeOffset = seek * (double) LogMel.HOP / LogMel.SAMPLE_RATE;
            int segSize = Math.min(windowFrames, contentFrames - seek);
            double segDuration = segSize * (double) LogMel.HOP / LogMel.SAMPLE_RATE;
            float[][] window = new float[cfg.nMels()][windowFrames];
            for (int i = 0; i < cfg.nMels(); i++) {
                System.arraycopy(m[i], seek, window[i], 0, Math.min(windowFrames, m[i].length - seek));
            }
            try (WhisperNet.Session s = net.encode(window)) {
                if (language == null) {
                    Map<String, Double> probs = detect(s);
                    language = probs.keySet().iterator().next();
                    langProbs = probs;
                }
                int[] prev;
                if (o.conditionOnPrevious() || all.isEmpty()) {
                    List<Integer> tail = all.subList(promptResetSince, all.size());
                    int keep = Math.min(tail.size(), cfg.targetPos() / 2 - 1 - promptTokens.length);
                    int[] p = new int[promptTokens.length + Math.max(0, keep)];
                    System.arraycopy(promptTokens, 0, p, 0, promptTokens.length);
                    for (int i = 0; i < keep; i++) {
                        p[promptTokens.length + i] = tail.get(tail.size() - keep + i);
                    }
                    prev = p;
                } else {
                    prev = promptTokens;
                }
                DecodeResult r = decodeWithFallback(s, prev, language, o, rnd);
                int[] tokens = r.tokens();
                if (noSpeechId >= 0 && r.noSpeech() > NO_SPEECH_LIMIT && r.avgLogprob() <= LOGPROB_LIMIT) {
                    seek += segSize;       // silence: skip the window
                    o.progress().accept(Math.min(1.0, (double) seek / Math.max(1, contentFrames)));
                    continue;
                }
                List<int[]> pieces = new ArrayList<>();          // token runs of the segments
                List<double[]> times = new ArrayList<>();        // start, end
                int previousSeek = seek;
                if (o.timestamps()) {
                    boolean[] isTs = new boolean[tokens.length];
                    for (int i = 0; i < tokens.length; i++) {
                        isTs[i] = tokens[i] >= timestampBegin;
                    }
                    boolean singleEnding = tokens.length >= 2 && !isTs[tokens.length - 2] && isTs[tokens.length - 1];
                    List<Integer> slices = new ArrayList<>();
                    for (int i = 0; i + 1 < tokens.length; i++) {
                        if (isTs[i] && isTs[i + 1]) {
                            slices.add(i + 1);
                        }
                    }
                    if (!slices.isEmpty()) {
                        if (singleEnding) {
                            slices.add(tokens.length);
                        }
                        int last = 0;
                        for (int cur : slices) {
                            int[] sl = Arrays.copyOfRange(tokens, last, cur);
                            double st = timeOffset + (sl[0] - timestampBegin) * timePrecision;
                            double en = timeOffset + (sl[sl.length - 1] - timestampBegin) * timePrecision;
                            pieces.add(sl);
                            times.add(new double[]{st, en});
                            last = cur;
                        }
                        if (singleEnding) {
                            seek += segSize;
                        } else {
                            int lastTs = tokens[last - 1] - timestampBegin;
                            seek += lastTs * stride;
                        }
                    } else {
                        double dur = segDuration;
                        int lastTsTok = -1;
                        for (int i = tokens.length - 1; i >= 0; i--) {
                            if (isTs[i]) {
                                lastTsTok = tokens[i];
                                break;
                            }
                        }
                        if (lastTsTok >= 0 && lastTsTok != timestampBegin) {
                            dur = (lastTsTok - timestampBegin) * timePrecision;
                        }
                        pieces.add(tokens);
                        times.add(new double[]{timeOffset, timeOffset + dur});
                        seek += segSize;
                    }
                } else {
                    pieces.add(tokens);
                    times.add(new double[]{timeOffset, timeOffset + segDuration});
                    seek += segSize;
                }
                if (seek <= previousSeek) {
                    seek = previousSeek + segSize;        // never stand still
                }
                List<int[]> segTokens = new ArrayList<>();
                List<String> segTexts = new ArrayList<>();
                List<double[]> segTimes = new ArrayList<>();
                for (int i = 0; i < pieces.size(); i++) {
                    int[] textTok = Arrays.stream(pieces.get(i)).filter(t -> t < eot).toArray();
                    String text = textTok.length == 0 ? "" : tok.decode(textTok);
                    if (text.isBlank()) {
                        continue;
                    }
                    segTokens.add(textTok);
                    segTexts.add(text);
                    segTimes.add(times.get(i).clone());
                }
                List<List<Transcript.Word>> segWords = new ArrayList<>();
                for (int i = 0; i < segTokens.size(); i++) {
                    segWords.add(List.of());
                }
                if (o.wordTimestamps() && !segTokens.isEmpty()) {
                    lastSpeech = addWordTimestamps(s, language, o.translate(), segTokens, segTimes, segWords, segSize,
                            previousSeek * (double) LogMel.HOP / LogMel.SAMPLE_RATE, lastSpeech);
                }
                for (int i = 0; i < segTokens.size(); i++) {
                    double[] tm = segTimes.get(i);
                    Transcript.Segment seg = new Transcript.Segment(segments.size(), previousSeek, tm[0], tm[1], segTexts.get(i),
                            segTokens.get(i), r.temperature(), r.avgLogprob(), compressionRatio(segTexts.get(i)), r.noSpeech(),
                            segWords.get(i));
                    segments.add(seg);
                    for (int t : segTokens.get(i)) {
                        all.add(t);
                    }
                    o.onSegment().accept(seg);
                }
                if (r.temperature() > 0.5) {
                    promptResetSince = all.size();
                }
            }
            o.progress().accept(Math.min(1.0, (double) seek / Math.max(1, contentFrames)));
        }
        StringBuilder text = new StringBuilder();
        for (Transcript.Segment sg : segments) {
            text.append(sg.text());
        }
        return new Transcript(language, o.translate() ? "translate" : "transcribe", duration, text.toString().strip(),
                segments, langProbs);
    }


    // ------------------------------------------------------------------ word timestamps

    /**
     * OpenAI's {@code add_word_timestamps} for the segments of one window: one pass of the decoder over
     * {@code [sot sequence, <|notimestamps|>, text tokens, <|endoftext|>]} returns the cross-attention of the alignment heads;
     * {@link WordAligner} turns it into word times, which are cut up per segment. Segment start / end times are adjusted to
     * the words at their edges like the reference does. Returns the new last speech timestamp.
     */
    private double addWordTimestamps(WhisperNet.Session s, String language, boolean translate, List<int[]> segTokens,
                                     List<double[]> segTimes, List<List<Transcript.Word>> segWords, int numFrames,
                                     double timeOffset, double lastSpeech) {
        List<Integer> text = new ArrayList<>();
        for (int[] t : segTokens) {
            for (int x : t) {
                text.add(x);
            }
        }
        int[] textArr = text.stream().mapToInt(Integer::intValue).toArray();
        List<Integer> seq = new ArrayList<>();
        seq.add(sot);
        if (multilingual()) {
            seq.add(langIds.get(language));
            seq.add(translate ? translateId : transcribeId);
        }
        int skip = seq.size();
        seq.add(notimestamps);
        for (int x : textArr) {
            seq.add(x);
        }
        seq.add(eot);
        int[] tokens = seq.stream().mapToInt(Integer::intValue).toArray();
        int used = Math.min(cfg.sourcePos(), numFrames / 2);
        float[][] weights;
        try {
            weights = s.crossAttention(tokens, alignmentHeads);
        } catch (UnsupportedOperationException e) {
            return lastSpeech;
        }
        List<WordAligner.Timing> alignment = WordAligner.align(weights, tokens.length, cfg.sourcePos(), used, skip, textArr,
                language, tok, eot);
        if (alignment.isEmpty()) {
            return lastSpeech;
        }
        double[] durations = alignment.stream().mapToDouble(t -> t.end - t.start).filter(d -> d != 0).sorted().toArray();
        double median = 0;
        if (durations.length > 0) {
            median = durations.length % 2 == 1 ? durations[durations.length / 2]
                    : (durations[durations.length / 2 - 1] + durations[durations.length / 2]) / 2;
        }
        median = Math.min(0.7, median);
        double maxDur = median * 2;
        WordAligner.tidy(alignment);
        int wordIndex = 0;
        for (int si = 0; si < segTokens.size(); si++) {
            int saved = 0;
            List<Transcript.Word> words = new ArrayList<>();
            double[] sw = null;                     // start / end of each kept word, mutable for the clean-up below
            List<double[]> times = new ArrayList<>();
            List<String> labels = new ArrayList<>();
            while (wordIndex < alignment.size() && saved < segTokens.get(si).length) {
                WordAligner.Timing t = alignment.get(wordIndex);
                if (!t.word.isEmpty()) {
                    labels.add(t.word);
                    times.add(new double[]{round2(timeOffset + t.start), round2(timeOffset + t.end)});
                }
                saved += t.tokens.length;
                wordIndex++;
            }
            double[] seg = segTimes.get(si);
            if (!times.isEmpty()) {
                double[] first = times.getFirst();
                double[] lastW = times.getLast();
                if (first[1] - lastSpeech > median * 4
                        && (first[1] - first[0] > maxDur || (times.size() > 1 && times.get(1)[1] - first[0] > maxDur * 2))) {
                    if (times.size() > 1 && times.get(1)[1] - times.get(1)[0] > maxDur) {
                        double boundary = Math.max(times.get(1)[1] / 2, times.get(1)[1] - maxDur);
                        first[1] = boundary;
                        times.get(1)[0] = boundary;
                    }
                    first[0] = Math.max(timeOffset, first[1] - maxDur);
                }
                if (seg[0] < first[1] && seg[0] - 0.5 > first[0]) {
                    first[0] = Math.max(0, Math.min(first[1] - median, seg[0]));
                } else {
                    seg[0] = first[0];
                }
                if (seg[1] > lastW[0] && seg[1] + 0.5 < lastW[1]) {
                    lastW[1] = Math.max(lastW[0] + median, seg[1]);
                } else {
                    seg[1] = lastW[1];
                }
                lastSpeech = seg[1];
            }
            for (int i = 0; i < labels.size(); i++) {
                words.add(new Transcript.Word(labels.get(i), times.get(i)[0], times.get(i)[1]));
            }
            segWords.set(si, words);
        }
        return lastSpeech;
    }

    private static double round2(double v) {
        return Math.round(v * 100.0) / 100.0;
    }

    // ------------------------------------------------------------------ language detection

    /** Probabilities of the languages after {@code <|startoftranscript|>}, most likely first (the top five). */
    public Map<String, Double> detectLanguage(float[] audio) {
        int windowSamples = cfg.windowFrames() * LogMel.HOP;
        float[][] m = mel.compute(Arrays.copyOf(audio, Math.min(audio.length, windowSamples)),
                windowSamples - Math.min(audio.length, windowSamples));
        try (WhisperNet.Session s = net.encode(m)) {
            return detect(s);
        }
    }

    private Map<String, Double> detect(WhisperNet.Session s) {
        float[] logits = new float[cfg.vocab()];
        s.decode(new int[]{sot}, 0, 1, 0, logits);
        List<Map.Entry<String, Integer>> langs = new ArrayList<>(langIds.entrySet());
        double max = Double.NEGATIVE_INFINITY;
        for (var e : langs) {
            max = Math.max(max, logits[e.getValue()]);
        }
        double sum = 0;
        double[] p = new double[langs.size()];
        for (int i = 0; i < p.length; i++) {
            p[i] = Math.exp(logits[langs.get(i).getValue()] - max);
            sum += p[i];
        }
        Integer[] order = new Integer[p.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        final double[] pp = p;
        Arrays.sort(order, (a, b) -> Double.compare(pp[b], pp[a]));
        Map<String, Double> out = new LinkedHashMap<>();
        for (int i = 0; i < Math.min(5, order.length); i++) {
            out.put(langs.get(order[i]).getKey(), p[order[i]] / sum);
        }
        return out;
    }

    // ------------------------------------------------------------------ decoding one window

    private DecodeResult decodeWithFallback(WhisperNet.Session s, int[] prompt, String language, Options o, Random rnd) {
        double[] temps = o.temperature() > 0 ? new double[]{o.temperature()}
                : o.fallback() ? FALLBACK : new double[]{0.0};
        DecodeResult r = null;
        for (double t : temps) {
            r = decodeAt(s, prompt, language, o, t, rnd);
            boolean fb = false;
            if (r.compression() > COMPRESSION_LIMIT) {
                fb = true;
            }
            if (r.avgLogprob() < LOGPROB_LIMIT) {
                fb = true;
            }
            if (noSpeechId >= 0 && r.noSpeech() > NO_SPEECH_LIMIT && r.avgLogprob() < LOGPROB_LIMIT) {
                fb = false;       // silence is not a reason to retry
            }
            if (!fb) {
                break;
            }
        }
        return r;
    }

    private DecodeResult decodeAt(WhisperNet.Session s, int[] prompt, String language, Options o, double temperature,
                                  Random rnd) {
        int n = temperature > 0 ? BEST_OF : 1;
        DecodeResult best = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            DecodeResult r = decodeOnce(s, prompt, language, o, temperature, rnd);
            double score = r.sumLogprob() / Math.max(1, r.tokens().length);
            if (best == null || score > bestScore) {
                best = r;
                bestScore = score;
            }
        }
        return best;
    }

    private DecodeResult decodeOnce(WhisperNet.Session s, int[] prompt, String language, Options o, double temperature,
                                    Random rnd) {
        int[] head = new int[prompt.length > 0 ? prompt.length + 2 : 1];
        int h = 0;
        if (prompt.length > 0) {
            head[h++] = prevId;
            for (int t : prompt) {
                head[h++] = t;
            }
        }
        head[h] = sot;
        int[] tail;
        if (multilingual()) {
            tail = o.timestamps() ? new int[]{langIds.get(language), o.translate() ? translateId : transcribeId}
                    : new int[]{langIds.get(language), o.translate() ? translateId : transcribeId, notimestamps};
        } else {
            tail = o.timestamps() ? new int[0] : new int[]{notimestamps};
        }
        // head includes prev tokens; the sot is its last element
        int[] head2 = Arrays.copyOf(head, h + 1);
        float[] logits = new float[cfg.vocab()];
        s.decode(head2, 0, head2.length, 0, logits);
        double noSpeech = 0;
        if (noSpeechId >= 0) {
            noSpeech = softmaxAt(logits, noSpeechId);
        }
        int pos = head2.length;
        if (tail.length > 0) {
            s.decode(tail, 0, tail.length, pos, logits);
            pos += tail.length;
        }
        int sampleBegin = pos;
        List<Integer> out = new ArrayList<>();
        double sumLogprob = 0;
        int maxNew = Math.min(Math.min(o.maxTokens(), cfg.targetPos() / 2), cfg.targetPos() - sampleBegin);
        int[] one = new int[1];
        for (int step = 0; step < maxNew; step++) {
            applyFilters(logits, out, step == 0, o.timestamps());
            int next = temperature > 0 ? sample(logits, temperature, rnd) : argmax(logits);
            sumLogprob += logSoftmaxAt(logits, next);
            if (next == eot) {
                break;
            }
            out.add(next);
            if (step + 1 == maxNew) {
                break;
            }
            if (o.cancelled().getAsBoolean()) {
                throw new Cancelled();
            }
            one[0] = next;
            s.decode(one, 0, 1, pos++, logits);
        }
        int[] tokens = out.stream().mapToInt(Integer::intValue).toArray();
        int[] textTok = Arrays.stream(tokens).filter(t -> t < eot).toArray();
        String text = textTok.length == 0 ? "" : tok.decode(textTok);
        return new DecodeResult(tokens, sumLogprob, sumLogprob / (tokens.length + 1), noSpeech, temperature,
                compressionRatio(text), text);
    }

    /** The logit filters of OpenAI's DecodingTask: blank / eot at the start, suppress list, timestamp rules. */
    private void applyFilters(float[] logits, List<Integer> out, boolean first, boolean timestamps) {
        if (first) {
            if (blankId >= 0) {
                logits[blankId] = Float.NEGATIVE_INFINITY;
            }
            logits[eot] = Float.NEGATIVE_INFINITY;
        }
        for (int i = 0; i < suppress.length; i++) {
            if (suppress[i]) {
                logits[i] = Float.NEGATIVE_INFINITY;
            }
        }
        if (!timestamps) {
            return;
        }
        logits[notimestamps] = Float.NEGATIVE_INFINITY;
        int len = out.size();
        boolean lastWasTs = len >= 1 && out.get(len - 1) >= timestampBegin;
        boolean penultimateWasTs = len < 2 || out.get(len - 2) >= timestampBegin;
        if (lastWasTs) {
            if (penultimateWasTs) {       // has to be text (or eot)
                Arrays.fill(logits, timestampBegin, logits.length, Float.NEGATIVE_INFINITY);
            } else {                      // cannot be text
                Arrays.fill(logits, 0, eot, Float.NEGATIVE_INFINITY);
            }
        }
        int lastTs = -1;
        for (int i = len - 1; i >= 0; i--) {
            if (out.get(i) >= timestampBegin) {
                lastTs = out.get(i);
                break;
            }
        }
        if (lastTs >= 0) {
            int from = lastWasTs && !penultimateWasTs ? lastTs : lastTs + 1;
            Arrays.fill(logits, timestampBegin, Math.min(logits.length, from), Float.NEGATIVE_INFINITY);
        }
        if (len == 0) {
            Arrays.fill(logits, 0, timestampBegin, Float.NEGATIVE_INFINITY);
            int lastAllowed = timestampBegin + maxInitialTimestamp;
            if (lastAllowed + 1 < logits.length) {
                Arrays.fill(logits, lastAllowed + 1, logits.length, Float.NEGATIVE_INFINITY);
            }
        }
        // sample a timestamp when their total probability beats every single text token
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v);
        }
        double sumAll = 0;
        double sumTs = 0;
        double maxText = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.length; i++) {
            double e = Math.exp(logits[i] - max);
            sumAll += e;
            if (i >= timestampBegin) {
                sumTs += e;
            } else {
                maxText = Math.max(maxText, logits[i]);
            }
        }
        double tsLogprob = Math.log(sumTs) + 0 - Math.log(sumAll);
        double maxTextLogprob = maxText - max - Math.log(sumAll);
        if (tsLogprob > maxTextLogprob) {
            Arrays.fill(logits, 0, timestampBegin, Float.NEGATIVE_INFINITY);
        }
    }

    private static int argmax(float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) {
            if (a[i] > a[best]) {
                best = i;
            }
        }
        return best;
    }

    private static int sample(float[] logits, double temperature, Random rnd) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v);
        }
        double sum = 0;
        double[] p = new double[logits.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = Float.isInfinite(logits[i]) && logits[i] < 0 ? 0 : Math.exp((logits[i] - max) / temperature);
            sum += p[i];
        }
        double u = rnd.nextDouble() * sum;
        double acc = 0;
        for (int i = 0; i < p.length; i++) {
            acc += p[i];
            if (u < acc) {
                return i;
            }
        }
        return argmax(logits);
    }

    private static double softmaxAt(float[] logits, int id) {
        return Math.exp(logSoftmaxAt(logits, id));
    }

    private static double logSoftmaxAt(float[] logits, int id) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v);
        }
        double sum = 0;
        for (float v : logits) {
            sum += Math.exp(v - max);
        }
        return logits[id] - max - Math.log(sum);
    }

    static double compressionRatio(String text) {
        byte[] b = text.getBytes(StandardCharsets.UTF_8);
        if (b.length == 0) {
            return 0;
        }
        Deflater d = new Deflater();
        d.setInput(b);
        d.finish();
        byte[] buf = new byte[b.length + 64];
        int n = 0;
        while (!d.finished()) {
            if (n == buf.length) {
                buf = Arrays.copyOf(buf, buf.length * 2);
            }
            n += d.deflate(buf, n, buf.length - n);
        }
        d.end();
        return (double) b.length / n;
    }
}
