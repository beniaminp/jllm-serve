package org.robomiri.jllm.audio;

import org.robomiri.jllm.ModelLoader;
import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.hub.HuggingFaceHub;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * OuteTTS 0.2 / 0.3 text-to-speech. A Qwen2 language model (run by the ordinary {@link Engine}: the same CPU and Metal
 * backends, quantization and KV cache as a chat model) reads a prompt of the words to say and writes the speech as
 * audio-codec tokens ({@code <|1234|>}, 75 per second, a code of the WavTokenizer codebook each); {@link WavTokenizerDecoder}
 * turns the codes into a 24 kHz waveform.
 *
 * <p>Prompt (OuteTTS interface v2): {@code <|im_start|>\n<|text_start|>} the reference speaker's words and the new words
 * joined by {@code <|space|>}, with {@code <|comma|>}-style punctuation tokens fused to the words ({@link OuteText}),
 * {@code <|text_end|>\n<|audio_start|>\n}, then the reference speaker's audio as {@code word<|t_0.40|><|7|><|1090|>...}
 * per word, each followed by {@code <|space|>\n}. The model continues with the same pattern for the new words; every audio
 * token it emits is a codec frame. A voice is a reference speaker (a short recording with its transcript, word by word
 * with the codes of each word) shipped with the OuteTTS package; without one the model invents a voice and is unstable.
 */
public final class OuteTts implements SpeechModel {

    private static final int AUDIO_CODES = 4100;     // <|0|> .. <|4099|> (the codebook has 4096 entries)
    private static final String DEFAULT_VOICE = "en_male_1";
    private static final int MAX_ATTEMPTS = 3;
    private static final boolean TRACE = Boolean.getBoolean("jllm.audio.trace");

    private record SWord(String word, double duration, int[] codes) {
    }

    private record Speaker(String name, String text, List<SWord> words) {
    }

    private final ModelLoader.Loaded lm;
    private final Tokenizer tok;
    private final WavTokenizerDecoder codec;
    private final Map<String, Speaker> speakers;
    private final AudioSettings cfg;
    private final int audioBase;
    private final int ctx;
    private final String lmName;

    private OuteTts(ModelLoader.Loaded lm, WavTokenizerDecoder codec, Map<String, Speaker> speakers, AudioSettings cfg, int ctx) {
        this.lm = lm;
        this.tok = lm.engine().tokenizer();
        this.codec = codec;
        this.speakers = speakers;
        this.cfg = cfg;
        this.ctx = ctx;
        this.audioBase = tok.tokenId("<|0|>");
        this.lmName = lm.engine().backend().name();
        if (audioBase < 0 || tok.tokenId("<|" + (AUDIO_CODES - 1) + "|>") != audioBase + AUDIO_CODES - 1) {
            throw new IllegalArgumentException("the tokenizer has no contiguous <|0|> .. <|4099|> audio tokens: not an OuteTTS model");
        }
    }

    /** Whether {@code dir} holds an OuteTTS language model: a Qwen2 checkpoint whose tokenizer has the audio tokens. */
    public static boolean isOuteTts(Path dir) {
        Path cfg = dir.resolve("tokenizer_config.json");
        Path tj = dir.resolve("tokenizer.json");
        try {
            if (!Files.isRegularFile(dir.resolve("config.json")) || !Files.isRegularFile(cfg)) {
                return false;
            }
            String c = Files.readString(cfg);
            return c.contains("<|audio_start|>") && c.contains("<|text_start|>") && c.contains("<|space|>")
                    && !c.contains("<|c1_0|>") && Files.isRegularFile(tj);       // OuteTTS 1.0 uses another prompt format
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    /**
     * Loads the language model from {@code dir} (a model folder under some {@code model.dir}) and the WavTokenizer GGUF
     * ({@code audio.codec-repo} / {@code audio.codec-file}, looked for in the model folders first, downloaded when missing).
     */
    public static OuteTts load(Settings s, HuggingFaceHub hub, Path dir) throws IOException, InterruptedException {
        AudioSettings a = AudioSettings.of(s);
        Map<String, Speaker> speakers = loadSpeakers();
        Path codecFile = findCodec(s, hub, a, dir);
        int threads = AudioSettings.cpuThreads(s);
        WavTokenizerDecoder codec = new WavTokenizerDecoder(codecFile, threads);
        try {
            Settings ls = s.with(Map.of("model.dir", dir.toAbsolutePath().getParent().toString(), "model.repo",
                    dir.getFileName().toString(), "model.quant", a.ttsQuant(), "backend.ctx", String.valueOf(a.ttsCtx()),
                    "model.gguf-file", "", "model.gguf-quant", "", "model.revision", "main"));
            ModelLoader.Loaded lm;
            try {
                lm = ModelLoader.load(ls, hub, dir.getFileName().toString());
            } catch (IOException | RuntimeException e) {
                throw e;
            } catch (Exception e) {
                throw new IOException("cannot load " + dir.getFileName() + ": " + e.getMessage(), e);
            }
            try {
                return new OuteTts(lm, codec, speakers, a, a.ttsCtx());
            } catch (RuntimeException e) {
                lm.close();
                throw e;
            }
        } catch (IOException | RuntimeException e) {
            codec.close();
            throw e;
        }
    }

    private static Path findCodec(Settings s, HuggingFaceHub hub, AudioSettings a, Path dir) throws IOException, InterruptedException {
        String repoDir = a.codecRepo().replace('/', '_');
        Path[] candidates = {dir.toAbsolutePath().getParent().resolve(repoDir).resolve(a.codecFile()),
                Path.of(s.get("model.dir")).resolve(repoDir).resolve(a.codecFile())};
        for (Path c : candidates) {
            if (Files.isRegularFile(c)) {
                return c;
            }
        }
        Path target = candidates[0].getParent();
        Files.createDirectories(target);
        return hub.ensure(a.codecRepo(), "main", target, a.codecFile());
    }

    private static Map<String, Speaker> loadSpeakers() throws IOException {
        try (InputStream in = OuteTts.class.getResourceAsStream("outetts-speakers.json")) {
            if (in == null) {
                throw new IOException("outetts-speakers.json is missing from the jar");
            }
            List<Object> all = (List<Object>) Json.parse(new String(in.readAllBytes(), StandardCharsets.UTF_8));
            Map<String, Speaker> out = new LinkedHashMap<>();
            for (Object o : all) {
                Map<String, Object> m = (Map<String, Object>) o;
                List<SWord> words = new ArrayList<>();
                for (Object w : Json.list(m, "words")) {
                    Map<String, Object> wm = (Map<String, Object>) w;
                    List<Object> codes = Json.list(wm, "codes");
                    words.add(new SWord(Json.str(wm, "word", ""), Json.dbl(wm, "duration", 0.0),
                            codes.stream().mapToInt(c -> ((Number) c).intValue()).toArray()));
                }
                Speaker sp = new Speaker(Json.str(m, "speaker", ""), Json.str(m, "text", ""), words);
                // the profile's words must be what the text processor makes of its text, else the prompt cannot be built
                List<OuteText.Word> proc = OuteText.process(sp.text());
                boolean ok = proc.size() == words.size();
                for (int i = 0; ok && i < proc.size(); i++) {
                    ok = proc.get(i).word().equals(words.get(i).word());
                }
                if (ok) {
                    out.put(sp.name(), sp);
                } else {
                    System.err.println("outetts: speaker " + sp.name() + " skipped (its words do not match its text)");
                }
            }
            return out;
        }
    }

    // ------------------------------------------------------------------ SpeechModel

    @Override
    public int sampleRate() {
        return WavTokenizerDecoder.SAMPLE_RATE;
    }

    @Override
    public List<String> voices() {
        return List.copyOf(speakers.keySet());
    }

    @Override
    public String describe() {
        return "OuteTTS (" + lm.weights().arch.core().architecture() + " LM " + lmName + " + WavTokenizer codec on the CPU), "
                + speakers.size() + " voices";
    }

    /** The prompt for {@code text} read by {@code speaker} (null: the model's own voice), as the text the tokenizer sees. */
    String prompt(String text, String voice) {
        List<OuteText.Word> words = OuteText.process(text);
        if (words.isEmpty()) {
            throw new IllegalArgumentException("nothing to say: the text has no letters or digits (OuteTTS reads Latin script;"
                    + " Chinese, Japanese and Korean are not supported)");
        }
        String wordsText = OuteText.promptText(words);
        Speaker sp = voice == null ? null : speakers.get(voice);
        if (voice != null && sp == null) {
            throw new IllegalArgumentException("unknown voice '" + voice + "'; available: " + voices());
        }
        StringBuilder audio = new StringBuilder();
        if (sp != null) {
            List<OuteText.Word> sw = OuteText.process(sp.text());
            for (int i = 0; i < sw.size(); i++) {
                SWord w = sp.words().get(i);
                OuteText.Word pw = sw.get(i);
                if (i > 0) {
                    audio.append("<|space|>\n");
                }
                pw.before().forEach(audio::append);
                audio.append(w.word());
                pw.after().forEach(audio::append);
                audio.append(String.format(Locale.ROOT, "<|t_%.2f|>", w.duration()));
                for (int c : w.codes()) {
                    audio.append("<|").append(c).append("|>");
                }
            }
            wordsText = OuteText.promptText(sw) + "<|space|>" + wordsText;
        }
        StringBuilder p = new StringBuilder("<|im_start|>\n<|text_start|>").append(wordsText).append("<|text_end|>\n")
                .append("<|audio_start|>\n");
        if (sp != null) {
            p.append(audio).append("<|space|>\n");
        }
        return p.toString();
    }

    @Override
    public float[] synthesize(String text, String voice, Options o) {
        String v = voice == null || voice.isBlank() ? (speakers.containsKey(DEFAULT_VOICE) ? DEFAULT_VOICE
                : speakers.keySet().stream().findFirst().orElse(null)) : voice;
        int[] ids = tok.encode(prompt(text, v), true);
        int maxNew = Math.min(o.maxFrames() > 0 ? o.maxFrames() * 2 : 4096, ctx - ids.length - 8);
        if (maxNew < 64) {
            throw new IllegalArgumentException("the prompt (" + ids.length + " tokens) leaves no room in the context of " + ctx
                    + " tokens (audio.tts.ctx)");
        }
        double temp = o.temperature();
        int words = OuteText.process(text).size();
        // a sampled run sometimes ends after a few frames, writes silence, or never ends (the model is not infallible at
        // temperature > 0): the speech of a word lasts between ~0.08 s and 2 s, so a result outside that is generated again with
        // another seed, up to 3 times
        double minSeconds = 0.08 * words;
        double maxSeconds = 2.0 * words + 3.0;
        Engine.Result r = null;
        int n = 0;
        float[] pcm = null;
        long c0 = 0;
        for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
            long seed = o.seed() < 0 ? -1 : o.seed() + 7919L * attempt;
            SamplingParams sp = new SamplingParams((float) temp, cfg.ttsTopK(), (float) cfg.ttsTopP(), 0f,
                    (float) cfg.ttsRepetitionPenalty(), 0f, 0f, seed, maxNew, List.of());
            r = lm.engine().generate(ids, sp, t -> { }, o.cancelled());
            if (r.finishReason().equals("cancelled")) {
                throw new WhisperTranscriber.Cancelled();
            }
            n = 0;
            int[] codes = new int[r.tokens().length];
            for (int t : r.tokens()) {
                int c = t - audioBase;
                if (c >= 0 && c < AUDIO_CODES) {
                    codes[n++] = c;
                }
            }
            if (n == 0) {
                if (attempt == MAX_ATTEMPTS - 1) {
                    throw new IllegalStateException("the model produced no audio for this text");
                }
                continue;
            }
            c0 = System.nanoTime();
            pcm = codec.decode(java.util.Arrays.copyOf(codes, n));
            float[] trimmed = Speech.trimSilence(pcm, sampleRate());
            double spoken = (double) trimmed.length / sampleRate() - 0.32;       // minus the margins trimSilence keeps
            boolean plausible = spoken >= minSeconds && n / 75.0 <= maxSeconds;
            if (cfg.ttsTrim()) {
                pcm = trimmed;
            }
            if (plausible || temp == 0 || attempt == MAX_ATTEMPTS - 1) {
                break;
            }
            if (TRACE) {
                System.err.printf("tts: %d words came out as %.2f s of speech in %.2f s of audio, trying again%n", words, spoken,
                        n / 75.0);
            }
        }
        o.progress().accept(n);
        if (TRACE) {
            Engine.Stats st = r.stats();
            System.err.printf("tts: prompt %d tokens (%d cached), prefill %.0f ms; %d tokens in %.0f ms (%.1f tok/s), %d frames = %.2f s;"
                    + " codec %.0f ms%n", st.promptTokens(), st.cachedTokens(), st.prefillMs(), st.generatedTokens(),
                    st.decodeMs(), st.tokPerSec(), n, n / 75.0, (System.nanoTime() - c0) / 1e6);
        }
        return pcm;
    }

    /** The tokens the model writes for {@code text} (for tests and tools). */
    public int[] generateTokens(String text, String voice, double temperature, int maxNew) {
        int[] ids = tok.encode(prompt(text, voice), true);
        SamplingParams sp = new SamplingParams((float) temperature, cfg.ttsTopK(), (float) cfg.ttsTopP(), 0f,
                (float) cfg.ttsRepetitionPenalty(), 0f, 0f, 1, maxNew, List.of());
        return lm.engine().generate(ids, sp, t -> { }, () -> false).tokens();
    }

    /** The audio codes among {@code tokens}. */
    public int[] codesOf(int[] tokens) {
        return java.util.Arrays.stream(tokens).map(t -> t - audioBase).filter(c -> c >= 0 && c < AUDIO_CODES).toArray();
    }

    /** The audio codes the model writes for {@code text} (for tests and tools), without the decoder. */
    public int[] generateCodes(String text, String voice, double temperature, int maxNew) {
        return codesOf(generateTokens(text, voice, temperature, maxNew));
    }

    /** The prompt text for {@code text} and {@code voice} (null voice: none, the model invents one). */
    public String promptText(String text, String voice) {
        return prompt(text, voice);
    }

    /** The prompt token ids for {@code text} and {@code voice}. */
    public int[] promptIds(String text, String voice) {
        return tok.encode(prompt(text, voice), true);
    }

    public Engine engine() {
        return lm.engine();
    }

    public WavTokenizerDecoder codec() {
        return codec;
    }

    @Override
    public void close() {
        lm.close();
        codec.close();
    }
}
