package org.robomiri.jllm.audio;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.hub.HuggingFaceHub;
import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

/**
 * The audio models in memory, by the id clients use in the {@code model} field, and the loader that recognises what a
 * downloaded checkpoint is: a Whisper checkpoint is a speech-to-text model, an OuteTTS language model (a Qwen2
 * checkpoint whose vocabulary has the {@code <|audio_start|>} / {@code <|text_start|>} tokens) a text-to-speech model.
 */
public final class AudioModels implements AutoCloseable {

    public enum Kind {
        SPEECH_TO_TEXT("speech-to-text"), TEXT_TO_SPEECH("text-to-speech");

        private final String label;

        Kind(String label) {
            this.label = label;
        }

        public String label() {
            return label;
        }
    }

    /** A loaded model; exactly one of {@code stt} and {@code tts} is set. */
    public record Entry(String id, Kind kind, String source, String backend, long loadedAt, double loadSeconds,
                        WhisperModel stt, SpeechModel tts) {
        public String description() {
            return stt != null ? "Whisper " + stt.weights.cfg.dModel() + "-wide, " + stt.weights.cfg.encLayers() + "+"
                    + stt.weights.cfg.decLayers() + " layers, " + (stt.transcriber.multilingual() ? "multilingual" : "English only")
                    : tts.describe();
        }
    }

    private final Map<String, Entry> models = new LinkedHashMap<>();

    /** What the checkpoint in {@code dir} is, or null when it is not an audio model. */
    public static Kind detect(Path dir) {
        if (WhisperWeights.isWhisper(dir)) {
            return Kind.SPEECH_TO_TEXT;
        }
        return OuteTts.isOuteTts(dir) ? Kind.TEXT_TO_SPEECH : null;
    }

    /** Loads the checkpoint in {@code dir} (its kind is detected) and registers it as {@code id}. */
    public Entry load(Settings s, HuggingFaceHub hub, String id, String source, Path dir) throws IOException,
            InterruptedException {
        Kind kind = detect(dir);
        if (kind == null) {
            throw new IllegalArgumentException(dir.getFileName() + " is not an audio model");
        }
        AudioSettings a = AudioSettings.of(s);
        long t0 = System.nanoTime();
        Entry e;
        if (kind == Kind.SPEECH_TO_TEXT) {
            WhisperModel m = WhisperModel.load(dir, a.backend(), AudioSettings.cpuThreads(s), a.residentMb());
            e = new Entry(id, kind, source, m.net.name(), System.currentTimeMillis(), (System.nanoTime() - t0) / 1e9, m, null);
        } else {
            OuteTts m = OuteTts.load(s, hub, dir);
            e = new Entry(id, kind, source, m.describe(), System.currentTimeMillis(), (System.nanoTime() - t0) / 1e9, null, m);
        }
        synchronized (this) {
            if (models.containsKey(id)) {
                close(e);
                throw new IllegalStateException("a model named '" + id + "' is already loaded");
            }
            models.put(id, e);
        }
        return e;
    }

    public synchronized List<Entry> list() {
        return new ArrayList<>(models.values());
    }

    public synchronized boolean has(String id) {
        return models.containsKey(id);
    }

    public synchronized Entry get(String id) {
        return models.get(id);
    }

    /** The model of {@code kind} a request names; with no (or an unknown) name and exactly one of that kind loaded, that one. */
    public synchronized Entry resolve(Kind kind, String id) {
        if (id != null) {
            Entry e = models.get(id);
            if (e != null && e.kind() == kind) {
                return e;
            }
        }
        List<Entry> same = models.values().stream().filter(e -> e.kind() == kind).toList();
        if (same.size() == 1) {
            return same.getFirst();
        }
        if (same.isEmpty()) {
            throw new NoSuchElementException("no " + kind.label() + " model is loaded: load one (a Whisper checkpoint such as"
                    + " openai/whisper-small for speech-to-text, OuteAI/OuteTTS-0.3-500M for text-to-speech)");
        }
        throw new NoSuchElementException("model '" + id + "' is not a loaded " + kind.label() + " model; loaded: "
                + String.join(", ", same.stream().map(Entry::id).toList()));
    }

    public void unload(String id) {
        Entry e;
        synchronized (this) {
            e = models.remove(id);
        }
        if (e == null) {
            throw new NoSuchElementException("model '" + id + "' is not loaded");
        }
        close(e);
    }

    private static void close(Entry e) {
        if (e.stt() != null) {
            e.stt().close();
        }
        if (e.tts() != null) {
            e.tts().close();
        }
    }

    @Override
    public void close() {
        List<Entry> all;
        synchronized (this) {
            all = new ArrayList<>(models.values());
            models.clear();
        }
        all.forEach(AudioModels::close);
    }

    /** Whether {@code dir} holds a checkpoint config naming one of the audio model types (cheap check for listings). */
    public static boolean looksLikeAudio(Path dir) {
        try {
            return Files.isRegularFile(dir.resolve("config.json")) && detect(dir) != null;
        } catch (RuntimeException e) {
            return false;
        }
    }

    static Map<String, Object> readJson(Path p) throws IOException {
        return Json.parseObject(Files.readString(p));
    }
}
