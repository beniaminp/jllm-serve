package org.robomiri.jllm.audio;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.hub.HuggingFaceHub;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** OuteTTS 0.2 / 0.3 (a Qwen2 language model that writes audio-codec tokens) + the WavTokenizer decoder. */
public final class OuteTts implements SpeechModel {

    private OuteTts() {
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
            return c.contains("<|audio_start|>") && c.contains("<|text_start|>") && Files.isRegularFile(tj);
        } catch (java.io.IOException | RuntimeException e) {
            return false;
        }
    }

    public static OuteTts load(Settings s, HuggingFaceHub hub, Path dir) {
        throw new UnsupportedOperationException("text-to-speech is not implemented yet");
    }

    @Override
    public int sampleRate() {
        return 24000;
    }

    @Override
    public List<String> voices() {
        return List.of();
    }

    @Override
    public String describe() {
        return "OuteTTS";
    }

    @Override
    public float[] synthesize(String text, String voice, Options o) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void close() {
    }
}
