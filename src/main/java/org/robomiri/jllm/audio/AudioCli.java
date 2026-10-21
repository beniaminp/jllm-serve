package org.robomiri.jllm.audio;

import org.robomiri.jllm.ModelLoader;
import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.hub.HuggingFaceHub;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * The command line side of the audio models: {@code ./jllm transcribe --model openai/whisper-small --audio clip.wav
 * [--language de] [--translate true] [--format text|json|srt|vtt|verbose_json] [--words true]} and
 * {@code ./jllm speak --model OuteAI/OuteTTS-0.3-500M --prompt "Hello" --voice en_male_1 --out hello.wav [--speed 1.2]},
 * and loading the models {@code ./jllm serve} offers on the audio endpoints ({@code audio.stt-model},
 * {@code audio.tts-model}).
 */
public final class AudioCli {

    private AudioCli() {
    }

    /** Downloads (when missing) and loads the checkpoint named by {@code repo} as an audio model. */
    public static AudioModels.Entry load(Settings s, HuggingFaceHub hub, AudioModels models, String repo) throws Exception {
        Path dir = Path.of(s.get("model.dir")).resolve(repo.replace('/', '_'));
        if (!Files.isRegularFile(dir.resolve("config.json"))) {
            ModelLoader.download(s.with(java.util.Map.of("model.repo", repo, "model.gguf-file", "", "model.gguf-quant", "")), hub,
                    repo);
        }
        return models.load(s, hub, repo, repo, dir);
    }

    public static void transcribe(Settings s, HuggingFaceHub hub, String repo, String audioFile, String language, boolean translate,
                                  String format, boolean words) throws Exception {
        if (audioFile == null || audioFile.isBlank()) {
            throw new IllegalArgumentException("transcribe needs --audio <file>");
        }
        try (AudioModels models = new AudioModels()) {
            AudioModels.Entry e = load(s, hub, models, repo);
            if (e.stt() == null) {
                throw new IllegalArgumentException(repo + " is not a speech-to-text model");
            }
            AudioSettings a = AudioSettings.of(s);
            float[] pcm = new AudioDecoder(a.ffmpeg(), a.maxSeconds()).decode16k(Files.readAllBytes(Path.of(audioFile)));
            WhisperTranscriber.Options o = AudioApi.WhisperDefaults.of(a).apply(WhisperTranscriber.Options.defaults()
                    .withLanguage(language == null || language.isBlank() ? null : language).withTranslate(translate)
                    .withWords(words));
            long t0 = System.nanoTime();
            Transcript t = e.stt().transcriber.transcribe(pcm, o);
            double sec = (System.nanoTime() - t0) / 1e9;
            System.out.print(t.render(format));
            System.err.printf("[%s: %.1f s of audio in %.2f s, real-time factor %.3f, language %s]%n", e.backend(), t.duration(),
                    sec, sec / Math.max(1e-9, t.duration()), t.language());
        }
    }

    public static void speak(Settings s, HuggingFaceHub hub, String repo, String text, String voice, double speed, String out)
            throws Exception {
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("speak needs --prompt \"text\"");
        }
        try (AudioModels models = new AudioModels()) {
            AudioModels.Entry e = load(s, hub, models, repo);
            if (e.tts() == null) {
                throw new IllegalArgumentException(repo + " is not a text-to-speech model");
            }
            AudioSettings a = AudioSettings.of(s);
            long t0 = System.nanoTime();
            float[] pcm = Speech.synthesizeLong(e.tts(), text, voice == null || voice.isBlank() ? null : voice,
                    SpeechModel.Options.defaults().withSpeed(speed).withTemperature(a.ttsTemperature()), a.ttsMaxChars());
            double sec = (System.nanoTime() - t0) / 1e9;
            double audio = (double) pcm.length / e.tts().sampleRate();
            Path file = Path.of(out == null || out.isBlank() ? "speech.wav" : out);
            Files.write(file, Wav.write16(pcm, e.tts().sampleRate()));
            System.err.printf("[%s: %.1f s of audio in %.2f s, real-time factor %.2f, wrote %s]%n", e.tts().describe(), audio,
                    sec, sec / Math.max(1e-9, audio), file);
        }
    }

    static IOException unreachable() {
        return new IOException("unreachable");
    }
}
