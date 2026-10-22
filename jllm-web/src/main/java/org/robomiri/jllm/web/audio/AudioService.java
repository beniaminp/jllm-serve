package org.robomiri.jllm.web.audio;

import org.robomiri.jllm.ModelLoader;
import org.robomiri.jllm.audio.AudioApi;
import org.robomiri.jllm.audio.AudioDecoder;
import org.robomiri.jllm.audio.AudioModels;
import org.robomiri.jllm.audio.AudioSettings;
import org.robomiri.jllm.audio.Wav;
import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.web.settings.SettingsService;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * The console's audio models (speech-to-text and text-to-speech) and the clip store the chat tools use. Loading goes
 * through {@code ModelManager} (the same Models page, jobs and downloads as for text models); this service holds what
 * was loaded and builds the decoder from the current settings.
 */
@Service
public class AudioService implements DisposableBean {

    private static volatile AudioService instance;

    private final AudioModels models = new AudioModels();
    private final AudioStore store = new AudioStore();
    private final SettingsService settings;
    private volatile AudioDecoder decoder;
    private volatile String decoderKey = "";

    public AudioService(SettingsService settings) {
        this.settings = settings;
        instance = this;
    }

    /** The running service, for the built-in tools (which are not Spring beans); null in a context without one. */
    public static AudioService current() {
        return instance;
    }

    public AudioModels models() {
        return models;
    }

    public AudioStore store() {
        return store;
    }

    /** Whether {@code dir} holds an audio checkpoint (checked after a download, before the text-model loader runs). */
    public boolean isAudio(Path dir) {
        return AudioModels.looksLikeAudio(dir);
    }

    /** Loads the audio checkpoint in {@code dir} as {@code id}. */
    public AudioModels.Entry load(Settings s, String id, String source, Path dir) throws Exception {
        return models.load(s, ModelLoader.hub(s), id, source, dir);
    }

    public boolean has(String id) {
        return models.has(id);
    }

    public void unload(String id) {
        models.unload(id);
    }

    public List<AudioModels.Entry> loaded() {
        return models.list();
    }

    /** Decoder for uploads with the current {@code audio.ffmpeg} and {@code audio.max-seconds}. */
    public AudioDecoder decoder() {
        AudioSettings a = AudioSettings.of(settings.current());
        String key = a.ffmpeg() + "|" + a.maxSeconds();
        if (decoder == null || !key.equals(decoderKey)) {
            decoder = new AudioDecoder(a.ffmpeg(), a.maxSeconds());
            decoderKey = key;
        }
        return decoder;
    }

    public AudioSettings audioSettings() {
        return AudioSettings.of(settings.current());
    }

    /** Stores a complete WAV file and returns its clip. */
    public AudioStore.Clip storeWav(byte[] wav, String label) throws IOException {
        Wav.Audio a = Wav.read(wav);
        return store.put(wav, a.sampleRate(), a.seconds(), label);
    }

    public AudioApi.WhisperDefaults whisperDefaults() {
        return AudioApi.WhisperDefaults.of(audioSettings());
    }

    @Override
    public void destroy() {
        models.close();
    }
}
