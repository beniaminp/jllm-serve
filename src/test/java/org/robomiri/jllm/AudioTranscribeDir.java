package org.robomiri.jllm;

import org.robomiri.jllm.audio.AudioDecoder;
import org.robomiri.jllm.audio.Transcript;
import org.robomiri.jllm.audio.WhisperModel;
import org.robomiri.jllm.audio.WhisperTranscriber;
import org.robomiri.jllm.json.Json;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Transcribes every {@code *.wav} of a directory with one loaded Whisper model (greedy, language given) and writes
 * {@code {"clip": "text", ...}} as JSON, for scripts/audio-accuracy.py:
 * {@code AudioTranscribeDir <whisper dir> <clips dir> <out.json> [cpu|metal] [en]}.
 */
public final class AudioTranscribeDir {

    public static void main(String[] a) throws Exception {
        String backend = a.length > 3 ? a[3] : "auto";
        String language = a.length > 4 ? a[4] : null;
        WhisperModel m = WhisperModel.load(Path.of(a[0]), backend, Runtime.getRuntime().availableProcessors(), 1500);
        AudioDecoder dec = new AudioDecoder("auto", 7200);
        List<Path> files = new ArrayList<>();
        try (var st = Files.list(Path.of(a[1]))) {
            st.filter(p -> p.toString().endsWith(".wav")).sorted().forEach(files::add);
        }
        Map<String, Object> out = new TreeMap<>();
        double audio = 0;
        long t0 = System.nanoTime();
        for (Path f : files) {
            float[] pcm = dec.decode16k(Files.readAllBytes(f));
            audio += pcm.length / 16000.0;
            Transcript t = m.transcriber.transcribe(pcm, WhisperTranscriber.Options.defaults().withLanguage(language));
            out.put(f.getFileName().toString().replace(".wav", ""), t.text());
        }
        double sec = (System.nanoTime() - t0) / 1e9;
        System.err.printf("%d clips, %.0f s of audio in %.1f s on %s (real-time factor %.3f)%n", files.size(), audio, sec,
                m.net.name(), sec / audio);
        Files.writeString(Path.of(a[2]), Json.write(out));
        m.close();
    }
}
