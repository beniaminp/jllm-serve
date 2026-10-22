package org.robomiri.jllm;

import org.robomiri.jllm.audio.AudioDecoder;
import org.robomiri.jllm.audio.LogMel;
import org.robomiri.jllm.audio.OuteTts;
import org.robomiri.jllm.audio.SpeechModel;
import org.robomiri.jllm.audio.Transcript;
import org.robomiri.jllm.audio.WavTokenizerDecoder;
import org.robomiri.jllm.audio.WhisperModel;
import org.robomiri.jllm.audio.WhisperNet;
import org.robomiri.jllm.audio.WhisperTranscriber;
import org.robomiri.jllm.config.Settings;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Speed of the audio models (run under the heavy lock, scripts/audio-bench.sh):
 * <pre>
 *   AudioBench stt   &lt;whisper dir&gt; &lt;audio file&gt; [cpu|metal] [decoder auto|on|off] [reps]
 *   AudioBench tts   &lt;OuteTTS dir&gt; "text" [voice] [reps] [--key=value ...]
 *   AudioBench codec &lt;WavTokenizer gguf&gt; [frames] [reps]
 * </pre>
 * stt prints the stages of one 30 s window (log-mel, encoder, one decoder step) and the real-time factor of the whole
 * transcription (median of reps); tts the prompt, generation and codec times and the real-time factor.
 */
public final class AudioBench {

    private static double median(List<Double> v) {
        List<Double> s = new ArrayList<>(v);
        s.sort(null);
        return s.get(s.size() / 2);
    }

    public static void main(String[] a) throws Exception {
        switch (a[0]) {
            case "stt" -> stt(a);
            case "tts" -> tts(a);
            case "codec" -> codec(a);
            default -> throw new IllegalArgumentException("stt | tts | codec");
        }
    }

    static void stt(String[] a) throws Exception {
        String backend = a.length > 3 ? a[3] : "metal";
        String decoder = a.length > 4 ? a[4] : "auto";
        int reps = a.length > 5 ? Integer.parseInt(a[5]) : 3;
        int threads = Runtime.getRuntime().availableProcessors();
        long t0 = System.nanoTime();
        WhisperModel m = WhisperModel.load(Path.of(a[1]), backend, threads, 1500, decoder);
        double loadS = (System.nanoTime() - t0) / 1e9;
        AudioDecoder dec = new AudioDecoder("auto", 7200);
        float[] audio = dec.decode16k(Files.readAllBytes(Path.of(a[2])));
        double seconds = audio.length / 16000.0;
        System.out.printf("%s on %s: load %.1f s, %.1f s of audio%n", Path.of(a[1]).getFileName(), m.net.name(), loadS, seconds);
        int frames = m.net.config().windowFrames();
        LogMel mel = new LogMel(m.net.config().nMels());
        int win = Math.min(audio.length, frames * 160);
        float[] w = Arrays.copyOf(audio, win);
        List<Double> tm = new ArrayList<>();
        List<Double> te = new ArrayList<>();
        List<Double> td = new ArrayList<>();
        float[][] feat = null;
        for (int i = 0; i < reps + 1; i++) {
            long s = System.nanoTime();
            feat = mel.compute(w, frames * 160 - win);
            long e = System.nanoTime();
            tm.add((e - s) / 1e6);
            float[][] window = new float[feat.length][frames];
            for (int k = 0; k < feat.length; k++) {
                System.arraycopy(feat[k], 0, window[k], 0, frames);
            }
            long s2 = System.nanoTime();
            try (WhisperNet.Session ses = m.net.encode(window)) {
                te.add((System.nanoTime() - s2) / 1e6);
                float[] logits = new float[m.net.config().vocab()];
                int[] prompt = {m.tokenizer.tokenId("<|startoftranscript|>")};
                ses.decode(prompt, 0, 1, 0, logits);
                long s3 = System.nanoTime();
                int n = 20;
                for (int k = 0; k < n; k++) {
                    ses.decode(prompt, 0, 1, 1 + k, logits);
                }
                td.add((System.nanoTime() - s3) / 1e6 / n);
            }
        }
        System.out.printf("  log-mel %.0f ms, encoder + cross-attention keys/values %.0f ms, decoder step %.1f ms (medians of %d)%n",
                median(tm.subList(1, tm.size())), median(te.subList(1, te.size())), median(td.subList(1, td.size())), reps);
        List<Double> rt = new ArrayList<>();
        Transcript t = null;
        for (int i = 0; i < reps + 1; i++) {
            long s = System.nanoTime();
            t = m.transcriber.transcribe(audio, WhisperTranscriber.Options.defaults());
            rt.add((System.nanoTime() - s) / 1e9);
        }
        double med = median(rt.subList(1, rt.size()));
        int tokens = t.segments().stream().mapToInt(x -> x.tokens().length).sum();
        System.out.printf("  transcription: %.2f s for %.1f s of audio, real-time factor %.3f (%d segments, %d tokens)%n  \"%s\"%n", med,
                seconds, med / seconds, t.segments().size(), tokens, t.text().length() > 100 ? t.text().substring(0, 100) + "..." : t.text());
        m.close();
    }

    static void tts(String[] a) throws Exception {
        List<String> args = new ArrayList<>(List.of("tts"));
        for (int i = 5; i < a.length; i++) {
            args.add(a[i]);
        }
        Settings s = Settings.load(args.toArray(new String[0])).validate();
        int reps = a.length > 4 ? Integer.parseInt(a[4]) : 3;
        long t0 = System.nanoTime();
        OuteTts m = OuteTts.load(s, ModelLoader.hub(s), Path.of(a[1]));
        System.out.printf("%s: load %.1f s%n", m.describe(), (System.nanoTime() - t0) / 1e9);
        List<Double> rt = new ArrayList<>();
        double audio = 0;
        for (int i = 0; i < reps + 1; i++) {
            long st = System.nanoTime();
            float[] pcm = m.synthesize(a[2], a.length > 3 && !a[3].equals("-") ? a[3] : null, SpeechModel.Options.defaults());
            double sec = (System.nanoTime() - st) / 1e9;
            audio = pcm.length / 24000.0;
            if (i > 0) {
                rt.add(sec);
            }
        }
        double med = median(rt);
        System.out.printf("  %.2f s of audio in %.2f s: real-time factor %.2f (median of %d, first run excluded)%n", audio, med,
                med / audio, reps);
        m.close();
    }

    static void codec(String[] a) throws Exception {
        int frames = a.length > 2 ? Integer.parseInt(a[2]) : 750;
        int reps = a.length > 3 ? Integer.parseInt(a[3]) : 3;
        int[] codes = new int[frames];
        java.util.Random r = new java.util.Random(1);
        for (int i = 0; i < frames; i++) {
            codes[i] = r.nextInt(4096);
        }
        try (WavTokenizerDecoder d = new WavTokenizerDecoder(Path.of(a[1]), Runtime.getRuntime().availableProcessors())) {
            List<Double> rt = new ArrayList<>();
            for (int i = 0; i < reps + 1; i++) {
                long s = System.nanoTime();
                d.decode(codes);
                if (i > 0) {
                    rt.add((System.nanoTime() - s) / 1e9);
                }
            }
            double audio = frames * 320 / 24000.0;
            System.out.printf("codec: %d frames = %.1f s of audio in %.2f s, real-time factor %.3f%n", frames, audio, median(rt),
                    median(rt) / audio);
        }
    }
}
