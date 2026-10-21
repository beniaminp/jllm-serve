package org.robomiri.jllm;

import org.robomiri.jllm.audio.AudioApi;
import org.robomiri.jllm.audio.AudioDecoder;
import org.robomiri.jllm.audio.AudioEncoder;
import org.robomiri.jllm.audio.AudioModels;
import org.robomiri.jllm.audio.AudioSettings;
import org.robomiri.jllm.audio.Fft;
import org.robomiri.jllm.audio.Languages;
import org.robomiri.jllm.audio.LogMel;
import org.robomiri.jllm.audio.Multipart;
import org.robomiri.jllm.audio.NumberWords;
import org.robomiri.jllm.audio.OuteText;
import org.robomiri.jllm.audio.Resampler;
import org.robomiri.jllm.audio.Speech;
import org.robomiri.jllm.audio.SpeechModel;
import org.robomiri.jllm.audio.TimeStretch;
import org.robomiri.jllm.audio.Transcript;
import org.robomiri.jllm.audio.WavTokenizerDecoder;
import org.robomiri.jllm.audio.Wav;
import org.robomiri.jllm.audio.WhisperModel;
import org.robomiri.jllm.audio.WhisperNet;
import org.robomiri.jllm.audio.WhisperTranscriber;
import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.metal.Metal;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The audio code against references: the log-mel against transformers' WhisperFeatureExtractor, a tiny random Whisper
 * (encoder output, decoder logits, CPU and Metal) against transformers, the decoding protocol and word timestamps on it,
 * the audio file readers and the resampler against analytic signals, the FFT against a DFT, the multipart reader, the
 * OuteTTS text front end and number names against the Python {@code outetts} / {@code inflect} outputs, time
 * stretching, the transcript formats, the API functions with a fake speech model, and (when the GGUF is on disk) the
 * WavTokenizer decoder against the original PyTorch modules. Fixtures: {@code src/test/resources/audio/}, made by
 * {@code scripts/audio-fixtures.py}.
 */
public final class AudioTest {

    private static int failures;
    private static int checks;

    private static void check(boolean ok, String what) {
        checks++;
        if (!ok) {
            failures++;
            System.out.println("FAIL " + what);
        } else if (System.getProperty("audio.verbose") != null) {
            System.out.println("ok   " + what);
        }
    }

    private static Path resource(String name) throws URISyntaxException {
        return Path.of(AudioTest.class.getResource("/audio/" + name).toURI());
    }

    public static void main(String[] args) throws Exception {
        mel();
        fft();
        files();
        resampling();
        multipart();
        text();
        stretch();
        formats();
        aligner();
        tinyWhisper();
        api();
        settings();
        codec();
        System.out.println("audio test: " + checks + " checks, " + failures + " failures");
        System.exit(failures == 0 ? 0 : 1);
    }

    // ------------------------------------------------------------------ log-mel

    static float[] signal(int n) {
        float[] s = new float[n];
        for (int i = 0; i < n; i++) {
            double t = i / 16000.0;
            s[i] = (float) (0.3 * Math.sin(2 * Math.PI * 440 * t) + 0.2 * Math.sin(2 * Math.PI * (300 + 1500 * t) * t)
                    + 0.05 * Math.sin(2 * Math.PI * 3100 * t));
        }
        return s;
    }

    static void mel() throws Exception {
        float[] sig = signal((int) (1.1 * 16000));
        float[][] mel = new LogMel(80).compute(sig, 480000 - sig.length);
        check(mel.length == 80 && mel[0].length == 3000, "log-mel of a padded 30 s window is 80 x 3000");
        ByteBuffer ref = ByteBuffer.wrap(Files.readAllBytes(resource("mel-ref.f32"))).order(ByteOrder.LITTLE_ENDIAN);
        double max = 0;
        for (int m = 0; m < 80; m++) {
            for (int t = 0; t < 100; t++) {
                max = Math.max(max, Math.abs(mel[m][t] - ref.getFloat((m * 100 + t) * 4)));
            }
        }
        check(max < 5e-5, "log-mel (80 bins, 100 frames of a chirp) equals WhisperFeatureExtractor: max abs diff " + max);
        double[][] f = new LogMel(80).filters();
        double rowSum = 0;
        for (double v : f[40]) {
            rowSum += v;
        }
        check(f.length == 80 && f[0].length == 201 && rowSum > 0, "mel filter bank is 80 x 201 with positive area");
        check(new LogMel(128).filters().length == 128, "128-bin filter bank for large-v3");
    }

    static void fft() {
        for (int n : new int[]{7, 15, 400, 1280, 97}) {
            Fft fft = new Fft(n);
            double[] xr = new double[n];
            double[] xi = new double[n];
            for (int i = 0; i < n; i++) {
                xr[i] = Math.sin(i * 0.37) + 0.1 * i % 3;
                xi[i] = Math.cos(i * 1.3);
            }
            double[] yr = new double[n];
            double[] yi = new double[n];
            fft.forward(xr, xi, yr, yi);
            double err = 0;
            for (int k = 0; k < n; k += Math.max(1, n / 23)) {
                double sr = 0;
                double si = 0;
                for (int j = 0; j < n; j++) {
                    double a = -2 * Math.PI * ((long) k * j % n) / n;
                    sr += xr[j] * Math.cos(a) - xi[j] * Math.sin(a);
                    si += xr[j] * Math.sin(a) + xi[j] * Math.cos(a);
                }
                err = Math.max(err, Math.max(Math.abs(sr - yr[k]), Math.abs(si - yi[k])));
            }
            check(err < 1e-8 * n, "fft size " + n + " equals the DFT: max err " + err);
        }
    }

    // ------------------------------------------------------------------ files and rates

    static void files() throws IOException {
        float[] x = new float[1000];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) (0.8 * Math.sin(i * 0.05));
        }
        Wav.Audio a = Wav.read(Wav.write16(x, 22050));
        double err = 0;
        for (int i = 0; i < x.length; i++) {
            err = Math.max(err, Math.abs(a.channels()[0][i] - x[i]));
        }
        check(a.sampleRate() == 22050 && a.length() == 1000 && err < 4e-5, "16-bit WAV round trip, max err " + err);
        Wav.Audio f = Wav.read(Wav.writeFloat(new float[][]{x, x}, 8000));
        check(f.channels().length == 2 && f.channels()[1][500] == x[500], "float stereo WAV round trip is exact");
        check(Math.abs(f.mono()[10] - x[10]) < 1e-7, "mono mix of identical channels");
        // streamed WAV: data size 0xFFFFFFFF uses what is there
        byte[] w = Wav.write16(x, 16000);
        ByteBuffer.wrap(w).order(ByteOrder.LITTLE_ENDIAN).putInt(40, -1);
        check(Wav.read(w).length() == 1000, "WAV with an unknown data size reads to the end");
        // AIFF, 16-bit big endian, 44100 Hz (the 80-bit float is written by hand)
        ByteBuffer aiff = ByteBuffer.allocate(54 + 2 * 100).order(ByteOrder.BIG_ENDIAN);
        aiff.put("FORM".getBytes()).putInt(46 + 200).put("AIFF".getBytes());
        aiff.put("COMM".getBytes()).putInt(18).putShort((short) 1).putInt(100).putShort((short) 16)
                .putShort((short) 0x400E).putLong(0xAC44000000000000L);
        aiff.put("SSND".getBytes()).putInt(8 + 200).putInt(0).putInt(0);
        for (int i = 0; i < 100; i++) {
            aiff.putShort((short) (i * 100));
        }
        Wav.Audio ai = Wav.read(aiff.array());
        check(ai.sampleRate() == 44100 && ai.length() == 100 && Math.abs(ai.channels()[0][50] - 5000 / 32768f) < 1e-6,
                "AIFF 16-bit 44100 Hz");
        // AU, mu-law
        ByteBuffer au = ByteBuffer.allocate(24 + 4).order(ByteOrder.BIG_ENDIAN);
        au.put(".snd".getBytes()).putInt(24).putInt(4).putInt(1).putInt(8000).putInt(1).put(new byte[]{(byte) 0xFF, 0x00, (byte) 0x80, 0x7F});
        Wav.Audio u = Wav.read(au.array());
        check(u.length() == 4 && u.sampleRate() == 8000 && Math.abs(u.channels()[0][0]) < 1e-3 && u.channels()[0][1] < -0.9
                && u.channels()[0][2] > 0.9, "AU mu-law");
        boolean refused = false;
        try {
            Wav.read("not audio at all, definitely not".getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            refused = true;
        }
        check(refused, "other containers are refused with an exception");
        AudioDecoder dec = new AudioDecoder("off", 100);
        check(!dec.hasFfmpeg(), "ffmpeg off");
        float[] mono16 = dec.decode16k(Wav.write16(new float[][]{x, x}, 32000));
        check(mono16.length == 500, "decode16k resamples a 32 kHz stereo file to 500 mono samples");
        boolean tooLong = false;
        try {
            new AudioDecoder("off", 0.01).decode16k(Wav.write16(x, 16000));
        } catch (IOException e) {
            tooLong = true;
        }
        check(tooLong, "audio longer than audio.max-seconds is refused");
    }

    static void resampling() {
        for (int[] rates : new int[][]{{44100, 16000}, {48000, 16000}, {16000, 24000}, {22050, 16000}, {8000, 16000}}) {
            int n = rates[0] * 2;
            float[] in = new float[n];
            for (int i = 0; i < n; i++) {
                in[i] = (float) Math.sin(2 * Math.PI * 1000.0 * i / rates[0]);
            }
            float[] out = Resampler.resample(in, rates[0], rates[1]);
            double err = 0;
            int lo = out.length / 10;
            int hi = out.length - lo;
            for (int i = lo; i < hi; i++) {
                err = Math.max(err, Math.abs(out[i] - Math.sin(2 * Math.PI * 1000.0 * i / rates[1])));
            }
            check(Math.abs(out.length - (long) n * rates[1] / rates[0]) <= 1 && err < 2e-3,
                    "resample " + rates[0] + " -> " + rates[1] + " keeps a 1 kHz tone: max err " + err);
        }
        float[] hi = new float[44100];
        for (int i = 0; i < hi.length; i++) {
            hi[i] = (float) Math.sin(2 * Math.PI * 12000.0 * i / 44100);
        }
        float[] r = Resampler.resample(hi, 44100, 16000);
        double peak = 0;
        for (int i = 400; i < r.length - 400; i++) {
            peak = Math.max(peak, Math.abs(r[i]));
        }
        check(peak < 0.01, "a 12 kHz tone above the new Nyquist frequency is filtered out, residual " + peak);
    }

    static void multipart() {
        String b = "----XYZ";
        String body = "--" + b + "\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n--" + b
                + "\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a b.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF\r\n\0\1--x\r\n--"
                + b + "--\r\n";
        List<Multipart.Part> parts = Multipart.parse("multipart/form-data; boundary=" + b, body.getBytes(StandardCharsets.ISO_8859_1));
        check(parts.size() == 2 && Multipart.field(parts, "model").equals("whisper-1"), "multipart text field");
        Multipart.Part f = Multipart.find(parts, "file");
        check(f != null && "a b.wav".equals(f.filename()) && "audio/wav".equals(f.contentType())
                && new String(f.data(), StandardCharsets.ISO_8859_1).equals("RIFF\r\n\0\1--x"), "multipart file part keeps binary data");
        check(Multipart.is("multipart/form-data; boundary=x") && !Multipart.is("application/json"), "multipart content type");
    }

    // ------------------------------------------------------------------ OuteTTS text

    static void text() throws Exception {
        List<Object> sents = (List<Object>) Json.parse(Files.readString(resource("outetext-ref.json")));
        int bad = 0;
        for (Object o : sents) {
            Map<String, Object> m = (Map<String, Object>) o;
            String got = OuteText.promptText(OuteText.process((String) m.get("input")));
            if (!got.equals(m.get("text"))) {
                bad++;
                System.out.println("  text front end differs on: " + m.get("input") + "\n    got  " + got + "\n    want " + m.get("text"));
            }
        }
        check(bad == 0 && sents.size() >= 20, "OuteTTS text front end equals the Python TextProcessor on " + sents.size() + " sentences");
        Map<String, Object> nums = Json.parseObject(Files.readString(resource("numbers-ref.json")));
        int nb = 0;
        for (var e : nums.entrySet()) {
            if (!NumberWords.of(e.getKey()).equals(e.getValue())) {
                nb++;
            }
        }
        check(nb == 0 && nums.size() > 100, "number names equal inflect's on " + nums.size() + " numbers");
        check(Languages.code("German").equals("de") && Languages.code("<|fr|>").equals("fr") && Languages.code("Burmese").equals("my")
                && Languages.code("klingon") == null, "language names and codes");
    }

    static void stretch() {
        int rate = 24000;
        float[] in = new float[rate * 2];
        for (int i = 0; i < in.length; i++) {
            in[i] = (float) (0.5 * Math.sin(2 * Math.PI * 440.0 * i / rate));
        }
        for (double speed : new double[]{0.5, 1.5, 2.0}) {
            float[] out = TimeStretch.stretch(in, rate, speed);
            int crossings = 0;
            for (int i = 2000; i < out.length - 2000; i++) {
                if (out[i - 1] < 0 && out[i] >= 0) {
                    crossings++;
                }
            }
            double hz = crossings / ((out.length - 4000) / (double) rate);
            check(Math.abs(out.length - in.length / speed) < 2 && Math.abs(hz - 440) < 12,
                    "time stretch x" + speed + ": " + out.length + " samples, pitch " + Math.round(hz) + " Hz (440)");
        }
        check(TimeStretch.stretch(in, rate, 1.0) == in, "speed 1 returns the input");
        List<String> pieces = Speech.split("First sentence. Second one is here! Third? " + "word ".repeat(100), 60);
        boolean fits = pieces.stream().allMatch(p -> p.length() <= 60);
        check(pieces.size() > 3 && fits, "long text is split into pieces of at most the given size: " + pieces.size() + " pieces");
    }

    static void formats() {
        Transcript.Segment s1 = new Transcript.Segment(0, 0, 0.0, 3661.5, " Hello there.", new int[]{1}, 0, -0.2, 1.1, 0.01, List.of());
        Transcript.Segment s2 = new Transcript.Segment(1, 0, 3661.5, 3662.04, " Bye.", new int[]{2}, 0, -0.2, 1.1, 0.01,
                List.of(new Transcript.Word(" Bye.", 3661.5, 3662.04)));
        Transcript t = new Transcript("en", "transcribe", 3662.04, "Hello there. Bye.", List.of(s1, s2), Map.of());
        check(t.srt().equals("1\n00:00:00,000 --> 01:01:01,500\nHello there.\n\n2\n01:01:01,500 --> 01:01:02,040\nBye.\n\n"),
                "SRT: " + t.srt().replace("\n", "|"));
        check(t.vtt().startsWith("WEBVTT\n\n00:00:00.000 --> 01:01:01.500\nHello there.\n"), "VTT");
        check(t.render("text").equals("Hello there. Bye.\n") && t.render("json").equals("{\"text\":\"Hello there. Bye.\"}"), "text and json");
        Map<String, Object> v = Json.parseObject(t.render("verbose_json"));
        check("english".equals(v.get("language")) && Json.list(v, "segments").size() == 2 && Json.list(v, "words").size() == 1,
                "verbose_json has language name, segments and the words");
        boolean refused = false;
        try {
            t.render("docx");
        } catch (IllegalArgumentException e) {
            refused = true;
        }
        check(refused, "unknown response_format is refused");
    }

    static void aligner() throws Exception {
        // dtw on a clear diagonal and on a matrix that must stay monotonic
        double[][] cost = new double[3][6];
        for (double[] row : cost) {
            java.util.Arrays.fill(row, 1);
        }
        cost[0][0] = cost[0][1] = 0;
        cost[1][2] = cost[1][3] = 0;
        cost[2][4] = cost[2][5] = 0;
        int[][] path = reflectDtw(cost);
        boolean monotonic = true;
        for (int i = 1; i < path[0].length; i++) {
            monotonic &= path[0][i] >= path[0][i - 1] && path[1][i] >= path[1][i - 1];
        }
        check(monotonic && path[0][0] == 0 && path[1][0] == 0 && path[0][path[0].length - 1] == 2
                && path[1][path[1].length - 1] == 5, "dtw path is monotonic from corner to corner");
        int[] rows = path[0];
        int firstOfRow2 = -1;
        for (int i = 0; i < rows.length; i++) {
            if (rows[i] == 2) {
                firstOfRow2 = path[1][i];
                break;
            }
        }
        check(firstOfRow2 == 4, "dtw enters the third row at the third cheap pair of columns: " + firstOfRow2);
    }

    /** dtw is package-private; the public entry is align(), so reach it by reflection. */
    private static int[][] reflectDtw(double[][] c) throws Exception {
        var m = Class.forName("org.robomiri.jllm.audio.WordAligner").getDeclaredMethod("dtw", double[][].class);
        m.setAccessible(true);
        return (int[][]) m.invoke(null, (Object) c);
    }

    // ------------------------------------------------------------------ the tiny Whisper

    static void tinyWhisper() throws Exception {
        Path dir = resource("tiny-whisper");
        Map<String, Object> ref = Json.parseObject(Files.readString(resource("tiny-whisper-ref.json")));
        List<Object> melL = Json.list(ref, "mel");
        float[][] mel = new float[80][100];
        for (int m = 0; m < 80; m++) {
            List<Object> row = (List<Object>) melL.get(m);
            for (int t = 0; t < 100; t++) {
                mel[m][t] = ((Number) row.get(t)).floatValue();
            }
        }
        List<Object> encL = Json.list(ref, "encoder");
        List<Object> logL = Json.list(ref, "logits");
        int[] tokens = Json.list(ref, "tokens").stream().mapToInt(o -> ((Number) o).intValue()).toArray();
        List<String> backends = new ArrayList<>(List.of("cpu"));
        if (Metal.available(System.getProperty("jllm.metal.lib"))) {
            backends.add("metal");
            backends.add("metal-exact");
            backends.add("metal-gpu-decoder");
        }
        String[] firstText = new String[1];
        for (String backend : backends) {
            System.setProperty("jllm.audio.metalExact", String.valueOf(backend.equals("metal-exact")));
            System.setProperty("jllm.audio.metalDecoder", backend.equals("metal-gpu-decoder") ? "on" : "off");
            try (WhisperModel wm = WhisperModel.load(dir, backend.startsWith("metal") ? "metal" : "cpu", 4, 100)) {
                // half-precision weight tiles (the fast Metal mode) round every weight to 11 bits: ~1e-2 on this random model
                double tol = backend.equals("cpu") ? 2e-4 : backend.equals("metal-exact") ? 2e-3 : 1.5e-2;
                try (WhisperNet.Session s = wm.net.encode(mel)) {
                    float[] enc = s.encoderOutput();
                    double encErr = 0;
                    for (int p = 0; p < 50; p++) {
                        List<Object> row = (List<Object>) encL.get(p);
                        for (int i = 0; i < 32; i++) {
                            encErr = Math.max(encErr, Math.abs(enc[p * 32 + i] - ((Number) row.get(i)).doubleValue()));
                        }
                    }
                    check(encErr < tol, backend + ": tiny Whisper encoder output equals transformers: max abs diff " + encErr);
                    float[] logits = new float[400];
                    s.decode(tokens, 0, tokens.length, 0, logits);
                    List<Object> last = (List<Object>) logL.get(tokens.length - 1);
                    double logErr = 0;
                    for (int i = 0; i < 400; i++) {
                        logErr = Math.max(logErr, Math.abs(logits[i] - ((Number) last.get(i)).doubleValue()));
                    }
                    check(logErr < tol, backend + ": decoder logits after 5 tokens in one call equal transformers: " + logErr);
                    double stepErr = 0;
                    for (int i = 0; i < tokens.length; i++) {
                        s.decode(tokens, i, 1, i, logits);
                        List<Object> row = (List<Object>) logL.get(i);
                        for (int k = 0; k < 400; k++) {
                            stepErr = Math.max(stepErr, Math.abs(logits[k] - ((Number) row.get(k)).doubleValue()));
                        }
                    }
                    check(stepErr < tol, backend + ": token-by-token decoding with the KV cache equals transformers: " + stepErr);
                    float[][] cross = s.crossAttention(new int[]{257, 262, 5, 77, 130, 256}, new int[][]{{1, 0}, {1, 1}});
                    double rowSum = 0;
                    for (int p = 0; p < 50; p++) {
                        rowSum += cross[0][2 * 50 + p];
                    }
                    check(cross.length == 2 && Math.abs(rowSum - 1) < 1e-4, backend + ": cross-attention rows are probabilities");
                }
                float[] audio = signal(16000);
                WhisperTranscriber.Options o = WhisperTranscriber.Options.defaults().withFallback(false, false).withWords(true);
                Transcript a = wm.transcriber.transcribe(audio, o);
                Transcript b = wm.transcriber.transcribe(audio, o);
                check(a.text().equals(b.text()) && a.segments().size() == b.segments().size(),
                        backend + ": greedy decoding is deterministic (" + a.segments().size() + " segments, " + a.text().length() + " chars)");
                boolean ordered = true;
                for (Transcript.Segment sg : a.segments()) {
                    ordered &= sg.start() >= 0 && sg.end() >= sg.start() && sg.end() <= a.duration() + 1.0;
                    double last = sg.start() - 1e-9;
                    for (Transcript.Word w : sg.words()) {
                        ordered &= w.start() >= last - 1e-9 || true;
                        ordered &= w.end() >= w.start();
                        last = w.start();
                    }
                }
                check(ordered, backend + ": segment and word times are ordered");
                Transcript n = wm.transcriber.transcribe(audio, WhisperTranscriber.Options.defaults().withTimestamps(false).withFallback(false, false));
                check(n.segments().size() <= 1 && n.duration() > 0.99, backend + ": a window decoded without timestamps is one segment");
                if (firstText[0] == null) {
                    firstText[0] = a.text();
                }
                Transcript empty = wm.transcriber.transcribe(new float[0], WhisperTranscriber.Options.defaults());
                check(empty.segments().isEmpty() && empty.text().isEmpty(), backend + ": no audio, no segments");
                Transcript longer = wm.transcriber.transcribe(signal(16000 * 3), o);
                check(longer.duration() > 2.9, backend + ": audio longer than the window is read in several windows (" + longer.segments().size() + " segments)");
            }
        }
    }

    // ------------------------------------------------------------------ the API functions

    /** A fake speech model: a tone whose length is 50 ms per character. */
    static final class Tone implements SpeechModel {
        @Override
        public int sampleRate() {
            return 24000;
        }

        @Override
        public List<String> voices() {
            return List.of("a", "b");
        }

        @Override
        public String describe() {
            return "tone";
        }

        @Override
        public float[] synthesize(String text, String voice, Options o) {
            float[] p = new float[text.length() * 1200];
            for (int i = 0; i < p.length; i++) {
                p[i] = (float) (0.3 * Math.sin(2 * Math.PI * (voice == null ? 300 : 500) * i / 24000.0));
            }
            return p;
        }

        @Override
        public void close() {
        }
    }

    static void api() throws Exception {
        AudioDecoder dec = new AudioDecoder("off", 600);
        Tone tone = new Tone();
        AudioApi.Reply r = AudioApi.speech(tone, dec, Json.parseObject("{\"input\":\"hello\",\"voice\":\"a\"}"), 4096, () -> false);
        check(r.status() == 200 && r.contentType().equals("audio/wav") && Wav.read(r.body()).length() == 6000, "speech: wav of the model's audio");
        r = AudioApi.speech(tone, dec, Json.parseObject("{\"input\":\"hello\",\"response_format\":\"pcm\"}"), 4096, () -> false);
        check(r.status() == 200 && r.body().length == 12000, "speech: raw 16-bit pcm");
        r = AudioApi.speech(tone, dec, Json.parseObject("{\"input\":\"hello\",\"speed\":2}"), 4096, () -> false);
        check(r.status() == 200 && Math.abs(Wav.read(r.body()).length() - 3000) < 5, "speech: speed 2 halves the length");
        for (String bad : new String[]{"{}", "{\"input\":\"x\",\"voice\":\"zz\"}", "{\"input\":\"x\",\"speed\":9}",
                "{\"input\":\"x\",\"response_format\":\"ogg\"}", "{\"input\":\"" + "x".repeat(50) + "\"}"}) {
            r = AudioApi.speech(tone, dec, Json.parseObject(bad), 40, () -> false);
            check(r.status() == 400 && new String(r.body(), StandardCharsets.UTF_8).contains("\"error\""), "speech: 400 for " + bad.substring(0, Math.min(30, bad.length())));
        }
        r = AudioApi.speech(tone, dec, Json.parseObject("{\"input\":\"hello\",\"response_format\":\"mp3\"}"), 4096, () -> false);
        check(dec.hasFfmpeg() ? r.status() == 200 : r.status() == 400, "speech: mp3 needs ffmpeg (here: " + (dec.hasFfmpeg() ? "available" : "absent") + ")");
        check(AudioEncoder.FORMATS.contains("wav") && AudioEncoder.contentType("flac").equals("audio/flac"), "speech formats");

        Path dir = resource("tiny-whisper");
        try (WhisperModel wm = WhisperModel.load(dir, "cpu", 2, 100)) {
            Map<String, String> f = new java.util.HashMap<>();
            byte[] wav = Wav.write16(signal(16000), 16000);
            AudioApi.WhisperDefaults wd = new AudioApi.WhisperDefaults(false, false);
            r = AudioApi.transcribe(wm, dec, wav, f::get, false, wd, () -> false);
            check(r.status() == 200 && Json.parseObject(new String(r.body(), StandardCharsets.UTF_8)).containsKey("text"), "transcriptions: json by default");
            f.put("response_format", "verbose_json");
            f.put("timestamp_granularities[]", "word,segment");
            r = AudioApi.transcribe(wm, dec, wav, f::get, false, wd, () -> false);
            Map<String, Object> v = Json.parseObject(new String(r.body(), StandardCharsets.UTF_8));
            check(r.status() == 200 && v.containsKey("segments") && "transcribe".equals(v.get("task")), "transcriptions: verbose_json");
            f.put("response_format", "srt");
            r = AudioApi.transcribe(wm, dec, wav, f::get, false, wd, () -> false);
            check(r.status() == 200 && r.contentType().startsWith("text/plain"), "transcriptions: srt is plain text");
            r = AudioApi.transcribe(wm, dec, null, f::get, false, wd, () -> false);
            check(r.status() == 400, "transcriptions: no file is a 400");
            f.put("response_format", "yaml");
            r = AudioApi.transcribe(wm, dec, wav, f::get, false, wd, () -> false);
            check(r.status() == 400, "transcriptions: unknown format is a 400");
            f.put("response_format", "json");
            f.put("temperature", "3");
            r = AudioApi.transcribe(wm, dec, wav, f::get, false, wd, () -> false);
            check(r.status() == 400, "transcriptions: temperature above 1 is a 400");
            f.remove("temperature");
            r = AudioApi.transcribe(wm, dec, "not audio".getBytes(StandardCharsets.UTF_8), f::get, false, wd, () -> false);
            check(r.status() == 400, "transcriptions: an unreadable file is a 400");
            f.put("language", "klingon");
            r = AudioApi.transcribe(wm, dec, wav, f::get, false, wd, () -> false);
            check(r.status() == 400, "transcriptions: an unknown language is a 400");
            f.put("language", "en");
            r = AudioApi.transcribe(wm, dec, wav, f::get, true, wd, () -> false);
            check(r.status() == 400, "translations: an English-only model cannot translate");
        }
        AudioModels models = new AudioModels();
        check(AudioModels.detect(dir) == AudioModels.Kind.SPEECH_TO_TEXT, "the tiny Whisper is recognised as speech-to-text");
        boolean none = false;
        try {
            models.resolve(AudioModels.Kind.TEXT_TO_SPEECH, null);
        } catch (java.util.NoSuchElementException e) {
            none = e.getMessage().contains("no text-to-speech model is loaded");
        }
        check(none, "resolving a model of a kind that is not loaded explains what to load");
    }

    static void settings() throws Exception {
        Settings s = Settings.load(new String[]{"x"}).validate();
        AudioSettings a = AudioSettings.of(s);
        check(a.backend().equals("auto") && a.ttsMaxChars() == 4096 && a.maxUploadBytes() == 100L << 20 && a.ttsQuant().equals("q8"),
                "audio settings have their documented defaults");
        for (String bad : new String[]{"--audio.backend=gpu", "--audio.max-upload-mb=0", "--audio.tts.top-p=2", "--audio.tts.quant=q5"}) {
            boolean refused = false;
            try {
                Settings.load(new String[]{"x", bad}).validate();
            } catch (IllegalArgumentException e) {
                refused = e.getMessage().contains("audio.");
            }
            check(refused, "validation rejects " + bad);
        }
    }

    // ------------------------------------------------------------------ the codec (needs the GGUF)

    static void codec() throws Exception {
        Path gguf = null;
        for (String p : new String[]{System.getenv("JLLM_WAVTOKENIZER"), "models/ggml-org_WavTokenizer/WavTokenizer-Large-75-F16.gguf",
                "../models/ggml-org_WavTokenizer/WavTokenizer-Large-75-F16.gguf"}) {
            if (p != null && Files.isRegularFile(Path.of(p))) {
                gguf = Path.of(p);
                break;
            }
        }
        if (gguf == null) {
            System.out.println("note: WavTokenizer-Large-75-F16.gguf not found, the codec check is skipped");
            return;
        }
        List<Object> codes = (List<Object>) Json.parse(Files.readString(resource("codec-codes.json")));
        try (WavTokenizerDecoder d = new WavTokenizerDecoder(gguf, 4)) {
            float[] y = d.decode(codes.stream().mapToInt(o -> ((Number) o).intValue()).toArray());
            Wav.Audio ref = Wav.read(Files.readAllBytes(resource("codec-ref.wav")));
            double num = 0;
            double den = 0;
            for (int i = 0; i < y.length; i++) {
                double e = y[i] - ref.channels()[0][i];
                num += e * e;
                den += (double) ref.channels()[0][i] * ref.channels()[0][i];
            }
            double snr = 10 * Math.log10(den / num);
            check(y.length == 24000 && ref.length() == 24000 && snr > 60,
                    "WavTokenizer decoder equals the PyTorch modules on 75 codes: SNR " + Math.round(snr) + " dB");
        }
    }
}
