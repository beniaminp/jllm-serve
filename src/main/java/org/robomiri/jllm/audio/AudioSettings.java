package org.robomiri.jllm.audio;

import org.robomiri.jllm.config.Settings;

/**
 * The {@code audio.*} settings, validated once ({@link Settings#validate} calls {@link #of}), so a typo fails at start-up
 * instead of at the first request.
 */
public record AudioSettings(String backend, String metalDecoder, int threads, long residentMb, String ffmpeg, double maxSeconds, long maxUploadBytes,
                            int ttsMaxChars, String ttsQuant, int ttsCtx, String ttsModel, String sttModel, String codecRepo, String codecFile,
                            double ttsTemperature, int ttsTopK, double ttsTopP, double ttsRepetitionPenalty,
                            boolean sttFallback, boolean sttConditionOnPrevious) {

    public static AudioSettings of(Settings s) {
        String backend = s.get("audio.backend").trim().toLowerCase(java.util.Locale.ROOT);
        require(backend.equals("cpu") || backend.equals("metal") || backend.equals("auto"), s, "audio.backend",
                "must be cpu, metal or auto");
        String metalDecoder = s.get("audio.metal.decoder").trim().toLowerCase(java.util.Locale.ROOT);
        require(metalDecoder.equals("auto") || metalDecoder.equals("on") || metalDecoder.equals("off"), s, "audio.metal.decoder",
                "must be auto, on or off");
        int threads = s.getInt("audio.threads");
        require(threads >= 0, s, "audio.threads", "must be 0 (= backend.threads) or positive");
        long resident = s.getLong("audio.resident-mb");
        require(resident >= 0, s, "audio.resident-mb", "must be 0 or positive");
        double maxSeconds = s.getFloat("audio.max-seconds");
        require(maxSeconds > 0, s, "audio.max-seconds", "must be positive");
        long upload = s.getLong("audio.max-upload-mb");
        require(upload >= 1 && upload <= 4096, s, "audio.max-upload-mb", "must be 1..4096");
        int chars = s.getInt("audio.tts.max-chars");
        require(chars >= 16 && chars <= 100_000, s, "audio.tts.max-chars", "must be 16..100000");
        String quant = s.get("audio.tts.quant").trim();
        try {
            org.robomiri.jllm.model.QType.parse(quant);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("audio.tts.quant: " + e.getMessage());
        }
        int ttsCtx = s.getInt("audio.tts.ctx");
        require(ttsCtx >= 1024, s, "audio.tts.ctx", "must be at least 1024");
        double temp = s.getFloat("audio.tts.temperature");
        require(temp >= 0 && temp <= 2, s, "audio.tts.temperature", "must be 0..2");
        int topK = s.getInt("audio.tts.top-k");
        require(topK >= 0, s, "audio.tts.top-k", "must be 0 (off) or positive");
        double topP = s.getFloat("audio.tts.top-p");
        require(topP > 0 && topP <= 1, s, "audio.tts.top-p", "must be in (0, 1]");
        double rep = s.getFloat("audio.tts.repetition-penalty");
        require(rep > 0 && rep <= 5, s, "audio.tts.repetition-penalty", "must be in (0, 5]");
        String ffmpeg = s.get("audio.ffmpeg").trim();
        require(!ffmpeg.isEmpty(), s, "audio.ffmpeg", "must be auto, off or a path");
        String codecFile = s.get("audio.codec-file").trim();
        require(codecFile.isEmpty() || codecFile.toLowerCase(java.util.Locale.ROOT).endsWith(".gguf"), s, "audio.codec-file",
                "must be the name of a .gguf file");
        return new AudioSettings(backend, metalDecoder, threads, resident, ffmpeg, maxSeconds, upload << 20, chars, quant, ttsCtx,
                s.get("audio.tts-model").trim(), s.get("audio.stt-model").trim(), s.get("audio.codec-repo").trim(), codecFile,
                temp, topK, topP, rep, s.getBool("audio.stt.fallback"), s.getBool("audio.stt.condition-on-previous"));
    }

    private static void require(boolean ok, Settings s, String key, String what) {
        if (!ok) {
            throw new IllegalArgumentException(key + " " + what + ", got '" + s.get(key) + "'");
        }
    }

    /** CPU threads for the audio models: {@code audio.threads}, else {@code backend.threads}, else all processors. */
    public static int cpuThreads(Settings s) {
        int t = s.getInt("audio.threads");
        if (t == 0) {
            t = s.getInt("backend.threads");
        }
        return t == 0 ? Runtime.getRuntime().availableProcessors() : t;
    }
}
