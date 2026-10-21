package org.robomiri.jllm.audio;

import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.function.BooleanSupplier;
import java.util.function.Function;

/**
 * The OpenAI audio endpoints as plain functions over the models, shared by the engine's own server and the web
 * console: {@code /v1/audio/transcriptions}, {@code /v1/audio/translations} and {@code /v1/audio/speech}.
 */
public final class AudioApi {

    private AudioApi() {
    }

    /** An HTTP answer. */
    public record Reply(int status, String contentType, byte[] body) {
        public static Reply json(int status, Object json) {
            return new Reply(status, "application/json", Json.write(json).getBytes(StandardCharsets.UTF_8));
        }

        public static Reply error(int status, String message, String type, String code) {
            return json(status, Json.obj("error", Json.obj("message", message, "type", type, "param", null, "code", code)));
        }

        public static Reply bad(String message) {
            return error(400, message, "invalid_request_error", null);
        }
    }

    /**
     * Transcribes (or translates) an uploaded file.
     *
     * @param field  the multipart form fields by name (null for an absent one)
     * @param file   the file's bytes
     */
    public static Reply transcribe(WhisperModel model, AudioDecoder decoder, byte[] file, Function<String, String> field,
                                   boolean translate, WhisperDefaults defaults, BooleanSupplier cancelled) {
        if (file == null || file.length == 0) {
            return Reply.bad("the 'file' field is required (an audio file)");
        }
        String format = field.apply("response_format");
        if (format == null || format.isBlank()) {
            format = "json";
        }
        if (!Transcript.FORMATS.contains(format)) {
            return Reply.bad("response_format must be one of " + Transcript.FORMATS);
        }
        double temperature = 0;
        String t = field.apply("temperature");
        if (t != null && !t.isBlank()) {
            try {
                temperature = Double.parseDouble(t.trim());
            } catch (NumberFormatException e) {
                return Reply.bad("temperature must be a number between 0 and 1");
            }
            if (temperature < 0 || temperature > 1) {
                return Reply.bad("temperature must be between 0 and 1");
            }
        }
        float[] audio;
        try {
            audio = decoder.decode16k(file);
        } catch (IOException e) {
            return Reply.bad("cannot read the audio: " + e.getMessage());
        }
        String language = blankToNull(field.apply("language"));
        if (translate) {
            language = blankToNull(language);
        }
        WhisperTranscriber.Options o = WhisperTranscriber.Options.defaults()
                .withLanguage(language).withTranslate(translate).withPrompt(blankToNull(field.apply("prompt")))
                .withTemperature(temperature).withCallbacks(cancelled, f -> { }, s -> { });
        String gran = field.apply("timestamp_granularities[]");
        if (gran == null) {
            gran = field.apply("timestamp_granularities");
        }
        if (gran != null && gran.contains("word")) {
            o = o.withWords(true);        // "words" in verbose_json (and in every segment)
        }
        o = defaults.apply(o);
        try {
            Transcript tr = model.transcriber.transcribe(audio, o);
            return new Reply(200, Transcript.contentType(format), tr.render(format).getBytes(StandardCharsets.UTF_8));
        } catch (IllegalArgumentException e) {
            return Reply.bad(e.getMessage());
        } catch (WhisperTranscriber.Cancelled e) {
            return Reply.error(499, "cancelled", "invalid_request_error", "cancelled");
        }
    }

    /** The transcription settings of {@code audio.stt.*}. */
    public record WhisperDefaults(boolean fallback, boolean conditionOnPrevious) {
        public static WhisperDefaults of(AudioSettings s) {
            return new WhisperDefaults(s.sttFallback(), s.sttConditionOnPrevious());
        }

        WhisperTranscriber.Options apply(WhisperTranscriber.Options o) {
            return o.withFallback(fallback, conditionOnPrevious);
        }
    }

    /**
     * {@code POST /v1/audio/speech} with the JSON {@code body}: {@code input}, {@code voice}, {@code response_format}
     * ({@code wav} by default), {@code speed}, plus {@code temperature} and {@code seed} of this server.
     */
    public static Reply speech(SpeechModel model, AudioDecoder tools, Map<String, Object> body, int maxChars,
                               BooleanSupplier cancelled) {
        String input = Json.str(body, "input", null);
        if (input == null || input.isBlank()) {
            return Reply.bad("'input' (the text to speak) is required");
        }
        String format = Json.str(body, "response_format", "wav");
        if (!AudioEncoder.FORMATS.contains(format)) {
            return Reply.bad("response_format must be one of " + AudioEncoder.FORMATS);
        }
        double speed = Json.dbl(body, "speed", 1.0);
        if (!(speed >= 0.25 && speed <= 4.0)) {
            return Reply.bad("speed must be between 0.25 and 4.0");
        }
        String voice = Json.str(body, "voice", null);
        if (voice != null && !voice.isBlank() && !model.voices().contains(voice) && !voice.equals("default")) {
            return Reply.bad("unknown voice '" + voice + "'; available: " + model.voices());
        }
        SpeechModel.Options o = SpeechModel.Options.defaults().withSpeed(speed).withCancel(cancelled);
        if (body.containsKey("temperature")) {
            o = o.withTemperature(Json.dbl(body, "temperature", 0.1));
        }
        if (body.containsKey("seed")) {
            o = o.withSeed(Json.num(body, "seed", -1));
        }
        try {
            float[] pcm = Speech.synthesizeLong(model, input, voice == null || voice.equals("default") ? null : voice, o,
                    maxChars);
            byte[] out = AudioEncoder.encode(format, pcm, model.sampleRate(), tools);
            return new Reply(200, AudioEncoder.contentType(format).replace("rate=24000", "rate=" + model.sampleRate()), out);
        } catch (IllegalArgumentException e) {
            return Reply.bad(e.getMessage());
        } catch (IOException e) {
            return Reply.bad(e.getMessage());
        } catch (WhisperTranscriber.Cancelled e) {
            return Reply.error(499, "cancelled", "invalid_request_error", "cancelled");
        }
    }

    private static String blankToNull(String s) {
        return s == null || s.isBlank() ? null : s.trim();
    }

    /** The loaded models as {@code /v1/models} entries. */
    public static List<Object> modelObjects(AudioModels models) {
        List<Object> out = new java.util.ArrayList<>();
        for (AudioModels.Entry e : models.list()) {
            out.add(Json.obj("id", e.id(), "object", "model", "created", e.loadedAt() / 1000, "owned_by", "jllm-serve",
                    "kind", e.kind().label(), "backend", e.backend()));
        }
        return out;
    }
}
