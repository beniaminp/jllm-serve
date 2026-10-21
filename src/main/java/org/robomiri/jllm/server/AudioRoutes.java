package org.robomiri.jllm.server;

import org.robomiri.jllm.audio.AudioApi;
import org.robomiri.jllm.audio.AudioDecoder;
import org.robomiri.jllm.audio.AudioModels;
import org.robomiri.jllm.audio.AudioSettings;
import org.robomiri.jllm.audio.Multipart;
import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

/**
 * The audio endpoints of {@code ./jllm serve}: {@code POST /v1/audio/transcriptions}, {@code /v1/audio/translations} and
 * {@code /v1/audio/speech} (OpenAI's), served by the Whisper / OuteTTS models loaded with {@code audio.stt-model} and
 * {@code audio.tts-model}. The console (jllm-web) has the same endpoints on Spring; both call {@link AudioApi}.
 */
public final class AudioRoutes {

    private final AudioModels models;
    private final AudioDecoder decoder;
    private final AudioSettings settings;

    public AudioRoutes(AudioModels models, AudioDecoder decoder, AudioSettings settings) {
        this.models = models;
        this.decoder = decoder;
        this.settings = settings;
    }

    /** The loaded audio models as {@code /v1/models} entries. */
    public List<Object> modelObjects() {
        return AudioApi.modelObjects(models);
    }

    /** @return true when the path was an audio endpoint (and the response has been sent) */
    boolean handle(String path, HttpServer.Request req, HttpServer.Response res) throws IOException {
        switch (path) {
            case "/v1/audio/transcriptions" -> transcribe(req, res, false);
            case "/v1/audio/translations" -> transcribe(req, res, true);
            case "/v1/audio/speech" -> speech(req, res);
            case "/api/audio/models" -> res.send(200, "application/json", Json.write(list()));
            default -> {
                return false;
            }
        }
        return true;
    }

    private List<Object> list() {
        List<Object> out = new ArrayList<>();
        for (AudioModels.Entry e : models.list()) {
            Map<String, Object> m = Json.obj("id", e.id(), "kind", e.kind().label(), "source", e.source(), "backend", e.backend(),
                    "description", e.description());
            if (e.tts() != null) {
                m.put("voices", new ArrayList<Object>(e.tts().voices()));
                m.put("sampleRate", (long) e.tts().sampleRate());
            }
            out.add(m);
        }
        return out;
    }

    private void transcribe(HttpServer.Request req, HttpServer.Response res, boolean translate) throws IOException {
        if (!req.method.equals("POST")) {
            send(res, AudioApi.Reply.error(405, "use POST", "invalid_request_error", null));
            return;
        }
        String ct = req.headers.get("content-type");
        if (!Multipart.is(ct)) {
            send(res, AudioApi.Reply.bad("Content-Type must be multipart/form-data"));
            return;
        }
        List<Multipart.Part> parts;
        try {
            parts = Multipart.parse(ct, req.body);
        } catch (IllegalArgumentException e) {
            send(res, AudioApi.Reply.bad("invalid multipart body: " + e.getMessage()));
            return;
        }
        Multipart.Part file = Multipart.find(parts, "file");
        AudioModels.Entry m;
        try {
            m = models.resolve(AudioModels.Kind.SPEECH_TO_TEXT, Multipart.field(parts, "model"));
        } catch (NoSuchElementException e) {
            send(res, AudioApi.Reply.error(404, e.getMessage(), "invalid_request_error", "model_not_found"));
            return;
        }
        StringBuilder gran = new StringBuilder();
        for (Multipart.Part p : parts) {
            if (p.name().equals("timestamp_granularities[]") || p.name().equals("timestamp_granularities")) {
                gran.append(gran.isEmpty() ? "" : ",").append(p.text());
            }
        }
        java.util.function.Function<String, String> field = name -> {
            if (name.startsWith("timestamp_granularities")) {
                return gran.isEmpty() ? null : gran.toString();
            }
            return Multipart.field(parts, name);
        };
        byte[] data = file == null ? null : file.data();
        if ("true".equalsIgnoreCase(Multipart.field(parts, "stream"))) { // server-sent events, one per segment
            boolean[] started = {false};
            AudioApi.Reply bad = AudioApi.transcribeStream(m.stt(), decoder, data, field, translate,
                    AudioApi.WhisperDefaults.of(settings), req::clientGone, event -> {
                        try {
                            if (!started[0]) {
                                res.startStream(200, "text/event-stream");
                                started[0] = true;
                            }
                            res.write("data: " + Json.write(event) + "\n\n");
                        } catch (IOException e) {
                            throw new java.io.UncheckedIOException(e);
                        }
                    });
            if (bad != null) {
                send(res, bad);
            } else {
                res.write("data: [DONE]\n\n");
                res.end();
            }
            return;
        }
        send(res, AudioApi.transcribe(m.stt(), decoder, data, field, translate, AudioApi.WhisperDefaults.of(settings),
                req::clientGone));
    }

    private void speech(HttpServer.Request req, HttpServer.Response res) throws IOException {
        if (!req.method.equals("POST")) {
            send(res, AudioApi.Reply.error(405, "use POST", "invalid_request_error", null));
            return;
        }
        Map<String, Object> body;
        try {
            body = Json.parseObject(req.bodyText());
        } catch (RuntimeException e) {
            send(res, AudioApi.Reply.bad("invalid JSON: " + e.getMessage()));
            return;
        }
        AudioModels.Entry m;
        try {
            m = models.resolve(AudioModels.Kind.TEXT_TO_SPEECH, Json.str(body, "model", null));
        } catch (NoSuchElementException e) {
            send(res, AudioApi.Reply.error(404, e.getMessage(), "invalid_request_error", "model_not_found"));
            return;
        }
        send(res, AudioApi.speech(m.tts(), decoder, body, settings.ttsMaxChars(), req::clientGone));
    }

    private static void send(HttpServer.Response res, AudioApi.Reply r) throws IOException {
        res.send(r.status(), r.contentType(), r.body());
    }
}
