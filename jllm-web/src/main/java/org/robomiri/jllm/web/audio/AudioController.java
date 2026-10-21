package org.robomiri.jllm.web.audio;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.robomiri.jllm.audio.AudioApi;
import org.robomiri.jllm.audio.AudioModels;
import org.robomiri.jllm.audio.Languages;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.web.chat.ChatSupport;
import org.robomiri.jllm.web.chat.ClientGone;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.function.BooleanSupplier;

/**
 * The audio API: OpenAI's {@code POST /v1/audio/transcriptions}, {@code /v1/audio/translations} and
 * {@code /v1/audio/speech}, and the console's {@code /api/audio/*} (loaded audio models, voices and the clip store the
 * chat uses). The work is {@link AudioApi}; this class only adapts the requests.
 */
@RestController
public class AudioController {

    private final AudioService audio;

    public AudioController(AudioService audio) {
        this.audio = audio;
    }

    // ------------------------------------------------------------------ OpenAI

    @PostMapping(value = "/v1/audio/transcriptions", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public void transcriptions(@RequestParam(value = "file", required = false) MultipartFile file,
                               @RequestParam Map<String, String> fields, HttpServletRequest req, HttpServletResponse res)
            throws IOException {
        transcribe(file, fields, false, req, res);
    }

    @PostMapping(value = "/v1/audio/translations", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public void translations(@RequestParam(value = "file", required = false) MultipartFile file,
                             @RequestParam Map<String, String> fields, HttpServletRequest req, HttpServletResponse res)
            throws IOException {
        transcribe(file, fields, true, req, res);
    }

    private void transcribe(MultipartFile file, Map<String, String> fields, boolean translate, HttpServletRequest req,
                            HttpServletResponse res) throws IOException {
        AudioModels.Entry m;
        try {
            m = audio.models().resolve(AudioModels.Kind.SPEECH_TO_TEXT, fields.get("model"));
        } catch (NoSuchElementException e) {
            ChatSupport.send(res, 404, ChatSupport.error(e.getMessage(), "invalid_request_error", "model_not_found"));
            return;
        }
        long limit = audio.audioSettings().maxUploadBytes();
        if (file != null && file.getSize() > limit) {
            ChatSupport.send(res, 413, ChatSupport.error("the audio file is larger than audio.max-upload-mb ("
                    + (limit >> 20) + " MB)", "invalid_request_error", "file_too_large"));
            return;
        }
        byte[] bytes = file == null ? null : file.getBytes();
        BooleanSupplier gone = ClientGone.of(req);
        String[] gran = req.getParameterValues("timestamp_granularities[]");   // repeated: segment and word
        if (gran != null) {
            fields = new java.util.HashMap<>(fields);
            fields.put("timestamp_granularities[]", String.join(",", gran));
        }
        Map<String, String> f = fields;
        if ("true".equalsIgnoreCase(f.get("stream"))) {        // server-sent events, one per segment (see AudioApi.transcribeStream)
            boolean[] started = {false};
            AudioApi.Reply bad = AudioApi.transcribeStream(m.stt(), audio.decoder(), bytes, f::get, translate,
                    audio.whisperDefaults(), gone, event -> {
                        try {
                            if (!started[0]) {
                                res.setStatus(200);
                                res.setContentType("text/event-stream");
                                res.setCharacterEncoding("UTF-8");
                                res.setHeader("Cache-Control", "no-cache");
                                started[0] = true;
                            }
                            res.getWriter().write("data: " + Json.write(event) + "\n\n");
                            res.flushBuffer();
                        } catch (IOException e) {
                            throw new java.io.UncheckedIOException(e);
                        }
                    });
            if (bad != null) {
                reply(res, bad);
            } else {
                res.getWriter().write("data: [DONE]\n\n");
                res.flushBuffer();
            }
            return;
        }
        AudioApi.Reply r = AudioApi.transcribe(m.stt(), audio.decoder(), bytes, f::get, translate,
                audio.whisperDefaults(), gone);
        reply(res, r);
    }

    @PostMapping(value = "/v1/audio/speech", consumes = MediaType.APPLICATION_JSON_VALUE)
    public void speech(@RequestBody(required = false) String text, HttpServletRequest req, HttpServletResponse res)
            throws IOException {
        Map<String, Object> body;
        try {
            body = Json.parseObject(text == null ? "" : text);
        } catch (RuntimeException e) {
            ChatSupport.send(res, 400, ChatSupport.error("invalid JSON: " + e.getMessage(), "invalid_request_error", null));
            return;
        }
        AudioModels.Entry m;
        try {
            m = audio.models().resolve(AudioModels.Kind.TEXT_TO_SPEECH, Json.str(body, "model", null));
        } catch (NoSuchElementException e) {
            ChatSupport.send(res, 404, ChatSupport.error(e.getMessage(), "invalid_request_error", "model_not_found"));
            return;
        }
        reply(res, AudioApi.speech(m.tts(), audio.decoder(), body, audio.audioSettings().ttsMaxChars(), ClientGone.of(req)));
    }

    private static void reply(HttpServletResponse res, AudioApi.Reply r) throws IOException {
        res.setStatus(r.status());
        res.setContentType(r.contentType());
        res.setContentLength(r.body().length);
        res.getOutputStream().write(r.body());
    }

    // ------------------------------------------------------------------ the console

    /** The loaded audio models with their kind. */
    @GetMapping(value = "/api/audio/models", produces = "application/json")
    public String models() {
        List<Object> out = new ArrayList<>();
        for (AudioModels.Entry e : audio.loaded()) {
            Map<String, Object> m = Json.obj("id", e.id(), "kind", e.kind().label(), "source", e.source(),
                    "backend", e.backend(), "description", e.description(), "loadedAt", e.loadedAt(),
                    "loadSeconds", Math.round(e.loadSeconds() * 10) / 10.0);
            if (e.tts() != null) {
                m.put("voices", new ArrayList<Object>(e.tts().voices()));
                m.put("sampleRate", (long) e.tts().sampleRate());
            } else {
                m.put("multilingual", e.stt().transcriber.multilingual());
                m.put("languages", new ArrayList<Object>(e.stt().transcriber.multilingual() ? Languages.NAMES.keySet()
                        : List.of("en")));
            }
            out.add(m);
        }
        return Json.write(out);
    }

    /** The voices of a speech model (the only one loaded when {@code model} is omitted). */
    @GetMapping(value = "/api/audio/voices", produces = "application/json")
    public String voices(@RequestParam(required = false) String model, HttpServletResponse res) throws IOException {
        try {
            AudioModels.Entry m = audio.models().resolve(AudioModels.Kind.TEXT_TO_SPEECH, model);
            return Json.write(Json.obj("model", m.id(), "voices", new ArrayList<Object>(m.tts().voices()),
                    "sampleRate", (long) m.tts().sampleRate()));
        } catch (NoSuchElementException e) {
            res.setStatus(404);
            return Json.write(Json.obj("error", e.getMessage()));
        }
    }

    /** Stores a WAV file (a recording, an edit) and returns its id, so a chat message and its tools can refer to it. */
    @PostMapping(value = "/api/audio/files", produces = "application/json")
    public String upload(@RequestBody byte[] wav, @RequestParam(required = false, defaultValue = "") String label,
                         HttpServletResponse res) {
        try {
            AudioStore.Clip c = audio.storeWav(wav, label);
            return Json.write(clip(c));
        } catch (IOException e) {
            res.setStatus(400);
            return Json.write(Json.obj("error", "not a WAV, AIFF or AU file: " + e.getMessage()));
        }
    }

    @GetMapping("/api/audio/files/{id}")
    public void download(@PathVariable String id, HttpServletResponse res) throws IOException {
        try {
            AudioStore.Clip c = audio.store().get(id);
            res.setContentType("audio/wav");
            res.setContentLength(c.wav().length);
            res.setHeader("Content-Disposition", "inline; filename=\"" + id + ".wav\"");
            res.getOutputStream().write(c.wav());
        } catch (NoSuchElementException e) {
            ChatSupport.send(res, 404, Json.write(Json.obj("error", e.getMessage())));
        }
    }

    @DeleteMapping(value = "/api/audio/files/{id}", produces = "application/json")
    public String delete(@PathVariable String id) {
        return Json.write(Json.obj("deleted", audio.store().remove(id)));
    }

    static Map<String, Object> clip(AudioStore.Clip c) {
        return Json.obj("id", c.id(), "seconds", Math.round(c.seconds() * 1000) / 1000.0, "sampleRate", (long) c.sampleRate(),
                "bytes", (long) c.wav().length, "label", c.label(), "url", "/api/audio/files/" + c.id());
    }
}
