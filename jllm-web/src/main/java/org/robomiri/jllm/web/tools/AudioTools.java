package org.robomiri.jllm.web.tools;

import org.robomiri.jllm.audio.AudioModels;
import org.robomiri.jllm.audio.Transcript;
import org.robomiri.jllm.audio.WhisperTranscriber;
import org.robomiri.jllm.audio.Wav;
import org.robomiri.jllm.audio.SpeechModel;
import org.robomiri.jllm.audio.Speech;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.web.audio.AudioService;
import org.robomiri.jllm.web.audio.AudioStore;

import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Tools that give a chat model ears and a voice: {@code transcribe_audio} (speech-to-text on a clip the user recorded or
 * attached) and {@code speak} (text-to-speech; the chat shows the result with a player). Clips are referred to by the id
 * the console's audio store gave them ({@code [audio:ID]} in a message); they need a Whisper / OuteTTS model loaded on the
 * Models page.
 */
public final class AudioTools {

    private AudioTools() {
    }

    public static List<Tool> all() {
        return List.of(new TranscribeAudio(), new Speak());
    }

    private static AudioService service() {
        AudioService s = AudioService.current();
        if (s == null) {
            throw new IllegalStateException("audio is not available");
        }
        return s;
    }

    /** Speech to text. */
    public static final class TranscribeAudio implements Tool {
        @Override
        public String name() {
            return "transcribe_audio";
        }

        @Override
        public String description() {
            return "Transcribe a recording to text with the loaded speech-to-text (Whisper) model. Give the audio_id of a clip"
                    + " the user recorded or attached (messages mention it as [audio:ID]). Optionally translate to English"
                    + " or ask for timestamps.";
        }

        @Override
        public Map<String, Object> parameters() {
            return Json.obj("type", "object", "properties", Json.obj(
                    "audio_id", Json.obj("type", "string", "description", "id of the clip, from [audio:ID]"),
                    "language", Json.obj("type", "string", "description", "spoken language code such as en or de (default: detect)"),
                    "translate", Json.obj("type", "boolean", "description", "translate the speech into English"),
                    "timestamps", Json.obj("type", "boolean", "description", "prefix every segment with its start time")),
                    "required", List.of("audio_id"));
        }

        @Override
        public String source() {
            return "builtin";
        }

        @Override
        public String call(Map<String, Object> args) throws Exception {
            AudioService svc = service();
            String id = Json.str(args, "audio_id", "").strip();
            if (id.startsWith("[audio:") && id.endsWith("]")) {
                id = id.substring(7, id.length() - 1);
            }
            if (id.isEmpty()) {
                throw new IllegalArgumentException("audio_id is required");
            }
            AudioStore.Clip clip = svc.store().get(id);
            AudioModels.Entry m = svc.models().resolve(AudioModels.Kind.SPEECH_TO_TEXT, null);
            float[] pcm = svc.decoder().decode16k(clip.wav());
            WhisperTranscriber.Options o = WhisperTranscriber.Options.defaults();
            String lang = Json.str(args, "language", "").strip();
            if (!lang.isEmpty()) {
                o = o.withLanguage(lang);
            }
            o = o.withTranslate(Json.bool(args, "translate", false));
            Transcript t = m.stt().transcriber.transcribe(pcm, o);
            if (!Json.bool(args, "timestamps", false)) {
                return t.text().isEmpty() ? "(no speech detected)" : t.text();
            }
            StringBuilder sb = new StringBuilder();
            for (Transcript.Segment s : t.segments()) {
                sb.append(String.format(Locale.ROOT, "[%02d:%05.2f] %s%n", (int) (s.start() / 60), s.start() % 60, s.text().strip()));
            }
            return sb.isEmpty() ? "(no speech detected)" : sb.toString().strip();
        }
    }

    /** Text to speech. */
    public static final class Speak implements Tool {
        @Override
        public String name() {
            return "speak";
        }

        @Override
        public String description() {
            return "Say text aloud with the loaded text-to-speech model. Returns an audio clip id; the chat shows a player"
                    + " for it, so just mention that you spoke it. Use for short passages (a few sentences).";
        }

        @Override
        public Map<String, Object> parameters() {
            return Json.obj("type", "object", "properties", Json.obj(
                    "text", Json.obj("type", "string", "description", "the text to speak"),
                    "voice", Json.obj("type", "string", "description", "voice name (default: the model's default voice)"),
                    "speed", Json.obj("type", "number", "description", "speaking speed, 0.25 to 4 (default 1)")),
                    "required", List.of("text"));
        }

        @Override
        public String source() {
            return "builtin";
        }

        @Override
        public String call(Map<String, Object> args) throws Exception {
            AudioService svc = service();
            String text = Json.str(args, "text", "").strip();
            if (text.isEmpty()) {
                throw new IllegalArgumentException("text is required");
            }
            AudioModels.Entry m = svc.models().resolve(AudioModels.Kind.TEXT_TO_SPEECH, null);
            String voice = Json.str(args, "voice", "").strip();
            double speed = Json.dbl(args, "speed", 1.0);
            if (!(speed >= 0.25 && speed <= 4.0)) {
                throw new IllegalArgumentException("speed must be between 0.25 and 4");
            }
            SpeechModel tts = m.tts();
            if (!voice.isEmpty() && !tts.voices().contains(voice)) {
                throw new IllegalArgumentException("unknown voice '" + voice + "'; available: " + tts.voices());
            }
            float[] pcm = Speech.synthesizeLong(tts, text, voice.isEmpty() ? null : voice,
                    SpeechModel.Options.defaults().withSpeed(speed), svc.audioSettings().ttsMaxChars());
            AudioStore.Clip c = svc.store().put(Wav.write16(pcm, tts.sampleRate()), tts.sampleRate(),
                    (double) pcm.length / tts.sampleRate(), "speech");
            return String.format(Locale.ROOT, "Spoke %d characters (%.1f s of audio). [audio:%s]", text.length(), c.seconds(),
                    c.id());
        }
    }
}
