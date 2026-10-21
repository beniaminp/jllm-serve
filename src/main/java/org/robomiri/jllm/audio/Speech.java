package org.robomiri.jllm.audio;

import java.util.ArrayList;
import java.util.List;

/** Text-to-speech for texts of any length: sentence-sized pieces, joined with short pauses, then the speed change. */
public final class Speech {

    private Speech() {
    }

    /** About this many characters per piece: the models are trained on utterances of a few seconds to a few tens of seconds. */
    static final int PIECE_CHARS = 220;

    /**
     * @param maxChars the longest input accepted
     * @throws IllegalArgumentException when the text is longer than {@code maxChars}
     */
    public static float[] synthesizeLong(SpeechModel model, String text, String voice, SpeechModel.Options o, int maxChars) {
        if (text.length() > maxChars) {
            throw new IllegalArgumentException("input is " + text.length() + " characters; at most " + maxChars
                    + " per request (audio.tts.max-chars)");
        }
        List<String> pieces = split(text, PIECE_CHARS);
        List<float[]> parts = new ArrayList<>();
        SpeechModel.Options plain = o.withSpeed(1.0);
        int pause = model.sampleRate() * 25 / 100;        // 250 ms between sentences
        for (int i = 0; i < pieces.size(); i++) {
            if (o.cancelled().getAsBoolean()) {
                throw new WhisperTranscriber.Cancelled();
            }
            float[] p = model.synthesize(pieces.get(i), voice, plain);
            parts.add(p);
            if (i + 1 < pieces.size()) {
                parts.add(new float[pause]);
            }
        }
        float[] pcm = Wav.concat(parts);
        return TimeStretch.stretch(pcm, model.sampleRate(), o.speed());
    }

    /** Splits at sentence ends, then at commas and spaces, into pieces of at most {@code max} characters. */
    static List<String> split(String text, int max) {
        List<String> out = new ArrayList<>();
        String t = text.strip().replaceAll("\\s+", " ");
        StringBuilder cur = new StringBuilder();
        int i = 0;
        int n = t.length();
        while (i < n) {
            int end = i;
            while (end < n) {
                char c = t.charAt(end);
                end++;
                if ((c == '.' || c == '!' || c == '?' || c == '。' || c == '！' || c == '？' || c == '\n') && (end >= n || t.charAt(end) == ' ')) {
                    break;
                }
            }
            String sentence = t.substring(i, end).strip();
            i = end;
            if (sentence.isEmpty()) {
                continue;
            }
            if (sentence.length() > max) {
                if (!cur.isEmpty()) {
                    out.add(cur.toString());
                    cur.setLength(0);
                }
                splitLong(sentence, max, out);
            } else if (cur.length() + 1 + sentence.length() > max && !cur.isEmpty()) {
                out.add(cur.toString());
                cur.setLength(0);
                cur.append(sentence);
            } else {
                if (!cur.isEmpty()) {
                    cur.append(' ');
                }
                cur.append(sentence);
            }
        }
        if (!cur.isEmpty()) {
            out.add(cur.toString());
        }
        if (out.isEmpty()) {
            out.add(t);
        }
        return out;
    }

    private static void splitLong(String s, int max, List<String> out) {
        String rest = s;
        while (rest.length() > max) {
            int cut = lastIndexBefore(rest, ",;:", max);
            if (cut < max / 3) {
                cut = rest.lastIndexOf(' ', max);
            }
            if (cut <= 0) {
                cut = max;
            }
            out.add(rest.substring(0, cut + (cut < rest.length() && ",;:".indexOf(rest.charAt(cut)) >= 0 ? 1 : 0)).strip());
            rest = rest.substring(Math.min(rest.length(), cut + 1)).strip();
        }
        if (!rest.isEmpty()) {
            out.add(rest);
        }
    }

    private static int lastIndexBefore(String s, String chars, int limit) {
        int best = -1;
        for (int i = 0; i < Math.min(limit, s.length()); i++) {
            if (chars.indexOf(s.charAt(i)) >= 0) {
                best = i;
            }
        }
        return best;
    }
}
