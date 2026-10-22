package org.robomiri.jllm.audio;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/** The languages Whisper knows: ISO-639-1 style codes (the {@code <|en|>} tokens) and their English names. */
public final class Languages {

    private Languages() {
    }

    /** Code to lower-case English name, in the order of the Whisper vocabulary (so token id = first + index). */
    public static final Map<String, String> NAMES = new LinkedHashMap<>();

    static {
        String[] t = {
            "en", "english", "zh", "chinese", "de", "german", "es", "spanish", "ru", "russian", "ko", "korean",
            "fr", "french", "ja", "japanese", "pt", "portuguese", "tr", "turkish", "pl", "polish", "ca", "catalan",
            "nl", "dutch", "ar", "arabic", "sv", "swedish", "it", "italian", "id", "indonesian", "hi", "hindi",
            "fi", "finnish", "vi", "vietnamese", "he", "hebrew", "uk", "ukrainian", "el", "greek", "ms", "malay",
            "cs", "czech", "ro", "romanian", "da", "danish", "hu", "hungarian", "ta", "tamil", "no", "norwegian",
            "th", "thai", "ur", "urdu", "hr", "croatian", "bg", "bulgarian", "lt", "lithuanian", "la", "latin",
            "mi", "maori", "ml", "malayalam", "cy", "welsh", "sk", "slovak", "te", "telugu", "fa", "persian",
            "lv", "latvian", "bn", "bengali", "sr", "serbian", "az", "azerbaijani", "sl", "slovenian", "kn", "kannada",
            "et", "estonian", "mk", "macedonian", "br", "breton", "eu", "basque", "is", "icelandic", "hy", "armenian",
            "ne", "nepali", "mn", "mongolian", "bs", "bosnian", "kk", "kazakh", "sq", "albanian", "sw", "swahili",
            "gl", "galician", "mr", "marathi", "pa", "punjabi", "si", "sinhala", "km", "khmer", "sn", "shona",
            "yo", "yoruba", "so", "somali", "af", "afrikaans", "oc", "occitan", "ka", "georgian", "be", "belarusian",
            "tg", "tajik", "sd", "sindhi", "gu", "gujarati", "am", "amharic", "yi", "yiddish", "lo", "lao",
            "uz", "uzbek", "fo", "faroese", "ht", "haitian creole", "ps", "pashto", "tk", "turkmen", "nn", "nynorsk",
            "mt", "maltese", "sa", "sanskrit", "lb", "luxembourgish", "my", "myanmar", "bo", "tibetan", "tl", "tagalog",
            "mg", "malagasy", "as", "assamese", "tt", "tatar", "haw", "hawaiian", "ln", "lingala", "ha", "hausa",
            "ba", "bashkir", "jw", "javanese", "su", "sundanese", "yue", "cantonese"};
        for (int i = 0; i < t.length; i += 2) {
            NAMES.put(t[i], t[i + 1]);
        }
    }

    /**
     * The code for a language given as a code ("en", "EN", "<|en|>") or English name ("English", "german") or common
     * alias ("burmese", "castilian", ...), or null when unknown.
     */
    public static String code(String s) {
        if (s == null) {
            return null;
        }
        String k = s.trim().toLowerCase(Locale.ROOT);
        if (k.startsWith("<|") && k.endsWith("|>")) {
            k = k.substring(2, k.length() - 2);
        }
        if (NAMES.containsKey(k)) {
            return k;
        }
        for (Map.Entry<String, String> e : NAMES.entrySet()) {
            if (e.getValue().equals(k)) {
                return e.getKey();
            }
        }
        return switch (k) {
            case "burmese" -> "my";
            case "castilian" -> "es";
            case "flemish" -> "nl";
            case "valencian" -> "ca";
            case "moldavian", "moldovan" -> "ro";
            case "letzeburgesch" -> "lb";
            case "pushto" -> "ps";
            case "panjabi" -> "pa";
            case "sinhalese" -> "si";
            case "haitian" -> "ht";
            case "mandarin" -> "zh";
            default -> null;
        };
    }
}
