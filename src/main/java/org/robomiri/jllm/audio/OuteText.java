package org.robomiri.jllm.audio;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * OuteTTS 0.2 / 0.3's text front end for Latin-script text (English, German, French, ...), a port of the {@code
 * TextProcessor} of the Python package {@code outetts} without its native dependencies:
 * <ol>
 *   <li>the apostrophes and backticks go ("don't" is read "dont" - the model was trained that way);</li>
 *   <li>the text is split into letter runs, digit runs and single symbols (what MeCab's {@code -Owakati} does to Latin text),
 *       and joined again with two spaces between words and no space before punctuation;</li>
 *   <li>letters are transliterated to ASCII (accents dropped, ß to ss, Æ to AE, ...), dashes become spaces, numbers are
 *       spelled out ({@link NumberWords}) and everything but letters and the punctuation the model knows is removed;</li>
 *   <li>each word keeps the punctuation tokens that stood before and after it ({@code <|comma|>}, {@code <|period|>}, ...).</li>
 * </ol>
 * Chinese, Japanese and Korean text needs MeCab / uroman and is not supported (those characters are dropped).
 */
public final class OuteText {

    private OuteText() {
    }

    /** A word with the punctuation tokens before and after it. */
    public record Word(String word, List<String> before, List<String> after) {
        /** The prompt text of the word: punctuation tokens around the letters. */
        public String text() {
            StringBuilder sb = new StringBuilder();
            before.forEach(sb::append);
            sb.append(word);
            after.forEach(sb::append);
            return sb.toString();
        }
    }

    static final Map<String, String> PUNCT = new LinkedHashMap<>();

    static {
        PUNCT.put(".", "<|period|>");
        PUNCT.put("!", "<|exclamation_mark|>");
        PUNCT.put("?", "<|question_mark|>");
        PUNCT.put(",", "<|comma|>");
        PUNCT.put("\"", "<|double_quote|>");
        PUNCT.put("„", "<|low_double_quote|>");
        PUNCT.put("¡", "<|inverted_exclamation|>");
        PUNCT.put("¿", "<|inverted_question|>");
        PUNCT.put("…", "<|ellipsis|>");
        PUNCT.put("...", "<|ellipsis|>");
        PUNCT.put("。", "<|cjk_period|>");
        PUNCT.put("！", "<|cjk_exclamation|>");
        PUNCT.put("？", "<|cjk_question|>");
        PUNCT.put("，", "<|cjk_comma|>");
        PUNCT.put("؟", "<|arabic_question|>");
    }

    private static final String ASCII_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    private static final Pattern WHITESPACE = Pattern.compile("(?<!\\s)\\s(?!\\s)", Pattern.UNICODE_CHARACTER_CLASS);
    private static final Pattern MULTIPLE_SPACES = Pattern.compile("\\s+", Pattern.UNICODE_CHARACTER_CLASS);
    private static final Pattern NUMBER = Pattern.compile("\\d+(\\.\\d+)?");
    private static final Pattern PUNCT_REMOVE =
            Pattern.compile("[^a-zA-Z\\u0080-\\uffff.!?,\"„¡¿…。！？，؟\\s]", Pattern.UNICODE_CHARACTER_CLASS);
    private static final Pattern CLEAN = Pattern.compile("[^a-z\\s]");

    /** The words of {@code text} as OuteTTS reads them. */
    public static List<Word> process(String text) {
        List<String> tokens = wakati(text.replace("'", "").replace("`", ""));
        if (tokens.isEmpty()) {
            return List.of();
        }
        StringBuilder joined = new StringBuilder(tokens.getFirst());
        for (String t : tokens.subList(1, tokens.size())) {
            joined.append(PUNCT.containsKey(t) ? " " : "  ").append(t);
        }
        String s = WHITESPACE.matcher(joined).replaceAll("");
        s = asciify(s);
        s = s.replace('—', ' ').replace('–', ' ').replace('-', ' ');
        s = replaceNumbers(s.toLowerCase(Locale.ROOT));
        s = PUNCT_REMOVE.matcher(s).replaceAll("");
        s = MULTIPLE_SPACES.matcher(s).replaceAll(" ").strip().toLowerCase(Locale.ROOT);
        String original = s;
        List<String> words = new ArrayList<>();
        for (String w : CLEAN.matcher(s).replaceAll("").split("\\s+")) {
            if (!w.isEmpty()) {
                words.add(w);
            }
        }
        List<String> originalSplit = joinPunctuation(original);
        List<Word> out = new ArrayList<>();
        for (int i = 0; i < words.size(); i++) {
            List<String> before = List.of();
            List<String> after = List.of();
            if (i < originalSplit.size()) {
                String orig = originalSplit.get(i);
                int at = orig.toLowerCase(Locale.ROOT).indexOf(words.get(i));
                if (at >= 0) {
                    before = marks(orig.substring(0, at));
                    after = marks(orig.substring(at + words.get(i).length()));
                }
            }
            out.add(new Word(words.get(i), before, after));
        }
        return out;
    }

    /** The prompt form of a word list: words joined by {@code <|space|>}. */
    public static String promptText(List<Word> words) {
        StringBuilder sb = new StringBuilder();
        for (Word w : words) {
            if (!sb.isEmpty()) {
                sb.append("<|space|>");
            }
            sb.append(w.text());
        }
        return sb.toString();
    }

    private static List<String> marks(String s) {
        List<String> out = new ArrayList<>();
        s.codePoints().forEach(cp -> {
            String m = PUNCT.get(new String(Character.toChars(cp)));
            if (m != null) {
                out.add(m);
            }
        });
        return out;
    }

    private static List<String> joinPunctuation(String text) {
        List<String> t = new ArrayList<>();
        for (String w : MULTIPLE_SPACES.split(text.strip())) {
            if (!w.isEmpty()) {
                t.add(w);
            }
        }
        int i = 0;
        while (i < t.size()) {
            if (PUNCT.containsKey(t.get(i)) && i > 0) {
                t.set(i - 1, t.get(i - 1) + t.get(i));
                t.remove(i);
            } else {
                i++;
            }
        }
        List<String> out = new ArrayList<>();
        for (String w : t) {
            if (PUNCT.containsKey(w)) {
                continue;
            }
            boolean allPunct = true;
            for (int k = 0; k < w.length(); k++) {
                if (ASCII_PUNCTUATION.indexOf(w.charAt(k)) < 0) {
                    allPunct = false;
                    break;
                }
            }
            if (!allPunct) {
                out.add(w);
            }
        }
        return out;
    }

    private static String replaceNumbers(String s) {
        Matcher m = NUMBER.matcher(s);
        StringBuilder sb = new StringBuilder();
        while (m.find()) {
            m.appendReplacement(sb, Matcher.quoteReplacement(NumberWords.of(m.group())));
        }
        m.appendTail(sb);
        return sb.toString();
    }

    /** Letter runs, digit runs and single other symbols; white space separates (the MeCab {@code -Owakati} of Latin text). */
    static List<String> wakati(String text) {
        List<String> out = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        int kind = 0;                  // 1 letters, 2 digits
        boolean prevSymbol = false;    // the previous character was a symbol token, with no white space since
        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            i += Character.charCount(cp);
            int k = Character.isLetter(cp) || Character.getType(cp) == Character.NON_SPACING_MARK ? 1
                    : Character.isDigit(cp) ? 2 : Character.isWhitespace(cp) ? -1 : 0;
            if (k > 0 && k == kind) {
                cur.appendCodePoint(cp);
                continue;
            }
            if (!cur.isEmpty()) {
                out.add(cur.toString());
                cur.setLength(0);
            }
            kind = 0;
            if (k > 0) {
                cur.appendCodePoint(cp);
                kind = k;
                prevSymbol = false;
            } else if (k == 0) {
                String sym = new String(Character.toChars(cp));
                // the dictionary has entries joining a quotation mark and an adjacent . , ! ? (",\"" -> ,") into one token,
                // which then is not punctuation the model knows and is dropped with it
                if (prevSymbol && !out.isEmpty()) {
                    String p = out.getLast();
                    boolean pq = p.equals("\"");
                    boolean tq = sym.equals("\"");
                    boolean pp = p.length() == 1 && ".,!?".contains(p);
                    boolean tp = ".,!?".contains(sym);
                    if ((pp && tq) || (pq && tp)) {
                        out.set(out.size() - 1, p + sym);
                        continue;
                    }
                }
                out.add(sym);
                prevSymbol = true;
            } else {
                prevSymbol = false;
            }
        }
        if (!cur.isEmpty()) {
            out.add(cur.toString());
        }
        return out;
    }

    /** Latin letters with their accents removed (what {@code anyascii} does to them); other scripts are dropped. */
    static String asciify(String s) {
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            i += Character.charCount(cp);
            if (cp < 128 || PUNCT.containsKey(new String(Character.toChars(cp)))) {
                sb.appendCodePoint(cp);
                continue;
            }
            String special = switch (cp) {
                case 'Æ' -> "AE";
                case 'æ' -> "ae";
                case 'Œ' -> "OE";
                case 'œ' -> "oe";
                case 'ß' -> "ss";
                case 'Ø' -> "O";
                case 'ø' -> "o";
                case 'Ł' -> "L";
                case 'ł' -> "l";
                case 'Đ', 'Ð' -> "D";
                case 'đ', 'ð' -> "d";
                case 'Þ' -> "Th";
                case 'þ' -> "th";
                case 'ı' -> "i";
                case 'Ħ' -> "H";
                case 'ħ' -> "h";
                case '’', '‘' -> "";
                case '“', '”' -> "\"";
                case '«', '»' -> "\"";
                case '€' -> "";
                default -> null;
            };
            if (special != null) {
                sb.append(special);
                continue;
            }
            String d = Normalizer.normalize(new String(Character.toChars(cp)), Normalizer.Form.NFKD);
            for (int k = 0; k < d.length(); k++) {
                char c = d.charAt(k);
                if (c < 128) {
                    sb.append(c);
                }
            }
        }
        return sb.toString();
    }
}
