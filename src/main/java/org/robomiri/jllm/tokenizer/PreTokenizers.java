package org.robomiri.jllm.tokenizer;

import org.robomiri.jllm.json.Json;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The {@code pre_tokenizer} stage of a tokenizer.json: a chain of stages, each of which splits (and possibly
 * rewrites) the pieces produced by the previous one. Supported: Sequence, Split (all behaviors, regex or literal,
 * invert), ByteLevel (add_prefix_space, use_regex), Metaspace (prepend_scheme / add_prefix_space / split),
 * Digits, WhitespaceSplit, Whitespace, BertPreTokenizer and Punctuation.
 */
final class PreTokenizers {

    /** HF's ByteLevel default pattern (GPT-2). */
    static final String GPT2_REGEX =
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

    private static final int FLAGS = Pattern.UNICODE_CHARACTER_CLASS;

    /**
     * One punctuation character as BERT defines it: ASCII punctuation (which includes symbols such as {@code $},
     * {@code +}, {@code <}, {@code |}) or Unicode General_Category P. Matching one character at a time is what
     * makes {@code "?!"} two pieces under the Isolated behavior.
     */
    private static final String PUNCTUATION = "[!-/:-@\\[-`{-~]|\\p{P}";

    private PreTokenizers() {
    }

    interface Stage {
        /**
         * @param first true when the text being processed starts at offset 0 of the user's input (only
         *              Metaspace with prepend_scheme "first" cares)
         */
        List<String> apply(List<String> pieces, boolean first);
    }

    /** A composed chain of stages. */
    static final class Chain {
        final List<Stage> stages = new ArrayList<>();
        boolean byteLevel;
        boolean metaspace;

        List<String> apply(String text, boolean first) {
            List<String> pieces = new ArrayList<>(1);
            if (!text.isEmpty()) {
                pieces.add(text);
            }
            for (Stage s : stages) {
                pieces = s.apply(pieces, first);
            }
            return pieces;
        }
    }

    static Chain build(Map<String, Object> node) {
        Chain chain = new Chain();
        if (node != null) {
            collect(node, chain);
        }
        return chain;
    }

    private static void collect(Map<String, Object> node, Chain chain) {
        String type = Json.str(node, "type", "");
        switch (type) {
            case "Sequence" -> {
                List<Object> children = Json.list(node, "pretokenizers");
                if (children != null) {
                    for (Object c : children) {
                        collect(Tokenizer.asMap(c), chain);
                    }
                }
            }
            case "Split" -> {
                Map<String, Object> pattern = Json.map(node, "pattern");
                String regex = Json.str(pattern, "Regex", null);
                String literal = Json.str(pattern, "String", null);
                if (regex == null && literal == null) {
                    throw new IllegalArgumentException("Split pre-tokenizer has no String/Regex pattern");
                }
                Pattern p = Pattern.compile(regex != null ? regex : Pattern.quote(literal), FLAGS);
                String behavior = Json.str(node, "behavior", "Isolated");
                boolean invert = Json.bool(node, "invert", false);
                chain.stages.add((pieces, first) -> splitAll(pieces, p, behavior, invert));
            }
            case "ByteLevel" -> {
                chain.byteLevel = true;
                boolean prefix = Json.bool(node, "add_prefix_space", false);
                boolean useRegex = Json.bool(node, "use_regex", true);
                Pattern p = Pattern.compile(GPT2_REGEX, FLAGS);
                chain.stages.add((pieces, first) -> {
                    List<String> in = pieces;
                    if (prefix) {
                        in = new ArrayList<>(pieces.size());
                        for (String s : pieces) {
                            in.add(s.startsWith(" ") ? s : " " + s);
                        }
                    }
                    return useRegex ? splitAll(in, p, "Isolated", false) : in;
                });
            }
            case "Metaspace" -> {
                chain.metaspace = true;
                String repl = Json.str(node, "replacement", "▁");
                String scheme = Json.str(node, "prepend_scheme", null);
                if (scheme == null) {
                    scheme = Json.bool(node, "add_prefix_space", true) ? "always" : "never";
                }
                boolean split = Json.bool(node, "split", true);
                String sch = scheme;
                Pattern rp = Pattern.compile(Pattern.quote(repl));
                chain.stages.add((pieces, first) -> {
                    List<String> out = new ArrayList<>(pieces.size() + 1);
                    for (int i = 0; i < pieces.size(); i++) {
                        String s = pieces.get(i).replace(" ", repl);
                        if (!s.isEmpty() && !s.startsWith(repl)
                                && (sch.equals("always") || (sch.equals("first") && first && i == 0))) {
                            s = repl + s;
                        }
                        out.add(s);
                    }
                    return split ? splitAll(out, rp, "MergedWithNext", false) : out;
                });
            }
            case "Digits" -> {
                boolean individual = Json.bool(node, "individual_digits", false);
                Pattern p = Pattern.compile("\\p{Nd}", FLAGS);
                chain.stages.add((pieces, first) ->
                        splitAll(pieces, p, individual ? "Isolated" : "Contiguous", false));
            }
            case "WhitespaceSplit" -> {
                Pattern p = Pattern.compile("\\s+", FLAGS);
                chain.stages.add((pieces, first) -> splitAll(pieces, p, "Removed", false));
            }
            case "Whitespace" -> {
                Pattern p = Pattern.compile("\\w+|[^\\w\\s]+", FLAGS);
                chain.stages.add((pieces, first) -> splitAll(pieces, p, "Removed", true));
            }
            case "BertPreTokenizer" -> { // whitespace removed, then every punctuation character on its own
                Pattern ws = Pattern.compile("\\s+", FLAGS);
                Pattern punct = Pattern.compile(PUNCTUATION, FLAGS);
                chain.stages.add((pieces, first) -> splitAll(splitAll(pieces, ws, "Removed", false), punct,
                        "Isolated", false));
            }
            case "Punctuation" -> {
                Pattern punct = Pattern.compile(PUNCTUATION, FLAGS);
                String behavior = Json.str(node, "behavior", "Isolated");
                chain.stages.add((pieces, first) -> splitAll(pieces, punct, behavior, false));
            }
            default -> throw new IllegalArgumentException("unsupported pre_tokenizer type: " + type);
        }
    }

    private static List<String> splitAll(List<String> pieces, Pattern p, String behavior, boolean invert) {
        List<String> out = new ArrayList<>(pieces.size() + 8);
        for (String s : pieces) {
            split(s, p, behavior, invert, out);
        }
        return out;
    }

    /**
     * HF {@code NormalizedString::split}: the text is cut into alternating "match" and "gap" spans (invert swaps
     * their roles) and the behavior decides what to keep and how to glue spans.
     */
    static void split(String s, Pattern p, String behavior, boolean invert, List<String> out) {
        if (s.isEmpty()) {
            return;
        }
        // spans as (start, end, isMatch)
        List<int[]> spans = new ArrayList<>();
        Matcher m = p.matcher(s);
        int last = 0;
        while (m.find()) {
            if (m.end() == m.start()) {
                continue;
            }
            if (m.start() > last) {
                spans.add(new int[]{last, m.start(), invert ? 1 : 0});
            }
            spans.add(new int[]{m.start(), m.end(), invert ? 0 : 1});
            last = m.end();
        }
        if (last < s.length()) {
            spans.add(new int[]{last, s.length(), invert ? 1 : 0});
        }
        switch (behavior) {
            case "Removed" -> {
                for (int[] sp : spans) {
                    if (sp[2] == 0) {
                        out.add(s.substring(sp[0], sp[1]));
                    }
                }
            }
            case "Isolated" -> {
                for (int[] sp : spans) {
                    out.add(s.substring(sp[0], sp[1]));
                }
            }
            case "Contiguous" -> {
                int i = 0;
                while (i < spans.size()) {
                    int j = i;
                    while (j + 1 < spans.size() && spans.get(j + 1)[2] == spans.get(i)[2]) {
                        j++;
                    }
                    out.add(s.substring(spans.get(i)[0], spans.get(j)[1]));
                    i = j + 1;
                }
            }
            case "MergedWithPrevious" -> {
                List<int[]> merged = new ArrayList<>(); // (start, end)
                boolean prevMatch = false;
                for (int[] sp : spans) {
                    boolean match = sp[2] == 1;
                    if (match && !prevMatch && !merged.isEmpty()) {
                        merged.get(merged.size() - 1)[1] = sp[1];
                    } else {
                        merged.add(new int[]{sp[0], sp[1]});
                    }
                    prevMatch = match;
                }
                for (int[] sp : merged) {
                    out.add(s.substring(sp[0], sp[1]));
                }
            }
            case "MergedWithNext" -> {
                List<int[]> merged = new ArrayList<>();
                boolean prevMatch = false;
                for (int k = spans.size() - 1; k >= 0; k--) {
                    int[] sp = spans.get(k);
                    boolean match = sp[2] == 1;
                    if (match && !prevMatch && !merged.isEmpty()) {
                        merged.get(merged.size() - 1)[0] = sp[0];
                    } else {
                        merged.add(new int[]{sp[0], sp[1]});
                    }
                    prevMatch = match;
                }
                for (int k = merged.size() - 1; k >= 0; k--) {
                    out.add(s.substring(merged.get(k)[0], merged.get(k)[1]));
                }
            }
            default -> throw new IllegalArgumentException("unsupported Split behavior: " + behavior);
        }
    }
}
