package org.robomiri.jllm.server;

import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.TokenLogprob;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.sampling.JsonConstraint;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The OpenAI request fields beyond plain sampling, parsed and turned into engine options, plus the response pieces
 * they add: {@code n}, {@code best_of}, {@code logprobs} / {@code top_logprobs}, {@code logit_bias},
 * {@code response_format}, {@code echo} and {@code suffix}. Shared by {@link ApiServer} and the web console's
 * OpenAI endpoints so that both behave the same; see docs/10 §10.3 for the semantics.
 */
public final class OpenAiFeatures {

    /** Largest {@code n} / {@code best_of}: the choices run one after the other, so this bounds a request's time. */
    public static final int MAX_N = 16;
    /** Largest {@code top_logprobs} (chat) / {@code logprobs} (completions), as OpenAI's. */
    public static final int MAX_TOP_LOGPROBS = 20;

    private OpenAiFeatures() {
    }

    /**
     * @param n           choices returned
     * @param bestOf      sequences generated (completions; {@code >= n}); the n with the highest cumulative
     *                    log-probability are returned
     * @param logprobs    whether the response carries logprobs
     * @param topLogprobs alternatives per position
     * @param echo        completions: the prompt is returned in front of the completion
     * @param suffix      completions: text after the insertion point (fill-in-the-middle), or null
     * @param logitBias   dense bias by token id, or null
     * @param format      the JSON constraint of {@code response_format}, or null for text
     */
    public record Options(int n, int bestOf, boolean logprobs, int topLogprobs, boolean echo, String suffix,
                          float[] logitBias, JsonConstraint.Schema format) {

        /** How many sequences the engine generates. */
        public int count() {
            return Math.max(n, bestOf);
        }

        /** Whether the output is constrained to JSON. */
        public boolean json() {
            return format != null;
        }

        /** The engine options for this request (logprobs are also computed when {@code best_of} needs them for ranking). */
        public Engine.Decoding decoding(Tokenizer tok) {
            JsonConstraint.Schema f = format;
            byte[][] table = f == null ? null : JsonConstraint.tokenTable(tok);
            return new Engine.Decoding(logitBias, logprobs || bestOf > n, topLogprobs, echo && logprobs,
                    f == null ? null : () -> new JsonConstraint(f, table));
        }
    }

    /**
     * Parses and validates the fields.
     *
     * @param vocab number of token ids (for {@code logit_bias} keys)
     * @throws IllegalArgumentException with a client-facing message for an invalid or unsupported value
     */
    public static Options parse(Map<String, Object> b, boolean chat, int vocab) {
        boolean stream = Json.bool(b, "stream", false);
        int n = intField(b, "n", 1, 1, MAX_N);
        int bestOf = n;
        boolean logprobs;
        int top = 0;
        if (chat) {
            Object lp = b.get("logprobs");
            if (lp != null && !(lp instanceof Boolean)) {
                throw new IllegalArgumentException("'logprobs' must be a boolean on /v1/chat/completions");
            }
            logprobs = Boolean.TRUE.equals(lp);
            if (b.get("top_logprobs") != null) {
                top = intField(b, "top_logprobs", 0, 0, MAX_TOP_LOGPROBS);
                if (!logprobs && top > 0) {
                    throw new IllegalArgumentException("'top_logprobs' needs 'logprobs': true");
                }
            }
            for (String f : new String[]{"echo", "suffix", "best_of"}) {
                if (b.get(f) != null && !Boolean.FALSE.equals(b.get(f))) {
                    throw new IllegalArgumentException("'" + f + "' is only supported on /v1/completions");
                }
            }
        } else {
            logprobs = b.get("logprobs") != null && !Boolean.FALSE.equals(b.get("logprobs"));
            if (logprobs) {
                top = intField(b, "logprobs", 0, 0, MAX_TOP_LOGPROBS);
            }
            bestOf = intField(b, "best_of", n, 1, MAX_N);
            if (bestOf < n) {
                throw new IllegalArgumentException("'best_of' (" + bestOf + ") must be at least 'n' (" + n + ")");
            }
            if (stream && bestOf > 1) {
                throw new IllegalArgumentException("'best_of' cannot be used with 'stream'");
            }
        }
        Object echo = b.get("echo");
        if (echo != null && !(echo instanceof Boolean)) {
            throw new IllegalArgumentException("'echo' must be a boolean");
        }
        Object suffix = b.get("suffix");
        if (suffix != null && !(suffix instanceof String)) {
            throw new IllegalArgumentException("'suffix' must be a string");
        }
        if (Boolean.TRUE.equals(echo) && suffix != null) {
            throw new IllegalArgumentException("'echo' and 'suffix' cannot be used together");
        }
        return new Options(n, bestOf, logprobs, top, Boolean.TRUE.equals(echo), (String) suffix,
                logitBias(b, vocab), format(b));
    }

    /** {@code logit_bias}: token id (as a string) to a bias in [-100, 100], as a dense array; null when absent. */
    static float[] logitBias(Map<String, Object> b, int vocab) {
        Object o = b.get("logit_bias");
        if (o == null) {
            return null;
        }
        if (!(o instanceof Map<?, ?> m)) {
            throw new IllegalArgumentException("'logit_bias' must be an object of token id -> bias");
        }
        if (m.isEmpty()) {
            return null;
        }
        float[] bias = new float[vocab];
        for (Map.Entry<?, ?> e : m.entrySet()) {
            int id;
            try {
                id = Integer.parseInt(String.valueOf(e.getKey()).trim());
            } catch (NumberFormatException ex) {
                throw new IllegalArgumentException("'logit_bias' keys must be token ids, got \"" + e.getKey() + "\"");
            }
            if (id < 0 || id >= vocab) {
                throw new IllegalArgumentException("'logit_bias' token id " + id + " is outside the vocabulary [0, " + vocab + ")");
            }
            if (!(e.getValue() instanceof Number v) || !(v.doubleValue() >= -100 && v.doubleValue() <= 100)) {
                throw new IllegalArgumentException("'logit_bias' values must be numbers in [-100, 100] (token " + id + ")");
            }
            bias[id] = v.floatValue();
        }
        return bias;
    }

    /** {@code response_format}: null for text, else the schema the output is constrained to. */
    static JsonConstraint.Schema format(Map<String, Object> b) {
        Object o = b.get("response_format");
        if (o == null) {
            return null;
        }
        if (!(o instanceof Map<?, ?>)) {
            throw new IllegalArgumentException("'response_format' must be an object such as {\"type\": \"json_object\"}");
        }
        String type = Json.str(o, "type", "text");
        return switch (type) {
            case "text" -> null;
            case "json_object" -> JsonConstraint.Schema.anyObject();
            case "json_schema" -> {
                Map<String, Object> js = Json.map(o, "json_schema");
                Map<String, Object> schema = js == null ? null : Json.map(js, "schema");
                if (schema == null) {
                    throw new IllegalArgumentException("'response_format' of type json_schema needs"
                            + " {\"json_schema\": {\"schema\": {...}}}");
                }
                try {
                    yield JsonConstraint.Schema.compile(schema);
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException("response_format: " + e.getMessage());
                }
            }
            default -> throw new IllegalArgumentException("'response_format' type must be text, json_object or"
                    + " json_schema (got \"" + type + "\")");
        };
    }

    /** An integer field in {@code [min, max]}; fractions and other types are a 400. */
    private static int intField(Map<String, Object> b, String key, int def, int min, int max) {
        Object v = b.get(key);
        if (v == null) {
            return def;
        }
        if (!(v instanceof Number num) || (v instanceof Double x && x != Math.rint(x))) {
            throw new IllegalArgumentException("'" + key + "' must be an integer");
        }
        double x = num.doubleValue();
        if (x < min || x > max) {
            throw new IllegalArgumentException("'" + key + "' must be between " + min + " and " + max);
        }
        return (int) x;
    }

    // ------------------------------------------------------------------ prompts

    /**
     * The fill-in-the-middle prompt for a completion between {@code prefix} and {@code suffix}, in the
     * prefix-suffix-middle order of the Qwen2.5-Coder family: {@code <|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>};
     * the model then writes the middle.
     *
     * @throws IllegalArgumentException when the tokenizer has no such tokens (the model was not trained for it)
     */
    public static int[] fimPrompt(Tokenizer tok, String prefix, String suffix) {
        int pre = tok.tokenId("<|fim_prefix|>");
        int suf = tok.tokenId("<|fim_suffix|>");
        int mid = tok.tokenId("<|fim_middle|>");
        if (pre < 0 || suf < 0 || mid < 0) {
            throw new IllegalArgumentException("'suffix' (fill-in-the-middle) needs a model whose tokenizer has the"
                    + " <|fim_prefix|>, <|fim_suffix|> and <|fim_middle|> tokens; this one has not");
        }
        int[] a = tok.encode(prefix);
        int[] c = tok.encode(suffix);
        int[] out = new int[a.length + c.length + 3];
        out[0] = pre;
        System.arraycopy(a, 0, out, 1, a.length);
        out[a.length + 1] = suf;
        System.arraycopy(c, 0, out, a.length + 2, c.length);
        out[out.length - 1] = mid;
        return out;
    }

    // ------------------------------------------------------------------ results

    /** The results to return: all of them, or for {@code best_of} the {@code n} most likely, most likely first. */
    public static List<Engine.Result> select(List<Engine.Result> all, Options o) {
        if (o.bestOf() <= o.n()) {
            return all;
        }
        List<Engine.Result> sorted = new ArrayList<>(all);
        sorted.sort(Comparator.comparingDouble(Engine.Result::cumulativeLogprob).reversed());
        return sorted.subList(0, o.n());
    }

    /**
     * The usage of all generated sequences: the prompt once, the completion tokens of every sequence (also those
     * {@code best_of} discarded), the times added up.
     */
    public static Engine.Stats total(List<Engine.Result> all) {
        Engine.Stats f = all.getFirst().stats();
        int gen = 0;
        int drafted = 0;
        int accepted = 0;
        double prefill = 0;
        double decode = 0;
        for (Engine.Result r : all) {
            Engine.Stats s = r.stats();
            gen += s.generatedTokens();
            drafted += s.draftedTokens();
            accepted += s.acceptedTokens();
            prefill += s.prefillMs();
            decode += s.decodeMs();
        }
        return new Engine.Stats(f.promptTokens(), f.cachedTokens(), gen, prefill, decode, drafted, accepted);
    }

    /** A token's text: its bytes as UTF-8 (a fragment of a multi-byte character shows as U+FFFD). */
    public static String tokenText(Tokenizer tok, int id) {
        return new String(tok.tokenBytes(id), StandardCharsets.UTF_8);
    }

    private static List<Integer> bytes(Tokenizer tok, int id) {
        byte[] b = tok.tokenBytes(id);
        List<Integer> out = new ArrayList<>(b.length);
        for (byte x : b) {
            out.add(x & 0xff);
        }
        return out;
    }

    /** Chat {@code choices[].logprobs}: {@code {"content": [{token, logprob, bytes, top_logprobs}, ...]}}. */
    public static Map<String, Object> chatLogprobs(List<TokenLogprob> lps, Tokenizer tok) {
        List<Object> content = new ArrayList<>(lps.size());
        for (TokenLogprob lp : lps) {
            List<Object> top = new ArrayList<>(lp.topTokens().length);
            for (int i = 0; i < lp.topTokens().length; i++) {
                int id = lp.topTokens()[i];
                top.add(Json.obj("token", tokenText(tok, id), "logprob", lp.topLogprobs()[i], "bytes", bytes(tok, id)));
            }
            content.add(Json.obj("token", tokenText(tok, lp.token()), "logprob", lp.logprob(),
                    "bytes", bytes(tok, lp.token()), "top_logprobs", top));
        }
        return Json.obj("content", content, "refusal", null);
    }

    /**
     * Completions {@code choices[].logprobs}: parallel lists {@code tokens}, {@code token_logprobs},
     * {@code top_logprobs} (a token-to-logprob object per position that, like OpenAI's, also holds the chosen token,
     * so {@code logprobs: 0} still reports it) and {@code text_offset} (each token's character offset in the text).
     *
     * @param ids    the tokens, in order
     * @param lps    their logprobs (an entry may be null: the first prompt token of an echo)
     * @param offset {@code offset[0]} is the text offset of the first token; advanced past the last one
     */
    public static Map<String, Object> completionLogprobs(int[] ids, List<TokenLogprob> lps, Tokenizer tok, int[] offset) {
        List<Object> tokens = new ArrayList<>(ids.length);
        List<Object> logprobs = new ArrayList<>(ids.length);
        List<Object> tops = new ArrayList<>(ids.length);
        List<Object> offsets = new ArrayList<>(ids.length);
        for (int i = 0; i < ids.length; i++) {
            String text = tokenText(tok, ids[i]);
            TokenLogprob lp = lps.get(i);
            tokens.add(text);
            logprobs.add(lp == null ? null : lp.logprob());
            if (lp == null) {
                tops.add(null);
            } else {
                Map<String, Object> t = new LinkedHashMap<>();
                for (int j = 0; j < lp.topTokens().length; j++) {
                    t.putIfAbsent(tokenText(tok, lp.topTokens()[j]), lp.topLogprobs()[j]);
                }
                t.putIfAbsent(text, lp.logprob());
                tops.add(t);
            }
            offsets.add(offset[0]);
            offset[0] += text.length();
        }
        return Json.obj("tokens", tokens, "token_logprobs", logprobs, "top_logprobs", tops, "text_offset", offsets);
    }

    /**
     * The {@code logprobs} of an echoed prompt (completions with {@code echo}): the prompt tokens from
     * {@code from} on (skipping a BOS the engine put in front), the first without a log-probability when nothing
     * precedes it.
     *
     * @param promptLps {@link Engine.Result#promptLogprobs()}; null (a cancelled request) gives null entries
     */
    public static Map<String, Object> echoLogprobs(int[] prompt, int from, List<TokenLogprob> promptLps,
                                                   Tokenizer tok, int[] offset) {
        int[] ids = java.util.Arrays.copyOfRange(prompt, from, prompt.length);
        List<TokenLogprob> lps = new ArrayList<>(ids.length);
        for (int i = from; i < prompt.length; i++) {
            lps.add(promptLps == null ? null : promptLps.get(i));
        }
        return completionLogprobs(ids, lps, tok, offset);
    }

    /**
     * The {@code logprobs} of a whole choice in a non-streamed response.
     *
     * @param prompt the prompt the engine saw; with {@code echo} its tokens from {@code echoFrom} come first
     */
    public static Map<String, Object> choiceLogprobs(boolean chat, Engine.Result r, Options o, int[] prompt,
                                                     int echoFrom, Tokenizer tok) {
        List<TokenLogprob> gen = r.logprobs() == null ? List.of() : r.logprobs();
        if (chat) {
            return chatLogprobs(gen, tok);
        }
        List<TokenLogprob> lps = new ArrayList<>();
        int[] ids = ids(gen);
        if (o.echo()) {
            for (int i = echoFrom; i < prompt.length; i++) {
                lps.add(r.promptLogprobs() == null ? null : r.promptLogprobs().get(i));
            }
            int[] both = java.util.Arrays.copyOfRange(prompt, echoFrom, prompt.length + ids.length);
            System.arraycopy(ids, 0, both, prompt.length - echoFrom, ids.length);
            ids = both;
        }
        lps.addAll(gen);
        return completionLogprobs(ids, lps, tok, new int[]{0});
    }

    /**
     * The {@code logprobs} of a streamed chunk: the tokens generated since the last chunk ({@code pending}, which is
     * cleared), or null when there are none.
     *
     * @param offset completions: {@code offset[0]} is the running text offset of the choice
     */
    public static Map<String, Object> drainLogprobs(boolean chat, List<TokenLogprob> pending, Tokenizer tok, int[] offset) {
        if (pending.isEmpty()) {
            return null;
        }
        List<TokenLogprob> lps = List.copyOf(pending);
        pending.clear();
        return chat ? chatLogprobs(lps, tok) : completionLogprobs(ids(lps), lps, tok, offset);
    }

    /** The token ids of a list of logprobs. */
    public static int[] ids(List<TokenLogprob> lps) {
        int[] ids = new int[lps.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = lps.get(i).token();
        }
        return ids;
    }
}
