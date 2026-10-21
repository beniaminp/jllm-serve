package org.robomiri.jllm;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.ChatTemplate.Message;
import org.robomiri.jllm.tokenizer.PromptCache;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * {@link PromptCache}: rendering a conversation through the cache gives exactly the ids of a full render, for every
 * chat template, for conversations that grow turn by turn (the agent case), that are edited in the middle, that run
 * side by side, with a cache small enough to evict, and with added tokens that swallow whitespace (lstrip / rstrip).
 * Then measures render + tokenize of a 50-turn, ~100k-character conversation without and with the cache.
 *
 * <p>usage: {@code PromptCacheTest <model-dir>} (uses its tokenizer.json only).
 */
public final class PromptCacheTest {

    private static int checks;

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args.length > 0 ? args[0] : "models/Qwen_Qwen3-0.6B");
        String json = Files.readString(dir.resolve("tokenizer.json"));
        Tokenizer tok = Tokenizer.fromJson(json);
        List<ChatTemplate.Format> formats = List.of(ChatTemplate.forModelType("qwen3"), ChatTemplate.forModelType("qwen2"),
                ChatTemplate.llama3("26 Jul 2024"), ChatTemplate.forModelType("mistral"), ChatTemplate.forModelType("gemma3"),
                ChatTemplate.forModelType("phi4"), ChatTemplate.forModelType("glm4"), ChatTemplate.forModelType("deepseek"),
                ChatTemplate.forModelType("deepseek_v2"), ChatTemplate.forModelType("mamba"));
        for (ChatTemplate.Format f : formats) {
            exactness(tok, f, 1L << 20, 11);
            exactness(tok, f, 20_000, 12); // a cache too small for most prompts: misses and evictions
        }
        // whitespace-swallowing added tokens (Phi-3's <|end|> style): no boundary after an rstrip token
        Tokenizer strip = Tokenizer.fromMaps(stripFlags(Json.parseObject(json)), Map.of());
        exactness(strip, ChatTemplate.forModelType("qwen3"), 1L << 20, 13);
        exactness(strip, ChatTemplate.forModelType("phi4"), 1L << 20, 14);
        sideBySide(tok, ChatTemplate.forModelType("qwen3"));
        measure(tok, ChatTemplate.forModelType("qwen3"));
        System.out.println("prompt cache tests passed (" + checks + " renders compared)");
    }

    /** {@code <|im_end|>} and {@code </think>} strip whitespace before them, {@code <|im_start|>} and {@code <think>} after. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> stripFlags(Map<String, Object> tj) {
        for (Object o : (List<Object>) tj.get("added_tokens")) {
            Map<String, Object> t = (Map<String, Object>) o;
            String c = (String) t.get("content");
            if (c.equals("<|im_end|>") || c.equals("</think>") || c.equals("<|end|>")) {
                t.put("lstrip", true);
            }
            if (c.equals("<|im_start|>") || c.equals("<think>") || c.equals("<|user|>")) {
                t.put("rstrip", true);
            }
        }
        return tj;
    }

    private static final String[] WORDS = {"hello", " world", "\n", "\n\n", "  ", " ", "\t", "def f(x):\n    return x",
            "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_response>", "</tool_response>", "<tool_call>",
            "{\"a\": 1}", "é", "日本語", "🙂", "<|endoftext|>", "[INST]", "<s>", "<|user|>", "<|end|>", "<｜User｜>",
            "  indented", "x", ".", ",", "'s", "123", " 4567", "\r\n", "<start_of_turn>", "<|eot_id|>"};

    private static String text(Random r, int words) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < words; i++) {
            sb.append(WORDS[r.nextInt(WORDS.length)]);
        }
        return sb.toString();
    }

    /** A conversation the template accepts: optional system, then user / assistant turns (Qwen: with tool turns). */
    private static List<Message> conversation(Random r, ChatTemplate.Format f, int turns) {
        boolean tools = f.name().startsWith("chatml");
        List<Message> m = new ArrayList<>();
        if (r.nextBoolean()) {
            m.add(new Message("system", text(r, 1 + r.nextInt(30))));
        }
        for (int t = 0; t < turns; t++) {
            m.add(new Message("user", text(r, r.nextInt(40))));
            String answer = text(r, r.nextInt(40));
            if (f.name().equals("chatml-qwen3") && r.nextInt(3) == 0) {
                answer = "<think>\n" + text(r, 5) + "\n</think>\n\n" + answer;
            }
            if (tools && r.nextInt(3) == 0) { // a tool call, its results, then the answer
                m.add(new Message("assistant", "<tool_call>\n{\"name\": \"f\"}\n</tool_call>"));
                int n = 1 + r.nextInt(2);
                for (int i = 0; i < n; i++) {
                    m.add(new Message("tool", text(r, r.nextInt(20))));
                }
                m.add(new Message("user", text(r, 3)));
            }
            m.add(new Message("assistant", answer));
        }
        return m;
    }

    /** Every request of growing conversations (with an edit now and then) through the cache equals a full render. */
    private static void exactness(Tokenizer tok, ChatTemplate.Format f, long budget, long seed) {
        Random r = new Random(seed);
        PromptCache cache = new PromptCache(tok, budget);
        for (int c = 0; c < 6; c++) {
            List<Message> all = conversation(r, f, 8);
            for (int end = 1; end <= all.size(); end++) {
                List<Message> req = new ArrayList<>(all.subList(0, end));
                if (!req.getLast().role().equals("user") && !f.name().equals("chatml-qwen3")) {
                    continue; // templates that require alternation end requests on a user turn
                }
                if (r.nextInt(8) == 0 && req.size() > 2) { // the agent compacted / edited an earlier message
                    int i = r.nextInt(req.size() - 1);
                    req.set(i, new Message(req.get(i).role(), text(r, 5)));
                    all.set(i, req.get(i));
                }
                boolean think = r.nextBoolean();
                compare(tok, f, cache, req, think);
                compare(tok, f, cache, req, !think); // the generation prompt differs: a hit before it
            }
        }
        PromptCache.Stats st = cache.stats();
        System.out.printf("  %-14s budget %7d: %4d requests, %4d hits, %6d ids reused, %6d encoded, %d entries%n",
                f.name() + (seed >= 13 ? "+strip" : ""), budget, st.requests(), st.hits(),
                st.reusedTokens(), st.encodedTokens(), st.entries());
        check(budget < 100_000 || st.hits() > 0, f.name() + ": the cache is used");
        check(st.bytes() <= budget, f.name() + ": within the budget");
    }

    private static void compare(Tokenizer tok, ChatTemplate.Format f, PromptCache cache, List<Message> req, boolean think) {
        int[] want;
        try {
            want = f.renderTokens(tok, req, true, think);
        } catch (IllegalArgumentException e) {
            return; // the template refuses this shape (e.g. Mistral's alternation): nothing to compare
        }
        int[] got = f.renderTokens(tok, req, true, think, cache);
        checks++;
        if (!Arrays.equals(want, got)) {
            int i = 0;
            while (i < Math.min(want.length, got.length) && want[i] == got[i]) {
                i++;
            }
            throw new AssertionError(f.name() + ": cached render differs at id " + i + " of " + want.length + " ("
                    + got.length + " ids): " + req);
        }
    }

    /** Two conversations sharing their system prompt, requests interleaved, both growing. */
    private static void sideBySide(Tokenizer tok, ChatTemplate.Format f) {
        Random r = new Random(5);
        PromptCache cache = new PromptCache(tok, 1L << 22);
        String system = text(r, 200);
        List<List<Message>> convs = List.of(new ArrayList<>(), new ArrayList<>());
        for (List<Message> c : convs) {
            c.add(new Message("system", system));
        }
        for (int t = 0; t < 10; t++) {
            for (List<Message> c : convs) {
                c.add(new Message("user", text(r, 20)));
                compare(tok, f, cache, c, false);
                c.add(new Message("assistant", text(r, 30)));
            }
        }
        PromptCache.Stats st = cache.stats();
        check(st.hits() == st.requests() - 1, "side by side: every request but the first finds a prefix (" + st + ")");
        System.out.println("  side by side: " + st);
    }

    /** Render + tokenize of the 50th request of a 50-turn, ~100k-character conversation, without and with the cache. */
    private static void measure(Tokenizer tok, ChatTemplate.Format f) {
        Random r = new Random(7);
        List<Message> conv = new ArrayList<>();
        conv.add(new Message("system", prose(r, 8000)));
        for (int t = 0; t < 50; t++) {
            conv.add(new Message("user", prose(r, 600)));
            conv.add(new Message("assistant", prose(r, 1240)));
        }
        long chars = conv.stream().mapToLong(m -> m.content().length()).sum();
        List<Message> last = conv.subList(0, conv.size() - 1); // ends with the 50th user message
        List<Message> before = conv.subList(0, conv.size() - 3); // the 49th request
        double full = Double.MAX_VALUE;
        double cached = Double.MAX_VALUE;
        int[] want = null;
        for (int rep = 0; rep < 15; rep++) {
            long t0 = System.nanoTime();
            want = f.renderTokens(tok, last, true, false);
            full = Math.min(full, (System.nanoTime() - t0) / 1e6);
            PromptCache cache = new PromptCache(tok, 64L << 20);
            f.renderTokens(tok, before, true, false, cache); // the previous request was seen
            long t1 = System.nanoTime();
            int[] got = f.renderTokens(tok, last, true, false, cache);
            cached = Math.min(cached, (System.nanoTime() - t1) / 1e6);
            check(Arrays.equals(want, got), "the measured render is exact");
        }
        System.out.printf("  measured: %d turns, %d chars, %d ids: render + tokenize %.1f ms in full, %.2f ms with the"
                + " previous request cached (%.0fx)%n", 50, chars, want.length, full, cached, full / cached);
    }

    private static final String[] PROSE = {"the", "function", "returns", "a", "list", "of", "files", "in", "src/main",
            "and", "then", "we", "call", "it", "again", "with", "x", "=", "42;", "if", "(err)", "{", "}", "\n", "    "};

    private static String prose(Random r, int chars) {
        StringBuilder sb = new StringBuilder();
        while (sb.length() < chars) {
            sb.append(PROSE[r.nextInt(PROSE.length)]).append(r.nextInt(6) == 0 ? "\n" : " ");
        }
        return sb.toString();
    }

    private static void check(boolean ok, String what) {
        if (!ok) {
            throw new AssertionError("FAILED: " + what);
        }
    }
}
