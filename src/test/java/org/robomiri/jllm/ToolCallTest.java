package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.engine.DraftModel;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.Speculator;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.QuantFile;
import org.robomiri.jllm.model.ShardProvider;
import org.robomiri.jllm.sampling.JsonConstraint;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.sampling.TokenConstraint;
import org.robomiri.jllm.sampling.ToolCallConstraint;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;

import static org.robomiri.jllm.SmokeTest.check;

/**
 * Constrained tool calls ({@link ToolCallConstraint}): the automaton on its own, the vocabulary masks, jump-forward,
 * speculation under the constraint (greedy output identical to plain constrained decoding) against a scripted fake
 * model, and on the real Qwen3-0.6B. Run in the test phase (pom: {@code tool-call-test}) with
 * {@code -Djllm.cpu.deterministic=true}, which makes multi-token CPU passes bit-identical to single-token ones.
 */
public final class ToolCallTest {

    static final String TOOLS = """
            [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
              "parameters": {"type": "object", "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer", "minimum": 1, "maximum": 7},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
              "required": ["city"], "additionalProperties": false}}},
             {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "pattern": "^[^\\\\0]+$"},
                "lines": {"type": "array", "items": {"type": "integer"}},
                "encoding": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "options": {"oneOf": [{"type": "object", "properties": {"a": {"type": "boolean"}}},
                                      {"type": "object", "properties": {"b": {"type": "number"}}}]}},
              "required": ["path"]}}},
             {"type": "function", "function": {"name": "read"}}]
            """;

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args.length > 0 ? args[0] : "models/Qwen_Qwen3-0.6B");
        lenientSchemas();
        automaton();
        Tokenizer tok = Tokenizer.load(dir.resolve("tokenizer.json"));
        masks(tok);
        jumpTokens(tok);
        fakeEngine(tok);
        if (args.length > 1 && args[1].equals("--no-model")) {
            System.out.println("tool call tests passed (no model)");
            return;
        }
        realModel(dir, tok);
        System.out.println("tool call tests passed");
    }

    @SuppressWarnings("unchecked")
    static List<Object> tools() {
        return (List<Object>) Json.parse(TOOLS);
    }

    static ToolCallConstraint.Spec spec() {
        return ToolCallConstraint.Spec.compile(tools());
    }

    /** Tool schemas are compiled leniently: unknown keywords are ignored, overlapping alternatives merged. */
    static void lenientSchemas() {
        String s = "{\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"integer\",\"minimum\":3},"
                + "\"x\":{\"oneOf\":[{\"type\":\"string\",\"format\":\"date\"},{\"type\":\"string\",\"pattern\":\"a+\"}]},"
                + "\"t\":{\"type\":\"array\",\"items\":[{\"type\":\"string\"}]},\"r\":{\"$ref\":\"http://x/y\"}},\"required\":[\"n\"]}";
        try {
            JsonConstraint.Schema.compile(Json.parseObject(s));
            check(false, "strict compile rejects 'minimum'");
        } catch (IllegalArgumentException e) {
            check(e.getMessage().contains("minimum"), "strict compile rejects unsupported keywords: " + e.getMessage());
        }
        JsonConstraint c = new JsonConstraint(JsonConstraint.Schema.compile(Json.parseObject(s), true), new byte[0][]);
        check(c.completesWith("{\"n\": 1, \"x\": \"zzz\", \"t\": [1, \"a\"], \"r\": {\"any\": [true]}}")
                        && !c.allowsText("{\"n\": 1.5") && !c.allowsText("{\"x\": 5") && !c.completesWith("{\"x\": \"a\"}"),
                "lenient compile: ignored keywords allow more, the rest is enforced (integer, string, required)");
    }

    /** The tool-call automaton on text: envelope, names, argument schemas, modes. */
    static void automaton() {
        ToolCallConstraint.Spec spec = spec();
        String call = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>";
        String[][] cases = {
                {"Sure, let me look.\n" + call, "true"},
                {call + "\n" + call.replace("Paris", "Rome"), "true"},
                {"<think>maybe <tool_call>{oops</think>\n\n" + call, "true"}, // a call while thinking is free text
                {"<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\", \"lines\": [1, 2],"
                        + " \"encoding\": null, \"options\": {\"b\": 2.5}}}\n</tool_call>", "true"},
                {"<tool_call>\n{\"name\": \"read\", \"arguments\": {\"anything\": [1]}}\n</tool_call>", "true"},
                {"<tool_call>\n{\"name\": \"read\", \"arguments\": {}}\n</tool_call>", "true"},
                {"<tool_call>\n{\"name\": \"get_time\"", "false"},             // unknown tool
                {"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"days\": 3}}", "false"}, // city required
                {"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": 5", "false"},   // type
                {"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"x\", \"zip\": 1", "false"}, // closed
                {"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"x\", \"unit\": \"kelvin\"", "false"},
                {"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"x\", \"days\": 2.5", "false"},
                {"<tool_call>\n{\"name\":\"get_weather\"", "false"},          // the fixed envelope
                {"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\":\"x\"", "false"}, // canonical spacing
                {"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"x\"}} trailing", "false"},
                {"<tool_call>\n{\"name\": \"read_file\", \"arguments\": []", "false"}, // arguments are an object
        };
        for (String[] c : cases) {
            ToolCallConstraint tc = new ToolCallConstraint(spec, new byte[0][], false);
            boolean ok = c[1].equals("true") ? tc.completesWith(c[0]) : tc.allowsText(c[0]);
            check(ok == c[1].equals("true"), "tool-call automaton: " + c[0].replace("\n", "\\n") + " -> " + ok);
        }
        ToolCallConstraint free = new ToolCallConstraint(spec, new byte[0][], false);
        free.acceptText("I'll check. <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Pa");
        check(!free.relaxed() && !free.canEnd(), "inside a call: strict, may not end");
        free.acceptText("ris\"}}\n</tool_call>");
        check(free.relaxed() && free.canEnd() && free.calls() == 1, "after the call: free text again, may end");
        check(new ToolCallConstraint(spec, new byte[0][], true).allowsText("<tool_call>{bad</think>")
                && new ToolCallConstraint(spec, new byte[0][], true).completesWith("thinking...</think>\n\n" + call),
                "a prompt that opened <think>: free until </think>");

        ToolCallConstraint.Spec req = spec.required(true);
        ToolCallConstraint r = new ToolCallConstraint(req, new byte[0][], false);
        check(!r.canEnd() && !r.allowsText("Sure") && !r.allowsText(" <tool_call>") && r.allowsText("<tool_call>")
                        && !r.allowsText("<think>") && r.forcedBytes(64).length > 11,
                "tool_choice required: the output starts with the call tag (forced: "
                        + new String(r.forcedBytes(64), StandardCharsets.UTF_8).replace("\n", "\\n") + ")");
        check(new ToolCallConstraint(req.thinking(true), new byte[0][], false).completesWith("<think>\nhm\n</think>\n\n" + call)
                        && !new ToolCallConstraint(req.thinking(true), new byte[0][], false).completesWith("<think>x</think>done"),
                "required with thinking: a think block, then the call");
        ToolCallConstraint named = new ToolCallConstraint(spec.only("read_file"), new byte[0][], false);
        String forced = new String(named.forcedBytes(256), StandardCharsets.UTF_8);
        check(forced.equals("<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\""),
                "named tool_choice: tag, name and the opening of the (non-empty) arguments are forced: " + forced.replace("\n", "\\n"));
        try {
            spec.only("nope");
            check(false, "unknown named tool rejected");
        } catch (IllegalArgumentException e) {
            check(true, "a named tool_choice must be one of the tools");
        }

        // response_format next to tools: either calls or the JSON answer
        JsonConstraint.Schema ans = JsonConstraint.Schema.compile(Json.parseObject(
                "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"]}"));
        ToolCallConstraint.Spec both = spec.answer(ans);
        ToolCallConstraint a = new ToolCallConstraint(both, new byte[0][], false);
        check(!a.canEnd() && a.completesWith("{\"answer\": \"42\"}") && a.completesWith(call) && a.completesWith(call + "\n" + call)
                        && !a.allowsText("Sure") && !a.allowsText(call + "text") && !a.allowsText("{\"x\": 1}"),
                "tools + response_format: calls, or an answer matching the schema");
        a.acceptText("{\"answer\": \"42\"}");
        check(a.mustEnd(), "the JSON answer ends the output");
    }

    /** The trie walk computes exactly the tokens that feeding their bytes one by one accepts. */
    static void masks(Tokenizer tok) {
        byte[][] table = JsonConstraint.tokenTable(tok);
        ToolCallConstraint.Spec spec = spec();
        String[] prefixes = {"<tool_call>", "<tool_call>\n{\"name\": \"", "<tool_call>\n{\"name\": \"read",
                "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\", \"lines\": [1",
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"",
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"unit\": \"c"};
        for (String pre : prefixes) {
            ToolCallConstraint c = new ToolCallConstraint(spec, table, false);
            c.acceptText(pre);
            ToolCallConstraint ref = c.copy();
            int allowed = 0;
            int wrong = 0;
            for (int id = 0; id < table.length; id++) { // after 300 checks allows() answers from the trie mask
                boolean viaMask = c.allows(id);
                boolean direct = table[id] != null && ref.allowsBytes(table[id]);
                wrong += viaMask != direct ? 1 : 0;
                allowed += direct ? 1 : 0;
            }
            check(wrong == 0, "trie mask equals per-token checks after '" + pre.replace("\n", "\\n") + "' ("
                    + allowed + " tokens allowed)");
        }
    }

    /** Jump-forward: the forced text, tokenized, accepted by the constraint; the last token is held back. */
    static void jumpTokens(Tokenizer tok) {
        byte[][] table = JsonConstraint.tokenTable(tok);
        ToolCallConstraint c = new ToolCallConstraint(spec(), table, false);
        check(c.jump(tok, 64).length == 0, "free text: nothing forced");
        c.accept(tok.tokenId("<tool_call>"));
        int[] j = c.jump(tok, 64);
        String text = tok.decode(j);
        check(j.length >= 2 && "\n{\"name\": \"".startsWith(text) && text.length() < "\n{\"name\": \"".length(),
                "after the tag: " + j.length + " tokens of the forced head, the last one held back: "
                        + Json.write(text) + " " + Arrays.toString(j));
        for (int t : j) {
            check(c.allows(t), "forced token " + t + " is allowed");
            c.accept(t);
        }
        c.acceptText(" \"get_weather\", \"arguments\": {\"city\": \"Paris\"}");
        int[] tail = c.jump(tok, 64);
        check(tok.decode(tail).equals("}\n</tool_call>") && tail[tail.length - 1] == tok.tokenId("</tool_call>"),
                "after the arguments: the closing text including </tool_call> (never extended) is forced: "
                        + Arrays.toString(tail));
        check(c.jump(tok, 1).length == 1, "at most max tokens");
        ToolCallConstraint n = new ToolCallConstraint(spec().only("get_weather"), table, false);
        int[] all = n.jump(tok, 64);
        check(tok.decode(all).startsWith("<tool_call>\n{\"name\": \"get_weather\", \"arguments\":")
                        && all[0] == tok.tokenId("<tool_call>"),
                "named tool_choice: the whole head is one jump: " + Json.write(tok.decode(all)));
    }

    // ------------------------------------------------------------------ a scripted fake model

    /**
     * A model that wants to write {@code script}: at every position the longest vocabulary token that continues the
     * script from the text generated so far gets logit 10, a few unrelated tokens get 12 inside tool calls
     * (the constraint must mask them), everything else small deterministic noise; once the text left the script it
     * prefers the end-of-sequence token. Logits depend on the whole cached history, so a stale or wrongly fed
     * position changes the output.
     */
    static final class ScriptBackend implements Backend {
        final Tokenizer tok;
        final byte[] script;
        final int promptLen;
        final int vocab;
        final int eos;
        final int[] kv = new int[2048];
        final int[] distract;
        final byte[][] table;
        int forwardCalls;

        ScriptBackend(Tokenizer tok, String script, int promptLen, int vocab, int eos) {
            this.tok = tok;
            this.script = script.getBytes(StandardCharsets.UTF_8);
            this.promptLen = promptLen;
            this.vocab = vocab;
            this.eos = eos;
            this.table = JsonConstraint.tokenTable(tok);
            // never valid inside a call: raw line breaks and tabs are neither JSON string content nor canonical spacing
            this.distract = new int[]{tok.encode("\n\n")[0], tok.encode(".\n")[0], tok.encode(":\n")[0],
                    tok.encode("\t\t")[0], tok.encode("\r\n")[0]};
        }

        public String name() {
            return "script";
        }

        public int contextLength() {
            return 2048;
        }

        public int maxBatch() {
            return 32;
        }

        void logitsAt(int p, float[] out, int base) {
            ByteArrayOutputStream g = new ByteArrayOutputStream();
            for (int i = promptLen; i <= p; i++) {
                g.writeBytes(table[kv[i]] == null ? new byte[0] : table[kv[i]]);
            }
            byte[] gen = g.toByteArray();
            for (int t = 0; t < vocab; t++) {
                out[base + t] = (float) (((t * 2654435761L + p * 40503L) & 1023) / 1024.0); // noise in [0, 1)
            }
            boolean onScript = gen.length <= script.length && Arrays.equals(gen, 0, gen.length, script, 0, gen.length);
            if (!onScript || gen.length == script.length) {
                out[base + eos] = 20;
                return;
            }
            int best = -1;
            int bestLen = 0;
            for (int t = 0; t < vocab && t < table.length; t++) {
                byte[] b = table[t];
                if (b == null) {
                    continue;
                }
                if (b.length > bestLen && gen.length + b.length <= script.length
                        && Arrays.equals(b, 0, b.length, script, gen.length, gen.length + b.length)) {
                    best = t;
                    bestLen = b.length;
                }
            }
            out[base + best] = 10;
            String g8 = new String(gen, StandardCharsets.UTF_8);
            boolean inCall = g8.lastIndexOf("<tool_call>") > g8.lastIndexOf("</tool_call>");
            if (inCall && p % 3 == 0) {
                out[base + distract[p % distract.length]] = 12; // a token the constraint must mask
            }
        }

        public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
            forwardCalls++;
            for (int i = 0; i < n; i++) {
                kv[pos + i] = tokens[off + i];
            }
            if (logits != null) {
                logitsAt(pos + n - 1, logits, 0);
            }
        }

        @Override
        public void forwardAll(int[] tokens, int off, int n, int pos, float[] out, int stride) {
            forwardCalls++;
            for (int i = 0; i < n; i++) {
                kv[pos + i] = tokens[off + i];
            }
            for (int i = 0; i < n; i++) {
                logitsAt(pos + i, out, i * stride);
            }
        }

        public void close() {
        }
    }

    static SamplingParams greedy(int max) {
        return new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, max, List.of());
    }

    record Run(Engine.Result r, int forwards) {
    }

    static Run fakeRun(Tokenizer tok, Architecture cfg, int[] prompt, String script, int eos, boolean spec, boolean jump,
                       Supplier<TokenConstraint> con) {
        return fakeRun(tok, cfg, prompt, script, eos, spec, jump, con, null);
    }

    /**
     * @param draftScript when not null, speculation is by a draft model that writes this script (its own scripted
     *                    backend, greedy, with no constraint: it proposes the masked distractions too)
     */
    static Run fakeRun(Tokenizer tok, Architecture cfg, int[] prompt, String script, int eos, boolean spec, boolean jump,
                       Supplier<TokenConstraint> con, String draftScript) {
        ScriptBackend b = new ScriptBackend(tok, script, prompt.length, cfg.core().vocabSize(), eos);
        Engine e = new Engine(b, tok, cfg, Set.of(eos));
        if (draftScript != null) {
            ScriptBackend db = new ScriptBackend(tok, draftScript, prompt.length, cfg.core().vocabSize(), eos);
            check(e.enableSpeculation(new Speculator.Config(Speculator.Mode.DRAFT, 8, 4, 2, 0),
                    new DraftModel(db, cfg.core().vocabSize(), null)), "draft speculation on");
        } else if (spec) {
            check(e.enableSpeculation(new Speculator.Config(Speculator.Mode.NGRAM, 8, 4, 2, 0), null), "speculation on");
        }
        e.setJumpForward(jump);
        Engine.Result r = e.generate(prompt, greedy(200), new Engine.Decoding(null, false, 0, false, con), 1, null,
                () -> false).getFirst();
        return new Run(r, b.forwardCalls);
    }

    /**
     * The scripted model under the tool-call constraint: the output is the script whatever the distracting tokens;
     * jump-forward produces the same text in fewer passes; speculation (prompt lookup, whose drafts the constraint
     * cuts) produces the identical tokens.
     */
    static void fakeEngine(Tokenizer tok) throws Exception {
        Path tmp = Files.createTempDirectory("jllm-tools");
        try {
            Architecture cfg = SmokeTest.cfg(tmp, "{\"model_type\":\"mamba\",\"hidden_size\":64,\"num_hidden_layers\":1,"
                    + "\"vocab_size\":" + tok.vocabSize() + "}");
            int eos = tok.tokenId("<|im_end|>");
            byte[][] table = JsonConstraint.tokenTable(tok);
            ToolCallConstraint.Spec spec = spec();
            Supplier<TokenConstraint> con = () -> new ToolCallConstraint(spec, table, false);
            String script = "Let me check the weather.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
                    + "{\"city\": \"Paris\", \"days\": 3, \"unit\": \"celsius\"}}\n</tool_call>\n<tool_call>\n{\"name\": "
                    + "\"read_file\", \"arguments\": {\"path\": \"src/main/java/Foo.java\", \"lines\": [1, 2, 3]}}\n</tool_call>";
            // the prompt shows an earlier call, so prompt lookup has drafts (partly for the wrong tool or arguments)
            int[] prompt = tok.encode("Earlier:\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\","
                    + " \"days\": 2}}\n</tool_call>\nNow:\n");
            Run plain = fakeRun(tok, cfg, prompt, script, eos, false, false, con);
            check(plain.r.text().equals(script) && plain.r.finishReason().equals("stop"),
                    "constrained greedy writes the script despite masked distractions: " + plain.r.tokens().length
                            + " tokens, " + plain.forwards + " passes: " + Json.write(plain.r.text()));
            Run unconstrained = fakeRun(tok, cfg, prompt, script, eos, false, false, null);
            check(!unconstrained.r.text().equals(script), "without the constraint the distractions win (sanity check)");
            Run spec1 = fakeRun(tok, cfg, prompt, script, eos, true, false, con);
            check(Arrays.equals(plain.r.tokens(), spec1.r.tokens()) && spec1.r.stats().draftedTokens() > 0,
                    "speculation under the constraint: identical tokens (" + spec1.r.stats().acceptedTokens() + "/"
                            + spec1.r.stats().draftedTokens() + " drafts accepted, " + spec1.forwards + " vs "
                            + plain.forwards + " passes)");
            // the per-token path of the masks (engine.constraint-masks=false) chooses the same tokens
            Run slow;
            JsonConstraint.setFastMasks(false);
            try {
                slow = fakeRun(tok, cfg, prompt, script, eos, false, false, con);
            } finally {
                JsonConstraint.setFastMasks(true);
            }
            check(Arrays.equals(plain.r.tokens(), slow.r.tokens()), "masks off: identical tokens");
            // draft-model speculation under the constraint (docs/17 17.7): a draft that writes the script too (it
            // proposes the distractions the constraint masks as well, so its drafts are cut), and one that writes
            // another call (rejected at the first different token); the tokens equal the plain ones,
            // with and without jump-forward
            String wrong = script.replaceFirst("get_weather", "read_file").replace("Paris", "Rome").replace("celsius", "fahrenheit");
            // (the draft length follows the doubling rule: the cost model's measured timings mean nothing on a fake)
            System.setProperty("jllm.engine.adaptiveDraft", "false");
            for (boolean j : new boolean[]{false, true}) {
                Run ref = j ? fakeRun(tok, cfg, prompt, script, eos, false, true, con) : plain;
                for (String ds : new String[]{script, wrong}) {
                    Run d = fakeRun(tok, cfg, prompt, script, eos, false, j, con, ds);
                    check(Arrays.equals(ref.r.tokens(), d.r.tokens()) && d.r.stats().draftedTokens() > 0,
                            "draft model " + (ds == script ? "writing the script" : "writing another call") + ", jump=" + j
                                    + ": identical tokens (" + d.r.stats().acceptedTokens() + "/" + d.r.stats().draftedTokens()
                                    + " drafts accepted, " + d.forwards + " vs " + ref.forwards + " passes)");
                }
            }
            System.clearProperty("jllm.engine.adaptiveDraft");
            Run jump = fakeRun(tok, cfg, prompt, script, eos, false, true, con);
            check(jump.r.text().equals(script) && jump.r.stats().forcedTokens() > 0 && jump.forwards < plain.forwards,
                    "jump-forward: the same text, " + jump.r.stats().forcedTokens() + " tokens forced, " + jump.forwards
                            + " vs " + plain.forwards + " passes");
            Run both = fakeRun(tok, cfg, prompt, script, eos, true, true, con);
            check(Arrays.equals(jump.r.tokens(), both.r.tokens()),
                    "jump-forward + speculation: identical to jump-forward alone (" + both.forwards + " passes, "
                            + both.r.stats().acceptedTokens() + "/" + both.r.stats().draftedTokens() + " drafts)");
            // forced call: the scripted model wants prose first, the constraint starts with the call
            Supplier<TokenConstraint> req = () -> new ToolCallConstraint(spec.required(true), table, false);
            Run forced = fakeRun(tok, cfg, prompt, script, eos, false, true, req);
            check(forced.r.text().startsWith("<tool_call>\n{\"name\": \"") && forced.r.stats().forcedTokens() >= 5,
                    "tool_choice required: a call although the model wanted prose, the head jump-forwarded ("
                            + forced.r.stats().forcedTokens() + " tokens): "
                            + Json.write(forced.r.text().substring(0, Math.min(60, forced.r.text().length()))));
            // a model that wants to call: required changes nothing but the forced head
            String direct = script.substring(script.indexOf("<tool_call>"));
            Run want = fakeRun(tok, cfg, prompt, direct, eos, true, true, req);
            check(want.r.text().equals(direct) && want.r.stats().forcedTokens() > 0,
                    "tool_choice required, model calling anyway: the same text, " + want.r.stats().forcedTokens()
                            + " tokens forced, " + want.forwards + " passes");
            Map<String, Object> obj = Json.parseObject(between(want.r.text()));
            check(obj.get("arguments") instanceof Map<?, ?> m && validArgs((String) obj.get("name"), m),
                    "the call parses: " + Json.write(obj));
        } finally {
            Files.deleteIfExists(tmp.resolve("config.json"));
            Files.deleteIfExists(tmp);
        }
    }

    static String between(String text) {
        int a = text.indexOf("<tool_call>");
        int b = text.indexOf("</tool_call>", a);
        return text.substring(a + "<tool_call>".length(), b).strip();
    }

    // ------------------------------------------------------------------ the real model

    /**
     * CPU threads for the real model ({@code -Djllm.test.threads}): 1 by default, since on a busy machine every
     * thread barrier of a multi-threaded matmul waits for the slowest, descheduled thread.
     */
    static final int THREADS = Integer.getInteger("jllm.test.threads", 1);

    static void realModel(Path dir, Tokenizer tok) throws Exception {
        if (!Files.exists(dir.resolve("config.json"))) {
            System.out.println("skip: no model at " + dir);
            return;
        }
        Architecture cfg = Architecture.load(dir);
        ChatTemplate.Format tpl = ChatTemplate.forModelType(cfg.core().modelType(), tok);
        // a short prompt: in deterministic mode the CPU prefills token by token
        String system = "You can call get_weather(city: string, days: integer, unit: \"celsius\" or \"fahrenheit\") by"
                + " writing:\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {...}}\n</tool_call>";
        String[] questions = {"What's the weather in Paris for the next 3 days, in celsius?"};
        byte[][] table = JsonConstraint.tokenTable(tok);
        ToolCallConstraint.Spec spec = spec();
        boolean det = Boolean.getBoolean("jllm.cpu.deterministic");
        // int8 activations: in deterministic mode every token goes through the same single-token int8 kernel either
        // way; the float Q8 kernel took C2 minutes to compile on a loaded machine (docs/07, "What the JIT needs from a Vector API kernel")
        try (ModelWeights w = ModelWeights.load(dir, cfg, QType.Q8_0, QuantFile.DEFAULT_HEAD, ShardProvider.local(dir), null);
             Engine plain = new Engine(new CpuBackend(w, 1024, 32, THREADS, KvDtype.F32, true), tok, cfg, Set.of());
             Engine specE = new Engine(new CpuBackend(w, 1024, 32, THREADS, KvDtype.F32, true), tok, cfg, Set.of())) {
            check(specE.enableSpeculation(new Speculator.Config(Speculator.Mode.NGRAM, 8, 4, 2, 0), null), "speculation on");
            for (String q : questions) {
                int[] prompt = plain.renderChat(List.of(new ChatTemplate.Message("system", system),
                        new ChatTemplate.Message("user", q)), false);
                System.out.println("    prompt: " + prompt.length + " tokens");
                for (boolean required : new boolean[]{false, true}) { // (auto: with jump-forward only, to save time)
                    ToolCallConstraint.Spec s = spec.required(required);
                    Supplier<TokenConstraint> con = () -> new ToolCallConstraint(s, table, false);
                    Engine.Decoding d = new Engine.Decoding(null, false, 0, false, con);
                    String[] texts = new String[4];
                    int[][] toks = new int[4][];
                    int i = 0;
                    for (boolean jump : required ? new boolean[]{false, true} : new boolean[]{true}) {
                        plain.setJumpForward(jump);
                        specE.setJumpForward(jump);
                        Engine.Result a = plain.generate(prompt, greedy(40), d, 1, null, () -> false).getFirst();
                        specE.adoptPrefix(plain, prompt); // the speculating engine skips the prefill
                        Engine.Result b = specE.generate(prompt, greedy(40), d, 1, null, () -> false).getFirst();
                        texts[i] = a.text();
                        toks[i++] = a.tokens();
                        texts[i] = b.text();
                        toks[i++] = b.tokens();
                        System.out.printf("    %s required=%b jump=%b: %d tokens (%d forced) %.0f ms | speculative %d/%d"
                                        + " drafts %.0f ms%n", q.substring(0, 20), required, jump, a.tokens().length,
                                a.stats().forcedTokens(), a.stats().decodeMs(), b.stats().acceptedTokens(),
                                b.stats().draftedTokens(), b.stats().decodeMs());
                        check(!det || Arrays.equals(a.tokens(), b.tokens()), "Qwen3-0.6B, required=" + required + ", jump="
                                + jump + ": speculative constrained greedy output identical to plain constrained greedy"
                                + (det ? "" : " (not checked: needs -Djllm.cpu.deterministic=true)"));
                        String call = a.text().contains("<tool_call>") ? between(a.text() + "</tool_call>") : null;
                        check(!required || call != null, "required: a call was made");
                        if (call != null && a.text().contains("</tool_call>")) {
                            Map<String, Object> obj = Json.parseObject(call);
                            check(obj.get("arguments") instanceof Map<?, ?> m && validArgs(String.valueOf(obj.get("name")), m),
                                    "the call parses and its arguments match the schema: " + Json.write(obj));
                        }
                    }
                    System.out.println("    output: " + Json.write(texts[0]) + (texts[2] == null ? ""
                            : "\n    with jump-forward: " + Json.write(texts[2])));
                }
            }
        }
    }

    /** The arguments against the two schemas (the subset the constraint enforces). */
    static boolean validArgs(String name, Map<?, ?> a) {
        return switch (name) {
            case "get_weather" -> a.get("city") instanceof String && Set.of("city", "days", "unit").containsAll(a.keySet())
                    && (a.get("days") == null || a.get("days") instanceof Long)
                    && (a.get("unit") == null || List.of("celsius", "fahrenheit").contains(a.get("unit")));
            case "read_file" -> a.get("path") instanceof String
                    && (a.get("lines") == null || a.get("lines") instanceof List<?> l && l.stream().allMatch(x -> x instanceof Long));
            default -> false;
        };
    }
}
