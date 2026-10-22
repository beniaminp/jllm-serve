package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.ChatTemplate;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Speculative decoding on agent-like prompts instead of the repeating bench prompt (docs/10 §10.9, docs/11 §11.11): the three
 * kinds of turn a coding agent produces, each sent as a chat conversation and generated greedily, with the speculation
 * mode the settings name ({@code --engine.speculative off|ngram|draft}). One line per scenario with the median decode
 * speed of the timed rounds, the drafted / accepted counts and the md5 of the text (greedy output must be the same in
 * every mode).
 *
 * <ul>
 *   <li>{@code edit}: a source file is resent with the request to write it out again with a change (the answer copies
 *       most of the prompt);
 *   <li>{@code tool}: a tool-calling system prompt with JSON schemas, asked for one call (a short JSON answer whose keys
 *       and values come from the prompt);
 *   <li>{@code prose}: a question answered in free text (nothing to copy).
 * </ul>
 *
 * <pre>java ... org.robomiri.jllm.SpecAgentBench [--rounds 3] [--max-tokens 192] [--file src/.../Foo.java] [settings...]</pre>
 * The model comes from the settings ({@code --model Qwen/Qwen3-0.6B}); {@code --ctx} defaults to 6144, {@code --temp} to 0.
 */
public final class SpecAgentBench {

    private static final String TOOLS = """
            # Tools

            You may call one or more functions to assist with the user query.

            You are provided with function signatures within <tools></tools> XML tags:
            <tools>
            {"type": "function", "function": {"name": "read_file", "description": "Read lines of a file in the workspace", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "path relative to the workspace root"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"]}}}
            {"type": "function", "function": {"name": "search_code", "description": "Search the workspace for a regular expression", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "glob": {"type": "string", "description": "only files matching this glob"}, "max_results": {"type": "integer"}}, "required": ["pattern"]}}}
            {"type": "function", "function": {"name": "run_tests", "description": "Run the project's tests", "parameters": {"type": "object", "properties": {"module": {"type": "string"}, "test_class": {"type": "string"}}, "required": []}}}
            </tools>

            For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
            <tool_call>
            {"name": <function-name>, "arguments": <args-json-object>}
            </tool_call>""";

    public static void main(String[] args) throws Exception {
        int rounds = 3;
        int maxTokens = 192;
        Path file = Path.of("src/main/java/org/robomiri/jllm/engine/NgramIndex.java");
        List<String> rest = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--rounds" -> rounds = Integer.parseInt(args[++i]);
                case "--max-tokens" -> maxTokens = Integer.parseInt(args[++i]);
                case "--file" -> file = Path.of(args[++i]);
                default -> rest.add(args[i]);
            }
        }
        if (!rest.contains("--ctx") && !rest.contains("--backend.ctx")) {
            rest.add("--ctx");
            rest.add("6144");
        }
        if (!rest.contains("--temp") && !rest.contains("--temperature")) { // greedy: every mode must write the same text
            rest.add("--temp");
            rest.add("0");
        }
        Settings st = Settings.load(rest.toArray(new String[0]));
        String source = Files.readString(file);

        List<String> names = List.of("edit", "tool", "prose");
        List<List<ChatTemplate.Message>> conversations = List.of(
                List.of(new ChatTemplate.Message("system", "You are a coding agent working in the user's repository. Answer with code only."),
                        new ChatTemplate.Message("user", "Here is " + file.getFileName() + ":\n\n```java\n" + source + "```\n\n"
                                + "Rename the field `used` to `entries` everywhere and write the whole file out again.")),
                List.of(new ChatTemplate.Message("system", TOOLS),
                        new ChatTemplate.Message("user", "Open src/main/java/org/robomiri/jllm/engine/Speculator.java from line 120 to line 200, "
                                + "I want to see how the draft is verified.")),
                List.of(new ChatTemplate.Message("user", "Explain how a hash map resolves collisions, and compare open addressing with "
                        + "chaining: when would you pick each, and what does resizing cost? Answer in about 250 words.")));

        try (ModelLoader.Loaded l = ModelLoader.load(st, ModelLoader.hub(st), st.get("model.repo"))) {
            Engine engine = l.engine();
            SamplingParams p = st.sampling().withMaxTokens(maxTokens);
            System.out.printf("SPEC model=%s speculative=%s backend=%s ctx=%d rounds=%d max-tokens=%d%n", st.get("model.repo"),
                    st.get("engine.speculative"), engine.backend().name(), engine.backend().contextLength(), rounds, maxTokens);
            for (int c = 0; c < names.size(); c++) {
                int[] prompt = engine.renderChat(conversations.get(c), false);
                engine.generate(prompt, p, null, () -> false); // warm-up: pipelines, caches, the n-gram index
                double[] rate = new double[rounds];
                Engine.Result last = null;
                for (int r = 0; r < rounds; r++) {
                    Engine.Result res = engine.generate(prompt, p, null, () -> false);
                    rate[r] = res.stats().tokPerSec();
                    last = res;
                }
                Arrays.sort(rate);
                Engine.Stats s = last.stats();
                System.out.printf("SPEC scenario=%s prompt=%d generated=%d decode_tok_s=%.1f (min %.1f max %.1f) drafted=%d accepted=%d "
                                + "md5=%s%n", names.get(c), s.promptTokens(), s.generatedTokens(), rate[rounds / 2], rate[0],
                        rate[rounds - 1], s.draftedTokens(), s.acceptedTokens(), md5(last.text()));
            }
        }
    }

    private static String md5(String s) throws Exception {
        byte[] d = MessageDigest.getInstance("MD5").digest(s.getBytes(StandardCharsets.UTF_8));
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 4; i++) {
            sb.append(String.format("%02x", d[i]));
        }
        return sb.toString();
    }
}
