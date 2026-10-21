package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.hub.HuggingFaceHub;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.sampling.JsonConstraint;
import org.robomiri.jllm.sampling.TokenConstraint;
import org.robomiri.jllm.sampling.ToolCallConstraint;
import org.robomiri.jllm.tokenizer.ChatTemplate;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Real-model check that speculation does not change constrained greedy output (docs/17 §17.7): loads the model once
 * per speculation setting (off, prompt lookup, and each draft model given), runs tool-call requests under the
 * tool-call constraint (tool_choice auto and required, jump-forward on and off) and compares the output tokens with
 * the speculation-off run. Exits 1 on any difference. Not run by {@code mvn install} (it needs the GPU for the Metal
 * rows and takes minutes; the CPU rows with a deterministic kernel are in {@link ToolCallTest}); by hand:
 *
 * <pre>
 * java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED [-Djllm.cpu.deterministic=true] \
 *   -cp target/classes:target/test-classes org.robomiri.jllm.ToolCallCheck --backend metal \
 *   --dir models --model Qwen/Qwen3-0.6B [--check-draft Qwen/Qwen3-0.6B,hf-internal-testing/tiny-random-Qwen3MoeForCausalLM] \
 *   [--check-max 64] [--check-rounds 1] [--check-table]
 * </pre>
 *
 * {@code --check-rounds n} repeats every generation and reports the median decode time; {@code --check-table} prints
 * markdown rows (for {@code scripts/mac-validate.sh}) instead of the aligned lines.
 *
 * Every other flag is an ordinary jllm setting (the model, the backend, the quantization). {@code scripts/tool-call-check.sh}
 * wraps it for CPU and Metal. A self-draft accepts every draft (it exercises long accepted runs, cuts by the
 * constraint, and rollback after the cut); a random tiny model with the same tokenizer is wrong nearly every time
 * (rejections at the first position).
 */
public final class ToolCallCheck {

    record Scenario(String name, String system, String user) {
    }

    static final List<Scenario> SCENARIOS = List.of(
            new Scenario("weather",
                    "You can call get_weather(city: string, days: integer, unit: \"celsius\" or \"fahrenheit\") by writing:\n"
                            + "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {...}}\n</tool_call>",
                    "What's the weather in Paris for the next 3 days, in celsius?"),
            // earlier calls in the context: prompt lookup has text to copy (partly for the wrong tool or arguments)
            new Scenario("read_file",
                    "Tools: get_weather(city, days, unit), read_file(path, lines). Example calls:\n"
                            + "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/main/java/Foo.java\", "
                            + "\"lines\": [1, 2, 3]}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
                            + "{\"city\": \"Rome\", \"days\": 2, \"unit\": \"celsius\"}}\n</tool_call>",
                    "Show me lines 1 to 5 of src/main/java/Bar.java"));

    public static void main(String[] args) throws Exception {
        List<String> rest = new ArrayList<>();
        List<String> drafts = new ArrayList<>();
        int max = 64;
        int rounds = 1;
        boolean tableOut = false;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--check-draft" -> drafts.addAll(Arrays.asList(args[++i].split(",")));
                case "--check-max" -> max = Integer.parseInt(args[++i]);
                case "--check-rounds" -> rounds = Integer.parseInt(args[++i]);
                case "--check-table" -> tableOut = true;
                default -> rest.add(args[i]);
            }
        }
        // (label, settings for the speculation mode)
        Map<String, List<String>> modes = new LinkedHashMap<>();
        modes.put("off", List.of("--engine.speculative", "off"));
        modes.put("ngram", List.of("--engine.speculative", "ngram"));
        for (String d : drafts) {
            modes.put("draft " + d, List.of("--engine.speculative", "draft", "--engine.draft-model", d,
                    "--engine.draft-tokens", "8"));
        }
        // key "scenario/required/jump" -> output tokens, per mode
        Map<String, int[]> reference = new LinkedHashMap<>();
        int bad = 0;
        for (Map.Entry<String, List<String>> mode : modes.entrySet()) {
            List<String> a = new ArrayList<>(List.of("run"));
            a.addAll(rest);
            a.addAll(mode.getValue());
            Settings s = Settings.load(a.toArray(new String[0])).validate();
            HuggingFaceHub hub = ModelLoader.hub(s);
            try (ModelLoader.Loaded l = ModelLoader.load(s, hub, s.get("model.repo"))) {
                Engine engine = l.engine();
                byte[][] table = JsonConstraint.tokenTable(engine.tokenizer());
                ToolCallConstraint.Spec spec = ToolCallTest.spec();
                for (Scenario sc : SCENARIOS) {
                    int[] prompt = engine.renderChat(List.of(new ChatTemplate.Message("system", sc.system()),
                            new ChatTemplate.Message("user", sc.user())), false);
                    for (boolean required : new boolean[]{false, true}) {
                        ToolCallConstraint.Spec sp = spec.required(required);
                        Supplier<TokenConstraint> con = () -> new ToolCallConstraint(sp, table, false);
                        Engine.Decoding d = new Engine.Decoding(null, false, 0, false, con);
                        for (boolean jump : new boolean[]{false, true}) {
                            engine.setJumpForward(jump);
                            Engine.Result r = null;
                            double[] ms = new double[rounds];
                            for (int k = 0; k < rounds; k++) {
                                r = engine.generate(prompt, ToolCallTest.greedy(max), d, 1, null, () -> false).getFirst();
                                ms[k] = r.stats().decodeMs();
                            }
                            Arrays.sort(ms);
                            double median = ms[rounds / 2];
                            String key = sc.name() + "/required=" + required + "/jump=" + jump;
                            String verdict;
                            if (mode.getKey().equals("off")) {
                                reference.put(key, r.tokens());
                                verdict = "reference";
                            } else if (Arrays.equals(reference.get(key), r.tokens())) {
                                verdict = "identical";
                            } else {
                                verdict = "DIFFERENT";
                                bad++;
                            }
                            if (tableOut) {
                                System.out.printf("| %s | %s | %d (%d forced) | %d/%d | %.0f | %s |%n", mode.getKey(), key,
                                        r.tokens().length, r.stats().forcedTokens(), r.stats().acceptedTokens(),
                                        r.stats().draftedTokens(), median, verdict);
                            } else {
                                System.out.printf("%-34s %-26s %-9s %3d tokens (%2d forced) drafts %2d/%-2d %6.0f ms  %s%n",
                                        mode.getKey(), key, verdict, r.tokens().length, r.stats().forcedTokens(),
                                        r.stats().acceptedTokens(), r.stats().draftedTokens(), median,
                                        verdict.equals("DIFFERENT") ? Json.write(r.text()) + " vs the reference" : "");
                            }
                        }
                    }
                }
            }
        }
        if (bad > 0) {
            System.out.println(bad + " constrained greedy outputs differ with speculation on");
            System.exit(1);
        }
        if (!tableOut) {
            System.out.println("tool call check passed: constrained greedy output identical with and without speculation");
        }
    }
}
