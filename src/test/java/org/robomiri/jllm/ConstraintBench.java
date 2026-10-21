package org.robomiri.jllm;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.sampling.JsonConstraint;
import org.robomiri.jllm.sampling.Sampler;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.nio.file.Path;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Per-token cost of constrained sampling (the JSON mask) with a real vocabulary and synthetic logits, teacher-forced
 * along a reference tool-call argument object: every step samples under the constraint (timed), then the reference
 * token is accepted so that all runs walk the same automaton states. Three logit shapes: the model agrees (the
 * reference token is the most likely), it is distracted (40 other tokens are more likely), and it disagrees (the
 * reference token is buried in noise, so in structural positions none of the 256 most likely tokens is allowed and
 * the full vocabulary is searched). Not a test: run by hand,
 * {@code java -cp target/classes:target/test-classes org.robomiri.jllm.ConstraintBench models/Qwen_Qwen3-0.6B}.
 */
public final class ConstraintBench {

    static final String SCHEMA = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},"
            + "\"content\":{\"type\":\"string\"},\"line\":{\"type\":\"integer\"},\"overwrite\":{\"type\":\"boolean\"},"
            + "\"mode\":{\"enum\":[\"append\",\"replace\"]}},\"required\":[\"path\",\"content\"],\"additionalProperties\":false}";
    static final String TEXT = "{\"path\": \"src/main/java/org/example/Foo.java\", \"content\": \"public class Foo {\\n"
            + "    private final int x = 1;\\n\\n    int x() {\\n        return x;\\n    }\\n}\\n\", \"line\": 42,"
            + " \"overwrite\": true, \"mode\": \"replace\"}";

    public static void main(String[] args) throws Exception {
        Tokenizer tok = Tokenizer.load(Path.of(args.length > 0 ? args[0] : "models/Qwen_Qwen3-0.6B").resolve("tokenizer.json"));
        int rounds = args.length > 1 ? Integer.parseInt(args[1]) : 20;
        JsonConstraint.Schema schema = JsonConstraint.Schema.compile(Json.parseObject(SCHEMA));
        byte[][] table = JsonConstraint.tokenTable(tok);
        int[] ref = tok.encode(TEXT);
        int vocab = tok.vocabSize();
        System.out.println("reference: " + ref.length + " tokens, vocabulary " + vocab);
        for (String shape : new String[]{"agree", "distracted", "disagree"}) {
            for (float temp : new float[]{0f, 0.7f}) {
                long best = Long.MAX_VALUE;
                Object cache = WARM ? JsonConstraint.newMaskCache() : null; // shared by the rounds: masks memoized
                for (int r = 0; r < rounds; r++) {
                    best = Math.min(best, run(shape, temp, schema, table, ref, vocab, r, cache));
                }
                long[] st = JsonConstraint.maskStats();
                System.out.printf("%-10s %s: %7.1f us/token (best of %d; %d trie masks so far, %d nodes)%n", shape,
                        temp == 0 ? "greedy " : "T=0.7  ", best / 1000.0 / ref.length, rounds, st[0], st[1]);
            }
        }
    }

    /** @return nanoseconds spent sampling, for the whole reference */
    /** {@code -Dwarm=true}: the rounds share a mask cache (as the requests of one schema or tool set do). */
    static final boolean WARM = Boolean.getBoolean("warm");

    static long run(String shape, float temp, JsonConstraint.Schema schema, byte[][] table, int[] ref, int vocab, int round,
                    Object cache) {
        SplittableRandom rnd = new SplittableRandom(round * 31L + 7);
        SamplingParams p = new SamplingParams(temp, 0, temp == 0 ? 1f : 0.95f, 0f, 1f, 0f, 0f, 1, 1000, List.of());
        Sampler s = new Sampler(p, vocab, new int[0]);
        JsonConstraint c = new JsonConstraint(schema, table, cache);
        float[] logits = new float[vocab];
        long ns = 0;
        for (int t : ref) {
            for (int i = 0; i < vocab; i++) {
                logits[i] = (float) (rnd.nextGaussian() * 2);
            }
            switch (shape) {
                case "agree" -> logits[t] = 20;
                case "distracted" -> {
                    for (int k = 0; k < 40; k++) {
                        logits[rnd.nextInt(vocab)] = 15 + (float) rnd.nextDouble() * 4;
                    }
                    logits[t] = 14;
                }
                default -> logits[t] += 3;
            }
            long t0 = System.nanoTime();
            int got = s.sample(logits, c, id -> false); // the engine's path (no end-of-sequence id here)
            ns += System.nanoTime() - t0;
            if (got < 0) {
                throw new IllegalStateException("nothing allowed");
            }
            c.accept(t);
        }
        return ns;
    }
}
