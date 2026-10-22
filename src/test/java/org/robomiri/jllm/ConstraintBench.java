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
 * the full vocabulary is searched). A fourth row, "unconstrained", is the floor: the same sampling without a
 * constraint.
 *
 * <p>Two modes. {@code --quick} (run by {@code mvn install}, a few seconds) checks the answers instead of timing
 * them: every sampled token is allowed by the automaton, a greedy token is the most likely allowed token (found by
 * brute force over the vocabulary, feeding each token's bytes to the automaton, independent of the masks), the
 * reference text completes the constraint, and a mask costs a few hundred trie nodes, not the vocabulary. The long
 * mode (the default, by hand) prints microseconds per token:
 * {@code java --add-modules jdk.incubator.vector -cp target/classes:target/test-classes
 * org.robomiri.jllm.ConstraintBench models/Qwen_Qwen3-0.6B [rounds]}, with {@code -Dwarm=true} for rounds that share
 * the mask memo (as the requests of an agent with one tool set do).
 */
public final class ConstraintBench {

    static final String SCHEMA = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},"
            + "\"content\":{\"type\":\"string\"},\"line\":{\"type\":\"integer\"},\"overwrite\":{\"type\":\"boolean\"},"
            + "\"mode\":{\"enum\":[\"append\",\"replace\"]}},\"required\":[\"path\",\"content\"],\"additionalProperties\":false}";
    static final String TEXT = "{\"path\": \"src/main/java/org/example/Foo.java\", \"content\": \"public class Foo {\\n"
            + "    private final int x = 1;\\n\\n    int x() {\\n        return x;\\n    }\\n}\\n\", \"line\": 42,"
            + " \"overwrite\": true, \"mode\": \"replace\"}";

    /** {@code -Dwarm=true}: the rounds share a mask cache (as the requests of one schema or tool set do). */
    static final boolean WARM = Boolean.getBoolean("warm");

    public static void main(String[] args) throws Exception {
        boolean quick = false;
        String model = "models/Qwen_Qwen3-0.6B";
        int rounds = 20;
        int positional = 0;
        for (String a : args) {
            if (a.equals("--quick")) {
                quick = true;
            } else if (positional++ == 0) {
                model = a;
            } else {
                rounds = Integer.parseInt(a);
            }
        }
        if (quick) {
            rounds = Math.min(rounds, 2);
        }
        Tokenizer tok = Tokenizer.load(Path.of(model).resolve("tokenizer.json"));
        JsonConstraint.Schema schema = JsonConstraint.Schema.compile(Json.parseObject(SCHEMA));
        byte[][] table = JsonConstraint.tokenTable(tok);
        int[] ref = tok.encode(TEXT);
        int vocab = tok.vocabSize();
        System.out.println("reference: " + ref.length + " tokens, vocabulary " + vocab
                + (quick ? " (quick: answers checked)" : ""));
        long masks0 = JsonConstraint.maskStats()[0];
        long nodes0 = JsonConstraint.maskStats()[1];
        for (String shape : new String[]{"unconstrained", "agree", "distracted", "disagree"}) {
            for (float temp : new float[]{0f, 0.7f}) {
                long best = Long.MAX_VALUE;
                Object cache = WARM ? JsonConstraint.newMaskCache() : null; // shared by the rounds: masks memoized
                for (int r = 0; r < rounds; r++) {
                    best = Math.min(best, run(shape, temp, schema, table, ref, vocab, r, cache, quick));
                }
                long[] st = JsonConstraint.maskStats();
                System.out.printf("%-13s %s: %7.1f us/token (best of %d; %d trie masks so far, %d nodes)%n", shape,
                        temp == 0 ? "greedy " : "T=0.7  ", best / 1000.0 / ref.length, rounds, st[0], st[1]);
            }
        }
        if (quick) {
            long masks = JsonConstraint.maskStats()[0] - masks0;
            long nodes = JsonConstraint.maskStats()[1] - nodes0;
            if (masks == 0) {
                throw new AssertionError("the disagreeing logits never needed a trie mask");
            }
            if (nodes / masks > 5000) { // ~780 on the Qwen3 vocabulary, which has 151 669 tokens
                throw new AssertionError("a trie mask visits " + nodes / masks + " nodes on average, expected a few hundred");
            }
            System.out.println("constraint bench passed (" + masks + " trie masks, " + nodes / masks + " nodes each)");
        }
    }

    /**
     * @param shape one of unconstrained, agree, distracted, disagree
     * @param check verify the answers (slow: a brute-force oracle over the vocabulary every few steps)
     * @return nanoseconds spent sampling, for the whole reference
     */
    static long run(String shape, float temp, JsonConstraint.Schema schema, byte[][] table, int[] ref, int vocab,
                    int round, Object cache, boolean check) {
        SplittableRandom rnd = new SplittableRandom(round * 31L + 7);
        SamplingParams p = new SamplingParams(temp, 0, temp == 0 ? 1f : 0.95f, 0f, 1f, 0f, 0f, 1, 1000, List.of());
        Sampler s = new Sampler(p, vocab, new int[0]);
        boolean constrained = !shape.equals("unconstrained");
        JsonConstraint c = new JsonConstraint(schema, table, cache);
        float[] logits = new float[vocab];
        float[] original = check ? new float[vocab] : null;
        long ns = 0;
        int step = 0;
        for (int t : ref) {
            for (int i = 0; i < vocab; i++) {
                logits[i] = (float) (rnd.nextGaussian() * 2);
            }
            switch (shape) {
                case "distracted" -> {
                    for (int k = 0; k < 40; k++) {
                        logits[rnd.nextInt(vocab)] = 15 + (float) rnd.nextDouble() * 4;
                    }
                    logits[t] = 14;
                }
                case "disagree" -> logits[t] += 3;
                default -> logits[t] = 20; // agree (and the unconstrained floor)
            }
            if (check) {
                System.arraycopy(logits, 0, original, 0, vocab);
            }
            long t0 = System.nanoTime();
            int got = constrained ? s.sample(logits, c, id -> false) // the engine's path (no end-of-sequence id here)
                    : s.sample(logits);
            ns += System.nanoTime() - t0;
            if (got < 0) {
                throw new IllegalStateException("nothing allowed");
            }
            if (check && constrained) {
                if (!c.allowsBytes(table[got])) {
                    throw new AssertionError("step " + step + ": sampled token " + got + " is not allowed");
                }
                if (temp == 0 && step % 4 == 0) { // greedy: the most likely allowed token, by brute force
                    int best = -1;
                    for (int id = 0; id < vocab; id++) {
                        if (table[id] != null && (best < 0 || original[id] > original[best]) && c.allowsBytes(table[id])) {
                            best = id;
                        }
                    }
                    if (original[got] != original[best]) {
                        throw new AssertionError("step " + step + " (" + shape + "): greedy picked " + got + " ("
                                + original[got] + "), the most likely allowed token is " + best + " (" + original[best] + ")");
                    }
                }
            }
            c.accept(t);
            step++;
        }
        if (check && !c.canEnd()) {
            throw new AssertionError("the reference text does not complete the constraint");
        }
        return ns;
    }
}
