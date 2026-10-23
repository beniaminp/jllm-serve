package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.sampling.SamplingParams;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * A measuring program for docs/26 §26.5: loads a model with the settings given on the command line, then grows one
 * conversation through the given context lengths, and after each stage prints the prefill and decode speed and the memory
 * report (experts on the CPU, the KV tiers), so tiered and resident runs can be put side by side.
 *
 * <p>usage: TierStudy &lt;model dir&gt; &lt;lengths, e.g. 2048,8192,16384&gt; [settings: --key value ...]
 */
public final class TierStudy {

    private static final SamplingParams GREEDY = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 32, List.of());

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]);
        int[] lengths = Arrays.stream(args[1].split(",")).mapToInt(Integer::parseInt).toArray();
        List<String> a = new ArrayList<>(List.of("--model.dir", dir.toAbsolutePath().getParent().toString(), "--engine.speculative", "off"));
        a.addAll(Arrays.asList(args).subList(2, args.length));
        Settings st = Settings.load(a.toArray(new String[0]));
        long t0 = System.nanoTime();
        try (ModelLoader.Loaded l = ModelLoader.load(st, ModelLoader.hub(st), dir.getFileName().toString(), 1)) {
            Engine e = l.engine();
            System.out.printf("loaded %s in %.1f s%n", e.backend().name(), (System.nanoTime() - t0) / 1e9);
            StringBuilder sb = new StringBuilder();
            for (int i = 0; sb.length() < lengths[lengths.length - 1] * 9; i++) {
                sb.append("item ").append(i).append(" of the list, ");
            }
            int[] all = e.tokenizer().encode(sb.toString());
            System.out.println("context | prefill tok/s (new tokens) | decode tok/s | experts on the CPU (GB) | KV hot cap / committed (blocks) | KV mode");
            for (int len : lengths) {
                int[] prompt = Arrays.copyOf(all, len);
                Engine.Result r = e.generate(prompt, GREEDY, null, () -> false);
                Engine.Stats s = r.stats();
                int fresh = s.promptTokens() - s.cachedTokens();
                Map<String, Object> rep = e.backend().memoryReport();
                double cpuGb = 0;
                String hot = "-";
                String mode = "-";
                if (rep != null) {
                    Map<String, Object> ex = (Map<String, Object>) rep.get("experts");
                    cpuGb = ex == null ? 0 : ((Number) ex.get("onCpuBytes")).doubleValue() / 1e9;
                    Map<String, Object> kv = (Map<String, Object>) rep.get("kv");
                    if (kv != null) {
                        mode = String.valueOf(kv.get("mode"));
                        Map<String, Object> h = (Map<String, Object>) kv.get("hot");
                        if (h != null) {
                            hot = h.get("capBlocks") + " / " + h.get("committedBlocks");
                        }
                    }
                }
                System.out.printf("%d | %.0f (%d) | %.1f | %.2f | %s | %s%n", len, fresh / (s.prefillMs() / 1000), fresh,
                        s.generatedTokens() / (s.decodeMs() / 1000), cpuGb, hot, mode);
            }
            Map<String, Object> rep = e.backend().memoryReport();
            if (rep != null && rep.get("plan") instanceof Map<?, ?> p) {
                System.out.println("plan: " + p);
            }
        }
    }
}
