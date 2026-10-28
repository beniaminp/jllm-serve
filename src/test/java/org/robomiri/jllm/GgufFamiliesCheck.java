package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.cpu.MambaCpuBackend;
import org.robomiri.jllm.gguf.GgufModel;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.MambaArch;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.ShardProvider;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * GGUF files of every model family jllm runs from safetensors, written the way llama.cpp's converter writes them
 * ({@link HfToGguf}) from the tiny random checkpoints of {@code scripts/make-tiny-model.py}, loaded, and compared:
 * <ol>
 *   <li><b>jllm, safetensors vs GGUF</b>: the same weights loaded from the checkpoint (converted to Q8_0) and from the
 *       F32 GGUF (quantized to Q8_0 at load) must give the same logits (CPU backend, f32 cache): this proves the
 *       metadata mapping, the tensor names and layouts, the permutations and the norm conventions of every family;</li>
 *   <li><b>the Q8_0 GGUF read in place</b> against the F32 file holding its dequantized values;</li>
 *   <li><b>llama.cpp</b> ({@code llama-debug --save-logits}, when installed) on the F32 file of the Q8_0-rounded
 *       weights against jllm on the same file: this proves the writer agrees with llama.cpp's reader, not just
 *       with jllm's. The prompt is tokenized by llama.cpp; both engines run the same ids.</li>
 * </ol>
 * usage: {@code GgufFamiliesCheck [--llama-cpp] [--keep DIR] [name...]}; run from the repository root (it calls
 * {@code python3 scripts/make-tiny-model.py}; without python3 the check is skipped).
 */
public final class GgufFamiliesCheck {

    /**
     * One tiny model: a name, the make-tiny-model.py family and arguments (config overrides that make the model one
     * llama.cpp's converter can express), the tolerance of the safetensors comparison, and whether llama.cpp is
     * expected to run it identically.
     */
    record Model(String name, double tol, double llamaCppTol, String... args) {
    }

    static final String YARN_EQ = "rope_scaling={\"type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":64,"
            + "\"beta_fast\":32,\"beta_slow\":1,\"mscale\":0.707,\"mscale_all_dim\":0.707}";

    static final List<Model> MODELS = List.of(
            new Model("llama", 2e-5, 2e-3, "llama"),
            new Model("llama-bias", 2e-5, 2e-3, "llama", "--attention-bias"),
            new Model("mistral3", 2e-5, 2e-3, "llama", "--gguf-arch", "mistral3"),
            new Model("llama-fused-qkv", 2e-5, 2e-3, "llama", "--fuse-qkv"),
            new Model("qwen2-fused-qkv", 2e-5, 2e-3, "qwen2", "--set", "use_sliding_window=false", "--fuse-qkv"),
            new Model("llama3rope", 2e-5, 2e-3, "llama3rope"),
            new Model("mistral", 2e-5, 2e-3, "mistral", "--set", "sliding_window=null"),
            new Model("qwen2", 2e-5, 2e-3, "qwen2", "--set", "use_sliding_window=false"),
            new Model("qwen3moe", 2e-5, 2e-3, "qwen3moeall"),
            new Model("gemma2", 2e-5, 2e-3, "gemma2", "--set", "query_pre_attn_scalar=64"),
            new Model("gemma3", 2e-5, 2e-3, "gemma3", "--set", "query_pre_attn_scalar=32"),
            new Model("gemma3mm", 2e-5, 2e-3, "gemma3mm", "--set", "query_pre_attn_scalar=32"),
            new Model("phi3-long", 2e-5, 2e-3, "phi3nowin"),
            new Model("phi3-short", 2e-5, 2e-3, "phi3short", "--set", "original_max_position_embeddings=256"),
            new Model("phi3-plain", 2e-5, 2e-3, "phi3mha", "--set", "sliding_window=null"),
            new Model("glm", 2e-5, 2e-3, "glm", "--set", "rms_norm_eps=1e-5"),
            new Model("glm4", 2e-5, 2e-3, "glm4"),
            new Model("mamba", 2e-5, 2e-3, "mamba"),
            new Model("mamba3", 2e-5, 2e-3, "mamba3"),
            // DeepSeek-V2-Lite sized in layers (27: llama.cpp takes fewer layers for the full model with a q_lora_rank), no
            // query low-rank; one expert group (llama.cpp scores groups as V3 does, V2's group_limited_greedy differs)
            new Model("deepseek-v2", 2e-5, 2e-3, "deepseek_v2", "--set", "num_hidden_layers=27", "--set", "n_group=1",
                    "--set", "topk_group=1", "--set", "topk_method=\"greedy\""),
            // V2 with the low-rank query, mscale == mscale_all_dim (all the GGUF keeps), normalized weights and no scale
            new Model("deepseek-v2q", 2e-5, 2e-3, "deepseek_v2q", "--set", "n_group=1", "--set", "topk_group=1",
                    "--set", "topk_method=\"greedy\"", "--set", "routed_scaling_factor=1.0", "--set", YARN_EQ),
            // V2-236B's router: weights not normalized, scaled
            new Model("deepseek-v2-scaled", 2e-5, 2e-3, "deepseek_v2q", "--set", "n_group=1", "--set", "topk_group=1",
                    "--set", "topk_method=\"greedy\"", "--set", "norm_topk_prob=false", "--set", "routed_scaling_factor=2.5",
                    "--set", YARN_EQ),
            new Model("deepseek-v3", 2e-5, 2e-3, "deepseek_v3"),
            new Model("deepseek-v3-fused-exps", 2e-5, 2e-3, "deepseek_v3", "--fuse-gate-up-exps"),
            // gpt-oss: llama.cpp's graph clamps the SwiGLU at 7.0 whatever the config says
            new Model("gpt-oss", 2e-5, 2e-3, "gptoss", "--set", "swiglu_limit=7.0"),
            new Model("gpt-oss-mxfp4", 2e-5, 2e-3, "gptoss_mxfp4", "--set", "swiglu_limit=7.0"),
            // Llama 4: llama.cpp hard-codes the chunk (8192) and the NoPE temperature (floor 8192, scale 0.1)
            new Model("llama4", 2e-5, 2e-3, "llama4", "--set", "attention_chunk_size=8192", "--set", "floor_scale=8192",
                    "--set", "attn_scale=0.1"),
            new Model("llama4-mix", 2e-5, 2e-3, "llama4mix", "--set", "attention_chunk_size=8192", "--set", "floor_scale=8192",
                    "--set", "attn_scale=0.1"),
            // Granite 4: llama.cpp's gated norm is per group (equal to transformers' with one group)
            new Model("granite4", 2e-5, 2e-3, "granite", "--set", "mamba_n_groups=1"),
            // Nemotron-H: llama.cpp's scan does not clamp the step size at time_step_min
            new Model("nemotron-h", 2e-5, 2e-3, "nemotronh", "--set", "time_step_min=0.0"),
            new Model("jamba", 2e-5, 2e-3, "jamba"));

    static int passed;
    static int failed;

    static void check(boolean ok, String what) {
        if (ok) {
            passed++;
        } else {
            failed++;
            System.out.println("FAIL: " + what);
        }
    }

    static final int[] PROMPT = {1, 7, 42, 99, 13, 5, 250, 8, 77, 31, 600, 9};

    static boolean python(Path out, Model m) throws Exception {
        List<String> cmd = new ArrayList<>(List.of("python3", "scripts/make-tiny-model.py", out.toString(), m.args()[0]));
        for (int i = 1; i < m.args().length; i++) {
            cmd.add(m.args()[i]);
        }
        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        String log = new String(p.getInputStream().readAllBytes());
        if (p.waitFor() != 0) {
            System.out.println("make-tiny-model.py failed for " + m.name() + ": " + log);
            return false;
        }
        return true;
    }

    static Backend backend(ModelWeights w) {
        return w.arch instanceof MambaArch ? new MambaCpuBackend(w, 64, 16, 4)
                : w.arch instanceof org.robomiri.jllm.model.HybridArch ? new org.robomiri.jllm.cpu.HybridCpuBackend(w, 64, 16, 4, KvDtype.F32, false)
                : new CpuBackend(w, 64, 16, 4, KvDtype.F32, false);
    }

    /** Logits of the prompt and of three greedy continuation steps. */
    static float[][] logits(ModelWeights w, int[] prompt) {
        int vocab = w.arch.core().vocabSize();
        try (Backend b = backend(w)) {
            float[][] out = new float[4][vocab];
            b.forward(prompt, 0, prompt.length, 0, out[0]);
            int pos = prompt.length;
            for (int i = 0; i < 3; i++) {
                b.forward(new int[]{GgufModelTest.argmax(out[i])}, 0, 1, pos++, out[i + 1]);
            }
            return out;
        }
    }

    public static void main(String[] args) throws Exception {
        boolean llamaCpp = false;
        Path keep = null;
        List<String> only = new ArrayList<>();
        List<String> hfDirs = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--llama-cpp" -> llamaCpp = true;
                case "--keep" -> keep = Path.of(args[++i]);
                case "--hf" -> hfDirs.add(args[++i]);
                default -> only.add(args[i]);
            }
        }
        try {
            Process p = new ProcessBuilder("python3", "--version").redirectErrorStream(true).start();
            p.getInputStream().readAllBytes();
            if (p.waitFor() != 0) {
                throw new IOException("python3 failed");
            }
        } catch (IOException e) {
            System.out.println("gguf families: skipped (python3 is needed for scripts/make-tiny-model.py)");
            return;
        }
        String llamaDebug = llamaCpp ? findLlamaDebug() : null;
        if (llamaCpp && llamaDebug == null) {
            System.out.println("llama.cpp: llama-debug not found (set LLAMA_DEBUG), skipping the llama.cpp comparison");
        }
        Path dir = keep != null ? Files.createDirectories(keep) : Files.createTempDirectory("gguf-families");
        try {
            for (Model m : MODELS) {
                if ((!only.isEmpty() || !hfDirs.isEmpty()) && !only.contains(m.name())) {
                    continue;
                }
                run(dir, m, llamaDebug);
            }
            for (String d : hfDirs) { // an existing tiny checkpoint, with the tolerances of a plain model
                run(dir, new Model(d, 5e-3, 5e-3), llamaDebug);
            }
        } finally {
            if (keep == null) {
                try (var s = Files.walk(dir)) {
                    s.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
                }
            }
        }
        System.out.println("gguf families: " + passed + " checks passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        }
    }

    static String findLlamaDebug() {
        String env = System.getenv("LLAMA_DEBUG");
        if (env != null && Files.isExecutable(Path.of(env))) {
            return env;
        }
        for (String d : System.getenv().getOrDefault("PATH", "").split(":")) {
            Path c = Path.of(d, "llama-debug");
            if (Files.isExecutable(c)) {
                return c.toString();
            }
        }
        return null;
    }

    static void run(Path root, Model m, String llamaDebug) throws Exception {
        Path hf = m.args().length == 0 ? Path.of(m.name()) : root.resolve("hf-" + m.name());
        if (m.args().length > 0 && !python(hf, m)) {
            check(false, m.name() + ": tiny model");
            return;
        }
        if (m.args().length == 0) {
            m = new Model(hf.getFileName().toString(), m.tol(), m.llamaCppTol());
        }
        Path f32 = root.resolve(m.name() + "-f32.gguf");
        Path q8 = root.resolve(m.name() + "-q8.gguf");
        Path deq = root.resolve(m.name() + "-deq.gguf");
        HfToGguf.archOverride = java.util.Arrays.asList(m.args()).contains("--gguf-arch") ? m.args()[java.util.Arrays.asList(m.args()).indexOf("--gguf-arch") + 1] : null;
        Architecture hfArch = Architecture.load(hf);
        // DeepSeek: the safetensors comparison uses the unsplit attn_kv_b (the same Q8_0 blocks as the checkpoint's
        // conversion, so it can be tight); the split attn_k_b / attn_v_b, quantized along other rows, are the files
        // llama.cpp is compared on. The quantization of the split tensors alone moves the logits of these tiny,
        // very sensitive models by ~2e-2 (reported below as the floor).
        boolean mla = hfArch instanceof org.robomiri.jllm.model.Decoder d && d.attention() instanceof org.robomiri.jllm.model.Attention.Mla;
        boolean fq = java.util.Arrays.asList(m.args()).contains("--fuse-qkv");
        boolean fg = java.util.Arrays.asList(m.args()).contains("--fuse-gate-up-exps");
        String arch = HfToGguf.convert(hf, f32, new HfToGguf.Options(HfToGguf.Outtype.F32, mla, fq, fg));
        Path f32split = f32;
        if (mla) {
            f32split = root.resolve(m.name() + "-f32split.gguf");
            HfToGguf.convert(hf, f32split, new HfToGguf.Options(HfToGguf.Outtype.F32, false, fq, fg));
        }
        HfToGguf.convert(hf, q8, new HfToGguf.Options(HfToGguf.Outtype.Q8_0, false, fq, fg));
        HfToGguf.convert(hf, deq, new HfToGguf.Options(HfToGguf.Outtype.DEQ_Q8_0, false, fq, fg));
        int vocab = hfArch.core().vocabSize();
        int[] prompt = PROMPT;
        try (ModelWeights hfw = ModelWeights.load(hf, hfArch, QType.Q8_0, ShardProvider.local(hf))) {
            float[][] ref = logits(hfw, prompt);
            try (GgufModel g = GgufModel.open(f32)) {
                String d = g.architecture().core().description();
                System.out.printf("%-12s %-10s %s%n", m.name(), arch, d);
                check(g.architecture().core().vocabSize() == vocab, m.name() + ": vocab");
                try (ModelWeights w = g.weights()) {
                    double diff = GgufModelTest.maxDiff(ref, logits(w, prompt));
                    check(diff < m.tol(), m.name() + ": GGUF F32 vs safetensors, max relative logit diff " + diff + " (bound " + m.tol() + ")");
                    System.out.printf("%-12s   F32 gguf vs safetensors: %.2e%n", "", diff);
                }
            }
            // the Q8_0 file read in place against the F32 file of its dequantized values
            float[][] a;
            float[][] b;
            try (GgufModel g = GgufModel.open(q8); ModelWeights w = g.weights()) {
                a = logits(w, prompt);
            }
            try (GgufModel g = GgufModel.open(deq); ModelWeights w = g.weights()) {
                b = logits(w, prompt);
            }
            double diff = GgufModelTest.maxDiff(b, a);
            check(diff < 2e-4, m.name() + ": Q8_0 in place vs the same values as F32: " + diff);
            double vs = GgufModelTest.maxDiff(ref, a);
            check(vs < (hfArch.core().layers() > 20 ? 0.5 : 0.05), m.name() + ": Q8_0 GGUF vs safetensors within quantization noise: " + vs);
            System.out.printf("%-12s   Q8_0 in place vs F32 of the same values: %.2e; vs safetensors %.2e%n", "", diff, vs);
            if (llamaDebug != null) {
                llamaCpp(llamaDebug, root, m, deq, vocab, true);
                llamaCpp(llamaDebug, root, m, f32split, vocab, false);
            }
        }
    }

    /** llama.cpp's logits for the last token of a prompt (it tokenizes) against jllm's on the same ids. */
    static void llamaCpp(String llamaDebug, Path root, Model m, Path gguf, int vocab, boolean strict) throws Exception {
        Path out = Files.createDirectories(root.resolve("llcpp-" + gguf.getFileName()));
        Process p = new ProcessBuilder(llamaDebug, "-m", gguf.toString(), "-p", "Hello my name is", "--save-logits",
                "--logits-output-dir", out.toString(), "-ngl", "0", "-ctk", "f32", "-ctv", "f32", "-fa", "off", "-c", "64")
                .redirectErrorStream(true).start();
        String log = new String(p.getInputStream().readAllBytes());
        if (!p.waitFor(300, TimeUnit.SECONDS) || p.exitValue() != 0) {
            check(false, m.name() + ": llama.cpp could not run the file: " + log.lines().filter(l -> l.contains("error") || l.contains("ASSERT")
                    || l.contains("failed")).findFirst().orElse(log.substring(Math.max(0, log.length() - 300))));
            return;
        }
        Path bin = null;
        Path tokens = null;
        try (var s = Files.list(out)) {
            for (Path f : (Iterable<Path>) s::iterator) {
                String n = f.getFileName().toString();
                if (n.endsWith("-tokens.bin")) {
                    tokens = f;
                } else if (n.endsWith(".bin")) {
                    bin = f;
                }
            }
        }
        ByteBuffer tb = ByteBuffer.wrap(Files.readAllBytes(tokens)).order(ByteOrder.LITTLE_ENDIAN);
        int[] ids = new int[tb.remaining() / 4];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = tb.getInt();
        }
        ByteBuffer lb = ByteBuffer.wrap(Files.readAllBytes(bin)).order(ByteOrder.LITTLE_ENDIAN);
        float[] theirs = new float[lb.remaining() / 4];
        for (int i = 0; i < theirs.length; i++) {
            theirs[i] = lb.getFloat();
        }
        check(theirs.length == vocab, m.name() + ": llama.cpp logits have " + theirs.length + " entries, expected " + vocab);
        try (GgufModel g = GgufModel.open(gguf); ModelWeights w = g.weights(); Backend b = backend(w)) {
            float[] ours = new float[vocab];
            b.forward(ids, 0, ids.length, 0, ours);
            double worst = 0;
            double mag = 1e-9;
            for (int i = 0; i < vocab; i++) {
                worst = Math.max(worst, Math.abs(ours[i] - theirs[i]));
                mag = Math.max(mag, Math.abs(theirs[i]));
            }
            boolean same = GgufModelTest.argmax(ours) == GgufModelTest.argmax(theirs);
            if (strict) {
                check(worst / mag < m.llamaCppTol(), m.name() + ": jllm vs llama.cpp max logit diff " + worst / mag + " (bound " + m.llamaCppTol() + ")");
                check(same, m.name() + ": same top token as llama.cpp");
            }
            System.out.printf("%-12s   %s (%d prompt tokens): max logit diff %.2e of max |logit| %.2f, top token %s%n", "",
                    strict ? "llama.cpp vs jllm on Q8_0-rounded F32 weights" : "llama.cpp float weights vs jllm Q8_0 (the quantization floor)",
                    ids.length, worst / mag, mag, same ? "same" : "DIFFERENT");
        }
    }
}
