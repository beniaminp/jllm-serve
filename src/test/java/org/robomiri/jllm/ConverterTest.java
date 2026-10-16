package org.robomiri.jllm;

import org.robomiri.jllm.hub.HuggingFaceHub;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.ModelConfig;
import org.robomiri.jllm.model.ModelConverter;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.QuantFile;
import org.robomiri.jllm.model.ShardProvider;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

/**
 * Converter and cache checks (dependency-free, plain main like {@link SmokeTest}): resume after a crash
 * between units, reconversion on a layout-version / identity / corrupt-index mismatch, shard release under a
 * low-disk provider when biases live in their own shard, the tensor rank check, duplicate / overlapping
 * safetensors entries, and the config defaults.
 *
 * <pre>
 * java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
 *      -cp target/classes:target/test-classes org.robomiri.jllm.ConverterTest --dir SCRATCH
 * </pre>
 * {@code SCRATCH} holds private copies of {@code Qwen_Qwen3-0.6B} and
 * {@code hf-internal-testing_tiny-random-Qwen3MoeForCausalLM} (checkpoint, config, tokenizer; no jllm-* files
 * are needed, and any present are replaced). The real models directory must never be used: the tests delete
 * and rewrite cache files.
 */
public final class ConverterTest {

    private static final String QWEN = "Qwen_Qwen3-0.6B";
    private static final String MOE = "hf-internal-testing_tiny-random-Qwen3MoeForCausalLM";
    private static int failures;

    public static void main(String[] args) throws Exception {
        if (args.length == 4 && args[0].equals("--child-convert")) {
            childConvert(Path.of(args[1]), QType.parse(args[2]), Integer.parseInt(args[3]));
            return;
        }
        Path dir = null;
        for (int i = 0; i + 1 < args.length; i++) {
            if (args[i].equals("--dir")) {
                dir = Path.of(args[i + 1]);
            }
        }
        if (dir == null || !Files.isDirectory(dir.resolve(QWEN)) || !Files.isDirectory(dir.resolve(MOE))) {
            System.err.println("usage: ConverterTest --dir <scratch dir holding copies of " + QWEN + " and " + MOE + ">");
            System.exit(2);
        }
        configs(dir);
        hubFileSets(dir);
        writerResume(dir);
        resumeAfterCrash(dir.resolve(QWEN));
        staleCacheReconversion(dir.resolve(MOE));
        identityMismatch(dir.resolve(MOE));
        biasShardRelease(dir);
        rankCheck(dir);
        duplicateAndOverlap(dir);
        if (failures > 0) {
            System.out.println("FAILED: " + failures);
            System.exit(1);
        }
        System.out.println("all converter tests passed");
    }

    // ------------------------------------------------------------------ config defaults (M1, L8-L15)

    static void configs(Path scratch) throws IOException {
        Path d = fresh(scratch.resolve("synthetic-config"));
        try {
            ModelConfig llama = cfg(d, """
                    {"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":64,"intermediate_size":128,
                     "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"vocab_size":100,
                     "rope_parameters":{"rope_type":"llama3","factor":8.0,"low_freq_factor":1.0,"high_freq_factor":4.0,
                     "original_max_position_embeddings":8192,"rope_theta":500000.0}}""");
            check(llama.ropeScaling() != null && llama.ropeScaling().type().equals("llama3") && llama.ropeScaling().factor() == 8.0
                    && llama.ropeTheta() == 500000.0 && llama.normEps() == 1e-6f, "llama: scaling and theta from rope_parameters");
            ModelConfig qwen3 = cfg(d, """
                    {"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","hidden_size":64,"intermediate_size":128,
                     "num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":100,
                     "use_sliding_window":true,"sliding_window":16,"max_window_layers":2}""");
            check(qwen3.windowFor(1) == 0 && qwen3.windowFor(2) == 16 && qwen3.windowFor(3) == 16,
                    "qwen3: use_sliding_window from max_window_layers");
            ModelConfig g2 = cfg(d, """
                    {"architectures":["Gemma2ForCausalLM"],"model_type":"gemma2","hidden_size":64,"intermediate_size":128,
                     "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"vocab_size":100,
                     "sliding_window":8,"query_pre_attn_scalar":32}""");
            check(g2.attnSoftcap() == 50f && g2.finalSoftcap() == 30f && g2.ropeTheta() == 1e4, "gemma2: softcap defaults 50/30, theta 1e4");
            ModelConfig g2off = cfg(d, """
                    {"architectures":["Gemma2ForCausalLM"],"model_type":"gemma2","hidden_size":64,"intermediate_size":128,
                     "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"vocab_size":100,
                     "sliding_window":8,"query_pre_attn_scalar":32,"attn_logit_softcapping":null,"final_logit_softcapping":null}""");
            check(g2off.attnSoftcap() == 0f && g2off.finalSoftcap() == 0f, "gemma2: explicit null soft-capping is off");
            ModelConfig g3 = cfg(d, """
                    {"architectures":["Gemma3ForCausalLM"],"model_type":"gemma3_text","hidden_size":64,"intermediate_size":128,
                     "num_hidden_layers":6,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"vocab_size":100,
                     "sliding_window":8,"query_pre_attn_scalar":32}""");
            check(g3.attnSoftcap() == 0f && g3.finalSoftcap() == 0f && g3.ropeTheta() == 1e6 && g3.gemma().localRopeTheta() == 1e4
                    && g3.usesLocalRope(0) && !g3.usesLocalRope(5), "gemma3: no softcap, theta 1e6, local 1e4");
            check(cfg(d, """
                    {"architectures":["Phi3ForCausalLM"],"model_type":"phi3","hidden_size":64,"intermediate_size":128,
                     "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"vocab_size":100}""").normEps() == 1e-5f,
                    "phi3: rms_norm_eps defaults to 1e-5");
            check(cfg(d, """
                    {"architectures":["GlmForCausalLM"],"model_type":"glm","hidden_size":64,"intermediate_size":128,
                     "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"vocab_size":100}""").normEps() == 1.5625e-7f,
                    "glm: rms_norm_eps defaults to 1.5625e-7");
            ModelConfig v3 = cfg(d, """
                    {"architectures":["DeepseekV3ForCausalLM"],"model_type":"deepseek_v3","hidden_size":64,"intermediate_size":128,
                     "num_hidden_layers":2,"num_attention_heads":2,"vocab_size":100,"q_lora_rank":0,"kv_lora_rank":32,
                     "qk_nope_head_dim":16,"qk_rope_head_dim":16,"v_head_dim":16,"n_routed_experts":16,"num_experts_per_tok":2,
                     "moe_intermediate_size":32,"first_k_dense_replace":1}""");
            check(v3.deepseek().sigmoid() && v3.deepseek().topkMethod().equals("noaux_tc") && v3.deepseek().nGroup() == 8
                    && v3.deepseek().topkGroup() == 4 && v3.deepseek().routedScale() == 2.5 && v3.normTopK(),
                    "deepseek_v3: sigmoid / noaux_tc / 8 groups keep 4 / 2.5 / norm_topk_prob defaults");
            String shortTypes = rejects(d, """
                    {"architectures":["Gemma3ForCausalLM"],"model_type":"gemma3_text","hidden_size":64,"intermediate_size":128,
                     "num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"vocab_size":100,
                     "layer_types":["sliding_attention","full_attention"]}""");
            check(shortTypes != null && shortTypes.contains("layer_types"), "gemma3: short layer_types rejected: " + shortTypes);
            String badEos = rejects(d, """
                    {"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":64,"intermediate_size":128,
                     "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"vocab_size":100,
                     "eos_token_id":[2,"</s>"]}""");
            check(badEos != null && badEos.contains("eos_token_id"), "eos_token_id with a string entry rejected: " + badEos);
            String badMlp = rejects(d, """
                    {"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe","hidden_size":64,"intermediate_size":128,
                     "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"vocab_size":100,
                     "num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":32,"mlp_only_layers":"0"}""");
            check(badMlp != null && badMlp.contains("mlp_only_layers"), "mlp_only_layers that is not a list rejected: " + badMlp);
        } finally {
            deleteTree(d);
        }
    }

    private static ModelConfig cfg(Path dir, String json) throws IOException {
        Files.writeString(dir.resolve("config.json"), json);
        return ModelConfig.load(dir);
    }

    private static String rejects(Path dir, String json) throws IOException {
        try {
            cfg(dir, json);
            return null;
        } catch (IllegalArgumentException e) {
            return e.getMessage();
        }
    }

    // ------------------------------------------------------------------ hub file sets (M7, L21)

    static void hubFileSets(Path scratch) throws IOException {
        Path d = fresh(scratch.resolve("synthetic-hub"));
        try {
            check(HuggingFaceHub.weightFileNames(d).equals(List.of("model.safetensors")), "no index: only model.safetensors");
            Files.writeString(d.resolve("model.safetensors.index.json"), """
                    {"metadata":{"total_size":1},"weight_map":{"a":"model-00002-of-00002.safetensors","b":"model-00001-of-00002.safetensors",
                     "c":"model-00001-of-00002.safetensors"}}""");
            check(HuggingFaceHub.weightFileNames(d).equals(List.of("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")),
                    "index: exactly the weight_map shards, once each, sorted");
            Files.writeString(d.resolve("model.safetensors.index.json"), """
                    {"weight_map":{"a":"../x.safetensors"}}""");
            String bad = null;
            try {
                HuggingFaceHub.weightFileNames(d);
            } catch (IOException e) {
                bad = e.getMessage();
            }
            check(bad != null && bad.contains("invalid shard file name"), "index: shard name with .. rejected: " + bad);
            check(HuggingFaceHub.isMetadataFile("config.json") && HuggingFaceHub.isMetadataFile("model.safetensors.index.json")
                    && !HuggingFaceHub.isMetadataFile("model.safetensors") && !HuggingFaceHub.isMetadataFile("consolidated.safetensors"),
                    "metadata file set");
            check(HuggingFaceHub.isLoopback("localhost") && HuggingFaceHub.isLoopback("127.0.0.1") && HuggingFaceHub.isLoopback("[::1]")
                    && !HuggingFaceHub.isLoopback("huggingface.co") && !HuggingFaceHub.isLoopback("10.0.0.1") && !HuggingFaceHub.isLoopback(null),
                    "loopback endpoint detection");
        } finally {
            deleteTree(d);
        }
    }

    // ------------------------------------------------------------------ writer resume rules (M2, M3)

    /** The Writer's own resume logic: a committed unit resumes; another layout version or identity does not. */
    static void writerResume(Path scratch) throws IOException {
        Path d = fresh(scratch.resolve("synthetic-writer"));
        try {
            QType q = QType.Q4_0;
            try (QuantFile.Writer w = new QuantFile.Writer(d, q)) {
                w.putFloats("x", new float[]{1, 2, 3});
                w.commitUnit("u1");
                // closed without finish(): a crash after unit u1
            }
            check(Files.isRegularFile(QuantFile.partialPath(d, q)) && !Files.isRegularFile(QuantFile.jsonPath(d, q)),
                    "writer: sidecar kept, no index, after a close without finish");
            List<String> plan = List.of("u1", "u2");
            try (QuantFile.Writer w = QuantFile.Writer.openOrResume(d, q, "", null, plan)) {
                check(w.completedUnits().equals(List.of("u1")), "writer: resumes after the committed unit");
            }
            try (QuantFile.Writer w = QuantFile.Writer.openOrResume(d, q, "", QuantFile.Identity.parse("org/other@main", null), plan)) {
                check(w.completedUnits().isEmpty(), "writer: sidecar for local files is refused for a repo identity");
            }
            // start over so a sidecar exists again, then patch its layout version
            try (QuantFile.Writer w = new QuantFile.Writer(d, q)) {
                w.putFloats("x", new float[]{1, 2, 3});
                w.commitUnit("u1");
            }
            Path partial = QuantFile.partialPath(d, q);
            Files.writeString(partial, Files.readString(partial).replace("\"layout\":" + QuantFile.LAYOUT_VERSION, "\"layout\":1"));
            try (QuantFile.Writer w = QuantFile.Writer.openOrResume(d, q, "", null, plan)) {
                check(w.completedUnits().isEmpty(), "writer: sidecar from another layout version is refused");
            }
            check(!Files.isRegularFile(QuantFile.partialPath(d, q).resolveSibling("jllm-q4_0.partial.json.tmp")), "writer: no tmp file left");
        } finally {
            deleteTree(d);
        }
    }

    // ------------------------------------------------------------------ crash between units (resume)

    /** Child JVM: converts {@code dir} and halts (no cleanup, like a crash) once the sidecar records {@code units} units. */
    private static void childConvert(Path dir, QType quant, int units) throws Exception {
        Path partial = QuantFile.partialPath(dir, quant);
        Thread.ofPlatform().daemon(true).start(() -> {
            while (true) {
                try {
                    Thread.sleep(2);
                    if (Files.isRegularFile(partial)) {
                        List<Object> done = Json.list(Json.parseObject(Files.readString(partial)), "units");
                        if (done != null && done.size() >= units) {
                            System.err.println("simulated crash after " + done.size() + " units");
                            Runtime.getRuntime().halt(137);
                        }
                    }
                } catch (Exception ignored) {
                    // a sidecar being replaced: try again
                }
            }
        });
        ModelConverter.convert(dir, ModelConfig.load(dir), quant, ShardProvider.local(dir));
        System.err.println("conversion finished before the simulated crash");
        System.exit(3);
    }

    static void resumeAfterCrash(Path dir) throws Exception {
        QType q = QType.Q4_0;
        QuantFile.delete(dir, q);
        int crashAfter = 5;
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        List<String> cmd = new ArrayList<>(List.of(java, "--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED",
                "-Xmx4g", "-cp", System.getProperty("java.class.path"), ConverterTest.class.getName(),
                "--child-convert", dir.toString(), "q4", String.valueOf(crashAfter)));
        Path log = dir.resolve("child.log");
        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).redirectOutput(log.toFile()).start();
        if (!p.waitFor(180, TimeUnit.SECONDS)) {
            p.destroyForcibly();
            check(false, "child conversion did not finish within 180 s");
            return;
        }
        String out = Files.readString(log);
        check(p.exitValue() == 137 && out.contains("simulated crash"), "child JVM halted mid-conversion (exit " + p.exitValue() + ")");
        Path partial = QuantFile.partialPath(dir, q);
        int recorded = Json.list(Json.parseObject(Files.readString(partial)), "units").size();
        check(recorded >= crashAfter && !Files.isRegularFile(QuantFile.jsonPath(dir, q)) && Files.isRegularFile(QuantFile.binPath(dir, q)),
                "after the crash: sidecar with " + recorded + " units, .bin present, no index");
        long binAfterCrash = Files.size(QuantFile.binPath(dir, q));

        ModelConfig cfg = ModelConfig.load(dir);
        List<String> plan = new ArrayList<>(List.of("embed"));
        for (int l = 0; l < cfg.layers(); l++) {
            plan.add("layer." + l);
        }
        plan.add("final");
        try (QuantFile.Writer w = QuantFile.Writer.openOrResume(dir, q, cfg.toString(), null, plan)) {
            check(w.completedUnits().size() == recorded && w.completedUnits().equals(plan.subList(0, recorded)),
                    "sidecar resumes exactly the recorded units");
        }
        CountingProvider resumed = new CountingProvider(dir, false);
        try (ModelWeights w = ModelWeights.load(dir, cfg, q, resumed)) {
            check(w.layers.length == cfg.layers(), "resumed conversion loads");
        }
        check(resumed.ensured > 0, "resumed conversion read the checkpoint");
        String resumedHash = sha256(QuantFile.binPath(dir, q));
        Map<String, Object> resumedIndex = tensors(QuantFile.jsonPath(dir, q));
        check(!Files.isRegularFile(partial), "sidecar removed after finish");
        check(Files.size(QuantFile.binPath(dir, q)) >= binAfterCrash, "resumed .bin grew from the crash point");

        QuantFile.delete(dir, q);
        try (ModelWeights w = ModelWeights.load(dir, cfg, q, ShardProvider.local(dir))) {
            check(w.layers.length == cfg.layers(), "fresh conversion loads");
        }
        check(sha256(QuantFile.binPath(dir, q)).equals(resumedHash), "resumed .bin is byte-identical to a fresh conversion");
        check(tensors(QuantFile.jsonPath(dir, q)).equals(resumedIndex), "resumed index equals the fresh one");
        Files.deleteIfExists(log);
    }

    // ------------------------------------------------------------------ stale caches (M2, M4)

    static void staleCacheReconversion(Path dir) throws Exception {
        QType q = QType.Q4_0;
        ModelConfig cfg = ModelConfig.load(dir);
        QuantFile.delete(dir, q);
        try (ModelWeights w = ModelWeights.load(dir, cfg, q, ShardProvider.local(dir))) {
            check(w.layers.length == cfg.layers(), "tiny MoE converts");
        }
        String hash = sha256(QuantFile.binPath(dir, q));
        Path json = QuantFile.jsonPath(dir, q);
        try (Stream<Path> s = Files.list(dir)) {
            check(s.noneMatch(f -> f.getFileName().toString().endsWith(".tmp")), "no .tmp files after finish");
        }

        // 1. another layout version
        Files.writeString(json, Files.readString(json).replace("\"layout\":" + QuantFile.LAYOUT_VERSION, "\"layout\":1"));
        String stale = null;
        try {
            QuantFile.open(dir, q).close();
        } catch (QuantFile.StaleCacheException e) {
            stale = e.getMessage();
        }
        check(stale != null && stale.contains("layout 1"), "open() rejects another layout version: " + stale);
        CountingProvider p1 = loadCounting(dir, cfg, q, null);
        check(p1.ensured > 0 && Json.num(Json.parseObject(Files.readString(json)), "layout", 0) == QuantFile.LAYOUT_VERSION
                && sha256(QuantFile.binPath(dir, q)).equals(hash), "layout mismatch: reconverted, same bytes");

        // 2. unparseable index
        Files.writeString(json, "{\"version\":1,\"layout\":" + QuantFile.LAYOUT_VERSION + ",\"tensors\":{\"x\":");
        CountingProvider p2 = loadCounting(dir, cfg, q, null);
        check(p2.ensured > 0 && sha256(QuantFile.binPath(dir, q)).equals(hash), "unparseable index: reconverted");

        // 3. a tensor the config needs is missing from the index (shape-check path)
        String text = Files.readString(json);
        int at = text.indexOf("\"model.layers.0.self_attn.q_norm.weight\"");
        int end = text.indexOf("},", at) + 2;
        Files.writeString(json, text.substring(0, at) + text.substring(end));
        CountingProvider p3 = loadCounting(dir, cfg, q, null);
        check(p3.ensured > 0 && sha256(QuantFile.binPath(dir, q)).equals(hash), "missing tensor: reconverted");

        // 4. an index entry that does not fit the .bin
        text = Files.readString(json);
        Files.writeString(json, text.replaceFirst("\"rows\":151936", "\"rows\":151937"));
        CountingProvider p4 = loadCounting(dir, cfg, q, null);
        check(p4.ensured > 0 && sha256(QuantFile.binPath(dir, q)).equals(hash), "corrupt entry: reconverted");

        // 5. a good cache is not reconverted
        CountingProvider p5 = loadCounting(dir, cfg, q, null);
        check(p5.ensured == 0, "valid cache is mapped without conversion");
    }

    // ------------------------------------------------------------------ identity (M3)

    static void identityMismatch(Path dir) throws Exception {
        QType q = QType.Q8_0;
        ModelConfig cfg = ModelConfig.load(dir);
        QuantFile.delete(dir, q);
        CountingProvider a = loadCounting(dir, cfg, q, "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM@main");
        Map<String, Object> header = Json.parseObject(Files.readString(QuantFile.jsonPath(dir, q)));
        check(a.ensured > 0 && "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM".equals(header.get("repo"))
                && "main".equals(header.get("revision")) && "0123abcd".equals(header.get("sha")), "index records repo, revision and sha");
        try (QuantFile f = QuantFile.open(dir, q)) {
            check(f.identity() != null && f.identity().name().equals("hf-internal-testing/tiny-random-Qwen3MoeForCausalLM@main"),
                    "QuantFile.identity()");
        }
        CountingProvider same = loadCounting(dir, cfg, q, "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM@main");
        check(same.ensured == 0, "same identity: no reconversion");
        CountingProvider local = loadCounting(dir, cfg, q, null);
        check(local.ensured == 0, "no identity given: cache is used as is");
        CountingProvider other = loadCounting(dir, cfg, q, "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM@refs/pr/1");
        header = Json.parseObject(Files.readString(QuantFile.jsonPath(dir, q)));
        check(other.ensured > 0 && "refs/pr/1".equals(header.get("revision")), "other revision: reconverted and recorded");
        CountingProvider repo = loadCounting(dir, cfg, q, "someone-else/model@refs/pr/1");
        check(repo.ensured > 0, "other repo: reconverted");
    }

    private static CountingProvider loadCounting(Path dir, ModelConfig cfg, QType q, String identity) throws IOException {
        CountingProvider p = new CountingProvider(dir, false);
        try (ModelWeights w = ModelWeights.load(dir, cfg, q, p, identity)) {
            check(w.layers.length == cfg.layers(), "load " + q + (identity != null ? " as " + identity : ""));
        }
        return p;
    }

    /** Counts ensure / release; with {@code lowDisk} deletes released shards and refuses to supply them again. */
    private static final class CountingProvider implements ShardProvider {
        final Path dir;
        final boolean lowDisk;
        final List<String> released = new ArrayList<>();
        int ensured;

        CountingProvider(Path dir, boolean lowDisk) {
            this.dir = dir;
            this.lowDisk = lowDisk;
        }

        @Override
        public Path ensure(String fileName) throws IOException {
            ensured++;
            Path p = dir.resolve(ShardProvider.checkName(dir, fileName));
            if (!Files.isRegularFile(p)) {
                throw new IOException("missing " + p + (released.contains(fileName) ? " (released before its last read)" : ""));
            }
            return p;
        }

        @Override
        public void release(String fileName) throws IOException {
            released.add(ShardProvider.checkName(dir, fileName));
            if (lowDisk) {
                Files.deleteIfExists(dir.resolve(fileName));
            }
        }

        @Override
        public String commitSha() {
            return "0123abcd";
        }
    }

    // ------------------------------------------------------------------ synthetic checkpoints (M5, M6, L24)

    private static final String QWEN2_CONFIG = """
            {"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2","hidden_size":32,"intermediate_size":64,
             "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"vocab_size":64,
             "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true,"bos_token_id":0,"eos_token_id":1,
             "max_position_embeddings":64}""";

    /** The tensors of the tiny Qwen2 above, in checkpoint order: name -> shape. */
    private static Map<String, int[]> qwen2Tensors() {
        Map<String, int[]> t = new LinkedHashMap<>();
        t.put("model.embed_tokens.weight", new int[]{64, 32});
        for (int l = 0; l < 2; l++) {
            String p = "model.layers." + l + ".";
            t.put(p + "input_layernorm.weight", new int[]{32});
            t.put(p + "post_attention_layernorm.weight", new int[]{32});
            t.put(p + "self_attn.q_proj.weight", new int[]{32, 32});
            t.put(p + "self_attn.q_proj.bias", new int[]{32});
            t.put(p + "self_attn.k_proj.weight", new int[]{16, 32});
            t.put(p + "self_attn.k_proj.bias", new int[]{16});
            t.put(p + "self_attn.v_proj.weight", new int[]{16, 32});
            t.put(p + "self_attn.v_proj.bias", new int[]{16});
            t.put(p + "self_attn.o_proj.weight", new int[]{32, 32});
            t.put(p + "mlp.gate_proj.weight", new int[]{64, 32});
            t.put(p + "mlp.up_proj.weight", new int[]{64, 32});
            t.put(p + "mlp.down_proj.weight", new int[]{32, 64});
        }
        t.put("model.norm.weight", new int[]{32});
        return t;
    }

    private static float[] values(String name, int[] shape) {
        int n = 1;
        for (int d : shape) {
            n *= d;
        }
        float[] v = new float[n];
        Random r = new Random(name.hashCode());
        for (int i = 0; i < n; i++) {
            v[i] = name.endsWith("layernorm.weight") || name.equals("model.norm.weight") ? 1f + 0.1f * r.nextFloat()
                    : (float) r.nextGaussian() * 0.1f;
        }
        return v;
    }

    /** Writes a safetensors file with F32 tensors; {@code offsets} may force specific data_offsets for a name. */
    private static void writeSafetensors(Path file, Map<String, int[]> shapes, Map<String, long[]> offsets) throws IOException {
        StringBuilder header = new StringBuilder("{");
        List<float[]> data = new ArrayList<>();
        long pos = 0;
        for (Map.Entry<String, int[]> e : shapes.entrySet()) {
            float[] v = values(e.getKey(), e.getValue());
            long begin = pos;
            long end = pos + 4L * v.length;
            if (offsets != null && offsets.containsKey(e.getKey())) {
                begin = offsets.get(e.getKey())[0];
                end = offsets.get(e.getKey())[1];
            } else {
                pos = end;
                data.add(v);
            }
            if (header.length() > 1) {
                header.append(',');
            }
            header.append('"').append(e.getKey()).append("\":{\"dtype\":\"F32\",\"shape\":")
                    .append(Arrays.toString(e.getValue()).replace(" ", "")).append(",\"data_offsets\":[").append(begin).append(',').append(end).append("]}");
        }
        header.append('}');
        while (header.length() % 8 != 0) {
            header.append(' ');
        }
        byte[] h = header.toString().getBytes(StandardCharsets.UTF_8);
        ByteBuffer bb = ByteBuffer.allocate(8 + h.length + (int) pos).order(ByteOrder.LITTLE_ENDIAN);
        bb.putLong(h.length).put(h);
        for (float[] v : data) {
            for (float f : v) {
                bb.putFloat(f);
            }
        }
        Files.write(file, bb.array());
    }

    private static void writeIndex(Path dir, Map<String, String> shardOf) throws IOException {
        StringBuilder sb = new StringBuilder("{\"metadata\":{},\"weight_map\":{");
        boolean first = true;
        for (Map.Entry<String, String> e : shardOf.entrySet()) {
            sb.append(first ? "" : ",").append('"').append(e.getKey()).append("\":\"").append(e.getValue()).append('"');
            first = false;
        }
        Files.writeString(dir.resolve("model.safetensors.index.json"), sb.append("}}").toString());
    }

    /**
     * Shard B holds layer 0's o_proj and layer 1's q/k/v biases only. A converter whose unit inputs omit the
     * biases counts one read for B, releases (deletes) it after layer 0 and then fails to read layer 1's biases.
     */
    static void biasShardRelease(Path scratch) throws Exception {
        Path d = fresh(scratch.resolve("synthetic-qwen2-shards"));
        try {
            Files.writeString(d.resolve("config.json"), QWEN2_CONFIG);
            Map<String, int[]> all = qwen2Tensors();
            Map<String, int[]> a = new LinkedHashMap<>();
            Map<String, int[]> b = new LinkedHashMap<>();
            Map<String, String> shardOf = new LinkedHashMap<>();
            for (Map.Entry<String, int[]> e : all.entrySet()) {
                boolean inB = e.getKey().equals("model.layers.0.self_attn.o_proj.weight")
                        || (e.getKey().startsWith("model.layers.1.") && e.getKey().endsWith(".bias"));
                (inB ? b : a).put(e.getKey(), e.getValue());
                shardOf.put(e.getKey(), inB ? "model-00002-of-00002.safetensors" : "model-00001-of-00002.safetensors");
            }
            writeSafetensors(d.resolve("model-00001-of-00002.safetensors"), a, null);
            writeSafetensors(d.resolve("model-00002-of-00002.safetensors"), b, null);
            writeIndex(d, shardOf);
            ModelConfig cfg = ModelConfig.load(d);
            check(cfg.attentionBias(), "synthetic qwen2 has q/k/v biases");
            CountingProvider p = new CountingProvider(d, true);
            try (ModelWeights w = ModelWeights.load(d, cfg, QType.Q8_0, p)) {
                float[] bq1 = values("model.layers.1.self_attn.q_proj.bias", new int[]{32});
                float[] bv1 = values("model.layers.1.self_attn.v_proj.bias", new int[]{16});
                check(Arrays.equals(w.layers[1].bq, bq1) && Arrays.equals(w.layers[1].bv, bv1), "layer 1 biases converted from shard B");
                check(w.layers[0].wo.rows == 32 && w.layers[0].wo.cols == 32, "layer 0 o_proj converted from shard B");
            }
            check(p.released.equals(List.of("model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors")),
                    "low-disk: shard B released once, after its last read in layer 1, then shard A: " + p.released);
            check(!Files.exists(d.resolve("model-00001-of-00002.safetensors")) && !Files.exists(d.resolve("model-00002-of-00002.safetensors")),
                    "low-disk: both shards deleted after conversion");
            check(p.ensured == 2, "each shard opened exactly once (ensure calls: " + p.ensured + ")");
        } finally {
            deleteTree(d);
        }
    }

    static void rankCheck(Path scratch) throws Exception {
        Path d = fresh(scratch.resolve("synthetic-rank"));
        try {
            Files.writeString(d.resolve("config.json"), QWEN2_CONFIG);
            Map<String, int[]> all = qwen2Tensors();
            all.put("model.layers.1.self_attn.q_proj.weight", new int[]{2, 16, 32}); // same numel, wrong rank
            writeSafetensors(d.resolve("model.safetensors"), all, null);
            String msg = convertFails(d, QType.Q8_0);
            check(msg != null && msg.contains("2-D") && msg.contains("q_proj"), "3-D weight rejected by the converter: " + msg);
            check(!QuantFile.exists(d, QType.Q8_0), "no cache left after the failed conversion");

            all = qwen2Tensors();
            all.put("model.layers.0.mlp.up_proj.weight", new int[]{64, 32, 1});
            writeSafetensors(d.resolve("model.safetensors"), all, null);
            msg = convertFails(d, QType.Q8_0);
            check(msg != null && msg.contains("2-D") && msg.contains("up_proj"), "3-D stack part rejected: " + msg);
        } finally {
            deleteTree(d);
        }
    }

    private static String convertFails(Path d, QType q) throws IOException {
        QuantFile.delete(d, q);
        try {
            ModelConverter.convert(d, ModelConfig.load(d), q, ShardProvider.local(d));
            return null;
        } catch (IOException e) {
            return e.getMessage();
        }
    }

    static void duplicateAndOverlap(Path scratch) throws Exception {
        Path d = fresh(scratch.resolve("synthetic-dup"));
        try {
            Map<String, int[]> one = new LinkedHashMap<>();
            one.put("a", new int[]{4, 32});
            one.put("b", new int[]{32});
            Map<String, int[]> two = new LinkedHashMap<>();
            two.put("b", new int[]{32});
            two.put("c", new int[]{32});
            writeSafetensors(d.resolve("model-00001-of-00002.safetensors"), one, null);
            writeSafetensors(d.resolve("model-00002-of-00002.safetensors"), two, null);
            Map<String, String> shardOf = new LinkedHashMap<>();
            shardOf.put("a", "model-00001-of-00002.safetensors");
            shardOf.put("b", "model-00001-of-00002.safetensors");
            shardOf.put("c", "model-00002-of-00002.safetensors");
            writeIndex(d, shardOf);
            String msg = openFails(d);
            check(msg != null && msg.contains("duplicate tensor b") && msg.contains("also in"), "duplicate name across shards rejected: " + msg);

            // one file whose second tensor's bytes overlap the first's
            Map<String, int[]> shapes = new LinkedHashMap<>();
            shapes.put("a", new int[]{4, 32});
            shapes.put("b", new int[]{32});
            Map<String, long[]> forced = Map.of("b", new long[]{64, 192});
            writeSafetensors(d.resolve("overlap.safetensors"), shapes, forced);
            msg = null;
            try {
                SafeTensors.open(d.resolve("overlap.safetensors")).close();
            } catch (IOException e) {
                msg = e.getMessage();
            }
            check(msg != null && msg.contains("overlap"), "overlapping data_offsets rejected: " + msg);
        } finally {
            deleteTree(d);
        }
    }

    private static String openFails(Path d) {
        try {
            SafeTensors.open(d).close();
            return null;
        } catch (IOException e) {
            return e.getMessage();
        }
    }

    // ------------------------------------------------------------------ helpers

    private static Path fresh(Path d) throws IOException {
        deleteTree(d);
        Files.createDirectories(d);
        return d;
    }

    private static void deleteTree(Path d) throws IOException {
        if (!Files.exists(d)) {
            return;
        }
        try (Stream<Path> s = Files.walk(d)) {
            for (Path p : s.sorted(java.util.Comparator.reverseOrder()).toList()) {
                Files.delete(p);
            }
        }
    }

    private static Map<String, Object> tensors(Path json) throws IOException {
        return Json.map(Json.parseObject(Files.readString(json)), "tensors");
    }

    private static String sha256(Path p) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        try (var in = Files.newInputStream(p)) {
            byte[] buf = new byte[1 << 20];
            int n;
            while ((n = in.read(buf)) >= 0) {
                md.update(buf, 0, n);
            }
        }
        return HexFormat.of().formatHex(md.digest());
    }

    static void check(boolean ok, String what) {
        if (ok) {
            System.out.println("ok  " + what);
        } else {
            System.out.println("FAIL " + what);
            failures++;
        }
    }
}
