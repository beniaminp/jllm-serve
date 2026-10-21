package org.robomiri.jllm.gguf;

import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.tokenizer.GgufTokenizer;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A model stored in a GGUF file, ready to run without conversion: its {@link Architecture} resolved from the
 * metadata ({@link GgufArchitecture}), its {@link Tokenizer} built from the {@code tokenizer.ggml.*} keys
 * ({@link GgufTokenizer}) and its {@link ModelWeights} reading the file's pages in place ({@link GgufWeights}).
 *
 * <p>{@link #weights()} hands the open file over to the returned {@link ModelWeights}, which unmaps it on close;
 * {@link #close()} releases the file only if that never happened.
 */
public final class GgufModel implements AutoCloseable {

    private static final Pattern HF_URL = Pattern.compile("^https?://(?:www\\.)?huggingface\\.co/([^/\\s]+/[^/\\s?#]+)");
    private static final Pattern REPO_ID = Pattern.compile("^[A-Za-z0-9][\\w.-]*/[\\w.-]+$");

    private final GgufFile file;
    private final GgufArchitecture.Result resolved;
    private boolean transferred;

    private GgufModel(GgufFile file, GgufArchitecture.Result resolved) {
        this.file = file;
        this.resolved = resolved;
    }

    /**
     * Opens {@code path} (and the other parts of a split model) and resolves its architecture.
     *
     * @throws IOException if the file is not a valid GGUF or describes a model this build cannot run
     */
    public static GgufModel open(Path path) throws IOException {
        GgufFile f = GgufFile.open(path);
        try {
            return new GgufModel(f, GgufArchitecture.parse(f));
        } catch (IOException | RuntimeException e) {
            f.close();
            throw e;
        }
    }

    public Architecture architecture() {
        return resolved.arch();
    }

    public GgufFile file() {
        return file;
    }

    /** One line about the file for the log: name, architecture, quantization mix. */
    public String describe() {
        java.util.Map<GgmlType, Long> mix = new java.util.EnumMap<>(GgmlType.class);
        for (GgufFile.Tensor t : file.tensors().values()) {
            if (t.dims().length >= 2) {
                mix.merge(t.type(), t.size(), Long::sum);
            }
        }
        StringBuilder sb = new StringBuilder();
        mix.forEach((t, b) -> sb.append(sb.isEmpty() ? "" : "+").append(t));
        return String.format("%s (%s, %s, %.0f MB)", file.str("general.name", file.paths().getFirst().getFileName().toString()),
                file.str("general.architecture", "?"), sb, file.byteSize() / 1e6);
    }

    /**
     * The tokenizer described by the file.
     *
     * @throws GgufTokenizer.UnsupportedPreTokenizer if the file's pre-tokenizer has no regex here: the caller may
     *         fetch the original model's tokenizer.json ({@link #originalRepo()}) instead
     * @throws IOException for a file without usable tokenizer metadata
     */
    public Tokenizer tokenizer() throws IOException {
        return tokenizer(file);
    }

    /** The tokenizer described by the {@code tokenizer.ggml.*} keys of any GGUF file (also a vocabulary-only one). */
    public static Tokenizer tokenizer(GgufFile file) throws IOException {
        String[] tokens = file.strings("tokenizer.ggml.tokens");
        if (tokens == null) {
            throw new IOException("the GGUF file has no tokenizer.ggml.tokens");
        }
        GgufTokenizer.Spec spec = new GgufTokenizer.Spec(file.str("tokenizer.ggml.model", null),
                file.str("tokenizer.ggml.pre", null), tokens, file.longs("tokenizer.ggml.token_type"),
                file.doubles("tokenizer.ggml.scores"), file.strings("tokenizer.ggml.merges"),
                (int) file.num("tokenizer.ggml.bos_token_id", -1), (int) file.num("tokenizer.ggml.eos_token_id", -1),
                (int) file.num("tokenizer.ggml.unknown_token_id", -1),
                file.get("tokenizer.ggml.add_bos_token") instanceof Boolean b ? b : null,
                file.bool("tokenizer.ggml.add_space_prefix", true), bytes(file.longs("tokenizer.ggml.precompiled_charsmap")),
                file.bool("tokenizer.ggml.remove_extra_whitespaces", false), file.str("general.name", null),
                file.get("tokenizer.ggml.add_eos_token") instanceof Boolean e ? e : null);
        try {
            return GgufTokenizer.build(spec);
        } catch (GgufTokenizer.UnsupportedPreTokenizer e) {
            throw e;
        } catch (IllegalArgumentException e) {
            throw new IOException("cannot build the tokenizer of the GGUF file: " + e.getMessage(), e);
        }
    }

    /** An integer array (u8 or i8 elements) as bytes, or null. */
    private static byte[] bytes(long[] a) {
        if (a == null) {
            return null;
        }
        byte[] b = new byte[a.length];
        for (int i = 0; i < a.length; i++) {
            b[i] = (byte) a[i];
        }
        return b;
    }

    /**
     * The Hugging Face repository the model was converted from ({@code owner/name}), from
     * {@code general.base_model.0.repo_url}, {@code general.source.url} or {@code general.source.huggingface.repository};
     * null if the file names none.
     */
    public String originalRepo() {
        for (String key : new String[]{"general.base_model.0.repo_url", "general.source.url",
                "general.source.huggingface.repository", "general.source.repo_url"}) {
            String v = file.str(key, null);
            if (v == null) {
                continue;
            }
            Matcher m = HF_URL.matcher(v.trim());
            if (m.find()) {
                return m.group(1);
            }
            if (REPO_ID.matcher(v.trim()).matches()) {
                return v.trim();
            }
        }
        return null;
    }

    /**
     * A warning when the file's Jinja chat template does not look like the built-in template of the model's family
     * (the engine renders its own and ignores the file's), else null.
     */
    public String chatTemplateWarning() {
        String jinja = file.str("tokenizer.chat_template", null);
        if (jinja == null) {
            return null;
        }
        String type = resolved.arch().core().modelType();
        String marker = switch (type) {
            case "qwen2", "qwen3", "qwen3_moe" -> "<|im_start|>";
            case "llama" -> "<|start_header_id|>";
            case "mistral" -> "[INST]";
            default -> null;
        };
        if (marker == null || jinja.contains(marker)) {
            return null;
        }
        return "the chat template stored in the GGUF file does not contain " + marker + ", which the built-in '" + type
                + "' template uses; the engine ignores the stored template and renders its own";
    }

    /**
     * The weights, read in place from the file. The returned {@link ModelWeights} owns the open file from now on.
     *
     * @throws IOException if a tensor is missing, has an unsupported type or does not match the architecture
     */
    public ModelWeights weights() throws IOException {
        GgufWeights src = new GgufWeights(file);
        try {
            ModelWeights w = ModelWeights.of(resolved.arch(), dominantType(), src, () -> {
                src.close();
                file.close();
            }, file.byteSize());
            transferred = true;
            String conv = src.summary();
            if (!conv.isEmpty()) {
                System.err.println("GGUF weights: " + conv);
            }
            return w;
        } catch (IllegalArgumentException e) {
            src.close();
            throw new IOException(file.paths().getFirst().getFileName() + ": " + e.getMessage().replace(ModelWeights.STALE_HINT, ""), e);
        }
    }

    /** The format of most weight bytes, as the engine's {@link QType}, for labels. */
    private QType dominantType() {
        java.util.Map<GgmlType, Long> bytes = new java.util.EnumMap<>(GgmlType.class);
        for (GgufFile.Tensor t : file.tensors().values()) {
            if (t.dims().length >= 2) {
                bytes.merge(t.type(), t.size(), Long::sum);
            }
        }
        GgmlType best = bytes.entrySet().stream().max(java.util.Map.Entry.comparingByValue()).map(java.util.Map.Entry::getKey)
                .orElse(GgmlType.Q8_0);
        QType q = GgmlCodec.nativeType(best);
        return q != null ? q : QType.Q8_0;
    }

    @Override
    public void close() {
        if (!transferred) {
            file.close();
        }
    }
}
