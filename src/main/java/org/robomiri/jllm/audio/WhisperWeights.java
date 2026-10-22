package org.robomiri.jllm.audio;

import org.robomiri.jllm.audio.Dense.Mat;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * The weights of a Whisper checkpoint ({@code model.safetensors} + {@code config.json}) in the layouts the backends
 * multiply with: every {@code nn.Linear} as a {@link Mat} ({@code [out, in]}), the two convolutions as matrices over
 * {@code (channel, tap)}, norms and biases as float arrays. The tensors stay memory-mapped; matrices are copied into
 * heap floats when the model is small enough ({@code residentLimitBytes}), else converted per use.
 */
public final class WhisperWeights implements AutoCloseable {

    /** A self- or cross-attention block: q and v and out have biases, k does not. */
    public static final class Attn {
        public Mat q, k, v, out;
        public float[] lnGamma, lnBeta;   // the layer norm in front of the block
    }

    /** One transformer layer. {@code cross} is null in the encoder. */
    public static final class Layer {
        public Attn self;
        public Attn cross;
        public float[] ffnLnGamma, ffnLnBeta;
        public Mat fc1, fc2;
    }

    public final WhisperConfig cfg;
    public Mat conv1, conv2;                       // [d, nMels*3], [d, d*3]
    public float[] encPos;                         // [sourcePos * d]
    public Layer[] enc, dec;
    public float[] encLnGamma, encLnBeta;
    public Mat embed;                              // [vocab, d]: token embedding and (tied) output projection
    public float[] decPos;                         // [targetPos * d]
    public float[] decLnGamma, decLnBeta;

    private final SafeTensors st;
    public final Path dir;

    private WhisperWeights(WhisperConfig cfg, SafeTensors st, Path dir) {
        this.cfg = cfg;
        this.st = st;
        this.dir = dir;
    }

    public static boolean isWhisper(Path dir) {
        Path c = dir.resolve("config.json");
        if (!Files.isRegularFile(c)) {
            return false;
        }
        try {
            Object mt = Json.parseObject(Files.readString(c)).get("model_type");
            return "whisper".equals(mt);
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    public static WhisperWeights load(Path dir, long residentLimitBytes) throws IOException {
        return load(dir, residentLimitBytes, false);
    }

    /**
     * @param gpuEncoder the encoder runs on the GPU, which reads the checkpoint's tensors itself: its matrices are not
     *                   copied into heap floats. The decoder's (used for every token) and the embedding always are when they fit.
     */
    public static WhisperWeights load(Path dir, long residentLimitBytes, boolean gpuEncoder) throws IOException {
        WhisperConfig cfg = WhisperConfig.parse(Json.parseObject(Files.readString(dir.resolve("config.json"))));
        Path file = dir.resolve("model.safetensors");
        if (!Files.isRegularFile(file)) {
            throw new IOException("no model.safetensors in " + dir + " (a sharded Whisper checkpoint is not supported)");
        }
        SafeTensors st = SafeTensors.open(file);
        WhisperWeights w = new WhisperWeights(cfg, st, dir);
        boolean all = totalBytes(cfg) <= residentLimitBytes;
        w.fill(all && !gpuEncoder, all || decoderBytes(cfg) <= residentLimitBytes);
        return w;
    }

    /** The bytes the weights take as floats (to decide whether to keep them resident). */
    static long totalBytes(WhisperConfig c) {
        long d = c.dModel();
        long enc = c.encLayers() * (4 * d * d + 2 * d * c.encFfn());
        long dec = c.decLayers() * (8 * d * d + 2 * d * c.decFfn());
        return 4 * (enc + dec + (long) c.vocab() * d + 3 * d * (c.nMels() + d));
    }

    /** The bytes of the decoder's layers and the embedding as floats. */
    static long decoderBytes(WhisperConfig c) {
        long d = c.dModel();
        return 4 * (c.decLayers() * (8 * d * d + 2 * d * c.decFfn()) + (long) c.vocab() * d);
    }

    private String prefix;

    private SafeTensors.Tensor t(String name) {
        SafeTensors.Tensor x = st.tensors().get(prefix + name);
        if (x == null) {
            throw new IllegalArgumentException("checkpoint has no tensor " + prefix + name);
        }
        return x;
    }

    private boolean has(String name) {
        return st.tensors().containsKey(prefix + name);
    }

    private float[] vec(String name) {
        return t(name).toFloatArray();
    }

    private float[] vecOrNull(String name) {
        return has(name) ? vec(name) : null;
    }

    private Mat mat(String base, int rows, int cols, boolean bias, boolean resident) {
        return Mat.of(t(base + ".weight"), rows, cols, bias ? vec(base + ".bias") : null, resident);
    }

    private Attn attn(String base, boolean resident) {
        int d = cfg.dModel();
        Attn a = new Attn();
        a.q = mat(base + ".q_proj", d, d, true, resident);
        a.k = mat(base + ".k_proj", d, d, has(base + ".k_proj.bias"), resident);
        a.v = mat(base + ".v_proj", d, d, true, resident);
        a.out = mat(base + ".out_proj", d, d, true, resident);
        return a;
    }

    private void fill(boolean encResident, boolean resident) {
        prefix = st.tensors().containsKey("model.encoder.conv1.weight") ? "model." : "";
        int d = cfg.dModel();
        conv1 = Mat.of(t("encoder.conv1.weight"), d, cfg.nMels() * 3, vec("encoder.conv1.bias"), true);
        conv2 = Mat.of(t("encoder.conv2.weight"), d, d * 3, vec("encoder.conv2.bias"), true);
        encPos = has("encoder.embed_positions.weight") ? vec("encoder.embed_positions.weight")
                : sinusoids(cfg.sourcePos(), d);
        List<Layer> el = new ArrayList<>();
        for (int i = 0; i < cfg.encLayers(); i++) {
            String b = "encoder.layers." + i;
            Layer l = new Layer();
            l.self = attn(b + ".self_attn", encResident);
            l.self.lnGamma = vec(b + ".self_attn_layer_norm.weight");
            l.self.lnBeta = vec(b + ".self_attn_layer_norm.bias");
            l.fc1 = mat(b + ".fc1", cfg.encFfn(), d, true, encResident);
            l.fc2 = mat(b + ".fc2", d, cfg.encFfn(), true, encResident);
            l.ffnLnGamma = vec(b + ".final_layer_norm.weight");
            l.ffnLnBeta = vec(b + ".final_layer_norm.bias");
            el.add(l);
        }
        enc = el.toArray(new Layer[0]);
        encLnGamma = vec("encoder.layer_norm.weight");
        encLnBeta = vec("encoder.layer_norm.bias");
        embed = Mat.of(t("decoder.embed_tokens.weight"), cfg.vocab(), d, null, resident);
        decPos = vec("decoder.embed_positions.weight");
        List<Layer> dl = new ArrayList<>();
        for (int i = 0; i < cfg.decLayers(); i++) {
            String b = "decoder.layers." + i;
            Layer l = new Layer();
            l.self = attn(b + ".self_attn", resident);
            l.self.lnGamma = vec(b + ".self_attn_layer_norm.weight");
            l.self.lnBeta = vec(b + ".self_attn_layer_norm.bias");
            l.cross = attn(b + ".encoder_attn", resident);
            l.cross.lnGamma = vec(b + ".encoder_attn_layer_norm.weight");
            l.cross.lnBeta = vec(b + ".encoder_attn_layer_norm.bias");
            l.fc1 = mat(b + ".fc1", cfg.decFfn(), d, true, resident);
            l.fc2 = mat(b + ".fc2", d, cfg.decFfn(), true, resident);
            l.ffnLnGamma = vec(b + ".final_layer_norm.weight");
            l.ffnLnBeta = vec(b + ".final_layer_norm.bias");
            dl.add(l);
        }
        dec = dl.toArray(new Layer[0]);
        decLnGamma = vec("decoder.layer_norm.weight");
        decLnBeta = vec("decoder.layer_norm.bias");
        if (st.tensors().containsKey("proj_out.weight") && !sameTensor("proj_out.weight", prefix + "decoder.embed_tokens.weight")) {
            throw new IllegalArgumentException("a Whisper checkpoint with an untied proj_out.weight is not supported");
        }
    }

    private boolean sameTensor(String a, String b) {
        SafeTensors.Tensor x = st.tensors().get(a);
        SafeTensors.Tensor y = st.tensors().get(b);
        return x != null && y != null && x.data().address() == y.data().address();
    }

    /** Whisper's sinusoidal positions: {@code sin/cos(pos * exp(-ln(10000) i / (d/2 - 1)))}, sines first. */
    static float[] sinusoids(int length, int channels) {
        float[] out = new float[length * channels];
        int half = channels / 2;
        double inc = Math.log(10000) / (half - 1);
        for (int p = 0; p < length; p++) {
            for (int i = 0; i < half; i++) {
                double a = p * Math.exp(-inc * i);
                out[p * channels + i] = (float) Math.sin(a);
                out[p * channels + half + i] = (float) Math.cos(a);
            }
        }
        return out;
    }

    /** The raw checkpoint (for the Metal backend, which uploads tensors as they are). */
    public SafeTensors checkpoint() {
        return st;
    }

    public String prefix() {
        return prefix;
    }

    /** Total parameter bytes in the file. */
    public long fileBytes() {
        long n = 0;
        for (SafeTensors.Tensor x : st.tensors().values()) {
            n += x.data().byteSize();
        }
        return n;
    }

    @Override
    public void close() {
        st.close();
    }
}
