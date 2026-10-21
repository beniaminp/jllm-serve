package org.robomiri.jllm.vision;

import org.robomiri.jllm.gguf.GgufFile;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * A multimodal projector file ({@code mmproj-*.gguf}, llama.cpp's {@code clip} format of tools/mtmd): the vision
 * transformer and the small network that maps its output into the text model's embedding space. This class reads the
 * hyper-parameters ({@code clip.*} keys) and hands out the tensors; the computation is in {@link ClipCpu} and
 * {@code ClipMetal}.
 *
 * <p>Projector types supported here: {@code idefics3} (SmolVLM: SigLIP + pixel shuffle), {@code gemma3} (SigLIP +
 * average pooling), {@code qwen2vl_merger} / {@code qwen2.5vl_merger} / {@code qwen3vl_merger} (Qwen-VL: windowed
 * ViT with 2-D rotary embeddings + 2x2 patch merger).
 */
public final class ClipModel implements AutoCloseable {

    public enum Projector {
        IDEFICS3("idefics3"),
        GEMMA3("gemma3"),
        QWEN2VL("qwen2vl_merger"),
        QWEN25VL("qwen2.5vl_merger"),
        QWEN3VL("qwen3vl_merger");

        public final String id;

        Projector(String id) {
            this.id = id;
        }

        static Projector of(String id) {
            for (Projector p : values()) {
                if (p.id.equals(id)) {
                    return p;
                }
            }
            return null;
        }
    }

    /** The feed-forward activation of the encoder ({@code clip.use_gelu} / {@code clip.use_silu}, else quick-gelu). */
    public enum Act { GELU, SILU, GELU_QUICK }

    /** One transformer block's tensors; absent ones are null. */
    public static final class Layer {
        public Mat q, k, v, qkv, o, up, gate, down;
        public float[] qB, kB, vB, qkvB, oB, upB, gateB, downB;
        public float[] ln1W, ln1B, ln2W, ln2B;
        /** Qwen3-VL deepstack merger after this block (null when the block has none): norm, fc1, fc2 over 2 x 2 merged tokens. */
        public Mat dsFc1, dsFc2;
        public float[] dsNormW, dsNormB, dsFc1B, dsFc2B;

        public boolean hasDeepstack() {
            return dsFc1 != null;
        }
    }

    private final GgufFile file;
    public final Path path;
    public final Projector projector;
    public final int imageSize;
    public final int patchSize;
    public final int nEmbd;
    public final int nFf;
    public final int nLayer;
    public final int nHead;
    public final int projectionDim;
    public final float eps;
    public final float[] mean;
    public final float[] std;
    public final Act act;
    public final int nMerge;
    public final int nWaPattern;
    public final int windowSize;
    public final int imageLongestEdge;
    public final int minPixels;
    public final int maxPixels;
    public final int[] featureLayers;
    public final List<Layer> layers = new ArrayList<>();

    private ClipModel(Path path, GgufFile file) throws IOException {
        this.path = path;
        this.file = file;
        if (!"clip".equals(file.str("general.architecture", ""))) {
            throw new IOException(path.getFileName() + " is not a multimodal projector (general.architecture is '"
                    + file.str("general.architecture", "?") + "', expected 'clip')");
        }
        if (!file.bool("clip.has_vision_encoder", false)) {
            throw new IOException(path.getFileName() + " has no vision encoder (an audio projector?)");
        }
        String pt = file.str("clip.projector_type", file.str("clip.vision.projector_type", ""));
        Projector p = Projector.of(pt);
        if (p == null) {
            throw new IOException(path.getFileName() + ": projector type '" + pt + "' is not supported (supported: "
                    + java.util.Arrays.stream(Projector.values()).map(x -> x.id).toList() + ")");
        }
        this.projector = p;
        this.nEmbd = req("clip.vision.embedding_length");
        this.nFf = (int) file.num("clip.vision.feed_forward_length", 0);
        this.nLayer = req("clip.vision.block_count");
        this.nHead = req("clip.vision.attention.head_count");
        this.projectionDim = req("clip.vision.projection_dim");
        this.imageSize = req("clip.vision.image_size");
        this.patchSize = req("clip.vision.patch_size");
        this.eps = (float) file.dbl("clip.vision.attention.layer_norm_epsilon", 1e-6);
        double[] m = file.doubles("clip.vision.image_mean");
        double[] sd = file.doubles("clip.vision.image_std");
        if (m == null || sd == null || m.length < 3 || sd.length < 3) {
            throw new IOException(path.getFileName() + ": clip.vision.image_mean / image_std are missing");
        }
        this.mean = new float[]{(float) m[0], (float) m[1], (float) m[2]};
        this.std = new float[]{(float) sd[0], (float) sd[1], (float) sd[2]};
        boolean gelu = file.bool("clip.use_gelu", false);
        boolean silu = file.bool("clip.use_silu", false);
        this.act = gelu ? Act.GELU : silu ? Act.SILU : Act.GELU_QUICK;
        int merge = switch (p) {
            case GEMMA3 -> 4;
            case QWEN2VL, QWEN25VL, QWEN3VL -> 2;
            default -> 1;
        };
        merge = (int) file.num("clip.vision.projector.scale_factor", merge);
        if (p == Projector.QWEN2VL || p == Projector.QWEN25VL || p == Projector.QWEN3VL) {
            merge = (int) file.num("clip.vision.spatial_merge_size", merge);
        }
        this.nMerge = merge;
        this.nWaPattern = (int) file.num("clip.vision.n_wa_pattern", 0);
        this.windowSize = (int) file.num("clip.vision.window_size", 0);
        this.imageLongestEdge = (int) file.num("clip.vision.preproc_image_size", 0);
        this.minPixels = (int) file.num("clip.vision.image_min_pixels", -1);
        this.maxPixels = (int) file.num("clip.vision.image_max_pixels", -1);
        long[] fl = file.longs("clip.vision.feature_layer");
        this.featureLayers = fl == null ? new int[0] : java.util.Arrays.stream(fl).mapToInt(x -> (int) x).toArray();
        for (int il = 0; il < nLayer; il++) {
            layers.add(layer(il));
        }
    }

    public static ClipModel open(Path path) throws IOException {
        GgufFile f = GgufFile.open(path);
        try {
            return new ClipModel(path, f);
        } catch (IOException | RuntimeException e) {
            f.close();
            throw e;
        }
    }

    private int req(String key) throws IOException {
        if (!file.has(key)) {
            throw new IOException(path.getFileName() + ": the key " + key + " is missing");
        }
        return (int) file.num(key, 0);
    }

    private Layer layer(int il) {
        String b = "v.blk." + il + ".";
        Layer l = new Layer();
        l.q = matOpt(b + "attn_q.weight");
        l.k = matOpt(b + "attn_k.weight");
        l.v = matOpt(b + "attn_v.weight");
        l.qkv = matOpt(b + "attn_qkv.weight");
        l.o = mat(b + "attn_out.weight");
        l.up = matOpt(b + "ffn_up.weight");
        l.gate = matOpt(b + "ffn_gate.weight");
        l.down = matOpt(b + "ffn_down.weight");
        l.qB = vecOpt(b + "attn_q.bias");
        l.kB = vecOpt(b + "attn_k.bias");
        l.vB = vecOpt(b + "attn_v.bias");
        l.qkvB = vecOpt(b + "attn_qkv.bias");
        l.oB = vecOpt(b + "attn_out.bias");
        l.upB = vecOpt(b + "ffn_up.bias");
        l.gateB = vecOpt(b + "ffn_gate.bias");
        l.downB = vecOpt(b + "ffn_down.bias");
        l.ln1W = vecOpt(b + "ln1.weight");
        l.ln1B = vecOpt(b + "ln1.bias");
        l.ln2W = vecOpt(b + "ln2.weight");
        l.ln2B = vecOpt(b + "ln2.bias");
        String ds = "v.deepstack." + il + ".";
        l.dsFc1 = matOpt(ds + "fc1.weight");
        l.dsFc2 = matOpt(ds + "fc2.weight");
        l.dsNormW = vecOpt(ds + "norm.weight");
        l.dsNormB = vecOpt(ds + "norm.bias");
        l.dsFc1B = vecOpt(ds + "fc1.bias");
        l.dsFc2B = vecOpt(ds + "fc2.bias");
        // some projectors were exported with the feed-forward names swapped (ggml's mul_mat takes the weight's ne[0]
        // as its input width: an "up" matrix whose ne[0] is not the embedding width is really the down projection)
        if (l.up != null && l.down != null && l.down.cols == nEmbd) {
            Mat t = l.up;
            l.up = l.down;
            l.down = t;
            float[] tb = l.upB;
            l.upB = l.downB;
            l.downB = tb;
        }
        return l;
    }

    public boolean has(String name) {
        return file.hasTensor(name);
    }

    public Mat mat(String name) {
        if (!file.hasTensor(name)) {
            throw new IllegalArgumentException(path.getFileName() + ": tensor " + name + " is missing");
        }
        return new Mat(file, file.tensor(name));
    }

    /** A convolution kernel (ggml dims {@code kx, ky, c, out}) as the matrix {@code [out][c * ky * kx]}. */
    public Mat convMat(String name) {
        if (!file.hasTensor(name)) {
            throw new IllegalArgumentException(path.getFileName() + ": tensor " + name + " is missing");
        }
        GgufFile.Tensor t = file.tensor(name);
        int out = Math.toIntExact(t.dims()[t.dims().length - 1]);
        return new Mat(file, t, out, Math.toIntExact(t.elements() / out));
    }

    public Mat matOpt(String name) {
        return file.hasTensor(name) ? new Mat(file, file.tensor(name)) : null;
    }

    public float[] vec(String name) {
        float[] v = vecOpt(name);
        if (v == null) {
            throw new IllegalArgumentException(path.getFileName() + ": tensor " + name + " is missing");
        }
        return v;
    }

    public float[] vecOpt(String name) {
        return file.hasTensor(name) ? matOpt(name).toFloats() : null;
    }

    public GgufFile file() {
        return file;
    }

    /** Size of the tensors in bytes, for the log. */
    public long byteSize() {
        return file.byteSize();
    }

    @Override
    public void close() {
        file.close();
    }
}
