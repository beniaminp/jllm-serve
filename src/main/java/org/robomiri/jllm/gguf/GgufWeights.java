package org.robomiri.jllm.gguf;

import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.TensorSource;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.EnumMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A {@link TensorSource} over a {@link GgufFile}: the canonical tensor names the loader asks for
 * ({@code model.layers.N.self_attn.q_proj.weight}, ...) are translated to llama.cpp's
 * ({@code blk.N.attn_q.weight}, ...), and the tensors are handed over as follows.
 *
 * <ul>
 *   <li><b>Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, MXFP4, F16, BF16</b>: viewed in
 *       place, no copy ({@link QTensor#mapped}); the kernels read the file's pages. {@code -Djllm.gguf.requantize=<types>}
 *       turns that off for the listed types (see {@link GgmlCodec}).</li>
 *   <li><b>F32 matrices</b> (and the types listed in {@code jllm.gguf.requantize}): dequantized row by row and quantized
 *       to Q8_0 in memory at load.</li>
 *   <li><b>gate and up of a dense MLP</b> (separate tensors in a GGUF, one stacked {@code [2·hidden, dim]} matrix in
 *       the engine): both are viewed in place and joined as a {@linkplain QTensor#split split} tensor (the GPU
 *       kernels bind the two halves separately, the CPU kernels route each row). Only Q2_K, whose kernels read
 *       one stacked block, copies them into one block of memory.</li>
 *   <li><b>MoE experts</b>: {@code ffn_down_exps} is viewed in place; {@code ffn_gate_exps} and {@code ffn_up_exps} are
 *       a split pair interleaved per expert (the stacked layout {@code [e·2m + k]}), also without a copy.</li>
 *   <li><b>norms, biases, the router</b>: copied into float arrays.</li>
 * </ul>
 *
 * <p>The q and k rows of the {@code llama} architecture are permuted in the file; they are not touched here, the
 * architecture selects interleaved RoPE instead (see {@link GgufArchitecture}).
 */
public final class GgufWeights implements TensorSource, AutoCloseable {

    private static final Pattern LAYER = Pattern.compile("^model\\.layers\\.(\\d+)\\.(.+)$");

    private final GgufFile gguf;
    private final Arena arena = Arena.ofShared();
    private final Map<GgmlType, Integer> converted = new EnumMap<>(GgmlType.class);
    private long convertedBytes;
    /** Bytes of memory the conversions allocated (the Q8_0 results, the copies of Q2_K tensors). */
    private long residentBytes;
    private long viewedBytes;

    public GgufWeights(GgufFile gguf) {
        this.gguf = gguf;
    }

    /** The llama.cpp names behind a canonical name: one, two for a fused pair, or null if the name is unknown. */
    private String[] resolve(String name) {
        switch (name) {
            case "model.embed_tokens.weight":
                return new String[]{"token_embd.weight"};
            case "lm_head.weight":
                return new String[]{"output.weight"};
            case "model.norm.weight":
                return new String[]{"output_norm.weight"};
            default:
                break;
        }
        Matcher m = LAYER.matcher(name);
        if (!m.matches()) {
            return null;
        }
        String b = "blk." + m.group(1) + ".";
        return switch (m.group(2)) {
            case "input_layernorm.weight" -> new String[]{b + "attn_norm.weight"};
            case "post_attention_layernorm.weight" -> new String[]{b + "ffn_norm.weight"};
            case "self_attn.q_norm.weight" -> new String[]{b + "attn_q_norm.weight"};
            case "self_attn.k_norm.weight" -> new String[]{b + "attn_k_norm.weight"};
            case "self_attn.q_proj.weight" -> new String[]{b + "attn_q.weight"};
            case "self_attn.k_proj.weight" -> new String[]{b + "attn_k.weight"};
            case "self_attn.v_proj.weight" -> new String[]{b + "attn_v.weight"};
            case "self_attn.q_proj.bias" -> new String[]{b + "attn_q.bias"};
            case "self_attn.k_proj.bias" -> new String[]{b + "attn_k.bias"};
            case "self_attn.v_proj.bias" -> new String[]{b + "attn_v.bias"};
            case "self_attn.o_proj.weight" -> new String[]{b + "attn_output.weight"};
            case "mlp.gate_up" -> new String[]{b + "ffn_gate.weight", b + "ffn_up.weight"};
            case "mlp.down_proj.weight" -> new String[]{b + "ffn_down.weight"};
            case "mlp.router" -> new String[]{b + "ffn_gate_inp.weight"};
            case "mlp.experts.gate_up" -> new String[]{b + "ffn_gate_exps.weight", b + "ffn_up_exps.weight"};
            case "mlp.experts.down" -> new String[]{b + "ffn_down_exps.weight"};
            default -> null;
        };
    }

    @Override
    public boolean has(String name) {
        String[] g = resolve(name);
        if (g == null) {
            return false;
        }
        for (String n : g) {
            if (!gguf.hasTensor(n)) {
                return false;
            }
        }
        return true;
    }

    private String[] require(String name) {
        String[] g = resolve(name);
        if (g == null || !has(name)) {
            throw new IllegalArgumentException("tensor not in the GGUF file: " + name
                    + (g == null ? "" : " (" + String.join(" + ", g) + ")"));
        }
        return g;
    }

    @Override
    public QTensor tensor(String name) {
        String[] g = require(name);
        if (g.length == 1) {
            return matrix(gguf.tensor(g[0]));
        }
        return name.endsWith("experts.gate_up") ? stackExperts(gguf.tensor(g[0]), gguf.tensor(g[1]))
                : stackPair(gguf.tensor(g[0]), gguf.tensor(g[1]));
    }

    @Override
    public float[] floats(String name) {
        GgufFile.Tensor t = gguf.tensor(require(name)[0]);
        if (!GgmlCodec.isFloat(t.type())) {
            throw new IllegalArgumentException(t + " is quantized; a float vector (norm, bias, router) was expected");
        }
        long n = t.elements();
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(t + " is too large for a float array");
        }
        float[] out = new float[(int) n];
        GgmlCodec.dequantizeRow(t.type(), gguf.data(t), 0, out.length, out, 0);
        return out;
    }

    // ------------------------------------------------------------------ matrices

    private static int rowsOf(GgufFile.Tensor t) {
        return Math.toIntExact(t.rows());
    }

    private static int colsOf(GgufFile.Tensor t) {
        return Math.toIntExact(t.cols());
    }

    /** A matrix (or the rows of a stack of matrices) in the engine's formats: a view when the format matches, else Q8_0. */
    private QTensor matrix(GgufFile.Tensor t) {
        return matrix(t, true);
    }

    private QTensor matrix(GgufFile.Tensor t, boolean count) {
        QType q = GgmlCodec.nativeType(t.type());
        if (q != null && colsOf(t) % q.blockSize == 0) {
            viewedBytes += count ? t.size() : 0;
            return QTensor.mapped(q, rowsOf(t), colsOf(t), gguf.data(t));
        }
        return requantize(rowsOf(t), colsOf(t), t);
    }

    /** Q8_0 matrix of {@code rows} rows whose row r comes from {@code src[...]}. */
    private QTensor requantize(int rows, int cols, GgufFile.Tensor... parts) {
        for (GgufFile.Tensor t : parts) {
            if (!GgmlCodec.canDequantize(t.type())) {
                throw new IllegalArgumentException(t + ": ggml type " + t.type() + " is not supported (supported: "
                        + GgmlCodec.SUPPORTED + ")");
            }
            converted.merge(t.type(), 1, Integer::sum);
            convertedBytes += t.size();
        }
        if (cols % QType.BLOCK != 0) {
            throw new IllegalArgumentException(parts[0] + ": a row of " + cols + " elements cannot be quantized to Q8_0 (needs a"
                    + " multiple of " + QType.BLOCK + ")");
        }
        int[] starts = new int[parts.length + 1];
        for (int i = 0; i < parts.length; i++) {
            starts[i + 1] = starts[i] + rowsOf(parts[i]);
        }
        if (starts[parts.length] != rows) {
            throw new IllegalArgumentException("requantize: " + rows + " rows requested from " + starts[parts.length]);
        }
        MemorySegment[] segs = new MemorySegment[parts.length];
        for (int i = 0; i < parts.length; i++) {
            segs[i] = gguf.data(parts[i]);
        }
        residentBytes += QTensor.byteSize(QType.Q8_0, rows, cols);
        return QTensor.quantize(arena, QType.Q8_0, rows, cols, (r, dst) -> {
            int i = 0;
            while (r >= starts[i + 1]) {
                i++;
            }
            GgmlCodec.dequantizeRow(parts[i].type(), segs[i], (long) (r - starts[i]) * parts[i].rowBytes(), cols, dst, 0);
        });
    }

    /**
     * {@code [gate; up]} as one {@code [2·rows, cols]} matrix: two views of the file (llama.cpp sorts the tensors by
     * name, so {@code ffn_norm} sits between them) joined by {@link QTensor#split}, for the formats whose kernels
     * take the two halves separately; Q2_K is copied into one block of memory; mixed or other types are requantized.
     */
    private QTensor stackPair(GgufFile.Tensor gate, GgufFile.Tensor up) {
        checkSame(gate, up);
        QType q = GgmlCodec.nativeType(gate.type());
        if (q != null && gate.type() == up.type()) {
            if (q.gguf) {
                viewedBytes += 2 * gate.size();
                return QTensor.split(matrix(gate, false), matrix(up, false), 0);
            }
            MemorySegment dst = arena.allocate(2 * gate.size(), 64);
            MemorySegment.copy(gguf.data(gate), 0, dst, 0, gate.size());
            MemorySegment.copy(gguf.data(up), 0, dst, gate.size(), gate.size());
            convertedBytes += 2 * gate.size();
            residentBytes += 2 * gate.size();
            return QTensor.mapped(q, 2 * rowsOf(gate), colsOf(gate), dst);
        }
        return requantize(2 * rowsOf(gate), colsOf(gate), gate, up);
    }

    /**
     * Per expert, gate rows then up rows, from the two {@code [dim, moeDim, experts]} tensors: a
     * {@linkplain QTensor#split split} pair of views (interleaved per expert) for the formats whose kernels take
     * them separately, else one block of memory.
     */
    private QTensor stackExperts(GgufFile.Tensor gate, GgufFile.Tensor up) {
        checkSame(gate, up);
        if (gate.dims().length != 3) {
            throw new IllegalArgumentException(gate + " is not a 3-d expert tensor");
        }
        int experts = Math.toIntExact(gate.dims()[2]);
        int m = Math.toIntExact(gate.dims()[1]);
        int cols = colsOf(gate);
        QType q = GgmlCodec.nativeType(gate.type());
        if (q != null && q.gguf && gate.type() == up.type()) {
            viewedBytes += 2 * gate.size();
            return QTensor.split(matrix(gate, false), matrix(up, false), m);
        }
        if (q != null && gate.type() == up.type()) {
            long rb = gate.rowBytes();
            MemorySegment dst = arena.allocate(2 * gate.size(), 64);
            MemorySegment g = gguf.data(gate);
            MemorySegment u = gguf.data(up);
            for (int e = 0; e < experts; e++) {
                MemorySegment.copy(g, (long) e * m * rb, dst, (long) e * 2 * m * rb, (long) m * rb);
                MemorySegment.copy(u, (long) e * m * rb, dst, ((long) e * 2 + 1) * m * rb, (long) m * rb);
            }
            convertedBytes += 2 * gate.size();
            residentBytes += 2 * gate.size();
            return QTensor.mapped(q, 2 * experts * m, cols, dst);
        }
        // mixed or non-native types: interleave the rows through float
        if (!GgmlCodec.canDequantize(gate.type()) || !GgmlCodec.canDequantize(up.type())) {
            throw new IllegalArgumentException(gate + " / " + up + ": ggml type not supported");
        }
        converted.merge(gate.type(), 1, Integer::sum);
        converted.merge(up.type(), 1, Integer::sum);
        convertedBytes += gate.size() + up.size();
        residentBytes += QTensor.byteSize(QType.Q8_0, 2 * experts * m, cols);
        MemorySegment g = gguf.data(gate);
        MemorySegment u = gguf.data(up);
        return QTensor.quantize(arena, QType.Q8_0, 2 * experts * m, cols, (r, dst) -> {
            int e = r / (2 * m);
            int k = r % (2 * m);
            boolean isUp = k >= m;
            int row = e * m + (isUp ? k - m : k);
            GgufFile.Tensor t = isUp ? up : gate;
            GgmlCodec.dequantizeRow(t.type(), isUp ? u : g, (long) row * t.rowBytes(), cols, dst, 0);
        });
    }

    private static void checkSame(GgufFile.Tensor a, GgufFile.Tensor b) {
        if (a.cols() != b.cols() || a.rows() != b.rows() || a.dims().length != b.dims().length) {
            throw new IllegalArgumentException(a + " and " + b + " differ in shape");
        }
    }

    /** One line about what was converted at load (empty when every tensor is used in place). */
    public String summary() {
        if (converted.isEmpty() && convertedBytes == 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        converted.forEach((t, n) -> sb.append(sb.isEmpty() ? "" : ", ").append(n).append(" x ").append(t));
        return String.format("%.0f MB of weights used in place, %.0f MB converted or copied at load (%.0f MB of memory)%s",
                viewedBytes / 1e6, convertedBytes / 1e6, residentBytes / 1e6,
                sb.isEmpty() ? "" : " (requantized to Q8_0: " + sb + ")");
    }

    @Override
    public void close() {
        arena.close();
    }
}
