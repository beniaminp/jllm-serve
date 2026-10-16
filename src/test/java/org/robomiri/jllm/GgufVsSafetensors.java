package org.robomiri.jllm;

import org.robomiri.jllm.gguf.GgufFile;
import org.robomiri.jllm.gguf.GgufWeights;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.nio.file.Path;
import java.util.Map;
import java.util.TreeMap;

/**
 * Dequantizes every weight tensor of a GGUF file through the engine's own loading path ({@link GgufWeights}) and
 * compares it with the bf16 original in a safetensors checkpoint of the same model: proves the tensor name mapping,
 * the dimension order and (for llama) the absence or presence of the q/k permutation, and measures the error of each
 * format. A tensor passes when its relative RMS error is below the bound of its format.
 *
 * <pre>
 * java ... org.robomiri.jllm.GgufVsSafetensors model.gguf /path/to/Qwen_Qwen3-0.6B [--permuted-llama]
 * </pre>
 * With {@code --permuted-llama} the original's q and k rows are permuted like llama.cpp's converter does before
 * comparing (for a llama-architecture file).
 */
public final class GgufVsSafetensors {

    public static void main(String[] args) throws Exception {
        Path gguf = Path.of(args[0]);
        Path dir = Path.of(args[1]);
        boolean permuted = args.length > 2 && args[2].equals("--permuted-llama");
        int heads = 0;
        int kvHeads = 0;
        int tensors = 0;
        int bad = 0;
        Map<String, double[]> byType = new TreeMap<>(); // type -> {count, worst rel rms, sum rel rms}
        try (GgufFile g = GgufFile.open(gguf); GgufWeights src = new GgufWeights(g);
             SafeTensors st = SafeTensors.open(dir.resolve("model.safetensors"))) {
            heads = (int) g.num(g.str("general.architecture", "") + ".attention.head_count", 0);
            kvHeads = (int) g.num(g.str("general.architecture", "") + ".attention.head_count_kv", heads);
            for (Map.Entry<String, SafeTensors.Tensor> e : st.tensors().entrySet()) {
                String name = e.getKey();
                if (name.equals("lm_head.weight") && !src.has(name)) {
                    continue; // tied head
                }
                String cacheName = name.endsWith("mlp.gate_proj.weight") || name.endsWith("mlp.up_proj.weight") ? null : name;
                if (cacheName != null && !src.has(cacheName)) {
                    System.out.println("not in the GGUF: " + name);
                    continue;
                }
                SafeTensors.Tensor ref = e.getValue();
                if (ref.shape().length != 2) {
                    float[] want = ref.toFloatArray();
                    float[] got = src.floats(name);
                    double err = rel(want, got);
                    tensors++;
                    if (err > 1e-6) {
                        bad++;
                        System.out.printf("BAD %-50s F32 rel err %.3e%n", name, err);
                    }
                    continue;
                }
                int rows = ref.dim(0);
                int cols = ref.dim(1);
                float[] want = ref.toFloatArray();
                if (permuted && (name.endsWith("q_proj.weight") || name.endsWith("k_proj.weight"))) {
                    want = permute(want, rows, cols, name.endsWith("q_proj.weight") ? heads : kvHeads);
                }
                QTensor t;
                int offset = 0;
                if (cacheName == null) {
                    // gate and up are stacked in the engine: [gate; up]
                    String gateName = name.replace("up_proj", "gate_proj");
                    if (name.endsWith("up_proj.weight")) {
                        offset = rows;
                    }
                    t = src.tensor(gateName.replace("mlp.gate_proj.weight", "mlp.gate_up"));
                } else {
                    t = src.tensor(name);
                }
                if (t.cols != cols || (cacheName != null && t.rows != rows)) {
                    bad++;
                    System.out.printf("BAD %-50s shape %dx%d, expected %dx%d%n", name, t.rows, t.cols, rows, cols);
                    continue;
                }
                double sse = 0;
                double ss = 0;
                float[] row = new float[cols];
                for (int r = 0; r < rows; r++) {
                    t.dequantizeRow(offset + r, row, 0);
                    for (int c = 0; c < cols; c++) {
                        double d = row[c] - want[r * cols + c];
                        sse += d * d;
                        ss += (double) want[r * cols + c] * want[r * cols + c];
                    }
                }
                double err = Math.sqrt(sse / ss);
                String type = t.type + (g.tensor(ggufName(name)).type().name().equals(t.type.name()) ? "" : " (from " + g.tensor(ggufName(name)).type() + ")");
                double[] agg = byType.computeIfAbsent(type, k -> new double[3]);
                agg[0]++;
                agg[1] = Math.max(agg[1], err);
                agg[2] += err;
                tensors++;
                double bound = switch (g.tensor(ggufName(name)).type()) {
                    case Q8_0, F16, BF16, F32 -> 0.01;
                    case Q6_K -> 0.03;
                    case Q5_0, Q5_1, Q5_K -> 0.06;
                    case Q4_K, Q4_0, Q4_1 -> 0.12;
                    case Q3_K -> 0.2;
                    default -> 0.4;
                };
                if (err > bound) {
                    bad++;
                    System.out.printf("BAD %-50s %s rel rms err %.4f > %.2f%n", name, type, err, bound);
                }
            }
        }
        byType.forEach((k, v) -> System.out.printf("%-22s %3d tensors, rel rms err: mean %.4f, worst %.4f%n", k, (int) v[0],
                v[2] / v[0], v[1]));
        System.out.println(tensors + " tensors compared, " + bad + " outside their format's bound");
        if (bad > 0) {
            System.exit(1);
        }
    }

    /** The GGUF name of a canonical (Hugging Face) tensor name, for the ggml type. */
    private static String ggufName(String hf) {
        if (hf.equals("model.embed_tokens.weight")) {
            return "token_embd.weight";
        }
        if (hf.equals("lm_head.weight")) {
            return "output.weight";
        }
        String b = "blk." + hf.replaceAll("model\\.layers\\.(\\d+)\\..*", "$1") + ".";
        String rest = hf.replaceAll("model\\.layers\\.\\d+\\.", "");
        return b + switch (rest) {
            case "self_attn.q_proj.weight" -> "attn_q.weight";
            case "self_attn.k_proj.weight" -> "attn_k.weight";
            case "self_attn.v_proj.weight" -> "attn_v.weight";
            case "self_attn.o_proj.weight" -> "attn_output.weight";
            case "mlp.gate_proj.weight" -> "ffn_gate.weight";
            case "mlp.up_proj.weight" -> "ffn_up.weight";
            case "mlp.down_proj.weight" -> "ffn_down.weight";
            default -> throw new IllegalArgumentException(hf);
        };
    }

    private static double rel(float[] want, float[] got) {
        double sse = 0;
        double ss = 0;
        for (int i = 0; i < want.length; i++) {
            double d = got[i] - want[i];
            sse += d * d;
            ss += (double) want[i] * want[i];
        }
        return Math.sqrt(sse / Math.max(ss, 1e-30));
    }

    private static float[] permute(float[] w, int rows, int cols, int nHead) {
        int hd = rows / nHead;
        int half = hd / 2;
        float[] out = new float[w.length];
        for (int h = 0; h < nHead; h++) {
            for (int j = 0; j < half; j++) {
                for (int c = 0; c < 2; c++) {
                    System.arraycopy(w, (h * hd + c * half + j) * cols, out, (h * hd + 2 * j + c) * cols, cols);
                }
            }
        }
        return out;
    }
}
