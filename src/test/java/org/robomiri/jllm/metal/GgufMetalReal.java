package org.robomiri.jllm.metal;

import org.robomiri.jllm.gguf.GgufFile;
import org.robomiri.jllm.gguf.GgufWeights;
import org.robomiri.jllm.model.QTensor;

import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Random;

/**
 * Metal matmul against the dequantized weights on every GGUF-layout tensor of a real file: for each tensor, the
 * single-token and the tiled prefill kernel are compared with a double-precision product built from
 * {@link QTensor#dequantizeRow}. Usage: {@code GgufMetalReal model.gguf [maxTensors]}.
 */
public final class GgufMetalReal {

    public static void main(String[] args) throws Exception {
        int max = args.length > 1 ? Integer.parseInt(args[1]) : 1000;
        int bad = 0;
        int count = 0;
        try (GgufFile g = GgufFile.open(Path.of(args[0])); GgufWeights src = new GgufWeights(g);
             MetalKernels k = new MetalKernels(Metal.create(null), 1e-6f)) {
            for (String hf : g.tensors().keySet()) {
                String name = switch (hf) {
                    case "token_embd.weight" -> "model.embed_tokens.weight";
                    default -> hf.startsWith("blk.") ? canonical(hf) : null;
                };
                if (name == null || !src.has(name) || count >= max) {
                    continue;
                }
                GgufFile.Tensor gt = g.tensor(hf);
                if (gt.dims().length != 2 || gt.type().id < 2) {
                    continue;
                }
                QTensor t = src.tensor(name);
                Metal.Buffer w = k.wrap(t.data);
                int cols = t.cols;
                for (int n : new int[]{1, 9}) {
                    Random r = new Random(3);
                    float[] x = new float[n * cols];
                    for (int i = 0; i < x.length; i++) {
                        x[i] = r.nextFloat() * 2 - 1;
                    }
                    Metal.Buffer in = k.upload(x);
                    Metal.Buffer out = k.scratch((long) n * t.rows * 4);
                    k.begin();
                    k.matmul(t, w, in, 0, out, 0, n, false, null, 0f);
                    k.commitAndWait();
                    float[] got = new float[n * t.rows];
                    java.lang.foreign.MemorySegment.copy(out.contents(), ValueLayout.JAVA_FLOAT, 0, got, 0, got.length);
                    float[] row = new float[cols];
                    double worst = 0;
                    for (int rr = 0; rr < Math.min(t.rows, 64); rr++) {
                        t.dequantizeRow(rr, row, 0);
                        for (int tok = 0; tok < n; tok++) {
                            double want = 0;
                            double mag = 0;
                            for (int c = 0; c < cols; c++) {
                                double p = (double) row[c] * x[tok * cols + c];
                                want += p;
                                mag += Math.abs(p);
                            }
                            worst = Math.max(worst, Math.abs(got[tok * t.rows + rr] - want) / (mag + 1e-9));
                        }
                    }
                    count++;
                    if (worst > 1e-4) {
                        bad++;
                        System.out.printf("BAD %-32s %s %dx%d n=%d: max error relative to sum|w x| %.3e%n", hf, t.type, t.rows, t.cols, n, worst);
                    }
                }
            }
        }
        System.out.println(count + " tensor products compared, " + bad + " differ");
    }

    private static String canonical(String g) {
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("blk\\.(\\d+)\\.(.*)").matcher(g);
        if (!m.matches()) {
            return null;
        }
        String p = "model.layers." + m.group(1) + ".";
        return switch (m.group(2)) {
            case "attn_q.weight" -> p + "self_attn.q_proj.weight";
            case "attn_k.weight" -> p + "self_attn.k_proj.weight";
            case "attn_v.weight" -> p + "self_attn.v_proj.weight";
            case "attn_output.weight" -> p + "self_attn.o_proj.weight";
            case "ffn_down.weight" -> p + "mlp.down_proj.weight";
            default -> null;
        };
    }
}
