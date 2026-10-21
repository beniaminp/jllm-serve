package org.robomiri.jllm.cpu;

import org.robomiri.jllm.ModelLoader;
import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.model.DecoderLayer;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * The real tensors of a model (first layer and LM head): native and Java int8 dot products against a double-precision
 * reference of the same quantized weights times the same int8 activation. Both kernels should be at float-rounding
 * distance from the reference (relative to {@code Σ|w_i x_i|}).
 *
 * <pre>java ... org.robomiri.jllm.cpu.NativeRealCheck models/Qwen_Qwen3-0.6B</pre>
 */
public final class NativeRealCheck {

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]).toAbsolutePath();
        Settings st = Settings.load(new String[]{"--backend", "cpu", "--model.dir", dir.getParent().toString(),
                "--backend.ctx", "256", "--engine.speculative", "off", "--backend.cpu.int8-activations", "false"});
        NativeCpu n = NativeCpu.load(NativeCpu.locate(null));
        boolean ok = true;
        try (ModelLoader.Loaded l = ModelLoader.load(st, ModelLoader.hub(st), dir.getFileName().toString())) {
            ModelWeights w = l.weights();
            List<QTensor> ts = new ArrayList<>();
            List<String> names = new ArrayList<>();
            DecoderLayer d = (DecoderLayer) w.layers[0];
            for (var c : d.attn().getClass().getRecordComponents()) {
                if (c.getAccessor().invoke(d.attn()) instanceof QTensor q) {
                    ts.add(q);
                    names.add("attn." + c.getName());
                }
            }
            ts.add(d.wo());
            names.add("wo");
            for (var c : d.mlp().getClass().getRecordComponents()) {
                if (c.getAccessor().invoke(d.mlp()) instanceof QTensor q) {
                    ts.add(q);
                    names.add("mlp." + c.getName());
                }
            }
            ts.add(w.lmHead);
            names.add("lmHead");
            Random rnd = new Random(5);
            for (int k = 0; k < ts.size(); k++) {
                QTensor t = ts.get(k);
                int cols = t.cols;
                float[] x = new float[cols];
                for (int i = 0; i < cols; i++) {
                    x[i] = (float) rnd.nextGaussian();
                }
                short[] xq = new short[cols];
                float[] xs = new float[cols / QType.BLOCK];
                int[] sums = new int[xs.length];
                Int8Kernels.quantize(x, 0, cols, xq, xs);
                KQuantKernels.blockSums(xq, cols, sums);
                NativeCpu.Activation a = new NativeCpu.Activation();
                n.quantize(x, 0, cols, a);
                int rows = Math.min(t.rows, 4096);
                float[] nat = new float[rows];
                n.dotRows(t, 0, rows, a, 0, nat, 0, false);
                float[] row = new float[cols];
                double worstN = 0;
                double worstJ = 0;
                for (int r = 0; r < rows; r++) {
                    t.dequantizeRow(r, row, 0);
                    double ref = 0;
                    double mag = 0;
                    for (int i = 0; i < cols; i++) {
                        double xi = (double) a.get(i) * a.scale(i / QType.BLOCK);
                        ref += row[i] * xi;
                        mag += Math.abs(row[i] * xi);
                    }
                    float java = KQuantKernels.supports(t) ? KQuantKernels.dotRow(t, r, xq, xs, sums, 0)
                            : Int8Kernels.dotRow(t, r, xq, xs, 0);
                    worstN = Math.max(worstN, Math.abs(nat[r] - ref) / mag);
                    worstJ = Math.max(worstJ, Math.abs(java - ref) / mag);
                }
                System.out.printf("%-12s %s %dx%d: native %.2e, java %.2e (max |dot - ref| / sum|w x|)%n", names.get(k),
                        t.type, t.rows, cols, worstN, worstJ);
                ok &= worstN < 1e-5 && worstJ < 1e-5;
            }
        }
        System.out.println(ok ? "native real-weights check passed" : "native real-weights check FAILED");
        if (!ok) {
            System.exit(1);
        }
    }
}
