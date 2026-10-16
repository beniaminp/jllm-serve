package org.robomiri.jllm;

import org.robomiri.jllm.gguf.GgufFile;
import org.robomiri.jllm.model.Ggml;
import org.robomiri.jllm.model.Q2K;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Byte parity of the dequantizers with {@code gguf.quants} on the real blocks of a GGUF file: for every tensor that
 * {@code src/test/python/gguf_parity.py} dumped, the same rows dequantized here (the engine's in-place formats through
 * {@link QTensor#dequantizeRow}, the formats it only converts through {@link Ggml}) must equal the reference.
 *
 * <pre>
 * java ... org.robomiri.jllm.GgufRealParity model.gguf DUMPDIR
 * </pre>
 */
public final class GgufRealParity {

    public static void main(String[] args) throws Exception {
        Path dump = Path.of(args[1]);
        List<String> names = new ArrayList<>();
        try (var s = Files.list(dump)) {
            s.map(p -> p.getFileName().toString()).filter(n -> n.endsWith(".type")).sorted()
                    .forEach(n -> names.add(n.substring(0, n.length() - 5)));
        }
        int tensors = 0;
        int bad = 0;
        long weights = 0;
        long exact = 0;
        java.util.Map<String, long[]> perType = new java.util.TreeMap<>();
        try (GgufFile g = GgufFile.open(Path.of(args[0]))) {
            for (String name : names) {
                String type = Files.readString(dump.resolve(name + ".type")).trim();
                byte[] raw = Files.readAllBytes(dump.resolve(name + ".f32"));
                float[] ref = new float[raw.length / 4];
                ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(ref);
                GgufFile.Tensor t = g.tensor(name);
                int cols = (int) t.cols();
                int rows = ref.length / cols;
                MemorySegment seg = g.data(t);
                float[] got = new float[cols];
                long ex = 0;
                double worst = 0;
                for (int r = 0; r < rows; r++) {
                    long off = (long) r * t.rowBytes();
                    switch (type) {
                        case "Q4_0" -> QTensor.mapped(QType.Q4_0G, 1, cols, seg.asSlice(off, t.rowBytes())).dequantizeRow(0, got, 0);
                        case "Q8_0" -> QTensor.mapped(QType.Q8_0G, 1, cols, seg.asSlice(off, t.rowBytes())).dequantizeRow(0, got, 0);
                        case "Q4_K" -> QTensor.mapped(QType.Q4_K, 1, cols, seg.asSlice(off, t.rowBytes())).dequantizeRow(0, got, 0);
                        case "Q5_K" -> QTensor.mapped(QType.Q5_K, 1, cols, seg.asSlice(off, t.rowBytes())).dequantizeRow(0, got, 0);
                        case "Q2_K" -> Q2K.dequantizeRow(seg, off, cols, got, 0);
                        case "Q3_K" -> Ggml.dequantQ3_K(seg, off, cols, got, 0);
                        case "Q6_K" -> Ggml.dequantQ6_K(seg, off, cols, got, 0);
                        case "Q4_1" -> Ggml.dequantQ4_1(seg, off, cols, got, 0);
                        case "Q5_0" -> Ggml.dequantQ5_0(seg, off, cols, got, 0);
                        case "Q5_1" -> Ggml.dequantQ5_1(seg, off, cols, got, 0);
                        case "F32" -> {
                            for (int c = 0; c < cols; c++) {
                                got[c] = seg.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), off + 4L * c);
                            }
                        }
                        default -> throw new IllegalArgumentException(name + ": " + type);
                    }
                    for (int c = 0; c < cols; c++) {
                        float e = ref[r * cols + c];
                        if (got[c] == e) {
                            ex++;
                        }
                        worst = Math.max(worst, Math.abs(got[c] - e) / (Math.abs(e) + 1e-6));
                    }
                }
                tensors++;
                weights += (long) rows * cols;
                exact += ex;
                long[] agg = perType.computeIfAbsent(type, k -> new long[3]);
                agg[0]++;
                agg[1] += (long) rows * cols;
                agg[2] += ex;
                if (ex != (long) rows * cols && worst > 1e-6) {
                    bad++;
                    System.out.printf("DIFF %-34s %-5s %d/%d equal, worst rel err %.2e%n", name, type, ex, (long) rows * cols, worst);
                }
            }
        }
        perType.forEach((k, v) -> System.out.printf("%-5s %3d tensors, %d weights, %d bit-identical to gguf.quants%n", k, v[0], v[1], v[2]));
        System.out.println(tensors + " tensors, " + weights + " weights, " + exact + " bit-identical, " + bad + " tensors beyond 1e-6");
        if (bad > 0) {
            System.exit(1);
        }
    }
}
