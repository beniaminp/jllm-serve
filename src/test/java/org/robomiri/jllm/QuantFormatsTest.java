package org.robomiri.jllm;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Fp8Quant;
import org.robomiri.jllm.model.MlxQuant;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.ShardProvider;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Pre-quantized checkpoint formats the converter dequantizes (no download, CPU backend):
 * <ul>
 *   <li>FP8: every {@code F8_E4M3} / {@code F8_E5M2} byte against an independent decoder written from the format's
 *       definition (subnormals, NaN, infinities, signed zero); block-wise {@code weight_scale_inv} with partial blocks
 *       and a per-tensor scale;</li>
 *   <li>MLX {@code mxfp4} / {@code mxfp8}: hand-computed packed words with E8M0 scales;</li>
 *   <li>end to end: a tiny Llama written as an FP8 checkpoint (and as MLX mxfp4 / mxfp8) by a reference encoder,
 *       and as an F32 checkpoint holding the values the encoder's output means; both are converted to Q8_0 and
 *       must give identical logits.</li>
 * </ul>
 */
public final class QuantFormatsTest {

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

    public static void main(String[] args) throws Exception {
        fp8Codes();
        fp8Blocks();
        mx();
        Path dir = Files.createTempDirectory("quant-formats-test");
        try {
            tinyModels(dir);
        } finally {
            try (var s = Files.walk(dir)) {
                s.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
        System.out.println("quant formats: " + passed + " checks passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        }
    }

    // ------------------------------------------------------------------ reference codecs (from the definitions)

    /** E4M3 ("fn"): bias 7, no infinities, S.1111.111 is NaN; exponent field 0 is subnormal, m/8 * 2^-6. */
    static double refE4m3(int b) {
        int s = (b >> 7) & 1;
        int e = (b >> 3) & 15;
        int m = b & 7;
        double v;
        if (e == 15 && m == 7) {
            return Double.NaN;
        } else if (e == 0) {
            v = m / 8.0 * Math.pow(2, -6);
        } else {
            v = (1 + m / 8.0) * Math.pow(2, e - 7);
        }
        return s == 1 ? -v : v;
    }

    /** E5M2: IEEE-style, bias 15, exponent 31 is infinity (m = 0) or NaN. */
    static double refE5m2(int b) {
        int s = (b >> 7) & 1;
        int e = (b >> 2) & 31;
        int m = b & 3;
        double v;
        if (e == 31) {
            v = m == 0 ? Double.POSITIVE_INFINITY : Double.NaN;
        } else if (e == 0) {
            v = m / 4.0 * Math.pow(2, -14);
        } else {
            v = (1 + m / 4.0) * Math.pow(2, e - 15);
        }
        return s == 1 ? -v : v;
    }

    /** Round to the nearest finite E4M3 value (ties to the even code), saturating at ±448. */
    static int encodeE4m3(float x) {
        int sign = x < 0 || (x == 0 && 1 / x < 0) ? 0x80 : 0;
        double a = Math.abs((double) x);
        int best = 0;
        double bestErr = Double.MAX_VALUE;
        for (int c = 0; c <= 0x7E; c++) {
            double err = Math.abs(refE4m3(c) - a);
            if (err < bestErr || (err == bestErr && (c & 1) == 0)) {
                best = c;
                bestErr = err;
            }
        }
        return sign | best;
    }

    static final double[] FP4 = {0, 0.5, 1, 1.5, 2, 3, 4, 6};

    /** Round to the nearest E2M1 code (ties to even), saturating at ±6. */
    static int encodeE2m1(double x) {
        int best = 0;
        double bestErr = Double.MAX_VALUE;
        for (int c = 0; c < 8; c++) {
            double err = Math.abs(FP4[c] - Math.abs(x));
            if (err < bestErr || (err == bestErr && (c & 1) == 0)) {
                best = c;
                bestErr = err;
            }
        }
        return (x < 0 ? 8 : 0) | best;
    }

    // ------------------------------------------------------------------ unit checks

    static void fp8Codes() {
        boolean e4 = true;
        boolean e5 = true;
        for (int b = 0; b < 256; b++) {
            double want = refE4m3(b);
            float got = SafeTensors.e4m3ToFloat(b);
            e4 &= Double.isNaN(want) ? Float.isNaN(got) : got == want && (want != 0 || (1 / got < 0) == ((b & 0x80) != 0));
            want = refE5m2(b);
            got = SafeTensors.e5m2ToFloat(b);
            e5 &= Double.isNaN(want) ? Float.isNaN(got) : got == want && (want != 0 || (1 / got < 0) == ((b & 0x80) != 0));
        }
        check(e4, "all 256 E4M3 codes decode exactly (incl. subnormals, NaN, -0)");
        check(e5, "all 256 E5M2 codes decode exactly (incl. subnormals, inf, NaN, -0)");
        // hand-computed corners
        check(SafeTensors.e4m3ToFloat(0x7E) == 448f && SafeTensors.e4m3ToFloat(0x01) == 0.001953125f
                        && SafeTensors.e4m3ToFloat(0x08) == 0.015625f && SafeTensors.e4m3ToFloat(0x38) == 1f
                        && SafeTensors.e4m3ToFloat(0xC4) == -3f && Float.isNaN(SafeTensors.e4m3ToFloat(0x7F))
                        && Float.isNaN(SafeTensors.e4m3ToFloat(0xFF)) && SafeTensors.e4m3ToFloat(0x78) == 256f,
                "E4M3 max 448, min subnormal 2^-9, min normal 2^-6, 1.0 = 0x38, -3 = 0xC4, 256 = 0x78, NaN 0x7F/0xFF");
        check(SafeTensors.e5m2ToFloat(0x7B) == 57344f && SafeTensors.e5m2ToFloat(0x7C) == Float.POSITIVE_INFINITY
                        && SafeTensors.e5m2ToFloat(0x01) == (float) Math.pow(2, -16) && SafeTensors.e5m2ToFloat(0x3C) == 1f,
                "E5M2 max 57344, inf 0x7C, min subnormal 2^-16, 1.0 = 0x3C");
        check(encodeE4m3(1f) == 0x38 && encodeE4m3(-3f) == 0xC4 && encodeE4m3(1000f) == 0x7E && encodeE4m3(1.0625f) == 0x38
                && encodeE4m3(1.1875f) == 0x3A, "reference encoder: exact, saturating, ties to even");
    }

    /** A segment holding {@code bytes}. */
    static MemorySegment seg(byte[] bytes) {
        MemorySegment s = Arena.ofAuto().allocate(Math.max(1, bytes.length));
        MemorySegment.copy(bytes, 0, s, ValueLayout.JAVA_BYTE, 0, bytes.length);
        return s;
    }

    static byte[] f32Bytes(float[] v) {
        ByteBuffer b = ByteBuffer.allocate(4 * v.length).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : v) {
            b.putFloat(f);
        }
        return b.array();
    }

    static void fp8Blocks() throws IOException {
        // [5, 6] weight, 2x4 blocks: grid [3, 2], the last block row and column partial. Codes 0x38 + (r*6+c) % 8
        // = 1, 1.125, ..., 1.875; the scale of block (i, j) is 2^i * (j + 1) (exact products).
        int rows = 5;
        int cols = 6;
        byte[] w = new byte[rows * cols];
        for (int i = 0; i < w.length; i++) {
            w[i] = (byte) (0x38 + i % 8);
        }
        float[] scale = new float[3 * 2];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                scale[i * 2 + j] = (1 << i) * (j + 1);
            }
        }
        var wt = new SafeTensors.Tensor("x.weight", SafeTensors.DType.F8_E4M3, new long[]{rows, cols}, seg(w));
        var st = new SafeTensors.Tensor("x.weight_scale_inv", SafeTensors.DType.F32, new long[]{3, 2}, seg(f32Bytes(scale)));
        float[] got = new Fp8Quant(2, 4).dequantized("x.weight", wt, st).toFloatArray();
        boolean ok = got.length == 30;
        for (int r = 0; r < rows && ok; r++) {
            for (int c = 0; c < cols; c++) {
                double want = (1 + ((r * cols + c) % 8) / 8.0) * (1 << (r / 2)) * (c / 4 + 1);
                ok &= got[r * cols + c] == want;
            }
        }
        check(ok, "FP8 block dequantization w = fp8 * scale_inv[r/2][c/4] with partial blocks");
        // reading a slice that starts mid-row (the converter reads row by row, other callers may not)
        float[] part = new float[7];
        new Fp8Quant(2, 4).dequantized("x.weight", wt, st).toFloat(9, part, 0, 7);
        check(part[0] == got[9] && part[6] == got[15], "FP8 dequantized view: arbitrary start");
        // per-tensor scale (one element)
        var one = new SafeTensors.Tensor("s", SafeTensors.DType.F32, new long[]{}, seg(f32Bytes(new float[]{0.5f})));
        got = new Fp8Quant(128, 128).dequantized("x.weight", wt, one).toFloatArray();
        check(got[0] == 0.5f && got[7] == 0.9375f, "FP8 per-tensor scale");
        // the default block: [130, 40] has a grid of [2, 1]
        var big = new SafeTensors.Tensor("y", SafeTensors.DType.F8_E5M2, new long[]{130, 40}, seg(new byte[130 * 40]));
        try {
            new Fp8Quant(128, 128).dequantized("y", big, st);
            check(false, "a [3, 2] scale for a [130, 40] weight must be refused");
        } catch (IOException e) {
            check(e.getMessage().contains("expected [2, 1]"), "scale grid mismatch: " + e.getMessage());
        }
        try {
            new Fp8Quant(2, 4).dequantized("x", st, st);
            check(false, "an F32 'weight' is not FP8");
        } catch (IOException e) {
            check(e.getMessage().contains("F8_E4M3"), "non-FP8 weight refused: " + e.getMessage());
        }
    }

    /** Words of 4-bit codes, lowest bits first, as MLX packs them. */
    static int word4(int... codes) {
        int w = 0;
        for (int i = 0; i < 8; i++) {
            w |= (codes[i] & 0xF) << (4 * i);
        }
        return w;
    }

    static MlxQuant mlx(Path dir, String mode, int group, int bits) throws IOException {
        Files.writeString(dir.resolve("config.json"), "{\"quantization\": {\"group_size\": " + group + ", \"bits\": "
                + bits + ", \"mode\": \"" + mode + "\"}}");
        return MlxQuant.read(dir);
    }

    static void mx() throws IOException {
        check(MlxQuant.e2m1ToFloat(0) == 0 && MlxQuant.e2m1ToFloat(7) == 6 && MlxQuant.e2m1ToFloat(0xB) == -1.5f
                        && MlxQuant.e2m1ToFloat(3) == 1.5f && MlxQuant.e8m0ToFloat(127) == 1 && MlxQuant.e8m0ToFloat(130) == 8
                        && MlxQuant.e8m0ToFloat(0) == (float) Math.pow(2, -127) && Float.isNaN(MlxQuant.e8m0ToFloat(255)),
                "E2M1 codes 0, 7 = 6, 0xB = -1.5; E8M0 127 = 1, 130 = 8, 0 = 2^-127, 255 = NaN");
        Path tmp = Files.createTempDirectory("jllm-mx");
        try {
            MlxQuant q = mlx(tmp, "mxfp4", 32, 4);
            check(q.mode == MlxQuant.Mode.MXFP4, "mxfp4 mode read");
            // one row of 64: group 0 has scale 2^-1 and the codes 0..7 then 8..15 (signs) repeated twice;
            // group 1 has scale 2^2 and the codes 7, 6, ..., 0 repeated
            int[] words = new int[8];
            for (int k = 0; k < 4; k++) {
                words[k] = word4(k % 2 == 0 ? new int[]{0, 1, 2, 3, 4, 5, 6, 7} : new int[]{8, 9, 10, 11, 12, 13, 14, 15});
                words[4 + k] = word4(7, 6, 5, 4, 3, 2, 1, 0);
            }
            ByteBuffer wb = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN);
            for (int w : words) {
                wb.putInt(w);
            }
            var packed = new SafeTensors.Tensor("x.weight", SafeTensors.DType.U32, new long[]{1, 8}, seg(wb.array()));
            var scales = new SafeTensors.Tensor("x.scales", SafeTensors.DType.U8, new long[]{1, 2}, seg(new byte[]{126, (byte) 129}));
            float[] v = q.dequantized("x.weight", packed, scales, null).toFloatArray();
            float[] want0 = {0, 0.25f, 0.5f, 0.75f, 1, 1.5f, 2, 3, -0f, -0.25f, -0.5f, -0.75f, -1, -1.5f, -2, -3};
            float[] want1 = {24, 16, 12, 8, 6, 4, 2, 0};
            boolean ok = v.length == 64;
            for (int c = 0; c < 64 && ok; c++) {
                float want = c < 32 ? want0[c % 16] : want1[c % 8];
                ok &= Float.compare(v[c], want) == 0;
            }
            check(ok, "mxfp4 dequantization: E2M1 value * 2^(scale - 127), hand-computed row");

            q = mlx(tmp, "mxfp8", 32, 8);
            // 8-bit codes, 4 per word: 0x38 (1.0), 0xC4 (-3), 0x01 (2^-9), 0x7E (448); scale 2^3
            ByteBuffer b8 = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < 8; i++) {
                b8.putInt(0x7E01C438);
            }
            packed = new SafeTensors.Tensor("x.weight", SafeTensors.DType.U32, new long[]{1, 8}, seg(b8.array()));
            scales = new SafeTensors.Tensor("x.scales", SafeTensors.DType.U8, new long[]{1, 1}, seg(new byte[]{(byte) 130}));
            v = q.dequantized("x.weight", packed, scales, null).toFloatArray();
            check(v.length == 32 && v[0] == 8 && v[1] == -24 && v[2] == 0.015625f && v[3] == 3584 && v[31] == 3584,
                    "mxfp8 dequantization: E4M3 value * 2^(scale - 127)");
            try {
                mlx(tmp, "nvfp4", 16, 4);
                check(false, "nvfp4 must be refused");
            } catch (IOException e) {
                check(e.getMessage().contains("nvfp4") && e.getMessage().contains("mxfp4"), "nvfp4 refused: " + e.getMessage());
            }
            try {
                mlx(tmp, "mxfp4", 64, 4);
                check(false, "mxfp4 with groups of 64 must be refused");
            } catch (IOException e) {
                check(e.getMessage().contains("group_size 32"), "mxfp4 group size: " + e.getMessage());
            }
        } finally {
            Files.deleteIfExists(tmp.resolve("config.json"));
            Files.deleteIfExists(tmp);
        }
    }

    // ------------------------------------------------------------------ tiny models end to end

    /** One safetensors entry. */
    record Entry(String dtype, long[] shape, byte[] data) {
    }

    static void write(Path file, Map<String, Entry> tensors) throws IOException {
        Map<String, Object> header = new LinkedHashMap<>();
        long off = 0;
        for (Map.Entry<String, Entry> e : tensors.entrySet()) {
            List<Object> dims = new ArrayList<>();
            for (long d : e.getValue().shape()) {
                dims.add(d);
            }
            long len = e.getValue().data().length;
            header.put(e.getKey(), Json.obj("dtype", e.getValue().dtype(), "shape", dims, "data_offsets", List.of(off, off + len)));
            off += len;
        }
        byte[] h = Json.write(header).getBytes(StandardCharsets.UTF_8);
        ByteBuffer b = ByteBuffer.allocate((int) (8 + h.length + off)).order(ByteOrder.LITTLE_ENDIAN);
        b.putLong(h.length).put(h);
        for (Entry e : tensors.values()) {
            b.put(e.data());
        }
        Files.write(file, b.array());
    }

    /**
     * FP8 with {@code [bh, bw]} blocks: per block {@code s = amax / 448}, codes {@code encode(w / s)}, the stored
     * {@code weight_scale_inv} is {@code s}; {@code ref} receives {@code fp8(code) * s} (the same f32 product).
     */
    static void fp8Matrix(String name, float[] w, long[] shape, int bh, int bw, Map<String, Entry> out, float[] ref) {
        int rows = (int) shape[0];
        int cols = (int) shape[1];
        int gr = (rows + bh - 1) / bh;
        int gc = (cols + bw - 1) / bw;
        float[] s = new float[gr * gc];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int k = (r / bh) * gc + c / bw;
                s[k] = Math.max(s[k], Math.abs(w[r * cols + c]) / 448f);
            }
        }
        byte[] codes = new byte[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                float sc = s[(r / bh) * gc + c / bw];
                int code = encodeE4m3(sc == 0 ? 0 : w[r * cols + c] / sc);
                codes[r * cols + c] = (byte) code;
                ref[r * cols + c] = (float) refE4m3(code) * sc;
            }
        }
        out.put(name, new Entry("F8_E4M3", shape, codes));
        out.put(name + "_scale_inv", new Entry("F32", new long[]{gr, gc}, f32Bytes(s)));
    }

    /** MLX mxfp4 / mxfp8: per 32 columns the E8M0 exponent {@code floor(log2 amax) - emax}, codes of {@code w / 2^e}. */
    static void mxMatrix(String name, float[] w, long[] shape, boolean fp4, Map<String, Entry> out, float[] ref) {
        int rows = (int) shape[0];
        int cols = (int) shape[1];
        int bits = fp4 ? 4 : 8;
        int groups = cols / 32;
        int perWord = 32 / bits;
        ByteBuffer packed = ByteBuffer.allocate(rows * cols * bits / 8).order(ByteOrder.LITTLE_ENDIAN);
        byte[] scales = new byte[rows * groups];
        for (int r = 0; r < rows; r++) {
            for (int g = 0; g < groups; g++) {
                float amax = 0;
                for (int c = g * 32; c < g * 32 + 32; c++) {
                    amax = Math.max(amax, Math.abs(w[r * cols + c]));
                }
                int e = amax == 0 ? -127 : Math.getExponent(amax) - (fp4 ? 2 : 8); // emax of E2M1 is 2, of E4M3 8
                scales[r * groups + g] = (byte) (e + 127);
                double scale = Math.pow(2, e);
                for (int c = g * 32; c < g * 32 + 32; c++) {
                    int code = fp4 ? encodeE2m1(w[r * cols + c] / scale) : encodeE4m3((float) (w[r * cols + c] / scale));
                    double v = fp4 ? (code >= 8 ? -FP4[code - 8] : FP4[code]) : refE4m3(code);
                    ref[r * cols + c] = (float) (v * scale);
                    int word = (r * cols + c) / perWord;
                    packed.putInt(word * 4, packed.getInt(word * 4) | (code << (((r * cols + c) % perWord) * bits)));
                }
            }
        }
        out.put(name, new Entry("U32", new long[]{rows, (long) cols * bits / 32}, packed.array()));
        out.put(name.substring(0, name.length() - ".weight".length()) + ".scales", new Entry("U8", new long[]{rows, groups}, scales));
    }

    static void tinyModels(Path root) throws Exception {
        GgufModelTest.Tiny t = GgufModelTest.tiny("llama", 7);
        int[] prompt = {1, 5, 17, 33, 2, 40, 9, 77};
        for (String format : new String[]{"fp8", "mxfp4", "mxfp8"}) {
            Map<String, Object> config = new LinkedHashMap<>(t.config());
            Map<String, Entry> quantized = new LinkedHashMap<>();
            Map<String, Entry> reference = new LinkedHashMap<>();
            if (format.equals("fp8")) {
                config.put("quantization_config", Json.obj("quant_method", "fp8", "weight_block_size", List.of(24L, 40L)));
            } else {
                config.put("quantization", Json.obj("group_size", 32L, "bits", format.equals("mxfp4") ? 4L : 8L, "mode", format));
            }
            int n = 0;
            for (Map.Entry<String, float[]> e : t.hf().entrySet()) {
                String name = e.getKey();
                long[] shape = t.shapes().get(name);
                float[] w = e.getValue();
                boolean embedOrHead = name.equals("model.embed_tokens.weight") || name.equals("lm_head.weight");
                // FP8 releases keep the embedding and head in bf16; MLX quantizes them too
                boolean quantize = shape.length == 2 && (format.equals("fp8") ? !embedOrHead : true);
                float[] ref = w;
                if (quantize) {
                    ref = new float[w.length];
                    if (format.equals("fp8")) {
                        fp8Matrix(name, w, shape, 24, 40, quantized, ref);
                    } else {
                        mxMatrix(name, w, shape, format.equals("mxfp4"), quantized, ref);
                    }
                    n++;
                } else {
                    quantized.put(name, new Entry("F32", shape, f32Bytes(w)));
                }
                reference.put(name, new Entry("F32", shape, f32Bytes(ref)));
            }
            Path qd = Files.createDirectories(root.resolve(format));
            Path rd = Files.createDirectories(root.resolve(format + "-ref"));
            Files.writeString(qd.resolve("config.json"), Json.write(config));
            Files.writeString(rd.resolve("config.json"), Json.write(t.config()));
            write(qd.resolve("model.safetensors"), quantized);
            write(rd.resolve("model.safetensors"), reference);
            float[][] a;
            float[][] b;
            try (ModelWeights w = ModelWeights.load(qd, Architecture.load(qd), QType.Q8_0, ShardProvider.local(qd))) {
                a = GgufModelTest.logits(w, prompt, 4);
            }
            try (ModelWeights w = ModelWeights.load(rd, Architecture.load(rd), QType.Q8_0, ShardProvider.local(rd))) {
                b = GgufModelTest.logits(w, prompt, 4);
            }
            double diff = GgufModelTest.maxDiff(b, a);
            // the same Q8_0 weights either way; the last float bits of a run can still change when the JIT compiles the
            // Vector API kernels between the two runs (interpreted and intrinsic lane reductions add in different orders)
            check(diff <= 1e-5 && n > 0, format + ": " + n + " quantized matrices converted, logits equal to the F32 checkpoint"
                    + " of the same values: max diff " + diff);
            // and the quantization is real: the logits differ from those of the original weights
            Path od = Files.createDirectories(root.resolve(format + "-orig"));
            Files.writeString(od.resolve("config.json"), Json.write(t.config()));
            GgufModelTest.writeSafetensors(od.resolve("model.safetensors"), t.hf(), t.shapes());
            try (ModelWeights w = ModelWeights.load(od, Architecture.load(od), QType.Q8_0, ShardProvider.local(od))) {
                double vs = GgufModelTest.maxDiff(GgufModelTest.logits(w, prompt, 4), a);
                check(vs > 1e-4 && vs < (format.equals("mxfp4") ? 2 : 0.5), format + ": vs the unquantized weights "
                        + vs + " (quantization noise)");
                System.out.printf("%-6s checkpoint vs its F32 dequantization: max logit diff %.1e; vs the original weights %.2e%n",
                        format, diff, vs);
            }
        }
    }
}
