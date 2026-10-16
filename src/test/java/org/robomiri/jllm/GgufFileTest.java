package org.robomiri.jllm;

import org.robomiri.jllm.gguf.GgmlType;
import org.robomiri.jllm.gguf.GgufFile;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/** Checks of the GGUF reader: every value type, tensor geometry, alignment, split files and malformed input. */
public final class GgufFileTest {

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
        Path dir = Files.createTempDirectory("gguf-test");
        try {
            roundTrip(dir);
            alignment(dir);
            split(dir);
            malformed(dir);
        } finally {
            try (var s = Files.list(dir)) {
                for (Path p : s.toList()) {
                    Files.deleteIfExists(p);
                }
            }
            Files.deleteIfExists(dir);
        }
        System.out.println("gguf reader: " + passed + " checks passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        }
    }

    private static byte[] seq(int n, int seed) {
        byte[] b = new byte[n];
        for (int i = 0; i < n; i++) {
            b[i] = (byte) (i * 7 + seed);
        }
        return b;
    }

    private static GgufWriter sample() {
        return new GgufWriter()
                .str("general.architecture", "qwen3")
                .u32("qwen3.block_count", 28)
                .i32("neg", -5)
                .u64("big", 1L << 40)
                .f32("qwen3.rope.freq_base", 1000000f)
                .f64("d", 0.25)
                .bool("flag", true)
                .u8("u8", 200)
                .i16("i16", -300)
                .strings("tokens", "a", "bé", "<|im_end|>")
                .i32s("types", 1, 3, 4)
                .f32s("scores", 0.5f, -1.5f)
                .nested("nested", new int[]{1, 2}, new int[]{3})
                .tensor("blk.0.attn_q.weight", GgmlType.Q8_0, new long[]{64, 3}, seq(2 * 34 * 3, 1))
                .f32Tensor("blk.0.attn_norm.weight", new long[]{4}, new float[]{1, 2, 3, 4})
                .tensor("rope", GgmlType.F16, new long[]{8, 2, 2}, seq(8 * 2 * 2 * 2, 9));
    }

    private static void roundTrip(Path dir) throws Exception {
        Path p = dir.resolve("a.gguf");
        sample().write(p);
        try (GgufFile g = GgufFile.open(p)) {
            check(g.version() == 3, "version");
            check(g.str("general.architecture", "").equals("qwen3"), "string");
            check(g.num("qwen3.block_count", 0) == 28, "u32");
            check(g.num("neg", 0) == -5, "i32");
            check(g.num("big", 0) == 1L << 40, "u64");
            check(g.dbl("qwen3.rope.freq_base", 0) == 1000000.0, "f32");
            check(g.dbl("d", 0) == 0.25, "f64");
            check(g.bool("flag", false), "bool");
            check(g.num("u8", 0) == 200, "u8 is unsigned");
            check(g.num("i16", 0) == -300, "i16");
            check(Arrays.equals(g.strings("tokens"), new String[]{"a", "bé", "<|im_end|>"}), "string array (utf-8)");
            check(Arrays.equals(g.longs("types"), new long[]{1, 3, 4}), "int array");
            check(Arrays.equals(g.doubles("scores"), new double[]{0.5, -1.5}), "float array");
            check(g.get("nested") instanceof Object[] n && n.length == 2 && Arrays.equals((long[]) n[0], new long[]{1, 2})
                    && Arrays.equals((long[]) n[1], new long[]{3}), "nested arrays");
            check(g.num("missing", 7) == 7 && g.str("missing", "x").equals("x") && g.strings("flag") == null, "defaults");
            GgufFile.Tensor q = g.tensor("blk.0.attn_q.weight");
            check(q.type() == GgmlType.Q8_0 && q.cols() == 64 && q.rows() == 3 && q.size() == 2 * 34 * 3
                    && q.rowBytes() == 68, "q8_0 geometry (ne[0] = columns)");
            check(Arrays.equals(g.data(q).toArray(ValueLayout.JAVA_BYTE), seq(2 * 34 * 3, 1)), "tensor bytes");
            GgufFile.Tensor r = g.tensor("rope");
            check(r.rows() == 4 && r.cols() == 8 && r.elements() == 32, "3-d tensor: rows are the product of the outer dims");
            check(Arrays.equals(g.data(r).toArray(ValueLayout.JAVA_BYTE), seq(64, 9)), "f16 tensor bytes");
            MemorySegment n = g.data(g.tensor("blk.0.attn_norm.weight"));
            check(n.get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), 8) == 3f, "f32 tensor");
            check(g.tensors().size() == 3 && g.hasTensor("rope") && !g.hasTensor("nope"), "directory");
            try {
                g.tensor("nope");
                check(false, "missing tensor throws");
            } catch (IllegalArgumentException e) {
                check(e.getMessage().contains("nope"), "missing tensor message");
            }
        }
    }

    private static void alignment(Path dir) throws Exception {
        for (long a : new long[]{16, 64, 4096}) {
            Path p = dir.resolve("al" + a + ".gguf");
            sample().u32("general.alignment", a).alignment(a).write(p);
            try (GgufFile g = GgufFile.open(p)) {
                check(g.alignment() == a, "alignment " + a);
                GgufFile.Tensor q = g.tensor("blk.0.attn_q.weight");
                check(q.offset() % a == 0, "data offset aligned to " + a);
                check(Arrays.equals(g.data(q).toArray(ValueLayout.JAVA_BYTE), seq(2 * 34 * 3, 1)), "bytes with alignment " + a);
                check(Arrays.equals(g.data(g.tensor("rope")).toArray(ValueLayout.JAVA_BYTE), seq(64, 9)), "bytes (2) with alignment " + a);
            }
        }
        Path p = dir.resolve("badal.gguf");
        sample().u32("general.alignment", 48).write(p);
        expectFail(p, "alignment", "non power of two alignment");
        sample().u32("general.alignment", 0).write(p);
        expectFail(p, "alignment", "zero alignment");
    }

    private static void split(Path dir) throws Exception {
        Path p1 = dir.resolve("m-00001-of-00002.gguf");
        Path p2 = dir.resolve("m-00002-of-00002.gguf");
        new GgufWriter().str("general.architecture", "llama").u32("split.no", 0).u32("split.count", 2)
                .tensor("a", GgmlType.Q4_0, new long[]{32}, seq(18, 3)).write(p1);
        new GgufWriter().u32("split.no", 1).u32("split.count", 2)
                .tensor("b", GgmlType.Q4_0, new long[]{64}, seq(36, 5)).write(p2);
        for (Path open : new Path[]{p1, p2}) {
            try (GgufFile g = GgufFile.open(open)) {
                check(g.tensors().size() == 2 && g.paths().size() == 2, "split: both parts via " + open.getFileName());
                check(g.str("general.architecture", "").equals("llama"), "split: metadata of part 1");
                check(Arrays.equals(g.data(g.tensor("a")).toArray(ValueLayout.JAVA_BYTE), seq(18, 3)), "split: tensor of part 1");
                check(Arrays.equals(g.data(g.tensor("b")).toArray(ValueLayout.JAVA_BYTE), seq(36, 5)), "split: tensor of part 2");
            }
        }
        check(GgufFile.splitNames("m-00002-of-00003.gguf").equals(java.util.List.of("m-00001-of-00003.gguf",
                "m-00002-of-00003.gguf", "m-00003-of-00003.gguf")), "split names");
        check(GgufFile.splitNames("plain.gguf").equals(java.util.List.of("plain.gguf")), "plain name");
        Files.delete(p2);
        expectFail(p1, "m-00002-of-00002.gguf is missing", "missing split part");
    }

    private static void expectFail(Path p, String contains, String what) {
        try (GgufFile g = GgufFile.open(p)) {
            check(false, what + ": expected an IOException");
        } catch (IOException e) {
            check(e.getMessage().contains(contains), what + ": message '" + e.getMessage() + "' should contain '" + contains + "'");
        } catch (RuntimeException e) {
            check(false, what + ": " + e);
        }
    }

    private static void expectFail(byte[] bytes, Path dir, String contains, String what) throws IOException {
        Path p = dir.resolve("bad.gguf");
        Files.write(p, bytes);
        expectFail(p, contains, what);
    }

    private static void malformed(Path dir) throws Exception {
        byte[] good = sample().toBytes();
        // every truncation fails with an IOException, never anything else
        int bad = 0;
        for (int n = 0; n < good.length; n += (n < 400 ? 1 : 7)) {
            Path p = dir.resolve("trunc.gguf");
            Files.write(p, Arrays.copyOf(good, n));
            try (GgufFile g = GgufFile.open(p)) {
                // a truncation inside the final padding-free tensor data is still caught by the bounds check
                bad++;
            } catch (IOException e) {
                // expected
            } catch (RuntimeException e) {
                check(false, "truncated to " + n + " bytes threw " + e);
            }
        }
        check(bad == 0, "no truncated prefix opens (" + bad + " did)");

        byte[] m = good.clone();
        m[0] = 'X';
        expectFail(m, dir, "not a GGUF file", "bad magic");
        m = good.clone();
        m[4] = 1;
        expectFail(m, dir, "version 1", "version 1");
        m = good.clone();
        m[4] = 9;
        expectFail(m, dir, "unsupported GGUF version", "version 9");
        m = good.clone();
        ByteBuffer.wrap(m).order(ByteOrder.LITTLE_ENDIAN).putLong(8, 1L << 50);
        expectFail(m, dir, "implausible counts", "huge tensor count");
        m = good.clone();
        ByteBuffer.wrap(m).order(ByteOrder.LITTLE_ENDIAN).putLong(16, -1);
        expectFail(m, dir, "invalid metadata count", "negative kv count");

        // tensor directory violations, built with the writer
        expectFail(new GgufWriter().tensor("t", GgmlType.Q8_0, new long[]{33}, seq(34, 0)).toBytes(), dir,
                "not a multiple", "row length not a multiple of the block size");
        expectFail(new GgufWriter().tensor("t", GgmlType.F32, new long[]{Long.MAX_VALUE / 2, 4}, new byte[0]).toBytes(), dir,
                "overflows", "shape overflow");
        expectFail(new GgufWriter().tensor("t", GgmlType.F32, new long[]{0}, new byte[0]).toBytes(), dir,
                "non-positive", "zero dimension");
        expectFail(new GgufWriter().f32Tensor("t", new long[]{4}, new float[]{1, 2, 3, 4}).f32Tensor("t", new long[]{4},
                new float[4]).toBytes(), dir, "duplicate tensor", "duplicate tensor");
        expectFail(new GgufWriter().u32("k", 1).u32("k", 2).toBytes(), dir, "duplicate metadata key", "duplicate key");
        // tensor data cut short: the last tensor extends beyond the file
        byte[] full = new GgufWriter().f32Tensor("t", new long[]{64}, new float[64]).toBytes();
        expectFail(Arrays.copyOf(full, full.length - 4), dir, "extends beyond the end", "tensor data truncated");
        // an offset that is not a multiple of the alignment: patch the offset of the second tensor
        byte[] two = new GgufWriter().f32Tensor("a", new long[]{3}, new float[3]).f32Tensor("b", new long[]{3}, new float[3]).toBytes();
        ByteBuffer bb = ByteBuffer.wrap(two).order(ByteOrder.LITTLE_ENDIAN);
        // header: 24 + info a (8+1 name, 4 nd, 8 dim, 4 type, 8 offset) then info b: name at 24+33=57; offset field at 57+9+4+8+4
        bb.putLong(57 + 9 + 4 + 8 + 4, 12);
        expectFail(two, dir, "alignment", "misaligned tensor offset");
        // overlapping tensors: second tensor placed on top of the first
        two = new GgufWriter().f32Tensor("a", new long[]{16}, new float[16]).f32Tensor("b", new long[]{16}, new float[16]).toBytes();
        bb = ByteBuffer.wrap(two).order(ByteOrder.LITTLE_ENDIAN);
        bb.putLong(24 + 9 + 4 + 8 + 4 + 8 + 9 + 4 + 8 + 4, 32);
        expectFail(two, dir, "overlap", "overlapping tensors");
        // metadata: array length larger than the file, string length larger than the file
        byte[] arr = new GgufWriter().strings("t", "x").toBytes();
        bb = ByteBuffer.wrap(arr).order(ByteOrder.LITTLE_ENDIAN);
        bb.putLong(24 + 9 + 4 + 4, 1L << 40);
        expectFail(arr, dir, "claims", "array length beyond the file");
        byte[] str = new GgufWriter().str("t", "xy").toBytes();
        bb = ByteBuffer.wrap(str).order(ByteOrder.LITTLE_ENDIAN);
        bb.putLong(24 + 9 + 4, 1L << 40);
        expectFail(str, dir, "truncated header", "string length beyond the file");
        // unknown metadata value type and unknown ggml type
        byte[] ty = new GgufWriter().u32("k", 1).toBytes();
        ByteBuffer.wrap(ty).order(ByteOrder.LITTLE_ENDIAN).putInt(24 + 9, 99);
        expectFail(ty, dir, "unknown value type", "unknown value type");
        byte[] gt = new GgufWriter().f32Tensor("a", new long[]{4}, new float[4]).toBytes();
        ByteBuffer.wrap(gt).order(ByteOrder.LITTLE_ENDIAN).putInt(24 + 9 + 4 + 8, 777);
        expectFail(gt, dir, "unknown ggml type id 777", "unknown tensor type");
        expectFail(dir.resolve("absent.gguf"), "no such GGUF file", "missing file");
    }
}
