package org.robomiri.jllm.vision;

import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Manual check against llama.cpp: preprocesses and encodes an image and writes each chunk's embeddings as little-endian
 * f32 ({@code <out>.<i>.f32}, tiles first, the overview last, like mtmd's chunk order) for comparison with the files
 * the {@code mtmd} dump tool writes. Usage: {@code VisionDump <mmproj.gguf> <image> <out-prefix> [threads]}.
 */
public final class VisionDump {

    private VisionDump() {
    }

    public static void main(String[] a) throws Exception {
        Path mm = Path.of(a[0]);
        Rgb img = ImageDecoder.decode(Files.readAllBytes(Path.of(a[1])));
        int threads = a.length > 3 ? Integer.parseInt(a[3]) : Runtime.getRuntime().availableProcessors();
        String backend = System.getProperty("jllm.vision.backend", "cpu");
        try (ClipModel m = ClipModel.open(mm);
             VisionEncoder enc = VisionEncoders.create(m, new VisionContext.Options(backend, threads, Preprocess.Limits.NONE, 0))) {
            System.out.println(m.projector + " layers=" + m.nLayer + " embd=" + m.nEmbd + " image " + img.width() + "x" + img.height()
                    + " on " + enc.name());
            long t0 = System.nanoTime();
            Preprocess.Prepared pp = Preprocess.preprocess(m, img, Preprocess.Limits.NONE);
            System.out.printf("preprocess %.1f ms: %d tiles, grid %dx%d, overview %s%n", (System.nanoTime() - t0) / 1e6,
                    pp.tiles().size(), pp.gridX(), pp.gridY(), pp.overview() != null);
            int i = 0;
            java.util.List<Preprocess.Pixels> all = new java.util.ArrayList<>(pp.tiles());
            if (pp.overview() != null) {
                all.add(pp.overview());
            }
            for (Preprocess.Pixels px : all) {
                long t1 = System.nanoTime();
                VisionEncoder.Encoded e = enc.encode(px);
                if (Boolean.getBoolean("jllm.vision.twice")) { // the second run: weights uploaded, pipelines compiled
                    t1 = System.nanoTime();
                    e = enc.encode(px);
                }
                System.out.printf("chunk %d: %dx%d px -> %d tokens (%dx%d) in %.0f ms%n", i, px.width(), px.height(), e.tokens(),
                        e.nx(), e.ny(), (System.nanoTime() - t1) / 1e6);
                float[] flat = e.embd();
                if (e.deepstack() != null) { // like mtmd: per token, the main features then each deepstack layer's
                    int d = e.embd().length / e.tokens();
                    int k = e.deepstack().length + 1;
                    flat = new float[e.tokens() * d * k];
                    for (int tk = 0; tk < e.tokens(); tk++) {
                        System.arraycopy(e.embd(), tk * d, flat, tk * d * k, d);
                        for (int j = 0; j < k - 1; j++) {
                            System.arraycopy(e.deepstack()[j], tk * d, flat, tk * d * k + (j + 1) * d, d);
                        }
                    }
                }
                ByteBuffer bb = ByteBuffer.allocate(flat.length * 4).order(ByteOrder.LITTLE_ENDIAN);
                bb.asFloatBuffer().put(flat);
                Files.write(Path.of(a[2] + "." + i + ".f32"), bb.array());
                // also the preprocessed pixels, for comparison with other implementations
                ByteBuffer pb = ByteBuffer.allocate(px.hwc().length * 4).order(ByteOrder.LITTLE_ENDIAN);
                pb.asFloatBuffer().put(px.hwc());
                Files.write(Path.of(a[2] + "." + i + ".pix.f32"), pb.array());
                i++;
            }
        }
    }
}
