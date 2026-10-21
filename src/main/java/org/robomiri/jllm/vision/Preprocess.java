package org.robomiri.jllm.vision;

import org.robomiri.jllm.vision.Resize.Algo;
import org.robomiri.jllm.vision.Resize.Pad;
import org.robomiri.jllm.vision.Resize.Size;
import org.robomiri.jllm.vision.Resize.SizeRule;

import java.util.ArrayList;
import java.util.List;

/**
 * Turns a picture into the float images the vision transformer takes, the way llama.cpp's {@code mtmd} does for each
 * projector type (so the same file gives the same numbers):
 * <ul>
 *   <li>{@code idefics3} (SmolVLM): the picture is fitted to a multiple of the tile size (512), cut into tiles, and a
 *       tile-sized overview of the whole picture is added;</li>
 *   <li>{@code gemma3}: one square image of the encoder's size, aspect ratio kept, black padding;</li>
 *   <li>Qwen-VL: one image whose sides are multiples of {@code patch * merge}, within a token budget, aspect ratio
 *       kept, black padding for the rounding.</li>
 * </ul>
 * Pixels are scaled to [0, 1] and normalised with the file's mean and std.
 */
public final class Preprocess {

    /** A normalised float image, {@code height x width} pixels, 3 floats (R, G, B) per pixel, row by row. */
    public record Pixels(int width, int height, float[] hwc) {
    }

    /**
     * The result: the images to encode ({@code tiles} in row-major grid order, then the optional {@code overview}
     * is kept apart because the prompt layout puts it elsewhere).
     */
    public record Prepared(List<Pixels> tiles, Pixels overview, int gridX, int gridY) {
        public boolean tiled() {
            return gridX > 0 && gridY > 0 || overview != null;
        }
    }

    /** Token budget overrides (0 = the model's own limits). */
    public record Limits(int minTokens, int maxTokens) {
        public static final Limits NONE = new Limits(0, 0);
    }

    private Preprocess() {
    }

    public static Prepared preprocess(ClipModel m, Rgb img, Limits lim) {
        return switch (m.projector) {
            case IDEFICS3 -> idefics3(m, img, lim);
            case GEMMA3 -> single(m, fixedSize(m, img, Algo.BILINEAR));
            case QWEN2VL, QWEN25VL, QWEN3VL -> single(m, dynSize(m, img, lim));
        };
    }

    private static Prepared single(ClipModel m, Rgb r) {
        return new Prepared(List.of(normalize(m, r)), null, 0, 0);
    }

    static Pixels normalize(ClipModel m, Rgb r) {
        int n = r.width() * r.height();
        float[] out = new float[3 * n];
        byte[] d = r.data();
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < 3; c++) {
                float v = (float) (d[3 * i + c] & 0xFF) / 255.0f;
                out[3 * i + c] = (v - m.mean[c]) / m.std[c];
            }
        }
        return new Pixels(r.width(), r.height(), out);
    }

    // ------------------------------------------------------------------ fixed size (Gemma 3)

    private static Rgb fixedSize(ClipModel m, Rgb img, Algo algo) {
        int s = m.imageSize;
        return Resize.resize(img, new Size(s, s), algo, Pad.CEIL, new int[]{0, 0, 0});
    }

    // ------------------------------------------------------------------ dynamic size (Qwen-VL)

    /** The pixel limits of a dynamic-resolution encoder: its own token limits, or the caller's overrides. */
    static int[] pixelLimits(ClipModel m, int defMinTokens, int defMaxTokens, Limits lim) {
        int area = m.patchSize * m.patchSize * m.nMerge * m.nMerge;
        int minTok = lim.minTokens() > 0 ? lim.minTokens() : defMinTokens;
        int maxTok = lim.maxTokens() > 0 ? lim.maxTokens() : defMaxTokens;
        return new int[]{minTok * area, maxTok * area};
    }

    private static Rgb dynSize(ClipModel m, Rgb img, Limits lim) {
        int[] px = pixelLimits(m, 8, 4096, lim);
        Size target = Resize.calcSizePreservedRatio(new Size(img.width(), img.height()),
                new SizeRule(m.patchSize * m.nMerge, px[0], px[1], 0));
        return Resize.resize(img, target, Algo.BICUBIC, Pad.CEIL, new int[]{0, 0, 0});
    }

    /** The size {@link #dynSize} gives {@code (w, h)}: lets a caller count tokens without resizing. */
    public static Size dynTarget(ClipModel m, int w, int h, Limits lim) {
        int[] px = pixelLimits(m, 8, 4096, lim);
        return Resize.calcSizePreservedRatio(new Size(w, h), new SizeRule(m.patchSize * m.nMerge, px[0], px[1], 0));
    }

    // ------------------------------------------------------------------ idefics3 (SmolVLM)

    private static Prepared idefics3(ClipModel m, Rgb img, Limits lim) {
        int tile = m.imageSize;
        int minPx = lim.minTokens() > 0 ? lim.minTokens() * m.patchSize * m.patchSize * m.nMerge * m.nMerge : -1;
        int maxPx = lim.maxTokens() > 0 ? lim.maxTokens() * m.patchSize * m.patchSize * m.nMerge * m.nMerge : -1;
        Size orig = new Size(img.width(), img.height());
        List<Rgb> slices = new ArrayList<>();
        Rgb overview;
        int gridX;
        int gridY;
        if (m.imageLongestEdge > 0 && minPx <= 0 && maxPx <= 0) {
            // the processor's own recipe: longest side to the preprocessor size, then up to whole tiles, everything with
            // Lanczos and no padding
            int longest = m.imageLongestEdge;
            double aspect = (double) orig.width() / orig.height();
            int rw;
            int rh;
            if (orig.width() >= orig.height()) {
                rw = longest;
                rh = (int) (longest / aspect);
                rh += rh % 2;
            } else {
                rh = longest;
                rw = (int) (longest * aspect);
                rw += rw % 2;
            }
            gridX = (rw + tile - 1) / tile;
            gridY = (rh + tile - 1) / tile;
            Size refined = new Size(gridX * tile, gridY * tile);
            Rgb resized = Resize.resize(img, new Size(rw, rh), Algo.LANCZOS, Pad.NONE, null);
            Rgb ref = Resize.resize(resized, refined, Algo.LANCZOS, Pad.NONE, null);
            overview = Resize.resize(ref, new Size(tile, tile), Algo.LANCZOS, Pad.NONE, null);
            for (int y = 0; y < gridY; y++) {
                for (int x = 0; x < gridX; x++) {
                    slices.add(Resize.crop(ref, x * tile, y * tile, tile, tile));
                }
            }
        } else {
            // files without the preprocessor size (and token limits given by the caller): the generic llava-uhd slicing
            Size refined = Resize.calcSizePreservedRatio(orig, new SizeRule(tile, Math.max(0, minPx), Math.max(0, maxPx),
                    m.imageLongestEdge));
            gridX = (refined.width() + tile - 1) / tile;
            gridY = (refined.height() + tile - 1) / tile;
            int[] black = {0, 0, 0};
            overview = Resize.resize(img, new Size(tile, tile), Algo.BICUBIC, Pad.NONE, black);
            Rgb ref = Resize.resize(img, refined, Algo.BICUBIC, Pad.CEIL, black);
            for (int y = 0; y < refined.height(); y += tile) {
                for (int x = 0; x < refined.width(); x += tile) {
                    slices.add(Resize.crop(ref, x, y, Math.min(tile, refined.width() - x), Math.min(tile, refined.height() - y)));
                }
            }
        }
        List<Pixels> tiles = new ArrayList<>(slices.size());
        for (Rgb s : slices) {
            tiles.add(normalize(m, s));
        }
        return new Prepared(tiles, normalize(m, overview), gridX, gridY);
    }
}
