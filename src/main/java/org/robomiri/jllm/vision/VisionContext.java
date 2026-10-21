package org.robomiri.jllm.vision;

import org.robomiri.jllm.engine.MediaPrompt;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Decoder;
import org.robomiri.jllm.model.RopePositions;
import org.robomiri.jllm.model.RopeTable;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A loaded multimodal projector bound to a text model: encodes pictures into embeddings and builds the
 * {@link MediaPrompt} the engine runs (docs/20). It knows how each model family frames a picture in the prompt -
 * SmolVLM's tiles with {@code <row_1_col_2>} markers and a global overview, Gemma 3's
 * {@code <start_of_image> ... <end_of_image>}, Qwen-VL's {@code <|vision_start|> ... <|vision_end|>} - and, for
 * Qwen-VL, the three-axis rotary positions of the picture tokens.
 *
 * <p>Encoded pictures are kept (least recently used, bounded in size): a chat resends its earlier pictures with every
 * turn, and encoding them again would cost more than the answer.
 */
public final class VisionContext implements AutoCloseable {

    /**
     * @param backend  {@code cpu}, {@code metal} or {@code auto}
     * @param threads  CPU threads (0 = all cores)
     * @param limits   image token budget overrides
     * @param cacheMb  megabytes of encoded pictures to keep
     */
    public record Options(String backend, int threads, Preprocess.Limits limits, int cacheMb) {
        public static Options defaults() {
            return new Options("cpu", 0, Preprocess.Limits.NONE, 256);
        }
    }

    private sealed interface Part permits Ids, Chunk {
    }

    private record Ids(int[] ids) implements Part {
    }

    private record Chunk(float[] embd, int tokens, int nx, int ny, String hash) implements Part {
    }

    private record Encoded(String hash, List<Part> parts, long bytes, double ms) {
    }

    private final ClipModel clip;
    private final VisionEncoder encoder;
    private final Tokenizer tok;
    private final int textDim;
    private final Options opts;
    private final int placeholder;
    private final List<Integer> mrope;
    private final boolean mropeInterleaved;
    private final String fingerprint;
    private final Map<String, Encoded> cache = new LinkedHashMap<>(16, 0.75f, true);
    private long cacheBytes;
    private double lastEncodeMs;
    private int lastImages;

    private VisionContext(ClipModel clip, VisionEncoder encoder, Tokenizer tok, Architecture arch, Options opts) {
        this.clip = clip;
        this.encoder = encoder;
        this.tok = tok;
        this.textDim = arch.core().dim();
        this.opts = opts;
        List<Integer> sections = List.of();
        boolean inter = false;
        if (arch instanceof Decoder d && !d.ropeTables().isEmpty()) {
            RopeTable t = d.ropeTables().getFirst();
            sections = t.mrope();
            inter = t.mropeInterleaved();
        }
        this.mrope = sections;
        this.mropeInterleaved = inter;
        boolean qwen = clip.projector == ClipModel.Projector.QWEN2VL || clip.projector == ClipModel.Projector.QWEN25VL
                || clip.projector == ClipModel.Projector.QWEN3VL;
        if (qwen && sections.isEmpty()) {
            throw new IllegalArgumentException("the text model has no multimodal rotary sections (rope.dimension_sections), which "
                    + clip.projector.id + " needs: it is not a Qwen-VL model");
        }
        String padName = switch (clip.projector) {
            case IDEFICS3 -> "<image>";
            case GEMMA3 -> "<image_soft_token>";
            default -> "<|image_pad|>";
        };
        int id = tok.tokenId(padName);
        if (id < 0) {
            throw new IllegalArgumentException("the text model's vocabulary has no " + padName + " token, which "
                    + clip.projector.id + " pictures use as their placeholder");
        }
        this.placeholder = id;
        this.fingerprint = clip.path.getFileName() + ":" + clip.byteSize() + ":" + opts.limits().minTokens() + ":"
                + opts.limits().maxTokens();
        // fail early on the usual mistake: a projector made for another text model
        if (clip.projectionDim != textDim) {
            throw new IllegalArgumentException("the projector " + clip.path.getFileName() + " produces " + clip.projectionDim
                    + "-wide embeddings, the text model is " + textDim + " wide: it is the wrong mmproj file for this model");
        }
    }

    /**
     * Opens a projector file for a text model.
     *
     * @throws IOException if the file is not a usable vision projector, or it does not fit the text model
     */
    public static VisionContext open(Path mmproj, Tokenizer tok, Architecture arch, Options opts) throws IOException {
        ClipModel clip = ClipModel.open(mmproj);
        VisionEncoder enc = null;
        try {
            enc = VisionEncoders.create(clip, opts);
            return new VisionContext(clip, enc, tok, arch, opts);
        } catch (IllegalArgumentException e) {
            if (enc != null) {
                enc.close();
            }
            clip.close();
            throw new IOException(e.getMessage(), e);
        } catch (RuntimeException e) {
            if (enc != null) {
                enc.close();
            }
            clip.close();
            throw e;
        }
    }

    public ClipModel clip() {
        return clip;
    }

    public ClipModel.Projector projector() {
        return clip.projector;
    }

    public Path file() {
        return clip.path;
    }

    /** One line for logs and the console. */
    public String describe() {
        return clip.projector.id + " (" + clip.path.getFileName() + ", " + clip.nLayer + " layers x " + clip.nEmbd
                + ", " + encoder.name() + ")";
    }

    /** Milliseconds the last {@link #build} spent encoding new pictures (0 when all came from the cache). */
    public synchronized double lastEncodeMs() {
        return lastEncodeMs;
    }

    /** How many pictures the last {@link #build} had to encode (the others were cached). */
    public synchronized int lastEncodedImages() {
        return lastImages;
    }

    // ------------------------------------------------------------------ building prompts

    /**
     * Replaces each {@link Tokenizer#IMAGE_MARK} of {@code ids} by the tokens of the next picture and returns the
     * prompt for the engine.
     *
     * @param ids    the prepared prompt ids (BOS in place) with one image mark per picture
     * @param images the pictures in order
     */
    public synchronized MediaPrompt build(int[] ids, List<Rgb> images) {
        int marks = 0;
        for (int id : ids) {
            if (id == Tokenizer.IMAGE_MARK) {
                marks++;
            }
        }
        if (marks != images.size()) {
            throw new IllegalArgumentException("the prompt has " + marks + " image placeholders and " + images.size() + " images");
        }
        lastEncodeMs = 0;
        lastImages = 0;
        int[] out = new int[ids.length + 16];
        int n = 0;
        List<MediaPrompt.Span> spans = new ArrayList<>();
        List<int[]> grids = new ArrayList<>(); // nx, ny per span, for the rotary positions
        int k = 0;
        for (int id : ids) {
            if (id != Tokenizer.IMAGE_MARK) {
                if (n == out.length) {
                    out = java.util.Arrays.copyOf(out, n * 2);
                }
                out[n++] = id;
                continue;
            }
            Encoded e = encode(images.get(k++));
            for (Part p : e.parts()) {
                int len = p instanceof Ids i ? i.ids().length : ((Chunk) p).tokens();
                if (n + len > out.length) {
                    out = java.util.Arrays.copyOf(out, Math.max(out.length * 2, n + len));
                }
                switch (p) {
                    case Ids i -> {
                        System.arraycopy(i.ids(), 0, out, n, len);
                    }
                    case Chunk c -> {
                        java.util.Arrays.fill(out, n, n + len, placeholder);
                        spans.add(new MediaPrompt.Span(n, len, c.hash(), c.embd()));
                        grids.add(new int[]{c.nx(), c.ny()});
                    }
                }
                n += len;
            }
        }
        int[] tokens = java.util.Arrays.copyOf(out, n);
        RopePositions rope = mrope.isEmpty() ? null : ropePositions(tokens.length, spans, grids);
        return new MediaPrompt(tokens, spans, rope, clip.projector == ClipModel.Projector.GEMMA3);
    }

    /** M-RoPE: text counts up; a picture takes {@code (t, row, column)} from its start and advances by its larger side. */
    private RopePositions ropePositions(int n, List<MediaPrompt.Span> spans, List<int[]> grids) {
        int[] t = new int[n];
        int[] h = new int[n];
        int[] w = new int[n];
        int next = 0;
        int si = 0;
        int i = 0;
        while (i < n) {
            if (si < spans.size() && spans.get(si).start() == i) {
                MediaPrompt.Span s = spans.get(si);
                int nx = Math.max(1, grids.get(si)[0]);
                int ny = Math.max(1, grids.get(si)[1]);
                for (int j = 0; j < s.length(); j++) {
                    t[i + j] = next;
                    h[i + j] = next + j / nx;
                    w[i + j] = next + j % nx;
                }
                next += Math.max(nx, ny);
                i += s.length();
                si++;
            } else {
                t[i] = next;
                h[i] = next;
                w[i] = next;
                next++;
                i++;
            }
        }
        int[] sec = mrope.stream().mapToInt(Integer::intValue).toArray();
        return new RopePositions(t, h, w, sec, mropeInterleaved, n - next);
    }

    // ------------------------------------------------------------------ encoding

    private Encoded encode(Rgb img) {
        String key = digest(img.hash() + "|" + fingerprint);
        Encoded hit = cache.get(key);
        if (hit != null) {
            return hit;
        }
        long t0 = System.nanoTime();
        Preprocess.Prepared pp = Preprocess.preprocess(clip, img, opts.limits());
        List<Part> parts = switch (clip.projector) {
            case IDEFICS3 -> idefics3Parts(pp, key);
            case GEMMA3 -> framed("<start_of_image>", "<end_of_image>", pp.tiles().getFirst(), key);
            default -> framed("<|vision_start|>", "<|vision_end|>", pp.tiles().getFirst(), key);
        };
        long bytes = 0;
        for (Part p : parts) {
            if (p instanceof Chunk c) {
                bytes += 4L * c.embd().length;
            }
        }
        double ms = (System.nanoTime() - t0) / 1e6;
        lastEncodeMs += ms;
        lastImages++;
        Encoded e = new Encoded(key, parts, bytes, ms);
        cache.put(key, e);
        cacheBytes += bytes;
        long limit = (long) opts.cacheMb() << 20;
        var it = cache.entrySet().iterator();
        while (cacheBytes > limit && cache.size() > 1 && it.hasNext()) {
            var en = it.next();
            if (en.getValue() == e) {
                break;
            }
            cacheBytes -= en.getValue().bytes();
            it.remove();
        }
        return e;
    }

    private Chunk chunk(Preprocess.Pixels px, String hash, int index) {
        VisionEncoder.Encoded e = encoder.encode(px);
        return new Chunk(e.embd(), e.tokens(), e.nx(), e.ny(), hash + "#" + index);
    }

    private List<Part> framed(String begin, String end, Preprocess.Pixels px, String hash) {
        return List.of(new Ids(special(begin)), chunk(px, hash, 0), new Ids(special(end)));
    }

    /**
     * SmolVLM / Idefics3: each tile is introduced by {@code <fake_token_around_image><row_R_col_C>}, rows are
     * separated by a newline, then {@code \n\n<fake_token_around_image><global-img>}, the overview and a closing
     * {@code <fake_token_around_image>} (transformers' Idefics3 processor, as llama.cpp's mtmd lays it out).
     */
    private List<Part> idefics3Parts(Preprocess.Prepared pp, String hash) {
        List<Part> parts = new ArrayList<>();
        int[] fake = special("<fake_token_around_image>");
        int idx = 0;
        for (int y = 0; y < pp.gridY(); y++) {
            for (int x = 0; x < pp.gridX(); x++) {
                parts.add(new Ids(special(String.format("<fake_token_around_image><row_%d_col_%d>", y + 1, x + 1))));
                parts.add(chunk(pp.tiles().get(y * pp.gridX() + x), hash, idx++));
            }
            if (y != pp.gridY() - 1) {
                parts.add(new Ids(single("\n")));
            }
        }
        int[] nn = single("\n\n");
        int[] glob = special("<global-img>");
        int[] head = new int[nn.length + fake.length + glob.length];
        System.arraycopy(nn, 0, head, 0, nn.length);
        System.arraycopy(fake, 0, head, nn.length, fake.length);
        System.arraycopy(glob, 0, head, nn.length + fake.length, glob.length);
        parts.add(new Ids(head));
        parts.add(chunk(pp.overview(), hash, idx));
        parts.add(new Ids(fake));
        return parts;
    }

    private int[] special(String text) {
        int[] ids = tok.encode(text, true);
        if (ids.length == 0) {
            throw new IllegalArgumentException("the vocabulary cannot tokenize " + text);
        }
        return ids;
    }

    private int[] single(String text) {
        int[] ids = tok.encode(text, true);
        if (ids.length != 1) {
            throw new IllegalArgumentException("expected " + text.replace("\n", "\\n") + " to be one token, got " + ids.length);
        }
        return ids;
    }

    private static String digest(String s) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(s.getBytes(java.nio.charset.StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void close() {
        encoder.close();
        clip.close();
        cache.clear();
    }
}
