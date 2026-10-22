package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.engine.CacheFile;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Decoder;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.ShardProvider;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Saved and copied caches on tiny random checkpoints, standard attention and DeepSeek's latent attention (MLA), f32
 * and f16: a state saved and loaded, or copied, continues with exactly the original logits; loading only the first
 * positions of a longer saved state (seekable file or plain stream) equals prefilling just those; and an engine save
 * that writes its rows after the engine moved on ({@link Engine#beginSave}) holds the rows of the moment it began,
 * while a request that would rewrite them waits for it.
 *
 * <p>usage: {@code KvStateTest <tokenizer-model-dir> [--metal <lib>] <tiny-model-dir>...} (the tiny models of
 * scripts/arch-smoke.sh; the tokenizer is only needed by the engine checks and must cover their vocabulary). With
 * {@code --metal} the backend checks also run on the Metal backend (or its emulator, -Djllm.metal.emulator=true), and a
 * CPU-saved state must load into it.
 */
public final class KvStateTest {

    private static final int[] PROMPT = {1, 7, 42, 99, 13, 5, 250, 8, 77, 31, 600, 9};
    private static int failures;

    public static void main(String[] args) throws Exception {
        Path tokFile = Path.of(args[0]).resolve("tokenizer.json");
        Tokenizer tok = Files.isRegularFile(tokFile) ? Tokenizer.load(tokFile) : null;
        if (tok == null) {
            System.out.println("note: no " + tokFile + ", the engine checks are skipped");
        }
        String lib = null;
        int first = 1;
        if (args.length > 2 && args[1].equals("--metal")) {
            lib = args[2];
            first = 3;
            if (!Metal.available(lib)) {
                System.out.println("note: Metal unavailable, CPU checks only");
                lib = null;
            }
        }
        for (int i = first; i < args.length; i++) {
            Path dir = Path.of(args[i]);
            Architecture cfg = Architecture.load(dir);
            String kind = ((Decoder) cfg).attention() instanceof Attention.Mla ? "mla" : "gqa";
            try (ModelWeights w = ModelWeights.load(dir, cfg, QType.Q8_0, ShardProvider.local(dir))) {
                // q8 with plain and with Hadamard-rotated keys (backend.kv-q8-rotate) for standard attention; a q8 latent
                // cache is not saved
                List<Object[]> dtypes = new java.util.ArrayList<>(List.of(new Object[]{KvDtype.F32, false}, new Object[]{KvDtype.F16, false}));
                if (kind.equals("gqa")) {
                    dtypes.add(new Object[]{KvDtype.Q8, false});
                    dtypes.add(new Object[]{KvDtype.Q8, true});
                }
                for (Object[] dr : dtypes) {
                    KvDtype dt = (KvDtype) dr[0];
                    boolean rot = (Boolean) dr[1];
                    KvDtype.setQ8Rotate(rot);
                    String label = dir.getFileName() + " (" + kind + ", " + dt + (rot ? " rotated keys" : "") + ")";
                    warmUp(new CpuBackend(w, 64, 16, 4, dt, false), cfg);
                    backendChecks(label + " cpu", () -> new CpuBackend(w, 64, 16, 4, dt, false), cfg);
                    if (lib != null) {
                        String l = lib;
                        backendChecks(label + " metal", () -> new MetalBackend(w, 64, 16, l, dt), cfg);
                        crossCheck(label, w, l, dt, cfg);
                    }
                    if (tok != null) {
                        engineChecks(label, w, cfg, tok, dt);
                    }
                    KvDtype.setQ8Rotate(false);
                }
                if (kind.equals("gqa") && tok != null) {
                    rotateRefusal(dir.getFileName().toString(), w, cfg, tok);
                }
            }
        }
        if (failures > 0) {
            System.out.println("FAILED: " + failures);
            System.exit(1);
        }
        System.out.println("kv state tests passed");
    }

    /**
     * Runs the kernels until the JIT has compiled them: an interpreted Vector API reduction may add in another order
     * than the compiled one, so logits computed across that switch can differ in the last bits. The exact comparisons
     * below then compare compiled code with compiled code.
     */
    private static void warmUp(Backend b, Architecture cfg) {
        float[] logits = new float[cfg.core().vocabSize()];
        try (b) {
            for (int i = 0; i < 20; i++) {
                b.forward(PROMPT, 0, PROMPT.length, 0, logits);
                b.forward(PROMPT, 3, 1, PROMPT.length, logits);
                b.forward(PROMPT, 0, 7, 0, null);
                b.forward(PROMPT, 7, 5, 7, logits);
            }
        }
    }

    interface Maker {
        Backend make() throws Exception;
    }

    private static void check(boolean ok, String what) {
        System.out.println("  " + (ok ? "ok      " : "FAILED  ") + what);
        if (!ok) {
            failures++;
        }
    }

    private static byte[] save(Backend b, int n) throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        b.saveState(n, Channels.newChannel(bytes));
        return bytes.toByteArray();
    }

    /** Save / load / copy / partial load on one backend kind: exact logits. */
    private static void backendChecks(String label, Maker make, Architecture cfg) throws Exception {
        int vocab = cfg.core().vocabSize();
        int n = PROMPT.length;
        int part = 7;
        float[] want = new float[vocab];
        float[] got = new float[vocab];
        try (Backend a = make.make(); Backend b = make.make(); Backend c = make.make(); Backend d = make.make();
             Backend e = make.make()) {
            check(a.stateFormat() != null && a.canSaveWhileServing(), label + ": state format " + a.stateFormat());
            a.forward(PROMPT, 0, n, 0, null);
            byte[] bytes = save(a, n);
            b.loadState(n, Channels.newChannel(new ByteArrayInputStream(bytes)));
            c.copyStateFrom(a, n);
            ByteArrayOutputStream concurrent = new ByteArrayOutputStream();
            a.saveStateWhileServing(n, Channels.newChannel(concurrent));
            int[] next = {PROMPT[3]};
            a.forward(next, 0, 1, n, want);
            b.forward(next, 0, 1, n, got);
            check(Arrays.equals(want, got), label + ": saved + loaded continues exactly");
            c.forward(next, 0, 1, n, got);
            check(Arrays.equals(want, got), label + ": copied continues exactly");
            check(Arrays.equals(bytes, concurrent.toByteArray()), label + ": saveStateWhileServing writes the same bytes");
            check(Arrays.equals(save(b, n), bytes), label + ": a loaded state saves to the same bytes");

            // the first `part` positions of the 12 saved: from a stream (skipped by reading) and from a file (seek)
            e.forward(PROMPT, 0, part, 0, null);
            e.forward(PROMPT, part, n - part, part, want);
            d.loadState(part, n, Channels.newChannel(new ByteArrayInputStream(bytes)));
            check(Arrays.equals(save(d, part), save(b, part)), label + ": a partial load holds the first rows bit for bit");
            d.forward(PROMPT, part, n - part, part, got);
            check(Arrays.equals(want, got), label + ": the first " + part + " of " + n + " saved positions (stream)");
            Path f = Files.createTempFile("kvstate", ".bin");
            try {
                Files.write(f, bytes);
                try (FileChannel ch = FileChannel.open(f, StandardOpenOption.READ)) {
                    d.loadState(part, n, ch);
                    check(ch.position() == bytes.length, label + ": a partial load consumes the whole state");
                }
                d.forward(PROMPT, part, n - part, part, got);
                check(Arrays.equals(want, got), label + ": the first " + part + " of " + n + " saved positions (file)");
            } finally {
                Files.deleteIfExists(f);
            }
        }
    }

    /** A CPU-saved state in the Metal backend (and back): same bytes, logits within the usual CPU / Metal tolerance. */
    private static void crossCheck(String label, ModelWeights w, String lib, KvDtype dt, Architecture cfg) throws Exception {
        int vocab = cfg.core().vocabSize();
        int n = PROMPT.length;
        float[] want = new float[vocab];
        float[] got = new float[vocab];
        try (CpuBackend cpu = new CpuBackend(w, 64, 16, 4, dt, false); MetalBackend gpu = new MetalBackend(w, 64, 16, lib, dt)) {
            check(cpu.stateFormat().equals(gpu.stateFormat()), label + ": cpu and metal formats agree (" + cpu.stateFormat() + ")");
            cpu.forward(PROMPT, 0, n, 0, null);
            byte[] bytes = save(cpu, n);
            gpu.loadState(n, Channels.newChannel(new ByteArrayInputStream(bytes)));
            check(Arrays.equals(save(gpu, n), bytes), label + ": a cpu state saved again by metal is the same bytes");
            int[] next = {PROMPT[5]};
            cpu.forward(next, 0, 1, n, want);
            gpu.forward(next, 0, 1, n, got);
            float scale = 0;
            float diff = 0;
            for (int i = 0; i < vocab; i++) {
                scale = Math.max(scale, Math.abs(want[i]));
                diff = Math.max(diff, Math.abs(want[i] - got[i]));
            }
            check(diff <= 1e-3f * scale + 1e-4f, String.format("%s: cpu state continued on metal (max diff %.1e)", label, diff));
        }
    }

    /**
     * A saved q8 cache of rotated keys has another layout than a plain q8 one ({@code backend.kv-q8-rotate}): each is
     * refused by an engine of the other kind, and loads into one of its own.
     */
    private static void rotateRefusal(String label, ModelWeights w, Architecture cfg, Tokenizer tok) throws Exception {
        ChatTemplate.Format fmt = ChatTemplate.forModelType(cfg.core().modelType(), tok);
        SamplingParams greedy = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 4, List.of());
        Path dir = Files.createTempDirectory("kvstate-rot");
        try {
            for (boolean saveRot : new boolean[]{true, false}) {
                Path file = dir.resolve(saveRot ? "rot.kv" : "plain.kv");
                KvDtype.setQ8Rotate(saveRot);
                try (Engine a = new Engine(new CpuBackend(w, 64, 16, 4, KvDtype.Q8, false), tok, cfg, Set.of(), 0, fmt)) {
                    a.generate(PROMPT, greedy, null, () -> false);
                    Engine.PendingSave p = a.beginSave(file, "fp");
                    p.write();
                }
                KvDtype.setQ8Rotate(!saveRot);
                boolean refused = false;
                try (Engine b = new Engine(new CpuBackend(w, 64, 16, 4, KvDtype.Q8, false), tok, cfg, Set.of(), 0, fmt)) {
                    try {
                        b.loadCache(file, "fp");
                    } catch (java.io.IOException e) {
                        refused = e.getMessage().contains("another layout");
                    }
                }
                KvDtype.setQ8Rotate(saveRot);
                boolean loaded = false;
                try (Engine b = new Engine(new CpuBackend(w, 64, 16, 4, KvDtype.Q8, false), tok, cfg, Set.of(), 0, fmt)) {
                    loaded = b.loadCache(file, "fp") > 0;
                }
                check(refused && loaded, label + ": a q8 cache saved " + (saveRot ? "with" : "without") + " rotated keys is refused by "
                        + (saveRot ? "a plain" : "a rotating") + " q8 backend and loads into its own kind");
            }
        } finally {
            KvDtype.setQ8Rotate(false);
            try (var files = Files.list(dir)) {
                for (Path f : files.toList()) {
                    Files.delete(f);
                }
            }
            Files.delete(dir);
        }
    }

    /**
     * {@link Engine#beginSave}: the rows are written after the engine served a request that extends the saved tokens
     * (it does not wait), and a request that rewrites a saved row waits until the write is done; the file holds the
     * rows of the moment the save began. Then {@link Engine#loadCache} with a limit loads a prefix.
     */
    private static void engineChecks(String label, ModelWeights w, Architecture cfg, Tokenizer tok, KvDtype dt) throws Exception {
        ChatTemplate.Format fmt = ChatTemplate.forModelType(cfg.core().modelType(), tok);
        SamplingParams greedy = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 4, List.of());
        Path dir = Files.createTempDirectory("kvstate-engine");
        Path file = dir.resolve("a.kv");
        try (Engine e1 = new Engine(new CpuBackend(w, 64, 16, 4, dt, false), tok, cfg, Set.of(), 0, fmt);
             Engine e2 = new Engine(new CpuBackend(w, 64, 16, 4, dt, false), tok, cfg, Set.of(), 0, fmt)) {
            e1.generate(PROMPT, greedy, null, () -> false);
            int[] saved = e1.cachedTokens();
            byte[] rows = save(e1.backend(), saved.length); // the rows right now (the engine is idle)
            Engine.PendingSave p = e1.beginSave(file, "fp");
            check(p != null && Arrays.equals(p.tokens(), saved), label + ": a save of " + saved.length + " positions began");

            // the next turn: extends the saved tokens, so it runs at once although the rows are not written yet
            int[] longer = Arrays.copyOf(saved, saved.length + 3);
            longer[saved.length] = 5;
            longer[saved.length + 1] = 6;
            longer[saved.length + 2] = 7;
            Engine.Result r = e1.generate(longer, greedy, null, () -> false);
            check(r.stats().cachedTokens() == saved.length, label + ": the extending request reused " + r.stats().cachedTokens()
                    + " positions without waiting");

            // a request that diverges inside the saved part must wait for the write
            int[] other = PROMPT.clone();
            other[3] = 500;
            Thread t = Thread.ofPlatform().start(() -> e1.generate(other, greedy, null, () -> false));
            Thread.sleep(300);
            check(t.isAlive() && e1.busy(), label + ": a request rewriting saved rows waits for the save");
            int n = p.write();
            t.join(30_000);
            check(!t.isAlive() && n == saved.length, label + ": ... and runs once the rows are written");

            CacheFile.Header h = CacheFile.readHeader(file);
            byte[] all = Files.readAllBytes(file);
            byte[] body = Arrays.copyOfRange(all, all.length - rows.length, all.length);
            check(Arrays.equals(h.tokens(), saved) && Arrays.equals(body, rows),
                    label + ": the file holds the tokens and rows of the moment the save began");

            int got = e2.loadCache(file, "fp", 5);
            check(got == 5 && Arrays.equals(e2.cachedTokens(), Arrays.copyOf(saved, 5)), label + ": loadCache with a limit"
                    + " restores the first 5 positions");
            got = e2.loadCache(file, "fp");
            check(got == saved.length && Arrays.equals(save(e2.backend(), got), rows), label + ": a full load restores the rows");
        } finally {
            Files.deleteIfExists(file);
            Files.deleteIfExists(dir);
        }
    }
}
