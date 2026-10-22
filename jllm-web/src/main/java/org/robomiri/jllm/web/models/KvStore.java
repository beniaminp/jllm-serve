package org.robomiri.jllm.web.models;

import org.robomiri.jllm.engine.CacheFile;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.web.WebProperties;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Conversation KV caches on disk, so that a restart (or a conversation that was pushed out of its slot) does not pay
 * the long first prefill again: a coding agent's first request carries ~10 000 tokens of instructions and tool
 * definitions, minutes of prefill on a CPU, and every later session of the agent starts with the same tokens.
 * See docs/17 §17.3.
 *
 * <ul>
 *   <li><b>Save</b>: a slot's cache is saved after a request whose prefill took at least
 *       {@code jllm.web.kv-cache-save-after-ms}, when a model is unloaded or the server stops, and before a restore
 *       overwrites it; only caches of at least {@code jllm.web.kv-cache-min-tokens} tokens. The save holds the engine
 *       only to take the token list ({@link Engine#beginSave}); the rows are written by a background thread while the
 *       next request runs (it waits only if it rewrites a row not written yet). After a request, a save is skipped
 *       when a saved file already covers most of the cache (fewer than {@code max(min-tokens, covered / 4)} new
 *       tokens), so a growing conversation is rewritten every so often, not after every turn. A cache that extends a
 *       saved one replaces it (one file per conversation); one that a saved file already covers is not written.</li>
 *   <li><b>Restore</b>: when a request is assigned a slot, the saved cache sharing the longest prefix with its
 *       prompt is found by a binary search in the saved token lists sorted lexicographically, and only that prefix is
 *       loaded (the rest of the file is skipped), if it covers at least {@link #MIN_GAIN} more tokens than the slot
 *       holds.</li>
 *   <li><b>Budget</b>: the least recently used files are deleted beyond {@code jllm.web.kv-cache-max-gb}.</li>
 * </ul>
 *
 * Files live in {@code <kv-cache-dir>/<fingerprint hash>/}: a saved cache is only reused by the same model with the
 * same load settings that change keys and values (quantization, backend, cache type, int8 activations).
 */
@Service
public class KvStore {

    /**
     * A model with a paged KV cache (docs/24): the pool's evicted blocks go to {@code <dir>/blocks/<fingerprint>/} and
     * come back when a prompt needs them, so this store is also the block store of the pool. The files survive a restart
     * (found again by the hash of the path of tokens they hold) and are trimmed to {@code jllm.web.kv-spill-max-gb}.
     * Does nothing for a model with a flat cache or with {@code jllm.web.kv-spill=false}.
     */
    public void attachSpill(LoadedModel m) {
        if (!props.kvSpill() || props.kvSpillMaxGb() <= 0) {
            return;
        }
        org.robomiri.jllm.kv.KvSequence seq = m.engine().backend().kvSequence();
        if (seq == null) {
            return;
        }
        org.robomiri.jllm.kv.KvPool pool = seq.pool();
        if (pool.spill() != null) {
            return; // backend.kv-tier already gave the pool its tiered block store (host memory, then files)
        }
        try {
            pool.setSpill(new org.robomiri.jllm.kv.FileSpill(dir.resolve("blocks"), fingerprint(m) + "|" + pool.format(),
                    (long) (props.kvSpillMaxGb() * 1e9)));
        } catch (IOException e) {
            System.err.println("kv-cache: no block store for " + m.id() + ": " + e.getMessage());
        }
    }

    /** Restoring must win at least this many tokens over what the slot already holds. */
    static final int MIN_GAIN = 256;

    /** A saved cache, with its tokens kept in memory for prefix matching. */
    private static final class Entry {
        final Path file;
        final int[] tokens;
        final long bytes;
        volatile long used;

        Entry(Path file, int[] tokens, long bytes, long used) {
            this.file = file;
            this.tokens = tokens;
            this.bytes = bytes;
            this.used = used;
        }
    }

    /** One saved cache, as the console lists it. */
    public record SavedView(String model, int tokens, long bytes, long usedAt) {
    }

    public record View(boolean enabled, String dir, int files, long bytes, long maxBytes, int minTokens,
                       long saveAfterMs, List<SavedView> saved) {
    }

    private final WebProperties props;
    private final Path dir;
    /**
     * Fingerprint hash -> entries by token list, in lexicographic order: the saved list sharing the longest prefix
     * with a prompt is next to where the prompt would be inserted. A directory is scanned the first time its model is
     * used.
     */
    private final Map<String, TreeMap<int[], Entry>> index = new LinkedHashMap<>();
    /** Fingerprint hash -> model id, for the console. */
    private final Map<String, String> names = new TreeMap<>();

    public KvStore(WebProperties props) {
        this.props = props;
        this.dir = Path.of(props.kvCacheDir()).toAbsolutePath();
    }

    public boolean enabled() {
        return props.kvCache();
    }

    /**
     * What identifies the keys and values a model computes: its source and the load settings that change them (not
     * the context size, batch, threads or slots).
     */
    static String fingerprint(LoadedModel m) {
        StringBuilder sb = new StringBuilder(m.source());
        new TreeMap<>(m.settings()).forEach((k, v) -> {
            boolean irrelevant = k.equals("slots") || k.equals("backend.ctx") || k.equals("backend.batch")
                    || k.equals("backend.threads") || k.startsWith("server.") || k.startsWith("engine.");
            if (!irrelevant) {
                sb.append('|').append(k).append('=').append(v);
            }
        });
        return sb.toString();
    }

    private static String hash(String fingerprint) {
        try {
            byte[] d = MessageDigest.getInstance("SHA-256").digest(fingerprint.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(d, 0, 12);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    /** The entries of a model's directory, scanning it on first use. */
    private synchronized TreeMap<int[], Entry> entries(String h, String fingerprint, String modelId) {
        names.put(h, modelId);
        TreeMap<int[], Entry> map = index.get(h);
        if (map != null) {
            return map;
        }
        map = new TreeMap<>(Arrays::compare);
        Path d = dir.resolve(h);
        if (Files.isDirectory(d)) {
            try (var files = Files.list(d)) {
                for (Path f : files.filter(p -> p.getFileName().toString().endsWith(".kv")).toList()) {
                    try {
                        CacheFile.Header hd = CacheFile.readHeader(f);
                        if (hd.fingerprint().equals(fingerprint)) {
                            Entry en = new Entry(f, hd.tokens(), Files.size(f), Files.getLastModifiedTime(f).toMillis());
                            Entry dup = map.put(en.tokens, en);
                            if (dup != null) { // the same conversation twice: keep one
                                Files.deleteIfExists(dup.file);
                            }
                        }
                    } catch (IOException e) {
                        System.err.println("kv-cache: ignoring unreadable " + f + ": " + e.getMessage());
                    }
                }
            } catch (IOException e) {
                System.err.println("kv-cache: cannot list " + d + ": " + e.getMessage());
            }
        }
        index.put(h, map);
        return map;
    }

    /** Length of the common prefix of {@code a} and {@code b}. */
    static int commonPrefix(int[] a, int[] b) {
        int n = Math.min(a.length, b.length);
        int i = 0;
        while (i < n && a[i] == b[i]) {
            i++;
        }
        return i;
    }

    /**
     * The entry sharing the longest prefix with {@code tokens}. In lexicographic order the common prefix with
     * {@code tokens} only shrinks moving away from where {@code tokens} would be inserted, so it is the entry just
     * before or just after that point: two comparisons instead of one per saved file.
     */
    static <V> Map.Entry<int[], V> longestPrefix(TreeMap<int[], V> map, int[] tokens) {
        Map.Entry<int[], V> lo = map.floorEntry(tokens);
        Map.Entry<int[], V> hi = map.higherEntry(tokens);
        if (lo == null) {
            return hi;
        }
        if (hi == null) {
            return lo;
        }
        return commonPrefix(hi.getKey(), tokens) > commonPrefix(lo.getKey(), tokens) ? hi : lo;
    }

    /**
     * Before a request runs on {@code slot}: loads the prefix of a saved cache that covers the most of {@code prompt},
     * when that beats what the slot holds. The slot's own cache is saved first if it is worth keeping.
     */
    public void restore(LoadedModel m, LoadedModel.Slot slot, int[] prompt) {
        Engine e = slot.engine();
        if (!enabled() || !e.canPersistCache() || prompt.length <= MIN_GAIN || blockSessions(m, slot)) {
            return; // (block sessions come back through the pool's own match / adopt)
        }
        String fp = fingerprint(m);
        String h = hash(fp);
        int have = e.cachedPrefix(prompt);
        Entry best;
        int bestLen;
        synchronized (this) {
            Map.Entry<int[], Entry> x = longestPrefix(entries(h, fp, m.id()), prompt);
            best = x == null ? null : x.getValue();
            bestLen = x == null ? 0 : Math.min(commonPrefix(x.getKey(), prompt), prompt.length - 1);
        }
        if (best == null || bestLen - have < MIN_GAIN) {
            return;
        }
        save(m, slot, "before restoring another conversation into slot " + slot.index(), Mode.BEFORE_RESTORE);
        long t0 = System.nanoTime();
        try {
            int n = e.loadCache(best.file, fp, bestLen); // only the shared prefix: the rest would be overwritten
            best.used = System.currentTimeMillis();
            touch(best.file);
            System.err.printf("kv-cache: restored %d of the %d saved tokens of %s into slot %d from %s in %.1f s (the"
                            + " prompt has %d)%n", n, best.tokens.length, m.id(), slot.index(), best.file.getFileName(),
                    (System.nanoTime() - t0) / 1e9, prompt.length);
        } catch (IOException | RuntimeException ex) {
            System.err.println("kv-cache: cannot restore " + best.file + ": " + ex.getMessage() + " (deleting it)");
            forget(h, best);
        }
    }

    /**
     * After a request, while its slot is still reserved: saves the slot's cache when the prefill was expensive. Only
     * the token list is taken here; the rows are written in the background.
     */
    public void afterRequest(LoadedModel m, LoadedModel.Slot slot, Engine.Stats stats) {
        if (enabled() && slot.engine().canPersistCache() && stats.prefillMs() >= props.kvCacheSaveAfterMs()
                && !slot.engine().saving()) {
            save(m, slot, "after a " + String.format("%.1f", stats.prefillMs() / 1000) + " s prefill", Mode.AFTER_REQUEST);
        }
    }

    /** Saves every slot of a model (unload, shutdown, or on request), waiting for the files to be written. */
    public void saveAll(LoadedModel m) {
        if (!enabled()) {
            return;
        }
        for (LoadedModel.Slot s : m.slots()) {
            save(m, s, "on unload", Mode.NOW);
        }
    }

    /** Why a save happens: what is skipped and whether the caller waits for the file. */
    enum Mode {
        /** after a request: skipped when a saved file already covers most of the cache; written in the background */
        AFTER_REQUEST,
        /** before a restore overwrites the slot: written in the background (the restore waits for it) */
        BEFORE_RESTORE,
        /** unload / shutdown / the console's button: written before returning */
        NOW
    }

    /**
     * With {@code backend.kv-tier} a conversation is saved as the pool's own block files (docs/26 §26.2.5): the blocks of its
     * token path that the store does not hold yet are written (in the configured disk format) and flushed, and nothing else.
     * Reloading needs no file of its own: the next prompt's tokens find the blocks by their path hashes, and the engine adopts
     * them like any evicted block (prefetched, and only as far as the prompt matches).
     */
    private boolean blockSessions(LoadedModel m, LoadedModel.Slot slot) {
        var seq = slot.engine().backend().kvSequence();
        return seq != null && seq.pool().spill() != null && "true".equals(m.settings().get("backend.kv-tier"));
    }

    private void saveBlocks(LoadedModel m, LoadedModel.Slot slot, String why) {
        Engine e = slot.engine();
        int[] tokens = e.cachedTokens();
        if (tokens.length < props.kvCacheMinTokens()) {
            return;
        }
        var pool = slot.engine().backend().kvSequence().pool();
        long t0 = System.nanoTime();
        int written = pool.persist(tokens, tokens.length);
        if (written > 0) {
            pool.flushSpill();
            System.err.printf("kv-cache: saved %d tokens of %s (slot %d, %s) as %d block files in %.2f s%n", tokens.length, m.id(),
                    slot.index(), why, written, (System.nanoTime() - t0) / 1e9);
        }
    }

    /** Saves the slot's cache unless it is short or a saved file already holds it. */
    void save(LoadedModel m, LoadedModel.Slot slot, String why, Mode mode) {
        Engine e = slot.engine();
        if (blockSessions(m, slot)) {
            saveBlocks(m, slot, why);
            return;
        }
        if (!e.canPersistCache()) {
            return;
        }
        int[] tokens = e.cachedTokens();
        if (tokens.length < props.kvCacheMinTokens()) {
            return;
        }
        String fp = fingerprint(m);
        String h = hash(fp);
        synchronized (this) {
            TreeMap<int[], Entry> map = entries(h, fp, m.id());
            Map.Entry<int[], Entry> ceil = map.ceilingEntry(tokens); // the first list that may extend tokens
            if (ceil != null && commonPrefix(ceil.getKey(), tokens) == tokens.length) {
                ceil.getValue().used = System.currentTimeMillis();
                return; // already on disk (the same tokens, or more)
            }
            if (mode == Mode.AFTER_REQUEST) {
                int covered = 0;
                for (Entry en : map.values()) {
                    if (en.tokens.length < tokens.length && commonPrefix(en.tokens, tokens) == en.tokens.length) {
                        covered = Math.max(covered, en.tokens.length);
                    }
                }
                if (tokens.length - covered < Math.max(props.kvCacheMinTokens(), covered / 4)) {
                    return; // most of it is on disk: saved again once it has grown more
                }
            }
        }
        Path file = dir.resolve(h).resolve(System.currentTimeMillis() + "-" + tokens.length + "-" + slot.index() + ".kv");
        long t0 = System.nanoTime();
        Engine.PendingSave pending;
        try {
            Files.createDirectories(file.getParent());
            Path about = file.getParent().resolve("model.txt");
            if (!Files.exists(about)) {
                Files.writeString(about, m.id() + "\n" + fp + "\n");
            }
            pending = e.beginSave(file, fp); // takes the token list now; the rows are written below
        } catch (IOException | RuntimeException ex) {
            System.err.println("kv-cache: cannot save " + file + ": " + ex.getMessage());
            return;
        }
        if (pending == null) {
            return;
        }
        double beginMs = (System.nanoTime() - t0) / 1e6;
        Runnable write = () -> finish(m, slot, why, pending, fp, h, t0, beginMs);
        if (mode == Mode.NOW) {
            write.run();
        } else {
            Thread.ofVirtual().name("kv-cache-save").start(write);
        }
    }

    /** Writes the rows of a begun save and indexes the file. */
    private void finish(LoadedModel m, LoadedModel.Slot slot, String why, Engine.PendingSave pending, String fp,
                        String h, long t0, double beginMs) {
        Path file = pending.file();
        try {
            pending.write();
            int[] saved = pending.tokens();
            Entry fresh = new Entry(file, saved, Files.size(file), System.currentTimeMillis());
            System.err.printf("kv-cache: saved %d tokens of %s (slot %d, %s) to %s, %.0f MB in %.1f s (the slot was"
                            + " held %.0f ms)%n", saved.length, m.id(), slot.index(), why, file, fresh.bytes / 1e6,
                    (System.nanoTime() - t0) / 1e9, beginMs);
            List<Entry> stale = new ArrayList<>();
            synchronized (this) {
                TreeMap<int[], Entry> map = entries(h, fp, m.id());
                for (Entry en : map.values()) {
                    if (en.tokens.length < saved.length && commonPrefix(en.tokens, saved) == en.tokens.length) {
                        stale.add(en); // the conversation grew: the new file covers the old one
                    }
                }
                for (Entry en : stale) {
                    map.remove(en.tokens);
                }
                Entry dup = map.put(saved, fresh);
                if (dup != null && !dup.file.equals(file)) {
                    stale.add(dup); // saved twice meanwhile (two saves of the same tokens): keep the new file
                }
            }
            for (Entry en : stale) {
                Files.deleteIfExists(en.file);
            }
            evict();
        } catch (IOException | RuntimeException ex) {
            System.err.println("kv-cache: cannot save " + file + ": " + ex.getMessage());
            try {
                Files.deleteIfExists(file);
            } catch (IOException ignored) {
                // nothing more to do
            }
        }
    }

    private static void touch(Path f) {
        try {
            Files.setLastModifiedTime(f, java.nio.file.attribute.FileTime.fromMillis(System.currentTimeMillis()));
        } catch (IOException ignored) {
            // the in-memory time still orders the evictions of this run
        }
    }

    private void forget(String h, Entry en) {
        synchronized (this) {
            TreeMap<int[], Entry> map = index.get(h);
            if (map != null && map.get(en.tokens) == en) {
                map.remove(en.tokens);
            }
        }
        try {
            Files.deleteIfExists(en.file);
        } catch (IOException ignored) {
            // already gone or not ours to delete
        }
    }

    /** Deletes the least recently used files of the indexed models beyond the budget. */
    private void evict() {
        long max = maxBytes();
        List<Path> delete = new ArrayList<>();
        synchronized (this) {
            List<Map.Entry<String, Entry>> all = new ArrayList<>();
            index.forEach((h, map) -> map.values().forEach(en -> all.add(Map.entry(h, en))));
            long total = all.stream().mapToLong(x -> x.getValue().bytes).sum();
            all.sort(Comparator.comparingLong(x -> x.getValue().used));
            for (var x : all) {
                if (total <= max) {
                    break;
                }
                index.get(x.getKey()).remove(x.getValue().tokens);
                total -= x.getValue().bytes;
                delete.add(x.getValue().file);
            }
        }
        for (Path f : delete) {
            try {
                Files.deleteIfExists(f);
                System.err.println("kv-cache: deleted " + f + " (over the " + props.kvCacheMaxGb() + " GB budget)");
            } catch (IOException ignored) {
                // retried at the next eviction
            }
        }
    }

    private long maxBytes() {
        return (long) (props.kvCacheMaxGb() * 1e9);
    }

    /** Deletes every saved cache. */
    public synchronized int clear() throws IOException {
        int n = 0;
        for (TreeMap<int[], Entry> map : index.values()) {
            for (Entry en : map.values()) {
                if (Files.deleteIfExists(en.file)) {
                    n++;
                }
            }
            map.clear();
        }
        if (Files.isDirectory(dir)) {
            try (var walk = Files.walk(dir)) {
                for (Path f : walk.filter(p -> p.getFileName().toString().endsWith(".kv")).toList()) {
                    if (Files.deleteIfExists(f)) {
                        n++;
                    }
                }
            }
        }
        return n;
    }

    public synchronized View view() {
        if (Files.isDirectory(dir)) { // models not used since the start: index them from their model.txt
            try (var dirs = Files.list(dir)) {
                for (Path d : dirs.filter(Files::isDirectory).toList()) {
                    String h = d.getFileName().toString();
                    Path about = d.resolve("model.txt");
                    if (!index.containsKey(h) && Files.isRegularFile(about)) {
                        List<String> lines = Files.readAllLines(about);
                        if (lines.size() >= 2 && hash(lines.get(1)).equals(h)) {
                            entries(h, lines.get(1), lines.get(0));
                        }
                    }
                }
            } catch (IOException e) {
                System.err.println("kv-cache: cannot list " + dir + ": " + e.getMessage());
            }
        }
        List<SavedView> saved = new ArrayList<>();
        long bytes = 0;
        for (var en : index.entrySet()) {
            for (Entry e : en.getValue().values()) {
                saved.add(new SavedView(names.getOrDefault(en.getKey(), en.getKey()), e.tokens.length, e.bytes, e.used));
                bytes += e.bytes;
            }
        }
        saved.sort(Comparator.comparingLong(SavedView::usedAt).reversed());
        return new View(enabled(), dir.toString(), saved.size(), bytes, maxBytes(), props.kvCacheMinTokens(),
                props.kvCacheSaveAfterMs(), saved);
    }
}
