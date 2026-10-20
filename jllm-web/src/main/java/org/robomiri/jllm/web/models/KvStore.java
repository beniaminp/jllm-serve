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
 *
 * <ul>
 *   <li><b>Save</b>: a slot's cache is written after a request whose prefill took at least
 *       {@code jllm.web.kv-cache-save-after-ms}, when a model is unloaded or the server stops, and before a restore
 *       overwrites it; only caches of at least {@code jllm.web.kv-cache-min-tokens} tokens. A cache that extends a
 *       saved one replaces it (one file per conversation); one that a saved file already covers is not written.</li>
 *   <li><b>Restore</b>: when a request is assigned a slot, the saved cache sharing the longest prefix with its
 *       prompt is loaded into the slot if it covers at least 256 more tokens than the slot holds.</li>
 *   <li><b>Budget</b>: the least recently used files are deleted beyond {@code jllm.web.kv-cache-max-gb}.</li>
 * </ul>
 *
 * Files live in {@code <kv-cache-dir>/<fingerprint hash>/}: a saved cache is only reused by the same model with the
 * same load settings that change keys and values (quantization, backend, cache type, int8 activations).
 */
@Service
public class KvStore {

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
    /** Fingerprint hash -> entries; a directory is scanned the first time its model is used. */
    private final Map<String, List<Entry>> index = new LinkedHashMap<>();
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
    private synchronized List<Entry> entries(String h, String fingerprint, String modelId) {
        names.put(h, modelId);
        List<Entry> list = index.get(h);
        if (list != null) {
            return list;
        }
        list = new ArrayList<>();
        Path d = dir.resolve(h);
        if (Files.isDirectory(d)) {
            try (var files = Files.list(d)) {
                for (Path f : files.filter(p -> p.getFileName().toString().endsWith(".kv")).toList()) {
                    try {
                        CacheFile.Header hd = CacheFile.readHeader(f);
                        if (hd.fingerprint().equals(fingerprint)) {
                            list.add(new Entry(f, hd.tokens(), Files.size(f), Files.getLastModifiedTime(f).toMillis()));
                        }
                    } catch (IOException e) {
                        System.err.println("kv-cache: ignoring unreadable " + f + ": " + e.getMessage());
                    }
                }
            } catch (IOException e) {
                System.err.println("kv-cache: cannot list " + d + ": " + e.getMessage());
            }
        }
        index.put(h, list);
        return list;
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
     * Before a request runs on {@code slot}: loads the saved cache that covers the most of {@code prompt}, when that
     * beats what the slot holds. The slot's own cache is saved first if it is worth keeping.
     */
    public void restore(LoadedModel m, LoadedModel.Slot slot, int[] prompt) {
        Engine e = slot.engine();
        if (!enabled() || !e.canPersistCache() || prompt.length <= MIN_GAIN) {
            return;
        }
        String fp = fingerprint(m);
        String h = hash(fp);
        int have = e.cachedPrefix(prompt);
        Entry best = null;
        int bestLen = 0;
        synchronized (this) {
            for (Entry en : entries(h, fp, m.id())) {
                int len = Math.min(commonPrefix(en.tokens, prompt), prompt.length - 1);
                if (len > bestLen) {
                    best = en;
                    bestLen = len;
                }
            }
        }
        if (best == null || bestLen - have < MIN_GAIN) {
            return;
        }
        save(m, slot, "before restoring another conversation into slot " + slot.index());
        long t0 = System.nanoTime();
        try {
            int n = e.loadCache(best.file, fp);
            best.used = System.currentTimeMillis();
            touch(best.file);
            System.err.printf("kv-cache: restored %d tokens of %s into slot %d from %s in %.1f s (%d of the prompt's %d"
                            + " tokens reusable)%n", n, m.id(), slot.index(), best.file.getFileName(),
                    (System.nanoTime() - t0) / 1e9, bestLen, prompt.length);
        } catch (IOException | RuntimeException ex) {
            System.err.println("kv-cache: cannot restore " + best.file + ": " + ex.getMessage() + " (deleting it)");
            forget(h, best);
        }
    }

    /** After a request: saves the slot's cache in the background when the prefill was expensive. */
    public void afterRequest(LoadedModel m, LoadedModel.Slot slot, Engine.Stats stats) {
        if (enabled() && slot.engine().canPersistCache() && stats.prefillMs() >= props.kvCacheSaveAfterMs()) {
            Thread.ofVirtual().name("kv-cache-save").start(() -> save(m, slot, "after a " + Math.round(stats.prefillMs() / 1000)
                    + " s prefill"));
        }
    }

    /** Saves every slot of a model (unload, shutdown, or on request). */
    public void saveAll(LoadedModel m) {
        if (!enabled()) {
            return;
        }
        for (LoadedModel.Slot s : m.slots()) {
            save(m, s, "on unload");
        }
    }

    /** Writes the slot's cache unless it is short or a saved file already holds it. */
    void save(LoadedModel m, LoadedModel.Slot slot, String why) {
        Engine e = slot.engine();
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
            for (Entry en : entries(h, fp, m.id())) {
                if (en.tokens.length >= tokens.length && commonPrefix(en.tokens, tokens) == tokens.length) {
                    en.used = System.currentTimeMillis();
                    return; // already on disk (the same tokens, or more)
                }
            }
        }
        Path file = dir.resolve(h).resolve(System.currentTimeMillis() + "-" + tokens.length + ".kv");
        long t0 = System.nanoTime();
        try {
            Files.createDirectories(file.getParent());
            Path about = file.getParent().resolve("model.txt");
            if (!Files.exists(about)) {
                Files.writeString(about, m.id() + "\n" + fp + "\n");
            }
            if (e.saveCache(file, fp) == 0) {
                return;
            }
            int[] saved = CacheFile.readHeader(file).tokens(); // the cache may have moved on since the check above
            Entry fresh = new Entry(file, saved, Files.size(file), System.currentTimeMillis());
            System.err.printf("kv-cache: saved %d tokens of %s (slot %d, %s) to %s, %.0f MB in %.1f s%n", saved.length,
                    m.id(), slot.index(), why, file, fresh.bytes / 1e6, (System.nanoTime() - t0) / 1e9);
            List<Entry> stale = new ArrayList<>();
            synchronized (this) {
                List<Entry> list = entries(h, fp, m.id());
                for (Entry en : list) {
                    if (en.tokens.length < saved.length && commonPrefix(en.tokens, saved) == en.tokens.length) {
                        stale.add(en); // the conversation grew: the new file covers the old one
                    }
                }
                list.removeAll(stale);
                list.add(fresh);
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
            List<Entry> list = index.get(h);
            if (list != null) {
                list.remove(en);
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
            index.forEach((h, list) -> list.forEach(en -> all.add(Map.entry(h, en))));
            long total = all.stream().mapToLong(x -> x.getValue().bytes).sum();
            all.sort(Comparator.comparingLong(x -> x.getValue().used));
            for (var x : all) {
                if (total <= max) {
                    break;
                }
                index.get(x.getKey()).remove(x.getValue());
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
        for (List<Entry> list : index.values()) {
            for (Entry en : list) {
                if (Files.deleteIfExists(en.file)) {
                    n++;
                }
            }
            list.clear();
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
            for (Entry e : en.getValue()) {
                saved.add(new SavedView(names.getOrDefault(en.getKey(), en.getKey()), e.tokens.length, e.bytes, e.used));
                bytes += e.bytes;
            }
        }
        saved.sort(Comparator.comparingLong(SavedView::usedAt).reversed());
        return new View(enabled(), dir.toString(), saved.size(), bytes, maxBytes(), props.kvCacheMinTokens(),
                props.kvCacheSaveAfterMs(), saved);
    }
}
