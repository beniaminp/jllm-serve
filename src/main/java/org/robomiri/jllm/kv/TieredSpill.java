package org.robomiri.jllm.kv;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The warm tier of the KV blocks (docs/26): a bounded, least recently used cache of block payloads in host memory
 * (off the Java heap) in front of a slower {@link KvSpill}, usually the SSD block store ({@link FileSpill}, possibly
 * through a {@link CodecSpill} that stores the rows smaller).
 *
 * <ul>
 *   <li>{@link #put} (a block leaves the hot pool) keeps the payload in host memory. When the tier is over its budget the
 *       least recently used payloads are written to the lower store by a background thread and then dropped from memory;
 *       until the write is done the payload stays readable here, so a block is never in neither tier.</li>
 *   <li>{@link #get} serves from memory (fast) or, failing that, from the lower store.</li>
 *   <li>{@link #prefetch} (the pool saw a prompt that will adopt this block) reads a block from the lower store into
 *       memory in the background, so that the adoption that follows finds it in host memory.</li>
 * </ul>
 * With a budget of 0 it is a plain pass-through. Thread safe.
 */
public final class TieredSpill implements KvSpill, AutoCloseable {

    private static final class Entry {
        final int[] tokens;
        final ByteBuffer data;
        boolean demoting;

        Entry(int[] tokens, ByteBuffer data) {
            this.tokens = tokens;
            this.data = data;
        }
    }

    /** Counters for the console and the tests. */
    public record Stats(int ramBlocks, long ramBytes, long ramBudget, long ramHits, long lowerHits, long demotions,
                        long prefetches, long prefetchHits, long dropped) {
    }

    private final KvSpill lower;
    private final long budget;
    private final Map<Long, Entry> ram = new LinkedHashMap<>(64, 0.75f, true);
    private final Set<Long> prefetching = new HashSet<>();
    private final ExecutorService io;
    private long ramBytes;
    private final AtomicLong ramHits = new AtomicLong();
    private final AtomicLong lowerHits = new AtomicLong();
    private final AtomicLong demotions = new AtomicLong();
    private final AtomicLong prefetches = new AtomicLong();
    private final AtomicLong prefetchHits = new AtomicLong();
    private final AtomicLong dropped = new AtomicLong();
    private final Set<Long> prefetched = new HashSet<>();
    private int pendingTasks;

    /**
     * @param lower  where payloads go when this tier is full (null: they are dropped)
     * @param budget bytes of host memory this tier may hold (0: nothing is kept here)
     */
    public TieredSpill(KvSpill lower, long budget) {
        this.lower = lower;
        this.budget = Math.max(0, budget);
        this.io = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "jllm-kv-tier");
            t.setDaemon(true);
            return t;
        });
    }

    public long budget() {
        return budget;
    }

    @Override
    public synchronized boolean has(long pathHash) {
        return ram.containsKey(pathHash) || (lower != null && lower.has(pathHash));
    }

    @Override
    public boolean put(long pathHash, int[] tokens, ByteBuffer payload) {
        int n = payload.remaining();
        if (budget == 0 || n > budget) {
            return lower != null && lower.put(pathHash, tokens, payload);
        }
        ByteBuffer copy = ByteBuffer.allocateDirect(n).order(payload.order());
        copy.put(payload.duplicate()).flip();
        Entry e = new Entry(tokens.clone(), copy);
        ArrayList<Long> overflow = new ArrayList<>();
        ArrayList<Entry> overflowEntries = new ArrayList<>();
        synchronized (this) {
            Entry old = ram.put(pathHash, e);
            ramBytes += n - (old == null ? 0 : old.data.capacity());
            prefetched.remove(pathHash);
            collect(overflow, overflowEntries);
        }
        demote(overflow, overflowEntries);
        return true;
    }

    /** Picks the least recently used payloads to push down until the rest fits the budget (under the lock). */
    private void collect(ArrayList<Long> hashes, ArrayList<Entry> entries) {
        long over = ramBytes;
        for (Entry x : ram.values()) {
            if (x.demoting) {
                over -= x.data.capacity();
            }
        }
        Iterator<Map.Entry<Long, Entry>> it = ram.entrySet().iterator();
        while (over > budget && it.hasNext()) {
            Map.Entry<Long, Entry> me = it.next();
            Entry x = me.getValue();
            if (x.demoting) {
                continue;
            }
            x.demoting = true;
            hashes.add(me.getKey());
            entries.add(x);
            over -= x.data.capacity();
        }
    }

    private void demote(ArrayList<Long> hashes, ArrayList<Entry> entries) {
        for (int i = 0; i < hashes.size(); i++) {
            long h = hashes.get(i);
            Entry x = entries.get(i);
            Runnable task = () -> {
                try {
                    boolean ok = lower != null && lower.put(h, x.tokens, x.data.duplicate());
                    if (ok) {
                        demotions.incrementAndGet();
                    } else {
                        dropped.incrementAndGet();
                    }
                } finally {
                    synchronized (TieredSpill.this) {
                        if (ram.get(h) == x) {
                            ram.remove(h);
                            ramBytes -= x.data.capacity();
                        }
                        pendingTasks--;
                        TieredSpill.this.notifyAll();
                    }
                }
            };
            synchronized (this) {
                pendingTasks++;
            }
            try {
                io.execute(task);
            } catch (RuntimeException closed) {
                task.run(); // the executor is shut down: write on the caller's thread
            }
        }
    }

    @Override
    public boolean get(long pathHash, int[] tokens, ByteBuffer payload) {
        Entry e;
        synchronized (this) {
            e = ram.get(pathHash);
            if (e != null && Arrays.equals(e.tokens, tokens) && e.data.capacity() == payload.remaining()) {
                payload.put(e.data.duplicate());
                ramHits.incrementAndGet();
                if (prefetched.remove(pathHash)) {
                    prefetchHits.incrementAndGet();
                }
                return true;
            }
        }
        if (lower != null && lower.get(pathHash, tokens, payload)) {
            lowerHits.incrementAndGet();
            return true;
        }
        return false;
    }

    @Override
    public void prefetch(long pathHash, int[] tokens, int payloadBytes) {
        if (lower == null || budget == 0 || payloadBytes > budget) {
            return;
        }
        synchronized (this) {
            if (ram.containsKey(pathHash) || !prefetching.add(pathHash)) {
                return;
            }
            pendingTasks++;
        }
        int[] tk = tokens.clone();
        Runnable task = () -> {
            try {
                if (!lower.has(pathHash)) {
                    return;
                }
                ByteBuffer buf = ByteBuffer.allocateDirect(payloadBytes);
                if (!lower.get(pathHash, tk, buf)) {
                    return;
                }
                buf.flip();
                ArrayList<Long> overflow = new ArrayList<>();
                ArrayList<Entry> overflowEntries = new ArrayList<>();
                synchronized (TieredSpill.this) {
                    if (!ram.containsKey(pathHash)) {
                        ram.put(pathHash, new Entry(tk, buf));
                        ramBytes += payloadBytes;
                        prefetched.add(pathHash);
                        prefetches.incrementAndGet();
                        collect(overflow, overflowEntries);
                    }
                }
                demote(overflow, overflowEntries);
            } finally {
                synchronized (TieredSpill.this) {
                    prefetching.remove(pathHash);
                    pendingTasks--;
                    TieredSpill.this.notifyAll();
                }
            }
        };
        try {
            io.execute(task);
        } catch (RuntimeException closed) {
            synchronized (this) {
                prefetching.remove(pathHash);
                pendingTasks--;
            }
        }
    }

    @Override
    public void remove(long pathHash) {
        synchronized (this) {
            Entry e = ram.remove(pathHash);
            if (e != null) {
                ramBytes -= e.data.capacity();
            }
            prefetched.remove(pathHash);
        }
        if (lower != null) {
            lower.remove(pathHash);
        }
    }

    @Override
    public synchronized int files() {
        return ram.size() + (lower == null ? 0 : lower.files());
    }

    @Override
    public synchronized long bytes() {
        return ramBytes + (lower == null ? 0 : lower.bytes());
    }

    public synchronized Stats stats() {
        return new Stats(ram.size(), ramBytes, budget, ramHits.get(), lowerHits.get(), demotions.get(), prefetches.get(),
                prefetchHits.get(), dropped.get());
    }

    /** Waits until the background writes and prefetches queued so far are done (tests, shutdown). */
    public synchronized void drain() {
        boolean interrupted = false;
        while (pendingTasks > 0) {
            try {
                wait(100);
            } catch (InterruptedException e) {
                interrupted = true;
                break;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    /** Writes everything held in memory to the lower store (a clean shutdown keeps the blocks). */
    public void flush() {
        ArrayList<Long> hashes = new ArrayList<>();
        ArrayList<Entry> entries = new ArrayList<>();
        synchronized (this) {
            for (Map.Entry<Long, Entry> me : ram.entrySet()) {
                if (!me.getValue().demoting) {
                    me.getValue().demoting = true;
                    hashes.add(me.getKey());
                    entries.add(me.getValue());
                }
            }
        }
        demote(hashes, entries);
        drain();
    }

    @Override
    public void close() {
        flush();
        io.shutdown();
    }
}
