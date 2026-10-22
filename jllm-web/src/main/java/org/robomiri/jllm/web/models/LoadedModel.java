package org.robomiri.jllm.web.models;

import org.robomiri.jllm.ModelLoader;
import org.robomiri.jllm.engine.Admission;
import org.robomiri.jllm.engine.BatchGroup;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.Priority;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * A model in memory, ready to serve. Requests hold a {@link Lease} while they use the engine, so that unloading
 * waits for them (after asking them to stop) before the backend's buffers and the mapped weights are released.
 *
 * <p>A model may have several <em>slots</em>: engines with their own KV cache on the same mapped weights. Each
 * request goes to the slot that already holds the longest prefix of its prompt, so a coding agent's long
 * conversation stays cached while its short side requests (titles, summaries) run in another slot, and two
 * requests can run at the same time.
 */
public final class LoadedModel {

    /** Thrown when a request arrives for a model that is being unloaded. */
    public static final class UnavailableException extends RuntimeException {
        UnavailableException(String message) {
            super(message);
        }
    }

    public interface Lease extends AutoCloseable {
        @Override
        void close();
    }

    /**
     * @param settings the load-time settings of the model (backend, context, quantization, speculation)
     * @param slots    the KV cache slots; {@code busy} of them are running or reserved by a request
     */
    public record View(String id, String source, String description, String backend, int contextLength,
                       String quant, long weightBytes, String template, long loadedAt, double loadSeconds,
                       long requests, long generatedTokens, double lastTokensPerSecond, int active,
                       Map<String, String> settings, int slots, int busy, boolean vision, String projector, String kind) {
    }

    /** One engine (KV cache) of the model, reserved for a request until closed. */
    public static final class Slot implements AutoCloseable {
        private final Engine engine;
        private final int index;
        private final AtomicInteger users = new AtomicInteger();
        private volatile long lastUsed;
        private volatile Admission<Slot> admission; // the model's queue under engine.scheduler=mixed

        Slot(Engine engine, int index) {
            this.engine = engine;
            this.index = index;
        }

        public Engine engine() {
            return engine;
        }

        public int index() {
            return index;
        }

        @Override
        public void close() {
            lastUsed = System.nanoTime();
            users.decrementAndGet();
            Admission<Slot> q = admission;
            if (q != null) {
                if (engine.backend() instanceof BatchGroup.Member m) {
                    m.resume(); // a pause asked for after the last checkpoint must not catch the next request
                }
                engine.releaseCache(); // the conversation stays in the prefix tree, the slot stops pinning its blocks
                q.release(this); // the next request in line gets the slot
            }
        }
    }

    private final String id;
    private final String source;
    private final KvStore store; // null: no saved caches
    private final ModelLoader.Loaded loaded;
    private final List<Slot> slots = new ArrayList<>();
    private final Map<String, String> settings;
    private final long loadedAt = System.currentTimeMillis();
    private final double loadSeconds;
    private final ReentrantReadWriteLock guard = new ReentrantReadWriteLock();
    private final AtomicLong requests = new AtomicLong();
    private final AtomicLong generated = new AtomicLong();
    private final AtomicInteger active = new AtomicInteger();
    private volatile double lastTokPerSec;
    private volatile boolean closing;
    /** The one queue of the model under {@code engine.scheduler=mixed} (docs/25); null with the slots scheduler. */
    private final Admission<Slot> admission;

    /** @param extra the engines of slots 1.. (slot 0 is {@code loaded}'s) */
    LoadedModel(String id, String source, ModelLoader.Loaded loaded, List<Engine> extra, Map<String, String> settings,
                double loadSeconds, KvStore store) {
        this.id = id;
        this.source = source;
        this.store = store;
        this.loaded = loaded;
        slots.add(new Slot(loaded.engine(), 0));
        for (Engine e : extra) {
            slots.add(new Slot(e, slots.size()));
        }
        this.settings = Map.copyOf(settings);
        this.loadSeconds = loadSeconds;
        this.admission = admissionFor(this.settings);
        if (admission != null) {
            slots.forEach(sl -> sl.admission = admission);
        }
    }

    /** The queue of a model whose slots form a mixed group; null for the slots scheduler (or a model that cannot mix). */
    private Admission<Slot> admissionFor(Map<String, String> st) {
        if (!(slots.getFirst().engine.backend() instanceof BatchGroup.Member m) || !m.group().mixed()) {
            return null;
        }
        int maxRunning = Integer.parseInt(st.getOrDefault("engine.max-running", "0"));
        boolean preempt = Boolean.parseBoolean(st.getOrDefault("engine.preempt", "true"));
        int sessionMax = Integer.parseInt(st.getOrDefault("engine.session-max-running", "0"));
        long agePromote = Long.parseLong(st.getOrDefault("engine.age-promote-ms", "10000")) * 1_000_000L;
        return new Admission<>(slots, new Admission.Ops<>() {
            @Override
            public int match(Slot slot, int[] prompt) {
                return slot.engine.cachedPrefix(prompt);
            }

            @Override
            public boolean pause(Slot slot) {
                if (!(slot.engine.backend() instanceof BatchGroup.Member mm) || !mm.requestPause()) {
                    return false;
                }
                // the blocks are back in the pool before the newcomer asks for its own (unless the request ends meanwhile)
                long end = System.nanoTime() + PAUSE_WAIT_MS * 1_000_000L;
                while (System.nanoTime() < end && slot.engine.busy() && !mm.awaitPaused(25)) {
                    // waiting for the generation to reach its next step boundary
                }
                return true;
            }

            @Override
            public void resume(Slot slot) {
                if (slot.engine.backend() instanceof BatchGroup.Member mm) {
                    mm.resume();
                }
            }
        }, new Admission.Config(maxRunning, preempt, sessionMax, agePromote));
    }

    /** How long a request waits for a generation it pauses to reach its next checkpoint (one step, normally milliseconds). */
    static final long PAUSE_WAIT_MS = 3000;

    /** The queue of the mixed scheduler, or null. */
    public Admission<Slot> admission() {
        return admission;
    }

    public String id() {
        return id;
    }

    public String source() {
        return source;
    }

    /** The first slot's engine: tokenizer, template and context are the same for all slots. */
    public Engine engine() {
        return loaded.engine();
    }

    /** The load-time settings (those of {@link View#settings}). */
    public Map<String, String> settings() {
        return settings;
    }

    List<Slot> slots() {
        return slots;
    }

    /**
     * Reserves the slot to run {@code prompt} (prepared ids) on: an idle slot with the longest cached prefix, unless
     * a busy slot holds a much longer one (waiting for it beats prefilling a long conversation again). A conversation
     * saved to disk ({@link KvStore}) that covers more of the prompt is then loaded into the slot.
     */
    public Slot pick(int[] prompt) {
        return pick(prompt, null, () -> false);
    }

    /**
     * {@link #pick(int[])} for the request of {@code session} (the OpenAI {@code user}, or null) whose client's presence
     * is {@code gone}. Under {@code engine.scheduler=mixed} the request waits in the model's one queue (docs/25): the
     * waiting request with the most cached tokens, in the highest priority class ({@link Priority#current()}), gets a
     * freed slot, and a request whose client left while it waited is dropped from the queue with
     * {@link Admission.Abandoned}.
     */
    public Slot pick(int[] prompt, String session, java.util.function.BooleanSupplier gone) {
        if (admission != null) {
            Slot s = admission.acquire(prompt, Priority.current().ordinal(), session, () -> closing || gone.getAsBoolean());
            s.users.incrementAndGet();
            restore(s, prompt);
            return s;
        }
        Choice c = choose(prompt);
        Slot s = c.slot();
        if (c.donor() != null) {
            try {
                long t0 = System.nanoTime();
                int got = s.engine.adoptPrefix(c.donor().engine, prompt);
                if (got > 0) {
                    System.err.printf("slots: copied a %d-token prefix of %s from slot %d into slot %d in %.0f ms (slot %d"
                                    + " keeps its conversation)%n", s.engine.cachedLength(), id, c.donor().index, s.index,
                            (System.nanoTime() - t0) / 1e6, c.donor().index);
                }
            } catch (RuntimeException e) {
                System.err.println("slots: prefix copy failed: " + e);
            } finally {
                c.donor().users.decrementAndGet();
            }
        }
        restore(s, prompt);
        return s;
    }

    private void restore(Slot s, int[] prompt) {
        if (store != null) {
            try {
                store.restore(this, s, prompt);
            } catch (RuntimeException e) {
                System.err.println("kv-cache: restore failed: " + e);
            }
        }
    }

    /** Copying a prefix from another slot must save at least this many tokens of prefill. */
    static final int MIN_COPY = 256;

    /** The slot a request runs on, and the idle slot to copy its prompt's prefix from first (or null). */
    private record Choice(Slot slot, Slot donor) {
    }

    /**
     * Slot policy, a fork-style prefix cache over the slots:
     * <ol>
     *   <li>an idle slot whose whole cache the prompt continues (the next turn of its conversation) runs it in place;</li>
     *   <li>otherwise the prompt runs on the idle slot that matters least (empty, else least recently used), after
     *       copying the longest matching prefix from another idle slot into it when that saves at least
     *       {@link #MIN_COPY} tokens of prefill: the donor keeps its own conversation instead of being overwritten
     *       from the point where the prompts diverge (an agent's sessions share ~10k tokens of instructions);</li>
     *   <li>when every slot is busy, the one with the longest prefix is waited for.</li>
     * </ol>
     */
    private synchronized Choice choose(int[] prompt) {
        // a paged KV cache shares blocks through the pool's prefix tree, whichever slot runs the prompt: the engine takes
        // over what any slot or conversation computed (Engine.generate), so no slot needs to be picked for its prefix
        // and nothing is copied; the slot whose own conversation the prompt continues still saves the table rebuild
        boolean paged = slots.getFirst().engine.backend().kvSequence() != null;
        Slot cont = null;
        int contLen = 0;
        Slot donor = null;
        int donorLen = 0;
        Slot victim = null;
        for (Slot s : slots) {
            if (s.users.get() > 0) {
                continue;
            }
            int len = s.engine.cachedPrefix(prompt);
            int all = s.engine.cachedLength();
            if (len > 0 && len >= Math.min(all, prompt.length - 1) && len > contLen) {
                cont = s;
                contLen = len;
            }
            if (len > donorLen && !paged) {
                donor = s;
                donorLen = len;
            }
            if (victim == null || worthLess(s, victim)) {
                victim = s;
            }
        }
        if (cont != null && (paged || contLen >= donorLen)) {
            cont.users.incrementAndGet();
            return new Choice(cont, null);
        }
        if (victim == null) { // all busy: wait for the one holding the longest prefix
            Slot best = slots.getFirst();
            int bestLen = -1;
            for (Slot s : slots) {
                int len = s.engine.cachedPrefix(prompt);
                if (len > bestLen || (len == bestLen && s.users.get() < best.users.get())) {
                    best = s;
                    bestLen = len;
                }
            }
            best.users.incrementAndGet();
            return new Choice(best, null);
        }
        if (donor == null || donor == victim) {
            victim.users.incrementAndGet();
            return new Choice(victim, null);
        }
        int victimLen = victim.engine.cachedPrefix(prompt);
        int donorTail = donor.engine.cachedLength() - donorLen; // what the donor would lose if the prompt ran there
        if (donorLen - victimLen < MIN_COPY || donorTail < MIN_COPY) {
            donor.users.incrementAndGet(); // little to gain by copying, or little to lose: run on the donor itself
            return new Choice(donor, null);
        }
        victim.users.incrementAndGet();
        donor.users.incrementAndGet(); // kept idle until the copy is done
        return new Choice(victim, donor);
    }

    /** Empty slots first, then the least recently used. */
    private static boolean worthLess(Slot a, Slot b) {
        int la = a.engine.cachedLength();
        int lb = b.engine.cachedLength();
        if ((la == 0) != (lb == 0)) {
            return la == 0;
        }
        return a.lastUsed < b.lastUsed;
    }

    public int slotCount() {
        return slots.size();
    }

    /** Slots a request is running on or has reserved right now. */
    public int busySlots() {
        return (int) slots.stream().filter(s -> s.users.get() > 0).count();
    }

    /** True once unloading started: running generations should stop. */
    public boolean closing() {
        return closing;
    }

    /** Marks the start of a request on this model. */
    public Lease lease() {
        if (closing) {
            throw new UnavailableException("model " + id + " is being unloaded");
        }
        guard.readLock().lock();
        if (closing) {
            guard.readLock().unlock();
            throw new UnavailableException("model " + id + " is being unloaded");
        }
        requests.incrementAndGet();
        active.incrementAndGet();
        return () -> {
            active.decrementAndGet();
            guard.readLock().unlock();
        };
    }

    /** Records a finished request on {@code slot}; an expensive prefill gets its cache saved to disk. */
    public void record(Slot slot, Engine.Stats stats) {
        record(stats);
        if (store != null) {
            store.afterRequest(this, slot, stats);
        }
    }

    public void record(Engine.Stats stats) {
        generated.addAndGet(stats.generatedTokens());
        if (stats.generatedTokens() > 1) {
            lastTokPerSec = stats.tokPerSec();
        }
    }

    /** Stops new requests, waits for the running ones (which see {@link #closing()}) and frees the model. */
    boolean close(long timeoutMs) throws InterruptedException {
        closing = true;
        if (!guard.writeLock().tryLock(timeoutMs, TimeUnit.MILLISECONDS)) {
            return false;
        }
        try {
            if (store != null) {
                try {
                    store.saveAll(this);
                } catch (RuntimeException e) {
                    System.err.println("kv-cache: saving on unload failed: " + e);
                }
            }
            for (int i = 1; i < slots.size(); i++) {
                slots.get(i).engine.close();
            }
            loaded.close();
        } finally {
            guard.writeLock().unlock();
        }
        return true;
    }

    public View view() {
        var w = loaded.weights();
        var e = loaded.engine();
        return new View(id, source, w.arch.toString(), e.backend().name(), e.backend().contextLength(),
                String.valueOf(w.quant), w.byteSize(), e.template().name(), loadedAt, loadSeconds, requests.get(),
                generated.get(), Math.round(lastTokPerSec * 10) / 10.0, active.get(), settings, slots.size(),
                (int) slots.stream().filter(s -> s.users.get() > 0).count(), e.vision() != null,
                e.vision() == null ? null : e.vision().describe(), "text");
    }
}
