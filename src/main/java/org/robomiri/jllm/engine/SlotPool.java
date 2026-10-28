package org.robomiri.jllm.engine;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;

/**
 * The slots of one model: engines with their own KV cache on the same mapped weights, so several requests (several users,
 * or an agent and its side requests) run at the same time. Their decode steps share forward passes when the engines are
 * members of one {@link BatchGroup} ({@code engine.batch-decode}), and with {@code engine.scheduler=mixed} every request
 * waits in the model's one {@link Admission} queue (docs/25). Used by {@code ./jllm serve} ({@code server.slots}, docs/10
 * §10.9c) and by the web console's models.
 *
 * <p>A request {@linkplain #pick picks} the slot that serves it and {@linkplain Slot#close closes} it when done. The choice
 * is a fork-style prefix cache over the slots (see {@link #choose}): the slot whose conversation the prompt continues, else
 * the idle slot that matters least, after copying a long matching prefix into it from another idle slot; with a paged KV
 * cache (the default from two slots on) the pool's prefix tree shares the blocks whichever slot runs the prompt.
 */
public final class SlotPool {

    /** Loads a conversation saved elsewhere (the console's disk store) into a slot that is about to run {@code prompt}. */
    public interface Restorer {
        void restore(Slot slot, int[] prompt);
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

    /** Copying a prefix from another slot must save at least this many tokens of prefill. */
    public static final int MIN_COPY = 256;

    /** How long a request waits for a generation it pauses to reach its next checkpoint (one step, normally milliseconds). */
    public static final long PAUSE_WAIT_MS = 3000;

    private final List<Slot> slots = new ArrayList<>();
    private final Restorer restorer;
    private final Admission<Slot> admission;
    private final String name;
    private volatile Thread sweeper;
    private volatile boolean closed;

    /**
     * @param engines the slots' engines, the first one owning the tokenizer, template and vision encoder
     * @param mixed   the queue settings of {@code engine.scheduler=mixed}; used only when the engines form a mixed
     *                {@link BatchGroup} (else the slots scheduler applies), null for the defaults
     * @param restorer called with the chosen slot before a request runs (null: nothing to restore)
     * @param name     for the log lines (the model's id)
     */
    public SlotPool(List<Engine> engines, Admission.Config mixed, Restorer restorer, String name) {
        if (engines.isEmpty()) {
            throw new IllegalArgumentException("a slot pool needs at least one engine");
        }
        for (Engine e : engines) {
            slots.add(new Slot(e, slots.size()));
        }
        this.restorer = restorer;
        this.name = name;
        this.admission = admissionFor(mixed);
        if (admission != null) {
            slots.forEach(sl -> sl.admission = admission);
        }
    }

    /** The queue settings of {@code engine.scheduler=mixed} from {@code s}. */
    public static Admission.Config mixedConfig(org.robomiri.jllm.config.Settings s) {
        return new Admission.Config(s.getInt("engine.max-running"), s.getBool("engine.preempt"),
                s.getInt("engine.session-max-running"), s.getLong("engine.age-promote-ms") * 1_000_000L);
    }

    /** The queue of a pool whose slots form a mixed group; null for the slots scheduler (or a model that cannot mix). */
    private Admission<Slot> admissionFor(Admission.Config cfg) {
        if (!(slots.getFirst().engine.backend() instanceof BatchGroup.Member m) || !m.group().mixed()) {
            return null;
        }
        Admission.Config c = cfg != null ? cfg : new Admission.Config(0, true, 0, 10_000L * 1_000_000L);
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
        }, c);
    }

    /** The queue of the mixed scheduler, or null. */
    public Admission<Slot> admission() {
        return admission;
    }

    /** The first slot's engine: tokenizer, template, context and vision encoder are the same for all slots. */
    public Engine first() {
        return slots.getFirst().engine;
    }

    public List<Slot> slots() {
        return slots;
    }

    public int size() {
        return slots.size();
    }

    /** Slots a request is running on or has reserved right now. */
    public int busy() {
        return (int) slots.stream().filter(s -> s.users.get() > 0).count();
    }

    /** {@link #pick(int[], String, BooleanSupplier)} without a session, for a client that stays. */
    public Slot pick(int[] prompt) {
        return pick(prompt, null, () -> false);
    }

    /** {@link #pick(int[], String, BooleanSupplier, long)} with no limit on the time queued. */
    public Slot pick(int[] prompt, String session, BooleanSupplier gone) {
        return pick(prompt, session, gone, 0);
    }

    /**
     * Reserves the slot to run {@code prompt} (prepared ids) on, for the request of {@code session} (the OpenAI {@code user},
     * or null) whose client's presence is {@code gone}. With the slots scheduler: {@link #choose}. Under
     * {@code engine.scheduler=mixed} the request waits in the pool's one queue (docs/25): the waiting request with the most
     * cached tokens, in the highest priority class ({@link Priority#current()}), gets a freed slot, and a request whose client
     * left while it waited is dropped with {@link Admission.Abandoned}.
     *
     * @param queueTimeoutMs under the mixed scheduler, how long the request may wait in the queue before
     *                       {@link Engine.OverloadedException} ({@code server.queue-timeout-ms}, the 503 of the API server);
     *                       0 = as long as the client stays. With the slots scheduler the slot's engine applies its own
     *                       queue timeout.
     */
    public Slot pick(int[] prompt, String session, BooleanSupplier gone, long queueTimeoutMs) {
        if (admission != null) {
            long t0 = System.nanoTime();
            boolean[] late = new boolean[1];
            Slot s;
            try {
                s = admission.acquire(prompt, Priority.current().ordinal(), session, () -> closed || gone.getAsBoolean()
                        || (late[0] = queueTimeoutMs > 0 && System.nanoTime() - t0 > queueTimeoutMs * 1_000_000L));
            } catch (Admission.Abandoned e) {
                if (late[0]) {
                    throw new Engine.OverloadedException("all " + slots.size() + " slots of " + name + " stayed busy for "
                            + queueTimeoutMs + " ms (server.queue-timeout-ms)");
                }
                throw e;
            }
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
                                    + " keeps its conversation)%n", s.engine.cachedLength(), name, c.donor().index, s.index,
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
        if (restorer != null) {
            try {
                restorer.restore(s, prompt);
            } catch (RuntimeException e) {
                System.err.println("kv-cache: restore failed: " + e);
            }
        }
    }

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
     *   <li>when every slot is busy, the one with the longest prefix is waited for (on its engine's queue).</li>
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
        if (victim == null) { // all busy: wait for the one holding the longest prefix, else the least loaded
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

    /**
     * A paged model's slots keep their conversation's blocks referenced until they move on to another conversation. This
     * gives the memory back (docs/24 §24.2.8): a slot that has been idle for {@code idleSeconds} releases its blocks (the
     * conversation stays in the pool's prefix tree and is found again by the next prompt, or evicted when the memory is
     * needed), and when the pool runs out of blocks it asks every idle slot at once. Does nothing for a flat cache.
     */
    public void enablePagedHousekeeping(long idleSeconds) {
        org.robomiri.jllm.kv.KvSequence seq = first().backend().kvSequence();
        if (seq == null) {
            return;
        }
        seq.pool().setReclaimer(() -> releaseIdle(0));
        if (idleSeconds > 0) {
            long idleMs = idleSeconds * 1000;
            sweeper = Thread.ofVirtual().name("kv-idle-release").start(() -> {
                try {
                    while (!closed) {
                        Thread.sleep(Math.max(100, Math.min(10_000, idleMs / 4)));
                        releaseIdle(idleMs);
                    }
                } catch (InterruptedException e) {
                    // closing
                }
            });
        }
    }

    /** Releases the blocks of the slots no request holds and whose engine has been idle for {@code minIdleMs}; true if any were freed. */
    public boolean releaseIdle(long minIdleMs) {
        int freed = 0;
        for (Slot s : slots) {
            if (s.users.get() == 0) {
                freed += s.engine.releaseKv(minIdleMs);
            }
        }
        return freed > 0;
    }

    /** Stops the housekeeping and lets queued requests leave; the engines stay open (their owner closes them). */
    public void stop() {
        closed = true;
        Thread sw = sweeper;
        if (sw != null) {
            sw.interrupt();
        }
    }

    /** True once {@link #stop} was called. */
    public boolean stopped() {
        return closed;
    }
}
