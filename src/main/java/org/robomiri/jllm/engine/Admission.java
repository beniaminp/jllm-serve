package org.robomiri.jllm.engine;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BooleanSupplier;

/**
 * The one queue of a model under {@code engine.scheduler=mixed} (docs/25): requests wait here for a slot instead of in the
 * lock queue of a slot chosen at arrival. When a slot and a place to run are free, the waiting request that gains the most
 * from the cache goes first, on the slot that holds the most of its prompt; priority classes, aging, per-session limits and
 * pausing a background generation for an interactive one are decided here too.
 *
 * <p>A slot is {@code IDLE}, {@code RUNNING} (takes part in passes) or {@code PAUSED} (it still holds its engine, but its blocks
 * were given back to the pool and it takes no part in passes). At most {@code maxRunning} slots run at once.
 *
 * <h3>Who goes next</h3>
 * Among the waiting requests whose session is under its limit, and the paused slots, by
 * <ol>
 *   <li>effective class: the request's class (0 interactive, 1 background) lowered by one per {@code agePromote} waited;</li>
 *   <li>a paused slot before a request of the same class (it has work done and its blocks may still be in memory);</li>
 *   <li>score = tokens of the prompt already in the cache + {@value #AGING_TOKENS_PER_SECOND} per second waited (so a long wait
 *       beats any cache hit: nothing starves);</li>
 *   <li>arrival.</li>
 * </ol>
 *
 * @param <S> the caller's slot type
 */
public final class Admission<S> {

    /** What the queue needs from the slots. */
    public interface Ops<S> {
        /** Leading tokens of {@code prompt} the cache holds for a run on {@code slot}. */
        int match(S slot, int[] prompt);

        /** Asks the generation on {@code slot} to pause; false when it cannot. */
        boolean pause(S slot);

        /** Lets a paused generation go on. */
        void resume(S slot);
    }

    /**
     * @param maxRunning  slots that may take part in passes at once; 0 or at least the slot count = all of them
     * @param preempt     pause a running background slot for an interactive request when no place to run is free
     * @param sessionMax  most slots one session may hold (running or paused); 0 = no limit
     * @param agePromote  nanoseconds of waiting that lift a request one class; 0 = no aging
     */
    public record Config(int maxRunning, boolean preempt, int sessionMax, long agePromote) {
    }

    /** Thrown to a waiting request whose client went away. */
    public static final class Abandoned extends RuntimeException {
        Abandoned() {
            super("the client went away while the request was queued", null, false, false);
        }
    }

    /** A request waiting for a slot gains this many tokens of cache-hit score per second it waits. */
    static final int AGING_TOKENS_PER_SECOND = 2000;

    private enum Phase { IDLE, RUNNING, PAUSED }

    private final class Entry {
        final S slot;
        Phase phase = Phase.IDLE;
        int rank;
        String session;
        long granted;
        long enqueued; // the request's arrival, for its class aging

        Entry(S slot) {
            this.slot = slot;
        }
    }

    private final class Waiter {
        final int[] prompt;
        final int rank;
        final String session;
        final long enqueued = System.nanoTime();
        Entry granted;

        Waiter(int[] prompt, int rank, String session) {
            this.prompt = prompt;
            this.rank = rank;
            this.session = session;
        }
    }

    private final Ops<S> ops;
    private final Config cfg;
    private final List<Entry> entries = new ArrayList<>();
    private final List<Waiter> waiting = new ArrayList<>();
    private long grants;
    private long preemptions;
    private long resumes;
    private long abandoned;
    private long serial;

    public Admission(List<S> slots, Ops<S> ops, Config cfg) {
        this.ops = ops;
        this.cfg = cfg;
        for (S s : slots) {
            entries.add(new Entry(s));
        }
    }

    private int limit() {
        return cfg.maxRunning() <= 0 ? entries.size() : Math.min(cfg.maxRunning(), entries.size());
    }

    /**
     * Waits for a slot to run {@code prompt} on. The slot is the caller's until {@link #release}.
     *
     * @param rank    0 = interactive, 1 = background
     * @param session the request's session (the OpenAI {@code user}), or null
     * @param gone    polled while waiting (about every 50 ms): true = the client left, the request is dropped from the queue
     * @throws Abandoned if {@code gone} became true while waiting
     */
    public S acquire(int[] prompt, int rank, String session, BooleanSupplier gone) {
        Waiter w = new Waiter(prompt, rank, session);
        synchronized (this) {
            waiting.add(w);
            dispatch();
        }
        while (true) {
            synchronized (this) {
                if (w.granted != null) {
                    return w.granted.slot;
                }
                try {
                    wait(50);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    waiting.remove(w);
                    throw new IllegalStateException("interrupted while queued", e);
                }
                if (w.granted != null) {
                    return w.granted.slot;
                }
            }
            if (gone.getAsBoolean()) {
                synchronized (this) {
                    if (w.granted != null) {
                        return w.granted.slot; // granted just now: the caller finds the client gone at its first check
                    }
                    waiting.remove(w);
                    abandoned++;
                    notifyAll();
                    throw new Abandoned();
                }
            }
            synchronized (this) {
                dispatch(); // aging may have changed the order
            }
        }
    }

    /** Gives the slot back (the request is finished, failed or cancelled). */
    public synchronized void release(S slot) {
        Entry e = entry(slot);
        e.phase = Phase.IDLE;
        e.session = null;
        dispatch();
    }

    private Entry entry(S slot) {
        for (Entry e : entries) {
            if (e.slot == slot) {
                return e;
            }
        }
        throw new IllegalArgumentException("not a slot of this queue");
    }

    private int running() {
        int n = 0;
        for (Entry e : entries) {
            n += e.phase == Phase.RUNNING ? 1 : 0;
        }
        return n;
    }

    private int holding(String session) {
        int n = 0;
        for (Entry e : entries) {
            n += e.phase != Phase.IDLE && session.equals(e.session) ? 1 : 0;
        }
        return n;
    }

    private int effective(int rank, long enqueued, long now) {
        if (cfg.agePromote() <= 0) {
            return rank;
        }
        return (int) Math.max(0, rank - Math.max(0, now - enqueued) / cfg.agePromote());
    }

    /** The idle slot with the longest cached prefix of {@code prompt} (the first when none has any); null if none is idle. */
    private Entry bestIdle(int[] prompt) {
        Entry best = null;
        int bestLen = -1;
        for (Entry e : entries) {
            if (e.phase != Phase.IDLE) {
                continue;
            }
            int len = ops.match(e.slot, prompt);
            if (len > bestLen) {
                best = e;
                bestLen = len;
            }
        }
        return best;
    }

    /** Hands out places while there are requests that can have them; pauses a background slot for an interactive request. */
    private void dispatch() {
        boolean changed = false;
        while (true) {
            long now = System.nanoTime();
            Waiter bestW = null;
            double bestScore = 0;
            int bestEff = 0;
            for (Waiter w : waiting) {
                if (w.granted != null || (cfg.sessionMax() > 0 && w.session != null && holding(w.session) >= cfg.sessionMax())) {
                    continue;
                }
                Entry idle = bestIdle(w.prompt);
                int hit = idle == null ? 0 : ops.match(idle.slot, w.prompt);
                double score = hit + AGING_TOKENS_PER_SECOND * Math.max(0, now - w.enqueued) / 1e9;
                int eff = effective(w.rank, w.enqueued, now);
                if (bestW == null || eff < bestEff || (eff == bestEff && score > bestScore)
                        || (eff == bestEff && score == bestScore && w.enqueued < bestW.enqueued)) {
                    bestW = w;
                    bestScore = score;
                    bestEff = eff;
                }
            }
            Entry pausedBest = null;
            int pausedEff = 0;
            for (Entry e : entries) {
                if (e.phase == Phase.PAUSED) {
                    int eff = effective(e.rank, e.enqueued, now);
                    if (pausedBest == null || eff < pausedEff || (eff == pausedEff && e.granted < pausedBest.granted)) {
                        pausedBest = e;
                        pausedEff = eff;
                    }
                }
            }
            boolean room = running() < limit();
            if (pausedBest != null && room && (bestW == null || pausedEff <= bestEff)) {
                pausedBest.phase = Phase.RUNNING;
                resumes++;
                ops.resume(pausedBest.slot);
                changed = true;
                continue;
            }
            if (bestW == null) {
                break;
            }
            Entry idle = bestIdle(bestW.prompt);
            if (idle == null) {
                break; // every slot holds a request: wait for a release
            }
            if (!room) {
                if (!(cfg.preempt() && bestEff == 0 && pauseBackground())) {
                    break;
                }
                changed = true;
                continue; // a place is free now
            }
            idle.phase = Phase.RUNNING;
            idle.rank = bestW.rank;
            idle.session = bestW.session;
            idle.granted = ++serial;
            idle.enqueued = bestW.enqueued;
            bestW.granted = idle;
            waiting.remove(bestW);
            grants++;
            changed = true;
        }
        if (changed) {
            notifyAll();
        }
    }

    /** Pauses the background slot that started last; false if there is none or it cannot pause. */
    private boolean pauseBackground() {
        Entry victim = null;
        for (Entry e : entries) {
            if (e.phase == Phase.RUNNING && e.rank > 0 && (victim == null || e.granted > victim.granted)) {
                victim = e;
            }
        }
        if (victim == null || !ops.pause(victim.slot)) {
            return false;
        }
        victim.phase = Phase.PAUSED;
        preemptions++;
        return true;
    }

    /** Counters: {requests granted a slot, background slots paused, paused slots resumed, requests dropped while queued}. */
    public synchronized long[] stats() {
        return new long[]{grants, preemptions, resumes, abandoned};
    }

    public synchronized int queued() {
        return waiting.size();
    }

    public synchronized int pausedSlots() {
        int n = 0;
        for (Entry e : entries) {
            n += e.phase == Phase.PAUSED ? 1 : 0;
        }
        return n;
    }

    public synchronized int runningSlots() {
        return running();
    }
}
