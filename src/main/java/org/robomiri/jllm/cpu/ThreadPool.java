package org.robomiri.jllm.cpu;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;

/**
 * Low-latency fork/join for the forward pass. A decode step issues a few hundred small parallel loops, so
 * workers spin briefly between jobs instead of parking, and the calling thread takes part in every job.
 *
 * <p>Jobs are not objects: the pool holds one set of job fields (body, size, chunk) that the caller writes
 * before publishing the job, and one atomic {@link #next} packing the job's epoch (high 32 bits), its task
 * count (16 bits) and the next task index (low 16 bits). A claim ({@code getAndIncrement}) is valid when its
 * index is below its own task count, which the thread sees in the same atomic value; only then does it read the
 * job fields, and then that job is provably still running (its claimed task is not done), so the fields are the
 * job's own. A thread that lagged behind the end of one job and claims a task of the next one therefore just
 * reloads the fields and runs it. Nothing is allocated per call.
 *
 * <p>The first exception a task throws is kept and rethrown on the caller once every task has been claimed;
 * tasks claimed after it are skipped (they are counted as done without running).
 */
public final class ThreadPool implements AutoCloseable {

    @FunctionalInterface
    public interface RangeBody {
        void run(int start, int end);
    }

    private static final long SPIN_NANOS = 200_000;

    private final class Worker extends Thread {
        volatile boolean parked;

        Worker(int i) {
            super("jllm-cpu-" + i);
            setDaemon(true);
        }

        @Override
        public void run() {
            long last = 0;
            while (!closed) {
                long e = next.get() >>> 32;
                if (e != last) {
                    last = work(e);
                    continue;
                }
                long deadline = System.nanoTime() + SPIN_NANOS;
                while (next.get() >>> 32 == last && !closed && System.nanoTime() < deadline) {
                    Thread.onSpinWait();
                }
                if (next.get() >>> 32 == last && !closed) {
                    parked = true;
                    if (next.get() >>> 32 == last && !closed) { // re-check after publishing 'parked' (pairs with run)
                        LockSupport.park(this);
                    }
                    parked = false;
                }
            }
        }
    }

    private final Worker[] workers;
    private final int threads;
    private volatile boolean closed;

    // the current job; written by the caller before it publishes the epoch in `next`, read by the workers after
    // they observe that epoch (the atomic orders the two)
    private RangeBody body;
    private int n;
    private int chunk;
    private int tasks;
    private long epoch;
    /** {@code epoch << 32 | tasks << 16 | next task index}. */
    private final AtomicLong next = new AtomicLong();
    /** Tasks per job are capped so that the index can never carry into the task-count bits (at most one over-claim per thread). */
    private static final int MAX_TASKS = 0xFFFF - 256;
    private final AtomicInteger done = new AtomicInteger();
    private final AtomicReference<Throwable> error = new AtomicReference<>();

    public ThreadPool(int threads) {
        this.threads = Math.max(1, threads);
        this.workers = new Worker[this.threads - 1];
        for (int i = 0; i < workers.length; i++) {
            workers[i] = new Worker(i);
            workers[i].start();
        }
    }

    public int threads() {
        return threads;
    }

    /**
     * Runs {@code body} over {@code [0, n)} split into contiguous chunks of at least {@code grain} items.
     */
    public void forRange(int n, int grain, RangeBody body) {
        run(n, Math.min(threads * 4, Math.max(1, n / Math.max(1, grain))), body);
    }

    /**
     * Runs {@code body} over {@code [0, n)} in chunks of exactly {@code chunk} items with no cap on the number of
     * tasks. For loops whose items have very uneven cost (causal attention: later tokens see more keys) the fine
     * grain keeps the tail short, since threads keep pulling small chunks until the last one is handed out.
     */
    public void forRangeFine(int n, int chunk, RangeBody body) {
        run(n, Math.max(1, (n + chunk - 1) / Math.max(1, chunk)), body);
    }

    private void run(int n, int tasks, RangeBody body) {
        if (tasks <= 1 || threads == 1) {
            body.run(0, n);
            return;
        }
        tasks = Math.min(tasks, MAX_TASKS);
        this.body = body;
        this.n = n;
        this.tasks = tasks;
        this.chunk = (n + tasks - 1) / tasks;
        error.set(null);
        done.set(0);
        long e = ++epoch;
        next.set(e << 32 | (long) tasks << 16);
        for (Worker w : workers) {
            if (w.parked) {
                LockSupport.unpark(w);
            }
        }
        work(e);
        while (done.get() < tasks) {
            Thread.onSpinWait();
        }
        Throwable t = error.get();
        if (t != null) {
            if (t instanceof RuntimeException re) {
                throw re;
            }
            if (t instanceof Error err) {
                throw err;
            }
            throw new RuntimeException(t);
        }
    }

    /**
     * Claims and runs tasks until a claim comes back past the end of its job; moves on to a newer job if a claim
     * belongs to one (see the class comment). Returns the epoch of the last job it worked on.
     */
    private long work(long e) {
        RangeBody body = null;
        int n = 0;
        int chunk = 0;
        boolean loaded = false;
        for (;;) {
            long v = next.getAndIncrement();
            int t = (int) (v & 0xFFFF);
            if (t >= (int) ((v >>> 16) & 0xFFFF)) {
                return e;
            }
            long ve = v >>> 32;
            if (ve != e || !loaded) {
                // a valid claim of job ve: it is still running, so the fields are its own
                e = ve;
                body = this.body;
                n = this.n;
                chunk = this.chunk;
                loaded = true;
            }
            try {
                if (error.get() == null) {
                    int s = t * chunk;
                    body.run(s, Math.min(n, s + chunk));
                }
            } catch (Throwable x) {
                error.compareAndSet(null, x);
            } finally {
                done.incrementAndGet();
            }
        }
    }

    @Override
    public void close() {
        closed = true;
        for (Worker w : workers) {
            LockSupport.unpark(w);
        }
    }
}
