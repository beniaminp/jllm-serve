package org.robomiri.jllm.cpu;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

/**
 * Low-latency fork/join for the forward pass. A decode step issues a few hundred small parallel loops, so
 * workers spin briefly between jobs instead of parking, and the calling thread takes part in every job.
 */
public final class ThreadPool implements AutoCloseable {

    @FunctionalInterface
    public interface RangeBody {
        void run(int start, int end);
    }

    private static final long SPIN_NANOS = 200_000;

    private static final class Job {
        final RangeBody body;
        final int n;
        final int chunk;
        final int tasks;
        final AtomicInteger next = new AtomicInteger();
        final AtomicInteger done = new AtomicInteger();
        volatile Throwable error;

        Job(RangeBody body, int n, int tasks) {
            this.body = body;
            this.n = n;
            this.tasks = tasks;
            this.chunk = (n + tasks - 1) / tasks;
        }

        void work() {
            int t;
            while ((t = next.getAndIncrement()) < tasks) {
                try {
                    int s = t * chunk;
                    body.run(s, Math.min(n, s + chunk));
                } catch (Throwable e) {
                    error = e;
                } finally {
                    done.incrementAndGet();
                }
            }
        }
    }

    private final class Worker extends Thread {
        volatile boolean parked;

        Worker(int i) {
            super("jllm-cpu-" + i);
            setDaemon(true);
        }

        @Override
        public void run() {
            Job last = null;
            while (!closed) {
                Job j = current;
                if (j != null && j != last) {
                    last = j;
                    j.work();
                    continue;
                }
                long deadline = System.nanoTime() + SPIN_NANOS;
                while (current == last && !closed && System.nanoTime() < deadline) {
                    Thread.onSpinWait();
                }
                if (current == last && !closed) {
                    parked = true;
                    if (current == last && !closed) { // re-check after publishing 'parked' (pairs with submit)
                        LockSupport.park(this);
                    }
                    parked = false;
                }
            }
        }
    }

    private final Worker[] workers;
    private final int threads;
    private volatile Job current;
    private volatile boolean closed;

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
        Job j = new Job(body, n, tasks);
        current = j;
        for (Worker w : workers) {
            if (w.parked) {
                LockSupport.unpark(w);
            }
        }
        j.work();
        while (j.done.get() < j.tasks) {
            Thread.onSpinWait();
        }
        if (j.error != null) {
            if (j.error instanceof RuntimeException re) {
                throw re;
            }
            if (j.error instanceof Error err) {
                throw err;
            }
            throw new RuntimeException(j.error);
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
