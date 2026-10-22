package org.robomiri.jllm.engine;

import org.robomiri.jllm.model.Backend;

import java.io.IOException;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Batched decoding across the slots of one model (a server's engines on the same weights): when several
 * conversations generate at the same time, their next tokens are computed in one forward pass
 * ({@link Backend#forwardBatch}), so the weights are read once for all of them instead of once per conversation.
 * Decoding is limited by reading the weights, so this is close to n times the tokens per second for n
 * conversations.
 *
 * <p>Each slot's engine gets a {@link Member}, a {@link Backend} that forwards everything to the slot's own
 * backend. A single-token {@code forward} (a decode step) joins a queue. Whichever thread takes the device lock
 * first becomes the leader: it waits up to {@link #WINDOW_NANOS} for the other slots that are generating right now,
 * runs every queued step as one batch and hands each its logits; the others just wait for their result. Every other
 * use of a backend (prefill, verification passes, saved caches) also takes the device lock, so the slots never run
 * on the shared hardware at once. A slot alone in the group decodes exactly as before, with no waiting.
 */
public final class BatchGroup {

    /** How long a leader waits for the other generating slots' next steps (they arrive within a sampling step). */
    static final long WINDOW_NANOS = Long.getLong("jllm.engine.batchWindowMicros", 2000) * 1000;
    /** A slot that decoded within this much time counts as generating. */
    private static final long ACTIVE_NANOS = 200_000_000L;

    /** The longest a background request lets interactive ones go first when it waits for the device ({@link Priority}). */
    static final long BACKGROUND_MAX_WAIT_MS = Long.getLong("jllm.engine.backgroundMaxWaitMs", 2000);

    private final ReentrantLock device = new ReentrantLock();
    /** Honour {@link Priority} in the order of the device waits ({@code engine.priority-order}). */
    private final boolean priorityOrder;
    /** Interactive requests waiting for, or in, a decode step: background ones let them have the device first. */
    private final java.util.concurrent.atomic.AtomicInteger interactiveWaiting = new java.util.concurrent.atomic.AtomicInteger();
    /** The CPU threads the members' CPU backends share (they never compute at the same time); closed with the last. */
    private org.robomiri.jllm.cpu.ThreadPool cpuPool;
    private final List<Member> members = new ArrayList<>();
    private final List<Step> pending = new ArrayList<>();
    private long batches;
    private long batchedSteps;

    private static final class Step {
        final Member member;
        final int token;
        final int pos;
        final float[] logits;
        final Priority priority = Priority.current();
        boolean done;
        RuntimeException error;

        Step(Member member, int token, int pos, float[] logits) {
            this.member = member;
            this.token = token;
            this.pos = pos;
            this.logits = logits;
        }
    }

    public BatchGroup() {
        this(true);
    }

    /** @param priorityOrder whether {@link Priority#BACKGROUND} requests give way to interactive ones at the device */
    public BatchGroup(boolean priorityOrder) {
        this.priorityOrder = priorityOrder;
    }

    /** Takes the device lock; a background request first lets the interactive ones that wait for the device go ahead. */
    private void lockDevice() {
        if (!priorityOrder || device.isHeldByCurrentThread()) {
            device.lock();
        } else if (Priority.current() == Priority.INTERACTIVE) {
            interactiveWaiting.incrementAndGet();
            try {
                device.lock();
            } finally {
                interactiveWaiting.decrementAndGet();
            }
        } else {
            // polls instead of queueing on the lock: a background request already in the lock's queue could not be
            // overtaken by an interactive one that arrives later
            long give = System.nanoTime() + BACKGROUND_MAX_WAIT_MS * 1_000_000;
            while (System.nanoTime() < give) {
                if (interactiveWaiting.get() == 0 && device.tryLock()) {
                    return;
                }
                java.util.concurrent.locks.LockSupport.parkNanos(200_000);
            }
            device.lock();
        }
    }

    /** {@code inner} as a member of this group. */
    public synchronized Member join(Backend inner) {
        Member m = new Member(inner);
        members.add(m);
        return m;
    }

    /** The CPU thread pool for a member's CPU backend: created by the first, shared by the others. */
    public synchronized org.robomiri.jllm.cpu.ThreadPool cpuPool(int threads) {
        if (cpuPool == null) {
            cpuPool = new org.robomiri.jllm.cpu.ThreadPool(threads);
        }
        return cpuPool;
    }

    /** Decode steps run, and how many of them shared a pass with another (for the logs and tests). */
    public synchronized long[] stats() {
        return new long[]{batches, batchedSteps};
    }

    private synchronized int othersGenerating(Member self, long now) {
        int n = 0;
        for (Member m : members) {
            if (m != self && now - m.lastDecode < ACTIVE_NANOS) {
                n++;
            }
        }
        return n;
    }

    private void decode(Member self, int token, int pos, float[] logits) {
        if (priorityOrder && Priority.current() == Priority.INTERACTIVE) {
            interactiveWaiting.incrementAndGet();
            try {
                decodeStep(self, token, pos, logits);
            } finally {
                interactiveWaiting.decrementAndGet();
            }
        } else {
            decodeStep(self, token, pos, logits);
        }
    }

    private void decodeStep(Member self, int token, int pos, float[] logits) {
        Step mine = new Step(self, token, pos, logits);
        long now = System.nanoTime();
        long give = now + BACKGROUND_MAX_WAIT_MS * 1_000_000;
        self.lastDecode = now;
        int others = othersGenerating(self, now);
        synchronized (this) {
            pending.add(mine);
            notifyAll();
        }
        while (true) {
            // a background step does not lead a batch while an interactive one waits (it may still ride in that batch)
            boolean yield = priorityOrder && mine.priority == Priority.BACKGROUND && interactiveWaiting.get() > 0
                    && System.nanoTime() < give;
            if (!yield && device.tryLock()) {
                try {
                    synchronized (this) {
                        if (mine.done) {
                            break;
                        }
                        long deadline = System.nanoTime() + WINDOW_NANOS;
                        // wait for the others that are generating, up to the window (never for a slot alone)
                        while (pending.size() < Math.min(others + 1, self.inner.maxBatch())) {
                            long left = deadline - System.nanoTime();
                            if (left <= 0) {
                                break;
                            }
                            try {
                                wait(left / 1_000_000, (int) (left % 1_000_000));
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                                break;
                            }
                        }
                    }
                    run(self);
                } finally {
                    device.unlock();
                    synchronized (this) {
                        notifyAll(); // the device is free: a waiting step may lead the next batch
                    }
                }
            } else {
                synchronized (this) {
                    if (!mine.done) {
                        try {
                            wait(1);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new IllegalStateException("interrupted while waiting to decode", e);
                        }
                    }
                }
            }
            synchronized (this) {
                if (mine.done) {
                    break;
                }
            }
        }
        if (mine.error != null) {
            throw mine.error;
        }
    }

    /** Runs the queued steps (the leader holds the device): one batched pass, or one by one where that cannot be. */
    private void run(Member leader) {
        List<Step> batch;
        synchronized (this) {
            int max = leader.inner.maxBatch();
            // the leader's own step first: its backend lends the buffers
            // (then the interactive steps before the background ones, if more are queued than one pass takes)
            pending.sort((a, b) -> {
                int c = Boolean.compare(b.member == leader, a.member == leader);
                return c != 0 || !priorityOrder ? c : a.priority.compareTo(b.priority);
            });
            batch = new ArrayList<>(pending.subList(0, Math.min(max, pending.size())));
            pending.removeAll(batch);
        }
        boolean batched = false;
        RuntimeException error = null;
        try {
            if (batch.size() > 1 && batch.get(0).member == leader) {
                int n = batch.size();
                Backend[] backends = new Backend[n];
                int[] tokens = new int[n];
                int[] positions = new int[n];
                float[][] logits = new float[n][];
                for (int i = 0; i < n; i++) {
                    Step s = batch.get(i);
                    backends[i] = s.member.inner;
                    tokens[i] = s.token;
                    positions[i] = s.pos;
                    logits[i] = s.logits;
                }
                batched = leader.inner.forwardBatch(backends, tokens, positions, logits);
            }
            if (!batched) {
                for (Step s : batch) {
                    s.member.inner.forward(new int[]{s.token}, 0, 1, s.pos, s.logits);
                }
            }
        } catch (RuntimeException e) {
            error = e;
        }
        synchronized (this) {
            batches++;
            if (batched) {
                batchedSteps += batch.size();
            }
            for (Step s : batch) {
                s.error = error;
                s.done = true;
            }
            notifyAll();
        }
    }

    /** One slot's backend in the group (see {@link BatchGroup}). */
    public final class Member implements Backend {
        final Backend inner;
        volatile long lastDecode = Long.MIN_VALUE / 2;

        Member(Backend inner) {
            this.inner = inner;
        }

        /** The slot's own backend. */
        public Backend inner() {
            return inner;
        }

        /** The group this slot decodes with. */
        public BatchGroup group() {
            return BatchGroup.this;
        }

        private void locked(Runnable r) {
            lockDevice();
            try {
                r.run();
            } finally {
                device.unlock();
                synchronized (BatchGroup.this) {
                    BatchGroup.this.notifyAll();
                }
            }
        }

        @Override
        public String name() {
            return inner.name();
        }

        @Override
        public int contextLength() {
            return inner.contextLength();
        }

        @Override
        public int maxBatch() {
            return inner.maxBatch();
        }

        @Override
        public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
            if (n == 1 && logits != null) {
                decode(this, tokens[off], pos, logits);
                return;
            }
            locked(() -> inner.forward(tokens, off, n, pos, logits));
        }

        @Override
        public void forwardAll(int[] tokens, int off, int n, int pos, float[] logitsOut, int stride) {
            locked(() -> inner.forwardAll(tokens, off, n, pos, logitsOut, stride));
        }

        @Override
        public boolean supportsEmbeds() {
            return inner.supportsEmbeds();
        }

        @Override
        public void forwardEmbeds(float[] embeds, int off, int n, int pos, float[][] deepstack, float[] logits) {
            locked(() -> inner.forwardEmbeds(embeds, off, n, pos, deepstack, logits));
        }

        @Override
        public boolean supportsBidirectionalSpan() {
            return inner.supportsBidirectionalSpan();
        }

        @Override
        public void forwardEmbedsBidirectional(float[] embeds, int off, int n, int pos, float[] logits) {
            locked(() -> inner.forwardEmbedsBidirectional(embeds, off, n, pos, logits));
        }

        @Override
        public boolean setRopePositions(int from, org.robomiri.jllm.model.RopePositions p) {
            boolean[] ok = new boolean[1];
            locked(() -> ok[0] = inner.setRopePositions(from, p));
            return ok[0];
        }

        @Override
        public boolean ropePositionsCustom() {
            return inner.ropePositionsCustom();
        }

        @Override
        public boolean supportsPrefixReuse() {
            return inner.supportsPrefixReuse();
        }

        // multi-token prediction (a Qwen3.5 GGUF): the head runs on the device like any pass, so under its lock
        @Override
        public boolean hasMtp() {
            return inner.hasMtp();
        }

        @Override
        public void enableMtp(boolean on) {
            locked(() -> inner.enableMtp(on));
        }

        @Override
        public boolean mtpReady() {
            return inner.mtpReady();
        }

        @Override
        public void mtpDraft(int token, float[] logits) {
            locked(() -> inner.mtpDraft(token, logits));
        }

        @Override
        public boolean supportsRollback() {
            return inner.supportsRollback();
        }

        @Override
        public void checkpointState() {
            locked(inner::checkpointState);
        }

        @Override
        public void rollbackState(int keep) {
            locked(() -> inner.rollbackState(keep));
        }

        @Override
        public int thermalState() {
            return inner.thermalState();
        }

        @Override
        public String stateFormat() {
            return inner.stateFormat();
        }

        @Override
        public void saveState(int n, WritableByteChannel out) throws IOException {
            lockDevice();
            try {
                inner.saveState(n, out);
            } finally {
                device.unlock();
            }
        }

        @Override
        public void loadState(int n, ReadableByteChannel in) throws IOException {
            lockDevice();
            try {
                inner.loadState(n, in);
            } finally {
                device.unlock();
            }
        }

        @Override
        public void copyStateFrom(Backend other, int n) {
            Backend o = other instanceof Member m ? m.inner : other;
            locked(() -> inner.copyStateFrom(o, n));
        }

        @Override
        public GreedyPipe greedyPipe(int vocab) {
            synchronized (BatchGroup.this) {
                // the GPU running ahead on its own would bypass the device lock: only while the slot is alone
                return members.size() == 1 ? inner.greedyPipe(vocab) : null;
            }
        }

        @Override
        public boolean forwardBatch(Backend[] backends, int[] tokens, int[] positions, float[][] logits) {
            return false; // batches are formed by the group
        }

        @Override
        public void close() {
            synchronized (BatchGroup.this) {
                members.remove(this);
            }
            locked(inner::close);
            synchronized (BatchGroup.this) {
                if (members.isEmpty() && cpuPool != null) {
                    cpuPool.close();
                    cpuPool = null;
                }
            }
        }
    }
}
