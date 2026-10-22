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
 *
 * <p><b>Mixed passes</b> ({@link #enableMixed}, {@code engine.scheduler=mixed}, docs/25): prefill chunks and verification
 * runs join the queue too, and the leader forms each pass with {@link PassPlanner}: one prefill chunk plus the next token
 * of every generating conversation, in a single {@link Backend#forwardMixed} pass. A slot can also be
 * paused between two passes ({@link Member#requestPause}): its blocks go back to the pool (which may spill them) and it
 * waits until {@link Member#resume}.
 */
public final class BatchGroup {

    /** How long a leader waits for the other generating slots' next steps (they arrive within a sampling step). */
    static final long WINDOW_NANOS = Long.getLong("jllm.engine.batchWindowMicros", 2000) * 1000;
    /** A slot that decoded within this much time counts as generating. */
    private static final long ACTIVE_NANOS = 200_000_000L;

    private final ReentrantLock device = new ReentrantLock();
    /** The CPU threads the members' CPU backends share (they never compute at the same time); closed with the last. */
    private org.robomiri.jllm.cpu.ThreadPool cpuPool;
    private final List<Member> members = new ArrayList<>();
    private final List<Step> pending = new ArrayList<>();
    private long batches;
    private long batchedSteps;

    /**
     * One forward call waiting for a pass: {@code n} tokens at {@code pos}, logits wanted as {@link Backend.Segment} says. In
     * slots mode only single decode tokens; in mixed mode also prefill chunks (run in parts when trimmed to the pass
     * budget: {@code ran} rows are done) and verification runs.
     */
    private static final class Step {
        final Member member;
        final int[] tokens;
        final int off;
        final int n;
        final int pos;
        final float[] logits;
        final int stride;
        final long enqueued = System.nanoTime();
        final int rank;
        int ran;
        boolean done;
        RuntimeException error;

        Step(Member member, int[] tokens, int off, int n, int pos, float[] logits, int stride) {
            this.member = member;
            this.tokens = tokens;
            this.off = off;
            this.n = n;
            this.pos = pos;
            this.logits = logits;
            this.stride = stride;
            this.rank = Priority.current().ordinal(); // the request thread's class (docs/10 section 10.14)
        }

        PassPlanner.Kind kind() {
            return stride > 0 ? PassPlanner.Kind.VERIFY : n == 1 && logits != null ? PassPlanner.Kind.DECODE : PassPlanner.Kind.CHUNK;
        }
    }

    /**
     * The settings of mixed passes.
     *
     * @param budget    most rows in a pass (0 = everything the backend's buffers hold)
     * @param reserve   rows the backend's buffers hold beyond {@code backend.batch}, for the riders of a full chunk
     * @param minChunk  a chunk is trimmed to the leftover budget only down to this many rows
     * @param agePromote nanoseconds of waiting that lift an item one priority class (0 = no aging)
     * @param backgroundCap rows of a background chunk while an interactive stream rides the pass (0 = no cap)
     */
    public record MixedConfig(int budget, int reserve, int minChunk, long agePromote, int backgroundCap) {
    }

    private volatile MixedConfig mixed; // null = slots mode
    private long mixedPasses;           // passes that carried a prefill chunk and at least one other segment
    private long riderRows;             // rows of other segments those passes carried
    private long trimmedChunks;         // prefill chunks run in parts because the budget left no room for all of it

    /** {@code inner} as a member of this group. */
    public synchronized Member join(Backend inner) {
        return join(inner, 0);
    }

    /**
     * {@code inner} as a member whose backend holds {@code reserve} rows more than the engine may use for one chunk
     * ({@link Member#maxBatch} hides them): room for the riders of a mixed pass.
     */
    public synchronized Member join(Backend inner, int reserve) {
        Member m = new Member(inner, reserve);
        members.add(m);
        return m;
    }

    /** Turns on mixed passes for this group's members (call before they run). */
    public synchronized void enableMixed(MixedConfig cfg) {
        this.mixed = cfg;
    }

    /** Whether the group forms mixed passes. */
    public boolean mixed() {
        return mixed != null;
    }

    /** Mixed-pass counters: {passes with a chunk and riders, rows of riders, chunks run in parts}. */
    public synchronized long[] mixedStats() {
        return new long[]{mixedPasses, riderRows, trimmedChunks};
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
        submit(new Step(self, new int[]{token}, 0, 1, pos, logits, 0));
    }

    /**
     * Queues {@code mine} and returns when a pass has run it. Whichever thread takes the device lock first leads the pass.
     */
    private void submit(Step mine) {
        Member self = mine.member;
        long now = System.nanoTime();
        self.lastDecode = now;
        int others = othersGenerating(self, now);
        synchronized (this) {
            pending.add(mine);
            notifyAll();
        }
        while (true) {
            if (device.tryLock()) {
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
                    if (mixed != null) {
                        runMixed(self);
                    } else {
                        run(self);
                    }
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
            pending.sort((a, b) -> Boolean.compare(b.member == leader, a.member == leader));
            batch = new ArrayList<>(pending.subList(0, Math.min(max, pending.size())));
            pending.removeAll(batch);
        }
        boolean batched = false;
        RuntimeException error = null;
        try {
            if (batch.size() > 1 && batch.get(0).member == leader) {
                batched = forwardBatchOf(leader.inner, batch);
            }
            if (!batched) {
                for (Step s : batch) {
                    s.member.inner.forward(s.tokens, 0, 1, s.pos, s.logits);
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

    /** {@code lender.forwardBatch} over single decode steps. */
    private static boolean forwardBatchOf(Backend lender, List<Step> steps) {
        int n = steps.size();
        Backend[] backends = new Backend[n];
        int[] tokens = new int[n];
        int[] positions = new int[n];
        float[][] logits = new float[n][];
        for (int i = 0; i < n; i++) {
            Step s = steps.get(i);
            backends[i] = s.member.inner;
            tokens[i] = s.tokens[0];
            positions[i] = s.pos;
            logits[i] = s.logits;
        }
        return lender.forwardBatch(backends, tokens, positions, logits);
    }

    /**
     * The mixed pass (the leader holds the device): {@link PassPlanner} picks one prefill chunk (trimmed to the budget if
     * need be) and the riders (decode tokens, verification runs), and the leader's backend runs them as one
     * {@link Backend#forwardMixed}. Where the backend cannot (Mamba, latent attention, a flat cache), the same picks run as
     * a batched decode pass plus the other segments one by one. A failure belongs to the step that caused it: the rest of
     * the pass is run again step by step.
     */
    private void runMixed(Member leader) {
        MixedConfig cfg = mixed;
        List<PassPlanner.Pick> picks;
        List<Step> steps = new ArrayList<>();
        List<Integer> rows = new ArrayList<>();
        synchronized (this) {
            if (pending.isEmpty()) {
                return;
            }
            int capacity = leader.inner.maxBatch();
            int budget = cfg.budget() > 0 ? Math.min(cfg.budget(), capacity) : capacity;
            List<PassPlanner.Item> items = new ArrayList<>(pending.size());
            for (Step s : pending) {
                items.add(new PassPlanner.Item(s, s.kind(), s.n - s.ran, s.rank, s.enqueued));
            }
            picks = PassPlanner.plan(items, System.nanoTime(),
                    new PassPlanner.Config(budget, cfg.minChunk(), cfg.agePromote(), cfg.backgroundCap()));
            for (PassPlanner.Pick p : picks) {
                Step s = (Step) p.ref();
                steps.add(s);
                rows.add(p.rows());
                pending.remove(s);
            }
        }
        int m = steps.size();
        if (m == 0) {
            return;
        }
        Backend.Segment[] segs = new Backend.Segment[m];
        for (int i = 0; i < m; i++) {
            Step s = steps.get(i);
            int r = rows.get(i);
            boolean last = s.ran + r == s.n;
            segs[i] = new Backend.Segment(s.member.inner, s.tokens, s.off + s.ran, r, s.pos + s.ran,
                    last || s.stride > 0 ? s.logits : null, s.stride);
        }
        RuntimeException[] errors = new RuntimeException[m];
        boolean ran;
        try {
            ran = leader.inner.forwardMixed(segs);
        } catch (RuntimeException e) {
            ran = false; // the segments are run one by one below, so that only the step that fails reports it
        }
        if (!ran) {
            runApart(steps, rows, segs, errors);
        }
        synchronized (this) {
            batches++;
            boolean hasChunk = steps.get(0).kind() == PassPlanner.Kind.CHUNK;
            if (hasChunk && m > 1) {
                mixedPasses++;
                for (int i = 1; i < m; i++) {
                    riderRows += rows.get(i);
                }
            }
            if (ran && m > 1) {
                for (Step s : steps) {
                    batchedSteps += s.kind() == PassPlanner.Kind.DECODE ? 1 : 0;
                }
            }
            for (int i = 0; i < m; i++) {
                Step s = steps.get(i);
                s.error = errors[i];
                boolean first = s.ran == 0;
                s.ran += rows.get(i);
                if (s.ran < s.n && errors[i] == null) {
                    trimmedChunks += first ? 1 : 0;
                    pending.add(s); // the rest of a trimmed chunk waits for the next pass, keeping its place in line
                } else {
                    s.done = true;
                }
            }
            notifyAll();
        }
    }

    /** The picks of a mixed pass the backend would not take as one pass: decode steps batched if possible, the rest alone. */
    private static void runApart(List<Step> steps, List<Integer> rows, Backend.Segment[] segs, RuntimeException[] errors) {
        List<Integer> decodes = new ArrayList<>();
        for (int i = 0; i < steps.size(); i++) {
            Step s = steps.get(i);
            if (s.kind() == PassPlanner.Kind.DECODE && rows.get(i) == 1) {
                decodes.add(i);
            }
        }
        boolean[] handled = new boolean[steps.size()];
        if (decodes.size() > 1) {
            List<Step> ds = new ArrayList<>();
            for (int i : decodes) {
                ds.add(steps.get(i));
            }
            try {
                if (forwardBatchOf(ds.get(0).member.inner, ds)) {
                    for (int i : decodes) {
                        handled[i] = true;
                    }
                }
            } catch (RuntimeException e) {
                // below, one by one
            }
        }
        for (int i = 0; i < steps.size(); i++) {
            if (handled[i]) {
                continue;
            }
            Backend.Segment sg = segs[i];
            try {
                if (sg.stride() > 0) {
                    sg.owner().forwardAll(sg.tokens(), sg.off(), sg.n(), sg.pos(), sg.logits(), sg.stride());
                } else {
                    sg.owner().forward(sg.tokens(), sg.off(), sg.n(), sg.pos(), sg.logits());
                }
            } catch (RuntimeException e) {
                errors[i] = e;
            }
        }
    }

    /** One slot's backend in the group (see {@link BatchGroup}). */
    public final class Member implements Backend {
        final Backend inner;
        private final int reserve;
        volatile long lastDecode = Long.MIN_VALUE / 2;
        private volatile boolean pauseRequested;
        private boolean paused;

        Member(Backend inner, int reserve) {
            this.inner = inner;
            this.reserve = reserve;
        }

        /**
         * Asks the generation on this slot to pause at its next {@link #checkpoint}: its blocks go back to the pool (which
         * may spill them) until {@link #resume}. Returns false when this slot cannot (no paged cache).
         */
        public boolean requestPause() {
            org.robomiri.jllm.kv.KvSequence ks = inner.kvSequence();
            if (mixed == null || ks == null || !ks.suspendable()) {
                return false;
            }
            pauseRequested = true;
            return true;
        }

        /**
         * Waits until the generation has paused (it does at its next checkpoint, after the step it is in) or {@code millis}
         * have passed; true if it is paused. A scheduler that gives the freed place to another request waits for this, so
         * the paused blocks are back in the pool before the newcomer asks for its own.
         */
        public boolean awaitPaused(long millis) {
            long end = System.nanoTime() + millis * 1_000_000L;
            synchronized (BatchGroup.this) {
                while (!paused && pauseRequested) {
                    long left = end - System.nanoTime();
                    if (left <= 0) {
                        break;
                    }
                    try {
                        BatchGroup.this.wait(Math.max(1, left / 1_000_000));
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
                return paused;
            }
        }

        /** Lets a paused (or about to pause) generation go on. */
        public void resume() {
            synchronized (BatchGroup.this) {
                pauseRequested = false;
                BatchGroup.this.notifyAll();
            }
        }

        /** Whether the generation on this slot is waiting, paused, right now. */
        public boolean paused() {
            synchronized (BatchGroup.this) {
                return paused;
            }
        }

        public boolean pauseRequested() {
            return pauseRequested;
        }

        @Override
        public int checkpoint(java.util.function.Supplier<int[]> tokens, int n, java.util.function.BooleanSupplier cancelled) {
            if (!pauseRequested) {
                return n;
            }
            org.robomiri.jllm.kv.KvSequence ks = inner.kvSequence();
            if (ks == null || !ks.suspendable() || n <= 0) {
                return n;
            }
            int[] toks = tokens.get();
            org.robomiri.jllm.kv.KvSequence.Suspended[] held = new org.robomiri.jllm.kv.KvSequence.Suspended[1];
            locked(() -> held[0] = ks.suspend(toks, n));
            synchronized (BatchGroup.this) {
                paused = true;
                BatchGroup.this.notifyAll();
                try {
                    while (pauseRequested && !cancelled.getAsBoolean()) {
                        BatchGroup.this.wait(50);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    paused = false;
                    pauseRequested = false;
                }
            }
            int[] valid = new int[1];
            locked(() -> valid[0] = ks.resume(held[0], toks));
            return valid[0];
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
            device.lock();
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
            return inner.maxBatch() - reserve; // the rows beyond are the riders' of a mixed pass
        }

        @Override
        public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
            if (mixed != null) { // every forward call waits for a pass: a chunk may ride with others' tokens
                submit(new Step(this, tokens, off, n, pos, logits, 0));
                return;
            }
            if (n == 1 && logits != null) {
                decode(this, tokens[off], pos, logits);
                return;
            }
            locked(() -> inner.forward(tokens, off, n, pos, logits));
        }

        @Override
        public void forwardAll(int[] tokens, int off, int n, int pos, float[] logitsOut, int stride) {
            if (mixed != null && n >= 1 && n <= inner.maxBatch()) { // a verification run joins the pass of the others
                submit(new Step(this, tokens, off, n, pos, logitsOut, stride));
                return;
            }
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
        public org.robomiri.jllm.kv.KvSequence kvSequence() {
            return inner.kvSequence();
        }

        @Override
        public StateSnapshot snapshotState(int n) {
            StateSnapshot[] r = new StateSnapshot[1];
            locked(() -> r[0] = inner.snapshotState(n));
            return r[0];
        }

        @Override
        public void saveState(int n, WritableByteChannel out) throws IOException {
            device.lock();
            try {
                inner.saveState(n, out);
            } finally {
                device.unlock();
            }
        }

        @Override
        public void loadState(int n, ReadableByteChannel in) throws IOException {
            device.lock();
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
