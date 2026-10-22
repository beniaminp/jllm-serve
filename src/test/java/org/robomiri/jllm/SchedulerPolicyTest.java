package org.robomiri.jllm;

import org.robomiri.jllm.engine.Admission;
import org.robomiri.jllm.engine.PassPlanner;
import org.robomiri.jllm.kv.FileSpill;
import org.robomiri.jllm.kv.KvPool;
import org.robomiri.jllm.kv.KvSequence;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The policy of the mixed scheduler (docs/25) without a model: what a pass carries ({@link PassPlanner}), who gets a slot
 * ({@link Admission}: cache-aware routing, priority classes and aging, session limits, pausing a background slot, a client
 * that leaves the queue) and pausing a sequence by giving its blocks back to the pool ({@link KvSequence#suspend}).
 */
public final class SchedulerPolicyTest {

    private static int failures;

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "ok   " : "FAIL ") + what);
        if (!ok) {
            failures++;
        }
    }

    public static void main(String[] args) throws Exception {
        planner();
        admission();
        suspend();
        if (failures > 0) {
            System.out.println(failures + " FAILED");
            System.exit(1);
        }
        System.out.println("scheduler policy test passed");
    }

    // ---------------------------------------------------------------- the pass

    private static PassPlanner.Item item(String ref, PassPlanner.Kind k, int rows, int rank, long enq) {
        return new PassPlanner.Item(ref, k, rows, rank, enq);
    }

    private static String describe(List<PassPlanner.Pick> picks) {
        StringBuilder b = new StringBuilder();
        for (PassPlanner.Pick p : picks) {
            b.append(p.ref()).append(':').append(p.rows()).append(' ');
        }
        return b.toString().trim();
    }

    static void planner() {
        long ms = 1_000_000L;
        PassPlanner.Config cfg = new PassPlanner.Config(288, 16, 0, 0);
        // the worked example of docs/25: a 256-row chunk, six decode tokens and one verification of 5 rows
        List<PassPlanner.Item> pending = new ArrayList<>();
        pending.add(item("D1", PassPlanner.Kind.DECODE, 1, 0, 1 * ms));
        pending.add(item("P-A", PassPlanner.Kind.CHUNK, 256, 0, 2 * ms));
        pending.add(item("V1", PassPlanner.Kind.VERIFY, 5, 0, 3 * ms));
        for (int i = 2; i <= 6; i++) {
            pending.add(item("D" + i, PassPlanner.Kind.DECODE, 1, 0, (3 + i) * ms));
        }
        pending.add(item("P-B", PassPlanner.Kind.CHUNK, 256, 0, 20 * ms));
        List<PassPlanner.Pick> picks = describePicks(PassPlanner.plan(pending, 30 * ms, cfg));
        check(describe(picks).equals("P-A:256 D1:1 V1:5 D2:1 D3:1 D4:1 D5:1 D6:1"),
                "one chunk (the older), then every rider by arrival: " + describe(picks));
        int rows = 0;
        for (PassPlanner.Pick p : picks) {
            rows += p.rows();
        }
        check(rows == 267 && rows <= 288, "the pass carries " + rows + " of 288 rows");

        // a smaller budget trims the chunk, never the riders
        picks = PassPlanner.plan(pending, 30 * ms, new PassPlanner.Config(100, 16, 0, 0));
        check(picks.getFirst().ref().equals("P-A") && picks.getFirst().rows() == 100 - 11,
                "budget 100: the chunk is trimmed to " + picks.getFirst().rows() + " rows to make room for 11 rider rows");
        // too little room for a useful piece: the chunk waits, the riders go
        picks = PassPlanner.plan(pending, 30 * ms, new PassPlanner.Config(20, 16, 0, 0));
        check(describe(picks).startsWith("D1:1 V1:5") && !describe(picks).contains("P-"),
                "budget 20: leftover 9 rows < minChunk 16, so the chunk waits: " + describe(picks));
        // interactive chunk before an older background one; aging lifts the background one after the period
        List<PassPlanner.Item> two = List.of(item("bg", PassPlanner.Kind.CHUNK, 64, 1, 1 * ms),
                item("ia", PassPlanner.Kind.CHUNK, 64, 0, 5 * ms));
        check(PassPlanner.plan(two, 6 * ms, new PassPlanner.Config(300, 16, 10 * ms, 0)).getFirst().ref().equals("ia"),
                "an interactive chunk goes before an older background one");
        check(PassPlanner.plan(two, 25 * ms, new PassPlanner.Config(300, 16, 10 * ms, 0)).getFirst().ref().equals("bg"),
                "after 2 aging periods the background chunk counts as interactive and, being older, goes first");
        // the background cap
        List<PassPlanner.Item> mix = List.of(item("bg", PassPlanner.Kind.CHUNK, 256, 1, 1 * ms),
                item("d", PassPlanner.Kind.DECODE, 1, 0, 2 * ms));
        picks = PassPlanner.plan(mix, 3 * ms, new PassPlanner.Config(288, 16, 0, 64));
        check(picks.getFirst().rows() == 64, "a background chunk riding with an interactive token is capped: " + describe(picks));
        picks = PassPlanner.plan(List.of(item("bg", PassPlanner.Kind.CHUNK, 256, 1, 1 * ms)), 3 * ms,
                new PassPlanner.Config(288, 16, 0, 64));
        check(picks.getFirst().rows() == 256, "alone it is not");
        // never a stall
        picks = PassPlanner.plan(List.of(item("v", PassPlanner.Kind.VERIFY, 30, 0, 1 * ms)), 2 * ms,
                new PassPlanner.Config(10, 16, 0, 0));
        check(picks.size() == 1, "a run larger than the budget still goes, alone");
    }

    private static List<PassPlanner.Pick> describePicks(List<PassPlanner.Pick> p) {
        return p;
    }

    // ---------------------------------------------------------------- the queue

    /** Slots are strings; "match" is a table of how much of a prompt (identified by its first token) a slot's cache holds. */
    private static final class Fake implements Admission.Ops<String> {
        final java.util.Map<String, Integer> hit = new java.util.concurrent.ConcurrentHashMap<>();
        final List<String> log = java.util.Collections.synchronizedList(new ArrayList<>());

        @Override
        public int match(String slot, int[] prompt) {
            return hit.getOrDefault(slot + prompt[0], 0);
        }

        @Override
        public boolean pause(String slot) {
            log.add("pause " + slot);
            return true;
        }

        @Override
        public void resume(String slot) {
            log.add("resume " + slot);
        }
    }

    private static Thread waiter(Admission<String> q, int first, int rank, String session, AtomicReference<String> got,
                                 BooleanSupplierBox gone) {
        Thread t = new Thread(() -> {
            try {
                got.set(q.acquire(new int[]{first}, rank, session, gone));
            } catch (Admission.Abandoned e) {
                got.set("gone");
            }
        });
        t.start();
        return t;
    }

    private interface BooleanSupplierBox extends java.util.function.BooleanSupplier {
    }

    private static void waitFor(java.util.function.BooleanSupplier c, long ms) throws InterruptedException {
        long end = System.nanoTime() + ms * 1_000_000L;
        while (!c.getAsBoolean() && System.nanoTime() < end) {
            Thread.sleep(2);
        }
    }

    static void admission() throws Exception {
        BooleanSupplierBox never = () -> false;
        // 1. cache-aware routing: the idle slot with the longest prefix, and the waiting request with the most to gain
        Fake ops = new Fake();
        Admission<String> q = new Admission<>(List.of("s0", "s1", "s2"), ops, new Admission.Config(0, true, 0, 0));
        ops.hit.put("s1" + 7, 5000);
        ops.hit.put("s2" + 8, 300);
        String a = q.acquire(new int[]{7}, 0, null, never);
        check(a.equals("s1"), "a prompt goes to the idle slot holding its prefix (" + a + ")");
        String b = q.acquire(new int[]{7}, 0, null, never);
        check(!b.equals("s1"), "the slot is taken: the same prompt runs elsewhere (" + b + ")");
        q.release(a);
        q.release(b);

        // 2. the request with the longest cached prefix wins a freed slot
        Fake o2 = new Fake();
        Admission<String> q2 = new Admission<>(List.of("x"), o2, new Admission.Config(0, true, 0, 0));
        o2.hit.put("x" + 1, 10);
        o2.hit.put("x" + 2, 4000);
        String held = q2.acquire(new int[]{9}, 0, null, never);
        AtomicReference<String> g1 = new AtomicReference<>();
        AtomicReference<String> g2 = new AtomicReference<>();
        Thread t1 = waiter(q2, 1, 0, null, g1, never);
        waitFor(() -> q2.queued() == 1, 2000);
        Thread t2 = waiter(q2, 2, 0, null, g2, never);
        waitFor(() -> q2.queued() == 2, 2000);
        q2.release(held);
        waitFor(() -> g1.get() != null || g2.get() != null, 2000);
        check(g2.get() != null && g1.get() == null, "of two waiting requests the one with 4000 cached tokens gets the freed slot first");
        q2.release("x");
        t1.join(2000);
        t2.join(2000);
        check(g1.get() != null, "the other follows");
        q2.release("x");

        // 3. an interactive request goes before an older background one
        Fake o3 = new Fake();
        Admission<String> q3 = new Admission<>(List.of("x"), o3, new Admission.Config(0, true, 0, 0));
        String h3 = q3.acquire(new int[]{9}, 0, null, never);
        AtomicReference<String> bg = new AtomicReference<>();
        AtomicReference<String> ia = new AtomicReference<>();
        Thread tb = waiter(q3, 1, 1, null, bg, never);
        waitFor(() -> q3.queued() == 1, 2000);
        Thread ti = waiter(q3, 2, 0, null, ia, never);
        waitFor(() -> q3.queued() == 2, 2000);
        q3.release(h3);
        waitFor(() -> bg.get() != null || ia.get() != null, 2000);
        check(ia.get() != null && bg.get() == null, "an interactive request overtakes an older background one");
        q3.release("x");
        tb.join(2000);
        ti.join(2000);
        q3.release("x");

        // 4. aging: a background request that waited long enough counts as interactive (and, being older, goes first)
        Fake o4 = new Fake();
        Admission<String> q4 = new Admission<>(List.of("x"), o4, new Admission.Config(0, true, 0, 150_000_000L));
        String h4 = q4.acquire(new int[]{9}, 0, null, never);
        AtomicReference<String> bg4 = new AtomicReference<>();
        AtomicReference<String> ia4 = new AtomicReference<>();
        Thread tb4 = waiter(q4, 1, 1, null, bg4, never);
        waitFor(() -> q4.queued() == 1, 2000);
        Thread.sleep(250);
        Thread ti4 = waiter(q4, 2, 0, null, ia4, never);
        waitFor(() -> q4.queued() == 2, 2000);
        q4.release(h4);
        waitFor(() -> bg4.get() != null || ia4.get() != null, 2000);
        check(bg4.get() != null && ia4.get() == null, "after 250 ms (aging period 150 ms) the background request goes first");
        q4.release("x");
        tb4.join(2000);
        ti4.join(2000);
        q4.release("x");

        // 5. per-session limit
        Fake o5 = new Fake();
        Admission<String> q5 = new Admission<>(List.of("a", "b", "c"), o5, new Admission.Config(0, true, 1, 0));
        String s1 = q5.acquire(new int[]{1}, 0, "alice", never);
        AtomicReference<String> second = new AtomicReference<>();
        Thread ts = waiter(q5, 2, 0, "alice", second, never);
        String other = q5.acquire(new int[]{3}, 0, "bob", never);
        Thread.sleep(150);
        check(second.get() == null && other != null, "a session already holding its one slot waits although slots are free; another session does not");
        q5.release(s1);
        ts.join(2000);
        check(second.get() != null, "and goes when its first request ends");
        q5.release(second.get());
        q5.release(other);

        // 6. preemption: one running place, a background request holds it, an interactive one arrives
        Fake o6 = new Fake();
        Admission<String> q6 = new Admission<>(List.of("a", "b"), o6, new Admission.Config(1, true, 0, 0));
        String bgSlot = q6.acquire(new int[]{1}, 1, null, never);
        AtomicReference<String> iaSlot = new AtomicReference<>();
        Thread t6 = waiter(q6, 2, 0, null, iaSlot, never);
        t6.join(3000);
        check(iaSlot.get() != null && !iaSlot.get().equals(bgSlot) && o6.log.contains("pause " + bgSlot),
                "an interactive request pauses the running background slot and takes the place: " + o6.log);
        check(q6.pausedSlots() == 1 && q6.runningSlots() == 1, "one slot paused, one running");
        q6.release(iaSlot.get());
        check(o6.log.contains("resume " + bgSlot) && q6.runningSlots() == 1,
                "when the interactive request ends the paused background slot resumes: " + o6.log);
        q6.release(bgSlot);
        // an interactive slot is never paused for another interactive request
        Fake o7 = new Fake();
        Admission<String> q7 = new Admission<>(List.of("a", "b"), o7, new Admission.Config(1, true, 0, 0));
        String run7 = q7.acquire(new int[]{1}, 0, null, never);
        AtomicReference<String> w7 = new AtomicReference<>();
        Thread t7 = waiter(q7, 2, 0, null, w7, never);
        Thread.sleep(150);
        check(w7.get() == null && o7.log.isEmpty(), "an interactive request waits behind a running interactive one, nothing is paused");
        q7.release(run7);
        t7.join(2000);
        q7.release(w7.get());

        // 8. a client that leaves the queue is woken and dropped
        Fake o8 = new Fake();
        Admission<String> q8 = new Admission<>(List.of("x"), o8, new Admission.Config(0, true, 0, 0));
        String h8 = q8.acquire(new int[]{9}, 0, null, never);
        AtomicBoolean left = new AtomicBoolean();
        AtomicReference<String> r8 = new AtomicReference<>();
        Thread t8 = waiter(q8, 1, 0, null, r8, left::get);
        waitFor(() -> q8.queued() == 1, 2000);
        long t0 = System.nanoTime();
        left.set(true);
        t8.join(2000);
        double ms = (System.nanoTime() - t0) / 1e6;
        check("gone".equals(r8.get()) && q8.queued() == 0 && ms < 500,
                "a queued request whose client left is dropped within " + Math.round(ms) + " ms");
        q8.release(h8);
        String again = q8.acquire(new int[]{1}, 0, null, never);
        check(again.equals("x"), "and its place in line is not held");
        q8.release(again);
    }

    // ---------------------------------------------------------------- pausing a sequence

    static void suspend() throws Exception {
        KvPoolTest.FakeStorage st = new KvPoolTest.FakeStorage(16, 40);
        KvPool pool = new KvPool(st, 12, "fake");
        Path dir = Files.createTempDirectory("jllm-pause");
        try {
            pool.setSpill(new FileSpill(dir, "fake-model", 1L << 30));
            int[] t = KvPoolTest.tokens(11, 100); // 6 full blocks and a tail of 4 rows
            KvSequence a = pool.newSequence(512);
            KvPoolTest.fill(st, a, t, 0, t.length);
            int before = pool.residentBlocks();
            KvSequence.Suspended s = a.suspend(t, t.length);
            check(s.positions() == 100 && a.blocks() == 0 && a.length() == 0, "a paused sequence holds no table");
            check(pool.stats().usedBlocks() == 1 && pool.stats().cachedBlocks() == 6,
                    "its 6 full blocks are cached (evictable), only the 4-row tail is still held: " + pool.stats().usedBlocks()
                            + " used, " + pool.stats().cachedBlocks() + " cached");
            // another conversation needs the whole pool: the paused blocks are spilled
            int[] u = KvPoolTest.tokens(12, 160);
            KvSequence b = pool.newSequence(512);
            KvPoolTest.fill(st, b, u, 0, u.length); // 10 blocks of the 11 left: evicts paused blocks
            check(pool.stats().spills() > 0, "the other conversation pushed " + pool.stats().spills() + " paused blocks to disk");
            b.close();
            int back = a.resume(s, t);
            check(back == 100 && a.length() == 100 && KvPoolTest.holds(st, a, t, 100),
                    "resumed: all 100 positions are back and read the right rows (" + back + ")");
            check(pool.stats().reloads() > 0, "some of them came back from disk (" + pool.stats().reloads() + " reloads)");
            // without a store the blocks can be lost: resume reports how much is valid
            KvPoolTest.FakeStorage st2 = new KvPoolTest.FakeStorage(16, 40);
            KvPool p2 = new KvPool(st2, 9, "fake");
            KvSequence c = p2.newSequence(512);
            KvPoolTest.fill(st2, c, t, 0, t.length);
            KvSequence.Suspended s2 = c.suspend(t, t.length);
            KvSequence d = p2.newSequence(512);
            KvPoolTest.fill(st2, d, u, 0, 112); // 7 blocks: the 2 free ones and 5 of the paused, leaf first
            int got = c.resume(s2, t);
            check(got < 100 && got % 16 == 0 && KvPoolTest.holds(st2, c, t, got),
                    "blocks lost without a store: resume reports " + got + " valid positions, the caller computes the rest");
        } finally {
            try (var files = Files.walk(dir)) {
                files.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
    }
}
