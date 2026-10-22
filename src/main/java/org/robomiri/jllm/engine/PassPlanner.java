package org.robomiri.jllm.engine;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Chooses what one mixed forward pass carries (docs/25, {@code engine.scheduler=mixed}). The pending work of a model's
 * slots is of three kinds:
 * <ul>
 *   <li>{@link Kind#DECODE}: one token of a generating conversation (1 row);</li>
 *   <li>{@link Kind#VERIFY}: a speculative draft's {@code k + 1} tokens, all rows' logits wanted (a short run);</li>
 *   <li>{@link Kind#CHUNK}: a slice of a prompt being prefilled (up to {@code backend.batch} rows, compute bound).</li>
 * </ul>
 * A pass is capped at a budget of rows (the activation buffers hold {@code backend.batch + engine.mixed-reserve}).
 * The plan is Sarathi's: the next token of every generating conversation rides along (they are cheap and waiting makes
 * the stream stutter), then one prefill chunk takes what is left of the budget, trimmed if it does not fit (the rest
 * stays pending). Inside a kind the order is by priority class, then arrival, with <em>aging</em>: a request waiting
 * longer than {@code agePromote} counts one class higher per period, so background work is delayed, never starved.
 *
 * <p>Pure bookkeeping: no backend, no threads; the caller turns the picks into segments of a pass.
 */
public final class PassPlanner {

    public enum Kind { DECODE, VERIFY, CHUNK }

    /**
     * One piece of pending work.
     *
     * @param ref      the caller's handle, returned in the {@link Pick}
     * @param rows     rows the item still needs (a chunk may be run in parts)
     * @param rank     priority class, 0 = interactive (first), 1 = background, ...
     * @param enqueued {@link System#nanoTime()} when it started waiting
     */
    public record Item(Object ref, Kind kind, int rows, int rank, long enqueued) {
    }

    /** {@code rows} of {@code ref} go into the pass (fewer than the item's rows: a chunk run in parts). */
    public record Pick(Object ref, Kind kind, int rows) {
    }

    /**
     * @param budget        most rows in a pass (at most the backend's capacity)
     * @param minChunk      a chunk is trimmed to the leftover budget only if that is at least this many rows (or the chunk's
     *                      own size when smaller); else it waits for a pass with more room
     * @param agePromote    nanoseconds of waiting that lift an item one class; 0 = no aging
     * @param backgroundCap with a class-0 rider in the pass, a chunk of a lower class runs at most this many rows
     *                      (0 = no cap): keeps interactive streams smooth while a background prompt prefills
     */
    public record Config(int budget, int minChunk, long agePromote, int backgroundCap) {
        public Config {
            if (budget < 1) {
                throw new IllegalArgumentException("a pass needs a budget of at least one row");
            }
        }
    }

    private PassPlanner() {
    }

    /** The class of {@code it} at time {@code now}: its rank lowered by one per {@code agePromote} waited (not below 0). */
    static int effectiveRank(Item it, long now, Config cfg) {
        if (cfg.agePromote() <= 0) {
            return it.rank();
        }
        long periods = Math.max(0, now - it.enqueued()) / cfg.agePromote();
        return (int) Math.max(0, it.rank() - periods);
    }

    /**
     * The picks of the next pass, the chunk (if any) first so that its rows start the batch, then the riders by class and
     * arrival. Empty when nothing is pending. The chunk's {@link Pick#rows()} may be fewer than the item's.
     */
    public static List<Pick> plan(List<Item> pending, long now, Config cfg) {
        List<Item> riders = new ArrayList<>();
        List<Item> chunks = new ArrayList<>();
        for (Item it : pending) {
            (it.kind() == Kind.CHUNK ? chunks : riders).add(it);
        }
        Comparator<Item> order = Comparator.<Item>comparingInt(it -> effectiveRank(it, now, cfg))
                .thenComparingLong(Item::enqueued);
        riders.sort(order);
        chunks.sort(order);
        List<Pick> riderPicks = new ArrayList<>();
        int used = 0;
        boolean interactiveRider = false;
        for (Item r : riders) {
            if (used + r.rows() <= cfg.budget()) {
                riderPicks.add(new Pick(r.ref(), r.kind(), r.rows()));
                used += r.rows();
                interactiveRider |= effectiveRank(r, now, cfg) == 0;
            }
        }
        List<Pick> out = new ArrayList<>();
        if (!chunks.isEmpty()) {
            Item c = chunks.getFirst();
            int room = cfg.budget() - used;
            if (cfg.backgroundCap() > 0 && interactiveRider && effectiveRank(c, now, cfg) > 0) {
                room = Math.min(room, cfg.backgroundCap());
            }
            int rows = Math.min(c.rows(), room);
            if (rows >= Math.min(cfg.minChunk(), c.rows())) {
                out.add(new Pick(c.ref(), Kind.CHUNK, rows));
            }
        }
        out.addAll(riderPicks);
        if (out.isEmpty() && !pending.isEmpty()) { // nothing fits the budget as it is: the oldest item goes alone (never a stall)
            Item first = pending.stream().min(Comparator.comparingLong(Item::enqueued)).orElseThrow();
            out.add(new Pick(first.ref(), first.kind(), first.kind() == Kind.CHUNK ? Math.min(first.rows(), cfg.budget()) : first.rows()));
        }
        return out;
    }
}
