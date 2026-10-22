package org.robomiri.jllm.engine;

/**
 * How urgent a request is (docs/10 §10.14): a person waiting for an answer ({@link #INTERACTIVE}, the default) or a
 * background job such as a subagent or a batch of summaries ({@link #BACKGROUND}). The OpenAI endpoints take it as
 * {@code "priority": "interactive" | "background"} in the request body or the {@code X-Jllm-Priority} header; it travels
 * with the request as {@link Engine.Decoding#priority()}.
 *
 * <p>Today it orders one wait only: who gets the device next ({@link BatchGroup}'s device lock, taken by prefill chunks,
 * verification passes and decode batches). A background request never goes first while an interactive one waits for the
 * device (at most {@link BatchGroup#BACKGROUND_MAX_WAIT_MS} behind it, so it cannot starve), and in a decode batch its
 * step is the last to be taken when more steps are queued than fit. It does not preempt a pass that is running, does not
 * touch which slot serves the request and is ignored by the CPU-only engine without a batch group. The mixed scheduler
 * (docs/25, {@code engine.scheduler=mixed}) builds its queue classes, pass order and pausing on {@link #current()}.
 */
public enum Priority {
    INTERACTIVE,
    BACKGROUND;

    private static final ThreadLocal<Priority> CURRENT = new ThreadLocal<>();

    /** The priority of the request this thread is serving; {@link #INTERACTIVE} outside of one. */
    public static Priority current() {
        Priority p = CURRENT.get();
        return p == null ? INTERACTIVE : p;
    }

    /** Makes {@code p} the calling thread's priority until the returned scope is closed. */
    public static Scope enter(Priority p) {
        Priority before = CURRENT.get();
        CURRENT.set(p);
        return () -> {
            if (before == null) {
                CURRENT.remove();
            } else {
                CURRENT.set(before);
            }
        };
    }

    /** The priority named {@code s} (case-insensitive: interactive, background); null or empty = interactive. */
    public static Priority parse(String s) {
        if (s == null || s.isBlank()) {
            return INTERACTIVE;
        }
        return switch (s.trim().toLowerCase(java.util.Locale.ROOT)) {
            case "interactive" -> INTERACTIVE;
            case "background" -> BACKGROUND;
            default -> throw new IllegalArgumentException("priority must be 'interactive' or 'background', got '" + s + "'");
        };
    }

    /** Restores the previous priority when closed. */
    public interface Scope extends AutoCloseable {
        @Override
        void close();
    }
}
