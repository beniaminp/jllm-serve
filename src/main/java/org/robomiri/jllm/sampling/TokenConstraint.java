package org.robomiri.jllm.sampling;

import org.robomiri.jllm.tokenizer.Tokenizer;

/**
 * Restricts which tokens may come next (constrained decoding, e.g. {@link JsonConstraint} for OpenAI's
 * {@code response_format}, {@link ToolCallConstraint} for tool calls). One instance per generated sequence: it
 * tracks the output so far. The engine asks {@link #allows} only for the candidates the sampler considers (most
 * likely first), so a check may cost a little but must not be done for the whole vocabulary on every step.
 */
public interface TokenConstraint {

    /** Whether appending {@code token} keeps the output a valid prefix. Never true for end-of-sequence tokens. */
    boolean allows(int token);

    /** Appends a token the sampler chose (one {@link #allows} said yes to). */
    void accept(int token);

    /** Whether the output is complete as it stands, so that an end-of-sequence token may be chosen now. */
    boolean canEnd();

    /** Whether the output is complete and nothing more may follow: the engine then ends with finish reason "stop". */
    boolean mustEnd();

    /**
     * Whether almost every token is allowed right now (free text between tool calls): the sampler then draws as if
     * unconstrained and only falls back to the masked distribution when the drawn token is not allowed
     * ({@link Sampler#sample(float[], TokenConstraint, java.util.function.IntPredicate)}).
     */
    default boolean relaxed() {
        return false;
    }

    /**
     * An independent copy in the same state (speculative decoding checks drafted tokens on a copy), or null when the
     * constraint cannot be copied (the engine then does not speculate under it).
     */
    default TokenConstraint copy() {
        return null;
    }

    /**
     * Jump-forward: tokens that must come next whatever the model prefers, because the constraint allows exactly
     * one continuation (e.g. {@code {"name": "} after a tool-call tag). The engine appends them without sampling and
     * runs them through the model in one pass. At most {@code max}; empty when nothing is forced.
     *
     * @param tok the tokenizer the forced text is encoded with
     */
    default int[] jump(Tokenizer tok, int max) {
        return new int[0];
    }
}
