package org.robomiri.jllm.sampling;

/**
 * Restricts which tokens may come next (constrained decoding, e.g. {@link JsonConstraint} for OpenAI's
 * {@code response_format}). One instance per generated sequence: it tracks the output so far. The engine asks
 * {@link #allows} only for the candidates the sampler considers (most likely first), so a check may cost a little
 * but must not be done for the whole vocabulary on every step.
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
}
