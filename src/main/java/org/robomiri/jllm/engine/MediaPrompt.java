package org.robomiri.jllm.engine;

import org.robomiri.jllm.model.RopePositions;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.List;

/**
 * A prompt with pictures in it: the usual token ids, plus the ranges of ids ({@link Span spans}) that are not looked
 * up in the embedding table but given as vectors (what a vision encoder made of an image). The ids inside a span are
 * placeholders (the model's image token), so the text side, the sampler's history and the n-gram speculator see
 * ordinary ids ({@link #tokens()}).
 *
 * <p>The engine decides what it may reuse from the KV cache by comparing ids, which would treat two different
 * pictures as equal. {@link #identity()} therefore carries, inside each span, an id derived from the picture's hash
 * (a number far above any vocabulary), so the cache bookkeeping, prefix sharing between engines and saved caches see
 * two pictures as two different prefixes and the same picture as the same one. Those ids never reach a backend:
 * spans are fed as embeddings.
 */
public final class MediaPrompt {

    /** First of the ids {@link #identity()} puts inside spans; real vocabularies are far smaller. */
    public static final int VIRTUAL_BASE = 1 << 30;

    /**
     * One picture's place in the prompt.
     *
     * @param start  index of its first id in the prompt
     * @param length number of ids (= rows of {@code embeds})
     * @param hash   identifies the picture and how it was encoded (equal hashes, equal embeddings)
     * @param embeds {@code length x dim} floats
     * @param deepstack Qwen3-VL: extra features of the same shape that the model adds to its hidden state after layer 0, 1, ...
     *                  (null for the other models)
     */
    public record Span(int start, int length, String hash, float[] embeds, float[][] deepstack) {
        public Span(int start, int length, String hash, float[] embeds) {
            this(start, length, hash, embeds, null);
        }

        public int end() {
            return start + length;
        }
    }

    private final int[] tokens;
    private final int[] identity;
    private final List<Span> spans;
    private final RopePositions rope;
    private final boolean nonCausal;

    /**
     * @param tokens    the prompt ids, with the model's image placeholder id inside every span
     * @param spans     the spans in order, not overlapping
     * @param rope      rotary positions when slots are not positions (Qwen-VL), else null
     * @param nonCausal the picture tokens attend to each other in both directions (Gemma 3): a span is then fed in one
     *                  pass
     */
    public MediaPrompt(int[] tokens, List<Span> spans, RopePositions rope, boolean nonCausal) {
        this.tokens = tokens;
        this.spans = List.copyOf(spans);
        this.rope = rope;
        this.nonCausal = nonCausal;
        this.identity = tokens.clone();
        int prevEnd = 0;
        for (Span s : spans) {
            if (s.start() < prevEnd || s.end() > tokens.length) {
                throw new IllegalArgumentException("spans overlap or leave the prompt");
            }
            prevEnd = s.end();
            Arrays.fill(identity, s.start(), s.end(), virtualId(s.hash()));
        }
    }

    /** An id for a picture hash: {@link #VIRTUAL_BASE} + 30 bits of it. */
    static int virtualId(String hash) {
        try {
            byte[] d = MessageDigest.getInstance("SHA-256").digest(hash.getBytes(StandardCharsets.UTF_8));
            int v = (d[0] & 0xFF) << 24 | (d[1] & 0xFF) << 16 | (d[2] & 0xFF) << 8 | (d[3] & 0xFF);
            return VIRTUAL_BASE | (v & (VIRTUAL_BASE - 1));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    /** The prompt ids with placeholders in the spans: what the text model, the sampler and the speculator see. */
    public int[] tokens() {
        return tokens;
    }

    /** The prompt ids with a picture-specific id in each span: what cache bookkeeping compares. */
    public int[] identity() {
        return identity;
    }

    public List<Span> spans() {
        return spans;
    }

    public RopePositions rope() {
        return rope;
    }

    public boolean nonCausal() {
        return nonCausal;
    }

    public int length() {
        return tokens.length;
    }

    /** The span that contains position {@code pos}, or null. */
    public Span spanAt(int pos) {
        for (Span s : spans) {
            if (pos < s.start()) {
                return null;
            }
            if (pos < s.end()) {
                return s;
            }
        }
        return null;
    }

    /** The first position after {@code pos} where a span starts (the prompt's length if none). */
    public int nextSpanStart(int pos) {
        for (Span s : spans) {
            if (s.start() > pos) {
                return s.start();
            }
        }
        return tokens.length;
    }

    /** Same pictures at the same places with the same embeddings as {@code o}'s hashes say. */
    public boolean sameMedia(MediaPrompt o) {
        return Arrays.equals(identity, o.identity);
    }
}
