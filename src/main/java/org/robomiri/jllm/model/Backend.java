package org.robomiri.jllm.model;

/** A device that runs the transformer forward pass and owns the KV cache. */
public interface Backend extends AutoCloseable {

    String name();

    /** Maximum sequence length (KV cache capacity). */
    int contextLength();

    /** Maximum number of tokens per {@link #forward} call (prefill chunk size). */
    int maxBatch();

    /**
     * Runs {@code n} tokens {@code tokens[off .. off+n)} occupying sequence positions {@code [pos, pos+n)},
     * appending their keys/values to the cache. Positions after {@code pos+n} are considered invalid afterwards.
     *
     * <p><b>Prefill and decode are not bit-identical.</b> A backend may compute an {@code n > 1} call with
     * different kernels than an {@code n == 1} call (the CPU backend dequantizes four weight rows and tiles over
     * tokens for {@code n > 1}, streams quantized rows - or int8 products - for {@code n == 1}; Metal uses
     * different threadgroup shapes), so the same token's keys, values and logits differ in the last bits
     * depending on how many tokens its call carried. Consequently a result depends on history: when the engine
     * reuses cached positions, those were produced either by an earlier prefill or by earlier decode steps, and
     * the logits of the new tokens can differ (and after a near-tie, the greedy continuation can diverge) from
     * running the same prompt from position 0 in one go. Only the sequence of {@code forward} calls, not the
     * token sequence alone, determines the output. {@code -Djllm.cpu.deterministic=true} makes the CPU backend
     * run every call token by token through its single-token kernels for comparisons (docs/07 §7.14).
     *
     * @param logits if non-null, receives the logits of the last token
     */
    void forward(int[] tokens, int off, int n, int pos, float[] logits);

    /**
     * Like {@link #forward} but returns the logits of <em>every</em> token of the chunk: row {@code i} (the
     * distribution after {@code tokens[off + i]}) goes to {@code logitsOut[i * stride ..]}, {@code min(head rows,
     * stride)} values each. The K/V of all {@code n} positions are appended, exactly as for {@code forward}; a
     * caller that accepts only a prefix simply continues from the shorter position (later calls overwrite the
     * rest). Used to verify speculative drafts in one pass. The default runs {@code n} single-token steps.
     *
     * @param stride row stride of {@code logitsOut} (normally the model's vocabulary size)
     */
    default void forwardAll(int[] tokens, int off, int n, int pos, float[] logitsOut, int stride) {
        float[] row = new float[stride];
        for (int i = 0; i < n; i++) {
            forward(tokens, off + i, 1, pos + i, row);
            System.arraycopy(row, 0, logitsOut, i * stride, stride);
        }
    }

    /**
     * Whether {@link #forwardEmbeds} works: the backend can start a forward pass from embedding vectors (the output
     * of a vision encoder) instead of token ids.
     */
    default boolean supportsEmbeds() {
        return false;
    }

    /**
     * Like {@link #forward} for {@code n} positions {@code [pos, pos+n)} whose input is not looked up in the embedding
     * table but given: {@code embeds[(off + i) * dim ..]} is the residual-stream input of position {@code pos + i}
     * ({@code dim} = the model's hidden size, used as it is: no embedding scale). Used for image tokens. Only when
     * {@link #supportsEmbeds}.
     */
    default void forwardEmbeds(float[] embeds, int off, int n, int pos, float[] logits) {
        forwardEmbeds(embeds, off, n, pos, null, logits);
    }

    /**
     * {@link #forwardEmbeds(float[], int, int, int, float[])} with Qwen3-VL's deepstack: {@code deepstack[l]} (the same
     * layout as {@code embeds}, rows {@code off ..}) is added to the hidden state of these positions after layer
     * {@code l}, for {@code l < deepstack.length}; null = none.
     */
    default void forwardEmbeds(float[] embeds, int off, int n, int pos, float[][] deepstack, float[] logits) {
        throw new UnsupportedOperationException(name() + " cannot run from embeddings");
    }

    /**
     * Whether {@link #forwardEmbedsBidirectional} works: attention inside a picture that runs both ways (Gemma 3).
     */
    default boolean supportsBidirectionalSpan() {
        return false;
    }

    /**
     * {@link #forwardEmbeds(float[], int, int, int, float[])} for a picture whose {@code n} tokens attend to each other
     * in both directions (Gemma 3: the 256 tokens of a picture see each other and everything before them; the text
     * after them sees all of them as usual): the keys and values of all {@code n} rows are stored first, and every row
     * attends to the keys {@code [0, pos + n)} (its sliding window, if the layer has one, counted back from the last
     * row). The rows must be one pass: {@code n} is at most {@link #maxBatch}. Only when
     * {@link #supportsBidirectionalSpan}.
     */
    default void forwardEmbedsBidirectional(float[] embeds, int off, int n, int pos, float[] logits) {
        throw new UnsupportedOperationException(name() + " cannot attend both ways inside a picture");
    }

    /**
     * Replaces the rotary positions of cache slots {@code [from, contextLength)} with {@code p} (Qwen-VL's M-RoPE; see
     * {@link RopePositions}), or with the plain {@code slot} when {@code p} is null. Keys already in the cache keep
     * the rotation they were stored with. Returns false, changing nothing, when the backend cannot.
     */
    default boolean setRopePositions(int from, RopePositions p) {
        return p == null;
    }

    /**
     * Whether {@link #setRopePositions} has left non-plain rotary positions in some slots; such a backend cannot join
     * {@link #forwardBatch} (the batch uses one backend's tables for all rows).
     */
    default boolean ropePositionsCustom() {
        return false;
    }

    /**
     * Whether the state after tokens {@code [0, N)} can be rewound to an earlier position. A KV cache can (positions
     * past {@code pos} are simply overwritten); a recurrent state (Mamba) cannot, so the engine then reuses the
     * cache only when the new prompt extends everything that was fed before, and otherwise restarts at position 0.
     */
    /**
     * Decodes one token for each of several backends that run on the same weights, in one pass: the projections, the
     * MLP and the LM head go over all the rows at once (the weights are read once for the whole batch, which is what
     * makes serving several conversations cheaper than serving them one after another), while each row's attention
     * uses its own backend's cache at its own position. {@code members[i]} gets {@code tokens[i]} at
     * {@code positions[i]} and its logits in {@code logits[i]}; {@code this} lends its buffers and must be one of the
     * members. Returns false, having done nothing, when the backend cannot (the caller then runs them one by one).
     */
    default boolean forwardBatch(Backend[] members, int[] tokens, int[] positions, float[][] logits) {
        return false;
    }

    /**
     * One piece of one conversation in a {@link #forwardMixed mixed pass}: {@code n} tokens {@code tokens[off .. off+n)}
     * at positions {@code [pos, pos+n)} of {@code owner}'s cache.
     *
     * @param logits receives the logits after the segment: with {@code stride == 0} the last row's (null = none wanted:
     *               a prefill chunk that is not the prompt's last), with {@code stride > 0} every row's, row {@code i}
     *               at {@code i * stride} (the verification of a speculative draft)
     */
    record Segment(Backend owner, int[] tokens, int off, int n, int pos, float[] logits, int stride) {
    }

    /**
     * Runs several conversations' pieces in <em>one</em> forward pass: the projections, the MLP and the LM head go over all
     * the rows at once (the weights are read once for the whole pass), while rotary embedding, the cache store and
     * attention run per segment in its owner's cache, a multi-token segment exactly as {@link #forward} would run it alone
     * and a single-token one as {@link #forwardBatch} runs a row. This is what lets a prefill chunk of one conversation
     * carry the next decode token of every other one (docs/25). The owners must be distinct, idle backends of the same
     * weights and cache type; {@code this} lends its activation buffers and need not be one of them; the total number of
     * rows is at most {@link #maxBatch}. Returns false, having done nothing, when the backend cannot (the caller then
     * runs the segments one by one).
     */
    default boolean forwardMixed(Segment[] segments) {
        return false;
    }

    /**
     * A point between two passes of one generation, where {@code n} positions of the cache hold the first {@code n} tokens
     * of {@code tokens}: a scheduler may pause the conversation here (its blocks are given back to the pool, which may spill
     * them while others run) and bring it back before returning. Returns the number of positions that are valid afterwards,
     * {@code n} unless something could not be brought back, in which case the caller computes {@code [valid, n)} again.
     * The default does nothing. {@code cancelled} is polled while paused, so that a client that went away is not waited for.
     */
    default int checkpoint(java.util.function.Supplier<int[]> tokens, int n, java.util.function.BooleanSupplier cancelled) {
        return n;
    }

    /**
     * The machine's thermal state as the GPU driver reports it: 0 nominal, 1 fair, 2 serious (the GPU is being slowed
     * down), 3 critical; -1 when the backend cannot tell (the CPU backend, an old native library).
     */
    default int thermalState() {
        return -1;
    }

    default boolean supportsPrefixReuse() {
        return true;
    }

    // ------------------------------------------------------------------ multi-token prediction (Qwen3.5 GGUF)

    /** Whether the model has a multi-token-prediction head this backend can run ({@link Qwen3NextMtp}). */
    default boolean hasMtp() {
        return false;
    }

    /**
     * Turns the head on or off. While on, every {@link #forward} / {@link #forwardAll} also feeds the head (its own KV
     * cache and the hidden state of the last position), at the cost of one extra transformer layer per token. Takes
     * effect from the next prompt (position 0).
     */
    default void enableMtp(boolean on) {
        throw new UnsupportedOperationException(name() + " has no multi-token-prediction head");
    }

    /** Whether {@link #mtpDraft} can be called now: the head follows every position passed so far. */
    default boolean mtpReady() {
        return false;
    }

    /**
     * The head's guess for the token that will follow {@code token}, which the model has just chosen to follow the last
     * position it processed and which is fed next: the head's logits (vocabulary-sized) go to {@code logits}.
     */
    default void mtpDraft(int token, float[] logits) {
        throw new UnsupportedOperationException(name() + " has no multi-token-prediction head");
    }

    /**
     * Whether {@link #checkpointState} / {@link #rollbackState} work: before a verification pass of two tokens the
     * recurrent state is armed to be kept after the first of them, and when the draft (the second) is rejected the state
     * is put back to that point (the KV cache needs nothing: later positions are overwritten).
     */
    default boolean supportsRollback() {
        return false;
    }

    /**
     * Arms the next pass ({@link #forwardAll} of two tokens): the recurrent state and the head's bookkeeping as they are
     * after its <em>first</em> token can be restored by {@link #rollbackState}. One checkpoint is kept. A backend may
     * save the state as it is now and replay, or have the pass save it on the way.
     */
    default void checkpointState() {
        throw new UnsupportedOperationException(name() + " cannot checkpoint its state");
    }

    /**
     * Puts the state to what it was after the first {@code keep} (= 1) tokens of the pass armed by
     * {@link #checkpointState}, as if only they had been fed.
     */
    default void rollbackState(int keep) {
        throw new UnsupportedOperationException(name() + " cannot roll back its state");
    }

    /**
     * The layout of {@link #saveState}'s output (cache type and shape), or null when this backend cannot save its
     * state. A saved state is only loaded into a backend reporting the same format.
     */
    default String stateFormat() {
        return null;
    }

    /** Writes the state after positions {@code [0, n)} (the KV cache). Only when {@link #stateFormat} is not null. */
    default void saveState(int n, java.nio.channels.WritableByteChannel out) throws java.io.IOException {
        throw new UnsupportedOperationException(name() + " cannot save its cache");
    }

    /** Replaces the state of positions {@code [0, n)} with what {@link #saveState} wrote. */
    default void loadState(int n, java.nio.channels.ReadableByteChannel in) throws java.io.IOException {
        throw new UnsupportedOperationException(name() + " cannot load a saved cache");
    }

    /**
     * Replaces the state of positions {@code [0, n)} with the first {@code n} positions of what {@link #saveState}
     * wrote for {@code stored >= n} positions, and consumes all of it. Lets a saved conversation that shares only a
     * prefix with a prompt be loaded without reading the rest. The default handles {@code n == stored} only.
     */
    default void loadState(int n, int stored, java.nio.channels.ReadableByteChannel in) throws java.io.IOException {
        if (n != stored) {
            throw new UnsupportedOperationException(name() + " cannot load part of a saved cache");
        }
        loadState(n, in);
    }

    /**
     * Whether {@link #saveStateWhileServing} works: the rows of positions {@code [0, n)} can be written out by another
     * thread while this backend serves requests that leave them alone (they only write positions {@code >= n}).
     */
    default boolean canSaveWhileServing() {
        return false;
    }

    /**
     * {@link #saveState} from a thread other than the one using the backend. The caller guarantees that positions
     * {@code [0, n)} are not written meanwhile; this method only reads them.
     */
    default void saveStateWhileServing(int n, java.nio.channels.WritableByteChannel out) throws java.io.IOException {
        throw new UnsupportedOperationException(name() + " cannot save its cache while serving");
    }

    /**
     * The block table of this backend's paged KV cache ({@code backend.kv-paged}, docs/24), or null when its cache is
     * flat. The engine uses it to take over blocks other conversations computed ({@link org.robomiri.jllm.kv.KvSequence#adopt})
     * and to publish its own once their tokens are final.
     */
    default org.robomiri.jllm.kv.KvSequence kvSequence() {
        return null;
    }

    /**
     * Where this backend's memory is and what the planner decided (docs/26 §26.2.7), as a JSON-ready tree (maps, lists, numbers,
     * strings): the budget, the components (weights on the GPU, experts on the CPU, scratch, KV by tier, other models), the
     * KV pool and its tiers, the streaming counters, and the latest decisions. The default describes the paged pool when
     * there is one, else {@code null}.
     */
    default java.util.Map<String, Object> memoryReport() {
        org.robomiri.jllm.kv.KvSequence s = kvSequence();
        return s == null ? null : org.robomiri.jllm.kv.KvReports.pool(s.pool());
    }

    /** A frozen copy of the first {@code n} positions' rows that another thread can write out ({@link #snapshotState}). */
    interface StateSnapshot extends AutoCloseable {
        /** Writes what {@link #saveState} would have written for the positions at the time of the snapshot. */
        void write(java.nio.channels.WritableByteChannel out) throws java.io.IOException;

        @Override
        void close();
    }

    /**
     * Freezes positions {@code [0, n)} for a background save without copying them: a paged cache retains the blocks, and
     * whoever writes one afterwards copies it first. Null when the backend has no such cache (the flat caches are saved
     * with {@link #saveStateWhileServing} and the engine waits before overwriting what is still being written).
     */
    default StateSnapshot snapshotState(int n) {
        return null;
    }

    /** Skips {@code bytes} of a saved state: a seek when {@code in} is seekable (a file), else reads them. */
    static void skipState(java.nio.channels.ReadableByteChannel in, long bytes) throws java.io.IOException {
        if (bytes <= 0) {
            return;
        }
        if (in instanceof java.nio.channels.SeekableByteChannel s) {
            s.position(s.position() + bytes);
            return;
        }
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocate((int) Math.min(bytes, 1 << 16));
        while (bytes > 0) {
            buf.clear().limit((int) Math.min(bytes, buf.capacity()));
            int r = in.read(buf);
            if (r < 0) {
                throw new java.io.EOFException("the saved cache file is truncated");
            }
            bytes -= r;
        }
    }

    /**
     * Replaces the state of positions {@code [0, n)} with that of {@code other}, a backend of the same model with the
     * same {@link #stateFormat}: copying a finished prefix costs a memory copy instead of a prefill. Both backends
     * must be idle (the engines hold their locks).
     */
    default void copyStateFrom(Backend other, int n) {
        throw new UnsupportedOperationException(name() + " cannot copy a cache");
    }

    /**
     * Pipelined greedy decoding: each step feeds one token and leaves the argmax of its logits on the device, where the
     * next step can read it, so the next step is queued before the CPU has looked at the previous token. Used by the
     * engine for greedy requests without penalties; while a pipe has steps in flight, no other method of the backend
     * may be called (the engine drains it first).
     */
    interface GreedyPipe {
        /** Queues the step that feeds {@code token} at {@code pos}, or with {@code token < 0} the argmax of the step before. */
        void submit(int token, int pos);

        /** The argmax token of the oldest step not read yet (waits for that step only). */
        int next();

        /** Steps submitted and not read yet. */
        int pending();

        /** Waits for every step in flight and drops their unread results (their cache rows are simply overwritten later). */
        void drain();
    }

    /** A {@link GreedyPipe} over the token ids {@code [0, vocab)}, or null when this backend has none. */
    default GreedyPipe greedyPipe(int vocab) {
        return null;
    }

    @Override
    void close();
}
