package org.robomiri.jllm.sampling;

import org.robomiri.jllm.tokenizer.Tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/**
 * A {@link TokenConstraint} driven by a byte automaton: a token is allowed when the automaton, fed the token's bytes
 * one by one from (a copy of) the current state, accepts all of them. Shared by {@link JsonConstraint} and
 * {@link ToolCallConstraint}; provides the three things that make constrained decoding cheap:
 *
 * <ul>
 *   <li><b>allowed-token masks.</b> Checking a candidate costs a state copy and one step per byte, fine for the few
 *   most likely tokens the sampler looks at first. When the sampler has to search further (none of its top
 *   candidates is allowed, typically the first token of a JSON value when the model wanted to write prose) the
 *   whole allowed set is computed once by walking the vocabulary trie ({@link TokenTrie}) and later checks are a
 *   bit test. Masks of strict states are memoized by the state's {@linkplain State#key key}, per
 *   constraint family ({@link MaskCache}), so recurring states (the start of a tool's arguments, the position
 *   after a value in an object) cost a lookup the second time.</li>
 *   <li><b>jump-forward</b> ({@link #jump}): when exactly one byte is allowed, repeatedly, the forced text is
 *   tokenized and appended without sampling.</li>
 *   <li><b>copies</b> for speculative decoding.</li>
 * </ul>
 */
abstract class ByteConstraint implements TokenConstraint {

    /** A state of the byte automaton. */
    interface State extends TokenTrie.ByteState {
        @Override
        State copy();

        /** Whether the output may end here (an end-of-sequence token is allowed). */
        boolean canEnd();

        /** Whether the output is complete and nothing may follow. */
        boolean done();

        /** Whether a special token (no bytes, e.g. {@code <|im_start|>}) is allowed here. */
        default boolean allowsSpecial() {
            return false;
        }

        /**
         * Whether a large part of the vocabulary is allowed here (inside a JSON string): not worth computing the
         * allowed set ({@link ByteConstraint#allowedTokens}).
         */
        default boolean wide() {
            return false;
        }

        /** Whether nearly everything is allowed here ({@link TokenConstraint#relaxed}). */
        default boolean relaxed() {
            return false;
        }

        /**
         * A key that identifies the set of tokens this state allows (two states with equal keys allow the same
         * tokens), or null when the state is not worth memoizing.
         */
        default Object key() {
            return null;
        }
    }

    /** Memoized allowed-token masks by state key, bounded; shared by the constraints of one schema / tool set. */
    static final class MaskCache {
        private static final int MAX = 256;
        private final java.util.LinkedHashMap<Object, Masks> map = new java.util.LinkedHashMap<>(64, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(java.util.Map.Entry<Object, Masks> e) {
                return size() > MAX;
            }
        };

        synchronized Masks get(Object key) {
            return map.get(key);
        }

        synchronized void put(Object key, Masks m) {
            map.put(key, m);
        }
    }

    /**
     * Whether allowed-token masks are computed and memoized ({@code engine.constraint-masks}); off, every candidate
     * is checked by feeding its bytes to the automaton, as before the masks existed.
     */
    private static volatile boolean fastMasks = !"false".equals(System.getProperty("jllm.constraint.masks"));

    static void setFastMasks(boolean on) {
        fastMasks = on;
    }

    /** Candidate checks in one state after which the full allowed set is computed (the sampler is searching wide). */
    static final int MASK_AFTER = 300;
    /** Longest forced text a jump-forward computes. */
    static final int MAX_JUMP_BYTES = 256;

    final byte[][] tokens;
    final MaskCache cache;
    State state;
    private long[] mask; // the allowed set of the current state, once computed
    private int[] list; // the same as token ids, when small
    private int checks;

    /** Statistics for benchmarks: masks computed by a trie walk, and trie nodes visited by them. */
    static long masksComputed;
    static long nodesVisited;

    ByteConstraint(byte[][] tokens, State start, MaskCache cache) {
        this.tokens = tokens;
        this.state = start;
        this.cache = cache != null ? cache : new MaskCache(); // (at least the states of this sequence)
    }

    /** Copy constructor. */
    ByteConstraint(ByteConstraint o) {
        this.tokens = o.tokens;
        this.state = o.state.copy();
        this.cache = o.cache;
        this.mask = o.mask; // immutable once built
        this.list = o.list;
        this.tooBig = o.tooBig;
    }

    @Override
    public boolean allows(int token) {
        byte[] b = token >= 0 && token < tokens.length ? tokens[token] : null;
        if (b == null) {
            return token >= 0 && token < tokens.length && state.allowsSpecial();
        }
        if (mask == null && fastMasks) {
            if (++checks == MASK_AFTER) {
                masks(Integer.MAX_VALUE);
            } else if (checks == 1) {
                cached();
            }
        }
        if (mask != null) {
            return (mask[token >>> 6] & (1L << token)) != 0;
        }
        State s = state.copy();
        for (byte x : b) {
            if (!s.feed(x & 0xff)) {
                return false;
            }
        }
        return true;
    }

    /**
     * The tokens this state allows, in id order, when there are at most {@value #LIST_MAX} and finding them takes a
     * trie walk of at most {@value #LIST_BUDGET} nodes (a structural JSON position, a key of a closed object, an enum
     * value, the envelope of a tool call); else null (inside a string most of the vocabulary is allowed). The walk
     * is done when {@code compute} is set or when the state was seen before (the memo then marks it), so one-off
     * states do not pay for it; its result (or the fact that there is none) is memoized by state.
     */
    @Override
    public int[] allowedTokens(boolean compute) {
        if (!fastMasks || state.relaxed() || (mask == null && state.wide())) {
            return null;
        }
        if (mask == null && !tooBig) {
            boolean seen = cached();
            if (mask == null && !tooBig) {
                if (compute || seen) {
                    masks(LIST_BUDGET);
                } else {
                    Object key = state.key();
                    if (key != null) {
                        cache.put(key, SEEN);
                    }
                }
            }
        }
        return list;
    }

    /** Most tokens {@link #allowedTokens} lists. */
    static final int LIST_MAX = 2048;
    /** Most trie nodes {@link #allowedTokens} visits. */
    static final int LIST_BUDGET = 6000;
    /** Cached for a state whose allowed set is too large to compute within {@link #LIST_BUDGET}. */
    private static final Masks TOO_BIG = new Masks(null, null);
    /** Cached for a state that was met once without computing its allowed set. */
    private static final Masks SEEN = new Masks(null, null);

    /** A state's allowed set as a bit mask, and as a list when small. */
    record Masks(long[] mask, int[] list) {
    }

    private boolean tooBig;

    /**
     * Takes the memoized masks of the current state, if any.
     *
     * @return whether the state was met before without its masks being computed
     */
    private boolean cached() {
        Object key = state.key();
        Masks m = key == null ? null : cache.get(key);
        if (m == TOO_BIG) {
            tooBig = true;
        } else if (m != null && m != SEEN) {
            mask = m.mask;
            list = m.list;
        }
        return m == SEEN;
    }

    /**
     * Computes the allowed set of the current state (special tokens excluded) with the vocabulary trie, giving up
     * after {@code budget} nodes; memoizes the outcome.
     */
    private void masks(int budget) {
        long[] m = new long[(tokens.length + 63) >>> 6];
        int visited = TokenTrie.of(tokens).allowed(state.copy(), m, budget);
        synchronized (ByteConstraint.class) {
            masksComputed++;
            nodesVisited += Math.max(visited, 0);
        }
        Masks result;
        if (visited < 0) {
            tooBig = true;
            result = TOO_BIG;
        } else {
            int n = 0;
            for (long w : m) {
                n += Long.bitCount(w);
            }
            int[] l = null;
            if (n <= LIST_MAX) {
                l = new int[n];
                int k = 0;
                for (int i = 0; i < m.length; i++) {
                    for (long w = m[i]; w != 0; w &= w - 1) {
                        l[k++] = i * 64 + Long.numberOfTrailingZeros(w);
                    }
                }
            }
            mask = m;
            list = l;
            result = new Masks(m, l);
        }
        Object key = state.key();
        if (key != null) {
            cache.put(key, result);
        }
    }

    @Override
    public void accept(int token) {
        byte[] b = token >= 0 && token < tokens.length ? tokens[token] : null;
        if (b == null) {
            if (token >= 0 && token < tokens.length && state.allowsSpecial()) {
                return; // a special token in free text: no bytes, no state change
            }
            throw new IllegalArgumentException("token " + token + " cannot be part of this output");
        }
        State s = state.copy();
        for (byte x : b) {
            if (!s.feed(x & 0xff)) {
                throw new IllegalArgumentException("token " + token + " is not valid here");
            }
        }
        state = s;
        mask = null;
        list = null;
        tooBig = false;
        checks = 0;
    }

    @Override
    public boolean canEnd() {
        return state.canEnd();
    }

    @Override
    public boolean mustEnd() {
        return state.done();
    }

    @Override
    public boolean relaxed() {
        return state.relaxed();
    }

    /** Whether {@code text} (UTF-8) is accepted from the current state; for tests and diagnostics. */
    public boolean allowsText(String text) {
        return allowsBytes(text.getBytes(StandardCharsets.UTF_8));
    }

    /** Whether {@code bytes} are accepted from the current state, fed one by one (no mask); for tests. */
    public boolean allowsBytes(byte[] bytes) {
        State s = state.copy();
        for (byte b : bytes) {
            if (!s.feed(b & 0xff)) {
                return false;
            }
        }
        return true;
    }

    /** Whether {@code text} (UTF-8) is accepted from the current state and leaves it complete; for tests. */
    public boolean completesWith(String text) {
        State s = state.copy();
        for (byte b : text.getBytes(StandardCharsets.UTF_8)) {
            if (!s.feed(b & 0xff)) {
                return false;
            }
        }
        return s.canEnd();
    }

    /** Feeds {@code text} (UTF-8) as if generated; for tests. @throws IllegalArgumentException if not allowed */
    public void acceptText(String text) {
        State s = state.copy();
        for (byte b : text.getBytes(StandardCharsets.UTF_8)) {
            if (!s.feed(b & 0xff)) {
                throw new IllegalArgumentException("not allowed: " + text);
            }
        }
        state = s;
        mask = null;
        list = null;
        tooBig = false;
        checks = 0;
    }

    /**
     * The bytes that must come next: while exactly one byte is allowed and the output cannot end, that byte. Each
     * step tries the 256 byte values on copies of the state and stops at the second one allowed.
     */
    public byte[] forcedBytes(int max) {
        if (state.relaxed()) {
            return new byte[0];
        }
        State s = state.copy();
        byte[] out = new byte[16];
        int n = 0;
        while (n < max && !s.canEnd()) {
            State only = null;
            int onlyByte = -1;
            for (int b = 0; b < 256; b++) {
                State t = s.copy();
                if (t.feed(b)) {
                    if (only != null) {
                        only = null;
                        onlyByte = -2;
                        break;
                    }
                    only = t;
                    onlyByte = b;
                }
            }
            if (onlyByte < 0) {
                break;
            }
            if (n == out.length) {
                out = Arrays.copyOf(out, n * 2);
            }
            out[n++] = (byte) onlyByte;
            s = only;
            if (s.relaxed()) {
                break;
            }
        }
        return Arrays.copyOf(out, n);
    }

    /**
     * Jump-forward. The forced text ({@link #forcedBytes}, cut back to whole UTF-8 characters) is encoded with the
     * tokenizer as ordinary text, the way the model would see it in its training data. The tokens are kept while
     * their bytes spell the forced text and the constraint accepts them; the last one is then held back unless no
     * vocabulary token extends it ({@link TokenTrie#extendable}), because at the boundary between forced and free
     * text the model's own tokenization may merge across it (the forced {@code ": "} followed by a value
     * {@code "abc} is better written {@code ": "} + {@code abc} or {@code ":} + {@code  "abc}, which the model picks
     * itself from the constrained choices). An added token such as {@code </tool_call>} is never extended.
     */
    @Override
    public int[] jump(Tokenizer tok, int max) {
        if (max <= 0 || state.relaxed() || state.canEnd()) {
            return new int[0];
        }
        byte[] f = forcedBytes(MAX_JUMP_BYTES);
        int cut = utf8Boundary(f);
        if (cut == 0) {
            return new int[0];
        }
        int[] ids = tok.encode(new String(f, 0, cut, StandardCharsets.UTF_8), false);
        State s = state.copy();
        int off = 0;
        int n = 0;
        for (int id : ids) {
            byte[] b = id >= 0 && id < tokens.length ? tokens[id] : null;
            if (b == null || off + b.length > cut || !Arrays.equals(b, 0, b.length, f, off, off + b.length)) {
                break;
            }
            boolean ok = true;
            for (byte x : b) {
                ok &= s.feed(x & 0xff);
            }
            if (!ok) {
                break;
            }
            off += b.length;
            n++;
        }
        if (n > 0 && TokenTrie.of(tokens).extendable(tokens[ids[n - 1]])) {
            n--;
        }
        return Arrays.copyOf(ids, Math.min(n, max));
    }

    /** The length of the longest prefix of {@code b} that ends on a UTF-8 character boundary. */
    static int utf8Boundary(byte[] b) {
        int n = b.length;
        int i = n - 1;
        while (i >= 0 && (b[i] & 0xC0) == 0x80) {
            i--; // continuation bytes
        }
        if (i < 0) {
            return 0;
        }
        int lead = b[i] & 0xff;
        int len = lead < 0x80 ? 1 : lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : 2;
        return i + len <= n ? n : i;
    }
}
