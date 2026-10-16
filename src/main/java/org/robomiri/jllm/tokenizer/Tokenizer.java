package org.robomiri.jllm.tokenizer;

import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.IntPredicate;
import java.util.function.UnaryOperator;

/**
 * BPE tokenizer that reads a Hugging Face {@code tokenizer.json}. Two flavours of the same model type are
 * supported:
 *
 * <ul>
 *   <li><b>byte-level</b> (GPT-2 / Qwen / Llama 3 / Phi-4 / GLM / DeepSeek / GPT-NeoX): split on added tokens,
 *       normalize, regex pre-tokenize, map bytes to the GPT-2 printable alphabet, apply merges by rank;</li>
 *   <li><b>SentencePiece style</b> (Mistral, Gemma, Phi-3): split on added tokens, normalize and/or Metaspace
 *       (spaces become {@code U+2581}), start from Unicode characters, unknown characters fall back to
 *       {@code <0xXX>} byte tokens, apply merges by rank.</li>
 * </ul>
 *
 * <p>The normalizer, pre-tokenizer, added-token (lstrip / rstrip) and post-processor (BOS) constructs found in
 * the tokenizers of the common model families are interpreted from the JSON; unknown constructs fail loudly
 * rather than silently tokenizing differently. Unigram models are not supported.
 */
public final class Tokenizer {

    /** Qwen2/3 pre-tokenizer, used when tokenizer.json does not carry any pre-tokenizer. */
    private static final String DEFAULT_SPLIT =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private static final int CACHE_LIMIT = 100_000;
    /** Chunks longer than this (in UTF-8 bytes) are never cached, bounding the cache's memory as well as its count. */
    private static final int CACHE_MAX_PIECE_BYTES = 64;

    private final Map<String, Integer> vocab;
    private final byte[][] idToBytes;
    private final String[] idToString;
    /** key = (left id << 32 | right id), value = (rank << 32 | merged id). */
    private final LongLongMap merges;
    private final Map<String, Integer> addedTokens;
    private final Set<Integer> specialIds;
    private final Set<Integer> lstripIds = new HashSet<>();
    private final Set<Integer> rstripIds = new HashSet<>();
    private final AddedTokenTrie addedTrie;
    private static final IntPredicate ANY_TOKEN = id -> true;
    private final UnaryOperator<String> normalizer;
    private final PreTokenizers.Chain preTokenizer;
    private final boolean byteLevel;
    private final boolean byteFallback;
    private final boolean fuseUnk;
    private final int unkId;
    private final int[] byteTokenIds = new int[256];
    private final int stripOnDecode;
    private final int bosId;
    private final int eosId;
    private final boolean addBos;
    private final boolean ignoreMerges;
    private final char[] byteToChar = new char[256];
    private final int[] charToByte = new int[512];
    private final int[] byteCharIds = new int[256];
    private final Map<String, int[]> cache = new ConcurrentHashMap<>();

    private Tokenizer(Map<String, Object> json, Map<String, Object> config) {
        initByteMapping();
        Map<String, Object> model = Json.map(json, "model");
        if (model == null) {
            throw new IllegalArgumentException("tokenizer.json has no \"model\" object");
        }
        if (!"BPE".equals(Json.str(model, "type", "BPE"))) {
            throw new IllegalArgumentException("only BPE tokenizers are supported, got model.type="
                    + Json.str(model, "type", null));
        }
        ignoreMerges = Json.bool(model, "ignore_merges", false);
        byteFallback = Json.bool(model, "byte_fallback", false);
        fuseUnk = Json.bool(model, "fuse_unk", false);

        Map<String, Object> v = Json.map(model, "vocab");
        if (v == null) {
            throw new IllegalArgumentException("tokenizer.json model has no \"vocab\" object");
        }
        vocab = new HashMap<>(v.size() * 2);
        int maxId = -1;
        for (Map.Entry<String, Object> e : v.entrySet()) {
            if (!(e.getValue() instanceof Number n) || n.longValue() < 0 || n.longValue() > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("vocab entry " + Json.write(e.getKey()) + " has invalid id "
                        + e.getValue());
            }
            int id = n.intValue();
            vocab.put(e.getKey(), id);
            maxId = Math.max(maxId, id);
        }

        List<Object> mergeList = Json.list(model, "merges");
        merges = new LongLongMap(mergeList == null ? 16 : mergeList.size());
        if (mergeList != null) {
            int rank = 0;
            for (Object m : mergeList) {
                String a;
                String b;
                if (m instanceof String s) { // legacy "a b" format
                    int sp = s.indexOf(' ');
                    if (sp <= 0 || sp == s.length() - 1) {
                        throw new IllegalArgumentException("merge #" + rank + " " + Json.write(s)
                                + " is not of the form \"left right\"");
                    }
                    a = s.substring(0, sp);
                    b = s.substring(sp + 1);
                } else if (m instanceof List<?> pair && pair.size() == 2
                        && pair.get(0) instanceof String l && pair.get(1) instanceof String r) {
                    a = l;
                    b = r;
                } else {
                    throw new IllegalArgumentException("merge #" + rank + " " + Json.write(m)
                            + " is neither \"left right\" nor [\"left\", \"right\"]");
                }
                Integer ia = vocab.get(a);
                Integer ib = vocab.get(b);
                Integer merged = vocab.get(a + b);
                if (ia != null && ib != null && merged != null) {
                    merges.putIfAbsent(pairKey(ia, ib), ((long) rank << 32) | merged);
                }
                rank++;
            }
        }

        addedTokens = new HashMap<>();
        specialIds = new HashSet<>();
        List<Object> added = Json.list(json, "added_tokens");
        if (added != null) {
            for (Object o : added) {
                int id = (int) Json.num(o, "id", -1);
                String content = Json.str(o, "content", null);
                if (id < 0 || content == null) {
                    continue;
                }
                addedTokens.put(content, id);
                if (Json.bool(o, "special", false)) {
                    specialIds.add(id);
                }
                if (Json.bool(o, "lstrip", false)) {
                    lstripIds.add(id);
                }
                if (Json.bool(o, "rstrip", false)) {
                    rstripIds.add(id);
                }
                maxId = Math.max(maxId, id);
            }
        }

        // ---- pipeline stages
        Map<String, Object> normalizerNode = Json.map(json, "normalizer");
        Map<String, Object> preNode = Json.map(json, "pre_tokenizer");
        Map<String, Object> decoderNode = Json.map(json, "decoder");
        normalizer = Normalizers.build(normalizerNode);
        PreTokenizers.Chain pre = PreTokenizers.build(preNode);
        boolean spSignal = pre.metaspace || byteFallback || containsType(decoderNode, "Replace")
                || containsType(decoderNode, "Metaspace") || containsType(decoderNode, "ByteFallback")
                || containsType(normalizerNode, "Replace") || containsType(normalizerNode, "Prepend");
        byteLevel = pre.byteLevel || !spSignal; // tokenizers with no hints at all keep the historical byte-level default
        if (byteLevel && !pre.byteLevel && !pre.splitRegex) { // legacy default: Qwen's regex
            pre = PreTokenizers.build(Json.obj("type", "Split", "pattern", Json.obj("Regex", DEFAULT_SPLIT),
                    "behavior", "Isolated"));
        }
        preTokenizer = pre;
        stripOnDecode = decodeStrip(decoderNode);

        // ---- id tables
        idToBytes = new byte[maxId + 1][];
        idToString = new String[maxId + 1];
        for (Map.Entry<String, Integer> e : vocab.entrySet()) {
            idToString[e.getValue()] = e.getKey();
            idToBytes[e.getValue()] = byteLevel ? decodeByteLevel(e.getKey()) : decodeSentencePiece(e.getKey());
        }
        for (Map.Entry<String, Integer> e : addedTokens.entrySet()) {
            idToString[e.getValue()] = e.getKey();
            idToBytes[e.getValue()] = byteLevel ? e.getKey().getBytes(StandardCharsets.UTF_8)
                    : decodeSentencePiece(e.getKey());
        }
        if (byteLevel) {
            // Some vocabularies (GPT-NeoX) lack the symbols of bytes that never occur in UTF-8 (0xC0, 0xF5..);
            // like the reference, such bytes are dropped. A vocabulary with most of them missing is not byte-level.
            int missing = -1;
            int present = 0;
            for (int b = 0; b < 256; b++) {
                Integer id = vocab.get(String.valueOf(byteToChar[b]));
                byteCharIds[b] = id == null ? -1 : id;
                if (id == null) {
                    missing = b;
                } else {
                    present++;
                }
            }
            if (present < 128) {
                throw new IllegalArgumentException("vocab is missing byte-level symbol for byte " + missing);
            }
        }
        for (int b = 0; b < 256; b++) {
            Integer id = vocab.get(String.format("<0x%02X>", b));
            byteTokenIds[b] = id == null ? -1 : id;
        }

        String unkToken = Json.str(model, "unk_token", null);
        Integer unk = unkToken == null ? null : tokenIdOrNull(unkToken);
        unkId = unk == null ? -1 : unk;

        // Leftmost-longest matching, so e.g. "<|im_start|>" wins over any shorter overlapping token.
        addedTrie = AddedTokenTrie.build(addedTokens);

        // ---- BOS / EOS: tokenizer_config.json (when available) wins over the post-processor template
        int[] template = findTemplateBos(Json.map(json, "post_processor"));
        int tplBos = template[0];
        Integer cfgBos = config == null ? null : tokenIdOrNull(specialName(config.get("bos_token")));
        Integer cfgEos = config == null ? null : tokenIdOrNull(specialName(config.get("eos_token")));
        bosId = cfgBos != null ? cfgBos : tplBos;
        eosId = cfgEos != null ? cfgEos : -1;
        boolean cfgAddBos = config != null && config.get("add_bos_token") instanceof Boolean b
                ? b : tplBos >= 0;
        addBos = cfgAddBos && bosId >= 0;
    }

    public static Tokenizer load(Path tokenizerJson) throws IOException {
        Path cfg = tokenizerJson.resolveSibling("tokenizer_config.json");
        return fromJson(Files.readString(tokenizerJson), Files.isRegularFile(cfg) ? Files.readString(cfg) : null);
    }

    /** @throws IllegalArgumentException if the text is not a well-formed BPE tokenizer.json */
    public static Tokenizer fromJson(String tokenizerJson) {
        return fromJson(tokenizerJson, null);
    }

    /**
     * @param tokenizerConfigJson the sibling {@code tokenizer_config.json} (may be null); supplies bos/eos tokens
     *                            and {@code add_bos_token}
     */
    public static Tokenizer fromJson(String tokenizerJson, String tokenizerConfigJson) {
        Map<String, Object> config = null;
        if (tokenizerConfigJson != null) {
            try {
                config = Json.parseObject(tokenizerConfigJson);
            } catch (RuntimeException e) {
                config = null; // a broken tokenizer_config.json must not stop the tokenizer from loading
            }
        }
        return new Tokenizer(Json.parseObject(tokenizerJson), config);
    }

    // ------------------------------------------------------------------ public API

    /**
     * Encodes text; added/special tokens appearing literally in the text are mapped to their ids. No BOS is
     * added; see {@link #addBosByDefault()} and {@link #bosId()}.
     */
    public int[] encode(String text) {
        return encode(text, true);
    }

    /**
     * @param allowSpecial when false, added/special tokens written literally in the text are tokenized as
     *                     ordinary characters (use this for untrusted content such as chat messages)
     */
    public int[] encode(String text, boolean allowSpecial) {
        SequenceEncoder e = new SequenceEncoder();
        if (allowSpecial) {
            e.special(text);
        } else {
            e.text(text);
        }
        return e.toArray();
    }

    /**
     * Incremental encoder for text assembled from trusted control strings and untrusted content. The result is
     * exactly that of encoding the concatenation in one go while treating only the control strings as able to
     * contain special tokens: ordinary text between two special tokens is tokenized as one run (so
     * pre-tokenization and merges never see an artificial boundary between a control string and a message), a
     * Metaspace prefix is applied only at offset 0, and the whitespace swallowed by {@code lstrip} / {@code rstrip}
     * tokens (Phi's {@code <|user|>}) is swallowed across pieces too.
     */
    public SequenceEncoder newSequence() {
        return new SequenceEncoder();
    }

    public final class SequenceEncoder {
        private final IntList out = new IntList(64);
        private final StringBuilder run = new StringBuilder();
        private boolean consumed; // anything (text or token) precedes the current run
        private boolean runAtStart;
        private boolean pendingRstrip;

        private SequenceEncoder() {
        }

        /** Appends a control string; added tokens in it become their ids. */
        public SequenceEncoder special(String text) {
            return scan(text, ANY_TOKEN);
        }

        /** Appends ordinary text; added tokens in it are NOT recognized. */
        public SequenceEncoder text(String text) {
            appendRun(text);
            return this;
        }

        /** Leftmost-longest scan for the added tokens accepted by {@code allowed}; the rest is ordinary text. */
        private SequenceEncoder scan(String text, IntPredicate allowed) {
            if (addedTrie == null) {
                return text(text);
            }
            int len = text.length();
            int last = 0; // start of the plain text not yet appended
            int i = 0;
            while (i < len) {
                if (!addedTrie.mayStart(text.charAt(i))) {
                    i++;
                    continue;
                }
                long m = addedTrie.longest(text, i, allowed);
                if (m < 0) {
                    i++;
                    continue;
                }
                int id = (int) m;
                appendRun(text.substring(last, i));
                if (lstripIds.contains(id)) {
                    int n = run.length();
                    while (n > 0 && isWhitespace(run.charAt(n - 1))) {
                        n--;
                    }
                    run.setLength(n);
                }
                flushRun();
                out.add(id);
                consumed = true;
                int end = i + (int) (m >>> 32);
                boolean rs = rstripIds.contains(id);
                if (rs) {
                    while (end < len && isWhitespace(text.charAt(end))) {
                        end++;
                    }
                }
                pendingRstrip = rs && end == len;
                last = end;
                i = end;
            }
            if (last < len) {
                appendRun(text.substring(last));
            }
            return this;
        }

        public int[] toArray() {
            flushRun();
            return out.toArray();
        }

        private void appendRun(String t) {
            if (pendingRstrip) {
                int i = 0;
                while (i < t.length() && isWhitespace(t.charAt(i))) {
                    i++;
                }
                t = t.substring(i);
            }
            if (t.isEmpty()) {
                return;
            }
            pendingRstrip = false;
            if (run.length() == 0) {
                runAtStart = !consumed;
            }
            run.append(t);
        }

        private void flushRun() {
            if (run.length() > 0) {
                encodeOrdinary(run.toString(), out, runAtStart);
                run.setLength(0);
                consumed = true;
            }
        }
    }

    /** Raw bytes of a token (UTF-8 fragments; a multi-byte character may span several tokens). */
    public byte[] tokenBytes(int id) {
        if (id < 0 || id >= idToBytes.length || idToBytes[id] == null) {
            return new byte[0];
        }
        return idToBytes[id];
    }

    /**
     * Concatenates the token bytes and decodes them as UTF-8. Special tokens are written out literally. For
     * SentencePiece-style tokenizers whose decoder strips the leading space (Mistral, Phi-3) the first space
     * is dropped, like the reference implementation.
     */
    public String decode(int[] ids) {
        java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
        for (int id : ids) {
            bos.writeBytes(tokenBytes(id));
        }
        byte[] all = bos.toByteArray();
        int skip = 0;
        while (skip < stripOnDecode && skip < all.length && all[skip] == ' ') {
            skip++;
        }
        return new String(all, skip, all.length - skip, StandardCharsets.UTF_8);
    }

    /** Number of leading spaces the reference decoder removes from the decoded text (0 or 1 in practice). */
    public int decodeStripLeading() {
        return stripOnDecode;
    }

    /** The token's surface form in the vocabulary (byte-level alphabet), for debugging. */
    public String tokenString(int id) {
        return id >= 0 && id < idToString.length ? idToString[id] : null;
    }

    /** Id of a vocabulary or added token, or -1. */
    public int tokenId(String token) {
        Integer id = tokenIdOrNull(token);
        return id == null ? -1 : id;
    }

    public boolean isSpecial(int id) {
        return specialIds.contains(id);
    }

    /** One past the largest token id known to the tokenizer (may be smaller than the model's embedding rows). */
    public int vocabSize() {
        return idToBytes.length;
    }

    /** True for GPT-2-style byte-level BPE, false for SentencePiece-style (Metaspace / byte fallback) BPE. */
    public boolean isByteLevel() {
        return byteLevel;
    }

    /** Id of the beginning-of-sequence token, or -1 if the tokenizer defines none. */
    public int bosId() {
        return bosId;
    }

    /** Id of the end-of-sequence token named by tokenizer_config.json, or -1. */
    public int eosId() {
        return eosId;
    }

    /** Id of the unknown token, or -1. */
    public int unkId() {
        return unkId;
    }

    /**
     * Whether the tokenizer's post-processor / config asks for {@link #bosId()} in front of a plain prompt.
     * {@code encode} never adds it; the caller decides.
     */
    public boolean addBosByDefault() {
        return addBos;
    }

    // ------------------------------------------------------------------ internals

    private Integer tokenIdOrNull(String token) {
        if (token == null) {
            return null;
        }
        Integer id = addedTokens.get(token);
        return id != null ? id : vocab.get(token);
    }

    /** Rust's {@code char::is_whitespace}, which decides what lstrip / rstrip tokens swallow. */
    private static boolean isWhitespace(char c) {
        if (c >= 0x1C && c <= 0x1F) {
            return false;
        }
        return Character.isWhitespace(c) || Character.isSpaceChar(c) || c == 0x85;
    }

    private void encodeOrdinary(String text, IntList out, boolean first) {
        if (normalizer != null) {
            text = normalizer.apply(text);
        }
        for (String piece : preTokenizer.apply(text, first)) {
            bpe(piece, out);
        }
    }

    private void bpe(String piece, IntList out) {
        int[] cached = cache.get(piece);
        if (cached == null) {
            cached = byteLevel ? bpeUncached(piece) : bpeCharsUncached(piece);
            // Only short chunks are worth remembering (words, punctuation, short runs); long runs would let a
            // single request fill the cache with megabytes per entry.
            if (piece.length() <= CACHE_MAX_PIECE_BYTES && cache.size() < CACHE_LIMIT
                    && piece.getBytes(StandardCharsets.UTF_8).length <= CACHE_MAX_PIECE_BYTES) {
                cache.put(piece, cached);
            }
        }
        out.addAll(cached);
    }

    private int[] bpeUncached(String piece) {
        byte[] bytes = piece.getBytes(StandardCharsets.UTF_8);
        if (ignoreMerges) {
            Integer whole = vocab.get(encodeByteLevel(bytes));
            if (whole != null) {
                return new int[]{whole};
            }
        }
        int n = bytes.length;
        int[] sym = new int[n];
        int k = 0;
        for (int i = 0; i < n; i++) {
            int id = byteCharIds[bytes[i] & 0xFF];
            if (id >= 0) {
                sym[k++] = id;
            }
        }
        return mergeSymbols(sym, k);
    }

    /**
     * SentencePiece-style initial symbols: one per Unicode character; characters missing from the vocabulary
     * become their {@code <0xXX>} byte tokens (byte_fallback) or the unknown token (fused when fuse_unk),
     * or are dropped when neither exists.
     */
    private int[] bpeCharsUncached(String piece) {
        if (ignoreMerges) {
            Integer whole = vocab.get(piece);
            if (whole != null) {
                return new int[]{whole};
            }
        }
        IntList syms = new IntList(piece.length() + 1);
        boolean lastUnk = false;
        for (int i = 0; i < piece.length(); ) {
            int cp = piece.codePointAt(i);
            int cl = Character.charCount(cp);
            Integer id = vocab.get(piece.substring(i, i + cl));
            i += cl;
            if (id != null) {
                syms.add(id);
                lastUnk = false;
                continue;
            }
            if (byteFallback) {
                byte[] bs = new String(Character.toChars(cp)).getBytes(StandardCharsets.UTF_8);
                boolean all = true;
                for (byte b : bs) {
                    all &= byteTokenIds[b & 0xFF] >= 0;
                }
                if (all) {
                    for (byte b : bs) {
                        syms.add(byteTokenIds[b & 0xFF]);
                    }
                    lastUnk = false;
                    continue;
                }
            }
            if (unkId >= 0 && !(fuseUnk && lastUnk)) {
                syms.add(unkId);
            }
            lastUnk = unkId >= 0;
        }
        int[] sym = syms.toArray();
        return mergeSymbols(sym, sym.length);
    }

    private int[] mergeSymbols(int[] sym, int n) {
        if (n < 2) {
            return n == sym.length ? sym : Arrays.copyOf(sym, n);
        }
        // Symbols form a doubly linked list over their original positions; a min-heap of (rank, position)
        // yields the lowest-ranked pair, leftmost first, which is exactly the order a full rescan would pick.
        // Entries go stale when either side of their pair has been merged away; they are verified on pop by
        // re-reading the pair at that position (ranks are unique per pair, so an equal rank means the same pair).
        int[] next = new int[n];
        int[] prev = new int[n];
        LongHeap heap = new LongHeap(2 * n);
        for (int i = 0; i < n; i++) {
            next[i] = i + 1 < n ? i + 1 : -1;
            prev[i] = i - 1;
            if (i + 1 < n) {
                offer(heap, sym, i, i + 1);
            }
        }
        int alive = n;
        while (!heap.isEmpty()) {
            long e = heap.pop();
            int i = (int) e;
            int j = next[i];
            if (sym[i] < 0 || j < 0) {
                continue;
            }
            long v = merges.get(pairKey(sym[i], sym[j]));
            if (v == LongLongMap.MISSING || (v >>> 32) != (e >>> 32)) {
                continue;
            }
            sym[i] = (int) v;
            sym[j] = -1;
            int k = next[j];
            next[i] = k;
            if (k >= 0) {
                prev[k] = i;
                offer(heap, sym, i, k);
            }
            if (prev[i] >= 0) {
                offer(heap, sym, prev[i], i);
            }
            alive--;
        }
        int[] out = new int[alive];
        int o = 0;
        for (int i = 0; i >= 0; i = next[i]) {
            out[o++] = sym[i];
        }
        return out;
    }

    private void offer(LongHeap heap, int[] sym, int left, int right) {
        long v = merges.get(pairKey(sym[left], sym[right]));
        if (v != LongLongMap.MISSING) {
            heap.push((v & 0xFFFFFFFF00000000L) | left);
        }
    }

    private static long pairKey(int a, int b) {
        return ((long) a << 32) | (b & 0xFFFFFFFFL);
    }

    /** GPT-2 byte-to-unicode table: printable bytes map to themselves, the rest to U+0100 and up. */
    private void initByteMapping() {
        Arrays.fill(charToByte, -1);
        boolean[] direct = new boolean[256];
        for (int b = '!'; b <= '~'; b++) {
            direct[b] = true;
        }
        for (int b = 0xA1; b <= 0xAC; b++) {
            direct[b] = true;
        }
        for (int b = 0xAE; b <= 0xFF; b++) {
            direct[b] = true;
        }
        int extra = 0;
        for (int b = 0; b < 256; b++) {
            char c = direct[b] ? (char) b : (char) (256 + extra++);
            byteToChar[b] = c;
            charToByte[c] = b;
        }
    }

    private String encodeByteLevel(byte[] bytes) {
        char[] cs = new char[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            cs[i] = byteToChar[bytes[i] & 0xFF];
        }
        return new String(cs);
    }

    private byte[] decodeByteLevel(String token) {
        byte[] out = new byte[token.length()];
        int n = 0;
        for (int i = 0; i < token.length(); i++) {
            char c = token.charAt(i);
            int b = c < charToByte.length ? charToByte[c] : -1;
            if (b >= 0) {
                out[n++] = (byte) b;
            } else { // not a byte-level symbol: keep its UTF-8 form
                byte[] u = String.valueOf(c).getBytes(StandardCharsets.UTF_8);
                out = Arrays.copyOf(out, out.length + u.length);
                System.arraycopy(u, 0, out, n, u.length);
                n += u.length;
            }
        }
        return n == out.length ? out : Arrays.copyOf(out, n);
    }

    /** SentencePiece decoding of one token: {@code <0xXX>} is one raw byte, U+2581 is a space. */
    private byte[] decodeSentencePiece(String token) {
        if (byteFallback && token.length() == 6 && token.startsWith("<0x") && token.endsWith(">")) {
            int hi = Character.digit(token.charAt(3), 16);
            int lo = Character.digit(token.charAt(4), 16);
            if (hi >= 0 && lo >= 0) {
                return new byte[]{(byte) (hi << 4 | lo)};
            }
        }
        return token.replace('▁', ' ').getBytes(StandardCharsets.UTF_8);
    }

    private static boolean containsType(Map<String, Object> node, String type) {
        if (node == null) {
            return false;
        }
        if (type.equals(Json.str(node, "type", null))) {
            return true;
        }
        for (String key : new String[]{"normalizers", "pretokenizers", "decoders", "processors"}) {
            List<Object> children = Json.list(node, key);
            if (children != null) {
                for (Object c : children) {
                    if (c instanceof Map<?, ?> && containsType(asMap(c), type)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /** How many leading spaces the decoder removes: a {@code Strip} decoder with start>0, or Metaspace with a prefix. */
    private static int decodeStrip(Map<String, Object> node) {
        if (node == null) {
            return 0;
        }
        String type = Json.str(node, "type", "");
        int n = 0;
        if (type.equals("Strip") && " ".equals(Json.str(node, "content", " "))) {
            n = (int) Json.num(node, "start", 0);
        } else if (type.equals("Metaspace")) {
            String scheme = Json.str(node, "prepend_scheme", null);
            boolean prefix = scheme != null ? !scheme.equals("never") : Json.bool(node, "add_prefix_space", true);
            n = prefix ? 1 : 0;
        }
        List<Object> children = Json.list(node, "decoders");
        if (children != null) {
            for (Object c : children) {
                n = Math.max(n, decodeStrip(asMap(c)));
            }
        }
        return n;
    }

    /** @return {bosId or -1} from a TemplateProcessing {@code single} template that starts with a special token */
    private int[] findTemplateBos(Map<String, Object> pp) {
        int[] none = {-1};
        if (pp == null) {
            return none;
        }
        String type = Json.str(pp, "type", "");
        if (type.equals("Sequence")) {
            List<Object> procs = Json.list(pp, "processors");
            if (procs != null) {
                for (Object p : procs) {
                    int[] r = findTemplateBos(asMap(p));
                    if (r[0] >= 0) {
                        return r;
                    }
                }
            }
        } else if (type.equals("TemplateProcessing")) {
            List<Object> single = Json.list(pp, "single");
            if (single != null && !single.isEmpty() && single.get(0) instanceof Map<?, ?> first) {
                Map<String, Object> st = Json.map(first, "SpecialToken");
                String name = Json.str(st, "id", null);
                if (name != null) {
                    Map<String, Object> reg = Json.map(Json.map(pp, "special_tokens"), name);
                    List<Object> ids = Json.list(reg, "ids");
                    if (ids != null && !ids.isEmpty() && ids.get(0) instanceof Number n) {
                        return new int[]{n.intValue()};
                    }
                    int id = tokenId(name);
                    if (id >= 0) {
                        return new int[]{id};
                    }
                }
            }
        }
        return none;
    }

    /** A tokenizer_config special token is either a string or an object with a "content" field. */
    private static String specialName(Object o) {
        if (o instanceof String s) {
            return s;
        }
        return o instanceof Map<?, ?> ? Json.str(o, "content", null) : null;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> asMap(Object o) {
        return o instanceof Map ? (Map<String, Object>) o : null;
    }

    /** Binary min-heap of longs without boxing; used for (rank << 32 | position) merge candidates. */
    static final class LongHeap {
        private long[] a;
        private int n;

        LongHeap(int cap) {
            a = new long[Math.max(cap, 4)];
        }

        boolean isEmpty() {
            return n == 0;
        }

        void push(long v) {
            if (n == a.length) {
                a = Arrays.copyOf(a, a.length * 2);
            }
            int i = n++;
            while (i > 0) {
                int parent = (i - 1) >>> 1;
                if (a[parent] <= v) {
                    break;
                }
                a[i] = a[parent];
                i = parent;
            }
            a[i] = v;
        }

        long pop() {
            long top = a[0];
            long v = a[--n];
            int i = 0;
            while (true) {
                int l = 2 * i + 1;
                if (l >= n) {
                    break;
                }
                int r = l + 1;
                int m = r < n && a[r] < a[l] ? r : l;
                if (a[m] >= v) {
                    break;
                }
                a[i] = a[m];
                i = m;
            }
            a[i] = v;
            return top;
        }
    }

    /** Growable int array without boxing. */
    static final class IntList {
        private int[] a;
        private int n;

        IntList(int cap) {
            a = new int[Math.max(cap, 4)];
        }

        void add(int v) {
            if (n == a.length) {
                a = Arrays.copyOf(a, a.length * 2);
            }
            a[n++] = v;
        }

        void addAll(int[] vs) {
            if (n + vs.length > a.length) {
                a = Arrays.copyOf(a, Math.max(a.length * 2, n + vs.length));
            }
            System.arraycopy(vs, 0, a, n, vs.length);
            n += vs.length;
        }

        int[] toArray() {
            return Arrays.copyOf(a, n);
        }
    }
}
