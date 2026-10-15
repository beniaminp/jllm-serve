package org.robomiri.jllm.tokenizer;

import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Byte-level BPE tokenizer that reads a Hugging Face {@code tokenizer.json} (GPT-2 / Qwen style).
 *
 * <p>Pipeline: split on added tokens, NFC-normalize, regex pre-tokenize, map bytes to the GPT-2 printable
 * alphabet, apply BPE merges by rank.
 */
public final class Tokenizer {

    /** Qwen2/3 pre-tokenizer, used when tokenizer.json does not carry a Split regex. */
    private static final String DEFAULT_SPLIT =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private static final int CACHE_LIMIT = 100_000;

    private final Map<String, Integer> vocab;
    private final byte[][] idToBytes;
    private final String[] idToString;
    /** key = (left id << 32 | right id), value = (rank << 32 | merged id). */
    private final Map<Long, Long> merges;
    private final Map<String, Integer> addedTokens;
    private final Set<Integer> specialIds;
    private final Pattern addedPattern;
    private final Pattern splitPattern;
    private final boolean nfc;
    private final boolean ignoreMerges;
    private final char[] byteToChar = new char[256];
    private final int[] charToByte = new int[512];
    private final int[] byteCharIds = new int[256];
    private final Map<String, int[]> cache = new ConcurrentHashMap<>();

    private Tokenizer(Map<String, Object> json) {
        initByteMapping();
        Map<String, Object> model = Json.map(json, "model");
        if (model == null || !"BPE".equals(Json.str(model, "type", "BPE"))) {
            throw new IllegalArgumentException("only BPE tokenizers are supported");
        }
        ignoreMerges = Json.bool(model, "ignore_merges", false);

        Map<String, Object> v = Json.map(model, "vocab");
        vocab = new HashMap<>(v.size() * 2);
        int maxId = -1;
        for (Map.Entry<String, Object> e : v.entrySet()) {
            int id = ((Number) e.getValue()).intValue();
            vocab.put(e.getKey(), id);
            maxId = Math.max(maxId, id);
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
                maxId = Math.max(maxId, id);
            }
        }

        idToBytes = new byte[maxId + 1][];
        idToString = new String[maxId + 1];
        for (Map.Entry<String, Integer> e : vocab.entrySet()) {
            idToString[e.getValue()] = e.getKey();
            idToBytes[e.getValue()] = decodeByteLevel(e.getKey());
        }
        for (Map.Entry<String, Integer> e : addedTokens.entrySet()) {
            idToString[e.getValue()] = e.getKey();
            idToBytes[e.getValue()] = e.getKey().getBytes(StandardCharsets.UTF_8);
        }
        for (int b = 0; b < 256; b++) {
            Integer id = vocab.get(String.valueOf(byteToChar[b]));
            if (id == null) {
                throw new IllegalArgumentException("vocab is missing byte-level symbol for byte " + b);
            }
            byteCharIds[b] = id;
        }

        List<Object> mergeList = Json.list(model, "merges");
        merges = new HashMap<>(mergeList == null ? 16 : mergeList.size() * 2);
        if (mergeList != null) {
            int rank = 0;
            for (Object m : mergeList) {
                String a;
                String b;
                if (m instanceof String s) { // legacy "a b" format
                    int sp = s.indexOf(' ');
                    a = s.substring(0, sp);
                    b = s.substring(sp + 1);
                } else {
                    List<?> pair = (List<?>) m;
                    a = (String) pair.get(0);
                    b = (String) pair.get(1);
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

        // Longest-first alternation so e.g. "<|im_start|>" wins over any shorter overlapping token.
        addedPattern = addedTokens.isEmpty() ? null : Pattern.compile(addedTokens.keySet().stream()
                .sorted(Comparator.comparingInt(String::length).reversed())
                .map(Pattern::quote)
                .collect(Collectors.joining("|")));

        Map<String, Object> normalizer = Json.map(json, "normalizer");
        nfc = normalizer != null && containsType(normalizer, "NFC");
        String regex = findSplitRegex(Json.map(json, "pre_tokenizer"));
        splitPattern = Pattern.compile(regex == null ? DEFAULT_SPLIT : regex, Pattern.UNICODE_CHARACTER_CLASS);
    }

    public static Tokenizer load(Path tokenizerJson) throws IOException {
        return new Tokenizer(Json.parseObject(Files.readString(tokenizerJson)));
    }

    // ------------------------------------------------------------------ public API

    /** Encodes text; added/special tokens appearing literally in the text are mapped to their ids. */
    public int[] encode(String text) {
        IntList out = new IntList(Math.max(16, text.length() / 3));
        if (addedPattern == null) {
            encodeOrdinary(text, out);
            return out.toArray();
        }
        Matcher m = addedPattern.matcher(text);
        int last = 0;
        while (m.find()) {
            if (m.start() > last) {
                encodeOrdinary(text.substring(last, m.start()), out);
            }
            out.add(addedTokens.get(m.group()));
            last = m.end();
        }
        if (last < text.length()) {
            encodeOrdinary(text.substring(last), out);
        }
        return out.toArray();
    }

    /** Raw bytes of a token (UTF-8 fragments; a multi-byte character may span several tokens). */
    public byte[] tokenBytes(int id) {
        if (id < 0 || id >= idToBytes.length || idToBytes[id] == null) {
            return new byte[0];
        }
        return idToBytes[id];
    }

    public String decode(int[] ids) {
        java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
        for (int id : ids) {
            bos.writeBytes(tokenBytes(id));
        }
        return bos.toString(StandardCharsets.UTF_8);
    }

    /** The token's surface form in the vocabulary (byte-level alphabet), for debugging. */
    public String tokenString(int id) {
        return id >= 0 && id < idToString.length ? idToString[id] : null;
    }

    /** Id of a vocabulary or added token, or -1. */
    public int tokenId(String token) {
        Integer id = addedTokens.get(token);
        if (id == null) {
            id = vocab.get(token);
        }
        return id == null ? -1 : id;
    }

    public boolean isSpecial(int id) {
        return specialIds.contains(id);
    }

    /** One past the largest token id known to the tokenizer (may be smaller than the model's embedding rows). */
    public int vocabSize() {
        return idToBytes.length;
    }

    // ------------------------------------------------------------------ internals

    private void encodeOrdinary(String text, IntList out) {
        if (nfc) {
            text = Normalizer.normalize(text, Normalizer.Form.NFC);
        }
        Matcher m = splitPattern.matcher(text);
        int last = 0;
        while (m.find()) {
            if (m.start() > last) { // "Isolated" split behaviour keeps unmatched gaps as pieces
                bpe(text.substring(last, m.start()), out);
            }
            if (m.end() > m.start()) {
                bpe(m.group(), out);
            }
            last = m.end();
        }
        if (last < text.length()) {
            bpe(text.substring(last), out);
        }
    }

    private void bpe(String piece, IntList out) {
        int[] cached = cache.get(piece);
        if (cached == null) {
            cached = bpeUncached(piece);
            if (cache.size() < CACHE_LIMIT) {
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
        int[] sym = new int[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            sym[i] = byteCharIds[bytes[i] & 0xFF];
        }
        int n = sym.length;
        while (n > 1) {
            long bestRank = Long.MAX_VALUE;
            int bestPos = -1;
            int bestId = -1;
            for (int i = 0; i + 1 < n; i++) {
                Long v = merges.get(pairKey(sym[i], sym[i + 1]));
                if (v != null && (v >>> 32) < bestRank) {
                    bestRank = v >>> 32;
                    bestPos = i;
                    bestId = (int) (long) v;
                }
            }
            if (bestPos < 0) {
                break;
            }
            sym[bestPos] = bestId;
            System.arraycopy(sym, bestPos + 2, sym, bestPos + 1, n - bestPos - 2);
            n--;
        }
        return Arrays.copyOf(sym, n);
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

    private static boolean containsType(Map<String, Object> node, String type) {
        if (type.equals(Json.str(node, "type", null))) {
            return true;
        }
        List<Object> children = Json.list(node, "normalizers");
        if (children != null) {
            for (Object c : children) {
                if (c instanceof Map<?, ?> && containsType(asMap(c), type)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static String findSplitRegex(Map<String, Object> pre) {
        if (pre == null) {
            return null;
        }
        if ("Split".equals(Json.str(pre, "type", null))) {
            Map<String, Object> pattern = Json.map(pre, "pattern");
            return pattern == null ? null : Json.str(pattern, "Regex", null);
        }
        List<Object> children = Json.list(pre, "pretokenizers");
        if (children != null) {
            for (Object c : children) {
                String r = findSplitRegex(asMap(c));
                if (r != null) {
                    return r;
                }
            }
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object o) {
        return o instanceof Map ? (Map<String, Object>) o : null;
    }

    /** Growable int array without boxing. */
    static final class IntList {
        private int[] a;
        private int n;

        IntList(int cap) {
            a = new int[cap];
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
