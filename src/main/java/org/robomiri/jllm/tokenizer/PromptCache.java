package org.robomiri.jllm.tokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The token ids of recently rendered chat prompts, so that a prompt extending one rendered before (the next request of
 * a conversation, which resends everything) is tokenized only from where it is new. The result is always exactly the
 * ids of tokenizing the whole prompt (docs/17 §17.5).
 *
 * <p>A chat template renders a conversation as a list of pieces, control strings and message texts ({@link
 * ChatTemplate}), which the {@link Tokenizer.SequenceEncoder} turns into ids. Pieces are not tokenized separately
 * (the text between two special tokens is one run, whatever pieces it came from), but right after an added token the
 * encoding of what follows does not depend on what came before: the encoder reports these {@link
 * Tokenizer.SequenceEncoder.Boundary boundaries}. For every piece boundary {@code k} of a prompt, the cache stores
 * under a 128-bit hash of the pieces {@code [0, k)} (texts and kinds) the last such boundary inside them: its piece,
 * its offset, and how many ids precede it. A new prompt looks up its own piece prefixes from the longest down; on a
 * hit it copies the stored ids and encodes from the boundary on.
 *
 * <p>Bounded by bytes (least recently used prompts go first) and thread-safe; encoding runs outside the lock.
 */
public final class PromptCache {

    /** A rendered prompt: all its ids, and the hashes under which its piece prefixes are found. */
    private static final class Entry {
        final int[] ids;
        final List<Key> keys = new ArrayList<>();
        int live; // keys that still point here
        long bytes;

        Entry(int[] ids) {
            this.ids = ids;
        }
    }

    private record Key(long a, long b) {
    }

    /** Where to resume: {@code ids} ids of {@code entry} precede offset {@code offset} of piece {@code piece}. */
    private record Ref(Entry entry, int piece, int offset, int ids) {
    }

    /** Counters, for logs and tests. */
    public record Stats(long requests, long hits, long reusedTokens, long encodedTokens, int entries, long bytes) {
    }

    private static final long KEY_BYTES = 96; // a map node, the key and the reference
    private static final long ENTRY_BYTES = 64;

    private final Tokenizer tok;
    private final long maxBytes;
    private final Map<Key, Ref> refs = new HashMap<>();
    private final LinkedHashMap<Entry, Entry> lru = new LinkedHashMap<>(16, 0.75f, true);
    private long bytes;
    private long requests;
    private long hits;
    private long reused;
    private long encoded;

    /** @param maxBytes memory for the cached ids and their index; prompts larger than a quarter of it are not kept */
    public PromptCache(Tokenizer tok, long maxBytes) {
        if (maxBytes <= 0) {
            throw new IllegalArgumentException("maxBytes must be positive");
        }
        this.tok = tok;
        this.maxBytes = maxBytes;
    }

    public Tokenizer tokenizer() {
        return tok;
    }

    /**
     * The ids of the pieces, as {@code tok.newSequence(protectedTokens)} gives them when fed {@code texts[i]} with
     * {@code special} or {@code text} according to {@code control[i]}.
     *
     * @param scope what else decides the ids besides the pieces (the template family): prompts of different scopes
     *              never share ids
     */
    public int[] encode(String scope, List<String> texts, List<Boolean> control, Set<String> protectedTokens) {
        int n = texts.size();
        Key[] keys = keys(scope, protectedTokens, texts, control);
        Ref hit = null;
        int hitK = 0; // pieces [0, hitK) are known
        int[] prefix = null;
        synchronized (this) {
            requests++;
            for (int k = n; k >= 1; k--) {
                Ref r = refs.get(keys[k - 1]);
                if (r != null) {
                    hit = r;
                    hitK = k;
                    lru.get(r.entry); // most recently used
                    prefix = Arrays.copyOf(r.entry.ids, r.ids);
                    break;
                }
            }
        }
        // the last boundary seen: piece, offset, ids before it
        int[] last = hit == null ? new int[]{-1, 0, 0} : new int[]{hit.piece, hit.offset, hit.ids};
        int[] cur = new int[1];
        Tokenizer.SequenceEncoder enc = hit == null ? tok.newSequence(protectedTokens) : tok.resumeSequence(protectedTokens, prefix);
        enc.onBoundary((offset, ids) -> {
            last[0] = cur[0];
            last[1] = offset;
            last[2] = ids;
        });
        int start = hit == null ? 0 : hit.piece;
        List<Key> newKeys = new ArrayList<>();
        List<Ref> newRefs = new ArrayList<>();
        Entry entry = null;
        int[][] lastAt = new int[n][];
        for (int i = start; i < n; i++) {
            cur[0] = i;
            int from = hit != null && i == hit.piece ? hit.offset : 0;
            if (control.get(i)) {
                enc.special(texts.get(i), from);
            } else {
                enc.text(texts.get(i), from);
            }
            if (i + 1 > hitK && last[0] >= 0) {
                lastAt[i] = last.clone();
            }
        }
        int[] ids = enc.toArray();
        entry = new Entry(ids);
        for (int i = 0; i < n; i++) {
            if (lastAt[i] != null) {
                newKeys.add(keys[i]);
                newRefs.add(new Ref(entry, lastAt[i][0], lastAt[i][1], lastAt[i][2]));
            }
        }
        entry.bytes = ENTRY_BYTES + 4L * ids.length + KEY_BYTES * newKeys.size();
        synchronized (this) {
            hits += hit == null ? 0 : 1;
            reused += prefix == null ? 0 : prefix.length;
            encoded += ids.length - (prefix == null ? 0 : prefix.length);
            if (!newKeys.isEmpty() && entry.bytes <= maxBytes / 4) {
                for (int i = 0; i < newKeys.size(); i++) {
                    Ref old = refs.put(newKeys.get(i), newRefs.get(i));
                    if (old != null && --old.entry.live == 0) {
                        drop(old.entry);
                    }
                }
                entry.keys.addAll(newKeys);
                entry.live = newKeys.size();
                lru.put(entry, entry);
                bytes += entry.bytes;
                Iterator<Entry> it = lru.keySet().iterator();
                while (bytes > maxBytes && it.hasNext()) {
                    Entry e = it.next();
                    if (e == entry) {
                        break;
                    }
                    it.remove();
                    forget(e);
                }
            }
        }
        return ids.clone(); // the entry's array stays private
    }

    /** Removes an entry none of whose keys are left. */
    private void drop(Entry e) {
        if (lru.remove(e) != null) {
            bytes -= e.bytes;
        }
    }

    /** Removes an evicted entry's keys (those not taken over by a newer prompt). */
    private void forget(Entry e) {
        bytes -= e.bytes;
        for (Key k : e.keys) {
            Ref r = refs.get(k);
            if (r != null && r.entry == e) {
                refs.remove(k);
            }
        }
    }

    public synchronized Stats stats() {
        return new Stats(requests, hits, reused, encoded, lru.size(), bytes);
    }

    public synchronized void clear() {
        refs.clear();
        lru.clear();
        bytes = 0;
    }

    // ------------------------------------------------------------------ hashing

    private static final long M1 = 0x9E3779B97F4A7C15L;
    private static final long M2 = 0xC2B2AE3D27D4EB4FL;

    /**
     * {@code keys[k - 1]} identifies the pieces {@code [0, k)}: two 64-bit multiply-rotate hashes over the scope, then
     * per piece its kind and length followed by its characters. A collision would need both to agree: about 2^-128 for
     * a pair of different prompts.
     */
    static Key[] keys(String scope, Set<String> protectedTokens, List<String> texts, List<Boolean> control) {
        long a = 0x243F6A8885A308D3L;
        long b = 0x13198A2E03707344L;
        String seed = scope + '\u0000' + new java.util.TreeSet<>(protectedTokens);
        for (int i = 0; i < seed.length(); i++) {
            a = Long.rotateLeft((a ^ seed.charAt(i)) * M1, 29);
            b = Long.rotateLeft((b + seed.charAt(i)) * M2, 31);
        }
        Key[] keys = new Key[texts.size()];
        for (int p = 0; p < texts.size(); p++) {
            String t = texts.get(p);
            long head = ((long) t.length() << 1) | (control.get(p) ? 1 : 0);
            a = Long.rotateLeft((a ^ head) * M1, 29);
            b = Long.rotateLeft((b + head) * M2, 31);
            int len = t.length();
            int i = 0;
            for (; i + 4 <= len; i += 4) { // four UTF-16 units per round
                long v = t.charAt(i) | (long) t.charAt(i + 1) << 16 | (long) t.charAt(i + 2) << 32 | (long) t.charAt(i + 3) << 48;
                a = Long.rotateLeft((a ^ v) * M1, 29);
                b = Long.rotateLeft((b + v) * M2, 31);
            }
            for (; i < len; i++) {
                a = Long.rotateLeft((a ^ t.charAt(i)) * M1, 29);
                b = Long.rotateLeft((b + t.charAt(i)) * M2, 31);
            }
            keys[p] = new Key(fmix(a), fmix(b ^ a));
        }
        return keys;
    }

    /** MurmurHash3's 64-bit finalizer. */
    private static long fmix(long h) {
        h ^= h >>> 33;
        h *= 0xFF51AFD7ED558CCDL;
        h ^= h >>> 33;
        h *= 0xC4CEB9FE1A85EC53L;
        h ^= h >>> 33;
        return h;
    }
}
