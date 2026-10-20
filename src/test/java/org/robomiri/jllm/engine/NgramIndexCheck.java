package org.robomiri.jllm.engine;

import java.util.Random;

/** {@link NgramIndex} against a brute-force search, including a history cut back to a prefix and regrown. */
public final class NgramIndexCheck {

    private NgramIndexCheck() {
    }

    /** The latest e < end whose n-gram equals the one ending at end, or -1. */
    private static int brute(int[] h, int end, int n) {
        for (int e = end - 1; e >= n - 1; e--) {
            boolean eq = true;
            for (int j = 0; j < n && eq; j++) {
                eq = h[e - j] == h[end - j];
            }
            if (eq) {
                return e;
            }
        }
        return -1;
    }

    /** @return the number of mismatches */
    public static int run() {
        Random r = new Random(5);
        int bad = 0;
        for (int trial = 0; trial < 20; trial++) {
            int vocab = 3 + r.nextInt(6); // small vocabularies repeat a lot
            int[] h = new int[3000];
            NgramIndex idx = new NgramIndex(2, 8);
            int len = 0;
            for (int phase = 0; phase < 2; phase++) {
                if (phase == 1) { // the next request shares only a prefix with this one
                    len = r.nextInt(len);
                    idx.truncate(h, len);
                }
                for (int i = 0; i < 1500; i++) {
                    h[len++] = r.nextInt(vocab);
                    if (len >= 2) {
                        idx.add(h, len - 2);
                    }
                    for (int n = 2; n <= 8; n++) {
                        int want = brute(h, len - 1, n);
                        int got = idx.find(h, len - 1, n);
                        // the index keeps one occurrence per n-gram: it may miss an older one it overwrote, never
                        // return a wrong one, and always finds the latest one once the history has stopped changing
                        boolean ok = got == -1 ? want == -1 || phase == 1 : got == want || phase == 1 && brute(h, got, n) >= -1
                                && matches(h, got, len - 1, n);
                        if (!ok) {
                            bad++;
                        }
                    }
                }
            }
        }
        return bad;
    }

    private static boolean matches(int[] h, int e, int end, int n) {
        for (int j = 0; j < n; j++) {
            if (h[e - j] != h[end - j]) {
                return false;
            }
        }
        return e < end;
    }
}
