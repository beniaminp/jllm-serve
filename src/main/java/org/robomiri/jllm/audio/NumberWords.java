package org.robomiri.jllm.audio;

import java.math.BigInteger;

/**
 * English number names as the Python library {@code inflect} writes them (what OuteTTS's text front end uses):
 * "one hundred and one", "one thousand, two hundred and thirty-four", "one thousand and ten", "three point one four".
 * Groups of three digits ("thousand", "million", ...) are joined with ", ", except that a last group below 100 follows
 * with " and "; "and" also follows "hundred" inside a group.
 */
public final class NumberWords {

    private NumberWords() {
    }

    private static final String[] ONES = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
    private static final String[] TENS = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
    private static final String[] MAGNITUDES = {"", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion",
        "sextillion", "septillion", "octillion", "nonillion", "decillion"};

    /** The words of a non-negative decimal literal such as "42" or "3.14". */
    public static String of(String number) {
        int dot = number.indexOf('.');
        String whole = dot < 0 ? number : number.substring(0, dot);
        StringBuilder sb = new StringBuilder(integer(whole));
        if (dot >= 0) {
            sb.append(" point");
            for (char c : number.substring(dot + 1).toCharArray()) {
                sb.append(' ').append(ONES[c - '0']);
            }
        }
        return sb.toString();
    }

    static String integer(String digits) {
        String d = digits.replaceFirst("^0+(?=.)", "");
        if (d.length() > 36) {          // beyond decillions: read the digits one by one
            StringBuilder sb = new StringBuilder();
            for (char c : d.toCharArray()) {
                sb.append(sb.isEmpty() ? "" : " ").append(ONES[c - '0']);
            }
            return sb.toString();
        }
        BigInteger n = new BigInteger(d);
        if (n.signum() == 0) {
            return "zero";
        }
        int groups = (d.length() + 2) / 3;
        int[] g = new int[groups];
        BigInteger thousand = BigInteger.valueOf(1000);
        BigInteger rest = n;
        for (int i = 0; i < groups; i++) {
            BigInteger[] qr = rest.divideAndRemainder(thousand);
            g[i] = qr[1].intValue();
            rest = qr[0];
        }
        StringBuilder sb = new StringBuilder();
        boolean any = false;
        for (int i = groups - 1; i >= 0; i--) {
            if (g[i] == 0) {
                continue;
            }
            String chunk = three(g[i], i == 0 && any);
            if (any) {
                sb.append(i == 0 && g[0] < 100 ? " and " : ", ");
            }
            sb.append(chunk);
            if (i > 0) {
                sb.append(' ').append(MAGNITUDES[i]);
            }
            any = true;
        }
        return sb.toString();
    }

    /** 1..999 as "x hundred and y". */
    private static String three(int v, boolean lowGroupAfterOthers) {
        StringBuilder sb = new StringBuilder();
        int h = v / 100;
        int r = v % 100;
        if (h > 0) {
            sb.append(ONES[h]).append(" hundred");
            if (r > 0) {
                sb.append(" and ");
            }
        }
        if (r > 0) {
            sb.append(r < 20 ? ONES[r] : TENS[r / 10] + (r % 10 > 0 ? "-" + ONES[r % 10] : ""));
        }
        return sb.toString();
    }
}
