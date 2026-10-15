package org.robomiri.jllm.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/**
 * Incremental UTF-8 decoder for streaming generation: byte-level tokens can split a multi-byte character, so
 * incomplete trailing sequences are held back until the next token completes them.
 */
public final class StreamDecoder {

    private byte[] buf = new byte[32];
    private int len;
    /** Leading spaces still to be dropped (SentencePiece decoders such as Mistral's strip the first one). */
    private int strip;

    public StreamDecoder() {
    }

    /** A decoder that, like {@link Tokenizer#decode}, drops the leading space the tokenizer's decoder strips. */
    public StreamDecoder(Tokenizer tokenizer) {
        this.strip = tokenizer.decodeStripLeading();
    }

    /** Appends token bytes and returns the text that is now complete (possibly empty). */
    public String push(byte[] bytes) {
        if (len + bytes.length > buf.length) {
            buf = Arrays.copyOf(buf, Math.max(buf.length * 2, len + bytes.length));
        }
        System.arraycopy(bytes, 0, buf, len, bytes.length);
        len += bytes.length;
        while (strip > 0 && len > 0) {
            if (buf[0] == ' ') {
                System.arraycopy(buf, 1, buf, 0, --len);
                strip--;
            } else {
                strip = 0;
            }
        }
        int complete = completePrefix();
        if (complete == 0) {
            return "";
        }
        String s = new String(buf, 0, complete, StandardCharsets.UTF_8);
        System.arraycopy(buf, complete, buf, 0, len - complete);
        len -= complete;
        return s;
    }

    /** Returns whatever is left (invalid sequences become U+FFFD). */
    public String flush() {
        String s = new String(buf, 0, len, StandardCharsets.UTF_8);
        len = 0;
        return s;
    }

    /** Length of the longest prefix that does not end inside a multi-byte sequence. */
    private int completePrefix() {
        // walk back over at most 3 continuation bytes to the lead byte of the last character
        int i = len - 1;
        int cont = 0;
        while (i >= 0 && cont < 3 && (buf[i] & 0xC0) == 0x80) {
            i--;
            cont++;
        }
        if (i < 0) {
            return len;
        }
        int lead = buf[i] & 0xFF;
        int need = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3 : (lead >> 3) == 0x1E ? 4 : 1;
        return len - i >= need ? len : i;
    }
}
