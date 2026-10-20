package org.robomiri.jllm.engine;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;

/**
 * A KV cache saved to disk ({@link Engine#saveCache}): a header with the model's fingerprint, the backend's state
 * format and the token ids whose keys and values follow, then the backend's state ({@link
 * org.robomiri.jllm.model.Backend#saveState}). The header is small and can be read on its own, so a store of saved
 * caches can be indexed without reading the caches.
 *
 * <pre>
 * "JLLMKV01"  8 bytes
 * fingerprint, format    modified UTF-8 (DataOutput.writeUTF)
 * n                      int (big-endian), then n token ids as ints
 * state                  backend-specific (the CPU KV cache: little-endian f32 / f16 rows)
 * </pre>
 */
public final class CacheFile {

    private static final byte[] MAGIC = {'J', 'L', 'L', 'M', 'K', 'V', '0', '1'};

    /** The header of a saved cache. */
    public record Header(String fingerprint, String format, int[] tokens) {
    }

    interface Body {
        void write(FileChannel ch) throws IOException;
    }

    private CacheFile() {
    }

    /** Reads only the header of {@code file}. */
    public static Header readHeader(Path file) throws IOException {
        try (InputStream in = Files.newInputStream(file)) {
            return readHeader(new DataInputStream(new java.io.BufferedInputStream(in, 1 << 16)));
        }
    }

    static Header readHeader(DataInputStream in) throws IOException {
        byte[] magic = in.readNBytes(MAGIC.length);
        if (!java.util.Arrays.equals(magic, MAGIC)) {
            throw new IOException("not a saved jllm KV cache");
        }
        String fingerprint = in.readUTF();
        String format = in.readUTF();
        int n = in.readInt();
        if (n < 0 || n > 1 << 24) {
            throw new IOException("corrupt saved cache header (" + n + " tokens)");
        }
        byte[] ids = in.readNBytes(4 * n);
        if (ids.length != 4 * n) {
            throw new IOException("the saved cache file is truncated");
        }
        int[] tokens = new int[n];
        ByteBuffer.wrap(ids).order(ByteOrder.BIG_ENDIAN).asIntBuffer().get(tokens);
        return new Header(fingerprint, format, tokens);
    }

    /** Writes header and body to a temporary file next to {@code file}, then moves it into place. */
    static void write(Path file, Header h, Body body) throws IOException {
        Path dir = file.toAbsolutePath().getParent();
        Files.createDirectories(dir);
        Path tmp = Files.createTempFile(dir, file.getFileName().toString(), ".tmp");
        try {
            try (FileChannel ch = FileChannel.open(tmp, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
                var bytes = new java.io.ByteArrayOutputStream();
                DataOutputStream out = new DataOutputStream(bytes);
                out.write(MAGIC);
                out.writeUTF(h.fingerprint());
                out.writeUTF(h.format());
                out.writeInt(h.tokens().length);
                for (int t : h.tokens()) {
                    out.writeInt(t);
                }
                out.flush();
                ByteBuffer hb = ByteBuffer.wrap(bytes.toByteArray());
                while (hb.hasRemaining()) {
                    ch.write(hb);
                }
                body.write(ch);
            }
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    /** Opens {@code file}, checks its header against the expected fingerprint and format, and hands over the body. */
    static Header read(Path file, String fingerprint, String format, BodyReader body) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            DataInputStream in = new DataInputStream(Channels.newInputStream(ch));
            Header h = readHeader(in); // unbuffered: the channel stays positioned right after the header
            if (!h.fingerprint().equals(fingerprint)) {
                throw new IOException("the saved cache is of another model (" + h.fingerprint() + ")");
            }
            if (!h.format().equals(format)) {
                throw new IOException("the saved cache has another layout (" + h.format() + ", this backend: " + format + ")");
            }
            body.read(h, ch);
            return h;
        }
    }

    interface BodyReader {
        void read(Header h, FileChannel ch) throws IOException;
    }
}
