package org.robomiri.jllm.kv;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A {@link KvSpill} that keeps one file per block in a directory: {@code <dir>/<fingerprint hash>/<path hash>.kvb}.
 * The fingerprint (the model, its quantization and cache type, the block size) keeps blocks of another model or
 * setting from ever being read; a file also carries the block's tokens, which {@link #get} compares, so a hash collision
 * returns nothing instead of wrong rows. Files left by an earlier run are found again by listing the directory, so
 * a restart finds the blocks the last run spilled. The total size is kept under a budget by deleting the least
 * recently used files.
 */
public final class FileSpill implements KvSpill {

    private static final int MAGIC = 0x4B564231; // "KVB1"

    private record Info(long size, long used) {
    }

    private final Path dir;
    private final long fingerprint;
    private final long maxBytes;
    private final Map<Long, Info> index = new ConcurrentHashMap<>();
    private long bytes;
    private long clock;

    /**
     * @param root        the base directory
     * @param fingerprint what identifies the rows (model, quantization, cache type, layers, block size)
     * @param maxBytes    the budget; the oldest files go beyond it
     */
    public FileSpill(Path root, String fingerprint, long maxBytes) throws IOException {
        this.fingerprint = fingerprintHash(fingerprint);
        this.dir = root.resolve(HexFormat.of().formatHex(longBytes(this.fingerprint)));
        this.maxBytes = maxBytes;
        Files.createDirectories(dir);
        try (var files = Files.list(dir)) {
            for (Path f : files.toList()) {
                String n = f.getFileName().toString();
                if (n.endsWith(".kvb")) {
                    try {
                        long h = Long.parseUnsignedLong(n.substring(0, n.length() - 4), 16);
                        long size = Files.size(f);
                        index.put(h, new Info(size, Files.getLastModifiedTime(f).toMillis()));
                        bytes += size;
                    } catch (NumberFormatException e) {
                        // not ours
                    }
                } else if (n.endsWith(".tmp")) {
                    Files.deleteIfExists(f); // a write that did not finish
                }
            }
        }
    }

    private static byte[] longBytes(long v) {
        return ByteBuffer.allocate(8).putLong(v).array();
    }

    private static long fingerprintHash(String s) {
        try {
            byte[] d = MessageDigest.getInstance("SHA-256").digest(s.getBytes(StandardCharsets.UTF_8));
            return ByteBuffer.wrap(d).getLong();
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private Path file(long h) {
        return dir.resolve(Long.toHexString(h) + ".kvb");
    }

    @Override
    public boolean has(long pathHash) {
        return index.containsKey(pathHash);
    }

    @Override
    public boolean put(long pathHash, int[] tokens, ByteBuffer payload) {
        int payloadBytes = payload.remaining();
        ByteBuffer head = ByteBuffer.allocate(4 + 8 + 8 + 4 + 4 * tokens.length + 4).order(ByteOrder.LITTLE_ENDIAN);
        head.putInt(MAGIC).putLong(fingerprint).putLong(pathHash).putInt(tokens.length);
        for (int t : tokens) {
            head.putInt(t);
        }
        head.putInt(payloadBytes).flip();
        long size = head.remaining() + payloadBytes;
        if (size > maxBytes) {
            return false;
        }
        Path tmp = dir.resolve(Long.toHexString(pathHash) + ".tmp");
        try (FileChannel ch = FileChannel.open(tmp, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            while (head.hasRemaining()) {
                ch.write(head);
            }
            ByteBuffer p = payload.duplicate();
            while (p.hasRemaining()) {
                ch.write(p);
            }
            ch.close();
            Files.move(tmp, file(pathHash), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            try {
                Files.deleteIfExists(tmp);
            } catch (IOException ignored) {
                // nothing to do
            }
            return false;
        }
        synchronized (this) {
            Info old = index.put(pathHash, new Info(size, ++clock + System.currentTimeMillis()));
            bytes += size - (old == null ? 0 : old.size());
        }
        trim();
        return true;
    }

    @Override
    public boolean get(long pathHash, int[] tokens, ByteBuffer payload) {
        Info info = index.get(pathHash);
        if (info == null) {
            return false;
        }
        try (FileChannel ch = FileChannel.open(file(pathHash), StandardOpenOption.READ)) {
            ByteBuffer head = ByteBuffer.allocate(4 + 8 + 8 + 4 + 4 * tokens.length + 4).order(ByteOrder.LITTLE_ENDIAN);
            while (head.hasRemaining()) {
                if (ch.read(head) < 0) {
                    return false;
                }
            }
            head.flip();
            if (head.getInt() != MAGIC || head.getLong() != fingerprint || head.getLong() != pathHash
                    || head.getInt() != tokens.length) {
                return false;
            }
            for (int t : tokens) {
                if (head.getInt() != t) {
                    return false;
                }
            }
            int n = head.getInt();
            if (n != payload.remaining()) {
                return false;
            }
            ByteBuffer dst = payload.duplicate();
            while (dst.hasRemaining()) {
                if (ch.read(dst) < 0) {
                    return false;
                }
            }
            payload.position(payload.limit());
        } catch (IOException e) {
            return false;
        }
        synchronized (this) {
            index.computeIfPresent(pathHash, (k, v) -> new Info(v.size(), ++clock + System.currentTimeMillis()));
        }
        return true;
    }

    @Override
    public void remove(long pathHash) {
        Info old;
        synchronized (this) {
            old = index.remove(pathHash);
            if (old != null) {
                bytes -= old.size();
            }
        }
        if (old != null) {
            try {
                Files.deleteIfExists(file(pathHash));
            } catch (IOException ignored) {
                // retried never; a leftover file is reindexed on the next start
            }
        }
    }

    /** Deletes the least recently used files until the budget holds. */
    private void trim() {
        List<Map.Entry<Long, Info>> all;
        synchronized (this) {
            if (bytes <= maxBytes) {
                return;
            }
            all = new ArrayList<>(index.entrySet());
        }
        all.sort((a, b) -> Long.compare(a.getValue().used(), b.getValue().used()));
        for (var e : all) {
            if (bytes() <= maxBytes) {
                break;
            }
            remove(e.getKey());
        }
    }

    @Override
    public int files() {
        return index.size();
    }

    @Override
    public synchronized long bytes() {
        return bytes;
    }

    /** Deletes every block file of this fingerprint. */
    public synchronized void clear() {
        for (Long h : new ArrayList<>(index.keySet())) {
            remove(h);
        }
    }

    public Path directory() {
        return dir;
    }
}
