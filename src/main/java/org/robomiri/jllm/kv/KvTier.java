package org.robomiri.jllm.kv;

import java.io.IOException;
import java.nio.file.Path;

/** Builds the block store behind a pool: warm tier in host memory, then (possibly compressed) block files. */
public final class KvTier {

    private KvTier() {
    }

    /**
     * Gives {@code pool} the tiered spill store of {@code cfg}: blocks it evicts go to a host-memory tier of
     * {@code cfg.hostBytes()}, whose overflow goes to files under {@code dir} (at most {@code maxBytes}), stored in
     * {@code cfg.diskFormat()} where the pool's row type allows. Files are named by the model's fingerprint, the pool's
     * row layout and the disk format, so a run with other settings never reads them.
     */
    public static TieredSpill attach(KvPool pool, KvTierConfig cfg, String fingerprint, Path dir, long maxBytes)
            throws IOException {
        KvBlockCodec codec = pool.storage().codec(cfg.diskFormat());
        String fp = fingerprint + "|" + pool.format() + (codec != null ? "|disk=" + codec.describe() : "");
        FileSpill files = new FileSpill(dir, fp, maxBytes);
        KvSpill lowest = codec == null || codec.identity() ? files : new CodecSpill(files, codec);
        TieredSpill tiered = new TieredSpill(lowest, cfg.hostBytes());
        pool.setSpill(tiered);
        pool.setPrefetch(cfg.prefetch());
        return tiered;
    }
}
