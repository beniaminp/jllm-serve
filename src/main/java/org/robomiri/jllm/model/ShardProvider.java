package org.robomiri.jllm.model;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Supplies safetensors shards to the converter one at a time, so a 60 GB checkpoint can be quantized on a
 * machine that cannot hold all of it on disk.
 */
public interface ShardProvider {

    /** Returns the path of {@code fileName}, downloading it first if necessary. */
    Path ensure(String fileName) throws IOException;

    /** Called when the converter no longer needs {@code fileName} (a low-disk implementation deletes it). */
    void release(String fileName) throws IOException;

    /** Files are expected to be present already. */
    static ShardProvider local(Path dir) {
        return new ShardProvider() {
            @Override
            public Path ensure(String fileName) throws IOException {
                Path p = dir.resolve(fileName);
                if (!java.nio.file.Files.isRegularFile(p)) {
                    throw new IOException("missing " + p);
                }
                return p;
            }

            @Override
            public void release(String fileName) {
            }
        };
    }
}
