package org.robomiri.jllm.model;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;

/**
 * Supplies safetensors shards to the converter one at a time, so a 60 GB checkpoint can be quantized on a
 * machine that cannot hold all of it on disk.
 *
 * <p>Shard names come from {@code model.safetensors.index.json}, i.e. from the downloaded repository, and a
 * low-disk implementation deletes {@code dir/name} on {@link #release}. Implementations must therefore only
 * accept names that {@link #checkName} approves: plain file names of {@code .safetensors} files that live
 * directly in the model directory.
 */
public interface ShardProvider {

    /** Returns the path of {@code fileName}, downloading it first if necessary. */
    Path ensure(String fileName) throws IOException;

    /** Called when the converter no longer needs {@code fileName} (a low-disk implementation deletes it). */
    void release(String fileName) throws IOException;

    /**
     * Validates a shard file name taken from an untrusted index: it must be a plain {@code .safetensors} file
     * name (no directory separators, no {@code ..}) that resolves to a direct child of {@code dir}.
     *
     * @return {@code fileName}
     * @throws IOException if the name is rejected
     */
    static String checkName(Path dir, String fileName) throws IOException {
        if (fileName == null || fileName.isBlank() || fileName.contains("/") || fileName.contains("\\")
                || fileName.contains("..") || fileName.indexOf('\0') >= 0 || !fileName.endsWith(".safetensors")) {
            throw new IOException("invalid shard file name in index: '" + fileName + "'");
        }
        try {
            Path base = dir.toAbsolutePath().normalize();
            Path p = base.resolve(fileName).normalize();
            Path parent = p.getParent();
            if (parent == null || !parent.equals(base) || !fileName.equals(p.getFileName().toString())) {
                throw new IOException("shard file name escapes the model directory: '" + fileName + "'");
            }
        } catch (InvalidPathException e) {
            throw new IOException("invalid shard file name in index: '" + fileName + "'", e);
        }
        return fileName;
    }

    /** Files are expected to be present already. */
    static ShardProvider local(Path dir) {
        return new ShardProvider() {
            @Override
            public Path ensure(String fileName) throws IOException {
                Path p = dir.resolve(checkName(dir, fileName));
                if (!Files.isRegularFile(p)) {
                    throw new IOException("missing " + p);
                }
                return p;
            }

            @Override
            public void release(String fileName) throws IOException {
                checkName(dir, fileName);
            }
        };
    }
}
