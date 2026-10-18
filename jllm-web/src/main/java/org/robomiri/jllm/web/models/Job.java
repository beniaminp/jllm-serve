package org.robomiri.jllm.web.models;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.Future;

/** A background download or load, with its progress estimated from the files growing on disk. */
public final class Job {

    public enum Kind { DOWNLOAD, LOAD }

    public enum State { QUEUED, RUNNING, DONE, FAILED, CANCELLED }

    /**
     * @param downloadedBytes bytes of the expected files present on disk (complete or partial); -1 when unknown
     * @param totalBytes      bytes of the files the job downloads; -1 when unknown
     */
    public record View(String id, Kind kind, String model, String source, State state, String phase, String error,
                       long created, long finished, long downloadedBytes, long totalBytes) {
    }

    final String id;
    final Kind kind;
    final String model;
    final String source;
    final long created = System.currentTimeMillis();
    volatile State state = State.QUEUED;
    volatile String phase = "queued";
    volatile String error;
    volatile long finished;
    volatile Future<?> future;
    /** Directory the expected files land in, and their sizes by name relative to it. */
    private volatile Path dir;
    private volatile Map<String, Long> expected = Map.of();

    Job(String id, Kind kind, String model, String source) {
        this.id = id;
        this.kind = kind;
        this.model = model;
        this.source = source;
    }

    public String id() {
        return id;
    }

    void expect(Path dir, Map<String, Long> files) {
        this.dir = dir;
        this.expected = Map.copyOf(files);
    }

    boolean finished() {
        return state == State.DONE || state == State.FAILED || state == State.CANCELLED;
    }

    void finish(State s, String err) {
        state = s;
        error = err;
        finished = System.currentTimeMillis();
        phase = switch (s) {
            case DONE -> "done";
            case FAILED -> "failed";
            case CANCELLED -> "cancelled";
            default -> phase;
        };
    }

    public View view() {
        long total = -1;
        long have = -1;
        Path d = dir;
        Map<String, Long> files = expected;
        if (d != null && !files.isEmpty()) {
            total = 0;
            have = 0;
            for (Map.Entry<String, Long> e : files.entrySet()) {
                long size = Math.max(0, e.getValue());
                total += size;
                have += Math.min(size, sizeOnDisk(d.resolve(e.getKey())));
            }
        }
        return new View(id, kind, model, source, state, phase, error, created, finished, have, total);
    }

    private static long sizeOnDisk(Path p) {
        try {
            if (Files.isRegularFile(p)) {
                return Files.size(p);
            }
            Path part = p.resolveSibling(p.getFileName() + ".incomplete");
            return Files.isRegularFile(part) ? Files.size(part) : 0;
        } catch (IOException e) {
            return 0;
        }
    }
}
