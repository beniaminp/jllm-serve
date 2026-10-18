package org.robomiri.jllm.web.logs;

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

/**
 * Keeps the recent output of {@code System.out} and {@code System.err} for the console's log view. The engine
 * reports download and conversion progress on stderr with carriage returns, so a {@code \r} replaces the line
 * being written instead of starting a new one; that unfinished line is available as {@link #pending()}.
 */
public final class LogBuffer {

    public record Line(long seq, long time, String stream, String text) {
    }

    private static final LogBuffer INSTANCE = new LogBuffer();

    private final ArrayDeque<Line> lines = new ArrayDeque<>();
    private int capacity = 2000;
    private long seq;
    private String pending = "";

    private LogBuffer() {
    }

    public static LogBuffer get() {
        return INSTANCE;
    }

    /** Tees {@code System.out} and {@code System.err} into the buffer (once). */
    public static synchronized void install() {
        if (System.err instanceof TeePrintStream) {
            return;
        }
        System.setOut(new TeePrintStream(System.out, "out"));
        System.setErr(new TeePrintStream(System.err, "err"));
    }

    public synchronized void setCapacity(int capacity) {
        this.capacity = Math.max(100, capacity);
        trim();
    }

    /** The lines with a sequence number greater than {@code after}, oldest first. */
    public synchronized List<Line> after(long after) {
        List<Line> out = new ArrayList<>();
        for (Line l : lines) {
            if (l.seq() > after) {
                out.add(l);
            }
        }
        return out;
    }

    /** The line currently being rewritten (a progress bar), or empty. */
    public synchronized String pending() {
        return pending;
    }

    public synchronized long lastSeq() {
        return seq;
    }

    synchronized void add(String stream, String text) {
        lines.addLast(new Line(++seq, System.currentTimeMillis(), stream, text));
        trim();
    }

    synchronized void setPending(String text) {
        pending = text;
    }

    private void trim() {
        while (lines.size() > capacity) {
            lines.removeFirst();
        }
    }

    /** Passes everything through to the original stream and splits a copy into lines. */
    private static final class TeePrintStream extends PrintStream {
        TeePrintStream(PrintStream original, String name) {
            super(new LineSplitter(original, name), true, StandardCharsets.UTF_8);
        }
    }

    private static final class LineSplitter extends OutputStream {
        private final PrintStream original;
        private final String name;
        private final ByteArrayOutputStream line = new ByteArrayOutputStream();

        LineSplitter(PrintStream original, String name) {
            this.original = original;
            this.name = name;
        }

        @Override
        public synchronized void write(int b) {
            original.write(b);
            accept(b);
        }

        @Override
        public synchronized void write(byte[] b, int off, int len) {
            original.write(b, off, len);
            for (int i = off; i < off + len; i++) {
                accept(b[i]);
            }
            if (line.size() > 0) { // a progress line is shown as it is being written, not only once replaced
                INSTANCE.setPending(line.toString(StandardCharsets.UTF_8));
            }
        }

        @Override
        public void flush() {
            original.flush();
        }

        // '\n' and '\r' never occur inside a multi-byte UTF-8 sequence, so splitting bytes on them is safe
        private void accept(int b) {
            if (b == '\n') {
                INSTANCE.add(name, line.toString(StandardCharsets.UTF_8));
                INSTANCE.setPending("");
                line.reset();
            } else if (b == '\r') {
                line.reset();
            } else {
                line.write(b);
                if (line.size() > 64 * 1024) { // a runaway line: keep it, but bounded
                    INSTANCE.add(name, line.toString(StandardCharsets.UTF_8));
                    line.reset();
                }
            }
        }
    }
}
