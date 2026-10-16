package org.robomiri.jllm.server;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * A small HTTP/1.1 server on plain sockets, one virtual thread per connection. Supports keep-alive,
 * Content-Length request bodies and chunked streaming responses (used for SSE).
 */
public final class HttpServer implements AutoCloseable {

    private static final int MAX_HEADER = 64 * 1024;

    /**
     * Connection limits (the {@code server.*} keys in {@code jllm.properties}).
     *
     * @param readTimeoutMs  socket read timeout while waiting for a request or its body, 0 = none
     * @param writeTimeoutMs how long a streaming response may wait for a client that stopped reading before the
     *                       connection is dropped (the generation behind it is cancelled), 0 = forever
     * @param maxConnections simultaneous connections; further ones wait in the kernel accept queue
     * @param maxBodyBytes   largest accepted request body
     * @param corsOrigin     value of {@code Access-Control-Allow-Origin}; null or empty = no CORS headers at all
     */
    public record Options(int readTimeoutMs, int writeTimeoutMs, int maxConnections, long maxBodyBytes,
                          String corsOrigin) {
        public Options {
            if (readTimeoutMs < 0 || writeTimeoutMs < 0 || maxConnections < 1 || maxBodyBytes < 0) {
                throw new IllegalArgumentException("invalid server options: " + this);
            }
            if (corsOrigin != null && corsOrigin.isBlank()) {
                corsOrigin = null;
            }
        }

        public static Options defaults() {
            return new Options(30_000, 10_000, 256, 4L * 1024 * 1024, null);
        }
    }

    /** A malformed request: answered with the given status and the connection is closed. */
    private static final class HttpError extends IOException {
        final int status;

        HttpError(int status, String message) {
            super(message);
            this.status = status;
        }
    }

    public static final class Request {
        public final String method;
        public final String path;
        public final Map<String, String> headers;
        public final byte[] body;
        /** True for an HTTP/1.0 (or version-less) request: no keep-alive, no chunked responses. */
        public final boolean http10;
        Probe probe;

        Request(String method, String path, Map<String, String> headers, byte[] body, boolean http10) {
            this.method = method;
            this.path = path;
            this.headers = headers;
            this.body = body;
            this.http10 = http10;
        }

        public String bodyText() {
            return new String(body, StandardCharsets.UTF_8);
        }

        /**
         * Whether the client has closed the connection since the request was read (also true when it merely
         * shut down its sending side). Polled by long-running handlers so that they stop working for nobody.
         */
        public boolean clientGone() {
            return probe != null && probe.gone;
        }
    }

    /**
     * Watches the connection while its request is handled: a virtual thread blocks reading the socket, which
     * returns -1 (or throws) as soon as the client closes, so a generation can be cancelled even when nothing is
     * being written to the client (non-streaming requests), and before the first failed write for streams. A
     * byte it reads instead is the start of the next (pipelined) request and is pushed back with mark/reset
     * once the response is complete; after that a read timeout is the keep-alive idle timeout.
     */
    private static final class Probe implements Runnable {
        private static final int PENDING = -2;
        private static final int TIMEOUT = -3;

        private final InputStream in;
        private final Thread thread;
        volatile boolean gone;
        private volatile boolean responseDone;
        private volatile int result = PENDING;

        Probe(InputStream in) {
            this.in = in;
            this.thread = Thread.ofVirtual().name("http-probe").start(this);
        }

        @Override
        public void run() {
            try {
                in.mark(1);
                while (true) {
                    try {
                        int c = in.read();
                        result = c;
                        if (c < 0) {
                            gone = true;
                        }
                        return;
                    } catch (SocketTimeoutException e) {
                        if (responseDone) { // nothing arrived after the response: an idle connection
                            result = TIMEOUT;
                            return;
                        }
                        // the response is still being produced: keep watching
                    }
                }
            } catch (IOException e) {
                gone = true;
                result = -1;
            }
        }

        /**
         * Called once the response is complete; waits for the next request's first byte. Returns true when it
         * arrived (and is back in the stream), false when the client is gone or the idle timeout expired.
         */
        boolean nextRequest() throws IOException {
            responseDone = true;
            try {
                thread.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
            if (result < 0) {
                return false;
            }
            in.reset();
            return true;
        }
    }

    /**
     * Response writer; either call {@link #send} once, or {@link #startStream} followed by {@link #write}/{@link #end}.
     *
     * <p>A stream is written by a writer thread of its own: {@link #write} only queues the chunk, so the caller
     * (the engine thread, holding the engine lock) never blocks on a client that has stopped reading. When the
     * queue stays full for longer than {@link Options#writeTimeoutMs()} the socket is closed, which unblocks the
     * writer, and {@code write} throws so the caller stops producing.
     */
    public static final class Response {
        private static final int QUEUE_CHUNKS = 256;
        private static final byte[] END = new byte[0];

        private final OutputStream out;
        private final Socket socket;
        private final int writeTimeoutMs;
        private final boolean keepAlive;
        private final String corsOrigin;
        private final boolean head;    // HEAD request: headers only, never a body
        private final boolean chunked; // HTTP/1.1: streams use chunked encoding; HTTP/1.0: raw bytes until close
        private final StringBuilder extraHeaders = new StringBuilder();
        private boolean started;
        private boolean streaming;
        private boolean ended;
        private BlockingQueue<byte[]> queue;
        private Thread writer;
        private volatile IOException writeFailure;
        private volatile boolean aborted;

        Response(OutputStream out, Socket socket, int writeTimeoutMs, boolean keepAlive, String corsOrigin,
                 boolean head, boolean chunked) {
            this.out = out;
            this.socket = socket;
            this.writeTimeoutMs = writeTimeoutMs;
            this.keepAlive = keepAlive;
            this.corsOrigin = corsOrigin;
            this.head = head;
            this.chunked = chunked;
        }

        /** Adds a header to the response; must be called before {@link #send} or {@link #startStream}. */
        public Response header(String name, String value) {
            if (started) {
                throw new IllegalStateException("response already started");
            }
            extraHeaders.append(name).append(": ").append(value).append("\r\n");
            return this;
        }

        /** Whether CORS headers are configured (and therefore an OPTIONS preflight deserves an answer). */
        public boolean cors() {
            return corsOrigin != null;
        }

        private void commonHeaders(StringBuilder h) {
            if (corsOrigin != null) {
                h.append("Access-Control-Allow-Origin: ").append(corsOrigin).append("\r\n");
                if (!corsOrigin.equals("*")) {
                    h.append("Vary: Origin\r\n");
                }
            }
            h.append(extraHeaders);
            h.append("Connection: ").append(keepAlive ? "keep-alive" : "close").append("\r\n\r\n");
        }

        public void send(int status, String contentType, byte[] body) throws IOException {
            if (started) {
                throw new IllegalStateException("response already started");
            }
            started = true;
            ended = true;
            StringBuilder h = new StringBuilder();
            h.append("HTTP/1.1 ").append(status).append(' ').append(reason(status)).append("\r\n");
            h.append("Content-Type: ").append(contentType).append("\r\n");
            h.append("Content-Length: ").append(body.length).append("\r\n");
            commonHeaders(h);
            out.write(h.toString().getBytes(StandardCharsets.ISO_8859_1));
            if (!head) {
                out.write(body);
            }
            out.flush();
        }

        public void send(int status, String contentType, String body) throws IOException {
            send(status, contentType, body.getBytes(StandardCharsets.UTF_8));
        }

        public void startStream(int status, String contentType) throws IOException {
            if (started) {
                throw new IllegalStateException("response already started");
            }
            started = true;
            streaming = true;
            StringBuilder h = new StringBuilder();
            h.append("HTTP/1.1 ").append(status).append(' ').append(reason(status)).append("\r\n");
            h.append("Content-Type: ").append(contentType).append("\r\n");
            if (chunked) {
                h.append("Transfer-Encoding: chunked\r\n");
            }
            h.append("Cache-Control: no-cache\r\n");
            commonHeaders(h);
            queue = new ArrayBlockingQueue<>(QUEUE_CHUNKS);
            writer = Thread.ofVirtual().name("http-writer").start(this::drain);
            enqueue(h.toString().getBytes(StandardCharsets.ISO_8859_1));
        }

        public void write(String chunk) throws IOException {
            if (!streaming) {
                throw new IllegalStateException("not streaming");
            }
            byte[] b = chunk.getBytes(StandardCharsets.UTF_8);
            if (b.length == 0 || head) {
                return;
            }
            if (chunked) {
                byte[] size = (Integer.toHexString(b.length) + "\r\n").getBytes(StandardCharsets.ISO_8859_1);
                byte[] framed = new byte[size.length + b.length + 2];
                System.arraycopy(size, 0, framed, 0, size.length);
                System.arraycopy(b, 0, framed, size.length, b.length);
                framed[framed.length - 2] = '\r';
                framed[framed.length - 1] = '\n';
                b = framed;
            }
            enqueue(b);
        }

        /** Finishes the response; for a stream, waits until everything queued has reached the socket. */
        public void end() throws IOException {
            if (ended) {
                return;
            }
            ended = true;
            if (!streaming) {
                out.flush();
                return;
            }
            if (chunked && !head) {
                enqueue("0\r\n\r\n".getBytes(StandardCharsets.ISO_8859_1));
            }
            enqueue(END);
            boolean finished;
            try {
                finished = writeTimeoutMs > 0 ? writer.join(Duration.ofMillis(writeTimeoutMs)) : joinIndefinitely();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                abort();
                throw new IOException("interrupted while finishing the stream");
            }
            if (!finished) {
                abort();
                throw new IOException("client stopped reading the stream for " + writeTimeoutMs + " ms");
            }
            if (writeFailure != null) {
                throw new IOException("client connection failed: " + writeFailure.getMessage(), writeFailure);
            }
        }

        private boolean joinIndefinitely() throws InterruptedException {
            writer.join();
            return true;
        }

        /** The writer thread: copies queued chunks to the socket, flushing whenever the queue runs empty. */
        private void drain() {
            try {
                while (true) {
                    byte[] b = queue.take();
                    if (b == END) {
                        break;
                    }
                    out.write(b);
                    if (queue.isEmpty()) {
                        out.flush();
                    }
                }
                out.flush();
            } catch (IOException e) {
                writeFailure = e;
                queue.clear(); // a producer blocked on the full queue wakes up and sees the failure
            } catch (InterruptedException e) {
                writeFailure = new IOException("writer interrupted");
                queue.clear();
            }
        }

        private void enqueue(byte[] b) throws IOException {
            if (writeFailure != null) {
                throw new IOException("client connection failed: " + writeFailure.getMessage(), writeFailure);
            }
            if (aborted) {
                throw new IOException("stream aborted");
            }
            boolean queued;
            try {
                if (writeTimeoutMs > 0) {
                    queued = queue.offer(b, writeTimeoutMs, TimeUnit.MILLISECONDS);
                } else {
                    queue.put(b);
                    queued = true;
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                abort();
                throw new IOException("interrupted while writing the stream");
            }
            if (!queued) {
                abort();
                throw new IOException("client stopped reading the stream for " + writeTimeoutMs + " ms");
            }
        }

        /** Drops the connection: closes the socket so that a writer blocked in {@code write} unblocks. */
        void abort() {
            aborted = true;
            try {
                socket.close();
            } catch (IOException ignored) {
                // nothing to do
            }
        }

        boolean started() {
            return started;
        }

        boolean ended() {
            return ended;
        }
    }

    @FunctionalInterface
    public interface Handler {
        void handle(Request req, Response res) throws Exception;
    }

    private final ServerSocket socket;
    private final Handler handler;
    private final Options options;
    private final Semaphore connections;
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    private volatile boolean closed;

    public HttpServer(String host, int port, Handler handler) throws IOException {
        this(host, port, Options.defaults(), handler);
    }

    public HttpServer(String host, int port, Options options, Handler handler) throws IOException {
        this.handler = handler;
        this.options = options;
        this.connections = new Semaphore(options.maxConnections());
        this.socket = new ServerSocket();
        socket.setReuseAddress(true);
        socket.bind(new InetSocketAddress(host, port));
    }

    public int port() {
        return socket.getLocalPort();
    }

    /** Accepts connections until {@link #close()}; blocks the calling thread. */
    public void serve() {
        while (!closed) {
            try {
                // hold the permit before accept() so connections beyond the cap wait in the kernel backlog
                if (!connections.tryAcquire(100, TimeUnit.MILLISECONDS)) {
                    continue;
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
            Socket s;
            try {
                s = socket.accept();
            } catch (IOException e) {
                connections.release();
                if (closed) {
                    return;
                }
                try {
                    Thread.sleep(100); // e.g. out of file descriptors: back off instead of spinning
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return;
                }
                continue;
            }
            try {
                executor.submit(() -> connection(s));
            } catch (RuntimeException e) { // executor shut down
                connections.release();
                try {
                    s.close();
                } catch (IOException ignored) {
                    // nothing to do
                }
            }
        }
    }

    private void connection(Socket s) {
        try (s) {
            s.setTcpNoDelay(true);
            s.setSoTimeout(options.readTimeoutMs());
            InputStream in = new BufferedInputStream(s.getInputStream());
            OutputStream out = new BufferedOutputStream(s.getOutputStream(), 1 << 16);
            while (!closed) {
                Request req;
                try {
                    req = readRequest(in, out);
                } catch (HttpError e) {
                    new Response(out, s, options.writeTimeoutMs(), false, options.corsOrigin(), false, false)
                            .send(e.status, "application/json", errorJson("invalid_request_error", e.getMessage()));
                    return;
                }
                if (req == null) {
                    return;
                }
                boolean keepAlive = !req.http10 && !"close".equalsIgnoreCase(req.headers.getOrDefault("connection", ""));
                Probe probe = new Probe(in); // the body has been consumed: from here on the socket is idle inbound
                req.probe = probe;
                Response res = new Response(out, s, options.writeTimeoutMs(), keepAlive, options.corsOrigin(),
                        req.method.equals("HEAD"), !req.http10);
                try {
                    handler.handle(req, res);
                    if (!res.started()) {
                        res.send(404, "text/plain", "not found");
                    }
                    res.end();
                } catch (Exception e) {
                    if (res.started() && e instanceof IOException) {
                        return; // the client went away mid-stream: nothing to report
                    }
                    System.err.println("error handling " + req.method + " " + req.path + ":");
                    e.printStackTrace();
                    if (!res.started()) {
                        res.send(500, "application/json", errorJson("server_error", "internal server error"));
                    } else {
                        // A stream that died must not look complete: unless the handler ended it itself (after
                        // reporting the error in-band), cut it off without the chunked terminator.
                        if (!res.ended()) {
                            res.abort();
                        }
                        return;
                    }
                }
                if (!keepAlive || !probe.nextRequest()) {
                    return;
                }
            }
        } catch (IOException ignored) {
            // client went away (or the read timeout expired)
        } finally {
            connections.release();
        }
    }

    /**
     * Overall deadline for receiving one request, started at its first byte: {@code readTimeoutMs} bounds each
     * individual read, this bounds the whole request (headers and body), so a client trickling one byte per
     * read timeout cannot hold a connection open indefinitely. 408 when exceeded.
     */
    private final class RequestClock {
        private long deadline; // nanoTime; 0 = no byte received yet

        void tick() throws HttpError {
            if (options.readTimeoutMs() <= 0) {
                return;
            }
            long now = System.nanoTime();
            if (deadline == 0) {
                deadline = now + options.readTimeoutMs() * 1_000_000L;
            } else if (now - deadline > 0) {
                throw new HttpError(408, "request not received within " + options.readTimeoutMs() + " ms");
            }
        }
    }

    /** Returns null at EOF before any request bytes. */
    private Request readRequest(InputStream in, OutputStream out) throws IOException {
        RequestClock clock = new RequestClock();
        String requestLine = readLine(in, clock);
        if (requestLine == null) {
            return null;
        }
        while (requestLine.isEmpty()) { // tolerate stray CRLFs between keep-alive requests
            requestLine = readLine(in, clock);
            if (requestLine == null) {
                return null;
            }
        }
        String[] parts = requestLine.split(" ");
        if (parts.length < 2 || parts.length > 3 || parts[0].isEmpty() || parts[1].isEmpty()) {
            throw new HttpError(400, "malformed request line");
        }
        boolean http10 = parts.length < 3 || !parts[2].equals("HTTP/1.1");
        if (parts.length == 3 && !parts[2].startsWith("HTTP/1.")) {
            throw new HttpError(400, "unsupported HTTP version");
        }
        Map<String, String> headers = new LinkedHashMap<>();
        int total = 0;
        String line;
        while ((line = readLine(in, clock)) != null && !line.isEmpty()) {
            total += line.length();
            if (total > MAX_HEADER) {
                throw new HttpError(431, "request headers too large");
            }
            int c = line.indexOf(':');
            if (c > 0) {
                headers.put(line.substring(0, c).trim().toLowerCase(Locale.ROOT), line.substring(c + 1).trim());
            }
        }
        if (line == null) {
            throw new EOFException();
        }
        byte[] body = new byte[0];
        String cl = headers.get("content-length");
        String te = headers.get("transfer-encoding");
        if (cl != null && te != null) { // ambiguous framing (request smuggling vector): refuse it
            throw new HttpError(400, "both Content-Length and Transfer-Encoding present");
        }
        boolean chunked = te != null;
        if (chunked && !te.equalsIgnoreCase("chunked")) {
            throw new HttpError(400, "unsupported Transfer-Encoding");
        }
        long len = 0;
        if (cl != null) {
            try {
                len = Long.parseLong(cl.trim());
            } catch (NumberFormatException e) {
                throw new HttpError(400, "malformed Content-Length");
            }
            if (len < 0) {
                throw new HttpError(400, "malformed Content-Length");
            }
            if (len > options.maxBodyBytes()) {
                throw new HttpError(413, "request body larger than " + options.maxBodyBytes() + " bytes");
            }
        }
        if ((cl != null || chunked) && "100-continue".equalsIgnoreCase(headers.get("expect"))) {
            // the client holds the body back until told to go ahead (curl does this for bodies > 1 KB)
            out.write("HTTP/1.1 100 Continue\r\n\r\n".getBytes(StandardCharsets.ISO_8859_1));
            out.flush();
        }
        if (cl != null) {
            body = readFully(in, (int) len, clock);
        } else if (chunked) {
            body = readChunked(in, clock);
        }
        return new Request(parts[0], parts[1], headers, body, http10);
    }

    private byte[] readChunked(InputStream in, RequestClock clock) throws IOException {
        java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
        while (true) {
            String sizeLine = readLine(in, clock);
            if (sizeLine == null) {
                throw new EOFException();
            }
            int semi = sizeLine.indexOf(';');
            long size;
            try {
                size = Long.parseLong((semi >= 0 ? sizeLine.substring(0, semi) : sizeLine).trim(), 16);
            } catch (NumberFormatException e) {
                throw new HttpError(400, "malformed chunk size");
            }
            if (size < 0) {
                throw new HttpError(400, "malformed chunk size");
            }
            if (size == 0) {
                while ((sizeLine = readLine(in, clock)) != null && !sizeLine.isEmpty()) {
                    // trailers ignored
                }
                return bos.toByteArray();
            }
            if (bos.size() + size > options.maxBodyBytes()) {
                throw new HttpError(413, "request body larger than " + options.maxBodyBytes() + " bytes");
            }
            bos.write(readFully(in, (int) size, clock));
            if (!"".equals(readLine(in, clock))) { // CRLF after chunk
                throw new HttpError(400, "malformed chunk");
            }
        }
    }

    /** Reads exactly {@code n} body bytes (EOF before that is an {@link EOFException}), checking the deadline. */
    private static byte[] readFully(InputStream in, int n, RequestClock clock) throws IOException {
        byte[] body = new byte[n];
        int off = 0;
        while (off < n) {
            int r = in.read(body, off, Math.min(8192, n - off));
            if (r < 0) {
                throw new EOFException();
            }
            off += r;
            clock.tick();
        }
        return body;
    }

    private static String readLine(InputStream in, RequestClock clock) throws IOException {
        StringBuilder sb = new StringBuilder(64);
        int c;
        while ((c = in.read()) >= 0) {
            clock.tick();
            if (c == '\n') {
                int n = sb.length();
                if (n > 0 && sb.charAt(n - 1) == '\r') {
                    sb.setLength(n - 1);
                }
                return sb.toString();
            }
            sb.append((char) c);
            if (sb.length() > MAX_HEADER) {
                throw new IOException("line too long");
            }
        }
        return sb.isEmpty() ? null : sb.toString();
    }

    private static String errorJson(String type, String message) {
        StringBuilder sb = new StringBuilder("{\"error\":{\"message\":\"");
        for (int i = 0; i < message.length(); i++) {
            char c = message.charAt(i);
            if (c == '"' || c == '\\') {
                sb.append('\\').append(c);
            } else if (c < 0x20) {
                sb.append(String.format("\\u%04x", (int) c));
            } else {
                sb.append(c);
            }
        }
        return sb.append("\",\"type\":\"").append(type).append("\"}}").toString();
    }

    private static String reason(int status) {
        return switch (status) {
            case 200 -> "OK";
            case 204 -> "No Content";
            case 400 -> "Bad Request";
            case 404 -> "Not Found";
            case 405 -> "Method Not Allowed";
            case 408 -> "Request Timeout";
            case 413 -> "Content Too Large";
            case 415 -> "Unsupported Media Type";
            case 431 -> "Request Header Fields Too Large";
            case 500 -> "Internal Server Error";
            case 503 -> "Service Unavailable";
            default -> "Status";
        };
    }

    @Override
    public void close() {
        closed = true;
        try {
            socket.close();
        } catch (IOException ignored) {
            // nothing to do
        }
        executor.shutdownNow();
    }
}
