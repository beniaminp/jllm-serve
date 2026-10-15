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
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
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
     * @param maxConnections simultaneous connections; further ones wait in the kernel accept queue
     * @param maxBodyBytes   largest accepted request body
     * @param corsOrigin     value of {@code Access-Control-Allow-Origin}; null or empty = no CORS headers at all
     */
    public record Options(int readTimeoutMs, int maxConnections, long maxBodyBytes, String corsOrigin) {
        public Options {
            if (readTimeoutMs < 0 || maxConnections < 1 || maxBodyBytes < 0) {
                throw new IllegalArgumentException("invalid server options: " + this);
            }
            if (corsOrigin != null && corsOrigin.isBlank()) {
                corsOrigin = null;
            }
        }

        public static Options defaults() {
            return new Options(30_000, 256, 4L * 1024 * 1024, null);
        }
    }

    public static final class Request {
        public final String method;
        public final String path;
        public final Map<String, String> headers;
        public final byte[] body;

        Request(String method, String path, Map<String, String> headers, byte[] body) {
            this.method = method;
            this.path = path;
            this.headers = headers;
            this.body = body;
        }

        public String bodyText() {
            return new String(body, StandardCharsets.UTF_8);
        }
    }

    /** Response writer; either call {@link #send} once, or {@link #startStream} followed by {@link #write}/{@link #end}. */
    public static final class Response {
        private final OutputStream out;
        private final boolean keepAlive;
        private final String corsOrigin;
        private final StringBuilder extraHeaders = new StringBuilder();
        private boolean started;
        private boolean streaming;
        private boolean ended;

        Response(OutputStream out, boolean keepAlive, String corsOrigin) {
            this.out = out;
            this.keepAlive = keepAlive;
            this.corsOrigin = corsOrigin;
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
            out.write(body);
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
            h.append("Transfer-Encoding: chunked\r\n");
            h.append("Cache-Control: no-cache\r\n");
            commonHeaders(h);
            out.write(h.toString().getBytes(StandardCharsets.ISO_8859_1));
            out.flush();
        }

        public void write(String chunk) throws IOException {
            byte[] b = chunk.getBytes(StandardCharsets.UTF_8);
            if (b.length == 0) {
                return;
            }
            out.write((Integer.toHexString(b.length) + "\r\n").getBytes(StandardCharsets.ISO_8859_1));
            out.write(b);
            out.write("\r\n".getBytes(StandardCharsets.ISO_8859_1));
            out.flush();
        }

        public void end() throws IOException {
            if (streaming && !ended) {
                out.write("0\r\n\r\n".getBytes(StandardCharsets.ISO_8859_1));
                out.flush();
            }
            ended = true;
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
                Request req = readRequest(in, out);
                if (req == null) {
                    return;
                }
                boolean keepAlive = !"close".equalsIgnoreCase(req.headers.getOrDefault("connection", ""));
                Response res = new Response(out, keepAlive, options.corsOrigin());
                try {
                    handler.handle(req, res);
                    if (!res.started()) {
                        res.send(404, "text/plain", "not found");
                    }
                } catch (Exception e) {
                    if (!res.started()) {
                        res.send(500, "text/plain", "internal error: " + e);
                    } else {
                        return; // can't recover a half-written response; drop the connection
                    }
                } finally {
                    if (res.started() && !res.ended()) {
                        res.end();
                    }
                }
                if (!keepAlive) {
                    return;
                }
            }
        } catch (IOException ignored) {
            // client went away (or the read timeout expired)
        } finally {
            connections.release();
        }
    }

    /** Returns null at EOF before any request bytes. */
    private Request readRequest(InputStream in, OutputStream out) throws IOException {
        String requestLine = readLine(in);
        if (requestLine == null) {
            return null;
        }
        while (requestLine.isEmpty()) { // tolerate stray CRLFs between keep-alive requests
            requestLine = readLine(in);
            if (requestLine == null) {
                return null;
            }
        }
        String[] parts = requestLine.split(" ");
        if (parts.length < 2) {
            throw new IOException("bad request line");
        }
        Map<String, String> headers = new LinkedHashMap<>();
        int total = 0;
        String line;
        while ((line = readLine(in)) != null && !line.isEmpty()) {
            total += line.length();
            if (total > MAX_HEADER) {
                throw new IOException("headers too large");
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
        boolean chunked = "chunked".equalsIgnoreCase(headers.get("transfer-encoding"));
        if ((cl != null || chunked) && "100-continue".equalsIgnoreCase(headers.get("expect"))) {
            // the client holds the body back until told to go ahead (curl does this for bodies > 1 KB)
            out.write("HTTP/1.1 100 Continue\r\n\r\n".getBytes(StandardCharsets.ISO_8859_1));
            out.flush();
        }
        if (cl != null) {
            long len = Long.parseLong(cl.trim());
            if (len < 0 || len > options.maxBodyBytes()) {
                throw new IOException("body too large");
            }
            body = in.readNBytes((int) len);
            if (body.length != len) {
                throw new EOFException();
            }
        } else if (chunked) {
            body = readChunked(in);
        }
        return new Request(parts[0], parts[1], headers, body);
    }

    private byte[] readChunked(InputStream in) throws IOException {
        java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
        while (true) {
            String sizeLine = readLine(in);
            if (sizeLine == null) {
                throw new EOFException();
            }
            int semi = sizeLine.indexOf(';');
            int size = Integer.parseInt((semi >= 0 ? sizeLine.substring(0, semi) : sizeLine).trim(), 16);
            if (size == 0) {
                while ((sizeLine = readLine(in)) != null && !sizeLine.isEmpty()) {
                    // trailers ignored
                }
                return bos.toByteArray();
            }
            if (bos.size() + size > options.maxBodyBytes()) {
                throw new IOException("body too large");
            }
            byte[] chunk = in.readNBytes(size);
            if (chunk.length != size) {
                throw new EOFException();
            }
            bos.write(chunk);
            readLine(in); // CRLF after chunk
        }
    }

    private static String readLine(InputStream in) throws IOException {
        StringBuilder sb = new StringBuilder(64);
        int c;
        while ((c = in.read()) >= 0) {
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

    private static String reason(int status) {
        return switch (status) {
            case 200 -> "OK";
            case 204 -> "No Content";
            case 400 -> "Bad Request";
            case 404 -> "Not Found";
            case 405 -> "Method Not Allowed";
            case 413 -> "Content Too Large";
            case 415 -> "Unsupported Media Type";
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
