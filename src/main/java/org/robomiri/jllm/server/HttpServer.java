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

/**
 * A small HTTP/1.1 server on plain sockets, one virtual thread per connection. Supports keep-alive,
 * Content-Length request bodies and chunked streaming responses (used for SSE).
 */
public final class HttpServer implements AutoCloseable {

    private static final int MAX_HEADER = 64 * 1024;
    private static final long MAX_BODY = 32L * 1024 * 1024;

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
        private boolean started;
        private boolean streaming;
        private boolean ended;

        Response(OutputStream out, boolean keepAlive) {
            this.out = out;
            this.keepAlive = keepAlive;
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
            h.append("Access-Control-Allow-Origin: *\r\n");
            h.append("Connection: ").append(keepAlive ? "keep-alive" : "close").append("\r\n\r\n");
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
            String h = "HTTP/1.1 " + status + " " + reason(status) + "\r\n"
                    + "Content-Type: " + contentType + "\r\n"
                    + "Transfer-Encoding: chunked\r\n"
                    + "Cache-Control: no-cache\r\n"
                    + "Access-Control-Allow-Origin: *\r\n"
                    + "Connection: " + (keepAlive ? "keep-alive" : "close") + "\r\n\r\n";
            out.write(h.getBytes(StandardCharsets.ISO_8859_1));
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
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    private volatile boolean closed;

    public HttpServer(String host, int port, Handler handler) throws IOException {
        this.handler = handler;
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
            Socket s;
            try {
                s = socket.accept();
            } catch (IOException e) {
                if (closed) {
                    return;
                }
                continue;
            }
            executor.submit(() -> connection(s));
        }
    }

    private void connection(Socket s) {
        try (s) {
            s.setTcpNoDelay(true);
            InputStream in = new BufferedInputStream(s.getInputStream());
            OutputStream out = new BufferedOutputStream(s.getOutputStream(), 1 << 16);
            while (!closed) {
                Request req = readRequest(in);
                if (req == null) {
                    return;
                }
                boolean keepAlive = !"close".equalsIgnoreCase(req.headers.getOrDefault("connection", ""));
                Response res = new Response(out, keepAlive);
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
            // client went away
        }
    }

    /** Returns null at EOF before any request bytes. */
    private static Request readRequest(InputStream in) throws IOException {
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
        if (cl != null) {
            long len = Long.parseLong(cl.trim());
            if (len < 0 || len > MAX_BODY) {
                throw new IOException("body too large");
            }
            body = in.readNBytes((int) len);
            if (body.length != len) {
                throw new EOFException();
            }
        } else if ("chunked".equalsIgnoreCase(headers.get("transfer-encoding"))) {
            body = readChunked(in);
        }
        return new Request(parts[0], parts[1], headers, body);
    }

    private static byte[] readChunked(InputStream in) throws IOException {
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
            if (bos.size() + size > MAX_BODY) {
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
