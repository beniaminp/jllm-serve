package org.robomiri.jllm.web.mcp;

import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/**
 * JSON-RPC 2.0 to an MCP server over one of its transports. Subclasses send messages; responses (however they
 * arrive) go through {@link #dispatch}, which completes the waiting request, answers the server's own requests
 * ({@code ping}) and hands notifications to the listener.
 */
public abstract class McpTransport implements AutoCloseable {

    /** An error response from the server. */
    public static final class RpcException extends IOException {
        public final long code;

        RpcException(long code, String message) {
            super(message);
            this.code = code;
        }
    }

    private final AtomicLong ids = new AtomicLong();
    private final Map<Long, CompletableFuture<Map<String, Object>>> pending = new ConcurrentHashMap<>();
    private volatile Consumer<String> notifications = m -> { };
    private volatile IOException closedBecause;

    public void onNotification(Consumer<String> listener) {
        this.notifications = listener;
    }

    /** Sends a request and waits for its result. */
    public Map<String, Object> request(String method, Map<String, Object> params, Duration timeout) throws IOException {
        if (closedBecause != null) {
            throw new IOException("connection closed: " + closedBecause.getMessage(), closedBecause);
        }
        long id = ids.incrementAndGet();
        CompletableFuture<Map<String, Object>> f = new CompletableFuture<>();
        pending.put(id, f);
        try {
            Map<String, Object> msg = Json.obj("jsonrpc", "2.0", "id", id, "method", method);
            if (params != null) {
                msg.put("params", params);
            }
            send(msg, id);
            Map<String, Object> resp = f.get(timeout.toMillis(), TimeUnit.MILLISECONDS);
            Map<String, Object> err = Json.map(resp, "error");
            if (err != null) {
                throw new RpcException(Json.num(err, "code", 0), Json.str(err, "message", "error") );
            }
            Map<String, Object> result = Json.map(resp, "result");
            return result == null ? Map.of() : result;
        } catch (TimeoutException e) {
            throw new IOException(method + ": no answer within " + timeout.toSeconds() + " s");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(method + ": interrupted");
        } catch (ExecutionException e) {
            Throwable c = e.getCause();
            throw c instanceof IOException io ? io : new IOException(c.getMessage(), c);
        } finally {
            pending.remove(id);
        }
    }

    public void notify(String method, Map<String, Object> params) throws IOException {
        Map<String, Object> msg = Json.obj("jsonrpc", "2.0", "method", method);
        if (params != null) {
            msg.put("params", params);
        }
        send(msg, -1);
    }

    /**
     * Writes one message.
     *
     * @param id the request id, -1 for notifications and responses
     */
    protected abstract void send(Map<String, Object> message, long id) throws IOException;

    /** Routes a message received from the server. */
    @SuppressWarnings("unchecked")
    protected void dispatch(Object msg) {
        if (msg instanceof java.util.List<?> batch) {
            batch.forEach(this::dispatch);
            return;
        }
        if (!(msg instanceof Map<?, ?> raw)) {
            return;
        }
        Map<String, Object> m = (Map<String, Object>) raw;
        Object id = m.get("id");
        String method = Json.str(m, "method", null);
        if (method == null && id instanceof Number n) {
            CompletableFuture<Map<String, Object>> f = pending.get(n.longValue());
            if (f != null) {
                f.complete(m);
            }
        } else if (method != null && id != null) { // a request from the server
            Map<String, Object> reply = Json.obj("jsonrpc", "2.0", "id", id);
            switch (method) {
                case "ping" -> reply.put("result", Json.obj());
                case "roots/list" -> reply.put("result", Json.obj("roots", java.util.List.of()));
                default -> reply.put("error", Json.obj("code", -32601, "message", "method not supported by jllm-web: " + method));
            }
            try {
                send(reply, -1);
            } catch (IOException ignored) {
                // the connection is going away; the next request reports it
            }
        } else if (method != null) {
            notifications.accept(method);
        }
    }

    /** Fails every waiting request: the connection is gone. */
    protected void closed(IOException why) {
        closedBecause = why;
        for (CompletableFuture<Map<String, Object>> f : pending.values()) {
            f.completeExceptionally(why);
        }
    }

    public boolean isClosed() {
        return closedBecause != null;
    }

    public String closedReason() {
        return closedBecause == null ? null : closedBecause.getMessage();
    }

    @Override
    public abstract void close();
}
