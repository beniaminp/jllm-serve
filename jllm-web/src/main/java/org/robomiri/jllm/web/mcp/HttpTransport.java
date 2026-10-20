package org.robomiri.jllm.web.mcp;

import org.robomiri.jllm.json.Json;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Map;

/**
 * The Streamable HTTP transport (MCP 2025-03-26 and later): every message is a POST; the answer is either a JSON
 * body or an event stream that carries it. The session id the server assigns at initialization goes with every
 * later request.
 */
public final class HttpTransport extends McpTransport {

    static final HttpClient HTTP = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(15))
            .followRedirects(HttpClient.Redirect.NORMAL).build();

    private final URI url;
    private final Map<String, String> headers;
    private volatile String session;
    private volatile String protocolVersion;

    public HttpTransport(URI url, Map<String, String> headers) {
        this.url = url;
        this.headers = Map.copyOf(headers);
    }

    void setProtocolVersion(String v) {
        this.protocolVersion = v;
    }

    @Override
    protected void send(Map<String, Object> message, long id) throws IOException {
        HttpRequest.Builder b = HttpRequest.newBuilder(url).timeout(Duration.ofMinutes(10))
                .header("Content-Type", "application/json")
                .header("Accept", "application/json, text/event-stream")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(message)));
        headers.forEach(b::header);
        if (session != null) {
            b.header("Mcp-Session-Id", session);
        }
        if (protocolVersion != null) {
            b.header("MCP-Protocol-Version", protocolVersion);
        }
        HttpResponse<InputStream> r;
        try {
            r = HTTP.send(b.build(), HttpResponse.BodyHandlers.ofInputStream());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted");
        }
        r.headers().firstValue("mcp-session-id").ifPresent(s -> session = s);
        int sc = r.statusCode();
        String type = r.headers().firstValue("content-type").orElse("");
        try (InputStream in = r.body()) {
            if (sc == 202 || sc == 204) {
                return;
            }
            if (sc >= 400) {
                String body = new String(in.readNBytes(2000), StandardCharsets.UTF_8);
                throw new IOException("HTTP " + sc + (sc == 401 || sc == 403 ? " (check the server's authorization header)" : "")
                        + (sc == 404 && session != null ? " (session expired: reconnect)" : "") + (body.isBlank() ? "" : ": " + body.strip()));
            }
            if (type.startsWith("text/event-stream")) {
                readEvents(in, id);
            } else {
                String body = new String(in.readAllBytes(), StandardCharsets.UTF_8);
                if (!body.isBlank()) {
                    dispatch(Json.parse(body));
                }
            }
        }
    }

    /** Reads events until the response to {@code id} has come (the server may close the stream right after). */
    private void readEvents(InputStream in, long id) throws IOException {
        BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
        StringBuilder data = new StringBuilder();
        for (String line; (line = r.readLine()) != null; ) {
            if (line.isEmpty()) {
                if (!data.isEmpty()) {
                    Object msg = Json.parse(data.toString());
                    data.setLength(0);
                    dispatch(msg);
                    if (msg instanceof Map<?, ?> m && m.get("id") instanceof Number n && n.longValue() == id && m.get("method") == null) {
                        return;
                    }
                }
            } else if (line.startsWith("data:")) {
                data.append(data.isEmpty() ? "" : "\n").append(line.substring(5).stripLeading());
            }
        }
        if (!data.isEmpty()) {
            dispatch(Json.parse(data.toString()));
        }
    }

    @Override
    public void close() {
        if (session != null) {
            try {
                HttpRequest.Builder b = HttpRequest.newBuilder(url).timeout(Duration.ofSeconds(5)).DELETE().header("Mcp-Session-Id", session);
                headers.forEach(b::header);
                HTTP.send(b.build(), HttpResponse.BodyHandlers.discarding());
            } catch (IOException | InterruptedException | RuntimeException ignored) {
                // best effort
            }
        }
        closed(new IOException("closed"));
    }
}
