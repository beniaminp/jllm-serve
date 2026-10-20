package org.robomiri.jllm.web.mcp;

import org.robomiri.jllm.json.Json;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * The older HTTP+SSE transport (MCP 2024-11-05), still common: a long-lived GET event stream announces the URL to
 * POST messages to (the {@code endpoint} event) and carries every response.
 */
public final class SseTransport extends McpTransport {

    private final URI url;
    private final Map<String, String> headers;
    private final CompletableFuture<URI> endpoint = new CompletableFuture<>();
    private volatile InputStream stream;

    public SseTransport(URI url, Map<String, String> headers) throws IOException {
        this.url = url;
        this.headers = Map.copyOf(headers);
        HttpRequest.Builder b = HttpRequest.newBuilder(url).header("Accept", "text/event-stream").GET();
        headers.forEach(b::header);
        HttpResponse<InputStream> r;
        try {
            r = HttpTransport.HTTP.send(b.build(), HttpResponse.BodyHandlers.ofInputStream());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted");
        }
        if (r.statusCode() != 200) {
            r.body().close();
            throw new IOException("HTTP " + r.statusCode() + " from " + url);
        }
        stream = r.body();
        Thread.ofVirtual().name("mcp-sse").start(this::read);
        try {
            endpoint.get(15, TimeUnit.SECONDS);
        } catch (Exception e) {
            close();
            throw new IOException("the server sent no endpoint event (is this an SSE MCP server? try the http type)");
        }
    }

    private void read() {
        try (BufferedReader r = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String event = "message";
            StringBuilder data = new StringBuilder();
            for (String line; (line = r.readLine()) != null; ) {
                if (line.isEmpty()) {
                    if (!data.isEmpty()) {
                        if (event.equals("endpoint")) {
                            endpoint.complete(url.resolve(data.toString().strip()));
                        } else {
                            try {
                                dispatch(Json.parse(data.toString()));
                            } catch (RuntimeException ignored) {
                                // not JSON-RPC
                            }
                        }
                    }
                    event = "message";
                    data.setLength(0);
                } else if (line.startsWith("event:")) {
                    event = line.substring(6).strip();
                } else if (line.startsWith("data:")) {
                    data.append(data.isEmpty() ? "" : "\n").append(line.substring(5).stripLeading());
                }
            }
            closed(new IOException("the server closed the event stream"));
        } catch (IOException e) {
            closed(e);
        }
    }

    @Override
    protected void send(Map<String, Object> message, long id) throws IOException {
        HttpRequest.Builder b = HttpRequest.newBuilder(endpoint.join()).timeout(Duration.ofSeconds(60))
                .header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(Json.write(message)));
        headers.forEach(b::header);
        HttpResponse<String> r;
        try {
            r = HttpTransport.HTTP.send(b.build(), HttpResponse.BodyHandlers.ofString());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted");
        }
        if (r.statusCode() >= 400) {
            throw new IOException("HTTP " + r.statusCode() + ": " + r.body());
        }
    }

    @Override
    public void close() {
        try {
            InputStream s = stream;
            if (s != null) {
                s.close();
            }
        } catch (IOException ignored) {
            // closed
        }
        closed(new IOException("closed"));
    }
}
