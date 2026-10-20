package org.robomiri.jllm.web.mcp;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class McpTransportTest {

    private static final Duration T = Duration.ofSeconds(60);

    @Test
    void stdio() throws Exception {
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        String src = Path.of("src/test/resources/mcp/FakeMcpServer.java").toAbsolutePath().toString();
        try (StdioTransport t = new StdioTransport("fake", java, List.of(src), Map.of(), null)) {
            Map<String, Object> init = t.request("initialize", Json.obj("protocolVersion", McpManager.PROTOCOL_VERSION), T);
            assertEquals("fake", Json.str(Json.map(init, "serverInfo"), "name", ""));
            t.notify("notifications/initialized", null);
            List<Object> tools = Json.list(t.request("tools/list", Json.obj(), T), "tools");
            assertEquals(2, tools.size());
            Map<String, Object> r = t.request("tools/call", Json.obj("name", "echo", "arguments", Json.obj("text", "hi")), T);
            assertEquals("echo: hi", McpManager.contentText(r));
            Map<String, Object> f = t.request("tools/call", Json.obj("name", "fail", "arguments", Json.obj()), T);
            assertTrue(Json.bool(f, "isError", false));
            assertThrows(McpTransport.RpcException.class, () -> t.request("nope", null, T));
        }
    }

    /** Streamable HTTP: JSON answers, an event-stream answer and the session header. */
    @Test
    void http() throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        String[] seenSession = new String[1];
        server.createContext("/mcp", ex -> {
            String body = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            Map<String, Object> msg = Json.parseObject(body);
            String method = Json.str(msg, "method", "");
            Object id = msg.get("id");
            if (!method.equals("initialize")) {
                seenSession[0] = ex.getRequestHeaders().getFirst("Mcp-Session-Id");
            }
            if (id == null) {
                ex.sendResponseHeaders(202, -1);
                ex.close();
                return;
            }
            Object result = switch (method) {
                case "initialize" -> Json.obj("protocolVersion", "2025-06-18", "serverInfo", Json.obj("name", "web", "version", "1"));
                case "tools/list" -> Json.obj("tools", List.of(Json.obj("name", "add", "inputSchema", Json.obj("type", "object"))));
                default -> Json.obj("content", List.of(Json.obj("type", "text", "text", "3")));
            };
            String answer = Json.write(Json.obj("jsonrpc", "2.0", "id", id, "result", result));
            if (method.equals("initialize")) {
                ex.getResponseHeaders().add("Mcp-Session-Id", "s-42");
            }
            byte[] out;
            if (method.equals("tools/call")) {
                ex.getResponseHeaders().add("Content-Type", "text/event-stream");
                out = ("event: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\"}\n\ndata: " + answer + "\n\n")
                        .getBytes(StandardCharsets.UTF_8);
            } else {
                ex.getResponseHeaders().add("Content-Type", "application/json");
                out = answer.getBytes(StandardCharsets.UTF_8);
            }
            ex.sendResponseHeaders(200, out.length);
            try (OutputStream os = ex.getResponseBody()) {
                os.write(out);
            }
        });
        server.start();
        try (HttpTransport t = new HttpTransport(URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/mcp"), Map.of())) {
            t.request("initialize", Json.obj(), T);
            t.notify("notifications/initialized", null);
            assertEquals("s-42", seenSession[0]);
            assertEquals(1, Json.list(t.request("tools/list", Json.obj(), T), "tools").size());
            assertEquals("3", McpManager.contentText(t.request("tools/call", Json.obj("name", "add"), T)));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void configShapes() {
        assertEquals("stdio", McpManager.type(Map.of("command", "npx")));
        assertEquals("sse", McpManager.type(Map.of("url", "https://x/sse")));
        assertEquals("http", McpManager.type(Map.of("url", "https://x/mcp")));
        assertEquals("http", McpManager.type(Map.of("url", "https://x/sse", "type", "streamable-http")));
        assertThrows(IllegalArgumentException.class, () -> McpManager.validate("bad name", Map.of("command", "x")));
        assertThrows(IllegalArgumentException.class, () -> McpManager.validate("ok", Map.of("url", "ftp://x")));
        Map<String, Object> red = McpManager.redact(Map.of("command", "x", "env", Map.of("TOKEN", "secret")));
        assertEquals(Map.of("TOKEN", McpManager.MASK), red.get("env"));
    }

    @Test
    void contentText() throws IOException {
        assertEquals("a\n[image: image/png, not shown]\nres", McpManager.contentText(Json.obj("content", List.of(
                Json.obj("type", "text", "text", "a"), Json.obj("type", "image", "mimeType", "image/png", "data", "..."),
                Json.obj("type", "resource", "resource", Json.obj("uri", "x://y", "text", "res"))))));
    }
}
