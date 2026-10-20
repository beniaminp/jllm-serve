package org.robomiri.jllm.web.mcp;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.web.tools.Tool;
import org.robomiri.jllm.web.tools.ToolsConfig;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

/**
 * The MCP servers of the tool settings: connects them at startup, keeps their tool lists and exposes their tools
 * as {@code <server>__<tool>} functions. A server whose connection drops shows as failed until reconnected.
 */
@Service
public class McpManager implements DisposableBean {

    public static final String PROTOCOL_VERSION = "2025-06-18";
    static final String MASK = "••••••";
    private static final Duration INIT_TIMEOUT = Duration.ofSeconds(60); // npx / uvx may download the server first

    public enum State { DISABLED, CONNECTING, CONNECTED, FAILED }

    /** A tool as the server describes it. */
    public record ToolInfo(String name, String function, String description, Map<String, Object> inputSchema) {
    }

    /** One configured server and its connection. */
    public record ServerView(String name, String type, Map<String, Object> config, State state, String error,
                             String serverName, String serverVersion, String instructions, List<ToolInfo> tools,
                             long connectedAt) {
    }

    private final class Connection {
        final String name;
        final Map<String, Object> config;
        volatile State state = State.DISABLED;
        volatile String error;
        volatile McpTransport transport;
        volatile String serverName = "";
        volatile String serverVersion = "";
        volatile String instructions = "";
        volatile List<ToolInfo> tools = List.of();
        volatile long connectedAt;

        Connection(String name, Map<String, Object> config) {
            this.name = name;
            this.config = config;
        }

        ServerView view() {
            McpTransport t = transport;
            State s = state;
            String err = error;
            if (s == State.CONNECTED && t != null && t.isClosed()) {
                s = State.FAILED;
                err = t.closedReason();
            }
            return new ServerView(name, type(config), redact(config), s, err, serverName, serverVersion, instructions, tools, connectedAt);
        }
    }

    private final ToolsConfig config;
    private final Map<String, Connection> servers = new LinkedHashMap<>();

    public McpManager(ToolsConfig config) {
        this.config = config;
    }

    @EventListener(ApplicationReadyEvent.class)
    public void start() {
        config.mcpServers().forEach((name, entry) -> {
            Connection c = new Connection(name, entry);
            synchronized (this) {
                servers.put(name, c);
            }
            if (!Json.bool(entry, "disabled", false)) {
                Thread.ofVirtual().name("mcp-connect-" + name).start(() -> connect(c));
            }
        });
    }

    public synchronized List<ServerView> list() {
        return servers.values().stream().map(Connection::view).toList();
    }

    public synchronized ServerView get(String name) {
        Connection c = servers.get(name);
        if (c == null) {
            throw new NoSuchElementException("no MCP server " + name);
        }
        return c.view();
    }

    /** Adds or replaces a server (Claude Desktop format) and connects it unless disabled. */
    public ServerView put(String name, Map<String, Object> entry) {
        validate(name, entry);
        Map<String, Object> previous = config.mcpServers().get(name);
        for (String k : new String[]{"env", "headers"}) { // masked values the UI sent back keep their stored value
            Map<String, Object> now = Json.map(entry, k);
            Map<String, Object> was = Json.map(previous, k);
            if (now != null && was != null) {
                now.replaceAll((kk, v) -> MASK.equals(v) && was.containsKey(kk) ? was.get(kk) : v);
            }
        }
        config.putMcpServer(name, entry);
        Connection old;
        Connection c = new Connection(name, entry);
        synchronized (this) {
            old = servers.put(name, c);
        }
        if (old != null) {
            disconnect(old);
        }
        if (!Json.bool(entry, "disabled", false)) {
            connect(c);
        }
        return c.view();
    }

    public void remove(String name) {
        Connection c;
        synchronized (this) {
            c = servers.remove(name);
        }
        if (c == null) {
            throw new NoSuchElementException("no MCP server " + name);
        }
        disconnect(c);
        config.removeMcpServer(name);
    }

    public ServerView reconnect(String name) {
        Connection c;
        synchronized (this) {
            c = servers.get(name);
        }
        if (c == null) {
            throw new NoSuchElementException("no MCP server " + name);
        }
        disconnect(c);
        connect(c);
        return c.view();
    }

    /** The tools of the connected servers. */
    public synchronized List<Tool> tools() {
        List<Tool> out = new ArrayList<>();
        for (Connection c : servers.values()) {
            if (c.state != State.CONNECTED || c.transport == null || c.transport.isClosed()) {
                continue;
            }
            for (ToolInfo t : c.tools) {
                out.add(new McpTool(c, t));
            }
        }
        return out;
    }

    static void validate(String name, Map<String, Object> entry) {
        if (name == null || !name.matches("[A-Za-z0-9_-]{1,32}")) {
            throw new IllegalArgumentException("server names are 1 to 32 letters, digits, '_' or '-'");
        }
        String type = type(entry);
        switch (type) {
            case "stdio" -> {
                if (Json.str(entry, "command", "").isBlank()) {
                    throw new IllegalArgumentException(name + ": a stdio server needs a \"command\"");
                }
            }
            case "http", "sse" -> {
                String url = Json.str(entry, "url", "");
                if (!url.matches("(?i)https?://.+")) {
                    throw new IllegalArgumentException(name + ": \"url\" must be an http(s) URL");
                }
            }
            default -> throw new IllegalArgumentException(name + ": type must be stdio, http or sse");
        }
    }

    /** {@code type}, else stdio for a command, sse for a URL ending in /sse, else http. */
    static String type(Map<String, Object> entry) {
        String t = Json.str(entry, "type", Json.str(entry, "transport", "")).toLowerCase(Locale.ROOT);
        if (t.equals("streamable-http") || t.equals("streamablehttp") || t.equals("streamable_http")) {
            return "http";
        }
        if (!t.isEmpty()) {
            return t;
        }
        if (!Json.str(entry, "command", "").isBlank()) {
            return "stdio";
        }
        return Json.str(entry, "url", "").replaceAll("[?#].*", "").endsWith("/sse") ? "sse" : "http";
    }

    /** The configuration without secret values (env and header values are masked). */
    static Map<String, Object> redact(Map<String, Object> entry) {
        Map<String, Object> out = new LinkedHashMap<>(entry);
        for (String k : new String[]{"env", "headers"}) {
            Map<String, Object> m = Json.map(entry, k);
            if (m != null) {
                Map<String, Object> masked = new LinkedHashMap<>();
                m.forEach((kk, v) -> masked.put(kk, String.valueOf(v).isEmpty() ? "" : MASK));
                out.put(k, masked);
            }
        }
        return out;
    }

    private void connect(Connection c) {
        c.state = State.CONNECTING;
        c.error = null;
        McpTransport t = null;
        try {
            t = open(c.name, c.config);
            c.transport = t;
            Map<String, Object> init = t.request("initialize", Json.obj(
                    "protocolVersion", PROTOCOL_VERSION,
                    "capabilities", Json.obj(),
                    "clientInfo", Json.obj("name", "jllm-web", "version", "1.0")), INIT_TIMEOUT);
            String version = Json.str(init, "protocolVersion", PROTOCOL_VERSION);
            if (t instanceof HttpTransport h) {
                h.setProtocolVersion(version);
            }
            Map<String, Object> info = Json.map(init, "serverInfo");
            c.serverName = Json.str(info, "name", "");
            c.serverVersion = Json.str(info, "version", "");
            c.instructions = Json.str(init, "instructions", "");
            t.notify("notifications/initialized", null);
            McpTransport tt = t;
            t.onNotification(method -> {
                if (method.equals("notifications/tools/list_changed")) {
                    Thread.ofVirtual().start(() -> {
                        try {
                            c.tools = listTools(c.name, tt);
                        } catch (IOException e) {
                            System.err.println("[mcp " + c.name + "] refreshing tools: " + e.getMessage());
                        }
                    });
                }
            });
            c.tools = listTools(c.name, t);
            c.connectedAt = System.currentTimeMillis();
            c.state = State.CONNECTED;
            System.err.println("MCP server " + c.name + " connected (" + c.serverName + " " + c.serverVersion + ", "
                    + c.tools.size() + " tools)");
        } catch (Exception e) {
            c.state = State.FAILED;
            c.error = e.getMessage() == null ? e.toString() : e.getMessage();
            System.err.println("MCP server " + c.name + ": " + c.error);
            if (t != null) {
                t.close();
            }
            c.transport = null;
        }
    }

    private static void disconnect(Connection c) {
        McpTransport t = c.transport;
        c.transport = null;
        c.tools = List.of();
        c.state = State.DISABLED;
        if (t != null) {
            t.close();
        }
    }

    @SuppressWarnings("unchecked")
    private static McpTransport open(String name, Map<String, Object> entry) throws IOException {
        Map<String, String> headers = new LinkedHashMap<>();
        Map<String, Object> h = Json.map(entry, "headers");
        if (h != null) {
            h.forEach((k, v) -> headers.put(k, String.valueOf(v)));
        }
        return switch (type(entry)) {
            case "stdio" -> {
                List<String> args = new ArrayList<>();
                List<Object> a = Json.list(entry, "args");
                if (a != null) {
                    a.forEach(x -> args.add(String.valueOf(x)));
                }
                Map<String, String> env = new LinkedHashMap<>();
                Map<String, Object> e = Json.map(entry, "env");
                if (e != null) {
                    e.forEach((k, v) -> env.put(k, String.valueOf(v)));
                }
                yield new StdioTransport(name, Json.str(entry, "command", ""), args, env, Json.str(entry, "cwd", null));
            }
            case "sse" -> new SseTransport(URI.create(Json.str(entry, "url", "")), headers);
            default -> new HttpTransport(URI.create(Json.str(entry, "url", "")), headers);
        };
    }

    @SuppressWarnings("unchecked")
    private static List<ToolInfo> listTools(String server, McpTransport t) throws IOException {
        List<ToolInfo> out = new ArrayList<>();
        Set<String> used = new HashSet<>();
        String cursor = null;
        String prefix = server.replaceAll("[^A-Za-z0-9_-]", "_");
        do {
            Map<String, Object> r = t.request("tools/list", cursor == null ? Json.obj() : Json.obj("cursor", cursor), Duration.ofSeconds(30));
            List<Object> tools = Json.list(r, "tools");
            if (tools != null) {
                for (Object o : tools) {
                    String name = Json.str(o, "name", "");
                    if (name.isEmpty()) {
                        continue;
                    }
                    String fn = prefix + "__" + name.replaceAll("[^A-Za-z0-9_-]", "_");
                    if (fn.length() > 64) {
                        fn = fn.substring(0, 64);
                    }
                    for (int i = 2; !used.add(fn); i++) {
                        fn = fn.substring(0, Math.min(fn.length(), 60)) + "_" + i;
                    }
                    Map<String, Object> schema = Json.map(o, "inputSchema");
                    out.add(new ToolInfo(name, fn, Json.str(o, "description", ""),
                            schema == null ? Json.obj("type", "object", "properties", Json.obj()) : schema));
                }
            }
            cursor = Json.str(r, "nextCursor", null);
        } while (cursor != null && !cursor.isEmpty());
        return out;
    }

    /** An MCP tool as a function the model can call. */
    private record McpTool(Connection c, ToolInfo info) implements Tool {
        @Override
        public String name() {
            return info.function();
        }

        @Override
        public String description() {
            String d = info.description().isBlank() ? info.name() : info.description();
            return d + " (MCP server " + c.name + ")";
        }

        @Override
        public Map<String, Object> parameters() {
            return info.inputSchema();
        }

        @Override
        public String source() {
            return c.name;
        }

        @Override
        public String call(Map<String, Object> args) throws Exception {
            McpTransport t = c.transport;
            if (t == null || t.isClosed()) {
                throw new IOException("MCP server " + c.name + " is not connected");
            }
            Map<String, Object> r = t.request("tools/call", Json.obj("name", info.name(), "arguments", args), Duration.ofMinutes(5));
            String text = contentText(r);
            if (Json.bool(r, "isError", false)) {
                throw new IOException(text.isBlank() ? "the tool reported an error" : text);
            }
            return text;
        }
    }

    /** The text of a {@code tools/call} result: text parts, resource text, and placeholders for binary content. */
    static String contentText(Map<String, Object> result) {
        StringBuilder sb = new StringBuilder();
        List<Object> content = Json.list(result, "content");
        if (content != null) {
            for (Object p : content) {
                String type = Json.str(p, "type", "text");
                String piece = switch (type) {
                    case "text" -> Json.str(p, "text", "");
                    case "image", "audio" -> "[" + type + ": " + Json.str(p, "mimeType", "") + ", not shown]";
                    case "resource" -> {
                        Map<String, Object> res = Json.map(p, "resource");
                        String t = Json.str(res, "text", null);
                        yield t != null ? t : "[resource " + Json.str(res, "uri", "") + "]";
                    }
                    case "resource_link" -> "[" + Json.str(p, "name", "resource") + ": " + Json.str(p, "uri", "") + "]";
                    default -> "";
                };
                if (!piece.isEmpty()) {
                    sb.append(sb.isEmpty() ? "" : "\n").append(piece);
                }
            }
        }
        if (sb.isEmpty() && result.get("structuredContent") != null) {
            sb.append(Json.write(result.get("structuredContent")));
        }
        return sb.toString();
    }

    @Override
    public void destroy() {
        for (Connection c : new ArrayList<>(servers.values())) {
            disconnect(c);
        }
    }
}
