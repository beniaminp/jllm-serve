package org.robomiri.jllm.web.tools;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.web.WebProperties;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The tool settings, stored as JSON in {@code jllm.web.tools-file}:
 * <pre>{@code
 * {
 *   "search": {"provider": "auto", "searxngUrl": "", "braveApiKey": "", "maxResults": 5},
 *   "fetch":  {"allowPrivateAddresses": false, "maxChars": 8000},
 *   "agent":  {"maxRounds": 6, "maxResultChars": 8000, "timeoutSeconds": 60},
 *   "disabledBuiltins": [],
 *   "mcpServers": {"name": {"command": "npx", "args": [...], "env": {}} or {"url": "https://...", "headers": {}}}
 * }}</pre>
 * {@code mcpServers} uses the format of Claude Desktop / Cursor configuration files, so existing entries can be
 * pasted as they are ({@code "type": "stdio" | "http" | "sse"} is optional; {@code "disabled": true} keeps an entry
 * without connecting it).
 */
@Service
public class ToolsConfig {

    // boxed so that a request may leave fields out (they keep their defaults)
    public record Search(String provider, String searxngUrl, String braveApiKey, Integer maxResults) {
    }

    public record Fetch(Boolean allowPrivateAddresses, Integer maxChars) {
    }

    public record Agent(Integer maxRounds, Integer maxResultChars, Integer timeoutSeconds) {
    }

    /** What the UI edits (MCP servers are managed separately). */
    public record General(Search search, Fetch fetch, Agent agent, List<String> disabledBuiltins) {
    }

    private final Path file;
    private Map<String, Object> root = new LinkedHashMap<>();

    public ToolsConfig(WebProperties props) {
        this.file = Path.of(props.toolsFile()).toAbsolutePath();
        if (Files.isRegularFile(file)) {
            try {
                root = Json.parseObject(Files.readString(file));
            } catch (IOException e) {
                throw new UncheckedIOException("cannot read " + file, e);
            } catch (RuntimeException e) {
                throw new IllegalStateException(file + ": " + e.getMessage(), e);
            }
        }
    }

    public Path file() {
        return file;
    }

    public synchronized General general() {
        Map<String, Object> s = section("search");
        Map<String, Object> f = section("fetch");
        Map<String, Object> a = section("agent");
        List<String> disabled = new ArrayList<>();
        List<Object> d = Json.list(root, "disabledBuiltins");
        if (d != null) {
            d.forEach(x -> disabled.add(String.valueOf(x)));
        }
        return new General(
                new Search(Json.str(s, "provider", "auto"), Json.str(s, "searxngUrl", ""), Json.str(s, "braveApiKey", ""),
                        (int) clamp(Json.num(s, "maxResults", 5), 1, 20)),
                new Fetch(Json.bool(f, "allowPrivateAddresses", false), (int) clamp(Json.num(f, "maxChars", 8000), 500, 200_000)),
                new Agent((int) clamp(Json.num(a, "maxRounds", 6), 1, 50), (int) clamp(Json.num(a, "maxResultChars", 8000), 200, 200_000),
                        (int) clamp(Json.num(a, "timeoutSeconds", 60), 1, 3600)),
                disabled);
    }

    public synchronized void setGeneral(General g) {
        String provider = g.search().provider() == null ? "auto" : g.search().provider();
        if (!List.of("auto", "duckduckgo", "searxng", "brave").contains(provider)) {
            throw new IllegalArgumentException("search provider must be auto, duckduckgo, searxng or brave");
        }
        String searx = g.search().searxngUrl() == null ? "" : g.search().searxngUrl().trim();
        if (!searx.isEmpty() && !searx.matches("(?i)https?://.+")) {
            throw new IllegalArgumentException("the SearXNG URL must start with http:// or https://");
        }
        General cur = general();
        root.put("search", Json.obj("provider", provider, "searxngUrl", searx,
                "braveApiKey", g.search().braveApiKey() == null ? "" : g.search().braveApiKey().trim(),
                "maxResults", or(g.search().maxResults(), cur.search().maxResults())));
        root.put("fetch", Json.obj("allowPrivateAddresses", or(g.fetch().allowPrivateAddresses(), cur.fetch().allowPrivateAddresses()),
                "maxChars", or(g.fetch().maxChars(), cur.fetch().maxChars())));
        root.put("agent", Json.obj("maxRounds", or(g.agent().maxRounds(), cur.agent().maxRounds()),
                "maxResultChars", or(g.agent().maxResultChars(), cur.agent().maxResultChars()),
                "timeoutSeconds", or(g.agent().timeoutSeconds(), cur.agent().timeoutSeconds())));
        root.put("disabledBuiltins", g.disabledBuiltins() == null ? List.of() : new ArrayList<Object>(g.disabledBuiltins()));
        save();
    }

    /** The {@code mcpServers} entries, by name (copies). */
    public synchronized Map<String, Map<String, Object>> mcpServers() {
        Map<String, Map<String, Object>> out = new LinkedHashMap<>();
        Map<String, Object> m = Json.map(root, "mcpServers");
        if (m != null) {
            m.forEach((k, v) -> {
                if (v instanceof Map<?, ?> mm) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> copy = new LinkedHashMap<>((Map<String, Object>) mm);
                    out.put(k, copy);
                }
            });
        }
        return out;
    }

    public synchronized void putMcpServer(String name, Map<String, Object> entry) {
        Map<String, Object> m = Json.map(root, "mcpServers");
        if (m == null) {
            m = new LinkedHashMap<>();
            root.put("mcpServers", m);
        }
        m.put(name, entry);
        save();
    }

    public synchronized void removeMcpServer(String name) {
        Map<String, Object> m = Json.map(root, "mcpServers");
        if (m != null && m.remove(name) != null) {
            save();
        }
    }

    private static <T> T or(T value, T fallback) {
        return value != null ? value : fallback;
    }

    private Map<String, Object> section(String key) {
        Map<String, Object> m = Json.map(root, key);
        return m == null ? Map.of() : m;
    }

    private static long clamp(long v, long lo, long hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private void save() {
        try {
            Path parent = file.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
            Files.writeString(tmp, pretty(root, 0) + "\n");
            Files.move(tmp, file, java.nio.file.StandardCopyOption.REPLACE_EXISTING, java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot write " + file, e);
        }
    }

    /** Indented JSON, so the file stays pleasant to edit by hand. */
    public static String pretty(Object v, int indent) {
        String pad = "  ".repeat(indent + 1);
        String end = "  ".repeat(indent);
        if (v instanceof Map<?, ?> m) {
            if (m.isEmpty()) {
                return "{}";
            }
            StringBuilder sb = new StringBuilder("{\n");
            int i = 0;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                sb.append(pad).append(Json.write(String.valueOf(e.getKey()))).append(": ").append(pretty(e.getValue(), indent + 1));
                sb.append(++i < m.size() ? ",\n" : "\n");
            }
            return sb.append(end).append('}').toString();
        }
        if (v instanceof List<?> l && !l.isEmpty() && l.stream().anyMatch(x -> x instanceof Map || x instanceof List)) {
            StringBuilder sb = new StringBuilder("[\n");
            for (int i = 0; i < l.size(); i++) {
                sb.append(pad).append(pretty(l.get(i), indent + 1)).append(i + 1 < l.size() ? ",\n" : "\n");
            }
            return sb.append(end).append(']').toString();
        }
        return Json.write(v);
    }
}
