package org.robomiri.jllm.web.api;

import org.robomiri.jllm.web.mcp.McpManager;
import org.robomiri.jllm.web.tools.ToolRegistry;
import org.robomiri.jllm.web.tools.ToolsConfig;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Tools: the list, their settings, a manual test call, and the MCP servers. */
@RestController
@RequestMapping("/api/tools")
public class ToolsController {

    public record Overview(List<ToolRegistry.Info> tools, ToolsConfig.General settings, List<McpManager.ServerView> servers,
                           String file) {
    }

    public record TestCall(String name, Map<String, Object> arguments) {
    }

    private final ToolRegistry registry;
    private final ToolsConfig config;
    private final McpManager mcp;

    public ToolsController(ToolRegistry registry, ToolsConfig config, McpManager mcp) {
        this.registry = registry;
        this.config = config;
        this.mcp = mcp;
    }

    @GetMapping
    public Overview overview() {
        return new Overview(registry.describe(), config.general(), mcp.list(), config.file().toString());
    }

    @PutMapping("/settings")
    public Overview settings(@RequestBody ToolsConfig.General settings) {
        if (settings.search() == null || settings.fetch() == null || settings.agent() == null) {
            throw new IllegalArgumentException("search, fetch and agent are required");
        }
        config.setGeneral(settings);
        return overview();
    }

    /** Runs one tool with the given arguments, as the model would. */
    @PostMapping("/call")
    public ToolRegistry.Outcome call(@RequestBody TestCall call) {
        return registry.call(registry.find(call.name()), call.arguments() == null ? Map.of() : call.arguments());
    }

    /** Adds or replaces servers: {@code {"mcpServers": {"name": {...}}}} (Claude Desktop format) or {@code {"name": {...}}}. */
    @PostMapping("/mcp")
    @SuppressWarnings("unchecked")
    public List<McpManager.ServerView> addServers(@RequestBody Map<String, Object> body) {
        Object inner = body.get("mcpServers");
        Map<String, Object> entries = inner instanceof Map<?, ?> m ? (Map<String, Object>) m : body;
        if (entries.isEmpty()) {
            throw new IllegalArgumentException("no servers given");
        }
        List<McpManager.ServerView> out = new ArrayList<>();
        for (Map.Entry<String, Object> e : entries.entrySet()) {
            if (!(e.getValue() instanceof Map<?, ?> v)) {
                throw new IllegalArgumentException(e.getKey() + ": the entry must be an object");
            }
            out.add(mcp.put(e.getKey(), deepCopy((Map<String, Object>) v)));
        }
        return out;
    }

    @PostMapping("/mcp/{name}/reconnect")
    public McpManager.ServerView reconnect(@PathVariable String name) {
        return mcp.reconnect(name);
    }

    @PostMapping("/mcp/{name}/enabled")
    public McpManager.ServerView enable(@PathVariable String name, @RequestBody Map<String, Object> body) {
        Map<String, Object> entry = new LinkedHashMap<>(config.mcpServers().getOrDefault(name, Map.of()));
        if (entry.isEmpty()) {
            throw new java.util.NoSuchElementException("no MCP server " + name);
        }
        if (Boolean.TRUE.equals(body.get("enabled"))) {
            entry.remove("disabled");
        } else {
            entry.put("disabled", true);
        }
        return mcp.put(name, entry);
    }

    @DeleteMapping("/mcp/{name}")
    public Map<String, Object> remove(@PathVariable String name) {
        mcp.remove(name);
        return Map.of("removed", name);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> deepCopy(Map<String, Object> m) {
        Map<String, Object> out = new LinkedHashMap<>();
        m.forEach((k, v) -> out.put(k, v instanceof Map<?, ?> mm ? deepCopy((Map<String, Object>) mm)
                : v instanceof List<?> l ? new ArrayList<Object>(l) : v));
        return out;
    }
}
