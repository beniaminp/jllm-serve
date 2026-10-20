package org.robomiri.jllm.web.tools;

import org.robomiri.jllm.web.mcp.McpManager;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/** Every tool the chat can offer: the built-in ones that are switched on, plus those of the connected MCP servers. */
@Service
public class ToolRegistry {

    /** A tool for the UI. */
    public record Info(String name, String description, String source, boolean enabled, Map<String, Object> parameters) {
    }

    /** The outcome of one call. */
    public record Outcome(String content, boolean error, long millis) {
    }

    private final ToolsConfig config;
    private final McpManager mcp;
    private final List<Tool> builtins;
    private final ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor();

    public ToolRegistry(ToolsConfig config, McpManager mcp) {
        this.config = config;
        this.mcp = mcp;
        this.builtins = BuiltinTools.all(config::general);
    }

    /** All tools, the switched-off built-ins included (marked). */
    public List<Info> describe() {
        List<String> off = config.general().disabledBuiltins();
        List<Info> out = new ArrayList<>();
        for (Tool t : builtins) {
            out.add(new Info(t.name(), t.description(), t.source(), !off.contains(t.name()), t.parameters()));
        }
        for (Tool t : mcp.tools()) {
            out.add(new Info(t.name(), t.description(), t.source(), true, t.parameters()));
        }
        return out;
    }

    /** The usable tools. */
    public List<Tool> available() {
        List<String> off = config.general().disabledBuiltins();
        List<Tool> out = new ArrayList<>();
        for (Tool t : builtins) {
            if (!off.contains(t.name())) {
                out.add(t);
            }
        }
        out.addAll(mcp.tools());
        return out;
    }

    /** The usable tools among {@code names} (unknown names are skipped: a server may have gone away). */
    public List<Tool> select(List<String> names) {
        List<Tool> out = new ArrayList<>();
        if (names == null) {
            return out;
        }
        for (Tool t : available()) {
            if (names.contains(t.name())) {
                out.add(t);
            }
        }
        return out;
    }

    /** Runs a tool with the configured time limit; failures become an error outcome the model can read. */
    public Outcome call(Tool tool, Map<String, Object> args) {
        ToolsConfig.Agent cfg = config.general().agent();
        long t0 = System.nanoTime();
        Future<String> f = pool.submit(() -> tool.call(args));
        String text;
        boolean error = false;
        try {
            text = f.get(cfg.timeoutSeconds(), TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            f.cancel(true);
            text = "the tool did not finish within " + cfg.timeoutSeconds() + " s";
            error = true;
        } catch (InterruptedException e) {
            f.cancel(true);
            Thread.currentThread().interrupt();
            text = "cancelled";
            error = true;
        } catch (ExecutionException e) {
            Throwable c = e.getCause();
            text = c.getMessage() == null ? c.toString() : c.getMessage();
            error = true;
        }
        if (text == null) {
            text = "";
        }
        if (text.length() > cfg.maxResultChars()) {
            text = text.substring(0, cfg.maxResultChars()) + "\n[truncated at " + cfg.maxResultChars() + " characters]";
        }
        return new Outcome(text, error, (System.nanoTime() - t0) / 1_000_000);
    }

    public Tool find(String name) {
        for (Tool t : available()) {
            if (t.name().equals(name)) {
                return t;
            }
        }
        throw new NoSuchElementException("no tool " + name);
    }
}
