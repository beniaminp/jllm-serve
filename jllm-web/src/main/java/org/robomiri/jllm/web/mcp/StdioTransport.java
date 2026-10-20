package org.robomiri.jllm.web.mcp;

import org.robomiri.jllm.json.Json;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** An MCP server started as a child process, speaking newline-delimited JSON-RPC on stdin / stdout. */
public final class StdioTransport extends McpTransport {

    private final String name;
    private final Process process;
    private final OutputStream stdin;
    private final java.util.Deque<String> lastErrors = new java.util.ArrayDeque<>(); // for the exit message

    public StdioTransport(String name, String command, List<String> args, Map<String, String> env, String cwd) throws IOException {
        this.name = name;
        List<String> cmd = new ArrayList<>();
        boolean windows = System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("win");
        if (windows && !command.toLowerCase(Locale.ROOT).endsWith(".exe")) {
            cmd.add("cmd"); // npx, uvx and friends are .cmd scripts there
            cmd.add("/c");
        }
        cmd.add(command);
        cmd.addAll(args);
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.environment().putAll(env);
        if (cwd != null && !cwd.isBlank()) {
            pb.directory(new java.io.File(cwd));
        }
        try {
            process = pb.start();
        } catch (IOException e) {
            throw new IOException("cannot start '" + command + "': " + e.getMessage(), e);
        }
        stdin = process.getOutputStream();
        Thread.ofVirtual().name("mcp-" + name + "-out").start(this::readStdout);
        Thread.ofVirtual().name("mcp-" + name + "-err").start(this::readStderr);
    }

    private void readStdout() {
        try (BufferedReader r = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            for (String line; (line = r.readLine()) != null; ) {
                if (line.isBlank()) {
                    continue;
                }
                Object msg;
                try {
                    msg = Json.parse(line);
                } catch (RuntimeException e) {
                    System.err.println("[mcp " + name + "] (not JSON-RPC) " + line);
                    continue;
                }
                dispatch(msg);
            }
        } catch (IOException ignored) {
            // the process is gone; reported below
        }
        int code;
        try {
            code = process.waitFor();
        } catch (InterruptedException e) {
            code = -1;
        }
        String why;
        synchronized (lastErrors) {
            why = String.join(" | ", lastErrors);
        }
        closed(new IOException("the server process exited" + (code >= 0 ? " with status " + code : "")
                + (why.isEmpty() ? "" : ": " + why)));
    }

    private void readStderr() {
        try (BufferedReader r = new BufferedReader(new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8))) {
            for (String line; (line = r.readLine()) != null; ) {
                System.err.println("[mcp " + name + "] " + line);
                if (!line.isBlank() && !line.startsWith("npm notice") && !line.startsWith("npm warn")) {
                    synchronized (lastErrors) {
                        lastErrors.addLast(line.strip());
                        while (lastErrors.size() > 3) {
                            lastErrors.removeFirst();
                        }
                    }
                }
            }
        } catch (IOException ignored) {
            // closed
        }
    }

    @Override
    protected synchronized void send(Map<String, Object> message, long id) throws IOException {
        if (!process.isAlive()) {
            throw new IOException("the server process is not running");
        }
        stdin.write((Json.write(message) + "\n").getBytes(StandardCharsets.UTF_8));
        stdin.flush();
    }

    @Override
    public void close() {
        try {
            stdin.close(); // the polite way to ask an MCP stdio server to exit
        } catch (IOException ignored) {
            // already closed
        }
        try {
            if (!process.waitFor(2, java.util.concurrent.TimeUnit.SECONDS)) {
                process.descendants().forEach(ProcessHandle::destroy);
                process.destroy();
                if (!process.waitFor(3, java.util.concurrent.TimeUnit.SECONDS)) {
                    process.destroyForcibly();
                }
            }
        } catch (InterruptedException e) {
            process.destroyForcibly();
            Thread.currentThread().interrupt();
        }
        closed(new IOException("closed"));
    }
}
