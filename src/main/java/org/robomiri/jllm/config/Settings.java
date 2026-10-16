package org.robomiri.jllm.config;

import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.sampling.SamplingParams;

import java.io.IOException;
import java.io.InputStream;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.TreeMap;

/**
 * Configuration, layered: bundled {@code jllm.properties} &lt; external properties file &lt; command line.
 *
 * <p>Command line flags are either short aliases ({@code --ctx 2048}, see {@link #ALIASES}) or property keys
 * used directly ({@code --server.port 9000}). Boolean flags may omit the value ({@code --think}).
 */
public final class Settings {

    public static final String RESOURCE = "/jllm.properties";

    /** Short command line flag -> property key. */
    public static final Map<String, String> ALIASES = Map.ofEntries(
            Map.entry("model", "model.repo"),
            Map.entry("revision", "model.revision"),
            Map.entry("dir", "model.dir"),
            Map.entry("quant", "model.quant"),
            Map.entry("low-disk", "model.low-disk"),
            Map.entry("backend", "backend"),
            Map.entry("ctx", "backend.ctx"),
            Map.entry("batch", "backend.batch"),
            Map.entry("threads", "backend.threads"),
            Map.entry("temp", "sampling.temperature"),
            Map.entry("temperature", "sampling.temperature"),
            Map.entry("top-k", "sampling.top-k"),
            Map.entry("top-p", "sampling.top-p"),
            Map.entry("min-p", "sampling.min-p"),
            Map.entry("rep-penalty", "sampling.repetition-penalty"),
            Map.entry("seed", "sampling.seed"),
            Map.entry("max-tokens", "sampling.max-tokens"),
            Map.entry("think", "chat.thinking"),
            Map.entry("system", "chat.system"),
            Map.entry("host", "server.host"),
            Map.entry("port", "server.port"));

    /** Flags that are not properties: per-invocation inputs. */
    private static final java.util.Set<String> CLI_ONLY = java.util.Set.of("prompt", "raw", "config");

    private final Properties props = new Properties();
    private final Map<String, String> sources = new LinkedHashMap<>();

    private Settings() {
    }

    /** Loads defaults, the external file (if any) and the flags in {@code args} (a leading command word is ignored). */
    public static Settings load(String[] args) throws IOException {
        Settings s = new Settings();
        try (InputStream in = Settings.class.getResourceAsStream(RESOURCE)) {
            if (in == null) {
                throw new IOException("bundled " + RESOURCE + " is missing");
            }
            s.props.load(new java.io.InputStreamReader(in, StandardCharsets.UTF_8));
        }
        s.props.stringPropertyNames().forEach(k -> s.sources.put(k, "default"));

        Map<String, String> cli = parseArgs(args);
        Path external = findExternal(cli.get("config"));
        if (external != null) {
            try (Reader r = Files.newBufferedReader(external, StandardCharsets.UTF_8)) {
                Properties p = new Properties();
                p.load(r);
                for (String k : p.stringPropertyNames()) {
                    s.put(k, p.getProperty(k), external.toString());
                }
            }
        }
        cli.forEach((k, v) -> {
            if (!k.equals("config")) {
                s.put(k, v, "command line");
            }
        });
        return s;
    }

    private static Path findExternal(String fromCli) {
        if (fromCli != null) {
            return Path.of(fromCli);
        }
        String env = System.getenv("JLLM_CONFIG");
        if (env != null && !env.isBlank()) {
            return Path.of(env);
        }
        Path local = Path.of("jllm.properties");
        return Files.isRegularFile(local) ? local : null;
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> m = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (!a.startsWith("--")) {
                continue;
            }
            String key = a.substring(2);
            String value;
            int eq = key.indexOf('=');
            if (eq >= 0) {
                value = key.substring(eq + 1);
                key = key.substring(0, eq);
            } else if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                value = args[++i];
            } else {
                value = "true";
            }
            m.put(ALIASES.getOrDefault(key, key), value);
        }
        return m;
    }

    private void put(String key, String value, String source) {
        if (!props.containsKey(key) && !CLI_ONLY.contains(key)) {
            throw new IllegalArgumentException("unknown option --" + key + " (see jllm.properties for valid keys)");
        }
        props.setProperty(key, value.trim());
        sources.put(key, source);
    }

    /**
     * Parses every typed setting once, so that a typo fails here, before the (slow) weight load, instead of
     * minutes later when the value is first used. {@link org.robomiri.jllm.Main} prints the message and exits
     * with status 2, as for an unknown key.
     *
     * @return this
     * @throws IllegalArgumentException naming the offending key and value
     */
    public Settings validate() {
        for (String k : new String[]{"backend.ctx", "backend.batch", "bench.max-tokens", "bench.rounds",
                "server.max-connections"}) {
            require(getInt(k) >= 1, k, "must be at least 1");
        }
        require(getInt("backend.threads") >= 0, "backend.threads", "must be 0 (= all processors) or positive");
        require(getInt("server.port") >= 0 && getInt("server.port") <= 65535, "server.port", "must be 0..65535");
        for (String k : new String[]{"server.read-timeout-ms", "server.write-timeout-ms"}) {
            require(getInt(k) >= 0, k, "must be 0 (= none) or positive");
        }
        for (String k : new String[]{"server.max-body-bytes", "server.queue-timeout-ms", "server.request-timeout-ms",
                "engine.max-prompt-bytes"}) {
            require(getLong(k) >= 0, k, "must be 0 or positive");
        }
        require(getLong("hub.timeout-ms") > 0, "hub.timeout-ms", "must be positive");
        for (String k : new String[]{"model.low-disk", "backend.cpu.int8-activations", "chat.thinking", "raw"}) {
            if (has(k)) {
                getBool(k);
            }
        }
        try {
            sampling(); // numbers plus the SamplingParams range checks
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("sampling.*: " + e.getMessage());
        }
        try {
            QType.parse(get("model.quant"));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("model.quant: " + e.getMessage());
        }
        KvDtype.parse(get("backend.kv-dtype")); // its message names the key
        String backend = get("backend");
        require(backend.equals("cpu") || backend.equals("metal") || backend.equals("auto"), "backend",
                "must be cpu, metal or auto");
        return this;
    }

    private void require(boolean ok, String key, String what) {
        if (!ok) {
            throw new IllegalArgumentException(key + " " + what + ", got '" + get(key) + "'");
        }
    }

    // ------------------------------------------------------------------ typed access

    public String get(String key) {
        String v = props.getProperty(key);
        if (v == null) {
            throw new IllegalArgumentException("unknown setting " + key);
        }
        return v;
    }

    public boolean has(String key) {
        return props.containsKey(key);
    }

    /** Value or null when empty. */
    public String opt(String key) {
        String v = get(key);
        return v.isEmpty() ? null : v;
    }

    public int getInt(String key) {
        try {
            return Integer.parseInt(get(key));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + " must be an integer, got '" + get(key) + "'");
        }
    }

    public long getLong(String key) {
        try {
            return Long.parseLong(get(key));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + " must be an integer, got '" + get(key) + "'");
        }
    }

    public float getFloat(String key) {
        try {
            return Float.parseFloat(get(key));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + " must be a number, got '" + get(key) + "'");
        }
    }

    public boolean getBool(String key) {
        String v = get(key).toLowerCase();
        return v.equals("true") || v.equals("yes") || v.equals("on") || v.equals("1");
    }

    public SamplingParams sampling() {
        return new SamplingParams(
                getFloat("sampling.temperature"),
                getInt("sampling.top-k"),
                getFloat("sampling.top-p"),
                getFloat("sampling.min-p"),
                getFloat("sampling.repetition-penalty"),
                getFloat("sampling.presence-penalty"),
                getFloat("sampling.frequency-penalty"),
                getLong("sampling.seed"),
                getInt("sampling.max-tokens"),
                List.of());
    }

    /** Human readable dump of every setting and where it came from. */
    public String describe() {
        StringBuilder sb = new StringBuilder();
        new TreeMap<>(sources).forEach((k, src) ->
                sb.append(String.format("  %-30s = %-40s (%s)%n", k, props.getProperty(k), src)));
        return sb.toString();
    }
}
