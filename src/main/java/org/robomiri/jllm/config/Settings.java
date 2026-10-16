package org.robomiri.jllm.config;

import org.robomiri.jllm.engine.Speculator;
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
            Map.entry("head-quant", "model.head-quant"),
            Map.entry("low-disk", "model.low-disk"),
            Map.entry("gguf-file", "model.gguf-file"),
            Map.entry("gguf-quant", "model.gguf-quant"),
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
            Map.entry("speculative", "engine.speculative"),
            Map.entry("draft-model", "engine.draft-model"),
            Map.entry("draft-tokens", "engine.draft-tokens"),
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

        // keys whose flag may stand alone (--think) and whose value is only consumed when it is a boolean literal
        java.util.Set<String> booleanKeys = new java.util.HashSet<>(java.util.Set.of("raw"));
        for (String k : s.props.stringPropertyNames()) {
            String v = s.props.getProperty(k);
            if (v.equals("true") || v.equals("false")) {
                booleanKeys.add(k);
            }
        }
        Map<String, String> cli = parseArgs(args, booleanKeys);
        Path external = findExternal(cli.get("config"));
        if (external != null) {
            try (Reader r = Files.newBufferedReader(external, StandardCharsets.UTF_8)) {
                Properties p = new Properties();
                p.load(r);
                for (String k : p.stringPropertyNames()) {
                    if (CLI_ONLY.contains(k)) {
                        throw new IllegalArgumentException(k + " is a command line flag (--" + k + "), not a setting: "
                                + "remove it from " + external);
                    }
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

    /**
     * {@code --key value}, {@code --key=value} (the only way to pass a value that itself starts with {@code --})
     * or, for a boolean key, a bare {@code --flag}: it consumes the next argument only when that is a boolean
     * literal, so {@code --raw} never swallows what follows it. Nothing takes positional arguments after the
     * command word, so a stray one (a typo such as {@code --think flase}) is an error rather than ignored.
     */
    private static Map<String, String> parseArgs(String[] args, java.util.Set<String> booleanKeys) {
        Map<String, String> m = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (!a.startsWith("--")) {
                if (i == 0) {
                    continue; // the command word
                }
                throw new IllegalArgumentException("unexpected argument '" + a + "' (options are --key value or"
                        + " --key=value; boolean flags take true/false or stand alone)");
            }
            String key = a.substring(2);
            String value;
            int eq = key.indexOf('=');
            if (eq >= 0) {
                value = key.substring(eq + 1);
                key = key.substring(0, eq);
            } else {
                key = ALIASES.getOrDefault(key, key);
                String next = i + 1 < args.length ? args[i + 1] : null;
                if (next == null || next.startsWith("--")) {
                    value = "true";
                } else if (booleanKeys.contains(key) && !isBooleanLiteral(next)) {
                    value = "true"; // a bare flag; the next argument is reported as unexpected on the next round
                } else {
                    value = args[++i];
                }
            }
            m.put(ALIASES.getOrDefault(key, key), value);
        }
        return m;
    }

    private static boolean isBooleanLiteral(String v) {
        return switch (v.toLowerCase(java.util.Locale.ROOT)) {
            case "true", "false", "yes", "no", "on", "off", "1", "0" -> true;
            default -> false;
        };
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
        speculation(); // the messages name their keys
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
        String ggufFile = get("model.gguf-file");
        require(ggufFile.isEmpty() || ggufFile.toLowerCase(java.util.Locale.ROOT).endsWith(".gguf"), "model.gguf-file",
                "must be the exact name of a .gguf file in the repository (or empty)");
        require(get("model.gguf-quant").matches("[A-Za-z0-9_.-]*"), "model.gguf-quant",
                "must be a quantization label such as Q4_K_M or Q8_0 (or empty)");
        try {
            QType.parseHead(get("model.head-quant"));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("model.head-quant: " + e.getMessage());
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

    /** Accepts true/false, yes/no, on/off and 1/0 (case-insensitive); anything else is an error, not false. */
    public boolean getBool(String key) {
        return switch (get(key).toLowerCase(java.util.Locale.ROOT)) {
            case "true", "yes", "on", "1" -> true;
            case "false", "no", "off", "0" -> false;
            default -> throw new IllegalArgumentException(key + " must be true or false, got '" + get(key) + "'");
        };
    }

    /** The speculative decoding settings ({@code engine.speculative}, {@code engine.draft-tokens}, {@code engine.ngram-*}). */
    public Speculator.Config speculation() {
        Speculator.Mode mode = Speculator.Mode.parse(get("engine.speculative"));
        int k = getInt("engine.draft-tokens");
        int max = getInt("engine.ngram-max");
        int min = getInt("engine.ngram-min");
        int window = getInt("engine.ngram-window");
        require(k >= 1 && k <= 32, "engine.draft-tokens", "must be 1..32");
        require(max >= 1, "engine.ngram-max", "must be at least 1");
        require(min >= 1 && min <= max, "engine.ngram-min", "must be 1..engine.ngram-max");
        require(window >= 1, "engine.ngram-window", "must be at least 1");
        if (mode == Speculator.Mode.DRAFT) {
            require(!get("engine.draft-model").isBlank(), "engine.draft-model",
                    "is required when engine.speculative=draft");
        }
        return new Speculator.Config(mode, k, max, min, window);
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
