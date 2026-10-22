package org.robomiri.jllm.web.settings;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.web.WebProperties;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.TreeMap;

/**
 * The engine settings ({@code jllm.properties}) as the web console sees them: the bundled defaults, the external
 * file the command line would also read ({@code $JLLM_CONFIG} or {@code ./jllm.properties}) and, on top, the
 * overrides edited in the UI, stored in {@code jllm.web.settings-file}. Each loaded model takes a snapshot of these
 * plus its own load options; sampling and chat defaults apply to the next request.
 */
@Service
public class SettingsService {

    /** When a changed value takes effect. */
    public enum Scope {
        /** the next request (sampling defaults, chat, request limits, the Hub client of the next download) */
        LIVE,
        /** the next model load (weights, backend, context, speculation) */
        LOAD,
        /** only the command line tools use it (the CLI's own HTTP server, bench) */
        CLI
    }

    /**
     * One setting for the UI.
     *
     * @param source {@code default}, the external file's path or {@code web}
     */
    public record View(String key, String value, String defaultValue, String source, String group, Scope scope,
                       String help) {
    }

    private static final String WEB_SOURCE = "command line"; // overrides are passed as flags, see build()

    private final Path file;
    private final Map<String, String> overrides = new TreeMap<>();
    private final Map<String, String> defaults = new LinkedHashMap<>();
    private final Map<String, String> help = new LinkedHashMap<>();
    private volatile Settings current;

    public SettingsService(WebProperties props) {
        this.file = Path.of(props.settingsFile()).toAbsolutePath();
        readBundled();
        if (Files.isRegularFile(file)) {
            Properties p = new Properties();
            try (Reader r = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
                p.load(r);
            } catch (IOException e) {
                throw new UncheckedIOException("cannot read " + file, e);
            }
            p.stringPropertyNames().forEach(k -> overrides.put(k, p.getProperty(k)));
        }
        try {
            current = build(overrides, Map.of());
        } catch (IllegalArgumentException e) {
            throw new IllegalStateException(file + ": " + e.getMessage(), e);
        }
    }

    /** The effective settings (validated). */
    public Settings current() {
        return current;
    }

    public Path file() {
        return file;
    }

    /** The current settings with {@code extra} (per-model load options) on top; validated. */
    public synchronized Settings with(Map<String, String> extra) {
        return build(overrides, extra);
    }

    public synchronized List<View> views() {
        Settings s = current;
        List<View> out = new ArrayList<>();
        for (String k : s.keys()) {
            String src = s.source(k);
            out.add(new View(k, s.get(k), defaults.getOrDefault(k, ""), WEB_SOURCE.equals(src) ? "web" : src,
                    group(k), scope(k), help.getOrDefault(k, "")));
        }
        return out;
    }

    /**
     * Applies {@code changes} (key to new value) and removes the overrides of {@code reset}, validating the result
     * as a whole before anything is stored.
     *
     * @throws IllegalArgumentException for an unknown key or an invalid value (nothing changes)
     */
    public synchronized void update(Map<String, String> changes, List<String> reset) {
        Map<String, String> next = new TreeMap<>(overrides);
        if (reset != null) {
            reset.forEach(next::remove);
        }
        if (changes != null) {
            changes.forEach((k, v) -> {
                if (!defaults.containsKey(k)) {
                    throw new IllegalArgumentException("unknown setting " + k);
                }
                next.put(k, v == null ? "" : v.trim());
            });
        }
        // an override equal to the default that no external file contradicts is just noise
        Settings plain = build(Map.of(), Map.of());
        next.entrySet().removeIf(e -> e.getValue().equals(plain.get(e.getKey())));
        Settings s = build(next, Map.of());
        store(next);
        overrides.clear();
        overrides.putAll(next);
        current = s;
    }

    /** The bundled default of every key plus the comment block above it, as help text. */
    private void readBundled() {
        try (InputStream in = Settings.class.getResourceAsStream(Settings.RESOURCE)) {
            if (in == null) {
                throw new IllegalStateException("bundled " + Settings.RESOURCE + " is missing");
            }
            BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
            StringBuilder comment = new StringBuilder();
            for (String line; (line = r.readLine()) != null; ) {
                String t = line.trim();
                if (t.startsWith("#")) {
                    String c = t.replaceFirst("^#+\\s?", "");
                    if (c.startsWith("---")) { // a section rule
                        comment.setLength(0);
                    } else {
                        comment.append(comment.isEmpty() ? "" : " ").append(c); // wrapped lines of one text
                    }
                } else if (t.isEmpty()) {
                    comment.setLength(0);
                } else {
                    int eq = t.indexOf('=');
                    if (eq > 0) {
                        String k = t.substring(0, eq).trim();
                        defaults.put(k, t.substring(eq + 1).trim());
                        help.put(k, comment.toString());
                    }
                    comment.setLength(0);
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private Settings build(Map<String, String> base, Map<String, String> extra) {
        List<String> args = new ArrayList<>();
        args.add("web");
        Map<String, String> all = new TreeMap<>(base);
        all.putAll(extra);
        all.forEach((k, v) -> args.add("--" + k + "=" + v)); // --key=value: a value may start with "--"
        try {
            return Settings.load(args.toArray(String[]::new)).validate();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private void store(Map<String, String> values) {
        Properties p = new Properties() {
            @Override
            public synchronized java.util.Enumeration<Object> keys() { // sorted output
                return java.util.Collections.enumeration(new java.util.TreeSet<>(super.keySet()));
            }

            @Override
            public java.util.Set<Map.Entry<Object, Object>> entrySet() {
                java.util.Set<Map.Entry<Object, Object>> s = new java.util.TreeSet<>(
                        java.util.Comparator.comparing(e -> e.getKey().toString()));
                s.addAll(super.entrySet());
                return s;
            }
        };
        p.putAll(values);
        try {
            Path parent = file.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            try (Writer w = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
                p.store(w, "jllm-web: engine settings changed in the web console (overrides jllm.properties)");
            }
        } catch (IOException e) {
            throw new UncheckedIOException("cannot write " + file, e);
        }
    }

    static String group(String key) {
        int dot = key.indexOf('.');
        return dot < 0 ? key : key.substring(0, dot);
    }

    public static Scope scope(String key) {
        if (key.equals("audio.stt-model") || key.equals("audio.tts-model")) {
            return Scope.CLI; // the models `jllm serve` loads at start; the console loads models from its Models page
        }
        if (key.startsWith("audio.stt.") || key.equals("audio.ffmpeg") || key.equals("audio.max-seconds")
                || key.equals("audio.max-upload-mb") || key.equals("audio.tts.max-chars")) {
            return Scope.LIVE; // read per request
        }
        if (key.startsWith("sampling.") || key.startsWith("chat.") || key.startsWith("hub.")
                || key.equals("server.request-timeout-ms") || key.equals("engine.max-prompt-bytes")
                || key.equals("engine.constrained-tools")) {
            return Scope.LIVE;
        }
        if (key.startsWith("bench.") || (key.startsWith("server.") && !key.equals("server.queue-timeout-ms"))) {
            return Scope.CLI;
        }
        return Scope.LOAD;
    }
}
