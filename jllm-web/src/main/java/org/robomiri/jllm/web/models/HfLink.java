package org.robomiri.jllm.web.models;

import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.regex.Pattern;

/**
 * What a model link points at. Accepted forms:
 * <ul>
 *   <li>{@code Qwen/Qwen3-0.6B} or {@code https://huggingface.co/Qwen/Qwen3-0.6B} (also {@code hf.co}, without scheme)</li>
 *   <li>{@code https://huggingface.co/Qwen/Qwen3-0.6B/tree/<revision>}</li>
 *   <li>{@code https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q4_K_M.gguf} (or {@code /resolve/}):
 *       that GGUF file</li>
 *   <li>{@code unsloth/Qwen3-0.6B-GGUF:Q4_K_M} or {@code hf.co/unsloth/Qwen3-0.6B-GGUF:Q4_K_M}: a GGUF quantization label</li>
 *   <li>a local path ending in {@code .gguf}</li>
 * </ul>
 *
 * @param repo     {@code owner/name}, or the local file's path when {@link #local}
 * @param revision branch, tag or commit; null for the default
 * @param file     a GGUF file in the repository; null when not chosen
 * @param quant    a GGUF quantization label; null when not chosen
 */
public record HfLink(String repo, String revision, String file, String quant, boolean local) {

    private static final Pattern NAME = Pattern.compile("[A-Za-z0-9][A-Za-z0-9._-]*");

    public static HfLink parse(String link) {
        if (link == null || link.isBlank()) {
            throw new IllegalArgumentException("a Hugging Face link or repository name is required");
        }
        String s = link.trim();
        if (s.toLowerCase(Locale.ROOT).endsWith(".gguf") && (s.startsWith("/") || s.startsWith("./")
                || s.startsWith("~") || s.matches("^[A-Za-z]:[\\\\/].*"))) {
            if (s.startsWith("~")) {
                s = System.getProperty("user.home") + s.substring(1);
            }
            return new HfLink(s, null, null, null, true);
        }
        s = s.replaceFirst("(?i)^https?://", "");
        s = s.replaceFirst("(?i)^(www\\.)?(huggingface\\.co|hf\\.co)/", "");
        int q = s.indexOf('?');
        if (q >= 0) {
            s = s.substring(0, q);
        }
        int h = s.indexOf('#');
        if (h >= 0) {
            s = s.substring(0, h);
        }
        while (s.endsWith("/")) {
            s = s.substring(0, s.length() - 1);
        }
        if (s.startsWith("datasets/") || s.startsWith("spaces/")) {
            throw new IllegalArgumentException("not a model repository: " + link);
        }
        if (s.startsWith("models/")) {
            s = s.substring("models/".length());
        }
        String[] parts = s.split("/");
        if (parts.length < 2) {
            throw new IllegalArgumentException("expected owner/name or a huggingface.co model link, got '" + link + "'");
        }
        String owner = parts[0];
        String name = parts[1];
        String quant = null;
        int colon = name.indexOf(':');
        if (colon >= 0 && parts.length == 2) { // owner/name:Q4_K_M (the Ollama / llama.cpp -hf style)
            quant = name.substring(colon + 1);
            name = name.substring(0, colon);
            if (!quant.matches("[A-Za-z0-9_.-]+")) {
                throw new IllegalArgumentException("invalid quantization label '" + quant + "' in " + link);
            }
        }
        if (!NAME.matcher(owner).matches() || !NAME.matcher(name).matches()) {
            throw new IllegalArgumentException("invalid repository name '" + owner + "/" + name + "' in " + link);
        }
        String repo = owner + "/" + name;
        String revision = null;
        String file = null;
        if (parts.length > 2) {
            String kind = parts[2];
            if (parts.length < 4) {
                throw new IllegalArgumentException("unsupported link: " + link);
            }
            revision = decode(parts[3]);
            switch (kind) {
                case "tree" -> {
                    // a folder of the repository is not a separate model: only the revision matters
                }
                case "blob", "resolve" -> {
                    if (parts.length < 5) {
                        throw new IllegalArgumentException("the link names no file: " + link);
                    }
                    file = decode(String.join("/", java.util.Arrays.copyOfRange(parts, 4, parts.length)));
                    if (!file.toLowerCase(Locale.ROOT).endsWith(".gguf")) {
                        file = null; // a link to config.json or a shard means the whole checkpoint
                    }
                }
                default -> throw new IllegalArgumentException("unsupported link: " + link
                        + " (use the model page, a /tree/<revision> or a /blob/<revision>/<file>.gguf link)");
            }
        }
        return new HfLink(repo, revision, file, quant, false);
    }

    private static String decode(String s) {
        return URLDecoder.decode(s.replace("+", "%2B"), StandardCharsets.UTF_8);
    }

    /** The default id of a model loaded from this link: the repository, plus the GGUF file or label when chosen. */
    public String defaultId() {
        if (local) {
            return java.nio.file.Path.of(repo).getFileName().toString().replaceFirst("(?i)\\.gguf$", "");
        }
        if (file != null) {
            String base = file.contains("/") ? file.substring(file.lastIndexOf('/') + 1) : file;
            return repo + ":" + base.replaceFirst("(?i)\\.gguf$", "");
        }
        return quant != null ? repo + ":" + quant : repo;
    }
}
