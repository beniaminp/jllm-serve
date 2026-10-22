package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.hub.HuggingFaceHub;
import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Finding and fetching diffusers-format image models ({@code model_index.json} plus one folder per component) on the
 * Hugging Face Hub. Only what inference needs is fetched: the configs and tokenizer files of the components the pipeline
 * uses and, per weight file, the half-precision variant ({@code *.fp16.safetensors}) when the repository has one, else
 * the plain safetensors file (or its shards).
 */
public final class DiffusersRepo {

    private static final Set<String> SKIP_COMPONENTS = Set.of("safety_checker", "feature_extractor", "image_encoder", "watermarker");
    private static final Pattern SHARD = Pattern.compile("(.+?)(\\.fp16)?-\\d{5}-of-\\d{5}\\.safetensors");

    private DiffusersRepo() {
    }

    /** True for a repository (file listing) that is a diffusers pipeline. */
    public static boolean isDiffusers(List<HuggingFaceHub.RepoFile> files) {
        return files.stream().anyMatch(f -> f.name().equals("model_index.json"));
    }

    /** True for a local folder holding a diffusers pipeline. */
    public static boolean isDiffusersDir(Path dir) {
        return Files.isRegularFile(dir.resolve("model_index.json"));
    }

    /** The component folders a pipeline of this {@code model_index.json} needs. */
    public static List<String> components(Map<String, Object> modelIndex) {
        List<String> out = new ArrayList<>();
        for (Map.Entry<String, Object> e : modelIndex.entrySet()) {
            if (e.getKey().startsWith("_") || SKIP_COMPONENTS.contains(e.getKey())) {
                continue;
            }
            if (e.getValue() instanceof List<?> l && l.size() == 2 && l.get(0) != null && l.get(1) != null) {
                out.add(e.getKey());
            }
        }
        return out;
    }

    /** The files of {@code listing} to download for the pipeline described by {@code modelIndex} (name -> file). */
    public static Map<String, HuggingFaceHub.RepoFile> select(List<HuggingFaceHub.RepoFile> listing, Map<String, Object> modelIndex) {
        Map<String, HuggingFaceHub.RepoFile> byName = new LinkedHashMap<>();
        for (HuggingFaceHub.RepoFile f : listing) {
            byName.put(f.name(), f);
        }
        Map<String, HuggingFaceHub.RepoFile> out = new LinkedHashMap<>();
        if (byName.containsKey("model_index.json")) {
            out.put("model_index.json", byName.get("model_index.json"));
        }
        for (String comp : components(modelIndex)) {
            String prefix = comp + "/";
            // weight files by stem; everything that is not a weight file but small config / tokenizer data is taken
            Map<String, List<String>> weights = new LinkedHashMap<>(); // "fp16" / "" -> files
            for (HuggingFaceHub.RepoFile f : listing) {
                String n = f.name();
                if (!n.startsWith(prefix) || n.indexOf('/', prefix.length()) >= 0) {
                    continue; // another component, or a sub-folder (onnx, openvino, ...)
                }
                String leaf = n.substring(prefix.length());
                if (leaf.endsWith(".safetensors") || leaf.endsWith(".safetensors.index.json") || leaf.endsWith(".safetensors.index.fp16.json")) {
                    boolean fp16 = leaf.contains(".fp16");
                    weights.computeIfAbsent(fp16 ? "fp16" : "", k -> new ArrayList<>()).add(n);
                } else if (leaf.endsWith(".bin") || leaf.endsWith(".ckpt") || leaf.endsWith(".onnx") || leaf.endsWith(".msgpack")
                        || leaf.endsWith(".pt") || leaf.endsWith(".h5") || leaf.endsWith(".pb") || leaf.endsWith(".ot")) {
                    continue; // other weight formats
                } else {
                    out.put(n, f);
                }
            }
            List<String> chosen = weights.containsKey("fp16") ? weights.get("fp16") : weights.getOrDefault("", List.of());
            for (String n : chosen) {
                out.put(n, byName.get(n));
            }
        }
        return out;
    }

    /**
     * Downloads the files of a pipeline into {@code dir} (the folder is created); files already complete are skipped.
     *
     * @return the files that make the model, for progress reporting
     */
    public static Map<String, HuggingFaceHub.RepoFile> download(HuggingFaceHub hub, String repo, String revision, Path dir)
            throws IOException, InterruptedException {
        Files.createDirectories(dir);
        List<HuggingFaceHub.RepoFile> listing = hub.listFiles(repo, revision);
        hub.ensure(repo, revision, dir, "model_index.json");
        Map<String, Object> index = Json.parseObject(Files.readString(dir.resolve("model_index.json"), StandardCharsets.UTF_8));
        Map<String, HuggingFaceHub.RepoFile> files = select(listing, index);
        long total = 0;
        for (HuggingFaceHub.RepoFile f : files.values()) {
            Path t = dir.resolve(f.name());
            if (!(Files.isRegularFile(t) && Files.size(t) == f.size())) {
                total += Math.max(0, f.size());
            }
        }
        long free = Files.getFileStore(dir).getUsableSpace();
        if (total > free) {
            throw new IOException(String.format("need %.1f GB for %s but only %.1f GB are free", total / 1e9, repo, free / 1e9));
        }
        System.err.printf("Downloading %s@%s (%d files, %.1f MB) into %s%n", repo, revision, files.size(), total / 1e6, dir);
        for (HuggingFaceHub.RepoFile f : files.values()) {
            hub.ensure(repo, revision, dir, f.name());
        }
        return files;
    }

    /** The component folders of a downloaded pipeline that lack their weights (a quick completeness check). */
    public static List<String> missingWeights(Path dir) throws IOException {
        Map<String, Object> index = Json.parseObject(Files.readString(dir.resolve("model_index.json"), StandardCharsets.UTF_8));
        Set<String> missing = new LinkedHashSet<>();
        for (String comp : components(index)) {
            Path d = dir.resolve(comp);
            if (!Files.isDirectory(d)) {
                missing.add(comp);
                continue;
            }
            if (comp.equals("scheduler") || comp.startsWith("tokenizer")) {
                continue;
            }
            try (var s = Files.list(d)) {
                if (s.noneMatch(p -> p.getFileName().toString().endsWith(".safetensors"))) {
                    missing.add(comp);
                }
            }
        }
        return new ArrayList<>(missing);
    }
}
