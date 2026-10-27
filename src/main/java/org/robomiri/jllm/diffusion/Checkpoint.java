package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

/**
 * The tensors and {@code config.json} of one component folder of a diffusers repository ({@code unet/}, {@code vae/},
 * {@code text_encoder/} ...). The half-precision variant ({@code *.fp16.safetensors}) is preferred when present because
 * it is half the size; a sharded component ({@code *.safetensors.index.json}) is read shard by shard.
 */
public final class Checkpoint implements AutoCloseable {

    public final Path dir;
    public final Map<String, Object> config;
    private final List<SafeTensors> files = new ArrayList<>();
    private final List<Path> paths = new ArrayList<>();
    private final Map<String, SafeTensors.Tensor> tensors = new LinkedHashMap<>();

    private Checkpoint(Path dir, Map<String, Object> config) {
        this.dir = dir;
        this.config = config;
    }

    /**
     * @param stem {@code diffusion_pytorch_model} for diffusers models, {@code model} for transformers models
     */
    public static Checkpoint open(Path dir, String stem) throws IOException {
        Map<String, Object> cfg = Map.of();
        Path cfgFile = dir.resolve("config.json");
        if (Files.isRegularFile(cfgFile)) {
            cfg = Json.parseObject(Files.readString(cfgFile));
        }
        Checkpoint c = new Checkpoint(dir, cfg);
        try {
            c.load(stem);
        } catch (IOException | RuntimeException e) {
            c.close();
            throw e;
        }
        return c;
    }

    private void load(String stem) throws IOException {
        for (String variant : new String[]{".fp16", ""}) {
            Path single = dir.resolve(stem + variant + ".safetensors");
            if (Files.isRegularFile(single)) {
                add(single);
                return;
            }
            Path index = dir.resolve(stem + variant + ".safetensors.index.json");
            if (Files.isRegularFile(index)) {
                for (String shard : new TreeSet<>(SafeTensors.readWeightMap(index).values())) {
                    add(dir.resolve(shard));
                }
                return;
            }
        }
        throw new IOException("no " + stem + "[.fp16].safetensors (or index) in " + dir);
    }

    private void add(Path f) throws IOException {
        SafeTensors st = SafeTensors.open(f);
        files.add(st);
        paths.add(f);
        tensors.putAll(st.tensors());
    }

    public boolean has(String name) {
        return tensors.containsKey(name);
    }

    /** The {@code .safetensors} files the tensors were read from (one, or the shards in name order). */
    public List<Path> files() {
        return List.copyOf(paths);
    }

    public SafeTensors.Tensor get(String name) {
        SafeTensors.Tensor t = tensors.get(name);
        if (t == null) {
            throw new IllegalArgumentException("tensor " + name + " is not in " + dir);
        }
        return t;
    }

    public Map<String, SafeTensors.Tensor> tensors() {
        return tensors;
    }

    /** Total bytes of the tensors as stored. */
    public long bytes() {
        long n = 0;
        for (SafeTensors.Tensor t : tensors.values()) {
            n += t.numel() * t.dtype().bytes;
        }
        return n;
    }

    @Override
    public void close() {
        for (SafeTensors st : files) {
            st.close();
        }
        files.clear();
    }

    // ------------------------------------------------------------------ config helpers

    public int getInt(String key, int def) {
        return (int) Json.num(config, key, def);
    }

    public double getDouble(String key, double def) {
        return Json.dbl(config, key, def);
    }

    public String getString(String key, String def) {
        return Json.str(config, key, def);
    }

    public boolean getBool(String key, boolean def) {
        return Json.bool(config, key, def);
    }

    /** A config value that is an int or a list of ints, as a list of {@code n} entries (an int is repeated). */
    public int[] getIntList(String key, int n, int def) {
        Object v = config.get(key);
        int[] out = new int[n];
        if (v instanceof List<?> l) {
            for (int i = 0; i < n; i++) {
                out[i] = ((Number) l.get(Math.min(i, l.size() - 1))).intValue();
            }
        } else if (v instanceof Number x) {
            java.util.Arrays.fill(out, x.intValue());
        } else {
            java.util.Arrays.fill(out, def);
        }
        return out;
    }

    /** The entries of a config list of ints (or a single int as a one-entry list). */
    public int[] getIntArray(String key, int... def) {
        Object v = config.get(key);
        if (v instanceof List<?> l) {
            int[] out = new int[l.size()];
            for (int i = 0; i < out.length; i++) {
                out[i] = ((Number) l.get(i)).intValue();
            }
            return out;
        }
        if (v instanceof Number x) {
            return new int[]{x.intValue()};
        }
        return def;
    }

    public List<String> getStringList(String key) {
        List<String> out = new ArrayList<>();
        Object v = config.get(key);
        if (v instanceof List<?> l) {
            for (Object o : l) {
                out.add(String.valueOf(o));
            }
        }
        return out;
    }
}
