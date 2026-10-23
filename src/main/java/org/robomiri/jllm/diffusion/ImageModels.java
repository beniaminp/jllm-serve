package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.hub.HuggingFaceHub;
import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/** Resolving and loading an image model from the settings: a local diffusers folder, or a Hub repository downloaded once. */
public final class ImageModels {

    private ImageModels() {
    }

    /** The name a model is served under: the repository name, or a local folder's name. */
    public static String defaultId(String source) {
        String s = source.endsWith("/") ? source.substring(0, source.length() - 1) : source;
        int slash = s.lastIndexOf('/');
        return slash >= 0 ? s.substring(slash + 1) : s;
    }

    /** Where the model's folder is: {@code source} itself if it is a diffusers folder, else {@code model.dir/owner_name}. */
    public static Path folder(Settings s, String source) {
        Path local = Path.of(source);
        if (Files.isDirectory(local) && DiffusersRepo.isDiffusersDir(local)) {
            return local;
        }
        return Path.of(s.get("model.dir")).resolve(source.replace('/', '_'));
    }

    /** The folder of the model, downloading the missing files from the Hub first. */
    public static Path resolve(Settings s, HuggingFaceHub hub, String source) throws IOException, InterruptedException {
        Path dir = folder(s, source);
        if (DiffusersRepo.isDiffusersDir(dir) && DiffusersRepo.missingWeights(dir).isEmpty()) {
            return dir;
        }
        if (Files.isDirectory(Path.of(source))) {
            throw new IOException(source + " is not a complete diffusers folder (model_index.json and the components' weights)");
        }
        DiffusersRepo.download(hub, source, s.get("model.revision"), dir);
        return dir;
    }

    /** True when the source is (or, for a Hub name, would be) a diffusers pipeline: judged by the local folder only. */
    public static boolean isLocalImageModel(Settings s, String source) {
        return DiffusersRepo.isDiffusersDir(folder(s, source));
    }

    /**
     * {@code image.turbo}: {@code auto} keeps the name-based detection (a folder whose name contains {@code turbo}, or an
     * {@code lcm} model with trailing timestep spacing), {@code true} / {@code false} force the distilled defaults on or off
     * (so a renamed Turbo folder, or a fine-tune without "turbo" in its path, gets the right ones).
     */
    private static Boolean turboOverride(Settings s) {
        return switch (s.get("image.turbo").toLowerCase(java.util.Locale.ROOT)) {
            case "true" -> Boolean.TRUE;
            case "false" -> Boolean.FALSE;
            default -> null;
        };
    }

    /** Loads the pipeline with the {@code image.*} settings. */
    public static ImagePipeline load(Settings s, HuggingFaceHub hub, String source) throws IOException, InterruptedException {
        Path dir = resolve(s, hub, source);
        Map<String, Object> index = Json.parseObject(Files.readString(dir.resolve("model_index.json"), StandardCharsets.UTF_8));
        boolean xl = index.containsKey("text_encoder_2");
        Devices dev = Devices.open(s.get("image.backend"), s.get("image.precision"), xl);
        try {
            ImagePipeline p = new ImagePipeline(dir, dev, turboOverride(s));
            p.setVaeTiling(s.getInt("image.vae-tile"), s.getInt("image.vae-tile-overlap"));
            return p;
        } catch (IOException | RuntimeException e) {
            dev.close();
            throw e;
        }
    }
}
