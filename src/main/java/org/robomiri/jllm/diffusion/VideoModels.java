package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.hub.HuggingFaceHub;
import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Resolving and loading a video model from the settings. A video model is an AnimateDiff <em>motion adapter</em> (the
 * Hugging Face repository {@code guoyww/animatediff-motion-adapter-v1-5-2}, a {@code config.json} and one safetensors file) used
 * with a Stable Diffusion 1.x model as its base ({@code video.base-model}, by default
 * {@code stable-diffusion-v1-5/stable-diffusion-v1-5}; any SD 1.5 fine-tune in the diffusers layout works). Both are
 * downloaded once into {@code model.dir}.
 */
public final class VideoModels {

    private VideoModels() {
    }

    /** The name a video model is served under: the adapter repository's name, or a local folder's name. */
    public static String defaultId(String source) {
        return ImageModels.defaultId(source);
    }

    /** True for a folder that holds a motion adapter ({@code config.json} of class {@code MotionAdapter}). */
    public static boolean isAdapterDir(Path dir) {
        Path cfg = dir.resolve("config.json");
        if (!Files.isRegularFile(cfg)) {
            return false;
        }
        try {
            return "MotionAdapter".equals(Json.str(Json.parseObject(Files.readString(cfg, StandardCharsets.UTF_8)), "_class_name", ""));
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    /** True for the file listing of a motion adapter repository. */
    public static boolean isAdapterRepo(List<HuggingFaceHub.RepoFile> files, String configText) {
        if (files.stream().noneMatch(f -> f.name().equals("config.json"))) {
            return false;
        }
        try {
            return configText != null && "MotionAdapter".equals(Json.str(Json.parseObject(configText), "_class_name", ""));
        } catch (RuntimeException e) {
            return false;
        }
    }

    /** Where the adapter's folder is: {@code source} itself if it is an adapter folder, else {@code model.dir/owner_name}. */
    public static Path adapterFolder(Settings s, String source) {
        Path local = Path.of(source);
        if (Files.isDirectory(local) && isAdapterDir(local)) {
            return local;
        }
        return Path.of(s.get("model.dir")).resolve(source.replace('/', '_'));
    }

    private static boolean hasWeights(Path dir) {
        return Files.isRegularFile(dir.resolve("diffusion_pytorch_model.fp16.safetensors"))
                || Files.isRegularFile(dir.resolve("diffusion_pytorch_model.safetensors"));
    }

    /** The adapter's folder, downloading its config and (half-precision, if the repository has them) weights first when missing. */
    public static Path resolveAdapter(Settings s, HuggingFaceHub hub, String source) throws IOException, InterruptedException {
        Path dir = adapterFolder(s, source);
        if (isAdapterDir(dir) && hasWeights(dir)) {
            return dir;
        }
        if (Files.isDirectory(Path.of(source))) {
            throw new IOException(source + " is not a motion adapter folder (config.json of class MotionAdapter and diffusion_pytorch_model[.fp16].safetensors)");
        }
        String revision = s.get("model.revision");
        List<HuggingFaceHub.RepoFile> listing = hub.listFiles(source, revision);
        Files.createDirectories(dir);
        hub.ensure(source, revision, dir, "config.json");
        if (!isAdapterDir(dir)) {
            throw new IOException(source + " is not an AnimateDiff motion adapter repository (its config.json is not of class MotionAdapter)");
        }
        String weights = listing.stream().anyMatch(f -> f.name().equals("diffusion_pytorch_model.fp16.safetensors"))
                ? "diffusion_pytorch_model.fp16.safetensors" : "diffusion_pytorch_model.safetensors";
        HuggingFaceHub.RepoFile wf = listing.stream().filter(f -> f.name().equals(weights)).findFirst()
                .orElseThrow(() -> new IOException(source + " has no diffusion_pytorch_model[.fp16].safetensors"));
        long free = Files.getFileStore(dir).getUsableSpace();
        if (wf.size() > free) {
            throw new IOException(String.format("need %.1f GB for %s but only %.1f GB are free", wf.size() / 1e9, source, free / 1e9));
        }
        System.err.printf("Downloading %s@%s (%s, %.1f MB) into %s%n", source, revision, weights, wf.size() / 1e6, dir);
        hub.ensure(source, revision, dir, weights);
        return dir;
    }

    /** Loads the video pipeline (base model + motion adapter) with the {@code video.*} settings. */
    public static VideoPipeline load(Settings s, HuggingFaceHub hub, String adapterSource) throws IOException, InterruptedException {
        return load(s, hub, adapterSource, s.get("video.base-model"));
    }

    public static VideoPipeline load(Settings s, HuggingFaceHub hub, String adapterSource, String baseSource) throws IOException, InterruptedException {
        Path adapter = resolveAdapter(s, hub, adapterSource);
        Path base = ImageModels.resolve(s, hub, baseSource);
        Devices dev = Devices.open(s.get("video.backend"), s.get("video.precision"), false);
        try {
            VideoPipeline p = new VideoPipeline(base, adapter, dev);
            p.setVaeTiling(s.getInt("image.vae-tile"), s.getInt("image.vae-tile-overlap"));
            p.setDecodeChunk(s.getInt("video.decode-chunk"));
            return p;
        } catch (IOException | RuntimeException e) {
            dev.close();
            throw e;
        }
    }
}
