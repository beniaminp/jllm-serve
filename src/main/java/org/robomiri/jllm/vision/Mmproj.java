package org.robomiri.jllm.vision;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.hub.HuggingFaceHub;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

/**
 * Finds the multimodal projector of a GGUF model: the file named by {@code model.mmproj}, or else an
 * {@code mmproj*.gguf} next to the model file or in its Hugging Face repository (downloaded when
 * {@code vision.auto-download} is on). Several candidates are ranked by how well the name matches the model's
 * ({@code mmproj-SmolVLM-256M-Instruct-f16.gguf} goes with {@code SmolVLM-256M-Instruct-Q8_0.gguf}) and then by
 * precision: F16, BF16, Q8_0, the rest.
 */
public final class Mmproj {

    private static final Pattern QUANT = Pattern.compile("[-_.](?:f16|f32|bf16|q\\d[\\w]*|iq\\d\\w*)$", Pattern.CASE_INSENSITIVE);

    private Mmproj() {
    }

    /** True for a file name that is a projector: {@code mmproj*.gguf}. */
    public static boolean isMmproj(String name) {
        String base = name.substring(name.lastIndexOf('/') + 1).toLowerCase(Locale.ROOT);
        return base.startsWith("mmproj") && base.endsWith(".gguf");
    }

    /** The name of the best projector in {@code names} for the model file {@code modelFile}, or null if there is none. */
    public static String choose(List<String> names, String modelFile) {
        List<String> cand = new ArrayList<>();
        for (String n : names) {
            if (isMmproj(n) && !n.endsWith(".incomplete")) {
                cand.add(n);
            }
        }
        if (cand.isEmpty()) {
            return null;
        }
        String stem = modelFile == null ? "" : stem(modelFile);
        String best = null;
        int bestScore = Integer.MIN_VALUE;
        for (String n : cand) {
            String low = n.toLowerCase(Locale.ROOT);
            int score = 0;
            if (!stem.isEmpty() && low.contains(stem)) {
                score += 100;
            }
            if (low.contains("f16") && !low.contains("bf16")) {
                score += 30;
            } else if (low.contains("bf16")) {
                score += 20;
            } else if (low.contains("q8_0")) {
                score += 10;
            } else if (low.contains("f32")) {
                score += 5;
            }
            if (score > bestScore || score == bestScore && n.compareTo(best) < 0) {
                best = n;
                bestScore = score;
            }
        }
        return best;
    }

    /** {@code SmolVLM-256M-Instruct-Q8_0.gguf} to {@code smolvlm-256m-instruct}: the name without the quantization. */
    static String stem(String modelFile) {
        String base = modelFile.substring(modelFile.lastIndexOf('/') + 1);
        if (base.toLowerCase(Locale.ROOT).endsWith(".gguf")) {
            base = base.substring(0, base.length() - 5);
        }
        base = base.replaceFirst("-\\d{5}-of-\\d{5}$", "");
        String prev;
        do {
            prev = base;
            base = QUANT.matcher(base).replaceFirst("");
        } while (!base.equals(prev));
        return base.toLowerCase(Locale.ROOT);
    }

    /**
     * Resolves the projector for a model loaded from {@code modelFile} (a GGUF), or returns null when the model has
     * none (a text-only model, or {@code vision.enabled=false}).
     *
     * @param model  the {@code model.repo} the user gave (a repository id or a local path), for the Hub lookup
     * @param hub    the Hub client, or null to look only on disk
     * @throws IOException if {@code model.mmproj} names a file that cannot be found
     */
    public static Path locate(Settings s, HuggingFaceHub hub, String model, Path modelFile) throws IOException {
        if (!s.getBool("vision.enabled")) {
            return null;
        }
        Path dir = modelFile.toAbsolutePath().getParent();
        String explicit = s.opt("model.mmproj");
        String revision = s.get("model.revision");
        boolean local = model.toLowerCase(Locale.ROOT).endsWith(".gguf");
        if (explicit != null) {
            Path p = Path.of(explicit);
            if (Files.isRegularFile(p)) {
                return p.toAbsolutePath();
            }
            if (dir != null && Files.isRegularFile(dir.resolve(explicit))) {
                return dir.resolve(explicit);
            }
            if (!local && hub != null) {
                try {
                    return hub.ensure(model, revision, dir, explicit);
                } catch (IOException e) {
                    throw new IOException("model.mmproj=" + explicit + ": not a file, not in " + dir + " and not downloadable ("
                            + e.getMessage() + ")", e);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException("interrupted");
                }
            }
            throw new IOException("model.mmproj=" + explicit + " is not a file (looked in the working directory and " + dir + ")");
        }
        List<String> names = new ArrayList<>();
        if (dir != null && Files.isDirectory(dir)) {
            try (var st = Files.list(dir)) {
                st.filter(Files::isRegularFile).forEach(p -> names.add(p.getFileName().toString()));
            }
        }
        String pick = choose(names, modelFile.getFileName().toString());
        if (pick != null) {
            return dir.resolve(pick);
        }
        if (local || hub == null || !s.getBool("vision.auto-download")) {
            return null;
        }
        try {
            // only a listing the download of the model already fetched: a model that is on disk does not go online
            List<HuggingFaceHub.RepoFile> listing = hub.cachedListing(model, revision);
            if (listing == null) {
                return null;
            }
            List<String> remote = new ArrayList<>();
            for (HuggingFaceHub.RepoFile f : listing) {
                remote.add(f.name());
            }
            String r = choose(remote, modelFile.getFileName().toString());
            if (r == null) {
                return null;
            }
            System.err.println("vision: downloading the projector " + r + " of " + model);
            return hub.ensure(model, revision, dir, r);
        } catch (IOException e) {
            System.err.println("note: cannot look for a vision projector of " + model + ": " + e.getMessage());
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }
}
