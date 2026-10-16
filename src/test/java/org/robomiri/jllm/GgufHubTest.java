package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.hub.HuggingFaceHub;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/** Choosing the GGUF file of a repository (model.gguf-file / model.gguf-quant) and validating those settings. */
public final class GgufHubTest {

    static int passed;
    static int failed;

    static void check(boolean ok, String what) {
        if (ok) {
            passed++;
        } else {
            failed++;
            System.out.println("FAIL: " + what);
        }
    }

    private static final List<String> UNSLOTH = List.of(".gitattributes", "README.md", "config.json",
            "Qwen3-0.6B-BF16.gguf", "Qwen3-0.6B-IQ4_XS.gguf", "Qwen3-0.6B-Q2_K.gguf", "Qwen3-0.6B-Q4_0.gguf",
            "Qwen3-0.6B-Q4_1.gguf", "Qwen3-0.6B-Q4_K_M.gguf", "Qwen3-0.6B-Q4_K_S.gguf", "Qwen3-0.6B-Q8_0.gguf",
            "Qwen3-0.6B-UD-Q4_K_XL.gguf", "Qwen3-0.6B-UD-Q8_K_XL.gguf", "mmproj-F16.gguf", "imatrix_unsloth.dat");

    private static String select(List<String> names, String file, String quant) {
        try {
            return HuggingFaceHub.selectGguf(names, file, quant).getKey();
        } catch (IOException e) {
            return "ERROR " + e.getMessage();
        }
    }

    public static void main(String[] args) {
        check(select(UNSLOTH, null, "Q4_K_M").equals("Qwen3-0.6B-Q4_K_M.gguf"), "quant Q4_K_M");
        check(select(UNSLOTH, null, "q4_k_m").equals("Qwen3-0.6B-Q4_K_M.gguf"), "quant is case insensitive");
        check(select(UNSLOTH, null, "Q8_0").equals("Qwen3-0.6B-Q8_0.gguf"), "quant Q8_0 is not Q8_K_XL");
        check(select(UNSLOTH, null, "Q4_0").equals("Qwen3-0.6B-Q4_0.gguf"), "quant Q4_0");
        check(select(UNSLOTH, null, "Q4_K_XL").equals("Qwen3-0.6B-UD-Q4_K_XL.gguf"), "quant Q4_K_XL");
        String r = select(UNSLOTH, null, "Q4_K");
        check(r.startsWith("ERROR no GGUF file matches model.gguf-quant=Q4_K") && r.contains("Qwen3-0.6B-Q4_K_M.gguf")
                && r.contains("Qwen3-0.6B-Q8_0.gguf") && !r.contains("mmproj"), "Q4_K is a prefix of other labels: no match, files listed: " + r);
        check(select(UNSLOTH, null, "Q4").startsWith("ERROR no GGUF"), "Q4 alone matches nothing");
        check(select(UNSLOTH, "Qwen3-0.6B-Q4_1.gguf", null).equals("Qwen3-0.6B-Q4_1.gguf"), "exact file");
        check(select(UNSLOTH, "Qwen3-0.6B-Q4_1.gguf", "Q8_0").equals("Qwen3-0.6B-Q4_1.gguf"), "the exact file wins over the label");
        check(select(UNSLOTH, "Qwen3-0.6B-Q9.gguf", null).contains("no GGUF file matches model.gguf-file"), "unknown file lists the available ones");
        r = select(UNSLOTH, null, null);
        check(r.startsWith("ERROR more than one GGUF file matches") && r.contains("Q2_K") && r.contains("--gguf-quant"),
                "no selection in a repository of many files is an error that lists them: " + r.substring(0, Math.min(80, r.length())));
        check(select(List.of("model-Q4_K_M.gguf", "README.md"), null, null).equals("model-Q4_K_M.gguf"), "a single file needs no selection");
        check(select(List.of("a/b/model-Q4_K_M.gguf", "a/b/model-Q8_0.gguf"), null, "Q8_0").equals("a/b/model-Q8_0.gguf"),
                "files in a subdirectory");
        check(select(List.of("README.md"), null, null).startsWith("ERROR no .gguf model file"), "no gguf at all");

        List<String> split = List.of("M-Q4_K_M-00001-of-00003.gguf", "M-Q4_K_M-00002-of-00003.gguf", "M-Q4_K_M-00003-of-00003.gguf",
                "M-Q8_0.gguf");
        try {
            Map.Entry<String, List<String>> e = HuggingFaceHub.selectGguf(split, null, "Q4_K_M");
            check(e.getKey().equals("M-Q4_K_M-00001-of-00003.gguf") && e.getValue().size() == 3, "a split model is one choice: " + e);
            e = HuggingFaceHub.selectGguf(split, "M-Q4_K_M-00002-of-00003.gguf", null);
            check(e.getKey().equals("M-Q4_K_M-00001-of-00003.gguf") && e.getValue().size() == 3, "naming any part selects all parts");
        } catch (IOException ex) {
            check(false, "split: " + ex.getMessage());
        }
        check(HuggingFaceHub.isGgufRepo(List.of(new HuggingFaceHub.RepoFile("a.gguf", 1, null))), "repo of gguf files");
        check(!HuggingFaceHub.isGgufRepo(List.of(new HuggingFaceHub.RepoFile("a.gguf", 1, null),
                new HuggingFaceHub.RepoFile("model.safetensors", 1, null))), "a repo with safetensors is not a GGUF one");
        check(!HuggingFaceHub.isGgufRepo(List.of(new HuggingFaceHub.RepoFile("mmproj-x.gguf", 1, null))), "mmproj alone is not a model");

        // settings
        check(validates("--gguf-quant", "Q4_K_M"), "gguf-quant Q4_K_M validates");
        check(validates("--gguf-file", "x-Q8_0.gguf"), "gguf-file validates");
        check(!validates("--gguf-file", "x.bin"), "gguf-file must end in .gguf");
        check(!validates("--gguf-quant", "Q4 K"), "gguf-quant must be a label");
        check(validates("--model", "/tmp/some/model.gguf"), "a local path is a valid model");
        System.out.println("gguf hub: " + passed + " checks passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        }
    }

    private static boolean validates(String... flags) {
        try {
            String[] a = new String[flags.length + 1];
            a[0] = "run";
            System.arraycopy(flags, 0, a, 1, flags.length);
            Settings.load(a).validate();
            return true;
        } catch (IllegalArgumentException | IOException e) {
            return false;
        }
    }
}
