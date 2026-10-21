package org.robomiri.jllm;

import java.nio.file.Path;
import java.util.LinkedHashMap;

/**
 * Writes the tiny Qwen3.5 / Qwen3-Next GGUF files of {@link GgufQwenNextTest} (with a vocabulary, so llama.cpp can load
 * them) for comparing the engine with llama.cpp on the same file: {@code GgufTinyWrite qwen35|qwen3next out.gguf [q8|f32|deq [ba]]}.
 */
public final class GgufTinyWrite {

    public static void main(String[] args) throws Exception {
        boolean qwen35 = args[0].equals("qwen35");
        for (String a : args) {
            if (a.startsWith("layers=")) {
                GgufQwenNextTest.LAYERS = Integer.parseInt(a.substring(7));
            } else if (a.equals("focus")) {
                GgufQwenNextTest.focus = true;
            } else if (a.equals("allmoe")) {
                GgufQwenNextTest.allMoe = true;
            } else if (a.startsWith("zero=")) {
                GgufQwenNextTest.zero = a.substring(5);
            } else if (a.startsWith("interval=")) {
                GgufQwenNextTest.INTERVAL = Integer.parseInt(a.substring(9));
            }
        }
        GgufQwenNextTest.withVocab = true;
        GgufQwenNextTest.quantizeBa = args.length > 3 && args[3].equals("ba");
        GgufQwenNextTest.Tiny t = GgufQwenNextTest.tiny(!qwen35, 7 + args[0].hashCode());
        boolean q8 = args.length > 2 && !args[2].equals("f32");
        if (args.length > 2 && args[2].equals("deq")) { // F32 tensors holding exactly the values a Q8_0 file would: no quantization left
            java.util.Map<String, float[]> eff = new LinkedHashMap<>();
            GgufQwenNextTest.write(Path.of(args[1] + ".tmp"), t, qwen35, true, eff, false);
            GgufQwenNextTest.write(Path.of(args[1]), t, qwen35, false, eff, true);
            java.nio.file.Files.delete(Path.of(args[1] + ".tmp"));
            q8 = false;
        } else {
            GgufQwenNextTest.write(Path.of(args[1]), t, qwen35, q8, new LinkedHashMap<>(), false);
        }
        System.out.println("wrote " + args[1]);
    }
}
