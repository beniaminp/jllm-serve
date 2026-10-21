package org.robomiri.jllm;

import org.robomiri.jllm.metal.KernelParams;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Checks the Java side of every Metal kernel's parameter struct against {@code kernels.metal}: parses each
 * {@code struct ... { ... };} block of the shader source and asserts that {@link KernelParams} (the layouts every
 * dispatch is written against, see {@code MetalKernels} and {@code Params}) lists the same structs with the same
 * field count, kinds (int / float, in order) and byte size, and that no struct is left without a writer. Also
 * checks the constants the Java side mirrors from the shader's constants block. Dependency-free, no GPU needed:
 * {@code java -cp target/classes:target/test-classes org.robomiri.jllm.MetalParamsTest}.
 */
public final class MetalParamsTest {

    private static final Pattern STRUCT = Pattern.compile("struct\\s+(\\w+)\\s*\\{([^}]*)}\\s*;");
    private static final Pattern FIELD = Pattern.compile("^\\s*(int|float)\\s+(\\w+)\\s*;");
    private static final Pattern DEFINE = Pattern.compile("^#define\\s+(\\w+)\\s+(-?[0-9.]+)f?");

    public static void main(String[] args) throws Exception {
        run();
        System.out.println("MetalParamsTest passed");
    }

    public static void run() throws IOException {
        String src = source();
        Map<String, String> structs = structs(src);
        List<String> problems = new ArrayList<>();
        for (KernelParams p : KernelParams.values()) {
            String kinds = structs.get(p.struct);
            if (kinds == null) {
                problems.add(p + ": struct " + p.struct + " is not in kernels.metal");
                continue;
            }
            if (!kinds.equals(p.kinds)) {
                problems.add(p.struct + ": Java writer layout " + p.kinds + " (" + p.kinds.length() + " fields, "
                        + p.bytes() + " bytes) vs shader " + kinds + " (" + kinds.length() + " fields, " + kinds.length() * 4 + " bytes)");
            }
            structs.remove(p.struct);
        }
        for (String orphan : structs.keySet()) {
            problems.add("struct " + orphan + " has no KernelParams layout (no Java writer)");
        }
        checkConstants(src, problems);
        problems.addAll(org.robomiri.jllm.metal.ExpertPlacementCheck.run());
        if (!problems.isEmpty()) {
            throw new AssertionError("kernels.metal and the Java writers disagree:\n  " + String.join("\n  ", problems));
        }
        System.out.println("ok  " + KernelParams.values().length + " parameter structs match kernels.metal");
        System.out.println("ok  expert placement: uniform usage fills whole layers, skewed usage moves the coldest experts");
    }

    /** Every {@code #define NAME value} of the shader's constants block that the Java side mirrors. */
    private static void checkConstants(String src, List<String> problems) {
        Map<String, Double> defines = new LinkedHashMap<>();
        for (String line : src.split("\n")) {
            Matcher m = DEFINE.matcher(line);
            if (m.find()) {
                defines.put(m.group(1), Double.parseDouble(m.group(2)));
            }
        }
        Map<String, Double> mirrored = org.robomiri.jllm.metal.MetalConstants.mirrored();
        for (Map.Entry<String, Double> e : mirrored.entrySet()) {
            Double shader = defines.get(e.getKey());
            if (shader == null) {
                problems.add("constant " + e.getKey() + " is mirrored in Java but not defined in kernels.metal");
            } else if (!shader.equals(e.getValue())) {
                problems.add("constant " + e.getKey() + ": Java " + e.getValue() + " vs kernels.metal " + shader);
            }
        }
        System.out.println("ok  " + mirrored.size() + " mirrored constants match kernels.metal");
    }

    private static String source() throws IOException {
        try (InputStream in = KernelParams.class.getResourceAsStream("kernels.metal")) {
            if (in == null) {
                throw new IllegalStateException("kernels.metal resource missing");
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    /** struct name -> field kinds ("i" / "f" per field in order); comments are stripped first. */
    static Map<String, String> structs(String src) {
        String noComments = src.replaceAll("//[^\n]*", "");
        Map<String, String> out = new LinkedHashMap<>();
        Matcher m = STRUCT.matcher(noComments);
        while (m.find()) {
            StringBuilder kinds = new StringBuilder();
            for (String line : m.group(2).split("\n")) {
                Matcher f = FIELD.matcher(line);
                if (f.find()) {
                    kinds.append(f.group(1).equals("int") ? 'i' : 'f');
                } else if (!line.isBlank()) {
                    throw new AssertionError("struct " + m.group(1) + ": unexpected member '" + line.trim()
                            + "' (only int / float fields can be written from Java)");
                }
            }
            out.put(m.group(1), kinds.toString());
        }
        return out;
    }
}
