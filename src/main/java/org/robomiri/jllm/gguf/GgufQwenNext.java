package org.robomiri.jllm.gguf;

import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The tensors of the Gated DeltaNet hybrids ({@code qwen35}: Qwen3.5, {@code qwen3next}: Qwen3-Next) in the layout the
 * engine's {@link org.robomiri.jllm.model.Qwen3NextLayer} wants, from llama.cpp's (convert_hf_to_gguf.py,
 * {@code Qwen3NextModel} / {@code Qwen3_5TextModel}). What the converter did, and what is undone or kept here:
 * <ul>
 *   <li><b>full-attention layers</b>: {@code attn_q} is the checkpoint's {@code q_proj}, per head {@code headDim} query
 *       rows then {@code headDim} output-gate rows. The engine keeps them as two matrices ({@code q_proj},
 *       {@code q_gate}), so the rows are gathered (a copy of the compressed rows). {@code attn_k}, {@code attn_v},
 *       {@code attn_output} are read in place.</li>
 *   <li><b>Gated DeltaNet layers</b>: {@code attn_qkv} = [q | k | v] and {@code attn_gate} = z are read in place as
 *       they are (the converter already flattened the checkpoint's per-key-head groups of Qwen3-Next into q, k, v, z
 *       blocks). {@code ssm_conv1d} {@code [convDim, kernel]}, {@code ssm_a} (= {@code -exp(A_log)}), {@code ssm_dt.bias}
 *       and {@code ssm_norm} are floats; {@code ssm_out} is read in place.</li>
 *   <li><b>beta and alpha</b> ({@code b}, {@code a}): {@code ssm_beta} / {@code ssm_alpha} (Qwen3.5, F32) or the fused
 *       {@code ssm_ba} (Qwen3-Next, rows [b a] per key head, regrouped here to [b (all heads) | a]) become one tiny
 *       Q8_0 matrix {@code [2 * valueHeads, dim]}.</li>
 *   <li><b>value-head order</b>: for Qwen3.5 the converter reorders the value heads from grouped-by-key-head to tiled
 *       (v head p reads key head {@code p % keyHeads}) in all of {@code attn_qkv}'s v rows, {@code attn_gate},
 *       {@code ssm_alpha}/{@code beta}, {@code ssm_conv1d}'s v channels, {@code ssm_a}, {@code ssm_dt} and the
 *       <em>columns</em> of {@code ssm_out}. Nothing is permuted back (the columns of a Q5_K matrix cannot be moved
 *       without requantizing): the architecture is flagged {@code tiled} and the kernels pick the key head that way.</li>
 *   <li>The norm gains already contain the {@code 1 +} the checkpoint's zero-centred norms need (the engine's cache
 *       folds it in too), and {@code ssm_norm} does not.</li>
 * </ul>
 * The multi-token-prediction block ({@code blk.<layers>.nextn.*}) is not read here.
 */
final class GgufQwenNext {

    private static final Pattern NAME = Pattern.compile("^model\\.layers\\.(\\d+)\\.(.+)$");

    private final GgufWeights w;
    private final GgufFile gguf;
    private final Arena arena;
    private final boolean qwen35;
    private final int valueHeads;
    private final int keyHeads;
    private final int headDim;
    private final int heads;

    GgufQwenNext(GgufWeights w, GgufFile gguf, Arena arena, String arch) {
        this.w = w;
        this.gguf = gguf;
        this.arena = arena;
        this.qwen35 = arch.startsWith("qwen35"); // qwen35 and qwen35moe: tiled value heads, separate beta / alpha
        this.valueHeads = (int) gguf.num(arch + ".ssm.time_step_rank", 0);
        this.keyHeads = (int) gguf.num(arch + ".ssm.group_count", 0);
        this.headDim = (int) gguf.num(arch + ".attention.key_length", 0);
        this.heads = (int) gguf.num(arch + ".attention.head_count", 0);
    }

    /** The canonical names this class maps (everything else goes through {@link GgufWeights}' plain table). */
    boolean owns(String name) {
        Matcher m = NAME.matcher(name);
        if (!m.matches()) {
            return false;
        }
        return switch (m.group(2)) {
            case "self_attn.q_proj.weight", "self_attn.q_gate", "linear_attn.in_proj", "linear_attn.in_proj_z",
                 "linear_attn.in_proj_ba", "linear_attn.conv1d.weight", "linear_attn.A", "linear_attn.dt_bias",
                 "linear_attn.norm.weight", "linear_attn.out_proj.weight" -> true;
            default -> false;
        };
    }

    private String[] files(String name) {
        Matcher m = NAME.matcher(name);
        m.matches();
        String b = "blk." + m.group(1) + ".";
        return switch (m.group(2)) {
            case "self_attn.q_proj.weight", "self_attn.q_gate" -> new String[]{b + "attn_q.weight"};
            case "linear_attn.in_proj" -> new String[]{b + "attn_qkv.weight"};
            case "linear_attn.in_proj_z" -> new String[]{b + "attn_gate.weight"};
            case "linear_attn.in_proj_ba" -> qwen35 ? new String[]{b + "ssm_beta.weight", b + "ssm_alpha.weight"}
                    : new String[]{b + "ssm_ba.weight"};
            case "linear_attn.conv1d.weight" -> new String[]{b + "ssm_conv1d.weight"};
            case "linear_attn.A" -> new String[]{b + "ssm_a"};
            case "linear_attn.dt_bias" -> new String[]{b + "ssm_dt.bias"};
            case "linear_attn.norm.weight" -> new String[]{b + "ssm_norm.weight"};
            case "linear_attn.out_proj.weight" -> new String[]{b + "ssm_out.weight"};
            default -> throw new IllegalArgumentException(name);
        };
    }

    boolean has(String name) {
        for (String f : files(name)) {
            if (!gguf.hasTensor(f)) {
                return false;
            }
        }
        return true;
    }

    private String[] require(String name) {
        if (!has(name)) {
            throw new IllegalArgumentException("tensor not in the GGUF file: " + name + " (" + String.join(" + ", files(name)) + ")");
        }
        return files(name);
    }

    QTensor tensor(String name) {
        String[] f = require(name);
        GgufFile.Tensor t = gguf.tensor(f[0]);
        if (name.endsWith("self_attn.q_proj.weight") || name.endsWith("self_attn.q_gate")) {
            boolean gate = name.endsWith("q_gate");
            int rows = Math.toIntExact(t.rows());
            if (heads <= 0 || rows != 2 * heads * headDim) {
                throw new IllegalArgumentException(t + " is not [dim, 2 x heads x headDim] (" + heads + " heads of " + headDim + ")");
            }
            int[] src = new int[heads * headDim];
            for (int h = 0; h < heads; h++) {
                for (int i = 0; i < headDim; i++) {
                    src[h * headDim + i] = h * 2 * headDim + (gate ? headDim : 0) + i;
                }
            }
            return gatherRows(t, src);
        }
        if (name.endsWith("in_proj_ba")) {
            return ba(f);
        }
        return w.matrix(t);
    }

    /** {@code [b (all value heads) | a]} as a Q8_0 matrix, from llama.cpp's separate or fused (per key head) rows. */
    private QTensor ba(String[] f) {
        int nv = valueHeads;
        if (qwen35) {
            GgufFile.Tensor b = gguf.tensor(f[0]);
            GgufFile.Tensor a = gguf.tensor(f[1]);
            if (b.rows() != nv || a.rows() != nv) {
                throw new IllegalArgumentException(b + " / " + a + ": expected " + nv + " rows each");
            }
            return w.requantize(2 * nv, Math.toIntExact(b.cols()), b, a);
        }
        GgufFile.Tensor t = gguf.tensor(f[0]);
        if (keyHeads <= 0 || t.rows() != 2L * nv || nv % keyHeads != 0) {
            throw new IllegalArgumentException(t + ": expected " + 2 * nv + " rows for " + nv + " value heads");
        }
        int r = nv / keyHeads;
        int[] src = new int[2 * nv];
        for (int part = 0; part < 2; part++) {
            for (int g = 0; g < keyHeads; g++) {
                for (int j = 0; j < r; j++) {
                    src[part * nv + g * r + j] = g * 2 * r + part * r + j;
                }
            }
        }
        return w.requantizeRows(t, src);
    }

    float[] floats(String name) {
        return w.floatsOf(gguf.tensor(require(name)[0]));
    }

    /** The rows {@code src[i]} of {@code t} as a matrix of the same format when the format is viewed in place, else Q8_0. */
    private QTensor gatherRows(GgufFile.Tensor t, int[] src) {
        QType q = GgmlCodec.nativeType(t.type());
        if (q == null) {
            return w.requantizeRows(t, src);
        }
        long rb = t.rowBytes();
        long size = rb * src.length;
        // page aligned and padded to whole pages: the Metal backend wraps the memory without copying it
        MemorySegment dst = arena.allocate((size + QTensor.PAGE - 1) / QTensor.PAGE * QTensor.PAGE, QTensor.PAGE).asSlice(0, size);
        MemorySegment in = gguf.data(t);
        for (int i = 0; i < src.length; i++) {
            MemorySegment.copy(in, src[i] * rb, dst, i * rb, rb);
        }
        w.countConverted(t.type(), rb * src.length);
        return QTensor.mapped(q, src.length, Math.toIntExact(t.cols()), dst);
    }
}
