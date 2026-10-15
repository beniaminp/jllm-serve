package org.robomiri.jllm.model;

/**
 * GLM rotary embedding (transformers' {@code apply_rotary_pos_emb} of modeling_glm.py): only the first
 * {@code rotaryDim} dimensions of a head are rotated, and the pairs are the adjacent elements {@code (2i, 2i+1)}
 * rather than {@code (i, i + rotaryDim/2)}:
 * <pre>
 *   x'[2i]   = x[2i]   * cos_i - x[2i+1] * sin_i
 *   x'[2i+1] = x[2i+1] * cos_i + x[2i]   * sin_i
 * </pre>
 * Dimensions {@code >= rotaryDim} are left unchanged.
 */
public final class InterleavedRope {

    private InterleavedRope() {
    }

    /**
     * Rotates one head in place.
     *
     * @param v         buffer holding the head
     * @param off       index of the head's first element
     * @param rotaryDim number of leading dimensions that rotate (even)
     * @param cos       cosine table; entry {@code i} of this position is {@code cos[tableOff + i]}
     * @param sin       sine table, same layout
     * @param tableOff  offset of this position's {@code rotaryDim/2} entries
     */
    public static void rotate(float[] v, int off, int rotaryDim, float[] cos, float[] sin, int tableOff) {
        for (int i = 0; i < rotaryDim / 2; i++) {
            float a = v[off + 2 * i];
            float b = v[off + 2 * i + 1];
            float cs = cos[tableOff + i];
            float sn = sin[tableOff + i];
            v[off + 2 * i] = a * cs - b * sn;
            v[off + 2 * i + 1] = b * cs + a * sn;
        }
    }
}
