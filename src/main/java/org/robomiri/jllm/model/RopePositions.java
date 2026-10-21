package org.robomiri.jllm.model;

/**
 * The rotary positions of a sequence that does not simply count its cache slots: Qwen-VL's multimodal RoPE
 * ("M-RoPE"). Each rotary pair of a head is turned by the position of one of three axes: time {@code t}, row
 * {@code h} and column {@code w}. A text token has the same position on all three; an image token keeps {@code t} at
 * the image's start and counts rows and columns, and the text after an image continues from the largest position the
 * image used, which is less than the number of cache slots the image took. So "position of cache slot {@code i}" is a
 * function of the sequence, and a backend takes it as a table: {@link Rope#rows} fills the cos/sin rows of the slots
 * from it, and every kernel keeps reading {@code cos[slot * half + pair]} as before.
 *
 * <p>Slots at or after {@code t.length} are text after the last known token: slot {@code i} has position
 * {@code i - tailShift} on all axes.
 *
 * @param t, h, w  position per axis of slot {@code i < t.length}
 * @param sections rotary pairs per axis ({@code t, h, w}); they sum to the rotary dimension / 2 (a 4th section of
 *                 llama.cpp's {@code rope.dimension_sections} is unused by Qwen-VL and is not carried here)
 * @param interleaved Qwen3-VL's layout: pair {@code j} uses axis {@code j % 3} while that axis still has pairs left
 *                    ({@code j < 3 * sections[axis]}), else the time axis
 * @param tailShift cache slots minus positions for the slots after the known ones
 */
public record RopePositions(int[] t, int[] h, int[] w, int[] sections, boolean interleaved, int tailShift) {

    public RopePositions {
        if (t.length != h.length || t.length != w.length || sections.length != 3) {
            throw new IllegalArgumentException("inconsistent rope positions");
        }
    }

    /** The axis (0 = t, 1 = h, 2 = w) that rotary pair {@code pair} turns with. */
    public int axisOf(int pair) {
        if (interleaved) {
            int axis = pair % 3;
            return pair < 3 * sections[axis] ? axis : 0;
        }
        if (pair < sections[0]) {
            return 0;
        }
        return pair < sections[0] + sections[1] ? 1 : 2;
    }

    /** The position that rotary pair {@code pair} of slot {@code slot} turns by. */
    public int position(int slot, int pair) {
        if (slot >= t.length) {
            return slot - tailShift;
        }
        return switch (axisOf(pair)) {
            case 0 -> t[slot];
            case 1 -> h[slot];
            default -> w[slot];
        };
    }
}
