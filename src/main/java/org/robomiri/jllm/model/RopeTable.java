package org.robomiri.jllm.model;

import java.util.List;

/**
 * One rotary position embedding: which leading part of each head rotates, with which base, scaling and pairing.
 * {@link Rope} turns it into the cos/sin table of a context length.
 *
 * @param rotaryDim leading dimensions of each head that rotate ({@code head_dim * partial_rotary_factor}, even);
 *                  the frequencies are {@code theta^(-2i/rotaryDim)}
 * @param theta     {@code rope_theta}
 * @param mrope     Qwen-VL's multimodal RoPE: rotary pairs per axis (time, height, width), summing to
 *                  {@code rotaryDim / 2}; empty for a text-only rotary embedding. A model with sections still runs
 *                  with plain positions for text ({@link RopePositions} gives pictures theirs)
 * @param mropeInterleaved Qwen3-VL's interleaved assignment of the pairs to the axes
 * @param activePairs how many of the leading {@code rotaryDim / 2} pairs carry a frequency; the rest rotate by 0
 *                  (Gemma 4's proportional RoPE: {@code rotaryDim} is the whole head — so its pair offset is
 *                  {@code head_dim / 2} — but only {@code partial_rotary_factor} of the pairs are non-zero)
 */
public record RopeTable(int rotaryDim, double theta, Scaling scaling, Pairing pairing, List<Integer> mrope,
                        boolean mropeInterleaved, int activePairs) {

    public RopeTable(int rotaryDim, double theta, Scaling scaling, Pairing pairing, List<Integer> mrope,
                     boolean mropeInterleaved) {
        this(rotaryDim, theta, scaling, pairing, mrope, mropeInterleaved, rotaryDim / 2);
    }

    public RopeTable(int rotaryDim, double theta, Scaling scaling, Pairing pairing) {
        this(rotaryDim, theta, scaling, pairing, List.of(), false);
    }

    /** Which dimensions of a head form a rotated pair. */
    public enum Pairing {
        /** Rotate-half: {@code i} with {@code i + rotaryDim/2} (Llama and most families). */
        HALF,
        /** Adjacent: {@code 2i} with {@code 2i + 1} (GLM, the DeepSeek rotary slice). */
        INTERLEAVED
    }

    public RopeTable {
        if (rotaryDim < 2 || rotaryDim % 2 != 0 || !(theta > 0)) {
            throw new IllegalArgumentException("invalid rope table: rotaryDim=" + rotaryDim + " theta=" + theta);
        }
        mrope = List.copyOf(mrope);
        if (!mrope.isEmpty() && (mrope.size() != 3 || mrope.stream().mapToInt(Integer::intValue).sum() != rotaryDim / 2)) {
            throw new IllegalArgumentException("mrope sections " + mrope + " must be three numbers summing to " + rotaryDim / 2);
        }
        if (activePairs < 0 || activePairs > rotaryDim / 2) {
            throw new IllegalArgumentException("active rotary pairs " + activePairs + " of " + (rotaryDim / 2));
        }
    }
}
