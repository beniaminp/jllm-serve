package org.robomiri.jllm.model;

/**
 * One rotary position embedding: which leading part of each head rotates, with which base, scaling and pairing.
 * {@link Rope} turns it into the cos/sin table of a context length.
 *
 * @param rotaryDim leading dimensions of each head that rotate ({@code head_dim * partial_rotary_factor}, even);
 *                  the frequencies are {@code theta^(-2i/rotaryDim)}
 * @param theta     {@code rope_theta}
 */
public record RopeTable(int rotaryDim, double theta, Scaling scaling, Pairing pairing) {

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
    }
}
