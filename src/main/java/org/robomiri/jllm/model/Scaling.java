package org.robomiri.jllm.model;

import java.util.Arrays;

/** RoPE frequency scaling of a {@link RopeTable} ({@code rope_scaling} in config.json); see {@link Rope#invFreq}. */
public sealed interface Scaling permits Scaling.None, Scaling.Linear, Scaling.Llama3, Scaling.LongRope, Scaling.Yarn {

    None NONE = new None();

    /** Plain RoPE. */
    record None() implements Scaling {
    }

    /** Every frequency divided by {@code factor}. */
    record Linear(double factor) implements Scaling {
    }

    /** Llama 3.1+: long wavelengths divided by {@code factor}, short ones kept, a smooth blend in between. */
    record Llama3(double factor, double lowFreqFactor, double highFreqFactor, int originalMaxPositions) implements Scaling {
    }

    /**
     * Phi-3 LongRoPE: one divisor per rotary pair, the {@code shortFactor}s up to {@code originalMaxPositions}
     * and the {@code longFactor}s beyond; cos and sin are multiplied by {@code attentionFactor} (0 = derive it
     * from {@code factor}, see {@link Rope#attentionFactor}).
     */
    record LongRope(double factor, int originalMaxPositions, double[] shortFactor, double[] longFactor,
                    double attentionFactor) implements Scaling {
        @Override
        public boolean equals(Object o) {
            return o instanceof LongRope l && l.factor == factor && l.originalMaxPositions == originalMaxPositions
                    && Arrays.equals(l.shortFactor, shortFactor) && Arrays.equals(l.longFactor, longFactor)
                    && l.attentionFactor == attentionFactor;
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(shortFactor) * 31 + Arrays.hashCode(longFactor);
        }

        @Override
        public String toString() {
            return "LongRope[factor=" + factor + ", originalMaxPositions=" + originalMaxPositions + ", pairs="
                    + shortFactor.length + ", attentionFactor=" + attentionFactor + "]";
        }
    }

    /** DeepSeek YaRN: {@code mscale} / {@code mscaleAllDim} as in {@code modeling_deepseek.py}. */
    record Yarn(double factor, int originalMaxPositions, double betaFast, double betaSlow, double mscale,
                double mscaleAllDim) implements Scaling {
    }
}
