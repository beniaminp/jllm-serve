package org.robomiri.jllm.sampling;

import java.util.List;

/**
 * Decoding parameters.
 *
 * @param temperature       0 means greedy (argmax)
 * @param topK              0 disables top-k
 * @param topP              1 disables nucleus sampling
 * @param minP              0 disables min-p (keep tokens with p >= minP * p_max)
 * @param repetitionPenalty 1 disables; HF-style, applied to every token in prompt and output
 * @param presencePenalty   OpenAI-style, applied to tokens generated so far
 * @param frequencyPenalty  OpenAI-style, scaled by how often a generated token occurred
 * @param seed              -1 for a random seed
 * @param maxTokens         maximum number of generated tokens
 * @param stop              stop strings (not included in the output)
 */
public record SamplingParams(
        float temperature,
        int topK,
        float topP,
        float minP,
        float repetitionPenalty,
        float presencePenalty,
        float frequencyPenalty,
        long seed,
        int maxTokens,
        List<String> stop) {

    public SamplingParams {
        stop = stop == null ? List.of() : List.copyOf(stop);
        // (the record's fields are not assigned yet inside a compact constructor: describe the arguments instead)
        if (!Float.isFinite(temperature) || !Float.isFinite(topP) || !Float.isFinite(minP)
                || !Float.isFinite(repetitionPenalty) || !Float.isFinite(presencePenalty) || !Float.isFinite(frequencyPenalty)) {
            throw new IllegalArgumentException("sampling parameters must be finite numbers: "
                    + describe(temperature, topK, topP, minP, repetitionPenalty, presencePenalty, frequencyPenalty));
        }
        if (temperature < 0 || topK < 0 || topP <= 0 || topP > 1 || minP < 0 || minP > 1 || repetitionPenalty <= 0) {
            throw new IllegalArgumentException("invalid sampling parameters (temperature >= 0, top_k >= 0, 0 < top_p <= 1,"
                    + " 0 <= min_p <= 1, repetition_penalty > 0): "
                    + describe(temperature, topK, topP, minP, repetitionPenalty, presencePenalty, frequencyPenalty));
        }
    }

    private static String describe(float temperature, int topK, float topP, float minP, float repetitionPenalty,
                                   float presencePenalty, float frequencyPenalty) {
        return "temperature=" + temperature + ", top_k=" + topK + ", top_p=" + topP + ", min_p=" + minP
                + ", repetition_penalty=" + repetitionPenalty + ", presence_penalty=" + presencePenalty
                + ", frequency_penalty=" + frequencyPenalty;
    }

    public static SamplingParams defaults() {
        // Qwen3 generation_config.json values
        return new SamplingParams(0.6f, 20, 0.95f, 0f, 1f, 0f, 0f, -1, 2048, List.of());
    }

    /** The same parameters with another seed (OpenAI {@code n}: choice i of a seeded request uses seed + i). */
    public SamplingParams withSeed(long newSeed) {
        return new SamplingParams(temperature, topK, topP, minP, repetitionPenalty, presencePenalty,
                frequencyPenalty, newSeed, maxTokens, stop);
    }

    public SamplingParams withMaxTokens(int n) {
        return new SamplingParams(temperature, topK, topP, minP, repetitionPenalty, presencePenalty,
                frequencyPenalty, seed, n, stop);
    }
}
