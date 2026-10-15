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
        if (temperature < 0 || topK < 0 || topP <= 0 || topP > 1 || minP < 0 || minP > 1 || repetitionPenalty <= 0) {
            throw new IllegalArgumentException("invalid sampling parameters: " + this);
        }
    }

    public static SamplingParams defaults() {
        // Qwen3 generation_config.json values
        return new SamplingParams(0.6f, 20, 0.95f, 0f, 1f, 0f, 0f, -1, 2048, List.of());
    }

    public SamplingParams withMaxTokens(int n) {
        return new SamplingParams(temperature, topK, topP, minP, repetitionPenalty, presencePenalty,
                frequencyPenalty, seed, n, stop);
    }
}
