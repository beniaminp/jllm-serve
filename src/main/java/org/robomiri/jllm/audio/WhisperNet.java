package org.robomiri.jllm.audio;

/**
 * The numerical part of Whisper on one device: the encoder over one window of log-mel frames and the decoder with its
 * KV caches. The protocol around it (language detection, special tokens, timestamps, windows) is
 * {@link WhisperTranscriber}.
 */
public interface WhisperNet extends AutoCloseable {

    WhisperConfig config();

    /** "cpu" or "metal (device)". */
    String name();

    /**
     * Runs the encoder over a window of {@code mel[nMels][windowFrames]} and the cross-attention projections of every
     * decoder layer. The session owns the decoder's self-attention cache.
     */
    Session encode(float[][] mel);

    /** One window's decoder state. Not thread-safe. */
    interface Session extends AutoCloseable {

        /**
         * Feeds {@code n} tokens {@code tokens[off .. off+n)} at positions {@code [pos, pos+n)} (positions at and after
         * {@code pos} are overwritten) and writes the logits after the last one into {@code logits[0..vocab)}.
         */
        void decode(int[] tokens, int off, int n, int pos, float[] logits);

        /** The encoder output {@code [positions][dModel]} (tests and diagnostics; may be expensive). */
        float[] encoderOutput();

        @Override
        void close();
    }

    @Override
    void close();
}
