package org.robomiri.jllm.audio;

import java.util.List;
import java.util.function.BooleanSupplier;
import java.util.function.IntConsumer;

/** A text-to-speech model. */
public interface SpeechModel extends AutoCloseable {

    /** Output sample rate in Hz (24000 for the OuteTTS / WavTokenizer pair). */
    int sampleRate();

    /** The voices {@link #synthesize} accepts (a voice may also be omitted for the model's default). */
    List<String> voices();

    /** A description for the UI: model family, codec, device. */
    String describe();

    /**
     * Speech for {@code text} in {@code voice} ({@code null}: the default voice).
     *
     * @return mono samples in [-1, 1] at {@link #sampleRate()}
     */
    float[] synthesize(String text, String voice, Options o);

    /**
     * @param speed       0.25 .. 4, 1 = as generated (applied afterwards by time stretching, pitch kept)
     * @param temperature sampling temperature of the language model (0 = greedy)
     * @param seed        sampling seed, -1 for a random one
     * @param maxFrames   most codec frames (75 per second of audio) to generate for one request; 0 = the model's limit
     * @param cancelled   polled between tokens
     * @param progress    receives the number of codec frames generated so far
     */
    record Options(double speed, double temperature, long seed, int maxFrames, BooleanSupplier cancelled,
                   IntConsumer progress) {
        public static Options defaults() {
            return new Options(1.0, 0.1, -1, 0, () -> false, f -> { });
        }

        public Options withSpeed(double s) {
            return new Options(s, temperature, seed, maxFrames, cancelled, progress);
        }

        public Options withTemperature(double t) {
            return new Options(speed, t, seed, maxFrames, cancelled, progress);
        }

        public Options withSeed(long s) {
            return new Options(speed, temperature, s, maxFrames, cancelled, progress);
        }

        public Options withCancel(BooleanSupplier c) {
            return new Options(speed, temperature, seed, maxFrames, c, progress);
        }

        public Options withProgress(IntConsumer p) {
            return new Options(speed, temperature, seed, maxFrames, cancelled, p);
        }
    }

    @Override
    void close();
}
