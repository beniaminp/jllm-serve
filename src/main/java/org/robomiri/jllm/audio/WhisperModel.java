package org.robomiri.jllm.audio;

import org.robomiri.jllm.tokenizer.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;

/** A loaded Whisper checkpoint: weights, the net on a device, tokenizer and the decoding protocol. */
public final class WhisperModel implements AutoCloseable {

    public final WhisperWeights weights;
    public final WhisperNet net;
    public final Tokenizer tokenizer;
    public final WhisperTranscriber transcriber;
    public final Path dir;

    private WhisperModel(Path dir, WhisperWeights weights, WhisperNet net, Tokenizer tokenizer, WhisperTranscriber t) {
        this.dir = dir;
        this.weights = weights;
        this.net = net;
        this.tokenizer = tokenizer;
        this.transcriber = t;
    }

    /**
     * @param backend "cpu", "metal" or "auto" (Metal when the library and a GPU are there, else CPU)
     * @param threads CPU threads
     * @param residentMb heap budget for the weights as floats; a bigger model is converted per use from the mapped file
     */
    public static WhisperModel load(Path dir, String backend, int threads, long residentMb) throws IOException {
        return load(dir, backend, threads, residentMb, "auto");
    }

    /** @param metalDecoder "auto", "on" or "off": whether the decoder runs on the GPU too (the Metal backend) */
    public static WhisperModel load(Path dir, String backend, int threads, long residentMb, String metalDecoder)
            throws IOException {
        WhisperWeights w = WhisperWeights.load(dir, residentMb << 20);
        try {
            Tokenizer tok = Tokenizer.load(dir.resolve("tokenizer.json"));
            WhisperNet net = null;
            if (!"cpu".equals(backend)) {
                try {
                    net = WhisperMetalLoader.tryCreate(w, threads, metalDecoder);
                } catch (RuntimeException | LinkageError e) {
                    if ("metal".equals(backend)) {
                        throw e;
                    }
                    System.err.println("whisper: Metal not used (" + e.getMessage() + "), falling back to the CPU");
                }
                if (net == null && "metal".equals(backend)) {
                    throw new IllegalStateException("the Metal backend is not available");
                }
            }
            if (net == null) {
                net = new WhisperCpu(w, threads);
            }
            return new WhisperModel(dir, w, net, tok, new WhisperTranscriber(net, tok, dir));
        } catch (IOException | RuntimeException e) {
            w.close();
            throw e;
        }
    }

    @Override
    public void close() {
        net.close();
        weights.close();
    }
}
