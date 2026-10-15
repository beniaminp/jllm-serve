package org.robomiri.jllm.engine;

import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.ModelConfig;
import org.robomiri.jllm.sampling.Sampler;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.StreamDecoder;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;

/**
 * Generation loop: tokenizes, prefills in chunks, samples one token at a time and streams text.
 * The backend owns a single KV cache, so requests are serialized with a fair lock (a bounded wait, after
 * which {@link OverloadedException} is thrown); a prefix cache lets consecutive chat turns skip re-prefilling
 * the shared prefix.
 */
public final class Engine implements AutoCloseable {

    /** Thrown when a request waited longer than the queue timeout for the engine. */
    public static final class OverloadedException extends RuntimeException {
        OverloadedException(String message) {
            super(message);
        }
    }

    public record Stats(int promptTokens, int cachedTokens, int generatedTokens, double prefillMs, double decodeMs) {
        public double tokPerSec() {
            return decodeMs > 0 ? generatedTokens / (decodeMs / 1000.0) : 0;
        }
    }

    public record Result(String text, int[] tokens, String finishReason, Stats stats) {
    }

    /** Receives generated text as it becomes available. */
    @FunctionalInterface
    public interface TextSink {
        void accept(String text);
    }

    private final Backend backend;
    private final Tokenizer tokenizer;
    private final ModelConfig cfg;
    private final Set<Integer> eosIds;
    private final ReentrantLock lock = new ReentrantLock(true);
    private final long queueTimeoutMs;

    private int[] cached = new int[0]; // tokens currently in the KV cache
    private int cachedLen;

    public Engine(Backend backend, Tokenizer tokenizer, ModelConfig cfg, Set<Integer> eosIds) {
        this(backend, tokenizer, cfg, eosIds, 0);
    }

    /**
     * @param queueTimeoutMs how long {@link #generate} may wait for the engine while another request runs
     *                       before throwing {@link OverloadedException}; 0 = wait indefinitely
     */
    public Engine(Backend backend, Tokenizer tokenizer, ModelConfig cfg, Set<Integer> eosIds, long queueTimeoutMs) {
        this.backend = backend;
        this.tokenizer = tokenizer;
        this.cfg = cfg;
        this.eosIds = eosIds;
        this.queueTimeoutMs = queueTimeoutMs;
    }

    public Tokenizer tokenizer() {
        return tokenizer;
    }

    public Backend backend() {
        return backend;
    }

    public Result generate(String prompt, SamplingParams p, TextSink sink) {
        return generate(prompt, p, sink, () -> false);
    }

    /**
     * @param cancelled polled once when generation starts and then before every token; generation stops with
     *                  finish reason "cancelled" when true
     */
    public Result generate(String prompt, SamplingParams p, TextSink sink, BooleanSupplier cancelled) {
        return generate(tokenizer.encode(prompt), p, sink, cancelled);
    }

    /**
     * Generates from already-tokenized prompt ids (used by the chat template, which controls special tokens).
     *
     * @throws OverloadedException if the engine stayed busy with other requests for longer than the queue timeout
     */
    public Result generate(int[] promptTokens, SamplingParams p, TextSink sink, BooleanSupplier cancelled) {
        if (promptTokens.length == 0) {
            promptTokens = new int[]{eosIds.iterator().next()};
        }
        acquire();
        try {
            return generateLocked(promptTokens, p, sink, cancelled);
        } finally {
            lock.unlock();
        }
    }

    private void acquire() {
        try {
            if (queueTimeoutMs <= 0) {
                lock.lockInterruptibly();
            } else if (!lock.tryLock(queueTimeoutMs, TimeUnit.MILLISECONDS)) {
                throw new OverloadedException("engine busy: no capacity after waiting " + queueTimeoutMs + " ms");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new OverloadedException("interrupted while waiting for the engine");
        }
    }

    private Result generateLocked(int[] prompt, SamplingParams p, TextSink sink, BooleanSupplier cancelled) {
        int ctx = backend.contextLength();
        if (prompt.length >= ctx) {
            throw new IllegalArgumentException("prompt has " + prompt.length + " tokens, context is " + ctx);
        }
        if (cancelled.getAsBoolean()) { // the client went away while this request was queued
            return new Result("", new int[0], "cancelled", new Stats(prompt.length, 0, 0, 0, 0));
        }
        int vocab = Math.min(cfg.vocabSize(), tokenizer.vocabSize());
        float[] logits = new float[cfg.vocabSize()];
        Sampler sampler = new Sampler(p, vocab, prompt);

        // prefix cache: reuse the longest common prefix, but always recompute at least the last token
        int reuse = 0;
        int max = Math.min(cachedLen, prompt.length - 1);
        while (reuse < max && cached[reuse] == prompt[reuse]) {
            reuse++;
        }
        // the KV cache is about to be overwritten: should forward() (or the sink) throw, nothing of it can be
        // trusted, so the bookkeeping stays invalid until this request completes and sets it again below
        cachedLen = 0;
        long t0 = System.nanoTime();
        int pos = reuse;
        int batch = backend.maxBatch();
        while (pos < prompt.length) {
            int n = Math.min(batch, prompt.length - pos);
            boolean last = pos + n == prompt.length;
            backend.forward(prompt, pos, n, pos, last ? logits : null);
            pos += n;
        }
        long t1 = System.nanoTime();

        List<Integer> out = new ArrayList<>();
        StreamDecoder dec = new StreamDecoder();
        StringBuilder text = new StringBuilder();
        String pending = ""; // text held back because it could be the start of a stop string
        int maxStop = p.stop().stream().mapToInt(String::length).max().orElse(0);
        String finish = "length";
        boolean stopHit = false; // a stop string was found: everything after it is discarded
        int maxTokens = Math.min(p.maxTokens(), ctx - prompt.length);
        int[] one = new int[1];

        for (int i = 0; i < maxTokens; i++) {
            if (cancelled.getAsBoolean()) {
                finish = "cancelled";
                break;
            }
            int tok = sampler.sample(logits);
            if (eosIds.contains(tok)) {
                finish = "stop";
                break;
            }
            sampler.accept(tok);
            out.add(tok);
            String piece = dec.push(tokenizer.tokenBytes(tok));
            if (!piece.isEmpty()) {
                pending += piece;
                int hit = findStop(pending, p.stop());
                if (hit >= 0) {
                    emit(pending.substring(0, hit), text, sink);
                    pending = "";
                    stopHit = true;
                    finish = "stop";
                    break;
                }
                int safe = maxStop == 0 ? pending.length() : Math.max(0, pending.length() - maxStop + 1);
                if (safe > 0) {
                    emit(pending.substring(0, safe), text, sink);
                    pending = pending.substring(safe);
                }
            }
            if (i + 1 < maxTokens) {
                one[0] = tok;
                backend.forward(one, 0, 1, pos, logits);
                pos++;
            }
        }
        if (!stopHit) { // EOS, length or cancelled: release the held-back text plus any incomplete UTF-8 sequence
            emit(pending + dec.flush(), text, sink);
        }
        long t2 = System.nanoTime();

        // remember what is in the cache: prompt + generated tokens that were fed through forward()
        int fed = pos; // positions [0, pos) hold valid K/V
        cached = new int[fed];
        System.arraycopy(prompt, 0, cached, 0, Math.min(fed, prompt.length));
        for (int i = 0; i < fed - prompt.length; i++) {
            cached[prompt.length + i] = out.get(i);
        }
        cachedLen = fed;

        int[] tokens = out.stream().mapToInt(Integer::intValue).toArray();
        Stats stats = new Stats(prompt.length, reuse, tokens.length, (t1 - t0) / 1e6, (t2 - t1) / 1e6);
        return new Result(text.toString(), tokens, finish, stats);
    }

    private static void emit(String s, StringBuilder text, TextSink sink) {
        if (s.isEmpty()) {
            return;
        }
        text.append(s);
        if (sink != null) {
            sink.accept(s);
        }
    }

    private static int findStop(String s, List<String> stops) {
        int best = -1;
        for (String st : stops) {
            if (st.isEmpty()) {
                continue;
            }
            int i = s.indexOf(st);
            if (i >= 0 && (best < 0 || i < best)) {
                best = i;
            }
        }
        return best;
    }

    @Override
    public void close() {
        backend.close();
    }
}
