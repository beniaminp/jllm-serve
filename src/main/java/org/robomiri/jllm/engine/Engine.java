package org.robomiri.jllm.engine;

import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.sampling.Sampler;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.sampling.TokenConstraint;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.StreamDecoder;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;
import java.util.function.IntPredicate;
import java.util.function.Supplier;

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

    /**
     * @param draftedTokens  speculative tokens proposed for verification during this request (0 without speculation)
     * @param acceptedTokens how many of them the target model accepted
     */
    public record Stats(int promptTokens, int cachedTokens, int generatedTokens, double prefillMs, double decodeMs,
                        int draftedTokens, int acceptedTokens) {
        public Stats(int promptTokens, int cachedTokens, int generatedTokens, double prefillMs, double decodeMs) {
            this(promptTokens, cachedTokens, generatedTokens, prefillMs, decodeMs, 0, 0);
        }

        public double tokPerSec() {
            return decodeMs > 0 ? generatedTokens / (decodeMs / 1000.0) : 0;
        }
    }

    /**
     * @param logprobs       one entry per generated token when {@link Decoding#logprobs()} was asked for, else null
     * @param promptLogprobs one entry per prompt token (the first is null: nothing predicts it) when
     *                       {@link Decoding#promptLogprobs()} was asked for, else null
     */
    public record Result(String text, int[] tokens, String finishReason, Stats stats, List<TokenLogprob> logprobs,
                         List<TokenLogprob> promptLogprobs) {
        public Result(String text, int[] tokens, String finishReason, Stats stats) {
            this(text, tokens, finishReason, stats, null, null);
        }

        /** The sum of the generated tokens' log-probabilities (the sequence's log-likelihood); needs logprobs. */
        public double cumulativeLogprob() {
            double s = 0;
            for (TokenLogprob lp : logprobs) {
                s += lp.logprob();
            }
            return s;
        }
    }

    /** Receives generated text as it becomes available. */
    @FunctionalInterface
    public interface TextSink {
        void accept(String text);
    }

    /**
     * Per-request decoding options beyond {@link SamplingParams} (the OpenAI fields {@code logit_bias},
     * {@code logprobs}, {@code echo} and {@code response_format}).
     *
     * @param logitBias      added to the logits before sampling, by token id (length at least the vocabulary); null = none
     * @param logprobs       report the log-probability of every generated token ({@link TokenLogprob})
     * @param topLogprobs    and this many alternatives per position (0 = none)
     * @param promptLogprobs report the prompt tokens' log-probabilities too: the whole prompt is run through
     *                       {@link Backend#forwardAll} (no prefix cache, small chunks)
     * @param constraint     creates a fresh {@link TokenConstraint} for each sequence; null = unconstrained
     */
    public record Decoding(float[] logitBias, boolean logprobs, int topLogprobs, boolean promptLogprobs,
                           Supplier<TokenConstraint> constraint) {
        public static final Decoding NONE = new Decoding(null, false, 0, false, null);

        /** Whether every step needs its own logits on the CPU (no speculation, no pipelined greedy decoding). */
        boolean exact() {
            return logprobs || promptLogprobs || constraint != null;
        }
    }

    /** Receives the output of {@link #generate(int[], SamplingParams, Decoding, int, ChoiceSink, BooleanSupplier)}, choice by choice. */
    public interface ChoiceSink {
        /** Choice {@code index} starts (the choices run one after the other). */
        default void start(int index) {
        }

        /**
         * The prompt of choice {@code index} has been prefilled; {@code promptLogprobs} as in {@link Result} (null
         * unless {@link Decoding#promptLogprobs()}, and then only for choice 0: the others repeat it).
         */
        default void prefilled(int index, List<TokenLogprob> promptLogprobs) {
        }

        /** Text of choice {@code index}, as it becomes available. */
        void text(int index, String text);

        /** A generated token's log-probability (with {@link Decoding#logprobs()}), before the text it adds. */
        default void token(int index, TokenLogprob lp) {
        }

        /** Choice {@code index} is finished. */
        default void done(int index, Result r) {
        }
    }

    private final Backend backend;
    private final Tokenizer tokenizer;
    private final Architecture.Core model;
    private final Set<Integer> eosIds;
    private final ReentrantLock lock = new ReentrantLock(true);
    private final long queueTimeoutMs;
    private final int emptyPromptToken;

    private volatile int[] cached = new int[0]; // tokens currently in the KV cache
    private volatile int cachedLen;
    private final ChatTemplate.Format template;
    private Speculator spec; // null = no speculative decoding

    public Engine(Backend backend, Tokenizer tokenizer, Architecture arch, Set<Integer> eosIds) {
        this(backend, tokenizer, arch, eosIds, 0);
    }

    /**
     * @param queueTimeoutMs how long {@link #generate} may wait for the engine while another request runs
     *                       before throwing {@link OverloadedException}; 0 = wait indefinitely
     */
    public Engine(Backend backend, Tokenizer tokenizer, Architecture arch, Set<Integer> eosIds, long queueTimeoutMs) {
        this(backend, tokenizer, arch, eosIds, queueTimeoutMs, ChatTemplate.forModelType(arch.core().modelType(), tokenizer));
    }

    /** @param template the chat format of the model family (prompt rendering and end-of-turn tokens) */
    public Engine(Backend backend, Tokenizer tokenizer, Architecture arch, Set<Integer> eosIds, long queueTimeoutMs,
                  ChatTemplate.Format template) {
        this.backend = backend;
        this.tokenizer = tokenizer;
        this.model = arch.core();
        this.template = template;
        Set<Integer> stops = new java.util.HashSet<>(eosIds);
        for (int id : template.stopIds(tokenizer)) {
            stops.add(id);
        }
        this.eosIds = Set.copyOf(stops);
        if (this.eosIds.isEmpty()) {
            System.err.println("note: no end-of-sequence token id is known for this model (config.json,"
                    + " generation_config.json, tokenizer and chat template have none): generation ends only at"
                    + " max_tokens or a stop string");
        }
        this.queueTimeoutMs = queueTimeoutMs;
        // the token an empty prompt is replaced with: the tokenizer's EOS, else config.json's first eos_token_id,
        // else the smallest known stop id (a Set has no stable order; the choice must not vary between runs)
        int empty = tokenizer.eosId();
        if (empty < 0 && !model.eosTokenIds().isEmpty()) {
            empty = model.eosTokenIds().getFirst();
        }
        if (empty < 0) {
            empty = this.eosIds.stream().mapToInt(Integer::intValue).min().orElse(-1);
        }
        this.emptyPromptToken = empty;
    }

    /**
     * Turns on speculative decoding (docs/10 §10.9). Ignored, with a note, for a backend whose state cannot be
     * rewound (Mamba). The engine owns {@code draft} and closes it.
     *
     * @return whether speculation is active
     */
    public boolean enableSpeculation(Speculator.Config cfg, DraftModel draft) {
        if (cfg.mode() == Speculator.Mode.OFF) {
            return false;
        }
        if (!Speculator.supported(backend)) {
            System.err.println("note: engine.speculative is ignored: " + backend.name() + " cannot roll back its state");
            if (draft != null) {
                draft.close();
            }
            return false;
        }
        lock.lock();
        try {
            spec = new Speculator(cfg, backend, draft, model.vocabSize(), Math.min(model.vocabSize(), tokenizer.vocabSize()), eosIds);
        } finally {
            lock.unlock();
        }
        return true;
    }

    public Tokenizer tokenizer() {
        return tokenizer;
    }

    /** The model family's chat template. */
    public ChatTemplate.Format template() {
        return template;
    }

    /** Renders a conversation to prompt ids with the family template (content cannot inject control tokens). */
    public int[] renderChat(java.util.List<ChatTemplate.Message> messages, boolean enableThinking) {
        return template.renderTokens(tokenizer, messages, true, enableThinking);
    }

    public Backend backend() {
        return backend;
    }

    /** The token ids that end generation (end-of-sequence plus the template's end-of-turn ids). */
    public Set<Integer> stopTokenIds() {
        return eosIds;
    }

    /** How long a request may wait for this engine before {@link OverloadedException}; 0 = indefinitely. */
    public long queueTimeoutMs() {
        return queueTimeoutMs;
    }

    /** Whether a request is running (or the lock is otherwise held) right now. */
    public boolean busy() {
        return lock.isLocked();
    }

    /**
     * How many leading tokens of {@code prompt} (as {@link #preparePrompt} returns it) are already in the KV cache,
     * i.e. would not be prefilled again. A snapshot: another request may change it at any time.
     */
    public int cachedPrefix(int[] prompt) {
        int[] c = cached;
        int n = Math.min(cachedLen, Math.min(c.length, prompt.length - 1));
        int i = 0;
        while (i < n && c[i] == prompt[i]) {
            i++;
        }
        return !backend.supportsPrefixReuse() && i < cachedLen ? 0 : i;
    }

    public Result generate(String prompt, SamplingParams p, TextSink sink) {
        return generate(prompt, p, sink, () -> false);
    }

    /**
     * @param cancelled polled when generation starts, between prefill chunks and before every token; generation
     *                  stops with finish reason "cancelled" when true (a prefill stopped half-way stays cached)
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
        promptTokens = preparePrompt(promptTokens);
        acquire();
        try {
            return generateLocked(promptTokens, p, Decoding.NONE, sink, null, 0, cancelled);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Generates {@code n} independent completions of one prompt (OpenAI {@code n} / {@code best_of}), one after the
     * other under a single hold of the engine: the prompt is prefilled once, each further choice reuses it from the
     * prefix cache and recomputes only its last token. Choice {@code i} of a seeded request samples with seed
     * {@code seed + i} (choice 0 exactly like a single request); an unseeded one draws a fresh seed per choice.
     *
     * @param sink receives the choices' text and logprobs as they are produced; may be null
     * @throws OverloadedException if the engine stayed busy with other requests for longer than the queue timeout
     */
    public List<Result> generate(int[] promptTokens, SamplingParams p, Decoding d, int n, ChoiceSink sink,
                                 BooleanSupplier cancelled) {
        int[] prompt = preparePrompt(promptTokens);
        acquire();
        try {
            List<Result> results = new ArrayList<>(n);
            List<TokenLogprob> promptLp = null;
            for (int i = 0; i < n; i++) {
                int c = i;
                SamplingParams pi = i == 0 || p.seed() < 0 ? p : p.withSeed(p.seed() + i);
                Decoding di = i == 0 ? d : new Decoding(d.logitBias(), d.logprobs(), d.topLogprobs(), false, d.constraint());
                if (sink != null) {
                    sink.start(c);
                }
                Result r = generateLocked(prompt, pi, di, sink == null ? null : text -> sink.text(c, text), sink, c,
                        cancelled);
                if (i == 0) {
                    promptLp = r.promptLogprobs();
                } else if (d.promptLogprobs()) { // the prompt is the same for every choice: computed once
                    r = new Result(r.text(), r.tokens(), r.finishReason(), r.stats(), r.logprobs(), promptLp);
                }
                if (sink != null) {
                    sink.done(c, r);
                }
                results.add(r);
            }
            return results;
        } finally {
            lock.unlock();
        }
    }

    /**
     * The ids the model will actually see for a prompt: the family's BOS token is put in front when the model
     * expects one (Llama, Mistral, Gemma, DeepSeek were trained with it at the start of every sequence) and an
     * empty prompt becomes a single EOS. {@link #generate} applies this itself; callers that enforce the context
     * limit must measure the prepared prompt rather than the raw ids. Idempotent.
     */
    public int[] preparePrompt(int[] promptTokens) {
        int bos = tokenizer.promptBos(); // the tokenizer's add_bos_token rule, not a per-family guess
        if (bos >= 0 && (promptTokens.length == 0 || promptTokens[0] != bos)) {
            int[] withBos = new int[promptTokens.length + 1];
            withBos[0] = bos;
            System.arraycopy(promptTokens, 0, withBos, 1, promptTokens.length);
            return withBos;
        }
        if (promptTokens.length == 0) {
            if (emptyPromptToken < 0) {
                throw new IllegalArgumentException("empty prompt, and the model has no EOS token to start from");
            }
            return new int[]{emptyPromptToken};
        }
        return promptTokens;
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

    /** @param choices with {@code index}: told about the prefill and each token's logprob; may be null */
    private Result generateLocked(int[] prompt, SamplingParams p, Decoding d, TextSink sink, ChoiceSink choices,
                                  int index, BooleanSupplier cancelled) {
        int ctx = backend.contextLength();
        if (prompt.length >= ctx) {
            throw new IllegalArgumentException("prompt has " + prompt.length + " tokens, context is " + ctx);
        }
        if (cancelled.getAsBoolean()) { // the client went away while this request was queued
            return new Result("", new int[0], "cancelled", new Stats(prompt.length, 0, 0, 0, 0));
        }
        int vocab = Math.min(model.vocabSize(), tokenizer.vocabSize());
        float[] logits = new float[model.vocabSize()];
        Sampler sampler = new Sampler(p, vocab, prompt, d.logitBias());
        TokenConstraint con = d.constraint() == null ? null : d.constraint().get();
        // an end-of-sequence token is allowed exactly when the constrained output is complete
        IntPredicate allowed = con == null ? null : id -> eosIds.contains(id) ? con.canEnd() : con.allows(id);
        List<TokenLogprob> promptLp = d.promptLogprobs() ? new ArrayList<>(prompt.length) : null;

        // prefix cache: reuse the longest common prefix, but always recompute at least the last token
        int reuse = 0;
        int max = Math.min(cachedLen, prompt.length - 1);
        while (reuse < max && cached[reuse] == prompt[reuse]) {
            reuse++;
        }
        if (!backend.supportsPrefixReuse() && reuse < cachedLen) {
            reuse = 0; // a recurrent state cannot be rewound: only an exact extension of the cached tokens is reusable
        }
        if (promptLp != null) {
            reuse = 0; // every prompt position's logits are needed
            promptLp.add(null); // nothing predicts the first token
        }
        // the KV cache is about to be overwritten: should forward() (or the sink) throw, nothing of it can be
        // trusted, so the bookkeeping stays invalid until this request completes and sets it again below
        cachedLen = 0;
        long t0 = System.nanoTime();
        int pos = reuse;
        int batch = backend.maxBatch();
        while (pos < prompt.length) {
            if (pos > reuse && cancelled.getAsBoolean()) {
                // the client gave up during a long prefill: keep what is done, so that a retry of the same
                // prompt (agents retry after a timeout) continues from here instead of starting over
                cached = java.util.Arrays.copyOf(prompt, pos);
                cachedLen = pos;
                return new Result("", new int[0], "cancelled",
                        new Stats(prompt.length, reuse, 0, (System.nanoTime() - t0) / 1e6, 0));
            }
            if (promptLp != null) {
                pos = prefillAll(prompt, pos, logits, vocab, d.topLogprobs(), promptLp);
                continue;
            }
            int n = Math.min(batch, prompt.length - pos);
            boolean last = pos + n == prompt.length;
            backend.forward(prompt, pos, n, pos, last ? logits : null);
            pos += n;
        }
        long t1 = System.nanoTime();
        if (choices != null) {
            choices.prefilled(index, promptLp);
        }

        List<Integer> out = new ArrayList<>();
        StreamDecoder dec = new StreamDecoder(tokenizer); // strips the leading space SentencePiece decoders drop, like decode()
        StringBuilder text = new StringBuilder();
        String pending = ""; // text held back because it could be the start of a stop string
        int maxStop = p.stop().stream().mapToInt(String::length).max().orElse(0);
        String finish = "length";
        boolean stopHit = false; // a stop string was found: everything after it is discarded
        int maxTokens = Math.min(p.maxTokens(), ctx - prompt.length);
        int[] one = new int[1];
        boolean exact = d.exact(); // every step's logits are needed here: no speculation, no pipelined greedy steps
        float[] raw = d.logprobs() ? new float[vocab] : null; // the model's logits before the sampler changes them
        List<TokenLogprob> lps = d.logprobs() ? new ArrayList<>() : null;
        Speculator sp = exact ? null : spec;
        long drafted0 = 0;
        long accepted0 = 0;
        int[] ready = null; // tokens a verification pass already decided: ready[rIdx..rLen), the first rPre accepted
        int rIdx = 0;
        int rLen = 0;
        int rPre = 0;
        if (sp != null) {
            sp.begin(prompt);
            drafted0 = sp.drafted();
            accepted0 = sp.accepted();
        }

        // greedy without penalties on a backend that can pick the argmax itself: each plain step is queued before the
        // CPU has seen the previous token (the device feeds its own argmax), so the GPU never waits for the CPU
        Backend.GreedyPipe pipe = sampler.greedyPlain() && !exact ? backend.greedyPipe(vocab) : null;
        Runnable drainPipe = pipe == null ? null : pipe::drain;
        boolean fromPipe = false; // the next token is the result of the oldest pipelined step
        try {
            for (int i = 0; i < maxTokens; i++) {
                if (cancelled.getAsBoolean()) {
                    finish = "cancelled";
                    break;
                }
                if (con != null && con.mustEnd()) { // the constrained output is complete: end it
                    finish = "stop";
                    break;
                }
                boolean queued = rIdx < rLen;
                if (raw != null) {
                    System.arraycopy(logits, 0, raw, 0, vocab);
                }
                int tok = queued ? ready[rIdx] : fromPipe ? pipe.next()
                        : con != null ? sampler.sample(logits, allowed) : sampler.sample(logits);
                if (tok < 0) { // the constraint allows no token at all (a vocabulary without the needed bytes)
                    finish = "stop";
                    break;
                }
                boolean preAccepted = queued && rIdx < rPre;
                if (queued) {
                    rIdx++;
                }
                if (eosIds.contains(tok)) {
                    finish = "stop";
                    break;
                }
                if (!preAccepted) {
                    sampler.accept(tok);
                }
                if (con != null) {
                    con.accept(tok);
                }
                if (lps != null) {
                    TokenLogprob lp = TokenLogprob.of(raw, vocab, tok, d.topLogprobs());
                    lps.add(lp);
                    if (choices != null) {
                        choices.token(index, lp);
                    }
                }
                out.add(tok);
                if (sp != null) {
                    sp.onToken(tok);
                }
                String piece = dec.push(tok); // id-based: lets the decoder apply the first-token rules (Metaspace)
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
                if (i + 1 < maxTokens && rIdx >= rLen) { // (a queued token's K/V is already in the cache)
                    if (sp != null && sp.step(tok, pos, ctx, maxTokens - i - 1, sampler, drainPipe)) {
                        ready = sp.ready();
                        rIdx = 0;
                        rLen = sp.readyCount();
                        rPre = sp.preAccepted();
                        pos = sp.newPos();
                        fromPipe = false;
                    } else if (pipe != null) {
                        if (pipe.pending() == 0) {
                            pipe.submit(tok, pos); // else the step feeding tok at pos is already in flight
                        }
                        pos++;
                        if (i + 2 < maxTokens) {
                            pipe.submit(-1, pos); // feeds the token the step above will choose
                        }
                        fromPipe = true;
                    } else {
                        one[0] = tok;
                        backend.forward(one, 0, 1, pos, logits);
                        pos++;
                    }
                }
            }
        } finally {
            if (pipe != null) {
                pipe.drain(); // steps queued past the end write cache rows that are simply overwritten later
            }
        }
        if (!stopHit) { // EOS, length or cancelled: release the held-back text plus any incomplete UTF-8 sequence
            emit(pending + dec.flush(), text, sink);
        }
        long t2 = System.nanoTime();

        // remember what is in the cache: prompt + generated tokens that were fed through forward()
        int fed = Math.min(pos, prompt.length + out.size()); // positions [0, pos) hold valid K/V
        cached = new int[fed];
        System.arraycopy(prompt, 0, cached, 0, Math.min(fed, prompt.length));
        for (int i = 0; i < fed - prompt.length; i++) {
            cached[prompt.length + i] = out.get(i);
        }
        cachedLen = fed;

        int[] tokens = out.stream().mapToInt(Integer::intValue).toArray();
        Stats stats = new Stats(prompt.length, reuse, tokens.length, (t1 - t0) / 1e6, (t2 - t1) / 1e6,
                sp == null ? 0 : (int) (sp.drafted() - drafted0), sp == null ? 0 : (int) (sp.accepted() - accepted0));
        return new Result(text.toString(), tokens, finish, stats, lps, promptLp);
    }

    /**
     * Prefills from {@code pos} with the logits of every position ({@link Backend#forwardAll}, in chunks of at
     * most 16 tokens so that the logits buffer stays small), recording each next prompt token's log-probability.
     * The last position's logits go to {@code logits}.
     *
     * @return the new position
     */
    private int prefillAll(int[] prompt, int pos, float[] logits, int vocab, int top, List<TokenLogprob> promptLp) {
        int stride = logits.length;
        int n = Math.min(Math.min(16, backend.maxBatch()), prompt.length - pos);
        float[] rows = new float[n * stride];
        float[] row = new float[stride];
        backend.forwardAll(prompt, pos, n, pos, rows, stride);
        for (int i = 0; i < n; i++) {
            System.arraycopy(rows, i * stride, row, 0, stride);
            int next = pos + i + 1;
            if (next < prompt.length) {
                promptLp.add(TokenLogprob.of(row, vocab, prompt[next], top));
            } else {
                System.arraycopy(row, 0, logits, 0, stride);
            }
        }
        return pos + n;
    }

    /**
     * Takes over the part of {@code src}'s cache that {@code prompt} starts with, when that is longer than what this
     * engine holds for it: the keys and values are copied from {@code src}'s backend (same model, same cache layout)
     * instead of being prefilled. Both engines are locked meanwhile (in a fixed order, so two engines adopting from
     * each other cannot deadlock); a busy {@code src} is waited for.
     *
     * @return the number of positions now shared with {@code src} that were not cached here before, 0 if none
     */
    public int adoptPrefix(Engine src, int[] prompt) {
        String format = backend.stateFormat();
        if (src == this || format == null || !format.equals(src.backend.stateFormat())) {
            return 0;
        }
        Engine first = System.identityHashCode(this) <= System.identityHashCode(src) ? this : src;
        Engine second = first == this ? src : this;
        first.acquire();
        try {
            second.acquire();
            try {
                int have = cachedPrefix(prompt);
                int n = src.cachedPrefix(prompt);
                if (n <= have) {
                    return 0;
                }
                cachedLen = 0;
                backend.copyStateFrom(src.backend, n);
                cached = java.util.Arrays.copyOf(src.cached, n);
                cachedLen = n;
                return n - have;
            } finally {
                second.lock.unlock();
            }
        } finally {
            first.lock.unlock();
        }
    }

    /** How many positions the cache holds now (0 while a request is running). */
    public int cachedLength() {
        return cachedLen;
    }

    /** The tokens whose keys and values are in the cache now (a copy). */
    public int[] cachedTokens() {
        int n = cachedLen;
        return java.util.Arrays.copyOf(cached, Math.min(n, cached.length));
    }

    /** Whether {@link #saveCache} / {@link #loadCache} work with this backend. */
    public boolean canPersistCache() {
        return backend.stateFormat() != null;
    }

    /**
     * Saves the cached tokens and their keys and values to {@code file} (atomically: a temporary file is moved into
     * place), waiting for a running request to finish first.
     *
     * @param fingerprint identifies the model and the settings that change its keys and values; a saved cache is
     *                    only loaded back under the same fingerprint
     * @return the number of positions saved (0: nothing cached, or the backend cannot save)
     */
    public int saveCache(java.nio.file.Path file, String fingerprint) throws java.io.IOException {
        String format = backend.stateFormat();
        if (format == null) {
            return 0;
        }
        acquire();
        try {
            int n = cachedLen;
            if (n == 0) {
                return 0;
            }
            int[] tokens = java.util.Arrays.copyOf(cached, n);
            CacheFile.write(file, new CacheFile.Header(fingerprint, format, tokens), ch -> backend.saveState(n, ch));
            return n;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Replaces the cache with the one saved in {@code file}, so that a prompt starting with its tokens skips their
     * prefill. Waits for a running request to finish first.
     *
     * @return the number of positions restored
     * @throws java.io.IOException when the file is not a cache of this model (fingerprint), not of this backend
     *                             layout, or longer than the context; the current cache is then unchanged unless the
     *                             failure happened while reading the keys and values, in which case it is emptied
     */
    public int loadCache(java.nio.file.Path file, String fingerprint) throws java.io.IOException {
        String format = backend.stateFormat();
        if (format == null) {
            throw new java.io.IOException(backend.name() + " cannot load a saved cache");
        }
        acquire();
        try {
            CacheFile.Header h = CacheFile.read(file, fingerprint, format, (hdr, ch) -> {
                int n = hdr.tokens().length;
                if (n >= backend.contextLength()) {
                    throw new java.io.IOException("the saved cache holds " + n + " tokens, the context is "
                            + backend.contextLength());
                }
                cachedLen = 0; // invalid until the whole state is in
                backend.loadState(n, ch);
            });
            cached = h.tokens();
            cachedLen = h.tokens().length;
            return cachedLen;
        } finally {
            lock.unlock();
        }
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
        if (spec != null) {
            spec.close();
        }
        backend.close();
    }
}
