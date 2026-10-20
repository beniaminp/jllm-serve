package org.robomiri.jllm.engine;

import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.sampling.Sampler;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.ChatTemplate;
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

    public record Result(String text, int[] tokens, String finishReason, Stats stats) {
    }

    /** Receives generated text as it becomes available. */
    @FunctionalInterface
    public interface TextSink {
        void accept(String text);
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
            return generateLocked(promptTokens, p, sink, cancelled);
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

    private Result generateLocked(int[] prompt, SamplingParams p, TextSink sink, BooleanSupplier cancelled) {
        int ctx = backend.contextLength();
        if (prompt.length >= ctx) {
            throw new IllegalArgumentException("prompt has " + prompt.length + " tokens, context is " + ctx);
        }
        if (cancelled.getAsBoolean()) { // the client went away while this request was queued
            return new Result("", new int[0], "cancelled", new Stats(prompt.length, 0, 0, 0, 0));
        }
        int vocab = Math.min(model.vocabSize(), tokenizer.vocabSize());
        float[] logits = new float[model.vocabSize()];
        Sampler sampler = new Sampler(p, vocab, prompt);

        // prefix cache: reuse the longest common prefix, but always recompute at least the last token
        int reuse = 0;
        int max = Math.min(cachedLen, prompt.length - 1);
        while (reuse < max && cached[reuse] == prompt[reuse]) {
            reuse++;
        }
        if (!backend.supportsPrefixReuse() && reuse < cachedLen) {
            reuse = 0; // a recurrent state cannot be rewound: only an exact extension of the cached tokens is reusable
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
            int n = Math.min(batch, prompt.length - pos);
            boolean last = pos + n == prompt.length;
            backend.forward(prompt, pos, n, pos, last ? logits : null);
            pos += n;
        }
        long t1 = System.nanoTime();

        List<Integer> out = new ArrayList<>();
        StreamDecoder dec = new StreamDecoder(tokenizer); // strips the leading space SentencePiece decoders drop, like decode()
        StringBuilder text = new StringBuilder();
        String pending = ""; // text held back because it could be the start of a stop string
        int maxStop = p.stop().stream().mapToInt(String::length).max().orElse(0);
        String finish = "length";
        boolean stopHit = false; // a stop string was found: everything after it is discarded
        int maxTokens = Math.min(p.maxTokens(), ctx - prompt.length);
        int[] one = new int[1];
        Speculator sp = spec;
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

        for (int i = 0; i < maxTokens; i++) {
            if (cancelled.getAsBoolean()) {
                finish = "cancelled";
                break;
            }
            boolean queued = rIdx < rLen;
            int tok = queued ? ready[rIdx] : sampler.sample(logits);
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
                if (sp != null && sp.step(tok, pos, ctx, maxTokens - i - 1, sampler)) {
                    ready = sp.ready();
                    rIdx = 0;
                    rLen = sp.readyCount();
                    rPre = sp.preAccepted();
                    pos = sp.newPos();
                } else {
                    one[0] = tok;
                    backend.forward(one, 0, 1, pos, logits);
                    pos++;
                }
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
        return new Result(text.toString(), tokens, finish, stats);
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
