package org.robomiri.jllm.sampling;

import org.robomiri.jllm.json.Json;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Constrained tool calls in the Hermes / Qwen format: free text, and wherever the model opens a call with
 * {@code <tool_call>}, exactly
 *
 * <pre>
 * &lt;tool_call&gt;
 * {"name": "NAME", "arguments": ARGS}
 * &lt;/tool_call&gt;
 * </pre>
 *
 * where NAME is one of the offered functions and ARGS a JSON object matching that function's {@code parameters}
 * schema (compiled leniently: what the automaton cannot enforce is allowed), written with the canonical
 * {@code json.dumps} whitespace of the chat templates ({@link JsonConstraint.State}). Text before, between and
 * after calls is not constrained, nor is a {@code <think>} block (a call written while thinking is not a call:
 * {@code ToolCallFormat.parse} ignores it too).
 *
 * <p>The envelope is fixed text, so after the opening tag the engine jump-forwards {@code \n{"name": "}, the rest of
 * the name once it is unambiguous, {@code ", "arguments": }, and after the arguments {@code }\n</tool_call>}.
 *
 * <p>Modes ({@link Spec}): with {@code tool_choice} "auto" the call is optional; "required" or a named function
 * forces it: the output must start with {@code <tool_call>} (after a thinking block when the model thinks), and may
 * end only after a call. With an answer schema ({@code response_format} next to {@code tools}) the output is either
 * calls or a JSON answer matching it.
 */
public final class ToolCallConstraint extends ByteConstraint {

    static final byte[] OPEN = "<tool_call>".getBytes(StandardCharsets.US_ASCII);
    static final byte[] THINK = "<think>".getBytes(StandardCharsets.US_ASCII);
    static final byte[] END_THINK = "</think>".getBytes(StandardCharsets.US_ASCII);
    /** The fixed parts of a call, around the name and the arguments. */
    static final byte[] HEAD = "\n{\"name\": \"".getBytes(StandardCharsets.US_ASCII);
    static final byte[] MID = ", \"arguments\": ".getBytes(StandardCharsets.US_ASCII);
    static final byte[] TAIL = "}\n</tool_call>".getBytes(StandardCharsets.US_ASCII);
    /** Whitespace allowed before a forced call or between calls in answer mode. */
    static final int MAX_WS = 8;

    /**
     * The compiled functions of a request: names and argument schemas, plus the mode. Immutable apart from its mask
     * cache; build it once per request (or keep it for a tool set that recurs) and create a constraint per sequence.
     */
    public static final class Spec {
        final byte[][] names; // as written between the quotes (JSON-escaped UTF-8)
        final JsonConstraint.Schema[] args;
        final boolean required;
        final boolean thinking;
        final JsonConstraint.Schema answer; // null = free text outside calls
        final MaskCache cache;

        private Spec(byte[][] names, JsonConstraint.Schema[] args, boolean required, boolean thinking,
                     JsonConstraint.Schema answer, MaskCache cache) {
            this.names = names;
            this.args = args;
            this.required = required;
            this.thinking = thinking;
            this.answer = answer;
            this.cache = cache;
        }

        /**
         * @param tools OpenAI tool entries ({@code {"type": "function", "function": {"name", "parameters"}}}, or
         *              the function objects themselves)
         * @throws IllegalArgumentException for a tool without a name or with a malformed schema
         */
        public static Spec compile(List<?> tools) {
            List<byte[]> names = new ArrayList<>();
            List<JsonConstraint.Schema> args = new ArrayList<>();
            for (Object t : tools) {
                Map<String, Object> fn = Json.map(t, "function");
                if (fn == null && t instanceof Map<?, ?>) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> m = (Map<String, Object>) t;
                    fn = m;
                }
                String name = Json.str(fn, "name", null);
                if (name == null || name.isEmpty()) {
                    throw new IllegalArgumentException("each tool needs a function name");
                }
                names.add(JsonConstraint.encode(name).getBytes(StandardCharsets.UTF_8));
                Map<String, Object> params = Json.map(fn, "parameters");
                JsonConstraint.Schema s;
                try {
                    s = params == null || params.isEmpty() ? JsonConstraint.Schema.anyObject()
                            : JsonConstraint.Schema.compile(params, true);
                } catch (IllegalArgumentException | IllegalStateException e) {
                    throw new IllegalArgumentException("tool " + name + ": " + e.getMessage());
                }
                if ((s.kinds() & K_OBJECT_KIND) == 0) {
                    s = JsonConstraint.Schema.anyObject(); // arguments are always an object
                }
                args.add(s);
            }
            if (names.isEmpty()) {
                throw new IllegalArgumentException("no tools");
            }
            return new Spec(names.toArray(new byte[0][]), args.toArray(new JsonConstraint.Schema[0]), false, false,
                    null, new MaskCache());
        }

        /** Only the function {@code name} may be called, and a call must be made (a named {@code tool_choice}). */
        public Spec only(String name) {
            byte[] n = JsonConstraint.encode(name).getBytes(StandardCharsets.UTF_8);
            for (int i = 0; i < names.length; i++) {
                if (Arrays.equals(names[i], n)) {
                    return new Spec(new byte[][]{n}, new JsonConstraint.Schema[]{args[i]}, true, thinking, answer,
                            new MaskCache());
                }
            }
            throw new IllegalArgumentException("tool_choice names the function '" + name + "', which is not in 'tools'");
        }

        /** @param required whether a call must be made ({@code tool_choice: "required"}) */
        public Spec required(boolean required) {
            return new Spec(names, args, required, thinking, answer, cache);
        }

        /**
         * @param thinking whether the model may open a {@code <think>} block before a forced call (Qwen3 with
         *                 thinking on); otherwise a forced call starts the output and is jump-forwarded at once
         */
        public Spec thinking(boolean thinking) {
            return new Spec(names, args, required, thinking, answer, cache);
        }

        /** @param answer the schema a non-call answer must match ({@code response_format}), or null for free text */
        public Spec answer(JsonConstraint.Schema answer) {
            if (answer == this.answer) {
                return this;
            }
            return new Spec(names, args, required, thinking, answer, new MaskCache());
        }

        public int size() {
            return names.length;
        }
    }

    private static final int K_OBJECT_KIND = 1; // JsonConstraint's K_OBJECT

    // phases
    private static final int FREE = 0, THINKING = 1, REQUIRE = 2, CALL = 3, ANSWER = 4, START = 5, AFTER = 6;
    // call segments
    private static final int S_HEAD = 0, S_NAME = 1, S_MID = 2, S_ARGS = 3, S_TAIL = 4;

    static final class TState implements ByteConstraint.State {
        final Spec spec;
        int phase;
        int openM; // free text: bytes of <tool_call> matched at the end
        int thinkM; // free text: bytes of <think> matched; thinking: bytes of </think>
        boolean thought; // a thinking block was closed (or the prompt closed it)
        int calls;
        // REQUIRE / AFTER / START: a tag being written, and the whitespace before it
        int tagLen;
        int tagMask; // 1: <tool_call> still possible, 2: <think> still possible
        int ws;
        // CALL
        int seg;
        int off;
        byte[] name = new byte[0];
        int chosen = -1;
        JsonConstraint.State args;
        // ANSWER / START
        JsonConstraint.State answer;

        TState(Spec spec) {
            this.spec = spec;
        }

        @Override
        public TState copy() {
            TState t = new TState(spec);
            t.phase = phase;
            t.openM = openM;
            t.thinkM = thinkM;
            t.thought = thought;
            t.calls = calls;
            t.tagLen = tagLen;
            t.tagMask = tagMask;
            t.ws = ws;
            t.seg = seg;
            t.off = off;
            t.name = name; // copied on write
            t.chosen = chosen;
            t.args = args == null ? null : args.copy();
            t.answer = answer == null ? null : answer.copy();
            return t;
        }

        /** The phase after a thinking block, or at the start (when the prompt opened none). */
        private void afterThinking() {
            if (spec.answer != null) {
                phase = spec.required ? REQUIRE : START;
                if (phase == START) {
                    answer = JsonConstraint.State.start(spec.answer, false);
                }
            } else {
                phase = spec.required && calls == 0 ? REQUIRE : FREE;
            }
            tagLen = 0;
            tagMask = 0;
            ws = 0;
            openM = 0;
            thinkM = 0;
        }

        @Override
        public boolean feed(int b) {
            return switch (phase) {
                case FREE -> free(b);
                case THINKING -> thinking(b);
                case CALL -> call(b);
                case ANSWER -> answer.feed(b);
                default -> tag(b); // REQUIRE, START, AFTER
            };
        }

        private boolean free(int b) {
            openM = OPEN[openM] == b ? openM + 1 : b == '<' ? 1 : 0;
            thinkM = THINK[thinkM] == b ? thinkM + 1 : b == '<' ? 1 : 0;
            if (openM == OPEN.length) {
                startCall();
            } else if (thinkM == THINK.length) {
                phase = THINKING;
                thinkM = 0;
                openM = 0;
            }
            return true;
        }

        private boolean thinking(int b) {
            thinkM = END_THINK[thinkM] == b ? thinkM + 1 : b == '<' ? 1 : 0;
            if (thinkM == END_THINK.length) {
                thought = true;
                afterThinking();
            }
            return true;
        }

        /** A tag must come (REQUIRE: a call, or a thinking block first; AFTER: another call; START: or the answer). */
        private boolean tag(int b) {
            if (tagLen == 0) {
                boolean wsOk = phase == AFTER || phase == START || thought;
                if (wsOk && (b == ' ' || b == '\n' || b == '\t' || b == '\r')) {
                    return ++ws <= MAX_WS;
                }
                if (phase == START && b != '<') { // the answer starts
                    phase = ANSWER;
                    return answer.feed(b);
                }
                tagMask = 1 | (phase == REQUIRE && spec.thinking && !thought ? 2 : 0);
            }
            int m = 0;
            if ((tagMask & 1) != 0 && tagLen < OPEN.length && OPEN[tagLen] == b) {
                m |= 1;
            }
            if ((tagMask & 2) != 0 && tagLen < THINK.length && THINK[tagLen] == b) {
                m |= 2;
            }
            if (m == 0) {
                return false;
            }
            tagMask = m;
            tagLen++;
            if ((m & 1) != 0 && tagLen == OPEN.length) {
                startCall();
            } else if ((m & 2) != 0 && tagLen == THINK.length) {
                phase = THINKING;
                thinkM = 0;
            }
            return true;
        }

        private void startCall() {
            phase = CALL;
            seg = S_HEAD;
            off = 0;
            name = new byte[0];
            chosen = -1;
            args = null;
            openM = 0;
            thinkM = 0;
            tagLen = 0;
            ws = 0;
        }

        private boolean call(int b) {
            switch (seg) {
                case S_HEAD, S_MID, S_TAIL -> {
                    byte[] lit = seg == S_HEAD ? HEAD : seg == S_MID ? MID : TAIL;
                    if (lit[off] != b) {
                        return false;
                    }
                    if (++off == lit.length) {
                        off = 0;
                        if (seg == S_HEAD) {
                            seg = S_NAME;
                        } else if (seg == S_MID) {
                            seg = S_ARGS;
                            args = JsonConstraint.State.start(spec.args[chosen], true);
                        } else { // the call is complete
                            calls++;
                            args = null;
                            phase = spec.answer != null ? AFTER : FREE;
                            openM = 0;
                            thinkM = 0;
                        }
                    }
                    return true;
                }
                case S_NAME -> {
                    byte[] n = Arrays.copyOf(name, name.length + 1);
                    n[name.length] = (byte) b;
                    for (byte[] cand : spec.names) {
                        if (cand.length >= n.length && Arrays.equals(cand, 0, n.length, n, 0, n.length)) {
                            name = n;
                            return true;
                        }
                    }
                    if (b == '"') {
                        for (int i = 0; i < spec.names.length; i++) {
                            if (Arrays.equals(spec.names[i], name)) {
                                chosen = i;
                                seg = S_MID;
                                off = 0;
                                return true;
                            }
                        }
                    }
                    return false;
                }
                default -> { // S_ARGS
                    if (!args.feed(b)) {
                        return false;
                    }
                    if (args.done()) {
                        seg = S_TAIL;
                        off = 0;
                    }
                    return true;
                }
            }
        }

        @Override
        public boolean canEnd() {
            return switch (phase) {
                case FREE -> !spec.required || calls > 0;
                case THINKING -> !spec.required && spec.answer == null;
                case AFTER -> true;
                case ANSWER -> answer.canEndValue();
                default -> false; // REQUIRE, START, CALL
            };
        }

        @Override
        public boolean done() {
            return phase == ANSWER && answer.done();
        }

        @Override
        public boolean wide() {
            return phase == CALL && seg == S_ARGS ? args.wide() : phase == ANSWER && answer.wide();
        }

        @Override
        public boolean allowsSpecial() {
            return phase == FREE || phase == THINKING;
        }

        @Override
        public boolean relaxed() {
            return phase == FREE || phase == THINKING;
        }

        @Override
        public Object key() {
            return switch (phase) {
                case CALL -> seg == S_ARGS ? keyOf("args", chosen, args.key())
                        : seg == S_NAME ? keyOf("name", new String(name, StandardCharsets.ISO_8859_1), 0)
                        : keyOf("lit", seg * 1000 + off, chosen); // (what follows the literal depends on the tool)
                case ANSWER -> keyOf("answer", 0, answer.key());
                case FREE, THINKING -> null;
                default -> keyOf("tag", phase * 1000 + tagLen * 10 + tagMask, List.of(ws, thought));
            };
        }

        private static Object keyOf(String what, Object a, Object b) {
            return b == null ? null : List.of(what, a, b);
        }
    }

    /**
     * @param tokens        the bytes of each token id ({@link JsonConstraint#tokenTable})
     * @param promptInThink whether the prompt left a {@code <think>} block open (the output starts inside it)
     */
    public ToolCallConstraint(Spec spec, byte[][] tokens, boolean promptInThink) {
        super(tokens, start(spec, promptInThink), spec.cache);
    }

    private ToolCallConstraint(ToolCallConstraint o) {
        super(o);
    }

    private static TState start(Spec spec, boolean inThink) {
        TState s = new TState(spec);
        if (inThink) {
            s.phase = THINKING;
        } else {
            s.afterThinking();
        }
        return s;
    }

    @Override
    public ToolCallConstraint copy() {
        return new ToolCallConstraint(this);
    }

    /** How many calls were completed so far. */
    public int calls() {
        return ((TState) state).calls;
    }
}
