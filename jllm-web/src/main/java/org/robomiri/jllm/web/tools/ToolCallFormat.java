package org.robomiri.jllm.web.tools;

import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.sampling.JsonConstraint;
import org.robomiri.jllm.sampling.TokenConstraint;
import org.robomiri.jllm.sampling.ToolCallConstraint;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

/**
 * Function calling on top of the plain chat templates, in the Hermes / Qwen format: the function signatures go into
 * the system prompt inside {@code <tools></tools>}, the model answers with {@code <tool_call>{"name": ...,
 * "arguments": {...}}</tool_call>} blocks, and results come back as {@code <tool_response>} blocks (Qwen's templates
 * render a {@code tool} message that way natively; for the other families the result is a user turn). Qwen2.5 /
 * Qwen3 were trained on exactly this; Llama 3, Mistral and others follow it reasonably well, and a bare JSON object
 * naming a known function (Llama 3.2's habit) is accepted as a call too.
 *
 * <p>Generation is constrained ({@link #constraint}): once the model writes {@code <tool_call>}, the call can only be
 * {@code {"name": <one of the tools>, "arguments": <an object matching its schema>}} in the template's layout, and
 * the fixed parts of it are appended without sampling (docs/17 §17.5). Earlier calls in the conversation are
 * rendered in the same layout ({@link #canonical}).
 */
public final class ToolCallFormat {

    public static final String OPEN = "<tool_call>";
    public static final String CLOSE = "</tool_call>";
    private static final AtomicLong IDS = new AtomicLong();

    /**
     * One call the model made.
     *
     * @param error why the call could not be read (then {@code name} may be null): reported back to the model
     */
    public record Call(String id, String name, Map<String, Object> arguments, String error) {
        public String argumentsJson() {
            return Json.write(arguments == null ? Map.of() : arguments);
        }
    }

    /** @param content the text outside the tool calls (with any {@code <think>} block) */
    public record Parsed(String content, List<Call> calls) {
    }

    private ToolCallFormat() {
    }

    /** The system prompt section describing {@code tools} (OpenAI {@code {"type": "function", "function": ...}} entries). */
    public static String toolsBlock(List<Map<String, Object>> tools, String instruction) {
        StringBuilder sb = new StringBuilder();
        sb.append("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n")
                .append("You are provided with function signatures within <tools></tools> XML tags:\n<tools>");
        for (Map<String, Object> t : tools) {
            sb.append('\n').append(Json.write(t));
        }
        sb.append("\n</tools>\n\nFor each function call, return a json object with function name and arguments within"
                + " <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
        if (instruction != null && !instruction.isBlank()) {
            sb.append("\n\n").append(instruction);
        }
        return sb.toString();
    }

    /**
     * Converts OpenAI chat messages (with {@code tool_calls} on assistant messages and {@code tool} results) to the
     * template's messages; {@code toolsBlock}, when not null, is appended to the system prompt (one is created).
     *
     * @param template the family template's name: Qwen's render {@code tool} messages themselves
     */
    public static List<ChatTemplate.Message> toTemplate(List<Object> messages, String toolsBlock, String template) {
        boolean qwen = template.startsWith("chatml-qwen");
        List<ChatTemplate.Message> out = new ArrayList<>();
        boolean systemSeen = false;
        for (Object m : messages) {
            String role = Json.str(m, "role", "user");
            String content = text(m instanceof Map<?, ?> mm ? mm.get("content") : null);
            switch (role) {
                case "system", "developer" -> {
                    if (toolsBlock != null && !systemSeen) {
                        content = content.isEmpty() ? toolsBlock : content + "\n\n" + toolsBlock;
                    }
                    systemSeen = true;
                    out.add(new ChatTemplate.Message("system", content));
                }
                case "assistant" -> {
                    StringBuilder sb = new StringBuilder(content);
                    List<Object> calls = Json.list(m, "tool_calls");
                    if (calls != null) {
                        for (Object c : calls) {
                            Map<String, Object> fn = Json.map(c, "function");
                            String name = Json.str(fn, "name", "");
                            Object args = fn == null ? null : fn.get("arguments");
                            if (args instanceof String s) {
                                try {
                                    args = Json.parse(s);
                                } catch (RuntimeException e) {
                                    // keep the string
                                }
                            }
                            if (!sb.isEmpty()) {
                                sb.append('\n');
                            }
                            sb.append(OPEN).append('\n').append(canonical(Json.obj("name", name, "arguments", args == null ? Map.of() : args)))
                                    .append('\n').append(CLOSE);
                        }
                    }
                    out.add(new ChatTemplate.Message("assistant", sb.toString()));
                }
                case "tool", "function" -> {
                    if (qwen) {
                        out.add(new ChatTemplate.Message("tool", content));
                    } else {
                        String block = "<tool_response>\n" + content + "\n</tool_response>";
                        ChatTemplate.Message prev = out.isEmpty() ? null : out.getLast();
                        if (prev != null && prev.role().equals("user") && prev.content().endsWith("</tool_response>")) {
                            out.set(out.size() - 1, new ChatTemplate.Message("user", prev.content() + "\n" + block));
                        } else {
                            out.add(new ChatTemplate.Message("user", block));
                        }
                    }
                }
                default -> out.add(new ChatTemplate.Message(role, content));
            }
        }
        if (toolsBlock != null && !systemSeen) {
            out.addFirst(new ChatTemplate.Message("system", toolsBlock));
        }
        return out;
    }

    /**
     * JSON in the layout of the chat templates' {@code tojson} (Python's {@code json.dumps}): {@code ": "} after keys
     * and {@code ", "} between items, nothing else, non-ASCII characters as they are. The tool-call constraint makes
     * the model write its calls this way, so earlier calls in the prompt look the same.
     */
    public static String canonical(Object v) {
        StringBuilder sb = new StringBuilder();
        canonical(v, sb);
        return sb.toString();
    }

    private static void canonical(Object v, StringBuilder sb) {
        if (v instanceof Map<?, ?> m) {
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                sb.append(first ? "" : ", ").append(Json.write(String.valueOf(e.getKey()))).append(": ");
                canonical(e.getValue(), sb);
                first = false;
            }
            sb.append('}');
        } else if (v instanceof List<?> l) {
            sb.append('[');
            for (int i = 0; i < l.size(); i++) {
                sb.append(i == 0 ? "" : ", ");
                canonical(l.get(i), sb);
            }
            sb.append(']');
        } else {
            Json.write(v, sb);
        }
    }

    /** Compiled tool sets by (tools, tokenizer): agents send the same tools with every request. */
    private static final Map<SpecKey, ToolCallConstraint.Spec> SPECS = new LinkedHashMap<>(16, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<SpecKey, ToolCallConstraint.Spec> e) {
            return size() > 16;
        }
    };

    /** The tools' JSON and the tokenizer (by identity: the memoized masks are over its token ids). */
    private record SpecKey(String tools, Tokenizer tok) {
        @Override
        public boolean equals(Object o) {
            return o instanceof SpecKey(String t, Tokenizer k) && t.equals(tools) && k == tok;
        }

        @Override
        public int hashCode() {
            return tools.hashCode() * 31 + System.identityHashCode(tok);
        }
    }

    /**
     * The constraint for a chat request's tool calls (OpenAI {@code tools} / {@code tool_choice}), or null when no
     * tools are in play. {@code "auto"}: calls are optional and free text stays free; {@code "required"}: the reply
     * must start with a call (after a thinking block when the model thinks); a named function: that call. With
     * {@code answer} ({@code response_format}) a reply that is not a call must be that JSON.
     *
     * @param thinking      whether the model may think before a forced call (Qwen3 with thinking on)
     * @param promptInThink whether the prompt left a {@code <think>} block open
     * @throws IllegalArgumentException for a malformed parameter schema or an unknown named function
     */
    public static Supplier<TokenConstraint> constraint(Map<String, Object> body, Tokenizer tok,
                                                       JsonConstraint.Schema answer, boolean thinking,
                                                       boolean promptInThink) {
        List<Object> tools = Json.list(body, "tools");
        Object choice = body.get("tool_choice");
        if (tools == null || tools.isEmpty() || "none".equals(choice)) {
            return null;
        }
        SpecKey key = new SpecKey(Json.write(tools), tok);
        ToolCallConstraint.Spec spec;
        synchronized (SPECS) {
            spec = SPECS.get(key);
        }
        if (spec == null) {
            spec = ToolCallConstraint.Spec.compile(tools);
            synchronized (SPECS) {
                SPECS.put(key, spec);
            }
        }
        if (choice instanceof Map<?, ?>) {
            spec = spec.only(Json.str(Json.map(choice, "function"), "name", ""));
        } else if ("required".equals(choice)) {
            spec = spec.required(true);
        }
        ToolCallConstraint.Spec s = spec.thinking(thinking).answer(answer);
        byte[][] table = JsonConstraint.tokenTable(tok);
        return () -> new ToolCallConstraint(s, table, promptInThink);
    }

    /**
     * {@code decoding} with the tool-call constraint of the request in place of its {@code response_format} one
     * (which the tool constraint takes over as the answer schema); unchanged when no tools are in play.
     */
    public static Engine.Decoding decoding(Engine.Decoding decoding, Map<String, Object> body, Engine engine,
                                           int[] prompt, JsonConstraint.Schema answer, boolean thinking) {
        Supplier<TokenConstraint> c = constraint(body, engine.tokenizer(), answer,
                thinking && engine.template().name().equals("chatml-qwen3"),
                org.robomiri.jllm.web.chat.ReasoningSplitter.promptOpensThink(engine, prompt));
        return c == null ? decoding : new Engine.Decoding(decoding.logitBias(), decoding.logprobs(),
                decoding.topLogprobs(), decoding.promptLogprobs(), c);
    }

    /** A message's text: a string, or the text parts of a multi-part content array. */
    public static String text(Object content) {
        if (content instanceof String s) {
            return s;
        }
        if (content instanceof List<?> parts) {
            StringBuilder sb = new StringBuilder();
            for (Object p : parts) {
                String type = p instanceof Map<?, ?> ? Json.str(p, "type", "text") : "?";
                if (!type.equals("text")) {
                    throw new IllegalArgumentException("message content parts must be of type \"text\" (got \"" + type
                            + "\"); images and audio are not supported");
                }
                sb.append(Json.str(p, "text", ""));
            }
            return sb.toString();
        }
        return "";
    }

    /**
     * Splits generated text into the answer and the tool calls. Calls inside an unfinished {@code <think>} block do
     * not count; a reply that is only a JSON object naming one of {@code known} is a call as well.
     */
    public static Parsed parse(String text, Set<String> known) {
        int thinkEnd = text.lastIndexOf("</think>");
        int searchFrom = thinkEnd >= 0 ? thinkEnd + "</think>".length() : (text.contains("<think>") ? text.length() : 0);
        int first = text.indexOf(OPEN, searchFrom);
        List<Call> calls = new ArrayList<>();
        if (first < 0) {
            String tail = text.substring(searchFrom).strip();
            if (tail.startsWith("{") && tail.endsWith("}")) { // bare JSON
                Call c = readCall(tail);
                if (c.error() == null && known.contains(c.name())) {
                    return new Parsed(text.substring(0, searchFrom).strip(), List.of(c));
                }
            }
            return new Parsed(text, List.of());
        }
        StringBuilder content = new StringBuilder(text.substring(0, first));
        int pos = first;
        while (pos >= 0 && pos < text.length()) {
            int open = text.indexOf(OPEN, pos);
            if (open < 0) {
                content.append(text.substring(pos));
                break;
            }
            content.append(text, pos, open);
            int close = text.indexOf(CLOSE, open);
            String body = text.substring(open + OPEN.length(), close < 0 ? text.length() : close);
            calls.add(readCall(body.strip()));
            pos = close < 0 ? -1 : close + CLOSE.length();
        }
        return new Parsed(content.toString().strip(), calls);
    }

    private static Call readCall(String body) {
        String id = "call_" + Long.toHexString(System.nanoTime() & 0xffffffL) + IDS.incrementAndGet();
        Object o;
        try {
            o = Json.parse(body);
        } catch (RuntimeException e) {
            int a = body.indexOf('{');
            int b = body.lastIndexOf('}');
            try {
                o = a >= 0 && b > a ? Json.parse(body.substring(a, b + 1)) : null;
            } catch (RuntimeException e2) {
                o = null;
            }
            if (o == null) {
                return new Call(id, null, Map.of(), "the tool call is not valid JSON: " + e.getMessage());
            }
        }
        if (!(o instanceof Map<?, ?>)) {
            return new Call(id, null, Map.of(), "the tool call must be a JSON object with \"name\" and \"arguments\"");
        }
        String name = Json.str(o, "name", null);
        Object args = ((Map<?, ?>) o).get("arguments");
        if (args == null) {
            args = ((Map<?, ?>) o).get("parameters"); // Llama 3.x
        }
        if (args instanceof String s) {
            try {
                args = Json.parse(s);
            } catch (RuntimeException e) {
                return new Call(id, name, Map.of(), "the arguments of " + name + " are not valid JSON");
            }
        }
        if (name == null || name.isBlank()) {
            return new Call(id, null, Map.of(), "the tool call has no \"name\"");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> map = args instanceof Map<?, ?> m ? (Map<String, Object>) m : Map.of();
        return new Call(id, name, map, null);
    }

    /**
     * Passes generated text on until a tool call starts (outside a {@code <think>} block); from there on it holds
     * everything back, so a client streaming the answer never sees the raw call. Text that could be the start of
     * {@code <tool_call>} waits until it is decided.
     */
    public static final class Holdback implements Engine.TextSink {
        private final Engine.TextSink out;
        private final StringBuilder all = new StringBuilder();
        private int emitted;
        private boolean held;

        public Holdback(Engine.TextSink out) {
            this.out = out;
        }

        @Override
        public void accept(String text) {
            all.append(text);
            if (held) {
                return;
            }
            int think = all.lastIndexOf("<think>");
            int thinkEnd = all.lastIndexOf("</think>");
            boolean thinking = think >= 0 && thinkEnd < think;
            int from = Math.max(Math.max(0, emitted - OPEN.length()), thinkEnd < 0 ? 0 : thinkEnd + "</think>".length());
            int call = thinking ? -1 : all.indexOf(OPEN, from);
            if (call >= 0) {
                held = true;
                flushTo(call);
                return;
            }
            int safe = all.length();
            if (!thinking) { // keep back a suffix that may grow into "<tool_call>"
                for (int k = Math.min(OPEN.length() - 1, all.length()); k > 0; k--) {
                    if (OPEN.startsWith(all.substring(all.length() - k))) {
                        safe = all.length() - k;
                        break;
                    }
                }
            }
            flushTo(safe);
        }

        /** Releases what is still held back unless a call started. */
        public void finish() {
            if (!held) {
                flushTo(all.length());
            }
        }

        public boolean callStarted() {
            return held;
        }

        private void flushTo(int end) {
            if (end > emitted) {
                out.accept(all.substring(emitted, end));
                emitted = end;
            }
        }
    }
}
