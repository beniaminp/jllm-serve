package org.robomiri.jllm.tokenizer;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.function.Supplier;

/**
 * Chat templates of the supported model families, each derived from the Jinja {@code chat_template} in the
 * model's tokenizer_config.json (minus tool calling) and checked against the official Jinja render.
 *
 * <p>Get one with {@link #forModelType(String)} / {@link #forModelType(String, Tokenizer)}; {@code enableThinking}
 * only matters to Qwen3 (when false, an empty think block is pre-filled so the model answers directly).
 *
 * <p>Control tokens come from the template; message content is encoded with special tokens disabled (plus the
 * format's {@link Format#protectedTokens()}), so text like {@code <|im_end|>} inside a message cannot forge a
 * turn boundary, while non-special added tokens such as {@code <think>} are matched as the reference does. The
 * rendered prompt already contains the
 * family's BOS token when its template has one (Llama 3, Mistral, Gemma, GLM-4, DeepSeek), so callers must not
 * prepend {@link Tokenizer#bosId()} again.
 */
public final class ChatTemplate {

    public record Message(String role, String content) {
    }

    /** A chat template bound to one model family. */
    public interface Format {
        /** Short family name, e.g. {@code chatml-qwen3}, {@code llama3}, {@code gemma}. */
        String name();

        /** The prompt as text, exactly as the official Jinja template renders it. */
        String render(List<Message> messages, boolean addGenerationPrompt, boolean enableThinking);

        /** The prompt as token ids (content tokenized with special tokens disabled). */
        int[] renderTokens(Tokenizer tok, List<Message> messages, boolean addGenerationPrompt, boolean enableThinking);

        /**
         * {@link #renderTokens(Tokenizer, List, boolean, boolean)} that reuses the ids of an earlier prompt this one
         * extends; the ids are the same. {@code cache} may be null, and is ignored when it is for another tokenizer.
         */
        default int[] renderTokens(Tokenizer tok, List<Message> messages, boolean addGenerationPrompt,
                                   boolean enableThinking, PromptCache cache) {
            return renderTokens(tok, messages, addGenerationPrompt, enableThinking);
        }

        /** Ids of the tokens that end an assistant turn (end-of-turn and end-of-text), those the vocabulary has. */
        int[] stopIds(Tokenizer tok);

        /**
         * Added tokens the format uses as control markers although the tokenizer does not flag them
         * {@code special} (DeepSeek's {@code <｜User｜>} / {@code <｜Assistant｜>}); message content never
         * produces them. Special tokens are always protected.
         */
        default Set<String> protectedTokens() {
            return Set.of();
        }
    }

    private static final Set<String> QWEN3_ROLES = Set.of("system", "user", "assistant", "tool");
    /** Python's {@code %b} in the C locale, for the Llama 3.x {@code Today Date:} line. */
    private static final String[] MONTHS = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
            "Nov", "Dec"};
    /** Llama 3.x with the date taken at render time, like transformers' {@code strftime_now("%d %b %Y")}. */
    private static final Format LLAMA3_TODAY = llama3(ChatTemplate::today);
    /** gpt-oss with the date taken at render time, like transformers' {@code strftime_now("%Y-%m-%d")}. */
    private static final Format HARMONY_TODAY = harmony(() -> LocalDate.now().toString());

    private ChatTemplate() {
    }

    // ------------------------------------------------------------------ factory

    /** Same as {@link #forModelType(String, Tokenizer)} without a tokenizer to disambiguate. */
    public static Format forModelType(String modelType) {
        return forModelType(modelType, null);
    }

    /**
     * @param modelType config.json {@code model_type}: qwen3, qwen2, llama, mistral, gemma2, gemma3(_text),
     *                  phi3, glm (GLM-Edge), glm4, mamba. Also accepted: phi4, deepseek.
     * @param tok       optional; lets the factory tell apart families that share a model_type: Phi-3 (SentencePiece
     *                  tokenizer, newline-delimited roles) vs Phi-4-mini (byte-level, no newlines), and
     *                  DeepSeek-R1-Distill (a qwen2 or llama model whose vocabulary has {@code <｜User｜>})
     * @throws IllegalArgumentException for a model type without a template here
     */
    public static Format forModelType(String modelType, Tokenizer tok) {
        String t = modelType == null ? "" : modelType.toLowerCase(Locale.ROOT).replace('-', '_');
        boolean deepseekVocab = tok != null && tok.tokenId("<｜User｜>") >= 0 && tok.tokenId("<｜Assistant｜>") >= 0;
        switch (t) {
            case "qwen3", "qwen3_moe", "qwen3_next":
                return QWEN3;
            case "qwen3_5": // Qwen3.5 (GGUF qwen35): the 248k vocabulary and a template of its own
                return QWEN35;
            case "qwen2", "qwen2_5", "qwen2_moe":
                return deepseekVocab ? DEEPSEEK : QWEN2;
            case "smolvlm", "idefics3":
                return SMOLVLM;
            case "qwen2vl", "qwen2_vl", "qwen2_5_vl", "qwen25vl":
                return QWEN2;
            case "qwen3vl", "qwen3_vl":
                return QWEN3VL;
            case "llama":
                if (deepseekVocab) {
                    return DEEPSEEK;
                }
                if (tok != null && tok.tokenId("<end_of_utterance>") >= 0 && tok.tokenId("<fake_token_around_image>") >= 0) {
                    return SMOLVLM; // SmolVLM / Idefics3: a Llama text model with its own turn format
                }
                if (tok != null && !tok.isByteLevel() && tok.tokenId("<|start_header_id|>") < 0) {
                    throw new IllegalArgumentException("model_type llama with a SentencePiece tokenizer (Llama 2) has no"
                            + " chat template here; only Llama 3.x is supported");
                }
                return LLAMA3_TODAY;
            case "deepseek_v2":
                return DEEPSEEK_V2;
            case "deepseek", "deepseek_r1", "deepseek_v3":
                return DEEPSEEK;
            case "mistral", "mixtral":
                return MISTRAL;
            case "gemma", "gemma2", "gemma3", "gemma3_text":
                return GEMMA;
            case "phi3", "phimoe": // Phi-3.5-MoE uses the Phi-3 SentencePiece tokenizer and newline template
                if (tok != null && tok.tokenId("<|im_sep|>") >= 0) {
                    return PHI4_14B;
                }
                return tok != null && tok.isByteLevel() ? PHI4 : PHI3;
            case "phi4", "phi4_mini":
                return PHI4;
            case "glm":
                return GLM_EDGE;
            case "glm4":
                return GLM4;
            case "mamba", "mamba2", "falcon_mamba":
                return MAMBA;
            case "gpt_oss":
                return HARMONY_TODAY;
            default:
                throw new IllegalArgumentException("no chat template for model_type \"" + modelType + "\"");
        }
    }

    /**
     * Llama 3.1/3.2 template with a fixed {@code Today Date:} string such as {@code "26 Jul 2024"} (the
     * template's own fallback) instead of the date of the render; use it for reproducible prompts.
     */
    public static Format llama3(String dateString) {
        return llama3(() -> dateString);
    }

    /** Today as Python's {@code strftime("%d %b %Y")} in the C locale. */
    static String today() {
        LocalDate d = LocalDate.now();
        return String.format("%02d %s %d", d.getDayOfMonth(), MONTHS[d.getMonthValue() - 1], d.getYear());
    }

    private static Format llama3(Supplier<String> date) {
        return new Family("llama3", (m, gen, think, o) -> {
            o.ctl("<|begin_of_text|>");
            int first = 0;
            String system = "";
            if (!m.isEmpty() && "system".equals(m.get(0).role())) {
                system = pyTrim(content(m.get(0)));
                first = 1;
            }
            o.ctl("<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: "
                    + date.get() + "\n\n");
            o.txt(system);
            o.ctl("<|eot_id|>");
            for (int i = first; i < m.size(); i++) {
                String role = role(m.get(i), "system", "user", "assistant");
                o.ctl("<|start_header_id|>" + role + "<|end_header_id|>\n\n");
                o.txt(pyTrim(content(m.get(i))));
                o.ctl("<|eot_id|>");
            }
            if (gen) {
                o.ctl("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
        }, "<|eot_id|>", "<|end_of_text|>", "<|eom_id|>");
    }

    /**
     * gpt-oss's harmony format with a fixed {@code Current date:} such as {@code "2025-08-05"} instead of the date of the
     * render; use it for reproducible prompts.
     */
    public static Format harmony(String dateString) {
        return harmony(() -> dateString);
    }

    /**
     * gpt-oss's harmony format as far as plain chat goes (no tools, no tool calls; the Jinja template of the model is the
     * reference): a system message with the model identity, the knowledge cutoff, today's date, the reasoning effort and
     * the valid channels; the first {@code system} / {@code developer} message as a developer message with "# Instructions";
     * {@code <|start|>user<|message|>...<|end|>} turns; earlier assistant turns as their {@code final} channel only (the
     * analysis channel is dropped, as the template does); the generation prompt {@code <|start|>assistant}. A last assistant
     * message without a generation prompt ends with {@code <|return|>} instead of {@code <|end|>}.
     *
     * <p>{@code enableThinking} chooses the reasoning effort: true is the template's default {@code medium} and the model
     * writes its analysis channel first; false is {@code low} and the prompt already opens the {@code final} channel
     * ({@code <|channel|>final<|message|>} after the assistant header), so the answer comes directly and the output has no
     * analysis to separate. The model ends its answer with {@code <|return|>}; {@code <|end|>} ends the analysis message,
     * after which it goes on, so only {@code <|return|>}, {@code <|call|>} and {@code <|endoftext|>} stop generation.
     */
    private static Format harmony(Supplier<String> date) {
        return new Family("harmony", (m, gen, think, o) -> {
            o.ctl("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n"
                    + "Current date: " + date.get() + "\n\nReasoning: " + (think ? "medium" : "low")
                    + "\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>");
            int first = 0;
            if (!m.isEmpty() && ("developer".equals(m.get(0).role()) || "system".equals(m.get(0).role()))) {
                String dev = content(m.get(0));
                first = 1;
                if (!dev.isEmpty()) {
                    o.ctl("<|start|>developer<|message|># Instructions\n\n");
                    o.txt(dev);
                    o.ctl("\n\n<|end|>");
                }
            }
            for (int i = first; i < m.size(); i++) {
                String role = m.get(i).role();
                if ("assistant".equals(role)) {
                    o.ctl("<|start|>assistant<|channel|>final<|message|>");
                    o.txt(content(m.get(i)));
                    o.ctl(i == m.size() - 1 && !gen ? "<|return|>" : "<|end|>");
                } else if ("user".equals(role)) {
                    o.ctl("<|start|>user<|message|>");
                    o.txt(content(m.get(i)));
                    o.ctl("<|end|>");
                } else if ("tool".equals(role)) {
                    throw new IllegalArgumentException("tool messages are not supported by the gpt-oss chat format here");
                } // (like the template, any other role is skipped)
            }
            if (gen) {
                o.ctl(think ? "<|start|>assistant" : "<|start|>assistant<|channel|>final<|message|>");
            }
        }, "<|return|>", "<|call|>", "<|endoftext|>");
    }

    // ------------------------------------------------------------------ the families

    /**
     * Qwen3 ChatML with thinking. Assistant turns before the last user query drop their reasoning; those after it
     * (a conversation that ends with the assistant) keep it as {@code <think>...</think>} the way the official
     * template re-renders them ({@code last_query_index}). Tool results become {@code <tool_response>} blocks in
     * a user turn.
     */
    private static final Format QWEN3 = new Family("chatml-qwen3", (m, gen, think, o) -> {
        for (Message msg : m) {
            if (msg.role() == null || !QWEN3_ROLES.contains(msg.role())) {
                throw new IllegalArgumentException("invalid role: " + msg.role());
            }
        }
        // the last user message that is not a tool response is the last query
        int lastQuery = m.size() - 1;
        for (int i = m.size() - 1; i >= 0; i--) {
            String c = content(m.get(i));
            if (m.get(i).role().equals("user")
                    && !(c.startsWith("<tool_response>") && c.endsWith("</tool_response>"))) {
                lastQuery = i;
                break;
            }
        }
        for (int i = 0; i < m.size(); i++) {
            Message msg = m.get(i);
            String role = msg.role();
            String content = content(msg);
            boolean last = i == m.size() - 1;
            if (role.equals("assistant")) {
                String reasoning = "";
                int end = content.indexOf("</think>");
                if (end >= 0) { // content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
                    String before = rstripNl(content.substring(0, end));
                    int open = before.lastIndexOf("<think>");
                    reasoning = lstripNl(open >= 0 ? before.substring(open + "<think>".length()) : before);
                    content = lstripNl(content.substring(content.lastIndexOf("</think>") + "</think>".length()));
                }
                o.ctl("<|im_start|>assistant\n");
                if (i > lastQuery && (last || !reasoning.isEmpty())) {
                    o.ctl("<think>\n");
                    o.txt(stripNl(reasoning));
                    o.ctl("\n</think>\n\n");
                }
                o.txt(content);
                o.ctl("<|im_end|>\n");
            } else if (role.equals("tool")) {
                toolResponse(m, i, o);
            } else {
                o.ctl("<|im_start|>" + role + "\n");
                o.txt(content);
                o.ctl("<|im_end|>\n");
            }
        }
        if (gen) {
            o.ctl("<|im_start|>assistant\n");
            if (!think) {
                o.ctl("<think>\n\n</think>\n\n");
            }
        }
    }, "<|im_end|>", "<|endoftext|>");

    /** The line the Qwen3.5 template puts into the system turn when thinking is on and no effort is given (xhigh). */
    private static final String QWEN35_EFFORT = "Reasoning effort is set to xhigh. Please think carefully through the task,"
            + " validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and"
            + " clarity in the final answer.";

    /**
     * Qwen3.5 ChatML (the {@code chat_template} of the GGUF files, without vision and tool calls): leading
     * {@code system} / {@code developer} messages are merged (trimmed, joined by a newline) into one system turn, which
     * starts with the reasoning-effort line while thinking is on; every message's content is trimmed; every assistant
     * turn carries its {@code <think>} block (the template's {@code preserve_thinking} defaults to true; the reasoning
     * is what precedes {@code </think>} in the content, empty if none); the generation prompt opens the think block
     * ({@code <think>\n}), or closes an empty one when thinking is off. A system message after the first turn is an error.
     */
    private static final Format QWEN35 = new Family("chatml-qwen35", (m, gen, think, o) -> {
        int numSys = 0;
        StringBuilder sys = new StringBuilder();
        for (int i = 0; i < m.size(); i++) {
            String r = m.get(i).role();
            if (numSys == i && ("system".equals(r) || "developer".equals(r))) {
                String c = pyTrim(content(m.get(i)));
                if (!c.isEmpty()) {
                    sys.append(sys.isEmpty() ? "" : "\n").append(c);
                }
                numSys++;
            }
        }
        if (!sys.isEmpty() || think) {
            o.ctl("<|im_start|>system\n");
            if (think) {
                o.ctl(QWEN35_EFFORT + (sys.isEmpty() ? "" : "\n\n"));
            }
            o.txt(sys.toString());
            o.ctl("<|im_end|>\n");
        }
        for (int i = numSys; i < m.size(); i++) {
            Message msg = m.get(i);
            String role = msg.role();
            if ("system".equals(role) || "developer".equals(role)) {
                throw new IllegalArgumentException("System message must be at the beginning.");
            }
            String content = content(msg);
            if (role.equals("assistant")) {
                String reasoning = "";
                int end = content.indexOf("</think>");
                if (end >= 0) {
                    String before = rstripNl(content.substring(0, end));
                    int open = before.lastIndexOf("<think>");
                    reasoning = lstripNl(open >= 0 ? before.substring(open + "<think>".length()) : before);
                    content = lstripNl(content.substring(content.lastIndexOf("</think>") + "</think>".length()));
                }
                o.ctl("<|im_start|>assistant\n<think>\n");
                o.txt(pyTrim(reasoning));
                o.ctl("\n</think>\n\n");
                o.txt(pyTrim(content));
                o.ctl("<|im_end|>\n");
            } else if (role.equals("tool")) {
                toolResponse(m, i, o);
            } else if (role.equals("user")) {
                o.ctl("<|im_start|>user\n");
                o.txt(pyTrim(content));
                o.ctl("<|im_end|>\n");
            } else {
                throw new IllegalArgumentException("Unexpected message role.");
            }
        }
        if (gen) {
            o.ctl("<|im_start|>assistant\n" + (think ? "<think>\n" : "<think>\n\n</think>\n\n"));
        }
    }, "<|im_end|>", "<|endoftext|>");

    /**
     * Qwen3-VL (Instruct) ChatML: no default system prompt, no thinking blocks (the model's own template has none);
     * pictures are placeholders in the user turn's text.
     */
    private static final Format QWEN3VL = new Family("chatml-qwen3vl", (m, gen, think, o) -> {
        for (int i = 0; i < m.size(); i++) {
            Message msg = m.get(i);
            String role = role(msg, "system", "user", "assistant", "tool");
            if (role.equals("tool")) {
                toolResponse(m, i, o);
                continue;
            }
            o.ctl("<|im_start|>" + role + "\n");
            o.txt(content(msg));
            o.ctl("<|im_end|>\n");
        }
        if (gen) {
            o.ctl("<|im_start|>assistant\n");
        }
    }, "<|im_end|>", "<|endoftext|>");

    /** Qwen2 / Qwen2.5 ChatML: a default system prompt when none is given, no thinking blocks. */
    private static final Format QWEN2 = new Family("chatml-qwen2", (m, gen, think, o) -> {
        if (m.isEmpty() || !"system".equals(m.get(0).role())) {
            o.ctl("<|im_start|>system\n");
            o.txt("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.");
            o.ctl("<|im_end|>\n");
        }
        for (int i = 0; i < m.size(); i++) {
            Message msg = m.get(i);
            String role = role(msg, "system", "user", "assistant", "tool");
            if (role.equals("tool")) {
                toolResponse(m, i, o);
                continue;
            }
            o.ctl("<|im_start|>" + role + "\n");
            o.txt(content(msg));
            o.ctl("<|im_end|>\n");
        }
        if (gen) {
            o.ctl("<|im_start|>assistant\n");
        }
    }, "<|im_end|>", "<|endoftext|>");

    /** Qwen's tool results: consecutive tool messages share one user turn, each in a {@code <tool_response>} block. */
    private static void toolResponse(List<Message> m, int i, Out o) {
        if (i == 0 || !"tool".equals(m.get(i - 1).role())) {
            o.ctl("<|im_start|>user");
        }
        o.ctl("\n<tool_response>\n");
        o.txt(content(m.get(i)));
        o.ctl("\n</tool_response>");
        if (i == m.size() - 1 || !"tool".equals(m.get(i + 1).role())) {
            o.ctl("<|im_end|>\n");
        }
    }

    /** DeepSeek-R1-Distill: the system prompt follows BOS directly; the answer starts inside an open think block. */
    private static final Format DEEPSEEK = new Family("deepseek-r1", (m, gen, think, o) -> {
        o.ctl("<｜begin▁of▁sentence｜>");
        String system = "";
        for (Message msg : m) {
            if ("system".equals(msg.role())) {
                system = content(msg); // the last system message wins
            }
        }
        o.txt(system);
        for (Message msg : m) {
            String role = role(msg, "system", "user", "assistant");
            if (role.equals("user")) {
                o.ctl("<｜User｜>");
                o.txt(content(msg));
            } else if (role.equals("assistant")) {
                String c = content(msg);
                int end = c.lastIndexOf("</think>");
                if (end >= 0) {
                    c = c.substring(end + "</think>".length());
                }
                o.ctl("<｜Assistant｜>");
                o.txt(c);
                o.ctl("<｜end▁of▁sentence｜>");
            }
        }
        if (gen) {
            o.ctl("<｜Assistant｜><think>\n");
        }
    }, Set.of("<｜User｜>", "<｜Assistant｜>"), "<｜end▁of▁sentence｜>");

    /**
     * DeepSeek-V2 / V2-Lite / Coder-V2 chat: {@code <bos>system\n\nUser: ...\n\nAssistant: ...<eos>User: ...\n\nAssistant:}
     * (the tokenizer_config.json template; a system message is emitted in place, followed by a blank line).
     */
    private static final Format DEEPSEEK_V2 = new Family("deepseek-v2", (m, gen, think, o) -> {
        o.ctl("<｜begin▁of▁sentence｜>");
        for (Message msg : m) {
            String role = role(msg, "system", "user", "assistant");
            switch (role) {
                case "system" -> o.txt(content(msg) + "\n\n");
                case "user" -> o.txt("User: " + content(msg) + "\n\n");
                default -> {
                    o.txt("Assistant: " + content(msg));
                    o.ctl("<｜end▁of▁sentence｜>");
                }
            }
        }
        if (gen) {
            o.txt("Assistant:");
        }
    }, "<｜end▁of▁sentence｜>");

    /** Mistral v0.3 instruct: {@code <s>[INST] user[/INST] assistant</s>}; the system prompt joins the last user turn. */
    private static final Format MISTRAL = new Family("mistral", (m, gen, think, o) -> {
        int first = 0;
        String system = null;
        if (!m.isEmpty() && "system".equals(m.get(0).role())) {
            system = content(m.get(0));
            first = 1;
        }
        for (int i = first; i < m.size(); i++) {
            String r = m.get(i).role();
            if (!"user".equals(r) && !"assistant".equals(r)) {
                throw new IllegalArgumentException("Only user and assistant roles are supported, with the exception"
                        + " of an initial optional system message!");
            }
            if (r.equals("user") != ((i - first) % 2 == 0)) {
                throw new IllegalArgumentException("After the optional system message, conversation roles must"
                        + " alternate user/assistant/user/assistant/...");
            }
        }
        o.ctl("<s>");
        for (int i = first; i < m.size(); i++) {
            String role = role(m.get(i), "user", "assistant");
            if (role.equals("user")) {
                o.ctl("[INST]");
                o.txt(" " + (i == m.size() - 1 && system != null ? system + "\n\n" : "") + content(m.get(i)));
                o.ctl("[/INST]");
            } else {
                o.txt(" " + pyTrim(content(m.get(i))));
                o.ctl("</s>");
            }
        }
    }, "</s>");

    /**
     * Gemma 2 / 3 instruct. A system message is prepended to the first turn (Gemma 3's behaviour; the Gemma 2
     * template refuses system messages, here they are accepted the same way).
     */
    private static final Format GEMMA = new Family("gemma", (m, gen, think, o) -> {
        o.ctl("<bos>");
        String prefix = "";
        int first = 0;
        if (!m.isEmpty() && "system".equals(m.get(0).role())) {
            prefix = content(m.get(0)) + "\n\n";
            first = 1;
        }
        for (int i = first; i < m.size(); i++) {
            String role = role(m.get(i), "system", "user", "assistant");
            if (role.equals("user") != ((i - first) % 2 == 0)) {
                throw new IllegalArgumentException("Conversation roles must alternate user/assistant/user/assistant/...");
            }
            o.ctl("<start_of_turn>" + (role.equals("assistant") ? "model" : role) + "\n");
            o.txt((i == first ? prefix : "") + pyTrim(content(m.get(i))));
            o.ctl("<end_of_turn>\n");
        }
        if (gen) {
            o.ctl("<start_of_turn>model\n");
        }
    }, "<end_of_turn>", "<eos>");

    /** Phi-3 (SentencePiece vocabulary): {@code <|user|>\n...<|end|>\n}; ends with EOS when not prompting. */
    private static final Format PHI3 = new Family("phi3", (m, gen, think, o) -> {
        for (Message msg : m) {
            o.ctl("<|" + role(msg, "system", "user", "assistant") + "|>\n");
            o.txt(content(msg));
            o.ctl("<|end|>\n");
        }
        o.ctl(gen ? "<|assistant|>\n" : "<|endoftext|>");
    }, "<|end|>", "<|endoftext|>", "<|assistant|>");

    /** Phi-4-mini (byte-level vocabulary): {@code <|user|>...<|end|>}, no newlines. */
    private static final Format PHI4 = new Family("phi4", (m, gen, think, o) -> {
        for (Message msg : m) {
            o.ctl("<|" + role(msg, "system", "user", "assistant") + "|>");
            o.txt(content(msg));
            o.ctl("<|end|>");
        }
        o.ctl(gen ? "<|assistant|>" : "<|endoftext|>");
    }, "<|end|>", "<|endoftext|>");

    /** phi-4 (14B, ChatML with a separator token): {@code <|im_start|>user<|im_sep|>...<|im_end|>}, no newlines. */
    private static final Format PHI4_14B = new Family("phi4-chatml", (m, gen, think, o) -> {
        for (Message msg : m) {
            o.ctl("<|im_start|>" + role(msg, "system", "user", "assistant") + "<|im_sep|>");
            o.txt(content(msg));
            o.ctl("<|im_end|>");
        }
        if (gen) {
            o.ctl("<|im_start|>assistant<|im_sep|>");
        }
    }, "<|im_end|>", "<|endoftext|>");

    /** GLM-Edge: {@code <|user|>\n...<|assistant|>\n...}; no BOS and no end-of-turn marker. */
    private static final Format GLM_EDGE = new Family("glm-edge", (m, gen, think, o) -> {
        for (Message msg : m) {
            o.ctl("<|" + role(msg, "system", "user", "assistant") + "|>\n");
            o.txt(content(msg));
        }
        if (gen) {
            o.ctl("<|assistant|>\n");
        }
    }, "<|endoftext|>", "<|user|>", "<|observation|>");

    /** GLM-4-0414: {@code [gMASK]<sop>}, all system messages first, generation prompt without newline. */
    private static final Format GLM4 = new Family("glm4", (m, gen, think, o) -> {
        o.ctl("[gMASK]<sop>");
        for (Message msg : m) {
            if ("system".equals(msg.role())) {
                o.ctl("<|system|>\n");
                o.txt(content(msg));
            }
        }
        for (Message msg : m) {
            String role = role(msg, "system", "user", "assistant", "observation");
            if (!role.equals("system")) {
                o.ctl("<|" + role + "|>\n");
                o.txt(content(msg));
            }
        }
        if (gen) {
            o.ctl("<|assistant|>");
        }
    }, "<|endoftext|>", "<|user|>", "<|observation|>");

    /**
     * SmolVLM / Idefics3 (a Llama text model): {@code <|im_start|>} (its BOS, not ChatML), then {@code User: text
     * <end_of_utterance>} / {@code Assistant: text<end_of_utterance>} lines; a system message is its text and a blank
     * line. This is llama.cpp's built-in {@code smolvlm} format (the Jinja template of the model differs in leaving
     * out the space after {@code User:} when the message starts with an image).
     */
    private static final Format SMOLVLM = new Family("smolvlm", (m, gen, think, o) -> {
        o.ctl("<|im_start|>");
        for (Message msg : m) {
            String role = role(msg, "system", "user", "assistant");
            switch (role) {
                case "system" -> {
                    o.txt(content(msg));
                    o.ctl("\n\n");
                }
                case "user" -> {
                    o.ctl("User: ");
                    o.txt(content(msg));
                    o.ctl("<end_of_utterance>\n");
                }
                default -> {
                    o.ctl("Assistant: ");
                    o.txt(content(msg));
                    o.ctl("<end_of_utterance>\n");
                }
            }
        }
        if (gen) {
            o.ctl("Assistant:");
        }
    }, "<end_of_utterance>");

    /** Mamba / GPT-NeoX base models have no chat format: the message contents as plain text, one per line. */
    private static final Format MAMBA = new Family("plain", (m, gen, think, o) -> {
        for (int i = 0; i < m.size(); i++) {
            if (i > 0) {
                o.txt("\n");
            }
            o.txt(content(m.get(i)));
        }
    }, "<|endoftext|>");

    // ------------------------------------------------------------------ machinery

    /** Where a template writes: control strings (may contain special tokens) and untrusted text. */
    static final class Out {
        private final List<String> texts = new ArrayList<>();
        private final List<Boolean> control = new ArrayList<>();

        void ctl(String s) {
            texts.add(s);
            control.add(true);
        }

        void txt(String s) {
            // a picture in the message (U+FFFC) is a control piece of its own, so the content around it stays text
            int i = s.indexOf(Tokenizer.IMAGE_MARK_TEXT);
            while (i >= 0) {
                if (i > 0) {
                    texts.add(s.substring(0, i));
                    control.add(false);
                }
                texts.add(Tokenizer.IMAGE_MARK_TEXT);
                control.add(true);
                s = s.substring(i + 1);
                i = s.indexOf(Tokenizer.IMAGE_MARK_TEXT);
            }
            if (!s.isEmpty() || texts.isEmpty() || !texts.get(texts.size() - 1).equals(Tokenizer.IMAGE_MARK_TEXT)) {
                texts.add(s);
                control.add(false);
            }
        }
    }

    @FunctionalInterface
    private interface Script {
        void build(List<Message> messages, boolean addGenerationPrompt, boolean enableThinking, Out out);
    }

    private record Family(String name, Script script, Set<String> protectedTokens, String... stopTokens)
            implements Format {

        Family(String name, Script script, String... stopTokens) {
            this(name, script, Set.of(), stopTokens);
        }

        @Override
        public String render(List<Message> messages, boolean addGenerationPrompt, boolean enableThinking) {
            Out o = new Out();
            script.build(messages, addGenerationPrompt, enableThinking, o);
            return String.join("", o.texts);
        }

        @Override
        public int[] renderTokens(Tokenizer tok, List<Message> messages, boolean addGenerationPrompt,
                                  boolean enableThinking) {
            Out o = new Out();
            script.build(messages, addGenerationPrompt, enableThinking, o);
            Tokenizer.SequenceEncoder enc = tok.newSequence(protectedTokens);
            for (int i = 0; i < o.texts.size(); i++) {
                if (o.control.get(i)) {
                    enc.special(o.texts.get(i));
                } else {
                    enc.text(o.texts.get(i));
                }
            }
            return enc.toArray();
        }

        @Override
        public int[] renderTokens(Tokenizer tok, List<Message> messages, boolean addGenerationPrompt,
                                  boolean enableThinking, PromptCache cache) {
            if (cache == null || cache.tokenizer() != tok) {
                return renderTokens(tok, messages, addGenerationPrompt, enableThinking);
            }
            Out o = new Out();
            script.build(messages, addGenerationPrompt, enableThinking, o);
            return cache.encode(name, o.texts, o.control, protectedTokens);
        }

        @Override
        public int[] stopIds(Tokenizer tok) {
            Set<Integer> ids = new LinkedHashSet<>();
            for (String s : stopTokens) {
                int id = tok.tokenId(s);
                if (id >= 0) {
                    ids.add(id);
                }
            }
            if (ids.isEmpty() && tok.eosId() >= 0) {
                ids.add(tok.eosId());
            }
            return ids.stream().mapToInt(Integer::intValue).toArray();
        }
    }

    private static String content(Message m) {
        return m.content() == null ? "" : m.content();
    }

    private static String role(Message m, String... allowed) {
        String r = m.role();
        for (String a : allowed) {
            if (a.equals(r)) {
                return r;
            }
        }
        throw new IllegalArgumentException("invalid role: " + r);
    }

    /** Python {@code str.lstrip('\n')}. */
    static String lstripNl(String s) {
        int a = 0;
        while (a < s.length() && s.charAt(a) == '\n') {
            a++;
        }
        return s.substring(a);
    }

    /** Python {@code str.rstrip('\n')}. */
    static String rstripNl(String s) {
        int b = s.length();
        while (b > 0 && s.charAt(b - 1) == '\n') {
            b--;
        }
        return s.substring(0, b);
    }

    /** Python {@code str.strip('\n')}. */
    static String stripNl(String s) {
        return lstripNl(rstripNl(s));
    }

    /** Jinja's {@code trim} filter, i.e. Python {@code str.strip()}. */
    static String pyTrim(String s) {
        int a = 0;
        int b = s.length();
        while (a < b && isPyWhitespace(s.charAt(a))) {
            a++;
        }
        while (b > a && isPyWhitespace(s.charAt(b - 1))) {
            b--;
        }
        return s.substring(a, b);
    }

    private static boolean isPyWhitespace(char c) {
        return c <= ' ' ? (c >= 9 && c <= 13) || (c >= 0x1C && c <= ' ')
                : c == 0x85 || Character.isSpaceChar(c);
    }
}
