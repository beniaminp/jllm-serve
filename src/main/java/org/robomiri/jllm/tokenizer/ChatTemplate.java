package org.robomiri.jllm.tokenizer;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Chat templates of the supported model families, each derived from the Jinja {@code chat_template} in the
 * model's tokenizer_config.json (minus tool calling) and checked against the official Jinja render.
 *
 * <p>Get one with {@link #forModelType(String)} / {@link #forModelType(String, Tokenizer)}. The static
 * {@link #renderTokens} / {@link #render} methods are the original Qwen3 ChatML entry points.
 *
 * <p>Control tokens come from the template; message content is encoded with special tokens disabled, so text
 * like {@code <|im_end|>} inside a message cannot forge a turn boundary. The rendered prompt already contains the
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

        /** Ids of the tokens that end an assistant turn (end-of-turn and end-of-text), those the vocabulary has. */
        int[] stopIds(Tokenizer tok);
    }

    private static final Set<String> QWEN3_ROLES = Set.of("system", "user", "assistant", "tool");
    /** Fallback of the Llama 3.x template when the caller passes no date (transformers would use today's). */
    private static final String LLAMA_DEFAULT_DATE = "26 Jul 2024";

    private ChatTemplate() {
    }

    // ------------------------------------------------------------------ original Qwen3 entry points

    /**
     * Renders the Qwen3 template directly to token ids.
     *
     * @throws IllegalArgumentException for a role outside system/user/assistant/tool
     */
    public static int[] renderTokens(Tokenizer tok, List<Message> messages, boolean addGenerationPrompt,
                                     boolean enableThinking) {
        return QWEN3.renderTokens(tok, messages, addGenerationPrompt, enableThinking);
    }

    /**
     * @param enableThinking when false, an empty think block is pre-filled so the model answers directly
     */
    public static String render(List<Message> messages, boolean addGenerationPrompt, boolean enableThinking) {
        return QWEN3.render(messages, addGenerationPrompt, enableThinking);
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
            case "qwen3", "qwen3_moe":
                return QWEN3;
            case "qwen2", "qwen2_5", "qwen2_moe":
                return deepseekVocab ? DEEPSEEK : QWEN2;
            case "llama":
                if (deepseekVocab) {
                    return DEEPSEEK;
                }
                if (tok != null && !tok.isByteLevel() && tok.tokenId("<|start_header_id|>") < 0) {
                    throw new IllegalArgumentException("model_type llama with a SentencePiece tokenizer (Llama 2) has no"
                            + " chat template here; only Llama 3.x is supported");
                }
                return llama3(LLAMA_DEFAULT_DATE);
            case "deepseek", "deepseek_r1", "deepseek_v3":
                return DEEPSEEK;
            case "mistral", "mixtral":
                return MISTRAL;
            case "gemma", "gemma2", "gemma3", "gemma3_text":
                return GEMMA;
            case "phi3":
                if (tok != null && tok.tokenId("<|im_sep|>") >= 0) {
                    return PHI4_14B;
                }
                return tok != null && tok.isByteLevel() ? PHI4 : PHI3;
            case "phi4", "phi4_mini", "phimoe":
                return PHI4;
            case "glm":
                return GLM_EDGE;
            case "glm4":
                return GLM4;
            case "mamba", "mamba2", "falcon_mamba":
                return MAMBA;
            default:
                throw new IllegalArgumentException("no chat template for model_type \"" + modelType + "\"");
        }
    }

    /** Llama 3.1/3.2 template with an explicit {@code Today Date:} string such as {@code "26 Jul 2024"}. */
    public static Format llama3(String dateString) {
        return new Family("llama3", (m, gen, think, o) -> {
            o.ctl("<|begin_of_text|>");
            int first = 0;
            String system = "";
            if (!m.isEmpty() && "system".equals(m.get(0).role())) {
                system = pyTrim(content(m.get(0)));
                first = 1;
            }
            o.ctl("<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: "
                    + dateString + "\n\n");
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

    // ------------------------------------------------------------------ the families

    private static final Format QWEN3 = new Family("chatml-qwen3", (m, gen, think, o) -> {
        for (Message msg : m) {
            String role = msg.role();
            if (role == null || !QWEN3_ROLES.contains(role)) {
                throw new IllegalArgumentException("invalid role: " + role);
            }
            String content = content(msg);
            if (role.equals("assistant")) { // previous turns drop their reasoning, like the official template
                int end = content.lastIndexOf("</think>");
                if (end >= 0) {
                    content = content.substring(end + "</think>".length()).stripLeading();
                }
            }
            o.ctl("<|im_start|>" + role + "\n");
            o.txt(content);
            o.ctl("<|im_end|>\n");
        }
        if (gen) {
            o.ctl("<|im_start|>assistant\n");
            if (!think) {
                o.ctl("<think>\n\n</think>\n\n");
            }
        }
    }, "<|im_end|>", "<|endoftext|>");

    /** Qwen2 / Qwen2.5 ChatML: a default system prompt when none is given, no thinking blocks. */
    private static final Format QWEN2 = new Family("chatml-qwen2", (m, gen, think, o) -> {
        if (m.isEmpty() || !"system".equals(m.get(0).role())) {
            o.ctl("<|im_start|>system\n");
            o.txt("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.");
            o.ctl("<|im_end|>\n");
        }
        for (Message msg : m) {
            o.ctl("<|im_start|>" + role(msg, "system", "user", "assistant") + "\n");
            o.txt(content(msg));
            o.ctl("<|im_end|>\n");
        }
        if (gen) {
            o.ctl("<|im_start|>assistant\n");
        }
    }, "<|im_end|>", "<|endoftext|>");

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
    }, "<｜end▁of▁sentence｜>");

    /** Mistral v0.3 instruct: {@code <s>[INST] user[/INST] assistant</s>}; the system prompt joins the last user turn. */
    private static final Format MISTRAL = new Family("mistral", (m, gen, think, o) -> {
        int first = 0;
        String system = null;
        if (!m.isEmpty() && "system".equals(m.get(0).role())) {
            system = content(m.get(0));
            first = 1;
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
            texts.add(s);
            control.add(false);
        }
    }

    @FunctionalInterface
    private interface Script {
        void build(List<Message> messages, boolean addGenerationPrompt, boolean enableThinking, Out out);
    }

    private record Family(String name, Script script, String... stopTokens) implements Format {

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
            Tokenizer.SequenceEncoder enc = tok.newSequence();
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
