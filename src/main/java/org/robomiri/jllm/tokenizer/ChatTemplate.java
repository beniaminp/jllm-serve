package org.robomiri.jllm.tokenizer;

import java.util.List;

/**
 * The Qwen3 ChatML template (the Jinja template shipped in tokenizer_config.json, minus tool calling).
 *
 * <pre>
 * &lt;|im_start|&gt;system\n{system}&lt;|im_end|&gt;\n
 * &lt;|im_start|&gt;user\n{user}&lt;|im_end|&gt;\n
 * &lt;|im_start|&gt;assistant\n
 * </pre>
 */
public final class ChatTemplate {

    public record Message(String role, String content) {
    }

    private ChatTemplate() {
    }

    /**
     * @param enableThinking when false, an empty think block is pre-filled so the model answers directly
     */
    public static String render(List<Message> messages, boolean addGenerationPrompt, boolean enableThinking) {
        StringBuilder sb = new StringBuilder();
        for (Message m : messages) {
            String content = m.content() == null ? "" : m.content();
            if (m.role().equals("assistant")) {
                // like the official template: previous turns drop their reasoning
                int end = content.lastIndexOf("</think>");
                if (end >= 0) {
                    content = content.substring(end + "</think>".length()).stripLeading();
                }
            }
            sb.append("<|im_start|>").append(m.role()).append('\n').append(content).append("<|im_end|>\n");
        }
        if (addGenerationPrompt) {
            sb.append("<|im_start|>assistant\n");
            if (!enableThinking) {
                sb.append("<think>\n\n</think>\n\n");
            }
        }
        return sb.toString();
    }
}
