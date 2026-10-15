package org.robomiri.jllm.tokenizer;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

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

    private static final Set<String> ROLES = Set.of("system", "user", "assistant", "tool");

    private ChatTemplate() {
    }

    /**
     * Renders directly to token ids. Control tokens come from the template; message content is encoded with
     * special tokens disabled, so text like {@code <|im_end|>} inside a message cannot forge a turn boundary.
     * Encoding piecewise is exact: added tokens are always split points for the tokenizer anyway.
     *
     * @throws IllegalArgumentException for a role outside system/user/assistant/tool
     */
    public static int[] renderTokens(Tokenizer tok, List<Message> messages, boolean addGenerationPrompt,
                                     boolean enableThinking) {
        int[] out = new int[0];
        for (Message m : messages) {
            String role = m.role();
            if (role == null || !ROLES.contains(role)) {
                throw new IllegalArgumentException("invalid role: " + role);
            }
            String content = m.content() == null ? "" : m.content();
            if (role.equals("assistant")) {
                int end = content.lastIndexOf("</think>");
                if (end >= 0) {
                    content = content.substring(end + "</think>".length()).stripLeading();
                }
            }
            out = concat(out, tok.encode("<|im_start|>" + role + "\n", true));
            out = concat(out, tok.encode(content, false));
            out = concat(out, tok.encode("<|im_end|>\n", true));
        }
        if (addGenerationPrompt) {
            out = concat(out, tok.encode("<|im_start|>assistant\n", true));
            if (!enableThinking) {
                out = concat(out, tok.encode("<think>\n\n</think>\n\n", true));
            }
        }
        return out;
    }

    private static int[] concat(int[] a, int[] b) {
        int[] r = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, r, a.length, b.length);
        return r;
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
