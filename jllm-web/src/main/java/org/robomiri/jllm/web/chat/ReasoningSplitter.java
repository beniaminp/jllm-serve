package org.robomiri.jllm.web.chat;

import org.robomiri.jllm.engine.Engine;

/**
 * Separates a reasoning model's {@code <think>...</think>} block from its answer, as the DeepSeek API does with
 * {@code reasoning_content}: clients such as opencode, Open WebUI and Continue then show the thinking apart and do
 * not send it back as part of the answer. Works on streamed pieces, holding back text that may be the start of a
 * tag. Only a block at the very start of the answer counts (DeepSeek-R1's prompt already opens it).
 */
public final class ReasoningSplitter implements Engine.TextSink {

    private static final String OPEN = "<think>";
    private static final String CLOSE = "</think>";

    private final Engine.TextSink reasoning;
    private final Engine.TextSink content;
    private final StringBuilder pending = new StringBuilder();
    private int state; // 0 = before anything, 1 = inside the block, 2 = answer
    private boolean answerStarted;

    /** @param startsInThink the prompt ends inside an open think block (DeepSeek-R1, Qwen3 with a forced block) */
    public ReasoningSplitter(Engine.TextSink reasoning, Engine.TextSink content, boolean startsInThink) {
        this.reasoning = reasoning;
        this.content = content;
        this.state = startsInThink ? 1 : 0;
    }

    @Override
    public void accept(String text) {
        pending.append(text);
        drain(false);
    }

    /** Releases what is still held back. */
    public void finish() {
        drain(true);
    }

    private void drain(boolean end) {
        while (!pending.isEmpty()) {
            if (state == 0) {
                int lead = 0;
                while (lead < pending.length() && Character.isWhitespace(pending.charAt(lead))) {
                    lead++;
                }
                String rest = pending.substring(lead);
                if (rest.startsWith(OPEN)) {
                    pending.delete(0, lead + OPEN.length());
                    state = 1;
                } else if (!end && OPEN.startsWith(rest)) {
                    return; // maybe "<thi" so far
                } else {
                    state = 2;
                }
            } else if (state == 1) {
                int close = pending.indexOf(CLOSE);
                if (close >= 0) {
                    emit(reasoning, pending.substring(0, close));
                    pending.delete(0, close + CLOSE.length());
                    state = 2;
                    continue;
                }
                int keep = end ? 0 : partial(CLOSE);
                emit(reasoning, pending.substring(0, pending.length() - keep));
                pending.delete(0, pending.length() - keep);
                return;
            } else {
                String s = pending.toString();
                pending.setLength(0);
                if (!answerStarted) { // the blank lines between the block and the answer
                    s = s.stripLeading();
                    answerStarted = !s.isEmpty();
                }
                emit(content, s);
                return;
            }
        }
    }

    /** Length of the longest suffix of the pending text that is a prefix of {@code tag}. */
    private int partial(String tag) {
        for (int k = Math.min(tag.length() - 1, pending.length()); k > 0; k--) {
            if (tag.startsWith(pending.substring(pending.length() - k))) {
                return k;
            }
        }
        return 0;
    }

    private static void emit(Engine.TextSink sink, String s) {
        if (!s.isEmpty()) {
            sink.accept(s);
        }
    }

    /** {@code [reasoning, answer]} of a complete text; reasoning is empty when there is no block. */
    public static String[] split(String text, boolean startsInThink) {
        StringBuilder r = new StringBuilder();
        StringBuilder c = new StringBuilder();
        ReasoningSplitter sp = new ReasoningSplitter(r::append, c::append, startsInThink);
        sp.accept(text);
        sp.finish();
        return new String[]{r.toString().strip(), c.toString()};
    }

    /** Whether the rendered prompt leaves the model inside an open think block. */
    public static boolean promptOpensThink(Engine engine, int[] prompt) {
        int n = Math.min(6, prompt.length);
        String tail = engine.tokenizer().decode(java.util.Arrays.copyOfRange(prompt, prompt.length - n, prompt.length));
        String t = tail.stripTrailing();
        return t.endsWith(OPEN);
    }
}
