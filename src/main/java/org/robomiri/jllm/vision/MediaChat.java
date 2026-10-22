package org.robomiri.jllm.vision;

import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.MediaPrompt;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.util.List;

/**
 * Chat prompts with pictures. A message's content is text in which every picture stands for itself as
 * {@link Tokenizer#IMAGE_MARK_TEXT} (U+FFFC), in reading order; the chat template keeps those marks apart from the text
 * and the vision encoder of the engine expands each into the model's picture tokens.
 */
public final class MediaChat {

    private MediaChat() {
    }

    /** What stands for one picture in a message's content. */
    public static final String MARK = Tokenizer.IMAGE_MARK_TEXT;

    /** Text typed by a user or a client can contain U+FFFC itself; it must not become a picture. */
    public static String stripMarks(String text) {
        return text == null ? "" : text.replace(MARK, "");
    }

    /**
     * Renders {@code messages} with the engine's chat template and expands the pictures.
     *
     * @param images the pictures in the order their marks appear in the messages
     * @throws IllegalArgumentException when the model has no vision projector or the marks and pictures do not match
     */
    public static MediaPrompt render(Engine engine, List<ChatTemplate.Message> messages, boolean thinking, List<Rgb> images) {
        VisionContext v = engine.vision();
        if (v == null) {
            throw new IllegalArgumentException("this model cannot see images: no vision projector (mmproj) is loaded with it");
        }
        int[] ids = engine.preparePrompt(engine.renderChat(messages, thinking));
        return v.build(ids, images);
    }
}
