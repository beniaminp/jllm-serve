package org.robomiri.jllm.web.chat;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.MediaPrompt;
import org.robomiri.jllm.vision.ImageFetcher;
import org.robomiri.jllm.vision.ImageParts;
import org.robomiri.jllm.vision.Rgb;
import org.robomiri.jllm.web.chat.ChatSupport.RequestError;

import java.util.List;
import java.util.Map;

/**
 * Pictures in the web endpoints' chat requests (docs/20): the OpenAI {@code image_url} content parts are decoded (a
 * {@code data:} URL, or an {@code http(s)} URL fetched under the Tools tab's fetch rules), checked against the loaded
 * model, and the prompt is built with the picture tokens in it.
 */
public final class WebImages {

    private WebImages() {
    }

    /**
     * Replaces the picture parts of {@code messages} by marks and returns the decoded pictures.
     *
     * @param allowPrivate whether URLs on private networks may be fetched ({@code fetch.allowPrivateAddresses} of the tools)
     * @throws RequestError a picture that cannot be read or fetched, or a model without a vision projector
     */
    public static ImageParts.Extracted extract(List<Object> messages, Engine engine, Settings s, boolean allowPrivate) {
        if (engine.vision() == null) {
            throw RequestError.invalid("this model cannot see images: it was loaded without a vision projector (mmproj)."
                    + " Load a vision-language GGUF model together with its mmproj file (Models tab: projector)");
        }
        try {
            return ImageParts.extract(messages, new ImageFetcher(allowPrivate, s.getLong("vision.max-image-bytes")),
                    s.getLong("vision.max-image-bytes"), s.getLong("vision.max-image-pixels"));
        } catch (IllegalArgumentException e) {
            throw RequestError.invalid(e.getMessage());
        }
    }

    /** The request body with its messages replaced (the other fields are shared). */
    public static Map<String, Object> withMessages(Map<String, Object> body, List<Object> messages) {
        Map<String, Object> copy = new java.util.LinkedHashMap<>(body);
        copy.put("messages", messages);
        return copy;
    }

    /**
     * The prompt for the engine: {@code ids} (rendered by the chat template, with one mark per picture) expanded with the
     * pictures' tokens, after checking that the context has room for it and for at least one answer token.
     */
    public static MediaPrompt prompt(Engine engine, int[] ids, List<Rgb> images) {
        MediaPrompt mp;
        try {
            mp = engine.vision().build(engine.preparePrompt(ids), images);
        } catch (IllegalArgumentException e) {
            throw RequestError.invalid(e.getMessage());
        }
        int ctx = engine.backend().contextLength();
        if (mp.length() >= ctx) {
            throw RequestError.contextExceeded("prompt has " + mp.length() + " tokens (with the pictures), context is " + ctx
                    + " (at least one token must be left for the completion); send smaller pictures, lower"
                    + " vision.image-max-tokens or load the model with a larger context");
        }
        return mp;
    }
}
