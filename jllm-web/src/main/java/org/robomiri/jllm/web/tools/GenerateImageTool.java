package org.robomiri.jllm.web.tools;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.web.images.ImageService;

import java.util.HashMap;
import java.util.Map;

/** {@code generate_image}: a picture from a text prompt with the loaded image model; the chat shows it inline. */
public final class GenerateImageTool implements Tool {

    private final ImageService images;

    GenerateImageTool(ImageService images) {
        this.images = images;
    }

    @Override
    public String name() {
        return "generate_image";
    }

    @Override
    public String description() {
        return "Draw a picture from a text description with the image model that is loaded (Stable Diffusion). The picture is "
                + "shown to the user in the chat. Write the prompt in English as a short list of what the picture shows, its style and "
                + "lighting, e.g. 'a red fox in a snowy forest, golden hour, detailed fur, photo'. Returns the link of the picture.";
    }

    @Override
    public Map<String, Object> parameters() {
        return ImageToolSupport.schema(
                "prompt", "string", "what the picture should show (English works best)", true,
                "negative_prompt", "string", "what it should not show, e.g. 'blurry, low quality' (optional)", false,
                "size", "string", "width x height in pixels, e.g. 512x512 (optional; a multiple of 64)", false,
                "steps", "integer", "denoising steps (optional; the model's default is used when left out)", false,
                "seed", "integer", "random seed for a reproducible picture (optional)", false);
    }

    @Override
    public String source() {
        return "builtin";
    }

    @Override
    public String call(Map<String, Object> args) throws Exception {
        String prompt = Json.str(args, "prompt", "").strip();
        if (prompt.isEmpty()) {
            throw new IllegalArgumentException("prompt is required");
        }
        Map<String, Object> body = new HashMap<>();
        body.put("prompt", prompt);
        body.put("n", 1);
        ImageToolSupport.copyOptions(args, body);
        body.put("n", 1);
        return ImageToolSupport.run(images, body, null, null, false);
    }
}
