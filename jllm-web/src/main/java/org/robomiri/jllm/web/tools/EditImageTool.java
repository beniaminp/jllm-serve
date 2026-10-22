package org.robomiri.jllm.web.tools;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.web.images.ImageService;

import java.util.HashMap;
import java.util.Map;

/**
 * {@code edit_image}: changes a picture the chat made earlier -- redraws it with a new prompt (image to image), or only a
 * rectangle of it (inpainting).
 */
public final class EditImageTool implements Tool {

    private final ImageService images;

    EditImageTool(ImageService images) {
        this.images = images;
    }

    @Override
    public String name() {
        return "edit_image";
    }

    @Override
    public String description() {
        return "Change a picture that generate_image (or an earlier edit_image) made: give its link as `image` and describe how it "
                + "should look now. Without `region` the whole picture is redrawn, `strength` telling how much (0.3 keeps most of it, "
                + "0.8 changes a lot). With `region` (a rectangle in fractions of the width and height, x and y of its top-left corner) "
                + "only that part is repainted and the rest stays as it is. The result is shown to the user in the chat.";
    }

    @Override
    public Map<String, Object> parameters() {
        Map<String, Object> schema = ImageToolSupport.schema(
                "image", "string", "link of the picture to change, as returned by generate_image: /v1/images/files/<id>.png", true,
                "prompt", "string", "how the picture (or the region) should look now", true,
                "strength", "number", "0..1, how much of the picture is redrawn (optional, default 0.6; 1 for a region)", false,
                "negative_prompt", "string", "what it should not show (optional)", false,
                "steps", "integer", "denoising steps (optional)", false,
                "seed", "integer", "random seed (optional)", false);
        @SuppressWarnings("unchecked")
        Map<String, Object> props = (Map<String, Object>) schema.get("properties");
        props.put("region", Json.obj("type", "object", "description",
                "optional rectangle to repaint, fractions of the picture: x, y (top-left corner), width, height, each 0..1",
                "properties", Json.obj("x", Json.obj("type", "number"), "y", Json.obj("type", "number"),
                        "width", Json.obj("type", "number"), "height", Json.obj("type", "number"))));
        return schema;
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
        byte[] png = ImageToolSupport.stored(images, Json.str(args, "image", ""));
        Map<String, Object> body = new HashMap<>();
        body.put("prompt", prompt);
        ImageToolSupport.copyOptions(args, body);
        body.put("n", 1);
        body.remove("size"); // an edit keeps the picture's size
        body.put("size", "auto");
        byte[] mask = null;
        Map<String, Object> region = Json.map(args, "region");
        if (region != null) {
            mask = ImageToolSupport.regionMask(png, region);
        }
        double strength = Json.dbl(args, "strength", mask != null ? 1.0 : 0.6);
        body.put("strength", Math.max(0.05, Math.min(1.0, strength)));
        return ImageToolSupport.run(images, body, png, mask, true);
    }
}
