package org.robomiri.jllm.web.tools;

import org.robomiri.jllm.diffusion.ImageApi;
import org.robomiri.jllm.diffusion.Imaging;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.web.images.ImageService;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * What {@link GenerateImageTool} and {@link EditImageTool} share: running a request on a loaded image model, keeping the
 * pictures in the {@link ImageService} store, and telling the model (in text) where they are. The chat page shows a
 * picture inline wherever a tool result contains {@code ![...](/v1/images/files/<id>.png)}.
 */
final class ImageToolSupport {

    static final Pattern FILE = Pattern.compile("([0-9a-f]{24})");

    private ImageToolSupport() {
    }

    /** The two image tools, for registration. */
    static List<Tool> all(ImageService images) {
        return List.of(new GenerateImageTool(images), new EditImageTool(images));
    }

    static Map<String, Object> schema(Object... props) {
        Map<String, Object> properties = new LinkedHashMap<>();
        List<Object> required = new ArrayList<>();
        for (int i = 0; i < props.length; i += 4) {
            properties.put((String) props[i], Json.obj("type", props[i + 1], "description", props[i + 2]));
            if ((Boolean) props[i + 3]) {
                required.add(props[i]);
            }
        }
        return Json.obj("type", "object", "properties", properties, "required", required);
    }

    /** Runs the request and returns the text for the model. */
    @SuppressWarnings("unchecked")
    static String run(ImageService images, Map<String, Object> body, byte[] image, byte[] mask, boolean edit) throws Exception {
        ImageService.Loaded model;
        try {
            model = images.resolve(Json.str(body, "model", null));
        } catch (NoSuchElementException e) {
            throw new IllegalStateException("no image model is loaded; ask the user to load one on the Models page (for example stabilityai/sd-turbo)");
        }
        body.put("response_format", "url");
        try (ImageService.Lease lease = model.lease()) {
            Map<String, Object> out = ImageApi.run(model.pipeline(), model.id(), body, image, mask, edit, images.store(), "", null);
            List<Object> data = (List<Object>) out.get("data");
            model.countImages(data.size());
            StringBuilder sb = new StringBuilder();
            Map<String, Object> timing = Json.map(out, "timing");
            sb.append(data.size() == 1 ? "Generated 1 image" : "Generated " + data.size() + " images").append(" with ").append(model.id());
            Object first = data.get(0);
            sb.append(" (").append(Json.num(first, "width", 0)).append('x').append(Json.num(first, "height", 0)).append(", seed ")
                    .append(Json.num(first, "seed", 0)).append(", ").append(String.format("%.1f s", Json.dbl(timing, "seconds", 0))).append("):\n");
            int i = 1;
            for (Object d : data) {
                sb.append("![generated image ").append(i++).append("](").append(Json.str(d, "url", "")).append(")\n");
            }
            sb.append("The picture is shown to the user in the chat. Pass its link as `image` to edit_image to change it.");
            return sb.toString();
        } catch (ImageApi.ApiException e) {
            throw new IllegalArgumentException(e.getMessage());
        } catch (IllegalStateException e) {
            throw new IllegalStateException(e.getMessage());
        }
    }

    /** The PNG of a picture from this chat: its link ({@code /v1/images/files/<id>.png}) or just the id. */
    static byte[] stored(ImageService images, String ref) {
        Matcher m = FILE.matcher(ref == null ? "" : ref);
        if (!m.find()) {
            throw new IllegalArgumentException("image must be the link of a picture generated earlier (/v1/images/files/<id>.png)");
        }
        byte[] png = images.store().get(m.group(1));
        if (png == null) {
            throw new IllegalArgumentException("that picture is no longer available (the server keeps the latest ones): generate it again");
        }
        return png;
    }

    /** A black picture of the image's size with a white rectangle: the region (fractions of the width / height) to repaint. */
    static byte[] regionMask(byte[] png, Map<String, Object> region) throws IOException {
        Imaging.Image img = Imaging.decode(png);
        int w = img.width();
        int h = img.height();
        double x = clamp(Json.dbl(region, "x", 0));
        double y = clamp(Json.dbl(region, "y", 0));
        double rw = clamp(Json.dbl(region, "width", 1));
        double rh = clamp(Json.dbl(region, "height", 1));
        int x0 = (int) Math.round(x * w);
        int y0 = (int) Math.round(y * h);
        int x1 = (int) Math.round(Math.min(1.0, x + rw) * w);
        int y1 = (int) Math.round(Math.min(1.0, y + rh) * h);
        byte[] rgb = new byte[w * h * 3];
        for (int yy = y0; yy < y1; yy++) {
            for (int xx = x0; xx < x1; xx++) {
                int o = (yy * w + xx) * 3;
                rgb[o] = rgb[o + 1] = rgb[o + 2] = (byte) 255;
            }
        }
        return Imaging.png(new Imaging.Image(w, h, rgb));
    }

    private static double clamp(double v) {
        return Math.max(0, Math.min(1, v));
    }

    /** Copies the optional generation settings of a tool call into the request. */
    static void copyOptions(Map<String, Object> args, Map<String, Object> body) {
        for (String k : new String[]{"negative_prompt", "steps", "guidance", "seed", "size", "scheduler", "n"}) {
            if (args.get(k) != null) {
                body.put(k, args.get(k));
            }
        }
    }
}
