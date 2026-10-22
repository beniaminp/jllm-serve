package org.robomiri.jllm.web.tools;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.MediaPrompt;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.vision.ImageFetcher;
import org.robomiri.jllm.vision.ImageParts;
import org.robomiri.jllm.vision.MediaChat;
import org.robomiri.jllm.vision.Rgb;
import org.robomiri.jllm.web.chat.ImageStore;
import org.robomiri.jllm.web.models.LoadedModel;
import org.robomiri.jllm.web.models.ModelManager;
import org.robomiri.jllm.web.settings.SettingsService;

import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * {@code describe_image}: lets a model that cannot see look at a picture. The tool hands the picture and a question to
 * a loaded vision-language model and returns its answer as text (docs/20). The picture is a URL (fetched under the
 * Tools tab's fetch rules), a {@code data:} URL, or the id of a picture uploaded to {@code POST /api/images}.
 */
public final class DescribeImageTool implements Tool {

    private static final String DEFAULT_QUESTION = "Describe this image in detail.";

    private final ModelManager models;
    private final SettingsService settings;
    private final ImageStore store;
    private final Supplier<ToolsConfig.General> config;

    public DescribeImageTool(ModelManager models, SettingsService settings, ImageStore store, Supplier<ToolsConfig.General> config) {
        this.models = models;
        this.settings = settings;
        this.store = store;
        this.config = config;
    }

    @Override
    public String name() {
        return "describe_image";
    }

    @Override
    public String description() {
        return "Look at an image with a vision model and answer a question about it (read text in it, describe it, count"
                + " things). The image is an http(s) URL, a data: URL or the id of an uploaded image. Needs a"
                + " vision-language model to be loaded.";
    }

    @Override
    public Map<String, Object> parameters() {
        return Json.obj("type", "object",
                "properties", Json.obj(
                        "image", Json.obj("type", "string", "description", "the image: an http(s) URL, a data: URL or an uploaded image id (img-...)"),
                        "question", Json.obj("type", "string", "description", "what to find out about the image (default: describe it)"),
                        "model", Json.obj("type", "string", "description", "id of the loaded vision model to ask (default: any that can see)")),
                "required", List.of("image"));
    }

    @Override
    public String source() {
        return "builtin";
    }

    @Override
    public String call(Map<String, Object> args) throws Exception {
        String ref = Json.str(args, "image", "").strip();
        if (ref.isEmpty()) {
            throw new IllegalArgumentException("the argument \"image\" is required (a URL or an uploaded image id)");
        }
        String question = Json.str(args, "question", DEFAULT_QUESTION).strip();
        if (question.isEmpty()) {
            question = DEFAULT_QUESTION;
        }
        Settings s = settings.current();
        LoadedModel model = pickModel(Json.str(args, "model", null));
        Rgb img;
        if (ref.startsWith("img-")) {
            img = store.get(ref);
            if (img == null) {
                throw new IllegalArgumentException("no uploaded image " + ref + " (unknown or evicted: upload it again)");
            }
        } else {
            img = ImageParts.load(ref, new ImageFetcher(config.get().fetch().allowPrivateAddresses(), s.getLong("vision.max-image-bytes")),
                    s.getLong("vision.max-image-bytes"), s.getLong("vision.max-image-pixels"));
        }
        try (LoadedModel.Lease lease = model.lease()) {
            Engine engine = model.engine();
            MediaPrompt mp = MediaChat.render(engine,
                    List.of(new ChatTemplate.Message("user", MediaChat.MARK + MediaChat.stripMarks(question))), false, List.of(img));
            SamplingParams p = new SamplingParams(0.2f, 20, 0.9f, 0f, 1.05f, 0f, 0f, -1L,
                    (int) Math.max(1, Math.min(512, engine.backend().contextLength() - mp.length() - 1)), List.of());
            try (LoadedModel.Slot slot = model.pick(mp.identity())) {
                Engine.Result r = slot.engine().generate(mp, p, null, model::closing);
                model.record(slot, r.stats());
                String text = r.text().strip();
                return text.isEmpty() ? "(the vision model returned no text)" : text;
            }
        }
    }

    private LoadedModel pickModel(String id) {
        if (id != null && !id.isBlank()) {
            LoadedModel m = models.loaded().stream().filter(x -> x.id().equals(id.strip())).findFirst()
                    .orElseThrow(() -> new IllegalArgumentException("model '" + id + "' is not loaded"));
            if (m.engine().vision() == null) {
                throw new IllegalArgumentException("model '" + id + "' cannot see images: it was loaded without a vision projector");
            }
            return m;
        }
        for (LoadedModel m : models.loaded()) {
            if (m.engine().vision() != null) {
                return m;
            }
        }
        throw new IllegalArgumentException("no vision model is loaded: load a vision-language GGUF model together with its"
                + " mmproj file (Models tab) and try again");
    }
}
