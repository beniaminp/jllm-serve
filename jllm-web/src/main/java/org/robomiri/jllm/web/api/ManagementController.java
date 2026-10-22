package org.robomiri.jllm.web.api;

import org.robomiri.jllm.web.WebProperties;
import org.robomiri.jllm.web.logs.LogBuffer;
import org.robomiri.jllm.web.models.Job;
import org.robomiri.jllm.web.models.LoadedModel;
import org.robomiri.jllm.web.models.ModelManager;
import org.robomiri.jllm.web.settings.SettingsService;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.util.List;
import java.util.Map;

/** The web console's management API: models, jobs, settings, logs and status. */
@RestController
@RequestMapping("/api")
public class ManagementController {

    public record Status(long uptimeMs, String java, int processors, long heapUsed, long heapMax, long weightBytes,
                         int loadedModels, int runningJobs, boolean apiKeyRequired, String settingsFile,
                         int thermalState) {
    }

    public record SettingsUpdate(Map<String, String> changes, List<String> reset) {
    }

    public record Logs(List<LogBuffer.Line> lines, String pending, long last) {
    }

    private final ModelManager models;
    private final SettingsService settings;
    private final WebProperties props;
    private final org.robomiri.jllm.web.models.KvStore kvStore;
    private final org.robomiri.jllm.web.images.ImageService images;

    @org.springframework.beans.factory.annotation.Autowired(required = false)
    private org.robomiri.jllm.web.video.VideoService videos;

    public ManagementController(ModelManager models, SettingsService settings, WebProperties props,
                                org.robomiri.jllm.web.models.KvStore kvStore, org.robomiri.jllm.web.images.ImageService images) {
        this.models = models;
        this.settings = settings;
        this.props = props;
        this.kvStore = kvStore;
        this.images = images;
        LogBuffer.get().setCapacity(props.logLines());
    }

    @GetMapping("/status")
    public Status status() {
        Runtime rt = Runtime.getRuntime();
        long weights = 0;
        int thermal = -1; // the hottest any loaded model's backend reports (-1: none can tell, e.g. CPU only)
        for (LoadedModel m : models.loaded()) {
            weights += m.view().weightBytes();
            thermal = Math.max(thermal, m.engine().backend().thermalState());
        }
        for (var m : images.loaded()) {
            weights += m.view().weightBytes();
        }
        int videoModels = 0;
        if (videos != null) {
            for (var m : videos.loaded()) {
                weights += m.view().weightBytes();
                videoModels++;
            }
        }
        int running = (int) models.jobs().stream().filter(j -> j.state() == Job.State.RUNNING).count();
        return new Status(ManagementFactory.getRuntimeMXBean().getUptime(), Runtime.version().toString(),
                rt.availableProcessors(), rt.totalMemory() - rt.freeMemory(), rt.maxMemory(), weights,
                models.loaded().size() + images.loaded().size() + videoModels, running, props.apiKeyRequired(), settings.file().toString(), thermal);
    }

    // ------------------------------------------------------------------ models

    @GetMapping("/models/loaded")
    public List<Object> loaded() { // text models ("kind": "text") and image models ("kind": "image")
        List<Object> out = new java.util.ArrayList<>(models.loaded().stream().map(LoadedModel::view).toList());
        images.loaded().forEach(m -> out.add(m.view()));
        if (videos != null) {
            videos.loaded().forEach(m -> out.add(m.view())); // "kind": "video"
        }
        return out;
    }

    @PostMapping("/models/load")
    public Job.View load(@RequestBody ModelManager.LoadRequest request) {
        return models.load(request);
    }

    @DeleteMapping("/models/loaded")
    public Map<String, Object> unload(@RequestParam String id) throws InterruptedException {
        models.unload(id);
        return Map.of("unloaded", id);
    }

    @PostMapping("/models/download")
    public Job.View download(@RequestBody ModelManager.DownloadRequest request) {
        return models.download(request);
    }

    /** Conversation caches saved to disk. */
    @GetMapping("/kv-cache")
    public org.robomiri.jllm.web.models.KvStore.View kvCache() {
        return kvStore.view();
    }

    /** Saves every slot of every loaded model now (normally done after long prefills and on unload). */
    @PostMapping("/kv-cache/save")
    public org.robomiri.jllm.web.models.KvStore.View saveKvCache() {
        for (var m : models.loaded()) {
            kvStore.saveAll(m);
        }
        return kvStore.view();
    }

    @DeleteMapping("/kv-cache")
    public Map<String, Object> clearKvCache() throws IOException {
        return Map.of("deleted", kvStore.clear());
    }

    @GetMapping("/models/local")
    public List<ModelManager.LocalModel> local() throws IOException {
        return models.localModels();
    }

    @DeleteMapping("/models/local")
    public Map<String, Object> deleteLocal(@RequestParam String path) throws IOException {
        models.deleteLocal(path);
        return Map.of("deleted", path);
    }

    @GetMapping("/hub/inspect")
    public ModelManager.RepoInfo inspect(@RequestParam String source, @RequestParam(required = false) String revision)
            throws IOException, InterruptedException {
        return models.inspect(source, revision);
    }

    // ------------------------------------------------------------------ jobs

    @GetMapping("/jobs")
    public List<Job.View> jobs() {
        return models.jobs();
    }

    @PostMapping("/jobs/{id}/cancel")
    public Map<String, Object> cancel(@PathVariable String id) {
        models.cancel(id);
        return Map.of("cancelled", id);
    }

    @DeleteMapping("/jobs")
    public Map<String, Object> clearJobs() {
        models.clearFinishedJobs();
        return Map.of("cleared", true);
    }

    // ------------------------------------------------------------------ settings and logs

    @GetMapping("/settings")
    public List<SettingsService.View> settings() {
        return settings.views();
    }

    @PutMapping("/settings")
    public List<SettingsService.View> updateSettings(@RequestBody SettingsUpdate update) {
        settings.update(update.changes(), update.reset());
        return settings.views();
    }

    @GetMapping("/logs")
    public Logs logs(@RequestParam(defaultValue = "0") long after) {
        LogBuffer b = LogBuffer.get();
        return new Logs(b.after(after), b.pending(), b.lastSeq());
    }
}
