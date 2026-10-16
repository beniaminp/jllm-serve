package org.robomiri.jllm.hub;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Collections;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * Downloads model repositories from the Hugging Face Hub using only {@link HttpClient}.
 *
 * <p>Supports resumable downloads (HTTP Range), SHA-256 verification of LFS files, {@code HF_TOKEN}
 * authentication and the {@code HF_ENDPOINT} mirror override.
 *
 * <p>Every request has a response timeout and every download body has a stall watchdog (no bytes for the
 * timeout counts as a connection failure), so a hung connection is retried instead of blocking forever.
 * Client errors (4xx) are not retried; server errors (5xx), 429 Too Many Requests (honouring
 * {@code Retry-After}) and I/O failures are retried with back-off, for listings and downloads alike.
 */
public final class HuggingFaceHub {

    private static final int MAX_REDIRECTS = 10;
    private static final int MAX_ATTEMPTS = 5;
    /** Default response / stall timeout. */
    public static final long DEFAULT_TIMEOUT_MILLIS = 60_000;

    private final HttpClient http;
    private final String endpoint;
    private final String token;
    /** False when a token is configured but the endpoint is plain http to a non-loopback host, so it is never sent. */
    private final boolean tokenUsable;
    private volatile long timeoutMillis;

    public HuggingFaceHub() {
        this(null, null);
    }

    /**
     * @param endpoint Hub base URL, or null for {@code $HF_ENDPOINT} / https://huggingface.co
     * @param token    access token, or null for {@code $HF_TOKEN} / {@code ~/.cache/huggingface/token}
     */
    public HuggingFaceHub(String endpoint, String token) {
        this(endpoint, token, DEFAULT_TIMEOUT_MILLIS);
    }

    /**
     * @param endpoint      Hub base URL, or null for {@code $HF_ENDPOINT} / https://huggingface.co
     * @param token         access token, or null for {@code $HF_TOKEN} / {@code ~/.cache/huggingface/token}
     * @param timeoutMillis response timeout of each request and maximum time without progress on a download
     */
    public HuggingFaceHub(String endpoint, String token, long timeoutMillis) {
        this.http = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NEVER) // handled manually so the token never leaks to CDNs
                .connectTimeout(Duration.ofSeconds(30))
                .build();
        String ep = endpoint != null ? endpoint : System.getenv("HF_ENDPOINT");
        this.endpoint = ep == null || ep.isBlank() ? "https://huggingface.co" : ep.replaceAll("/+$", "");
        this.token = token != null && !token.isBlank() ? token.trim() : findToken();
        URI epUri = URI.create(this.endpoint);
        this.tokenUsable = "https".equalsIgnoreCase(epUri.getScheme()) || isLoopback(epUri.getHost());
        setTimeoutMillis(timeoutMillis);
    }

    /** {@code localhost} or a loopback IP literal: a local mirror, where the token may travel over plain http. */
    public static boolean isLoopback(String host) {
        if (host == null) {
            return false;
        }
        String h = host.toLowerCase();
        if (h.startsWith("[") && h.endsWith("]")) {
            h = h.substring(1, h.length() - 1);
        }
        return h.equals("localhost") || h.startsWith("127.") || h.equals("::1") || h.equals("0:0:0:0:0:0:0:1");
    }

    /** The hint appended to a 401 / 403: what to do about the token. */
    private String authHint(int status) {
        if (status != 401 && status != 403) {
            return "";
        }
        if (token == null) {
            return " (gated/private repo? set HF_TOKEN)";
        }
        return tokenUsable ? " (is HF_TOKEN valid and allowed to read this repo?)"
                : " (token withheld: endpoint " + endpoint + " is not https)";
    }

    /** Sets the response timeout of each request and the maximum time without progress on a download body. */
    public void setTimeoutMillis(long timeoutMillis) {
        if (timeoutMillis <= 0) {
            throw new IllegalArgumentException("timeout must be positive: " + timeoutMillis);
        }
        this.timeoutMillis = timeoutMillis;
    }

    public long timeoutMillis() {
        return timeoutMillis;
    }

    private static String findToken() {
        for (String name : new String[]{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}) {
            String v = System.getenv(name);
            if (v != null && !v.isBlank()) {
                return v.trim();
            }
        }
        Path p = Path.of(System.getProperty("user.home"), ".cache", "huggingface", "token");
        try {
            if (Files.isRegularFile(p)) {
                String v = Files.readString(p).trim();
                return v.isEmpty() ? null : v;
            }
        } catch (IOException ignored) {
            // no token
        }
        return null;
    }

    /** A file in a repository as reported by the Hub API. */
    public record RepoFile(String name, long size, String sha256) {
    }

    /** An HTTP response with an unexpected status; 4xx statuses other than 429 are not retried. */
    public static final class HttpStatusException extends IOException {
        private final int status;
        private final long retryAfterMillis;

        HttpStatusException(int status, String message) {
            this(status, message, 0);
        }

        HttpStatusException(int status, String message, long retryAfterMillis) {
            super(message);
            this.status = status;
            this.retryAfterMillis = retryAfterMillis;
        }

        public int status() {
            return status;
        }

        /** The server's {@code Retry-After} in milliseconds (429 / 503), 0 when it gave none. */
        public long retryAfterMillis() {
            return retryAfterMillis;
        }

        /** True for a failure that will not go away by retrying: a 4xx other than 429 Too Many Requests. */
        boolean isPermanent() {
            return status >= 400 && status < 500 && status != 429;
        }
    }

    /** Builds the exception for an unexpected status, carrying the response's {@code Retry-After} if any. */
    private static HttpStatusException statusException(HttpResponse<?> resp, String message) {
        long retryAfter = 0;
        String header = resp.headers().firstValue("retry-after").orElse(null);
        if (header != null) {
            try {
                retryAfter = Math.min(Long.parseLong(header.trim()), 300) * 1000L; // seconds; an HTTP date is ignored
            } catch (NumberFormatException ignored) {
                // HTTP-date form: fall back to the usual back-off
            }
        }
        return new HttpStatusException(resp.statusCode(), message, retryAfter);
    }

    /** True if {@code e} is worth another attempt: I/O trouble, a 5xx or a 429. */
    private static boolean retryable(IOException e) {
        return !(e instanceof HttpStatusException h) || !h.isPermanent();
    }

    /** Milliseconds to wait before attempt {@code attempt + 1}: the server's Retry-After, else linear back-off. */
    private static long backoffMillis(IOException e, int attempt) {
        if (e instanceof HttpStatusException h && h.retryAfterMillis() > 0) {
            return h.retryAfterMillis();
        }
        return 1000L * attempt;
    }

    /** True for the small files every model needs: configs, tokenizer and the shard index. */
    public static boolean isMetadataFile(String name) {
        return name.equals("config.json")
                || name.equals("generation_config.json")
                || name.equals("tokenizer.json")
                || name.equals("tokenizer_config.json")
                || name.equals("model.safetensors.index.json");
    }

    /**
     * The weight shards the checkpoint in {@code dir} consists of: exactly the files named by its
     * {@code model.safetensors.index.json} when there is one, otherwise {@code model.safetensors}. Other
     * {@code .safetensors} files of a repository (consolidated or duplicate checkpoints, adapters) are not part
     * of it and are never downloaded; the converter reads the same set.
     */
    public static List<String> weightFileNames(Path dir) throws IOException {
        Path index = dir.resolve("model.safetensors.index.json");
        if (Files.isRegularFile(index)) {
            return List.copyOf(new TreeSet<>(SafeTensors.readWeightMap(index).values()));
        }
        return List.of("model.safetensors");
    }

    // ------------------------------------------------------------------ GGUF files

    private static final Pattern GGUF_SPLIT = Pattern.compile("^(.*)-(\\d{5})-of-(\\d{5})\\.gguf$");

    /** A usage problem with a GGUF source (no such local file; no file, or more than one, fits the requested name / quantization): the message says what to change. */
    public static final class GgufSelectionException extends IOException {
        public GgufSelectionException(String message) {
            super(message);
        }
    }

    /** True for a {@code .gguf} file that holds a language model (not the {@code mmproj-*} vision projector). */
    public static boolean isGgufModelFile(String name) {
        String base = name.substring(name.lastIndexOf('/') + 1).toLowerCase(java.util.Locale.ROOT);
        return base.endsWith(".gguf") && !base.startsWith("mmproj");
    }

    /** True if a repository has GGUF model files and no safetensors checkpoint (so it can only be loaded as GGUF). */
    public static boolean isGgufRepo(List<RepoFile> listing) {
        boolean gguf = false;
        for (RepoFile f : listing) {
            if (f.name().equals("model.safetensors") || f.name().equals("model.safetensors.index.json")) {
                return false;
            }
            gguf |= isGgufModelFile(f.name());
        }
        return gguf;
    }

    /** The models in {@code names}: a split model counts once, under its first part. Sorted. */
    private static TreeMap<String, List<String>> ggufModels(List<String> names) {
        TreeMap<String, List<String>> models = new TreeMap<>();
        for (String n : names) {
            if (!isGgufModelFile(n)) {
                continue;
            }
            Matcher m = GGUF_SPLIT.matcher(n);
            String key = m.matches() ? m.group(1) + "-00001-of-" + m.group(3) + ".gguf" : n;
            models.computeIfAbsent(key, k -> new ArrayList<>()).add(n);
        }
        models.values().forEach(Collections::sort);
        return models;
    }

    /** True if the file name carries the quantization label {@code quant} ({@code Q4_K_M}) as a whole word. */
    static boolean matchesQuant(String name, String quant) {
        String base = name.substring(name.lastIndexOf('/') + 1);
        return Pattern.compile("(?<![A-Za-z0-9])" + Pattern.quote(quant) + "(?![A-Za-z0-9_])", Pattern.CASE_INSENSITIVE)
                .matcher(base).find();
    }

    /**
     * Picks the GGUF model to load from {@code names} (the repository's files, or the files in a local directory):
     * the one named exactly {@code file} (a split model: any of its parts), or the one whose name carries the
     * label {@code quant}, or the only one there is.
     *
     * @return the name of the first part and the names of all parts, in order
     * @throws IOException with the list of the available files when nothing or more than one model fits
     */
    public static Map.Entry<String, List<String>> selectGguf(List<String> names, String file, String quant) throws IOException {
        TreeMap<String, List<String>> models = ggufModels(names);
        if (models.isEmpty()) {
            throw new GgufSelectionException("no .gguf model file found (searched " + names.size() + " files)");
        }
        List<Map.Entry<String, List<String>>> hits = new ArrayList<>();
        String what;
        if (file != null && !file.isBlank()) {
            what = "model.gguf-file=" + file;
            for (Map.Entry<String, List<String>> e : models.entrySet()) {
                if (e.getValue().contains(file)) {
                    hits.add(e);
                }
            }
        } else if (quant != null && !quant.isBlank()) {
            what = "model.gguf-quant=" + quant;
            for (Map.Entry<String, List<String>> e : models.entrySet()) {
                if (matchesQuant(e.getKey(), quant)) {
                    hits.add(e);
                }
            }
        } else {
            what = "no model.gguf-file / model.gguf-quant";
            hits.addAll(models.entrySet());
        }
        if (hits.size() == 1) {
            Map.Entry<String, List<String>> e = hits.getFirst();
            return Map.entry(e.getKey(), e.getValue());
        }
        StringBuilder sb = new StringBuilder();
        sb.append(hits.isEmpty() ? "no GGUF file matches " : "more than one GGUF file matches ").append(what)
                .append("; available:");
        for (Map.Entry<String, List<String>> e : (hits.isEmpty() ? models.entrySet() : hits)) {
            sb.append("\n  ").append(e.getKey());
            if (e.getValue().size() > 1) {
                sb.append(" (").append(e.getValue().size()).append(" parts)");
            }
        }
        sb.append("\nset model.gguf-file to an exact file name (--gguf-file) or model.gguf-quant to a quantization label"
                + " such as Q4_K_M (--gguf-quant)");
        throw new GgufSelectionException(sb.toString());
    }

    /** The GGUF model files under {@code dir} (two levels deep), as names relative to it. */
    public static List<String> localGgufNames(Path dir) throws IOException {
        List<String> out = new ArrayList<>();
        if (!Files.isDirectory(dir)) {
            return out;
        }
        try (var s = Files.walk(dir, 2)) {
            s.filter(Files::isRegularFile).forEach(p -> out.add(dir.relativize(p).toString().replace('\\', '/')));
        }
        out.removeIf(n -> !isGgufModelFile(n) || n.endsWith(".incomplete"));
        return out;
    }

    /**
     * Downloads the GGUF model chosen by {@code file} / {@code quant} (see {@link #selectGguf}) into {@code dir}:
     * only that file and its split siblings, each resumable and SHA-256 verified like every other download.
     *
     * @return the path of the first (or only) file
     */
    public Path downloadGguf(String repo, String revision, Path dir, String file, String quant)
            throws IOException, InterruptedException {
        Map<String, RepoFile> byName = new HashMap<>();
        List<String> names = new ArrayList<>();
        for (RepoFile f : listFiles(repo, revision)) {
            byName.put(f.name(), f);
            names.add(f.name());
        }
        Map.Entry<String, List<String>> sel;
        try {
            sel = selectGguf(names, file, quant);
        } catch (GgufSelectionException e) {
            throw new GgufSelectionException(repo + ": " + e.getMessage());
        }
        Files.createDirectories(dir);
        long need = 0;
        for (String n : sel.getValue()) {
            Path t = dir.resolve(n);
            RepoFile f = byName.get(n);
            if (!(Files.isRegularFile(t) && Files.size(t) == f.size())) {
                need += Math.max(0, f.size());
            }
        }
        long free = Files.getFileStore(dir).getUsableSpace();
        if (need > free) {
            throw new IOException(String.format("need %.1f GB for %s but only %.1f GB are free", need / 1e9, sel.getKey(), free / 1e9));
        }
        System.err.printf("Downloading %s@%s: %s (%d file%s, %.1f MB) into %s%n", repo, revision, sel.getKey(),
                sel.getValue().size(), sel.getValue().size() == 1 ? "" : "s", need / 1e6, dir);
        for (String n : sel.getValue()) {
            downloadFile(repo, revision, byName.get(n), dir);
        }
        return dir.resolve(sel.getKey());
    }

    private final Map<String, List<RepoFile>> listings = new HashMap<>();
    private final Map<String, String> commits = new HashMap<>();

    /**
     * The commit {@code revision} of {@code repo} resolves to (the listing's {@code sha}), or null if the Hub
     * did not report one.
     */
    public String commitSha(String repo, String revision) throws IOException, InterruptedException {
        listFiles(repo, revision);
        return commits.get(repo + "@" + revision);
    }

    /** Lists the files of {@code repo} at {@code revision} (cached per repo/revision). */
    public List<RepoFile> listFiles(String repo, String revision) throws IOException, InterruptedException {
        String key = repo + "@" + revision;
        List<RepoFile> cached = listings.get(key);
        if (cached != null) {
            return cached;
        }
        URI uri = URI.create(endpoint + "/api/models/" + repo + "/revision/" + encodeSegment(revision) + "?blobs=true");
        HttpResponse<String> resp;
        for (int attempt = 1; ; attempt++) {
            try {
                resp = send(uri, HttpResponse.BodyHandlers.ofString(), null, false);
                if (resp.statusCode() != 200) {
                    throw statusException(resp, "Hub API " + uri + " returned HTTP " + resp.statusCode()
                            + authHint(resp.statusCode()) + ": " + truncate(resp.body()));
                }
                break;
            } catch (IOException e) {
                if (attempt >= MAX_ATTEMPTS || !retryable(e)) {
                    throw e;
                }
                System.err.printf("  listing %s: %s, retrying (%d/%d)%n", repo, e.getMessage(), attempt, MAX_ATTEMPTS);
                Thread.sleep(backoffMillis(e, attempt));
            }
        }
        Map<String, Object> info;
        try {
            info = Json.parseObject(resp.body());
        } catch (RuntimeException e) {
            throw new IOException("Hub API " + uri + " returned invalid JSON: " + e.getMessage(), e);
        }
        List<RepoFile> files = new ArrayList<>();
        List<Object> siblings = Json.list(info, "siblings");
        if (siblings == null) {
            throw new IOException("Hub API response has no 'siblings'");
        }
        for (Object o : siblings) {
            String name = Json.str(o, "rfilename", null);
            if (name == null) {
                continue;
            }
            Map<String, Object> lfs = Json.map(o, "lfs");
            long size = lfs != null ? Json.num(lfs, "size", -1) : Json.num(o, "size", -1);
            String sha = lfs != null ? Json.str(lfs, "sha256", null) : null;
            files.add(new RepoFile(name, size, sha));
        }
        listings.put(key, files);
        String commit = Json.str(info, "sha", null);
        if (commit != null && !commit.isBlank()) {
            commits.put(key, commit);
        }
        return files;
    }

    /**
     * Downloads the inference files of {@code repo} into {@code dir}, skipping files that are already complete:
     * the metadata files first, then exactly the weight shards {@link #weightFileNames} derives from them. The
     * free-space check covers that same set of shards.
     *
     * @return {@code dir}
     */
    public Path download(String repo, String revision, Path dir) throws IOException, InterruptedException {
        downloadMetadata(repo, revision, dir);
        Map<String, RepoFile> byName = new HashMap<>();
        for (RepoFile f : listFiles(repo, revision)) {
            byName.put(f.name(), f);
        }
        List<RepoFile> weights = new ArrayList<>();
        long total = 0;
        for (String name : weightFileNames(dir)) {
            RepoFile f = byName.get(name);
            if (f == null) {
                throw new IOException(repo + " has no " + name + (Files.isRegularFile(dir.resolve("model.safetensors.index.json"))
                        ? " although model.safetensors.index.json lists it" : " (and no model.safetensors.index.json): "
                        + "only safetensors checkpoints are supported"));
            }
            weights.add(f);
            Path t = dir.resolve(name);
            if (!(Files.isRegularFile(t) && Files.size(t) == f.size())) {
                total += Math.max(0, f.size());
            }
        }
        long free = Files.getFileStore(dir).getUsableSpace();
        if (total > free) {
            throw new IOException(String.format("need %.1f GB for %s but only %.1f GB are free; "
                    + "use 'run/serve ... --low-disk' to download, quantize and delete shards one at a time",
                    total / 1e9, repo, free / 1e9));
        }
        System.err.printf("Downloading %s@%s (%d shards, %.1f MB) into %s%n", repo, revision, weights.size(), total / 1e6, dir);
        for (RepoFile f : weights) {
            downloadFile(repo, revision, f, dir);
        }
        return dir;
    }

    /** Downloads the small config/tokenizer/index files only (shards are fetched on demand by {@link #ensure}). */
    public Path downloadMetadata(String repo, String revision, Path dir) throws IOException, InterruptedException {
        Files.createDirectories(dir);
        for (RepoFile f : listFiles(repo, revision)) {
            if (isMetadataFile(f.name())) {
                downloadFile(repo, revision, f, dir);
            }
        }
        return dir;
    }

    /** Makes sure one file of the repo is present in {@code dir} and returns its path. */
    public Path ensure(String repo, String revision, Path dir, String name) throws IOException, InterruptedException {
        RepoFile f = listFiles(repo, revision).stream().filter(x -> x.name().equals(name)).findFirst()
                .orElseThrow(() -> new IOException(name + " is not in " + repo));
        downloadFile(repo, revision, f, dir);
        return dir.resolve(name);
    }

    private void downloadFile(String repo, String revision, RepoFile f, Path dir) throws IOException, InterruptedException {
        Path target = dir.resolve(f.name()).normalize();
        if (!target.startsWith(dir.normalize())) {
            throw new IOException("refusing to write outside model dir: " + f.name());
        }
        if (Files.isRegularFile(target) && (f.size() < 0 || Files.size(target) == f.size())) {
            System.err.printf("  %-32s up to date%n", f.name());
            return;
        }
        Files.createDirectories(target.getParent());
        Path part = target.resolveSibling(target.getFileName() + ".incomplete");
        URI uri = URI.create(endpoint + "/" + repo + "/resolve/" + encodeSegment(revision) + "/" + encodePath(f.name()));

        // A checksum mismatch means the partial file on disk was corrupt (or the file changed on the Hub while
        // we were resuming), so the download is restarted once from byte 0 before giving up.
        for (int pass = 1; ; pass++) {
            MessageDigest sha = sha256();
            long have = Files.isRegularFile(part) ? Files.size(part) : 0;
            if (f.size() >= 0 && have > f.size()) {
                have = 0;
            }
            if (have > 0) {
                hashExisting(part, sha);
            }
            for (int attempt = 1; ; attempt++) {
                try {
                    have = transfer(uri, part, have, f, sha);
                    break;
                } catch (IOException e) {
                    if (attempt >= MAX_ATTEMPTS || !retryable(e)) {
                        throw e;
                    }
                    System.err.printf("%n  %s: %s, retrying (%d/%d)%n", f.name(), e.getMessage(), attempt, MAX_ATTEMPTS);
                    Thread.sleep(backoffMillis(e, attempt));
                    // resume from what is on disk; re-hash it since the stream state is unknown
                    have = Files.isRegularFile(part) ? Files.size(part) : 0;
                    sha = sha256();
                    if (have > 0) {
                        hashExisting(part, sha);
                    }
                }
            }
            if (f.size() >= 0 && have != f.size()) {
                throw new IOException(f.name() + ": size mismatch, got " + have + " expected " + f.size());
            }
            if (f.sha256() != null) {
                String got = HexFormat.of().formatHex(sha.digest());
                if (!got.equalsIgnoreCase(f.sha256())) {
                    Files.deleteIfExists(part);
                    if (pass == 1) {
                        System.err.printf("  %s: sha256 mismatch (got %s), downloading again from the start%n",
                                f.name(), got);
                        continue;
                    }
                    throw new IOException(f.name() + ": sha256 mismatch, got " + got + " expected " + f.sha256());
                }
            }
            Files.move(part, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            return;
        }
    }

    /** Streams {@code uri} into {@code part} starting at byte {@code have}; returns the final size. */
    private long transfer(URI uri, Path part, long have, RepoFile f, MessageDigest sha) throws IOException, InterruptedException {
        String range = have > 0 ? "bytes=" + have + "-" : null;
        HttpResponse<InputStream> resp = send(uri, HttpResponse.BodyHandlers.ofInputStream(), range, true);
        int status = resp.statusCode();
        if (status == 416 && have > 0 && (f.size() < 0 || have == f.size())) {
            // Range Not Satisfiable at the end of the file: the download is complete. With an unknown size
            // (no LFS entry) the 416 itself is the only evidence; the sha256 check does not apply there.
            resp.body().close();
            return have;
        }
        boolean append;
        if (status == 206) {
            append = true;
        } else if (status == 200) {
            append = false;
            if (have > 0) {
                sha.reset();
            }
            have = 0;
        } else {
            String body;
            try (InputStream in = resp.body()) {
                body = new String(in.readNBytes(512), StandardCharsets.UTF_8);
            }
            throw statusException(resp, "GET " + uri + " returned HTTP " + status + authHint(status) + ": " + truncate(body));
        }
        long size = f.size() >= 0 ? f.size() : have + resp.headers().firstValueAsLong("content-length").orElse(-1);
        Progress progress = new Progress(f.name(), size, have);
        long timeout = timeoutMillis;
        try (InputStream in = resp.body();
             OutputStream out = Files.newOutputStream(part, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                     append ? StandardOpenOption.APPEND : StandardOpenOption.TRUNCATE_EXISTING);
             StallWatchdog watchdog = new StallWatchdog(in, timeout)) {
            byte[] buf = new byte[1 << 20];
            int n;
            try {
                while ((n = in.read(buf)) >= 0) {
                    watchdog.progress();
                    out.write(buf, 0, n);
                    sha.update(buf, 0, n);
                    have += n;
                    progress.update(have);
                }
            } catch (IOException e) {
                if (watchdog.stalled()) {
                    throw new IOException("no data received for " + describe(timeout), e);
                }
                throw e;
            }
        }
        progress.done(have);
        return have;
    }

    /**
     * Closes a response body when no byte has arrived for {@code timeoutMillis}, which makes the blocked
     * {@code read} fail with an {@link IOException} and hands control back to the retry loop.
     */
    private static final class StallWatchdog implements AutoCloseable {
        private final InputStream body;
        private final long timeoutMillis;
        private final Thread thread;
        private volatile long lastProgress = System.nanoTime();
        private volatile boolean stalled;
        private volatile boolean closed;

        StallWatchdog(InputStream body, long timeoutMillis) {
            this.body = body;
            this.timeoutMillis = timeoutMillis;
            this.thread = Thread.ofVirtual().unstarted(this::watch);
            thread.start();
        }

        void progress() {
            lastProgress = System.nanoTime();
        }

        boolean stalled() {
            return stalled;
        }

        private void watch() {
            try {
                while (!closed) {
                    long idle = (System.nanoTime() - lastProgress) / 1_000_000;
                    if (idle >= timeoutMillis) {
                        stalled = true;
                        try {
                            body.close();
                        } catch (IOException ignored) {
                            // the reader will see the failure
                        }
                        return;
                    }
                    Thread.sleep(Math.min(1000, timeoutMillis - idle));
                }
            } catch (InterruptedException ignored) {
                // stopped
            }
        }

        @Override
        public void close() {
            closed = true;
            thread.interrupt();
        }
    }

    /**
     * Sends a GET, following redirects manually; the auth header is only sent to the Hub endpoint itself, and
     * only over https (or to a loopback mirror) so the token never travels in clear text.
     *
     * <p>Timeouts: {@link HttpRequest.Builder#timeout} runs until the whole body has been consumed, which would
     * abort any download longer than the timeout, so it is only set on {@code !streaming} (small API) requests.
     * A streaming request is bounded by the timeout until its headers arrive, and its body by the
     * {@link StallWatchdog} in {@link #transfer}.
     */
    private <T> HttpResponse<T> send(URI uri, HttpResponse.BodyHandler<T> handler, String range, boolean streaming)
            throws IOException, InterruptedException {
        URI current = uri;
        String hubHost = URI.create(endpoint).getHost();
        long timeout = timeoutMillis;
        for (int i = 0; i <= MAX_REDIRECTS; i++) {
            HttpRequest.Builder b = HttpRequest.newBuilder(current).GET()
                    .header("User-Agent", "jllm-serve/1.0 (java " + Runtime.version() + ")");
            if (!streaming) {
                b.timeout(Duration.ofMillis(timeout));
            }
            if (token != null && tokenUsable && hubHost != null && hubHost.equalsIgnoreCase(current.getHost())
                    && ("https".equalsIgnoreCase(current.getScheme()) || isLoopback(current.getHost()))) {
                b.header("Authorization", "Bearer " + token);
            }
            if (range != null) {
                b.header("Range", range);
            }
            HttpResponse<T> resp;
            if (streaming) {
                CompletableFuture<HttpResponse<T>> cf = http.sendAsync(b.build(), handler);
                try {
                    resp = cf.get(timeout, TimeUnit.MILLISECONDS);
                } catch (TimeoutException e) {
                    cf.cancel(true);
                    throw new HttpTimeoutException("no response from " + current + " within " + describe(timeout));
                } catch (ExecutionException e) {
                    Throwable c = e.getCause();
                    if (c instanceof IOException io) {
                        throw io;
                    }
                    if (c instanceof RuntimeException re) {
                        throw re;
                    }
                    throw new IOException(c);
                }
            } else {
                resp = http.send(b.build(), handler);
            }
            int s = resp.statusCode();
            if (s == 301 || s == 302 || s == 303 || s == 307 || s == 308) {
                String loc = resp.headers().firstValue("location")
                        .orElseThrow(() -> new IOException("redirect without Location from " + uri));
                closeQuietly(resp);
                current = current.resolve(loc);
                continue;
            }
            return resp;
        }
        throw new IOException("too many redirects for " + uri);
    }

    private static void closeQuietly(HttpResponse<?> resp) {
        if (resp.body() instanceof InputStream in) {
            try {
                in.close();
            } catch (IOException ignored) {
                // nothing to do
            }
        }
    }

    private static void hashExisting(Path p, MessageDigest sha) throws IOException {
        try (InputStream in = Files.newInputStream(p)) {
            byte[] buf = new byte[1 << 20];
            int n;
            while ((n = in.read(buf)) >= 0) {
                sha.update(buf, 0, n);
            }
        }
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Encodes a repository path, keeping its {@code /} separators. */
    private static String encodePath(String path) {
        StringBuilder sb = new StringBuilder();
        for (String seg : path.split("/")) {
            if (!sb.isEmpty()) {
                sb.append('/');
            }
            sb.append(encodeSegment(seg));
        }
        return sb.toString();
    }

    /** Encodes one URL path segment: a revision such as {@code refs/pr/1} becomes {@code refs%2Fpr%2F1}. */
    private static String encodeSegment(String segment) {
        return URLEncoder.encode(segment, StandardCharsets.UTF_8).replace("+", "%20");
    }

    private static String describe(long millis) {
        return millis % 1000 == 0 ? millis / 1000 + " s" : millis + " ms";
    }

    private static String truncate(String s) {
        return s == null ? "" : s.length() > 300 ? s.substring(0, 300) + "..." : s;
    }

    /** Single-line progress bar on stderr. */
    private static final class Progress {
        private final String name;
        private final long total;
        private final long start;
        private final long t0 = System.nanoTime();
        private long lastPrint;

        Progress(String name, long total, long start) {
            this.name = name;
            this.total = total;
            this.start = start;
        }

        void update(long have) {
            long now = System.nanoTime();
            if (now - lastPrint < 250_000_000L) {
                return;
            }
            lastPrint = now;
            print(have, now);
        }

        void done(long have) {
            print(have, System.nanoTime());
            System.err.println();
        }

        private void print(long have, long now) {
            double secs = Math.max(1e-3, (now - t0) / 1e9);
            double mbps = (have - start) / 1e6 / secs;
            if (total > 0) {
                System.err.printf("\r  %-32s %8.1f / %.1f MB  %5.1f%%  %6.1f MB/s", name, have / 1e6, total / 1e6,
                        100.0 * have / total, mbps);
            } else {
                System.err.printf("\r  %-32s %8.1f MB  %6.1f MB/s", name, have / 1e6, mbps);
            }
        }
    }
}
