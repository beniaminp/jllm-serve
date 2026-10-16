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
import java.util.TreeSet;
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
 * Client errors (4xx) are not retried; server errors (5xx) and I/O failures are retried with back-off.
 */
public final class HuggingFaceHub {

    private static final int MAX_REDIRECTS = 10;
    private static final int MAX_ATTEMPTS = 5;
    /** Default response / stall timeout. */
    public static final long DEFAULT_TIMEOUT_MILLIS = 60_000;

    private final HttpClient http;
    private final String endpoint;
    private final String token;
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
        setTimeoutMillis(timeoutMillis);
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

    /** An HTTP response with an unexpected status; 4xx statuses are not retried. */
    public static final class HttpStatusException extends IOException {
        private final int status;

        HttpStatusException(int status, String message) {
            super(message);
            this.status = status;
        }

        public int status() {
            return status;
        }

        boolean isClientError() {
            return status >= 400 && status < 500;
        }
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
        HttpResponse<String> resp = send(uri, HttpResponse.BodyHandlers.ofString(), null, false);
        if (resp.statusCode() != 200) {
            throw new HttpStatusException(resp.statusCode(), "Hub API " + uri + " returned HTTP " + resp.statusCode()
                    + (resp.statusCode() == 401 || resp.statusCode() == 403 ? " (gated/private repo? set HF_TOKEN)" : "")
                    + ": " + truncate(resp.body()));
        }
        Map<String, Object> info = Json.parseObject(resp.body());
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
                    if (attempt >= MAX_ATTEMPTS || e instanceof HttpStatusException h && h.isClientError()) {
                        throw e;
                    }
                    System.err.printf("%n  %s: %s, retrying (%d/%d)%n", f.name(), e.getMessage(), attempt, MAX_ATTEMPTS);
                    Thread.sleep(1000L * attempt);
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
        if (status == 416 && have > 0 && have == f.size()) {
            resp.body().close();
            return have; // already complete
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
            throw new HttpStatusException(status, "GET " + uri + " returned HTTP " + status + ": " + truncate(body));
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
     * only over https so the token never travels in clear text.
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
            if (token != null && hubHost != null && hubHost.equalsIgnoreCase(current.getHost())
                    && "https".equalsIgnoreCase(current.getScheme())) {
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
