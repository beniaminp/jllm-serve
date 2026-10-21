package org.robomiri.jllm.vision;

import java.io.IOException;
import java.io.InputStream;
import java.net.InetAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Locale;

/**
 * Downloads the picture a request names by URL, with the rules the console's {@code fetch_url} tool has: only
 * {@code http} and {@code https}, every redirect hop checked again, loopback / private / link-local / multicast
 * addresses refused unless allowed, a size limit that stops the download rather than checking afterwards.
 */
public final class ImageFetcher {

    private static final HttpClient HTTP = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .followRedirects(HttpClient.Redirect.NEVER) // each hop is checked here
            .build();

    private final boolean allowPrivate;
    private final long maxBytes;

    /**
     * @param allowPrivate whether addresses on private networks and the machine itself may be fetched
     * @param maxBytes     largest download accepted
     */
    public ImageFetcher(boolean allowPrivate, long maxBytes) {
        this.allowPrivate = allowPrivate;
        this.maxBytes = maxBytes;
    }

    /** The bytes behind an {@code http(s)} URL. */
    public byte[] fetch(String url) throws IOException {
        URI uri;
        try {
            uri = URI.create(url.strip());
        } catch (IllegalArgumentException e) {
            throw new IOException("invalid image URL: " + e.getMessage(), e);
        }
        try {
            HttpResponse<InputStream> r = null;
            for (int hop = 0; hop < 6; hop++) {
                check(uri, allowPrivate);
                r = HTTP.send(HttpRequest.newBuilder(uri).timeout(Duration.ofSeconds(30))
                        .header("User-Agent", "jllm-serve").header("Accept", "image/*,*/*;q=0.5").GET().build(),
                        HttpResponse.BodyHandlers.ofInputStream());
                int sc = r.statusCode();
                if (sc >= 300 && sc < 400 && r.headers().firstValue("location").isPresent()) {
                    r.body().close();
                    uri = uri.resolve(r.headers().firstValue("location").get());
                    r = null;
                    continue;
                }
                break;
            }
            if (r == null) {
                throw new IOException("too many redirects fetching " + url);
            }
            try (InputStream in = r.body()) {
                if (r.statusCode() >= 400) {
                    throw new IOException("HTTP " + r.statusCode() + " from " + uri);
                }
                long declared = r.headers().firstValueAsLong("content-length").orElse(-1);
                if (declared > maxBytes) {
                    throw new IOException("the image at " + uri + " is " + declared + " bytes, the limit is " + maxBytes);
                }
                byte[] bytes = in.readNBytes((int) Math.min(maxBytes + 1, Integer.MAX_VALUE - 8));
                if (bytes.length > maxBytes) {
                    throw new IOException("the image at " + uri + " is larger than the limit of " + maxBytes + " bytes");
                }
                return bytes;
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted while fetching " + url);
        }
    }

    /** Only http(s), and no loopback / private / link-local hosts unless allowed. */
    public static void check(URI uri, boolean allowPrivate) throws IOException {
        String scheme = uri.getScheme() == null ? "" : uri.getScheme().toLowerCase(Locale.ROOT);
        if (!scheme.equals("http") && !scheme.equals("https")) {
            throw new IllegalArgumentException("only http and https image URLs can be fetched (or a data: URL)");
        }
        if (uri.getHost() == null) {
            throw new IllegalArgumentException("the image URL has no host");
        }
        if (allowPrivate) {
            return;
        }
        for (InetAddress a : InetAddress.getAllByName(uri.getHost())) {
            if (a.isLoopbackAddress() || a.isSiteLocalAddress() || a.isLinkLocalAddress() || a.isAnyLocalAddress()
                    || a.isMulticastAddress() || isUniqueLocalV6(a)) {
                throw new IllegalArgumentException(uri.getHost() + " is a private address; fetching it is turned off"
                        + " (vision.allow-private-urls, or the Tools tab's fetch setting in the console)");
            }
        }
    }

    private static boolean isUniqueLocalV6(InetAddress a) {
        byte[] b = a.getAddress();
        return b.length == 16 && (b[0] & 0xfe) == 0xfc;
    }
}
