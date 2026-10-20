package org.robomiri.jllm.web.tools;

import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.io.InputStream;
import java.net.InetAddress;
import java.net.URI;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** The tools that need no MCP server: web search, fetching a page, a calculator and the clock. */
public final class BuiltinTools {

    static final String USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0";
    private static final HttpClient HTTP = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .followRedirects(HttpClient.Redirect.NEVER) // fetch_url checks every hop itself
            .build();

    private BuiltinTools() {
    }

    public static List<Tool> all(Supplier<ToolsConfig.General> config) {
        return List.of(new WebSearch(config), new FetchUrl(config), new Calculator(), new CurrentTime());
    }

    private static Map<String, Object> schema(Object... props) {
        Map<String, Object> properties = new java.util.LinkedHashMap<>();
        List<Object> required = new ArrayList<>();
        for (int i = 0; i < props.length; i += 4) {
            properties.put((String) props[i], Json.obj("type", props[i + 1], "description", props[i + 2]));
            if ((Boolean) props[i + 3]) {
                required.add(props[i]);
            }
        }
        return Json.obj("type", "object", "properties", properties, "required", required);
    }

    // ------------------------------------------------------------------ web search

    /** A search result. */
    public record Result(String title, String url, String snippet) {
    }

    /**
     * Web search. {@code auto} uses SearXNG when a URL is configured, else Brave when it has an API key, else
     * DuckDuckGo (the HTML page, then the lite page, then the Instant Answer API, since DuckDuckGo answers
     * datacenter addresses with a bot check now and then).
     */
    public static final class WebSearch implements Tool {
        private final Supplier<ToolsConfig.General> config;

        WebSearch(Supplier<ToolsConfig.General> config) {
            this.config = config;
        }

        @Override
        public String name() {
            return "web_search";
        }

        @Override
        public String description() {
            return "Search the web. Returns the top results with title, URL and a snippet. Use it for current events, "
                    + "facts you are unsure about and anything after your training data; then fetch_url a result to read it.";
        }

        @Override
        public Map<String, Object> parameters() {
            return schema("query", "string", "the search query", true,
                    "max_results", "integer", "how many results (default 5)", false);
        }

        @Override
        public String source() {
            return "builtin";
        }

        @Override
        public String call(Map<String, Object> args) throws Exception {
            String q = Json.str(args, "query", "").strip();
            if (q.isEmpty()) {
                throw new IllegalArgumentException("query is required");
            }
            ToolsConfig.Search cfg = config.get().search();
            int max = (int) Math.max(1, Math.min(20, Json.num(args, "max_results", cfg.maxResults())));
            List<Result> results = search(q, max, cfg);
            if (results.isEmpty()) {
                return "No results for: " + q;
            }
            StringBuilder sb = new StringBuilder("Results for: " + q + "\n");
            int i = 1;
            for (Result r : results) {
                sb.append('\n').append(i++).append(". ").append(r.title()).append('\n').append("   ").append(r.url()).append('\n');
                if (!r.snippet().isEmpty()) {
                    sb.append("   ").append(r.snippet()).append('\n');
                }
            }
            return sb.toString().strip();
        }

        static List<Result> search(String q, int max, ToolsConfig.Search cfg) throws Exception {
            String provider = cfg.provider();
            if (provider.equals("searxng") || (provider.equals("auto") && !cfg.searxngUrl().isBlank())) {
                return searxng(q, max, cfg.searxngUrl());
            }
            if (provider.equals("brave") || (provider.equals("auto") && !cfg.braveApiKey().isBlank())) {
                return brave(q, max, cfg.braveApiKey());
            }
            return duckDuckGo(q, max);
        }
    }

    static List<Result> duckDuckGo(String q, int max) throws Exception {
        List<String> problems = new ArrayList<>();
        try {
            HttpResponse<String> r = HTTP.send(HttpRequest.newBuilder(URI.create("https://html.duckduckgo.com/html/"))
                    .timeout(Duration.ofSeconds(20)).header("User-Agent", USER_AGENT)
                    .header("Content-Type", "application/x-www-form-urlencoded").header("Accept", "text/html")
                    .header("Accept-Language", "en-US,en;q=0.7")
                    .POST(HttpRequest.BodyPublishers.ofString("q=" + enc(q) + "&kl=wt-wt")).build(), HttpResponse.BodyHandlers.ofString());
            List<Result> res = parseDuckDuckGoHtml(r.body(), max);
            if (!res.isEmpty()) {
                return res;
            }
            problems.add("html: HTTP " + r.statusCode() + (r.body().contains("anomaly") ? " (bot check)" : ", no results"));
        } catch (IOException e) {
            problems.add("html: " + e.getMessage());
        }
        try {
            HttpResponse<String> r = HTTP.send(HttpRequest.newBuilder(URI.create("https://lite.duckduckgo.com/lite/?q=" + enc(q)))
                    .timeout(Duration.ofSeconds(20)).header("User-Agent", USER_AGENT).GET().build(), HttpResponse.BodyHandlers.ofString());
            List<Result> res = parseDuckDuckGoLite(r.body(), max);
            if (!res.isEmpty()) {
                return res;
            }
            problems.add("lite: HTTP " + r.statusCode());
        } catch (IOException e) {
            problems.add("lite: " + e.getMessage());
        }
        // the Instant Answer API: no web results, but an abstract and related topics for many entities
        HttpResponse<String> r = HTTP.send(HttpRequest.newBuilder(URI.create("https://api.duckduckgo.com/?format=json&no_html=1&skip_disambig=1&q=" + enc(q)))
                .timeout(Duration.ofSeconds(20)).header("User-Agent", USER_AGENT).GET().build(), HttpResponse.BodyHandlers.ofString());
        List<Result> res = r.statusCode() == 200 ? parseInstantAnswer(r.body(), max) : List.of();
        if (res.isEmpty()) {
            throw new IOException("DuckDuckGo returned no results (" + String.join("; ", problems) + "); configure a SearXNG"
                    + " instance or a Brave API key in the Tools tab for reliable search");
        }
        return res;
    }

    private static final Pattern DDG_LINK = Pattern.compile("(?is)<a\\b([^>]*class=\"result__a\"[^>]*)>(.*?)</a>");
    private static final Pattern DDG_SNIPPET = Pattern.compile("(?is)class=\"result__snippet\"[^>]*>(.*?)</(?:a|td|div)>");
    private static final Pattern HREF = Pattern.compile("(?i)href=\"([^\"]*)\"");

    static List<Result> parseDuckDuckGoHtml(String html, int max) {
        List<Result> out = new ArrayList<>();
        // one block per result; ads carry result--ad
        String[] blocks = html.split("(?i)<div class=\"result ");
        for (int i = 1; i < blocks.length && out.size() < max; i++) {
            String b = blocks[i];
            if (b.startsWith("result--ad") || b.contains("result--ad ")) {
                continue;
            }
            Matcher a = DDG_LINK.matcher(b);
            if (!a.find()) {
                continue;
            }
            Matcher h = HREF.matcher(a.group(1));
            if (!h.find()) {
                continue;
            }
            String url = ddgTarget(Html.decode(h.group(1)));
            if (url == null) {
                continue;
            }
            Matcher s = DDG_SNIPPET.matcher(b);
            out.add(new Result(Html.clean(a.group(2)), url, s.find() ? Html.clean(s.group(1)) : ""));
        }
        return out;
    }

    static List<Result> parseDuckDuckGoLite(String html, int max) {
        List<Result> out = new ArrayList<>();
        Matcher a = Pattern.compile("(?is)<a\\b([^>]*class=['\"]result-link['\"][^>]*)>(.*?)</a>").matcher(html);
        Matcher s = Pattern.compile("(?is)<td[^>]*class=['\"]result-snippet['\"][^>]*>(.*?)</td>").matcher(html);
        while (a.find() && out.size() < max) {
            Matcher h = Pattern.compile("(?i)href=['\"]([^'\"]*)['\"]").matcher(a.group(1));
            String url = h.find() ? ddgTarget(Html.decode(h.group(1))) : null;
            String snippet = s.find() ? Html.clean(s.group(1)) : "";
            if (url != null) {
                out.add(new Result(Html.clean(a.group(2)), url, snippet));
            }
        }
        return out;
    }

    static List<Result> parseInstantAnswer(String json, int max) {
        Map<String, Object> o = Json.parseObject(json);
        List<Result> out = new ArrayList<>();
        String abs = Json.str(o, "AbstractText", "");
        if (!abs.isEmpty()) {
            out.add(new Result(Json.str(o, "Heading", "Summary"), Json.str(o, "AbstractURL", ""), abs));
        }
        String answer = Json.str(o, "Answer", "");
        if (!answer.isEmpty()) {
            out.add(new Result("Instant answer", "", answer));
        }
        List<Object> topics = Json.list(o, "RelatedTopics");
        if (topics != null) {
            for (Object t : topics) {
                if (out.size() >= max) {
                    break;
                }
                String text = Json.str(t, "Text", "");
                if (!text.isEmpty()) {
                    out.add(new Result(text.length() > 80 ? text.substring(0, 80) + "…" : text, Json.str(t, "FirstURL", ""), text));
                }
            }
        }
        return out;
    }

    /** DuckDuckGo wraps result links in {@code //duckduckgo.com/l/?uddg=<target>}; ads go to y.js. */
    static String ddgTarget(String href) {
        if (href.startsWith("//")) {
            href = "https:" + href;
        }
        int u = href.indexOf("uddg=");
        if (u >= 0) {
            String v = href.substring(u + 5);
            int amp = v.indexOf('&');
            return URLDecoder.decode(amp >= 0 ? v.substring(0, amp) : v, StandardCharsets.UTF_8);
        }
        if (href.contains("duckduckgo.com/y.js")) {
            return null;
        }
        return href.startsWith("http") ? href : null;
    }

    static List<Result> searxng(String q, int max, String base) throws Exception {
        if (base == null || base.isBlank()) {
            throw new IllegalStateException("no SearXNG URL is configured (Tools tab)");
        }
        String url = base.replaceAll("/+$", "") + "/search?format=json&q=" + enc(q);
        HttpResponse<String> r = HTTP.send(HttpRequest.newBuilder(URI.create(url)).timeout(Duration.ofSeconds(20))
                .header("User-Agent", USER_AGENT).header("Accept", "application/json").GET().build(), HttpResponse.BodyHandlers.ofString());
        if (r.statusCode() != 200) {
            throw new IOException("SearXNG returned HTTP " + r.statusCode() + (r.statusCode() == 403
                    ? " (enable the json format under search.formats in its settings.yml)" : ""));
        }
        if (!r.body().stripLeading().startsWith("{")) {
            throw new IOException("SearXNG at " + base + " did not answer in JSON: add json to search.formats in its"
                    + " settings.yml (most public instances turn it off; a local one is a single docker run)");
        }
        List<Result> out = new ArrayList<>();
        List<Object> results = Json.list(Json.parseObject(r.body()), "results");
        if (results != null) {
            for (Object o : results) {
                if (out.size() >= max) {
                    break;
                }
                out.add(new Result(Json.str(o, "title", ""), Json.str(o, "url", ""), Json.str(o, "content", "")));
            }
        }
        return out;
    }

    static List<Result> brave(String q, int max, String key) throws Exception {
        if (key == null || key.isBlank()) {
            throw new IllegalStateException("no Brave Search API key is configured (Tools tab)");
        }
        HttpResponse<String> r = HTTP.send(HttpRequest.newBuilder(URI.create("https://api.search.brave.com/res/v1/web/search?count="
                        + max + "&q=" + enc(q))).timeout(Duration.ofSeconds(20)).header("Accept", "application/json")
                .header("X-Subscription-Token", key).GET().build(), HttpResponse.BodyHandlers.ofString());
        if (r.statusCode() != 200) {
            throw new IOException("Brave Search returned HTTP " + r.statusCode());
        }
        List<Result> out = new ArrayList<>();
        List<Object> results = Json.list(Json.map(Json.parseObject(r.body()), "web"), "results");
        if (results != null) {
            for (Object o : results) {
                if (out.size() >= max) {
                    break;
                }
                out.add(new Result(Html.clean(Json.str(o, "title", "")), Json.str(o, "url", ""), Html.clean(Json.str(o, "description", ""))));
            }
        }
        return out;
    }

    private static String enc(String s) {
        return URLEncoder.encode(s, StandardCharsets.UTF_8);
    }

    // ------------------------------------------------------------------ fetch

    /** Downloads a web page and returns its readable text (private and loopback addresses refused by default). */
    public static final class FetchUrl implements Tool {
        private static final int MAX_BYTES = 3 << 20;
        private final Supplier<ToolsConfig.General> config;

        FetchUrl(Supplier<ToolsConfig.General> config) {
            this.config = config;
        }

        @Override
        public String name() {
            return "fetch_url";
        }

        @Override
        public String description() {
            return "Fetch a web page (http or https) and return its title and readable text.";
        }

        @Override
        public Map<String, Object> parameters() {
            return schema("url", "string", "the address of the page", true,
                    "max_chars", "integer", "truncate the text to this many characters", false);
        }

        @Override
        public String source() {
            return "builtin";
        }

        @Override
        public String call(Map<String, Object> args) throws Exception {
            ToolsConfig.Fetch cfg = config.get().fetch();
            String url = Json.str(args, "url", "").strip();
            int maxChars = (int) Math.max(200, Math.min(cfg.maxChars(), Json.num(args, "max_chars", cfg.maxChars())));
            URI uri = URI.create(url);
            HttpResponse<InputStream> r = null;
            for (int hop = 0; hop < 6; hop++) {
                check(uri, cfg.allowPrivateAddresses());
                r = HTTP.send(HttpRequest.newBuilder(uri).timeout(Duration.ofSeconds(30)).header("User-Agent", USER_AGENT)
                        .header("Accept", "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.5")
                        .GET().build(), HttpResponse.BodyHandlers.ofInputStream());
                int sc = r.statusCode();
                if (sc >= 300 && sc < 400 && r.headers().firstValue("location").isPresent()) {
                    r.body().close();
                    uri = uri.resolve(r.headers().firstValue("location").get());
                    continue;
                }
                break;
            }
            if (r == null || (r.statusCode() >= 300 && r.statusCode() < 400)) {
                throw new IOException("too many redirects");
            }
            byte[] bytes;
            try (InputStream in = r.body()) {
                bytes = in.readNBytes(MAX_BYTES);
            }
            if (r.statusCode() >= 400) {
                throw new IOException("HTTP " + r.statusCode() + " from " + uri);
            }
            String type = r.headers().firstValue("content-type").orElse("text/html").toLowerCase(Locale.ROOT);
            Charset cs = StandardCharsets.UTF_8;
            Matcher m = Pattern.compile("charset=([\\w-]+)").matcher(type);
            if (m.find()) {
                try {
                    cs = Charset.forName(m.group(1));
                } catch (RuntimeException ignored) {
                    // keep UTF-8
                }
            }
            String body = new String(bytes, cs);
            String text;
            String title = "";
            if (type.contains("html") || type.contains("xml")) {
                title = Html.title(body);
                text = Html.toText(body);
            } else if (type.startsWith("text/") || type.contains("json")) {
                text = body;
            } else {
                throw new IOException("unsupported content type " + type + " at " + uri);
            }
            boolean cut = text.length() > maxChars;
            if (cut) {
                text = text.substring(0, maxChars);
            }
            return (title.isEmpty() ? "" : "Title: " + title + "\n") + "URL: " + uri + "\n\n" + text
                    + (cut ? "\n\n[truncated at " + maxChars + " characters]" : "");
        }

        /** Only http(s), and no loopback / private / link-local hosts unless allowed. */
        static void check(URI uri, boolean allowPrivate) throws IOException {
            String scheme = uri.getScheme() == null ? "" : uri.getScheme().toLowerCase(Locale.ROOT);
            if (!scheme.equals("http") && !scheme.equals("https")) {
                throw new IllegalArgumentException("only http and https URLs can be fetched");
            }
            if (uri.getHost() == null) {
                throw new IllegalArgumentException("the URL has no host");
            }
            if (allowPrivate) {
                return;
            }
            for (InetAddress a : InetAddress.getAllByName(uri.getHost())) {
                if (a.isLoopbackAddress() || a.isSiteLocalAddress() || a.isLinkLocalAddress() || a.isAnyLocalAddress()
                        || a.isMulticastAddress() || isUniqueLocalV6(a)) {
                    throw new IllegalArgumentException(uri.getHost() + " is a private address; fetching it is turned off"
                            + " (Tools tab, fetch: allow private addresses)");
                }
            }
        }

        private static boolean isUniqueLocalV6(InetAddress a) {
            byte[] b = a.getAddress();
            return b.length == 16 && (b[0] & 0xfe) == 0xfc;
        }
    }

    // ------------------------------------------------------------------ calculator

    /** Arithmetic that small models get wrong: + - * / % ^, parentheses, common functions and constants. */
    public static final class Calculator implements Tool {
        @Override
        public String name() {
            return "calculator";
        }

        @Override
        public String description() {
            return "Evaluate an arithmetic expression exactly, e.g. \"(17.5 * 3) / 4 + sqrt(2)^3\". Supports + - * / % ^,"
                    + " parentheses, sqrt, abs, ln, log10, log2, exp, sin, cos, tan, asin, acos, atan, floor, ceil, round, min, max, pow, pi, e.";
        }

        @Override
        public Map<String, Object> parameters() {
            return schema("expression", "string", "the expression to evaluate", true);
        }

        @Override
        public String source() {
            return "builtin";
        }

        @Override
        public String call(Map<String, Object> args) {
            String expr = Json.str(args, "expression", "");
            double v = new Expr(expr).parse();
            String s = v == Math.rint(v) && Math.abs(v) < 1e15 ? String.valueOf((long) v) : String.valueOf(v);
            return expr.strip() + " = " + s;
        }
    }

    /** Recursive descent over the usual precedence; ^ is right-associative and binds tighter than unary minus. */
    static final class Expr {
        private final String s;
        private int pos;

        Expr(String s) {
            this.s = s.replace('×', '*').replace('÷', '/').replace("**", "^");
        }

        double parse() {
            double v = sum();
            skip();
            if (pos < s.length()) {
                throw new IllegalArgumentException("unexpected '" + s.charAt(pos) + "' at position " + pos);
            }
            return v;
        }

        private double sum() {
            double v = product();
            for (; ; ) {
                if (eat('+')) {
                    v += product();
                } else if (eat('-')) {
                    v -= product();
                } else {
                    return v;
                }
            }
        }

        private double product() {
            double v = unary();
            for (; ; ) {
                if (eat('*')) {
                    v *= unary();
                } else if (eat('/')) {
                    v /= unary();
                } else if (eat('%')) {
                    v %= unary();
                } else {
                    return v;
                }
            }
        }

        private double unary() {
            if (eat('-')) {
                return -unary();
            }
            if (eat('+')) {
                return unary();
            }
            return power();
        }

        private double power() {
            double base = atom();
            return eat('^') ? Math.pow(base, unary()) : base;
        }

        private double atom() {
            skip();
            if (eat('(')) {
                double v = sum();
                expect(')');
                return v;
            }
            int start = pos;
            if (pos < s.length() && (Character.isDigit(s.charAt(pos)) || s.charAt(pos) == '.')) {
                while (pos < s.length() && (Character.isDigit(s.charAt(pos)) || s.charAt(pos) == '.' || s.charAt(pos) == '_'
                        || ((s.charAt(pos) == 'e' || s.charAt(pos) == 'E') && pos + 1 < s.length()
                        && (Character.isDigit(s.charAt(pos + 1)) || s.charAt(pos + 1) == '-' || s.charAt(pos + 1) == '+')))) {
                    if ((s.charAt(pos) == 'e' || s.charAt(pos) == 'E')) {
                        pos++;
                    }
                    pos++;
                }
                return Double.parseDouble(s.substring(start, pos).replace("_", ""));
            }
            while (pos < s.length() && Character.isLetterOrDigit(s.charAt(pos))) {
                pos++;
            }
            String name = s.substring(start, pos).toLowerCase(Locale.ROOT);
            if (name.isEmpty()) {
                throw new IllegalArgumentException(pos < s.length() ? "unexpected '" + s.charAt(pos) + "' at position " + pos : "unexpected end of expression");
            }
            switch (name) {
                case "pi" -> {
                    return Math.PI;
                }
                case "e" -> {
                    return Math.E;
                }
                default -> {
                }
            }
            expect('(');
            List<Double> a = new ArrayList<>();
            if (!eat(')')) {
                do {
                    a.add(sum());
                } while (eat(','));
                expect(')');
            }
            return apply(name, a);
        }

        private static double apply(String f, List<Double> a) {
            java.util.function.IntConsumer need = n -> {
                if (a.size() != n) {
                    throw new IllegalArgumentException(f + " takes " + n + " argument" + (n == 1 ? "" : "s"));
                }
            };
            switch (f) {
                case "min", "max" -> {
                    if (a.isEmpty()) {
                        throw new IllegalArgumentException(f + " needs arguments");
                    }
                    return f.equals("min") ? a.stream().mapToDouble(Double::doubleValue).min().orElseThrow()
                            : a.stream().mapToDouble(Double::doubleValue).max().orElseThrow();
                }
                case "pow" -> {
                    need.accept(2);
                    return Math.pow(a.get(0), a.get(1));
                }
                default -> need.accept(1);
            }
            double x = a.getFirst();
            return switch (f) {
                case "sqrt" -> Math.sqrt(x);
                case "abs" -> Math.abs(x);
                case "ln", "log" -> Math.log(x);
                case "log10" -> Math.log10(x);
                case "log2" -> Math.log(x) / Math.log(2);
                case "exp" -> Math.exp(x);
                case "sin" -> Math.sin(x);
                case "cos" -> Math.cos(x);
                case "tan" -> Math.tan(x);
                case "asin" -> Math.asin(x);
                case "acos" -> Math.acos(x);
                case "atan" -> Math.atan(x);
                case "floor" -> Math.floor(x);
                case "ceil" -> Math.ceil(x);
                case "round" -> Math.round(x);
                default -> throw new IllegalArgumentException("unknown function " + f);
            };
        }

        private void skip() {
            while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) {
                pos++;
            }
        }

        private boolean eat(char c) {
            skip();
            if (pos < s.length() && s.charAt(pos) == c) {
                pos++;
                return true;
            }
            return false;
        }

        private void expect(char c) {
            if (!eat(c)) {
                throw new IllegalArgumentException("expected '" + c + "' at position " + pos);
            }
        }
    }

    // ------------------------------------------------------------------ time

    public static final class CurrentTime implements Tool {
        @Override
        public String name() {
            return "current_time";
        }

        @Override
        public String description() {
            return "The current date and time, in the server's time zone or a given IANA time zone such as Europe/Paris.";
        }

        @Override
        public Map<String, Object> parameters() {
            return schema("timezone", "string", "an IANA time zone (optional)", false);
        }

        @Override
        public String source() {
            return "builtin";
        }

        @Override
        public String call(Map<String, Object> args) {
            String tz = Json.str(args, "timezone", "");
            ZoneId zone;
            try {
                zone = tz.isBlank() ? ZoneId.systemDefault() : ZoneId.of(tz.strip());
            } catch (RuntimeException e) {
                throw new IllegalArgumentException("unknown time zone " + tz);
            }
            ZonedDateTime now = ZonedDateTime.now(zone);
            return now.format(DateTimeFormatter.ofPattern("EEEE, d MMMM yyyy, HH:mm:ss", Locale.ENGLISH)) + " (" + zone + ", UTC"
                    + now.getOffset() + ")";
        }
    }
}
