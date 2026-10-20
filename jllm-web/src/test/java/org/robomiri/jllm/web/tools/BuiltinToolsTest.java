package org.robomiri.jllm.web.tools;

import org.junit.jupiter.api.Test;

import java.net.URI;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BuiltinToolsTest {

    @Test
    void calculator() {
        var c = new BuiltinTools.Calculator();
        assertEquals("1234.5 * 678 / 9 = 92999", c.call(Map.of("expression", "1234.5 * 678 / 9")));
        assertTrue(c.call(Map.of("expression", "2^3^2")).endsWith("= 512"));
        assertTrue(c.call(Map.of("expression", "-2^2")).endsWith("= -4"));
        assertTrue(c.call(Map.of("expression", "sqrt(16) + max(1, 7, 3) % 4")).endsWith("= 7"));
        assertTrue(c.call(Map.of("expression", "1.5e3 / 3")).endsWith("= 500"));
        assertTrue(c.call(Map.of("expression", "round(pi * 100)")).endsWith("= 314"));
        assertThrows(IllegalArgumentException.class, () -> c.call(Map.of("expression", "2 +")));
        assertThrows(IllegalArgumentException.class, () -> c.call(Map.of("expression", "foo(1)")));
        assertThrows(IllegalArgumentException.class, () -> c.call(Map.of("expression", "(1")));
    }

    @Test
    void duckDuckGoMarkup() {
        String html = """
                <div class="result results_links results_links_deep result--ad ">
                  <a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad_domain=x">Ad</a></div>
                <div class="result results_links results_links_deep web-result ">
                  <div class="links_main links_deep result__body">
                    <h2 class="result__title">
                      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fopenjdk.org%2Fjeps%2F508&amp;rut=abc">JEP 508: <b>Vector API</b> (Tenth Incubator)</a>
                    </h2>
                    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">Introduce an API to express &quot;vector&quot; computations&#x2026;</a>
                  </div>
                </div>
                <div class="result results_links results_links_deep web-result ">
                    <h2 class="result__title"><a class="result__a" rel="nofollow" href="https://example.org/direct">Second</a></h2>
                </div>""";
        List<BuiltinTools.Result> r = BuiltinTools.parseDuckDuckGoHtml(html, 5);
        assertEquals(2, r.size());
        assertEquals("JEP 508: Vector API (Tenth Incubator)", r.get(0).title());
        assertEquals("https://openjdk.org/jeps/508", r.get(0).url());
        assertEquals("Introduce an API to express \"vector\" computations…", r.get(0).snippet());
        assertEquals("https://example.org/direct", r.get(1).url());

        String lite = "<a rel=\"nofollow\" href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.example%2F\" class='result-link'>A</a>"
                + "<td class='result-snippet'>snippet <b>A</b></td>";
        assertEquals(List.of(new BuiltinTools.Result("A", "https://a.example/", "snippet A")), BuiltinTools.parseDuckDuckGoLite(lite, 5));
    }

    @Test
    void htmlToText() {
        String page = "<html><head><title>T &amp; U</title><style>x{}</style></head><body><nav>menu</nav>"
                + "<h1>Head</h1><p>One&nbsp;two<br>three</p><script>evil()</script><ul><li>a</li><li>b</li></ul></body></html>";
        assertEquals("T & U", Html.title(page));
        assertEquals("Head\nOne two\nthree\na\nb", Html.toText(page));
    }

    @Test
    void fetchRefusesPrivateAddresses() {
        assertThrows(IllegalArgumentException.class, () -> BuiltinTools.FetchUrl.check(URI.create("http://127.0.0.1/"), false));
        assertThrows(IllegalArgumentException.class, () -> BuiltinTools.FetchUrl.check(URI.create("http://10.1.2.3/"), false));
        assertThrows(IllegalArgumentException.class, () -> BuiltinTools.FetchUrl.check(URI.create("http://[::1]/"), false));
        assertThrows(IllegalArgumentException.class, () -> BuiltinTools.FetchUrl.check(URI.create("file:///etc/passwd"), true));
    }
}
