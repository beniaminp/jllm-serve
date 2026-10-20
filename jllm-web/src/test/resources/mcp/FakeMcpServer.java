// A minimal MCP stdio server for the tests, run with the JDK's source launcher: java FakeMcpServer.java
// Tools: echo(text) and fail(); it pings the client once to check that server requests are answered.
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class FakeMcpServer {
    static String field(String json, String name) {
        Matcher m = Pattern.compile("\"" + name + "\"\\s*:\\s*(\"((?:[^\"\\\\]|\\\\.)*)\"|-?\\d+)").matcher(json);
        return m.find() ? (m.group(2) != null ? m.group(2) : m.group(1)) : null;
    }

    public static void main(String[] args) throws Exception {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, "UTF-8"));
        System.err.println("fake server starting");
        for (String line; (line = in.readLine()) != null; ) {
            String method = field(line, "method");
            String id = field(line, "id");
            if (method == null || id == null) {
                continue; // a notification or the answer to our ping
            }
            String result = switch (method) {
                case "initialize" -> "{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{\"tools\":{}},"
                        + "\"serverInfo\":{\"name\":\"fake\",\"version\":\"0.1\"},\"instructions\":\"for tests\"}";
                case "tools/list" -> "{\"tools\":[{\"name\":\"echo\",\"description\":\"Echo the text\",\"inputSchema\":"
                        + "{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}},\"required\":[\"text\"]}},"
                        + "{\"name\":\"fail\",\"description\":\"Always fails\",\"inputSchema\":{\"type\":\"object\"}}]}";
                case "tools/call" -> line.contains("\"fail\"")
                        ? "{\"content\":[{\"type\":\"text\",\"text\":\"it failed\"}],\"isError\":true}"
                        : "{\"content\":[{\"type\":\"text\",\"text\":\"echo: " + field(line, "text") + "\"}]}";
                default -> null;
            };
            if (method.equals("tools/list")) {
                System.out.println("{\"jsonrpc\":\"2.0\",\"id\":\"srv-1\",\"method\":\"ping\"}");
            }
            System.out.println(result == null
                    ? "{\"jsonrpc\":\"2.0\",\"id\":" + id + ",\"error\":{\"code\":-32601,\"message\":\"no " + method + "\"}}"
                    : "{\"jsonrpc\":\"2.0\",\"id\":" + id + ",\"result\":" + result + "}");
            System.out.flush();
        }
    }
}
