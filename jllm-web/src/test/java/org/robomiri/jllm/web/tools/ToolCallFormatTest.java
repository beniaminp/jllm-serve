package org.robomiri.jllm.web.tools;

import org.junit.jupiter.api.Test;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.tokenizer.ChatTemplate;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ToolCallFormatTest {

    @Test
    void parsesCalls() {
        var p = ToolCallFormat.parse("Let me check.\n<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"x\"}}\n</tool_call>\n"
                + "<tool_call>{\"name\":\"calculator\",\"arguments\":\"{\\\"expression\\\":\\\"1+1\\\"}\"}</tool_call>", Set.of());
        assertEquals("Let me check.", p.content());
        assertEquals(2, p.calls().size());
        assertEquals("web_search", p.calls().get(0).name());
        assertEquals("x", p.calls().get(0).arguments().get("query"));
        assertEquals("1+1", p.calls().get(1).arguments().get("expression")); // arguments given as a JSON string
    }

    @Test
    void thinkingAndBareJson() {
        var thinking = ToolCallFormat.parse("<think>maybe <tool_call>{\"name\":\"a\"}</tool_call></think>\n\nNo tools needed.", Set.of("a"));
        assertTrue(thinking.calls().isEmpty());
        var unfinished = ToolCallFormat.parse("<think>I could use <tool_call>", Set.of("a"));
        assertTrue(unfinished.calls().isEmpty());
        var after = ToolCallFormat.parse("<think>ok</think>\n<tool_call>{\"name\":\"a\",\"arguments\":{}}", Set.of()); // no closing tag
        assertEquals(1, after.calls().size());
        assertEquals("<think>ok</think>", after.content());
        var bare = ToolCallFormat.parse("{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}", Set.of("get_weather"));
        assertEquals("Paris", bare.calls().getFirst().arguments().get("city"));
        var unknown = ToolCallFormat.parse("{\"name\": \"other\", \"parameters\": {}}", Set.of("get_weather"));
        assertTrue(unknown.calls().isEmpty());
        var broken = ToolCallFormat.parse("<tool_call>{name: oops</tool_call>", Set.of());
        assertNotNull(broken.calls().getFirst().error());
        assertNull(broken.calls().getFirst().name());
    }

    @Test
    void holdbackStopsAtTheCall() {
        StringBuilder out = new StringBuilder();
        var h = new ToolCallFormat.Holdback(out::append);
        for (String piece : new String[]{"Sure", ", <", "tool", "_call>\n{\"name\"", ": \"a\"}</tool_call>"}) {
            h.accept(piece);
        }
        h.finish();
        assertEquals("Sure, ", out.toString());
        assertTrue(h.callStarted());

        StringBuilder out2 = new StringBuilder();
        var h2 = new ToolCallFormat.Holdback(out2::append);
        h2.accept("a < b and <tool");
        assertEquals("a < b and ", out2.toString()); // "<tool" is held back until decided
        h2.accept("s>");
        h2.finish();
        assertEquals("a < b and <tools>", out2.toString());

        StringBuilder out3 = new StringBuilder();
        var h3 = new ToolCallFormat.Holdback(out3::append);
        h3.accept("<think>about <tool_call> syntax</think>");
        h3.finish();
        assertEquals("<think>about <tool_call> syntax</think>", out3.toString());
    }

    @Test
    void canonicalJson() {
        assertEquals("{\"a\": [1, 2.5, \"x\\n\"], \"b\": {\"c\": null, \"d\": true}, \"e\": {}, \"f\": []}",
                ToolCallFormat.canonical(Json.parseObject("{\"a\":[1,2.5,\"x\\n\"],\"b\":{\"c\":null,\"d\":true},\"e\":{},\"f\":[]}")));
    }

    @Test
    void templateMessages() {
        List<Object> msgs = List.of(
                Json.obj("role", "user", "content", "weather?"),
                Json.obj("role", "assistant", "content", null, "tool_calls", List.of(Json.obj("id", "c1", "type", "function",
                        "function", Json.obj("name", "w", "arguments", "{\"city\":\"Paris\"}")))),
                Json.obj("role", "tool", "tool_call_id", "c1", "content", "sunny"),
                Json.obj("role", "tool", "tool_call_id", "c2", "content", "warm"));
        List<ChatTemplate.Message> qwen = ToolCallFormat.toTemplate(msgs, "TOOLS", "chatml-qwen3");
        assertEquals("system", qwen.get(0).role());
        assertEquals("TOOLS", qwen.get(0).content());
        // earlier calls in the layout the constraint makes the model write (json.dumps spacing)
        assertEquals("<tool_call>\n{\"name\": \"w\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>", qwen.get(2).content());
        assertEquals("tool", qwen.get(3).role());

        List<ChatTemplate.Message> llama = ToolCallFormat.toTemplate(msgs, null, "llama3");
        assertEquals(3, llama.size()); // the two results share one user turn
        assertEquals("user", llama.get(2).role());
        assertEquals("<tool_response>\nsunny\n</tool_response>\n<tool_response>\nwarm\n</tool_response>", llama.get(2).content());

        List<ChatTemplate.Message> withSystem = ToolCallFormat.toTemplate(
                List.of(Json.obj("role", "system", "content", "Be brief."), Json.obj("role", "user", "content", "hi")), "TOOLS", "chatml-qwen3");
        assertEquals("Be brief.\n\nTOOLS", withSystem.getFirst().content());
    }

    @Test
    void toolsBlockListsFunctions() {
        String b = ToolCallFormat.toolsBlock(List.of(Map.of("type", "function", "function", Map.of("name", "f"))), "extra");
        assertTrue(b.contains("<tools>\n{"));
        assertTrue(b.endsWith("extra"));
    }
}
