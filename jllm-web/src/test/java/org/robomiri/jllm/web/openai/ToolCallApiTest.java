package org.robomiri.jllm.web.openai;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.sampling.JsonConstraint;
import org.robomiri.jllm.tokenizer.Tokenizer;
import org.robomiri.jllm.web.models.ModelManager;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.asyncDispatch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;

/**
 * {@code POST /v1/chat/completions} with {@code tools}, served by a scripted fake model on the Qwen3 tokenizer and
 * chat template: the model "wants" to write a call but is tempted, inside it, by tokens that would break the JSON
 * (raw line breaks). The constraint keeps the call valid, so the response carries {@code tool_calls} whose arguments
 * parse and match the tool's schema.
 */
@SpringBootTest(properties = {"jllm.web.settings-file=target/tool-call-api-test.properties", "jllm.web.kv-cache=false",
        "jllm.web.tools-file=target/tool-call-api-test-tools.json"})
@AutoConfigureMockMvc
class ToolCallApiTest {

    static final Path MODEL = Path.of("../models/Qwen_Qwen3-0.6B");
    static final String TOOLS = "[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {"
            + "\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, \"days\": {\"type\": \"integer\"}},"
            + " \"required\": [\"city\"], \"additionalProperties\": false}}},"
            + " {\"type\": \"function\", \"function\": {\"name\": \"read_file\", \"parameters\": {\"type\": \"object\","
            + " \"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"]}}}]";
    static final String CALL = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>";

    @Autowired
    MockMvc mvc;
    @Autowired
    ModelManager models;
    ScriptBackend backend;
    Engine engine;

    /**
     * A model that wants to write {@code script} after the prompt's (empty) thinking block: the longest token that
     * continues it gets logit 10; inside a call, every third position a token that is invalid there (a raw line
     * break or tab) gets 12; off the script it ends the turn.
     */
    static final class ScriptBackend implements Backend {
        final byte[][] table;
        final int[] kv = new int[8192];
        final int[] marker;
        final int[] distract;
        final int eos;
        final int vocab;
        volatile byte[] script = new byte[0];

        ScriptBackend(Tokenizer tok, int vocab) {
            this.table = JsonConstraint.tokenTable(tok);
            this.vocab = vocab;
            this.marker = tok.encode("</think>\n\n");
            this.eos = tok.tokenId("<|im_end|>");
            this.distract = new int[]{tok.encode("\n\n")[0], tok.encode(".\n")[0], tok.encode("\t\t")[0]};
        }

        public String name() {
            return "script";
        }

        public int contextLength() {
            return 8192;
        }

        public int maxBatch() {
            return 32;
        }

        void logitsAt(int p, float[] out, int base) {
            int start = 0;
            for (int i = p - marker.length + 1; i >= 0; i--) {
                if (Arrays.equals(kv, i, i + marker.length, marker, 0, marker.length)) {
                    start = i + marker.length;
                    break;
                }
            }
            ByteArrayOutputStream g = new ByteArrayOutputStream();
            for (int i = start; i <= p; i++) {
                g.writeBytes(table[kv[i]] == null ? new byte[0] : table[kv[i]]);
            }
            byte[] gen = g.toByteArray();
            for (int t = 0; t < vocab; t++) {
                out[base + t] = (float) (((t * 2654435761L + p * 40503L) & 1023) / 1024.0);
            }
            byte[] s = script;
            if (gen.length >= s.length || !Arrays.equals(gen, 0, gen.length, s, 0, gen.length)) {
                out[base + eos] = 20;
                return;
            }
            int best = eos;
            int bestLen = 0;
            for (int t = 0; t < table.length; t++) {
                byte[] b = table[t];
                if (b != null && b.length > bestLen && gen.length + b.length <= s.length
                        && Arrays.equals(b, 0, b.length, s, gen.length, gen.length + b.length)) {
                    best = t;
                    bestLen = b.length;
                }
            }
            out[base + best] = 10;
            String g8 = new String(gen, StandardCharsets.UTF_8);
            if (g8.lastIndexOf("<tool_call>") > g8.lastIndexOf("</tool_call>") && p % 3 == 0) {
                out[base + distract[p % distract.length]] = 12;
            }
        }

        public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
            System.arraycopy(tokens, off, kv, pos, n);
            if (logits != null) {
                logitsAt(pos + n - 1, logits, 0);
            }
        }

        @Override
        public void forwardAll(int[] tokens, int off, int n, int pos, float[] out, int stride) {
            System.arraycopy(tokens, off, kv, pos, n);
            for (int i = 0; i < n; i++) {
                logitsAt(pos + i, out, i * stride);
            }
        }

        public void close() {
        }
    }

    @BeforeEach
    void register() throws Exception {
        assumeTrue(Files.exists(MODEL.resolve("tokenizer.json")), "needs " + MODEL);
        Tokenizer tok = Tokenizer.load(MODEL.resolve("tokenizer.json"));
        Architecture arch = Architecture.load(MODEL); // config.json only: the Qwen3 chat template
        backend = new ScriptBackend(tok, arch.core().vocabSize());
        engine = new Engine(backend, tok, arch, Set.of());
        Method m = ModelManager.class.getDeclaredMethod("registerForTest", String.class, Engine.class);
        m.setAccessible(true);
        m.invoke(models, "fake", engine);
    }

    @AfterEach
    void unregister() throws Exception {
        if (engine != null) {
            Method m = ModelManager.class.getDeclaredMethod("removeForTest", String.class);
            m.setAccessible(true);
            m.invoke(models, "fake");
        }
    }

    String body(String extra) {
        return "{\"model\": \"fake\", \"messages\": [{\"role\": \"user\", \"content\": \"Weather in Paris for 3 days?\"}],"
                + " \"tools\": " + TOOLS + ", \"temperature\": 0, \"max_tokens\": 120, \"enable_thinking\": false" + extra + "}";
    }

    @SuppressWarnings("unchecked")
    Map<String, Object> chat(String body) throws Exception {
        MvcResult r = mvc.perform(post("/v1/chat/completions").contentType("application/json").content(body)).andReturn();
        if (r.getRequest().isAsyncStarted()) {
            r = mvc.perform(asyncDispatch(r)).andReturn();
        }
        assertEquals(200, r.getResponse().getStatus(), r.getResponse().getContentAsString());
        return Json.parseObject(r.getResponse().getContentAsString(StandardCharsets.UTF_8));
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> onlyCall(Map<String, Object> resp) {
        Map<String, Object> choice = (Map<String, Object>) Json.list(resp, "choices").getFirst();
        assertEquals("tool_calls", choice.get("finish_reason"), Json.write(resp));
        List<Object> calls = Json.list(Json.map(choice, "message"), "tool_calls");
        assertEquals(1, calls.size(), Json.write(resp));
        return Json.map(calls.getFirst(), "function");
    }

    @Test
    void toolCallArgumentsAreValid() throws Exception {
        backend.script = ("Let me check.\n" + CALL).getBytes(StandardCharsets.UTF_8);
        Map<String, Object> resp = chat(body(""));
        Map<String, Object> fn = onlyCall(resp);
        assertEquals("get_weather", fn.get("name"));
        Map<String, Object> args = Json.parseObject((String) fn.get("arguments"));
        assertEquals(Map.of("city", "Paris", "days", 3L), args, Json.write(resp));
        Map<String, Object> usage = Json.map(resp, "usage");
        assertTrue(Json.num(usage, "forced_tokens", 0) > 0, "jump-forward appended tokens: " + Json.write(usage));
    }

    @Test
    void requiredAndNamedToolChoice() throws Exception {
        backend.script = CALL.getBytes(StandardCharsets.UTF_8);
        Map<String, Object> fn = onlyCall(chat(body(", \"tool_choice\": \"required\"")));
        assertEquals(Map.of("city", "Paris", "days", 3L), Json.parseObject((String) fn.get("arguments")));
        // a named function: its name and the head of the call are forced (jump-forward)
        backend.script = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\"}}\n</tool_call>"
                .getBytes(StandardCharsets.UTF_8);
        Map<String, Object> named = onlyCall(chat(body(", \"tool_choice\": {\"type\": \"function\", \"function\": {\"name\": \"read_file\"}}")));
        assertEquals("read_file", named.get("name"));
        assertTrue(Json.parseObject((String) named.get("arguments")).get("path") instanceof String, Json.write(named));
        MvcResult bad = mvc.perform(post("/v1/chat/completions").contentType("application/json")
                .content(body(", \"tool_choice\": {\"type\": \"function\", \"function\": {\"name\": \"nope\"}}"))).andReturn();
        assertEquals(400, bad.getResponse().getStatus());
    }

    @Test
    void responseFormatWithTools() throws Exception {
        backend.script = CALL.getBytes(StandardCharsets.UTF_8);
        String fmt = ", \"response_format\": {\"type\": \"json_schema\", \"json_schema\": {\"name\": \"a\", \"schema\":"
                + " {\"type\": \"object\", \"properties\": {\"answer\": {\"type\": \"string\"}}, \"required\": [\"answer\"]}}}";
        Map<String, Object> fn = onlyCall(chat(body(fmt)));
        assertEquals("get_weather", fn.get("name"));
        backend.script = "{\"answer\": \"sunny\"}".getBytes(StandardCharsets.UTF_8);
        Map<String, Object> resp = chat(body(fmt));
        Map<String, Object> choice = Json.map(Json.list(resp, "choices").getFirst(), "message");
        assertEquals(Map.of("answer", "sunny"), Json.parseObject((String) choice.get("content")), Json.write(resp));
    }
}
