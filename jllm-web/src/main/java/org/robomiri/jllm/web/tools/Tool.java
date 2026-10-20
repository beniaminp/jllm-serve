package org.robomiri.jllm.web.tools;

import java.util.Map;

/** A function the model can call: a built-in one or a tool of a connected MCP server. */
public interface Tool {

    /** The function name the model uses ({@code [A-Za-z0-9_-]{1,64}}). */
    String name();

    String description();

    /** JSON schema of the arguments object. */
    Map<String, Object> parameters();

    /** {@code builtin}, or the MCP server's name. */
    String source();

    /** Runs the tool; the text goes back to the model. Throws with a message the model can read on failure. */
    String call(Map<String, Object> arguments) throws Exception;

    /** The OpenAI {@code tools} entry. */
    default Map<String, Object> spec() {
        return org.robomiri.jllm.json.Json.obj("type", "function", "function", org.robomiri.jllm.json.Json.obj(
                "name", name(), "description", description(), "parameters", parameters()));
    }
}
