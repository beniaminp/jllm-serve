package org.robomiri.jllm;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.tokenizer.ChatTemplate;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Renders conversations with a built-in chat template and prints the prompts as a JSON array, to diff with another
 * implementation's (llama.cpp's {@code /apply-template}): {@code ChatRender model_type cases.json}, where the file is
 * {@code [{"messages": [{"role", "content", "reasoning_content"?}], "think": true}]}. An assistant message's
 * {@code reasoning_content} is written into its content as {@code <think>reasoning</think>content}, the way the engine
 * keeps it.
 */
public final class ChatRender {

    public static void main(String[] args) throws Exception {
        ChatTemplate.Format f = ChatTemplate.forModelType(args[0]);
        List<Object> cases = Json.list(Map.of("c", Json.parse(Files.readString(Path.of(args[1])))), "c");
        List<Object> out = new ArrayList<>();
        for (Object o : cases) {
            List<ChatTemplate.Message> msgs = new ArrayList<>();
            for (Object m : Json.list(o, "messages")) {
                String content = Json.str(m, "content", "");
                String reasoning = Json.str(m, "reasoning_content", null);
                if (reasoning != null) {
                    content = "<think>" + reasoning + "</think>" + content;
                }
                msgs.add(new ChatTemplate.Message(Json.str(m, "role", ""), content));
            }
            out.add(f.render(msgs, true, Json.bool(o, "think", true)));
        }
        System.out.println(Json.write(out));
    }
}
