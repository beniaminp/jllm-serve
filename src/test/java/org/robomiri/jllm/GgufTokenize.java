package org.robomiri.jllm;

import org.robomiri.jllm.gguf.GgufFile;
import org.robomiri.jllm.gguf.GgufModel;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Prints the token ids of the GGUF file's own tokenizer for each prompt file, in {@code llama-tokenize --ids --no-bos}'s
 * format, so the two can be diffed: {@code GgufTokenize model.gguf prompt1.txt prompt2.txt ...}. Only the metadata of
 * the file is read (no weights).
 */
public final class GgufTokenize {

    public static void main(String[] args) throws Exception {
        try (GgufFile g = GgufFile.open(Path.of(args[0]))) {
            Tokenizer tok = GgufModel.tokenizer(g);
            for (int i = 1; i < args.length; i++) {
                String text = Files.readString(Path.of(args[i]));
                System.out.println(Arrays.toString(tok.encode(text)));
            }
        }
    }
}
