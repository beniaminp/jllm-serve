package org.robomiri.jllm.web;

import org.junit.jupiter.api.Test;
import org.robomiri.jllm.web.models.HfLink;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HfLinkTest {

    @Test
    void repositoryForms() {
        for (String s : new String[]{"Qwen/Qwen3-0.6B", "https://huggingface.co/Qwen/Qwen3-0.6B", "huggingface.co/Qwen/Qwen3-0.6B/",
                "https://hf.co/Qwen/Qwen3-0.6B", "https://huggingface.co/Qwen/Qwen3-0.6B?library=transformers",
                "https://huggingface.co/models/Qwen/Qwen3-0.6B"}) {
            HfLink l = HfLink.parse(s);
            assertEquals("Qwen/Qwen3-0.6B", l.repo(), s);
            assertNull(l.revision(), s);
            assertNull(l.file(), s);
            assertNull(l.quant(), s);
            assertFalse(l.local(), s);
            assertEquals("Qwen/Qwen3-0.6B", l.defaultId());
        }
    }

    @Test
    void revisionAndFile() {
        HfLink tree = HfLink.parse("https://huggingface.co/Qwen/Qwen3-0.6B/tree/refs%2Fpr%2F3");
        assertEquals("refs/pr/3", tree.revision());

        HfLink blob = HfLink.parse("https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q4_K_M.gguf");
        assertEquals("unsloth/Qwen3-0.6B-GGUF", blob.repo());
        assertEquals("main", blob.revision());
        assertEquals("Qwen3-0.6B-Q4_K_M.gguf", blob.file());
        assertEquals("unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M", blob.defaultId());

        HfLink resolve = HfLink.parse("https://huggingface.co/a/b-GGUF/resolve/v1/sub/dir/x-Q8_0.gguf?download=true");
        assertEquals("sub/dir/x-Q8_0.gguf", resolve.file());
        assertEquals("v1", resolve.revision());

        HfLink config = HfLink.parse("https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json");
        assertNull(config.file()); // the whole checkpoint
    }

    @Test
    void quantLabelAndLocal() {
        HfLink q = HfLink.parse("hf.co/unsloth/Qwen3-0.6B-GGUF:Q4_K_M");
        assertEquals("unsloth/Qwen3-0.6B-GGUF", q.repo());
        assertEquals("Q4_K_M", q.quant());
        assertEquals("unsloth/Qwen3-0.6B-GGUF:Q4_K_M", q.defaultId());

        HfLink local = HfLink.parse("/models/Llama-3.2-1B-Q8_0.gguf");
        assertTrue(local.local());
        assertEquals("Llama-3.2-1B-Q8_0", local.defaultId());
    }

    @Test
    void rejects() {
        assertThrows(IllegalArgumentException.class, () -> HfLink.parse(""));
        assertThrows(IllegalArgumentException.class, () -> HfLink.parse("just-a-name"));
        assertThrows(IllegalArgumentException.class, () -> HfLink.parse("https://huggingface.co/datasets/a/b"));
        assertThrows(IllegalArgumentException.class, () -> HfLink.parse("https://huggingface.co/a/b/discussions/1"));
        assertThrows(IllegalArgumentException.class, () -> HfLink.parse("a/../b"));
    }
}
