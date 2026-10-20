package org.robomiri.jllm.web.chat;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ReasoningSplitterTest {

    @Test
    void wholeText() {
        assertArrayEquals(new String[]{"plan it", "The answer."}, ReasoningSplitter.split("<think>\nplan it\n</think>\n\nThe answer.", false));
        assertArrayEquals(new String[]{"", "No block here <think>x</think>"}, ReasoningSplitter.split("No block here <think>x</think>", false));
        assertArrayEquals(new String[]{"", "Answer."}, ReasoningSplitter.split("<think>\n\n</think>\n\nAnswer.", false));
        // DeepSeek-R1: the prompt opened the block
        assertArrayEquals(new String[]{"hmm", "Done"}, ReasoningSplitter.split("hmm</think>Done", true));
        // still thinking when generation stopped
        assertArrayEquals(new String[]{"half a thou", ""}, ReasoningSplitter.split("<think>half a thou", false));
    }

    @Test
    void tagsSplitAcrossPieces() {
        StringBuilder r = new StringBuilder();
        StringBuilder c = new StringBuilder();
        ReasoningSplitter s = new ReasoningSplitter(r::append, c::append, false);
        for (String piece : new String[]{"<th", "ink>", "step 1 ", "< b", "</thi", "nk>", "\n\n", "Hel", "lo <think>"}) {
            s.accept(piece);
        }
        s.finish();
        assertEquals("step 1 < b", r.toString());
        assertEquals("Hello <think>", c.toString());
    }
}
