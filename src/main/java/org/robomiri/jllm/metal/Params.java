package org.robomiri.jllm.metal;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Typed writer for a kernel's parameter struct: {@code p.reset(layout).i(..).f(..).done()} writes the fields in
 * declaration order into the native scratch and returns the struct size to pass as {@code plen}. {@link #done()}
 * checks that exactly the fields of the {@link KernelParams} layout were written, with the right kinds, so a
 * writer that drifts from its struct fails at the first dispatch instead of feeding the kernel shifted bytes.
 */
final class Params {

    private final MemorySegment seg;
    private KernelParams layout;
    private int count;

    Params(MemorySegment seg) {
        this.seg = seg;
    }

    MemorySegment segment() {
        return seg;
    }

    Params reset(KernelParams layout) {
        this.layout = layout;
        this.count = 0;
        return this;
    }

    Params i(int v) {
        check('i');
        seg.setAtIndex(ValueLayout.JAVA_INT, count++, v);
        return this;
    }

    Params i(boolean flag) {
        return i(flag ? 1 : 0);
    }

    Params f(float v) {
        check('f');
        seg.setAtIndex(ValueLayout.JAVA_FLOAT, count++, v);
        return this;
    }

    private void check(char kind) {
        if (count >= layout.kinds.length() || layout.kinds.charAt(count) != kind) {
            throw new IllegalStateException(layout.struct + ": field " + count + " written as " + kind
                    + ", layout is " + layout.kinds);
        }
    }

    /** @return the struct size in bytes ({@code plen}) */
    int done() {
        if (count != layout.kinds.length()) {
            throw new IllegalStateException(layout.struct + ": " + count + " of " + layout.kinds.length() + " fields written");
        }
        return layout.bytes();
    }
}
