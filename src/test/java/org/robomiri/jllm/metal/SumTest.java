package org.robomiri.jllm.metal;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

/** Which add tree does simd_sum use? Compares it with explicit shuffle trees on random data. */
public final class SumTest {
    public static void main(String[] args) {
        Metal m = Metal.create(args[0], "");
        int groups = 4096;
        Metal.Buffer in = m.newBuffer(groups * 32 * 4L);
        Metal.Buffer out = m.newBuffer(groups * 4 * 4L);
        Random r = new Random(1);
        for (int i = 0; i < groups * 32; i++) {
            in.contents().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) (r.nextGaussian() * Math.exp(r.nextGaussian())));
        }
        try (Arena a = Arena.ofConfined()) {
            MemorySegment bufs = a.allocate(ValueLayout.ADDRESS, 2);
            MemorySegment offs = a.allocate(ValueLayout.JAVA_LONG, 2);
            bufs.setAtIndex(ValueLayout.ADDRESS, 0, in.handle());
            bufs.setAtIndex(ValueLayout.ADDRESS, 1, out.handle());
            m.begin();
            m.dispatch(m.pipeline("zz_sumtest"), bufs, offs, 2, a.allocate(16), 0, groups, 1, 1, 32, 1, 1);
            m.commitAndWait();
        }
        int[] diff = new int[3];
        for (int g = 0; g < groups; g++) {
            float s = out.contents().getAtIndex(ValueLayout.JAVA_FLOAT, g * 4);
            for (int k = 0; k < 3; k++) {
                if (Float.floatToRawIntBits(s) != Float.floatToRawIntBits(out.contents().getAtIndex(ValueLayout.JAVA_FLOAT, g * 4 + 1 + k))) {
                    diff[k]++;
                }
            }
        }
        System.out.println("simd_sum differs from xor16..1: " + diff[0] + ", xor1..16: " + diff[1] + ", down16..1: " + diff[2] + " of " + groups);
        m.close();
    }
}
