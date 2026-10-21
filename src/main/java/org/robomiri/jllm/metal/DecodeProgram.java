package org.robomiri.jllm.metal;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

/**
 * One recorded single-token decode step: the arguments of every dispatch of the step, packed the way {@code jm_run}
 * reads them ({@code JMCmd} in jllm_metal.m and the emulator's runtime.cpp: 384 bytes per command), plus the
 * <em>patches</em> that make it valid for another step: the fields that depend on the step's {@link #VAR_POS position}
 * (rope / attention positions, the KV cache rows written at the position) or on the token ring slots of pipelined
 * decoding ({@link #VAR_SLOT_IN}, {@link #VAR_SLOT_OUT}). Each patch is {@code value = base + coeff * var}: a parameter
 * word ({@link #KIND_PARAM}, a 32-bit int) or a buffer binding's byte offset ({@link #KIND_OFFSET}).
 *
 * <p>{@link MetalKernels#step} records a program the first time a step of a given shape runs (the dispatches are
 * appended here instead of crossing to the native side one by one), runs it with one {@code jm_run} call, and for
 * later steps of the same shape only applies the patches and calls {@code jm_run} again.
 */
final class DecodeProgram {

    static final int CMD_BYTES = 384;
    static final int OFF_PIPE = 0;
    static final int OFF_NBUF = 4;
    static final int OFF_PLEN = 8;
    static final int OFF_TGMEM = 12;
    static final int OFF_GRID = 16;
    static final int OFF_THREADS = 28;
    static final int OFF_BUFS = 48;
    static final int OFF_OFFS = 176;
    static final int OFF_PARAMS = 304;
    static final int MAX_BUFS = 16;
    static final int MAX_PARAMS = 64;

    /** The step's position: rope / attention positions, the KV cache rows written. */
    static final int VAR_POS = 0;
    /** Pipelined decoding: the ring slot embed_token reads the step's token from. */
    static final int VAR_SLOT_IN = 1;
    /** Pipelined decoding: the ring slot argmax writes the next token to. */
    static final int VAR_SLOT_OUT = 2;
    static final int VARS = 3;

    static final int KIND_PARAM = 0;
    static final int KIND_OFFSET = 1;

    private final Arena arena = Arena.ofAuto();
    private MemorySegment cmds;
    private int n;
    private final List<String> names = new ArrayList<>();
    // patches, one entry per index: command, kind, field / binding index, variable, base, coefficient
    private int[] pCmd = new int[16];
    private int[] pKind = new int[16];
    private int[] pIndex = new int[16];
    private int[] pVar = new int[16];
    private long[] pBase = new long[16];
    private long[] pCoeff = new long[16];
    private int np;
    /** A later step's fresh recording was compared with this one patched for it, and they were equal. */
    boolean verified;

    DecodeProgram() {
        cmds = arena.allocate((long) CMD_BYTES * 256, 16);
    }

    /** The packed commands ({@link #size()} of them). */
    MemorySegment commands() {
        return cmds;
    }

    int size() {
        return n;
    }

    int patches() {
        return np;
    }

    String kernel(int cmd) {
        return names.get(cmd);
    }

    /** Appends one dispatch; returns its index. {@code bufs} / {@code offs} hold at least {@code nbuf} entries. */
    int add(int pipe, String name, MemorySegment bufs, MemorySegment offs, int nbuf, MemorySegment params, int plen,
            int tgmem, int gx, int gy, int gz, int tx, int ty, int tz) {
        if (nbuf > MAX_BUFS || plen > MAX_PARAMS) {
            throw new IllegalStateException(name + ": " + nbuf + " buffers / " + plen + " parameter bytes do not fit a command");
        }
        if ((long) (n + 1) * CMD_BYTES > cmds.byteSize()) {
            MemorySegment bigger = arena.allocate(cmds.byteSize() * 2, 16);
            MemorySegment.copy(cmds, 0, bigger, 0, cmds.byteSize());
            cmds = bigger;
        }
        long b = (long) n * CMD_BYTES;
        cmds.set(ValueLayout.JAVA_INT, b + OFF_PIPE, pipe);
        cmds.set(ValueLayout.JAVA_INT, b + OFF_NBUF, nbuf);
        cmds.set(ValueLayout.JAVA_INT, b + OFF_PLEN, plen);
        cmds.set(ValueLayout.JAVA_INT, b + OFF_TGMEM, tgmem);
        cmds.set(ValueLayout.JAVA_INT, b + OFF_GRID, gx);
        cmds.set(ValueLayout.JAVA_INT, b + OFF_GRID + 4, gy);
        cmds.set(ValueLayout.JAVA_INT, b + OFF_GRID + 8, gz);
        cmds.set(ValueLayout.JAVA_INT, b + OFF_THREADS, tx);
        cmds.set(ValueLayout.JAVA_INT, b + OFF_THREADS + 4, ty);
        cmds.set(ValueLayout.JAVA_INT, b + OFF_THREADS + 8, tz);
        MemorySegment.copy(bufs, 0, cmds, b + OFF_BUFS, (long) nbuf * 8);
        MemorySegment.copy(offs, 0, cmds, b + OFF_OFFS, (long) nbuf * 8);
        if (plen > 0) {
            MemorySegment.copy(params, 0, cmds, b + OFF_PARAMS, plen);
        }
        names.add(name);
        return n++;
    }

    /** Field {@code index} of command {@code cmd} (a parameter word, or a binding's offset) is {@code base + coeff * var}. */
    void patch(int cmd, int kind, int index, int var, long base, long coeff) {
        if (np == pCmd.length) {
            int cap = np * 2;
            pCmd = java.util.Arrays.copyOf(pCmd, cap);
            pKind = java.util.Arrays.copyOf(pKind, cap);
            pIndex = java.util.Arrays.copyOf(pIndex, cap);
            pVar = java.util.Arrays.copyOf(pVar, cap);
            pBase = java.util.Arrays.copyOf(pBase, cap);
            pCoeff = java.util.Arrays.copyOf(pCoeff, cap);
        }
        pCmd[np] = cmd;
        pKind[np] = kind;
        pIndex[np] = index;
        pVar[np] = var;
        pBase[np] = base;
        pCoeff[np] = coeff;
        np++;
    }

    /** Rewrites every patched field for a step with the variables {@code vars} ({@link #VARS} of them). */
    void apply(int[] vars) {
        for (int i = 0; i < np; i++) {
            long v = pBase[i] + pCoeff[i] * vars[pVar[i]];
            long b = (long) pCmd[i] * CMD_BYTES;
            if (pKind[i] == KIND_PARAM) {
                cmds.set(ValueLayout.JAVA_INT, b + OFF_PARAMS + 4L * pIndex[i], (int) v);
            } else {
                cmds.set(ValueLayout.JAVA_LONG, b + OFF_OFFS + 8L * pIndex[i], v);
            }
        }
    }

    /**
     * The first command whose bytes differ from {@code o}'s (pipeline, bindings, offsets, parameters, grid), {@code size()}
     * if one program is a prefix of the other, or -1 when they are the same.
     */
    int firstDifference(DecodeProgram o) {
        int m = Math.min(n, o.n);
        for (int i = 0; i < m; i++) {
            long b = (long) i * CMD_BYTES;
            if (MemorySegment.mismatch(cmds, b, b + CMD_BYTES, o.cmds, b, b + CMD_BYTES) >= 0) {
                return i;
            }
        }
        return n == o.n ? -1 : m;
    }
}
