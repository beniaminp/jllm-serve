package org.robomiri.jllm.kv;

import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.KvQ4;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * How the block files of the SSD tier store the rows of a block (docs/26 §26.2.4): the pool's own format as it is
 * ({@link Format#NATIVE}), or, for an f32 or f16 pool, <b>q8</b> (int8 values and an f32 scale per head, the rows of
 * {@code backend.kv-dtype=q8}) or <b>q4</b> (nibbles and a half scale per group of 32, the rows of q4): a quarter or
 * an eighth of the f32 bytes. A block is the canonical payload of {@link KvStorage#save}: for every layer, for every
 * position, the K row then the V row of {@code kvDim} values.
 *
 * <p>Both lossy formats rotate each head's row with the Walsh–Hadamard transform of {@link KvQ4#rotate} before
 * quantizing (and rotate again, divided by the block size, after dequantizing): the rotation spreads a few large
 * channels over the whole head, which is what lets 8 or 4 bits suffice (docs/07 §7.13). The rotation is applied to
 * the values too, not only to the keys: they are stored in this format only to be read back into the pool, so nothing
 * has to be rotated at attention time, and a rotated, quantized value row has the same error as a key row.
 *
 * <p>A pool that already stores q8 (rotated keys) is stored as it is: re-quantizing would lose more for no saving.
 */
public final class KvBlockCodec {

    /** The on-disk row format. */
    public enum Format {
        NATIVE, Q8, Q4;

        public static Format parse(String s) {
            return switch (s.trim().toLowerCase()) {
                case "native", "off", "" -> NATIVE;
                case "q8", "int8" -> Q8;
                case "q4", "int4" -> Q4;
                default -> throw new IllegalArgumentException("must be native, q8 or q4, got '" + s + "'");
            };
        }

        @Override
        public String toString() {
            return name().toLowerCase();
        }
    }

    private final int layers;
    private final int B;
    private final int kvHeads;
    private final int headDim;
    private final int kvDim;
    private final KvDtype src;
    private final Format format;
    private final long srcRow;
    private final long dstRow;

    /**
     * @param src the pool's row type (f32 or f16; anything else is stored natively)
     */
    public KvBlockCodec(Format format, KvDtype src, int layers, int blockTokens, int kvHeads, int headDim) {
        this.layers = layers;
        this.B = blockTokens;
        this.kvHeads = kvHeads;
        this.headDim = headDim;
        this.kvDim = kvHeads * headDim;
        this.src = src;
        this.srcRow = src.rowBytes(kvDim, kvHeads);
        boolean usable = (src == KvDtype.F32 || src == KvDtype.F16) && KvQ4.supports(headDim) && format != Format.NATIVE;
        this.format = usable ? format : Format.NATIVE;
        this.dstRow = switch (this.format) {
            case NATIVE -> srcRow;
            case Q8 -> KvDtype.Q8.rowBytes(kvDim, kvHeads);
            case Q4 -> KvDtype.Q4.rowBytes(kvDim, kvHeads);
        };
    }

    /** The format actually used (native when the pool's type or head width does not allow the asked one). */
    public Format format() {
        return format;
    }

    public boolean identity() {
        return format == Format.NATIVE;
    }

    /** Bytes of one encoded block. */
    public int encodedBytes() {
        return (int) ((long) layers * B * 2 * dstRow);
    }

    /** Bytes of one block in the pool's format. */
    public int nativeBytes() {
        return (int) ((long) layers * B * 2 * srcRow);
    }

    /** A description for the file fingerprint: blocks of another format are never read as this one. */
    public String describe() {
        return format.toString();
    }

    /** Encodes {@code nativeBytes()} bytes from {@code in}'s position into {@code out} (both advance). */
    public void encode(ByteBuffer in, ByteBuffer out) {
        ByteBuffer s = in.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer d = out.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        if (format == Format.NATIVE) {
            s.limit(s.position() + nativeBytes());
            d.put(s);
        } else {
            float[] row = new float[kvDim];
            byte[] nib = new byte[kvDim / 2];
            short[] sc = new short[kvDim / KvQ4.GROUP];
            byte[] q = new byte[kvDim];
            float[] scales = new float[kvHeads];
            for (int i = 0; i < layers * B * 2; i++) {
                readRow(s, row);
                for (int h = 0; h < kvHeads; h++) {
                    KvQ4.rotate(row, h * headDim, headDim);
                }
                if (format == Format.Q8) { // int8 values, then one f32 scale per head
                    for (int h = 0; h < kvHeads; h++) {
                        float max = 0;
                        for (int j = 0; j < headDim; j++) {
                            max = Math.max(max, Math.abs(row[h * headDim + j]));
                        }
                        float scale = KvDtype.q8Scale(max);
                        scales[h] = scale;
                        for (int j = 0; j < headDim; j++) {
                            q[h * headDim + j] = KvDtype.q8(row[h * headDim + j], scale);
                        }
                    }
                    d.put(q);
                    for (float v : scales) {
                        d.putFloat(v);
                    }
                } else { // nibbles, then one half scale per group of 32
                    KvQ4.quantize(row, 0, kvDim, nib, 0, sc, 0);
                    d.put(nib);
                    for (short v : sc) {
                        d.putShort(v);
                    }
                }
            }
        }
        in.position(in.position() + nativeBytes());
        out.position(out.position() + encodedBytes());
    }

    /** The inverse of {@link #encode}: {@code encodedBytes()} bytes from {@code in} into {@code out}, in the pool's format. */
    public void decode(ByteBuffer in, ByteBuffer out) {
        ByteBuffer s = in.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer d = out.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        if (format == Format.NATIVE) {
            s.limit(s.position() + encodedBytes());
            d.put(s);
        } else {
            float[] row = new float[kvDim];
            byte[] nib = new byte[kvDim / 2];
            short[] sc = new short[kvDim / KvQ4.GROUP];
            byte[] q = new byte[kvDim];
            float[] scales = new float[kvHeads];
            float inv = 1f / KvQ4.block(headDim);
            for (int i = 0; i < layers * B * 2; i++) {
                if (format == Format.Q8) {
                    s.get(q);
                    for (int h = 0; h < kvHeads; h++) {
                        scales[h] = s.getFloat();
                    }
                    for (int h = 0; h < kvHeads; h++) {
                        for (int j = 0; j < headDim; j++) {
                            row[h * headDim + j] = q[h * headDim + j] * scales[h];
                        }
                    }
                } else {
                    s.get(nib);
                    for (int g = 0; g < sc.length; g++) {
                        sc[g] = s.getShort();
                    }
                    KvQ4.dequantize(nib, 0, sc, 0, kvDim, row, 0);
                }
                for (int h = 0; h < kvHeads; h++) { // H·H = block·I: rotate again and divide
                    KvQ4.rotate(row, h * headDim, headDim);
                }
                for (int j = 0; j < kvDim; j++) {
                    row[j] *= inv;
                }
                writeRow(d, row);
            }
        }
        in.position(in.position() + encodedBytes());
        out.position(out.position() + nativeBytes());
    }

    private void readRow(ByteBuffer s, float[] row) {
        if (src == KvDtype.F32) {
            for (int j = 0; j < kvDim; j++) {
                row[j] = s.getFloat();
            }
        } else {
            for (int j = 0; j < kvDim; j++) {
                row[j] = Float.float16ToFloat(s.getShort());
            }
        }
    }

    private void writeRow(ByteBuffer d, float[] row) {
        if (src == KvDtype.F32) {
            for (int j = 0; j < kvDim; j++) {
                d.putFloat(row[j]);
            }
        } else {
            for (int j = 0; j < kvDim; j++) {
                d.putShort(Float.floatToFloat16(row[j]));
            }
        }
    }
}
