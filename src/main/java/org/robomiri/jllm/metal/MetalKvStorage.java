package org.robomiri.jllm.metal;

import org.robomiri.jllm.kv.KvStorage;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.KvDtype;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

/**
 * The Metal backend's block memory (docs/24 §24.5): per layer one K buffer and one V buffer of {@code maxBlocks * B}
 * rows, row {@code r} being exactly the flat cache's row (f32, half, or q8 int8 values followed by the per-head
 * scales; {@link KvDtype#rowBytes}). A block is {@code B} consecutive rows, so block {@code b}'s position {@code t}
 * is row {@code b * B + t} and the kernels reach it through the sequence's block table (<code>kv_row</code> in
 * kernels.metal). The buffers are shared memory created uninitialised: pages are committed as blocks are first
 * written, which is what "allocated as a conversation grows" means on a unified-memory GPU (the address range is
 * reserved up front, as the flat cache's was). They belong to a Metal context of their own, so a slot closing does not
 * free what the others use; every slot's encoder binds them.
 */
public final class MetalKvStorage implements KvStorage {

    private final Metal metal;
    private final int layers;
    private final KvDtype dtype;
    private final int kvHeads;
    private final int headDim;
    private final int B;
    private final long rowBytes;
    private final int maxBlocks;
    final Metal.Buffer[] keys;
    final Metal.Buffer[] vals;

    MetalKvStorage(String metalLib, KvDtype dtype, int layers, Attention.Gqa a, int blockTokens, int maxBlocks) {
        if (dtype == KvDtype.Q4) {
            throw new IllegalArgumentException("the paged Metal cache has no q4 blocks");
        }
        this.metal = Metal.create(metalLib);
        this.layers = layers;
        this.dtype = dtype;
        this.kvHeads = a.kvHeads();
        this.headDim = a.headDim();
        this.B = blockTokens;
        this.rowBytes = dtype.rowBytes(a.kvDim(), a.kvHeads());
        this.maxBlocks = maxBlocks;
        long bytes = (long) maxBlocks * blockTokens * rowBytes;
        keys = new Metal.Buffer[layers];
        vals = new Metal.Buffer[layers];
        try {
            for (int l = 0; l < layers; l++) {
                keys[l] = metal.newBuffer(bytes, false);
                vals[l] = metal.newBuffer(bytes, false);
            }
        } catch (RuntimeException | Error e) {
            metal.close();
            throw e;
        }
    }

    @Override
    public int blockTokens() {
        return B;
    }

    @Override
    public int spillBytes() {
        long n = (long) layers * B * 2 * rowBytes;
        if (n > Integer.MAX_VALUE) {
            throw new IllegalStateException("a block of " + n + " bytes is too large to spill; lower backend.kv-block");
        }
        return (int) n;
    }

    @Override
    public org.robomiri.jllm.kv.KvBlockCodec codec(org.robomiri.jllm.kv.KvBlockCodec.Format f) {
        return new org.robomiri.jllm.kv.KvBlockCodec(f, dtype, layers, B, kvHeads, headDim);
    }

    @Override
    public long residentBytes() {
        return (long) layers * B * 2 * rowBytes;
    }

    @Override
    public void grow(int blocks) {
        if (blocks > maxBlocks) {
            throw new IllegalStateException("block " + blocks + " is past the pool buffers (" + maxBlocks + ")");
        }
        committedBlocks = Math.max(committedBlocks, blocks);
    }

    /** The bytes of row {@code t} of block {@code block} in {@code seg} (a layer's K or V buffer contents). */
    private long at(int block, int t) {
        return ((long) block * B + t) * rowBytes;
    }

    @Override
    public void copy(int src, int dst, int rows) {
        long n = (long) B * rowBytes; // the whole block: the tail past `rows` is overwritten before it is read
        for (int l = 0; l < layers; l++) {
            MemorySegment.copy(keys[l].contents(), at(src, 0), keys[l].contents(), at(dst, 0), n);
            MemorySegment.copy(vals[l].contents(), at(src, 0), vals[l].contents(), at(dst, 0), n);
        }
    }

    @Override
    public void save(int block, ByteBuffer dst) {
        MemorySegment out = MemorySegment.ofBuffer(dst);
        long o = 0; // ofBuffer's segment starts at the buffer's position
        for (int l = 0; l < layers; l++) {
            for (int t = 0; t < B; t++) {
                MemorySegment.copy(keys[l].contents(), at(block, t), out, o, rowBytes);
                o += rowBytes;
                MemorySegment.copy(vals[l].contents(), at(block, t), out, o, rowBytes);
                o += rowBytes;
            }
        }
        dst.position(dst.position() + spillBytes());
    }

    @Override
    public void load(int block, ByteBuffer src) {
        MemorySegment in = MemorySegment.ofBuffer(src);
        long o = 0;
        for (int l = 0; l < layers; l++) {
            for (int t = 0; t < B; t++) {
                MemorySegment.copy(in, o, keys[l].contents(), at(block, t), rowBytes);
                o += rowBytes;
                MemorySegment.copy(in, o, vals[l].contents(), at(block, t), rowBytes);
                o += rowBytes;
            }
        }
        src.position(src.position() + spillBytes());
    }

    private DeviceLedger.Resident ledger;

    /**
     * Counts the pool in the {@link DeviceLedger}: its buffers belong to a Metal context of their own, so no backend's own
     * buffers include them, and a model planning its GPU memory (the MoE expert placement, the memory planner) would
     * otherwise not see the KV blocks at all. What is counted is the pool's hot cap (the blocks it may commit), the
     * budget the GPU has to keep for it.
     */
    void register(org.robomiri.jllm.model.ModelWeights w, org.robomiri.jllm.kv.KvPool pool) {
        ledger = new DeviceLedger.Resident() {
            @Override
            public org.robomiri.jllm.model.ModelWeights weights() {
                return w;
            }

            @Override
            public long gpuWeightBytes() {
                return 0;
            }

            @Override
            public long ownBytes() {
                return (long) pool.hotCap() * residentBytes();
            }
        };
        DeviceLedger.add(ledger);
    }

    /** Bytes the pool's committed blocks occupy now (the high-water mark; pages of free blocks stay committed). */
    long committedBytes() {
        return (long) committedBlocks * residentBytes();
    }

    private volatile int committedBlocks;

    @Override
    public void close() {
        if (ledger != null) {
            DeviceLedger.remove(ledger);
            ledger = null;
        }
        metal.close();
    }
}
