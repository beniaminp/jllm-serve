package org.robomiri.jllm.diffusion;

import java.util.ArrayList;
import java.util.List;

/** Reads named tensors of a checkpoint onto an {@link Ops} device; remembers what it loaded so it can free it. */
final class Loader {

    final Ops ops;
    final Checkpoint ckpt;
    private final List<Tn> loaded = new ArrayList<>();

    Loader(Ops ops, Checkpoint ckpt) {
        this.ops = ops;
        this.ckpt = ckpt;
    }

    boolean has(String name) {
        return ckpt.has(name);
    }

    Tn vec(String name) {
        return put(ops.weight(ckpt.get(name), Ops.Kind.VECTOR));
    }

    Tn lin(String name) {
        return put(ops.weight(ckpt.get(name), Ops.Kind.LINEAR));
    }

    Tn conv(String name) {
        return put(ops.weight(ckpt.get(name), Ops.Kind.CONV));
    }

    /** The vector if the checkpoint has it, else null (a layer without bias). */
    Tn optVec(String name) {
        return ckpt.has(name) ? vec(name) : null;
    }

    /** Takes over a tensor this loader did not read from the checkpoint, to free it with the others. */
    Tn adopt(Tn t) {
        return put(t);
    }

    private Tn put(Tn t) {
        loaded.add(t);
        return t;
    }

    void free() {
        for (Tn t : loaded) {
            ops.free(t);
        }
        loaded.clear();
    }
}
