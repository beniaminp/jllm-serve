package org.robomiri.jllm.kv;

/** Thrown when every block of a {@link KvPool} is referenced by a sequence and none can be evicted. */
public final class KvPoolExhausted extends IllegalStateException {
    public KvPoolExhausted(String message) {
        super(message);
    }
}
