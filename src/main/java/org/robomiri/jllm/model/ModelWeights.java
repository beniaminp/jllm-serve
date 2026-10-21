package org.robomiri.jllm.model;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Quantized weights of a model, memory-mapped from a {@link QuantFile}: the embedding, one {@link LayerWeights}
 * per layer loaded for exactly what the {@link Architecture}'s layer spec asks, the final norm and the LM head.
 *
 * <p>Linear layers use the requested format (Q4_0 by default); the embedding / LM head always uses Q8_0 because
 * small models are sensitive to output-layer precision. Gate and up projections are fused into one matrix; in MoE
 * layers all experts are stacked into one gate/up tensor and one down tensor.
 */
public final class ModelWeights implements AutoCloseable {

    public final Architecture arch;
    public final QType quant;
    public final QTensor embed;
    public final QTensor lmHead;
    public final LayerWeights[] layers;
    public final float[] finalNorm;
    private final TensorSource file;
    private final AutoCloseable owner;
    private final long bytes;

    private ModelWeights(Architecture arch, QType quant, TensorSource file, AutoCloseable owner, long bytes) {
        this.arch = arch;
        this.quant = quant;
        this.file = file;
        this.owner = owner;
        this.bytes = bytes;
        Architecture.Core c = arch.core();
        boolean mamba = arch instanceof MambaArch;
        embed = file.tensor(mamba ? "backbone.embeddings.weight" : "model.embed_tokens.weight");
        lmHead = file.has("lm_head.weight") ? file.tensor("lm_head.weight") : embed;
        // The engine sizes its logits from vocabSize and backends write lmHead.rows outputs, so both must agree.
        if (embed.cols != c.dim() || embed.rows != c.vocabSize()
                || lmHead.cols != c.dim() || lmHead.rows != c.vocabSize()) {
            throw new IllegalArgumentException("embedding " + embed.rows + "x" + embed.cols + " / lm_head "
                    + lmHead.rows + "x" + lmHead.cols + " do not match config " + arch + STALE_HINT);
        }
        layers = new LayerWeights[c.layers()];
        String finalNormName;
        switch (arch) {
            case MambaArch m -> {
                for (int l = 0; l < c.layers(); l++) {
                    layers[l] = MambaLayer.load(file, "backbone.layers." + l + ".", m);
                }
                finalNormName = "backbone.norm_f.weight";
            }
            case Qwen3NextArch q -> {
                for (int l = 0; l < c.layers(); l++) {
                    layers[l] = Qwen3NextLayer.load(file, "model.layers." + l + ".", q, l);
                }
                finalNormName = "model.norm.weight";
            }
            case Decoder d -> {
                for (int l = 0; l < c.layers(); l++) {
                    LayerSpec spec = d.layers().get(l);
                    DecoderLayer L = DecoderLayer.load(file, "model.layers." + l + ".", spec);
                    if (!L.matches(spec, c.dim())) {
                        throw new IllegalArgumentException("layer " + l + " weight shapes do not match config " + arch + STALE_HINT);
                    }
                    layers[l] = L;
                }
                finalNormName = "model.norm.weight";
            }
        }
        finalNorm = file.floats(finalNormName);
        if (finalNorm.length != c.dim()) {
            throw new IllegalArgumentException(finalNormName + " has " + finalNorm.length + " entries, expected "
                    + c.dim() + STALE_HINT);
        }
    }

    /** The architecture as a decoder; throws for a Mamba model (a transformer backend was given the wrong model). */
    public Decoder decoder() {
        if (arch instanceof Decoder d) {
            return d;
        }
        throw new IllegalArgumentException("not a decoder model: " + arch);
    }

    /** The architecture as Mamba; throws for a decoder. */
    public MambaArch mambaArch() {
        if (arch instanceof MambaArch m) {
            return m;
        }
        throw new IllegalArgumentException("not a Mamba model: " + arch);
    }

    /** Layer {@code l} of a decoder. */
    public DecoderLayer decoderLayer(int l) {
        return (DecoderLayer) layers[l]; // a decoder's layers are DecoderLayers by construction
    }

    /** The architecture as Qwen3-Next; throws for any other. */
    public Qwen3NextArch qwen3Next() {
        if (arch instanceof Qwen3NextArch q) {
            return q;
        }
        throw new IllegalArgumentException("not a Qwen3-Next model: " + arch);
    }

    /** Layer {@code l} of a Qwen3-Next model. */
    public Qwen3NextLayer qwen3NextLayer(int l) {
        return (Qwen3NextLayer) layers[l];
    }

    /** Layer {@code l} of a Mamba model. */
    public MambaLayer mambaLayer(int l) {
        return (MambaLayer) layers[l];
    }

    /** Appended to a shape-mismatch message of a converted cache (the remedy there is to reconvert). */
    public static final String STALE_HINT = " (stale cache)";

    /**
     * Weights read from any {@link TensorSource} (a GGUF file): the same loading and shape checks as for a converted
     * cache. {@code owner} is closed with the weights and releases the memory behind the tensors.
     *
     * @param quant a label for the dominant weight format (informational)
     * @param bytes the size of the file(s) behind the weights
     * @throws IllegalArgumentException if a tensor is missing or does not match {@code arch}
     */
    public static ModelWeights of(Architecture arch, QType quant, TensorSource source, AutoCloseable owner, long bytes) {
        return new ModelWeights(arch, quant, source, owner, bytes);
    }

    /** Local checkpoint: no repo / revision is recorded or checked. See {@link #load(Path, Architecture, QType, ShardProvider, String)}. */
    public static ModelWeights load(Path dir, Architecture arch, QType quant, ShardProvider shards) throws IOException {
        return load(dir, arch, quant, shards, null);
    }

    /**
     * Opens the quantized cache in {@code dir}, converting the Hugging Face checkpoint first if there is none.
     *
     * <p>A cache that exists but cannot be used — written by another {@link QuantFile#LAYOUT_VERSION}, an
     * unparseable or corrupt index, converted from another repo or revision than {@code identity}, or tensors
     * whose shapes do not match {@code arch} — is deleted and converted again. The second attempt must succeed:
     * a shape mismatch after a fresh conversion is a real inconsistency between the checkpoint and its config
     * and is reported as such.
     *
     * @param identity {@code repo@revision} the model was requested as (recorded in the cache and compared with
     *                 what an existing cache records), or null to skip the check for local files
     */
    public static ModelWeights load(Path dir, Architecture arch, QType quant, ShardProvider shards, String identity)
            throws IOException {
        return load(dir, arch, quant, QuantFile.DEFAULT_HEAD, shards, identity);
    }

    /**
     * As above with the embedding / LM head stored as {@code head} (Q8_0 by default, or Q6_K). The head format is
     * part of the cache name, so a cache converted with another head is neither read nor overwritten.
     */
    public static ModelWeights load(Path dir, Architecture arch, QType quant, QType head, ShardProvider shards,
                                    String identity) throws IOException {
        QuantFile.Identity id = QuantFile.Identity.parse(identity, null);
        boolean converted = false;
        if (!QuantFile.exists(dir, quant, head)) {
            ModelConverter.convert(dir, arch, quant, head, shards, identity);
            converted = true;
        }
        while (true) {
            long t0 = System.nanoTime();
            String stale = null;
            try {
                QuantFile file = QuantFile.open(dir, quant, head);
                try {
                    if (id != null && !id.sameAs(file.identity())) {
                        throw new QuantFile.StaleCacheException("it was converted from "
                                + (file.identity() == null ? "local files" : file.identity().name()) + ", not " + id.name());
                    }
                    ModelWeights w = new ModelWeights(arch, quant, file, file::close, file.byteSize());
                    if (w.embed.type != head) {
                        throw new QuantFile.StaleCacheException("its embedding is " + w.embed.type + ", not " + head);
                    }
                    System.err.printf("Mapped %s weights (%s, %.0f MB) in %.2fs%n", arch.core().architecture(), quant,
                            file.byteSize() / 1e6, (System.nanoTime() - t0) / 1e9);
                    return w;
                } catch (IllegalArgumentException e) {
                    // a tensor is missing or has the wrong shape: the cache was written for another checkpoint or config
                    file.close();
                    stale = e.getMessage();
                } catch (IOException | RuntimeException e) {
                    file.close();
                    throw e;
                }
            } catch (QuantFile.StaleCacheException e) {
                stale = e.getMessage();
            }
            if (converted) {
                throw new IOException("the freshly converted cache " + QuantFile.binPath(dir, quant, head)
                        + " does not match the config: " + stale);
            }
            System.err.println("Reconverting " + QuantFile.binPath(dir, quant, head).getFileName() + ": " + stale);
            QuantFile.delete(dir, quant, head);
            ModelConverter.convert(dir, arch, quant, head, shards, identity);
            converted = true;
        }
    }

    public long byteSize() {
        return bytes;
    }

    @Override
    public void close() {
        try {
            owner.close();
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }
}
