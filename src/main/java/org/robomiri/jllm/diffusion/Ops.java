package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.safetensors.SafeTensors;

/**
 * The operations the diffusion networks are built from, on one device. {@link CpuOps} is the float32 reference;
 * {@link MetalOps} runs the same operations on the GPU (float16 or float32). The text encoders, the UNet and the VAE are
 * written once against this interface.
 *
 * <p>Layout: activations are channels-last. An image batch is {@code [B, H, W, C]}, a token sequence {@code [B, L, C]}
 * (or {@code [rows, C]}); a convolution weight is {@code [Cout, kh, kw, Cin]} (diffusers stores {@code [Cout, Cin, kh, kw]},
 * {@link #weight} transposes it while loading) and a linear weight {@code [out, in]} as in the checkpoint. Biases and norm
 * scales are float32 vectors ({@link Kind#VECTOR}).
 *
 * <p>Ownership: every op returns a new tensor that the caller {@link #free frees} when it is dead (a no-op on the CPU;
 * the GPU pools its buffers). Ops never modify their inputs.
 */
public interface Ops extends AutoCloseable {

    enum Kind {
        /** {@code [out, in]}, used as {@code x · Wᵀ}. */
        LINEAR,
        /** diffusers {@code [Cout, Cin, kh, kw]} stored as {@code [Cout, kh, kw, Cin]}. */
        CONV,
        /** a float32 vector (bias, norm scale); also used for tables. */
        VECTOR
    }

    enum Act {
        NONE, SILU, GELU, QUICK_GELU
    }

    String name();

    /** True when activations and weights are float16 in memory (arithmetic is float32 inside the kernels). */
    boolean half();

    /** A new tensor with unspecified contents. */
    Tn create(int... shape);

    /** A new activation holding {@code data} (float32 on the host). */
    Tn upload(float[] data, int... shape);

    /** The contents as float32 (waits for the device). */
    float[] download(Tn t);

    /** Returns the tensor's storage to the pool; the tensor must not be used afterwards. */
    void free(Tn t);

    /** Waits until everything submitted has finished. */
    void sync();

    /** Releases pooled scratch memory beyond {@code keepBytes} (the GPU pool grows with the largest image made so far). */
    default void trim(long keepBytes) {
    }

    /** A persistent float32 vector or table (a {@link Kind#VECTOR} weight) made from host data; freed with {@link #free}. */
    Tn vector(float[] data, int... shape);

    /** Loads a checkpoint tensor as a weight of the given kind (converted to this device's format). */
    Tn weight(SafeTensors.Tensor t, Kind kind);

    // ---------------------------------------------------------------- operations

    /**
     * {@code act(x · Wᵀ + bias) + residual} for {@code x [rows, K]}, {@code w [N, K]}; {@code bias [N]} and
     * {@code residual [rows, N]} may be null. The activation is applied before the residual is added.
     */
    Tn linear(Tn x, Tn w, Tn bias, Tn residual, Act act);

    /**
     * Convolution on {@code x [B, H, W, Cin]} with {@code w [Cout, kh, kw, Cin]}:
     * {@code out = conv(x) + bias + chanBias[b, :] + residual}, with zero padding {@code padT, padL, padB, padR}.
     * {@code chanBias [B, Cout]} (the time embedding of a ResNet block) and {@code residual [B, Ho, Wo, Cout]} may be null.
     */
    Tn conv2d(Tn x, Tn w, Tn bias, int stride, int padT, int padL, int padB, int padR, Tn chanBias, Tn residual);

    /** Group normalisation over {@code [B, H, W, C]} (or {@code [B, L, C]}), optionally followed by SiLU. */
    default Tn groupNorm(Tn x, Tn gamma, Tn beta, int groups, float eps, boolean silu) {
        return groupNorm(x, gamma, beta, groups, eps, silu, x.dim(0));
    }

    /**
     * Group normalisation where the tensor is read as {@code samples} independent items of {@code rows / samples} positions
     * each: the statistics of a group are taken over all positions of one item. The video networks use it to normalise over
     * the frames of a clip too ({@code [B·F, H, W, C]} with {@code samples = B}).
     */
    Tn groupNorm(Tn x, Tn gamma, Tn beta, int groups, float eps, boolean silu, int samples);

    /** Layer normalisation over the last dimension. */
    Tn layerNorm(Tn x, Tn gamma, Tn beta, float eps);

    /**
     * Scaled dot-product attention {@code softmax(q kᵀ / sqrt(hd)) v} per batch and head: {@code q [B, lq, heads·hd]},
     * {@code k, v [B, lk, heads·hd]}; with {@code causal}, position i attends to j ≤ i.
     */
    Tn attention(Tn q, Tn k, Tn v, int heads, boolean causal);

    /** {@code x[:, :n] · gelu(x[:, n:])} for {@code x [rows, 2n]}. */
    Tn geglu(Tn x);

    /** {@code a + b} (same shape). */
    Tn add(Tn a, Tn b);

    Tn silu(Tn x);

    /** Nearest-neighbour upsampling of {@code [B, H, W, C]} by 2. */
    Tn upsample2x(Tn x);

    /**
     * Nearest-neighbour 2x upsampling followed by a 3x3 convolution with padding 1 (the up-sampling layer of the UNet and the VAE).
     * The default does it in two steps; the GPU reads the small input directly in the convolution.
     */
    default Tn upsampleConv(Tn x, Tn w, Tn bias) {
        Tn big = upsample2x(x);
        Tn out = conv2d(big, w, bias, 1, 1, 1, 1, 1, null, null);
        free(big);
        return out;
    }

    /** Concatenation along the channel dimension of two {@code [B, H, W, *]} tensors. */
    Tn concat(Tn a, Tn b);

    /**
     * Swaps two axes: {@code x} read as {@code [outer, d1, d2, C]} becomes {@code [outer, d2, d1, C]}, returned with the given
     * shape (any shape of the same size). The video networks move between "frames of one pixel" and "pixels of one frame" with it.
     */
    Tn swapAxes(Tn x, int outer, int d1, int d2, int... shape);

    /** {@code out[r, :] = x[r, :] + table[r mod period, :]} for {@code x [rows, C]} and a float vector table {@code [>= period, C]}. */
    Tn addRows(Tn x, Tn table, int period);

    @Override
    void close();
}
