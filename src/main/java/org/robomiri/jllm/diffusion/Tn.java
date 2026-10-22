package org.robomiri.jllm.diffusion;

/**
 * A tensor of the diffusion stack: a shape and storage owned by the {@link Ops} that made it (a {@code float[]} on the
 * CPU, a Metal buffer on the GPU). Activations are laid out channels-last (NHWC, or [rows, features] for token
 * sequences); see {@link Ops}.
 */
public abstract class Tn {

    public final int[] shape;

    protected Tn(int... shape) {
        this.shape = shape.clone();
    }

    public final long numel() {
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        return n;
    }

    public final int dim(int i) {
        return shape[i];
    }

    /** Size of the last dimension (channels, or the feature width of a token sequence). */
    public final int last() {
        return shape[shape.length - 1];
    }

    /** Product of all dimensions but the last: the number of pixels of an image batch, or of tokens. */
    public final int rows() {
        return (int) (numel() / last());
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + java.util.Arrays.toString(shape);
    }
}
