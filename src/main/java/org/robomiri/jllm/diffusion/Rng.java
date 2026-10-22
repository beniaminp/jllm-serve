package org.robomiri.jllm.diffusion;

/**
 * The seeded random source of image generation, defined so that anyone can reproduce it (the reference scripts do, in
 * numpy):
 *
 * <ul>
 *   <li>a counter-based SplitMix64 stream: output {@code i} (1-based) is {@code mix(seed + i · 0x9E3779B97F4A7C15)} with
 *       the usual finalizer ({@code z = (z ^ z>>30) · 0xBF58476D1CE4E5B9; z = (z ^ z>>27) · 0x94D049BB133111EB; z ^ z>>31});</li>
 *   <li>a uniform in (0, 1): {@code u = ((out >>> 11) + 0.5) · 2⁻⁵³};</li>
 *   <li>standard normals by the Box–Muller transform, two per pair of uniforms {@code (u1, u2)}:
 *       {@code r = sqrt(−2 ln u1)}, {@code z0 = r cos(2π u2)}, {@code z1 = r sin(2π u2)}, rounded to float32.</li>
 * </ul>
 *
 * A generation with seed s draws, in this order, from one stream: the initial noise {@code [C, h, w]} in the
 * channels-first order PyTorch uses, then (for img2img) the noise added to the encoded image and the posterior sample,
 * then the noise of each ancestral step. Image k of a batch uses seed {@code s + k}.
 */
public final class Rng {

    private static final long GOLDEN = 0x9E3779B97F4A7C15L;

    private final long seed;
    private long counter;
    private boolean haveSpare;
    private float spare;

    public Rng(long seed) {
        this.seed = seed;
    }

    private long next() {
        long z = seed + (++counter) * GOLDEN;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    private double uniform() {
        return ((next() >>> 11) + 0.5) * 0x1.0p-53;
    }

    /** The next {@code n} standard normal numbers. */
    public float[] normal(int n) {
        float[] out = new float[n];
        int i = 0;
        if (haveSpare && n > 0) {
            out[i++] = spare;
            haveSpare = false;
        }
        while (i < n) {
            double u1 = uniform();
            double u2 = uniform();
            double r = Math.sqrt(-2.0 * Math.log(u1));
            double th = 2.0 * Math.PI * u2;
            out[i++] = (float) (r * Math.cos(th));
            float z1 = (float) (r * Math.sin(th));
            if (i < n) {
                out[i++] = z1;
            } else {
                spare = z1;
                haveSpare = true;
            }
        }
        return out;
    }
}
