package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.diffusion.Imaging.Image;

/**
 * A free preview of a latent image: the four latent channels of a Stable Diffusion model are linearly related to colour
 * to a good first approximation, so a 4×3 matrix turns the denoiser's current estimate into a small RGB picture (one
 * pixel per latent pixel, an 8th of the final size) without running the VAE. The coefficients are the commonly used
 * least-squares fits for the SD 1.x / 2.x and the SDXL latent spaces.
 */
final class LatentPreview {

    private static final float[][] SD = {
            {0.298f, 0.207f, 0.208f},
            {0.187f, 0.286f, 0.173f},
            {-0.158f, 0.189f, 0.264f},
            {-0.184f, -0.271f, -0.473f}};
    private static final float[][] XL = {
            {0.3651f, 0.4232f, 0.4341f},
            {-0.2533f, -0.0042f, 0.1068f},
            {0.1076f, 0.1111f, -0.0362f},
            {-0.3165f, -0.2492f, -0.2188f}};

    private LatentPreview() {
    }

    static Image render(float[] x0, int lh, int lw, boolean xl) {
        float[][] m = xl ? XL : SD;
        int c = x0.length / (lh * lw);
        byte[] rgb = new byte[lh * lw * 3];
        for (int p = 0; p < lh * lw; p++) {
            for (int k = 0; k < 3; k++) {
                float v = 0;
                for (int ch = 0; ch < Math.min(4, c); ch++) {
                    v += x0[p * c + ch] * m[ch][k];
                }
                v = (v + 1f) / 2f;
                rgb[p * 3 + k] = (byte) Math.max(0, Math.min(255, Math.round(v * 255f)));
            }
        }
        return new Image(lw, lh, rgb);
    }
}
