package org.robomiri.jllm.diffusion;

/**
 * The compute devices of a pipeline: where the text encoders and the UNet run, and where the VAE runs (the VAE of SDXL
 * overflows in float16 and stays float32).
 */
public final class Devices implements AutoCloseable {

    private final Ops main;
    private final Ops vae;
    private final String description;
    private final AutoCloseable owner;

    public Devices(Ops main, Ops vae, String description, AutoCloseable owner) {
        this.main = main;
        this.vae = vae;
        this.description = description;
        this.owner = owner;
    }

    public Ops main() {
        return main;
    }

    public Ops vae() {
        return vae;
    }

    public String description() {
        return description;
    }

    /** CPU in float32. */
    public static Devices cpu() {
        CpuOps ops = new CpuOps();
        return new Devices(ops, ops, "cpu float32", ops);
    }

    /**
     * @param backend   {@code auto} (Metal when available, else CPU), {@code metal} or {@code cpu}
     * @param precision {@code auto}, {@code f16} or {@code f32}: the GPU's storage type for weights and activations
     * @param xl        an SDXL-type model: its VAE runs in float32 on the GPU (it overflows in float16)
     */
    public static Devices open(String backend, String precision, boolean xl) {
        String b = backend == null ? "auto" : backend.toLowerCase(java.util.Locale.ROOT);
        boolean wantMetal = b.equals("metal") || (b.equals("auto") && MetalOps.available());
        if (!wantMetal) {
            if (b.equals("metal")) {
                throw new IllegalStateException("the Metal backend is not available on this machine");
            }
            return cpu();
        }
        return MetalOps.open(precision, xl);
    }

    @Override
    public void close() {
        try {
            if (owner != null) {
                owner.close();
            }
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }
}
