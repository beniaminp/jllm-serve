package org.robomiri.jllm.diffusion;

/** Test-side access to the profiling counters of {@code MetalOps} (package-private). */
final class MetalProfile {

    private MetalProfile() {
    }

    static boolean reset(Devices d) {
        if (d.main() instanceof MetalOps m) {
            m.resetProfile();
            return Boolean.getBoolean("jllm.diffusion.profile");
        }
        return false;
    }

    static String report(Devices d) {
        return d.main() instanceof MetalOps m ? m.profileReport() : "";
    }

    static long live(Devices d) {
        return d.main() instanceof MetalOps m ? m.liveBytes() : 0;
    }

    static long peak(Devices d) {
        return d.main() instanceof MetalOps m ? m.peakBytes() : 0;
    }
}
