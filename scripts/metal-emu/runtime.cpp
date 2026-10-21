// The emulator's side of the jm_* C API of src/main/native/jllm_metal.m, so MetalBackend runs unchanged on any
// machine (-Djllm.metal.emulator=true -Djllm.metal.lib=target/native/libjllm_metal_emu.so).
//
// A dispatch runs at once (the real backend's serial encoder gives the same ordering): threadgroup after threadgroup,
// each GPU thread of a threadgroup a fiber with its own stack. A fiber runs until it finishes or reaches a
// collective: threadgroup_barrier waits for every live thread of the threadgroup, a simd_* operation for every live
// lane of the simdgroup (their values are exchanged through a per-simdgroup buffer, double-buffered by phase so a
// fast lane can enter the next collective while slow lanes still read the last one). A round of the scheduler in
// which no fiber gets past anything is a deadlock (a collective in divergent code) and aborts with the kernel name.
#include "emu_args.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/mman.h>
#include <ucontext.h>
#include <unordered_map>
#include <vector>

namespace {

struct Kernel {
    std::string name;
    EmuKernel fn;
};
std::vector<Kernel> &kernels() {
    static std::vector<Kernel> k;
    return k;
}
std::unordered_map<std::string, int> &byName() {
    static std::unordered_map<std::string, int> m;
    return m;
}

struct Buffer {
    void *p;
    size_t len;
    bool owned;
};

constexpr size_t STACK = 256 * 1024;
constexpr int MAX_THREADS = 1024;

struct Fiber {
    ucontext_t ctx;
    EmuArgs args;
    bool done;
    int simdPhase;
    int waiting; // 0 running, 1 at a barrier, 2 in a simd collective (for the deadlock report)
};

struct Simd {
    int width;          // live lanes
    int arrived[2];
    int departed[2];
    int released[2];    // lanes a completed collective let through (they must all leave before it is reused)
    uint32_t mask[2];   // which lanes arrived
    alignas(16) unsigned char buf[2][32][32];
};

struct Group {
    Fiber fibers[MAX_THREADS];
    int n;
    int live;
    int barrierArrived;
    int barrierGen;
    Simd simd[MAX_THREADS / 32];
    EmuKernel fn;
};

Group *G;            // the running threadgroup
Fiber *cur;          // the running fiber
ucontext_t sched;
bool progressed;
char *stacks;
const char *curKernel = "";

void yieldToScheduler() { swapcontext(&cur->ctx, &sched); }

void fiberMain() {
    G->fn(cur->args);
    cur->done = true;
    G->live--;
    G->simd[cur->args.sg].width--;
    progressed = true;
    // a barrier the others wait at may now be complete without us
    if (G->live > 0 && G->barrierArrived > 0 && G->barrierArrived >= G->live) {
        G->barrierArrived = 0;
        G->barrierGen++;
    }
}

void runGroup(int ntg) {
    int stuck = 0;
    while (G->live > 0) {
        progressed = false;
        for (int i = 0; i < G->n; i++) {
            Fiber *f = &G->fibers[i];
            if (f->done) continue;
            cur = f;
            swapcontext(&sched, &f->ctx);
        }
        if (!progressed && ++stuck > 2) {
            fprintf(stderr, "metal emulator: kernel %s deadlocked in threadgroup %d (a barrier or simd_* call that"
                            " not every live thread reaches); live %d, barrier %d/%d\n", curKernel, ntg, G->live,
                    G->barrierArrived, G->live);
            for (int i = 0; i < G->n; i++) {
                Fiber *f = &G->fibers[i];
                Simd &s = G->simd[i / 32];
                if (i % 32 == 0) fprintf(stderr, "  simdgroup %d: width %d arrived %d/%d\n", i / 32, s.width, s.arrived[0], s.arrived[1]);
                if (!f->done) fprintf(stderr, "    thread %d waits %d phase %d\n", i, f->waiting, f->simdPhase);
            }
            abort();
        }
        if (progressed) stuck = 0;
    }
}

} // namespace

void emu_register(const char *name, EmuKernel k) {
    byName()[name] = (int) kernels().size();
    kernels().push_back({name, k});
}

void emu_barrier() {
    int gen = G->barrierGen;
    if (++G->barrierArrived >= G->live) {
        G->barrierArrived = 0;
        G->barrierGen++;
        progressed = true;
        return;
    }
    cur->waiting = 1;
    while (G->barrierGen == gen) yieldToScheduler();
    cur->waiting = 0;
    progressed = true;
}

uint32_t emu_simd_exchange(const void *mine, void *all, int size) {
    Simd &s = G->simd[cur->args.sg];
    int ph = cur->simdPhase;
    cur->simdPhase ^= 1;
    memcpy(s.buf[ph][cur->args.lane], mine, (size_t) size);
    s.arrived[ph]++;
    s.mask[ph] |= 1u << cur->args.lane;
    cur->waiting = 2;
    while (s.released[ph] == 0 && s.arrived[ph] < s.width) yieldToScheduler();
    cur->waiting = 0;
    if (s.released[ph] == 0) s.released[ph] = s.arrived[ph]; // lanes that exit from now on do not change it
    progressed = true;
    // lanes are numbered densely from 0 within a simdgroup, so the live ones are 0 .. width-1 unless some exited
    int base = cur->args.sg * 32;
    for (int l = 0; l < 32; l++) {
        int t = base + l;
        if (t < G->n) memcpy((char *) all + (size_t) l * size, s.buf[ph][l], (size_t) size);
    }
    uint32_t mask = s.mask[ph];
    if (++s.departed[ph] >= s.released[ph]) {
        s.arrived[ph] = 0;
        s.departed[ph] = 0;
        s.released[ph] = 0;
        s.mask[ph] = 0;
    }
    return mask;
}


int emu_lane() { return (int) cur->args.lane; }

extern "C" {

void *jm_create(const char *source, char *err, int errlen) {
    (void) source; // the kernels were transpiled from kernels.metal at build time
    if (!stacks) {
        stacks = (char *) mmap(nullptr, STACK * MAX_THREADS, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        G = new Group();
    }
    return (void *) 1;
}

void jm_destroy(void *ctx) { (void) ctx; }

int jm_device_name(void *ctx, char *out, int len) {
    snprintf(out, (size_t) len, "Metal emulator (CPU, %zu kernels)", kernels().size());
    return 0;
}

uint64_t jm_recommended_working_set(void *ctx) { return 64ull << 30; }
// no thermal sensor to read: nominal, or JLLM_EMU_THERMAL=0..3 to test the warning path
int jm_thermal_state(void *ctx) {
    const char *v = getenv("JLLM_EMU_THERMAL");
    return v ? atoi(v) : 0;
}

uint64_t jm_device_max_tg_mem(void *ctx) { return 32768; }

int jm_pipeline(void *ctx, const char *name, char *err, int errlen) {
    auto it = byName().find(name);
    if (it == byName().end()) {
        snprintf(err, (size_t) errlen, "kernel '%s' not found", name);
        return -1;
    }
    return it->second;
}

int jm_pipeline_max_threads(void *ctx, int pipeline) { return MAX_THREADS; }

void *jm_buffer_new(void *ctx, uint64_t len, int zero) {
    void *p = zero ? calloc(1, len ? len : 1) : malloc(len ? len : 1);
    return p ? new Buffer{p, len, true} : nullptr;
}

void *jm_buffer_wrap(void *ctx, void *ptr, uint64_t len) { return new Buffer{ptr, len, false}; }

void *jm_buffer_contents(void *buf) { return ((Buffer *) buf)->p; }

void jm_buffer_release(void *buf) {
    Buffer *b = (Buffer *) buf;
    if (b->owned) free(b->p);
    delete b;
}

void jm_begin(void *ctx) {}
void jm_commit(void *ctx) {}
void jm_abort(void *ctx) {}

int jm_commit_wait(void *ctx, char *err, int errlen) { return 0; }

int jm_wait_pending(void *ctx, int keep, char *err, int errlen) { return 0; } // dispatches already ran

void jm_dispatch(void *ctx, int pipeline, void **bufs, const uint64_t *offsets, int nbuf, const void *params, int plen,
                 int tgmem, int gx, int gy, int gz, int tx, int ty, int tz) {
    const Kernel &k = kernels()[(size_t) pipeline];
    curKernel = k.name.c_str();
    int n = tx * ty * tz;
    if (n > MAX_THREADS) {
        fprintf(stderr, "metal emulator: %s with %d threads per threadgroup\n", curKernel, n);
        abort();
    }
    std::vector<unsigned char> pbytes(params ? (const unsigned char *) params : nullptr,
                                      params ? (const unsigned char *) params + plen : nullptr);
    std::vector<unsigned char> dyn((size_t) (tgmem > 0 ? tgmem : 0) + 16);
    void *bp[32];
    for (int i = 0; i < nbuf; i++) bp[i] = (char *) ((Buffer *) bufs[i])->p + offsets[i];
    if (plen > 0) bp[nbuf] = pbytes.data();
    int nsg = (n + 31) / 32;
    int ntg = 0;
    for (int z = 0; z < gz; z++)
        for (int y = 0; y < gy; y++)
            for (int x = 0; x < gx; x++, ntg++) {
                G->n = n;
                G->live = n;
                G->barrierArrived = 0;
                G->barrierGen = 0;
                G->fn = k.fn;
                for (int s = 0; s < nsg; s++) {
                    G->simd[s] = Simd();
                    G->simd[s].width = (s + 1) * 32 <= n ? 32 : n - s * 32;
                }
                for (int i = 0; i < n; i++) {
                    Fiber *f = &G->fibers[i];
                    f->done = false;
                    f->simdPhase = 0;
                    EmuArgs &a = f->args;
                    memcpy(a.buf, bp, sizeof bp);
                    a.tgmem = dyn.data();
                    a.tgPos = (uint3){(uint) x, (uint) y, (uint) z};
                    a.tid = (uint3){(uint) (i % tx), (uint) ((i / tx) % ty), (uint) (i / (tx * ty))};
                    a.tpg = (uint3){(uint) tx, (uint) ty, (uint) tz};
                    a.gid = (uint3){(uint) (x * tx) + a.tid.x, (uint) (y * ty) + a.tid.y, (uint) (z * tz) + a.tid.z};
                    a.tidx = (uint) i;
                    a.sg = (uint) (i / 32);
                    a.lane = (uint) (i % 32);
                    a.nsg = (uint) nsg;
                    getcontext(&f->ctx);
                    f->ctx.uc_stack.ss_sp = stacks + (size_t) i * STACK;
                    f->ctx.uc_stack.ss_size = STACK;
                    f->ctx.uc_link = &sched;
                    makecontext(&f->ctx, fiberMain, 0);
                }
                runGroup(ntg);
            }
}

// jm_run: the commands of a recorded decode step (layout of JMCmd in jllm_metal.m / DecodeProgram.java), run in order
struct JMCmd {
    int32_t pipe, nbuf, plen, tgmem;
    int32_t grid[3], threads[3];
    int32_t pad[2];
    void *bufs[16];
    uint64_t offs[16];
    uint8_t params[64];
    uint8_t pad2[16];
};
static_assert(sizeof(JMCmd) == 384, "JMCmd layout is shared with DecodeProgram.java");

void jm_run(void *ctx, const JMCmd *cmds, int n) {
    for (int k = 0; k < n; k++) {
        const JMCmd &d = cmds[k];
        jm_dispatch(ctx, d.pipe, (void **) d.bufs, d.offs, d.nbuf, d.plen > 0 ? d.params : nullptr, d.plen, d.tgmem,
                    d.grid[0], d.grid[1], d.grid[2], d.threads[0], d.threads[1], d.threads[2]);
    }
}

} // extern "C"
