// jllm_metal: a tiny C ABI over Metal, called from Java through the FFM API (java.lang.foreign).
//
// The library knows nothing about transformers. It compiles Metal source handed over by Java, creates buffers
// (or wraps Java-owned page-aligned memory without copying, thanks to unified memory), and records generic compute
// dispatches into one command buffer per forward pass. All model logic lives in Java (MetalBackend) and in the
// shader source (kernels.metal).
//
// Build: clang -O2 -fobjc-arc -dynamiclib -framework Metal -framework Foundation jllm_metal.m -o libjllm_metal.dylib

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define JM_MAX_ARGS 31   // buffer bindings per kernel (Metal allows 31)
#define JM_SET_CAP 192   // buffers tracked between two barriers; a full set forces a barrier

// The access a kernel declares to each of its buffer arguments (from the pipeline's reflection).
enum { JM_NONE = 0, JM_READ = 1, JM_WRITE = 2 };

@interface JMContext : NSObject {
@public
    // ---- concurrent encoder mode (jm_set_concurrent) and its hazard tracker
    int mode;                // 0 serial encoder; 1 concurrent + barriers where a hazard is found; 2 concurrent + a barrier before every dispatch
    int useResources;        // barrier on the conflicting buffers only (memoryBarrierWithResources:) instead of on every buffer
    uint8_t *access;         // per pipeline, JM_MAX_ARGS entries: JM_NONE / JM_READ / JM_WRITE (read-write counts as write)
    int accessCap;           // pipelines the table has room for
    void *wBuf[JM_SET_CAP];  // buffers written since the last barrier ...
    int nW;
    void *rBuf[JM_SET_CAP];  // ... and buffers only read since then
    int nR;
    void *pwBuf[JM_SET_CAP]; // the same for the dispatches of the open group (their members are independent of each other)
    int npW;
    void *prBuf[JM_SET_CAP];
    int npR;
    int group;               // the tracker's open group: consecutive dispatches with the same non-zero id are independent
    int setGroup;            // the group id jm_set_group declared for the dispatches that follow
    int trace;               // JLLM_METAL_TRACE: print every tracked dispatch with a mark for the barrier before it
    long barriers;           // statistics since the context was created
    long dispatched;
}
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) NSMutableArray<id<MTLComputePipelineState>> *pipelines;
@property(nonatomic, strong) NSMutableArray<NSString *> *names;
@property(nonatomic, strong) id<MTLCommandBuffer> cmd;
@property(nonatomic, strong) id<MTLComputeCommandEncoder> enc;
// command buffers of the current forward pass that jm_commit already submitted (jm_commit_wait checks them)
@property(nonatomic, strong) NSMutableArray<id<MTLCommandBuffer>> *inflight;
@end

@implementation JMContext
@end

static void set_err(char *err, int len, NSString *msg) {
    if (err && len > 0) {
        const char *s = msg ? msg.UTF8String : "unknown error";
        strncpy(err, s ? s : "unknown error", (size_t) len - 1);
        err[len - 1] = 0;
    }
}

void *jm_create(const char *source, char *err, int errlen) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            set_err(err, errlen, @"no Metal device available");
            return NULL;
        }
        JMContext *c = [JMContext new];
        c.device = dev;
        c.queue = [dev newCommandQueue];
        c.pipelines = [NSMutableArray new];
        c.names = [NSMutableArray new];
        c->trace = getenv("JLLM_METAL_TRACE") != NULL;
        c.inflight = [NSMutableArray new];
        NSError *e = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        // The kernels use -INFINITY as the identity of their max reductions; the default (fast) math mode lets
        // the compiler assume no infinities exist. Relaxed keeps the fast-math optimisations but preserves
        // infinities and NaNs (macOS 15+; older systems keep the default, where the sentinels have worked so
        // far because the compiler never exploited the assumption in these reductions).
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeRelaxed;
        }
        c.library = [dev newLibraryWithSource:[NSString stringWithUTF8String:source] options:opts error:&e];
        if (!c.library) {
            set_err(err, errlen, e ? e.localizedDescription : @"shader compilation failed");
            return NULL;
        }
        return (__bridge_retained void *) c;
    }
}

void jm_destroy(void *ctx) {
    @autoreleasepool {
        JMContext *c = (__bridge_transfer JMContext *) ctx;
        c.enc = nil;
        c.cmd = nil;
        if (getenv("JLLM_METAL_STATS")) {
            fprintf(stderr, "jllm_metal: %ld dispatches tracked, %ld barriers\n", c->dispatched, c->barriers);
        }
        free(c->access);
        c->access = NULL;
        c = nil;
    }
}

int jm_device_name(void *ctx, char *out, int len) {
    @autoreleasepool {
        JMContext *c = (__bridge JMContext *) ctx;
        set_err(out, len, c.device.name);
        return 0;
    }
}

uint64_t jm_recommended_working_set(void *ctx) {
    JMContext *c = (__bridge JMContext *) ctx;
    return c.device.recommendedMaxWorkingSetSize;
}

// The system's thermal state (NSProcessInfoThermalState): 0 nominal, 1 fair, 2 serious, 3 critical. A hot Apple GPU
// runs at lower clocks, which shows up as slower tokens without any other sign.
int jm_thermal_state(void *ctx) {
    (void) ctx;
    return (int) [[NSProcessInfo processInfo] thermalState];
}

// Bytes of threadgroup memory one threadgroup may use (static arrays plus the dynamic [[threadgroup(0)]]
// argument together); 32 KB on every Apple GPU so far.
uint64_t jm_device_max_tg_mem(void *ctx) {
    JMContext *c = (__bridge JMContext *) ctx;
    return c.device.maxThreadgroupMemoryLength;
}

// Returns a pipeline index, or -1 with a message in err.
int jm_pipeline(void *ctx, const char *name, char *err, int errlen) {
    @autoreleasepool {
        JMContext *c = (__bridge JMContext *) ctx;
        id<MTLFunction> f = [c.library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!f) {
            set_err(err, errlen, [NSString stringWithFormat:@"kernel '%s' not found", name]);
            return -1;
        }
        NSError *e = nil;
        // the reflection tells which buffer arguments a kernel can write (`device T *`) and which it only reads
        // (`device const T *`): the concurrent encoder's hazard tracker needs that
        MTLComputePipelineReflection *__autoreleasing refl = nil;
        id<MTLComputePipelineState> p = [c.device newComputePipelineStateWithFunction:f
                                                                              options:MTLPipelineOptionBindingInfo
                                                                           reflection:&refl
                                                                                error:&e];
        if (!p) {
            set_err(err, errlen, e ? e.localizedDescription : @"pipeline creation failed");
            return -1;
        }
        int idx = (int) c.pipelines.count;
        [c.pipelines addObject:p];
        [c.names addObject:[NSString stringWithUTF8String:name]];
        if (idx >= c->accessCap) {
            int cap = c->accessCap ? c->accessCap * 2 : 256;
            c->access = (uint8_t *) realloc(c->access, (size_t) cap * JM_MAX_ARGS);
            c->accessCap = cap;
        }
        uint8_t *a = c->access + (size_t) idx * JM_MAX_ARGS;
        // without a reflection every argument counts as written: more barriers, never fewer
        memset(a, refl ? JM_NONE : JM_WRITE, JM_MAX_ARGS);
        if (refl) {
            for (id<MTLBinding> bind in refl.bindings) {
                if (bind.type == MTLBindingTypeBuffer && bind.index < JM_MAX_ARGS && bind.used) {
                    a[bind.index] = bind.access == MTLBindingAccessReadOnly ? JM_READ : JM_WRITE;
                }
            }
        }
        if (getenv("JLLM_METAL_DUMP_ACCESS")) {
            fprintf(stderr, "jllm_metal access %-40s", name);
            for (int i = 0; i < JM_MAX_ARGS; i++) {
                if (a[i]) {
                    fprintf(stderr, " %d%c", i, a[i] == JM_WRITE ? 'W' : 'r');
                }
            }
            fprintf(stderr, "\n");
        }
        return idx;
    }
}

int jm_pipeline_max_threads(void *ctx, int pipeline) {
    JMContext *c = (__bridge JMContext *) ctx;
    return (int) c.pipelines[(NSUInteger) pipeline].maxTotalThreadsPerThreadgroup;
}

// New buffer in shared (CPU+GPU visible) memory. With zero != 0 the contents are cleared up front; otherwise
// they are left untouched (pages are then only committed when first written), which is right for scratch
// buffers that are always fully written before they are read.
void *jm_buffer_new(void *ctx, uint64_t len, int zero) {
    @autoreleasepool {
        JMContext *c = (__bridge JMContext *) ctx;
        id<MTLBuffer> b = [c.device newBufferWithLength:(NSUInteger) len options:MTLResourceStorageModeShared];
        if (!b) {
            return NULL;
        }
        if (zero) {
            memset(b.contents, 0, (size_t) len);
        }
        return (__bridge_retained void *) b;
    }
}

// Wraps memory owned by the caller (must stay alive while the buffer is used) without copying. Metal requires
// ptr and len to be page aligned; when it refuses, the reason is printed to stderr and NULL is returned (the
// Java side throws). A silent copying fallback would double the memory of a multi-gigabyte model.
void *jm_buffer_wrap(void *ctx, void *ptr, uint64_t len) {
    @autoreleasepool {
        JMContext *c = (__bridge JMContext *) ctx;
        id<MTLBuffer> b = [c.device newBufferWithBytesNoCopy:ptr
                                                      length:(NSUInteger) len
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
        if (!b) {
            uint64_t page = (uint64_t) NSPageSize();
            fprintf(stderr,
                    "jllm_metal: newBufferWithBytesNoCopy refused %llu bytes at %p (page size %llu: ptr %s, "
                    "length %s, max buffer length %llu)\n",
                    (unsigned long long) len, ptr, (unsigned long long) page,
                    ((uintptr_t) ptr % page) == 0 ? "aligned" : "misaligned",
                    (len % page) == 0 ? "page multiple" : "not a page multiple",
                    (unsigned long long) c.device.maxBufferLength);
            return NULL;
        }
        return (__bridge_retained void *) b;
    }
}

void *jm_buffer_contents(void *buf) {
    return [(__bridge id<MTLBuffer>) buf contents];
}

void jm_buffer_release(void *buf) {
    @autoreleasepool {
        id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>) buf;
        b = nil;
    }
}

static void reset_tracker(JMContext *c) {
    c->nW = c->nR = c->npW = c->npR = 0;
    c->group = 0;
}

static void open_cmd(JMContext *c) {
    c.cmd = [c.queue commandBufferWithUnretainedReferences];
    // Serial: every dispatch sees the results of the one before it (the GPU drains between them). Concurrent: dispatches
    // may overlap, and track_hazards() inserts a barrier wherever one reads or overwrites what an earlier one wrote.
    // A new encoder needs no barrier against the previous one: encoders of one queue run in order.
    c.enc = [c.cmd computeCommandEncoderWithDispatchType:c->mode ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
    reset_tracker(c);
}

static int in_set(void **set, int n, void *b) {
    for (int i = 0; i < n; i++) {
        if (set[i] == b) {
            return 1;
        }
    }
    return 0;
}

// Adds b to a set; returns 0 when the set is full.
static int add_set(void **set, int *n, void *b) {
    if (in_set(set, *n, b)) {
        return 1;
    }
    if (*n >= JM_SET_CAP) {
        return 0;
    }
    set[(*n)++] = b;
    return 1;
}

// Called before the dispatch is encoded: decides whether it needs a barrier against the dispatches since the last one
// (it reads a buffer one of them writes, or writes a buffer one of them reads or writes) and records its own accesses.
// Dispatches of one group are independent by contract: they are compared with what came before the group only.
// Buffer granularity: a kernel's extent inside a buffer is not known, so two dispatches that touch different rows of the
// same buffer conflict unless the caller puts them in a group.
static void track_hazards_impl(JMContext *c, int pipeline, void *const *bufs, int nbuf, int group) {
    c->dispatched++;
    if (c->mode == 3) { // timing only (the results are wrong): no barriers at all, the upper bound of what overlap can gain
        return;
    }
    if (c->mode == 2) { // measuring aid: a barrier between every pair of dispatches
        if (c->dispatched > 1) {
            [c.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            c->barriers++;
        }
        return;
    }
    if (nbuf > JM_MAX_ARGS) {
        nbuf = JM_MAX_ARGS;
    }
    if (group != c->group || group == 0) { // the open group ends: its members become visible to what follows
        for (int i = 0; i < c->npW; i++) {
            add_set(c->wBuf, &c->nW, c->pwBuf[i]);
        }
        for (int i = 0; i < c->npR; i++) {
            add_set(c->rBuf, &c->nR, c->prBuf[i]);
        }
        c->npW = c->npR = 0;
        c->group = group;
    }
    const uint8_t *acc = c->access + (size_t) pipeline * JM_MAX_ARGS;
    int hazard = 0;
    for (int i = 0; i < nbuf && !hazard; i++) {
        if (acc[i] == JM_NONE) {
            continue;
        }
        void *b = bufs[i];
        if (in_set(c->wBuf, c->nW, b) || (acc[i] == JM_WRITE && in_set(c->rBuf, c->nR, b))) {
            hazard = 1;
        }
    }
    if (hazard) {
        if (c->useResources) {
            // every buffer touched since the last barrier (a read before a later write is a hazard too)
            id<MTLResource> res[4 * JM_SET_CAP];
            int n = 0;
            for (int i = 0; i < c->nW; i++) {
                res[n++] = (__bridge id<MTLResource>) c->wBuf[i];
            }
            for (int i = 0; i < c->nR; i++) {
                res[n++] = (__bridge id<MTLResource>) c->rBuf[i];
            }
            for (int i = 0; i < c->npW; i++) {
                res[n++] = (__bridge id<MTLResource>) c->pwBuf[i];
            }
            for (int i = 0; i < c->npR; i++) {
                res[n++] = (__bridge id<MTLResource>) c->prBuf[i];
            }
            [c.enc memoryBarrierWithResources:res count:(NSUInteger) n];
        } else {
            [c.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        c->barriers++;
        c->nW = c->nR = c->npW = c->npR = 0;
    }
    for (int i = 0; i < nbuf; i++) {
        if (acc[i] == JM_NONE) {
            continue;
        }
        void *b = bufs[i];
        int ok;
        if (group) {
            ok = acc[i] == JM_WRITE ? add_set(c->pwBuf, &c->npW, b) : add_set(c->prBuf, &c->npR, b);
        } else {
            ok = acc[i] == JM_WRITE ? add_set(c->wBuf, &c->nW, b) : add_set(c->rBuf, &c->nR, b);
        }
        if (!ok) { // a full set: a barrier now makes room, and orders this dispatch after everything before it
            [c.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            c->barriers++;
            c->nW = c->nR = c->npW = c->npR = 0;
            if (group) {
                acc[i] == JM_WRITE ? add_set(c->pwBuf, &c->npW, b) : add_set(c->prBuf, &c->npR, b);
            } else {
                acc[i] == JM_WRITE ? add_set(c->wBuf, &c->nW, b) : add_set(c->rBuf, &c->nR, b);
            }
        }
    }
}

static void track_hazards(JMContext *c, int pipeline, void *const *bufs, int nbuf, int group) {
    long before = c->barriers;
    track_hazards_impl(c, pipeline, bufs, nbuf, group);
    if (c->trace) { // JLLM_METAL_TRACE=1: "B" = a barrier precedes the dispatch, "g<id>" = it belongs to a group
        fprintf(stderr, "jllm_metal trace %c %-32s", c->barriers > before ? 'B' : '-', c.names[(NSUInteger) pipeline].UTF8String);
        if (group) {
            fprintf(stderr, " g%d", group);
        }
        fprintf(stderr, "\n");
    }
}

// Selects the encoder type for the command buffers opened from now on (call between forward passes, not while one is
// being recorded): 0 serial; 1 concurrent with barriers inserted by hazard tracking; 2 concurrent with a barrier
// before every dispatch (to measure what a barrier costs); +4 barriers name the conflicting buffers instead of
// covering every buffer. Returns the previous mode.
int jm_set_concurrent(void *ctx, int m) {
    JMContext *c = (__bridge JMContext *) ctx;
    int old = c->mode | (c->useResources ? 4 : 0);
    c->mode = m & 3;
    c->useResources = (m & 4) != 0;
    return old;
}

// Declares that the dispatches that follow, until the next jm_set_group call, are independent of each other (they read
// what none of them writes, and write disjoint memory), so they may overlap. Ids must differ between two adjacent groups;
// 0 closes the group. Only meaningful in the concurrent modes.
void jm_set_group(void *ctx, int id) {
    JMContext *c = (__bridge JMContext *) ctx;
    c->setGroup = id;
}

// Barriers and dispatches the tracker has seen on this context (for tests and the benchmark logs).
long jm_barriers(void *ctx) {
    return ((__bridge JMContext *) ctx)->barriers;
}

// Starts recording a new command buffer with a serial compute encoder: each dispatch sees the results of the
// previous one, so the Java side does not need explicit barriers.
void jm_begin(void *ctx) {
    @autoreleasepool {
        JMContext *c = (__bridge JMContext *) ctx;
        open_cmd(c);
    }
}

// Records one dispatch. Buffers are bound at indices [0, nbuf); params (if any) at index nbuf; tgmem > 0
// sets the size of the kernel's dynamic threadgroup memory argument [[threadgroup(0)]].
void jm_dispatch(void *ctx, int pipeline, void **bufs, const uint64_t *offsets, int nbuf,
                 const void *params, int plen, int tgmem, int gx, int gy, int gz, int tx, int ty, int tz) {
    @autoreleasepool {
        JMContext *c = (__bridge JMContext *) ctx;
        id<MTLComputeCommandEncoder> enc = c.enc;
        if (c->mode) {
            track_hazards(c, pipeline, bufs, nbuf, c->setGroup);
        }
        [enc setComputePipelineState:c.pipelines[(NSUInteger) pipeline]];
        for (int i = 0; i < nbuf; i++) {
            [enc setBuffer:(__bridge id<MTLBuffer>) bufs[i] offset:(NSUInteger) offsets[i] atIndex:(NSUInteger) i];
        }
        if (plen > 0) {
            [enc setBytes:params length:(NSUInteger) plen atIndex:(NSUInteger) nbuf];
        }
        if (tgmem > 0) {
            [enc setThreadgroupMemoryLength:(NSUInteger) tgmem atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger) gx, (NSUInteger) gy, (NSUInteger) gz)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger) tx, (NSUInteger) ty, (NSUInteger) tz)];
    }
}

// One command of a recorded decode step (DecodeProgram.java writes them; 384 bytes each, natural alignment):
// what jm_dispatch takes as arguments, packed, so a whole step crosses from Java in one call.
typedef struct {
    int32_t pipe, nbuf, plen, tgmem;
    int32_t grid[3], threads[3];
    int32_t pad[2];
    void *bufs[16];
    uint64_t offs[16];
    uint8_t params[64];
    uint8_t pad2[16];
} JMCmd;
_Static_assert(sizeof(JMCmd) == 384, "JMCmd layout is shared with DecodeProgram.java");

// Replays n recorded commands into the current encoder, exactly as n jm_dispatch calls would (the serial encoder
// orders them; params are copied with setBytes, so the caller may patch the array for the next step at once). State the
// previous command already set (pipeline, a buffer at an index, its offset) is not set again.
void jm_run(void *ctx, const JMCmd *cmds, int n) {
    @autoreleasepool {
        JMContext *c = (__bridge JMContext *) ctx;
        id<MTLComputeCommandEncoder> enc = c.enc;
        int lastPipe = -1;
        void *bound[16] = {0};
        uint64_t boundOff[16] = {0};
        for (int k = 0; k < n; k++) {
            const JMCmd *d = &cmds[k];
            if (c->mode) {
                track_hazards(c, d->pipe, d->bufs, d->nbuf, d->pad[0]);
            }
            if (d->pipe != lastPipe) {
                [enc setComputePipelineState:c.pipelines[(NSUInteger) d->pipe]];
                lastPipe = d->pipe;
            }
            for (int i = 0; i < d->nbuf; i++) {
                if (bound[i] != d->bufs[i]) {
                    [enc setBuffer:(__bridge id<MTLBuffer>) d->bufs[i] offset:(NSUInteger) d->offs[i] atIndex:(NSUInteger) i];
                    bound[i] = d->bufs[i];
                    boundOff[i] = d->offs[i];
                } else if (boundOff[i] != d->offs[i]) {
                    [enc setBufferOffset:(NSUInteger) d->offs[i] atIndex:(NSUInteger) i];
                    boundOff[i] = d->offs[i];
                }
            }
            if (d->plen > 0) {
                [enc setBytes:d->params length:(NSUInteger) d->plen atIndex:(NSUInteger) d->nbuf];
                if (d->nbuf < 16) {
                    bound[d->nbuf] = NULL; // setBytes replaced whatever buffer was bound there
                }
            }
            if (d->tgmem > 0) {
                [enc setThreadgroupMemoryLength:(NSUInteger) d->tgmem atIndex:0];
            }
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger) d->grid[0], (NSUInteger) d->grid[1], (NSUInteger) d->grid[2])
                threadsPerThreadgroup:MTLSizeMake((NSUInteger) d->threads[0], (NSUInteger) d->threads[1],
                                                  (NSUInteger) d->threads[2])];
        }
    }
}

// Submits what has been recorded so far without waiting and opens a fresh command buffer for the dispatches
// that follow, so the GPU runs the first part of a forward pass while the CPU is still encoding the rest.
// Command buffers of one queue execute in submission order, so the serial-dependency guarantee of jm_begin
// carries across the split. jm_commit_wait later waits for the last buffer and reports a failure of any.
void jm_commit(void *ctx) {
    @autoreleasepool {
        JMContext *c = (__bridge JMContext *) ctx;
        [c.enc endEncoding];
        [c.cmd commit];
        [c.inflight addObject:c.cmd];
        open_cmd(c);
    }
}

// Discards the command buffer being recorded without running it. Metal aborts the process if a compute
// encoder is released before endEncoding, so the Java side calls this when recording fails part-way.
// Parts already submitted with jm_commit are waited for (they may still read buffers the CPU writes next).
// A no-op when nothing is being recorded.
void jm_abort(void *ctx) {
    @autoreleasepool {
        JMContext *c = (__bridge JMContext *) ctx;
        if (c.enc) {
            [c.enc endEncoding];
        }
        c.enc = nil;
        c.cmd = nil;
        if (c.inflight.count > 0) {
            [c.inflight.lastObject waitUntilCompleted];
            [c.inflight removeAllObjects];
        }
    }
}

// Waits until at most `keep` of the command buffers submitted with jm_commit are still unfinished, oldest first
// (the queue runs them in order), and forgets the finished ones. Lets the CPU read the result of one pipelined
// decode step while the next is already running. Returns 0, or -1 with the error of a failed buffer.
int jm_wait_pending(void *ctx, int keep, char *err, int errlen) {
    @autoreleasepool {
        JMContext *c = (__bridge JMContext *) ctx;
        while ((int) c.inflight.count > keep) {
            id<MTLCommandBuffer> cmd = c.inflight.firstObject;
            [cmd waitUntilCompleted];
            [c.inflight removeObjectAtIndex:0];
            if (cmd.status == MTLCommandBufferStatusError) {
                set_err(err, errlen, cmd.error ? cmd.error.localizedDescription : @"command buffer failed");
                return -1;
            }
        }
        return 0;
    }
}

// Submits the recorded work and blocks until the GPU is done with it and with everything jm_commit
// submitted before. Returns 0 on success.
int jm_commit_wait(void *ctx, char *err, int errlen) {
    @autoreleasepool {
        JMContext *c = (__bridge JMContext *) ctx;
        [c.enc endEncoding];
        [c.cmd commit];
        [c.cmd waitUntilCompleted]; // in-order queue: the earlier buffers are complete too
        [c.inflight addObject:c.cmd];
        int rc = 0;
        for (id<MTLCommandBuffer> cmd in c.inflight) {
            if (cmd.status == MTLCommandBufferStatusError) {
                set_err(err, errlen, cmd.error ? cmd.error.localizedDescription : @"command buffer failed");
                rc = -1;
                break;
            }
        }
        [c.inflight removeAllObjects];
        c.enc = nil;
        c.cmd = nil;
        return rc;
    }
}
