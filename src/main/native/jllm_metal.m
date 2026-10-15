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

@interface JMContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) NSMutableArray<id<MTLComputePipelineState>> *pipelines;
@property(nonatomic, strong) id<MTLCommandBuffer> cmd;
@property(nonatomic, strong) id<MTLComputeCommandEncoder> enc;
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
        NSError *e = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
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
        id<MTLComputePipelineState> p = [c.device newComputePipelineStateWithFunction:f error:&e];
        if (!p) {
            set_err(err, errlen, e ? e.localizedDescription : @"pipeline creation failed");
            return -1;
        }
        [c.pipelines addObject:p];
        return (int) c.pipelines.count - 1;
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

// Starts recording a new command buffer with a serial compute encoder: each dispatch sees the results of the
// previous one, so the Java side does not need explicit barriers.
void jm_begin(void *ctx) {
    @autoreleasepool {
        JMContext *c = (__bridge JMContext *) ctx;
        c.cmd = [c.queue commandBufferWithUnretainedReferences];
        c.enc = [c.cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    }
}

// Records one dispatch. Buffers are bound at indices [0, nbuf); params (if any) at index nbuf.
void jm_dispatch(void *ctx, int pipeline, void **bufs, const uint64_t *offsets, int nbuf,
                 const void *params, int plen, int gx, int gy, int gz, int tx, int ty, int tz) {
    @autoreleasepool {
        JMContext *c = (__bridge JMContext *) ctx;
        id<MTLComputeCommandEncoder> enc = c.enc;
        [enc setComputePipelineState:c.pipelines[(NSUInteger) pipeline]];
        for (int i = 0; i < nbuf; i++) {
            [enc setBuffer:(__bridge id<MTLBuffer>) bufs[i] offset:(NSUInteger) offsets[i] atIndex:(NSUInteger) i];
        }
        if (plen > 0) {
            [enc setBytes:params length:(NSUInteger) plen atIndex:(NSUInteger) nbuf];
        }
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger) gx, (NSUInteger) gy, (NSUInteger) gz)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger) tx, (NSUInteger) ty, (NSUInteger) tz)];
    }
}

// Discards the command buffer being recorded without running it. Metal aborts the process if a compute
// encoder is released before endEncoding, so the Java side calls this when recording fails part-way.
// A no-op when nothing is being recorded.
void jm_abort(void *ctx) {
    @autoreleasepool {
        JMContext *c = (__bridge JMContext *) ctx;
        if (c.enc) {
            [c.enc endEncoding];
        }
        c.enc = nil;
        c.cmd = nil;
    }
}

// Submits the recorded work and blocks until the GPU is done. Returns 0 on success.
int jm_commit_wait(void *ctx, char *err, int errlen) {
    @autoreleasepool {
        JMContext *c = (__bridge JMContext *) ctx;
        [c.enc endEncoding];
        [c.cmd commit];
        [c.cmd waitUntilCompleted];
        int rc = 0;
        if (c.cmd.status == MTLCommandBufferStatusError) {
            set_err(err, errlen, c.cmd.error ? c.cmd.error.localizedDescription : @"command buffer failed");
            rc = -1;
        }
        c.enc = nil;
        c.cmd = nil;
        return rc;
    }
}
