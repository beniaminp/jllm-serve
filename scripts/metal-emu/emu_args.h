// The per-thread launch arguments shared by the transpiled kernels and the emulator runtime (no MSL macros here).
#pragma once
#include <stdint.h>

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef uint64_t ulong;
typedef uint uint2 __attribute__((ext_vector_type(2)));
typedef uint uint3 __attribute__((ext_vector_type(3)));
typedef uint uint4 __attribute__((ext_vector_type(4)));
typedef uint uint8 __attribute__((ext_vector_type(8)));

/** Filled per GPU thread by runtime.cpp before the kernel function is entered. */
struct EmuArgs {
    void *buf[32];
    void *tgmem;
    uint3 tgPos;      // threadgroup_position_in_grid
    uint3 tid;        // thread_position_in_threadgroup
    uint3 tpg;        // threads_per_threadgroup
    uint3 gid;        // thread_position_in_grid
    uint tidx;        // thread_index_in_threadgroup
    uint sg, lane, nsg;
};
typedef void (*EmuKernel)(EmuArgs &);
void emu_register(const char *name, EmuKernel k);
