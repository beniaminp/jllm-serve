"""ctypes binding of the Metal emulator (libjllm_metal_emu.so) for kernel tests: buffers from numpy arrays, one
dispatch per call. See kernel_tests.py."""
import ctypes
import os
import struct
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')


class Emu:
    def __init__(self, lib=None):
        lib = lib or os.path.join(ROOT, 'target', 'native', 'libjllm_metal_emu.' + ('dylib' if sys.platform == 'darwin' else 'so'))
        self.l = ctypes.CDLL(lib)
        l = self.l
        l.jm_create.restype = ctypes.c_void_p
        l.jm_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        l.jm_pipeline.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        l.jm_buffer_wrap.restype = ctypes.c_void_p
        l.jm_buffer_wrap.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        l.jm_buffer_release.argtypes = [ctypes.c_void_p]
        l.jm_dispatch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                  ctypes.c_int] + [ctypes.c_int] * 6
        err = ctypes.create_string_buffer(512)
        self.ctx = l.jm_create(b'', err, 512)
        self.pipes = {}

    def pipeline(self, name):
        if name not in self.pipes:
            err = ctypes.create_string_buffer(512)
            i = self.l.jm_pipeline(self.ctx, name.encode(), err, 512)
            if i < 0:
                raise KeyError(err.value.decode())
            self.pipes[name] = i
        return self.pipes[name]

    def dispatch(self, name, arrays, params, grid, threads, tgmem=0):
        """arrays: numpy arrays bound at 0..n-1 (contiguous; written in place); params: bytes bound at n."""
        bufs = []
        for a in arrays:
            assert a.flags['C_CONTIGUOUS']
            bufs.append(self.l.jm_buffer_wrap(self.ctx, a.ctypes.data, a.nbytes))
        n = len(bufs)
        b = (ctypes.c_void_p * max(1, n))(*bufs)
        o = (ctypes.c_uint64 * max(1, n))(*([0] * n))
        pb = ctypes.create_string_buffer(params, len(params)) if params else None
        g = list(grid) + [1] * (3 - len(grid))
        t = list(threads) + [1] * (3 - len(threads))
        self.l.jm_dispatch(self.ctx, self.pipeline(name), b, o, n, pb, len(params or b''), tgmem, *g, *t)
        for x in bufs:
            self.l.jm_buffer_release(x)


def pack(fmt, *values):
    """A params struct of 32-bit fields: fmt like 'iiiif' (i = int32, f = float32)."""
    return struct.pack('<' + fmt, *values)
