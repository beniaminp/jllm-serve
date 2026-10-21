#!/usr/bin/env python3
"""numpy reference dequantizers for the ggml block types the Metal kernels read in place besides Q4_0 / Q8_0 / the K
quants of kernel_tests.py: Q4_1, Q5_0, Q5_1, Q3_K, IQ4_NL, IQ4_XS, MXFP4 and plain F16 / BF16 (names as in the
kernel suffixes: q41 q50 q51 q3k iq4nl iq4xs mxfp4 f16w bf16w).

They are written from ggml-quants.c (dequantize_row_*), independently of the kernels and of the Java decoders, and used
by kernel_tests.py. Run as a script with the path of a shared library built from llama.cpp's ggml-quants.c they are
compared with, bit for bit, on random blocks:

    clang -O2 -fPIC -shared -I<llama.cpp>/ggml/include -I<llama.cpp>/ggml/src -o libggmlq.dylib \
          <llama.cpp>/ggml/src/ggml-quants.c stubs.c        # stubs.c: ggml_abort, ggml_row_size, ggml_type_name, ggml_type_size
    python3 scripts/metal-emu/ggml_ref.py libggmlq.dylib
"""
import ctypes
import sys

import numpy as np

KV_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)
KV_FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)

# fmt -> (weights per block, bytes per block, ggml function suffix)
INFO = {
    'q41': (32, 20, 'q4_1'), 'q50': (32, 22, 'q5_0'), 'q51': (32, 24, 'q5_1'), 'q3k': (256, 110, 'q3_K'),
    'iq4nl': (32, 18, 'iq4_nl'), 'iq4xs': (256, 136, 'iq4_xs'), 'mxfp4': (32, 17, 'mxfp4'),
}
NEW = tuple(INFO) + ('f16w', 'bf16w')


def f16(rng, shape, lo, hi):
    return rng.uniform(lo, hi, shape).astype(np.float16)


def _put_f16(blk, off, h):
    blk[..., off:off + 2] = h.view(np.uint8).reshape(h.shape + (2,))


def random_blocks(fmt, rows, cols, rng):
    """(blocks uint8 [rows, nb, size]) of random quants with sane scales (random bytes would often be NaN / inf)."""
    if fmt in ('f16w', 'bf16w'):
        nb = cols // 32
        v = (rng.standard_normal((rows, nb, 32)) * 0.1).astype(np.float32)
        if fmt == 'f16w':
            raw = v.astype(np.float16).view(np.uint8).reshape(rows, nb, 64)
        else:
            raw = (v.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8).reshape(rows, nb, 64)
        return raw.copy()
    bs, size, _ = INFO[fmt]
    nb = cols // bs
    blk = rng.integers(0, 256, (rows, nb, size)).astype(np.uint8)
    d = f16(rng, (rows, nb), 0.002, 0.02)
    if fmt in ('q41', 'q51'):
        _put_f16(blk, 0, d)
        _put_f16(blk, 2, f16(rng, (rows, nb), -0.05, 0.05))
    elif fmt in ('q50', 'iq4nl', 'iq4xs'):
        _put_f16(blk, 0, d)
    elif fmt == 'q3k':
        _put_f16(blk, 108, d)
    elif fmt == 'mxfp4':
        blk[:, :, 0] = rng.integers(100, 140, (rows, nb))
    return blk


def _bits32(b4):
    """[..., 4] little-endian bytes -> [..., 32] bits (bit i of the u32 is element i)."""
    return np.unpackbits(b4, axis=-1, bitorder='little')


def dequant(fmt, blk):
    """float32 [rows, cols] from blocks [rows, nb, size]."""
    rows, nb, size = blk.shape
    if fmt in ('f16w', 'bf16w'):
        raw = blk.reshape(rows, nb * 64)
        if fmt == 'f16w':
            return raw.view(np.float16).astype(np.float32).reshape(rows, nb * 32)
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(rows, nb * 32)
    f32 = np.float32
    d = blk[:, :, 0:2].copy().view(np.float16).reshape(rows, nb).astype(f32)
    if fmt in ('q41', 'q50', 'q51'):
        off = {'q41': 4, 'q50': 6, 'q51': 8}[fmt]
        qs = blk[:, :, off:off + 16]
        lo, hi = (qs & 15).astype(np.int32), (qs >> 4).astype(np.int32)
        if fmt != 'q41':
            qh = _bits32(blk[:, :, 2:6] if fmt == 'q50' else blk[:, :, 4:8]).reshape(rows, nb, 32).astype(np.int32)
            lo, hi = lo | (qh[:, :, :16] << 4), hi | (qh[:, :, 16:] << 4)
        v = np.concatenate([lo, hi], axis=2)
        if fmt == 'q50':
            w = (v - 16).astype(f32) * d[:, :, None]
        else:
            m = blk[:, :, 2:4].copy().view(np.float16).reshape(rows, nb).astype(f32)
            w = v.astype(f32) * d[:, :, None] + m[:, :, None]
        return w.reshape(rows, nb * 32)
    if fmt == 'iq4nl':
        qs = blk[:, :, 2:18]
        w = np.concatenate([KV_IQ4NL[qs & 15], KV_IQ4NL[qs >> 4]], axis=2) * d[:, :, None]
        return w.reshape(rows, nb * 32)
    if fmt == 'mxfp4':
        e = blk[:, :, 0].astype(np.uint32)
        bits = np.where(e < 2, np.uint32(0x00200000) << e, (e - 1) << 23).astype(np.uint32)
        scale = bits.view(f32)
        qs = blk[:, :, 1:17]
        w = np.concatenate([KV_FP4[qs & 15], KV_FP4[qs >> 4]], axis=2) * scale[:, :, None]
        return w.reshape(rows, nb * 32)
    if fmt == 'iq4xs':
        sh = blk[:, :, 2:4].copy().view(np.uint16).reshape(rows, nb).astype(np.int32)
        sl = blk[:, :, 4:8].astype(np.int32)
        w = np.zeros((rows, nb, 8, 32), f32)
        for ib in range(8):
            ls = ((sl[:, :, ib // 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4)
            dl = d * (ls - 32).astype(f32)
            qs = blk[:, :, 8 + 16 * ib:8 + 16 * ib + 16]
            w[:, :, ib, :16] = KV_IQ4NL[qs & 15] * dl[:, :, None]
            w[:, :, ib, 16:] = KV_IQ4NL[qs >> 4] * dl[:, :, None]
        return w.reshape(rows, nb * 256)
    if fmt == 'q3k':
        d = blk[:, :, 108:110].copy().view(np.float16).reshape(rows, nb).astype(f32)
        hm = blk[:, :, 0:32].astype(np.int32)
        qs = blk[:, :, 32:96].astype(np.int32)
        sc12 = blk[:, :, 96:108].astype(np.int32)
        sc = np.zeros((rows, nb, 16), np.int32)
        for s in range(16):
            g, i = s >> 2, s & 3
            lowb = sc12[:, :, 4 * (g & 1) + i]
            lo = lowb & 15 if g < 2 else lowb >> 4
            sc[:, :, s] = (lo | (((sc12[:, :, 8 + i] >> (2 * g)) & 3) << 4)) - 32
        w = np.zeros((rows, nb, 256), f32)
        for u in range(8):                       # unit u = 4 n + j: bytes 32 n .. 32 n + 31 of qs, shift 2 j, mask bit u
            n, j = u >> 2, u & 3
            for h in range(2):                   # two 16-weight runs with their own scale
                q = (qs[:, :, 32 * n + 16 * h:32 * n + 16 * h + 16] >> (2 * j)) & 3
                hb = (hm[:, :, 16 * h:16 * h + 16] >> u) & 1
                val = (q - 4 * (1 - hb)).astype(f32)
                dl = d * sc[:, :, 2 * u + h].astype(f32)
                w[:, :, 32 * u + 16 * h:32 * u + 16 * h + 16] = dl[:, :, None] * val
        return w.reshape(rows, nb * 256)
    raise ValueError(fmt)


def weights(fmt, rows, cols, rng):
    """(wq bytes, dummy scales, w float32 [rows, cols]) as fmt_weights of kernel_tests.py returns them."""
    blk = random_blocks(fmt, rows, cols, rng)
    return blk.reshape(-1).copy(), np.zeros(4, np.float32), dequant(fmt, blk)


def main(lib_path):
    lib = ctypes.CDLL(lib_path)
    rng = np.random.default_rng(99)
    bad = 0
    for fmt, (bs, size, fn_suffix) in INFO.items():
        rows, cols = 6, bs * 8 if bs == 256 else 512
        blk = random_blocks(fmt, rows, cols, rng)
        # exercise all exponents of MXFP4 including the denormal-scale ones, and unconstrained quant bits everywhere
        if fmt == 'mxfp4':
            blk[:, :, 0] = rng.integers(0, 256, blk.shape[:2])
            blk[:, :, 0] = np.where(blk[:, :, 0] == 255, 127, blk[:, :, 0])
        want = dequant(fmt, blk)
        fn = getattr(lib, 'dequantize_row_' + fn_suffix)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        got = np.zeros((rows, cols), np.float32)
        for r in range(rows):
            b = np.ascontiguousarray(blk[r].reshape(-1))
            fn(b.ctypes.data, got[r].ctypes.data, cols)
        same = np.array_equal(got.view(np.uint32), want.view(np.uint32))
        bad += not same
        print('%-6s numpy reference == llama.cpp dequantize_row_%s (%d weights, bitwise): %s' % (fmt, fn_suffix, got.size, same))
    for fmt in ('f16w', 'bf16w'):
        blk = random_blocks(fmt, 2, 128, rng)
        print('%-6s plain conversion (no ggml function)' % fmt)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main(sys.argv[1])
