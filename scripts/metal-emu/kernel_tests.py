#!/usr/bin/env python3
"""Kernel-level tests on the Metal emulator against numpy references (python3 -I scripts/metal-emu/kernel_tests.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from emu import Emu, pack

E = Emu()
failures = 0


def check(ok, what):
    global failures
    print(('ok   ' if ok else 'FAIL ') + what)
    if not ok:
        failures += 1


def attention_ref(q, kc, vc, n_heads, n_kv, hd, pos0, ntok, scale, window=0, softcap=0.0):
    out = np.zeros((ntok, n_heads * hd), np.float64)
    g = n_heads // n_kv
    for t in range(ntok):
        ln = pos0 + t + 1
        first = max(0, ln - window) if window > 0 else 0
        for h in range(n_heads):
            kvh = h // g
            qv = q[t, h * hd:(h + 1) * hd].astype(np.float64)
            k = kc[first:ln, kvh * hd:(kvh + 1) * hd].astype(np.float64)
            v = vc[first:ln, kvh * hd:(kvh + 1) * hd].astype(np.float64)
            s = k @ qv * scale
            if softcap > 0:
                s = np.tanh(s / softcap) * softcap
            p = np.exp(s - s.max())
            p /= p.sum()
            out[t, h * hd:(h + 1) * hd] = p @ v
    return out


def q8_rows(x, n_kv, hd):
    """The q8 cache rows of x ([rows, n_kv * hd] f32) as the engine stores them (KvDtype.q8): per (row, kv head)
    scale = max|x| / 127, q = rint(x / scale); each row is n_kv * hd int8 values then the n_kv f32 scales. Returns
    (bytes [rows, n_kv * hd + 4 * n_kv], the dequantized values q * scale)."""
    rows = x.shape[0]
    h = x.reshape(rows, n_kv, hd)
    scale = (np.abs(h).max(axis=2) / np.float32(127)).astype(np.float32)
    safe = np.where(scale > 0, scale, np.float32(1))
    q = np.clip(np.rint(h / safe[:, :, None]), -127, 127).astype(np.int8)
    q[scale == 0] = 0
    out = np.concatenate([q.reshape(rows, -1).view(np.uint8), scale.view(np.uint8).reshape(rows, -1)], axis=1)
    return np.ascontiguousarray(out), (q.astype(np.float32) * scale[:, :, None]).reshape(rows, -1)


def attention(name, hd, n_heads=4, n_kv=2, pos0=5, ntok=3, ctx=16, G=1, q8=False):
    rng = np.random.default_rng(1)
    q = rng.standard_normal((ntok, n_heads * hd)).astype(np.float32)
    kc = rng.standard_normal((ctx, n_kv * hd)).astype(np.float32)
    vc = rng.standard_normal((ctx, n_kv * hd)).astype(np.float32)
    stride = n_kv * hd
    if q8:  # the kernel reads int8 * scale; the reference sees the same dequantized values
        (kc, kd), (vc, vd), stride = q8_rows(kc, n_kv, hd), q8_rows(vc, n_kv, hd), n_kv * hd + 4 * n_kv
    else:
        kd, vd = kc, vc
    out = np.zeros((ntok, n_heads * hd), np.float32)
    scores = np.zeros((ntok * n_heads * ctx,), np.float32)
    scale = 1 / np.sqrt(hd)
    # AttnParams: nHeads nKvHeads qkDim vDim pos0 ctx scale qStride kStride vStride oStride window tokenOffset softcap
    p = pack('iiiiiifiiiiiif', n_heads, n_kv, hd, hd, pos0, ctx, scale, n_heads * hd, stride, stride,
             n_heads * hd, 0, 0, 0.0)
    E.dispatch(name, [q, kc, vc, out, scores], p, (n_heads // G, ntok), (256,))
    ref = attention_ref(q, kd, vd, n_heads, n_kv, hd, pos0, ntok, scale)
    err = np.abs(out - ref).max()
    check(err < 1e-4, '%s hd=%d: max err %.2e' % (name, hd, err))


def kv_store_q8(hd, n_kv=2, ntok=5, pos0=3, ctx=12):
    """kv_store_q8 quantizes the staging rows of k and v into the cache rows [pos0, pos0 + ntok) bit for bit as
    numpy (and the engine's Java side) does, and leaves the other rows alone."""
    rng = np.random.default_rng(4)
    k = (rng.standard_normal((ntok, n_kv * hd)) * rng.random((ntok, 1)) * 4).astype(np.float32)
    v = rng.standard_normal((ntok, n_kv * hd)).astype(np.float32)
    k[1, :hd] = 0  # an all-zero row: scale 0, values 0
    stride = n_kv * hd + 4 * n_kv
    kc = np.full((ctx, stride), 0x5A, np.uint8)
    vc = np.full((ctx, stride), 0x5A, np.uint8)
    # KvStoreParams: headDim kvDim nKvHeads stride; the cache is bound at row pos0
    kview = kc[pos0:]
    vview = vc[pos0:]
    kb = np.ascontiguousarray(kview)
    vb = np.ascontiguousarray(vview)
    E.dispatch('kv_store_q8', [k, v, kb, vb], pack('iiii', hd, n_kv * hd, n_kv, stride), (2 * n_kv, ntok), (32,))
    want_k, kd = q8_rows(k, n_kv, hd)
    want_v, vd = q8_rows(v, n_kv, hd)
    ok = (kb[:ntok] == want_k).all() and (vb[:ntok] == want_v).all() and (kb[ntok:] == 0x5A).all()
    err = max(np.abs(kd - k).max() / (np.abs(k).max() / 254), np.abs(vd - v).max() / (np.abs(v).max() / 254))
    check(ok and err <= 1.0001, 'kv_store_q8 hd=%d: rows as numpy %s, max error %.3f of max|x| / 254'
          % (hd, 'yes' if ok else 'NO', err))


def q4_weights(rows, cols, rng):
    """Q4_0 as the engine stores it: nibbles (low = weight j, high = j + 16 of each 32-block), f32 scales apart."""
    q = rng.integers(0, 16, (rows, cols)).astype(np.uint8)
    s = (rng.random((rows, cols // 32)) * 0.1 + 0.01).astype(np.float32)
    blocks = q.reshape(rows, cols // 32, 32)
    packed = (blocks[:, :, :16] | (blocks[:, :, 16:] << 4)).astype(np.uint8).reshape(-1)
    w = (blocks.astype(np.float32) - 8) * s[:, :, None]
    return packed.copy(), s.reshape(-1).copy(), w.reshape(rows, cols)


def matmul(name, rows, cols, ntok, grid, threads=128, tgmem=0, bias=False, acc=False):
    rng = np.random.default_rng(2)
    wq, ws, w = q4_weights(rows, cols, rng)
    x = rng.standard_normal((ntok, cols)).astype(np.float32)
    out0 = rng.standard_normal((ntok, rows)).astype(np.float32) if acc else np.zeros((ntok, rows), np.float32)
    out = out0.copy()
    nw = np.ones(cols, np.float32)
    b = rng.standard_normal(rows).astype(np.float32)
    # MatmulParams: rows cols ntok accumulate eps softcap hasBias (the o_proj bias, bound at 5)
    E.dispatch(name, [wq, ws, x, out, nw, b], pack('iiiiffi', rows, cols, ntok, int(acc), 1e-6, 0.0, int(bias)), grid,
               (threads,), tgmem)
    ref = x.astype(np.float64) @ w.T.astype(np.float64)
    if bias:
        ref += b
    if acc:
        ref += out0
    err = np.abs(out - ref).max() / np.abs(ref).max()
    check(err < 1e-5, '%s %dx%d, %d tokens%s%s: rel err %.2e' % (name, rows, cols, ntok, ', bias' if bias else '',
                                                               ', accumulate' if acc else '', err))


for cols in (128, 1024):
    matmul('matmul_q4', 128, cols, 3, (128 // 4, 1))
    matmul('matmul_pf_q4', 128, cols, 9, ((128 + 7) // 8, 2))
    matmul('matmul_mma_q4', 128, cols, 40, (2, 2), threads=128, tgmem=0)
matmul('matmul_q4', 128, 128, 3, (128 // 4, 1), bias=True, acc=True)
matmul('matmul_pf_q4', 96, 128, 9, ((96 + 7) // 8, 2), bias=True)
matmul('matmul_mma_q4', 96, 128, 40, (2, 2), threads=128, tgmem=0, bias=True, acc=True)

attention('attention_g1_d64', 64)
attention('attention_g1_d128', 128)
attention('attention_g1_d256', 256)
attention('attention', 96)
attention('attention_g1_d128_q8', 128, q8=True)
attention('attention_g1_d64_q8', 64, q8=True)
attention('attention_q8', 96, q8=True)
attention('attention_g4_d128_q8', 128, n_heads=8, n_kv=2, G=4, q8=True)
kv_store_q8(128)
kv_store_q8(64, n_kv=3)


def flash(hd, n_heads=4, n_kv=2, pos0=37, ntok=45, ctx=128, window=0, softcap=0.0, half=False, q8=False, kind='fa'):
    rng = np.random.default_rng(3)
    q = rng.standard_normal((ntok, n_heads * hd)).astype(np.float32)
    kc32 = rng.standard_normal((ctx, n_kv * hd)).astype(np.float32)
    vc32 = rng.standard_normal((ctx, n_kv * hd)).astype(np.float32)
    kc = kc32.astype(np.float16) if half else kc32
    vc = vc32.astype(np.float16) if half else vc32
    kd, vd, stride = kc.astype(np.float32), vc.astype(np.float32), n_kv * hd
    if q8:
        (kc, kd), (vc, vd), stride = q8_rows(kc32, n_kv, hd), q8_rows(vc32, n_kv, hd), n_kv * hd + 4 * n_kv
    out = np.full((ntok, n_heads * hd), np.nan, np.float32)
    scale = 1 / np.sqrt(hd)
    # FlashParams: nHeads nKvHeads pos0 ntok scale qStride kStride oStride window softcap
    p = pack('iiiifiiiif', n_heads, n_kv, pos0, ntok, scale, n_heads * hd, stride, n_heads * hd, window, softcap)
    name = 'attention_%s_d%d%s' % (kind, hd, '_h' if half else '_q8' if q8 else '')
    E.dispatch(name, [q, kc, vc, out], p, (n_heads, (ntok + 31) // 32), (128,))
    ref = attention_ref(q, kd, vd, n_heads, n_kv, hd, pos0, ntok, scale, window, softcap)
    err = np.abs(out - ref).max()
    check(err < 2e-5, '%s pos0=%d ntok=%d window=%d softcap=%g: max err %.2e' % (name, pos0, ntok, window, softcap, err))


flash(64)
flash(128)
flash(128, half=True)
flash(64, pos0=0, ntok=32)
flash(128, window=20, pos0=60, ntok=40)
flash(64, softcap=5.0)
flash(128, n_heads=8, n_kv=1, ntok=7)
flash(256, ntok=33)
flash(128, q8=True)
flash(64, window=20, pos0=60, ntok=40, q8=True)
# attention_fb (K and V staged together, half staging for the f16 / q8 caches, padded rows): the same cases
for kw in ({'hd': 64}, {'hd': 128}, {'hd': 128, 'half': True}, {'hd': 64, 'pos0': 0, 'ntok': 32},
           {'hd': 128, 'window': 20, 'pos0': 60, 'ntok': 40}, {'hd': 64, 'softcap': 5.0},
           {'hd': 128, 'n_heads': 8, 'n_kv': 1, 'ntok': 7}, {'hd': 256, 'ntok': 33}, {'hd': 256, 'ntok': 33, 'half': True},
           {'hd': 64, 'half': True, 'window': 20, 'pos0': 60, 'ntok': 40}, {'hd': 128, 'q8': True},
           {'hd': 64, 'window': 20, 'pos0': 60, 'ntok': 40, 'q8': True}, {'hd': 256, 'q8': True, 'ntok': 9}):
    flash(kind='fb', **kw)


def split_q8(hd, n_heads=4, n_kv=2, pos0=300, ctx=320, n_split=5):
    """Flash decoding (attention_split_q8 + attention_combine) of one token over a q8 cache."""
    rng = np.random.default_rng(5)
    q = rng.standard_normal((1, n_heads * hd)).astype(np.float32)
    (kc, kd), (vc, vd) = q8_rows(rng.standard_normal((ctx, n_kv * hd)).astype(np.float32), n_kv, hd), \
        q8_rows(rng.standard_normal((ctx, n_kv * hd)).astype(np.float32), n_kv, hd)
    stride = n_kv * hd + 4 * n_kv
    keys = pos0 + 1
    chunk = (keys + n_split - 1) // n_split
    part = np.zeros(n_heads * n_split * (hd + 2), np.float32)
    scores = np.zeros(n_heads * ctx, np.float32)
    out = np.zeros((1, n_heads * hd), np.float32)
    scale = 1 / np.sqrt(hd)
    # AttnSplitParams: nHeads nKvHeads headDim pos0 ctx scale qStride kvStride window softcap nSplit chunk
    E.dispatch('attention_split_q8', [q, kc, vc, part, scores],
               pack('iiiiifiiifii', n_heads, n_kv, hd, pos0, ctx, scale, n_heads * hd, stride, 0, 0.0, n_split, chunk),
               (n_heads, n_split), (256,))
    E.dispatch('attention_combine', [part, out], pack('iii', n_heads, hd, n_split), (n_heads,), (256,))
    ref = attention_ref(q, kd, vd, n_heads, n_kv, hd, pos0, 1, scale)
    err = np.abs(out - ref).max()
    check(err < 1e-4, 'attention_split_q8 hd=%d, %d splits: max err %.2e' % (hd, n_split, err))


split_q8(128)


# ---- half-weight MMA prefill kernels (*_hmma) in every weight format, against numpy.
# Each format's bytes are random (with sane f16 / f32 scales) and decoded here following the block layouts of
# kernels.metal (and llama.cpp for the GGUF ones); the established *_mma kernel is checked against the same decoder
# first, so a decoder slip cannot hide a kernel error. The *_hmma kernels store each dequantized weight as half: they
# are checked strictly (2e-5 of max|ref|) against the reference with the weights rounded to half (numpy's f16 rounding
# is round-to-nearest-even, as the GPU's), and loosely (2e-3) against the exact weights.

def f16(rng, shape, lo, hi):
    return rng.uniform(lo, hi, shape).astype(np.float16)


def fmt_weights(fmt, rows, cols, rng):
    """(wq bytes, ws f32 scales or a dummy, w [rows, cols] f32 as the kernels dequantize it) of one tensor."""
    dummy = np.zeros(4, np.float32)
    if fmt == 'q4':
        wq, ws, w = q4_weights(rows, cols, rng)
        return wq, ws, w
    if fmt == 'q8':
        q = rng.integers(-127, 128, (rows, cols)).astype(np.int8)
        s = (rng.random((rows, cols // 32)) * 0.01 + 0.001).astype(np.float32)
        w = (q.astype(np.float32).reshape(rows, cols // 32, 32) * s[:, :, None]).reshape(rows, cols)
        return q.view(np.uint8).reshape(-1).copy(), s.reshape(-1).copy(), w
    if fmt == 'q4g' or fmt == 'q8g':
        nb = cols // 32
        q4 = fmt == 'q4g'
        blk = np.zeros((rows, nb, 18 if q4 else 34), np.uint8)
        d = f16(rng, (rows, nb), 0.002, 0.02)
        blk[:, :, 0:2] = d.view(np.uint8).reshape(rows, nb, 2)
        blk[:, :, 2:] = rng.integers(0, 256, (rows, nb, blk.shape[2] - 2))
        if q4:
            qs = blk[:, :, 2:]
            v = np.concatenate([qs & 15, qs >> 4], axis=2).astype(np.float32) - 8
        else:
            v = blk[:, :, 2:].view(np.int8).astype(np.float32)
        w = (v * d.astype(np.float32)[:, :, None]).reshape(rows, cols)
        return blk.reshape(-1).copy(), dummy, w
    nsb = cols // 256
    w = np.zeros((rows, nsb, 256), np.float32)
    if fmt == 'q2k':
        blk = np.zeros((rows, nsb, 84), np.uint8)
        blk[:, :, :80] = rng.integers(0, 256, (rows, nsb, 80))
        d = f16(rng, (rows, nsb), 0.005, 0.05)
        dmin = f16(rng, (rows, nsb), 0.0, 0.02)
        blk[:, :, 80:82] = d.view(np.uint8).reshape(rows, nsb, 2)
        blk[:, :, 82:84] = dmin.view(np.uint8).reshape(rows, nsb, 2)
        for j in range(16):
            s = blk[:, :, j]
            dl = d.astype(np.float32) * (s & 15).astype(np.float32)
            ml = dmin.astype(np.float32) * (s >> 4).astype(np.float32)
            off = 16 + (j >> 3) * 32 + (j & 1) * 16
            q = ((blk[:, :, off:off + 16] >> (2 * ((j & 7) >> 1))) & 3).astype(np.float32)
            w[:, :, 16 * j:16 * j + 16] = q * dl[:, :, None] - ml[:, :, None]
    elif fmt == 'q6k':
        blk = np.zeros((rows, nsb, 210), np.uint8)
        blk[:, :, :192] = rng.integers(0, 256, (rows, nsb, 192))
        blk[:, :, 192:208] = rng.integers(-64, 64, (rows, nsb, 16)).astype(np.int8).view(np.uint8)
        d = f16(rng, (rows, nsb), 0.0005, 0.005)
        blk[:, :, 208:210] = d.view(np.uint8).reshape(rows, nsb, 2)
        sc = blk[:, :, 192:208].view(np.int8).astype(np.float32)
        for j in range(16):
            h, g, lb = j >> 3, (j & 7) >> 1, j & 1
            ql = blk[:, :, 64 * h + 32 * (g & 1) + 16 * lb:][:, :, :16]
            qh = blk[:, :, 128 + 32 * h + 16 * lb:][:, :, :16]
            q = ((ql >> ((g >> 1) * 4)) & 15) | (((qh >> (2 * g)) & 3) << 4)
            dl = d.astype(np.float32) * sc[:, :, j]
            w[:, :, 16 * j:16 * j + 16] = (q.astype(np.float32) - 32) * dl[:, :, None]
    else:  # q4k / q5k
        q5 = fmt == 'q5k'
        blk = np.zeros((rows, nsb, 176 if q5 else 144), np.uint8)
        blk[:, :, 4:] = rng.integers(0, 256, (rows, nsb, blk.shape[2] - 4))
        d = f16(rng, (rows, nsb), 0.0005, 0.005)
        dmin = f16(rng, (rows, nsb), 0.0, 0.002)
        blk[:, :, 0:2] = d.view(np.uint8).reshape(rows, nsb, 2)
        blk[:, :, 2:4] = dmin.view(np.uint8).reshape(rows, nsb, 2)
        sc = blk[:, :, 4:16].astype(np.uint32)
        for j in range(8):
            if j < 4:
                s, m = sc[:, :, j] & 63, sc[:, :, j + 4] & 63
            else:
                s = (sc[:, :, j + 4] & 15) | ((sc[:, :, j - 4] >> 6) << 4)
                m = (sc[:, :, j + 4] >> 4) | ((sc[:, :, j] >> 6) << 4)
            dl = d.astype(np.float32) * s.astype(np.float32)
            ml = dmin.astype(np.float32) * m.astype(np.float32)
            qs = blk[:, :, (48 if q5 else 16) + (j >> 1) * 32:][:, :, :32]
            v = (qs >> ((j & 1) * 4)) & 15
            if q5:
                v = v | (((blk[:, :, 16:48] >> j) & 1) << 4)
            w[:, :, 32 * j:32 * j + 32] = v.astype(np.float32) * dl[:, :, None] - ml[:, :, None]
    return blk.reshape(-1).copy(), dummy, w.reshape(rows, cols)


GGUF = ('q4g', 'q8g', 'q4k', 'q5k')


def rel_errs(out, w, x_ref_fn):
    """(error against the half-rounded weights, error against the exact ones), relative to max|ref|."""
    exact = x_ref_fn(w.astype(np.float64))
    halfw = x_ref_fn(w.astype(np.float16).astype(np.float64))
    scale = np.abs(exact).max()
    return np.abs(out - halfw).max() / scale, np.abs(out - exact).max() / scale


def report(name, desc, out, w, fn, half):
    eh, ee = rel_errs(out, w, fn)
    if half:
        check(eh < 2e-5 and ee < 2e-3, '%s %s: rel err %.1e (half weights), %.1e (exact)' % (name, desc, eh, ee))
    else:
        check(ee < 1e-5, '%s %s: rel err %.1e' % (name, desc, ee))


def mm_test(fmt, kind, rows, cols, ntok, acc=False, bias=False, softcap=0.0):
    rng = np.random.default_rng(10)
    wq, ws, w = fmt_weights(fmt, rows, cols, rng)
    x = rng.standard_normal((ntok, cols)).astype(np.float32)
    out0 = rng.standard_normal((ntok, rows)).astype(np.float32) if acc else np.zeros((ntok, rows), np.float32)
    out = out0.copy()
    b = rng.standard_normal(rows).astype(np.float32)
    name = 'matmul_%s_%s' % (kind, fmt)
    E.dispatch(name, [wq, ws, x, out, np.ones(cols, np.float32), b],
               pack('iiiiffi', rows, cols, ntok, int(acc), 1e-6, softcap, int(bias)),
               ((rows + 63) // 64, (ntok + 31) // 32), (128,))

    def ref(wm):
        r = x.astype(np.float64) @ wm.T + (b if bias else 0)
        if softcap > 0:
            r = np.tanh(r / softcap) * softcap
        return r + (out0 if acc else 0)
    report(name, '%dx%d, %d tokens%s%s%s' % (rows, cols, ntok, ', acc' if acc else '', ', bias' if bias else '',
                                             ', softcap' if softcap else ''), out, w, ref, kind == 'hmma')


def mm3_test(fmt, kind, rows, cols, ntok, half=False, bias=False):
    rng = np.random.default_rng(11)
    ws_ = [fmt_weights(fmt, r, cols, rng) for r in rows]
    x = rng.standard_normal((ntok, cols)).astype(np.float32)
    outs = [np.full((ntok, rows[0]), np.nan, np.float32), np.full((ntok, rows[1]), np.nan, np.float32),
            np.full((ntok, rows[2]), np.nan, np.float16 if half else np.float32)]
    bs = [rng.standard_normal(r).astype(np.float32) for r in rows]
    name = 'matmul3_%s_%s%s' % (kind, fmt, '_h' if half else '')
    bufs = [ws_[0][0], ws_[0][1], ws_[1][0], ws_[1][1], ws_[2][0], ws_[2][1], x] + outs \
        + [np.ones(cols, np.float32)] + bs
    E.dispatch(name, bufs, pack('iiiiifi', rows[0], rows[1], rows[2], cols, ntok, 1e-6, int(bias)),
               ((sum(rows) + 63) // 64, (ntok + 31) // 32), (128,))
    w = np.concatenate([t[2] for t in ws_])
    b = np.concatenate(bs)
    out = np.concatenate([o.astype(np.float32) for o in outs], axis=1)
    if half:  # the v rows are rounded to half once more on the way out: compare them at half precision
        eh, ee = rel_errs(out, w, lambda wm: x.astype(np.float64) @ wm.T + (b if bias else 0))
        ok = eh < 1e-3 and ee < 2e-3
        check(ok, '%s %s, %d tokens%s: rel err %.1e (half weights, v as half), %.1e (exact)'
              % (name, rows, ntok, ', bias' if bias else '', eh, ee))
        return
    report(name, '%s, %d tokens%s' % (rows, ntok, ', bias' if bias else ''), out, w,
           lambda wm: x.astype(np.float64) @ wm.T + (b if bias else 0), kind == 'hmma')


def silu(g):
    return g / (1 + np.exp(-g))


def gelu(g):
    return 0.5 * g * (1 + np.tanh(0.7978845608028654 * (g + 0.044715 * g ** 3)))


def swiglu_test(fmt, kind, hidden, cols, ntok, use_gelu=False):
    rng = np.random.default_rng(12)
    x = rng.standard_normal((ntok, cols)).astype(np.float32)
    if fmt in GGUF:  # gate and up are separate views; the up view is bound as buffer(1)
        gq, _, gw = fmt_weights(fmt, hidden, cols, rng)
        uq, _, uw = fmt_weights(fmt, hidden, cols, rng)
        wq, ws, w = gq, uq, np.concatenate([gw, uw])
    else:
        wq, ws, w = fmt_weights(fmt, 2 * hidden, cols, rng)
    out = np.full((ntok, hidden), np.nan, np.float32)
    name = 'matmul_swiglu_%s_%s' % (kind, fmt)
    E.dispatch(name, [wq, ws, x, out, np.ones(cols, np.float32)], pack('iiifi', hidden, cols, ntok, 1e-6, int(use_gelu)),
               ((hidden + 31) // 32, (ntok + 31) // 32), (128,))
    act = gelu if use_gelu else silu

    def ref(wm):
        y = x.astype(np.float64) @ wm.T
        return act(y[:, :hidden]) * y[:, hidden:]
    report(name, 'hidden %d x %d, %d tokens%s' % (hidden, cols, ntok, ', gelu' if use_gelu else ''), out, w, ref,
           kind == 'hmma')


def moe_tiles(counts, experts, rng):
    """Slots grouped by expert as moe_sort leaves them: order[] and tiles [n, (expert, start, count)...]."""
    order, tiles, start = [], [], 0
    for e, c in zip(experts, counts):
        order += list(range(start, start + c))
        tiles += [e, start, c]
        start += c
    perm = rng.permutation(start)  # the slots of a tile are scattered over the chunk's tokens
    return np.array([perm[i] for i in order], np.int32), np.array([len(counts)] + tiles, np.int32), start


def moe_test(fmt, kind, m, dim, n_exp=4, top_k=2, counts=(5, 32, 16, 1), experts=(2, 0, 3, 2)):
    rng = np.random.default_rng(13)
    order, tiles, slots = moe_tiles(counts, experts, rng)
    ntok = (slots + top_k - 1) // top_k
    x = rng.standard_normal((ntok, dim)).astype(np.float32)
    if fmt in GGUF:  # gate experts [E * m] in buffer 0, up experts [E * m] in buffer 1
        gq, _, gw = fmt_weights(fmt, n_exp * m, dim, rng)
        uq, _, uw = fmt_weights(fmt, n_exp * m, dim, rng)
        wq, ws = gq, uq
        wg, wu = gw.reshape(n_exp, m, dim), uw.reshape(n_exp, m, dim)
    else:            # expert e: rows e * 2m .. (gate m, then up m)
        wq, ws, w2 = fmt_weights(fmt, n_exp * 2 * m, dim, rng)
        w2 = w2.reshape(n_exp, 2 * m, dim)
        wg, wu = w2[:, :m], w2[:, m:]
    hid = np.full((ntok * top_k, m), np.nan, np.float32)
    name = 'moe_up_%s_%s' % (kind, fmt)
    E.dispatch(name, [wq, ws, x, order, tiles, hid], pack('iii', 2 * m, dim, top_k),
               ((m + 31) // 32, len(counts)), (128,))
    slot_e = np.zeros(slots, int)
    for e, s0, c in tiles[1:].reshape(-1, 3):
        slot_e[order[s0:s0 + c]] = e

    def up_ref(g, u):
        out = np.zeros((slots, m))
        for s in range(slots):
            xv = x[s // top_k].astype(np.float64)
            out[s] = silu(g[slot_e[s]] @ xv) * (u[slot_e[s]] @ xv)
        return out
    exact = up_ref(wg.astype(np.float64), wu.astype(np.float64))
    halfw = up_ref(wg.astype(np.float16).astype(np.float64), wu.astype(np.float16).astype(np.float64))
    got = hid[:slots]
    sc = np.abs(exact).max()
    eh, ee = np.abs(got - halfw).max() / sc, np.abs(got - exact).max() / sc
    ok = (eh < 2e-5 and ee < 2e-3) if kind == 'hmma' else ee < 1e-5
    check(ok, '%s m=%d dim=%d tiles %s: rel err %.1e (half weights), %.1e (exact)' % (name, m, dim, list(counts), eh, ee))

    # down: ys[s] = rw[s] * W_e h[s], rows = dim, cols = m (a fresh h, so the check is independent of moe_up)
    h = rng.standard_normal((slots, m)).astype(np.float32)
    rw = rng.random(slots).astype(np.float32)
    if fmt in GGUF and m % 256 != 0 and fmt in ('q4k', 'q5k'):
        return
    dq, ds, dw = fmt_weights(fmt, n_exp * dim, m, rng)
    dw = dw.reshape(n_exp, dim, m)
    ys = np.full((slots, dim), np.nan, np.float32)
    name = 'moe_down_%s_%s' % (kind, fmt)
    E.dispatch(name, [dq, ds, h, order, tiles, rw, ys], pack('iii', dim, m, top_k), ((dim + 63) // 64, len(counts)),
               (128,))

    def down_ref(wd):
        return np.stack([rw[s] * (wd[slot_e[s]] @ h[s].astype(np.float64)) for s in range(slots)])
    exact = down_ref(dw.astype(np.float64))
    halfw = down_ref(dw.astype(np.float16).astype(np.float64))
    sc = np.abs(exact).max()
    eh, ee = np.abs(ys - halfw).max() / sc, np.abs(ys - exact).max() / sc
    ok = (eh < 2e-5 and ee < 2e-3) if kind == 'hmma' else ee < 1e-5
    check(ok, '%s dim=%d m=%d tiles %s: rel err %.1e (half weights), %.1e (exact)' % (name, dim, m, list(counts), eh, ee))


for fmt in ('q4', 'q8', 'q2k', 'q6k', 'q4g', 'q8g', 'q4k', 'q5k'):
    kq = fmt in ('q2k', 'q6k', 'q4k', 'q5k')  # 256-weight super-blocks
    cols = 256 if kq else 128
    for kind in ('mma', 'hmma'):
        mm_test(fmt, kind, 96, cols, 37)                       # partial row and token tiles: staging epilogue
        mm_test(fmt, kind, 128, 2 * cols, 64, acc=True)        # full tiles: stored straight from the accumulators
        mm_test(fmt, kind, 64, cols, 32, bias=True)            # full tile with a bias: staging epilogue
        mm3_test(fmt, kind, (128, 64, 64), cols, 64)           # q / k tiles direct, v tile staged
        mm3_test(fmt, kind, (64, 32, 32), cols, 37, bias=True)
        mm3_test(fmt, kind, (64, 32, 32), cols, 40, half=True)
        swiglu_test(fmt, kind, 48, cols, 37)
        moe_test(fmt, kind, 64 if not kq else 256, cols)
mm_test('q4', 'hmma', 96, 256, 40, softcap=3.0)
mm_test('q4', 'hmma', 64, 128, 7)                              # one 8-token block of the tile only
swiglu_test('q8', 'hmma', 32, 128, 64, use_gelu=True)
moe_test('q4', 'hmma', 96, 128, counts=(32, 32, 3), experts=(1, 3, 1))
sys.exit(1 if failures else 0)
