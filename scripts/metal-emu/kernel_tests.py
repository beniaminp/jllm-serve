#!/usr/bin/env python3
"""Kernel-level tests on the Metal emulator against numpy references (python3 -I scripts/metal-emu/kernel_tests.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from emu import Emu, pack
import ggml_ref

E = Emu()
failures = 0
# python3 kernel_tests.py [fmt ...]: only the GGUF block-format tests of the listed formats (q41 q50 q51 q3k iq4nl iq4xs
# mxfp4 f16w bf16w q4g q8g q4k q5k ...); no arguments: everything
ONLY = sys.argv[1:]


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


def paged(caches, page):
    """The paged KV cache's layout (docs/24): the per-position arrays of `caches` (each [ctx, ...]) scattered, 2^page
    positions to a block, over a pool in shuffled block order with unused blocks between them filled with garbage (NaN, or
    0x5A bytes), so that a wrong address shows; returns the pools and the block table (int32 per logical block). With
    page == 0 the arrays as they are (the flat cache) and a one-int dummy table."""
    if page == 0:
        return list(caches) + [np.zeros(1, np.int32)]
    block = 1 << page
    ctx = caches[0].shape[0]
    nb = (ctx + block - 1) // block
    total = nb + 3
    ids = np.random.default_rng(page * 7919 + ctx).permutation(total)[:nb].astype(np.int32)
    out = []
    for c in caches:
        pool = np.empty((total * block,) + c.shape[1:], c.dtype)
        pool[...] = np.nan if c.dtype.kind == 'f' else 0x5A
        for i in range(nb):
            seg = c[i * block:(i + 1) * block]
            pool[ids[i] * block:ids[i] * block + len(seg)] = seg
        out.append(pool)
    return out + [ids]


def paged_rows(pool, tbl, page, first, n):
    """The rows of positions [first, first + n) of a paged cache."""
    block = 1 << page
    if n <= 0:
        return np.zeros((0,), pool.dtype)
    return np.stack([pool[tbl[p // block] * block + p % block] for p in range(first, first + n)])


def attention(name, hd, n_heads=4, n_kv=2, pos0=5, ntok=3, ctx=16, G=1, q8=False, page=0):
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
    # AttnParams: nHeads nKvHeads qkDim vDim pos0 ctx scale qStride kStride vStride oStride window tokenOffset softcap pageShift
    p = pack('iiiiiifiiiiiifi', n_heads, n_kv, hd, hd, pos0, ctx, scale, n_heads * hd, stride, stride,
             n_heads * hd, 0, 0, 0.0, page)
    kc, vc, tbl = paged([kc, vc], page)
    E.dispatch(name, [q, kc, vc, out, scores, tbl], p, (n_heads // G, ntok), (256,))
    ref = attention_ref(q, kd, vd, n_heads, n_kv, hd, pos0, ntok, scale)
    err = np.abs(out - ref).max()
    check(err < 1e-4, '%s hd=%d%s: max err %.2e' % (name, hd, ', paged (blocks of %d)' % (1 << page) if page else '', err))


def kv_store_q8(hd, n_kv=2, ntok=5, pos0=3, ctx=12, page=0):
    """kv_store_q8 quantizes the staging rows of k and v into the cache rows [pos0, pos0 + ntok) bit for bit as
    numpy (and the engine's Java side) does, and leaves the other rows alone."""
    rng = np.random.default_rng(4)
    k = (rng.standard_normal((ntok, n_kv * hd)) * rng.random((ntok, 1)) * 4).astype(np.float32)
    v = rng.standard_normal((ntok, n_kv * hd)).astype(np.float32)
    if ntok > 1:
        k[1, :hd] = 0  # an all-zero row: scale 0, values 0
    stride = n_kv * hd + 4 * n_kv
    kc = np.full((ctx, stride), 0x5A, np.uint8)
    vc = np.full((ctx, stride), 0x5A, np.uint8)
    # KvStoreParams: headDim kvDim nKvHeads stride pos pageShift; flat: the cache is bound at row pos0
    want_k, kd = q8_rows(k, n_kv, hd)
    want_v, vd = q8_rows(v, n_kv, hd)
    if page:  # paged: the pools bound at row 0, rows of positions pos0 .. pos0 + ntok - 1 through the block table
        kb, vb, tbl = paged([kc, vc], page)
        E.dispatch('kv_store_q8', [k, v, kb, vb, tbl], pack('iiiiii', hd, n_kv * hd, n_kv, stride, pos0, page),
                   (2 * n_kv, ntok), (32,))
        ok = (paged_rows(kb, tbl, page, pos0, ntok) == want_k).all() and (paged_rows(vb, tbl, page, pos0, ntok) == want_v).all()
        ok = ok and (paged_rows(kb, tbl, page, 0, pos0) == 0x5A).all() and (paged_rows(kb, tbl, page, pos0 + ntok, ctx - pos0 - ntok) == 0x5A).all()
    else:
        kview = kc[pos0:]
        vview = vc[pos0:]
        kb = np.ascontiguousarray(kview)
        vb = np.ascontiguousarray(vview)
        E.dispatch('kv_store_q8', [k, v, kb, vb, np.zeros(1, np.int32)], pack('iiiiii', hd, n_kv * hd, n_kv, stride, 0, 0),
                   (2 * n_kv, ntok), (32,))
        ok = (kb[:ntok] == want_k).all() and (vb[:ntok] == want_v).all() and (kb[ntok:] == 0x5A).all()
    err = max(np.abs(kd - k).max() / (np.abs(k).max() / 254), np.abs(vd - v).max() / (np.abs(v).max() / 254))
    check(ok and err <= 1.0001, 'kv_store_q8 hd=%d%s: rows as numpy %s, max error %.3f of max|x| / 254'
          % (hd, ', paged (blocks of %d)' % (1 << page) if page else '', 'yes' if ok else 'NO', err))


def kv_store_pg(half, n_kv=2, hd=64, ntok=21, pos0=13, ctx=70, page=4):
    """kv_store_pg (the paged f32 / f16 cache's store): the staging rows of k and v land in the rows of positions pos0 ..
    pos0 + ntok - 1 through the block table, as float32 (f16: rounded to half like numpy's astype), the other rows untouched."""
    rng = np.random.default_rng(11)
    kvd = n_kv * hd
    k = rng.standard_normal((ntok, kvd)).astype(np.float32)
    v = rng.standard_normal((ntok, kvd)).astype(np.float32)
    dt = np.float16 if half else np.float32
    kc = np.full((ctx, kvd), 7, dt)
    vc = np.full((ctx, kvd), 7, dt)
    kb, vb, tbl = paged([kc, vc], page)
    channel_threads = 64  # MetalConstants.CHANNEL_THREADS
    chunks = (kvd + channel_threads - 1) // channel_threads
    # KvPagedParams: kvDim stride pos pageShift
    E.dispatch('kv_store_pg_h' if half else 'kv_store_pg', [k, v, kb, vb, tbl], pack('iiii', kvd, kvd, pos0, page),
               (2 * chunks, ntok), (channel_threads,))
    ok = np.array_equal(paged_rows(kb, tbl, page, pos0, ntok), k.astype(dt)) and np.array_equal(paged_rows(vb, tbl, page, pos0, ntok), v.astype(dt))
    ok = ok and (paged_rows(kb, tbl, page, 0, pos0) == 7).all() and (paged_rows(vb, tbl, page, pos0 + ntok, ctx - pos0 - ntok) == 7).all()
    check(ok, 'kv_store_pg%s hd=%d kvHeads=%d, %d rows at %d, blocks of %d: rows as numpy' % ('_h' if half else '', hd, n_kv, ntok, pos0, 1 << page))


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


if not ONLY:
    for cols in (128, 1024):
        matmul('matmul_q4', 128, cols, 3, (128 // 4, 1))
        matmul('matmul_pf_q4', 128, cols, 9, ((128 + 7) // 8, 2))
        matmul('matmul_mma_q4', 128, cols, 40, (2, 2), threads=128, tgmem=0)
    matmul('matmul_q4', 128, 128, 3, (128 // 4, 1), bias=True, acc=True)
    matmul('matmul_pf_q4', 96, 128, 9, ((96 + 7) // 8, 2), bias=True)
    matmul('matmul_mma_q4', 96, 128, 40, (2, 2), threads=128, tgmem=0, bias=True, acc=True)

if not ONLY:
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
    # the paged cache (docs/24): the same kernels over a shuffled pool and a block table
    attention('attention_g1_d64', 64, ctx=50, pos0=30, ntok=9, page=4)
    attention('attention_g1_d128', 128, ctx=70, pos0=41, ntok=5, page=5)
    attention('attention', 96, ctx=40, pos0=17, page=4)
    attention('attention_g1_d128_q8', 128, ctx=70, pos0=33, ntok=6, q8=True, page=4)
    attention('attention_g4_d128_q8', 128, n_heads=8, n_kv=2, G=4, q8=True, ctx=45, pos0=31, page=4)
    attention('attention_g1_d64', 64, ctx=300, pos0=290, ntok=4, page=6)
    kv_store_q8(128, page=4)
    kv_store_q8(64, n_kv=3, ntok=40, pos0=11, ctx=70, page=4)
    kv_store_q8(128, ntok=1, pos0=31, ctx=70, page=5)
    for _half in (False, True):
        kv_store_pg(_half)
        kv_store_pg(_half, hd=128, ntok=1, pos0=47, page=5)
        kv_store_pg(_half, n_kv=3, hd=32, ntok=64, pos0=0, ctx=100, page=4)


def flash(hd, n_heads=4, n_kv=2, pos0=37, ntok=45, ctx=128, window=0, softcap=0.0, half=False, q8=False, kind='fa', page=0):
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
    # FlashParams: nHeads nKvHeads pos0 ntok scale qStride kStride oStride window softcap pageShift
    p = pack('iiiifiiiifi', n_heads, n_kv, pos0, ntok, scale, n_heads * hd, stride, n_heads * hd, window, softcap, page)
    name = 'attention_%s_d%d%s' % (kind, hd, '_h' if half else '_q8' if q8 else '')
    kc, vc, tbl = paged([kc, vc], page)
    E.dispatch(name, [q, kc, vc, out, tbl], p, (n_heads, (ntok + 31) // 32), (128,))
    ref = attention_ref(q, kd, vd, n_heads, n_kv, hd, pos0, ntok, scale, window, softcap)
    err = np.abs(out - ref).max()
    check(err < 2e-5, '%s pos0=%d ntok=%d window=%d softcap=%g%s: max err %.2e'
          % (name, pos0, ntok, window, softcap, ', paged (blocks of %d)' % (1 << page) if page else '', err))


if not ONLY:
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
    # the paged cache: both forms, every cache type, windows and soft-caps, chunk starts inside a block
    for _kind in ('fa', 'fb'):
        for kw in ({'hd': 64}, {'hd': 128, 'half': True}, {'hd': 128, 'q8': True}, {'hd': 64, 'window': 20, 'pos0': 60, 'ntok': 40},
                   {'hd': 128, 'softcap': 5.0, 'pos0': 17}, {'hd': 256, 'ntok': 33, 'half': True}, {'hd': 64, 'pos0': 0, 'ntok': 32}):
            flash(kind=_kind, page=4, **kw)
        flash(64, kind=_kind, page=6, pos0=100, ntok=45, ctx=192)


def split_q8(hd, n_heads=4, n_kv=2, pos0=300, ctx=320, n_split=5, page=0):
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
    # AttnSplitParams: nHeads nKvHeads headDim pos0 ctx scale qStride kvStride window softcap nSplit chunk pageShift
    kc, vc, tbl = paged([kc, vc], page)
    E.dispatch('attention_split_q8', [q, kc, vc, part, scores, tbl],
               pack('iiiiifiiifiii', n_heads, n_kv, hd, pos0, ctx, scale, n_heads * hd, stride, 0, 0.0, n_split, chunk, page),
               (n_heads, n_split), (256,))
    E.dispatch('attention_combine', [part, out], pack('iii', n_heads, hd, n_split), (n_heads,), (256,))
    ref = attention_ref(q, kd, vd, n_heads, n_kv, hd, pos0, 1, scale)
    err = np.abs(out - ref).max()
    check(err < 1e-4, 'attention_split_q8 hd=%d, %d splits%s: max err %.2e' % (hd, n_split, ', paged' if page else '', err))


if not ONLY:
    split_q8(128)
    split_q8(128, page=4)
    split_q8(64, page=5, pos0=250, n_split=3)


def attention_dec(hd, G, kv='f32', n_heads=8, n_kv=2, pos0=100, n_split=1, window=0, softcap=0.0, ctx=320, wide=True, page=0):
    """attention_dec (section 4b): one token over a cache, G query heads per threadgroup, n_split key ranges (merged by
    attention_combine when n_split > 1) against a float64 numpy reference on the values the cache holds."""
    rng = np.random.default_rng(6 + hd + G)
    q = rng.standard_normal((1, n_heads * hd)).astype(np.float32)
    k0 = rng.standard_normal((ctx, n_kv * hd)).astype(np.float32)
    v0 = rng.standard_normal((ctx, n_kv * hd)).astype(np.float32)
    suffix = ''
    if kv == 'q8':
        (kc, kd), (vc, vd), stride = q8_rows(k0, n_kv, hd), q8_rows(v0, n_kv, hd), n_kv * hd + 4 * n_kv
        suffix = '_q8'
    elif kv == 'f16':
        kc, vc, stride = k0.astype(np.float16), v0.astype(np.float16), n_kv * hd
        kd, vd = kc.astype(np.float32), vc.astype(np.float32)
        suffix = '_h' if wide else '_h4'   # 16-byte loads (8 halves per lane chunk) or 8-byte ones
    else:
        kc, vc, stride, kd, vd = k0, v0, n_kv * hd, k0, v0
    keys = pos0 + 1
    first = max(0, keys - window) if window > 0 else 0
    part = np.zeros(n_heads * n_split * (hd + 2), np.float32)
    out = np.zeros((1, n_heads * hd), np.float32)
    scale = 1 / np.sqrt(hd)
    # AttnSplitParams: nHeads nKvHeads headDim pos0 ctx scale qStride kvStride window softcap nSplit chunk pageShift
    kc, vc, tbl = paged([kc, vc], page)
    E.dispatch('attention_dec_g%d_d%d%s' % (G, hd, suffix), [q, kc, vc, part, out, tbl],
               pack('iiiiifiiifiii', n_heads, n_kv, hd, pos0, ctx, scale, n_heads * hd, stride, window, softcap, n_split,
                    0, page), (n_heads // G, (n_split + 3) // 4), (128,))
    if n_split > 1:
        E.dispatch('attention_combine', [part, out], pack('iii', n_heads, hd, n_split), (n_heads,), (256,))
    ref = attention_ref(q, kd, vd, n_heads, n_kv, hd, pos0, 1, scale, window, softcap)
    err = np.abs(out - ref).max()
    check(err < 1e-4, 'attention_dec_g%d_d%d%s, %d keys, %d split(s)%s%s%s: max err %.2e'
          % (G, hd, suffix, keys, n_split, ', window %d' % window if window else '', ', softcap' if softcap else '',
             ', paged (blocks of %d)' % (1 << page) if page else '', err))


for _hd in (64, 128, 256):
    for _kv in ('f32', 'f16', 'q8'):
        for _g in (1, 2) if _hd == 256 else (1, 2, 4):
            attention_dec(_hd, _g, _kv)                               # 101 keys: three full blocks and a partial one
            attention_dec(_hd, _g, _kv, pos0=300, n_split=3)          # merged partials
attention_dec(128, 2, pos0=300, n_split=3, kv='f16', wide=False)       # the 4-element half variant
attention_dec(64, 2, pos0=100, kv='f16', wide=False)
attention_dec(128, 2, pos0=37)                                        # fewer keys than one block
attention_dec(128, 2, pos0=0)                                         # one key
attention_dec(128, 2, pos0=300, window=64)                            # sliding window
attention_dec(64, 2, pos0=250, n_split=7, softcap=3.0)                # Gemma's soft-cap, a split count that is no multiple of 4
attention_dec(128, 1, pos0=200, n_split=40)                           # splits with no keys at all
for _hd in (64, 128, 256):                                              # the paged cache (docs/24)
    for _kv in ('f32', 'f16', 'q8'):
        attention_dec(_hd, 2, _kv, page=4)
        attention_dec(_hd, 1, _kv, pos0=300, n_split=3, page=5)
attention_dec(128, 2, pos0=300, window=64, page=4)
attention_dec(64, 2, pos0=250, n_split=7, softcap=3.0, page=6)
attention_dec(128, 2, pos0=0, page=4)


def gdn_conv(channels=40, kc=4, ntok=6, in_stride=50):
    """Qwen3-Next causal conv: y = silu(conv over (state, x)), the state advanced past the chunk."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((ntok, in_stride)).astype(np.float32)
    w = rng.standard_normal((channels, kc)).astype(np.float32)
    state = rng.standard_normal((channels, kc - 1)).astype(np.float32)
    y = np.zeros((ntok, channels), np.float32)
    seq = np.concatenate([state.T, x[:, :channels]], axis=0).astype(np.float64)  # [kc - 1 + ntok, channels]
    ref = np.stack([np.sum(seq[t:t + kc] * w.T, axis=0) for t in range(ntok)])
    ref = ref / (1 + np.exp(-ref))
    want_state = seq[-(kc - 1):].T
    s = state.copy()
    snap = np.zeros_like(state)
    snap_tok = ntok // 2
    E.dispatch('gdn_conv', [x, w, s, y, snap], pack('iiiiii', channels, kc, ntok, in_stride, channels, snap_tok),
               ((channels + 63) // 64,), (64,))
    # the snapshot is the conv state after token snap_tok: the last kc - 1 inputs up to and including it
    want_snap = seq[snap_tok + 1:snap_tok + kc].T
    err = max(np.abs(y - ref).max(), np.abs(s - want_state).max(), np.abs(snap - want_snap).max())
    check(err < 1e-5, 'gdn_conv channels=%d kernel=%d ntok=%d: max err %.2e' % (channels, kc, ntok, err))


def gdn_rule(nk=2, nv=4, dk=64, dv=12, ntok=5, tiled=0):
    """The gated delta rule: l2norm(q) / sqrt(dk), l2norm(k), beta = sigmoid(b), g = A softplus(a + dt_bias),
    S = exp(g) S; S += k ((v - S^T k) beta)^T; o = S^T q, against a float64 loop (state stored [head][j][dk]).
    tiled=1: value head h reads key head h % nk (Qwen3.5 GGUF order) instead of h // (nv // nk)."""
    rng = np.random.default_rng(5)
    kw = nk * dk
    conv = 2 * kw + nv * dv
    proj_stride = conv + nv * dv + 2 * nv
    qkv = rng.standard_normal((ntok, conv)).astype(np.float32)
    proj = rng.standard_normal((ntok, proj_stride)).astype(np.float32)
    A = -rng.uniform(0.1, 4, nv).astype(np.float32)
    dtb = rng.standard_normal(nv).astype(np.float32)
    S0 = (0.1 * rng.standard_normal((nv, dv, dk))).astype(np.float32)
    out = np.zeros((ntok, nv * dv), np.float32)
    S = S0.astype(np.float64).transpose(0, 2, 1).copy()  # [h, dk, dv] like transformers
    ref = np.zeros((ntok, nv, dv))
    r = nv // nk
    b_off = conv + nv * dv
    for t in range(ntok):
        for h in range(nv):
            kh = h % nk if tiled else h // r
            q = qkv[t, kh * dk:(kh + 1) * dk].astype(np.float64)
            k = qkv[t, kw + kh * dk:kw + (kh + 1) * dk].astype(np.float64)
            v = qkv[t, 2 * kw + h * dv:2 * kw + (h + 1) * dv].astype(np.float64)
            q = q / np.sqrt(q @ q + 1e-6) / np.sqrt(dk)
            k = k / np.sqrt(k @ k + 1e-6)
            beta = 1 / (1 + np.exp(-proj[t, b_off + h]))
            g = A[h] * np.log1p(np.exp(proj[t, b_off + nv + h] + dtb[h]))
            S[h] *= np.exp(g)
            d = (v - S[h].T @ k) * beta
            S[h] += np.outer(k, d)
            ref[t, h] = S[h].T @ q
        if t == ntok // 2:
            S_snap = S.copy()
    st = S0.copy()
    snap = np.zeros_like(S0)
    snap_tok = ntok // 2
    E.dispatch('gdn_rule', [qkv, proj, A, dtb, st, out, snap],
               pack('iiiiiiiifii', nk, nv, dk, dv, ntok, conv, proj_stride, b_off, 1e-6, tiled, snap_tok),
               ((dv + 3) // 4, nv), (128,))
    err = max(np.abs(out - ref.reshape(ntok, -1)).max(), np.abs(st - S.transpose(0, 2, 1)).max(),
              np.abs(snap - S_snap.transpose(0, 2, 1)).max())
    check(err < 1e-5, 'gdn_rule nk=%d nv=%d dk=%d dv=%d ntok=%d tiled=%d: max err %.2e' % (nk, nv, dk, dv, ntok, tiled, err))


def gdn_norm_gate(nv=3, dv=40, ntok=4, eps=1e-6):
    rng = np.random.default_rng(6)
    z_off = 7
    proj_stride = z_off + nv * dv + 5
    o = rng.standard_normal((ntok, nv * dv)).astype(np.float32)
    proj = rng.standard_normal((ntok, proj_stride)).astype(np.float32)
    w = rng.standard_normal(dv).astype(np.float32)
    x = o.reshape(ntok, nv, dv).astype(np.float64)
    z = proj[:, z_off:z_off + nv * dv].reshape(ntok, nv, dv).astype(np.float64)
    ref = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w * (z / (1 + np.exp(-z)))
    E.dispatch('gdn_norm_gate', [o, proj, w], pack('iiiif', dv, nv, proj_stride, z_off, eps), (ntok * nv,), (64,))
    err = np.abs(o - ref.reshape(ntok, -1)).max()
    check(err < 1e-5, 'gdn_norm_gate nv=%d dv=%d: max err %.2e' % (nv, dv, err))


if not ONLY:
    gdn_conv()
    gdn_conv(channels=70, kc=3, ntok=1, in_stride=80)
    gdn_rule()
    gdn_rule(nk=2, nv=2, dk=128, dv=32, ntok=3)
    gdn_rule(nk=1, nv=3, dk=32, dv=5, ntok=7)
    gdn_rule(nk=2, nv=6, dk=64, dv=12, ntok=4, tiled=1)
    gdn_rule(nk=4, nv=8, dk=32, dv=8, ntok=3, tiled=1)
    gdn_norm_gate()
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
    if fmt in ggml_ref.NEW:   # Q4_1 Q5_0 Q5_1 Q3_K IQ4_NL IQ4_XS MXFP4 F16 BF16: ggml_ref.py (checked against ggml-quants.c)
        return ggml_ref.weights(fmt, rows, cols, rng)
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


GGUF = ('q4g', 'q8g', 'q4k', 'q5k') + ggml_ref.NEW
KQ256 = ('q2k', 'q6k', 'q4k', 'q5k', 'q3k', 'iq4xs')  # formats of 256-weight super-blocks
# Formats whose *_hmma kernels are exact: the half tile holds the integers and the per-block scale is applied in f32
# after the multiply-accumulates (they are checked as strictly as the f32 *_mma kernels, against the exact weights).
HMMA_EXACT = ('q4', 'q8', 'q4g', 'q8g')


def rounds_half(fmt, kind):
    """True for a kernel that rounds each dequantized weight to half (hmma on the k-quants)."""
    return kind == 'hmma' and fmt not in HMMA_EXACT


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
                                             ', softcap' if softcap else ''), out, w, ref, rounds_half(fmt, kind))


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
           lambda wm: x.astype(np.float64) @ wm.T + (b if bias else 0), rounds_half(fmt, kind))


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
           rounds_half(fmt, kind))


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
    ok = (eh < 2e-5 and ee < 2e-3) if rounds_half(fmt, kind) else ee < 1e-5
    check(ok, '%s m=%d dim=%d tiles %s: rel err %.1e (half weights), %.1e (exact)' % (name, m, dim, list(counts), eh, ee))

    # down: ys[s] = rw[s] * W_e h[s], rows = dim, cols = m (a fresh h, so the check is independent of moe_up)
    h = rng.standard_normal((slots, m)).astype(np.float32)
    rw = rng.random(slots).astype(np.float32)
    if fmt in GGUF and m % 256 != 0 and fmt in KQ256:
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
    ok = (eh < 2e-5 and ee < 2e-3) if rounds_half(fmt, kind) else ee < 1e-5
    check(ok, '%s dim=%d m=%d tiles %s: rel err %.1e (half weights), %.1e (exact)' % (name, dim, m, list(counts), eh, ee))


def plain_test(fmt, rows=96, cols=None, ntok=3):
    """matmul_<fmt> (the decode kernel, mm_row_gguf), matmul_pf_<fmt> (the tiled prefill kernel, mm_tile_gguf),
    matmul_swiglu_<fmt> and the embedding lookup embed_token_<fmt>, which share gg_unit with the mma kernels but walk
    the units differently."""
    cols = cols or (256 if fmt in KQ256 else 128)
    rng = np.random.default_rng(20)
    wq, ws, w = fmt_weights(fmt, rows, cols, rng)
    x = rng.standard_normal((ntok, cols)).astype(np.float32)
    ref = x.astype(np.float64) @ w.T.astype(np.float64)
    for name, grid in (('matmul_' + fmt, (rows // 4, 1)), ('matmul_pf_' + fmt, ((rows + 7) // 8, (ntok + 7) // 8))):
        out = np.zeros((ntok, rows), np.float32)
        E.dispatch(name, [wq, ws, x, out, np.ones(cols, np.float32), np.zeros(rows, np.float32)],
                   pack('iiiiffi', rows, cols, ntok, 0, 1e-6, 0.0, 0), grid, (128,))
        err = np.abs(out - ref).max() / np.abs(ref).max()
        check(err < 1e-5, '%s %dx%d, %d tokens: rel err %.2e' % (name, rows, cols, ntok, err))
    hidden = 48
    gq, _, gw = fmt_weights(fmt, hidden, cols, rng)
    uq, _, uw = fmt_weights(fmt, hidden, cols, rng)
    out = np.full((ntok, hidden), np.nan, np.float32)
    E.dispatch('matmul_swiglu_' + fmt, [gq, uq, x, out, np.ones(cols, np.float32)], pack('iiifi', hidden, cols, ntok, 1e-6, 0),
               ((hidden + 3) // 4, 1), (128,))
    g = x.astype(np.float64) @ gw.T.astype(np.float64)
    u = x.astype(np.float64) @ uw.T.astype(np.float64)
    want = silu(g) * u
    err = np.abs(out - want).max() / np.abs(want).max()
    check(err < 1e-5, 'matmul_swiglu_%s hidden %d x %d, %d tokens: rel err %.2e' % (fmt, hidden, cols, ntok, err))
    # the embedding row of token 5, scaled by 1.5: every unit of the row decoded by embed_token
    tok = np.array([5], np.int32)
    xo = np.zeros(cols, np.float32)
    E.dispatch('embed_token_' + fmt, [wq, ws, tok, xo], pack('iif', cols, 0, 1.5), (1,), (128,))
    err = np.abs(xo - 1.5 * w[5]).max() / np.abs(w[5]).max()
    check(err < 1e-6, 'embed_token_%s row 5: rel err %.2e' % (fmt, err))


for fmt in ggml_ref.NEW:
    if not ONLY or fmt in ONLY:
        plain_test(fmt)

for fmt in ('q4', 'q8', 'q2k', 'q6k', 'q4g', 'q8g', 'q4k', 'q5k') + ggml_ref.NEW:
    if ONLY and fmt not in ONLY:
        continue
    kq = fmt in KQ256  # 256-weight super-blocks
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
if not ONLY:
    mm_test('q4', 'hmma', 96, 256, 40, softcap=3.0)
    mm_test('q4', 'hmma', 64, 128, 7)                              # one 8-token block of the tile only
    swiglu_test('q8', 'hmma', 32, 128, 64, use_gelu=True)
    moe_test('q4', 'hmma', 96, 128, counts=(32, 32, 3), experts=(1, 3, 1))
def split_chunk(hd=64, n_heads=4, n_kv=2, pos0=200, ctx=256, n_split=7, window=0):
    """attention_split with chunk = 0 (the keys per split derived in the kernel, as a recorded decode step passes it)
    gives the bytes of the explicit ceil(keys / nSplit)."""
    rng = np.random.default_rng(6)
    q = rng.standard_normal((1, n_heads * hd)).astype(np.float32)
    kc = rng.standard_normal((ctx, n_kv * hd)).astype(np.float32)
    vc = rng.standard_normal((ctx, n_kv * hd)).astype(np.float32)
    ln = pos0 + 1
    keys = ln - (max(0, ln - window) if window > 0 else 0)
    outs = []
    for chunk in ((keys + n_split - 1) // n_split, 0):
        part = np.zeros(n_heads * n_split * (hd + 2), np.float32)
        scores = np.zeros(n_heads * ctx, np.float32)
        out = np.zeros((1, n_heads * hd), np.float32)
        p = pack('iiiiifiiifiii', n_heads, n_kv, hd, pos0, ctx, 1 / np.sqrt(hd), n_heads * hd, n_kv * hd, window, 0.0,
                 n_split, chunk, 0)
        E.dispatch('attention_split', [q, kc, vc, part, scores, np.zeros(1, np.int32)], p, (n_heads, n_split), (256,))
        E.dispatch('attention_combine', [part, out], pack('iii', n_heads, hd, n_split), (n_heads,), (256,))
        outs.append(out)
    ref = attention_ref(q, kc, vc, n_heads, n_kv, hd, pos0, 1, 1 / np.sqrt(hd), window)
    err = np.abs(outs[1] - ref).max()
    check(np.array_equal(outs[0], outs[1]) and err < 1e-4,
          'attention_split chunk derived in the kernel (window %d): %s the explicit chunk, max err %.2e'
          % (window, 'same bytes as' if np.array_equal(outs[0], outs[1]) else 'DIFFERENT from', err))


if not ONLY:
    split_chunk()
    split_chunk(window=50)


def rope_ref(h, cos, sin, pos, rot_off, rot_half, adjacent, w=None, eps=1e-6):
    """One head h (float32 [headDim]) normalised (w) and rotated at pos, in float32 like the kernel."""
    h = h.astype(np.float32).copy()
    hd = h.shape[0]
    i = np.arange(rot_half)
    i0 = rot_off + (2 * i if adjacent else i)
    i1 = rot_off + (2 * i + 1 if adjacent else i + rot_half)
    a, b = h[i0], h[i1]
    if w is not None:
        ss = np.float32((a.astype(np.float64) ** 2 + b.astype(np.float64) ** 2).sum())
        sc = np.float32(1) / np.sqrt(ss / np.float32(hd) + np.float32(eps))
        a, b = a * sc * w[i0], b * sc * w[i1]
    c, s = cos[pos * rot_half + i], sin[pos * rot_half + i]
    h[i0], h[i1] = a * c - b * s, b * c + a * s
    return h


def rope_store_q8(hd, n_heads=4, n_kv=2, ntok=3, pos0=5, ctx=16, rot_half=None, adjacent=False, norm=True):
    """rope_store_q8 writes the q rows and the q8 cache rows of k and v that rope (in place on the f32 staging rows)
    followed by kv_store_q8 write, byte for byte; the cache rows also match numpy's rope + quantization."""
    rot_half = rot_half or hd // 2
    rng = np.random.default_rng(7)
    q0 = rng.standard_normal((ntok, n_heads * hd)).astype(np.float32)
    k0 = (rng.standard_normal((ntok, n_kv * hd)) * 3).astype(np.float32)
    v = rng.standard_normal((ntok, n_kv * hd)).astype(np.float32)
    v[1, hd:2 * hd] = 0  # an all-zero v row: scale 0
    qw = (rng.random(hd) + 0.5).astype(np.float32)
    kw = (rng.random(hd) + 0.5).astype(np.float32)
    ang = rng.random((ctx, rot_half)).astype(np.float32) * 3
    cos, sin = np.cos(ang).astype(np.float32).reshape(-1), np.sin(ang).astype(np.float32).reshape(-1)
    kv_dim = n_kv * hd
    stride = kv_dim + 4 * n_kv
    eps = 1e-6
    # reference: rope<float> in place, then kv_store_q8
    q_a, k_a = q0.copy(), k0.copy()
    E.dispatch('rope_il' if adjacent else 'rope', [q_a, k_a, qw, kw, cos, sin, k_a],
               pack('iiiiifiii', hd, n_heads * hd, kv_dim, n_heads, pos0, eps, int(norm), 0, rot_half),
               (n_heads + n_kv, ntok), (rot_half,))
    kc_a = np.full((ntok, stride), 0x5A, np.uint8)
    vc_a = np.full((ntok, stride), 0x5A, np.uint8)
    E.dispatch('kv_store_q8', [k_a, v, kc_a, vc_a, np.zeros(1, np.int32)], pack('iiiiii', hd, kv_dim, n_kv, stride, 0, 0),
               (2 * n_kv, ntok), (32,))
    # fused
    q_b, k_b = q0.copy(), k0.copy()
    kc_b = np.full((ntok, stride), 0x5A, np.uint8)
    vc_b = np.full((ntok, stride), 0x5A, np.uint8)
    E.dispatch('rope_il_store_q8' if adjacent else 'rope_store_q8', [q_b, k_b, qw, kw, cos, sin, kc_b, v, vc_b],
               pack('iiiiifiiiii', hd, n_heads * hd, kv_dim, n_heads, pos0, eps, int(norm), 0, rot_half, n_kv, stride),
               (n_heads + 2 * n_kv, ntok), (rot_half,))
    same = np.array_equal(q_a, q_b) and np.array_equal(kc_a, kc_b) and np.array_equal(vc_a, vc_b)
    # numpy: rotate every k head, quantize
    kr = np.stack([np.concatenate([rope_ref(k0[t, h * hd:(h + 1) * hd], cos, sin, pos0 + t, 0, rot_half, adjacent,
                                            kw if norm else None, eps) for h in range(n_kv)]) for t in range(ntok)])
    want_k, _ = q8_rows(kr, n_kv, hd)
    want_v, _ = q8_rows(v, n_kv, hd)
    # the float32 rotation may round differently from numpy's in the last bit: at most one quantization step apart
    kq = kc_b[:, :kv_dim].view(np.int8).astype(int)
    close_k = np.abs(kq - want_k[:, :kv_dim].view(np.int8).astype(int)).max() <= 1
    check(same and close_k and np.array_equal(vc_b, want_v),
          'rope_%sstore_q8 hd=%d rotHalf=%d norm=%s: %s rope + kv_store_q8, k within a step of numpy, v as numpy'
          % ('il_' if adjacent else '', hd, rot_half, norm, 'same bytes as' if same else 'DIFFERENT from'))


if not ONLY:
    rope_store_q8(128)
    rope_store_q8(64, n_kv=1, n_heads=3)
    rope_store_q8(64, rot_half=16, adjacent=True, norm=False)
    rope_store_q8(96, rot_half=24, norm=False)


def router_norm(ds=False, n_exp=16, dim=128, top_k=4, ntok=3, remap=False, method=2, sigmoid=1):
    """router_norm / router_ds_norm: the norm before the experts and the router in one kernel give the bytes of rmsnorm
    followed by router / router_ds (xb, assign, weights, and with remap the GPU slots and the pick counts)."""
    rng = np.random.default_rng(8)
    x = rng.standard_normal((ntok, dim)).astype(np.float32)
    nw = (rng.random(dim) + 0.5).astype(np.float32)
    w = (rng.standard_normal((n_exp, dim)) * 0.2).astype(np.float32)
    bias = (rng.standard_normal(n_exp) * 0.1).astype(np.float32)
    emap = np.array([-1 if e % 3 == 0 else e for e in range(n_exp)], np.int32)
    eps = 1e-6
    res = []
    for fused in (False, True):
        xb = np.zeros((ntok, dim), np.float32)
        assign = np.zeros(ntok * top_k, np.int32)
        rw = np.zeros(ntok * top_k, np.float32)
        gassign = np.zeros(ntok * top_k, np.int32)
        counts = np.zeros(n_exp, np.uint32)
        if ds:
            fields = (n_exp, dim, top_k, 1, 4, 2, sigmoid, method, 1, 2.5, int(remap))
            bufs = [w, assign, rw, bias, emap, gassign, counts]
            if fused:
                E.dispatch('router_ds_norm', [x] + bufs + [nw, xb], pack('iiiiiiiiifif', *fields, eps), (ntok,), (256,))
            else:
                E.dispatch('rmsnorm', [x, nw, xb], pack('if', dim, eps), (ntok,), (256,))
                E.dispatch('router_ds', [xb] + bufs, pack('iiiiiiiiifi', *fields), (ntok,), (256,))
        else:
            fields = (n_exp, dim, top_k, 1, int(remap))
            bufs = [w, assign, rw, emap, gassign, counts]
            if fused:
                E.dispatch('router_norm', [x] + bufs + [nw, xb], pack('iiiiif', *fields, eps), (ntok,), (256,))
            else:
                E.dispatch('rmsnorm', [x, nw, xb], pack('if', dim, eps), (ntok,), (256,))
                E.dispatch('router', [xb] + bufs, pack('iiiii', *fields), (ntok,), (256,))
        res.append((xb, assign, rw, gassign, counts))
    same = all(np.array_equal(a, b) for a, b in zip(res[0], res[1]))
    picks_ok = res[1][4].sum() == (ntok * top_k if remap else 0)
    check(same and picks_ok, '%s ntok=%d remap=%s: %s rmsnorm + %s' % ('router_ds_norm' if ds else 'router_norm', ntok,
                                                                      remap, 'same bytes as' if same else 'DIFFERENT from',
                                                                      'router_ds' if ds else 'router'))


# ---- multi-row single-row walk (section 2a of kernels.metal): mv<NR>_matmul / matmul3 / matmul_swiglu against numpy and,
# byte for byte, against the one-row kernels (matmul / matmul3 / matmul_swiglu, plain and _norm) they stand in for.
MV_FORMATS = ('q4', 'q8', 'q2k', 'q6k', 'q4g', 'q8g', 'q4k', 'q5k') + ggml_ref.NEW


def mv_norm_ref(x, nw, eps=1e-6):
    xd = x.astype(np.float64)
    return xd * (1.0 / np.sqrt((xd * xd).mean() + eps)) * nw


def mv_test(fmt, nr, rows, cols, ntok=1, norm=False, bias=False, acc=False, softcap=0.0):
    rng = np.random.default_rng(30)
    wq, ws, w = fmt_weights(fmt, rows, cols, rng)
    x = rng.standard_normal((ntok, cols)).astype(np.float32)
    nw = (rng.random(cols) + 0.5).astype(np.float32)
    b = rng.standard_normal(rows).astype(np.float32)
    out0 = rng.standard_normal((ntok, rows)).astype(np.float32) if acc else np.zeros((ntok, rows), np.float32)
    sfx = '_norm' if norm else ''
    par = pack('iiiiffi', rows, cols, ntok, int(acc), 1e-6, softcap, int(bias))
    tg = cols * 4 if norm else 16
    tiles = (ntok + 7) // 8
    old, new = out0.copy(), out0.copy()
    E.dispatch('matmul_%s%s' % (fmt, sfx), [wq, ws, x, old, nw, b], par, ((rows + 3) // 4, tiles), (128,), tg)
    E.dispatch('mv%d_matmul_%s%s' % (nr, fmt, sfx), [wq, ws, x, new, nw, b], par,
               ((rows + 4 * nr - 1) // (4 * nr), tiles), (128,), tg)
    xr = np.stack([mv_norm_ref(x[t], nw) for t in range(ntok)]) if norm else x.astype(np.float64)
    ref = xr @ w.astype(np.float64).T + (b if bias else 0)
    if softcap > 0:
        ref = np.tanh(ref / softcap) * softcap
    ref = ref + (out0 if acc else 0)
    err = np.abs(new - ref).max() / np.abs(ref).max()
    check(np.array_equal(old, new) and err < 1e-4, 'mv%d_matmul_%s%s %dx%d, %d token(s)%s%s%s: %s the one-row kernel, rel err %.1e'
          % (nr, fmt, sfx, rows, cols, ntok, ', bias' if bias else '', ', acc' if acc else '', ', softcap' if softcap else '',
             'same bytes as' if np.array_equal(old, new) else 'DIFFERENT from', err))


def mv3_test(fmt, nr, rows, cols, ntok=1, norm=False, half=False, bias=False):
    rng = np.random.default_rng(31)
    ws_ = [fmt_weights(fmt, r, cols, rng) for r in rows]
    x = rng.standard_normal((ntok, cols)).astype(np.float32)
    nw = (rng.random(cols) + 0.5).astype(np.float32)
    bs = [rng.standard_normal(r).astype(np.float32) for r in rows]
    sfx = ('_norm' if norm else '') + ('_h' if half else '')
    par = pack('iiiiifi', rows[0], rows[1], rows[2], cols, ntok, 1e-6, int(bias))
    tg = cols * 4 if norm else 16
    tiles = (ntok + 7) // 8
    res = []
    for name, per in (('matmul3_%s%s' % (fmt, sfx), 4), ('mv%d_matmul3_%s%s' % (nr, fmt, sfx), 4 * nr)):
        outs = [np.full((ntok, rows[0]), np.nan, np.float32), np.full((ntok, rows[1]), np.nan, np.float32),
                np.full((ntok, rows[2]), np.nan, np.float16 if half else np.float32)]
        bufs = [ws_[0][0], ws_[0][1], ws_[1][0], ws_[1][1], ws_[2][0], ws_[2][1], x] + outs + [nw] + bs
        E.dispatch(name, bufs, par, ((sum(rows) + per - 1) // per, tiles), (128,), tg)
        res.append(outs)
    same = all(a.tobytes() == c.tobytes() for a, c in zip(res[0], res[1]))
    w = np.concatenate([t[2] for t in ws_])
    xr = np.stack([mv_norm_ref(x[t], nw) for t in range(ntok)]) if norm else x.astype(np.float64)
    ref = xr @ w.astype(np.float64).T + (np.concatenate(bs) if bias else 0)
    out = np.concatenate([o.astype(np.float32) for o in res[1]], axis=1)
    err = np.abs(out - ref).max() / np.abs(ref).max()
    check(same and err < (2e-3 if half else 1e-4), 'mv%d_matmul3_%s%s %s, %d token(s)%s: %s the one-row kernel, rel err %.1e'
          % (nr, fmt, sfx, rows, ntok, ', bias' if bias else '', 'same bytes as' if same else 'DIFFERENT from', err))


def mv_swiglu_test(fmt, nr, hidden, cols, ntok=1, norm=False, use_gelu=False):
    rng = np.random.default_rng(32)
    x = rng.standard_normal((ntok, cols)).astype(np.float32)
    nw = (rng.random(cols) + 0.5).astype(np.float32)
    if fmt in GGUF:
        gq, _, gw = fmt_weights(fmt, hidden, cols, rng)
        uq, _, uw = fmt_weights(fmt, hidden, cols, rng)
        wq, ws, w = gq, uq, np.concatenate([gw, uw])
    else:
        wq, ws, w = fmt_weights(fmt, 2 * hidden, cols, rng)
    sfx = '_norm' if norm else ''
    par = pack('iiifi', hidden, cols, ntok, 1e-6, int(use_gelu))
    tg = cols * 4 if norm else 16
    tiles = (ntok + 7) // 8
    res = []
    for name, per in (('matmul_swiglu_%s%s' % (fmt, sfx), 4), ('mv%d_matmul_swiglu_%s%s' % (nr, fmt, sfx), 4 * nr)):
        out = np.full((ntok, hidden), np.nan, np.float32)
        E.dispatch(name, [wq, ws, x, out, nw], par, ((hidden + per - 1) // per, tiles), (128,), tg)
        res.append(out)
    xr = np.stack([mv_norm_ref(x[t], nw) for t in range(ntok)]) if norm else x.astype(np.float64)
    y = xr @ w.astype(np.float64).T
    ref = (gelu if use_gelu else silu)(y[:, :hidden]) * y[:, hidden:]
    err = np.abs(res[1] - ref).max() / np.abs(ref).max()
    same = np.array_equal(res[0], res[1])
    check(same and err < 1e-4, 'mv%d_matmul_swiglu_%s%s hidden %d x %d, %d token(s)%s: %s the one-row kernel, rel err %.1e'
          % (nr, fmt, sfx, hidden, cols, ntok, ', gelu' if use_gelu else '', 'same bytes as' if same else 'DIFFERENT from', err))


if not ONLY:
    for _fmt in MV_FORMATS:
        for _nr in (2, 4):
            mv_test(_fmt, _nr, 37, 512)                                      # tail simdgroup: 37 rows is no multiple of NR
            mv_test(_fmt, _nr, 64, 512, norm=True, bias=True, acc=True)
            mv_test(_fmt, _nr, 21, 512, ntok=11, bias=True)                  # batched rows: two token tiles, the second partial
            mv3_test(_fmt, _nr, (21, 11, 13), 512, bias=True)                # NR rows straddle the q / k / v ranges
            mv3_test(_fmt, _nr, (24, 8, 8), 512, norm=True, half=True)
            mv3_test(_fmt, _nr, (9, 5, 7), 512, ntok=3)
            mv_swiglu_test(_fmt, _nr, 19, 512)
            mv_swiglu_test(_fmt, _nr, 24, 512, norm=True, use_gelu=True)
            mv_swiglu_test(_fmt, _nr, 13, 512, ntok=5)
    for _nr in (2, 4):
        mv_test('q4', _nr, 40, 3072, softcap=3.0)                            # 3 blocks per lane (Wdown's width)
        mv_test('q8', _nr, 130, 1024, norm=True)
        mv_test('q4g', _nr, 33, 1024, norm=True)


if not ONLY:
    router_norm()
    router_norm(remap=True, ntok=5)
    router_norm(ds=True)
    router_norm(ds=True, method=1, sigmoid=0, remap=True)


# ---------------------------------------------------------------------------------------------- vision encoder (vision.metal)


def vit_act_ref(v, act):
    v = v.astype(np.float64)
    if act == 1:
        return 0.5 * v * (1 + np.tanh(0.7978845608028654 * v * (1 + 0.044715 * v * v)))
    if act == 2:
        return v / (1 + np.exp(-v))
    if act == 3:
        return v / (1 + np.exp(-1.702 * v))
    return v


def vit_gemm(m, n, k, act=0, acc=False, bias=True, pad_x=0, pad_o=0):
    """vit_gemm: out = act(x . W^T + bias) (+ out) with half weights; the reference rounds x to half as the kernel does
    (its tiles are half), everything else in float64, so the tolerance only covers float32 accumulation order."""
    rng = np.random.default_rng(m * 7 + n * 3 + k)
    ldx, ldo = k + pad_x, n + pad_o
    x = rng.standard_normal((m, ldx)).astype(np.float32)
    w = (rng.standard_normal((n, k)) * 0.3).astype(np.float16)
    b = (rng.standard_normal(n) * 0.5).astype(np.float32)
    out = (rng.standard_normal((m, ldo)) * 2).astype(np.float32)
    before = out.copy()
    xh = x[:, :k].astype(np.float16).astype(np.float64)
    ref = xh @ w.astype(np.float64).T
    if bias:
        ref = ref + b
    ref = vit_act_ref(ref, act)
    if acc:
        ref = before[:, :n] + ref
    E.dispatch('vit_gemm', [x, w, b, out], pack('iiiiiiii', m, n, k, ldx, ldo, act, int(acc), int(bias)),
               ((n + 63) // 64, (m + 63) // 64), (128,))
    err = np.abs(out[:, :n] - ref).max()
    untouched = np.array_equal(out[:, n:], before[:, n:])
    check(err < 2e-3 * max(1.0, np.abs(ref).max()) and untouched,
          'vit_gemm %dx%dx%d act=%d acc=%s bias=%s: max err %.2e%s' % (m, n, k, act, acc, bias, err,
                                                                      '' if untouched else ' (wrote outside the output)'))


vit_gemm(64, 64, 32)
vit_gemm(70, 100, 44, act=1, bias=True)
vit_gemm(33, 130, 68, act=2, acc=True, pad_x=4, pad_o=6)
vit_gemm(9, 17, 12, act=3, bias=False)


def vit_norm(rows, cols, rms, has_w=True, has_b=True):
    rng = np.random.default_rng(rows + cols)
    x = (rng.standard_normal((rows, cols)) * 3 + 1).astype(np.float32)
    w = (rng.random(cols) + 0.5).astype(np.float32)
    b = rng.standard_normal(cols).astype(np.float32)
    out = np.zeros((rows, cols), np.float32)
    eps = 1e-6
    E.dispatch('vit_norm', [x, out, w, b], pack('iiiifii', cols, int(rms), int(has_w), int(has_b), eps, cols, cols),
               (rows,), (256,))
    xd = x.astype(np.float64)
    if rms:
        ref = xd / np.sqrt((xd * xd).mean(1, keepdims=True) + eps)
    else:
        mu = xd.mean(1, keepdims=True)
        ref = (xd - mu) / np.sqrt(((xd - mu) ** 2).mean(1, keepdims=True) + eps)
    if has_w:
        ref = ref * w
    if has_b:
        ref = ref + b
    err = np.abs(out - ref).max()
    check(err < 2e-5, 'vit_norm %s %dx%d: max err %.2e' % ('rms' if rms else 'layer', rows, cols, err))


vit_norm(5, 768, False)
vit_norm(3, 1280, True, has_b=False)
vit_norm(2, 72, False, has_w=False, has_b=False)


def vit_attention(t, heads, dh, windows):
    """vit_attention: bidirectional attention, keys of row i in [lo[i], hi[i]) (windows: tokens regrouped in blocks)."""
    rng = np.random.default_rng(t + heads + dh)
    dim = heads * dh
    q, k, v = (rng.standard_normal((t, dim)).astype(np.float32) for _ in range(3))
    lo = np.zeros(t, np.int32)
    hi = np.full(t, t, np.int32)
    if windows:
        for i in range(t):
            lo[i] = (i // windows) * windows
            hi[i] = min(t, lo[i] + windows)
    out = np.zeros((t, dim), np.float32)
    scale = 1 / np.sqrt(dh)
    mma = dh in (64, 72, 80, 128)
    if mma:
        E.dispatch('vit_attention_mma_d%d' % dh, [q, k, v, out, lo, hi], pack('iiif', t, heads, dh, scale),
                   (heads, (t + 31) // 32), (128,))
        out2 = np.zeros_like(out)  # and the simple kernel must agree
        E.dispatch('vit_attention', [q, k, v, out2, lo, hi], pack('iiif', t, heads, dh, scale), ((t + 15) // 16, heads), (128,))
        check(np.abs(out - out2).max() < 3e-3, 'vit_attention_mma_d%d t=%d: agrees with vit_attention, which is float (%.1e; the mma kernel works in half)' % (dh, t, np.abs(out - out2).max()))
    else:
        E.dispatch('vit_attention', [q, k, v, out, lo, hi], pack('iiif', t, heads, dh, scale), ((t + 15) // 16, heads), (128,))
    ref = np.zeros((t, dim))
    for h in range(heads):
        qh, kh, vh = (a[:, h * dh:(h + 1) * dh].astype(np.float64) for a in (q, k, v))
        for i in range(t):
            sc = kh[lo[i]:hi[i]] @ qh[i] * scale
            e = np.exp(sc - sc.max())
            ref[i, h * dh:(h + 1) * dh] = (e / e.sum()) @ vh[lo[i]:hi[i]]
    err = np.abs(out - ref).max()
    check(err < (3e-3 if mma else 2e-5), 'vit_attention t=%d heads=%d dh=%d windows=%s: max err %.2e' % (t, heads, dh, windows, err))


vit_attention(21, 3, 64, 0)
vit_attention(70, 2, 80, 16)
vit_attention(37, 2, 80, 8)
vit_attention(40, 1, 72, 0)
vit_attention(10, 2, 72, 0)
vit_attention(9, 1, 128, 4)


def vit_rope(t, heads, dh):
    rng = np.random.default_rng(t + dh)
    x = rng.standard_normal((t, heads * dh)).astype(np.float32)
    ang = rng.random((t, dh // 2)) * 6
    cs = np.stack([np.cos(ang), np.sin(ang)], axis=2).astype(np.float32).reshape(t, dh)
    cs = np.ascontiguousarray(cs)
    want = x.astype(np.float64).reshape(t, heads, dh)
    x0, x1 = want[:, :, :dh // 2], want[:, :, dh // 2:]
    c, sn = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    ref = np.concatenate([x0 * c - x1 * sn, x0 * sn + x1 * c], axis=2).reshape(t, heads * dh)
    E.dispatch('vit_rope', [x, cs], pack('iii', t, heads, dh), ((dh // 2 + 7) // 8, heads, t), (8, 1, 1))
    err = np.abs(x - ref).max()
    check(err < 1e-5, 'vit_rope t=%d heads=%d dh=%d: max err %.2e' % (t, heads, dh, err))


vit_rope(6, 2, 16)
vit_rope(5, 3, 80)


def vit_elementwise(n=1000):
    rng = np.random.default_rng(n)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    g = a.copy()
    E.dispatch('vit_gated', [g, b], pack('ii', n, 2), ((n + 63) // 64,), (64,))
    err1 = np.abs(g - vit_act_ref(a, 2) * b).max()
    g = a.copy()
    E.dispatch('vit_act', [g], pack('ii', n, 1), ((n + 63) // 64,), (64,))
    err2 = np.abs(g - vit_act_ref(a, 1)).max()
    g = a.copy()
    E.dispatch('vit_add', [g, b], pack('ii', n, 0), ((n + 63) // 64,), (64,))
    err3 = np.abs(g - (a + b)).max()
    check(max(err1, err2, err3) < 2e-6, 'vit_gated / vit_act / vit_add: max err %.2e' % max(err1, err2, err3))


vit_elementwise()


def vit_gather(rows=7, width=37, src=11, group=1):
    rng = np.random.default_rng(rows + width)
    x = rng.standard_normal((src, width)).astype(np.float32)
    idx = rng.integers(0, src, rows * group).astype(np.int32)
    out = np.zeros((rows, width), np.float32)
    E.dispatch('vit_gather_mean' if group > 1 else 'vit_gather', [x, out, idx], pack('iii', rows, width, group),
               ((width + 15) // 16, (rows + 3) // 4), (16, 4))
    ref = x[idx].reshape(rows, group, width).mean(1)
    err = np.abs(out - ref).max()
    check(err < 1e-6, 'vit_gather%s rows=%d width=%d group=%d: max err %.2e' % ('_mean' if group > 1 else '', rows, width, group, err))


vit_gather()
vit_gather(group=4)


# ---------------------------------------------------------------------------------------------- audio.metal (Whisper)
import math


def np_gelu(v):
    return 0.5 * v * (1.0 + np.vectorize(math.erf)(v / math.sqrt(2.0)))


def wh_gemm(variant, rows, cols, ntok, batch=1, bias=True, gelu=False, acc=False, ld_pad=0):
    """wh_gemm_<w><t>: out[t][r] (+)= act(bias[r] + x[t] . w[r]) against float64 numpy on the (half-rounded) operands,
    batched over z with strides, for shapes that are not multiples of the 64 x 32 x 32 tiles."""
    rng = np.random.default_rng(11)
    half_w = variant[0] == 'h'
    half_tile = variant[1] == 'h'
    ldw = cols + ld_pad
    w32 = rng.standard_normal((batch, rows, ldw)).astype(np.float32) * 0.3
    w = w32.astype(np.float16) if half_w else w32
    x = rng.standard_normal((batch, ntok, cols)).astype(np.float32)
    b = rng.standard_normal(rows).astype(np.float32)
    out0 = rng.standard_normal((batch, ntok, rows)).astype(np.float32)
    out = out0.copy() if acc else np.zeros_like(out0)
    flags = (1 if bias else 0) | (2 if gelu else 0) | (4 if acc else 0)
    p = pack('iiiiiiiiii', rows, cols, ntok, ldw, cols, rows, flags, rows * ldw, ntok * cols, ntok * rows)
    E.dispatch('wh_gemm_' + variant, [w, x, out, b], p, ((rows + 63) // 64, (ntok + 31) // 32, batch), (128,))
    wd = w.astype(np.float64)
    if half_tile and not half_w:
        wd = w32.astype(np.float16).astype(np.float64)       # the float weight is rounded to half in the tile
    ref = np.einsum('btk,brk->btr', x.astype(np.float64), wd[:, :, :cols])
    if bias:
        ref = ref + b[None, None, :]
    if gelu:
        ref = np_gelu(ref)
    if acc:
        ref = ref + out0
    err = np.abs(out - ref).max() / max(1.0, np.abs(ref).max())
    check(err < 2e-5, 'wh_gemm_%s rows=%d cols=%d ntok=%d batch=%d bias=%s gelu=%s acc=%s: max rel err %.2e'
          % (variant, rows, cols, ntok, batch, bias, gelu, acc, err))


for v in ('hh', 'hf', 'fh', 'ff'):
    wh_gemm(v, 70, 100, 37)
wh_gemm('hh', 64, 32, 32, bias=False)
wh_gemm('ff', 128, 96, 64, gelu=True)
wh_gemm('hh', 100, 80, 5, acc=True, gelu=True)
wh_gemm('ff', 75, 33, 40, batch=3, bias=False)
wh_gemm('ff', 64, 70, 33, batch=2, bias=False, ld_pad=26)


def wh_layernorm(n, d):
    rng = np.random.default_rng(12)
    x = (rng.standard_normal((n, d)) * 3 + 1).astype(np.float32)
    g = rng.standard_normal(d).astype(np.float32)
    bt = rng.standard_normal(d).astype(np.float32)
    out = np.zeros_like(x)
    E.dispatch('wh_layernorm', [x, out, g, bt], pack('ifi', d, 1e-5, 1), (n,), (256,))
    xd = x.astype(np.float64)
    ref = (xd - xd.mean(1, keepdims=True)) / np.sqrt(xd.var(1, keepdims=True) + 1e-5) * g + bt
    err = np.abs(out - ref).max()
    check(err < 3e-5, 'wh_layernorm n=%d d=%d: max err %.2e' % (n, d, err))
    inplace = x.copy()
    E.dispatch('wh_layernorm', [inplace, inplace, g, bt], pack('ifi', d, 1e-5, 1), (n,), (256,))
    check(np.array_equal(inplace, out), 'wh_layernorm in place gives the same bytes')


wh_layernorm(5, 384)
wh_layernorm(3, 1280)


def wh_im2col():
    rng = np.random.default_rng(13)
    c, t = 5, 11
    mel = rng.standard_normal((c, t)).astype(np.float32)
    col = np.zeros((t, c * 3), np.float32)
    E.dispatch('wh_im2col_mel', [mel, col], pack('iii', t, c, t), ((t * c * 3 + 255) // 256,), (256,))
    ref = np.zeros((t, c * 3), np.float32)
    for tt in range(t):
        for ch in range(c):
            for k in range(3):
                i = tt + k - 1
                ref[tt, ch * 3 + k] = mel[ch, i] if 0 <= i < t else 0
    check(np.array_equal(col, ref), 'wh_im2col_mel: patches of the first convolution')
    tin, t2 = 12, 6
    h = rng.standard_normal((tin, c)).astype(np.float32)
    col2 = np.zeros((t2, c * 3), np.float32)
    E.dispatch('wh_im2col_s2', [h, col2], pack('iii', t2, c, tin), ((t2 * c * 3 + 255) // 256,), (256,))
    ref2 = np.zeros((t2, c * 3), np.float32)
    for tt in range(t2):
        for ch in range(c):
            for k in range(3):
                i = 2 * tt + k - 1
                ref2[tt, ch * 3 + k] = h[i, ch] if 0 <= i < tin else 0
    check(np.array_equal(col2, ref2), 'wh_im2col_s2: patches of the stride-2 convolution')


wh_im2col()


def wh_split_heads(n=7, heads=3, hd=8, cap=12, pos0=2):
    rng = np.random.default_rng(14)
    d = heads * hd
    q, k, v = [rng.standard_normal((n, d)).astype(np.float32) for _ in range(3)]
    qh = np.zeros((heads, n, hd), np.float32)
    kh = np.full((heads, cap, hd), 7, np.float32)
    vt = np.full((heads, hd, cap), 7, np.float32)
    scale = 0.125
    E.dispatch('wh_split_heads', [q, k, v, qh, kh, vt], pack('iiiiifii', n, heads, hd, cap, pos0, scale, 1, 1),
               ((n * d + 255) // 256,), (256,))
    rq = (q * np.float32(scale)).reshape(n, heads, hd).transpose(1, 0, 2)
    rk = np.full((heads, cap, hd), 7, np.float32)
    rk[:, pos0:pos0 + n, :] = k.reshape(n, heads, hd).transpose(1, 0, 2)
    rv = np.full((heads, hd, cap), 7, np.float32)
    rv[:, :, pos0:pos0 + n] = v.reshape(n, heads, hd).transpose(1, 2, 0)
    check(np.array_equal(qh, rq) and np.array_equal(kh, rk) and np.array_equal(vt, rv),
          'wh_split_heads: q scaled, k rows and v columns at position %d, the rest untouched' % pos0)


wh_split_heads()


def wh_softmax(n=9, heads=2, length=70, causal=True, pos0=3):
    rng = np.random.default_rng(15)
    s = (rng.standard_normal((heads, n, length)) * 4).astype(np.float32)
    got = s.copy()
    E.dispatch('wh_softmax', [got], pack('iiii', n, length, int(causal), pos0), (n, heads), (256,))
    ref = np.zeros_like(s, np.float64)
    for h in range(heads):
        for t in range(n):
            vis = min(length, pos0 + t + 1) if causal else length
            e = np.exp(s[h, t, :vis].astype(np.float64) - s[h, t, :vis].max())
            ref[h, t, :vis] = e / e.sum()
    err = np.abs(got - ref).max()
    check(err < 1e-6, 'wh_softmax n=%d len=%d causal=%s: max err %.2e, masked tail zero %s'
          % (n, length, causal, err, (got[:, 0, min(length, pos0 + 1):] == 0).all() if causal else True))


wh_softmax()
wh_softmax(causal=False, length=300)


def wh_embed(half_table, n=5, d=40, vocab=30, pos0=3):
    rng = np.random.default_rng(16)
    table = rng.standard_normal((vocab, d)).astype(np.float16 if half_table else np.float32)
    pos = rng.standard_normal((20, d)).astype(np.float32)
    tok = rng.integers(0, vocab, n).astype(np.int32)
    x = np.zeros((n, d), np.float32)
    E.dispatch('wh_embed_' + ('h' if half_table else 'f'), [table, pos, tok, x], pack('iii', n, d, pos0), ((n * d + 255) // 256,), (256,))
    ref = table[tok].astype(np.float32) + pos[pos0:pos0 + n]
    check(np.array_equal(x, ref), 'wh_embed_%s: token embedding plus learned position %d.., exact' % ('h' if half_table else 'f', pos0))


wh_embed(True)
wh_embed(False)


# ---- q4 KV cache (KvQ4): Hadamard-rotated K, 4-bit groups of 32 with a half scale, recent window of half rows


def q4_rotate(x):
    """Unnormalized Walsh-Hadamard transform of each head row (last axis, blocks of the largest power of two dividing
    it) in float32 with KvQ4.rotate's butterflies: stride 1, 2, 4, ..., (a, b) -> (a + b, a - b)."""
    x = np.array(x, np.float32, copy=True)
    hd = x.shape[-1]
    b = hd & -hd
    xs = x.reshape(-1, hd // b, b)
    h = 1
    while h < b:
        y = xs.reshape(xs.shape[0], xs.shape[1], b // (2 * h), 2, h)
        a = y[:, :, :, 0, :].copy()
        c = y[:, :, :, 1, :].copy()
        y[:, :, :, 0, :] = a + c
        y[:, :, :, 1, :] = a - c
        h *= 2
    return x


def q4_rows(x):
    """The q4 cache rows of x ([rows, kvDim] f32, heads of whole groups): per group of 32 d = half(max|x| / 7.5),
    code = clip(rint(x / d + 7.5), 0, 15) (8 when d = 0), value (code - 7.5) * d; a row is kvDim / 2 bytes (byte j of
    a group: code j | code j + 16 << 4) then kvDim / 32 halves. Returns (bytes, the dequantized values)."""
    rows = x.shape[0]
    g = x.reshape(rows, -1, 32)
    d16 = (np.abs(g).max(axis=2) / np.float32(7.5)).astype(np.float16)
    d = d16.astype(np.float32)
    safe = np.where(d != 0, d, np.float32(1))
    c = np.clip(np.rint(g / safe[:, :, None] + np.float32(7.5)), 0, 15).astype(np.uint8)
    c[d == 0] = 8
    packed = (c[:, :, :16] | (c[:, :, 16:] << 4)).astype(np.uint8).reshape(rows, -1)
    out = np.concatenate([packed, d16.view(np.uint8).reshape(rows, -1)], axis=1)
    return np.ascontiguousarray(out), ((c.astype(np.float32) - np.float32(7.5)) * d[:, :, None]).reshape(rows, -1)


def kv_store_q4(hd, n_kv=2, n_heads=4, ntok=5, pos0=3, ctx=12, ring=8, rotate=True):
    """kv_store_q4 rotates and quantizes k, quantizes v, rotates the queries in place and fills the ring and its tags,
    all bit for bit as numpy (and KvQ4 / KvCache.Q4) does. With rotate=False (the accuracy study's switch) k and the
    queries stay as they are."""
    rng = np.random.default_rng(6)
    kv_dim = n_kv * hd
    k = (rng.standard_normal((ntok, kv_dim)) * rng.random((ntok, 1)) * 4).astype(np.float32)
    k[:, 5] *= 20  # an outlier channel
    v = rng.standard_normal((ntok, kv_dim)).astype(np.float32)
    v[1, :32] = 0  # an all-zero group: scale 0, codes 8
    q = rng.standard_normal((ntok, n_heads * hd)).astype(np.float32)
    q0 = q.copy()
    stride = kv_dim // 2 + kv_dim // 16
    kc = np.full((ctx, stride), 0x5A, np.uint8)
    vc = np.full((ctx, stride), 0x5A, np.uint8)
    kb = np.ascontiguousarray(kc[pos0:])
    vb = np.ascontiguousarray(vc[pos0:])
    kr = np.zeros((ring, kv_dim), np.float16)
    vr = np.zeros((ring, kv_dim), np.float16)
    tags = np.full(ring, -1, np.int32)
    # KvStoreQ4Params: headDim kvDim nKvHeads qStride stride pos0 ringCap rotate
    E.dispatch('kv_store_q4', [k, v, kb, vb, q, kr, vr, tags],
               pack('iiiiiiii', hd, kv_dim, n_kv, n_heads * hd, stride, pos0, ring, int(rotate)), (2 * n_kv + n_heads, ntok), (32,))
    krot = q4_rotate(k.reshape(ntok, n_kv, hd)).reshape(ntok, kv_dim) if rotate else k
    want_k, kd = q4_rows(krot)
    want_v, vd = q4_rows(v)
    ok = (kb[:ntok] == want_k).all() and (vb[:ntok] == want_v).all() and (kb[ntok:] == 0x5A).all()
    okq = (q == (q4_rotate(q0.reshape(ntok, n_heads, hd)).reshape(ntok, -1) if rotate else q0)).all()
    slots = [(pos0 + t) % ring for t in range(ntok)]
    okr = all((kr[s] == krot[t].astype(np.float16)).all() and (vr[s] == v[t].astype(np.float16)).all()
              and tags[s] == pos0 + t for t, s in enumerate(slots))
    # error: within half a step max|x| / 15 of each group (plus the half rounding of the scale)
    gk = np.abs(krot.reshape(ntok, -1, 32)).max(axis=2)
    err = (np.abs(kd - krot).reshape(ntok, -1, 32).max(axis=2) / np.where(gk > 0, gk / 15, 1)).max()
    check(ok and okq and okr and err <= 1.002, 'kv_store_q4 hd=%d rotate=%s: rows as numpy %s, queries %s, ring %s, max error '
          '%.3f of max|x| / 15' % (hd, rotate, 'yes' if ok else 'NO', 'yes' if okq else 'NO', 'yes' if okr else 'NO', err))


kv_store_q4(128)
kv_store_q4(64, n_kv=3, n_heads=6)
kv_store_q4(96)
kv_store_q4(256, n_kv=1, ntok=3)
kv_store_q4(128, rotate=False)


# ---- q8 KV cache with Hadamard-rotated keys (backend.kv-q8-rotate): kv_store_q8r and the q8 attention kernels on it


def kv_store_q8r(hd, n_kv=2, n_heads=4, ntok=5, pos0=3, ctx=12, page=0):
    """kv_store_q8r rotates k (and the queries, in place) and quantizes it with q8's one-scale-per-head-row rule, v as
    kv_store_q8 does, all bit for bit as numpy (and KvCache.Q8 with rotate) does; the other cache rows stay."""
    rng = np.random.default_rng(11)
    kv_dim = n_kv * hd
    k = (rng.standard_normal((ntok, kv_dim)) * rng.random((ntok, 1)) * 4).astype(np.float32)
    k[:, 5] *= 20  # an outlier channel
    if ntok > 1:
        k[1, :hd] = 0  # an all-zero row: scale 0, values 0
    v = rng.standard_normal((ntok, kv_dim)).astype(np.float32)
    q = rng.standard_normal((ntok, n_heads * hd)).astype(np.float32)
    q0 = q.copy()
    stride = kv_dim + 4 * n_kv
    kc = np.full((ctx, stride), 0x5A, np.uint8)
    vc = np.full((ctx, stride), 0x5A, np.uint8)
    # KvStoreQ8RParams: headDim kvDim nKvHeads qStride stride pos pageShift; flat: the cache is bound at row pos0
    krot = q4_rotate(k.reshape(ntok, n_kv, hd)).reshape(ntok, kv_dim)
    want_k, kd = q8_rows(krot, n_kv, hd)
    want_v, vd = q8_rows(v, n_kv, hd)
    if page:  # paged: the pools bound at row 0, the rows of positions pos0 .. pos0 + ntok - 1 through the block table
        kb, vb, tbl = paged([kc, vc], page)
        E.dispatch('kv_store_q8r', [k, v, kb, vb, q, tbl], pack('iiiiiii', hd, kv_dim, n_kv, n_heads * hd, stride, pos0, page),
                   (2 * n_kv + n_heads, ntok), (32,))
        ok = (paged_rows(kb, tbl, page, pos0, ntok) == want_k).all() and (paged_rows(vb, tbl, page, pos0, ntok) == want_v).all()
        ok = ok and (paged_rows(kb, tbl, page, 0, pos0) == 0x5A).all()
    else:
        kb = np.ascontiguousarray(kc[pos0:])
        vb = np.ascontiguousarray(vc[pos0:])
        E.dispatch('kv_store_q8r', [k, v, kb, vb, q, np.zeros(1, np.int32)], pack('iiiiiii', hd, kv_dim, n_kv, n_heads * hd, stride, 0, 0),
                   (2 * n_kv + n_heads, ntok), (32,))
        ok = (kb[:ntok] == want_k).all() and (vb[:ntok] == want_v).all() and (kb[ntok:] == 0x5A).all()
    okq = (q == q4_rotate(q0.reshape(ntok, n_heads, hd)).reshape(ntok, -1)).all()
    # the rotated row is within half a step (max|x| / 254) of the rotated key
    err = np.abs(kd - krot).max() / (np.abs(krot).max() / 254)
    check(ok and okq and err <= 1.0001, 'kv_store_q8r hd=%d%s: rows as numpy %s, queries rotated %s, max error %.3f of max|x| / 254'
          % (hd, ', paged (blocks of %d)' % (1 << page) if page else '', 'yes' if ok else 'NO', 'yes' if okq else 'NO', err))


kv_store_q8r(128)
kv_store_q8r(64, n_kv=3, n_heads=6)
kv_store_q8r(96)
kv_store_q8r(256, n_kv=1, ntok=3)
kv_store_q8r(128, page=4)
kv_store_q8r(64, n_kv=3, n_heads=6, ntok=40, pos0=11, ctx=70, page=4)
kv_store_q8r(128, ntok=1, pos0=31, ctx=70, page=5)


def attention_q8_rot(name, hd, n_heads=4, n_kv=2, pos0=5, ntok=3, ctx=16, G=1):
    """A q8 attention kernel over rotated key rows with rotated queries and the score scale divided by the Hadamard
    block computes the attention of the original keys: against the f64 reference on the dequantized rotated rows to
    rounding, against the unrotated f32 keys within the quantization error, and with keys that have an outlier channel
    (like Qwen3's) clearly closer to them than the same kernel over plain q8 rows."""
    rng = np.random.default_rng(12)
    kv_dim = n_kv * hd
    q = rng.standard_normal((ntok, n_heads * hd)).astype(np.float32)
    kc = rng.standard_normal((ctx, kv_dim)).astype(np.float32)
    kc[:, 3] *= 12  # outlier channels: one scale per row fits them, not the rest
    kc[:, hd + 7] *= 12
    vc = rng.standard_normal((ctx, kv_dim)).astype(np.float32)
    B = hd & -hd
    scale = np.float32(1 / np.sqrt(hd))
    stride = kv_dim + 4 * n_kv
    krot = q4_rotate(kc.reshape(ctx, n_kv, hd)).reshape(ctx, kv_dim)
    (kb, kd), (vb, vd) = q8_rows(krot, n_kv, hd), q8_rows(vc, n_kv, hd)
    qr = q4_rotate(q.reshape(ntok, n_heads, hd)).reshape(ntok, -1)
    out = np.zeros((ntok, n_heads * hd), np.float32)
    scores = np.zeros((ntok * n_heads * ctx,), np.float32)
    # AttnParams: nHeads nKvHeads qkDim vDim pos0 ctx scale qStride kStride vStride oStride window tokenOffset softcap
    p = pack('iiiiiifiiiiiifi', n_heads, n_kv, hd, hd, pos0, ctx, scale / B, n_heads * hd, stride, stride, n_heads * hd, 0, 0, 0.0, 0)
    E.dispatch(name, [qr, kb, vb, out, scores, np.zeros(1, np.int32)], p, (n_heads // G, ntok), (256,))
    ref_rot = attention_ref(qr, kd, vd, n_heads, n_kv, hd, pos0, ntok, scale / B)
    exact = attention_ref(q, kc, vc, n_heads, n_kv, hd, pos0, ntok, scale)
    # the same kernel over plain q8 rows of the unrotated keys
    (pk, pkd) = q8_rows(kc, n_kv, hd)
    out0 = np.zeros_like(out)
    p0 = pack('iiiiiifiiiiiifi', n_heads, n_kv, hd, hd, pos0, ctx, scale, n_heads * hd, stride, stride, n_heads * hd, 0, 0, 0.0, 0)
    E.dispatch(name, [q, pk, vb, out0, scores.copy(), np.zeros(1, np.int32)], p0, (n_heads // G, ntok), (256,))
    e_rot, e_plain = np.abs(out - exact).max(), np.abs(out0 - exact).max()
    check(np.abs(out - ref_rot).max() < 1e-4 and e_rot < e_plain,
          '%s hd=%d rotated keys: max err %.2e vs the kernel on the dequantized rows, %.2e vs f32 keys (plain q8 %.2e)'
          % (name, hd, np.abs(out - ref_rot).max(), e_rot, e_plain))


def rope_store_q8r(hd, n_heads=4, n_kv=2, ntok=3, pos0=5, ctx=16, rot_half=None, adjacent=False, norm=True):
    """rope_store_q8r (rope, the Hadamard rotation in threadgroup memory and the q8 quantization of k in one dispatch)
    writes the q rows and the cache rows byte for byte as rope followed by kv_store_q8r does."""
    rot_half = rot_half or hd // 2
    rng = np.random.default_rng(13)
    q0 = rng.standard_normal((ntok, n_heads * hd)).astype(np.float32)
    k0 = (rng.standard_normal((ntok, n_kv * hd)) * 3).astype(np.float32)
    k0[:, 5] *= 15
    v = rng.standard_normal((ntok, n_kv * hd)).astype(np.float32)
    v[1, hd:2 * hd] = 0
    qw = (rng.random(hd) + 0.5).astype(np.float32)
    kw = (rng.random(hd) + 0.5).astype(np.float32)
    ang = rng.random((ctx, rot_half)).astype(np.float32) * 3
    cos, sin = np.cos(ang).astype(np.float32).reshape(-1), np.sin(ang).astype(np.float32).reshape(-1)
    kv_dim = n_kv * hd
    stride = kv_dim + 4 * n_kv
    eps = 1e-6
    q_a, k_a = q0.copy(), k0.copy()
    E.dispatch('rope_il' if adjacent else 'rope', [q_a, k_a, qw, kw, cos, sin, k_a],
               pack('iiiiifiii', hd, n_heads * hd, kv_dim, n_heads, pos0, eps, int(norm), 0, rot_half),
               (n_heads + n_kv, ntok), (rot_half,))
    kc_a = np.full((ntok, stride), 0x5A, np.uint8)
    vc_a = np.full((ntok, stride), 0x5A, np.uint8)
    E.dispatch('kv_store_q8r', [k_a, v, kc_a, vc_a, q_a, np.zeros(1, np.int32)], pack('iiiiiii', hd, kv_dim, n_kv, n_heads * hd, stride, 0, 0),
               (2 * n_kv + n_heads, ntok), (32,))
    q_b, k_b = q0.copy(), k0.copy()
    kc_b = np.full((ntok, stride), 0x5A, np.uint8)
    vc_b = np.full((ntok, stride), 0x5A, np.uint8)
    E.dispatch('rope_il_store_q8r' if adjacent else 'rope_store_q8r', [q_b, k_b, qw, kw, cos, sin, kc_b, v, vc_b],
               pack('iiiiifiiiii', hd, n_heads * hd, kv_dim, n_heads, pos0, eps, int(norm), 0, rot_half, n_kv, stride),
               (n_heads + 2 * n_kv, ntok), (rot_half,))
    check(np.array_equal(q_a, q_b) and np.array_equal(kc_a, kc_b) and np.array_equal(vc_a, vc_b),
          'rope_%sstore_q8r hd=%d rotHalf=%d norm=%s: same bytes (rotated queries, k and v rows) as rope + kv_store_q8r'
          % ('il_' if adjacent else '', hd, rot_half, norm))


rope_store_q8r(128)
rope_store_q8r(64, n_kv=1, n_heads=3)
rope_store_q8r(64, rot_half=16, adjacent=True, norm=False)
rope_store_q8r(96, rot_half=24, norm=False)
rope_store_q8r(256, n_kv=1)


attention_q8_rot('attention_g1_d128_q8', 128)
attention_q8_rot('attention_g1_d64_q8', 64)
attention_q8_rot('attention_q8', 96)
attention_q8_rot('attention_g4_d128_q8', 128, n_heads=8, n_kv=2, G=4)


def attention_dec_rot(hd, G, n_heads=8, n_kv=2, pos0=300, n_split=3, ctx=320):
    """attention_dec over a q8 cache of rotated keys (backend.kv-q8-rotate): the rotated query and scale / B, as the
    host passes them; against the float64 reference on the dequantized rotated rows to rounding, and (keys with outlier
    channels) closer to the attention of the original f32 keys than the same kernel over plain q8 rows."""
    rng = np.random.default_rng(21 + hd + G)
    q = rng.standard_normal((1, n_heads * hd)).astype(np.float32)
    k0 = rng.standard_normal((ctx, n_kv * hd)).astype(np.float32)
    k0[:, 3] *= 12
    k0[:, hd + 7] *= 12
    v0 = rng.standard_normal((ctx, n_kv * hd)).astype(np.float32)
    B = hd & -hd
    stride = n_kv * hd + 4 * n_kv
    scale = np.float32(1 / np.sqrt(hd))
    (vc, vd) = q8_rows(v0, n_kv, hd)

    def run(qq, kk, sc):
        kc, kd = q8_rows(kk, n_kv, hd)
        part = np.zeros(n_heads * n_split * (hd + 2), np.float32)
        out = np.zeros((1, n_heads * hd), np.float32)
        E.dispatch('attention_dec_g%d_d%d_q8' % (G, hd), [qq, kc, vc, part, out, np.zeros(1, np.int32)],
                   pack('iiiiifiiifiii', n_heads, n_kv, hd, pos0, ctx, sc, n_heads * hd, stride, 0, 0.0, n_split, 0, 0),
                   (n_heads // G, (n_split + 3) // 4), (128,))
        if n_split > 1:
            E.dispatch('attention_combine', [part, out], pack('iii', n_heads, hd, n_split), (n_heads,), (256,))
        return out, kd
    krot = q4_rotate(k0.reshape(ctx, n_kv, hd)).reshape(ctx, -1)
    qr = q4_rotate(q.reshape(1, n_heads, hd)).reshape(1, -1)
    out, kd = run(qr, krot, scale / B)
    ref_rot = attention_ref(qr, kd, vd, n_heads, n_kv, hd, pos0, 1, scale / B)
    exact = attention_ref(q, k0, v0, n_heads, n_kv, hd, pos0, 1, scale)
    out0, _ = run(q, k0, scale)
    e_rot, e_plain = np.abs(out - exact).max(), np.abs(out0 - exact).max()
    check(np.abs(out - ref_rot).max() < 1e-4 and e_rot < e_plain,
          'attention_dec_g%d_d%d_q8 rotated keys, %d split(s): max err %.2e vs the dequantized rows, %.2e vs f32 keys (plain q8 %.2e)'
          % (G, hd, n_split, np.abs(out - ref_rot).max(), e_rot, e_plain))


attention_dec_rot(128, 2)
attention_dec_rot(64, 4, n_split=1, pos0=100)
attention_dec_rot(256, 2)



def ex_from(tags, cap, recent, p):
    """KvCache.Q4.exactFrom: the first key a query at p reads from the ring."""
    if recent == 0:
        return p + 1
    lim = max(0, p - recent + 1)
    s = p
    while s >= lim and tags[s % cap] == s:
        s -= 1
    return s + 1


def q4_cache(ctx, n_kv, hd, recent, cap, top, lost=(), seed=7):
    """Random K / V for positions [0, ctx) as the q4 cache holds them: the q4 rows of the rotated K and of V, and a ring
    of cap slots holding the positions (top - cap, top] as halves, minus the lost ones. Returns the buffers and the
    float64 views a reference needs."""
    rng = np.random.default_rng(seed)
    kv_dim = n_kv * hd
    k = q4_rotate(rng.standard_normal((ctx, n_kv, hd)).astype(np.float32)).reshape(ctx, kv_dim)
    v = rng.standard_normal((ctx, kv_dim)).astype(np.float32)
    (kc, kd), (vc, vd) = q4_rows(k), q4_rows(v)
    kr = np.zeros((max(cap, 1), kv_dim), np.float16)
    vr = np.zeros((max(cap, 1), kv_dim), np.float16)
    tags = np.full(max(cap, 1), -1, np.int32)
    for s in range(max(0, top - cap + 1), top + 1):
        if cap > 0 and s not in lost:
            kr[s % cap], vr[s % cap], tags[s % cap] = k[s], v[s], s
    return kc, vc, kr, vr, tags, kd, vd, kr.astype(np.float32), vr.astype(np.float32)


def attention_ref_q4(q, kd, vd, kr, vr, tags, cap, recent, n_heads, n_kv, hd, pos0, ntok, scale, window=0, softcap=0.0):
    """attention_ref where each query at p reads the keys [ex_from(p), p] from the ring rows."""
    out = np.zeros((ntok, n_heads * hd), np.float64)
    g = n_heads // n_kv
    for t in range(ntok):
        ln = pos0 + t + 1
        first = max(0, ln - window) if window > 0 else 0
        ex = ex_from(tags, cap, recent, ln - 1)
        for h in range(n_heads):
            kvh = h // g
            sl = slice(kvh * hd, (kvh + 1) * hd)
            k = np.array([kr[s % cap, sl] if s >= ex else kd[s, sl] for s in range(first, ln)], np.float64)
            v = np.array([vr[s % cap, sl] if s >= ex else vd[s, sl] for s in range(first, ln)], np.float64)
            s = k @ q[t, h * hd:(h + 1) * hd].astype(np.float64) * scale
            if softcap > 0:
                s = np.tanh(s / softcap) * softcap
            pr = np.exp(s - s.max())
            pr /= pr.sum()
            out[t, h * hd:(h + 1) * hd] = pr @ v
    return out


def attention_q4(name, hd, n_heads=4, n_kv=2, pos0=30, ntok=5, ctx=48, recent=8, lost=(), window=0):
    """attention_q4_t: q4 rows before exactFrom(p), ring rows from it on (lost: positions the ring no longer holds)."""
    cap = recent + 8 if recent > 0 else 0
    kc, vc, kr, vr, tags, kd, vd, krf, vrf = q4_cache(ctx, n_kv, hd, recent, cap, pos0 + ntok - 1, lost)
    rng = np.random.default_rng(8)
    q = rng.standard_normal((ntok, n_heads * hd)).astype(np.float32)
    out = np.zeros((ntok, n_heads * hd), np.float32)
    scores = np.zeros((ntok * n_heads * ctx,), np.float32)
    scale = np.float32(1 / np.sqrt(hd) / (hd & -hd))
    stride = n_kv * hd // 2 + n_kv * hd // 16
    # AttnQ4Params: nHeads nKvHeads qkDim pos0 ctx scale qStride stride oStride window tokenOffset softcap recent ringCap
    p = pack('iiiiifiiiiifii', n_heads, n_kv, hd, pos0, ctx, scale, n_heads * hd, stride, n_heads * hd, window, 0, 0.0,
             recent, cap)
    E.dispatch(name, [q, kc, vc, out, scores, kr, vr, tags], p, (n_heads, ntok), (256,))
    ref = attention_ref_q4(q, kd, vd, krf, vrf, tags, cap, recent, n_heads, n_kv, hd, pos0, ntok, scale, window)
    err = np.abs(out - ref).max()
    check(err < 1e-4, '%s hd=%d recent=%d lost=%s window=%d: max err %.2e' % (name, hd, recent, list(lost), window, err))


attention_q4('attention_g1_d128_q4', 128)
attention_q4('attention_g1_d128_q4', 128, lost=(29,))
attention_q4('attention_g1_d128_q4', 128, lost=(31,))
attention_q4('attention_g1_d128_q4', 128, recent=0)
attention_q4('attention_g1_d64_q4', 64, n_kv=1, window=6)
attention_q4('attention_g1_d256_q4', 256)
attention_q4('attention_q4', 96)


def flash_q4(hd, n_heads=4, n_kv=2, pos0=37, ntok=45, ctx=128, recent=16, lost=(), window=0, softcap=0.0):
    """attention_fa_q4: the band corrections (ring scores, P_band (V_ring - V_q4)) against the same reference."""
    cap = recent + ntok if recent > 0 else 0
    kc, vc, kr, vr, tags, kd, vd, krf, vrf = q4_cache(ctx, n_kv, hd, recent, cap, pos0 + ntok - 1, lost, seed=9)
    rng = np.random.default_rng(10)
    q = rng.standard_normal((ntok, n_heads * hd)).astype(np.float32)
    out = np.full((ntok, n_heads * hd), np.nan, np.float32)
    scale = np.float32(1 / np.sqrt(hd) / (hd & -hd))
    stride = n_kv * hd // 2 + n_kv * hd // 16
    # FlashQ4Params: nHeads nKvHeads pos0 ntok scale qStride stride oStride window softcap recent ringCap
    p = pack('iiiifiiiifii', n_heads, n_kv, pos0, ntok, scale, n_heads * hd, stride, n_heads * hd, window, softcap,
             recent, cap)
    E.dispatch('attention_fa_d%d_q4' % hd, [q, kc, vc, out, kr, vr, tags], p, (n_heads, (ntok + 31) // 32), (128,))
    ref = attention_ref_q4(q, kd, vd, krf, vrf, tags, cap, recent, n_heads, n_kv, hd, pos0, ntok, scale, window, softcap)
    err = np.abs(out - ref).max()
    check(err < 2e-5, 'attention_fa_d%d_q4 pos0=%d ntok=%d recent=%d lost=%s window=%d softcap=%g: max err %.2e'
          % (hd, pos0, ntok, recent, list(lost), window, softcap, err))


flash_q4(128)
flash_q4(128, recent=0)
flash_q4(64, lost=(50,), window=20, pos0=60, ntok=40)
flash_q4(64, softcap=5.0, recent=40)
flash_q4(256, ntok=33)


def split_q4(hd, n_heads=4, n_kv=2, pos0=300, ctx=320, n_split=5, recent=64, lost=(), derive_chunk=False):
    """Flash decoding (attention_split_q4 + attention_combine) of one token over a q4 cache with a ring. With derive_chunk
    the chunk parameter is 0 and the kernel derives ceil(keys / nSplit) itself (what decode replay relies on)."""
    cap = recent + 8 if recent > 0 else 0
    kc, vc, kr, vr, tags, kd, vd, krf, vrf = q4_cache(ctx, n_kv, hd, recent, cap, pos0, lost, seed=11)
    rng = np.random.default_rng(12)
    q = rng.standard_normal((1, n_heads * hd)).astype(np.float32)
    stride = n_kv * hd // 2 + n_kv * hd // 16
    keys = pos0 + 1
    chunk = 0 if derive_chunk else (keys + n_split - 1) // n_split
    part = np.zeros(n_heads * n_split * (hd + 2), np.float32)
    scores = np.zeros(n_heads * ctx, np.float32)
    out = np.zeros((1, n_heads * hd), np.float32)
    scale = np.float32(1 / np.sqrt(hd) / (hd & -hd))
    # AttnSplitQ4Params: nHeads nKvHeads headDim pos0 ctx scale qStride stride window softcap nSplit chunk recent ringCap
    E.dispatch('attention_split_q4', [q, kc, vc, part, scores, kr, vr, tags],
               pack('iiiiifiiifiiii', n_heads, n_kv, hd, pos0, ctx, scale, n_heads * hd, stride, 0, 0.0, n_split, chunk,
                    recent, cap), (n_heads, n_split), (256,))
    E.dispatch('attention_combine', [part, out], pack('iii', n_heads, hd, n_split), (n_heads,), (256,))
    ref = attention_ref_q4(q, kd, vd, krf, vrf, tags, cap, recent, n_heads, n_kv, hd, pos0, 1, scale)
    err = np.abs(out - ref).max()
    check(err < 1e-4, 'attention_split_q4 hd=%d, %d splits, recent=%d lost=%s derive_chunk=%s: max err %.2e'
          % (hd, n_split, recent, list(lost), derive_chunk, err))


split_q4(128)
split_q4(128, lost=(290,))
split_q4(64, recent=0)
split_q4(128, derive_chunk=True)
split_q4(128, lost=(290,), n_split=7, derive_chunk=True)


# ---------------------------------------------------------------------------------------------- sparse decode attention (docs/26)
# kv_summary, sparse_score, sparse_select and attention_dec_sp: the block summaries of a paged cache, the Quest bound, the
# top-k selection and the attention over the selected blocks, each against numpy, then the four in a row against a numpy
# implementation of the whole step.

def half_down(x):
    """The largest half <= x (float32 x clamped to the finite halves): KvSummary.down."""
    x = np.clip(x, -65504, 65504).astype(np.float32)
    h = x.astype(np.float16)
    with np.errstate(over='ignore'):
        return np.where(h.astype(np.float32) > x, np.nextafter(h, np.float16(-np.inf)), h).astype(np.float16)


def half_up(x):
    x = np.clip(x, -65504, 65504).astype(np.float32)
    h = x.astype(np.float16)
    with np.errstate(over='ignore'):
        return np.where(h.astype(np.float32) < x, np.nextafter(h, np.float16(np.inf)), h).astype(np.float16)


def kv_cache_rows(kind, k0, n_kv, hd, rotate=False):
    """The key rows of a cache of `kind` ('f32', 'f16', 'q8') as the engine stores them for the float rows k0, with the
    row stride in elements and the dequantized values; `rotate`: q8 of Hadamard-rotated keys."""
    if kind == 'q8':
        if rotate:
            k0 = q4_rotate(k0.reshape(len(k0), n_kv, hd)).reshape(len(k0), -1)
        rows, deq = q8_rows(k0, n_kv, hd)
        return rows, n_kv * hd + 4 * n_kv, deq, '_q8'
    if kind == 'f16':
        h = k0.astype(np.float16)
        return h, n_kv * hd, h.astype(np.float32), '_h'
    return k0, n_kv * hd, k0, ''


def summary_ref(deq, n_kv, hd, lo_row, hi_row):
    """Per kv head, channel min / max over rows [lo_row, hi_row) of the dequantized keys, rounded outwards to halves:
    ([n_kv, hd] lo, [n_kv, hd] hi)."""
    x = deq[lo_row:hi_row].reshape(hi_row - lo_row, n_kv, hd)
    return half_down(x.min(axis=0)), half_up(x.max(axis=0))


def kv_summary(kind, hd, n_kv=2, pos=13, n=70, ctx=128, page=4, rotate=False):
    rng = np.random.default_rng(31 + hd + page)
    k0 = rng.standard_normal((ctx, n_kv * hd)).astype(np.float32)
    k0[:, 3] *= 9
    k0[:, hd + 5] *= 0.001
    k0[7, 2] = 70000.0 if kind == 'f32' else k0[7, 2]  # beyond the largest half: clamped
    rows, stride, deq, suffix = kv_cache_rows(kind, k0, n_kv, hd, rotate)
    block = 1 << page
    nb = (ctx + block - 1) // block
    kc, tbl = paged([rows], page)
    total = len(kc) // block
    sums = np.full(total * n_kv * 2 * hd, 0x7BFF, np.uint16)   # garbage (a large half) where nothing is written
    first, last = pos >> page, (pos + n - 1) >> page
    E.dispatch('kv_summary' + suffix, [kc, sums, tbl], pack('iiiiii', hd, n_kv, stride, pos, n, page), (n_kv, last - first + 1), (hd,))
    ok = True
    for lb in range(nb):
        phys = tbl[lb]
        got = sums[phys * n_kv * 2 * hd:(phys + 1) * n_kv * 2 * hd].reshape(n_kv, 2, hd)
        if first <= lb <= last:
            lo, hi = summary_ref(deq, n_kv, hd, lb * block, min((lb + 1) * block, pos + n))
            ok &= bool((got[:, 0].view(np.uint16) == lo.view(np.uint16)).all() and (got[:, 1].view(np.uint16) == hi.view(np.uint16)).all())
        else:
            ok &= bool((got == 0x7BFF).all())
    check(ok, 'kv_summary%s hd=%d%s, rows %d..%d in blocks of %d: bit for bit the outward-rounded min / max of the valid rows'
          % (suffix, hd, ' (rotated q8)' if rotate else '', pos, pos + n - 1, block))


for _hd in (64, 128, 256):
    for _kind in ('f32', 'f16', 'q8'):
        kv_summary(_kind, _hd)
kv_summary('q8', 128, rotate=True)
kv_summary('f16', 128, pos=0, n=1)                  # a decode step on the first position
kv_summary('f32', 64, pos=47, n=1, page=5)          # a decode step inside a block of 32
kv_summary('q8', 64, pos=100, n=3, page=3)          # blocks of 8
kv_summary('f32', 128, pos=64, n=64, page=6)        # one whole block


def sparse_summaries(deq, n_kv, hd, block, nb, upto):
    """lo / hi halves of every block of the first `upto` rows: [nb, n_kv, 2, hd] float16."""
    out = np.zeros((nb, n_kv, 2, hd), np.float16)
    for lb in range(nb):
        hi_row = min((lb + 1) * block, upto)
        if hi_row > lb * block:
            lo, hi = summary_ref(deq, n_kv, hd, lb * block, hi_row)
            out[lb, :, 0], out[lb, :, 1] = lo, hi
    return out


def bound_ref(q, sm, n_heads, n_kv, hd, nblocks):
    """The bound of every block j < nblocks for each kv head: the largest over the group of sum_c max(q_c lo_c, q_c hi_c)."""
    grp = n_heads // n_kv
    out = np.zeros((n_kv, nblocks), np.float64)
    for g in range(n_kv):
        for j in range(nblocks):
            lo, hi = sm[j, g, 0].astype(np.float64), sm[j, g, 1].astype(np.float64)
            out[g, j] = max(np.maximum(q[(g * grp + r) * hd:(g * grp + r + 1) * hd] * lo,
                                       q[(g * grp + r) * hd:(g * grp + r + 1) * hd] * hi).sum() for r in range(grp))
    return out


def sparse_score(hd, n_heads=8, n_kv=2, pos=600, ctx=800, page=4, per_group=8, seed=0):
    rng = np.random.default_rng(41 + hd + n_heads + seed)
    block = 1 << page
    nb = (ctx + block - 1) // block
    q = rng.standard_normal(n_heads * hd).astype(np.float32)
    k0 = (rng.standard_normal((ctx, n_kv * hd)) * rng.random(n_kv * hd) * 3).astype(np.float32)
    sm = sparse_summaries(k0, n_kv, hd, block, nb, ctx)
    # the pool: the summaries at the blocks' physical places, garbage elsewhere
    total = nb + 3
    ids = np.random.default_rng(page * 7919 + ctx).permutation(total)[:nb].astype(np.int32)
    pool = np.full((total, n_kv, 2, hd), 0x7BFF, np.uint16).view(np.float16)
    for lb in range(nb):
        pool[ids[lb]] = sm[lb]
    nblocks = (pos >> page) + 1
    stride = nb
    scores = np.full(n_kv * stride, np.nan, np.float32)
    groups = (nb + per_group - 1) // per_group
    E.dispatch('sparse_score_d%d' % hd, [q, pool, ids, scores], pack('iiiiii', n_heads, n_kv, pos, page, stride, per_group),
               (groups, n_kv), (128,))
    ref = bound_ref(q, sm, n_heads, n_kv, hd, nblocks)
    got = scores.reshape(n_kv, stride)[:, :nblocks].astype(np.float64)
    err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9)
    untouched = np.isnan(scores.reshape(n_kv, stride)[:, nblocks:]).all()
    check(err < 1e-5 and untouched, 'sparse_score_d%d, %d heads over %d kv heads, %d blocks: max relative error %.1e, later blocks untouched %s'
          % (hd, n_heads, n_kv, nblocks, err, untouched))


for _hd in (64, 128, 256):
    sparse_score(_hd)
sparse_score(128, n_heads=4, n_kv=4)                 # one query head per kv head
sparse_score(128, n_heads=8, n_kv=1)                 # a group of eight
sparse_score(64, pos=37, ctx=64, per_group=4)        # three blocks
sparse_score(128, pos=0, ctx=64)                     # one block
sparse_score(128, pos=799, ctx=800, per_group=64)    # last position, big groups


def select_ref(scores, tbl, nblocks, sink, win, topk):
    c1 = nblocks - win
    cand = np.arange(sink, c1)
    order = np.lexsort((cand, -scores[cand].astype(np.float64)))     # higher score first, the lower block on a tie
    top = np.sort(cand[order[:topk]])
    return np.concatenate([tbl[:sink], tbl[top], tbl[c1:nblocks]]).astype(np.int32)


def sparse_select(nblocks, sink, win, topk, kind='random', n_kv=2, page=4, seed=0):
    rng = np.random.default_rng(51 + nblocks + topk + seed)
    nb = nblocks + 5
    stride = nb
    if kind == 'ties':       # a handful of distinct values: most candidates tie with the k-th
        sc = rng.integers(-2, 3, (n_kv, stride)).astype(np.float32)
    elif kind == 'equal':    # every candidate equal: the lowest blocks win
        sc = np.full((n_kv, stride), 1.5, np.float32)
    elif kind == 'wide':     # magnitudes over many powers of two, both signs
        sc = ((rng.standard_normal((n_kv, stride)) * 10.0 ** rng.integers(-6, 6, (n_kv, stride)))).astype(np.float32)
    else:
        sc = rng.standard_normal((n_kv, stride)).astype(np.float32)
    sc = np.ascontiguousarray(sc)
    sel_stride = sink + topk + win + 3
    tbl = np.random.default_rng(page * 7919 + nb).permutation(nb + 7)[:nb].astype(np.int32)
    sel = np.full(n_kv * sel_stride, -1, np.int32)
    pos = (nblocks - 1) * (1 << page) + 5
    E.dispatch('sparse_select', [sc.reshape(-1), tbl, sel], pack('iiiiiiii', n_kv, pos, page, sink, win, topk, stride, sel_stride),
               (n_kv,), (256,))
    got = sel.reshape(n_kv, sel_stride)
    ok = True
    for g in range(n_kv):
        ref = select_ref(sc[g], tbl, nblocks, sink, win, topk)
        ok &= bool((got[g, :len(ref)] == ref).all()) and bool((got[g, len(ref):] == -1).all())
    check(ok, 'sparse_select, %d blocks, sink %d, window %d, top %d, %s scores: sinks, the chosen blocks ascending, the window'
          % (nblocks, sink, win, topk, kind))


for _kind in ('random', 'ties', 'equal', 'wide'):
    sparse_select(300, 1, 4, 16, _kind)
    sparse_select(40, 1, 2, 8, _kind)
sparse_select(5000, 2, 16, 64, 'random')
sparse_select(5000, 1, 16, 64, 'ties', seed=1)
sparse_select(90, 1, 2, 1, 'random')                  # top one
sparse_select(90, 0, 1, 7, 'random')                  # no sinks, one window block
sparse_select(30, 1, 2, 26, 'random')                 # nearly every candidate


def sparse_dec(hd, G, kind='f32', n_heads=8, n_kv=2, pos0=600, n_split=1, softcap=0.0, ctx=640, page=4, sel_blocks=24, rotate=False,
               seed=0):
    """attention_dec_sp over a random selection of `sel_blocks` blocks (always the one holding pos0, last), against the
    float64 attention over the selected positions of the dequantized cache."""
    rng = np.random.default_rng(61 + hd + G + seed)
    block = 1 << page
    q = rng.standard_normal((1, n_heads * hd)).astype(np.float32)
    k0 = rng.standard_normal((ctx, n_kv * hd)).astype(np.float32)
    v0 = rng.standard_normal((ctx, n_kv * hd)).astype(np.float32)
    krows, stride, kd, suffix = kv_cache_rows(kind, k0, n_kv, hd, rotate)
    vrows, _, vd, _ = kv_cache_rows(kind, v0, n_kv, hd)
    nblocks = (pos0 >> page) + 1
    last = nblocks - 1
    sels = []
    for g in range(n_kv):
        pick = np.sort(rng.choice(last, sel_blocks - 1, replace=False))
        sels.append(np.concatenate([pick, [last]]))
    kc, vc, tbl = paged([krows, vrows], page)
    sel_stride = sel_blocks + 2
    sel = np.full(n_kv * sel_stride, -1, np.int32)
    for g in range(n_kv):
        sel[g * sel_stride:g * sel_stride + sel_blocks] = tbl[sels[g]]
    keys = (sel_blocks - 1) * block + (pos0 & (block - 1)) + 1
    part = np.zeros(n_heads * n_split * (hd + 2), np.float32)
    out = np.zeros((1, n_heads * hd), np.float32)
    scale = 1 / np.sqrt(hd)
    # AttnDecSpParams: AttnSplitParams (nHeads nKvHeads headDim pos0 ctx scale qStride kvStride window softcap nSplit chunk pageShift) + selStride selBlocks
    E.dispatch('attention_dec_sp_g%d_d%d%s' % (G, hd, suffix), [q, kc, vc, part, out, sel],
               pack('iiiiifiiifiiiii', n_heads, n_kv, hd, pos0, ctx, scale, n_heads * hd, stride, 0, softcap, n_split, 0, page,
                    sel_stride, sel_blocks), (n_heads // G, (n_split + 3) // 4), (128,))
    if n_split > 1:
        E.dispatch('attention_combine', [part, out], pack('iii', n_heads, hd, n_split), (n_heads,), (256,))
    grp = n_heads // n_kv
    ref = np.zeros((1, n_heads * hd), np.float64)
    for h in range(n_heads):
        g = h // grp
        pos = np.concatenate([np.arange(b * block, min((b + 1) * block, pos0 + 1)) for b in sels[g]])
        assert len(pos) == keys
        s = kd[pos][:, g * hd:(g + 1) * hd].astype(np.float64) @ q[0, h * hd:(h + 1) * hd].astype(np.float64) * scale
        if softcap > 0:
            s = np.tanh(s / softcap) * softcap
        p = np.exp(s - s.max())
        p /= p.sum()
        ref[0, h * hd:(h + 1) * hd] = p @ vd[pos][:, g * hd:(g + 1) * hd].astype(np.float64)
    err = np.abs(out - ref).max()
    check(err < 1e-4, 'attention_dec_sp_g%d_d%d%s, %d selected keys, %d split(s)%s%s: max err %.2e'
          % (G, hd, suffix, keys, n_split, ', softcap' if softcap else '', ', blocks of %d' % block, err))


for _hd in (64, 128, 256):
    for _kind in ('f32', 'f16', 'q8'):
        for _g in (1, 2) if _hd == 256 else (1, 2, 4):
            sparse_dec(_hd, _g, _kind)
            sparse_dec(_hd, _g, _kind, n_split=3, pos0=611)
sparse_dec(128, 2, 'q8', rotate=True, n_split=3)
sparse_dec(64, 2, 'f32', softcap=3.0, n_split=5)
sparse_dec(128, 2, 'f16', page=5, pos0=1000, ctx=1024, sel_blocks=12, n_split=2)
sparse_dec(128, 2, 'f32', page=4, pos0=16 * 40, ctx=704, sel_blocks=10)             # the token is the first row of its block
sparse_dec(128, 2, 'f32', page=4, pos0=16 * 40 + 15, ctx=704, sel_blocks=10)        # ... and the last


def sparse_step(hd, kind, n_heads=8, n_kv=2, pos=1003, ctx=1024, page=4, sink=1, win=3, topk=20, rotate=False, needle=700):
    """The four kernels in a row, as MetalKernels.sparseDecodeAttention dispatches them, against a numpy step: summaries of
    the block of the token, bounds of every block, selection, attention over the selection. A key along the first query head
    is planted at `needle`; its block must be among the chosen."""
    rng = np.random.default_rng(71 + hd)
    block = 1 << page
    grp = n_heads // n_kv
    q = rng.standard_normal(n_heads * hd).astype(np.float32)
    k0 = (rng.standard_normal((ctx, n_kv * hd)) * (1 + 0.5 * rng.random(n_kv * hd))).astype(np.float32)
    v0 = rng.standard_normal((ctx, n_kv * hd)).astype(np.float32)
    for g in range(n_kv):
        k0[needle, g * hd:(g + 1) * hd] = 2.0 * q[g * grp * hd:(g * grp + 1) * hd]
    # a rotated cache scores the rotated query against rotated keys: the host hands the kernels the rotated query
    qq = q
    if rotate:
        qq = q4_rotate(q.reshape(n_heads, hd)).reshape(-1)
    krows, stride, kd, suffix = kv_cache_rows(kind, k0, n_kv, hd, rotate)
    vrows, _, vd, _ = kv_cache_rows(kind, v0, n_kv, hd)
    nb = (ctx + block - 1) // block
    kc, vc, tbl = paged([krows, vrows], page)
    total = len(kc) // block
    sums = np.full(total * n_kv * 2 * hd, 0x7BFF, np.uint16)
    # summaries of every block below the token's: written earlier (here by the same kernel), then the token's block alone
    nblocks = (pos >> page) + 1
    E.dispatch('kv_summary' + suffix, [kc, sums, tbl], pack('iiiiii', hd, n_kv, stride, 0, pos + 1, page), (n_kv, nblocks), (hd,))
    E.dispatch('kv_summary' + suffix, [kc, sums, tbl], pack('iiiiii', hd, n_kv, stride, pos, 1, page), (n_kv, 1), (hd,))
    scores = np.full(n_kv * nb, np.nan, np.float32)
    per_group = 8
    E.dispatch('sparse_score_d%d' % hd, [qq, sums.view(np.float16), tbl, scores],
               pack('iiiiii', n_heads, n_kv, pos, page, nb, per_group), ((nb + per_group - 1) // per_group, n_kv), (128,))
    nsel = sink + topk + win
    sel_stride = nsel
    sel = np.full(n_kv * sel_stride, -1, np.int32)
    E.dispatch('sparse_select', [scores, tbl, sel], pack('iiiiiiii', n_kv, pos, page, sink, win, topk, nb, sel_stride), (n_kv,), (256,))
    part = np.zeros(n_heads * 3 * (hd + 2), np.float32)
    out = np.zeros((1, n_heads * hd), np.float32)
    scale = np.float32(1 / np.sqrt(hd))
    if rotate:
        scale = np.float32(scale / (hd & -hd))
    n_split = 3
    E.dispatch('attention_dec_sp_g2_d%d%s' % (hd, suffix), [qq.reshape(1, -1), kc, vc, part, out, sel],
               pack('iiiiifiiifiiiii', n_heads, n_kv, hd, pos, ctx, scale, n_heads * hd, stride, 0, 0.0, n_split, 0, page,
                    sel_stride, nsel), (n_heads // 2, (n_split + 3) // 4), (128,))
    E.dispatch('attention_combine', [part, out], pack('iii', n_heads, hd, n_split), (n_heads,), (256,))
    # numpy: the same step
    sm = sparse_summaries(kd, n_kv, hd, block, nb, pos + 1)
    bound = bound_ref(qq, sm, n_heads, n_kv, hd, nblocks)
    ref = np.zeros((1, n_heads * hd), np.float64)
    chosen = []
    ok_sel = True
    for g in range(n_kv):
        cand = np.arange(sink, nblocks - win)
        order = np.lexsort((cand, -bound[g, cand]))
        top = np.sort(cand[order[:topk]])
        blocks = np.concatenate([np.arange(sink), top, np.arange(nblocks - win, nblocks)])
        chosen.append(blocks)
        ok_sel &= bool((sel.reshape(n_kv, sel_stride)[g] == tbl[blocks]).all())
        for r in range(grp):
            h = g * grp + r
            pos_l = np.concatenate([np.arange(b * block, min((b + 1) * block, pos + 1)) for b in blocks])
            s = kd[pos_l][:, g * hd:(g + 1) * hd].astype(np.float64) @ qq[h * hd:(h + 1) * hd].astype(np.float64) * float(scale)
            p = np.exp(s - s.max())
            p /= p.sum()
            ref[0, h * hd:(h + 1) * hd] = p @ vd[pos_l][:, g * hd:(g + 1) * hd].astype(np.float64)
    found = all((needle >> page) in c for c in chosen)
    err = np.abs(out - ref).max()
    check(ok_sel and found and err < 1e-4, 'sparse step%s hd=%d%s: selection equals numpy %s, planted key found %s, attention max err %.2e'
          % (suffix, hd, ' (rotated q8)' if rotate else '', ok_sel, found, err))


for _hd in (64, 128, 256):
    for _kind in ('f32', 'f16', 'q8'):
        sparse_step(_hd, _kind)
sparse_step(128, 'q8', rotate=True)


if not ONLY:
    # the image-generation kernels (diffusion.metal): gemm, convolution, normalisation, attention, layout
    import kernel_tests_diffusion

    kernel_tests_diffusion.run(E, check)
sys.exit(1 if failures else 0)
