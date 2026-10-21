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


def flash(hd, n_heads=4, n_kv=2, pos0=37, ntok=45, ctx=128, window=0, softcap=0.0, half=False, q8=False):
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
    name = 'attention_fa_d%d%s' % (hd, '_h' if half else '_q8' if q8 else '')
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
        p = pack('iiiiifiiifii', n_heads, n_kv, hd, pos0, ctx, 1 / np.sqrt(hd), n_heads * hd, n_kv * hd, window, 0.0,
                 n_split, chunk)
        E.dispatch('attention_split', [q, kc, vc, part, scores], p, (n_heads, n_split), (256,))
        E.dispatch('attention_combine', [part, out], pack('iii', n_heads, hd, n_split), (n_heads,), (256,))
        outs.append(out)
    ref = attention_ref(q, kc, vc, n_heads, n_kv, hd, pos0, 1, 1 / np.sqrt(hd), window)
    err = np.abs(outs[1] - ref).max()
    check(np.array_equal(outs[0], outs[1]) and err < 1e-4,
          'attention_split chunk derived in the kernel (window %d): %s the explicit chunk, max err %.2e'
          % (window, 'same bytes as' if np.array_equal(outs[0], outs[1]) else 'DIFFERENT from', err))


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
    E.dispatch('kv_store_q8', [k_a, v, kc_a, vc_a], pack('iiii', hd, kv_dim, n_kv, stride), (2 * n_kv, ntok), (32,))
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


router_norm()
router_norm(remap=True, ntok=5)
router_norm(ds=True)
router_norm(ds=True, method=1, sigmoid=0, remap=True)
sys.exit(1 if failures else 0)
