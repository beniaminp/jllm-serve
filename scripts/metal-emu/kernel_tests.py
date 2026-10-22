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


if not ONLY:
    split_q8(128)


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


if not ONLY:
    router_norm()
    router_norm(remap=True, ntok=5)
    router_norm(ds=True)
    router_norm(ds=True, method=1, sigmoid=0, remap=True)


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


sys.exit(1 if failures else 0)
