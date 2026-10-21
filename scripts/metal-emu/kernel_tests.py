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
    E.dispatch('gdn_conv', [x, w, s, y], pack('iiiii', channels, kc, ntok, in_stride, channels),
               ((channels + 63) // 64,), (64,))
    err = max(np.abs(y - ref).max(), np.abs(s - want_state).max())
    check(err < 1e-5, 'gdn_conv channels=%d kernel=%d ntok=%d: max err %.2e' % (channels, kc, ntok, err))


def gdn_rule(nk=2, nv=4, dk=64, dv=12, ntok=5):
    """The gated delta rule: l2norm(q) / sqrt(dk), l2norm(k), beta = sigmoid(b), g = A softplus(a + dt_bias),
    S = exp(g) S; S += k ((v - S^T k) beta)^T; o = S^T q, against a float64 loop (state stored [head][j][dk])."""
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
            kh = h // r
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
    st = S0.copy()
    E.dispatch('gdn_rule', [qkv, proj, A, dtb, st, out],
               pack('iiiiiiiif', nk, nv, dk, dv, ntok, conv, proj_stride, b_off, 1e-6), ((dv + 3) // 4, nv), (128,))
    err = max(np.abs(out - ref.reshape(ntok, -1)).max(), np.abs(st - S.transpose(0, 2, 1)).max())
    check(err < 1e-5, 'gdn_rule nk=%d nv=%d dk=%d dv=%d ntok=%d: max err %.2e' % (nk, nv, dk, dv, ntok, err))


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


gdn_conv()
gdn_conv(channels=70, kc=3, ntok=1, in_stride=80)
gdn_rule()
gdn_rule(nk=2, nv=2, dk=128, dv=32, ntok=3)
gdn_rule(nk=1, nv=3, dk=32, dv=5, ntok=7)
gdn_norm_gate()
sys.exit(1 if failures else 0)
