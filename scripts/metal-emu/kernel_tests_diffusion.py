"""Kernel tests of the image-generation kernels (src/main/resources/org/robomiri/jllm/metal/diffusion.metal) on the Metal
emulator against float64 numpy references; run from kernel_tests.py (which passes its emulator and its check function).

Every kernel is tested at its awkward sizes (rows / columns / k that are not multiples of the 64 x 64 x 32 tiles, a
width of 40 or 77 keys) in float32 and in float16 (half inputs and outputs; the references see the float16-rounded
inputs, the tolerance is the half rounding of the result)."""
import struct

import numpy as np

SILU, GELU, QUICK = 1, 2, 3


def erf64(x):
    from math import erf
    return np.vectorize(erf)(x)


def act_ref(v, act):
    if act == SILU:
        return v / (1 + np.exp(-v))
    if act == GELU:
        return 0.5 * v * (1 + erf64(v / np.sqrt(2)))
    if act == QUICK:
        return v / (1 + np.exp(-1.702 * v))
    return v


def pack(fmt, *values):
    return struct.pack('<' + fmt, *values)


def run(E, check):
    rng = np.random.default_rng(2024)

    def tol(dtype, ref):
        return (2e-5 if dtype == np.float32 else 1.2e-2) * max(1.0, float(np.abs(ref).max()))

    def rnd(dtype, *shape, scale=1.0):
        return (rng.standard_normal(shape) * scale).astype(dtype)

    sfx = {np.float32: '_f', np.float16: '_h'}

    # ---------------------------------------------------------------- df_gemm_nt: act(A W^T + bias) + res
    def gemm(dtype, m, n, k, act=0, bias=True, res=True, vec=None):
        a, w = rnd(dtype, m, k), rnd(dtype, n, k, scale=0.3)
        b = rnd(np.float32, n)
        r = rnd(dtype, m, n)
        out = np.zeros((m, n), dtype)
        if vec is None:
            vec = int(k % 4 == 0 and n % 4 == 0)
        # GemmParams: m n k lda ldb ldc heads aSB aSH bSB bSH cSB cSH alpha hasBias hasRes act vec
        p = pack('iiiiiiiiiiiiifiiii', m, n, k, k, k, n, 1, 0, 0, 0, 0, 0, 0, 1.0, int(bias), int(res), act, vec)
        E.dispatch('df_gemm_nt' + sfx[dtype], [a, w, out, b, r], p, ((n + 63) // 64, (m + 63) // 64, 1), (128,))
        ref = a.astype(np.float64) @ w.astype(np.float64).T
        if bias:
            ref = ref + b
        ref = act_ref(ref, act)
        if res:
            ref = ref + r.astype(np.float64)
        err = np.abs(out - ref).max()
        check(err < tol(dtype, ref), 'df_gemm_nt%s m=%d n=%d k=%d act=%d bias=%s res=%s vec=%d: max err %.2e'
              % (sfx[dtype], m, n, k, act, bias, res, vec, err))

    for dt in (np.float32, np.float16):
        gemm(dt, 70, 45, 75)                      # nothing a multiple of a tile
        gemm(dt, 64, 64, 64, act=SILU)
        gemm(dt, 33, 130, 40, act=GELU, vec=0)    # k = 40: a 32-wide slice and an 8-wide tail; scalar loads
        gemm(dt, 5, 72, 96, act=QUICK, res=False)
        gemm(dt, 3, 20, 8, bias=False, res=False)

    # ---------------------------------------------------------------- scores, softmax, P V: naive attention, batched
    def naive_attention(dtype, batch, heads, lq, lk, hd, causal):
        dm = heads * hd
        q, k, v = rnd(dtype, batch, lq, dm), rnd(dtype, batch, lk, dm), rnd(dtype, batch, lk, dm)
        ld = (lk + 7) // 8 * 8
        scale = 1 / np.sqrt(hd)
        out = np.zeros((batch, lq, dm), dtype)
        ref = np.zeros((batch, lq, dm), np.float64)
        for b in range(batch):
            for h in range(heads):
                qh = q[b, :, h * hd:(h + 1) * hd].astype(np.float64)
                kh = k[b, :, h * hd:(h + 1) * hd].astype(np.float64)
                vh = v[b, :, h * hd:(h + 1) * hd].astype(np.float64)
                s = qh @ kh.T * scale
                if causal:
                    s = np.where(np.arange(lk)[None, :] <= np.arange(lq)[:, None], s, -np.inf)
                pr = np.exp(s - s.max(1, keepdims=True))
                pr /= pr.sum(1, keepdims=True)
                ref[b, :, h * hd:(h + 1) * hd] = pr @ vh
        vec = int(dm % 4 == 0 and hd % 4 == 0)
        for b in range(batch):
            scores = np.zeros((heads, lq, ld), np.float32)
            qb, kb, vb = q[b].copy(), k[b].copy(), v[b].copy()
            ob = np.zeros((lq, dm), dtype)
            p = pack('iiiiiiiiiiiiifiiii', lq, lk, hd, dm, dm, ld, heads, 0, hd, 0, hd, 0, lq * ld, scale, 0, 0, 0, vec)
            name = 'df_scores_h' if dtype == np.float16 else 'df_gemm_nt_f'
            E.dispatch(name, [qb, kb, scores, scores, scores], p, ((lk + 63) // 64, (lq + 63) // 64, heads), (128,))
            E.dispatch('df_softmax', [scores], pack('iiii', lk, ld, lq, int(causal)), (heads * lq,), (256,))
            p = pack('iiiiiiiiiiiiifiiii', lq, hd, lk, ld, dm, dm, heads, 0, lq * ld, 0, hd, 0, hd, 1.0, 0, 0, 0, vec)
            E.dispatch('df_pv' + sfx[dtype], [scores, vb, ob, ob, ob], p, ((hd + 63) // 64, (lq + 63) // 64, heads), (128,))
            out[b] = ob
        err = np.abs(out - ref).max()
        check(err < tol(dtype, ref) * 0.5, 'naive attention%s b=%d h=%d lq=%d lk=%d hd=%d causal=%s: max err %.2e'
              % (sfx[dtype], batch, heads, lq, lk, hd, causal, err))

    for dt in (np.float32, np.float16):
        naive_attention(dt, 2, 3, 37, 77, 40, False)
        naive_attention(dt, 1, 2, 30, 30, 64, True)
        naive_attention(dt, 1, 1, 20, 50, 100, False)   # a head that is not a flash width

    # ---------------------------------------------------------------- df_flash_d{D}: flash attention
    def flash(dtype, batch, heads, lq, lk, hd, causal):
        dm = heads * hd
        q, k, v = rnd(dtype, batch, lq, dm), rnd(dtype, batch, lk, dm), rnd(dtype, batch, lk, dm)
        out = np.zeros((batch, lq, dm), dtype)
        scale = 1 / np.sqrt(hd)
        ref = np.zeros((batch, lq, dm), np.float64)
        for b in range(batch):
            for h in range(heads):
                qh = q[b, :, h * hd:(h + 1) * hd].astype(np.float64)
                kh = k[b, :, h * hd:(h + 1) * hd].astype(np.float64)
                vh = v[b, :, h * hd:(h + 1) * hd].astype(np.float64)
                s = qh @ kh.T * scale
                if causal:
                    s = np.where(np.arange(lk)[None, :] <= np.arange(lq)[:, None], s, -np.inf)
                pr = np.exp(s - s.max(1, keepdims=True))
                pr /= pr.sum(1, keepdims=True)
                ref[b, :, h * hd:(h + 1) * hd] = pr @ vh
        # FlashParams: heads lq lk qs ks vs os scale causal
        p = pack('iiiiiiifi', heads, lq, lk, dm, dm, dm, dm, scale, int(causal))
        E.dispatch('df_flash_d%d%s' % (hd, sfx[dtype]), [q, k, v, out], p, (heads, (lq + 31) // 32, batch), (128,))
        err = np.abs(out - ref).max()
        check(err < tol(dtype, ref) * 0.5, 'df_flash_d%d%s b=%d h=%d lq=%d lk=%d causal=%s: max err %.2e'
              % (hd, sfx[dtype], batch, heads, lq, lk, causal, err))

    for dt in (np.float32, np.float16):
        flash(dt, 2, 2, 45, 77, 40, False)       # cross attention: 77 text tokens, 45 queries (a partial tile)
        flash(dt, 1, 2, 40, 40, 64, True)        # causal self attention (the text encoder)
        flash(dt, 1, 1, 33, 70, 8, False)
    flash(np.float32, 1, 1, 32, 32, 160, False)

    # ---------------------------------------------------------------- df_conv: implicit gemm convolution
    def conv_ref(x, w, bias, chan, res, stride, pad_t, pad_l, ho, wo):
        b, h, wd, ci = x.shape
        co, kh, kw, _ = w.shape
        xp = np.zeros((b, h + pad_t + kh + stride * ho, wd + pad_l + kw + stride * wo, ci), np.float64)
        xp[:, pad_t:pad_t + h, pad_l:pad_l + wd] = x
        out = np.zeros((b, ho, wo, co), np.float64)
        for ky in range(kh):
            for kx in range(kw):
                patch = xp[:, ky:ky + stride * ho:stride, kx:kx + stride * wo:stride]
                out += patch @ w[:, ky, kx, :].astype(np.float64).T
        if bias is not None:
            out += bias
        if chan is not None:
            out += chan.astype(np.float64)[:, None, None, :]
        if res is not None:
            out += res.astype(np.float64)
        return out

    def conv(dtype, b, h, wd, ci, co, kh, kw, stride, pad_t, pad_l, pad_b, pad_r, use_bias=True, use_chan=True, use_res=True, direct=False):
        ho = (h + pad_t + pad_b - kh) // stride + 1
        wo = (wd + pad_l + pad_r - kw) // stride + 1
        x, w = rnd(dtype, b, h, wd, ci), rnd(dtype, co, kh, kw, ci, scale=0.2)
        bias = rnd(np.float32, co)
        chan = rnd(dtype, b, co)
        res = rnd(dtype, b, ho, wo, co)
        out = np.zeros((b, ho, wo, co), dtype)
        # ConvParams: batch h w cin ho wo cout kh kw stride padT padL hasBias hasChan hasRes
        p = pack('iiiiiiiiiiiiiii', b, h, wd, ci, ho, wo, co, kh, kw, stride, pad_t, pad_l, int(use_bias), int(use_chan), int(use_res))
        m = b * ho * wo
        if direct:
            E.dispatch('df_conv_direct' + sfx[dtype], [x, w, out, bias, chan, res], p, ((m * co + 255) // 256,), (256,))
        else:
            E.dispatch('df_conv' + sfx[dtype], [x, w, out, bias, chan, res], p, ((co + 63) // 64, (m + 63) // 64, 1), (128,))
        ref = conv_ref(x.astype(np.float64), w, bias if use_bias else None, chan if use_chan else None,
                       res if use_res else None, stride, pad_t, pad_l, ho, wo)
        err = np.abs(out - ref).max()
        check(err < tol(dtype, ref), 'df_conv%s%s %dx%dx%d ci=%d co=%d k=%dx%d s=%d pad=%d,%d,%d,%d: max err %.2e'
              % ('_direct' if direct else '', sfx[dtype], b, h, wd, ci, co, kh, kw, stride, pad_t, pad_l, pad_b, pad_r, err))

    for dt in (np.float32, np.float16):
        conv(dt, 2, 9, 7, 16, 24, 3, 3, 1, 1, 1, 1, 1)
        conv(dt, 1, 10, 12, 8, 70, 3, 3, 2, 1, 1, 1, 1, use_res=False)       # the strided down-sampling convolution
        conv(dt, 1, 8, 8, 16, 16, 3, 3, 2, 0, 0, 1, 1, use_chan=False)       # the VAE's asymmetric padding
        conv(dt, 1, 5, 6, 24, 40, 1, 1, 1, 0, 0, 0, 0)
        conv(dt, 1, 6, 5, 4, 3, 3, 3, 1, 1, 1, 1, 1, direct=True)           # first / last layers: 4 -> 3 channels
        conv(dt, 2, 7, 7, 9, 8, 3, 3, 1, 1, 1, 1, 1, direct=True)           # an inpainting UNet's 9 input channels

    # ---------------------------------------------------------------- normalisation
    def groupnorm(dtype, b, hw, c, groups, silu):
        x = rnd(dtype, b, hw, c, scale=2.0)
        gamma, beta = rnd(np.float32, c), rnd(np.float32, c)
        out = np.zeros_like(x)
        eps = 1e-5
        gs = c // groups
        ref = np.zeros((b, hw, c), np.float64)
        for bi in range(b):
            for g in range(groups):
                sl = x[bi, :, g * gs:(g + 1) * gs].astype(np.float64)
                ref[bi, :, g * gs:(g + 1) * gs] = (sl - sl.mean()) / np.sqrt(sl.var() + eps)
        ref = ref * gamma + beta
        if silu:
            ref = act_ref(ref, SILU)
        # NormParams: c hw groups eps silu hasAffine
        E.dispatch('df_groupnorm' + sfx[dtype], [x, gamma, beta, out], pack('iiifii', c, hw, groups, eps, int(silu), 0), (groups, b), (256,))
        err = np.abs(out - ref).max()
        check(err < tol(dtype, ref), 'df_groupnorm%s b=%d hw=%d c=%d groups=%d silu=%s: max err %.2e' % (sfx[dtype], b, hw, c, groups, silu, err))

    def layernorm(dtype, rows, c, affine):
        x = rnd(dtype, rows, c, scale=3.0)
        gamma, beta = rnd(np.float32, c), rnd(np.float32, c)
        out = np.zeros_like(x)
        xd = x.astype(np.float64)
        ref = (xd - xd.mean(1, keepdims=True)) / np.sqrt(xd.var(1, keepdims=True) + 1e-5)
        if affine:
            ref = ref * gamma + beta
        E.dispatch('df_layernorm' + sfx[dtype], [x, gamma, beta, out], pack('iiifii', c, 1, 1, 1e-5, 0, int(affine)), (rows,), (256,))
        err = np.abs(out - ref).max()
        check(err < tol(dtype, ref), 'df_layernorm%s rows=%d c=%d affine=%s: max err %.2e' % (sfx[dtype], rows, c, affine, err))

    for dt in (np.float32, np.float16):
        groupnorm(dt, 2, 50, 64, 8, True)
        groupnorm(dt, 1, 33, 40, 5, False)
        layernorm(dt, 9, 320, True)
        layernorm(dt, 3, 77, False)

    # ---------------------------------------------------------------- elementwise and layout
    def ew(dtype):
        n = 1001
        x, y = rnd(dtype, n), rnd(dtype, n)
        out = np.zeros(n, dtype)
        s = sfx[dtype]
        E.dispatch('df_add' + s, [x, y, out], pack('qiiii', n, 0, 0, 0, 0), ((n + 255) // 256,), (256,))
        check(np.abs(out - (x.astype(np.float64) + y)).max() < tol(dtype, x), 'df_add' + s)
        E.dispatch('df_silu' + s, [x, out], pack('qiiii', n, 0, 0, 0, 0), ((n + 255) // 256,), (256,))
        check(np.abs(out - act_ref(x.astype(np.float64), SILU)).max() < tol(dtype, x), 'df_silu' + s)
        rows, half = 7, 30
        g = rnd(dtype, rows, 2 * half)
        o2 = np.zeros((rows, half), dtype)
        E.dispatch('df_geglu' + s, [g, o2], pack('qiiii', rows * half, half, 0, 0, 0), ((rows * half + 255) // 256,), (256,))
        gd = g.astype(np.float64)
        ref = gd[:, :half] * act_ref(gd[:, half:], GELU)
        check(np.abs(o2 - ref).max() < tol(dtype, ref), 'df_geglu' + s)
        b, h, w, c = 2, 3, 5, 8
        im = rnd(dtype, b, h, w, c)
        up = np.zeros((b, 2 * h, 2 * w, c), dtype)
        E.dispatch('df_upsample2x' + s, [im, up], pack('qiiii', up.size, h, w, c, 0), ((up.size + 255) // 256,), (256,))
        check(np.array_equal(up, im.repeat(2, axis=1).repeat(2, axis=2)), 'df_upsample2x' + s)
        a1, a2 = rnd(dtype, b, h, w, 5), rnd(dtype, b, h, w, 11)
        cat = np.zeros((b, h, w, 16), dtype)
        E.dispatch('df_concat' + s, [a1, a2, cat], pack('qiiii', cat.size, 5, 11, 0, 0), ((cat.size + 255) // 256,), (256,))
        check(np.array_equal(cat, np.concatenate([a1, a2], axis=-1)), 'df_concat' + s)

    ew(np.float32)
    ew(np.float16)

    # the softmax kernel alone: rows of different lengths (causal), padding zeroed
    lq, n, ld = 6, 13, 16
    s = rnd(np.float32, 2 * lq, ld, scale=3.0)
    want = np.zeros_like(s, dtype=np.float64)
    for r in range(2 * lq):
        k = min(n, r % lq + 1)
        e = np.exp(s[r, :k].astype(np.float64) - s[r, :k].max())
        want[r, :k] = e / e.sum()
    E.dispatch('df_softmax', [s], pack('iiii', n, ld, lq, 1), (2 * lq,), (256,))
    check(np.abs(s - want).max() < 1e-5, 'df_softmax causal rows with zeroed padding')
