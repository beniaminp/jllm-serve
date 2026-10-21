#!/usr/bin/env python3
"""Independent float64 numpy reference of the Qwen3-Next forward pass (Qwen3NextForCausalLM), written after
transformers' modeling_qwen3_next.py: zero-centred RMSNorm (x * (1 + w)), gated full attention (q_proj = [query |
gate] per head, q/k norm, rotate-half RoPE on the first partial_rotary_factor * head_dim dims, output * sigmoid(gate)),
the Gated DeltaNet (fix_query_key_value_ordering, depthwise causal conv + SiLU, l2-normalized q / k, q scaled by
head_k_dim^-0.5, beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias), the gated delta rule, the gated RMSNorm
rmsnorm(o) * w * silu(z)) and the sparse MoE block (softmax router, top-k, optional renormalization, shared expert
scaled by sigmoid(shared_expert_gate . x)).

usage: qwen3next-ref.py <tiny-model-dir>    (a directory written by make-tiny-model.py qwen3next*)
Runs the 12-token prompt of ArchSmoke and writes <dir>/ref-logits.json: the logits of the last prompt token and the 8
greedy tokens that follow. The delta rule runs token by token (torch_recurrent_gated_delta_rule); REF_CHUNKED=1 uses
the chunked form instead (torch_chunk_gated_delta_rule: UT transform solved as a triangular system, chunk REF_CHUNK,
default 64) - both must give the same numbers. With REF_Q8=1 every matrix the engine quantizes is first rounded to
Q8_0 as its converter does (the router and the shared-expert gate stay f32), so the engine's Q8_0 conversion can be
compared to float rounding. Needs numpy.
"""
import json
import math
import os
import struct
import sys

import numpy as np

PROMPT = [1, 7, 42, 99, 13, 5, 250, 8, 77, 31, 600, 9]
NEW_TOKENS = 8
F32_MATRICES = ("mlp.gate.weight", "shared_expert_gate.weight")


def q8(m):
    """Q8_0 round trip of a 2-D matrix as QTensor does it: blocks of 32 along each row, f32 scale d = absmax / 127,
    int8 values round(x * (1 / d)) in f32."""
    rows, cols = m.shape
    b = m.reshape(rows, cols // 32, 32).astype(np.float32)
    d = (np.abs(b).max(axis=2, keepdims=True) / np.float32(127)).astype(np.float32)
    inv = np.where(d == 0, np.float32(0), np.float32(1) / np.where(d == 0, np.float32(1), d)).astype(np.float32)
    qv = np.floor(b * inv + np.float32(0.5))  # Java's Math.round: halves round up
    return (qv.astype(np.float64) * d.astype(np.float64)).reshape(rows, cols)


def load(dir_):
    with open(os.path.join(dir_, "model.safetensors"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            a, b = info["data_offsets"]
            f.seek(base + a)
            t = np.frombuffer(f.read(b - a), dtype="<f4").astype(np.float64).reshape(info["shape"])
            if os.environ.get("REF_Q8") and t.ndim == 2 and not name.endswith(F32_MATRICES):
                t = q8(t)
            out[name] = t
    return out


def rms1p(x, w, eps):
    """Qwen3NextRMSNorm: x / sqrt(mean(x^2) + eps) * (1 + w) over the last axis."""
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * (1.0 + w)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def l2norm(x, eps=1e-6):
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


class Model:
    def __init__(self, dir_):
        with open(os.path.join(dir_, "config.json")) as f:
            self.c = json.load(f)
        self.w = load(dir_)
        c = self.c
        n = c["num_hidden_layers"]
        self.types = c.get("layer_types") or [
            "full_attention" if (i + 1) % c.get("full_attention_interval", 4) == 0 else "linear_attention" for i in range(n)]
        self.eps = c.get("rms_norm_eps", 1e-6)

    # --------------------------------------------------------------------------------------------- full attention
    def attention(self, p, h):
        c, w = self.c, self.w
        T = h.shape[0]
        heads, kvh, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        qg = (h @ w[p + "q_proj.weight"].T).reshape(T, heads, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:].reshape(T, heads * hd)
        q = rms1p(q, w[p + "q_norm.weight"], self.eps)
        k = rms1p((h @ w[p + "k_proj.weight"].T).reshape(T, kvh, hd), w[p + "k_norm.weight"], self.eps)
        v = (h @ w[p + "v_proj.weight"].T).reshape(T, kvh, hd)
        rd = int(hd * c.get("partial_rotary_factor", 0.25))
        inv = 1.0 / (c.get("rope_theta", 10000.0) ** (np.arange(0, rd, 2, dtype=np.float64) / rd))
        freqs = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]
        emb = np.concatenate([freqs, freqs], axis=-1)
        cos, sin = np.cos(emb)[:, None, :], np.sin(emb)[:, None, :]
        q = np.concatenate([q[..., :rd] * cos + rotate_half(q[..., :rd]) * sin, q[..., rd:]], axis=-1)
        k = np.concatenate([k[..., :rd] * cos + rotate_half(k[..., :rd]) * sin, k[..., rd:]], axis=-1)
        rep = heads // kvh
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v, rep, axis=1)
        s = np.einsum("thd,uhd->htu", q, k) * hd ** -0.5
        s = s + np.triu(np.full((T, T), -np.inf), 1)[None]
        s = np.exp(s - s.max(axis=-1, keepdims=True))
        s = s / s.sum(axis=-1, keepdims=True)
        o = np.einsum("htu,uhd->thd", s, v).reshape(T, heads * hd)
        return (o * sigmoid(gate)) @ w[p + "o_proj.weight"].T

    # --------------------------------------------------------------------------------------------- Gated DeltaNet
    def delta_net(self, p, h):
        c, w = self.c, self.w
        T = h.shape[0]
        nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
        dk, dv, K = c["linear_key_head_dim"], c["linear_value_head_dim"], c["linear_conv_kernel_dim"]
        r = nv // nk
        qkvz = (h @ w[p + "in_proj_qkvz.weight"].T).reshape(T, nk, 2 * dk + 2 * r * dv)
        ba = (h @ w[p + "in_proj_ba.weight"].T).reshape(T, nk, 2 * r)
        q, k = qkvz[..., :dk], qkvz[..., dk:2 * dk]
        v = qkvz[..., 2 * dk:2 * dk + r * dv].reshape(T, nv, dv)
        z = qkvz[..., 2 * dk + r * dv:].reshape(T, nv, dv)
        b, a = ba[..., :r].reshape(T, nv), ba[..., r:].reshape(T, nv)
        mixed = np.concatenate([q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1)], axis=-1)  # [T, conv_dim]
        cw = w[p + "conv1d.weight"][:, 0, :]  # [conv_dim, K]
        padded = np.concatenate([np.zeros((K - 1, mixed.shape[1])), mixed], axis=0)
        conv = np.stack([np.sum(padded[t:t + K].T * cw, axis=1) for t in range(T)])
        mixed = silu(conv)
        kw = nk * dk
        q = mixed[:, :kw].reshape(T, nk, dk)
        k = mixed[:, kw:2 * kw].reshape(T, nk, dk)
        v = mixed[:, 2 * kw:].reshape(T, nv, dv)
        beta = sigmoid(b)
        g = -np.exp(w[p + "A_log"]) * softplus(a + w[p + "dt_bias"])
        q = np.repeat(q, r, axis=1)
        k = np.repeat(k, r, axis=1)
        q = l2norm(q) * dk ** -0.5
        k = l2norm(k)
        if os.environ.get("REF_CHUNKED"):
            o = chunk_rule(q, k, v, g, beta, int(os.environ.get("REF_CHUNK", "64")))
        else:
            o = recurrent_rule(q, k, v, g, beta)
        nw = w[p + "norm.weight"]
        o = o / np.sqrt(np.mean(o * o, axis=-1, keepdims=True) + self.eps) * nw * silu(z)
        return o.reshape(T, nv * dv) @ w[p + "out_proj.weight"].T

    # --------------------------------------------------------------------------------------------- MLP
    def mlp(self, p, h):
        w = self.w
        return (silu(h @ w[p + "gate_proj.weight"].T) * (h @ w[p + "up_proj.weight"].T)) @ w[p + "down_proj.weight"].T

    def moe(self, p, h):
        c, w = self.c, self.w
        logits = h @ w[p + "mlp.gate.weight"].T
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        out = np.zeros_like(h)
        topk = c["num_experts_per_tok"]
        for t in range(h.shape[0]):
            idx = np.argsort(-probs[t], kind="stable")[:topk]
            wt = probs[t, idx]
            if c.get("norm_topk_prob", True):
                wt = wt / wt.sum()
            for e, we in zip(idx, wt):
                out[t] += we * self.mlp(p + "mlp.experts.%d." % e, h[t:t + 1])[0]
        shared = self.mlp(p + "mlp.shared_expert.", h) * sigmoid(h @ w[p + "mlp.shared_expert_gate.weight"].T)
        return out + shared

    def is_moe(self, l):
        c = self.c
        return c["num_experts"] > 0 and l not in (c.get("mlp_only_layers") or []) \
            and (l + 1) % c.get("decoder_sparse_step", 1) == 0

    def logits(self, ids):
        c, w = self.c, self.w
        x = w["model.embed_tokens.weight"][ids]
        for l in range(c["num_hidden_layers"]):
            p = "model.layers.%d." % l
            h = rms1p(x, w[p + "input_layernorm.weight"], self.eps)
            if self.types[l] == "full_attention":
                x = x + self.attention(p + "self_attn.", h)
            else:
                x = x + self.delta_net(p + "linear_attn.", h)
            h = rms1p(x, w[p + "post_attention_layernorm.weight"], self.eps)
            x = x + (self.moe(p, h) if self.is_moe(l) else self.mlp(p + "mlp.", h))
        x = rms1p(x, w["model.norm.weight"], self.eps)
        head = w.get("lm_head.weight", w["model.embed_tokens.weight"])
        return x @ head.T


def recurrent_rule(q, k, v, g, beta):
    """torch_recurrent_gated_delta_rule for one sequence: q, k [T, H, dk], v [T, H, dv], g / beta [T, H]."""
    T, H, dk = k.shape
    dv = v.shape[-1]
    S = np.zeros((H, dk, dv))
    out = np.zeros((T, H, dv))
    for t in range(T):
        S = S * np.exp(g[t])[:, None, None]
        kv_mem = np.einsum("hkv,hk->hv", S, k[t])
        delta = (v[t] - kv_mem) * beta[t][:, None]
        S = S + np.einsum("hk,hv->hkv", k[t], delta)
        out[t] = np.einsum("hkv,hk->hv", S, q[t])
    return out


def chunk_rule(q, k, v, g, beta, C):
    """torch_chunk_gated_delta_rule for one sequence (q already scaled): per chunk the UT system
    (I + tril(diag(beta) K K^T * decay, -1)) X = [beta v | beta K * exp(cum)] solved exactly, then the scan over chunks."""
    T, H, dk = k.shape
    dv = v.shape[-1]
    pad = (C - T % C) % C
    zq = lambda x: np.concatenate([x, np.zeros((pad,) + x.shape[1:])], axis=0)
    q, k, v, g, beta = zq(q), zq(k), zq(v), zq(g), zq(beta)
    S = np.zeros((H, dk, dv))
    out = np.zeros((T + pad, H, dv))
    for c0 in range(0, T + pad, C):
        for hh in range(H):
            qc, kc, vc = q[c0:c0 + C, hh], k[c0:c0 + C, hh], v[c0:c0 + C, hh]
            bc, cum = beta[c0:c0 + C, hh], np.cumsum(g[c0:c0 + C, hh])
            decay = np.exp(np.tril(cum[:, None] - cum[None, :]))
            decay = np.tril(decay)  # pairwise decay, zero above the diagonal
            kb = kc * bc[:, None]
            A = np.tril((kb @ kc.T) * decay, -1)
            M = np.eye(C) + A
            new_v = np.linalg.solve(M, vc * bc[:, None])
            k_cum = np.linalg.solve(M, kb * np.exp(cum)[:, None])
            v_new = new_v - k_cum @ S[hh]
            attn = np.tril((qc @ kc.T) * decay)
            out[c0:c0 + C, hh] = (qc * np.exp(cum)[:, None]) @ S[hh] + attn @ v_new
            S[hh] = S[hh] * np.exp(cum[-1]) + (kc * np.exp(cum[-1] - cum)[:, None]).T @ v_new
    return out[:T]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    m = Model(sys.argv[1])
    ids = list(PROMPT)
    last = m.logits(np.array(ids))[-1]
    greedy = []
    for _ in range(NEW_TOKENS):
        nxt = int(np.argmax(m.logits(np.array(ids))[-1])) if greedy else int(np.argmax(last))
        greedy.append(nxt)
        ids.append(nxt)
    with open(os.path.join(sys.argv[1], "ref-logits.json"), "w") as f:
        json.dump({"logits": [float(v) for v in last], "greedy": greedy,
                   "chunked": bool(os.environ.get("REF_CHUNKED")), "q8": bool(os.environ.get("REF_Q8"))}, f)
    print("%s: argmax %d, max|logit| %.4f, greedy %s" % (sys.argv[1], int(np.argmax(last)), float(np.abs(last).max()), greedy))


if __name__ == "__main__":
    main()
