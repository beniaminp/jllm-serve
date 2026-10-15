#!/usr/bin/env python3
"""Independent pure-Python (float64, no dependencies) reference of the DeepSeek-V2/V3 forward pass, written
line by line after modeling_deepseek.py (DeepseekV2Attention, MoEGate, DeepseekV2MoE, the YaRN rotary embedding
and apply_rotary_pos_emb with its explicit "view(d/2, 2).transpose" de-interleave + rotate_half).

usage: deepseek-ref.py <tiny-model-dir>   (a directory written by make-tiny-model.py deepseek_*)
Runs the 12-token prompt of ArchSmoke and writes <dir>/ref-logits.json with the logits of the last token.
Used by DeepSeekRef.java to check the engine against this implementation. With REF_Q8=1 in the environment every
matrix is first rounded to Q8_0 as the engine's converter does, which makes the comparison exact up to float
rounding (without it the engine differs by the 1-6 % quantization noise).
"""
import json
import math
import os
import struct
import sys

PROMPT = [1, 7, 42, 99, 13, 5, 250, 8, 77, 31, 600, 9]


def load(dir_):
    with open(os.path.join(dir_, "model.safetensors"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in header.items():
            a, b = info["data_offsets"]
            f.seek(base + a)
            cnt = (b - a) // 4
            flat = list(struct.unpack("<%df" % cnt, f.read(b - a)))
            shape = info["shape"]
            if len(shape) == 2:
                rows = [flat[i * shape[1]:(i + 1) * shape[1]] for i in range(shape[0])]
                if os.environ.get("REF_Q8") and not name.endswith("mlp.gate.weight"):
                    rows = [q8_row(r) for r in rows]  # what the engine's Q8_0 converter does to every matrix
                out[name] = rows
            else:
                out[name] = flat
    return out


def q8_row(r):
    """Q8_0 round trip: blocks of 32 along the row, scale = absmax/127, values rounded to int8."""
    out = []
    for b in range(0, len(r), 32):
        blk = r[b:b + 32]
        amax = max(abs(v) for v in blk)
        d = amax / 127.0
        if d == 0:
            out += [0.0] * len(blk)
        else:
            out += [round(v / d) * d for v in blk]
    return out


def linear(w, x):
    return [sum(a * b for a, b in zip(row, x)) for row in w]


def rms(x, w, eps):
    s = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v * s * g for v, g in zip(x, w)]


def silu(v):
    return v / (1.0 + math.exp(-v))


def mlp(w, p, x):
    g = linear(w[p + "gate_proj.weight"], x)
    u = linear(w[p + "up_proj.weight"], x)
    return linear(w[p + "down_proj.weight"], [silu(a) * b for a, b in zip(g, u)])


def softmax(v):
    m = max(v)
    e = [math.exp(a - m) for a in v]
    s = sum(e)
    return [a / s for a in e]


# ---- YaRN (yarn_* functions of modeling_deepseek.py)
def yarn_get_mscale(scale=1.0, mscale=1.0):
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def correction_dim(rot, dim, base, maxpos):
    return (dim * math.log(maxpos / (rot * 2 * math.pi))) / (2 * math.log(base))


def inv_freq_and_mscale(cfg):
    dim, base = cfg["qk_rope_head_dim"], cfg["rope_theta"]
    rs = cfg.get("rope_scaling")
    extra = [1.0 / (base ** (i / dim)) for i in range(0, dim, 2)]
    if not rs:
        return extra, 1.0
    f = rs["factor"]
    inter = [1.0 / (f * base ** (i / dim)) for i in range(0, dim, 2)]
    low = max(math.floor(correction_dim(rs["beta_fast"], dim, base, rs["original_max_position_embeddings"])), 0)
    high = min(math.ceil(correction_dim(rs["beta_slow"], dim, base, rs["original_max_position_embeddings"])), dim - 1)
    if low == high:
        high += 0.001
    inv = []
    for i in range(dim // 2):
        ramp = min(max((i - low) / (high - low), 0), 1)
        mask = 1.0 - ramp
        inv.append(inter[i] * (1 - mask) + extra[i] * mask)
    return inv, yarn_get_mscale(f, rs.get("mscale", 1)) / yarn_get_mscale(f, rs.get("mscale_all_dim", 0))


def rotate_half(x):
    h = len(x) // 2
    return [-v for v in x[h:]] + x[:h]


def apply_rope(x, pos, inv, mscale):
    """apply_rotary_pos_emb for one head vector: de-interleave (d/2, 2) -> transpose, then x*cos + rotate_half(x)*sin."""
    d = len(x)
    xt = [x[2 * i] for i in range(d // 2)] + [x[2 * i + 1] for i in range(d // 2)]
    freqs = [pos * f for f in inv]
    cos = [math.cos(a) * mscale for a in freqs] * 2
    sin = [math.sin(a) * mscale for a in freqs] * 2
    rh = rotate_half(xt)
    return [a * c + b * s for a, c, b, s in zip(xt, cos, rh, sin)]


def gate(cfg, w, p, x):
    """MoEGate.forward for one token: returns (topk_idx, topk_weight)."""
    scores_raw = linear(w[p + "gate.weight"], x)
    sig = cfg["scoring_func"] == "sigmoid"
    scores = [1.0 / (1.0 + math.exp(-v)) for v in scores_raw] if sig else softmax(scores_raw)
    e, k, ng, tg = cfg["n_routed_experts"], cfg["num_experts_per_tok"], cfg["n_group"], cfg["topk_group"]
    method = cfg["topk_method"]
    if method == "greedy":
        idx = sorted(range(e), key=lambda i: (-scores[i], i))[:k]
    else:
        choice = scores
        if method == "noaux_tc":
            bias = w[p + "gate.e_score_correction_bias"]
            choice = [s + b for s, b in zip(scores, bias)]
        gs = e // ng
        if method == "noaux_tc":
            gsc = [sum(sorted(choice[g * gs:(g + 1) * gs], reverse=True)[:2]) for g in range(ng)]
        else:
            gsc = [max(choice[g * gs:(g + 1) * gs]) for g in range(ng)]
        groups = sorted(range(ng), key=lambda g: (-gsc[g], g))[:tg]
        masked = [choice[i] if (i // gs) in groups else -math.inf for i in range(e)]
        idx = sorted(range(e), key=lambda i: (-masked[i], i))[:k]
    weight = [scores[i] for i in idx]
    renorm = k > 1 and cfg["norm_topk_prob"]
    if renorm:
        den = sum(weight) + 1e-20
        weight = [v / den for v in weight]
    if cfg["model_type"] == "deepseek_v3" or not renorm:  # V3 always scales; V2 only in the else branch
        weight = [v * cfg["routed_scaling_factor"] for v in weight]
    return idx, weight


def main():
    dir_ = sys.argv[1]
    cfg = json.load(open(os.path.join(dir_, "config.json")))
    w = load(dir_)
    eps = cfg["rms_norm_eps"]
    heads = cfg["num_attention_heads"]
    nope, rope_d, vd = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
    qkd = nope + rope_d
    kvl = cfg["kv_lora_rank"]
    inv, mscale = inv_freq_and_mscale(cfg)
    softmax_scale = qkd ** -0.5
    rs = cfg.get("rope_scaling")
    if rs and rs.get("mscale_all_dim"):
        m = yarn_get_mscale(rs["factor"], rs["mscale_all_dim"])
        softmax_scale *= m * m

    hs = [list(w["model.embed_tokens.weight"][t]) for t in PROMPT]
    for l in range(cfg["num_hidden_layers"]):
        p = "model.layers.%d." % l
        a = p + "self_attn."
        keys, values = [], []
        new = []
        normed = [rms(h, w[p + "input_layernorm.weight"], eps) for h in hs]
        outs = []
        for pos, xn in enumerate(normed):
            if cfg["q_lora_rank"] is None:
                q = linear(w[a + "q_proj.weight"], xn)
            else:
                q = linear(w[a + "q_b_proj.weight"], rms(linear(w[a + "q_a_proj.weight"], xn), w[a + "q_a_layernorm.weight"], eps))
            ckv = linear(w[a + "kv_a_proj_with_mqa.weight"], xn)
            compressed, k_pe = ckv[:kvl], ckv[kvl:]
            kv = linear(w[a + "kv_b_proj.weight"], rms(compressed, w[a + "kv_a_layernorm.weight"], eps))
            k_pe = apply_rope(k_pe, pos, inv, mscale)
            qh, kh, vh = [], [], []
            for h in range(heads):
                qq = q[h * qkd:(h + 1) * qkd]
                qh.append(qq[:nope] + apply_rope(qq[nope:], pos, inv, mscale))
                kk = kv[h * (nope + vd):(h + 1) * (nope + vd)]
                kh.append(kk[:nope] + k_pe)
                vh.append(kk[nope:])
            keys.append(kh)
            values.append(vh)
            o = []
            for h in range(heads):
                sc = [sum(x * y for x, y in zip(qh[h], keys[s][h])) * softmax_scale for s in range(pos + 1)]
                pr = softmax(sc)
                o += [sum(pr[s] * values[s][h][d] for s in range(pos + 1)) for d in range(vd)]
            outs.append(linear(w[a + "o_proj.weight"], o))
        hs = [[x + y for x, y in zip(h, o)] for h, o in zip(hs, outs)]
        for pos, h in enumerate(hs):
            xn = rms(h, w[p + "post_attention_layernorm.weight"], eps)
            moe = l >= cfg["first_k_dense_replace"] and l % cfg["moe_layer_freq"] == 0
            if not moe:
                y = mlp(w, p + "mlp.", xn)
            else:
                idx, wt = gate(cfg, w, p + "mlp.", xn)
                y = [0.0] * len(xn)
                for i, c in zip(idx, wt):
                    ey = mlp(w, p + "mlp.experts.%d." % i, xn)
                    y = [u + c * v for u, v in zip(y, ey)]
                if cfg["n_shared_experts"]:
                    sy = mlp(w, p + "mlp.shared_experts.", xn)
                    y = [u + v for u, v in zip(y, sy)]
            hs[pos] = [u + v for u, v in zip(h, y)]
    last = rms(hs[-1], w["model.norm.weight"], eps)
    logits = linear(w["lm_head.weight"], last)
    json.dump({"logits": logits}, open(os.path.join(dir_, "ref-logits.json"), "w"))
    print("wrote ref-logits.json; argmax", max(range(len(logits)), key=lambda i: logits[i]))


main()
