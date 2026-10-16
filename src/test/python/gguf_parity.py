#!/usr/bin/env python3
"""Dequantizes real tensors of a GGUF file with the reference implementation (gguf.quants) for GgufRealParity.

    v/bin/python src/test/python/gguf_parity.py model.gguf OUTDIR [ROWS] [LAYERS]

For the embedding and the tensors of the blocks listed in LAYERS (comma separated, default 0,13,27) it writes
OUTDIR/<tensor name>.f32: the first ROWS (default 128) rows as little-endian float32, together with
OUTDIR/<tensor name>.type holding the ggml type. Java then dequantizes the same rows of the same file and compares.
"""
import sys
import numpy as np
from gguf import GGUFReader
from gguf import quants


def main():
    path, out = sys.argv[1], sys.argv[2]
    rows = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    layers = sys.argv[4].split(",") if len(sys.argv) > 4 else ["0", "13", "27"]
    wanted = tuple(f"blk.{l}." for l in layers)
    r = GGUFReader(path)
    n = 0
    for t in r.tensors:
        if not (t.name == "token_embd.weight" or t.name.startswith(wanted)):
            continue
        if len(t.shape) != 2:
            continue
        k = min(rows, int(t.shape[1]))
        data = np.asarray(t.data)[:k]  # (k, bytes per row) for quantized types, (k, cols) for floats
        if t.tensor_type.name in ("F32", "F16", "BF16"):
            ref = data.astype(np.float32)
        else:
            ref = quants.dequantize(data, t.tensor_type).astype(np.float32)
        ref.astype("<f4").tofile(f"{out}/{t.name}.f32")
        open(f"{out}/{t.name}.type", "w").write(t.tensor_type.name)
        n += 1
    print(n, "tensors written")


if __name__ == "__main__":
    main()
