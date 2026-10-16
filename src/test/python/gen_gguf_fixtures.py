#!/usr/bin/env python3
"""Generates src/test/resources/gguf/<type>.blocks (raw ggml blocks) and <type>.ref (their f32 dequantization by
the reference implementation gguf.quants) for the byte-parity checks of GgufKernelsCheck.

    python3 -m venv v && v/bin/pip install gguf numpy
    v/bin/python src/test/python/gen_gguf_fixtures.py src/test/resources/gguf

Rows are random bytes with the f16 scale fields replaced by small valid values (random bytes would often decode
to NaN / inf scales), so every quant code and every packed scale / min bit pattern occurs.
"""
import sys

import numpy as np
import gguf
from gguf import GGMLQuantizationType as T, quants

ROWS, COLS = 4, 512
# type -> byte offsets of the f16 scale fields inside a block
SCALES = {
    T.Q4_0: [0], T.Q8_0: [0], T.Q4_1: [0, 2], T.Q5_0: [0], T.Q5_1: [0, 2],
    T.Q2_K: [80, 82], T.Q3_K: [108], T.Q4_K: [0, 2], T.Q5_K: [0, 2], T.Q6_K: [208],
}


def main(out):
    rng = np.random.default_rng(12345)
    for t, offs in SCALES.items():
        block, size = gguf.GGML_QUANT_SIZES[t]
        nb = COLS // block
        data = rng.integers(0, 256, size=(ROWS, nb, size), dtype=np.uint8)
        for o in offs:
            f = rng.uniform(0.002, 0.05, size=(ROWS, nb)).astype(np.float16)
            data[:, :, o:o + 2] = f.view(np.uint8).reshape(ROWS, nb, 2)
        data = data.reshape(ROWS, nb * size)
        ref = quants.dequantize(data, t).astype(np.float32)
        assert ref.shape == (ROWS, COLS), ref.shape
        assert np.isfinite(ref).all()
        data.tofile(f"{out}/{t.name}.blocks")
        ref.astype("<f4").tofile(f"{out}/{t.name}.ref")
        print(t.name, data.shape, float(np.abs(ref).max()))


if __name__ == "__main__":
    main(sys.argv[1])
