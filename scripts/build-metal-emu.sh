#!/bin/sh
# Builds target/native/libjllm_metal_emu.so: the Metal kernels (kernels.metal, transpiled to C++) and a CPU runtime
# behind the same C API as libjllm_metal.dylib, so the Metal backend can be run and tested without an Apple GPU:
#   java ... -Djllm.metal.emulator=true -Djllm.metal.lib=target/native/libjllm_metal_emu.so
# It is slow (every GPU thread is a fiber): meant for the tiny test models, not for real ones. Needs clang++ and python3.
set -eu
cd "$(dirname "$0")/.."
OUT=target/native
mkdir -p "$OUT/emu"
python3 scripts/metal-emu/transpile.py src/main/resources/org/robomiri/jllm/metal/kernels.metal "$OUT/emu/kernels.cpp"
EXT=so
[ "$(uname)" = "Darwin" ] && EXT=dylib
clang++ -std=c++17 -O2 -g -fPIC -shared -Wno-unknown-attributes -Iscripts/metal-emu \
  -o "$OUT/libjllm_metal_emu.$EXT" "$OUT/emu/kernels.cpp" scripts/metal-emu/runtime.cpp
echo "$OUT/libjllm_metal_emu.$EXT"
