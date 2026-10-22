#!/bin/sh
# Builds target/native/libjllm_metal_emu.so: the Metal kernels (kernels.metal, transpiled to C++) and a CPU runtime
# behind the same C API as libjllm_metal.dylib, so the Metal backend can be run and tested without an Apple GPU:
#   java ... -Djllm.metal.emulator=true -Djllm.metal.lib=target/native/libjllm_metal_emu.so
# It is slow (every GPU thread is a fiber): meant for the tiny test models, not for real ones. Needs clang++ and python3.
set -eu
cd "$(dirname "$0")/.."
OUT=target/native
mkdir -p "$OUT/emu"
python3 scripts/metal-emu/transpile.py src/main/resources/org/robomiri/jllm/metal/kernels.metal "$OUT/emu/kernels.cpp" \
  src/main/resources/org/robomiri/jllm/metal/vision.metal src/main/resources/org/robomiri/jllm/audio/audio.metal \
  src/main/resources/org/robomiri/jllm/metal/diffusion.metal
EXT=so
DEFS=""
# macOS: the ucontext fiber calls need _XOPEN_SOURCE, and _DARWIN_C_SOURCE keeps the BSD extensions (MAP_ANON) visible
[ "$(uname)" = "Darwin" ] && { EXT=dylib; DEFS="-D_XOPEN_SOURCE=700 -D_DARWIN_C_SOURCE -Wno-deprecated-declarations"; }
# -fsigned-char: MSL's char is signed (the q8 KV cache), C++'s is unsigned on some ARM targets
# shellcheck disable=SC2086
clang++ -std=c++17 -O2 -g -fPIC -shared -fsigned-char -Wno-unknown-attributes $DEFS -Iscripts/metal-emu \
  -o "$OUT/libjllm_metal_emu.$EXT" "$OUT/emu/kernels.cpp" scripts/metal-emu/runtime.cpp
echo "$OUT/libjllm_metal_emu.$EXT"
