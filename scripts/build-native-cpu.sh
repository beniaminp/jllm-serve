#!/bin/sh
# Builds target/native/libjllm_cpu.so (Linux) or .dylib (macOS): the CPU backend's int8 dot products in C
# (src/main/native/jllm_cpu.c), called through the Foreign Function & Memory API. One binary covers every SIMD level
# of its architecture (x86-64: AVX2, AVX-VNNI, AVX512-VNNI; AArch64: NEON, dotprod): the SIMD functions carry their
# own target attributes and the library picks one at run time. Without it the engine uses its Java kernels.
#
#   scripts/build-native-cpu.sh                 native build with cc (or $CC: gcc or clang)
#   scripts/build-native-cpu.sh --optional      exit 0 with a note when there is no C compiler (the Maven test phase)
#   CC="clang --target=aarch64-linux-gnu" OUT=... scripts/build-native-cpu.sh    cross-compile (needs a sysroot)
#
# No -march / -ffast-math: the baseline code must run on any CPU of the architecture, and the activation quantizer
# must round exactly like the Java one (-ffp-contract=off: no fused multiply-adds the source does not ask for).
set -eu
cd "$(dirname "$0")/.."
optional=false
[ "${1:-}" = "--optional" ] && optional=true
CC="${CC:-cc}"
if ! command -v "${CC%% *}" >/dev/null 2>&1; then
  if $optional; then
    echo "build-native-cpu: no C compiler ($CC), libjllm_cpu not built (the Java kernels are used)"
    exit 0
  fi
  echo "build-native-cpu: no C compiler ($CC); install gcc or clang" >&2
  exit 1
fi
EXT=so
FLAGS="-shared -fPIC"
if [ "$(uname)" = "Darwin" ] && [ -z "${OUT:-}" ]; then
  EXT=dylib
  FLAGS="-dynamiclib"
fi
OUT="${OUT:-target/native/libjllm_cpu.$EXT}"
mkdir -p "$(dirname "$OUT")"
# shellcheck disable=SC2086
if ! $CC -std=c11 -O2 -fvisibility=hidden -ffp-contract=off -fno-fast-math -Wall -Wextra -Wno-unused-function \
    $FLAGS -o "$OUT" src/main/native/jllm_cpu.c -lm; then
  rm -f "$OUT"
  if $optional; then
    echo "build-native-cpu: compiling failed, libjllm_cpu not built (the Java kernels are used)"
    exit 0
  fi
  exit 1
fi
echo "built $OUT"
