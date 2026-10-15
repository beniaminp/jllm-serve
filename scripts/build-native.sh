#!/bin/sh
# Builds libjllm_metal.dylib (macOS only). Requires Xcode command line tools.
set -eu
cd "$(dirname "$0")/.."
mkdir -p target/native
clang -O2 -fobjc-arc -dynamiclib \
  -framework Metal -framework Foundation \
  src/main/native/jllm_metal.m \
  -o target/native/libjllm_metal.dylib
echo "built target/native/libjllm_metal.dylib"
