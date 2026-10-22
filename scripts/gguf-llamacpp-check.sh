#!/usr/bin/env bash
# GGUF files of every model family against llama.cpp itself.
#
# For each tiny random checkpoint of scripts/make-tiny-model.py (the variants llama.cpp's converter can express) this
# writes the GGUF the way convert_hf_to_gguf.py writes it (src/test/java/org/robomiri/jllm/HfToGguf.java: tensor names,
# dimension order, the llama q/k permutation, Gemma's +1 norms, Mamba's -exp(A_log), DeepSeek's split kv_b, rope_freqs /
# rope_factors_* ...), then
#   1. loads it with jllm and with the safetensors path and compares the logits (CPU, f32 cache), and
#   2. runs `llama-debug --save-logits` (llama.cpp, CPU, f32 cache) on the file holding the Q8_0-rounded weights as F32
#      and compares its logits for the last prompt token with jllm's on the same token ids.
# The second part proves that the writer, and so jllm's reading of the layout, agrees with llama.cpp's reader. Every
# family must agree to within float rounding (the printed "llama.cpp vs jllm" line, bound 2e-3 of the largest logit).
#
# usage: scripts/gguf-llamacpp-check.sh [name...]       names as in GgufFamiliesCheck.MODELS (default: all)
#        LLAMA_DEBUG=/path/to/llama-debug scripts/gguf-llamacpp-check.sh   (default: llama-debug on PATH; tested with
#        Homebrew llama.cpp 0.7.0 = build b11669, whose sources the layouts were taken from)
# Needs JDK 26 (JAVA_HOME or /opt/homebrew/opt/openjdk@26), python3 (standard library only) and llama-debug; without
# llama-debug it runs the jllm-only part.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x /opt/homebrew/opt/openjdk@26/bin/java ] && export JAVA_HOME=/opt/homebrew/opt/openjdk@26
export JAVA_HOME="${JAVA_HOME:?set JAVA_HOME to a JDK 26}"
mvn -q -o compile test-compile
exec "$JAVA_HOME/bin/java" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx3g \
  -cp target/classes:target/test-classes org.robomiri.jllm.GgufFamiliesCheck --llama-cpp "$@"
