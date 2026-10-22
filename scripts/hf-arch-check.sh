#!/usr/bin/env bash
# The long-context families against Hugging Face transformers on tiny random models (docs/14 14.18 - 14.21).
#
# usage: scripts/hf-arch-check.sh [family ...]     (gptoss gptoss-mxfp4 llama4 llama4-mix; default: all four)
#
# For every family scripts/hf-arch-ref.py builds a tiny random model with transformers (the real config's shape: sinks,
# alternating windows, YaRN without truncation, clamped-SwiGLU MoE with biases and MXFP4 experts for gpt-oss; chunked
# attention, NoPE layers with temperature tuning, L2 q/k norm, sigmoid top-1 MoE with a shared expert for Llama 4),
# saves it as a Hugging Face checkpoint and runs transformers on the prompt plus 12 greedy ids with the weights rounded
# to Q8_0; ArchRefCheck then converts the checkpoint to q8 and compares the engine's logits at every position, the
# argmax, the free-running greedy ids and the chunk-size invariance, on the CPU backend and (on a Mac) the Metal one.
#
# Environment: REF_PYTHON  python with torch + transformers (default python3), REF_WORK  where the models go
# (default /tmp/jllm-hf), JAVA_HOME. METAL_EMU=1 runs the Metal side on the CPU emulator of the kernels.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x /opt/homebrew/opt/openjdk@26/bin/java ] && [ -z "${JAVA_HOME:-}" ] && export JAVA_HOME=/opt/homebrew/opt/openjdk@26
export JAVA_HOME="${JAVA_HOME:?set JAVA_HOME to a JDK 26}"
PY="${REF_PYTHON:-python3}"
WORK="${REF_WORK:-/tmp/jllm-hf}"
FAMILIES=("$@")
[ ${#FAMILIES[@]} -gt 0 ] || FAMILIES=(gptoss gptoss-mxfp4 llama4 llama4-mix)

mvn -q compile test-compile
LIB=""
if [ "$(uname)" = Darwin ] || [ -n "${METAL_EMU:-}" ]; then
  if [ -n "${METAL_EMU:-}" ]; then
    LIB="$(./scripts/build-metal-emu.sh 2>/dev/null | tail -1)"
    JAVA_OPTS="${JAVA_OPTS:-} -Djllm.metal.emulator=true"
  else
    ./scripts/build-native.sh >/dev/null
    LIB=target/native/libjllm_metal.dylib
  fi
fi
rc=0
for f in "${FAMILIES[@]}"; do
  d="$WORK/$f"
  rm -rf "$d"
  "$PY" -I scripts/hf-arch-ref.py "$f" "$d" 2>&1 | grep -v 'Warning\|NOTE:\|transformers\]' || true
  "$JAVA_HOME/bin/java" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx4g \
    -Djllm.cpu.fastmath=false ${JAVA_OPTS:-} -cp target/classes:target/test-classes \
    org.robomiri.jllm.ArchRefCheck "$d" "${LIB:--}" q8 2>&1 | grep -v '^WARNING\|^Converting\|^Mapped' || rc=1
done
exit $rc
