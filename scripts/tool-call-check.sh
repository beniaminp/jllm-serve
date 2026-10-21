#!/bin/bash
# Constrained tool calls x speculation on the real Qwen3-0.6B (docs/17 section 17.7): the greedy output of tool-call
# requests under the tool-call constraint (tool_choice auto and required, jump-forward on and off) must be identical
# with speculation off, with prompt lookup, and with a draft model (the model drafting for itself, which accepts every
# draft, and on the CPU a random tiny model with the same tokenizer, which is wrong nearly every time). Exits 1 on any
# difference.
#
#   scripts/tool-call-check.sh                 CPU (deterministic kernels, native int8 kernels if built) and Metal
#   scripts/tool-call-check.sh cpu             only the CPU
#   scripts/tool-call-check.sh metal           only the GPU (a Mac)
#
# The models must be in ./models (Qwen/Qwen3-0.6B and hf-internal-testing/tiny-random-Qwen3MoeForCausalLM); nothing
# is downloaded if they are there.
# Extra flags are passed to jllm (e.g. --quant q8).
# Run it on an idle machine: the CPU rows take a few minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
JAVA="${JAVA_HOME:+$JAVA_HOME/bin/}java"
which=${1:-all}
[ $# -gt 0 ] && shift
mvn -q -o test-compile
OPTS=(--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx4g)
[ -f target/native/libjllm_cpu.dylib ] && OPTS+=(-Djllm.cpu.nativeLib=target/native/libjllm_cpu.dylib)
[ -f target/native/libjllm_cpu.so ] && OPTS+=(-Djllm.cpu.nativeLib=target/native/libjllm_cpu.so)
[ -f target/native/libjllm_metal.dylib ] && OPTS+=(-Djllm.metal.lib=target/native/libjllm_metal.dylib)
# draft models: the model itself (every draft is right unless the constraint disagrees with it) and, on the CPU, a
# random tiny model with the same tokenizer (wrong nearly every time; the Metal backend needs a head size of a
# multiple of 32, which the tiny model lacks)
DRAFTS_CPU=Qwen/Qwen3-0.6B,hf-internal-testing/tiny-random-Qwen3MoeForCausalLM
DRAFTS_METAL=Qwen/Qwen3-0.6B
# the draft length follows the doubling rule, so that drafts are tried in every pass: the cost model would stop
# drafting when it measures speculation as not paying, and then nothing would be checked
NOADAPT=-Djllm.engine.adaptiveDraft=false
check() { # backend drafts jvm-flags jllm-flags
  local backend=$1 drafts=$2 jvm=$3 flags=$4
  echo "== constrained greedy output with and without speculation: $backend"
  # shellcheck disable=SC2086
  "$JAVA" "${OPTS[@]}" $jvm -cp target/classes:target/test-classes org.robomiri.jllm.ToolCallCheck \
    --backend "$backend" --ctx 1024 --check-draft "$drafts" $flags ${EXTRA[@]+"${EXTRA[@]}"}
}
EXTRA=("$@")
if [ "$which" = all ] || [ "$which" = cpu ]; then
  # deterministic kernels: a multi-token pass (verification, jump-forward) is bit-identical to single-token steps;
  # int8 activations fixed, since "auto" times both ways at every load and may choose differently in two loads
  check cpu "$DRAFTS_CPU" "-Djllm.cpu.deterministic=true $NOADAPT" "--backend.cpu.int8-activations true"
fi
if [ "$which" = all ] || [ "$which" = metal ]; then
  [ "$(uname)" = "Darwin" ] || { echo "metal: needs a Mac" >&2; exit 2; }
  check metal "$DRAFTS_METAL" "$NOADAPT" ""
fi
