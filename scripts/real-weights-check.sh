#!/usr/bin/env bash
# Real-weights check: the engine (CPU backend) against Hugging Face transformers on a published checkpoint.
#
# usage: scripts/real-weights-check.sh <hf-repo | model-dir> [quant]     (quant: q8 by default; q4, q2k, q6k)
#
#   1. downloads the repository (config, tokenizer, safetensors) with huggingface_hub into $REAL_WORK/models/<repo>,
#      or, for a local model directory, links its files there (the engine writes its quantized cache next to the
#      checkpoint, so a local directory is never modified);
#   2. scripts/real_ref.py runs transformers in float32: the prompts chat-templated with apply_chat_template and as
#      raw text, 16 greedy steps each, top-20 and full logits of every step, plus tokenizer probes; then the same
#      steps again with every weight the engine quantizes rounded to Q8_0 (the engine's exact weights at q8);
#   3. RealWeightsCheck converts the checkpoint to <quant>, runs the same prompt ids on the CPU backend with float
#      activations (backend.cpu.int8-activations off, -Djllm.cpu.fastmath=false) and prints, per prompt, whether
#      the engine's tokenizer and chat template give the reference's ids / text, the greedy ids matched before the
#      first divergence with the reference's logit margin there, the teacher-forced argmax matches, the top-20
#      overlap and the Pearson correlation of the logits, and the correlation / largest error against the
#      Q8-rounded reference (at q8 only float rounding should remain there); it writes result-<quant>.json and
#      report-<quant>.txt next to ref.json (docs/14-model-families.md, 14.13).
#
# Environment:
#   REAL_WORK   working directory (default /tmp/jllm-real): models/ and out/<repo>/ (ref.json, result-*.json)
#   REAL_PYTHON a python with torch + transformers + huggingface_hub (default: python3). For example:
#                 python3 -m venv /tmp/realref-venv
#                 /tmp/realref-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
#                 /tmp/realref-venv/bin/pip install transformers safetensors accelerate huggingface_hub
#   REF_DTYPE   upcast: keep the bf16 weights and upcast each matrix when used (float32 activations, half the memory)
#   LOW_DISK=1  delete each safetensors shard once the engine has converted it (after the reference ran)
#   KEEP=1      keep the downloaded checkpoint and the quantized cache (default: delete them, keep the JSON)
#   HF_TOKEN    for gated repositories
#   MVN         the Maven command (default mvn)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x /opt/homebrew/opt/openjdk@26/bin/java ] && [ -z "${JAVA_HOME:-}" ] && export JAVA_HOME=/opt/homebrew/opt/openjdk@26
export JAVA_HOME="${JAVA_HOME:?set JAVA_HOME to a JDK 26}"
SRC="${1:?usage: real-weights-check.sh <hf-repo | model-dir> [quant]}"
QUANT="${2:-q8}"
WORK="${REAL_WORK:-/tmp/jllm-real}"
PY="${REAL_PYTHON:-python3}"

if [ -d "$SRC" ]; then
  NAME="$(basename "$SRC")"
  MODEL="$WORK/models/$NAME"
  mkdir -p "$MODEL"
  for f in "$SRC"/*.json "$SRC"/*.safetensors "$SRC"/*.jinja "$SRC"/tokenizer.model; do
    [ -f "$f" ] || continue
    case "$(basename "$f")" in jllm-*) continue ;; esac
    ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$MODEL/$(basename "$f")"
  done
else
  NAME="${SRC//\//_}"
  MODEL="$WORK/models/$NAME"
  "$PY" -I -c 'import sys; from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "*.jinja"])' "$SRC" "$MODEL"
fi
OUT="$WORK/out/$NAME"
mkdir -p "$OUT"

"$PY" -I scripts/real_ref.py "$MODEL" "$OUT"
${MVN:-mvn} -q compile test-compile
"$JAVA_HOME/bin/java" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx3g \
  -Djllm.cpu.fastmath=false ${LOW_DISK:+-Dreal.lowDisk=true} ${JAVA_OPTS:-} -cp target/classes:target/test-classes \
  org.robomiri.jllm.RealWeightsCheck "$MODEL" "$OUT" "$QUANT" 2>&1 | grep --line-buffered -v '^WARNING' | tee "$OUT/report-$QUANT.txt"

rm -f "$OUT"/*.f32
if [ -z "${KEEP:-}" ]; then
  rm -rf "$MODEL"
fi
echo "results: $OUT"
