#!/bin/bash
# Validates the Metal features on a real Mac: first correctness on the GPU (every family, CPU == Metal), then speed:
# each newer feature on (the default) against off, on a dense model and optionally on a mixture-of-experts model.
# Prints a markdown table; paste it into an issue or docs/11.
#
#   scripts/mac-validate.sh                                  Qwen/Qwen3-0.6B, ~15 minutes
#   scripts/mac-validate.sh --moe Qwen/Qwen3-30B-A3B         plus the MoE rows (needs ~20 GB free memory)
#   scripts/mac-validate.sh --skip-smoke                     speed only
#
# Keep the Mac plugged in and idle; the table notes the thermal state, since a hot GPU runs slower.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$(uname)" = "Darwin" ] || { echo "run this on a Mac (Apple Silicon): it measures the Metal backend" >&2; exit 2; }
DENSE=Qwen/Qwen3-0.6B
MOE=""
SMOKE=1
ROUNDS=3
while [ $# -gt 0 ]; do
  case "$1" in
    --model) DENSE="$2"; shift 2 ;;
    --moe) MOE="$2"; shift 2 ;;
    --skip-smoke) SMOKE=""; shift ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
OUT=$(mktemp -d)

if [ -n "$SMOKE" ]; then
  echo "== correctness on the GPU (scripts/arch-smoke.sh)"
  scripts/arch-smoke.sh > "$OUT/smoke.log" 2>&1 || { tail -30 "$OUT/smoke.log"; echo "arch smoke FAILED (log: $OUT/smoke.log)"; exit 1; }
  echo "   $(grep -c 'cpu == metal' "$OUT/smoke.log") CPU == Metal checks, $(grep -c 'experts on cpu.*ok' "$OUT/smoke.log") expert-split checks: passed"
fi

# median of the rounds' "prefill … (N tok/s), decode … (M tok/s)" lines
bench() { # label model java-opts settings...
  local label="$1" model="$2" opts="$3"
  shift 3
  JAVA_OPTS="$opts" ./jllm bench --model "$model" --temp 0 --bench.rounds "$ROUNDS" "$@" > "$OUT/b.log" 2>&1 || {
    echo "| $label | failed | | $(tail -1 "$OUT/b.log") |"; return; }
  local pre dec thermal
  pre=$(grep -o '([0-9.]* tok/s), decode' "$OUT/b.log" | grep -o '[0-9.]*' | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
  dec=$(grep -o 'decode [0-9]* tok in [0-9]* ms ([0-9.]* tok/s)' "$OUT/b.log" | grep -o '([0-9.]*' | tr -d '(' | sort -n \
    | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
  thermal=$(grep -c 'thermal state' "$OUT/b.log" || true)
  echo "| $label | $pre | $dec | $( [ "$thermal" != 0 ] && echo 'thermal warning' ) |"
}

echo
echo "machine: $(sysctl -n machdep.cpu.brand_string), $(($(sysctl -n hw.memsize) / 1073741824)) GB, $(sw_vers -productVersion)"
echo
echo "| configuration | prefill tok/s | decode tok/s | note |"
echo "|---|---|---|---|"
bench "$DENSE: defaults" "$DENSE" ""
bench "flash prefill off" "$DENSE" "-Djllm.metal.flashPrefill=false"
bench "flash decode off" "$DENSE" "-Djllm.metal.flashDecode=false"
bench "pipelined decoding off" "$DENSE" "-Djllm.metal.pipeline=false"
bench "decode replay off" "$DENSE" "-Djllm.metal.replay=false"
bench "decode replay off, pipelined decoding off" "$DENSE" "-Djllm.metal.replay=false -Djllm.metal.pipeline=false"
bench "q8 KV cache, decode fusions off" "$DENSE" "-Djllm.metal.fuse=false" --backend.kv-dtype q8
bench "simdgroup-matrix prefill off" "$DENSE" "-Djllm.metal.mmaPrefill=false"
bench "f16 KV cache" "$DENSE" "" --backend.kv-dtype f16
bench "q8 KV cache" "$DENSE" "" --backend.kv-dtype q8
if [ -n "$MOE" ]; then
  bench "$MOE: defaults (256-token chunks)" "$MOE" ""
  bench "64-token chunks" "$MOE" "" --batch 64
  bench "grouped MoE prefill off" "$MOE" "-Djllm.metal.moeMma=false"
  bench "decode replay off" "$MOE" "-Djllm.metal.replay=false"
  bench "experts of 4 layers on the CPU" "$MOE" "" --backend.metal.cpu-moe 4
  bench "experts of 4 layers on the CPU, int8 off" "$MOE" "" --backend.metal.cpu-moe 4 --backend.cpu.int8-activations false
  bench "all experts on the GPU" "$MOE" "" --backend.metal.cpu-moe off
fi
echo
echo "(median of $ROUNDS rounds; logs in $OUT)"
