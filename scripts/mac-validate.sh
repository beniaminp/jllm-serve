#!/bin/bash
# Validates the Metal features on a real Mac: first correctness on the GPU (every family, CPU == Metal), then speed:
# each newer feature on (the default) against off, on a dense model and optionally on a mixture-of-experts model;
# draft-model speculation (the dense model drafting for itself, adaptive length and draft pipelining on and off) and
# batched decoding of three slots in shared passes against taking turns.
# Prints a markdown table; paste it into an issue or docs/11.
#
#   scripts/mac-validate.sh                                  Qwen/Qwen3-0.6B, ~25 minutes
#   scripts/mac-validate.sh --moe Qwen/Qwen3-30B-A3B         plus the MoE rows (needs ~20 GB free memory)
#   scripts/mac-validate.sh --skip-smoke                     speed only
#   scripts/mac-validate.sh --vs-llama                       plus jllm vs llama.cpp (llama-bench) on the same GGUF file
#   scripts/mac-validate.sh --vs-llama --llama-moe unsloth/Qwen3-30B-A3B-GGUF   ... and on a MoE GGUF
#   scripts/mac-validate.sh --llama-only --vs-llama          only the llama.cpp comparison
#   scripts/mac-validate.sh --audio speech.wav               plus the audio rows (Whisper small and OuteTTS: CPU, Metal encoder,
#                                                            Metal decoder, half and float weight tiles; q8 and q4 speech model)
#   scripts/mac-validate.sh --gguf-types DIR                 plus every .gguf in DIR (e.g. made with llama-quantize from one
#                                                            model: Q4_1, Q5_0, IQ4_NL, MXFP4, ...) used in place against
#                                                            -Djllm.gguf.requantize=all (requantized to Q8_0 at load)
#
# The llama.cpp comparison needs llama-bench (brew install llama.cpp, or LLAMA_BENCH=/path/to/llama-bench) and
# python3. Both engines read the same .gguf file (default unsloth/Qwen3-0.6B-GGUF, Q4_0; --llama-quant to change),
# with the same prompt length and number of generated tokens; jllm runs without speculative decoding, as llama-bench
# has none, and a second jllm row shows it with its default speculation.
#
# Keep the Mac plugged in and idle; the table notes the thermal state, since a hot GPU runs slower.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$(uname)" = "Darwin" ] || { echo "run this on a Mac (Apple Silicon): it measures the Metal backend" >&2; exit 2; }
DENSE=Qwen/Qwen3-0.6B
MOE=""
SMOKE=1
ROUNDS=3
VS_LLAMA=""
LLAMA_ONLY=""
LLAMA_DENSE=unsloth/Qwen3-0.6B-GGUF
LLAMA_MOE=""
LLAMA_QUANT=Q4_0
AUDIO=""
GGUF_TYPES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) DENSE="$2"; shift 2 ;;
    --moe) MOE="$2"; shift 2 ;;
    --skip-smoke) SMOKE=""; shift ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --vs-llama) VS_LLAMA=1; shift ;;
    --llama-only) LLAMA_ONLY=1; SMOKE=""; shift ;;
    --llama-model) LLAMA_DENSE="$2"; shift 2 ;;
    --llama-moe) LLAMA_MOE="$2"; shift 2 ;;
    --llama-quant) LLAMA_QUANT="$2"; shift 2 ;;
    --audio) AUDIO="$2"; shift 2 ;;
    --gguf-types) GGUF_TYPES="$2"; shift 2 ;;
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

if [ -z "$LLAMA_ONLY" ]; then
echo
echo "| configuration | prefill tok/s | decode tok/s | note |"
echo "|---|---|---|---|"
bench "$DENSE: defaults" "$DENSE" ""
bench "flash prefill off" "$DENSE" "-Djllm.metal.flashPrefill=false"
bench "flash decode off" "$DENSE" "-Djllm.metal.flashDecode=false"
bench "pipelined decoding off" "$DENSE" "-Djllm.metal.pipeline=false"
bench "decode replay off" "$DENSE" "-Djllm.metal.replay=false"
bench "decode replay off, pipelined decoding off" "$DENSE" "-Djllm.metal.replay=false -Djllm.metal.pipeline=false"
bench "q8 KV cache, decode fusions off" "$DENSE" "-Djllm.metal.fuse=false" --backend.kv-dtype q8 --engine.speculative off
bench "simdgroup-matrix prefill off" "$DENSE" "-Djllm.metal.mmaPrefill=false"
# the prefill kernels' two forms: f32 *_mma vs half-weight *_hmma (default: per shape, as the start-up check timed them),
# and attention_fa vs attention_fb; the 2000-token rows weigh attention more
bench "half-weight MMA prefill forced on" "$DENSE" "-Djllm.metal.hmma=true"
bench "half-weight MMA prefill off (f32 MMA)" "$DENSE" "-Djllm.metal.hmma=false"
# the default 64-token chunk underfills the GPU; 256 tokens per chunk shows the matmul kernels' own throughput
bench "half-weight MMA prefill forced on, --batch 256" "$DENSE" "-Djllm.metal.hmma=true" --batch 256
bench "half-weight MMA prefill off, --batch 256" "$DENSE" "-Djllm.metal.hmma=false" --batch 256
bench "GGUF $LLAMA_QUANT: half-weight MMA prefill forced on, --batch 256" "$LLAMA_DENSE" "-Djllm.metal.hmma=true" \
  --gguf-quant "$LLAMA_QUANT" --batch 256
bench "GGUF $LLAMA_QUANT: half-weight MMA prefill off, --batch 256" "$LLAMA_DENSE" "-Djllm.metal.hmma=false" \
  --gguf-quant "$LLAMA_QUANT" --batch 256
bench "flash prefill form fa" "$DENSE" "-Djllm.metal.flashPrefillKernel=fa"
bench "flash prefill form fb" "$DENSE" "-Djllm.metal.flashPrefillKernel=fb"
bench "~2000-token prompt: defaults" "$DENSE" "" --bench.prompt-repeat 200
bench "~2000-token prompt: hmma off, fa" "$DENSE" "-Djllm.metal.hmma=false -Djllm.metal.flashPrefillKernel=fa" --bench.prompt-repeat 200
bench "~2000-token prompt: hmma on, fb" "$DENSE" "-Djllm.metal.hmma=true -Djllm.metal.flashPrefillKernel=fb" --bench.prompt-repeat 200
bench "f16 KV cache" "$DENSE" "" --backend.kv-dtype f16 --engine.speculative off
bench "q8 KV cache" "$DENSE" "" --backend.kv-dtype q8 --engine.speculative off
bench "q4 KV cache" "$DENSE" "" --backend.kv-dtype q4 --engine.speculative off
bench "q4 KV cache, no recent window" "$DENSE" "" --backend.kv-dtype q4 --backend.kv-q4-recent 0 --engine.speculative off
bench "q4 KV cache, recent window 512" "$DENSE" "" --backend.kv-dtype q4 --backend.kv-q4-recent 512 --engine.speculative off
bench "~2000-token prompt: f16 KV cache" "$DENSE" "" --backend.kv-dtype f16 --engine.speculative off --bench.prompt-repeat 200
bench "~2000-token prompt: q4 KV cache" "$DENSE" "" --backend.kv-dtype q4 --engine.speculative off --bench.prompt-repeat 200
# draft-model speculation on the GPU (the draft is the dense model itself: every draft is accepted, so this measures
# what a pass of k drafts costs against k plain steps; a real draft is smaller and accepts less)
bench "speculation off" "$DENSE" "" --engine.speculative off
bench "draft $DENSE for itself, k 4 (adaptive)" "$DENSE" "" --engine.speculative draft --engine.draft-model "$DENSE" \
  --engine.draft-tokens 4
bench "draft for itself, k 4, adaptive length off" "$DENSE" "-Djllm.engine.adaptiveDraft=false" --engine.speculative draft \
  --engine.draft-model "$DENSE" --engine.draft-tokens 4
bench "draft for itself, k 4, draft pipelining off" "$DENSE" "-Djllm.engine.draftPipeline=false" --engine.speculative draft \
  --engine.draft-model "$DENSE" --engine.draft-tokens 4

# batched decoding across slots: three engines (slots) on the weights, generating together in shared passes
# (MetalBackend.forwardBatch) against taking turns; BatchDecodeTest also checks that the answers equal those alone
batch() { # label java-opts
  local label="$1" opts="$2"
  # shellcheck disable=SC2086
  "$JAVA" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx4g $opts \
    -Djllm.metal.lib=target/native/libjllm_metal.dylib -cp target/classes:target/test-classes \
    org.robomiri.jllm.BatchDecodeTest "models/$(echo "$DENSE" | tr / _)" bench --backend metal > "$OUT/batch.log" 2>&1 || {
    echo "| $label | failed | | $(tail -1 "$OUT/batch.log") |"; return; }
  local one three
  one=$(grep -o 'one slot [0-9.]* tok/s' "$OUT/batch.log" | grep -o '[0-9.]*')
  three=$(grep -o 'three slots together [0-9.]* tok/s ([0-9.]*x)' "$OUT/batch.log" | sed 's/three slots together //')
  echo "| $label | | one slot $one, three $three | $(grep -o '[0-9]* steps in shared passes' "$OUT/batch.log") |"
}
JAVA="${JAVA_HOME:+$JAVA_HOME/bin/}java"
mvn -q compile test-compile
[ -f target/native/libjllm_metal.dylib ] || ./scripts/build-native.sh >/dev/null
batch "3 slots generating together (batched)" ""
batch "3 slots taking turns" "-Djllm.metal.batchDecode=false"
# constrained tool calls (docs/17 section 17.7). ToolCallCheck generates a tool call under the tool-call constraint
# with speculation off, with prompt lookup and with the model drafting for itself, jump-forward off and on, and fails
# the run when the tokens differ from speculation off; its decode time is the median of 5 generations.
# ConstraintBench times the token mask (CPU, microseconds per token) with the masks on and off (engine.constraint-masks).
echo
echo "| constrained tool call (Qwen3-0.6B, greedy, 'required') | tokens | drafts accepted | decode ms | tokens vs speculation off |"
echo "|---|---|---|---|---|"
"$JAVA" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx4g \
  -Djllm.metal.lib=target/native/libjllm_metal.dylib -cp target/classes:target/test-classes \
  org.robomiri.jllm.ToolCallCheck --backend metal --ctx 1024 --check-rounds 5 --check-table --check-max 48 \
  --check-draft "$DENSE" > "$OUT/tc.log" 2>&1 \
  && grep '^| .* | weather/required=true/' "$OUT/tc.log" \
    | awk -F'|' '{printf "|%s %s|%s|%s|%s|%s|\n", $2, $3, $4, $5, $6, $7}' \
  || { echo "| failed | | | | $(tail -1 "$OUT/tc.log") |"; }
masks() { # label java-opts
  # shellcheck disable=SC2086
  "$JAVA" --add-modules jdk.incubator.vector $2 -cp target/classes:target/test-classes org.robomiri.jllm.ConstraintBench \
    "models/$(echo "$DENSE" | tr / _)" 10 > "$OUT/cb.log" 2>&1 || { echo "| $label | failed | | $(tail -1 "$OUT/cb.log") |"; return; }
  local g d x
  g=$(grep '^agree *greedy' "$OUT/cb.log" | grep -o '[0-9.]* us/token' | grep -o '^[0-9.]*')
  d=$(grep '^distracted *greedy' "$OUT/cb.log" | grep -o '[0-9.]* us/token' | grep -o '^[0-9.]*')
  x=$(grep '^disagree *greedy' "$OUT/cb.log" | grep -o '[0-9.]* us/token' | grep -o '^[0-9.]*')
  echo "| $label | agree $g, distracted $d, disagree $x | | |"
}
echo
echo "| token mask of constrained sampling (greedy, us per token) | agree, distracted, disagree | | |"
echo "|---|---|---|---|"
masks "masks on (default)" ""
masks "masks on, shared memo (-Dwarm)" "-Dwarm=true"
masks "masks off (engine.constraint-masks=false)" "-Djllm.constraint.masks=false"

# audio (docs/21): the real-time factor (seconds of work per second of audio, lower is better) of one run of the command line
# tools, each in a fresh JVM, so it includes warm-up; scripts/audio-bench.sh has the steady-state stages
if [ -n "$AUDIO" ] && [ -z "$LLAMA_ONLY" ]; then
  audio_row() { # label java-opts args...
    local label="$1" opts="$2" rtf
    shift 2
    rtf=$(JAVA_OPTS="$opts" JLLM_CONFIG=/dev/null ./jllm "$@" 2>&1 >/dev/null | grep -o 'real-time factor [0-9.]*' | tail -1 | grep -o '[0-9.]*$' || true)
    echo "| $label | ${rtf:-failed} |"
  }
  echo
  echo "| audio | real-time factor |"
  echo "|---|---|"
  for k in cpu "metal --audio.metal.decoder off" "metal --audio.metal.decoder on"; do
    # shellcheck disable=SC2086
    audio_row "whisper-small, backend $k" "" transcribe --model openai/whisper-small --audio "$AUDIO" --audio.backend $k
  done
  audio_row "whisper-small, backend metal, float weight tiles" "-Djllm.audio.metalExact=true" transcribe --model openai/whisper-small \
    --audio "$AUDIO" --audio.backend metal
  audio_row "OuteTTS 0.3 500M, q8 language model" "" speak --model OuteAI/OuteTTS-0.3-500M --prompt "Speech synthesis is the artificial production of human speech." --out "$OUT/s8.wav"
  audio_row "OuteTTS 0.3 500M, q4 language model" "" speak --model OuteAI/OuteTTS-0.3-500M --prompt "Speech synthesis is the artificial production of human speech." --audio.tts.quant q4 --out "$OUT/s4.wav"
fi

if [ -n "$MOE" ]; then
  bench "$MOE: defaults (256-token chunks)" "$MOE" ""
  bench "64-token chunks" "$MOE" "" --batch 64
  bench "grouped MoE prefill off" "$MOE" "-Djllm.metal.moeMma=false"
  bench "half-weight MMA prefill forced on" "$MOE" "-Djllm.metal.hmma=true"
  bench "half-weight MMA prefill off (f32 MMA)" "$MOE" "-Djllm.metal.hmma=false"
  bench "decode replay off" "$MOE" "-Djllm.metal.replay=false"
  bench "experts of 4 layers on the CPU" "$MOE" "" --backend.metal.cpu-moe 4
  bench "experts of 4 layers on the CPU, int8 off" "$MOE" "" --backend.metal.cpu-moe 4 --backend.cpu.int8-activations false
  bench "all experts on the GPU" "$MOE" "" --backend.metal.cpu-moe off
  bench "$MOE with draft $DENSE, k 4 (adaptive)" "$MOE" "" --engine.speculative draft --engine.draft-model "$DENSE" \
    --engine.draft-tokens 4
fi
# GGUF block types read in place against the old behaviour (dequantize at load, requantize to Q8_0): the same file, a
# switch on the loader (-Djllm.gguf.requantize=all; a list of ggml type names limits it to those types)
if [ -n "$GGUF_TYPES" ]; then
  for f in "$GGUF_TYPES"/*.gguf; do
    [ -e "$f" ] || continue
    bench "$(basename "$f"): in place (default)" "$f" ""
    bench "$(basename "$f"): requantized to Q8_0 (jllm.gguf.requantize=all)" "$f" "-Djllm.gguf.requantize=all"
  done
fi
echo
echo "(median of $ROUNDS rounds; logs in $OUT)"
fi

# ---------------------------------------------------------------- jllm vs llama.cpp on the same GGUF file
# jllm's bench line gives the prompt length P and generated count N; llama-bench then runs -p P (prefill) and -n N
# (decode), at context depth P when this llama-bench supports -d, so both decode after a prompt of the same length.
# llama.cpp runs with flash attention off and on; the faster of the two is reported for each column.
versus() { # gguf-repo
  local repo="$1" line pre dec ptok ntok file depth=""
  ./jllm bench --model "$repo" --gguf-quant "$LLAMA_QUANT" --temp 0 --bench.rounds "$ROUNDS" \
      --engine.speculative off > "$OUT/vj.log" 2>&1 || { echo "| $repo | jllm failed: $(tail -1 "$OUT/vj.log") | | |"; return; }
  pre=$(grep -o '([0-9.]* tok/s), decode' "$OUT/vj.log" | grep -o '[0-9.]*' | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
  dec=$(grep -o 'decode [0-9]* tok in [0-9]* ms ([0-9.]* tok/s)' "$OUT/vj.log" | grep -o '([0-9.]*' | tr -d '(' | sort -n \
    | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
  line=$(grep -m1 '^prefill [0-9]* tok' "$OUT/vj.log")
  ptok=$(echo "$line" | awk '{print $2}')
  ntok=$(echo "$line" | grep -o 'decode [0-9]*' | awk '{print $2}')
  file=$(find models/"${repo/\//_}" -name "*${LLAMA_QUANT}*.gguf" ! -name '*-0000[2-9]-of-*' 2>/dev/null | sort | head -1)
  [ -n "$file" ] || { echo "| $repo | no ${LLAMA_QUANT} .gguf under models/${repo/\//_} | | |"; return; }
  local help depthNote="decode from an empty context (this llama-bench has no -d)"
  help=$("$LLAMA_BENCH" --help 2>&1 || true)
  if grep -q -- '--n-depth' <<< "$help"; then depth="-d $ptok"; depthNote="decode at depth $ptok"; fi
  # flash attention off and on; builds whose -fa takes on/off/auto instead get those values, then the default
  "$LLAMA_BENCH" -m "$file" -p "$ptok" -n "$ntok" $depth -r "$ROUNDS" -fa 0,1 -o json > "$OUT/vl.json" 2> "$OUT/vl.log" \
    || "$LLAMA_BENCH" -m "$file" -p "$ptok" -n "$ntok" $depth -r "$ROUNDS" -fa off,on -o json > "$OUT/vl.json" 2> "$OUT/vl.log" \
    || "$LLAMA_BENCH" -m "$file" -p "$ptok" -n "$ntok" $depth -r "$ROUNDS" -o json > "$OUT/vl.json" 2> "$OUT/vl.log" || {
    echo "| $repo | llama-bench failed: $(tail -1 "$OUT/vl.log") | | |"; return; }
  read -r lpre ldec < <(python3 - "$OUT/vl.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
pp = max((r["avg_ts"] for r in rows if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0), default=0)
tg = max((r["avg_ts"] for r in rows if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0), default=0)
print(f"{pp:.1f} {tg:.1f}")
PY
)
  local name; name=$(basename "$file")
  echo "| $name | jllm (Metal) | $pre | $dec | prompt $ptok tok, $ntok generated |"
  echo "| | llama.cpp | $lpre | $ldec | $depthNote |"
  awk -v a="$pre" -v b="$lpre" -v c="$dec" -v d="$ldec" \
    'BEGIN { printf "| | jllm / llama.cpp | %.2fx | %.2fx | |\n", (b > 0 ? a / b : 0), (d > 0 ? c / d : 0) }'
  ./jllm bench --model "$repo" --gguf-quant "$LLAMA_QUANT" --temp 0 --bench.rounds "$ROUNDS" > "$OUT/vs.log" 2>&1 && {
    dec=$(grep -o 'decode [0-9]* tok in [0-9]* ms ([0-9.]* tok/s)' "$OUT/vs.log" | grep -o '([0-9.]*' | tr -d '(' | sort -n \
      | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
    echo "| | jllm, default speculation | | $dec | the bench prompt repeats itself, so this favours speculation |"; }
}

if [ -n "$VS_LLAMA" ]; then
  LLAMA_BENCH="${LLAMA_BENCH:-$(command -v llama-bench || true)}"
  if [ -z "$LLAMA_BENCH" ]; then
    echo; echo "llama-bench not found: brew install llama.cpp, or set LLAMA_BENCH=/path/to/llama-bench"
  elif ! command -v python3 > /dev/null; then
    echo; echo "python3 not found (xcode-select --install): it reads llama-bench's JSON"
  else
    echo
    echo "llama.cpp: $("$LLAMA_BENCH" --version 2>&1 | head -1)"
    echo
    echo "| model file | engine | prefill tok/s | decode tok/s | note |"
    echo "|---|---|---|---|---|"
    versus "$LLAMA_DENSE"
    [ -z "$LLAMA_MOE" ] || versus "$LLAMA_MOE"
    echo
    echo "(jllm: median of $ROUNDS rounds; llama.cpp: mean of $ROUNDS repetitions, best of flash attention off/on)"
  fi
fi
