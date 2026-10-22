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
#   scripts/mac-validate.sh --qwen35 /path/to/Qwen3.5-27B-Q4_0.gguf   plus multi-token-prediction speculation on/off
#                                                                     (a qwen35 GGUF with its nextn layer; ~19 GB of memory)
#   scripts/mac-validate.sh --vs-llama                       plus jllm vs llama.cpp (llama-bench) on the same GGUF file
#   scripts/mac-validate.sh --vs-llama --llama-moe unsloth/Qwen3-30B-A3B-GGUF   ... and on a MoE GGUF
#   scripts/mac-validate.sh --llama-only --vs-llama          only the llama.cpp comparison
#   scripts/mac-validate.sh --vision mmproj-X.gguf pic.png   plus the vision encoder rows (docs/20): encoding the
#                                                            picture on the GPU (default) against the CPU, warm
#   scripts/mac-validate.sh --vision mmproj-X.gguf pic.png --vision-model text.gguf   ... and the whole answer to the picture
#                                                            (prefill including the encoding), a Gemma 3 file also with
#                                                            attention that runs both ways inside the picture on and off
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
QWEN35=""
SMOKE=1
ROUNDS=3
VS_LLAMA=""
LLAMA_ONLY=""
LLAMA_DENSE=unsloth/Qwen3-0.6B-GGUF
LLAMA_MOE=""
LLAMA_QUANT=Q4_0
VISION_MMPROJ=""
VISION_IMAGE=""
VISION_MODEL=""
AUDIO=""
GGUF_TYPES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) DENSE="$2"; shift 2 ;;
    --moe) MOE="$2"; shift 2 ;;
    --qwen35) QWEN35="$2"; shift 2 ;;
    --skip-smoke) SMOKE=""; shift ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --vs-llama) VS_LLAMA=1; shift ;;
    --llama-only) LLAMA_ONLY=1; SMOKE=""; shift ;;
    --llama-model) LLAMA_DENSE="$2"; shift 2 ;;
    --llama-moe) LLAMA_MOE="$2"; shift 2 ;;
    --llama-quant) LLAMA_QUANT="$2"; shift 2 ;;
    --vision) VISION_MMPROJ="$2"; VISION_IMAGE="$3"; shift 3 ;;
    --vision-model) VISION_MODEL="$2"; shift 2 ;;
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
# the serial compute encoder (default) against the concurrent one with hazard-tracked barriers (docs/08 8.28); "unsafe"
# has no barriers at all (wrong results), it times the ceiling of what overlapping dependent kernels could gain
bench "concurrent encoder off (serial, default)" "$DENSE" "-Djllm.metal.concurrent=false"
bench "concurrent encoder on" "$DENSE" "-Djllm.metal.concurrent=true"
bench "concurrent encoder on, pipelined decoding off" "$DENSE" "-Djllm.metal.concurrent=true -Djllm.metal.pipeline=false"
bench "concurrent encoder, a barrier before every dispatch" "$DENSE" "-Djllm.metal.concurrent=barriers"
bench "concurrent encoder, no barriers (WRONG RESULTS, timing only)" "$DENSE" "-Djllm.metal.concurrent=unsafe"
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
# prefill chunk (backend.batch=auto is 256 for dense decoders on Metal; 64 was the default before)
bench "64-token prefill chunks (the old default)" "$DENSE" "" --batch 64
bench "512-token prefill chunks" "$DENSE" "" --batch 512
# multi-row decode matvec (NR rows per simdgroup; mvRows=1 brings the one-row kernels back) and the decode attention
bench "decode matvec: one row per simdgroup (mvRows=1)" "$DENSE" "-Djllm.metal.mvRows=1"
bench "decode matvec: 2 rows per simdgroup" "$DENSE" "-Djllm.metal.mvRows=2"
bench "decode matvec: 4 rows per simdgroup" "$DENSE" "-Djllm.metal.mvRows=4"
bench "decode attention off (attention_t)" "$DENSE" "-Djllm.metal.decodeAttention=false"
bench "decode attention on (attention_dec)" "$DENSE" "-Djllm.metal.decodeAttention=true"
bench "~1000-token prompt: decode attention off" "$DENSE" "-Djllm.metal.decodeAttention=false" --bench.prompt-repeat 100
bench "~1000-token prompt: decode attention on" "$DENSE" "-Djllm.metal.decodeAttention=true" --bench.prompt-repeat 100
bench "~3000-token prompt: decode attention off" "$DENSE" "-Djllm.metal.decodeAttention=false" --bench.prompt-repeat 300
bench "~3000-token prompt: decode attention on" "$DENSE" "-Djllm.metal.decodeAttention=true" --bench.prompt-repeat 300
bench "~2000-token prompt: defaults" "$DENSE" "" --bench.prompt-repeat 200
bench "~2000-token prompt: hmma off, fa" "$DENSE" "-Djllm.metal.hmma=false -Djllm.metal.flashPrefillKernel=fa" --bench.prompt-repeat 200
bench "~2000-token prompt: hmma on, fb" "$DENSE" "-Djllm.metal.hmma=true -Djllm.metal.flashPrefillKernel=fb" --bench.prompt-repeat 200
bench "f16 KV cache" "$DENSE" "" --backend.kv-dtype f16 --engine.speculative off
bench "q8 KV cache (default: rotated keys, backend.kv-q8-rotate)" "$DENSE" "" --backend.kv-dtype q8 --engine.speculative off
bench "q8 KV cache, plain keys (backend.kv-q8-rotate false)" "$DENSE" "" --backend.kv-dtype q8 --backend.kv-q8-rotate false --engine.speculative off
bench "q4 KV cache" "$DENSE" "" --backend.kv-dtype q4 --engine.speculative off
# the paged KV cache (docs/24): one pool of 16-token blocks and a block table per slot instead of a flat array, against the
# flat cache on the same prompts (the flat rows are the defaults above; backend.kv-paged=false is the default)
bench "paged KV cache (f32, 16-token blocks)" "$DENSE" "" --backend.kv-paged true --engine.speculative off
bench "paged KV cache, f16" "$DENSE" "" --backend.kv-paged true --backend.kv-dtype f16 --engine.speculative off
bench "paged KV cache, q8" "$DENSE" "" --backend.kv-paged true --backend.kv-dtype q8 --engine.speculative off
bench "paged KV cache, 64-token blocks" "$DENSE" "" --backend.kv-paged true --backend.kv-block 64 --engine.speculative off
bench "~3000-token prompt: flat KV cache" "$DENSE" "" --bench.prompt-repeat 300 --engine.speculative off
bench "~3000-token prompt: paged KV cache" "$DENSE" "" --bench.prompt-repeat 300 --backend.kv-paged true --engine.speculative off
bench "~3000-token prompt: paged KV cache, decode replay off" "$DENSE" "-Djllm.metal.replay=false" --bench.prompt-repeat 300 --backend.kv-paged true --engine.speculative off
bench "q4 KV cache, no recent window" "$DENSE" "" --backend.kv-dtype q4 --backend.kv-q4-recent 0 --engine.speculative off
bench "q4 KV cache, recent window 512" "$DENSE" "" --backend.kv-dtype q4 --backend.kv-q4-recent 512 --engine.speculative off
bench "~2000-token prompt: f16 KV cache" "$DENSE" "" --backend.kv-dtype f16 --engine.speculative off --bench.prompt-repeat 200
bench "~2000-token prompt: q4 KV cache" "$DENSE" "" --backend.kv-dtype q4 --engine.speculative off --bench.prompt-repeat 200
bench "~2000-token prompt: q8 KV cache (rotated keys)" "$DENSE" "" --backend.kv-dtype q8 --engine.speculative off --bench.prompt-repeat 200
bench "~2000-token prompt: q8 KV cache, plain keys" "$DENSE" "" --backend.kv-dtype q8 --backend.kv-q8-rotate false --engine.speculative off --bench.prompt-repeat 200
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
batch "3 slots generating together, concurrent encoder on" "-Djllm.metal.concurrent=true"
# the vision encoder (docs/20): the second encoding of a picture (weights resident, pipelines built) on the GPU against
# the CPU encoder (vision.backend=cpu); VisionDump prints the time per chunk of the picture
vision() { # label backend
  local label="$1" backend="$2"
  "$JAVA" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g -Djllm.vision.backend="$backend" \
    -Djllm.vision.twice=true -Djllm.metal.lib=target/native/libjllm_metal.dylib -cp target/classes:target/test-classes \
    org.robomiri.jllm.vision.VisionDump "$VISION_MMPROJ" "$VISION_IMAGE" "$OUT/vision" > "$OUT/vision.log" 2>&1 || {
    echo "| $label | failed | | $(tail -1 "$OUT/vision.log") |"; return; }
  echo "| $label | $(grep '^chunk' "$OUT/vision.log" | sed 's/.* in \([0-9]*\) ms/\1 ms/' | paste -sd, -) | | $(grep -o '[0-9]* tokens' "$OUT/vision.log" | paste -sd, -) |"
}
if [ -n "$VISION_MMPROJ" ]; then
  echo
  echo "| vision encoder: $(basename "$VISION_MMPROJ"), $(basename "$VISION_IMAGE") | encode time per chunk | | tokens per chunk |"
  echo "|---|---|---|---|"
  vision "on the GPU (vision.backend=metal, default)" metal
  vision "on the CPU (vision.backend=cpu)" cpu
  if [ -n "$VISION_MODEL" ]; then
    # the whole answer: ./jllm run with the picture; its last line has the prefill time (the encoding is before it) and the speed
    answer() { # label java-opts
      local label="$1" opts="$2"
      JAVA_OPTS="$opts" JLLM_CONFIG=/dev/null ./jllm run --model "$VISION_MODEL" --mmproj "$VISION_MMPROJ" --image "$VISION_IMAGE" \
        --prompt "Describe the picture." --temp 0 --max-tokens 16 --backend metal --ctx 1024 > "$OUT/answer.log" 2>&1 || {
        echo "| $label | failed | | $(tail -1 "$OUT/answer.log") |"; return; }
      echo "| $label | $(grep -o '[0-9]* ms to encode' "$OUT/answer.log" | head -1) | $(grep -o 'prefill [0-9]* ms' "$OUT/answer.log" | head -1) | $(grep -o '[0-9.]* tok/s' "$OUT/answer.log" | tail -1) |"
    }
    echo
    echo "| whole answer, 16 tokens | encode | prefill | decode |"
    echo "|---|---|---|---|"
    answer "defaults" ""
    # Gemma 3 only (other families have no picture attention of their own): a picture attends to itself both ways by default
    answer "attention inside the picture causal (-Djllm.vision.causal=true)" "-Djllm.vision.causal=true"
  fi
fi
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
  local label="$1"
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
  bench "concurrent encoder on" "$MOE" "-Djllm.metal.concurrent=true"
  bench "concurrent encoder on, experts of 4 layers on the CPU" "$MOE" "-Djllm.metal.concurrent=true" --backend.metal.cpu-moe 4
  bench "experts of 4 layers on the CPU" "$MOE" "" --backend.metal.cpu-moe 4
  bench "experts of 4 layers on the CPU, int8 off" "$MOE" "" --backend.metal.cpu-moe 4 --backend.cpu.int8-activations false
  bench "all experts on the GPU" "$MOE" "" --backend.metal.cpu-moe off
  bench "$MOE with draft $DENSE, k 4 (adaptive)" "$MOE" "" --engine.speculative draft --engine.draft-model "$DENSE" \
    --engine.draft-tokens 4
fi
# image generation (docs/22): Stable Diffusion on the GPU, each switch on and off (needs the diffusers folder of a model in
# models/, default stabilityai/sd-turbo; IMAGE_MODEL=<folder> to choose another)
IMG="${IMAGE_MODEL:-models/stabilityai_sd-turbo}"
if [ -f "$IMG/model_index.json" ]; then
  imgbench() { # label java-opts backend precision steps
    local label="$1" opts="$2" backend="$3" prec="$4" steps="$5"
    # shellcheck disable=SC2086
    "$JAVA" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -XX:+UseParallelGC -Xmx6g $opts \
      -Djllm.metal.lib=target/native/libjllm_metal.dylib -cp target/classes:target/test-classes \
      org.robomiri.jllm.diffusion.DiffusionBench "$IMG" "$backend" "$prec" "$steps" 512 512 3 > "$OUT/img.log" 2>&1 || {
      echo "| $label | failed | | $(tail -1 "$OUT/img.log") |"; return; }
    local line total unet vae
    line=$(grep '^warm' "$OUT/img.log" | tail -1)
    total=$(echo "$line" | sed 's/.*: \([0-9.]*\) s total.*/\1/')
    unet=$(echo "$line" | sed 's/.*(\([0-9.]*\) s\/eval).*/\1/')
    vae=$(echo "$line" | sed 's/.*+ vae \([0-9.]*\) .*/\1/')
    echo "| $label | $total | $unet | $vae |"
  }
  echo
  echo "| image generation, $(basename "$IMG") 512x512 | s per image | UNet s/eval | VAE s |"
  echo "|---|---|---|---|"
  imgbench "Metal f16, 1 step (defaults)" "" metal f16 1
  imgbench "Metal f16, 4 steps" "" metal f16 4
  imgbench "Metal f32, 4 steps" "" metal f32 4
  imgbench "flash attention off (scores + softmax), 4 steps" "-Djllm.diffusion.flash=false" metal f16 4
  imgbench "chunked group norm off, 4 steps" "-Djllm.diffusion.simpleNorm=true" metal f16 4
  imgbench "CPU f32, 1 step" "" cpu f32 1
fi
# video generation (docs/23): AnimateDiff (motion adapter on Stable Diffusion 1.5) on the GPU, each switch on and off (needs the
# folders of the base model and of the motion adapter in models/; VIDEO_BASE / VIDEO_ADAPTER choose others)
VBASE="${VIDEO_BASE:-models/stable-diffusion-v1-5_stable-diffusion-v1-5}"
VADAPT="${VIDEO_ADAPTER:-models/guoyww_animatediff-motion-adapter-v1-5-2}"
if [ -f "$VBASE/model_index.json" ] && [ -f "$VADAPT/config.json" ]; then
  vidbench() { # label java-opts backend precision steps size frames
    local label="$1" opts="$2" backend="$3" prec="$4" steps="$5" size="$6" frames="$7"
    # shellcheck disable=SC2086
    "$JAVA" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -XX:+UseParallelGC -Xmx6g $opts \
      -Djllm.metal.lib=target/native/libjllm_metal.dylib -cp target/classes:target/test-classes \
      org.robomiri.jllm.diffusion.VideoBench "$VBASE" "$VADAPT" "$backend" "$prec" "$steps" "$size" "$size" "$frames" 2 > "$OUT/video.log" 2>&1 || {
      echo "| $label | failed | | | $(tail -1 "$OUT/video.log") |"; return; }
    local line total unet vae
    line=$(grep '^warm' "$OUT/video.log" | tail -1)
    total=$(echo "$line" | sed 's/.*: \([0-9.]*\) s total.*/\1/')
    unet=$(echo "$line" | sed 's/.*(\([0-9.]*\) s\/step).*/\1/')
    vae=$(echo "$line" | sed 's/.*(\([0-9.]*\) s\/frame).*/\1/')
    echo "| $label | $total | $unet | $vae |"
  }
  echo
  echo "| video generation, AnimateDiff on $(basename "$VBASE") | s per clip | UNet s/step | VAE s/frame |"
  echo "|---|---|---|---|"
  vidbench "Metal f16, 256x256, 16 frames, 4 steps (defaults)" "" metal f16 4 256 16
  vidbench "Metal f32, 256x256, 16 frames, 4 steps" "" metal f32 4 256 16
  vidbench "Metal f16, 512x512, 16 frames, 2 steps" "" metal f16 2 512 16
  vidbench "flash attention off (scores + softmax), 256x256, 16 frames, 2 steps" "-Djllm.diffusion.flash=false" metal f16 2 256 16
  vidbench "chunked group norm off, 256x256, 16 frames, 2 steps" "-Djllm.diffusion.simpleNorm=true" metal f16 2 256 16
  vidbench "CPU f32, 128x128, 8 frames, 1 step" "" cpu f32 1 128 8
fi
# Wan 2.1 (docs/23 section 23.12): the transformer and the VAE decoder on random inputs (the text encoder is not needed for timing); needs the
# transformer and vae folders of Wan-AI/Wan2.1-T2V-1.3B-Diffusers in models/ (WAN_MODEL=<folder> to choose another)
WANM="${WAN_MODEL:-models/Wan-AI_Wan2.1-T2V-1.3B-Diffusers}"
if [ -f "$WANM/transformer/config.json" ] && [ -f "$WANM/vae/config.json" ]; then
  wanbench() { # label java-opts backend precision latent-frames h w
    local label="$1" opts="$2" backend="$3" prec="$4" f="$5" h="$6" w="$7"
    # shellcheck disable=SC2086
    "$JAVA" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -XX:+UseParallelGC -Xmx6g $opts \
      -Djllm.metal.lib=target/native/libjllm_metal.dylib -cp target/classes:target/test-classes \
      org.robomiri.jllm.diffusion.WanBench "$WANM" "$backend" "$prec" "$f" "$h" "$w" 2 > "$OUT/wan.log" 2>&1 || {
      echo "| $label | failed | | $(tail -1 "$OUT/wan.log") |"; return; }
    local dit vae
    dit=$(grep '^warm  transformer' "$OUT/wan.log" | sed 's/.*: \([0-9.]*\) s per.*/\1/')
    vae=$(grep '^warm  vae' "$OUT/wan.log" | sed 's/.*: \([0-9.]*\) s$/\1/')
    echo "| $label | $dit | $vae |"
  }
  echo
  echo "| Wan 2.1 1.3B, 5 latent frames of 24 x 40 (1200 tokens; 17 frames of 192 x 320) | transformer s per evaluation | VAE decode s |"
  echo "|---|---|---|"
  wanbench "Metal f32 (defaults)" "" metal f32 5 24 40
  wanbench "Metal f16" "" metal f16 5 24 40
  wanbench "flash attention off (scores + softmax), f32" "-Djllm.diffusion.flash=false" metal f32 5 24 40
  wanbench "chunked group norm off, f32" "-Djllm.diffusion.simpleNorm=true" metal f32 5 24 40
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
# multi-token prediction (engine.speculative=mtp, docs/10 section 10.9, docs/16 section 16.14): the model's own nextn
# head drafts one token per step, the target verifies it with the next position. The bench prompt repeats itself,
# which flatters every speculation mode a little.
if [ -n "$QWEN35" ]; then
  bench "$QWEN35: speculation off" "$QWEN35" "" --engine.speculative off
  bench "$QWEN35: multi-token prediction (mtp)" "$QWEN35" "" --engine.speculative mtp
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
    echo "llama.cpp: $("$LLAMA_BENCH" --version 2>&1 | grep -m1 -i version)"
    echo
    echo "| model file | engine | prefill tok/s | decode tok/s | note |"
    echo "|---|---|---|---|---|"
    versus "$LLAMA_DENSE"
    [ -z "$LLAMA_MOE" ] || versus "$LLAMA_MOE"
    echo
    echo "(jllm: median of $ROUNDS rounds; llama.cpp: mean of $ROUNDS repetitions, best of flash attention off/on)"
  fi
fi
