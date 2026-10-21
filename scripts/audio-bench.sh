#!/bin/bash
# Speed of the audio models on this machine (docs/21-audio.md, "Measurements"): Whisper speech-to-text on the CPU and on
# Metal (decoder on the CPU, and on the GPU), against whisper.cpp when whisper-cli and its ggml models are there, and
# OuteTTS text-to-speech with the WavTokenizer codec. Run it idle (the repository's agents do it under the heavy lock).
#
#   scripts/audio-bench.sh <speech.wav> [--stt "tiny base small large-v3-turbo"] [--tts "OuteAI/OuteTTS-0.3-500M"]
#                          [--ggml-dir DIR] [--text "sentence to speak"] [--reps 3]
#
# <speech.wav> should be several seconds to a minute or two of speech (`say -o x.aiff "..."; afconvert -f WAVE -d LEI16@16000
# x.aiff x.wav` makes one). The ggml models are ggml-<size>.bin of https://huggingface.co/ggerganov/whisper.cpp.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -n "${JAVA_HOME:-}" ] || export JAVA_HOME=/opt/homebrew/opt/openjdk@26
AUDIO="${1:?usage: audio-bench.sh <speech.wav> [options]}"
shift
STT="tiny base small"
TTS="OuteAI/OuteTTS-0.3-500M"
GGML=""
TEXT="Speech synthesis is the artificial production of human speech. A computer system used for this purpose is called a speech synthesizer."
REPS=3
while [ $# -gt 0 ]; do
  case "$1" in
    --stt) STT="$2"; shift 2 ;;
    --tts) TTS="$2"; shift 2 ;;
    --ggml-dir) GGML="$2"; shift 2 ;;
    --text) TEXT="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
mvn -q -o test-compile
CP="target/classes:target/test-classes"
JAVA="$JAVA_HOME/bin/java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g"
[ -f target/native/libjllm_metal.dylib ] && JAVA="$JAVA -Djllm.metal.lib=target/native/libjllm_metal.dylib"
[ -f target/native/libjllm_cpu.dylib ] && JAVA="$JAVA -Djllm.cpu.nativeLib=target/native/libjllm_cpu.dylib"

echo "machine: $(sysctl -n machdep.cpu.brand_string), $(($(sysctl -n hw.memsize) / 1073741824)) GB, $(sw_vers -productVersion)"
for m in $STT; do
  dir="models/openai_whisper-$m"
  [ -d "$dir" ] || { echo "(no $dir, skipped)"; continue; }
  echo; echo "== whisper-$m"
  $JAVA -cp $CP org.robomiri.jllm.AudioBench stt "$dir" "$AUDIO" cpu auto "$REPS" 2>&1 | grep -v WARNING
  $JAVA -cp $CP org.robomiri.jllm.AudioBench stt "$dir" "$AUDIO" metal off "$REPS" 2>&1 | grep -v WARNING
  $JAVA -cp $CP org.robomiri.jllm.AudioBench stt "$dir" "$AUDIO" metal on "$REPS" 2>&1 | grep -v WARNING
  if [ -n "$GGML" ] && command -v whisper-cli >/dev/null && [ -f "$GGML/ggml-$m.bin" ]; then
    for mode in "" "-ng"; do
      s=$(date +%s.%N)
      whisper-cli -m "$GGML/ggml-$m.bin" -f "$AUDIO" -l auto $mode -nt > /tmp/whisper-cpp.out 2> /tmp/whisper-cpp.err || true
      e=$(date +%s.%N)
      dur=$(python3 -c "import wave,sys; w=wave.open(sys.argv[1]); print(w.getnframes()/w.getframerate())" "$AUDIO")
      echo "  whisper.cpp ${mode:-(Metal)}${mode:+(CPU)}: $(echo "$e $s" | awk '{printf "%.2f", $1-$2}') s wall for $dur s of audio (includes loading the model);" \
           "$(grep -o 'total time = *[0-9.]* ms' /tmp/whisper-cpp.err | tail -1)"
    done
  fi
done
for t in $TTS; do
  dir="models/$(echo "$t" | tr '/' '_')"
  [ -d "$dir" ] || { echo "(no $dir, skipped)"; continue; }
  echo; echo "== $t"
  $JAVA -cp $CP org.robomiri.jllm.AudioBench tts "$dir" "$TEXT" en_male_1 "$REPS" 2>&1 | grep -v WARNING | grep -v "^Metal\|^Backend\|^Speculative\|^Mapped\|^native\|^int8"
  $JAVA -cp $CP org.robomiri.jllm.AudioBench tts "$dir" "$TEXT" en_male_1 "$REPS" --audio.tts.quant=q4 2>&1 | grep -v WARNING | grep "real-time\|loaded\|load " | sed 's/^/  (q4) /'
done
codec=models/ggml-org_WavTokenizer/WavTokenizer-Large-75-F16.gguf
if [ -f "$codec" ]; then
  echo; echo "== WavTokenizer decoder (CPU)"
  $JAVA -cp $CP org.robomiri.jllm.AudioBench codec "$codec" 750 "$REPS" 2>&1 | grep -v WARNING
fi
