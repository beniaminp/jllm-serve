#!/bin/bash
# Qwen3.5 / Qwen3-Next GGUF against llama.cpp itself, on tiny random models (no download, ~1 minute):
#   scripts/qwen35-llamacpp-check.sh            needs llama-server (brew install llama.cpp, or LLAMA_SERVER=...), JDK 26, python3
#
# GgufTinyWrite writes the tiny hybrids of GgufQwenNextTest the way llama.cpp's converter writes them, with F32 tensors
# holding Q8_0-representable values (so llama.cpp and jllm both compute with exactly these weights and float activations),
# a vocabulary (so llama.cpp loads them) and a multi-token-prediction block. For three prompts, fed as token ids:
#   * the top-5 log-probabilities of the first position and 12 greedy ids must agree (llama.cpp vs jllm GgufGreedy);
#   * with the output head focused on four tokens (so drafts are often right): at each of 60 positions, whether llama.cpp's
#     own multi-token-prediction draft (--spec-type draft-mtp) equals the next token must be what jllm's head says.
# Prints both sides; exits 1 on a difference of the greedy ids or of the draft bits.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x /opt/homebrew/opt/openjdk@26/bin/java ] && export JAVA_HOME=/opt/homebrew/opt/openjdk@26
JAVA="${JAVA_HOME:?set JAVA_HOME to a JDK 26}/bin/java"
LLAMA_SERVER="${LLAMA_SERVER:-$(command -v llama-server)}"
mvn -q compile test-compile
OUT=$(mktemp -d)
CP=target/classes:target/test-classes
J="$JAVA --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -cp $CP"
fail=0
P1=$((20000 + RANDOM % 20000))
P2=$((P1 + 1))

serve() { # file port extra...: starts llama-server on the file, waits until it answers
  local f="$1" port="$2"
  shift 2
  "$LLAMA_SERVER" -m "$f" -c 512 -ngl 0 --port "$port" --no-webui -t 4 -ctk f32 -ctv f32 --no-warmup "$@" > "$OUT/server-$port.log" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 60); do curl -s "http://127.0.0.1:$port/health" | grep -q ok && return; sleep 1; done
  echo "llama-server did not start (see $OUT/server-$port.log)"; exit 1
}

PROMPTS='[[1,5,17,33,2,40,9,77,12,60,3,21],[7,7,7,8,9,10,11,12,13,14],[50,40,30,20,10,60,70,80,90,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]'
IDS=("ids:1,5,17,33,2,40,9,77,12,60,3,21" "ids:7,7,7,8,9,10,11,12,13,14" "ids:50,40,30,20,10,60,70,80,90,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")

compare() { # label arch extra-writer-args...
  local label="$1" arch="$2"
  shift 2
  $J org.robomiri.jllm.GgufTinyWrite "$arch" "$OUT/$label.gguf" deq ba "$@" | tail -1 > /dev/null
  serve "$OUT/$label.gguf" "$P1"
  python3 - "$PROMPTS" "$P1" > "$OUT/$label-ll.txt" <<'EOF'
import json, sys, urllib.request
for p in json.loads(sys.argv[1]):
    req = urllib.request.Request('http://127.0.0.1:%s/completion' % sys.argv[2], data=json.dumps(
        {"prompt": p, "n_predict": 12, "temperature": 0, "top_k": 1, "n_probs": 5, "cache_prompt": False,
         "return_tokens": True}).encode(), headers={'Content-Type': 'application/json'})
    d = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
    cp = d['completion_probabilities']
    print('ids', [e['id'] for e in cp])
    print('top', [(t['id'], round(t['logprob'], 3)) for t in cp[0]['top_logprobs']])
EOF
  kill "$SERVER"; wait "$SERVER" 2>/dev/null || true
  $J org.robomiri.jllm.GgufGreedy "$OUT/$label.gguf" cpu -n 12 --ctx 512 --kv f32 "${IDS[@]}" 2>&1 \
    | python3 -c '
import re, sys
ids = []; top = []; cur = None
for line in sys.stdin:
    if line.startswith("greedy ids:"):
        print("ids", eval(line.split(":", 1)[1]))
    m = re.match(r"\s+(\d+)\s+[-\d.]+\s+([-\d.]+)\s", line)
    if m:
        top.append((int(m.group(1)), round(float(m.group(2)), 3)))
        if len(top) == 5:
            print("top", top); top = []
' > "$OUT/$label-jl.txt"
  echo "== $label: llama.cpp vs jllm (first position top-5 log-probabilities may differ in the last printed digit)"
  # llama.cpp stops at the end-of-sequence id (2); jllm's tool does not: compare the ids llama.cpp produced
  python3 - "$OUT/$label-ll.txt" "$OUT/$label-jl.txt" <<'EOF' && echo "   greedy ids: identical (3 prompts, up to 12 tokens)" || { echo "   greedy ids DIFFER"; fail=1; }
import sys
def ids(f):
    return [eval(l.split(" ", 1)[1]) for l in open(f) if l.startswith("ids")]
ll, jl = ids(sys.argv[1]), ids(sys.argv[2])
sys.exit(0 if len(ll) == len(jl) and all(j[:len(l)] == l for l, j in zip(ll, jl)) else 1)
EOF
  grep '^top' "$OUT/$label-ll.txt" | head -1 | sed 's/^/   llama.cpp /'
  grep '^top' "$OUT/$label-jl.txt" | head -1 | sed 's/^/   jllm      /'
}

compare qwen35 qwen35
compare qwen3next qwen3next allmoe

# the multi-token-prediction head: per position, does the draft equal the next token?
$J org.robomiri.jllm.GgufTinyWrite qwen35 "$OUT/mtp.gguf" deq ba focus | tail -1 > /dev/null
serve "$OUT/mtp.gguf" "$P2" --spec-type draft-mtp --spec-draft-n-max 1
python3 - "$P2" > "$OUT/mtp-ll.txt" <<'EOF'
import json, sys, urllib.request
def post(prompt, n):
    req = urllib.request.Request('http://127.0.0.1:%s/completion' % sys.argv[1], data=json.dumps(
        {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "cache_prompt": False, "return_tokens": True,
         "ignore_eos": True}).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
p = [1, 5, 17, 33, 2, 40, 9, 77, 12, 60, 3, 21]
s = post(p, 61)['tokens']
bits = ''
for j in range(60):
    t = post(p + s[:j], 3)['timings']
    bits += '1' if (t.get('draft_n', 0) == 1 and t.get('draft_n_accepted', 0) == 1) else '0'
print(bits)
EOF
kill "$SERVER"; wait "$SERVER" 2>/dev/null || true
$J org.robomiri.jllm.GgufGreedy "$OUT/mtp.gguf" cpu -n 61 --ctx 512 --kv f32 --mtp "ids:1,5,17,33,2,40,9,77,12,60,3,21" 2>&1 \
  | sed -n 's/^mtp: .*: \([01]*\)$/\1/p' > "$OUT/mtp-jl.txt"
echo "== multi-token prediction: does the head's draft equal the next token, at 60 positions"
echo "   llama.cpp $(cat "$OUT/mtp-ll.txt")"
echo "   jllm      $(cat "$OUT/mtp-jl.txt")"
cmp -s "$OUT/mtp-ll.txt" "$OUT/mtp-jl.txt" && echo "   identical ($(tr -cd 1 < "$OUT/mtp-ll.txt" | wc -c | tr -d ' ') hits)" || { echo "   DIFFERENT"; fail=1; }
exit $fail
