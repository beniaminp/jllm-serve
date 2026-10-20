#!/bin/sh
# Writes the configuration of a coding agent for the models loaded in a running jllm console.
#   ./agent-config.sh opencode|continue|aider|cline [console URL, default http://127.0.0.1:8090] [-f to overwrite]
#
#   opencode   opencode.json in the current directory
#   continue   ~/.continue/config.yaml (an existing one is kept: the new file goes next to it as config.jllm.yaml)
#   aider      .aider.conf.yml, .aider.model.metadata.json, .aider.model.settings.yml in the current directory
#   cline      prints the values for Cline's "OpenAI Compatible" provider (Cline has no configuration file)
#
# Set JLLM_API_KEY if the console was started with jllm.web.api-key.
set -eu
AGENT="${1:-}"
[ -n "$AGENT" ] && shift
URL="http://127.0.0.1:8090"
FORCE=""
for a in "$@"; do
  case "$a" in
    -f) FORCE=1 ;;
    *) URL="${a%/}" ;;
  esac
done

get() { # get <path> <file>
  if ! curl -fsS ${JLLM_API_KEY:+-H "Authorization: Bearer $JLLM_API_KEY"} "$URL$1" -o "$2"; then
    rm -f "$2"
    echo "could not get the configuration from $URL (is the console running, and a model loaded?)" >&2
    exit 1
  fi
}

put() { # put <path> <target>: download, refusing to overwrite without -f
  if [ -f "$2" ] && [ -z "$FORCE" ]; then
    echo "$2 exists; rerun with -f to overwrite it" >&2
    exit 1
  fi
  TMP="$(mktemp)"
  get "$1" "$TMP"
  mkdir -p "$(dirname "$2")"
  mv "$TMP" "$2"
  echo "wrote $2"
}

case "$AGENT" in
  opencode)
    put "/api/integrations/opencode?raw=true" "$(pwd)/opencode.json"
    echo "run opencode here; warnings, if any, are on the console's Connect page ($URL/#api)" ;;
  continue)
    TARGET="$HOME/.continue/config.yaml"
    if [ -f "$TARGET" ] && [ -z "$FORCE" ]; then
      TARGET="$HOME/.continue/config.jllm.yaml"
      FORCE=1
      echo "~/.continue/config.yaml exists: copy the models from the new file into it (or rerun with -f to replace it)"
    fi
    put "/api/integrations/continue?raw=true" "$TARGET" ;;
  aider)
    for f in conf:.aider.conf.yml metadata:.aider.model.metadata.json settings:.aider.model.settings.yml; do
      put "/api/integrations/aider?file=${f%%:*}" "$(pwd)/${f#*:}"
    done
    echo "run aider here" ;;
  cline)
    TMP="$(mktemp)"
    get "/api/integrations/cline" "$TMP"
    python3 - "$TMP" <<'PY' || cat "$TMP"
import json, sys
r = json.load(open(sys.argv[1]))
print("In Cline's settings choose API Provider: " + r["provider"])
print("  Base URL:  " + r["baseUrl"])
print("  API Key:   " + r["apiKey"])
for m in r["models"]:
    print("  Model ID:  %s   (context window %d, max output tokens %d, images and computer use off)"
          % (m["modelId"], m["contextWindow"], m["maxOutputTokens"]))
for w in r["warnings"]:
    print("warning: " + w)
PY
    rm -f "$TMP" ;;
  *)
    sed -n '2,11p' "$0" >&2
    exit 2 ;;
esac
