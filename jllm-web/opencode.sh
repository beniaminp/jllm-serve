#!/bin/sh
# Writes an opencode.json for the models loaded in a running jllm console into the current directory.
#   ./opencode.sh [console URL, default http://127.0.0.1:8090] [-f to overwrite]
# Then run `opencode` here. Set JLLM_API_KEY if the console was started with jllm.web.api-key.
set -eu
URL="http://127.0.0.1:8090"
FORCE=""
for a in "$@"; do
  case "$a" in
    -f) FORCE=1 ;;
    *) URL="${a%/}" ;;
  esac
done
if [ -f opencode.json ] && [ -z "$FORCE" ]; then
  echo "opencode.json exists here; rerun with -f to overwrite (or merge the provider block by hand):" >&2
  curl -fsS ${JLLM_API_KEY:+-H "Authorization: Bearer $JLLM_API_KEY"} "$URL/api/integrations/opencode?raw=true"
  exit 1
fi
TMP="$(mktemp)"
if ! curl -fsS ${JLLM_API_KEY:+-H "Authorization: Bearer $JLLM_API_KEY"} "$URL/api/integrations/opencode?raw=true" -o "$TMP"; then
  rm -f "$TMP"
  echo "could not get the configuration from $URL (is the console running, and a model loaded?)" >&2
  exit 1
fi
mv "$TMP" opencode.json
echo "wrote $(pwd)/opencode.json; warnings, if any, are on the console's Connect page ($URL/#api)"
