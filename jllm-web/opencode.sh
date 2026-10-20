#!/bin/sh
# Writes an opencode.json for the models loaded in a running jllm console into the current directory.
#   ./opencode.sh [console URL, default http://127.0.0.1:8090] [-f to overwrite]
# Then run `opencode` here. Set JLLM_API_KEY if the console was started with jllm.web.api-key.
# (Same as ./agent-config.sh opencode; that script also writes Continue, aider and Cline settings.)
exec "$(dirname "$0")/agent-config.sh" opencode "$@"
