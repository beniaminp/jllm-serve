#!/bin/sh
# Launcher for the jllm web console: ./run.sh [Spring Boot options, e.g. --server.port=9000 --jllm.web.api-key=secret]
#
# Builds the engine (installed into the local Maven repository) and this module when needed, then starts the
# console at http://127.0.0.1:8090/. Needs JDK 26+. Environment:
#   JAVA_HOME      JDK to use (default: the java on PATH)
#   JLLM_XMX       Java heap size, default 60% of the RAM. The CPU backend keeps its KV caches on the heap: one per
#                  slot, e.g. 3.8 GB for a 0.6B Qwen3 at a 32768 context with an f16 cache (weights are memory-mapped
#                  outside the heap). The young generation is kept small (JLLM_XMN, default 512m) so that the
#                  caches, which live in the old generation, can use nearly all of it.
#   JAVA_OPTS      extra JVM options
#   JLLM_NO_BUILD  set to skip the Maven builds
#
# Relative paths (model.dir, jllm-web.properties, ./jllm.properties) resolve against the directory you run it from.
set -eu
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$DIR")"
JAVA="${JAVA_HOME:+$JAVA_HOME/bin/}java"
JAR="$DIR/target/jllm-web.jar"

if [ -z "${JLLM_NO_BUILD:-}" ]; then
  (cd "$ROOT" && mvn -q install -DskipTests) || { echo "engine build failed" >&2; exit 1; }
  if [ "$(uname)" = "Darwin" ] && [ ! -f "$ROOT/target/native/libjllm_metal.dylib" ]; then
    "$ROOT/scripts/build-native.sh" >&2 || echo "warning: Metal library not built, using the CPU backend" >&2
  fi
  # Maven considers the jar up to date when only the engine changed: rebuild it when the engine jar is newer
  ENGINE_JAR="$ROOT/target/jllm-serve.jar"
  if [ -f "$JAR" ] && [ -f "$ENGINE_JAR" ] && [ "$ENGINE_JAR" -nt "$JAR" ]; then
    rm -f "$JAR" "$JAR.original"
  fi
  (cd "$DIR" && mvn -q package -DskipTests) || { echo "jllm-web build failed" >&2; exit 1; }
fi

METAL_LIB="$ROOT/target/native/libjllm_metal.dylib"
if [ -f "$METAL_LIB" ]; then
  JAVA_OPTS="${JAVA_OPTS:-} -Djllm.metal.lib=$METAL_LIB"
fi

if [ -n "${JLLM_XMX:-}" ]; then
  HEAP="-Xmx$JLLM_XMX"
else
  HEAP="-XX:MaxRAMPercentage=60"
fi

# shellcheck disable=SC2086
exec "$JAVA" \
  --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -XX:+UseParallelGC $HEAP -Xmn"${JLLM_XMN:-512m}" \
  ${JAVA_OPTS:-} \
  -jar "$JAR" "$@"
