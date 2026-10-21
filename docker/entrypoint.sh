#!/bin/sh
# Starts the jllm web console in the container. Environment:
#   JLLM_API_KEY       the key the console and the /v1 API require (Authorization: Bearer <key>). Unset: a random key
#                      is generated on first start, kept in /data/.jllm-api-key and printed to the log. "none": no key
#                      (only when something in front of the container does authentication).
#   JLLM_HEAP_PERCENT  Java heap as a share of the container's memory limit (default 60; the CPU backend keeps its KV
#                      caches on the heap, the weights are memory-mapped outside it)
#   JLLM_AUTOLOAD      models to load at start, comma separated (as jllm.web.autoload)
#   HF_TOKEN           Hugging Face token for gated or private models
#   JLLM_EXTRA_CA_CERT a PEM file (mounted into the container) with the CA of a TLS-intercepting proxy, trusted in
#                      addition to the JDK's own CAs for downloads
#   JLLM_CPU_NATIVE    backend.cpu.native: auto (default: the image's native int8 kernels when they load and pass
#                      their self-check, else the Java kernels), true (require them), false (the Java kernels)
#   JAVA_OPTS          extra JVM options; arguments after the image name go to the console (--server.port=…, …)
set -eu
cd /data

key="${JLLM_API_KEY:-}"
if [ -z "$key" ]; then
  if [ ! -s .jllm-api-key ]; then
    (umask 077 && head -c 24 /dev/urandom | base64 | tr -d '/+=\n' > .jllm-api-key)
  fi
  key="$(cat .jllm-api-key)"
  echo "jllm: the console and the API require this key (kept in /data/.jllm-api-key):"
  echo "jllm:   $key"
elif [ "$key" = "none" ]; then
  key=""
  echo "jllm: warning: no API key (JLLM_API_KEY=none): anyone who reaches the port can download models and delete files"
fi
# passed through the environment (Spring's relaxed binding), so it does not appear in the process list
export JLLM_WEB_APIKEY="$key"
[ -n "${JLLM_AUTOLOAD:-}" ] && export JLLM_WEB_AUTOLOAD="$JLLM_AUTOLOAD"

if [ -n "${JLLM_EXTRA_CA_CERT:-}" ]; then
  # a private copy of the JDK trust store with the extra CA added (the JDK's own file is read-only for this user)
  store=/tmp/jllm-cacerts
  cp "$JAVA_HOME/lib/security/cacerts" "$store"
  chmod 600 "$store"
  # keytool takes one certificate per call: split a bundle and add each of its certificates
  dir=$(mktemp -d)
  awk -v d="$dir" '/-----BEGIN CERTIFICATE-----/ {n++} n {print > (d "/" n ".pem")}' "$JLLM_EXTRA_CA_CERT"
  for f in "$dir"/*.pem; do
    [ -e "$f" ] || { echo "jllm: no certificate in $JLLM_EXTRA_CA_CERT" >&2; exit 1; }
    keytool -importcert -noprompt -keystore "$store" -storepass changeit -alias "jllm-extra-$(basename "$f" .pem)" \
      -file "$f" >/dev/null 2>&1 || true   # already present in the JDK's store: fine
  done
  rm -rf "$dir"
  JAVA_OPTS="${JAVA_OPTS:-} -Djavax.net.ssl.trustStore=$store -Djavax.net.ssl.trustStorePassword=changeit"
fi

# the native int8 CPU kernels built for this platform (Dockerfile stage "native"); absent: the Java kernels are used
if [ -f /opt/jllm/native/libjllm_cpu.so ]; then
  JAVA_OPTS="${JAVA_OPTS:-} -Djllm.cpu.nativeLib=/opt/jllm/native/libjllm_cpu.so"
fi
if [ -n "${JLLM_CPU_NATIVE:-}" ]; then
  JAVA_OPTS="${JAVA_OPTS:-} -Djllm.cpu.native=$JLLM_CPU_NATIVE"
fi

# shellcheck disable=SC2086
exec java \
  --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -XX:+UseParallelGC -XX:MaxRAMPercentage="${JLLM_HEAP_PERCENT:-60}" -Xmn"${JLLM_XMN:-512m}" \
  ${JAVA_OPTS:-} \
  -jar /opt/jllm/jllm-web.jar --server.address=0.0.0.0 "$@"
