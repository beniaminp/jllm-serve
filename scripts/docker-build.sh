#!/bin/sh
# Builds the jllm-serve Docker image (engine + web console, CPU backend), optionally for several platforms, and
# optionally pushes it to a registry such as Docker Hub.
#
#   scripts/docker-build.sh                                    local image jllm-serve:latest for this machine
#   scripts/docker-build.sh -t you/jllm-serve:1.0              another name
#   scripts/docker-build.sh -t you/jllm-serve:1.0 --push       linux/amd64 + linux/arm64, pushed (docker login first)
#   scripts/docker-build.sh --platform linux/arm64 -t x:y --push
#
# Multi-platform images are built with docker buildx (a docker-container builder is created once, named jllm);
# the Java build runs once on this machine's platform and is copied into each platform's image, so no emulation is
# needed for it; the small C library of native CPU kernels (libjllm_cpu.so) is compiled per target platform. Proxy variables (HTTP_PROXY, HTTPS_PROXY, NO_PROXY) are passed to the build; EXTRA_CA_CERT=<pem>
# adds a CA for a TLS-intercepting proxy to the build only; BUILDX_CREATE_OPTS is passed to `docker buildx create`.
set -eu
cd "$(dirname "$0")/.."
TAG=jllm-serve:latest
PUSH=""
PLATFORMS=""
EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    -t|--tag) TAG="$2"; shift 2 ;;
    --push) PUSH=1; shift ;;
    --platform) PLATFORMS="$2"; shift 2 ;;
    --network) EXTRA="$EXTRA --network $2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$PUSH" ] && [ -z "$PLATFORMS" ] && PLATFORMS=linux/amd64,linux/arm64

ARGS=""
for v in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  eval "val=\${$v:-}"
  [ -n "$val" ] && ARGS="$ARGS --build-arg $v=$val"
done
[ -n "${EXTRA_CA_CERT:-}" ] && ARGS="$ARGS --secret id=extra_ca,src=$EXTRA_CA_CERT"

if [ -z "$PLATFORMS" ]; then
  # shellcheck disable=SC2086
  DOCKER_BUILDKIT=1 docker build $ARGS $EXTRA -t "$TAG" .
  echo "built $TAG; run: docker run -d -p 8090:8090 -v jllm-data:/data --name jllm $TAG"
  exit 0
fi

# BUILDX_CREATE_OPTS: extra options for creating the builder, e.g. "--buildkitd-config <toml>" with a registry CA
# for a TLS-intercepting proxy, or "--driver-opt network=host"
# shellcheck disable=SC2086
docker buildx inspect jllm >/dev/null 2>&1 || docker buildx create --name jllm --driver docker-container ${BUILDX_CREATE_OPTS:-} >/dev/null
OUT="--load"
[ -n "$PUSH" ] && OUT="--push"
case "$PLATFORMS" in *,*) [ -z "$PUSH" ] && { echo "several platforms need --push (a local image holds one)" >&2; exit 2; } ;; esac
# shellcheck disable=SC2086
docker buildx build --builder jllm --platform "$PLATFORMS" $ARGS $EXTRA $OUT -t "$TAG" .
echo "built $TAG for $PLATFORMS${PUSH:+ and pushed it}"
