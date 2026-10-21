# jllm-serve web console in a container: the engine and jllm-web, CPU backend (a container cannot reach Apple's Metal
# GPU, also not under Docker Desktop on a Mac: run ./jllm-web/run.sh natively there for the GPU).
#
#   docker build -t jllm-serve .                       (or scripts/docker-build.sh, also for multi-arch + push)
#   docker run -d -p 8090:8090 -v jllm-data:/data --name jllm jllm-serve
#   docker logs jllm        # prints the API key the console asks for on first use
#   open http://localhost:8090/
#
# Everything the console writes (downloaded models, converted caches, saved conversations, settings, tools) goes to
# the /data volume. See docs/19-docker.md.

# ---------------------------------------------------------------- build: Java is architecture independent, so the
# jars are built once on the build machine's platform and copied into each target platform's image
FROM --platform=$BUILDPLATFORM maven:3-eclipse-temurin-26 AS build
WORKDIR /src
# Optional, for builds behind a TLS-intercepting proxy: `--secret id=extra_ca,src=<pem file>` adds that CA to the
# build's JDK trust store. Nothing of it reaches the final image.
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then \
      d=$(mktemp -d); \
      awk -v d="$d" '/-----BEGIN CERTIFICATE-----/ {n++} n {print > (d "/" n ".pem")}' /run/secrets/extra_ca; \
      for f in "$d"/*.pem; do \
        keytool -importcert -noprompt -cacerts -storepass changeit -alias "extra-$(basename "$f" .pem)" -file "$f" \
          >/dev/null 2>&1 || true; \
      done; \
    fi
# Maven ignores the HTTP(S)_PROXY variables Docker passes to builds: maven-proxy turns them into a settings file
COPY docker/maven-proxy.sh /usr/local/bin/maven-proxy
COPY pom.xml .
COPY jllm-web/pom.xml jllm-web/pom.xml
RUN --mount=type=cache,target=/root/.m2 \
    mvn -q -B $(maven-proxy) dependency:go-offline -DskipTests >/dev/null 2>&1 || true
COPY src src
COPY jllm-web/src jllm-web/src
RUN --mount=type=cache,target=/root/.m2 \
    mvn -q -B $(maven-proxy) install -DskipTests \
    && cd jllm-web && mvn -q -B $(maven-proxy) package -DskipTests \
    && cp target/jllm-web.jar /jllm-web.jar

# ---------------------------------------------------------------- native int8 CPU kernels (libjllm_cpu.so): C, so
# compiled for each TARGET platform (a stage without --platform runs on the target platform: under emulation in a
# cross-platform build, like the runtime stage's apt-get; it is a few seconds of gcc). On the runtime image's base, so
# it links against the same C library. One binary covers all SIMD levels of its architecture (x86-64: AVX2 / AVX-VNNI
# / AVX512-VNNI, arm64: NEON / dotprod). If it does not build, /out stays empty and the engine uses its Java kernels.
FROM eclipse-temurin:26-jdk AS native
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY scripts/build-native-cpu.sh scripts/build-native-cpu.sh
COPY src/main/native src/main/native
RUN mkdir -p /out && CC=gcc OUT=/out/libjllm_cpu.so sh scripts/build-native-cpu.sh --optional

# ---------------------------------------------------------------- runtime
FROM eclipse-temurin:26-jdk
# the JDK image (not the JRE) because the engine needs the jdk.incubator.vector module; curl for the health check
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 jllm \
    && mkdir -p /data /opt/jllm && chown jllm:jllm /data
COPY --from=build /jllm-web.jar /opt/jllm/jllm-web.jar
# the native kernels, when they were built (entrypoint.sh passes their path; without them the Java kernels run)
COPY --from=native /out/ /opt/jllm/native/
COPY docker/entrypoint.sh /opt/jllm/entrypoint.sh
USER jllm
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8090
ENV JLLM_HEAP_PERCENT=60
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8090/ >/dev/null || exit 1
ENTRYPOINT ["/opt/jllm/entrypoint.sh"]
