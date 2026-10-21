# 19. Docker: one image with everything

The image contains the engine and the web console (`jllm-web`): load models from Hugging Face, chat, manage
settings, saved conversations, tools and integrations, and the OpenAI-compatible API, all from a browser.

| file | what it is |
|---|---|
| `Dockerfile` | multi-stage build: compiles the engine and the console, runtime on a JDK 26 image |
| `docker/entrypoint.sh` | starts the console in the container: data in `/data`, API key, heap size, extra CA |
| `docker/maven-proxy.sh` | lets the build's Maven use `HTTP(S)_PROXY` |
| `scripts/docker-build.sh` | builds for this machine, or for `linux/amd64` + `linux/arm64` and pushes |
| `docker-compose.yml` | the same as a compose service |

> **CPU only.** A container cannot use Apple's Metal GPU, not even under Docker Desktop on a Mac (it runs a Linux
> VM). The image always runs the CPU backend (Vector API: AVX2 / AVX-512 on x86, NEON on ARM). On a Mac, run
> `./jllm-web/run.sh` natively to get the GPU ([README](../README.md#web-console-jllm-web)). There is no NVIDIA
> backend.

---

## 19.1 For people who use the image

```sh
docker pull <you>/jllm-serve:latest
docker run -d --name jllm -p 8090:8090 -v jllm-data:/data <you>/jllm-serve:latest
docker logs jllm          # shows the API key, e.g.  jllm:   lAWvUZuXJuqcfTTEF7OP57WEJ5jcymb
```

Open <http://localhost:8090/>. The console asks for the key once and keeps it in the browser. From the
**Models** tab, load a model (`Qwen/Qwen3-0.6B` is a good first one: 1.5 GB to download, converted once). Then chat
in the **Chat** tab, or point any OpenAI client at `http://localhost:8090/v1` with the key as its API key.

Everything the console writes goes to the `/data` volume, so it survives restarts and upgrades:

```
/data/models/<owner>_<name>/   downloads and their converted, memory-mapped caches
/data/kv-cache/                saved conversations (so long prompts are not processed again after a restart)
/data/jllm-web.properties      engine settings changed in the console
/data/jllm-web-tools.json      tools and MCP servers
/data/.jllm-api-key            the generated API key
```

Upgrade: `docker pull` the new image, `docker rm -f jllm`, and run it again with the same volume.

### Settings (environment variables)

| variable | default | |
|---|---|---|
| `JLLM_API_KEY` | generated | the key the console and `/v1` require. Unset: generated on first start, kept in `/data/.jllm-api-key`, printed to the log. `none`: no key (only behind your own authentication) |
| `JLLM_HEAP_PERCENT` | 60 | Java heap as a share of the container's memory limit |
| `JLLM_AUTOLOAD` | | models to load at start, comma separated (`Qwen/Qwen3-0.6B,unsloth/Qwen3-4B-GGUF:Q4_K_M`) |
| `HF_TOKEN` | | Hugging Face token for gated or private models |
| `JLLM_EXTRA_CA_CERT` | | path (inside the container) of a PEM file with the CA of a TLS-intercepting proxy |
| `HTTPS_PROXY` | | not read by Java; add `JAVA_OPTS="-Dhttps.proxyHost=… -Dhttps.proxyPort=…"` instead |
| `JLLM_CPU_NATIVE` | auto | `backend.cpu.native` when the console's settings leave it unset: `auto` uses the image's native int8 CPU kernels (`/opt/jllm/native/libjllm_cpu.so`) when they load and pass their self-check, `true` requires them, `false` uses the Java kernels |
| `JAVA_OPTS` | | extra JVM options |

Arguments after the image name go to the console, e.g. `… <you>/jllm-serve --server.port=9000`.

### Memory

🧸 The model's weights are a big book on a shelf. They aren't copied into the program's memory; the program
reads the pages it needs, and the operating system keeps recently used pages in RAM. What the program does keep in
its own memory is its notes about the conversation, the KV cache.

📐 The weights are memory-mapped (page cache, outside the Java heap), while the CPU backend's KV caches live on the
heap (one per slot). Give the container at least

```
memory ≥ weights file  +  heap,     heap = JLLM_HEAP_PERCENT % of the limit  ≥  slots · KV cache
KV cache = layers · ctx · kvDim · 2 · (4 bytes f32 | 2 bytes f16)
```

For example, Qwen3-0.6B Q4_0 has 452 MB of weights. At `--ctx 4096` with an f32 cache its KV cache is
28 · 4096 · 1024 · 2 · 4 B = 0.94 GB, so `--memory 4g` is comfortable. A 30B-A3B MoE at Q4_0 (19.4 GB) needs about
24 GB. `docker run --memory …` sets the limit and the heap follows it.

### Security

The management API can download models and delete files. Keep the key, or publish the port only to this machine
with `-p 127.0.0.1:8090:8090`. For access from other machines put a TLS reverse proxy in front: the console speaks
plain HTTP.

---

## 19.2 For you: build and push

```sh
scripts/docker-build.sh                                   # jllm-serve:latest for this machine
docker run -d -p 8090:8090 -v jllm-data:/data jllm-serve  # try it

docker login                                              # Docker Hub account
scripts/docker-build.sh -t <you>/jllm-serve:1.0 --push    # linux/amd64 + linux/arm64, pushed
scripts/docker-build.sh -t <you>/jllm-serve:latest --push
```

Multi-platform images are built with `docker buildx`; the script creates a builder named `jllm` once. Java
bytecode is the same on every platform, so the build stage runs once on the build machine's own platform
(`FROM --platform=$BUILDPLATFORM`) and its jar is copied into each platform's runtime image. Only that runtime
stage's `apt-get` runs under emulation for the foreign platform (Docker Desktop has QEMU built in; on Linux run
`docker run --privileged --rm tonistiigi/binfmt --install arm64` once).

The one piece of C, the native int8 CPU kernels `libjllm_cpu.so` ([07 §7.16](07-cpu-backend.md#716-native-int8-kernels-libjllm_cpu-nativecpu)),
is built in its own stage, `native`, for each target platform: that stage has no `--platform`, so like the runtime
stage it runs on the target platform (emulated in a cross-platform build; it is one gcc call of a few seconds). It
starts from the runtime image's base, so the library links against the same C library, and one binary covers every
SIMD level of its architecture (x86-64: AVX2, AVX-VNNI, AVX512-VNNI; arm64: NEON, dotprod), chosen when it loads. The
runtime image copies it to `/opt/jllm/native/` and `entrypoint.sh` passes `-Djllm.cpu.nativeLib` when the file is
there. If the stage cannot build it (`build-native-cpu.sh --optional` exits 0 without output), the image still works
with the Java kernels; the log says which kernels run (`native CPU kernels: avx512-vnni (...)` or `... using the Java
kernels`), and so does the backend name.

Behind a TLS-intercepting proxy:

```sh
EXTRA_CA_CERT=/path/ca.pem scripts/docker-build.sh …        # the build's Maven trusts it (the image does not)
BUILDX_CREATE_OPTS="--buildkitd-config buildkitd.toml" …   # registry CA for the multi-platform builder
```

`docker compose up -d` builds and runs the same with the settings of `docker-compose.yml`.

### What was verified

- **amd64:** the image was built and run on a 4-core x86 Linux machine. It became healthy, printed and kept its
  generated key across a restart, and rejected requests without the key (401). It loaded Qwen3-0.6B from the
  volume (CPU backend, AVX-512) and answered a chat completion through `/v1/chat/completions`.
- **native kernels (amd64):** the `native` stage alone (`docker buildx build --target native`) compiled
  `libjllm_cpu.so` in 4.4 s, and `NativeCpuCheck` run inside that image against it passed (678 checks, levels
  scalar / avx2 / avx512-vnni). The full image with this stage has not been rebuilt here. For arm64 the same C code
  was cross-compiled and run under qemu-aarch64 (`src/test/native/jllm_cpu_levels.c`, NEON and dotprod against
  scalar), but not as a Docker build.
- **arm64:** the image was built (650 MB amd64, 673 MB arm64) but could not be run on that machine. Run it on an
  ARM machine or a Mac before publishing.
