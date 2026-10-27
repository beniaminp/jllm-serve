# Sourced by ./jllm, jllm-web/run.sh and docker/entrypoint.sh (POSIX sh): the JVM option for the direct-memory limit.
#
# The CPU backend's paged KV cache (docs/24) keeps its blocks in slabs of an automatic arena, outside the Java heap, and the
# JVM limits that memory like direct buffers (-XX:MaxDirectMemorySize). Without the option the limit is the maximum heap,
# 4 GB with ./jllm's default JLLM_XMX, far below what backend.ctx=auto plans from the physical memory (docs/10 section 10.13).
# So the launchers set the limit to the physical memory (or the container's memory limit, when it is lower); the context
# budget then decides how much of it a cache really takes.
#
#   JLLM_DIRECT_MEMORY   a size for -XX:MaxDirectMemorySize (16g, 8192m ...), or "default" to leave the JVM's own limit
#                        (the heap size); unset: the physical memory as described above
jllm_direct_memory_opt() {
  case "${JLLM_DIRECT_MEMORY:-}" in
    default) return 0 ;;
    "") ;;
    *) echo "-XX:MaxDirectMemorySize=$JLLM_DIRECT_MEMORY"; return 0 ;;
  esac
  jllm_bytes=""
  if [ "$(uname)" = "Darwin" ]; then
    jllm_bytes=$(sysctl -n hw.memsize 2>/dev/null || true)
  elif [ -r /proc/meminfo ]; then
    jllm_bytes=$(awk '/^MemTotal:/ {printf "%.0f", $2 * 1024}' /proc/meminfo)
    # a container's limit (cgroup v2, then v1); "max" or the v1 "unlimited" value leave the physical memory
    for jllm_f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
      if [ -r "$jllm_f" ]; then
        jllm_lim=$(cat "$jllm_f" 2>/dev/null || true)
        case "$jllm_lim" in
          ''|max|*[!0-9]*) ;;
          *) [ -n "$jllm_bytes" ] && [ "$jllm_lim" -lt "$jllm_bytes" ] && jllm_bytes=$jllm_lim ;;
        esac
      fi
    done
  fi
  case "$jllm_bytes" in
    ''|*[!0-9]*) return 0 ;; # unknown: the JVM's default
  esac
  echo "-XX:MaxDirectMemorySize=$jllm_bytes"
}
