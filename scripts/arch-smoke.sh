#!/usr/bin/env bash
# Architecture smoke test: Llama / Mistral / Qwen2 / Qwen3-MoE / GLM / Phi-3 / Gemma / Mamba / DeepSeek / Qwen3-Next / gpt-oss / Llama 4 pipeline on tiny random checkpoints.
#
# For every model it converts to q4, q8, q2k and q6k (plus q4 with a q6k embedding / LM head), loads, generates 8 tokens greedily
# on the CPU backend and on the Metal backend, and requires the two id sequences to be equal.
#
# Models (under $ARCH_MODELS, default /tmp/jllm-arch-models):
#   hf-internal-testing_tiny-random-LlamaForCausalLM      downloaded from the Hub (4 MB, random weights);
#                                                         hidden_size 16: not a multiple of the 32-wide quantization
#                                                         block, so the engine reports it as unsupported (skipped)
#   hf-internal-testing_tiny-random-LlamaPlainForCausalLM generated: plain Llama, GQA, untied head
#   hf-internal-testing_tiny-random-MistralForCausalLM    generated: the Hub repo only has pytorch_model.bin and a
#                                                         gelu config, so a silu Mistral config with
#                                                         sliding_window=4 and fresh random weights is written
#   hf-internal-testing_tiny-random-Qwen2ForCausalLM      generated: the repo does not exist on the Hub; q/k/v bias,
#                                                         tied embeddings, sliding window on layers >= 2
#   hf-internal-testing_tiny-random-Llama3RopeForCausalLM generated: llama3 rope_scaling, MHA, tied embeddings
#   hf-internal-testing_tiny-random-LlamaBiasForCausalLM  generated: plain Llama with attention_bias (q/k/v and o_proj
#                                                         biases; o_proj's accumulates into the residual)
#   hf-internal-testing_tiny-random-Phi3ForCausalLM       generated: fused qkv_proj / gate_up_proj, longrope (long factors
#                                                         at ctx 64), partial_rotary_factor 0.75, sliding window 6
#   hf-internal-testing_tiny-random-Phi3ShortForCausalLM  generated: as above with the short factors, tied embeddings
#   hf-internal-testing_tiny-random-Phi3MhaForCausalLM    generated: headDim 64, MHA, plain RoPE, sliding window 6
#   hf-internal-testing_tiny-random-Gemma2ForCausalLM     generated: head_dim 64, (1+w) sandwich norms, gelu_tanh,
#                                                         attention + final soft-capping, local/global alternation
#   hf-internal-testing_tiny-random-Gemma3ForCausalLM     generated: q/k norm, 5 local + 1 global layers, two RoPE bases
#   hf-internal-testing_tiny-random-Gemma3MMForCausalLM   generated: the same decoder in the multimodal wrapper layout
#   hf-internal-testing_tiny-random-Gemma{2,3}Hd256ForCausalLM  generated: head_dim 256, the largest Metal supports
#   hf-internal-testing_tiny-random-Gemma2BiasForCausalLM generated: Gemma 2 with attention_bias (the o_proj bias
#                                                         before the post-attention norm)
#   hf-internal-testing_tiny-random-MambaForCausalLM      generated: Mamba v1, d_model 64, state 16, conv 4, tied head
#   hf-internal-testing_tiny-random-Mamba3ForCausalLM     generated: Mamba v1, d_model 128, state 8, conv 3, untied head
#                                                         (Mamba models additionally check that chunking the prompt
#                                                         differently gives the same logits)
#   hf-internal-testing_tiny-random-Qwen3MoeForCausalLM   generated: q/k norm, 8 experts top 2, one dense layer; MoE models
#                                                         also run with their experts on the CPU (backend.metal.cpu-moe)
#   ...-DeepseekV2ForCausalLM / -DeepseekV2QLoraForCausalLM / -DeepseekV3ForCausalLM  generated: MLA + shared experts +
#                                                         yarn, group-limited softmax routing (q_lora null / set) and
#                                                         sigmoid noaux_tc routing; also checked against deepseek-ref.py
#
#   ...-Qwen3NextForCausalLM / -Qwen3Next2ForCausalLM    generated: Qwen3-Next hybrids (Gated DeltaNet + gated attention +
#                                                         MoE with a gated shared expert); checked against qwen3next-ref.py
#                                                         (Qwen3NextChecks: reference, chunked vs token loop, saved states)
#                                                         and, like Mamba, for chunking invariance
#   ...-GptOssForCausalLM / -GptOssMxfp4ForCausalLM       generated: gpt-oss (alternating sliding-window / full attention, attention
#                                                         sinks, YaRN without truncation, 8 experts top 2 with a biased router,
#                                                         expert biases and the clamped SwiGLU), the experts as float tensors or
#                                                         as MXFP4 blocks (read in place on both backends); checked against
#                                                         transformers by hf-arch-check.sh (see the end of this script)
#   ...-Llama4ForCausalLM / -Llama4MixForCausalLM         generated: Llama 4 text (chunked attention on 3 of 4 layers, a NoPE layer
#                                                         with attention temperature tuning, L2 q/k norm, interleaved llama3 RoPE,
#                                                         top-1 sigmoid MoE with a shared expert; the second is dense / MoE
#                                                         alternating in the multimodal wrapper layout)
#   hf-internal-testing_tiny-random-GlmForCausalLM        generated: glm, q/k/v bias, fused gate_up, partial (0.5) interleaved rope
#   hf-internal-testing_tiny-random-GlmEdgeForCausalLM    generated: glm, head_dim 64, full interleaved rope, tied
#   hf-internal-testing_tiny-random-Glm4ForCausalLM       generated: glm4, four norms per layer, partial interleaved rope
#   hf-internal-testing_tiny-random-LlamaDraftForCausalLM generated: the plain Llama's shapes with other random weights
#                                                         (another seed), the draft model of its speculation check
#
# With Metal, every attention model (Q4_0) also checks batched decoding across slots (three backends in shared passes vs.
# alone, f32 / f16 / q8 caches, a three-thread batch group) and draft-model speculation through the engine (greedy output
# with a draft equals the output without; drafts: the model itself, its Q2_K quantization, and for the plain Llama the
# other-seed model). The engine needs a tokenizer: models/Qwen_Qwen3-0.6B/tokenizer.json (-Djllm.smoke.tokenizer).
#
# usage: scripts/arch-smoke.sh          (needs JDK 26: JAVA_HOME or /opt/homebrew/opt/openjdk@26, python3, curl)
#        METAL_EMU=1 scripts/arch-smoke.sh   on any OS: the Metal side runs on the CPU emulator of the kernels
#                                            (scripts/build-metal-emu.sh, needs clang++), so no Apple GPU is needed
# The half-weight MMA prefill kernels (*_hmma, docs/08 8.21) are exact for Q4_0 / Q8_0 (the scale is applied in f32), so
# the whole run uses the tuner's per-shape choice (hmma=auto); the last pass forces them on every dense and MoE prefill
# matmul (-Djllm.metal.hmma=true, MMA from 2 tokens on) for the Q4_0 / Q8_0 models. The lossy K-quant variants are opt-in
# (-Djllm.metal.hmmaKQuants=true) and not part of this test; scripts/metal-emu/kernel_tests.py checks all of them.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x /opt/homebrew/opt/openjdk@26/bin/java ] && export JAVA_HOME=/opt/homebrew/opt/openjdk@26
export JAVA_HOME="${JAVA_HOME:?set JAVA_HOME to a JDK 26}"
M="${ARCH_MODELS:-/tmp/jllm-arch-models}"
P=hf-internal-testing_tiny-random

mvn -q compile test-compile
if [ -n "${METAL_EMU:-}" ]; then
  METAL_LIB="$(./scripts/build-metal-emu.sh 2>/dev/null | tail -1)"
  JAVA_OPTS="${JAVA_OPTS:-} -Djllm.metal.emulator=true"
else
  ./scripts/build-native.sh >/dev/null
  METAL_LIB=target/native/libjllm_metal.dylib
fi

LLAMA="$M/${P}-LlamaForCausalLM"
if [ ! -f "$LLAMA/model.safetensors" ]; then
  mkdir -p "$LLAMA"
  for f in config.json generation_config.json model.safetensors; do
    curl -sfL -o "$LLAMA/$f" "https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/main/$f"
  done
fi

gen() { # family dir-suffix
  local d="$M/${P}-$2"
  rm -rf "$d"
  python3 scripts/make-tiny-model.py "$d" "$1" "${@:3}" >/dev/null
}
gen llama LlamaPlainForCausalLM
gen mistral MistralForCausalLM
gen qwen2 Qwen2ForCausalLM
gen llama3rope Llama3RopeForCausalLM
gen llama LlamaBiasForCausalLM --attention-bias
gen glm GlmForCausalLM
gen glmedge GlmEdgeForCausalLM
gen glm4 Glm4ForCausalLM
gen phi3 Phi3ForCausalLM
gen phi3short Phi3ShortForCausalLM
gen phi3mha Phi3MhaForCausalLM
gen gemma2 Gemma2ForCausalLM
gen gemma3 Gemma3ForCausalLM
gen gemma3mm Gemma3MMForCausalLM
gen gemma2 Gemma2Hd256ForCausalLM --head-dim 256
gen gemma3 Gemma3Hd256ForCausalLM --head-dim 256
gen gemma2 Gemma2BiasForCausalLM --attention-bias
gen mamba MambaForCausalLM
gen mamba3 Mamba3ForCausalLM
gen qwen3moe Qwen3MoeForCausalLM
gen deepseek_v2 DeepseekV2ForCausalLM
gen deepseek_v2q DeepseekV2QLoraForCausalLM
gen deepseek_v3 DeepseekV3ForCausalLM
gen qwen3next Qwen3NextForCausalLM
gen qwen3next2 Qwen3Next2ForCausalLM
gen gptoss GptOssForCausalLM
gen gptoss_mxfp4 GptOssMxfp4ForCausalLM
gen llama4 Llama4ForCausalLM
gen llama4mix Llama4MixForCausalLM
gen llama LlamaDraftForCausalLM --seed 4321

# DeepSeek: engine vs. the independent float64 reference (scripts/deepseek-ref.py) on Q8_0-rounded weights
for n in DeepseekV2ForCausalLM DeepseekV2QLoraForCausalLM DeepseekV3ForCausalLM; do
  REF_Q8=1 python3 scripts/deepseek-ref.py "$M/${P}-$n" >/dev/null
done
"$JAVA_HOME/bin/java" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx4g -Djllm.cpu.fastmath=false ${JAVA_OPTS:-} \
  -cp target/classes:target/test-classes org.robomiri.jllm.DeepSeekRef "$METAL_LIB" \
  "$M/${P}-DeepseekV2ForCausalLM" "$M/${P}-DeepseekV2QLoraForCausalLM" "$M/${P}-DeepseekV3ForCausalLM"

# saved / copied / partially loaded caches, standard and latent (MLA), CPU and Metal; the engine checks use the
# Qwen3 tokenizer when models/Qwen_Qwen3-0.6B is there
"$JAVA_HOME/bin/java" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx4g -Djllm.cpu.fastmath=false ${JAVA_OPTS:-} \
  -cp target/classes:target/test-classes org.robomiri.jllm.KvStateTest models/Qwen_Qwen3-0.6B --metal "$METAL_LIB" \
  "$M/${P}-LlamaPlainForCausalLM" "$M/${P}-DeepseekV2ForCausalLM" "$M/${P}-DeepseekV2QLoraForCausalLM"
# Qwen3-Next: engine vs. the independent float64 reference (scripts/qwen3next-ref.py, needs numpy) on Q8_0-rounded
# weights, the chunked delta rule vs. the token loop, and exact saved / loaded / copied states
for n in Qwen3NextForCausalLM Qwen3Next2ForCausalLM; do
  REF_Q8=1 python3 scripts/qwen3next-ref.py "$M/${P}-$n" >/dev/null
done
"$JAVA_HOME/bin/java" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx4g -Djllm.cpu.fastmath=false ${JAVA_OPTS:-} \
  -cp target/classes:target/test-classes org.robomiri.jllm.Qwen3NextChecks "$METAL_LIB" \
  "$M/${P}-Qwen3NextForCausalLM" "$M/${P}-Qwen3Next2ForCausalLM"

# Qwen3.5 / Qwen3-Next as GGUF (tiny hybrids written the way llama.cpp's converter writes them): GGUF vs the converted
# checkpoint, CPU == Metal, the multi-token-prediction head and its speculation (output == plain decoding)
"$JAVA_HOME/bin/java" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx4g -Djllm.cpu.fastmath=false ${JAVA_OPTS:-} \
  -Djllm.metal.lib="$METAL_LIB" -cp target/classes:target/test-classes org.robomiri.jllm.GgufQwenNextTest

"$JAVA_HOME/bin/java" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx4g -Djllm.cpu.fastmath=false \
  -Djllm.metal.expertCountEvery=1 -Djllm.metal.cpuMoeSyncUs=0 \
  -Djllm.smoke.draftPair="${P}-LlamaPlainForCausalLM=$M/${P}-LlamaDraftForCausalLM" ${JAVA_OPTS:-} \
  -cp target/classes:target/test-classes org.robomiri.jllm.ArchSmoke "$METAL_LIB" \
  "$LLAMA" "$M/${P}-LlamaPlainForCausalLM" "$M/${P}-MistralForCausalLM" "$M/${P}-Qwen2ForCausalLM" "$M/${P}-Llama3RopeForCausalLM" \
  "$M/${P}-LlamaBiasForCausalLM" "$M/${P}-Gemma2BiasForCausalLM" \
  "$M/${P}-GlmForCausalLM" "$M/${P}-GlmEdgeForCausalLM" "$M/${P}-Glm4ForCausalLM" \
  "$M/${P}-Phi3ForCausalLM" "$M/${P}-Phi3ShortForCausalLM" "$M/${P}-Phi3MhaForCausalLM" \
  "$M/${P}-Gemma2ForCausalLM" "$M/${P}-Gemma3ForCausalLM" "$M/${P}-Gemma3MMForCausalLM" \
  "$M/${P}-Gemma2Hd256ForCausalLM" "$M/${P}-Gemma3Hd256ForCausalLM" \
  "$M/${P}-MambaForCausalLM" "$M/${P}-Mamba3ForCausalLM" "$M/${P}-Qwen3MoeForCausalLM" \
  "$M/${P}-DeepseekV2ForCausalLM" "$M/${P}-DeepseekV2QLoraForCausalLM" "$M/${P}-DeepseekV3ForCausalLM" \
  "$M/${P}-Qwen3NextForCausalLM" "$M/${P}-Qwen3Next2ForCausalLM" \
  "$M/${P}-GptOssForCausalLM" "$M/${P}-GptOssMxfp4ForCausalLM" "$M/${P}-Llama4ForCausalLM" "$M/${P}-Llama4MixForCausalLM"

# gpt-oss and Llama 4 against Hugging Face transformers (tiny models of the real configs' shape, logits and greedy ids at
# every position, CPU and Metal): only where a python with torch + transformers is named, e.g.
#   REF_PYTHON=/path/to/venv/bin/python scripts/arch-smoke.sh
if [ -n "${REF_PYTHON:-}" ]; then
  REF_WORK="${REF_WORK:-$M/hf}" METAL_EMU="${METAL_EMU:-}" ./scripts/hf-arch-check.sh
fi

# Second pass: every dense and MoE prefill matmul through the half-weight MMA kernels (hmma=true, MMA from 2 tokens
# on instead of 32 / 16), for the formats where they are exact (Q4_0, Q8_0): the same bounds as above must hold.
# (Not Phi3: with MMA prefill from 2 tokens its q8-cache replay x batch comparison flips an int8 rounding step, 4e-2 with
# the f32 MMA kernels as well, 3e-3 with the half-weight ones; its f32 / f16 caches and every other model pass.)
# The q4-cache cases are left out of this pass (-Djllm.smoke.q4=false) for the same reason: a batched pass of 3 rows and a
# lone step take different matmul kernels here, and a 4-bit code flips (4e-3 on Phi3Mha); the first pass checks them all.
exec "$JAVA_HOME/bin/java" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx4g -Djllm.cpu.fastmath=false \
  -Djllm.metal.hmma=true -Djllm.metal.mmaMinTokens=2 -Djllm.metal.moeMmaMin=2 -Djllm.smoke.quants=Q4_0,Q8_0 -Djllm.smoke.q4=false \
  -Djllm.metal.expertCountEvery=1 -Djllm.metal.cpuMoeSyncUs=0 ${JAVA_OPTS:-} \
  -cp target/classes:target/test-classes org.robomiri.jllm.ArchSmoke "$METAL_LIB" \
  "$LLAMA" "$M/${P}-LlamaPlainForCausalLM" "$M/${P}-Qwen2ForCausalLM" "$M/${P}-Gemma3ForCausalLM" "$M/${P}-Phi3MhaForCausalLM" \
  "$M/${P}-Qwen3MoeForCausalLM" "$M/${P}-DeepseekV3ForCausalLM" "$M/${P}-Qwen3NextForCausalLM" \
  "$M/${P}-GptOssForCausalLM" "$M/${P}-GptOssMxfp4ForCausalLM" "$M/${P}-Llama4ForCausalLM"
