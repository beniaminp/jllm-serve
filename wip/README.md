# Work in progress that is not merged yet

## `q4-kv-cache.patch`: 4-bit KV cache (`backend.kv-dtype=q4`)

The diff of an unfinished branch against commit `6d32ebc`, which is where the branch started. It was stopped before
its final validation, and it conflicts with work merged since then, so it is kept here as a patch rather than merged.

**What it contains** (19 files, +2133 / −78):
- `model/KvQ4.java`: the q4 format.
  - Keys are rotated with a normalized Walsh–Hadamard transform over each head's row after RoPE. The query gets the same
    rotation inside attention: (Hq)·(Hk) = q·k. The scores are then scaled by `1 / block(headDim)`.
  - Rotated keys and values are quantized to 4 bits in groups of 32 with one scale per group.
  - An optional recent window keeps the last `backend.kv-q4-recent` positions as half in a ring buffer.
- `cpu/KvCache.java` (`KvCache.Q4`) and `cpu/KvQ4Sim.java`: the CPU cache, and a simulator for the accuracy study
  (`-Dq4.sim`, `-Dq4.rotk`, `-Dq4.symh`, `-Dq4.recent`, `-Dstudy.kv=q4`).
- Metal:
  - `kv_store_q4` plus q4 variants of the attention, split (flash decode) and flash prefill kernels, with
    `KernelParams` entries `KV_STORE_Q4`, `ATTN_Q4`, `ATTN_SPLIT_Q4` and `FLASH_Q4`;
  - `MetalKernels.Q4Ring` (the recent window);
  - `MetalBackend` save and load of the window (`saveRecent` / `loadRecent`);
  - `MetalTuner` checks of the q4 kernels.
- Tests: `ArchSmoke` q4 cases (`q8Check(..., KvDtype.Q4, recent)`), SmokeTest cases, and `kernel_tests.py` numpy
  references. The settings are in `Settings`, `jllm.properties`, the web console dropdown and `mac-validate.sh`.

**Conflicts to resolve when applying it** (`git apply --3way wip/q4-kv-cache.patch`):
- `MetalBackend` GQA step. Decode replay's fusion now stores the q8 cache in `ropeStoreQ8` (`MetalKernels.FUSE`).
  q4 needs its own path there: `kvStoreQ4` and the q4 attention kernels. Decode replay must also record and replay it,
  or explicitly fall back.
- `MetalBackend.stateFormat`. It now also returns an `mla …` format. Keep q4's `recent=` suffix for the standard cache.
- `MetalTuner.decide/check`. Main now picks between two flash prefill forms (`attention_fa`, `attention_fb`) and
  returns a `Result(use, ms)`. q4 added a `recent` parameter and a ring.
- `ArchSmoke`. Main added the replay checks and the `cfg instanceof Decoder` guards. q4 changed `q8Check`'s signature.
- `KernelParams`, `kernels.metal`, `kernel_tests.py`, `mac-validate.sh`: both sides added entries; keep both.
- Batched decoding on Metal (`forwardBatch`, `kvStoreQ8Row` and the other per-row kernels) has no q4 path yet.
  Either add one, or exclude q4 from batching.

**Not done before it was stopped:**
- the accuracy table (q8, q4 without rotation, q4 with rotation, with and without the window, against f16 on
  Qwen3-0.6B, plus a needle-in-a-haystack retrieval check);
- choosing the defaults from that data;
- the full emulator arch smoke;
- the docs (05 / 07 / 08 / 11).
