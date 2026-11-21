# Upstream sync - 2026-11-21 (b12599, 52 commits)

Merge `dac8242ed` brings master from `f8caa5ea3` to upstream `b12599` (52 commits).
Both boards (.104/.105) synced to the merge and rebuilt clean.

Notable upstream picks for this hardware: gemma4 multi-seq prompt processing
(#32342), Vulkan RMS_NORM_MUL_ROPE loop control (#31149), chunked
GATED_DELTA_NET (#30207), graph reuse, fused-op resolution, MoE expert cache
(opt-in, `--moe-cache-mib`, default off).

## Conflict resolution

- ggml op enum: keep fork `MHC_ROUTE`/`MHC_MERGE` plus upstream `SLEEP` ->
  `GGML_OP_COUNT` 102. `RPC_PROTO_PATCH_VERSION` bumped to 2; client and peer
  must run the same build (both boards rebuilt together).
- `llama_context::graph_max_nodes`: adopt upstream's model-class
  `graph_nodes_per_tensor()` refactor and MoE-cache node reserve. The
  sensenova-loop mHC budget stays as a `max()` adjustment after the base;
  DFLASH selector block scaling already lives upstream in
  `llama_model_dflash::graph_nodes_per_tensor()`.
- Deduplicate `LLM_KV_LOOP_START/END/COUNT` in `llama-arch.h`/`.cpp`: the fork
  (sensenova-loop) and upstream (inkling2-loop) added the same keys; the
  auto-merge kept both. MHC keys kept once.
- `tests/test-llama-archs.cpp`: delete obsolete `check_loop_sinkhorn`. It called
  a graph helper removed by the fused-mHC refactor; the mHC contract is covered
  by `test_mhc_route`/`test_mhc_merge` in test-backend-ops.

## Verification

Local (x64, RADV Phoenix): test-backend-ops mHC 18/18 passed;
test-llama-archs sensenova_loop NMSE 8.9e-08 (Vulkan) / 0 (CPU), roundtrip and
mixed batch OK.

.104 (gfx1013, drirc unified-heap fix active, launcher defaults):

| Test | Pre-sync | Post-sync |
|---|---|---|
| gemma4 pp512 (ub1024, q8_0 KV) | 1008 | 1002.9 |
| gemma4 pp2048 | 1094 | 1102.3 |
| gemma4 tg128 | 93.4 | 93.1 |
| LFM3-1.7B Q4_0 pp2048 (ub512) | - | 3390.8 |
| LFM3-1.7B Q4_0 tg128 | - | 233.4 |
| RPC gemma4 pp512 (ub256, layer-split) | ~770 | 758 |
| RPC gemma4 tg128 | ~47 | 47.0 |

Server greedy quality: "Capital of France is Paris." - correct.

## New upstream serving behavior (measured on .104, gemma4, 256-token warm gens)

- Graph reuse (default on): 83.3 t/s. `LLAMA_GRAPH_REUSE_DISABLE=1`: 54.6 t/s.
  Keep default.
- `fused_kv_rows` auto-probes when KV is unified (`-kvu 1`). The fused variant
  is unsupported on gfx1013 and resolve_fused_ops disables it (warning at
  load). `LLAMA_KV_ROWS_DISABLE=1` measured slower (75.8 t/s) than default
  (83.3 t/s). Keep default.
- Server tg, same method (256-token warm greedy gens): post-sync 82.6 / 83.3
  t/s; pre-sync code (5c2ea2d1b, worktree build on .104) 76.6 / 80.0 t/s.
  The sync improves serving ~6%; bench tg unchanged (93.1). The 88.2 in the
  release notes used a different, unrecorded measurement and is not comparable.

## Known issues

Unchanged from release-verification-2026-11-20: the 15 GB shared-memory ceiling
(Qwen3.5-35B Q5 over RPC, ub1024 at >=32k context) is hardware, not code; use Q4
weight quants over RPC and ub512 / q4_0 KV for long context.
