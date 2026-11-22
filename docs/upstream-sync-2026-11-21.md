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

---

# Evening sync (b12614) + Inkling-2 RPC correctness/perf fixes

Merge: origin/master b12614, 2 commits (afe77e18f chat reasoning_effort=none
#32763, e6d624b9a mtmd Parakeet rel-pos OOB #32722). Zero conflicts; no fork
delta touched. Guide tests pass: MHC_ROUTE/MHC_MERGE 18/18, test-llama-archs
sensenova_loop OK. Both boards rebuilt from the same commit chain.

## New commits

- `107b9249a` llama: `LLAMA_PRE_SPLIT_DISABLE` gate. Pre-split before reserve
  over-reserves compute on split-heavy graphs (inkling2_loop: 433 splits,
  11.6 GiB compute projection vs 1.8 GiB lazy); the peer was OOM-killed during
  reserve. Gate skips pre-split in both reserve branches.
- `ca672f8cd` llama: loop-slot KV/conv-state device follows the physical
  weights. `llama_model::dev_layer(il)` maps through virtual `layer_phys(il)`;
  inkling2_loop slots 24..47 reuse layers 6..17, and slot-index placement put
  their KV on a different device than the compute. RPC tg 2.2 -> 5.9.
- `434b08557` rpc: forward `supports_op` queries to the peer
  (`RPC_CMD_SUPPORTS_OP`, client cache keyed by op signature, proto patch
  2 -> 3, both boards rebuilt together). ROOT CAUSE of the RPC garbage output:
  `ggml_backend_rpc_device_supports_op` returned true unconditionally
  (upstream TODO); the scheduler assigned ops the peer Vulkan cannot execute
  and the peer silently miscomputed them. Evidence chain: weights
  byte-identical on the peer (FNV-1a stream hashes 49/49), op kernels pass on
  both boards (MUL_MAT_ID 1180/1180 incl. IQ3_S, TOP_K, SUM_ROWS/SQR/SQRT),
  dense LFM3 coherent over the same RPC path, inkling2 garbage even all-peer
  (-ts 100,0, 2 splits); coherent after the fix.
- `1d7113e9b` llama: skip pipeline parallelism when an RPC backend is present.
  It auto-enables for multi-device layer-split and sets n_copies=4,
  quadrupling split-input copies - ruinous on 1GbE.
- `a062dd502` rpc: log forwarded supports_op answers at GGML_RPC_DEBUG=2.
- `5e5baa3ac` llama: `ggml_cont` on SITU gate/up. Vulkan SCALE requires
  contiguous src0 (scale.comp uses flat indexing); the strided gate/up views
  of the merged gate_up mul_mat_id output fell back to CPU per layer, and over
  RPC each fallback copied the full gate_up tensor peer->client (~80 MB/token).
  Q6_K -ts 25,75 ub128: pp 5.3 -> 13.1, tg 4.8 -> 9.8.

## Inkling-2-Mini results (.104 client, .105 peer, 1GbE)

| config | pp t/s | tg t/s | quality |
|---|---|---|---|
| AD-Q6_K 22 GB RPC -ts 25,75 ub128 c16384 q8_0 KV | 13.1 | 9.8 | Paris, reasoning, tool_calls correct |
| AD-IQ3_S 13 GB single-board ub128 c16384 | 41.0 | 17.6 | coherent |
| AD-IQ3_S RPC all-peer ub128 (before cont fix) | 30.5 | 23.9 | coherent after supports_op fix |

Memory: Q6_K -ts 25,75 splits 12/12 physical layers (peer 11.77 GiB weights,
client 10.43 GiB + embd); -ts 50,50 puts ALL 24 physical layers on RPC0
(slots 0..23 cover every physical layer) and overflows; -ts 12.5,87.5 leaves
18 layers + embd/out on the client (16.5 GiB) and overflows there. ub512 with
-ts 25,75 OOM-killed the peer at reserve (compute buffers); ub128/256 fit.

Grammar: the jinja tool-call grammar works after the correctness fix
(`get_weather({"city":"Paris"})`, finish_reason=tool_calls). The earlier
"Unexpected empty grammar stack" exception was a symptom of garbage output
hitting the lazy tool-call grammar, not a grammar bug.

## System changes on .105 (root)

- klogd added to the default runlevel: peer OOM-kills were invisible before
  (0 kernel lines in /var/log/messages); now captured.
- /etc/drirc created (radv_enable_unified_heap_on_apu=false): HOST_CACHED
  memory types 0 -> 3, mirrors .104. Perf only.
- .105 fell back to the r2 kernel during the earlier crash loop (bootguard).
  r2 OOM-kills cleanly; r3 sets panic_on_oops=1/hardlockup_panic=1/panic=60
  and reboots on GPU faults. The original "instant disconnects" were peer
  overcommit (21.7 GiB on one board with default -ts) -> OOM/panic.

## Diagnostics that worked

- GGML_SCHED_DEBUG=2 + -lv 5: per-node backend dump. Assignment causes need
  the temporary `#if 0` -> `#if 1` at ggml-backend.cpp:1327 (revert after).
- GGML_RPC_DEBUG=2: RPC DBG logs (not gated by -lv).
- FNV-1a stream hash of GGUF tensor data vs rpc cache filenames: proves
  upload/cache integrity (gguf-py t.data.offset is NOT a file offset; stream
  t.data bytes instead).
- All-peer (-ts 100,0) isolates split-composition bugs from single-device RPC
  bugs; dense LFM3 is a good RPC control model (SWA + shortconv + FA + q8_0 KV).

## Verification suite (post-fix)

Guide baselines (post-fix, .104, drirc active, HOST_CACHED=1 MMID_TUNE=1):

| test | baseline | now | delta |
|---|---|---|---|
| gemma4 MoE pp512 (ub1024) | ~1003 | 1012.9 +/- 2.2 | +1.0% |
| gemma4 MoE pp2048 (ub1024) | ~1102 | 1101.0 +/- 4.5 | -0.1% |
| gemma4 MoE tg128 | ~93 | 94.1 +/- 0.1 | +1.2% |
| LFM3 dense pp2048 (ub512) | ~3391 | 3385.8 +/- 29.3 | -0.2% |
| LFM3 dense tg128 | ~233 | 232.9 +/- 0.6 | -0.1% |
| gemma4 RPC pp512 (ub256, ts 50,50) | ~758 | 750.6 / 754.2 | -0.5..-1.0% |
| gemma4 RPC tg128 | ~47 | 45.3 / 44.7 | -3.6..-4.8% |
| gemma4 server tg (256-token warm greedy x2) | ~83 | 86.3 / 87.0 | +4% |

Server greedy quality: "The capital of France is Paris." - correct.

RPC tg128 is ~4% below baseline: the pipeline-parallel disable removes the
n_copies=4 copy/compute overlap. pp is at parity and the change is required
for inkling2 correctness/perf on 1GbE (4x activation copies). Kept; re-enable
in llama-context.cpp if gemma4 RPC tg ever takes priority over inkling2.

Inkling sweep (final builds): IQ3_S single-board ub512 c16384: pp 50.7,
tg 34.5 (fastest config; 13 GB quant fits one board). Q6_K 22 GB RPC
-ts 25,75 ub256: pp 13.6, tg 9.6 (ub128: pp 13.1, tg 9.8 - ub256 fits but
the wire, not compute, bounds RPC tg). Both coherent incl. tool calls.

---

# Perf optimization pass (2026-11-22, Inkling-2)

Goal: squeeze pp/tg on single-board and RPC, SenseNova-mHC style (study the
hot graph, fuse what is slow, measure). Baselines before this pass:
IQ3_S single ub512 pp 50.7 / tg 34.5; Q6_K RPC ts25,75 ub256 pp 13.6 / tg 9.6.

## Changes

- `e0e5b4228` ggml: populate use_counts on split graphs. The manual split
  assembly left them zero, which silently disabled every use-count-gated
  backend fusion (ggml_can_fuse_subgraph) on multi-backend local graphs and
  on the RPC peer (the serializer sends the counts). Measured alone: ~flat
  on inkling (its patterns mostly do not match the existing fusions), but
  required for the fusions below and correct in general.
- `5d89be2c0` ggml, vulkan: fused SiTU GLU op (GGML_GLU_OP_SITU). The SiTU
  activation was ~10 elementwise ops per MoE layer over strided gate/up
  views; now one shader reading both views directly (no cont copies).
  CPU reference f32/f16/bf16, 24 backend-ops cases pass on gfx1013 and
  Phoenix. No RPC protocol change (GLU variant, op_params carry beta/lb).
  Measured: Q6_K RPC pp 12.7 -> 14.6 (+15%), tg 9.77 -> 10.0; IQ3_S RPC
  ts12.5 pp 15.4 -> 16.2, tg 10.2 -> 10.6; single-board ~flat (pp 51.2,
  tg 29.9 - dispatch overhead is spread over many other small ops).
- LOOP_GATE op (pending commit): fuses the loop-control gating
  add(sub(cur, mul(forget,cur)), mul(gain,message)) into one 4-src shader
  (48 instances/token). GGML_OP_COUNT 102 -> 103, RPC proto patch 3 -> 4,
  both boards rebuilt together.

## Remaining levers (not done, documented)

- mix() hyper-connection mixing: ~800 ops/token (3 mixes x 48 slots, each
  concat+mul_mat+soft_max+mul_mat over the block bank). A fused
  block-attention kernel is the biggest remaining tg lever (single-board
  tg ~30 is dispatch-bound, not bandwidth-bound: ~57 GB/s effective vs
  ~190 GB/s achieved by gemma4).
- topk_moe routing fusion: existing fused_topk_moe expects a
  softmax/top_k pattern; inkling uses sigmoid+bias+argsort_top_k+sum_rows
  normalization - pattern extension needed.
- shortconv dynamic taps: 4 lags x (view/cont/mul/add) on full-attn layers.
