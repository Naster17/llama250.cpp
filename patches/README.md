# llama250.cpp patches

All BC-250 (gfx1013 / PS5 Oberon APU) deltas on top of upstream llama.cpp
(`ggml-org/llama.cpp`), in apply order. Each is a `git format-patch` file
(commit message + unified diff); apply with `git am` (or `git apply` for the
diff alone).

Patches 0001-0002 and 0006 predate the current session; 0003-0005 and 0007-0011
were added this session. All are already committed in this fork — the files here
are the exportable, reviewable form. 0007-0010 (fused mHC) build on the
`sensenova_loop` architecture merged from the blueswhen/OpenSenseNova fork; 0011
refines 0004 (gate the pre-split to RPC backends only).

## Patches

| # | File | Area | Purpose |
|---|------|------|---------|
| 0001 | `0001-vulkan-BC-250-gfx1013-scalar-path-tuning-and-UMA-mma.patch` | `ggml-vulkan.cpp`, `ggml-vulkan-types.h` | BC-250 gfx1013 scalar-path tuning (workgroup/row sizes for the no-matrix-core RDNA2 scalar FA/matvec) + UMA mmap handling. The base fork tuning. |
| 0002 | `0002-rpc-optional-BF16-wire-for-layer-split-activations.patch` | `ggml-rpc.cpp`, `transport.cpp` | Optional BF16 wire format for layer-split RPC activations (halves activation traffic over the 1GbE link). |
| 0003 | `0003-unicode-non-backtracking-flint-pre-tokenizer-handler.patch` | `src/unicode.cpp`, `tests/test-unicode.cpp` | Non-backtracking flint pre-tokenizer (digits/CJK/words). Fixes musl libc stack-overflow SIGSEGV on long whitespace/CJK runs. Mirrors upstream `3e528431e`+`6f55fb55e` (now also upstream after the merge). |
| 0004 | `0004-llama-pre-split-graph-before-sched_reserve.patch` | `src/llama-context.cpp` | Pre-split graph before `ggml_backend_sched_reserve`. Fixes RPC compute over-allocation (1790 -> 534 MiB) that OOM-killed the remote at ctx 200k. |
| 0005 | `0005-vulkan-env-gated-FA-quant-KV-LDS-staging-for-BC-250.patch` | `ggml-vulkan.cpp` | Extend BC-250 scalar FA LDS staging from F16-only to quantized K/V, env-gated `GGML_VK_FA_QUANT_STAGING=1`. Correct, ~neutral perf at tested depths. |
| 0006 | `0006-bc250-make-r3-the-persistent-default-on-.104-with-bo.patch` | `scripts/bc250/` | Boot GPU-readiness service + r3 persistent default for .104 (board-specific init scripts, not core code). |
| 0007 | `0007-ggml-add-fused-mHC-route-merge-ops.patch` | `ggml.h`, `ggml.c`, `ggml-rpc.h`, `ggml-cpu/` | Fused `MHC_ROUTE`/`MHC_MERGE` ggml ops + CPU reference. SenseNova Loop mHC router (gates + 20-pass Sinkhorn + pre-weighted branch) as a single node. Bumps RPC proto patch version. |
| 0008 | `0008-vulkan-fused-mHC-route-merge-kernels.patch` | `ggml-vulkan/`, `vulkan-shaders/` | Vulkan fused mHC kernels: route = one workgroup per (token,stream); merge = elementwise. Collapses ~100 tiny dispatches per router into one. |
| 0009 | `0009-sensenova_loop-use-fused-mHC-ops.patch` | `src/models/` | sensenova_loop graph routes/merges via the fused ops; crops the packed tensor for MTP. On BC-250: pp512 45->382 (8.5x), tg128 7.4->21.7 (2.9x). |
| 0010 | `0010-tests-mHC-route-merge-backend-ops.patch` | `tests/test-backend-ops.cpp` | mHC route/merge backend-op tests (view/non-view, d=4096/1000, n_tok=1/7/512); 18/18 OK on Vulkan0. |
| 0011 | `0011-llama-gate-pre-split-graph-to-RPC-backends.patch` | `src/llama-context.cpp` | Gate the 0004 pre-split to RPC backends only. Local-only graphs split better lazily; recovers qwen3vl local pp while keeping the RPC OOM fix. |
| 0012 | `0012-rpc-optional-Q8_0-activation-wire-GGML_RPC_WIRE_Q8_0.patch` | `ggml/src/ggml-rpc/ggml-rpc.cpp` | Opt-in Q8_0 activation wire (`GGML_RPC_WIRE_Q8_0=1`): quantize F32 split-boundary activations to Q8_0 (~3.8x smaller than F32, ~1.9x vs BF16) to cut pp wire traffic on slow RPC links. Block-aligned chunked transfer; server dequantizes unconditionally. |
| 0013 | `0013-vulkan-IQ3_XXS-dequant-sign-via-conditional-negation.patch` | `vulkan-shaders/dequant_funcs.glsl` | IQ3_XXS pp dequant: sign via conditional negation (free RDNA2 source modifier) instead of multiply by ±1. Ripple-65B pp512 57.6->60.9 t/s on 2xBC-250 RPC; tg unchanged (matvec already optimal). 2604/2604 backend ops OK. |
| 0014 | `0014-vulkan-env-gated-FA-staging-extensions-for-BC-250-GG.patch` | `ggml-vulkan.cpp` | Env-gated BC-250 FA staging extensions (`GGML_VK_FA_STAGING_EXT` bitmask: 1=prefill rows, 2=shallow KV, 4=head dims <=256 other than 128/256). Default off. A/B on q8_0/f16 KV: no gain, prefill staging -1..-10% pp@d4096; knob kept for i-quant KV experiments. FLASH_ATTN_EXT suite OK with EXT=7. |

## Apply (to a clean upstream checkout)

```sh
cd llama.cpp
git am /path/to/llama250.cpp/patches/000*.patch
```

If a patch no longer applies after upstream moves, rebase with
`git apply --3way patches/000X-*.patch`. Note 0003 is already in recent upstream
(flint support landed there), so it may be redundant on a fresh upstream tree.

## Companion driver patch: RETRACTED (do not use)

The old `Naster17/bc250-mesa` int-dot enablement patch is HARMFUL on gfx1013:
the dot opcodes (`v_dot2/4/8`, `v_dot4c`) write zero or SI-era legacy float
min, so `MUL_MAT_ID` returns ERR ~ 1.0 for every quant. The earlier "4.3x MoE
matvec, +30% pp" claim came from perf-only runs without correctness checks.
Use stock Mesa (int dot OFF). See `Naster17/bc250-mesa`
`docs/GFX1013-DOT-HAZARD.md` and docs/BC250-USAGE.md sec. 2.

## Not included here

- Docs/bench-log commits and the CI-workflow disable (fork housekeeping, not code
  patches).
- The upstream merge itself (this fork tracks `ggml-org/llama.cpp` master).

## Verification

- 0001/0005: `test-backend-ops -o MUL_MAT_ID / FLASH_ATTN_EXT` on Vulkan0 OK.
- 0003: `test-unicode` passes; flint tokenizes long whitespace/CJK without segfault.
- 0004: flint ctx 200k over RPC loads/listens (RPC0 compute ~534 MiB).
- 0007-0010: `test-backend-ops -o MHC_ROUTE,MHC_MERGE` 18/18 OK on Vulkan0; SenseNova-Loop-9B-Q8_0 loads and generates coherent text, pp512 382 / tg128 21.7.
- 0011: qwen3vl pp512 669->682 (local), qwen3.5-2b tg128 unchanged (~85); RPC pre-split still triggers when an RPC backend is present.
- 0012: Q8_0 wire functional on gemma4-26B-A4B RPC (pp512 ~677, tg ~47.8); gemma4 pp is compute-bound so wire compression shows no speedup there (it targets wire-bound large-hidden models). Block-aligned transfer verified.
- 0013: `test-backend-ops -o MUL_MAT,MUL_MAT_VEC` 2604/2604 OK on Vulkan0; Ripple-65B pp512 57.6->60.9 t/s (+6%) on 2xBC-250 RPC, tg unchanged (matvec already optimal).
- 0014: `test-backend-ops -o FLASH_ATTN_EXT` with `GGML_VK_FA_STAGING_EXT=7 GGML_VK_FA_QUANT_STAGING=1` OK on Vulkan0; A/B logs in `bench-results/bc250-20261119/` (LFM3/gemma4: neutral-to-negative on q8_0 KV, default unchanged).

## Release v1.1.0

On top of v1.0.0 (upstream merged to `f8caa5ea3`):

- **SenseNova-Loop fused mHC** (0007-0010): single-node `MHC_ROUTE`/`MHC_MERGE` ggml ops + Vulkan kernels replace ~100 tiny dispatches per router. SenseNova-Loop-9B-Q8_0 on one BC-250: pp512 45->382 (8.5x), tg128 7.4->21.7 (2.9x).
- **RPC pre-split gate** (0011): pre-split only for RPC backends; recovers local qwen3vl pp.
- **Q8_0 activation wire** (0012): opt-in `GGML_RPC_WIRE_Q8_0=1`, ~3.8x smaller boundary activations than F32 for wire-bound RPC pp.
- **IQ3_XXS dequant** (0013): sign via conditional negation; Ripple-65B pp +6%.

Notes: Ripple-65B (ricursive) measures via `llama-server` + HTTP, not `llama-bench` (bench requests a 1 GB contiguous RPC buffer that RADV cannot allocate on the 512 MB VRAM carveout; server splits buffers smaller). Vulkan FA already works for ricursive (dense-mask fallback); the kv_rows cell-list optimization is unsupported but gives ~0 gain for this model. Ripple pp is compute-bound on the gfx1013 scalar FP16 path (no WMMA/int-dot); GEMM tiles already tuned (0001), `-ub 512` gives +3% pp over 256.