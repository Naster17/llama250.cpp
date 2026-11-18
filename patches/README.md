# llama250.cpp patches

All BC-250 (gfx1013 / PS5 Oberon APU) deltas on top of upstream llama.cpp
(`ggml-org/llama.cpp`), in apply order. Each is a `git format-patch` file
(commit message + unified diff); apply with `git am` (or `git apply` for the
diff alone).

Patches 0001-0002 and 0006 predate the current session; 0003-0005 were added
this session. All are already committed in this fork — the files here are the
exportable, reviewable form.

## Patches

| # | File | Area | Purpose |
|---|------|------|---------|
| 0001 | `0001-vulkan-BC-250-gfx1013-scalar-path-tuning-and-UMA-mma.patch` | `ggml-vulkan.cpp`, `ggml-vulkan-types.h` | BC-250 gfx1013 scalar-path tuning (workgroup/row sizes for the no-matrix-core RDNA2 scalar FA/matvec) + UMA mmap handling. The base fork tuning. |
| 0002 | `0002-rpc-optional-BF16-wire-for-layer-split-activations.patch` | `ggml-rpc.cpp`, `transport.cpp` | Optional BF16 wire format for layer-split RPC activations (halves activation traffic over the 2.5GbE link). |
| 0003 | `0003-unicode-non-backtracking-flint-pre-tokenizer-handler.patch` | `src/unicode.cpp`, `tests/test-unicode.cpp` | Non-backtracking flint pre-tokenizer (digits/CJK/words). Fixes musl libc stack-overflow SIGSEGV on long whitespace/CJK runs. Mirrors upstream `3e528431e`+`6f55fb55e` (now also upstream after the merge). |
| 0004 | `0004-llama-pre-split-graph-before-sched_reserve.patch` | `src/llama-context.cpp` | Pre-split graph before `ggml_backend_sched_reserve`. Fixes RPC compute over-allocation (1790 -> 534 MiB) that OOM-killed the remote at ctx 200k. |
| 0005 | `0005-vulkan-env-gated-FA-quant-KV-LDS-staging-for-BC-250.patch` | `ggml-vulkan.cpp` | Extend BC-250 scalar FA LDS staging from F16-only to quantized K/V, env-gated `GGML_VK_FA_QUANT_STAGING=1`. Correct, ~neutral perf at tested depths. |
| 0006 | `0006-bc250-make-r3-the-persistent-default-on-.104-with-bo.patch` | `scripts/bc250/` | Boot GPU-readiness service + r3 persistent default for .104 (board-specific init scripts, not core code). |

## Apply (to a clean upstream checkout)

```sh
cd llama.cpp
git am /path/to/llama250.cpp/patches/000*.patch
```

If a patch no longer applies after upstream moves, rebase with
`git apply --3way patches/000X-*.patch`. Note 0003 is already in recent upstream
(flint support landed there), so it may be redundant on a fresh upstream tree.

## Companion driver patch (separate repo)

The biggest BC-250 win is the Mesa RADV int-dot enablement — `Naster17/bc250-mesa`
(`patches/0001-ac-gpu_info-enable-int-dot-gfx1013.patch` there). It unlocks
`VK_KHR_shader_integer_dot_product` on gfx1013: ~4.3x MoE matvec, +30% prompt
processing. Use the patched `libvulkan_radeon.so` via `VK_ICD_FILENAMES`.

## Not included here

- Docs/bench-log commits and the CI-workflow disable (fork housekeeping, not code
  patches).
- The upstream merge itself (this fork tracks `ggml-org/llama.cpp` master).

## Verification

- 0001/0005: `test-backend-ops -o MUL_MAT_ID / FLASH_ATTN_EXT` on Vulkan0 OK.
- 0003: `test-unicode` passes; flint tokenizes long whitespace/CJK without segfault.
- 0004: flint ctx 200k over RPC loads/listens (RPC0 compute ~534 MiB).