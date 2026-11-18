# llama250.cpp patches

Deltas applied on top of upstream llama.cpp (`ggml-org/llama.cpp`) for the AMD
BC-250 (gfx1013 / PS5 Oberon APU) inference fork. Each patch is a plain unified
diff, applyable with `git apply` (or `patch -p1`).

Base fork tuning (already committed in this repo, not repeated here):
`1915a3517 vulkan : BC-250 gfx1013 scalar-path tuning and UMA mmap` and
`93ed3ec1b rpc : optional BF16 wire for layer-split activations`.

## Patches (in apply order)

| # | File | Area | Purpose |
|---|------|------|---------|
| 0001 | `0001-flint-pre-tokenizer-non-backtracking.patch` | `src/unicode.cpp`, `tests/test-unicode.cpp` | Non-backtracking flint pre-tokenizer handlers (digits/CJK/words). Fixes the musl libc stack-overflow SIGSEGV on long whitespace/CJK runs (std::regex recursion). Mirrors upstream commits `3e528431e` + `6f55fb55e`. |
| 0002 | `0002-rpc-reserve-pre-split-fix.patch` | `src/llama-context.cpp` | Pre-split the graph before `ggml_backend_sched_reserve`. Fixes the RPC compute-buffer over-allocation (1790 -> 534 MiB) that OOM-killed the remote rpc-server at large ctx (200k). |
| 0003 | `0003-vulkan-fa-quant-kv-staging.patch` | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Extend the BC-250 scalar flash-attention LDS-staging gate from F16-only to quantized K/V (q8_0/q4_0/q4_1/q5_0/q5_1), env-gated `GGML_VK_FA_QUANT_STAGING=1`. Correct but ~neutral perf at tested depths (quant FA is dequant-compute-bound). |

## Apply

```sh
cd llama250.cpp
for p in patches/000*.patch; do git apply "$p"; done
```

If a patch no longer applies after an upstream merge (code moved), rebase with
`git apply --3way patches/000X-*.patch`.

## Companion driver patch

The biggest BC-250 win is the Mesa RADV int-dot enablement (separate repo
`Naster17/bc250-mesa`): it unlocks `VK_KHR_shader_integer_dot_product` on
gfx1013, giving ~4.3x MoE matvec and +30% prompt processing. Use the patched
`libvulkan_radeon.so` via `VK_ICD_FILENAMES`. See ../README.md.

## Verification

- 0001: `test-unicode` passes; flint tokenizes long whitespace/CJK without segfault.
- 0002: flint ctx 200k over RPC loads and listens (RPC0 compute 534 MiB).
- 0003: `test-backend-ops -o FLASH_ATTN_EXT` quant K/V OK; flint generates coherent
  text at depth with `GGML_VK_FA_QUANT_STAGING=1`.