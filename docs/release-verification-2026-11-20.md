# Release verification - master 5c2ea2d1b (2026-11-20)

Stabilization pass over the main branch. Both boards (.104/.105) synced to master
5c2ea2d1b and rebuilt clean. The experimental sparse-attention work was moved OFF
master to the `sparse-full-attn` branch (quality-for-speed tradeoff not wanted on
the main line). Master is quality-first.

## Configuration (verified intact)

- RADV driconf: /etc/drirc `radv_enable_unified_heap_on_apu=false` on .104
  (backup /etc/drirc.bak-unifiedheap). .105 has no override. Both boards expose
  HOST_CACHED memory types (vulkaninfo count 3). This is the fix that recovers
  server tg (was halved by write-combining host buffers under the unified heap).
  Persists across reboot (verified after a panic/reboot cycle).
- Launcher defaults: scripts/llama250.sh and scripts/rpc-server.sh export
  GGML_VK_HOST_MALLOC_CACHED=1 and GGML_VK_BC250_MMID_TUNE=1.
- Kernel 7.2.10-bc250rocm-r3, Mesa 26.1.6 RADV, 40 CU unlocked, governor min 2000
  MHz, PCIe Gen4 x16. r3 boot guard + bc250-r3-gpu-ready service recover the GPU
  automatically after a reboot (verified).

## Single-board benchmarks (.104, launcher env, -ngl 99 -fa 1 -t 10, r2)

| Model | Type | pp512 | pp2048 | tg128 |
|---|---|---|---|---|
| gemma4-26B-A4B Q4_K_XL (ub1024, q8_0 KV) | MoE | 1008 | 1094 | 93.4 |
| LFM3-1.7B Q8_0 (ub512, f16 KV) | dense | 3255 | 3154 | 161.4 |
| Qwen3-VL-8B Q8_0 (ub512, q8_0 KV) | dense | 697 | 664 | 40.7 |

## Server generation + quality (greedy, temp 0)

- gemma4 single-board server tg 88.2 t/s; "why is the sky blue" -> correct
  Rayleigh-scattering explanation.
- LFM3 single-board: "capital of France" -> "Paris." (correct).
- gemma4 RPC two-board server tg 73.1 t/s; "capital of Japan" -> "Tokyo." (correct).

## RPC two-board (layer-split, BF16 wire, .104 client + .105 peer)

- gemma4 (fits one board): bench pp512 ~770, tg128 ~47; server tg 73.1. RPC is
  functional and quality-correct. As documented, a model that fits one board is
  faster on a single board (tg 93 vs RPC 47 bench) due to the pipeline bubble;
  use RPC for models that do not fit.

## Known issues (not release-blockers for the tested paths)

- Qwen3.5-35B-A3B RPC layer-split bench PANICKED .104 (GPU hang -> oops=panic ->
  reboot; board auto-recovered). Model-specific RPC instability; needs separate
  investigation before that model is used over RPC. gemma4 RPC is stable.
- llama-cli + RPC offload segfaults at load (upstream bug); llama-bench/server work.
- RPC row-split (-sm row) unsupported (no split buffers).
- pp at ub1024 + very long context (>=32k) can hit a 1GB buffer allocation limit;
  use ub512 for long context.
- Sparse full-attn (GGML_VK_FA_FULL_WINDOW) is on the sparse-full-attn branch only:
  real speedup (+13% @8k, +52% @32k) but simple selection policies degrade needle
  recall (window 0/3, landmarks 1/3). Not on master; quality-first decision.

## Reproduction

```sh
# single board (launcher sets HOST_CACHED=1 MMID_TUNE=1)
./scripts/llama250.sh ~/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
# two boards: peer on .105, client on .104
./scripts/rpc-server.sh                                   # .105
RPC=192.168.1.105:50052 ./scripts/llama250.sh ~/models/MODEL.gguf   # .104
```

Verdict: master is stable, fast, and quality-correct for the tested single-board
and gemma4-RPC paths. The drirc unified-heap fix is the key serving-speed win and
is persistent. Qwen3.5-35B RPC is the one flagged instability to investigate next.