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

- Qwen3.5-35B-A3B RPC hang/panic: ROOT CAUSE = memory ceiling, not a code bug.
  The Q5_K_XL file is 26.4 GB; layer-split 50/50 puts 13.2 GB of weights on each
  15 GB board, leaving ~1.8 GB for KV + compute + RPC weight-transfer staging.
  That overflows -> allocation hang -> GPU hang -> oops=panic -> reboot (auto-
  recovered). Even a minimal ub128/pp64 config hangs at load. FIX: use a smaller
  quant (Q4_K_XL ~21 GB -> 10.5 GB/board fits with headroom) or a smaller model.
  gemma4 (14.2 GB) RPC is stable because it leaves ample headroom.
- pp ub1024 + long context (>=32k) "1GB buffer" failure: ROOT CAUSE = OOM (exit
  137), same memory ceiling. gemma4 weights occupy 14.2 GB of 15 GB (GTT 13.8 GB
  + VRAM 0.4 GB measured), leaving ~0.8 GB; the 32k KV + ub1024 compute buffers
  (the FA split_k temporary grows to ~1 GB) exceed it. FIX: ub512 for long
  context (halves compute buffers, verified 520 t/s @32k), and/or q4_0 KV (halves
  KV memory), and/or a smaller weight quant. Not a hard buffer-size limit
  (maxStorageBufferRange is 4 GB).
- llama-cli + RPC offload segfaults at load: upstream bug (not fork-specific);
  llama-bench and llama-server work. Use server/bench, not cli, with RPC.
- RPC row-split (-sm row) unsupported: the RPC backend has no split buffers; this
  is a missing feature (significant to implement), not a regression. Layer-split
  (-sm layer) is the supported RPC mode.
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
is persistent. The flagged instabilities (Qwen3.5-35B RPC, ub1024 long-context
"1GB buffer") were investigated and root-caused to the 15 GB shared-memory ceiling
(OOM), not code bugs: use a smaller weight quant for >20 GB models over RPC, and
ub512 / q4_0 KV for long context. The cli+RPC segfault is upstream; RPC row-split
is a missing feature. No code change warranted for the memory-ceiling cases.