# llama250.cpp

A [llama.cpp](https://github.com/ggml-org/llama.cpp) fork tuned for the **AMD BC-250** -
the cheap mining board built around the Cyan Skillfish / Oberon APU (`gfx1013`, the
PS5-class chip). 40 compute units, 16 GB of shared GDDR6, Zen 2 CPU, Vulkan via RADV.

Upstream llama.cpp runs on this board, but its Vulkan backend makes conservative
choices for hardware like this. This fork collects the patches, settings and
measurements that close that gap. Everything here was benchmarked on real BC-250
boards, not guessed.

## Why the BC-250 needs its own tuning

`gfx1013` is RDNA1-class with a few RDNA2 touches, and it breaks several assumptions
the Vulkan backend makes:

| Fact | Consequence |
|---|---|
| No `VK_KHR_cooperative_matrix` (no WMMA) | All GEMM runs on the scalar shader path; tile choice dominates prefill speed |
| No accelerated integer dot product (`shaderIntegerDotProduct` all false) | No dp4a shortcut for quantized math; dequant goes through FP16 |
| Full-rate packed FP16 (`v_pk_fma_f16`) | FP16 shader math is the compute lever: ~20.5 TFLOPS FP16 vs ~7.7 TFLOPS FP32 at 40 CU / 2 GHz |
| 40 CUs, but factory boards ship with 24 active | Needs the community dual-register unlock (kernel/umr, outside this repo) |
| UMA: one DEVICE_LOCAL heap, ReBAR enabled | Weight loading path matters; mmap is a win, not a loss |
| Wave32, 64 KB LDS per WGP | Tile and workgroup sizes tuned for this, not for RDNA3 |

## What is patched

All patches live on top of upstream master (base commit `278a89cb7`) and are kept as
small, reviewable diffs:

1. **Vulkan scalar-path tuning for gfx1013** (`ggml-vulkan.cpp`, `ggml-vulkan-types.h`)
   - device detection by PCI ID `0x13FE` + RADV + no coopmat + wave32
   - large GEMM tiles (256x128x128, workgroup 256) enabled on the scalar path
   - occupancy-aware tile selector: fall back to medium tiles when the large tile
     would leave CUs idle
   - decode matvec: extra rows-per-workgroup variants (rows2/rows4) for Q8_0 and
     larger workgroups for long-K shapes
   - FlashAttention decode: shared-memory staging for deep KV (>= 4096)
2. **mmap auto on UMA with ReBAR** (`ggml-vulkan.cpp`)
   - upstream disables mmap for all iGPUs; on this UMA board mmap is ~12% faster for
     prefill, so AUTO now resolves to mmap when ReBAR is present
3. **RPC cluster: BF16 wire for layer-split activations** (`ggml-rpc.cpp`, opt-in
   `GGML_RPC_WIRE_BF16=1`)
   - F32 activation tensors crossing a device boundary are sent as BF16 (2x less
     traffic), +16% prefill on a 2-board 1 GbE cluster; greedy output verified identical
   - `GGML_RPC_QUICKACK=1` experiment kept gated (measured: no effect, NODELAY already set)

Measured effect of the tuning on the new base: +2-5% prefill, +2% decode; the mmap
change adds +12% prefill on top. Full numbers in [docs/bc250.md](docs/bc250.md).

## Quickstart

Build (on the board, Alpine + Vulkan SDK):

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DGGML_RPC=ON
ninja -C build
```

Run (single board):

```sh
build/bin/llama-cli -m model.gguf -ngl 99 -fa 1 -ub 512 -t 6
```

- `-fa 1` is worth +37-41% prefill on this chip.
- `-ub 512` is the measured sweet spot; 2048 costs up to -32%.
- mmap is automatic after the patch (or force with `-lm mmap`).
- Keep KV cache f16; q8_0 KV measured slightly slower here.

Two-board cluster over 1 GbE:

```sh
# board 2
build/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -c
# board 1
GGML_RPC_WIRE_BF16=1 build/bin/llama-bench -m model.gguf -ngl 99 -fa 1 \
    --rpc 192.168.1.105:50052 -sm layer -ub 256
```

`-sm row` is not supported by the RPC backend (split buffers unimplemented).

## System setup matters as much as the code

In order of measured impact:

1. **Mesa version.** RADV/ACO 26.1 emits 12-31% fewer instructions for the same
   shaders than 25.2: +10% prefill, +3% decode, free. Keep Mesa current.
2. **40 CU unlock** (community kernel patch or umr dual-register write). Without it
   you have 24 CUs and ~60% of the performance.
3. **Governor.** [cyan-skillfish-governor-smu](https://github.com/nikarh/cyan-skillfish-governor-smu)
   with the SMU method; see docs/bc250.md for the cold-start config.
4. Kernel cmdline: `amdgpu.gttsize=15200 ttm.pages_limit=3891200`, THP `always`.

## Benchmarks (LFM3-1.7B-Q8_0, 40 CU @ 2 GHz, Mesa 26.1)

| Config | pp512 tok/s | tg128 tok/s |
|---|---|---|
| Upstream defaults (fa auto, no mmap) | ~2517 | ~154 |
| This fork, recommended flags | ~2826 | ~155 |
| This fork, Q4_0 weights | ~2906 | ~213 |
| 2-board RPC cluster, BF16 wire | ~2558 (pp2048) | ~124 |

Q4_0 is the best quant for this chip: prefill on par with Q8_0, decode +36%.
Details, per-op profiles and the full hypothesis log: [docs/bc250.md](docs/bc250.md).

## Repository layout

- Upstream llama.cpp tree, unmodified except the files listed above
- `docs/bc250.md` - hardware facts, all measurements, verdicts, reproduction scripts
- `docs/upstream-README.md` - the original llama.cpp README
- `bench-results/bc250-20261115/` - raw logs, CSVs and scripts from the benchmark rounds

Sync with upstream:

```sh
git remote add upstream https://github.com/ggml-org/llama.cpp
git fetch upstream && git merge upstream/master
```

## Status and caveats

- Private fork for BC-250 owners. Not intended for upstream submission; the
  device-specific heuristics are deliberately gated by PCI ID.
- Known issues: `llama-cli` + RPC offload segfaults at model load (upstream bug,
  `llama-bench`/`llama-server` work); RPC row-split unsupported.
- AI-assisted development; all changes benchmarked and reviewed on hardware.

## License

MIT, as upstream llama.cpp.