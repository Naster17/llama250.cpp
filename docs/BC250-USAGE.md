# Using llama250.cpp on the AMD BC-250

This fork = upstream llama.cpp + BC-250 (gfx1013 / PS5 Oberon) patches. See
`patches/README.md` for the delta list and `README.md` for the project log.

## 1. Build

Needs Vulkan + RPC. On Alpine (musl): `apk add cmake ninja gcc g++ vulkan-headers
spirv-headers glslang-dev libdrm-dev wayland-dev python3 mako`.

```sh
cmake -B build -DGGML_VULKAN=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target llama-server llama-bench ggml-rpc-server
```

Binaries: `build/bin/llama-server`, `build/bin/ggml-rpc-server`, `build/bin/llama-bench`.

## 2. Patched RADV driver (the big win)

Upstream Mesa disables integer dot product on gfx1013. The `Naster17/bc250-mesa`
patch enables it: ~4.3x MoE expert matvec, +30% prompt processing. Build it once:

```sh
git clone https://github.com/Naster17/bc250-mesa && cd bc250-mesa && ./build.sh
```

Then point the Vulkan loader at the patched driver (no root, no system change):

```sh
export VK_ICD_FILENAMES=$PWD/mesa/build/src/amd/vulkan/radeon_devenv_icd.x86_64.json
```

Set this on **every** node (server and each rpc-server). Confirm with
`vulkaninfo --summary` and the llama.cpp startup line `int dot: 1` (was `0`).

## 3. Single node

```sh
VK_ICD_FILENAMES=/path/radeon_devenv_icd.x86_64.json \
./build/bin/llama-server -m model.gguf --host 0.0.0.0 --port 8080 \
  -ngl 99 -t 10 -tb 10 -fa on -ctk q8_0 -ctv q8_0 -c 32768 -b 4096 -ub 1024
```

## 4. Two-node RPC (layer split)

On the remote node (.105):

```sh
VK_ICD_FILENAMES=/path/radeon_icd.json \
./build/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -t 6
```

On the main node (.104):

```sh
VK_ICD_FILENAMES=/path/radeon_devenv_icd.x86_64.json \
./build/bin/llama-server -m flint-UD-Q4_K_XL.gguf --host 0.0.0.0 --port 8080 \
  -t 10 -tb 10 -C 0xffff --cpu-strict 1 --poll 50 \
  -b 4096 -ub 1024 -ctk q8_0 -ctv q8_0 -fa on \
  -c 200000 --parallel 1 -kvu -ctxcp 8 --swa-checkpoints 8 \
  -cram 2048 -cms 8192 --cache-prompt --jinja --reasoning-format auto \
  --split-mode layer -ts 50,50 --rpc 192.168.1.105:50052
```

Notes:
- `-ts 50,50` splits layers 50/50 across the two boards. The model must fit the
  combined 2x16 GB; flint Q4_K_XL (23.3 GiB) does.
- The RPC pre-split fix (patch 0002) is what makes `-c 200000` reserve correctly
  (RPC compute ~534 MiB instead of an OOM-inducing 1790 MiB).
- `-kvu` (unified KV) + `-ctxcp`/`--swa-checkpoints` enable prompt caching.

## 5. Optional env knobs

| Env | Effect |
|-----|--------|
| `GGML_VK_FA_QUANT_STAGING=1` | FA LDS staging for quantized K/V (patch 0003). Correct, ~neutral at tested depths; may help at very long ctx. Off by default. |
| `VK_ICD_FILENAMES=...` | Select the patched RADV driver (int dot). Recommended. |

## 6. Benchmark

```sh
./build/bin/llama-bench -m model.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
  -sm layer -ts 50/50 -b 4096 -ub 1024 -p 1024,2048 -n 256 -r 2 \
  --rpc 192.168.1.105:50052
```

`llama-bench` uses `-ts 50/50` (slash) and has no `-tb`; the server uses `-ts 50,50`
(comma). Pure tg from llama-bench is the clean kernel metric; the server's
`timings.predicted_ms` is pure generation, while a client wall-time "TG tok/s"
also includes prompt processing + HTTP.

## 7. What each patch buys

- flint pre-tokenizer (0001): no musl stack-overflow segfault on long whitespace/CJK.
- RPC pre-split (0002): large-ctx RPC reserve no longer OOM-kills the remote.
- FA quant-KV staging (0003): optional, neutral perf.
- bc250-mesa int dot (driver): +30% pp, ~4.3x MoE matvec — the main speedup.