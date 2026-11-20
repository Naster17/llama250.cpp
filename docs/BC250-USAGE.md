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

## 2. Use the STOCK Mesa driver (do NOT enable integer dot)

**Do not use the old `bc250-mesa` int-dot patch.** It was retracted: the gfx1013
dot opcodes (`v_dot2/4/8`) write zero (or `v_dot4c` runs an SI-era legacy float
min), so enabling `has_accelerated_dot_product` makes llama.cpp produce garbage
(`MUL_MAT_ID` ERR ~ 1.0 for every quant). Stock Mesa correctly keeps integer dot
**off** on gfx1013 (`int dot: 0`). See `Naster17/bc250-mesa`
`docs/GFX1013-DOT-HAZARD.md`.

Just use the system RADV driver (no `VK_ICD_FILENAMES` override). The startup
line should read `int dot: 0`. The working accelerators on this chip are packed
FP16 / mixed-precision FMA / INT24 / DPP, which llama.cpp's FP16 path already
uses — no driver patch needed.

## 3. Single node

```sh
./build/bin/llama-server -m model.gguf --host 0.0.0.0 --port 8080 \
  -ngl 99 -t 10 -tb 10 -fa on -ctk q8_0 -ctv q8_0 -c 32768 -b 4096 -ub 1024
```

## 4. Two-node RPC (layer split)

On the remote node (.105):

```sh
./build/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -t 6
```

On the main node (.104):

```sh
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
| `GGML_VK_FA_STAGING_EXT=N` | BC-250 FA staging extensions (patch 0014), bitmask: 1=prefill rows, 2=shallow KV (<4096), 4=head dims <=256 other than 128/256. Off by default; A/B on q8_0/f16 KV showed no gain (prefill staging slightly negative). For i-quant KV experiments. |
| `GGML_VK_BC250_MMID_TUNE=N` | MoE MUL_MAT_ID tile tuning (patch 0015): 1=per-expert tile fit (+7.6% MoE pp at ub512 on gemma4), 2=WG-256 large ID warptile variant. Off by default in code; llama250.sh sets 1. MoE pp also favors `-ub 1024`; avoid 2048 (device-lost on gemma4). |
| `GGML_VK_HOST_MALLOC_CACHED=1` | Cached host buffers via malloc+import (patch 0019). Fixes ~10 ms/token write-combining logits reads on BC-250: serving tg +85% measured (gemma4 47.8->88.3), pp2048 +6%. Off by default in code; llama250.sh sets 1. Set 0 if a model regresses. |
| `GGML_VK_BC250_SPLITK=1` | Small-n split-k for gfx1013 (patch 0017): +27-31% on pp16/pp32 (spec-verify and small-batch shapes), large pp unchanged. Off by default. |
| `GGML_RPC_ASYNC_COPY=1` | Async RPC cross-backend copies + async-path wire compression (patch 0018). Correct (protocol + greedy identity verified); perf neutral on 1GbE - for faster interconnects. Off by default. |

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
- bc250-mesa int dot (driver): **RETRACTED — harmful.** gfx1013 dot opcodes write
  zero/garbage; stock Mesa (int dot off) is correct. The real levers are packed
  FP16 / mix / INT24 / DPP, already used by the FP16 path. See bc250-mesa docs.