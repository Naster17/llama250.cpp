# Usage scenarios (model-agnostic)

Universal commands for llama250.cpp on AMD BC-250 (gfx1013). Works for any GGUF:
dense, MoE, embeddings, vision, etc. Uses the **stock Mesa driver** (int dot OFF
— the dot opcodes are broken on gfx1013; see `bc250-mesa/docs/GFX1013-DOT-HAZARD.md`).

Helper scripts: `scripts/llama250.sh` (server), `scripts/rpc-server.sh` (RPC peer).
Build first: `cmake -B build -DGGML_VULKAN=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`.

## 1. Small/medium dense model, single node (7B–14B)

Fits one 16 GB board. Everyday chat/completion.

```sh
./scripts/llama250.sh ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
# or raw:
./build/bin/llama-server -m model.gguf --host 0.0.0.0 --port 8080 \
  -ngl 99 -t 10 -tb 10 -fa on -ctk q8_0 -ctv q8_0 -c 32768 -b 4096 -ub 1024
```

## 2. Large MoE, two-node RPC (flint, Qwen3-30B-A3B, gpt-oss-20b, ...)

Model > 16 GB → split layers across both boards.

On the peer (.105):
```sh
./scripts/rpc-server.sh                 # ggml-rpc-server -H 0.0.0.0 -p 50052 -t 6
```
On the main node (.104):
```sh
RPC=192.168.1.105:50052 CTX=200000 ./scripts/llama250.sh ~/models/flint-UD-Q4_K_XL.gguf
# or raw:
./build/bin/llama-server -m model.gguf --host 0.0.0.0 --port 8080 \
  -t 10 -tb 10 -fa on -ctk q8_0 -ctv q8_0 -c 200000 --parallel 1 -kvu \
  -ctxcp 8 --swa-checkpoints 8 -cram 2048 -cms 8192 --cache-prompt \
  --split-mode layer -ts 50,50 --rpc 192.168.1.105:50052
```
`-ts 50,50` = 50/50 layer split. Adjust if boards have different free memory.
The RPC pre-split fix (patch 0002) makes large-ctx reserve correctly (~534 MiB
compute instead of an OOM-inducing 1790 MiB).

## 3. Long context (128k–200k)

```sh
CTX=200000 KVU=1 CACHE=1 ./scripts/llama250.sh model.gguf
```
- `-kvu` unified KV + `--cache-prompt -ctxcp N` checkpoints → reuse long prefixes.
- KV quant: `-ctk q8_0 -ctv q8_0` (safe). For very long ctx where KV bandwidth
  dominates tg, try `-ctk q4_0 -ctv q4_0` (halves KV reads; check quality).
- SWA models (gemma, flint, Ministral) only keep full KV for the non-SWA layers.

## 4. Maximum token-generation bandwidth (MoE)

tg on MoE is GDDR6-bandwidth-bound (expert weight reads), not compute-bound.
Levers that actually help:
```sh
# smaller expert quant + q4_0 KV:
./scripts/llama250.sh model-IQ4_XS.gguf   # fewer bytes/token
CTK=q4_0 CTV=q4_0 ./scripts/llama250.sh model.gguf
```
- Smaller quant (IQ4_XS/IQ3) or REAP-pruned experts → fewer weight bytes.
- RPC **row-split** (`--split-mode row`) makes both boards read weights in
  parallel per layer (vs layer-split serializing them) — A/B for your model.
- Integer dot does NOT help (broken on gfx1013); packed FP16 already used.

## 5. Embeddings

```sh
./build/bin/llama-server -m embedding-model.gguf --embeddings \
  --host 0.0.0.0 --port 8080 -ngl 99 -t 10 -fa on -c 8192 -b 4096 -ub 1024
```

## 6. Speculative decoding (draft + target)

```sh
./build/bin/llama-server -m target.gguf -md draft.gguf \
  --host 0.0.0.0 --port 8080 -ngl 99 -ngld 99 -t 10 -fa on \
  --draft-max 16 --draft-min 4 -c 32768
```
Small dense draft (e.g. 1–2B) + larger target; acceptance-dependent tg speedup.

## 7. Vision / multimodal

```sh
./build/bin/llama-server -m model.gguf --mmproj mmproj.gguf \
  --host 0.0.0.0 --port 8080 -ngl 99 -t 10 -fa on -c 32768
```

## 8. Benchmark (pure kernel throughput)

```sh
./build/bin/llama-bench -m model.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
  -sm layer -ts 50/50 -b 4096 -ub 1024 -p 1024,2048 -n 256 -r 2 \
  --rpc 192.168.1.105:50052
```
`llama-bench` uses `-ts 50/50` (slash), no `-tb`. Its tg is pure generation;
a client wall-time "TG tok/s" also includes prompt processing + HTTP.

## General tuning notes (BC-250)

- `-t 10 -tb 10`: 8 Zen2 cores / 16 threads (core unlock active); 10 is a safe
  default, sweep 12-16. `-C 0xffff --cpu-strict 1 --poll 50` pins affinity
  (optional; can reduce jitter).
- `-fa on`: flash attention (scalar path on RDNA2, no matrix cores).
- `-ngl 99`: offload all layers; reduce if a model + KV exceeds 16 GB.
- Governor: keep GPU clock high under load (cyan-skillfish-governor); tg is
  bandwidth-bound so clock matters less than for pp.
- 40-CU unlock + custom kernel are board-level (outside Mesa/llama.cpp).
- Optional `GGML_VK_FA_QUANT_STAGING=1`: FA LDS staging for quant KV (patch
  0003), correct but ~neutral at tested depths.