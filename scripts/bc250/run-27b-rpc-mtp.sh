#!/usr/bin/env bash
set -euo pipefail

# Recommended Qwen3.8-27B RPC config for 2x BC-250 (1GbE).
# Baseline (2026-11-25, .104+.105, -fa 1 -c 2048, reasoning-budget 0): 24.0/17.8 pp/tg t/s
# With draft-mtp n-max 2: 22.1/22.7 pp/tg t/s (+27.5% tg)

RPC_HOST="${RPC_HOST:-192.168.1.105:50052}"
MODEL="${MODEL:-/home/nik/models/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
DRAFT="${DRAFT:-/home/nik/models/mtp-Qwen3.8-27B-Q4_0.gguf}"
LLAMA_CLI="${LLAMA_CLI:-/home/nik/llama250.cpp/build/bin/llama-cli}"

exec "$LLAMA_CLI" \
  -m "$MODEL" \
  -md "$DRAFT" \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  -ngld 99 \
  -ngl 99 \
  -fa 1 \
  -c "${CTX:-4096}" \
  --rpc "$RPC_HOST" \
  --reasoning-budget 0 \
  "$@"
