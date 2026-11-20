#!/usr/bin/env bash
# RPC peer server for 2-node BC-250 layer-split inference.
# Run on the SECOND board; point the main node's llama250.sh RPC=host:port at it.
#
# Usage: ./rpc-server.sh [extra ggml-rpc-server args...]
# Env: HOST=0.0.0.0  PORT=50052  THREADS=6  CACHE=1 (weight hash cache)
#      HOST_CACHED=1 (cached host buffers, WC logits fix)  MMID_TUNE=1 (MoE ID tiles)
#
# Uses the STOCK Mesa driver (int dot OFF) - correct for gfx1013.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$HERE/build/bin/ggml-rpc-server}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-50052}"
THREADS="${THREADS:-6}"
CACHE="${CACHE:-1}"
export GGML_VK_HOST_MALLOC_CACHED="${HOST_CACHED:-1}"
export GGML_VK_BC250_MMID_TUNE="${MMID_TUNE:-1}"
echo "== llama250 rpc-server (BC-250) host=$HOST port=$PORT threads=$THREADS cache=$CACHE ==" >&2
args=( -H "$HOST" -p "$PORT" -t "$THREADS" )
[ "$CACHE" = 1 ] && args+=( -c )
exec "$BIN" "${args[@]}" "$@"