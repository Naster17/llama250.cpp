#!/usr/bin/env bash
# RPC peer server for 2-node BC-250 layer-split inference.
# Run on the SECOND board; point the main node's llama250.sh RPC=host:port at it.
#
# Usage: ./rpc-server.sh [extra ggml-rpc-server args...]
# Env: HOST=0.0.0.0  PORT=50052  THREADS=6
#
# Uses the STOCK Mesa driver (int dot OFF) - correct for gfx1013.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$HERE/build/bin/ggml-rpc-server}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-50052}"
THREADS="${THREADS:-6}"
echo "== llama250 rpc-server (BC-250) host=$HOST port=$PORT threads=$THREADS ==" >&2
exec "$BIN" -H "$HOST" -p "$PORT" -t "$THREADS" "$@"