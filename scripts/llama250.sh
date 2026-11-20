#!/usr/bin/env bash
# Universal llama250.cpp launcher for AMD BC-250 (gfx1013 / RDNA2-class).
# Model-agnostic: works for dense, MoE, embeddings, any GGUF.
#
# Usage:
#   ./llama250.sh MODEL.gguf [extra llama-server args...]
#
# Env overrides:
#   PORT=8080            HTTP port
#   CTX=0                context size (0 = model default / train ctx)
#   NGL=99               GPU layers (99 = all)
#   THREADS=10           CPU threads (BC-250: 8 cores / 16 threads unlocked; sweep 10-16)
#   RPC=host:port        enable 2-node layer-split RPC (run rpc-server.sh on peer)
#   TS=50,50             tensor split when RPC set
#   KVU=1                unified KV (prompt caching); set 0 for separate SWA cache
#   FA=on                flash attention
#   CTK=q8_0 CTV=q8_0    KV cache quant (q8_0 safe; q4_0 halves KV bandwidth)
#   UB=1024 B=4096       ubatch / batch (MoE pp favors 1024; avoid 2048: device-lost)
#   LM=dio               load mode (dio: 2x faster load, +1-6% pp measured on BC-250;
#                        mmap/auto/none/mlock also valid; use mmap on tmpfs/network FS)
#   CACHE=1              prompt caching + checkpoints (set 0 to disable)
#   MMID_TUNE=1          MoE ID tile tuning, per-expert fit (+7.6% MoE pp at ub512
#                        on gemma4, neutral elsewhere; 0 disables)
#   HOST_CACHED=1        cached host buffers via malloc+import (fixes ~10ms/token WC logits
#                        reads on BC-250; +80% serving tg measured; 0 disables)
#   EXTRA_ARGS           appended verbatim
#
# Notes:
#   - Uses the STOCK Mesa driver (int dot OFF). Do NOT enable the retracted
#     bc250-mesa int-dot patch: gfx1013 dot opcodes write garbage.
#   - Packed FP16 / mixed-precision / INT24 / DPP accelerators are used by the
#     FP16 path automatically. No driver patch needed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$HERE/build/bin/llama-server}"
MODEL="${1:?usage: llama250.sh MODEL.gguf [args...]}"
shift || true

PORT="${PORT:-8080}"
CTX="${CTX:-0}"
NGL="${NGL:-99}"
THREADS="${THREADS:-10}"
FA="${FA:-on}"
CTK="${CTK:-q8_0}"
CTV="${CTV:-q8_0}"
UB="${UB:-1024}"
B="${B:-4096}"
KVU="${KVU:-1}"
CACHE="${CACHE:-1}"
LM="${LM:-dio}"
export GGML_VK_BC250_MMID_TUNE="${MMID_TUNE:-1}"
export GGML_VK_HOST_MALLOC_CACHED="${HOST_CACHED:-1}"

args=( -m "$MODEL" --host 0.0.0.0 --port "$PORT"
       -ngl "$NGL" -t "$THREADS" -tb "$THREADS"
       -fa "$FA" -ctk "$CTK" -ctv "$CTV" -b "$B" -ub "$UB" -lm "$LM"
       --parallel "${PARALLEL:-1}" )
[ "$CTX" != 0 ] && args+=( -c "$CTX" )
[ "$KVU" = 1 ] && args+=( -kvu )
if [ "$CACHE" = 1 ]; then
  args+=( --cache-prompt -ctxcp "${CTXCP:-8}" --swa-checkpoints "${SWACP:-8}"
          -cram "${CRAM:-2048}" -cms "${CMS:-8192}" )
fi
if [ -n "${RPC:-}" ]; then
  args+=( --split-mode layer -ts "${TS:-50,50}" --rpc "$RPC" )
fi
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && args+=( ${EXTRA_ARGS} )
args+=( "$@" )

echo "== llama250 (BC-250) ==" >&2
echo "model=$MODEL port=$PORT ctx=$CTX ngl=$NGL rpc=${RPC:-none} kvu=$KVU cache=$CACHE lm=$LM" >&2
exec "$BIN" "${args[@]}"