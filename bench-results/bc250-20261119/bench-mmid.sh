#!/bin/bash
# BC-250 MoE MUL_MAT_ID tile-tuning A/B (2026-11-19, GGML_VK_BC250_MMID_TUNE).
# Run on .104. Logs: ~/bench-20261119/mmid-*.log
B=$HOME/llama250.cpp/build/bin/llama-bench
OUT=$HOME/bench-20261119
mkdir -p "$OUT"
GEMMA=$HOME/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
COMMON="-ngl 99 -fa 1 -ub 512 -ctk q8_0 -ctv q8_0 -r 2 -t 10"

run() { # name envs args...
  local name=$1 envs=$2; shift 2
  echo "== RUN $name"
  { echo "### $name env:[$envs] args:$*"; env $envs "$B" "$@"; echo; } >> "$OUT/$name.log" 2>&1
  grep -E "^\|" "$OUT/$name.log" | tail -5
}

echo "== PHASE mmid-gemma"
run mmid-gemma-base "" -m "$GEMMA" $COMMON -p 512,2048 -n 128
run mmid-gemma-l1 "GGML_VK_BC250_MMID_TUNE=1" -m "$GEMMA" $COMMON -p 512,2048 -n 128
run mmid-gemma-l3 "GGML_VK_BC250_MMID_TUNE=3" -m "$GEMMA" $COMMON -p 512,2048 -n 128

echo "== PHASE mmid-ub"
run mmid-ub1024-base "" -m "$GEMMA" -ngl 99 -fa 1 -ub 1024 -ctk q8_0 -ctv q8_0 -r 2 -t 10 -p 2048 -n 0
run mmid-ub1024-l1 "GGML_VK_BC250_MMID_TUNE=1" -m "$GEMMA" -ngl 99 -fa 1 -ub 1024 -ctk q8_0 -ctv q8_0 -r 2 -t 10 -p 2048 -n 0
run mmid-ub2048-base "" -m "$GEMMA" -ngl 99 -fa 1 -ub 2048 -ctk q8_0 -ctv q8_0 -r 2 -t 10 -p 2048 -n 0
run mmid-ub2048-l1 "GGML_VK_BC250_MMID_TUNE=1" -m "$GEMMA" -ngl 99 -fa 1 -ub 2048 -ctk q8_0 -ctv q8_0 -r 2 -t 10 -p 2048 -n 0

echo "== PHASE mmid-profile"
GGML_VK_PERF_LOGGER=1 "$B" -m "$GEMMA" $COMMON -p 512 -n 0 -r 1 > "$OUT/mmid-prof-base.log" 2>&1
GGML_VK_BC250_MMID_TUNE=1 GGML_VK_PERF_LOGGER=1 "$B" -m "$GEMMA" $COMMON -p 512 -n 0 -r 1 > "$OUT/mmid-prof-l1.log" 2>&1
echo "-- base MUL_MAT_ID:"; grep -E "MUL_MAT_ID|Total time" "$OUT/mmid-prof-base.log" | tail -8
echo "-- l1 MUL_MAT_ID:"; grep -E "MUL_MAT_ID|Total time" "$OUT/mmid-prof-l1.log" | tail -8
echo "== PHASE mmid-done"
