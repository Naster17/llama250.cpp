#!/bin/bash
# BC-250 step 2+3 benches (2026-11-19): gemma4 single-board / -t sweep / dio A/B,
# FA staging extension A/B (GGML_VK_FA_STAGING_EXT). Run on .104.
# llama-bench auto-sizes ctx per test (n_prompt + n_gen + n_depth); no -c flag.
# Logs: ~/bench-20261119/<name>.log
B=$HOME/llama250.cpp/build/bin/llama-bench
OUT=$HOME/bench-20261119
mkdir -p "$OUT"
GEMMA=$HOME/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
LFM=$HOME/models/LFM3-1.7B-Q8_0.gguf
RIP=$HOME/models/Ripple-65B-Preview-AD-IQ2_S.gguf
COMMON="-ngl 99 -fa 1 -ub 512 -ctk q8_0 -ctv q8_0 -r 2 -t 10"
QS=GGML_VK_FA_QUANT_STAGING=1

run() { # name envs args...
  local name=$1 envs=$2; shift 2
  echo "== RUN $name"
  { echo "### $name env:[$envs] args:$*"; env $envs "$B" "$@"; echo; } >> "$OUT/$name.log" 2>&1
  grep -E "^\|" "$OUT/$name.log" | tail -6
}

probe() { # name mode (load-time probe: drop caches if root, then timed run + warm run)
  local name=$1 mode=$2
  echo "== RUN $name"
  sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
  { echo "### $name load probe ($mode)"; time "$B" -m "$GEMMA" -ngl 99 -fa 1 -lm "$mode" -p 1 -n 0 -r 1 -t 10; echo; } >> "$OUT/$name.log" 2>&1
  "$B" -m "$GEMMA" -ngl 99 -fa 1 -lm "$mode" -p 1 -n 0 -r 1 -t 10 >> "$OUT/$name.log" 2>&1
  grep -E "real|pp1" "$OUT/$name.log" | tail -3
}

echo "== PHASE step2-gemma"
run gemma-single-t10 "" -m "$GEMMA" $COMMON -p 512,2048 -n 128 -d 0,4096
run gemma-t6  "" -m "$GEMMA" $COMMON -t 6  -p 512 -n 128
run gemma-t14 "" -m "$GEMMA" $COMMON -t 14 -p 512 -n 128
run gemma-t16 "" -m "$GEMMA" $COMMON -t 16 -p 512 -n 128

echo "== PHASE step2-dio"
probe gemma-load-mmap mmap
probe gemma-load-dio dio
run gemma-mmap "" -m "$GEMMA" $COMMON -lm mmap -p 512 -n 128
run gemma-dio  "" -m "$GEMMA" $COMMON -lm dio  -p 512 -n 128
run lfm-mmap "" -m "$LFM" $COMMON -lm mmap -p 512,2048 -n 128
run lfm-dio  "" -m "$LFM" $COMMON -lm dio  -p 512,2048 -n 128

echo "== PHASE step3-lfm"
run lfm-base "$QS" -m "$LFM" $COMMON -p 512,2048 -n 128 -d 0,4096
run lfm-ext1 "$QS GGML_VK_FA_STAGING_EXT=1" -m "$LFM" $COMMON -p 512,2048 -n 128 -d 0,4096
run lfm-ext2 "$QS GGML_VK_FA_STAGING_EXT=2" -m "$LFM" $COMMON -p 512,2048 -n 128 -d 0,4096
run lfm-ext3 "$QS GGML_VK_FA_STAGING_EXT=3" -m "$LFM" $COMMON -p 512,2048 -n 128 -d 0,4096
run lfm-f16kv-base "" -m "$LFM" -ngl 99 -fa 1 -ub 512 -ctk f16 -ctv f16 -r 2 -t 10 -p 512,2048 -n 128 -d 0,4096
run lfm-f16kv-ext3 "GGML_VK_FA_STAGING_EXT=3" -m "$LFM" -ngl 99 -fa 1 -ub 512 -ctk f16 -ctv f16 -r 2 -t 10 -p 512,2048 -n 128 -d 0,4096

echo "== PHASE step3-gemma"
run gemma-ext1 "$QS GGML_VK_FA_STAGING_EXT=1" -m "$GEMMA" $COMMON -p 512,2048 -n 128 -d 0,4096
run gemma-ext3 "$QS GGML_VK_FA_STAGING_EXT=3" -m "$GEMMA" $COMMON -p 512,2048 -n 128 -d 0,4096

echo "== PHASE step3-ripple"
run rip-base "$QS" -m "$RIP" $COMMON -p 512 -n 128 -d 0,4096
run rip-ext4 "$QS GGML_VK_FA_STAGING_EXT=4" -m "$RIP" $COMMON -p 512 -n 128 -d 0,4096
run rip-ext6 "$QS GGML_VK_FA_STAGING_EXT=6" -m "$RIP" $COMMON -p 512 -n 128 -d 0,4096
run rip-ext7 "$QS GGML_VK_FA_STAGING_EXT=7" -m "$RIP" $COMMON -p 512 -n 128 -d 0,4096
echo "== PHASE local-done"
