#!/bin/bash
# Speculative decoding A/B via llama-server + /completion (temp 0, 256 tokens).
# Run on .104. Logs: ~/bench-20261119/spec-*.json / spec-*-server.log
BIN=$HOME/llama250.cpp/build/bin/llama-server
OUT=$HOME/bench-20261119
PORT=8091
mkdir -p "$OUT"
COMMON="-ngl 99 -fa on -c 4096 -ub 512 -t 10 -lm dio -ctk q8_0 -ctv q8_0 --port $PORT --host 127.0.0.1 -np 1"
PROMPT_CHAT='Explain step by step how a flash attention kernel computes softmax online, and why it saves memory. Be concise but complete.'
PROMPT_NGRAM=$(for i in $(seq -w 1 30); do echo "log line $i: sensor=alpha value=$((10#$i * 7 % 97)) status=ok"; done; echo "Continue the exact same pattern for 60 more lines:")

run_spec() { # name model flags prompt
  local name=$1 model=$2 flags=$3 prompt=$4
  echo "== RUN $name"
  $BIN -m "$model" $COMMON $flags > "$OUT/spec-$name-server.log" 2>&1 &
  local pid=$!
  local ok=0
  for i in $(seq 1 180); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null)
    [ "$code" = "200" ] && ok=1 && break
    sleep 1
  done
  if [ "$ok" != 1 ]; then
    echo "$name: SERVER FAILED TO LOAD"; tail -5 "$OUT/spec-$name-server.log"
    kill $pid 2>/dev/null; wait $pid 2>/dev/null; return
  fi
  curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json,sys; print(json.dumps({'prompt': sys.argv[1], 'n_predict': 256, 'temperature': 0, 'cache_prompt': False}))" "$prompt")" \
    > "$OUT/spec-$name.json"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  python3 - "$OUT/spec-$name.json" "$name" <<'EOF'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    t=d.get("timings",{})
    tg=t.get("predicted_n",0)/max(t.get("predicted_ms",1),1)*1000
    print(f"{sys.argv[2]}: tg={tg:.1f} t/s n={t.get('predicted_n')} prompt_n={t.get('prompt_n')} head={d.get('content','')[:60]!r}")
except Exception as e:
    print(sys.argv[2], "PARSE FAIL", e)
EOF
  grep -i -E "acceptance|draft.*rate|speculative" "$OUT/spec-$name-server.log" | tail -2
}

echo "== PHASE spec-lfm"
run_spec lfm-base "$HOME/models/LFM3-1.7B-Q8_0.gguf" "" "$PROMPT_NGRAM"
run_spec lfm-ngram "$HOME/models/LFM3-1.7B-Q8_0.gguf" "--spec-type ngram-mod" "$PROMPT_NGRAM"
echo "== PHASE spec-loop"
run_spec loop-base "$HOME/models/SenseNova-Loop-9B-Q8_0.gguf" "" "$PROMPT_CHAT"
run_spec loop-mtp "$HOME/models/SenseNova-Loop-9B-Q8_0.gguf" "--spec-type draft-mtp --spec-draft-n-max 3" "$PROMPT_CHAT"
echo "== PHASE spec-orca"
run_spec orca-base "$HOME/models/OrcaSAQ-2-27B-Uncensored.gguf" "" "$PROMPT_CHAT"
run_spec orca-mtp "$HOME/models/OrcaSAQ-2-27B-Uncensored.gguf" "--spec-type draft-mtp --spec-draft-n-max 3" "$PROMPT_CHAT"
echo "== PHASE spec-gemma"
run_spec gemma-base "$HOME/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf" "" "$PROMPT_CHAT"
run_spec gemma-eagle3 "$HOME/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf" "-md $HOME/models/draft/gemma-4-26B-A4B-it-speculator.eagle3-Q4_K_M.gguf --spec-draft-n-max 8" "$PROMPT_CHAT"
echo "== PHASE spec-done"
