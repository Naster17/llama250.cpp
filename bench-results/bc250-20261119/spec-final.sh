#!/bin/bash
# Final speculation round on cleared GPU: gemma4 EAGLE3 (ub256/c2048), OrcaSAQ MTP (-ngl 60).
# Run on .104. Uses port 8095.
BIN=$HOME/llama250.cpp/build/bin/llama-server
O=$HOME/bench-20261119
PORT=8095
PROMPT='Explain step by step how a flash attention kernel computes softmax online, and why it saves memory. Be concise but complete.'

run_spec() { # name model flags extra
  local name=$1 model=$2 flags=$3 extra=$4
  echo "== RUN $name"
  $BIN -m "$model" -fa on -t 10 -ctk q8_0 -ctv q8_0 --port $PORT --host 127.0.0.1 -np 1 $extra $flags > "$O/spec-$name-server.log" 2>&1 &
  local pid=$! ok=0
  for i in $(seq 1 180); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null)
    [ "$code" = "200" ] && ok=1 && break
    sleep 1
  done
  if [ "$ok" != 1 ]; then
    echo "$name: SERVER FAILED TO LOAD"; tail -6 "$O/spec-$name-server.log"
    kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; return
  fi
  curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json,sys; print(json.dumps({'prompt': sys.argv[1], 'n_predict': 256, 'temperature': 0, 'cache_prompt': False}))" "$PROMPT")" \
    > "$O/spec-$name.json"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; kill -9 $pid 2>/dev/null
  python3 - "$O/spec-$name.json" "$name" <<'EOF'
import json,sys
try:
    d=json.load(open(sys.argv[1])); t=d.get("timings",{})
    tg=t.get("predicted_n",0)/max(t.get("predicted_ms",1),1)*1000
    print(f"{sys.argv[2]}: tg={tg:.1f} t/s n={t.get('predicted_n')} head={d.get('content','')[:60]!r}")
except Exception as e:
    print(sys.argv[2], "PARSE FAIL", e)
EOF
  grep -i acceptance "$O/spec-$name-server.log" | tail -1
  sleep 2
}

echo "== PHASE final-gemma"
run_spec gemma3-base "$HOME/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf" "-ngl 99" "-lm dio -c 2048 -ub 256"
run_spec gemma3-eagle3 "$HOME/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf" "-ngl 99 -md $HOME/models/draft/gemma-4-26B-A4B-it-speculator.eagle3-Q4_K_M.gguf --spec-draft-n-max 8" "-lm dio -c 2048 -ub 256"
echo "== PHASE final-orca"
run_spec orca3-base "$HOME/models/OrcaSAQ-2-27B-Uncensored.gguf" "-ngl 60" "-lm dio -c 2048 -ub 256"
run_spec orca3-mtp "$HOME/models/OrcaSAQ-2-27B-Uncensored.gguf" "-ngl 60 --spec-type draft-mtp --spec-draft-n-max 3" "-lm dio -c 2048 -ub 256"
echo "== PHASE final-done"
