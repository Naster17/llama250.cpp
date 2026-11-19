#!/bin/bash
# OrcaSAQ with -ngl 64 (1 CPU layer) - 60 gave 1.8 t/s (GDN on CPU). Run on .104.
BIN=$HOME/llama250.cpp/build/bin/llama-server
O=$HOME/bench-20261119
PORT=8097
PROMPT='Explain step by step how a flash attention kernel computes softmax online, and why it saves memory. Be concise but complete.'
run_spec() { # name flags
  local name=$1 flags=$2
  echo "== RUN $name"
  $BIN -m $HOME/models/OrcaSAQ-2-27B-Uncensored.gguf -ngl 64 -fa on -t 10 -ctk q8_0 -ctv q8_0 --port $PORT --host 127.0.0.1 -np 1 -lm dio -c 2048 -ub 256 $flags \
    > $O/spec-$name-server.log 2>&1 &
  local pid=$! ok=0
  for i in $(seq 1 180); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null)
    [ "$code" = "200" ] && ok=1 && break
    sleep 1
  done
  if [ "$ok" != 1 ]; then echo "$name: SERVER FAILED"; tail -6 $O/spec-$name-server.log; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; return; fi
  curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json,sys; print(json.dumps({'prompt': sys.argv[1], 'n_predict': 256, 'temperature': 0, 'cache_prompt': False}))" "$PROMPT")" \
    > $O/spec-$name.json
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; kill -9 $pid 2>/dev/null
  python3 - "$O/spec-$name.json" "$name" <<'EOF'
import json,sys
try:
    d=json.load(open(sys.argv[1])); t=d.get("timings",{})
    tg=t.get("predicted_n",0)/max(t.get("predicted_ms",1),1)*1000
    print(f"{sys.argv[2]}: tg={tg:.1f} n={t.get('predicted_n')} head={d.get('content','')[:60]!r}")
except Exception as e:
    print(sys.argv[2], "PARSE FAIL", e)
EOF
  grep -i acceptance $O/spec-$name-server.log | tail -1
  sleep 2
}
echo "== PHASE orca64"
run_spec orca4-base ""
run_spec orca4-mtp "--spec-type draft-mtp --spec-draft-n-max 3"
echo "== PHASE orca64-done"
