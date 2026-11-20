#!/bin/bash
# Sampling overhead A/B: default chain vs top_k=1 greedy shortcut (temp 0).
# Same model server, two requests: default vs {"top_k":1,"samplers":["top_k"]}.
# Run on .104. Logs: ~/bench-20261119/samp-*.json
BIN=$HOME/llama250.cpp/build/bin/llama-server
O=$HOME/bench-20261119
PORT=8101
PROMPT='Explain step by step how a flash attention kernel computes softmax online, and why it saves memory. Be concise but complete.'

req() { # name extra_json
  curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
    -d "$(python3 -c "
import json,sys
d={'prompt': sys.argv[1], 'n_predict': 256, 'temperature': 0, 'cache_prompt': False}
d.update(json.loads(sys.argv[2]))
print(json.dumps(d))" "$PROMPT" "$2")" > "$O/samp-$1.json"
}

run_model() { # tag model
  local tag=$1 model=$2
  echo "== PHASE samp-$tag"
  $BIN -m "$model" -ngl 99 -fa on -t 10 -ctk q8_0 -ctv q8_0 --port $PORT --host 127.0.0.1 -np 1 -lm dio -c 2048 -ub 256 > "$O/samp-$tag-server.log" 2>&1 &
  local pid=$! ok=0
  for i in $(seq 1 180); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null)
    [ "$code" = "200" ] && ok=1 && break
    sleep 1
  done
  if [ "$ok" != 1 ]; then echo "$tag: SERVER FAILED"; tail -4 "$O/samp-$tag-server.log"; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; return; fi
  req "$tag-default" '{}'
  req "$tag-topk1" '{"top_k": 1, "samplers": ["top_k"]}'
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; kill -9 $pid 2>/dev/null
  python3 - "$O" "$tag" <<'EOF'
import json,sys
O,tag=sys.argv[1],sys.argv[2]
a=json.load(open(f"{O}/samp-{tag}-default.json")); b=json.load(open(f"{O}/samp-{tag}-topk1.json"))
for name,d in (("default",a),("topk1",b)):
    t=d.get("timings",{})
    tg=t.get("predicted_n",0)/max(t.get("predicted_ms",1),1)*1000
    print(f"{tag}-{name}: tg={tg:.1f} n={t.get('predicted_n')}")
print("IDENTICAL" if a.get("content")==b.get("content") else "CONTENT DIFFERS")
EOF
  sleep 2
}

run_model gemma "$HOME/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
run_model lfm "$HOME/models/LFM3-1.7B-Q8_0.gguf"
echo "== PHASE samp-done"
