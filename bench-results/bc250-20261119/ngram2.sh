#!/bin/bash
# ngram-mod retry with a strongly repetitive prompt (LFM3). Run on .104.
BIN=$HOME/llama250.cpp/build/bin/llama-server
O=$HOME/bench-20261119
PROMPT=$(printf 'Repeat the following block exactly 30 times with no changes and no extra text:\nBLOCK: The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.\nEND BLOCK\nStart:\n')
for mode in base ngram; do
  FLAGS=""
  [ $mode = ngram ] && FLAGS="--spec-type ngram-mod"
  $BIN -m $HOME/models/LFM3-1.7B-Q8_0.gguf -ngl 99 -fa on -c 4096 -ub 512 -t 10 -lm dio -ctk q8_0 -ctv q8_0 --port 8093 --host 127.0.0.1 -np 1 $FLAGS > $O/ngram2-$mode-server.log 2>&1 &
  pid=$!
  for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8093/health 2>/dev/null)
    [ "$code" = 200 ] && break
    sleep 1
  done
  curl -s http://127.0.0.1:8093/completion -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json,sys; print(json.dumps({'prompt': sys.argv[1], 'n_predict': 512, 'temperature': 0, 'cache_prompt': False}))" "$PROMPT")" > $O/ngram2-$mode.json
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  python3 - "$O/ngram2-$mode.json" "$mode" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1])); t=d.get("timings",{})
tg=t.get("predicted_n",0)/max(t.get("predicted_ms",1),1)*1000
print(f"{sys.argv[2]}: tg={tg:.1f} n={t.get('predicted_n')} head={d.get('content','')[:60]!r}")
EOF
  grep -i acceptance $O/ngram2-$mode-server.log | tail -1
done
