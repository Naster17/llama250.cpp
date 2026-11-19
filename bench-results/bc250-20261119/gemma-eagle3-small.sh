#!/bin/bash
# gemma4 EAGLE3 with EXPLICIT --spec-type (auto-detect ran no drafts). Run on .104.
BIN=$HOME/llama250.cpp/build/bin/llama-server
O=$HOME/bench-20261119
PORT=8098
PROMPT='Explain step by step how a flash attention kernel computes softmax online, and why it saves memory. Be concise but complete.'
echo "== RUN gemma4-eagle3-small"
$BIN -m $HOME/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf -ngl 99 -fa on -t 10 -ctk q8_0 -ctv q8_0 --port $PORT --host 127.0.0.1 -np 1 -lm dio -c 1024 -ub 128 \
  -md $HOME/models/draft/gemma-4-26B-A4B-it-speculator.eagle3-Q4_K_M.gguf --spec-type draft-eagle3 --spec-draft-n-max 8 \
  > $O/spec-gemma4-eagle3-small-server.log 2>&1 &
pid=$!; ok=0
for i in $(seq 1 180); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null)
  [ "$code" = "200" ] && ok=1 && break
  sleep 1
done
if [ "$ok" != 1 ]; then echo "SERVER FAILED"; tail -8 $O/spec-gemma4-eagle3-small-server.log; kill -9 $pid 2>/dev/null; exit 1; fi
curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d "$(python3 -c "import json,sys; print(json.dumps({'prompt': sys.argv[1], 'n_predict': 256, 'temperature': 0, 'cache_prompt': False}))" "$PROMPT")" \
  > $O/spec-gemma4-eagle3-small.json
kill $pid 2>/dev/null; wait $pid 2>/dev/null; kill -9 $pid 2>/dev/null
python3 - "$O/spec-gemma4-eagle3-small.json" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1])); t=d.get("timings",{})
tg=t.get("predicted_n",0)/max(t.get("predicted_ms",1),1)*1000
print(f"gemma4-eagle3-small: tg={tg:.1f} n={t.get('predicted_n')} head={d.get('content','')[:60]!r}")
EOF
grep -i -E "acceptance|speculative|eagle" $O/spec-gemma4-eagle3-small-server.log | tail -4
