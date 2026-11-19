#!/bin/bash
# Qwen3.5-35B-A3B (RPC) + qwen3.5-2b-f16 draft (same family, vocab 248320 verified).
# Orchestrates .105 peer from workstation. Base tg reference: 69.0 (llama-bench).
set -u
H4=nik@192.168.1.104
H5=nik@192.168.1.105
O='~/bench-20261119'
PROMPT='Explain step by step how a flash attention kernel computes softmax online, and why it saves memory. Be concise but complete.'
echo "== PHASE qwen35-draft"
ssh $H5 'nohup ~/llama250.cpp/scripts/rpc-server.sh >/tmp/rpc-q35d.log 2>&1 & sleep 3; head -1 /tmp/rpc-q35d.log'
ssh $H4 "cd ~/llama250.cpp && build/bin/llama-server -m ~/models/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf -md ~/models/qwen3.5-2b-f16.gguf --spec-type draft-simple --spec-draft-n-max 8 -ngl 99 -fa on -c 2048 -ub 512 -t 10 -lm dio -ctk q8_0 -ctv q8_0 --split-mode layer -ts 50,50 --rpc 192.168.1.105:50052 --port 8099 --host 127.0.0.1 -np 1 > $O/spec-qwen35-draft-server.log 2>&1 & sleep 1; ok=0; for i in \$(seq 1 300); do code=\$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8099/health 2>/dev/null); [ \"\$code\" = 200 ] && ok=1 && break; sleep 1; done; if [ \"\$ok\" != 1 ]; then echo SERVER-FAILED; tail -6 $O/spec-qwen35-draft-server.log; pkill -x llama-server; else curl -s http://127.0.0.1:8099/completion -H 'Content-Type: application/json' -d '{\"prompt\": \"$PROMPT\", \"n_predict\": 256, \"temperature\": 0, \"cache_prompt\": false}' > $O/spec-qwen35-draft.json; pkill -x llama-server; fi"
ssh $H5 'pkill -x ggml-rpc-server; sleep 1; echo PEER-STOPPED'
python3 - <<'EOF'
import json, subprocess
out = subprocess.run(["ssh", "nik@192.168.1.104", "cat ~/bench-20261119/spec-qwen35-draft.json"], capture_output=True, text=True).stdout
try:
    d = json.loads(out); t = d.get("timings", {})
    tg = t.get("predicted_n", 0) / max(t.get("predicted_ms", 1), 1) * 1000
    print(f"qwen35-draft: tg={tg:.1f} n={t.get('predicted_n')} head={d.get('content','')[:60]!r}")
except Exception as e:
    print("qwen35-draft PARSE FAIL", e)
EOF
ssh $H4 'grep -i acceptance ~/bench-20261119/spec-qwen35-draft-server.log | tail -1'
echo "== PHASE qwen35-draft-done"
