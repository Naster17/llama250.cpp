#!/bin/bash
# RPC async-copy A/B (2026-11-19). Orchestrates .104 (client) + .105 (peer) from workstation.
# GGML_RPC_ASYNC_COPY is client-side; server decodes wire flags unconditionally.
set -u
H4=nik@192.168.1.104
H5=nik@192.168.1.105
B4='~/llama250.cpp/build/bin'
O='~/bench-20261119'
GEMMA='~/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf'
LFM='~/models/LFM3-1.7B-Q8_0.gguf'
COMMON="-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -r 2 -t 10 --rpc 192.168.1.105:50052 -sm layer -ts 50/50"
scp -q ggml/src/ggml-rpc/ggml-rpc.cpp $H4:llama250.cpp/ggml/src/ggml-rpc/ggml-rpc.cpp || exit 1
scp -q ggml/src/ggml-rpc/ggml-rpc.cpp $H5:llama250.cpp/ggml/src/ggml-rpc/ggml-rpc.cpp || exit 1
ssh $H4 "bash -c 'set -o pipefail; cd ~/llama250.cpp && ninja -C build ggml-rpc llama-bench llama-server ggml-rpc-server test-rpc-multi-server 2>&1 | tail -1'" || exit 1
ssh $H5 "bash -c 'set -o pipefail; cd ~/llama250.cpp && ninja -C build ggml-rpc llama-bench llama-server ggml-rpc-server 2>&1 | tail -1'" || exit 1

echo "== PHASE rpc-protocol-test"
ssh $H4 "cd ~/llama250.cpp && GGML_RPC_ASYNC_COPY=1 GGML_RPC_WIRE_BF16=1 bash tests/test-rpc-multi-server.sh build/bin/ggml-rpc-server build/bin/test-rpc-multi-server && echo PROTOCOL-OK" || echo PROTOCOL-FAIL

echo "== PHASE rpc-perf"
ssh $H5 'nohup ~/llama250.cpp/scripts/rpc-server.sh >/tmp/rpc-async.log 2>&1 & sleep 3; head -1 /tmp/rpc-async.log'
ssh $H4 "GGML_RPC_WIRE_BF16=1 $B4/llama-bench -m $GEMMA $COMMON -ub 512 -p 512,2048 -n 128 > $O/rpc-async-base.log 2>&1; GGML_RPC_WIRE_BF16=1 $B4/llama-bench -m $GEMMA $COMMON -ub 1024 -p 2048 -n 0 > $O/rpc-async-base-ub1024.log 2>&1; grep -hE '^\|.*(pp|tg)' $O/rpc-async-base*.log | tail -5"
ssh $H4 "GGML_RPC_WIRE_BF16=1 GGML_RPC_ASYNC_COPY=1 $B4/llama-bench -m $GEMMA $COMMON -ub 512 -p 512,2048 -n 128 > $O/rpc-async-on.log 2>&1; GGML_RPC_WIRE_BF16=1 GGML_RPC_ASYNC_COPY=1 $B4/llama-bench -m $GEMMA $COMMON -ub 1024 -p 2048 -n 0 > $O/rpc-async-on-ub1024.log 2>&1; grep -hE '^\|.*(pp|tg)' $O/rpc-async-on*.log | tail -5"
ssh $H5 'pkill -x ggml-rpc-server; sleep 1; echo PEER-STOPPED'

echo "== PHASE rpc-identity"
PROMPT='Write a short numbered list of five uses for a blank sheet of paper.'
for mode in base async; do
  ENVS="GGML_RPC_WIRE_BF16=1"
  [ $mode = async ] && ENVS="GGML_RPC_WIRE_BF16=1 GGML_RPC_ASYNC_COPY=1"
  ssh $H5 'nohup ~/llama250.cpp/scripts/rpc-server.sh >/tmp/rpc-async.log 2>&1 & sleep 2'
  ssh $H4 "cd ~/llama250.cpp && env $ENVS build/bin/llama-server -m $LFM -ngl 99 -fa on -c 2048 -ub 512 -t 10 -lm dio --port 8092 --host 127.0.0.1 -np 1 --split-mode layer -ts 50,50 --rpc 192.168.1.105:50052 > $O/rpc-ident-$mode-server.log 2>&1 & sleep 1; for i in \$(seq 1 120); do code=\$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8092/health 2>/dev/null); [ \"\$code\" = 200 ] && break; sleep 1; done; curl -s http://127.0.0.1:8092/completion -H 'Content-Type: application/json' -d '{\"prompt\": \"$PROMPT\", \"n_predict\": 128, \"temperature\": 0, \"cache_prompt\": false}' > $O/rpc-ident-$mode.json; pkill -x llama-server"
  ssh $H5 'pkill -x ggml-rpc-server; sleep 1'
done
python3 - <<'EOF'
import json, subprocess
for mode in ("base", "async"):
    out = subprocess.run(["ssh", "nik@192.168.1.104", f"cat ~/bench-20261119/rpc-ident-{mode}.json"], capture_output=True, text=True).stdout
    try:
        d = json.loads(out)
        print(mode, "content:", repr(d.get("content", "")[:120]))
    except Exception as e:
        print(mode, "FAIL", e, out[:200])
EOF

echo "== PHASE ngram-retry"
scp -q bench-results/bc250-20261119/ngram2.sh $H4:bench-20261119/ngram2.sh
ssh $H4 'bash ~/bench-20261119/ngram2.sh'
echo "== PHASE rpc-async-done"
