# Upstream merge guide (for agents)

Read this BEFORE merging `origin/master` (ggml-org/llama.cpp) into this fork.
It exists so a sync never silently drops BC-250 tuning, and so competing
upstream changes get a human decision instead of a guess.

## 0. Ground rules

- Private fork. NEVER push to `origin`, NEVER open PRs. Push only to `bc250`,
  only with user approval. Commit trailer: `Assisted-by: <model>`.
- Every fork delta was benchmarked on real boards. Never drop one silently.
  Never accept an upstream change that regresses BC-250 without a user decision.
- If upstream touches the same code as our patch and looks better: STOP and ASK
  the user (section 4). Do not pick a side alone.
- Verification is mandatory after every sync (section 5). "It compiles" is not
  verification.

## 1. Know our deltas before touching anything

Source of truth is the committed code. `patches/0001-0019` are historical
format-patch exports (pre-sync base); they may not apply to a new base. Do not
re-export them unless the user asks.

| Area | Files | What it does | Gate |
|---|---|---|---|
| Vulkan scalar path | ggml/src/ggml-vulkan/ggml-vulkan.cpp, -types.h, -common.h, -buffers.cpp, -push-constants.h | gfx1013 tile/row tuning, UMA host buffers, FA quant-KV LDS staging, MoE ID tiles, small-n split-k, IQ dequant sign, cached-host NT staging | PCI ID 0x13FE + env below |
| Vulkan shaders | vulkan-shaders/mhc_route.comp, mhc_merge.comp, dequant_funcs.glsl, vulkan-shaders-gen.cpp | fused mHC kernels, dequant fixes | registry entries |
| ggml ops | ggml/include/ggml.h, ggml/src/ggml.c, ggml-cpu/ops.cpp | GGML_OP_MHC_ROUTE / MHC_MERGE + CPU fallback | op enum |
| RPC wire | ggml/src/ggml-rpc/ggml-rpc.cpp, transport.cpp, ggml/include/ggml-rpc.h | BF16 + Q8_0 activation wire, async cross-backend copies | GGML_RPC_WIRE_Q8_0 |
| Graph split | src/llama-context.cpp | pre-split graph before sched_reserve, RPC backends only; sensenova mHC node budget in graph_max_nodes | RPC presence |
| SenseNova Loop | src/models/sensenova-loop.cpp, models.h, src/llama-arch.*, llama-hparams.h, conversion/, gguf-py/ | arch, KV keys, mHC hparams, fused-op graph | arch enum |
| Tokenizer | src/unicode.cpp, tests/test-unicode.cpp | non-backtracking flint pre-tokenizer handler | none |
| Launchers | scripts/llama250.sh, scripts/rpc-server.sh, scripts/bc250/* | optimal defaults, peer server, r3 boot guard | none |

Env knobs that must survive every merge (grep for them after resolving):
`GGML_VK_BC250_MMID_TUNE`, `GGML_VK_BC250_SPLITK`, `GGML_VK_FA_QUANT_STAGING`,
`GGML_VK_HOST_MALLOC_CACHED`, `GGML_RPC_WIRE_Q8_0`.

System-level, NOT in this repo (do not lose, do not "fix" in code):
- `/etc/drirc` on both boards: `radv_enable_unified_heap_on_apu=false`.
  Without it HOST_CACHED memory types vanish and serving tg halves.
  Diagnose: `vulkaninfo | grep -c HOST_CACHED` (0 = broken, 3 = fixed).
- kernel r3 + cyan-skillfish governor (min 2000 MHz). See docs/bc250.md.

## 2. Merge procedure

1. `git fetch origin`; `git rev-list --count master..origin/master`; read the
   new commit log for vulkan / rpc / llama-graph / gemma4 / model-loader changes.
2. Map the conflict surface:
   `BASE=$(git merge-base master origin/master)` then intersect
   `git diff --name-only $BASE..master` with `git diff --name-only $BASE..origin/master`.
3. `git merge origin/master --no-edit`.
4. Resolve conflicts per section 3. `git add` every resolved file (the index
   keeps unmerged entries until you do; commit fails otherwise).
5. Hunt auto-merge traps (section 3) even in files that merged "cleanly".
6. Compile-check in a FRESH build dir:
   `cmake -B build-sync -DGGML_VULKAN=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release`
   then build `llama-bench llama-server ggml-rpc-server test-backend-ops test-llama-archs`.
   WARNING: an existing `build/` may be configured for a DIFFERENT source tree
   (check `CMAKE_HOME_DIRECTORY` in its CMakeCache.txt). Never trust it blindly.
7. Run local tests: `test-backend-ops -o MHC_ROUTE,MHC_MERGE` and
   `test-llama-archs -a sensenova_loop`. Both must pass.
8. Commit the merge (plain `git commit`; it records both parents), push `bc250`
   with approval.
9. Boards (.104 and .105, parallel): `git fetch bc250 && git reset --hard
   bc250/master && cmake --build build -j16 --target llama-server llama-bench
   ggml-rpc-server`. Both boards must run the SAME build (RPC wire compat).
10. Verification suite, section 5. Parity or better, else investigate or ask.
11. Write a dated sync note `docs/upstream-sync-YYYY-MM-DD.md` with the numbers
    and every resolution decision. Commit + push. Kill test servers and peers.

## 3. Resolution rules and auto-merge traps

- Additive on both sides (ops, KV keys, archs, shaders, test cases): keep BOTH,
  exactly once. Real case: fork sensenova and upstream inkling2-loop both added
  `LLM_KV_LOOP_START/END/COUNT`; auto-merge kept two copies and broke the build.
- Positional tables must track the enums: `GGML_OP_NAME`, `GGML_OP_SYMBOL`,
  the llm_kv name table in llama-arch.cpp. Update every `static_assert` count.
- Op count change => bump `RPC_PROTO_PATCH_VERSION` in ggml-rpc.h and rebuild
  BOTH boards together. Client and peer exchange op indices over the wire.
- Upstream refactors that move our logic: adopt the upstream structure, then
  re-express our calibration inside it. Check first whether upstream already
  absorbed our change (real case: DFLASH block scaling already lived in
  `llama_model_dflash::graph_nodes_per_tensor()`; only the sensenova mHC budget
  needed re-adding in `graph_max_nodes`).
- Tests calling a fork helper that no longer exists: delete the obsolete test
  ONLY if the contract is covered elsewhere (mHC sinkhorn case: covered by
  test_mhc_route/test_mhc_merge in test-backend-ops). Otherwise port the test.
- Python dicts (gguf-py constants.py, tensor_mapping.py): duplicate keys do NOT
  error, they silently overwrite. After merging run
  `grep -oE '"[a-z0-9_.]+"\s*:' <file> | sort | uniq -d`.
- Shader registry: confirm mhc_route/mhc_merge AND new upstream shaders are all
  registered in vulkan-shaders-gen.cpp.
- Never take `theirs` wholesale in a gated file (loses tuning). Never take
  `ours` wholesale (loses upstream fixes). Resolve hunk by hunk.

## 4. Upstream competes with our change: ASK the user

Classify every overlap:

a) Orthogonal or additive -> keep both. No question needed.
b) Upstream solves the SAME problem better (general support instead of our
   env-gated workaround, faster kernel, cleaner design) -> gather evidence
   (bench A/B on .104, quality check, or code argument), then ASK:
   - what ours does, what upstream does, 2-4 sentences;
   - why upstream is better, with numbers if measured;
   - options: keep ours / take upstream / hybrid, with your recommendation
     preselected. Use the ask tool. Wait for the answer; do not decide alone.
c) Upstream regresses BC-250 (bench drop, quality drop, hang) -> keep ours,
   report the evidence in the sync note. Ask only if the tradeoff is unclear.
d) Hardware-specific gate (gfx1013 / PCI ID) that upstream cannot cover ->
   keep ours, no question needed.

Precedents: the drirc unified-heap fix superseded the NEED for
GGML_VK_HOST_MALLOC_CACHED (kept as a redundant safety net, user-approved);
the graph-node-budget refactor was adopted with our mHC calibration preserved
(case b/hybrid, decided with evidence); FA quant-KV LDS staging stayed
env-gated because it carries hang risk (case c/d).

## 5. Verification suite and baselines

Baselines = post-sync b12599 (merge dac8242ed), .104, drirc fix active.
Tolerance: bench +/-2%, tg +/-2%, greedy quality exact, RPC no hang.

```sh
export GGML_VK_HOST_MALLOC_CACHED=1 GGML_VK_BC250_MMID_TUNE=1
M=~/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
# MoE: expect pp512 ~1003, pp2048 ~1102, tg128 ~93
build/bin/llama-bench -m $M -ngl 99 -fa 1 -ub 1024 -ctk q8_0 -ctv q8_0 -t 10 -r 2 -p 512,2048 -n 128
# dense: expect pp2048 ~3391, tg128 ~233 (LFM3-1.7B Q4_0)
build/bin/llama-bench -m ~/models/LFM3-1.7B-Q4_0.gguf -ngl 99 -fa 1 -ub 512 -t 10 -r 2 -p 2048 -n 128
# RPC: peer on .105 (scripts/rpc-server.sh); expect pp512 ~758, tg128 ~47
build/bin/llama-bench -m $M -ngl 99 -fa 1 -ub 256 -ctk q8_0 -ctv q8_0 -t 10 -r 1 -p 512 -n 128 \
  --rpc 192.168.1.105:50052 -sm layer -ts 50,50
# server: scripts/llama250.sh $M ; greedy "capital of France" must say Paris
# server tg method: 256-token warm greedy gen, cache_prompt false, twice;
#   tg = predicted_n*1000/predicted_ms. Expect ~83. The old 88.2 figure used a
#   different unrecorded method - never compare against it.
```

If server tg looks off: A/B the pre-sync commit via `git worktree add` + a
separate build dir, SAME curl method. Probe `LLAMA_GRAPH_REUSE_DISABLE=1` and
`LLAMA_KV_ROWS_DISABLE=1` only as diagnostics; defaults measured best
(83.3 vs 54.6 vs 75.8). Remove the worktree afterwards.

Safety: NEVER read `/sys/kernel/debug/dri/0/amdgpu_error_*` (GPU hang ->
oops=panic -> reboot). Use gpu_metrics sysfs for clocks/temps.
Do not bench Qwen3.5-35B Q5 over RPC or ub1024 at >=32k context: known OOM
(memory ceiling, section 6), not a merge regression.

## 6. Hardware limits are not bugs

16 GB shared GDDR6 (~15 GB usable). Models >~20 GB at Q5 overflow even split
over two boards; ub1024 + >=32k context OOMs on gemma4. Answers are config and
quants (Q4 weights over RPC, ub512 / q4_0 KV for long context), not code.
See docs/release-verification-2026-11-20.md.
