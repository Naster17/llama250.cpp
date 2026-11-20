# Context parallelism for long-context prefill - design scope

Status: SCOPED, not implemented. Verdict up front: the design is sound and reuses
existing ggml mechanisms, but it is **net-negative on the current 1GbE inter-board
link**. It becomes beneficial only with 10GbE+. See section 5 (scaling) before
reading the rest. This document is the concrete design to revisit if the cluster
gets a faster interconnect.

## 1. Goal

Flatten the long-context prefill (pp) curve. pp drops 926 -> 912 -> 815 -> ~450
t/s as context grows 512 -> 2048 -> 8192 -> 30k+ because the full-attention
layers (5 of 30 in gemma4-26B, head_dim 512, causal dense mask) are O(n^2). The
SWA layers (25) are already cheap (fixed window). Context parallelism splits the
full-attn KV computation across the two boards so the O(n^2) term divides by the
board count. It does NOT help tg (single query token, no O(n^2) prefill term).

## 2. Existing mechanisms we reuse

- FA split-KV / flash-decoding (ggml-vulkan.cpp ~9233-9306): the FA op already
  partitions KV into `split_k` chunks. Each chunk computes a partial output O
  [HSV, ne1, ne2, ne3] plus per-row softmax stats m (running max) and L (running
  sum), laid out in `prealloc_split_k`. `fa_split_k_reduce` merges the partials
  via online softmax. This is exactly the partial-attention + reduction math
  context parallelism needs; today it runs within one device.
- RPC backend (ggml-rpc.cpp): dispatches ops to remote devices; supports_op
  returns true (delegates to remote). Layer-split graph crosses activations once
  per ubatch (small). BF16 wire compression (GGML_RPC_WIRE_BF16) and async copies
  already exist.
- ISWA KV cache (llama-kv-cache-iswa): gemma4 uses interleaved SWA. The cache
  separates `kv_swa` (windowed) from the full-attn cache (`get_swa()` /
  full). Only the full-attn cache needs sharding; kv_swa stays local.
- Attention build (llama-graph.cpp build_attn ~3397): for each layer it does
  `mctx->cpy_k/cpy_v` (write new K/V into the cache at k_idxs/v_idxs), then
  `mctx->get_k/get_v` (read the cache K/V), then `build_attn_mha(q,k,v,mask,
  kv_rows,...)`. These are the insertion points for sharding.

## 3. Design

### 3.1 KV sharding

Shard the full-attn KV cache cells across the N boards by context position:
board b owns cells [b*KV/N, (b+1)*KV/N). kv_swa is replicated/local (cheap).
New K/V for the current ubatch (k_cur/v_cur) are written to the board owning
their position (cpy_k/cpy_v routed by idx). gemma4 shared-KV layers (layers that
read another layer's KV) must follow the same shard map as their source layer.

Two layout options:
- (A) Sharded KV, replicated Q: each board holds a KV shard and the FULL Q. Each
  board computes partial O for all Q against its KV shard; partials are reduced.
  Simpler graph; Q is small so replication is cheap. Partial O is large (section
  5) -> reduction transfer dominates.
- (B) Sharded KV + sharded Q (ring attention): each board holds Q shard + KV
  shard; KV shards ring around, each board accumulates O for its Q shard online.
  Transfer is KV blocks (not partial O); O stays local. More complex (ring
  schedule, N-1 steps), but transfers KV once instead of partial O per layer.

Recommendation: (B) ring attention for pp, because partial O (option A) is
larger than the KV shard transfer for long context. Both reuse the split-KV
partial O + m/L representation.

### 3.2 Reduction op

A cross-backend online-softmax merge of partial attention results. Given partial
(O_i, m_i, L_i) from each board:
  m = max_i(m_i);  L = sum_i(L_i * exp(m_i - m));
  O = sum_i(O_i * L_i * exp(m_i - m)) / L.
This is the fa_split_k_reduce math, generalized to inputs living on different
backends. Implement as a ggml op (e.g. GGML_OP_FLASH_ATTN_EXT_REDUCE) whose srcs
are the partial O/m/L tensors; the scheduler places it on one backend and inserts
cross-backend copies (RPC GET_TENSOR) for the remote partials. The Vulkan shader
already exists (fa_split_k_reduce); the new work is the cross-backend wiring and
a CPU fallback reduce.

### 3.3 Graph changes (llama-graph.cpp build_attn)

For full-attn layers when context-parallel is enabled:
1. cpy_k/cpy_v: route new K/V to the owning shard (local or remote via RPC).
2. get_k/get_v: return the LOCAL KV shard (not the whole cache).
3. build_attn_mha: emit a partial FA over the local shard, producing O_local +
   m_local + L_local (extend ggml_flash_attn_ext to output m/L, or add a variant).
4. Ring step (option B): exchange KV shards with the peer (RPC), compute partial
   FA against the received shard, merge online locally. After N-1 steps each
   board has the full O for its Q shard.
5. The output wo projection proceeds on the local O shard.
SWA layers and the MoE FFN are unchanged (local). The graph splits naturally:
full-attn FA + reduce span both backends; everything else stays layer-local.

### 3.4 RPC dispatch

The sharded FA ops are assigned to RPC devices by the scheduler (buffer placement
of the KV shard). KV shard exchange and partial gather use the existing RPC
GET_TENSOR / SET_TENSOR paths with BF16 wire compression. The reduce op gathers
remote partials. No new transport needed; the existing async-copy + BF16 paths
apply. The current RPC limitation ("row split unsupported", no split buffers) is
about MUL_MAT row splitting and does not block this op-level sharding, but the
scheduler must be taught to place the sharded FA + reduce correctly.

## 4. Correctness considerations

- Online-softmax merge must be numerically stable (subtract global max before
  exp); F32 accumulation for m/L/O partials. Greedy output must match the
  non-parallel path bit-for-bit (or within F32 reduction rounding).
- Causal mask: each Q position attends only to KV positions <= it; with sharded
  KV the mask per shard is a slice of the global causal mask. Ring steps must
  apply the correct mask slice (fully-masked shards can be skipped).
- Shared-KV layers (gemma4) must read the source layer's shard map.
- KV cache management (defrag, seq_rm, context shift) must operate on the sharded
  cache consistently across boards.

## 5. Expected scaling (the decisive constraint)

Inter-board link measured: eth0 speed = 1000 Mb/s (~110 MB/s practical).

Per ubatch chunk at pp8192, gemma4 (5 full-attn layers, head_dim 512, 2 KV heads):
- Full-attn FA compute today: ~264 ms/chunk. Split across 2 boards saves ~130 ms.
- Option A partial O transfer: O is [512, 1024 tokens, 16 heads] * 4 B ~ 32 MB per
  layer; 5 layers ~ 160 MB/chunk. At 110 MB/s ~ 1.5 s. Net ~10x LOSS.
- Option B ring KV transfer: K+V shard ~ 512*8192*2*2 B*2 ~ 32 MB/layer; 5 layers
  ~ 160 MB exchanged. At 110 MB/s ~ 1.5 s. Net ~10x LOSS. BF16 halves it -> ~5x
  LOSS still.

| Interconnect | Transfer/chunk | Compute saved | Net |
|---|---|---|---|
| 1GbE (~110 MB/s) | ~1.5 s | ~130 ms | ~10x loss |
| 10GbE (~1.1 GB/s) | ~150 ms | ~130 ms | ~breakeven/slight win |
| 25GbE+ | ~60 ms | ~130 ms | clear win, scales with context |

At longer context (30k+) the FA compute saved grows O(n^2) while transfer grows
O(n), so faster links benefit more. But on 1GbE context parallelism is always
net-negative for pp. Layer-split RPC works on 1GbE only because it crosses small
activations once per ubatch, not the whole KV.

## 6. Risks and complexity

- High implementation complexity: sharded KV cache + ring schedule + new reduce
  op + graph restructuring + scheduler placement. Touches llama-kv-cache-iswa,
  llama-graph, ggml (new op), ggml-vulkan (partial m/L output), ggml-rpc.
- Stability: boards run gpu_recovery=0 / oops=panic; a hung cross-device FA
  reduce or RPC deadlock can hang the GPU queue (requires reboot, as seen in the
  FA-staging experiment). Cross-device synchronization adds deadlock surface.
- Network sensitivity: any link slowdown directly negates the gain.
- Precision: cross-device F32 reduction rounding; must validate greedy parity.
- Maintenance: a large invasive feature in a private fork; upstream ggml has no
  context-parallel RPC, so this is fork-only and must be rebased carefully.

## 7. Recommendation and phased plan

DO NOT implement on the current 1GbE cluster (net-negative). Prerequisite: a
10GbE+ inter-board link (NIC upgrade), which also speeds layer-split RPC pp.

If 10GbE+ is acquired, phased plan:
1. Prototype the reduce op + partial m/L FA output on a single device (validate
   online-softmax parity vs fa_split_k_reduce).
2. Two-device KV sharding for ONE full-attn layer, option A (replicated Q),
   measure transfer vs compute at 10GbE.
3. Generalize to all full-attn layers + ISWA shared-KV handling.
4. Switch to ring attention (option B) if partial O transfer dominates.
5. Full greedy-parity + long-context pp scaling validation; stability soak.

Estimated effort: several weeks of focused engineering. Only justified with the
faster interconnect and a concrete need for very long context pp.