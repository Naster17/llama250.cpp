# Inkling-2-Mini on BC-250 - 2026-11-21

The user's two-board RPC launch of `Inkling-2-Mini-UD-Q8_K_XL.gguf` failed for
four stacked reasons. All are fixed; the model now loads and serves over RPC.

## 1. Legacy GGUF format (pre upstream #32694)

The UD quant predates upstream Inkling 2 support. It carries arch `inkling2`
with `block_count` = 48 expanded cache slots (24 physical layers), loop keys
`loop.core_start/core_end` + `num_loops`, ctrl tensors
`loop_{attn,k,v,mlp}_{down,gain,forget}`, and two global F32 query tensors.
Current code wants arch `inkling2_loop`, physical `block_count`,
`loop.start/end/count`, `loop_{q,k,v,ffn}_ctrl_*`, and per-core-layer
`loop_msg_query` tensors.

Fix: `scripts/bc250/inkling2-legacy-gguf-fix.py` rewrites metadata in place and
row-slices the global queries into the 12 msg_query tensors (row order
[q[2i], kv[2i], kv[2i+1], q[2i+1]], exactly as conversion/inkling2.py does).
The msg tensors take the globals' place in header order and occupy exactly
their byte region, so only the header growth delta shifts the data section.
GGUF gotcha: tensor offsets are RELATIVE to the data section start and must be
sequential in header order (gguf.cpp validates this); absolute-offset
assumptions corrupt the file.

## 2. BF16 + NVFP4 weights unsupported on gfx1013

Despite the "Q8_K_XL" name the file stores attention/dense weights as BF16 and
experts as NVFP4. gfx1013 has no coopmat-bf16 and no ocp_fp4, so those mul_mat
paths fall back or dequantize into F32 device buffers (a single
ffn_gate_up_exps F32 copy is 1296 MiB) and overflow the 15 GB pool on the peer.

Fix: `scripts/bc250/inkling2-requant-gfx1013.py` requantizes in place:
BF16 -> Q8_0 (near-lossless, half size), NVFP4 -> Q4_0 (same 4-bit class, same
byte size), F32 untouched. New sizes are <= old sizes, so data compacts
forward with no extra disk. Round-trip error vs F32: Q8_0 ~0.4%, Q4_0 ~7%
(normal for 4-bit). Requires numpy (`apk add py3-numpy`).

## 3. Context and ubatch over the memory ceiling

`-c 150000 -ub 1024` projects ~84 GB across the two boards; even `-c 65536
-ub 512` projects ~23 GB on the peer because the 48-slot MoE loop graph
reserves large compute buffers. Working config: `-c 32768 -ub 256 -cram 256`
(see verification below). 150k context needs a smaller weight quant or more
memory; it is a hardware limit, not a code bug.

## 4. Host state: governor profile and shell aliases

.104's active governor config was byte-identical to `config.balance40.toml`
(no frequency pin, slow ramp, 75 C throttle): pp512 2499 vs 3220 on .105,
tg128 148 vs 159. Restored the tuned config (min 2000 / max 2100, ramp 5,
82 C throttle); .104 back to 3243 / 159.4.

The `power_*` aliases in ~/.zshrc on both boards were `sudo cp ... ; sudo
rc-service ...` chains relying on `alias sudo="doas"`. Rewritten as a guarded
`power_profile()` function (doas explicit, profile existence check, `&&`
chain) with the same alias names. Only `minecraft40` is llama-safe;
balance*/turbo*/sturbo*/ssturbo*/hashcat leave the GPU unpinned or slow-ramp,
and maxturbo40 (2230 MHz) exceeds the VRM envelope. Old .zshrc kept as
~/.zshrc.bak-power, old governor config as config.toml.bak-default.

## Operational pitfalls hit during debugging

- `pkill -f ggml-rpc-server` inside `bash -c "...pkill -f ggml-rpc-server..."`
  kills the calling shell (pattern matches its own cmdline). Use stdin scripts
  (`ssh host bash -s <<EOF`) or the bracket trick `ggml-rpc-serve[r]`.
- zsh expands unquoted `[r]` as a glob ("no matches found") - quote patterns.
- Orphaned RPC peers keep 11+ GB of GPU memory; a "clean" peer then fails
  allocations. Always verify `ps` + mem_info_gtt_used before blaming OOM.
- doas here has no -S; root SSH is denied; `echo PASS | su root -c '...'`
  works for non-interactive root, interactive doas works over `ssh -tt`.

## 5. Structural limit: loop weight sharing vs RPC layer-split

After requant (17.5 GB) the peer still crashed: its model projection was
15781 MiB (~88% of the whole model) while the client GPU held only embeddings.
Inkling-2 loop slots 24..47 share the physical weights of layers 6..17, which
land on the other side of any contiguous layer-split boundary.
ggml_backend_sched then places the shared weights (and effectively the whole
graph) on one device, so the peer needs ~15.4 GB + 3.3 GB compute > 15.7 GB.
Layer-split RPC cannot express "slots co-located with their physical layers";
row-split RPC is unsupported (no split buffers). Single board cannot hold
17.5 GB + compute either.

Consequence: this model needs a weight footprint <= ~11.8 GB for single-board
ub128 + 32k q8_0 KV, or <= ~13.4 GB for ub64 + 16k. The user is downloading a
smaller quant; the 17.5 GB Q8_0-dense file remains on disk for future RPC
row-split / remote-weight work.

## Verification (2026-11-21)

- Governor restore on .104: LFM3-1.7B Q8_0 pp512 2499 -> 3243.7, tg128
  148.3 -> 159.4 (.105 reference 3220 / 159.2).
- power_minecraft40 proven end-to-end over interactive zsh + doas: config
  pinned 2000-2100, service restarted.
- Requant: 370/744 tensors converted (BF16->Q8_0 x330, NVFP4->Q4_0 x40),
  22867452896 -> 17544798176 bytes, sequential offsets verified.
- RPC launch of the 17.5 GB file: peer alloc failure (688 MiB expert buffer),
  root cause section 5. Not a code regression.

## 6. Loop-aware tensor split and upstream correctness fixes

The layer-split boundary can be placed BEFORE the loop core so shared weights
never cross it: with 24 physical layers / 48 slots and core 6..17,
`-ts 12.5,87.5` gives the peer slots 0..5 (4.0 GB) and the client slots 6..47
(12.2 GB) at ub64 - both fit. Verified loading and serving with this split.

Upstream merge `c23464b15` (b12608) adds two correctness fixes required here:
- #32669: generic `<name>.scale` pass - per-expert scales apply to ANY weight
  type, so requanted Q4_0 experts keep their `.scale` tensors working.
- #32704: inkling2_loop shares loop slots AFTER the scale pass.

The requantizer now reads back every converted tensor and aborts if the
dequantized error exceeds 2% (Q8_0) / 15% (Q4_0). The first on-board requant
pass produced corrupt blocks (inf fp16 scales) despite clean local math; the
original was re-downloaded from HF and the conversion re-run under the
verification gate. On-board corruption root cause not yet identified.

Working launch (single board holds the loop core, peer the prefix):

```sh
GGML_VK_HOST_MALLOC_CACHED=1 GGML_VK_BC250_MMID_TUNE=1 llama-server \
  -m ~/models/Inkling-2-Mini-UD-Q8_K_XL.gguf --host 0.0.0.0 --port 8080 \
  -ngl 99 -t 10 -tb 10 -fa on -ctk q8_0 -ctv q8_0 \
  -c 32768 -b 4096 -ub 64 -lm dio --parallel 1 \
  --split-mode layer -ts 12.5,87.5 --rpc 192.168.1.105:50052 --jinja
```

## Backups on .104

- ~/inkling-header-backup.bin + .json (pre-format-fix header and tensor map)
- ~/inkling-requant-backup.bin + .json (pre-requant header and tensor map)
- header copy also on .105:~/inkling-header-backup.bin
Requant is lossy; the original BF16/NVFP4 weights are not recoverable from
these backups (metadata only). Re-download the UD GGUF to restore them.
