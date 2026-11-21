#!/usr/bin/env python3
"""Convert legacy Inkling-2 loop GGUFs to the current inkling2_loop format.

Legacy UD quants predating upstream #32694 carry: arch "inkling2", block_count
= expanded cache slots, loop keys "loop.core_start/core_end" + "num_loops",
ctrl tensors "loop_{attn,k,v,mlp}_{down,gain,forget}", and two global F32
query tensors. Current code wants arch "inkling2_loop", block_count = physical
layers, "loop.start/end/count", "loop_{q,k,v,ffn}_ctrl_*" and per-core-layer
"loop_msg_query" tensors.

Weights are copied bit-exact. The global queries are row-sliced into the
per-layer msg_query tensors exactly as conversion/inkling2.py does:
row order [q[2i], kv[2i], kv[2i+1], q[2i+1]]. The 12 msg_query tensors occupy
exactly the byte region of the two global queries and take their place in
header order, so GGUF sequential relative offsets stay valid and --inplace
only shifts the data section by the header growth delta.

Usage:
  inkling2-legacy-gguf-fix.py INPUT.gguf OUTPUT.gguf
  inkling2-legacy-gguf-fix.py INPUT.gguf --inplace HEADER_BACKUP.bin
"""
import io
import os
import struct
import sys
ALIGN = 32
# ggml type -> (block elems, block bytes)
TRAITS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
          8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
          13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74),
          18: (256, 98), 19: (256, 50), 20: (256, 56), 21: (32, 18), 22: (256, 110),
          23: (256, 82), 24: (256, 136), 30: (1, 2), 34: (256, 54), 35: (256, 66),
          39: (32, 17), 40: (64, 36)}

CHUNK = 1 << 26


def rs(f):
    n, = struct.unpack("<Q", f.read(8))
    return f.read(n).decode()


def ws(f, s):
    b = s.encode()
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
       7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_val(f, t):
    if t == 8:
        return rs(f)
    if t == 9:
        et, n = struct.unpack("<IQ", f.read(12))
        return (et, [read_val(f, et) for _ in range(n)])
    return struct.unpack(FMT[t], f.read(struct.calcsize(FMT[t])))[0]


def write_val(f, t, v):
    if t == 8:
        ws(f, v)
    elif t == 9:
        et, vals = v
        f.write(struct.pack("<IQ", et, len(vals)))
        for x in vals:
            write_val(f, et, x)
    else:
        f.write(struct.pack(FMT[t], v))


def nbytes(dims, ty):
    blck, ts = TRAITS[ty]
    n = 1
    for d in dims:
        n *= d
    assert n % blck == 0, (dims, ty)
    return n // blck * ts


def align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def parse(f):
    magic, ver, nt, nk = struct.unpack("<4sIQQ", f.read(24))
    assert magic == b"GGUF"
    kv = []
    for _ in range(nk):
        k = rs(f)
        t, = struct.unpack("<I", f.read(4))
        kv.append([k, t, read_val(f, t)])
    tensors = []
    for _ in range(nt):
        name = rs(f)
        nd, = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
        ty, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        tensors.append({"name": name, "dims": dims, "ty": ty, "off": off,
                        "size": nbytes(dims, ty)})
    return ver, kv, tensors, align(f.tell())


def transform(f, kv, tensors, s0):
    old = dict((k, v) for k, t, v in kv)
    assert old["general.architecture"] == "inkling2"
    start = old["inkling2.loop.core_start"]
    end = old["inkling2.loop.core_end"]
    passes = old["inkling2.num_loops"]
    n_phys = old["inkling2.block_count"] - (passes - 1) * (end - start)
    assert n_phys == max(int(t["name"].split(".")[1]) for t in tensors
                         if t["name"].startswith("blk.")) + 1

    out_kv = []
    for k, t, v in kv:
        if k == "general.architecture":
            v = "inkling2_loop"
        elif k.startswith("inkling2."):
            k = "inkling2_loop." + k[len("inkling2."):]
        if k == "inkling2_loop.block_count":
            v = n_phys
        elif k == "inkling2_loop.num_loops":
            k = "inkling2_loop.loop.count"
        elif k == "inkling2_loop.loop.core_start":
            k = "inkling2_loop.loop.start"
        elif k == "inkling2_loop.loop.core_end":
            k = "inkling2_loop.loop.end"
        elif k in ("inkling2_loop.attention.head_count",
                   "inkling2_loop.attention.head_count_kv",
                   "inkling2_loop.attention.sliding_window_pattern"):
            v = (v[0], v[1][:n_phys])
        out_kv.append([k, t, v])
    out_kv.append(["inkling2_loop.loop.varying", 7, False])

    gnames = ("loop_res_query.weight", "loop_kv_res_query.weight")
    glob = [t for t in tensors if t["name"] in gnames]
    assert len(glob) == 2
    q = [t for t in glob if t["name"] == gnames[0]][0]
    kvq = [t for t in glob if t["name"] == gnames[1]][0]
    assert q["ty"] == 0 and kvq["ty"] == 0
    assert q["dims"][1] == 2 * (end - start)
    row = q["dims"][0] * 4
    f.seek(s0 + q["off"])
    qdata = f.read(q["size"])
    f.seek(s0 + kvq["off"])
    kvdata = f.read(kvq["size"])
    # the two globals must be adjacent so their region fits the msg tensors
    gmin = min(q["off"], kvq["off"])
    assert max(q["off"] + q["size"], kvq["off"] + kvq["size"]) == gmin + q["size"] + kvq["size"]
    msgs = []
    for i in range(end - start):
        payload = b"".join((qdata[2 * i * row:(2 * i + 1) * row],
                            kvdata[2 * i * row:(2 * i + 1) * row],
                            kvdata[(2 * i + 1) * row:(2 * i + 2) * row],
                            qdata[(2 * i + 1) * row:(2 * i + 2) * row]))
        msgs.append({"name": "blk.%d.loop_msg_query.weight" % (start + i),
                     "dims": (q["dims"][0], 4), "ty": 0,
                     "off": gmin + i * len(payload),
                     "size": len(payload), "payload": payload})
    assert sum(m["size"] for m in msgs) == q["size"] + kvq["size"]

    ordered = []
    for t in tensors:
        if t in glob:
            if t["off"] == gmin:
                ordered.extend(msgs)
            continue
        n = t["name"]
        for a, b in ((".loop_attn_", ".loop_q_ctrl_"), (".loop_k_", ".loop_k_ctrl_"),
                     (".loop_v_", ".loop_v_ctrl_"), (".loop_mlp_", ".loop_ffn_ctrl_")):
            n = n.replace(a, b)
        ordered.append(dict(t, name=n))
    return out_kv, ordered


def header_bytes(ver, out_kv, ordered):
    hdr = io.BytesIO()
    hdr.write(struct.pack("<4sIQQ", b"GGUF", ver, len(ordered), len(out_kv)))
    for k, t, v in out_kv:
        ws(hdr, k)
        hdr.write(struct.pack("<I", t))
        write_val(hdr, t, v)
    for t in ordered:
        ws(hdr, t["name"])
        hdr.write(struct.pack("<I", len(t["dims"])))
        hdr.write(struct.pack("<%dQ" % len(t["dims"]), *t["dims"]))
        hdr.write(struct.pack("<I", t["ty"]))
        hdr.write(struct.pack("<Q", t["off"]))
    return hdr.getvalue()


def check_sequential(ordered):
    pos = 0
    for t in ordered:
        assert t["off"] == pos, (t["name"], t["off"], pos)
        pos = align(pos + t["size"])


def copy_mode(src, dst):
    f = open(src, "rb")
    ver, kv, tensors, s0 = parse(f)
    out_kv, ordered = transform(f, kv, tensors, s0)
    for t in ordered:
        t["src"] = t["off"]
    pos = 0
    for t in ordered:
        t["off"] = pos
        pos = align(pos + t["size"])
    check_sequential(ordered)
    header = header_bytes(ver, out_kv, ordered)
    with open(dst, "wb") as o:
        o.write(header)
        o.write(b"\0" * (align(len(header)) - len(header)))
        for t in ordered:
            if "payload" in t:
                o.write(t["payload"])
            else:
                f.seek(s0 + t["src"])
                left = t["size"]
                while left:
                    b = f.read(min(left, CHUNK))
                    o.write(b)
                    left -= len(b)
            o.write(b"\0" * (align(t["size"]) - t["size"]))
    print("wrote %s: %d tensors, %d data bytes" % (dst, len(ordered), pos))


def inplace_mode(src, backup):
    f = open(src, "r+b")
    ver, kv, tensors, s0 = parse(f)
    out_kv, ordered = transform(f, kv, tensors, s0)
    check_sequential(ordered)
    header = header_bytes(ver, out_kv, ordered)
    s_new = align(len(header))
    delta = s_new - s0
    assert delta >= 0
    size = os.path.getsize(src)
    with open(backup, "wb") as b:
        f.seek(0)
        b.write(f.read(s0))
    # shift data backwards from the tail so overlapping copies stay intact
    end = size
    while end > s0:
        start = max(s0, end - CHUNK)
        f.seek(start)
        b = f.read(end - start)
        f.seek(start + delta)
        f.write(b)
        end = start
    for t in ordered:
        if "payload" in t:
            f.seek(s_new + t["off"])
            f.write(t["payload"])
    f.seek(0)
    f.write(header)
    f.write(b"\0" * (s_new - len(header)))
    f.truncate(size + delta)
    f.close()
    print("inplace %s: delta %d, %d tensors, backup %s" %
          (src, delta, len(ordered), backup))


if __name__ == "__main__":
    if sys.argv[2] == "--inplace":
        inplace_mode(sys.argv[1], sys.argv[3])
    else:
        copy_mode(sys.argv[1], sys.argv[2])
