#!/usr/bin/env python3
"""Requantize an Inkling-2 GGUF for gfx1013 (BC-250) Vulkan scalar path.

The UD "Q8_K_XL" Inkling-2 files store attention/dense weights as BF16 and
experts as NVFP4. gfx1013 has no coopmat-bf16 and no ocp_fp4, so those mul_mat
paths fall back or dequantize to F32 device buffers and overflow the 15 GB
pool. BF16 -> Q8_0 is near-lossless (8 mantissa bits -> 7-8 effective) and
halves the size; NVFP4 -> Q4_0 keeps the same 4-bit class and byte size.
F32 tensors are copied untouched. Data is compacted in place (new sizes are
<= old sizes), so no extra disk space is needed.

Usage: inkling2-requant-gfx1013.py MODEL.gguf HEADER_BACKUP.bin
"""
import json
import os
import struct
import sys

import numpy as np

ALIGN = 32
E2M1x2 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)


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


def align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def ue4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for x in range(256):
        if x == 0 or x == 0x7F:
            continue
        e, m = (x >> 3) & 0xF, x & 7
        raw = m * 2.0 ** -9 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[x] = raw * 0.5
    return t


UE4M3 = ue4m3_table()


def nvfp4_to_f32(raw, ne):
    nb = ne // 64
    b = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 36)
    d = UE4M3[b[:, :4]]
    qs = b[:, 4:].reshape(nb, 4, 8)
    lo = E2M1x2[qs & 0x0F]
    hi = E2M1x2[qs >> 4]
    out = np.empty((nb, 4, 16), dtype=np.float32)
    out[:, :, :8] = lo * d[:, :, None]
    out[:, :, 8:] = hi * d[:, :, None]
    return out.reshape(-1)


def bf16_to_f32(raw):
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def f32_to_q8_0(x):
    nb = x.size // 32
    xb = x.reshape(nb, 32)
    amax = np.abs(xb).max(axis=1)
    d = amax / 127.0
    id_ = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0)
    q = np.sign(xb) * np.floor(np.abs(xb) * id_[:, None] + 0.5)
    q = np.clip(q, -128, 127).astype(np.int8)
    out = np.empty((nb, 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(nb, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def f32_to_q4_0(x):
    nb = x.size // 32
    xb = x.reshape(nb, 32)
    amax = np.abs(xb).max(axis=1)
    mx = xb[np.arange(nb), np.abs(xb).argmax(axis=1)]
    d = mx / -8.0
    id_ = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0)
    v = xb * id_[:, None] + 8.5
    xi = np.minimum(15, np.floor(v)).astype(np.uint8)
    out = np.empty((nb, 18), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(nb, 2)
    out[:, 2:] = xi[:, :16] | (xi[:, 16:] << 4)
    return out.tobytes()


def q8_0_deq(raw):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    return (b[:, 2:].view(np.int8).astype(np.float32) * d[:, None]).reshape(-1)


def q4_0_deq(raw):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
    d = b[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    out = np.empty((b.shape[0], 32), dtype=np.float32)
    out[:, :16] = ((b[:, 2:] & 0xF).astype(np.float32) - 8) * d[:, None]
    out[:, 16:] = ((b[:, 2:] >> 4).astype(np.float32) - 8) * d[:, None]
    return out.reshape(-1)


def main(src, backup):
    f = open(src, "r+b")
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
        n = 1
        for d in dims:
            n *= d
        tensors.append({"name": name, "dims": dims, "ty": ty, "off": off, "n": n})
    s0 = align(f.tell())
    with open(backup, "wb") as b:
        f.seek(0)
        b.write(f.read(s0))
    with open(backup + ".json", "w") as j:
        json.dump([{"name": t["name"], "ty": t["ty"], "off": t["off"]} for t in tensors], j)

    TR = {0: (1, 4), 1: (1, 2), 30: (1, 2), 40: (64, 36), 2: (32, 18), 8: (32, 34)}
    for t in tensors:
        bl, bs = TR[t["ty"]]
        t["size"] = t["n"] // bl * bs
        if t["ty"] == 30:
            t["new_ty"], t["new_size"] = 8, t["n"] // 32 * 34
        elif t["ty"] == 40:
            t["new_ty"], t["new_size"] = 2, t["n"] // 32 * 18
        else:
            t["new_ty"], t["new_size"] = t["ty"], t["size"]
    pos = 0
    for t in tensors:
        t["new_off"] = pos
        pos = align(pos + t["new_size"])

    import io
    hdr = io.BytesIO()
    hdr.write(struct.pack("<4sIQQ", b"GGUF", ver, nt, nk))
    for k, t, v in kv:
        if k == "general.file_type":
            v = 7
        ws(hdr, k)
        hdr.write(struct.pack("<I", t))
        write_val(hdr, t, v)
    for t in tensors:
        ws(hdr, t["name"])
        hdr.write(struct.pack("<I", len(t["dims"])))
        hdr.write(struct.pack("<%dQ" % len(t["dims"]), *t["dims"]))
        hdr.write(struct.pack("<I", t["new_ty"]))
        hdr.write(struct.pack("<Q", t["new_off"]))
    header = hdr.getvalue()
    s0_new = align(len(header))

    for t in tensors:
        f.seek(s0 + t["off"])
        raw = f.read(t["size"])
        if t["ty"] == 30:
            x = bf16_to_f32(raw)
            new = f32_to_q8_0(x)
        elif t["ty"] == 40:
            x = nvfp4_to_f32(raw, t["n"])
            new = f32_to_q4_0(x)
        else:
            x = None
            new = raw
        assert len(new) == t["new_size"], (t["name"], len(new), t["new_size"])
        f.seek(s0_new + t["new_off"])
        f.write(new)
        if x is not None:
            f.seek(s0_new + t["new_off"])
            deq = q8_0_deq(f.read(len(new))) if t["ty"] == 30 else q4_0_deq(f.read(len(new)))
            lim = 0.02 if t["ty"] == 30 else 0.15
            err = float(np.abs(deq - x).max()) / max(float(np.abs(x).max()), 1e-6)
            assert err <= lim, (t["name"], err)
    f.seek(0)
    f.write(header)
    f.write(b"\0" * (s0_new - len(header)))
    f.truncate(s0_new + pos)
    f.close()
    nq = sum(1 for t in tensors if t["ty"] in (30, 40))
    print("requant %s: %d/%d tensors converted, %d -> %d bytes" %
          (src, nq, nt, s0 + sum(t["size"] for t in tensors), s0_new + pos))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
