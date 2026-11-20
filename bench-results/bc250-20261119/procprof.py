#!/usr/bin/env python3
# Poor-man's profiler: /proc syscall/wchan sampling + per-thread CPU tick deltas.
# usage: procprof.py PID SECONDS
import os, sys, time, glob, collections

pid, secs = sys.argv[1], float(sys.argv[2])

def snap():
    d = {}
    for t in glob.glob(f"/proc/{pid}/task/*"):
        tid = t.split("/")[-1]
        try:
            st = open(t + "/stat").read().split()
            comm = open(t + "/comm").read().strip()
            d[tid] = (int(st[13]), int(st[14]), comm)
        except Exception:
            pass
    return d

before = snap()
cnt = collections.Counter()
n = 0
t0 = time.time()
while time.time() - t0 < secs:
    for t in glob.glob(f"/proc/{pid}/task/*"):
        try:
            sc = open(t + "/syscall").read().split()
            wc = open(t + "/wchan").read().strip() or "-"
        except Exception:
            continue
        if sc and sc[0] == "-1":
            cnt["userspace"] += 1
        else:
            cnt[f"sc{sc[0]}:{wc}"] += 1
        n += 1
    time.sleep(0.002)
after = snap()
print(f"samples={n}")
print("== syscall/wchan histogram:")
for k, v in cnt.most_common(12):
    print(f"  {v*100//max(n,1):3d}%  {k}")
print("== per-thread CPU ticks during window (utime+stime, 100Hz ticks):")
deltas = []
for tid, (u, s, comm) in after.items():
    if tid in before:
        du = u - before[tid][0]; ds = s - before[tid][1]
        if du + ds > 0:
            deltas.append((du + ds, du, ds, comm, tid))
for tot, du, ds, comm, tid in sorted(deltas, reverse=True)[:12]:
    print(f"  {tot:5d} ticks (u{du}/s{ds})  {comm} [{tid}]")
