#!/usr/bin/env python3
"""BC-250 .104 r3 GPU readiness: load amdgpu, apply SMU-OC, restart governor, ACK boot guard.

Runs at boot on the r3 kernel (amdgpu is blacklisted there). Only acknowledges the
boot guard after amdgpu loads cleanly, so a failed load still lets the guard revert.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

KREL = "7.2.10-bc250rocm-r3"
GUARD = Path("/run/bc250-r3-guard")
LOG = Path("/var/log/bc250-r3-gpu-ready.log")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def main():
    import os
    if os.uname().release != KREL:
        log(f"SKIP: kernel {os.uname().release} != {KREL}")
        return 0
    if os.geteuid() != 0:
        log("ERROR: must run as root")
        return 1

    # Wait for the boot guard to arm.
    state = None
    for _ in range(120):
        try:
            state = json.loads((GUARD / "state.json").read_text())
            if state.get("status") == "armed":
                break
        except (OSError, ValueError):
            pass
        time.sleep(0.5)
    if not state or state.get("status") != "armed":
        log("ERROR: boot guard not armed; not touching amdgpu")
        return 1
    if state["deadline_monotonic"] - time.monotonic() < 30:
        log("ERROR: too little guard deadline left")
        return 1

    # Load the custom amdgpu module.
    if not Path("/sys/module/amdgpu").exists():
        r = run(["modprobe", "amdgpu"], timeout=90)
        if r.returncode != 0:
            log(f"ERROR: modprobe amdgpu failed: {r.stderr.strip()[:300]}")
            return 1
    if not Path("/sys/module/amdgpu").exists():
        log("ERROR: amdgpu not present after modprobe")
        return 1
    log("amdgpu loaded")

    # Apply SMU-OC (undervolt + boost ceiling) and restart the governor now that amdgpu is up.
    r = run(["/home/nik/bc250_smu_oc/.venv/bin/bc250-apply", "-a", "/etc/bc250-smu-oc.conf"], timeout=60)
    log(f"smu-oc: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[:120]}")
    r = run(["rc-service", "cyan-skillfish-governor-smu", "restart"], timeout=60)
    log(f"governor restart rc={r.returncode}")

    # Only ACK once amdgpu is confirmed loaded.
    if state["deadline_monotonic"] - time.monotonic() < 10:
        log("ERROR: guard deadline too close; not ACKing")
        return 1
    (GUARD / "ack.json").write_text(
        json.dumps({k: state[k] for k in ("boot_id", "nonce", "phase")})
    )
    for _ in range(25):
        if json.loads((GUARD / "state.json").read_text())["status"] == "acknowledged":
            break
        time.sleep(0.2)
    acked = json.loads((GUARD / "state.json").read_text())["status"] == "acknowledged"
    log(f"boot guard acknowledged={acked}")
    return 0 if acked else 1


if __name__ == "__main__":
    sys.exit(main())