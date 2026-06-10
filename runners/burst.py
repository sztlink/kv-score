#!/usr/bin/env python3
"""Load generator only (numbers come from the engine logger, never from here).
Usage: burst.py <completions-url> <concurrency> <duration-s>"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
base, N, DUR = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
para = ("A modern out of order CPU core fetches an instruction, decodes it into micro ops, "
        "renames registers, dispatches, executes, accesses cache, and retires in program order. ") * 8
deadline = time.monotonic() + DUR
def worker(_):
    while time.monotonic() < deadline:
        b = json.dumps({"model": "spikemodel", "prompt": para, "max_tokens": 256, "temperature": 0}).encode()
        r = urllib.request.Request(base, data=b, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=900) as x:
                x.read()
        except Exception:
            pass
with ThreadPoolExecutor(max_workers=N) as ex:
    list(ex.map(worker, range(N)))
print("burst done")
