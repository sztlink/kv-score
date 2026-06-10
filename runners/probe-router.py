#!/usr/bin/env python3
"""Q axis: action-trace exact-match against an OpenAI-compatible endpoint.

Usage: probe-router.py <completions-url> [limit] [cases.jsonl]
Each case requires emitting `FINAL_ACTION=...; FINAL_TARGET=...; SOURCE_RANK=n`
exactly. Greedy decode, 8-way concurrency, errors counted separately (a timeout
is not a wrong answer). Stdlib only.
"""
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001/v1/completions"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 120
CASES = sys.argv[3] if len(sys.argv) > 3 else "data/router_cases.jsonl"

cases = [json.loads(l) for l in open(CASES)][:LIMIT]
errs = [0]

PAT = re.compile(
    r"FINAL_ACTION\s*=\s*([A-Za-z_]+)\s*;\s*FINAL_TARGET\s*=\s*([A-Za-z0-9_:/\.\-]+)"
    r"\s*;\s*SOURCE_RANK\s*=\s*(\d+)"
)


def parse(text):
    raw = "FINAL_ACTION=" + text
    raw = re.split(r"<\|im_end\|>|\n\s*\n", raw)[0]
    m = PAT.search(raw)
    return (m.group(1), m.group(2), int(m.group(3))) if m else (None, None, None)


def ask(prompt):
    body = json.dumps({
        "model": "spikemodel", "prompt": prompt, "max_tokens": 56,
        "temperature": 0, "stop": ["<|im_end|>"],
    }).encode()
    req = urllib.request.Request(BASE, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())["choices"][0].get("text", "")
    except Exception:
        errs[0] += 1
        return "ERR"


def work(case):
    a, t, rk = parse(ask(case["prompt"]))
    return (a == case["expected_action"] and t == case["expected_target"]
            and rk == case["expected_rank"])


with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(work, cases))

n = len(results)
print(f"ROUTER exact={sum(results)}/{n} ({sum(results)/n*100:.0f}%) errors={errs[0]}/{n}")
