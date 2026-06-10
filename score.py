#!/usr/bin/env python3
"""kv-score: Q/T/C axes + geometric-mean composite, relative to the fp16 row.

Input CSV columns: config,router_exact,router_total,tput_n128,kv_tokens
The row named 'fp16' is the reference. Stdlib only.
"""
import csv
import sys


def main(path: str) -> None:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    ref = next((r for r in rows if r["config"] == "fp16"), None)
    if ref is None:
        sys.exit("error: no fp16 reference row")
    rq = int(ref["router_exact"]) / int(ref["router_total"])
    rt = float(ref["tput_n128"])
    rc = float(ref["kv_tokens"])

    print(f"{'config':16} {'Q':>7} {'T':>7} {'C':>7}  {'KV-Score':>8}")
    for r in rows:
        q = (int(r["router_exact"]) / int(r["router_total"])) / rq * 100
        t = float(r["tput_n128"]) / rt * 100
        c = float(r["kv_tokens"]) / rc * 100
        score = (q * t * c) ** (1 / 3)
        print(f"{r['config']:16} {q:7.1f} {t:7.1f} {c:7.1f}  {score:8.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/results.csv")
