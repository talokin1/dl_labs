"""Незалежна перевірка пошуку порога на малих наборах A, B, C.

Повільний перебір написано окремо на чистому Python і не використовує жодної
функції з threshold.py: кандидати, лічильники, вартість і вибір порога
обчислюються заново прямим застосуванням правила s >= t.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from threshold import candidate_table, select_threshold

COLUMNS = ["threshold", "TP", "FP", "FN", "TN", "C"]

SETS = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 10 + [0]),
}


def brute_force(scores: list[float], labels: list[int]) -> tuple[list[dict], float]:
    candidates = [math.inf] + sorted(set(scores), reverse=True)
    rows = []
    for t in candidates:
        tp = fp = fn = tn = 0
        for s, y in zip(scores, labels, strict=True):
            predicted = s >= t
            if predicted and y == 1:
                tp += 1
            elif predicted and y == 0:
                fp += 1
            elif not predicted and y == 1:
                fn += 1
            else:
                tn += 1
        rows.append({"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": 10 * fn + fp})
    best_t, best_c = None, None
    for row in rows:
        if best_c is None or row["C"] < best_c or (row["C"] == best_c and row["threshold"] > best_t):
            best_t, best_c = row["threshold"], row["C"]
    return rows, best_t


def run_checks(out: Path | None = None) -> bool:
    all_ok = True
    for name, (scores, labels) in SETS.items():
        slow_rows, slow_t = brute_force(scores, labels)
        slow = pd.DataFrame(slow_rows, columns=COLUMNS)
        fast = candidate_table(np.asarray(labels), np.asarray(scores))
        fast_t = float(select_threshold(fast)["threshold"])

        same_table = (
            slow.shape == fast.shape
            and (slow["threshold"].to_numpy() == fast["threshold"].to_numpy()).all()
            and all((slow[c].to_numpy() == fast[c].to_numpy()).all() for c in COLUMNS[1:])
        )
        same_t = slow_t == fast_t
        ok = bool(same_table and same_t)
        all_ok &= ok

        print(f"\nНабір {name}: s={scores}, y={labels}")
        print(slow.to_string(index=False))
        print(f"  t* перебір = {slow_t}, t* cumsum = {fast_t}")
        print(f"  таблиці збігаються: {same_table}; поріг збігається: {same_t} -> {'OK' if ok else 'FAIL'}")
        if out is not None:
            slow.assign(selected=slow["threshold"] == slow_t).to_csv(out / f"check_{name}.csv", index=False)
    return all_ok


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    passed = run_checks()
    raise SystemExit(0 if passed else 1)
