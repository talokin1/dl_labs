"""Матриця плутанини, вартість C(t) = 10*FN + FP і пошук порога за мінімумом C.

Правило: прогноз "шахрайство", якщо s >= t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COST_FN = 10
COST_FP = 1
COLUMNS = ["threshold", "TP", "FP", "FN", "TN", "C"]


def cost(fn: int, fp: int) -> int:
    return COST_FN * fn + COST_FP * fp


def confusion(y: np.ndarray, s: np.ndarray, t: float) -> dict[str, int]:
    """Пряме застосування правила s >= t до всієї вибірки."""
    y = np.asarray(y)
    pred = np.asarray(s) >= t
    pos = y == 1
    tp = int(np.count_nonzero(pred & pos))
    fp = int(np.count_nonzero(pred & ~pos))
    fn = int(np.count_nonzero(~pred & pos))
    tn = int(np.count_nonzero(~pred & ~pos))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp)}


def candidate_table(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """Лічильники для всіх кандидатів за O(n log n).

    Кандидати: +inf та всі унікальні оцінки, у порядку спадання порога.
    Після сортування оцінок за спаданням поріг t = s_(i) відносить до позитивних
    рівно префікс до останнього елемента групи рівних оцінок, тому TP і FP
    беруться з кумулятивних сум саме на межах груп.
    """
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y та s мають бути одновимірними масивами однакової довжини.")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Мітки мають бути 0/1.")
    if not np.isfinite(s).all():
        raise ValueError("Оцінки мають бути скінченними.")

    order = np.argsort(-s, kind="stable")
    s_desc = s[order]
    y_desc = y[order]
    cum_tp = np.cumsum(y_desc)
    cum_fp = np.cumsum(1 - y_desc)

    # Останній індекс кожної групи рівних оцінок.
    group_end = np.flatnonzero(np.append(s_desc[1:] != s_desc[:-1], True))

    tp = np.concatenate(([0], cum_tp[group_end]))
    fp = np.concatenate(([0], cum_fp[group_end]))
    thresholds = np.concatenate(([np.inf], s_desc[group_end]))
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    fn = n_pos - tp
    tn = n_neg - fp
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "C": cost(fn, fp),
        }
    )


def select_threshold(table: pd.DataFrame) -> pd.Series:
    """Рядок з мінімальною C; за рівності — з найбільшим порогом."""
    best = table["C"].min()
    tied = table[table["C"] == best]
    return tied.loc[tied["threshold"].idxmax()]


def precision_recall(tp: int, fp: int, fn: int) -> tuple[float | None, float | None]:
    """None, якщо знаменник нульовий і метрика не визначена."""
    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    return precision, recall
