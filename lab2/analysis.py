"""ЛР2, пункти 2-4: аналіз порогів зі збережених прогнозів, без навчання.

    uv run --locked analysis.py --saved results --out report

Порядок: перевірки A/B/C -> вибір t* на validation -> оцінка t* на test.
Test не впливає ні на поріг, ні на модель.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from checks import run_checks
from lab2_starter import load_predictions, split_summary, write_json
from threshold import candidate_table, confusion, precision_recall, select_threshold

N_EXAMPLES = 5


def fmt(x: float | None, digits: int = 4) -> str:
    return "не визначено" if x is None else f"{x:.{digits}f}"


def plot_curves(y: np.ndarray, s: np.ndarray, t_star: float, out: Path) -> dict[str, float]:
    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    precision, recall, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)

    marks = {}
    for label, t in (("t = 0.5", 0.5), (f"t* = {t_star:.4g}", t_star)):
        m = confusion(y, s, t)
        p, r = precision_recall(m["TP"], m["FP"], m["FN"])
        marks[label] = (p, r, m["FP"] / (m["FP"] + m["TN"]))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, drawstyle="steps-post", label=f"PR (AP = {ap:.4f})")
    ax.axhline(y.mean(), color="grey", ls="--", lw=1, label=f"частка шахрайства = {y.mean():.4f}")
    for (label, (p, r, _)), marker in zip(marks.items(), "os", strict=True):
        if p is not None:
            ax.plot(r, p, marker, ms=8, label=f"{label}: P={p:.3f}, R={r:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="PR-крива (validation)", xlim=(0, 1.01), ylim=(0, 1.02))
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pr_curve_validation.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, xmax in zip(axes, (1.0, 0.01), strict=True):
        ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
        for (label, (_, r, f)), marker in zip(marks.items(), "os", strict=True):
            ax.plot(f, r, marker, ms=8, label=f"{label}: FPR={f:.5f}, TPR={r:.3f}")
        ax.set(xlabel="FPR", ylabel="TPR", xlim=(0, xmax), ylim=(0, 1.02))
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    axes[0].set_title("ROC-крива (validation)")
    axes[1].set_title("ROC, збільшено: FPR ≤ 0.01")
    fig.tight_layout()
    fig.savefig(out / "roc_curve_validation.png", dpi=150)
    plt.close(fig)
    return {"roc_auc": float(auc), "average_precision": float(ap)}


def error_examples(data: dict[str, np.ndarray], t_star: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(
        {
            "csv_row": data["test_row_ids"],
            "label": data["test_y"],
            "score": data["test_scores"].astype(np.float64),
            "amount": data["test_amount"],
        }
    )
    pred = df["score"] >= t_star
    df["error"] = np.select([pred & (df["label"] == 0), ~pred & (df["label"] == 1)], ["FP", "FN"], "")
    df["distance_to_threshold"] = (df["score"] - t_star).abs()
    errors = df[df["error"] != ""].sort_values("csv_row")
    table = pd.concat(
        [errors[errors["error"] == kind].head(N_EXAMPLES) for kind in ("FP", "FN")], ignore_index=True
    )
    table = table[["csv_row", "error", "label", "score", "amount", "distance_to_threshold"]]

    stats = []
    for name, mask in (
        ("всі FP", df["error"] == "FP"),
        ("всі FN", df["error"] == "FN"),
        ("TN", ~pred & (df["label"] == 0)),
        ("TP", pred & (df["label"] == 1)),
    ):
        part = df[mask]
        stats.append(
            {
                "group": name,
                "n": len(part),
                "amount_median": part["amount"].median() if len(part) else None,
                "amount_mean": part["amount"].mean() if len(part) else None,
                "distance_median": part["distance_to_threshold"].median() if len(part) else None,
            }
        )
    return table, pd.DataFrame(stats)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saved", type=Path, default=Path("results"), help="Тека з predictions.npz")
    parser.add_argument("--out", type=Path, default=Path("report"), help="Тека для таблиць і графіків")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)

    data = load_predictions(args.saved)
    summary = split_summary(data)
    summary.to_csv(out / "split_summary.csv", index=False)
    print("== Поділ ==")
    print(summary.to_string(index=False))

    # Пункт 3: перевірки виконуються до будь-якої роботи з test.
    print("\n== Перевірка пошуку порога на наборах A/B/C ==")
    if not run_checks(out):
        raise SystemExit("Перевірки не пройдено: тестову оцінку не виконано.")

    # Пункт 2: вибір порога лише на validation.
    y_va = data["validation_y"]
    s_va = data["validation_scores"].astype(np.float64)
    table = candidate_table(y_va, s_va)
    table.to_csv(out / "validation_candidates.csv", index=False)
    best = select_threshold(table)
    t_star = float(best["threshold"])
    write_json(
        out / "threshold.json",
        {
            "t_star": t_star,
            "t_star_hex": t_star.hex(),
            "rule": "fraud if s >= t_star",
            "cost": "C = 10*FN + FP",
            "chosen_on": "validation",
            "tie_break": "largest threshold among minimal C",
        },
    )

    rules = [("Стандартний поріг", 0.5), ("Мінімум вартості", t_star), ("Усі негативні", np.inf)]
    val_rows = [{"rule": name, "threshold": t, **confusion(y_va, s_va, t)} for name, t in rules]
    val_summary = pd.DataFrame(val_rows)
    val_summary.to_csv(out / "validation_summary.csv", index=False)
    curves = plot_curves(y_va, s_va, t_star, out)
    n_min = int((table["C"] == best["C"]).sum())
    print(f"\n== Validation: {len(table)} кандидатів, ROC-AUC = {curves['roc_auc']:.6f}, AP = {curves['average_precision']:.6f} ==")
    print(f"t* = {t_star!r} (кандидатів з мінімальною C: {n_min})")
    print(val_summary.to_string(index=False))

    # Пункт 4: фіксований t* на test.
    y_te = data["test_y"]
    s_te = data["test_scores"].astype(np.float64)
    test_rows = [{"rule": name, "threshold": t, **confusion(y_te, s_te, t)} for name, t in rules]
    for row in test_rows:
        row["precision"], row["recall"] = precision_recall(row["TP"], row["FP"], row["FN"])
    test_summary = pd.DataFrame(test_rows)
    test_summary.to_csv(out / "test_summary.csv", index=False)
    m = test_rows[1]
    write_json(out / "test_metrics.json", {"t_star": t_star, **{k: m[k] for k in ("TP", "FP", "FN", "TN", "C", "precision", "recall")}})
    print("\n== Test (t* зафіксовано на validation) ==")
    print(f"TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']} C={m['C']}")
    print(f"precision={fmt(m['precision'])} recall={fmt(m['recall'])}")
    print("Для порівняння (не для вибору порога):")
    print(test_summary.to_string(index=False))

    examples, stats = error_examples(data, t_star)
    examples.to_csv(out / "test_errors.csv", index=False)
    stats.to_csv(out / "test_error_stats.csv", index=False)
    print("\n== Перші FP і FN на test за рядком CSV ==")
    print(examples.to_string(index=False))
    print("\n== Amount і відстань до порога за групами ==")
    print(stats.to_string(index=False))
    print(f"\nРезультати збережено: {out.resolve()}")


if __name__ == "__main__":
    main()
