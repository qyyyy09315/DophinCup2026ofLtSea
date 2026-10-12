"""Describe label-level I130 threshold cross-fit errors; no post-hoc blend selection.

Input: the immutable audit report, saved held-out decisions and fold thresholds.
Output: label_diagnostics.csv and label_summary.json in the same diagnostic dir.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rel/i130_calibration_audit_20261012"
N_LABELS = 500


def main():
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    with np.load(OUT / "heldout_decisions.npz", allow_pickle=False) as d:
        indices = d["validation_indices"]
        preds = {k: np.unpackbits(d[k], axis=1)[:, :N_LABELS].astype(bool)
                 for k in ("penta_i130", "hexa_i144")}
    columns = [f"label_{j}" for j in range(N_LABELS)]
    y = pd.read_csv(ROOT / "data/train.csv", usecols=columns).to_numpy(dtype=np.int8)[indices].astype(bool)
    data = {"label": columns, "validation_positive_count": y.sum(axis=0)}
    for k, pred in preds.items():
        tp = (pred & y).sum(axis=0)
        fp = (pred & ~y).sum(axis=0)
        fn = (~pred & y).sum(axis=0)
        data[f"{k}_f1"] = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        data[f"{k}_fp"] = fp
        data[f"{k}_fn"] = fn
        measured = float(data[f"{k}_f1"].mean())
        if not np.isclose(measured, report["pooled"][k]["macro_f1"], atol=1e-12, rtol=0):
            raise ValueError("Saved held-out decisions do not reproduce report F1")
    with np.load(OUT / "thresholds.npz", allow_pickle=False) as d:
        for k in preds:
            for stage in ("row", "final"):
                ts = np.stack([d[f"{k}_fold{j}_{stage}"] for j in range(5)])
                data[f"{k}_{stage}_threshold_mean"] = ts.mean(axis=0)
                data[f"{k}_{stage}_threshold_std"] = ts.std(axis=0)
                data[f"{k}_{stage}_threshold_range"] = np.ptp(ts, axis=0)
    df = pd.DataFrame(data)
    df["hexa_minus_penta_f1"] = df["hexa_i144_f1"] - df["penta_i130_f1"]
    df["support_quartile"] = pd.qcut(df["validation_positive_count"], q=4,
                                    labels=["Q1", "Q2", "Q3", "Q4"])
    df.to_csv(OUT / "label_diagnostics.csv", index=False, encoding="utf-8-sig")
    strata = df.groupby("support_quartile", observed=True).agg(
        labels=("label", "count"), positive_min=("validation_positive_count", "min"),
        positive_max=("validation_positive_count", "max"), penta_f1=("penta_i130_f1", "mean"),
        hexa_f1=("hexa_i144_f1", "mean"), delta=("hexa_minus_penta_f1", "mean"),
        final_threshold_std=("penta_i130_final_threshold_std", "mean"))
    summary = {
        "strata": strata.reset_index().to_dict(orient="records"),
        "penta_zero_f1_labels": int((df["penta_i130_f1"] == 0).sum()),
        "penta_f1_below_020_labels": int((df["penta_i130_f1"] < 0.20).sum()),
        "cmat_improved_labels": int((df["hexa_minus_penta_f1"] > 0).sum()),
        "cmat_worsened_labels": int((df["hexa_minus_penta_f1"] < 0).sum()),
        "penta_final_threshold_range_at_least_005_labels": int(
            (df["penta_i130_final_threshold_range"] >= 0.05).sum()),
        "lowest_penta_f1_labels": df.nsmallest(10, "penta_i130_f1")[[
            "label", "validation_positive_count", "penta_i130_f1",
            "penta_i130_fp", "penta_i130_fn", "penta_i130_final_threshold_std"
        ]].to_dict(orient="records"),
        "interpretation_limit": "Descriptive label analysis on previously inspected validation data; "
                                "do not optimize label-specific blend weights on these scores and claim independent gain.",
    }
    (OUT / "label_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
