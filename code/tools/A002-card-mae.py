"""
A002-card-mae — C011a 上半折估实例基数（分析性，CPU）
================================================================
I044 oracle 真基数 clip_k=6 +0.027，但门控分不开。
I006 在 CK005a 上线性估 k hold MAE=15。本脚本冻 C011a 分数，
半折 Ridge / 分位数回归估真基数，只报 MAE/corr，不改提交、不写腾讯表。
若 hold MAE 仍 ≥10，则 MetaLabeler 同族不再开。
用法:
  python code/tools/A002-card-mae.py
输出:
  rel/A002_report.json
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))

REPORT_PATH = os.path.join(i011.REL_DIR, "A002_report.json")
C011A_ALPHA = 0.50
CNN_W = 0.2
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
CARD_CUT = 10


def build_c011a(deep, sim, y_train, y_val):
    s011 = deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
        i011.rrf_pair(deep, sim, K_RRF))
    n = s011.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t = i011.search_t(s011, y_val)
    row_m = np.maximum((s011 >= t).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    return i019.mix(s011, p, ALPHA_SK)


def metrics(y, pred, low):
    err = np.abs(y - pred)
    return {
        "mae": round(float(err.mean()), 3),
        "mae_low": round(float(err[low].mean()), 3) if low.any() else None,
        "mae_high": round(float(err[~low].mean()), 3) if (~low).any() else None,
        "corr": round(float(np.corrcoef(y, pred)[0, 1]), 3),
        "pred_mean": round(float(pred.mean()), 3),
        "pred_mean_low": round(float(pred[low].mean()), 3) if low.any() else None,
    }


def oof_predict(X, y, folds, kind="ridge"):
    pred = np.zeros(len(y), np.float64)
    for hold, fit in ((folds[0], folds[1]), (folds[1], folds[0])):
        scaler = StandardScaler()
        xf = scaler.fit_transform(X[fit])
        xh = scaler.transform(X[hold])
        if kind == "ridge":
            m = Ridge(alpha=1.0)
            m.fit(xf, y[fit])
            pred[hold] = m.predict(xh)
        else:
            m = QuantileRegressor(quantile=0.5, alpha=1e-4, solver="highs")
            m.fit(xf, y[fit])
            pred[hold] = m.predict(xh)
    return np.clip(pred, 1.0, 80.0)


def main():
    t0 = time.time()
    print("=" * 60)
    print("A002: C011a cardinality MAE (CPU, analytical)")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    val_len = np.array([len(s) for s in train_df["sequence"].values[idx_val]], np.float32)
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    card = y_val.sum(1).astype(np.float64)
    low = card <= CARD_CUT
    print(f"  n={len(y_val)} low={int(low.sum())} miss={int(miss.sum())} "
          f"card_mean={card.mean():.2f}")

    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    cnn42 = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_val = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))["val"]
    cnn_lock = (1.0 - C011A_ALPHA) * cnn42 + C011A_ALPHA * c005_val
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + CNN_W * cnn_lock
    score = build_c011a(deep, sim, y_train, y_val)
    t_g = i011.search_t(score, y_val)
    scut = (score >= t_g).astype(np.int8)
    pcard = scut.sum(1).astype(np.float64)

    neigh_card = np.zeros(len(y_val), np.float64)
    neigh_card[hit] = y_train[idx_v[hit, 0]].sum(1)
    bit1 = bits_v[:, 0].astype(np.float64)
    bit1[miss] = 0.0
    p70 = np.quantile(score, 0.70, axis=1)
    p80 = np.quantile(score, 0.80, axis=1)

    X = np.column_stack([
        pcard, score.sum(1), np.log1p(val_len),
        (score >= 0.4).sum(1), (score >= 0.5).sum(1), (score >= 0.6).sum(1),
        p70, p80, score.max(1),
        hit.astype(np.float64), neigh_card, np.log1p(bit1),
    ])

    rng = np.random.RandomState(i011.RANDOM_STATE)
    perm = rng.permutation(len(y_val))
    mid = len(y_val) // 2
    folds = [perm[:mid], perm[mid:]]

    recs = {
        "scut_card": metrics(card, pcard, low),
        "neigh_or_scut": metrics(card, np.where(hit, neigh_card, pcard), low),
        "ridge": metrics(card, oof_predict(X, card, folds, "ridge"), low),
        "q50": metrics(card, oof_predict(X, card, folds, "q50"), low),
    }
    recs["scut_card"]["pred_mean"] = round(float(pcard.mean()), 3)
    recs["scut_card"]["pred_mean_low"] = round(float(pcard[low].mean()), 3)
    for name, rec in recs.items():
        print(f"  {name:14s} MAE={rec['mae']:.2f} low={rec['mae_low']:.2f} "
              f"high={rec['mae_high']:.2f} corr={rec['corr']:.3f} "
              f"pred_low={rec['pred_mean_low']}")

    best = min((k for k in recs if k != "scut_card"), key=lambda k: recs[k]["mae"])
    useful = bool(recs[best]["mae"] < 8.0 and recs[best]["mae_low"] < 5.0)
    report = {
        "n": int(len(y_val)),
        "n_low": int(low.sum()),
        "card_mean": round(float(card.mean()), 3),
        "card_mean_low": round(float(card[low].mean()), 3),
        "models": recs,
        "best": best,
        "useful_for_rcut": useful,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "估 k MAE 够低，可再开 MetaLabeler 式 RCut"
            if useful else "估 k MAE 仍高，MetaLabeler / How Many Labels 不再开"
        ),
    }
    os.makedirs(i011.REL_DIR, exist_ok=True)
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}")
    print(f"  用时 {report['seconds']}s  → {report['conclusion']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
