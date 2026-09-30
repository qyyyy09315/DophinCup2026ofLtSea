"""
TCK004-ensemble — 六路集成检验 (+R001 BiLSTM 概率)
===================================================
方案:
  - 基座: TCK003 最优权重 (t001a=0.1, t002=0.2, cnn=0.2, sim=0.5, kmer=0)
  - 检验 R001 边际贡献: w_r ∈ {0, .05, ..., .30}, 基座按 (1-w_r) 等比缩放
  - 选权: 固定阈值 0.5 F1 → 最优 w_r → 100 档阈值 → 提交
  - 附加: R001 概率与各路的按标签 Pearson 相关 (多样性报告)
输出:
  submit/submission_TCK004_ensemble.csv
  rel/TCK004_ensemble_val_metrics.json
  models/tck004-ensemble.json
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUB_PATH = f"{SUB_DIR}/submission_TCK004_ensemble.csv"
METRICS_PATH = f"{REL_DIR}/TCK004_ensemble_val_metrics.json"
ENSEMBLE_CONFIG_PATH = f"{MODEL_DIR}/tck004-ensemble.json"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
NEIGH_K6_PATH = f"{REL_DIR}/CK002_neigh_k6.npz"
NEIGH_FULL_PATH = f"{REL_DIR}/TCK002a_neigh_full.npz"
T001A_PROBS_PATH = f"{REL_DIR}/TCK001_t001a_probs.npz"
T002_VAL_PROBS_PATH = f"{REL_DIR}/T002_val_probs.npz"
T002_TEST_PROBS_PATH = f"{REL_DIR}/T002_test_probs.npz"
KMER_PROBS_PATH = f"{REL_DIR}/TCK003_kmer_probs.npz"
R001_PROBS_PATH = f"{REL_DIR}/TCK004_r001_probs.npz"

BASE_W = {"t001a": 0.1, "t002": 0.2, "cnn": 0.2, "sim": 0.5, "kmer": 0.0}
R_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
THRESHOLD_STEPS = 100
VAL_SIZE = 0.2
RANDOM_STATE = 42


def macro_auc(y_true, y_prob):
    aucs = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, y_prob[:, i]))
    return float(np.mean(aucs))


def macro_f1_fixed_threshold(ens, y_true, t=0.5):
    pred = (ens >= t)
    f1s = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def evaluate_at_thresholds(y_prob, y_true, thresholds):
    per_f1, per_prec, per_rec = [], [], []
    y_pred = np.zeros_like(y_prob, dtype=np.int8)
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        p = (y_prob[:, i] >= thresholds[i]).astype(int)
        y_pred[:, i] = p
        per_f1.append(f1_score(y_true[:, i], p, zero_division=0))
        per_prec.append(precision_score(y_true[:, i], p, zero_division=0))
        per_rec.append(recall_score(y_true[:, i], p, zero_division=0))
    return y_pred, np.mean(per_f1), np.mean(per_prec), np.mean(per_rec)


def find_optimal_thresholds(y_prob, y_true):
    print("  搜索逐标签最优阈值 (100 档)...")
    n_labels = y_true.shape[1]
    best_thresholds = np.full(n_labels, 0.5, dtype=np.float32)
    candidates = np.linspace(0.01, 0.99, THRESHOLD_STEPS)
    for i in range(n_labels):
        yt = y_true[:, i]
        if yt.sum() == 0:
            continue
        best_f1, best_t = 0, 0.5
        for t in candidates:
            pred = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(yt, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[i] = best_t
    return best_thresholds


def predict_with_thresholds(y_prob, thresholds):
    preds = np.zeros_like(y_prob, dtype=np.int8)
    for i, t in enumerate(thresholds):
        preds[:, i] = (y_prob[:, i] >= t).astype(np.int8)
    return preds


def build_sim_prob(neigh_idx, neigh_sim, y_train, top_k, mode):
    n = neigh_idx.shape[0]
    labels = y_train[neigh_idx[:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
    if mode == "mean":
        return labels.mean(axis=1)
    w = np.maximum(neigh_sim[:, :top_k], 1e-7)
    return (labels * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]


def main():
    t0 = time.time()
    print("=" * 60)

    print("[1/6] 读取数据与六路概率...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)),
                                          test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train = y_full[idx_train]

    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, val_true = vp["cnn"], vp["y_true"]
    cnn_test = tp["cnn"]

    d6 = np.load(NEIGH_K6_PATH)
    sim_val = build_sim_prob(d6["idx_val"], d6["sim_val"], y_train, 3, "sim_weighted")
    dfull = np.load(NEIGH_FULL_PATH)
    sim_test = build_sim_prob(dfull["idx_test"], dfull["sim_test"], y_full, 3, "sim_weighted")

    t1 = np.load(T001A_PROBS_PATH)
    t001a_val, t001a_test = t1["t001a_val"], t1["t001a_test"]
    t2v = np.load(T002_VAL_PROBS_PATH)
    t2t = np.load(T002_TEST_PROBS_PATH)
    t002_val, t002_test = t2v["t002"], t2t["t002"]
    km = np.load(KMER_PROBS_PATH)
    kmer_val, kmer_test = km["kmer_val"], km["kmer_test"]
    r1 = np.load(os.path.join(REL_DIR, "R001_embeddings.npz"))
    # R001 概率: 缓存或从模型提取
    if os.path.exists(R001_PROBS_PATH):
        rp = np.load(R001_PROBS_PATH)
        r001_val, r001_test = rp["r001_val"], rp["r001_test"]
        print(f"  R001 概率读缓存: {r001_val.shape}")
    else:
        import importlib.util, sys, torch
        spec = importlib.util.spec_from_file_location(
            "r001", os.path.join(PROJECT_ROOT, "code", "R", "R001-BiLSTM.py"))
        rmod = importlib.util.module_from_spec(spec)
        sys.modules["r001"] = rmod
        spec.loader.exec_module(rmod)
        rmodel = rmod.ProteinLSTM(n_labels=len(label_cols)).to(rmod.DEVICE)
        ck = torch.load(os.path.join(MODEL_DIR, "r001-bilstm.pt"),
                        map_location=rmod.DEVICE, weights_only=True)
        rmodel.load_state_dict(ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
        rmodel.eval()
        val_ds = rmod.ProteinDataset(train_df["sequence"].values[idx_val],
                                     y_full[idx_val], rmod.MAX_LEN)
        test_ds = rmod.ProteinDataset(test_df["sequence"].values,
                                      np.zeros((len(test_df), len(label_cols)), dtype=np.int8),
                                      rmod.MAX_LEN)
        vl = torch.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False,
                                         pin_memory=True, num_workers=0)
        tl = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False,
                                         pin_memory=True, num_workers=0)
        r001_val, _ = rmod.get_probs(rmodel, vl)
        r001_test, _ = rmod.get_probs(rmodel, tl)
        np.savez_compressed(R001_PROBS_PATH, r001_val=r001_val, r001_test=r001_test)
        print(f"  R001 概率已缓存: val={r001_val.shape}, test={r001_test.shape}")
    print(f"  R001 概率: val={r001_val.shape}, test={r001_test.shape}")

    print("[2/6] R001 与各路相关性...")
    others = {"t001a": t001a_val, "t002": t002_val, "cnn": cnn_val,
              "sim": sim_val, "kmer": kmer_val}
    for name, p in others.items():
        corrs = [np.corrcoef(r001_val[:, i], p[:, i])[0, 1]
                 for i in range(len(label_cols)) if not np.isnan(np.corrcoef(r001_val[:, i], p[:, i])[0, 1])]
        print(f"  R001 vs {name}: {np.mean(corrs):.4f}")

    print("[3/6] w_r 扫描 (固定阈值 0.5 F1)...")
    base = (BASE_W["t001a"] * t001a_val + BASE_W["t002"] * t002_val
            + BASE_W["cnn"] * cnn_val + BASE_W["sim"] * sim_val
            + BASE_W["kmer"] * kmer_val)
    best_wr, best_f1 = 0.0, -1.0
    for wr in R_GRID:
        ens = (1.0 - wr) * base + wr * r001_val
        f1 = macro_f1_fixed_threshold(ens, val_true)
        print(f"  w_r={wr:.2f}: F1(0.5)={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_wr = f1, wr
    print(f"  最优 w_r={best_wr:.2f}")

    print("[4/6] 集成阈值搜索与评估...")
    ens_val = (1.0 - best_wr) * base + best_wr * r001_val
    base_test = (BASE_W["t001a"] * t001a_test + BASE_W["t002"] * t002_test
                 + BASE_W["cnn"] * cnn_test + BASE_W["sim"] * sim_test
                 + BASE_W["kmer"] * kmer_test)
    ens_test = (1.0 - best_wr) * base_test + best_wr * r001_test
    opt_thresholds = find_optimal_thresholds(ens_val, val_true)
    print(f"  阈值: mean={opt_thresholds.mean():.3f}, <0.5: {(opt_thresholds < 0.5).sum()}/{len(label_cols)}")
    y_val_pred, macro_f1, macro_prec, macro_rec = evaluate_at_thresholds(
        ens_val, val_true, opt_thresholds)
    auc = macro_auc(val_true, ens_val)
    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {macro_f1:.4f}      │")
    print(f"  │ Macro Precision:  {macro_prec:.4f}      │")
    print(f"  │ Macro Recall:     {macro_rec:.4f}      │")
    print(f"  │ Macro AUC:        {auc:.4f}      │")
    print(f"  │ 预测/真实:        {int(y_val_pred.sum())}/{int(val_true.sum())}      │")
    print(f"  └─────────────────────────────┘")

    print("[5/6] 测试集提交...")
    y_test_pred = predict_with_thresholds(ens_test, opt_thresholds)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    print("[6/6] 保存...")
    config = {"model": "tck004-ensemble", "base_w": BASE_W,
              "best_w_r": round(float(best_wr), 4), "thresholds": opt_thresholds.tolist(),
              "label_cols": label_cols}
    with Path(ENSEMBLE_CONFIG_PATH).open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    metrics = {"model": "tck004-ensemble", "best_w_r": round(float(best_wr), 4),
               "base_w": BASE_W,
               "val_macro_f1": round(macro_f1, 6),
               "val_macro_precision": round(macro_prec, 6),
               "val_macro_recall": round(macro_rec, 6),
               "val_macro_auc": round(auc, 6),
               "val_total_positive_true": int(val_true.sum()),
               "val_total_positive_pred": int(y_val_pred.sum()),
               "total_time_seconds": round(time.time() - t0, 1)}
    with Path(METRICS_PATH).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  配置: {ENSEMBLE_CONFIG_PATH}, 指标: {METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
