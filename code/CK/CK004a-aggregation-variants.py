"""
CK004a-aggregation-variants — 邻居聚合方式变体探索 (CK004 衍生)
=================================================================
方案:
  - CK004 基线: top-3 相似度线性加权聚合, 多 k 融合 (k6 主导), w_sim=0.7
  - 本实验: 固定多 k 融合权重与 w_sim, 只变聚合方式:
      top-k ∈ {3, 5} (缓存为 top-5)
      权重 w ∈ {sim, sim^2, sim^0.5, softmax(sim/0.1), softmax(sim/0.5)}
  - 评估: 100 档阈值 Macro F1 (与 CK004 同口径, 基线 0.4295)
  - 若最优变体 Δ > 0.001: 用全量 test 邻居缓存重聚合 → 覆盖提交文件
输出:
  rel/CK004a_report.json
  (可选) submit/submission_CK004_ensemble.csv 覆盖
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
REPORT_PATH = f"{REL_DIR}/CK004a_report.json"
SUB_PATH = f"{SUB_DIR}/submission_CK004a_ensemble.csv"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
T001A_PROBS_PATH = f"{REL_DIR}/TCK001_t001a_probs.npz"
T002_VAL_PROBS_PATH = f"{REL_DIR}/T002_val_probs.npz"
T002_TEST_PROBS_PATH = f"{REL_DIR}/T002_test_probs.npz"
TCK002_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

K_RANGE = [3, 4, 5, 6, 7, 8]
FUSION_WTS = [0, 0.1, 0.2, 0.4, 0.2, 0.1]   # CK004 最优
SIM_W = 0.7
THRESHOLD_STEPS = 100
VAL_SIZE = 0.2
RANDOM_STATE = 42

# (top_k, weight_mode) 变体
VARIANTS = [
    (3, "sim"), (5, "sim"),
    (3, "sim2"), (5, "sim2"),
    (3, "sim05"), (5, "sim05"),
    (5, "softmax01"), (5, "softmax05"),
    (3, "softmax01"),
]


def build_sim_prob_v(neigh_idx, neigh_sim, y_space, top_k, mode):
    n = neigh_idx.shape[0]
    labels = y_space[neigh_idx[:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
    s = neigh_sim[:, :top_k].astype(np.float32)
    if mode == "sim":
        w = np.maximum(s, 1e-7)
    elif mode == "sim2":
        w = np.maximum(s, 1e-7) ** 2
    elif mode == "sim05":
        w = np.maximum(s, 1e-7) ** 0.5
    elif mode == "softmax01":
        w = np.exp((s - s.max(axis=1, keepdims=True)) / 0.1)
    elif mode == "softmax05":
        w = np.exp((s - s.max(axis=1, keepdims=True)) / 0.5)
    else:
        raise ValueError(mode)
    return (labels * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]


def macro_f1_100thr(ens, y_true):
    """向量化 100 档阈值 Macro F1。"""
    cands = np.linspace(0.01, 0.99, THRESHOLD_STEPS)
    f1s = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if yt.sum() == 0:
            continue
        p = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (p & ytm).sum(0); fp = (p & ~ytm).sum(0); fn = (~p & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        f1s.append(f1t.max())
    return float(np.mean(f1s))


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)

    print("[1/5] 读取数据与概率...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)),
                                          test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]

    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, cnn_test = vp["cnn"], tp["cnn"]
    t1 = np.load(T001A_PROBS_PATH)
    t001a_val, t001a_test = t1["t001a_val"], t1["t001a_test"]
    t2v = np.load(T002_VAL_PROBS_PATH)
    t2t = np.load(T002_TEST_PROBS_PATH)
    t002_val, t002_test = t2v["t002"], t2t["t002"]

    with open(TCK002_CONFIG_PATH, "r", encoding="utf-8") as f:
        w = json.load(f)["best_w"]

    print("[2/5] 加载各 k 邻居缓存 (val)...")
    neigh_val = {}
    for k in K_RANGE:
        prefix = "CK003" if k >= 7 else "CK002"
        d = np.load(f"{REL_DIR}/{prefix}_neigh_k{k}.npz")
        neigh_val[k] = (d["idx_val"], d["sim_val"])
        print(f"  k={k}: {d['idx_val'].shape} ✓")

    def fused_sim(top_k, mode):
        return sum(wt * build_sim_prob_v(idx, sim, y_train, top_k, mode)
                   for k, (idx, sim) in neigh_val.items()
                   for wt in [FUSION_WTS[K_RANGE.index(k)]] if wt > 0)

    print("[3/5] 变体评估 (固定融合权重 + w_sim=0.7, 100 档阈值)...")
    results = {}
    for top_k, mode in VARIANTS:
        sim_f = fused_sim(top_k, mode)
        ens = (w["t001a"] * t001a_val + w["t002"] * t002_val
               + w["cnn"] * cnn_val + SIM_W * sim_f)
        f1 = macro_f1_100thr(ens, y_val)
        results[f"top{top_k}_{mode}"] = round(f1, 6)
        print(f"  top{top_k} {mode:10s}: F1={f1:.4f}")

    baseline = results["top3_sim"]
    best_name = max(results, key=results.get)
    best_f1 = results[best_name]
    print(f"  CK004 基线 (top3_sim): {baseline:.4f} | 最优: {best_name} {best_f1:.4f} "
          f"(Δ={best_f1 - baseline:+.4f})")

    report = {"variants": results, "baseline_top3_sim": baseline,
              "best": best_name, "best_f1": best_f1, "delta": best_f1 - baseline,
              "fusion_wts": FUSION_WTS, "sim_w": SIM_W}
    gain = best_f1 - baseline

    if not (gen_sub and gain > 0.001):
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  报告: {REPORT_PATH}")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    print("[4/5] 最优变体重聚合 test (全量缓存) + 提交...")
    top_k, mode = best_name.split("_", 1)
    top_k = int(top_k[3:])
    sim_test = 0.0
    for k in K_RANGE:
        wt = FUSION_WTS[K_RANGE.index(k)]
        if wt <= 0:
            continue
        if k == 6:
            d = np.load(f"{REL_DIR}/TCK002a_neigh_full.npz")
        else:
            d = np.load(f"{REL_DIR}/CK004_neigh_k{k}_full.npz")
        sim_test += wt * build_sim_prob_v(d["idx_test"], d["sim_test"], y_full, top_k, mode)
    ens_test = (w["t001a"] * t001a_test + w["t002"] * t002_test
                + w["cnn"] * cnn_test + SIM_W * sim_test)

    # val 上取最优阈值
    sim_f = fused_sim(top_k, mode)
    ens_val = (w["t001a"] * t001a_val + w["t002"] * t002_val
               + w["cnn"] * cnn_val + SIM_W * sim_f)
    cands = np.linspace(0.01, 0.99, THRESHOLD_STEPS)
    best_t = np.full(500, 0.5)
    for i in tqdm(range(500), desc="    阈值搜索", unit="label"):
        yt = y_val[:, i]
        if yt.sum() == 0:
            continue
        p = ens_val[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (p & ytm).sum(0); fp = (p & ~ytm).sum(0); fn = (~p & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best_t[i] = cands[f1t.argmax()]

    y_test_pred = np.zeros_like(ens_test, dtype=np.int8)
    for i, t in enumerate(best_t):
        y_test_pred[:, i] = (ens_test[:, i] >= t).astype(np.int8)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    report["submission"] = SUB_PATH
    print(f"  提交已覆盖: {SUB_PATH}")

    print("[5/5] 报告...")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
