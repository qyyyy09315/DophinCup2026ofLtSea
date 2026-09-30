"""
I057-stack — 冻结四路 Super Learner 短训 stacking（不重开 I010 / TC003c）
================================================================
依据:
  - Super Learner: van der Laan, Polley & Hubbard, Stat Appl Genet Mol Biol
    2007, doi:10.2202/1544-6115.1309。交叉验证选基学习器凸组合。
  - ProtBoost: Chervov et al., arXiv:2412.04529。CAFA5 第 2 用 Py-Boost +
    GCN stacking。GCN 要 GO DAG，第八轮已关。本实验只借「多路冻结专家
    短训元学习」，不要层级图。
  - I010：只扫 deep/sim 的 SIM_W，半折负。TC003c：CNN×Transformer 逐标签
    LR，当时还没有 k-mer/DIAMOND/I019。
自己的改法:
  冻 C011a 四路：t001a / t002 / cnn=C002b×C005 α=0.50 / sim。
  sl3：三深度头学 deep'，再走 I011+I019。
  sl4：四路直接学分数，只走 I019。
  权在半折上拟合（全局 NNLS 凸组合；对照逐标签 LR），另一折评估。
  门槛相对 C011a 半折 OOF +0.001。CPU，分钟级。
用法:
  python code/I/I057-stack.py
输出:
  rel/I057_report.json
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I057_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
CNN_W = 0.2
CNN_ALPHA = 0.50
N_LABELS = 500


def i019_from_s011(s011, y_train, y_val):
    n = s011.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t = i011.search_t(s011, y_val)
    row_m = np.maximum((s011 >= t).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    return i019.mix(s011, p, ALPHA_SK)


def i011_then_i019(deep, sim, y_train, y_val):
    s011 = deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
        i011.rrf_pair(deep, sim, K_RRF))
    return i019_from_s011(s011, y_train, y_val)


def mix_paths(paths, w):
    out = np.zeros_like(paths[0], dtype=np.float32)
    for wi, p in zip(w, paths):
        out += np.float32(wi) * p
    return out


def fit_nnls(paths, y, fit):
    a = np.stack([p[fit].ravel() for p in paths], 1).astype(np.float64)
    b = y[fit].ravel().astype(np.float64)
    w, _ = nnls(a, b)
    s = w.sum()
    if s <= 1e-12:
        w = np.full(len(paths), 1.0 / len(paths))
    else:
        w = w / s
    return w.astype(np.float64)


def apply_lr_label(paths, coefs, intercepts):
    out = np.empty_like(paths[0], dtype=np.float32)
    stacked = np.stack(paths, 0)
    for j in range(N_LABELS):
        z = intercepts[j] + stacked[:, :, j].T @ coefs[j]
        out[:, j] = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
    return out


def fit_lr(paths, y, fit):
    n_p = len(paths)
    coefs = np.zeros((N_LABELS, n_p), np.float64)
    intercepts = np.zeros(N_LABELS, np.float64)
    eq = np.full(n_p, 1.0 / n_p)
    x_fit = [p[fit] for p in paths]
    y_fit = y[fit]
    n = len(fit)
    for j in range(N_LABELS):
        yj = y_fit[:, j]
        pos = int(yj.sum())
        if pos == 0 or pos == n:
            coefs[j] = eq
            continue
        x = np.stack([p[:, j] for p in x_fit], 1)
        lr = LogisticRegression(
            C=1.0, max_iter=200, solver="lbfgs", class_weight="balanced")
        lr.fit(x, yj)
        intercepts[j] = float(lr.intercept_[0])
        coefs[j] = lr.coef_[0]
    return coefs, intercepts


def main():
    t0 = time.time()
    print("=" * 60)
    print("I057: 冻结四路 Super Learner")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]

    cnn42 = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005z = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))
    c005 = c005z["val"]
    y5 = c005z["y_true"]
    if y5.shape != y_val.shape or not np.array_equal(y5, y_val):
        raise RuntimeError("C005 y_true 与当前切分不对齐")
    cnn = (1.0 - CNN_ALPHA) * cnn42 + CNN_ALPHA * c005

    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer

    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        wcfg = json.load(f)["best_w"]
    deep_lin = (
        wcfg["t001a"] * t1 + wcfg["t002"] * t2 + CNN_W * cnn).astype(np.float32)

    s_c011a = i011_then_i019(deep_lin, sim, y_train, y_val)
    rec0 = i011.eval_ens(s_c011a, y_val, hit, miss, "c011a")
    print(f"  C011a F1={rec0['f1_100']:.4f} miss={rec0['f1_miss']:.4f}")

    deep3 = [t1.astype(np.float32), t2.astype(np.float32), cnn.astype(np.float32)]
    four = deep3 + [sim.astype(np.float32)]

    rng = np.random.RandomState(i011.RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_from_scores(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            th = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= th).astype(np.int8)
        return pred

    oof_c011a = oof_from_scores(s_c011a)
    rec_oof0 = {
        "name": "oof_c011a",
        "f1_100": round(i011.macro_f1(oof_c011a, y_val), 6),
        "f1_miss": round(i011.macro_f1(oof_c011a[miss], y_val[miss]), 6),
    }
    print(f"  OOF C011a F1={rec_oof0['f1_100']:.4f}")

    def oof_stack(kind):
        score = np.zeros_like(s_c011a, dtype=np.float32)
        weights = []
        for i, hold in enumerate(tqdm(folds, desc=f"    {kind}", unit="fold")):
            fit = folds[1 - i]
            if kind == "sl3_nnls":
                w = fit_nnls(deep3, y_val, fit)
                deep = mix_paths(deep3, w)
                score[hold] = i011_then_i019(deep, sim, y_train, y_val)[hold]
                weights.append(w.tolist())
            elif kind == "sl4_nnls":
                w = fit_nnls(four, y_val, fit)
                mixed = mix_paths(four, w)
                score[hold] = i019_from_s011(mixed, y_train, y_val)[hold]
                weights.append(w.tolist())
            elif kind == "sl3_lr":
                coefs, intercepts = fit_lr(deep3, y_val, fit)
                deep = apply_lr_label(deep3, coefs, intercepts)
                score[hold] = i011_then_i019(deep, sim, y_train, y_val)[hold]
                weights.append({
                    "coef_mean": coefs.mean(0).tolist(),
                    "intercept_mean": float(intercepts.mean()),
                })
            elif kind == "sl4_lr":
                coefs, intercepts = fit_lr(four, y_val, fit)
                mixed = apply_lr_label(four, coefs, intercepts)
                score[hold] = i019_from_s011(mixed, y_train, y_val)[hold]
                weights.append({
                    "coef_mean": coefs.mean(0).tolist(),
                    "intercept_mean": float(intercepts.mean()),
                })
            else:
                raise ValueError(kind)
        pred = oof_from_scores(score)
        rec = {
            "name": f"oof_{kind}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
            "weights": weights,
        }
        rec["delta_c011a"] = round(rec["f1_100"] - rec_oof0["f1_100"], 6)
        full = i011.eval_ens(score, y_val, hit, miss, kind)
        full["delta_c011a"] = round(full["f1_100"] - rec0["f1_100"], 6)
        return rec, full, score

    kinds = ("sl3_nnls", "sl4_nnls", "sl3_lr", "sl4_lr")
    oof_rows = [rec_oof0]
    full_rows = [rec0]
    for kind in kinds:
        rec, full, _ = oof_stack(kind)
        oof_rows.append(rec)
        full_rows.append(full)
        print(f"  {kind:10s} full={full['f1_100']:.4f}({full['delta_c011a']:+.4f}) "
              f"OOF={rec['f1_100']:.4f}({rec['delta_c011a']:+.4f}) "
              f"missOOF={rec['f1_miss']:.4f}")

    cand = [r for r in oof_rows if r["name"] != "oof_c011a"]
    best = max(cand, key=lambda r: r["f1_100"])
    gain = best["f1_100"] - rec_oof0["f1_100"]
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best['name']} {best['f1_100']:.4f} vs C011a {gain:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "baseline_c011a": rec0,
        "oof_c011a": rec_oof0,
        "best_oof": best,
        "gain_oof_vs_c011a": round(gain, 6),
        "useful": useful,
        "grid": full_rows,
        "oof_grid": oof_rows,
        "cnn_alpha": CNN_ALPHA,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(i011.REL_DIR, exist_ok=True)
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  用时 {report['seconds']}s")


if __name__ == "__main__":
    main()
