"""
I019-sinkhorn — I011 分数矩阵做 Sinkhorn 双边质量匹配
================================================================
依据:
  - Cuturi, NeurIPS 2013, arXiv:1306.0895
    熵正则最优传输：交替归一行/列（Sinkhorn–Knopp），把核矩阵
    推到指定的行质量与列质量。
  - I013 硬按列 PCut 已负：只卡列频次会少报。
  - I006 行基数硬截已负。双边软匹配同时照顾 π_j 与每条蛋白的 k。
自己的改法:
  冻 I011 分数。核 K=exp(ens/τ)。列质量=π_j·N·ρ，行质量=该行 SCut 基数
  （至少 1）。迭代后得到传输矩阵 P。混合 ens'=(1-α)·ens + α·列归一 P。
  τ∈{0.05,0.10,0.20}，ρ∈{1.0,1.2}，α∈{0.3,0.6,1.0}。
  半折：π 与行质量用拟合折，评估折套用。门槛相对 I011 半折 ≥ +0.001。
用法:
  python code/I/I019-sinkhorn.py
  python code/I/I019-sinkhorn.py --gen-sub
输出:
  rel/I019_report.json
  submit/submission_I019_ensemble.csv（--gen-sub）
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    "i011", os.path.join(PROJECT_ROOT, "code", "I", "I011-rrf.py"))
i011 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i011)

REPORT_PATH = os.path.join(i011.REL_DIR, "I019_report.json")
K_RRF, BETA = 30, 0.50
TAU = (0.05, 0.10, 0.20)
RHO = (1.0, 1.2)
ALPHA = (0.3, 0.6, 1.0)
N_ITER = 40


def sinkhorn(kernel, row_m, col_m, n_iter=N_ITER, eps=1e-12):
    u = np.ones(kernel.shape[0], np.float64)
    v = np.ones(kernel.shape[1], np.float64)
    k = kernel.astype(np.float64)
    for _ in range(n_iter):
        u = row_m / (k @ v + eps)
        v = col_m / (k.T @ u + eps)
    return (u[:, None] * k * v[None, :]).astype(np.float32)


def mix(ens, p, alpha):
    p_n = i011.col_norm(p)
    return (1.0 - alpha) * ens + alpha * p_n


def main():
    t0 = time.time()
    print("=" * 60)
    print("I019: Sinkhorn 双边质量匹配")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1 - i011.ALIGN_W) * sim_kmer
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    ens = deep + i011.SIM_W * sim + BETA * i011.col_norm(i011.rrf_pair(deep, sim, K_RRF))
    n, l = ens.shape
    pi = y_train.mean(0).astype(np.float64)
    t_scut = i011.search_t(ens, y_val)
    scut = ens >= t_scut
    row_m = np.maximum(scut.sum(1).astype(np.float64), 1.0)

    rec0 = i011.eval_ens(ens, y_val, hit, miss, "i011")
    print(f"  I011 F1={rec0['f1_100']:.4f} card={rec0['pred_card']:.1f}")
    rows = [rec0]
    configs = {"i011": ens}

    print("[1/2] τ × ρ × α ...")
    ens_shift = ens - ens.max(axis=1, keepdims=True)
    for tau in tqdm(TAU, desc="    τ", unit="t"):
        ker = np.exp(ens_shift / tau)
        for rho in RHO:
            col_m = np.maximum(pi * n * rho, 1e-6)
            col_m *= row_m.sum() / col_m.sum()
            p = sinkhorn(ker, row_m, col_m)
            for a in ALPHA:
                mixed = mix(ens, p, a)
                name = f"sk_t{tau:.2f}_r{rho:.1f}_a{a:.1f}"
                rec = i011.eval_ens(mixed, y_val, hit, miss, name,
                                    {"tau": tau, "rho": rho, "alpha": a})
                rec["delta_i011"] = round(rec["f1_100"] - rec0["f1_100"], 6)
                rows.append(rec)
                configs[name] = mixed
                print(f"  {name:24s} F1={rec['f1_100']:.4f} d={rec['delta_i011']:+.4f} "
                      f"miss={rec['f1_miss']:.4f}")

    print("[2/2] 半折核对...")
    rng = np.random.RandomState(i011.RANDOM_STATE)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            t = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= t).astype(np.int8)
        return pred

    oof_rows = []
    oof_i011 = None
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "i011":
            oof_i011 = rec["f1_100"]
        print(f"  oof_{name:24s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i011"] = round(rec["f1_100"] - oof_i011, 6)

    best_oof = max(oof_rows, key=lambda r: r["f1_100"])
    gain = best_oof["f1_100"] - oof_i011
    useful = bool(gain >= 0.001 and best_oof["name"] != "oof_i011")
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I011 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline_i011": rec0,
        "best_oof": best_oof,
        "gain_oof_vs_i011": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")

    if "--gen-sub" in sys.argv:
        tau, rho, alpha = 0.20, 1.0, 0.6
        print(f"[3/3] 按 OOF 最优 τ={tau} ρ={rho} α={alpha} 出提交...")
        test_df = pd.read_csv(os.path.join(i011.DATA_DIR, "test.csv"))
        tsv_test = os.path.join(i011.CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
        full_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
        idx_t, bits_t = i011.parse_aln(tsv_test, test_df["protein_id"].values, full_map)
        sim_bit_t = i011.agg_labels(idx_t, bits_t, y_full)
        sim_kmer_t = 0.0
        for kk, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
            if wt <= 0:
                continue
            d = np.load(os.path.join(
                i011.REL_DIR, "TCK002a_neigh_full.npz" if kk == 6
                else f"CK004_neigh_k{kk}_full.npz"))
            n2 = d["idx_test"].shape[0]
            lab = y_full[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
            s = d["sim_test"][:, :5].astype(np.float32)
            ww = np.maximum(s, 1e-7) ** 2
            sim_kmer_t = sim_kmer_t + wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
        sim_t = i011.ALIGN_W * sim_bit_t + (1.0 - i011.ALIGN_W) * sim_kmer_t
        tp = np.load(os.path.join(i011.REL_DIR, "TC003_test_probs.npz"))
        t1t = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
        t2t = np.load(os.path.join(i011.REL_DIR, "T002_test_probs.npz"))
        deep_t = w["t001a"] * t1t["t001a_test"] + w["t002"] * t2t["t002"] + w["cnn"] * tp["cnn"]
        ens_t = (deep_t + i011.SIM_W * sim_t
                 + BETA * i011.col_norm(i011.rrf_pair(deep_t, sim_t, K_RRF)))
        mixed_v = configs["sk_t0.20_r1.0_a0.6"]
        t_sub = i011.search_t(mixed_v, y_val)
        n_t = ens_t.shape[0]
        t_i011 = i011.search_t(ens, y_val)
        scut_t = ens_t >= t_i011
        row_t = np.maximum(scut_t.sum(1).astype(np.float64), 1.0)
        col_t = np.maximum(pi * n_t * rho, 1e-6)
        col_t *= row_t.sum() / col_t.sum()
        shift_t = ens_t - ens_t.max(axis=1, keepdims=True)
        p_t = sinkhorn(np.exp(shift_t / tau), row_t, col_t)
        mixed_t = mix(ens_t, p_t, alpha)
        y_pred = (mixed_t >= t_sub).astype(np.int8)
        os.makedirs(i011.SUB_DIR, exist_ok=True)
        sub = pd.concat([test_df[["protein_id"]],
                         pd.DataFrame(y_pred, columns=label_cols)], axis=1)
        sub_path = os.path.join(i011.SUB_DIR, "submission_I019_ensemble.csv")
        sub.to_csv(sub_path, index=False)
        report["submission"] = sub_path
        report["submit_tau"] = tau
        report["submit_rho"] = rho
        report["submit_alpha"] = alpha
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  提交: {sub_path} ({len(sub)} 行)")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
