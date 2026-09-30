"""
I105 — 重复半折方差分析与微传导点翻案检查（评估方法学，分析性）
====================================================================
背景:
  - 本地换锁判据 = 单次半折 OOF Δ ≥ +0.001。但半折 = 一半样本搜阈值 + 另一半评估，
    阈值采样噪声 + 评估集噪声使单次读数方差未知。I085 top20（+0.0007）、I101 注入
    （+0.0003）这类「微传导」点可能是被单次半折噪声误杀的真增益（I097 平台先例:
    本地中性、平台 +0.0090）。
  - 本刀: 对关键变体各算一次 mixed 分数矩阵，然后 R=20 次重复半折（不同随机切分），
    报告相对基线的配对 Δ 分布（mean ± SE、95% 置信下界、P(Δ>0)）。两个用途:
    ① 若某变体配对 Δ 的置信下界 > 0 或均值 ≥ +0.001 -> 翻案重开;
    ② 量化单次半折的统计功效 -> 评估方法学结论（写报告）。
变体: 基线(w_plm=0.35 弱头) / I098a 弱头 w=0.25 / I085 top20 sim / I101 swiglu-RRF w6=0.10
输出: rel/I105_report.json（分析性，不写腾讯表格）
用法: python code/I/I105-repeated-halffold-variance.py
"""

import os
import json
import time

import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
SEED = 42
R = 20

i096 = SourceFileLoader("i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")).load_module()
i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
i085 = SourceFileLoader("i085", os.path.join(PROJECT_ROOT, "code", "I", "I085-top100-gated-agg.py")).load_module()
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe
ranks, col_norm = i064.ranks, i064.col_norm
build_sim = i096.build_sim


def repeated_half_fold(mixed, y_va, reps=R):
    """R 次不同切分的半折 OOF（阈值在半 A 搜、在半 B 评，双向）"""
    vals = []
    yb = y_va.astype(bool)
    for r in range(reps):
        a_idx, b_idx = train_test_split(np.arange(len(y_va)), test_size=0.5, random_state=1000 + r)
        preds = np.zeros_like(y_va, dtype=bool)
        for fi, ei in ((a_idx, b_idx), (b_idx, a_idx)):
            t_sub = search_scut(mixed[fi], y_va[fi])
            preds[ei] = mixed[ei] >= t_sub
        tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
        p_c = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        r_c = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        vals.append(float(np.mean(np.where(p_c + r_c > 0, 2 * p_c * r_c / np.maximum(p_c + r_c, 1e-9), 0.0))))
    return np.array(vals)


def main():
    t0 = time.time()
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_all = np.arange(len(train_df))
    idx_tr, idx_va = train_test_split(idx_all, test_size=0.2, random_state=SEED)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)

    print("=" * 88)
    print(f"  I105: Repeated Half-Fold Variance (R={R}) & Micro-Gain Vindication Check")
    print("=" * 88)

    print("[1/3] 构建概率缓存与 top20 sim...")
    cnn_raw = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn_raw + 0.5 * c005)
    plm_weak = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    swiglu = np.load(os.path.join(REL_DIR, "i098_swiglu_val_probs.npz"))["val"]
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}
    sim = build_sim(y_tr, n_va, val_pids, train_map, val_map)

    idx100, bits100 = i085.parse_topk(
        os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001_top100.tsv"),
        val_pids, train_map)
    sim_top20 = i085.agg_topk(idx100, bits100, y_tr.astype(np.int8), 20)
    kmer = i085.build_kmer(y_tr.astype(np.int8))
    sim_top20_full = 0.4 * sim_top20 + 0.6 * kmer  # 与 I085 同款: 0.4*(比对支) + 0.6*(kmer)

    print("[2/3] 每变体一次 canonical 全流程得 mixed 矩阵...")
    mixed_sets = {}
    _, _, mixed_sets["基线(w_plm=0.35)"], _, _ = full_pipe(
        0.65 * deep4 + 0.35 * plm_weak, sim, y_tr, y_va, n_va)
    _, _, mixed_sets["I098a 弱头 w=0.25"], _, _ = full_pipe(
        0.75 * deep4 + 0.25 * plm_weak, sim, y_tr, y_va, n_va)
    _, _, mixed_sets["I085 top20 sim"], _, _ = full_pipe(
        0.65 * deep4 + 0.35 * plm_weak, sim_top20_full, y_tr, y_va, n_va)
    d5_inj = (0.65 * deep4 + 0.35 * plm_weak) + 0.10 * col_norm(1.0 / (30 + ranks(swiglu) + 1))
    _, _, mixed_sets["I101 注入 w6=0.10"], _, _ = full_pipe(d5_inj, sim, y_tr, y_va, n_va)

    print(f"[3/3] R={R} 次重复半折与配对差分析...")
    all_vals = {name: repeated_half_fold(mixed, y_va) for name, mixed in mixed_sets.items()}
    base_vals = all_vals["基线(w_plm=0.35)"]
    print(f"  基线 OOF: mean={base_vals.mean():.4f} ± {base_vals.std(ddof=1):.4f} "
          f"(单次半折读数标准差 ≈ {base_vals.std(ddof=1):.4f})")

    finals = {}
    for name, vals in all_vals.items():
        if name == "基线(w_plm=0.35)":
            continue
        deltas = vals - base_vals  # 配对差（同切分）
        d_mean = float(deltas.mean())
        d_se = float(deltas.std(ddof=1) / np.sqrt(R))
        ci_lo = d_mean - 1.96 * d_se
        finals[name] = {"delta_mean": round(d_mean, 4), "delta_se": round(d_se, 4),
                        "ci95_lo": round(ci_lo, 4), "p_gt0": round(float((deltas > 0).mean()), 3)}
        if ci_lo > 0:
            verdict = "显著为正(翻案)"
        elif d_mean >= 0.001:
            verdict = "均值≥+0.001(考虑换锁)"
        else:
            verdict = "未翻案"
        print(f"  {name:22s} | Δ={d_mean:+.4f} ± {d_se:.4f} | CI95下界={ci_lo:+.4f} | "
              f"P(Δ>0)={finals[name]['p_gt0']:.2f} -> {verdict}")

    report = {
        "experiment": "I105",
        "method": f"Repeated half-fold variance (R={R}) & micro-gain vindication",
        "baseline_oof_mean": round(float(base_vals.mean()), 4),
        "baseline_half_fold_std": round(float(base_vals.std(ddof=1)), 4),
        "variants": finals,
        "note": "配对差值（同一随机切分下变体-基线），CI 为配对差均值 ± 1.96SE",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with Path(os.path.join(REL_DIR, "I105_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 88)
    print(f"  I105 完成! 报告: {os.path.join(REL_DIR, 'I105_report.json')} | 总用时: {(time.time()-t0)/60:.1f} min")
    print("=" * 88)


if __name__ == "__main__":
    main()
