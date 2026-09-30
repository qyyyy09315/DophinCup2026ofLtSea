"""
I098a — 强头混权 w 网格与失配假设检验
========================================
背景:
  - I098 实测: SwiGLU 强头（独立 0.3141）替换 PLM 支后融合 ΔOOF -0.0094，且幅度随头强度放大
    （I096 平坦收敛头 -0.0016 -> I098 -0.0094）。机理假设: deep5 = 0.65*deep4 + 0.35*plm 中的
    w=0.35 是 I064 为 35ep 弱头（独立 0.27 量级）调的锁定值；强头校准更好（P=0.365 vs 弱头 0.25），
    在 0.35 权重下过度主导 deep5、扰乱与 sim 支的秩平衡。
  - 本刀网格检验失配假设（纯后处理，val 概率缓存）:
    A. 替换式强头网格: deep5' = (1-w)*deep4 + w*swiglu, w ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.35}
       （w=0.35 应复现 I098 OOF 0.4292 作校验）;
    B. 弱头对照网格: plm1664_multiseed 同样扫 w ∈ {0.25, 0.35}（现锁口径 w=0.35 应复现 0.4386）;
    C. 强弱共存联合网格: deep5' = (1-w1-w2)*deep4 + w1*plm_multiseed + w2*swiglu。
判据: 融合半折 OOF >= 0.4396（基线 0.4386 + 0.001，新锁 I097 口径不变）。
用法:
  python code/I/I098a-stronghead-weight-grid.py
输出:
  rel/I098a_report.json
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
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
SEED = 42

i096 = SourceFileLoader(
    "i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")
).load_module()
i064 = SourceFileLoader(
    "i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")
).load_module()
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe
build_sim = i096.build_sim


def half_fold_oof(mixed, y_va):
    a_idx, b_idx = train_test_split(np.arange(len(y_va)), test_size=0.5, random_state=SEED)
    preds = np.zeros_like(y_va, dtype=bool)
    for fi, ei in ((a_idx, b_idx), (b_idx, a_idx)):
        t_sub = search_scut(mixed[fi], y_va[fi])
        preds[ei] = mixed[ei] >= t_sub
    yb = y_va.astype(bool)
    tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
    p_c = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r_c = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    return float(np.mean(np.where(p_c + r_c > 0, 2 * p_c * r_c / np.maximum(p_c + r_c, 1e-9), 0.0)))


def eval_mix(deep4, sim, y_tr, y_va, n_va, w_plm, w_strong):
    """deep5' = (1-w_plm-w_strong)*deep4 + w_plm*弱头 + w_strong*强头，走 canonical 全流程"""
    plm_weak = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    head_strong = np.load(os.path.join(REL_DIR, "i098_swiglu_val_probs.npz"))["val"]
    deep5 = (1.0 - w_plm - w_strong) * deep4 + w_plm * plm_weak + w_strong * head_strong
    f1_in, auc_in, mixed, ths, s = full_pipe(deep5, sim, y_tr, y_va, n_va)
    return f1_in, auc_in, half_fold_oof(mixed, y_va)


def main():
    t_start = time.time()
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)

    idx_all = np.arange(len(train_df))
    idx_tr, idx_va = train_test_split(idx_all, test_size=0.2, random_state=SEED)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)

    print("=" * 84)
    print("  I098a: Strong-Head Weight Grid & Mismatch Hypothesis Test")
    print("=" * 84)

    print("[1/3] 构建 deep4 与 sim（canonical 口径）...")
    cnn_raw = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn_raw + 0.5 * c005)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}
    sim = build_sim(y_tr, n_va, val_pids, train_map, val_map)

    results = []
    best = None

    def run(tag, w_plm, w_strong, expect=None):
        nonlocal best
        f1_in, auc_in, oof = eval_mix(deep4, sim, y_tr, y_va, n_va, w_plm, w_strong)
        rec = {"tag": tag, "w_plm": w_plm, "w_strong": w_strong,
               "f1_in": round(f1_in, 4), "auc_in": round(auc_in, 4),
               "oof": round(oof, 4), "delta_oof": round(oof - 0.4386, 4)}
        results.append(rec)
        mark = ""
        if expect is not None:
            assert abs(oof - expect) < 0.0006, f"校验失败 {tag}: OOF={oof} vs 期望 {expect}"
            mark = f" (校验 {expect} ✓)"
        if best is None or oof > best["oof"]:
            best = rec
            mark += " (*)"
        print(f"  {tag} w_plm={w_plm:.2f} w_strong={w_strong:.2f} | 样内 {f1_in:.4f} | "
              f"OOF={oof:.4f} (Δ={oof-0.4386:+.4f}){mark}", flush=True)
        return rec

    print("[2/3] 校验点...")
    run("弱头基准", 0.35, 0.0, expect=0.4386)   # 现锁口径
    run("强头替换@0.35", 0.0, 0.35, expect=0.4292)  # I098 已测点

    print("[3/3] 网格...")
    print("  -- B. 弱头对照 --")
    for w in (0.25, 0.30):
        run("弱头对照", w, 0.0)
    print("  -- A. 替换式强头 --")
    for w in (0.10, 0.15, 0.20, 0.25, 0.30):
        run("强头替换", 0.0, w)
    print("  -- C. 强弱共存 --")
    for w1, w2 in ((0.25, 0.05), (0.25, 0.10), (0.20, 0.05), (0.20, 0.10), (0.20, 0.15), (0.15, 0.15)):
        run("强弱共存", w1, w2)

    gate_pass = best["delta_oof"] >= 0.001
    conclusion = ("有效，强头混权最优点找到，进入 I100 全量重训出提交流程" if gate_pass
                  else "全网格未过 +0.001 换锁线，失配假设不足以换锁")
    report = {
        "experiment": "I098a",
        "method": "Strong-head weight grid & mismatch hypothesis test",
        "baseline_check": {"weak_035": 0.4386, "strong_035": 0.4292},
        "grid": results,
        "best": best,
        "gate_pass": gate_pass,
        "conclusion": conclusion,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with Path(os.path.join(REL_DIR, "I098a_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 84)
    print(f"  I098a 完成! 最优: {best['tag']} w_plm={best['w_plm']} w_strong={best['w_strong']} | "
          f"OOF={best['oof']:.4f} (Δ={best['delta_oof']:+.4f}) -> {conclusion}")
    print(f"  报告: {os.path.join(REL_DIR, 'I098a_report.json')} | 总用时: {(time.time()-t_start)/60:.1f} min")
    print("=" * 84)


if __name__ == "__main__":
    main()
