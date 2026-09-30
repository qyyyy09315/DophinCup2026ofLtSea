"""
I101 — 新头作为第 6 路 deep 支的融合突围（I 线备选刀，I098 触发）
===================================================
背景:
  - I098 实测: SwiGLU 收敛头独立 0.3141（+0.0105 过 Gate 1），但替换 PLM 支后融合
    ΔOOF -0.0094（法则第 7 次，且幅度随头强度放大: I096 -0.0016 -> I098 -0.0094）。
    机理: w=0.35 线性混权是 I064 为弱头调的，强头在此口径下过度主导 deep5、
    扰乱与 sim 支的秩平衡。
  - 本刀不再替换 PLM 支，而是把新头作为**第 6 路**注入（改变信息集合而非替换同源支）:
    PLM 支保持 I065 口径不变（0.35*plm1664_multiseed），新头以两种方式注入后重跑
    canonical 全流程（RRF + Sinkhorn + SCut + 半折 OOF）:
      方案 A 秩注入: deep5' = deep5 + w6 * col_norm(1/(30+ranks(head6)+1))   [I064 use_rrf_plm 同机制]
      方案 B 概率注入: deep5' = (1-w6)*deep5 + w6*col_norm(head6)            [列归一校准后线性]
    网格 w6 ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40}，w6=0 为基线复现校验（必须 0.4386）。
  - 多头支持: --heads 可传多个 val 概率 npz（I098 SwiGLU / I099 ML-Decoder），
    先逐路独立扫网格，再测双路同时注入（各 w6/2）。
判据: 半折 OOF >= 0.4396（基线 0.4386 + 0.001）。
用法:
  python code/I/I101-head-sixth-branch.py                     # 默认注入 rel/i098_swiglu_val_probs.npz
  python code/I/I101-head-sixth-branch.py --heads a.npz b.npz # 多路
输出:
  rel/I101_report.json（纯后处理，无训练产物）
"""

import os
import json
import time
import argparse

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
W_PLM = 0.35
GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

i096 = SourceFileLoader(
    "i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")
).load_module()
i064 = SourceFileLoader(
    "i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")
).load_module()
ranks, col_norm = i064.ranks, i064.col_norm
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


def eval_config(deep4, sim, y_tr, y_va, n_va, inject_heads, w6, mode):
    """注入第 6 路后重跑 canonical 全流程；w6=0 时必须复现基线"""
    deep5 = (1.0 - W_PLM) * deep4 + W_PLM * np.load(
        os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    if inject_heads and w6 > 0:
        k = len(inject_heads)
        wi = w6 / k
        for h in inject_heads:
            if mode == "rrf":
                deep5 = deep5 + wi * col_norm(1.0 / (30 + ranks(h) + 1))
            else:
                deep5 = (1.0 - wi) * deep5 + wi * col_norm(h)
    f1_in, auc_in, mixed, ths, s = full_pipe(
        deep5, sim, y_tr, y_va, n_va
    )
    return f1_in, auc_in, half_fold_oof(mixed, y_va)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", nargs="+",
                        default=[os.path.join(REL_DIR, "i098_swiglu_val_probs.npz")],
                        help="第 6 路 val 概率 npz 列表（I098/I099 产物）")
    args = parser.parse_args()

    t_start = time.time()
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)

    idx_all = np.arange(len(train_df))
    idx_tr, idx_va = train_test_split(idx_all, test_size=0.2, random_state=SEED)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)

    print("=" * 84)
    print("  I101: New Head as Sixth Deep Branch (RRF / linear injection grid)")
    print(f"  Heads: {[os.path.basename(p) for p in args.heads]}")
    print("=" * 84)

    # 1. deep4 与 sim（canonical 口径）
    print("[1/3] 构建 deep4 与 sim 支（canonical 口径）...")
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

    # 2. 基线复现校验（w6=0）
    print("[2/3] 基线复现校验（w6=0，必须 OOF≈0.4386）...")
    f1_in0, auc_in0, oof0 = eval_config(deep4, sim, y_tr, y_va, n_va, None, 0.0, "rrf")
    print(f"  基线: 样内 F1={f1_in0:.4f} (应为 0.4475), OOF={oof0:.4f} (应为 0.4386)")
    assert abs(oof0 - 0.4386) < 0.0005, f"基线漂移: OOF={oof0}"

    # 3. 注入网格
    print("[3/3] 第 6 路注入网格（逐路 + 双路联合）...")
    head_names = [os.path.splitext(os.path.basename(p))[0] for p in args.heads]
    heads = [np.load(p)["val"].astype(np.float32) for p in args.heads]

    results = []
    combos = [( [i], "rrf" ) for i in range(len(heads))]
    combos += [( [i], "linear" ) for i in range(len(heads))]
    if len(heads) >= 2:
        combos.append((list(range(len(heads))), "rrf"))

    best = None
    for idxs, mode in combos:
        tag = "+".join(f"{head_names[i]}({mode})" for i in idxs)
        inject = [heads[i] for i in idxs]
        for w6 in GRID:
            if w6 == 0.0:
                continue
            f1_in, auc_in, oof = eval_config(deep4, sim, y_tr, y_va, n_va, inject, w6, mode)
            results.append({
                "combo": tag, "mode": mode, "w6": w6,
                "f1_in": round(f1_in, 4), "auc_in": round(auc_in, 4), "oof": round(oof, 4),
                "delta_oof": round(oof - 0.4386, 4),
            })
            mark = " (*)" if best is None or oof > best["oof"] else ""
            print(f"  {tag} w6={w6:.2f} | 样内 F1={f1_in:.4f} | OOF={oof:.4f} "
                  f"(Δ={oof-0.4386:+.4f}){mark}", flush=True)
            if best is None or oof > best["oof"]:
                best = results[-1]

    gate_pass = best is not None and best["delta_oof"] >= 0.001
    conclusion = ("有效，第 6 路新支路传导成功，进入 I100 全量换锁流程" if gate_pass
                  else "全网格未过 +0.001 换锁线，I 线突围关闭")
    report = {
        "experiment": "I101",
        "method": "New head as sixth deep branch (RRF/linear injection grid)",
        "heads": head_names,
        "baseline_check": {"f1_in": round(f1_in0, 4), "oof": round(oof0, 4)},
        "grid": results,
        "best": best,
        "gate_pass": gate_pass,
        "conclusion": conclusion,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with Path(os.path.join(REL_DIR, "I101_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 84)
    if best:
        print(f"  I101 完成! 最优: {best['combo']} w6={best['w6']} | "
              f"OOF={best['oof']:.4f} (Δ={best['delta_oof']:+.4f}) -> {conclusion}")
    else:
        print(f"  I101 完成! 无有效网格 -> {conclusion}")
    print(f"  报告: {os.path.join(REL_DIR, 'I101_report.json')} | 总用时: {(time.time()-t_start)/60:.1f} min")
    print("=" * 84)


if __name__ == "__main__":
    main()
