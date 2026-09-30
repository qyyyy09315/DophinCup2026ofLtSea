"""
I012-disres — 分歧残差：只在 deep/sim 名次打架的标签上按头尾信任平移
================================================================
依据:
  - GOLabeler: You et al., Bioinformatics 2018, doi:10.1093/bioinformatics/bty130
    把功能预测做成「每个蛋白对 500 标签的排序」，listwise Learning-to-Rank。
  - ListNet: Cao et al., ICML 2007, doi:10.1145/1273496.1273513
  - I010：尾标签共享 SIM_W=1.125、头标签 0.5 —— 尾更信同源，头更信 CNN。
  - I011：全局 RRF 有效；I011d 按样本调 β 失败（样本级太粗）。
自己的改法:
  不重训 ListNet。冻 I011（k=30 β=0.50）。
  对每个 (蛋白, 标签)：δ = (rank_deep − rank_sim)/499。
  δ>0 表示 sim 排得更靠前。
  训练频次三分位：尾 sign=+1（信 sim），头 sign=−1（信 deep），中=0。
  仅当 |rank 差|≥D 才加残差：ens' = I011 + α · sign · δ。
  D∈{20,50,100}，α∈{0.02,0.05,0.10}。门槛相对 I011 半折 ≥ +0.001。
用法:
  python code/I/I012-disres.py
输出:
  rel/I012_report.json
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    "i011", os.path.join(PROJECT_ROOT, "code", "I", "I011-rrf.py"))
i011 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i011)

REPORT_PATH = os.path.join(i011.REL_DIR, "I012_report.json")
K, BETA = 30, 0.50
D_GRID = (20, 50, 100)
A_GRID = (0.02, 0.05, 0.10)
N = i011.N_LABELS


def freq_sign(pos):
    """尾 +1 信 sim，头 −1 信 deep，中 0。"""
    active = pos > 0
    q1, q2 = np.quantile(pos[active], [1 / 3, 2 / 3])
    s = np.zeros(N, np.float32)
    s[pos <= q1] = 1.0
    s[pos >= q2] = -1.0
    s[~active] = 0.0
    return s, float(q1), float(q2)


def residual(deep, sim, sign, d_min, alpha):
    rd, rs = i011.ranks(deep), i011.ranks(sim)
    gap = (rd - rs).astype(np.float32) / 499.0
    mask = (np.abs(rd - rs) >= d_min).astype(np.float32)
    return alpha * sign[None, :] * gap * mask


def main():
    t0 = time.time()
    print("=" * 60)
    print("I012: 分歧残差（头尾信任 × 名次差）")

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
    base = deep + i011.SIM_W * sim
    i011_ens = base + BETA * i011.col_norm(i011.rrf_pair(deep, sim, K))
    sign, q1, q2 = freq_sign(y_train.sum(0).astype(np.float64))
    print(f"  频次切点 q1={q1:.0f} q2={q2:.0f}  尾 {(sign > 0).sum()} 头 {(sign < 0).sum()}")

    rows = []
    rec = i011.eval_ens(base, y_val, hit, miss, "ck005a")
    rows.append(rec)
    rec = i011.eval_ens(i011_ens, y_val, hit, miss, "i011")
    rows.append(rec)
    print(f"  ck005a {rows[0]['f1_100']:.4f}  i011 {rows[1]['f1_100']:.4f}")

    print("[1/2] D × α 网格...")
    configs = {"i011": i011_ens}
    for d in tqdm(D_GRID, desc="    D", unit="d"):
        for a in A_GRID:
            ens = i011_ens + residual(deep, sim, sign, d, a)
            name = f"res_D{d}_a{a:.2f}"
            rec = i011.eval_ens(ens, y_val, hit, miss, name, {"D": d, "alpha": a})
            rec["delta_i011"] = round(rec["f1_100"] - rows[1]["f1_100"], 6)
            rows.append(rec)
            configs[name] = ens
            print(f"  {name:18s} F1={rec['f1_100']:.4f} d_i011={rec['delta_i011']:+.4f}")

    print("[2/2] 半折核对...")
    rng = np.random.RandomState(i011.RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(ens):
        pred = np.zeros_like(ens, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            t = i011.search_t(ens[fit], y_val[fit])
            pred[hold] = (ens[hold] >= t).astype(np.int8)
        return pred

    oof_rows = []
    oof_i011 = None
    for name, ens in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(ens)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "i011":
            oof_i011 = rec["f1_100"]
        print(f"  oof_{name:18s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i011"] = round(rec["f1_100"] - oof_i011, 6)

    best_oof = max(oof_rows, key=lambda r: r["f1_100"])
    gain = best_oof["f1_100"] - oof_i011
    useful = bool(gain >= 0.001 and best_oof["name"] != "oof_i011")
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I011 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "q1": q1, "q2": q2,
        "n_tail": int((sign > 0).sum()),
        "n_head": int((sign < 0).sum()),
        "baseline_i011": rows[1],
        "best_oof": best_oof,
        "gain_oof_vs_i011": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
