"""
I043-misscnn — miss 上用 C002b×C005 平均 CNN，hit 保持 I019
================================================================
依据:
  - Kulmanov & Hoehndorf, Bioinformatics 2021, doi:10.1093/bioinformatics/btaa763
    DeepGOPlus：有 BLAST 命中走同源，无命中走 CNN。不是全局加权。
  - C011 全样本平均独立 F1(0.5) 0.2527→0.2815，接到 I019 只 OOF +0.0006：
    CNN 只占 deep 0.2，hit 上 sim 已经够强，平均被冲淡。
  - I027 关的是 I011/I019 两套系统路由。本实验不换融合公式，
    只在 miss 行替换 CNN 一元后再建 I019。
自己的改法:
  cnn_miss = (1-α)·C002b + α·C005，α∈{0.50,1.00}。
  hit 行仍用 C002b。重建 I019。门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I043-misscnn.py
输出:
  rel/I043_report.json
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))

REPORT_PATH = os.path.join(i011.REL_DIR, "I043_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
A_GRID = (0.50, 1.00)
CNN_W = 0.2


def i019_from_cnn(cnn, deep_rest, sim, y_train, y_val):
    deep = deep_rest + CNN_W * cnn
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


def f1_05(p, y):
    return i011.macro_f1((p >= 0.5).astype(np.int8), y)


def main():
    t0 = time.time()
    print("=" * 60)
    print("I043: miss 上换 CNN（DeepGOPlus 式）")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]

    cnn42 = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))["val"]
    y5 = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))["y_true"]
    if y5.shape != y_val.shape or not np.array_equal(y5, y_val):
        raise RuntimeError("C005 y_true 与当前切分不对齐")

    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    print(f"  hit {int(hit.sum())}/{len(hit)} miss {int(miss.sum())}")
    print(f"  miss F1(0.5) C002b={f1_05(cnn42[miss], y_val[miss]):.4f} "
          f"C005={f1_05(c005[miss], y_val[miss]):.4f}")

    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep_rest = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"]

    s019 = i019_from_cnn(cnn42, deep_rest, sim, y_train, y_val)
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f} miss={rec0['f1_miss']:.4f}")

    rows = [rec0]
    configs = {"i019": s019}
    print("[1/1] miss 替换 CNN ...")
    for a in A_GRID:
        cnn = cnn42.copy()
        cnn[miss] = (1.0 - a) * cnn42[miss] + a * c005[miss]
        s = i019_from_cnn(cnn, deep_rest, sim, y_train, y_val)
        name = f"miss_a{a:.2f}"
        rec = i011.eval_ens(s, y_val, hit, miss, name, {"alpha": a})
        rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rec["d_miss"] = round(rec["f1_miss"] - rec0["f1_miss"], 6)
        rows.append(rec)
        configs[name] = s
        print(f"  {name:12s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f} "
              f"miss={rec['f1_miss']:.4f} dm={rec['d_miss']:+.4f}")

    rng = np.random.RandomState(i011.RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            th = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= th).astype(np.int8)
        return pred

    oof_rows = []
    oof_i019 = None
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "i019":
            oof_i019 = rec["f1_100"]
        print(f"  oof_{name:12s} F1={rec['f1_100']:.4f} miss={rec['f1_miss']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    cand = [r for r in oof_rows if r["name"] != "oof_i019"]
    best_oof = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "n_miss": int(miss.sum()),
        "baseline_i019": rec0,
        "best_oof": best_oof,
        "gain_oof_vs_i019": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(i011.REL_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  用时 {report['seconds']}s")


if __name__ == "__main__":
    main()
