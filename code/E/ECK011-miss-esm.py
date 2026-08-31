"""
ECK011-miss-esm — 只在 DIAMOND miss 上用 ESM 嵌入邻居补比对槽
================================================================
依据:
  - PLMSearch: Liu et al., Nat Commun 2024,
    doi:10.1038/s41467-024-46808-5。PLM 表示搜远同源，输入只有序列。
  - ECK010：全库 ESM kNN 传标签写进 I019，OOF −0.0025，关闭。
    失败点是强同源已被 DIAMOND 吃干，再写一层会冲淡。
  - A001：miss 1748 条 C011a F1=0.0767，DIAMOND 路为 0。
自己的改法:
  冻 C011a 的 deep。hit 仍 0.4·DIAMOND+0.6·k-mer。
  miss 把空的 DIAMOND 槽换成 ESM2 top-5 sim² 传标签（E001 缓存，
  下标已在 idx_train）。不重训、不占 GPU。
门槛:
  相对 C011a 半折 OOF ≥ +0.001。
用法:
  python code/E/ECK011-miss-esm.py
输出:
  rel/ECK011_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "ECK011_report.json")
NEIGH_PATH = os.path.join(i011.REL_DIR, "E001_neigh_esm.npz")
C011A_ALPHA = 0.50
CNN_W = 0.2
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
TOP = 5


def esm_transfer(idx, sim, y_space, top=TOP):
    n = idx.shape[0]
    valid = idx[:, :top] >= 0
    lab = np.zeros((n, top, y_space.shape[1]), np.float32)
    for j in range(top):
        lab[:, j] = y_space[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(sim[:, :top], 1e-7) ** 2
    ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)


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


def main():
    t0 = time.time()
    print("=" * 60)
    print("ECK011: miss-only ESM neighbor fills DIAMOND slot")

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
    print(f"  miss={int(miss.sum())}/{len(y_val)}")

    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim_lock = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer

    d = np.load(NEIGH_PATH)
    esm = esm_transfer(d["idx_val"], d["sim_val"], y_train)
    sim_miss = sim_lock.copy()
    sim_miss[miss] = i011.ALIGN_W * esm[miss] + (1.0 - i011.ALIGN_W) * sim_kmer[miss]

    cnn42 = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_val = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))["val"]
    cnn_lock = (1.0 - C011A_ALPHA) * cnn42 + C011A_ALPHA * c005_val
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + CNN_W * cnn_lock

    s_lock = build_c011a(deep, sim_lock, y_train, y_val)
    s_miss = build_c011a(deep, sim_miss, y_train, y_val)
    rec0 = i011.eval_ens(s_lock, y_val, hit, miss, "c011a")
    rec_m = i011.eval_ens(s_miss, y_val, hit, miss, "miss_esm")
    rec_m["delta_c011a"] = round(rec_m["f1_100"] - rec0["f1_100"], 6)
    rec_e = i011.eval_ens(esm, y_val, hit, miss, "esm_ind")
    print(f"  C011a F1={rec0['f1_100']:.4f} miss={rec0['f1_miss']:.4f}")
    print(f"  miss_esm F1={rec_m['f1_100']:.4f} d={rec_m['delta_c011a']:+.4f} "
          f"miss={rec_m['f1_miss']:.4f}")
    print(f"  esm 独立 F1={rec_e['f1_100']:.4f} miss={rec_e['f1_miss']:.4f}")

    rows = [rec0, rec_m, rec_e]
    configs = {"c011a": s_lock, "miss_esm": s_miss}

    rng = np.random.RandomState(i011.RANDOM_STATE)
    perm = rng.permutation(len(y_val))
    mid = len(y_val) // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            th = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= th).astype(np.int8)
        return pred

    oof_rows = []
    oof_lock = None
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "c011a":
            oof_lock = rec["f1_100"]
        print(f"  oof_{name:12s} F1={rec['f1_100']:.4f} miss={rec['f1_miss']:.4f}")
    for rec in oof_rows:
        rec["delta_c011a"] = round(rec["f1_100"] - oof_lock, 6)

    cand = [r for r in oof_rows if r["name"] != "oof_c011a"]
    best_oof = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best_oof["f1_100"] - oof_lock
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} vs C011a {gain:+.4f} → "
          f"{'有效' if useful else '无增益'}")

    report = {
        "n_miss": int(miss.sum()),
        "baseline_c011a": rec0,
        "miss_esm": rec_m,
        "esm_ind": rec_e,
        "oof_c011a": oof_lock,
        "best_oof": best_oof,
        "gain_oof_vs_c011a": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "C011a OOF≥+0.001，可在 miss 上补 ESM 检索"
            if useful else "miss 上补 ESM 邻居无增益，关闭；不重开全库 kNN"
        ),
    }
    os.makedirs(i011.REL_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}")
    print(f"  用时 {report['seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
