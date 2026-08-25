"""
CK015-diff — train 同源图上对邻居标签做一步扩散，再传给 val
================================================================
依据:
  - Yuan et al., Briefings in Bioinformatics 2023, doi:10.1093/bib/bbad117
    SPROF-GO：序列模型之后用同源信息做 label diffusion，
    蛋白功能社区重叠，邻居的邻居标签也能传。
  - Zhou et al., NIPS 2004。LGC 是图上平滑；本实验平滑的是
    train 标签，不是 val 分数。
  - ECK005：标签共现图（标签–标签），负。
  - ECK008：val–val 图平滑 I019，负。
  - ECK003 / CK006e：二跳/共享近邻过滤邻居集合，负。
    本实验不改 val 邻居名单，改的是邻居携带的标签。
自己的改法:
  冻 I019。ECK003 train–train k6 top-5：
    y' = (1-α) y + α · sim² 加权邻居标签
  再用 CK002 val→train k6 对 y' 做 sim² 传递。
  α∈{0.15,0.30,0.50}。ens'=(1-γ)·I019 + γ·col_norm(diff)。
  γ∈{0.15,0.30}。门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/CK/CK015-diff.py
输出:
  rel/CK015_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "CK015_report.json")
TRAIN_NEIGH = os.path.join(i011.REL_DIR, "ECK003_neigh_train_k6.npz")
VAL_NEIGH = os.path.join(i011.REL_DIR, "CK002_neigh_k6.npz")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
A_GRID = (0.15, 0.30, 0.50)
G_GRID = (0.15, 0.30)
TOP = 5


def diffuse_train(y_train, idx_tr, sim_tr, alpha):
    n = y_train.shape[0]
    lab = y_train[idx_tr[:, :TOP].ravel()].reshape(n, TOP, -1).astype(np.float32)
    ww = np.maximum(sim_tr[:, :TOP], 1e-7) ** 2
    neigh = (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return ((1.0 - alpha) * y_train.astype(np.float32) + alpha * neigh).astype(np.float32)


def transfer(idx, sim, y_soft, top=TOP):
    n = idx.shape[0]
    lab = y_soft[idx[:, :top].ravel()].reshape(n, top, -1)
    ww = np.maximum(sim[:, :top], 1e-7) ** 2
    return (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]


def build_i019(train_df, y_full, idx_train, idx_val):
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    s011 = deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
        i011.rrf_pair(deep, sim, K_RRF))
    n = s011.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t_i011 = i011.search_t(s011, y_val)
    row_m = np.maximum((s011 >= t_i011).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    return i019.mix(s011, p, ALPHA_SK), hit, y_train, y_val, sim_kmer


def main():
    t0 = time.time()
    print("=" * 60)
    print("CK015: train 同源图标签扩散")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s019, hit, y_train, y_val, sim_kmer = build_i019(
        train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    tr = np.load(TRAIN_NEIGH)
    va = np.load(VAL_NEIGH)
    idx_tr, sim_tr = tr["idx"], tr["sim"]
    idx_va, sim_va = va["idx_val"], va["sim_val"]
    if idx_tr.shape[0] != y_train.shape[0]:
        raise RuntimeError(f"ECK003 n={idx_tr.shape[0]} vs y_train n={y_train.shape[0]}")
    if idx_va.shape[0] != y_val.shape[0]:
        raise RuntimeError(f"CK002 val n={idx_va.shape[0]} vs y_val n={y_val.shape[0]}")

    rows = [rec0]
    configs = {"i019": s019}
    corr = {}

    print("[1/2] α × γ ...")
    for a in tqdm(A_GRID, desc="    α", unit="a"):
        y_soft = diffuse_train(y_train, idx_tr, sim_tr, a)
        sf = transfer(idx_va, sim_va, y_soft)
        c = float(np.corrcoef(sf.ravel(), sim_kmer.ravel())[0, 1])
        corr[f"a{a:.2f}"] = round(c, 4)
        rec_r = i011.eval_ens(sf, y_val, hit, miss, f"diff_a{a:.2f}")
        rec_r["delta_i019"] = round(rec_r["f1_100"] - rec0["f1_100"], 6)
        rec_r["corr_kmer"] = corr[f"a{a:.2f}"]
        rows.append(rec_r)
        print(f"  diff α={a:.2f} 独立 F1={rec_r['f1_100']:.4f} corr={c:.3f}")
        s_n = i011.col_norm(sf)
        for g in G_GRID:
            mixed = (1.0 - g) * s019 + g * s_n
            name = f"mix_a{a:.2f}_g{g:.2f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"alpha": a, "gamma": g})
            rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            configs[name] = mixed
            print(f"  {name:20s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

    print("[2/2] 半折核对...")
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
        print(f"  oof_{name:20s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    cand = [r for r in oof_rows if r["name"] != "oof_i019"]
    best_oof = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline_i019": rec0,
        "best_oof": best_oof,
        "gain_oof_vs_i019": round(gain, 6),
        "useful": useful,
        "corr_kmer": corr,
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
