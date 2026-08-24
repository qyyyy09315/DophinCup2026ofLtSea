"""
I023-listnet — 半折 ListNet 对 500 标签做 listwise 重排
================================================================
依据:
  - ListNet: Cao, Qin, Liu, Tsai & Li, ICML 2007,
    doi:10.1145/1273496.1273513。把排序看成标签上的 softmax 分布，
    用交叉熵对齐真实正标签分布。
  - GOLabeler: You et al., Bioinformatics 2018,
    doi:10.1093/bioinformatics/bty130。蛋白功能预测做成「对 GO 排序」。
  - I012：引用了 ListNet 但只做名次差残差，OOF +0.0002，关闭。
    本实验真正训线性 ListNet。
自己的改法:
  冻 deep/sim/I011/I019。每条蛋白 × 500 标签的特征：
    deep, sim, I011, I019, rank_deep/499, rank_sim/499, |Δrank|/499。
  半折：拟合折上标准化 + 线性 ListNet（softmax CE），评估折打分。
  混合 ens'=(1-α)·base + α·col_norm(s)，base∈{I011, I019}，α∈{0.3,0.6,1.0}。
  门槛相对 I011 半折 ≥ +0.001；替换 I019 还要相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I023-listnet.py
输出:
  rel/I023_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I023_report.json")
K_RRF, BETA = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
A_GRID = (0.3, 0.6, 1.0)
N_EPOCH = 25
LR = 0.25
L2 = 1e-4
BATCH = 256


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(np.clip(z, -40.0, 40.0))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def make_x(deep, sim, s011, s019):
    rd = i011.ranks(deep).astype(np.float32) / 499.0
    rs = i011.ranks(sim).astype(np.float32) / 499.0
    return np.stack([deep, sim, s011, s019, rd, rs, np.abs(rd - rs)], axis=-1)


def fit_listnet(x, y, n_epoch=N_EPOCH, lr=LR, l2=L2, batch=BATCH, seed=42):
    n, _, d = x.shape
    w = np.zeros(d, np.float64)
    b = 0.0
    q = y.astype(np.float64)
    q = q / np.maximum(q.sum(1, keepdims=True), 1.0)
    rng = np.random.RandomState(seed)
    for _ in range(n_epoch):
        perm = rng.permutation(n)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            xb = x[idx].astype(np.float64)
            s = xb @ w + b
            g = softmax(s) - q[idx]
            dw = np.einsum("bld,bl->d", xb, g) / len(idx)
            db = float(g.mean())
            w -= lr * (dw + l2 * w)
            b -= lr * db
    return w.astype(np.float32), np.float32(b)


def apply_listnet(x, w, b):
    return (x.astype(np.float64) @ w.astype(np.float64) + float(b)).astype(np.float32)


def build_scores(train_df, y_full, idx_train, idx_val):
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1 - i011.ALIGN_W) * sim_kmer
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    s011 = deep + i011.SIM_W * sim + BETA * i011.col_norm(i011.rrf_pair(deep, sim, K_RRF))
    n = s011.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t_i011 = i011.search_t(s011, y_val)
    row_m = np.maximum((s011 >= t_i011).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    s019 = i019.mix(s011, p, ALPHA_SK)
    return deep, sim, s011, s019, hit, y_val


def oof_listnet(x, y, folds):
    s = np.zeros(x.shape[:2], np.float32)
    weights = []
    for i, hold in enumerate(tqdm(folds, desc="    ListNet", unit="fold")):
        fit = folds[1 - i]
        mu = x[fit].reshape(-1, x.shape[-1]).mean(0)
        sd = x[fit].reshape(-1, x.shape[-1]).std(0)
        sd = np.maximum(sd, 1e-6)
        xf = (x[fit] - mu) / sd
        xh = (x[hold] - mu) / sd
        w, b = fit_listnet(xf, y[fit], seed=i011.RANDOM_STATE + i)
        s[hold] = apply_listnet(xh, w, b)
        weights.append({"w": w.tolist(), "b": float(b), "mu": mu.tolist(), "sd": sd.tolist()})
    return s, weights


def main():
    t0 = time.time()
    print("=" * 60)
    print("I023: 半折 ListNet 重排")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    deep, sim, s011, s019, hit, y_val = build_scores(train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec011 = i011.eval_ens(s011, y_val, hit, miss, "i011")
    rec019 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I011 F1={rec011['f1_100']:.4f}")
    print(f"  I019 F1={rec019['f1_100']:.4f}")

    x = make_x(deep, sim, s011, s019)
    rng = np.random.RandomState(i011.RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    print("[1/2] 半折训 ListNet...")
    s_ln, weights = oof_listnet(x, y_val, folds)
    s_ln_n = i011.col_norm(s_ln)
    rec_ln = i011.eval_ens(s_ln, y_val, hit, miss, "listnet_raw", scale=True)
    print(f"  listnet_raw F1={rec_ln['f1_100']:.4f}")

    rows = [rec011, rec019, rec_ln]
    configs = {"i011": s011, "i019": s019, "listnet_raw": s_ln}

    print("[2/2] α 混合 + 半折阈值...")
    for base_name, base in (("i011", s011), ("i019", s019)):
        for a in A_GRID:
            mixed = (1.0 - a) * base + a * s_ln_n
            name = f"ln_{base_name}_a{a:.1f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, name,
                                {"base": base_name, "alpha": a})
            rec["delta_i011"] = round(rec["f1_100"] - rec011["f1_100"], 6)
            rec["delta_i019"] = round(rec["f1_100"] - rec019["f1_100"], 6)
            rows.append(rec)
            configs[name] = mixed
            print(f"  {name:22s} F1={rec['f1_100']:.4f} "
                  f"d011={rec['delta_i011']:+.4f} d019={rec['delta_i019']:+.4f}")

    def oof_pred(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            t = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= t).astype(np.int8)
        return pred

    oof_rows = []
    oof_i011 = oof_i019 = None
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
        elif name == "i019":
            oof_i019 = rec["f1_100"]
        print(f"  oof_{name:22s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i011"] = round(rec["f1_100"] - oof_i011, 6)
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    ln_oof = [r for r in oof_rows if "ln_" in r["name"] or "listnet" in r["name"]]
    best_oof = max(ln_oof, key=lambda r: r["f1_100"]) if ln_oof else oof_rows[0]
    gain011 = best_oof["f1_100"] - oof_i011
    gain019 = best_oof["f1_100"] - oof_i019
    useful = bool(gain011 >= 0.001)
    replace = bool(gain019 >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I011 {gain011:+.4f} vs I019 {gain019:+.4f} "
          f"→ {'有效' if useful else '无增益'}"
          f"{'，替换 I019' if replace else '，不替换 I019'}")

    report = {
        "baseline_i011": rec011,
        "baseline_i019": rec019,
        "listnet_raw": rec_ln,
        "best_oof": best_oof,
        "gain_oof_vs_i011": round(gain011, 6),
        "gain_oof_vs_i019": round(gain019, 6),
        "useful": useful,
        "replace_i019": replace,
        "weights": weights,
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
