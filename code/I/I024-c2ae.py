"""
I024-c2ae — 冻结 I011 分数，半折 C2AE 潜空间对齐后解码
================================================================
依据:
  - C2AE: Yeh, Wu, Ko & Wang, AAAI 2017, doi:10.1609/aaai.v31i1.10769
    特征编码器 F(x) 与标签编码器 G(y) 映到同一潜空间，解码器还原标签。
    相关损失让 F(x)≈G(y)，重建损失让 D(F(x))≈y。
  - I022：线性 SVD 子空间投影 OOF +0.0004，关闭。
  - I023：线性 ListNet OOF −0.0006，关闭。
    本实验改非线性潜空间，仍冻分数、半折训，不碰 CNN/PLM。
自己的改法:
  x=I011 分数。半折：
    Fe: 500→h→r，Ge: 500→h→r，D: r→h→500。
    L = BCEWithLogits(D(Fe(x)), y) + λ ||Fe(x)−Ge(y)||^2。
  评估折 ŷ=σ(D(Fe(x)))，混合 ens'=(1-α)·base + α·ŷ，base∈{I011,I019}。
  h=256 r=64 λ=1.0，α∈{0.3,0.6,1.0}，40 epoch。
  门槛相对 I011 半折 ≥ +0.001；替换 I019 还要相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I024-c2ae.py
输出:
  rel/I024_report.json
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I024_report.json")
K_RRF, BETA = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
HIDDEN, RANK, LAM = 256, 64, 1.0
N_EPOCH, LR, BATCH = 40, 1e-3, 256
A_GRID = (0.3, 0.6, 1.0)
DEVICE = torch.device("cpu")


class C2AE(nn.Module):
    def __init__(self, n_label=500, hidden=HIDDEN, rank=RANK):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Linear(n_label, hidden), nn.ReLU(), nn.Linear(hidden, rank))
        self.ge = nn.Sequential(
            nn.Linear(n_label, hidden), nn.ReLU(), nn.Linear(hidden, rank))
        self.dec = nn.Sequential(
            nn.Linear(rank, hidden), nn.ReLU(), nn.Linear(hidden, n_label))

    def forward(self, x, y=None):
        zx = self.fe(x)
        logits = self.dec(zx)
        if y is None:
            return logits
        zy = self.ge(y)
        return logits, zx, zy


def fit_c2ae(x, y, seed):
    torch.manual_seed(seed)
    model = C2AE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(x.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    n = len(x)
    model.train()
    for _ in range(N_EPOCH):
        perm = torch.randperm(n)
        for start in range(0, n, BATCH):
            idx = perm[start:start + BATCH]
            xb, yb = xt[idx].to(DEVICE), yt[idx].to(DEVICE)
            logits, zx, zy = model(xb, yb)
            loss = bce(logits, yb) + LAM * ((zx - zy) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_c2ae(model, x):
    model.eval()
    xt = torch.from_numpy(x.astype(np.float32)).to(DEVICE)
    out = []
    for start in range(0, len(x), 1024):
        logits = model(xt[start:start + 1024])
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


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
    return s011, s019, hit, y_val


def main():
    t0 = time.time()
    print("=" * 60)
    print("I024: C2AE 潜空间对齐")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s011, s019, hit, y_val = build_scores(train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec011 = i011.eval_ens(s011, y_val, hit, miss, "i011")
    rec019 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I011 F1={rec011['f1_100']:.4f}")
    print(f"  I019 F1={rec019['f1_100']:.4f}")

    rng = np.random.RandomState(i011.RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    print("[1/2] 半折训 C2AE...")
    mu = s011.mean(0, keepdims=True)
    sd = np.maximum(s011.std(0, keepdims=True), 1e-6)
    x = ((s011 - mu) / sd).astype(np.float32)
    yhat = np.zeros_like(s011, np.float32)
    for i, hold in enumerate(tqdm(folds, desc="    C2AE", unit="fold")):
        fit = folds[1 - i]
        model = fit_c2ae(x[fit], y_val[fit], seed=i011.RANDOM_STATE + i)
        yhat[hold] = predict_c2ae(model, x[hold])

    rec_raw = i011.eval_ens(yhat, y_val, hit, miss, "c2ae_raw")
    print(f"  c2ae_raw F1={rec_raw['f1_100']:.4f}")
    rows = [rec011, rec019, rec_raw]
    configs = {"i011": s011, "i019": s019, "c2ae_raw": yhat}

    print("[2/2] α 混合 + 半折阈值...")
    for base_name, base in (("i011", s011), ("i019", s019)):
        for a in A_GRID:
            mixed = (1.0 - a) * base + a * yhat
            name = f"c2_{base_name}_a{a:.1f}"
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

    c2_oof = [r for r in oof_rows if "c2" in r["name"]]
    best_oof = max(c2_oof, key=lambda r: r["f1_100"]) if c2_oof else oof_rows[0]
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
        "c2ae_raw": rec_raw,
        "best_oof": best_oof,
        "gain_oof_vs_i011": round(gain011, 6),
        "gain_oof_vs_i019": round(gain019, 6),
        "useful": useful,
        "replace_i019": replace,
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
