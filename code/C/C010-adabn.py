"""
C010-adabn — C002b 测试时 AdaBN（只改 BN 统计，不改权重）
================================================================
依据:
  - Li, Wang, Shi, Hou & Liu, Pattern Recognition 2018,
    doi:10.1016/j.patcog.2018.03.005。AdaBN：目标域前向时
    用目标 batch 的均值/方差替换源域 running stats，权重不动。
  - C009 MC Dropout 开的是 Dropout，BN 仍 eval，负。
    TENT 还要反传改仿射参数，更危险。本实验只换 BN 统计。
自己的改法:
  加载 cnn-c002b.pt。val 上 BN.train()、其余 eval，大 batch
  前向收集概率（动量式更新 running stats 后，再 eval 前向一次）。
  替换 CNN 路，重建 I019。独立对照 C002b F1(0.5)。
  融合门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/C/C010-adabn.py
输出:
  rel/C010_report.json
  rel/C010_adabn_cnn_val.npz
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
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


c002b = load_mod("c002b", os.path.join("C", "C002b-cnn.py"))
i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))

REPORT_PATH = os.path.join(i011.REL_DIR, "C010_report.json")
PROBS_PATH = os.path.join(i011.REL_DIR, "C010_adabn_cnn_val.npz")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cnn-c002b.pt")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
G_GRID = (0.15, 0.30, 1.0)
DEVICE = c002b.DEVICE


def enable_bn_train(m):
    if isinstance(m, nn.BatchNorm1d):
        m.train()
        m.momentum = 1.0


@torch.no_grad()
def collect_bn(model, loader):
    model.eval()
    model.apply(enable_bn_train)
    for x, _ in tqdm(loader, desc="    AdaBN", unit="b"):
        x = x.to(DEVICE)
        with c002b.autocast_ctx():
            model(x)


@torch.no_grad()
def det_probs(model, loader):
    model.eval()
    chunks = []
    for x, _ in loader:
        x = x.to(DEVICE)
        with c002b.autocast_ctx():
            logits = model(x)
        chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def i019_from_cnn(cnn, deep_rest, sim, y_train, y_val):
    deep = deep_rest + 0.2 * cnn
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
    print("C010: AdaBN")
    print(f"  device={DEVICE}")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    seqs = [s.replace("U", "X").replace("Z", "X").replace("O", "X")
            for s in train_df["sequence"].values[idx_val]]
    ds = c002b.ProteinDataset(seqs, y_val.astype(np.float32), c002b.MAX_LEN)
    loader = DataLoader(ds, batch_size=c002b.BATCH_SIZE, shuffle=False,
                        num_workers=0, collate_fn=c002b.collate_onehot)

    model = c002b.MultiScaleCNNv3(n_labels=500).to(DEVICE)
    c002b.load_checkpoint(model, MODEL_PATH)
    print(f"  加载 {MODEL_PATH}")

    cnn42 = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))["cnn"]
    f1_det = f1_05(cnn42, y_val)
    print(f"  C002b det F1(0.5)={f1_det:.4f}")

    collect_bn(model, loader)
    cnn_ad = det_probs(model, loader)
    np.savez_compressed(PROBS_PATH, cnn=cnn_ad)
    f1_ad = f1_05(cnn_ad, y_val)
    corr = float(np.corrcoef(cnn42.ravel(), cnn_ad.ravel())[0, 1])
    print(f"  AdaBN F1(0.5)={f1_ad:.4f} d={f1_ad - f1_det:+.4f} corr={corr:.4f}")

    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep_rest = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"]

    s019 = i019_from_cnn(cnn42, deep_rest, sim, y_train, y_val)
    sad = i019_from_cnn(cnn_ad, deep_rest, sim, y_train, y_val)
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    reca = i011.eval_ens(sad, y_val, hit, miss, "adabn")
    reca["delta_i019"] = round(reca["f1_100"] - rec0["f1_100"], 6)
    print(f"  I019 F1={rec0['f1_100']:.4f}")
    print(f"  AdaBN F1={reca['f1_100']:.4f} d={reca['delta_i019']:+.4f}")

    rows = [rec0, reca]
    configs = {"i019": s019, "adabn": sad}
    sn = i011.col_norm(sad)
    print("[2/2] γ 混合 ...")
    for g in tqdm(G_GRID, desc="    γ", unit="g"):
        if g >= 0.999:
            continue
        mixed = (1.0 - g) * s019 + g * sn
        name = f"mix_g{g:.2f}"
        rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"gamma": g})
        rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        configs[name] = mixed
        print(f"  {name:12s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

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
        print(f"  oof_{name:12s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    cand = [r for r in oof_rows if r["name"] != "oof_i019"]
    best_oof = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001 and f1_ad > 0.2714)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} 独立F1(0.5)={f1_ad:.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "f1_05_det": round(f1_det, 6),
        "f1_05_adabn": round(f1_ad, 6),
        "corr_det_adabn": round(corr, 6),
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
