"""
I030-fbeta — 尾标签用 F-β 搜阈值，冻 I019 分数
================================================================
依据:
  - van Rijsbergen C.J. Information Retrieval, 2nd ed., 1979.
    F_β=(1+β²)PR/(β²P+R)，β>1 更重召回。
  - I009：hit/miss 两套 SCut，miss 阈值 0.32 全局 −0.015，关闭。
    本实验仍一套阈值，只改尾标签的目标从 F1 换成 F-β。
  - I029：改分数被 SCut 吃掉。本实验不改分数。
自己的改法:
  冻 I019。按 train 频次三分位：头仍 F1，尾 F-β。
  β∈{1.2,1.5,2.0}。半折：拟合折搜阈值，评估折套用。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I030-fbeta.py
输出:
  rel/I030_report.json
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


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))

REPORT_PATH = os.path.join(i011.REL_DIR, "I030_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
B_GRID = (1.2, 1.5, 2.0)
CANDS = np.linspace(0.01, 0.99, 100)


def search_t_beta(ens, y, beta_vec):
    best = np.full(i011.N_LABELS, 0.5, dtype=np.float64)
    for i in range(i011.N_LABELS):
        yt = y[:, i]
        if yt.sum() == 0:
            continue
        pr = ens[:, i][:, None] >= CANDS[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0)
        fp = (pr & ~ytm).sum(0)
        fn = (~pr & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / np.maximum(tp + fn, 1)
        b2 = float(beta_vec[i]) ** 2
        f = (1.0 + b2) * prec * rec / np.maximum(b2 * prec + rec, 1e-9)
        best[i] = CANDS[f.argmax()]
    return best


def beta_by_freq(freq, beta_tail):
    b = np.ones(len(freq), np.float64)
    active = freq > 0
    if active.sum() == 0:
        return b
    q1 = np.quantile(freq[active], 1 / 3)
    b[freq <= q1] = beta_tail
    b[~active] = 1.0
    return b


def build_i019(train_df, y_full, idx_train, idx_val):
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
    s011 = deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(i011.rrf_pair(deep, sim, K_RRF))
    n = s011.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t_i011 = i011.search_t(s011, y_val)
    row_m = np.maximum((s011 >= t_i011).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    s019 = i019.mix(s011, p, ALPHA_SK)
    return s019, hit, y_train, y_val


def pack(pred, y, hit, miss, name, extra=None):
    r, p = i011.macro_rp(pred, y)
    rec = {
        "name": name,
        "f1_100": round(i011.macro_f1(pred, y), 6),
        "recall": round(r, 6),
        "precision": round(p, 6),
        "f1_miss": round(i011.macro_f1(pred[miss], y[miss]), 6),
        "f1_hit": round(i011.macro_f1(pred[hit], y[hit]), 6),
        "pred_card": round(float(pred.sum(1).mean()), 3),
    }
    if extra:
        rec.update(extra)
    return rec


def main():
    t0 = time.time()
    print("=" * 60)
    print("I030: 尾标签 F-β 阈值")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s019, hit, y_train, y_val = build_i019(train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    freq = y_train.sum(0).astype(np.float64)
    rows = [rec0]
    pred0 = (s019 >= i011.search_t(s019, y_val)).astype(np.int8)
    preds = {"i019": pred0}

    print("[1/2] β 网格（全 val 阈值）...")
    for b in tqdm(B_GRID, desc="    β", unit="b"):
        bv = beta_by_freq(freq, b)
        t = search_t_beta(s019, y_val, bv)
        pred = (s019 >= t).astype(np.int8)
        rec = pack(pred, y_val, hit, miss, f"fb_b{b:.1f}", {"beta": b})
        rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rec["mean_t"] = round(float(t.mean()), 4)
        rows.append(rec)
        preds[rec["name"]] = pred
        print(f"  {rec['name']:12s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f} "
              f"t={rec['mean_t']:.3f} card={rec['pred_card']:.1f}")

    print("[2/2] 半折核对...")
    rng = np.random.RandomState(i011.RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(beta_tail):
        pred = np.zeros_like(s019, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            if beta_tail is None:
                t = i011.search_t(s019[fit], y_val[fit])
            else:
                t = search_t_beta(s019[fit], y_val[fit], beta_by_freq(freq, beta_tail))
            pred[hold] = (s019[hold] >= t).astype(np.int8)
        return pred

    oof_rows = []
    oof_i019 = None
    configs = [("i019", None)] + [(f"fb_b{b:.1f}", b) for b in B_GRID]
    for name, b in tqdm(configs, desc="    OOF", unit="cfg"):
        pred = oof_pred(b)
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

    fb_oof = [r for r in oof_rows if r["name"].startswith("oof_fb_")]
    best_oof = max(fb_oof, key=lambda r: r["f1_100"]) if fb_oof else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline_i019": rec0,
        "best_oof": best_oof,
        "gain_oof_vs_i019": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(i011.REL_DIR, exist_ok=True)
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  用时 {report['seconds']}s")


if __name__ == "__main__":
    main()
