"""
I027-route — 样本级路由：过预测走 I019，其余走 I011
================================================================
依据:
  - Dynamic classifier selection: Woloszynski & Kurzynski, Pattern Recognition
    2011, doi:10.1016/j.patcog.2011.03.020。按样本选分类器，不是全局加权。
  - I011 召回更高，I019 压过预测。I020/I026 截断都负：截的是决策，
    本实验改成换分数。
  - I008 门控在 CK005a 上有平台增益，在 I011/I019 上截断失效，但仍可当路由信号。
自己的改法:
  冻 I011 与 I019 分数。半折 Logistic 学 P(I011 SCut 多报)。
  路由：
    hard: p≥τ 用 I019，否则 I011
    soft: (1-p)·I011 + p·I019
    hit: hit 用 I019，miss 用 I011
  τ∈{0.50,0.60,0.70}。门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I027-route.py
输出:
  rel/I027_report.json
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
i008 = load_mod("i008", os.path.join("I", "I008-gated-clip.py"))
i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))

REPORT_PATH = os.path.join(i011.REL_DIR, "I027_report.json")
K_RRF, BETA = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
GATE_T = (0.50, 0.60, 0.70)


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
    print("I027: 样本级路由 I011 / I019")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s011, s019, hit, y_val = build_scores(train_df, y_full, idx_train, idx_val)
    miss = ~hit
    val_len = train_df["sequence"].str.len().values[idx_val].astype(np.float32)
    rec011 = i011.eval_ens(s011, y_val, hit, miss, "i011")
    rec019 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I011 F1={rec011['f1_100']:.4f}")
    print(f"  I019 F1={rec019['f1_100']:.4f}")

    t011 = i011.search_t(s011, y_val)
    scut011 = (s011 >= t011).astype(np.int8)
    card = y_val.sum(1).astype(np.float32)
    y_gate = (scut011.sum(1) > card).astype(np.int8)
    X = i008.make_features(s011, scut011, val_len, hit)
    gprob, gauc = i008.oof_gate(X, y_gate, hit)
    print(f"  门控 OOF AUC(hit)={gauc:.3f}  多报率={y_gate[hit].mean():.3f}")

    rows = [rec011, rec019]
    configs = {"i011": s011, "i019": s019}

    print("[1/2] 路由 ...")
    mixed_hit = np.where(hit[:, None], s019, s011)
    rec = i011.eval_ens(mixed_hit, y_val, hit, miss, "hit_i019_miss_i011")
    rec["delta_i019"] = round(rec["f1_100"] - rec019["f1_100"], 6)
    rows.append(rec)
    configs[rec["name"]] = mixed_hit
    print(f"  {rec['name']:24s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

    mixed_soft = (1.0 - gprob)[:, None] * s011 + gprob[:, None] * s019
    rec = i011.eval_ens(mixed_soft, y_val, hit, miss, "soft_p")
    rec["delta_i019"] = round(rec["f1_100"] - rec019["f1_100"], 6)
    rows.append(rec)
    configs[rec["name"]] = mixed_soft
    print(f"  {rec['name']:24s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

    for tau in tqdm(GATE_T, desc="    τ", unit="t"):
        use19 = (gprob >= tau)[:, None]
        mixed = np.where(use19, s019, s011)
        name = f"hard_t{tau:.2f}"
        rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"tau": tau})
        rec["delta_i019"] = round(rec["f1_100"] - rec019["f1_100"], 6)
        rec["n_i019"] = int((gprob >= tau).sum())
        rows.append(rec)
        configs[name] = mixed
        print(f"  {name:24s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f} "
              f"n19={rec['n_i019']}")

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
        print(f"  oof_{name:24s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i011"] = round(rec["f1_100"] - oof_i011, 6)
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    route_oof = [r for r in oof_rows if r["name"] not in ("oof_i011", "oof_i019")]
    best_oof = max(route_oof, key=lambda r: r["f1_100"]) if route_oof else oof_rows[0]
    gain011 = best_oof["f1_100"] - oof_i011
    gain019 = best_oof["f1_100"] - oof_i019
    useful = bool(gain019 >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I011 {gain011:+.4f} vs I019 {gain019:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "baseline_i011": rec011,
        "baseline_i019": rec019,
        "best_oof": best_oof,
        "gain_oof_vs_i011": round(gain011, 6),
        "gain_oof_vs_i019": round(gain019, 6),
        "useful": useful,
        "gate_auc_hit": round(gauc, 6),
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
