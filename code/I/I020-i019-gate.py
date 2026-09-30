"""
I020-i019-gate — I008 门控 hit 截断叠到 I019 分数
================================================================
依据:
  - I008 平台 45.3902 超过 CK005a 45.3039（本地仅 +0.0008）
  - I019 Sinkhorn τ=0.20 α=0.60 OOF 较 I011 +0.0011
  - Yang, SIGIR 2001, doi:10.1145/383952.383975
自己的改法:
  冻 I019 分数与 SCut。I008 的半折 Logistic 学 P(过预测)，
  对 P≥τ 的 hit 行按比例截。τ×r 网格重扫（分数变了，I008 的 0.80/0.85
  不一定最优）。门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I020-i019-gate.py
  python code/I/I020-i019-gate.py --gen-sub
输出:
  rel/I020_report.json
  submit/submission_I020_ensemble.csv（--gen-sub）
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
i008 = load_mod("i008", os.path.join("I", "I008-gated-clip.py"))
i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))

REPORT_PATH = os.path.join(i011.REL_DIR, "I020_report.json")
TAU_SK, RHO, ALPHA = 0.20, 1.0, 0.60
GATE_T = (0.50, 0.60, 0.70, 0.80)
GATE_R = (0.70, 0.80, 0.85)


def build_i019_val(train_df, y_full, idx_train, idx_val):
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
    ens = (deep + i011.SIM_W * sim
           + i019.BETA * i011.col_norm(i011.rrf_pair(deep, sim, i019.K_RRF)))
    n = ens.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t_i011 = i011.search_t(ens, y_val)
    row_m = np.maximum((ens >= t_i011).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((ens - ens.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    mixed = i019.mix(ens, p, ALPHA)
    return mixed, hit, y_train, y_val, w, pi, t_i011


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print("I020: I008 门控截断 × I019 分数")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    mixed, hit, y_train, y_val, w, pi, t_i011 = build_i019_val(
        train_df, y_full, idx_train, idx_val)
    miss = ~hit
    val_len = train_df["sequence"].str.len().values[idx_val].astype(np.float32)
    t = i011.search_t(mixed, y_val)
    scut = (mixed >= t).astype(np.int8)
    card = y_val.sum(1).astype(np.float32)
    y_gate = (scut.sum(1) > card).astype(np.int8)
    print(f"  hit 多报率={y_gate[hit].mean():.3f}")

    X = i008.make_features(mixed, scut, val_len, hit)
    gprob, gauc = i008.oof_gate(X, y_gate, hit)
    print(f"  门控 OOF AUC(hit)={gauc:.3f}")

    rec0 = {
        "name": "i019_scut",
        "f1_100": round(i011.macro_f1(scut, y_val), 6),
        "f1_miss": round(i011.macro_f1(scut[miss], y_val[miss]), 6),
        "f1_hit": round(i011.macro_f1(scut[hit], y_val[hit]), 6),
        "pred_card": round(float(scut.sum(1).mean()), 3),
    }
    print(f"  I019 SCut F1={rec0['f1_100']:.4f}")
    rows = [rec0]

    print("[1/2] 门控网格...")
    for tau in tqdm(GATE_T, desc="    τ", unit="t"):
        gated = hit & (gprob >= tau)
        for r in GATE_R:
            pred = i008.clip_ratio_rows(mixed, scut, gated, r)
            rec = {
                "name": f"gate_t{tau:.2f}_r{r:.2f}",
                "tau": tau, "ratio": r, "n_clip": int(gated.sum()),
                "f1_100": round(i011.macro_f1(pred, y_val), 6),
                "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
                "f1_hit": round(i011.macro_f1(pred[hit], y_val[hit]), 6),
                "pred_card": round(float(pred.sum(1).mean()), 3),
            }
            rec["delta"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            print(f"  {rec['name']:22s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f} "
                  f"n={rec['n_clip']}")

    ranked = [r for r in rows if r["name"] != "i019_scut"]
    best = max(ranked, key=lambda r: r["f1_100"])
    gain = best["f1_100"] - rec0["f1_100"]
    useful = bool(gain >= 0.001)
    print(f"  全 val 最优 {best['name']} F1={best['f1_100']:.4f} Δ={gain:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "baseline": rec0,
        "best": best,
        "gain_all_f1": round(gain, 6),
        "useful": useful,
        "gate_auc_hit": round(gauc, 6),
        "grid": rows,
        "seconds": round(time.time() - t0, 1),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")

    if gen_sub:
        print("[2/2] 按最优门控出提交...")
        test_df = pd.read_csv(os.path.join(i011.DATA_DIR, "test.csv"))
        tsv_test = os.path.join(i011.CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
        full_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
        idx_t, bits_t = i011.parse_aln(tsv_test, test_df["protein_id"].values, full_map)
        hit_t = idx_t[:, 0] >= 0
        sim_bit_t = i011.agg_labels(idx_t, bits_t, y_full)
        sim_kmer_t = 0.0
        for kk, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
            if wt <= 0:
                continue
            d = np.load(os.path.join(
                i011.REL_DIR, "TCK002a_neigh_full.npz" if kk == 6
                else f"CK004_neigh_k{kk}_full.npz"))
            n2 = d["idx_test"].shape[0]
            lab = y_full[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
            s = d["sim_test"][:, :5].astype(np.float32)
            ww = np.maximum(s, 1e-7) ** 2
            sim_kmer_t = sim_kmer_t + wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
        sim_t = i011.ALIGN_W * sim_bit_t + (1.0 - i011.ALIGN_W) * sim_kmer_t
        tp = np.load(os.path.join(i011.REL_DIR, "TC003_test_probs.npz"))
        t1t = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
        t2t = np.load(os.path.join(i011.REL_DIR, "T002_test_probs.npz"))
        deep_t = w["t001a"] * t1t["t001a_test"] + w["t002"] * t2t["t002"] + w["cnn"] * tp["cnn"]
        ens_t = (deep_t + i011.SIM_W * sim_t
                 + i019.BETA * i011.col_norm(i011.rrf_pair(deep_t, sim_t, i019.K_RRF)))
        n_t = ens_t.shape[0]
        scut_raw_t = ens_t >= t_i011
        row_t = np.maximum(scut_raw_t.sum(1).astype(np.float64), 1.0)
        col_t = np.maximum(pi * n_t * RHO, 1e-6)
        col_t *= row_t.sum() / col_t.sum()
        p_t = i019.sinkhorn(
            np.exp((ens_t - ens_t.max(axis=1, keepdims=True)) / TAU_SK), row_t, col_t)
        mixed_t = i019.mix(ens_t, p_t, ALPHA)
        scut_t = (mixed_t >= t).astype(np.int8)
        test_len = test_df["sequence"].str.len().values.astype(np.float32)
        X_t = i008.make_features(mixed_t, scut_t, test_len, hit_t)
        clf = LogisticRegression(max_iter=400, class_weight="balanced")
        clf.fit(X[hit], y_gate[hit])
        gprob_t = np.zeros(len(test_df), np.float32)
        gprob_t[hit_t] = clf.predict_proba(X_t[hit_t])[:, 1].astype(np.float32)
        tau = float(best.get("tau", 0.80))
        gated_t = hit_t & (gprob_t >= tau)
        y_pred = i008.clip_ratio_rows(mixed_t, scut_t, gated_t, float(best.get("ratio", 0.85)))
        os.makedirs(i011.SUB_DIR, exist_ok=True)
        sub = pd.concat([test_df[["protein_id"]],
                         pd.DataFrame(y_pred, columns=label_cols)], axis=1)
        sub_path = os.path.join(i011.SUB_DIR, "submission_I020_ensemble.csv")
        sub.to_csv(sub_path, index=False)
        report["submission"] = sub_path
        report["n_clip_test"] = int(gated_t.sum())
        with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  提交: {sub_path} 截 {int(gated_t.sum())} hit")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
