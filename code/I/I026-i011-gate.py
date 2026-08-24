"""
I026-i011-gate — I008 门控 hit 截断叠到 I011 分数
================================================================
依据:
  - I008 平台 45.3902 超过 CK005a 45.3039（本地仅 +0.0008）
  - I011 RRF k=30 β=0.50 本地 0.4443 / OOF +0.0055
  - I020：同一套门控叠到 I019，门控 AUC 0.652，最优 −0.0007，关闭。
    Sinkhorn 已压过预测，再截伤召回。I011 还没压过预测，门控可能仍有效。
  - Yang, SIGIR 2001, doi:10.1145/383952.383975
自己的改法:
  冻 I011 分数与 SCut。I008 半折 Logistic 学 P(过预测)，
  对 P≥τ 的 hit 行按比例截。τ×r 重扫。
  门槛相对 I011 ≥ +0.001。
用法:
  python code/I/I026-i011-gate.py
输出:
  rel/I026_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I026_report.json")
K_RRF, BETA = 30, 0.50
GATE_T = (0.50, 0.60, 0.70, 0.80)
GATE_R = (0.70, 0.80, 0.85)


def build_i011(train_df, y_full, idx_train, idx_val):
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
    ens = deep + i011.SIM_W * sim + BETA * i011.col_norm(i011.rrf_pair(deep, sim, K_RRF))
    return ens, hit, y_val


def main():
    t0 = time.time()
    print("=" * 60)
    print("I026: I008 门控截断 × I011 分数")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    ens, hit, y_val = build_i011(train_df, y_full, idx_train, idx_val)
    miss = ~hit
    val_len = train_df["sequence"].str.len().values[idx_val].astype(np.float32)
    t = i011.search_t(ens, y_val)
    scut = (ens >= t).astype(np.int8)
    card = y_val.sum(1).astype(np.float32)
    y_gate = (scut.sum(1) > card).astype(np.int8)
    print(f"  hit 多报率={y_gate[hit].mean():.3f}")

    X = i008.make_features(ens, scut, val_len, hit)
    gprob, gauc = i008.oof_gate(X, y_gate, hit)
    print(f"  门控 OOF AUC(hit)={gauc:.3f}")

    rec0 = i011.eval_ens(ens, y_val, hit, miss, "i011")
    rec0["name"] = "i011_scut"
    print(f"  I011 SCut F1={rec0['f1_100']:.4f}")
    rows = [rec0]

    print("[1/2] 门控网格...")
    for tau in tqdm(GATE_T, desc="    τ", unit="t"):
        gated = hit & (gprob >= tau)
        for r in GATE_R:
            pred = i008.clip_ratio_rows(ens, scut, gated, r)
            rec = {
                "name": f"gate_t{tau:.2f}_r{r:.2f}",
                "tau": tau, "ratio": r, "n_clip": int(gated.sum()),
                "f1_100": round(i011.macro_f1(pred, y_val), 6),
                "recall": round(i011.macro_rp(pred, y_val)[0], 6),
                "precision": round(i011.macro_rp(pred, y_val)[1], 6),
                "auc": rec0["auc"],
                "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
                "f1_hit": round(i011.macro_f1(pred[hit], y_val[hit]), 6),
                "pred_card": round(float(pred.sum(1).mean()), 3),
            }
            rec["delta"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            print(f"  {rec['name']:22s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f} "
                  f"n={rec['n_clip']}")

    ranked = [r for r in rows if r["name"] != "i011_scut"]
    best = max(ranked, key=lambda r: r["f1_100"])
    gain = best["f1_100"] - rec0["f1_100"]
    useful = bool(gain >= 0.001)
    print(f"  全 val 最优 {best['name']} F1={best['f1_100']:.4f} Δ={gain:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    print("[2/2] 半折核对最优与基线...")
    rng = np.random.RandomState(i011.RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]
    oof011 = np.zeros_like(scut)
    oofg = np.zeros_like(scut)
    mask_full = hit & (gprob >= best["tau"])
    for i, hold in enumerate(folds):
        fit = folds[1 - i]
        t_fit = i011.search_t(ens[fit], y_val[fit])
        scut_h = (ens[hold] >= t_fit).astype(np.int8)
        oof011[hold] = scut_h
        tmp = np.zeros_like(scut)
        tmp[hold] = scut_h
        clipped = i008.clip_ratio_rows(ens, tmp, mask_full, best["ratio"])
        oofg[hold] = clipped[hold]
    oof011_f1 = i011.macro_f1(oof011, y_val)
    oofg_f1 = i011.macro_f1(oofg, y_val)
    print(f"  OOF I011={oof011_f1:.4f}  gate={oofg_f1:.4f}  d={oofg_f1 - oof011_f1:+.4f}")
    useful_oof = bool(oofg_f1 - oof011_f1 >= 0.001)

    report = {
        "baseline": rec0,
        "best": best,
        "gain_all_f1": round(gain, 6),
        "oof_i011": round(oof011_f1, 6),
        "oof_best": round(oofg_f1, 6),
        "gain_oof": round(oofg_f1 - oof011_f1, 6),
        "useful": useful_oof,
        "gate_auc_hit": round(gauc, 6),
        "grid": rows,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(i011.REL_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  用时 {report['seconds']}s")


if __name__ == "__main__":
    main()
