"""
I035-sysrrf — 平台锁 I008 与本地锁 I011 做系统级 RRF
================================================================
依据:
  - Cormack, Clarke & Buettcher, SIGIR 2009, doi:10.1145/1571941.1572114
    RRF 融两路排序，不校准分数。
  - Wolpert, Neural Networks 1992, doi:10.1016/S0893-6080(05)80023-1
    stacked generalization：组合的是完整系统，不是同一套分数的截断。
  - I008 平台 45.3902，压 hit 过预测；I011 本地 +0.0055，抬召回。
  - I016：两套 SCut 投票，负。本实验融分数，不融硬决策。
  - I020/I026：把 I008 截断叠回 I011/I019，负。本实验不截 I011。
自己的改法:
  冻 CK005a 分数。I008：半折门控，P≥0.80 的 hit 把非保留标签分数置 0
  （保留 top round(n_scut×0.85)），得到 clipped 分数。
  与 I011 做 RRF：ens' = I011 + β·col_norm(RRF(I011, clipped))。
  k=30，β∈{0.20,0.50}。另混 (1-γ)·I019 + γ·ens'，γ∈{0.15,0.30}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I035-sysrrf.py
输出:
  rel/I035_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I035_report.json")
K_RRF, BETA_I011 = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
GATE_T, GATE_R = 0.80, 0.85
BETA_GRID = (0.20, 0.50)
G_GRID = (0.15, 0.30)


def clip_scores(ens, scut, mask, ratio):
    out = ens.copy()
    idx = np.where(mask)[0]
    scard = scut.sum(1)
    order = np.argsort(-ens, axis=1)
    for i in tqdm(idx, desc="    clip", unit="row", leave=False):
        keep = max(1, int(round(float(scard[i]) * ratio)))
        ranked = [j for j in order[i] if scut[i, j]]
        if len(ranked) <= keep:
            continue
        drop = ranked[keep:]
        out[i, drop] = 0.0
    return out


def sinkhorn_on(ens, y_train, y_val):
    n = ens.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t = i011.search_t(ens, y_val)
    row_m = np.maximum((ens >= t).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((ens - ens.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    return i019.mix(ens, p, ALPHA_SK)


def main():
    t0 = time.time()
    print("=" * 60)
    print("I035: I008 × I011 系统级 RRF")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    val_len = train_df["sequence"].str.len().values[idx_val].astype(np.float32)
    card = y_val.sum(1).astype(np.float32)
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    print(f"  hit {int(hit.sum())}/{len(hit)}")

    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    ck = deep + i011.SIM_W * sim
    s011 = ck + BETA_I011 * i011.col_norm(i011.rrf_pair(deep, sim, K_RRF))
    s019 = sinkhorn_on(s011, y_train, y_val)

    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    rec011 = i011.eval_ens(s011, y_val, hit, miss, "i011")
    rec_ck = i011.eval_ens(ck, y_val, hit, miss, "ck005a")
    print(f"  I019 F1={rec0['f1_100']:.4f}  I011 F1={rec011['f1_100']:.4f}  "
          f"CK005a F1={rec_ck['f1_100']:.4f}")

    t_ck = i011.search_t(ck, y_val)
    scut = (ck >= t_ck).astype(np.int8)
    y_gate = (scut.sum(1) > card).astype(np.int8)
    X = i008.make_features(ck, scut, val_len, hit)
    gprob, gauc = i008.oof_gate(X, y_gate, hit)
    gated = hit & (gprob >= GATE_T)
    print(f"  门控 AUC={gauc:.3f}  τ={GATE_T:.2f} 触发 {int(gated.sum())}/{int(hit.sum())}")
    clipped = clip_scores(ck, scut, gated, GATE_R)
    rec_clip = i011.eval_ens(clipped, y_val, hit, miss, "i008clip")
    print(f"  I008-clip F1={rec_clip['f1_100']:.4f}")

    rows = [rec0, rec011, rec_ck, rec_clip]
    configs = {"i019": s019, "i011": s011}

    print("[1/2] β × γ ...")
    rrf = i011.rrf_pair(s011, clipped, K_RRF)
    rrf_n = i011.col_norm(rrf)
    for b in tqdm(BETA_GRID, desc="    β", unit="b"):
        ens = s011 + b * rrf_n
        rec = i011.eval_ens(ens, y_val, hit, miss, f"rrf_b{b:.2f}", {"beta": b})
        rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        configs[f"rrf_b{b:.2f}"] = ens
        print(f"  rrf_b{b:.2f} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")
        s_n = i011.col_norm(ens)
        for g in G_GRID:
            mixed = (1.0 - g) * s019 + g * s_n
            name = f"mix_b{b:.2f}_g{g:.2f}"
            rec_m = i011.eval_ens(mixed, y_val, hit, miss, name, {"beta": b, "gamma": g})
            rec_m["delta_i019"] = round(rec_m["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec_m)
            configs[name] = mixed
            print(f"  {name:20s} F1={rec_m['f1_100']:.4f} d={rec_m['delta_i019']:+.4f}")

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

    cand = [r for r in oof_rows if r["name"] not in ("oof_i019", "oof_i011")]
    best_oof = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline_i019": rec0,
        "i011": rec011,
        "ck005a": rec_ck,
        "i008clip": rec_clip,
        "best_oof": best_oof,
        "gain_oof_vs_i019": round(gain, 6),
        "useful": useful,
        "gate_auc": round(gauc, 4),
        "n_clip": int(gated.sum()),
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
