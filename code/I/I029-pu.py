"""
I029-pu — Elkan–Noto PU 校正 I019：把 0 当未标注
================================================================
依据:
  - Elkan & Noto, KDD 2008, doi:10.1145/1401890.1401920
    只观察到正例与未标注。c=P(s=1|y=1) 用已标注正例的平均分数估计，
    P(y=1|x)=P(s=1|x)/c。
  - 第二轮调研把 PU 列为待评估：标签若是「已知注释」则 0≠负。
  - I004：残差校准头，列单调映射被 SCut 吞掉。
    本实验 clip(s/c,0,1) 在 s>c 处饱和，不再严格单调。
自己的改法:
  冻 I019。逐标签 c_j=mean(s|y=1)，下限 0.05。s'=clip(s/c,0,1)。
  混合 ens'=(1-α)·I019 + α·s'。α∈{0.3,0.6,1.0}。
  对照：只改频次低于中位数的尾标签。
  半折：c 在拟合折正例上估，评估折套用。门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I029-pu.py
输出:
  rel/I029_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I029_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
A_GRID = (0.3, 0.6, 1.0)
CMIN = 0.05


def estimate_c(score, y, cmin=CMIN):
    c = np.ones(score.shape[1], np.float64)
    for j in range(score.shape[1]):
        pos = y[:, j] == 1
        if pos.sum() == 0:
            continue
        c[j] = float(score[pos, j].mean())
    return np.clip(c, cmin, 1.0)


def pu_clip(score, c):
    return np.clip(score / c[None, :], 0.0, 1.0).astype(np.float32)


def mix_tail(base, pu, freq, alpha):
    out = base.copy()
    med = np.median(freq[freq > 0]) if (freq > 0).any() else 0.0
    tail = freq <= med
    out[:, tail] = (1.0 - alpha) * base[:, tail] + alpha * pu[:, tail]
    return out


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


def main():
    t0 = time.time()
    print("=" * 60)
    print("I029: Elkan–Noto PU 校正")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s019, hit, y_train, y_val = build_i019(train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    c_full = estimate_c(s019, y_val)
    print(f"  c: min={c_full.min():.3f} med={np.median(c_full):.3f} max={c_full.max():.3f}")
    pu_full = pu_clip(s019, c_full)
    freq = y_train.sum(0).astype(np.float64)

    rows = [rec0]
    configs = {"i019": s019}

    print("[1/2] α 全标签 / 只改尾 ...")
    rec_raw = i011.eval_ens(pu_full, y_val, hit, miss, "pu_raw")
    rec_raw["delta_i019"] = round(rec_raw["f1_100"] - rec0["f1_100"], 6)
    rows.append(rec_raw)
    configs["pu_raw"] = pu_full
    print(f"  pu_raw               F1={rec_raw['f1_100']:.4f} d={rec_raw['delta_i019']:+.4f}")

    for a in tqdm(A_GRID, desc="    α", unit="a"):
        mixed = (1.0 - a) * s019 + a * pu_full
        name = f"pu_a{a:.1f}"
        rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"alpha": a, "scope": "all"})
        rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        configs[name] = mixed
        print(f"  {name:20s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

        mixed_t = mix_tail(s019, pu_full, freq, a)
        name_t = f"pu_tail_a{a:.1f}"
        rec_t = i011.eval_ens(mixed_t, y_val, hit, miss, name_t,
                              {"alpha": a, "scope": "tail"})
        rec_t["delta_i019"] = round(rec_t["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec_t)
        configs[name_t] = mixed_t
        print(f"  {name_t:20s} F1={rec_t['f1_100']:.4f} d={rec_t['delta_i019']:+.4f}")

    print("[2/2] 半折核对（c 在拟合折估）...")
    rng = np.random.RandomState(i011.RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pu_score(scope, alpha):
        out = np.zeros_like(s019)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            c = estimate_c(s019[fit], y_val[fit])
            pu = pu_clip(s019[hold], c)
            if scope == "raw":
                out[hold] = pu
            elif scope == "all":
                out[hold] = (1.0 - alpha) * s019[hold] + alpha * pu
            else:
                tmp = s019[hold].copy()
                med = np.median(freq[freq > 0]) if (freq > 0).any() else 0.0
                tail = freq <= med
                tmp[:, tail] = (1.0 - alpha) * s019[hold][:, tail] + alpha * pu[:, tail]
                out[hold] = tmp
        return out

    oof_scores = {"i019": s019}
    oof_scores["pu_raw"] = oof_pu_score("raw", 1.0)
    for a in A_GRID:
        oof_scores[f"pu_a{a:.1f}"] = oof_pu_score("all", a)
        oof_scores[f"pu_tail_a{a:.1f}"] = oof_pu_score("tail", a)

    def oof_pred(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            t = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= t).astype(np.int8)
        return pred

    oof_rows = []
    oof_i019 = None
    for name, score in tqdm(oof_scores.items(), desc="    OOF", unit="cfg"):
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

    pu_oof = [r for r in oof_rows if r["name"] != "oof_i019"]
    best_oof = max(pu_oof, key=lambda r: r["f1_100"]) if pu_oof else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline_i019": rec0,
        "c_stats": {
            "min": round(float(c_full.min()), 4),
            "median": round(float(np.median(c_full)), 4),
            "max": round(float(c_full.max()), 4),
        },
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
