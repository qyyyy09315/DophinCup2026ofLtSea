"""
C011b-logitavg — C002b 与 C005 logits 平均替换 CNN 路，接到 C011a
================================================================
依据:
  - Njieutcheu Tassi C, Gawlikowski J, Fitri AU, Triebel R.
    The impact of averaging logits over probabilities on ensembles of
    neural networks. AISafety 2022, CEUR-WS Vol. 3215.
    https://ceur-ws.org/Vol-3215/19.pdf
    欠自信时 logits 平均抬置信、降 ECE；过自信时伤 TP/FP 可分。
  - C011 / C011a：概率算术平均 α=0.50 接 I019，平台锁。
    本实验不重训、不蒸馏（C023 已关），只把平均从 σ 空间改到 logit 空间。
自己的改法:
  cnn' = σ( (1-α)·logit(C002b) + α·logit(C005) )，α∈{0.30,0.50,0.70}。
  重建 C011a（deep = 0.1·T001a + 0.2·T002 + 0.2·cnn'，再 I011+I019）。
  对照：同 α 的概率平均（复现 C011a 点）。门槛半折 OOF 较 C011a ≥ +0.001。
用法:
  python code/C/C011b-logitavg.py
输出:
  rel/C011b_report.json
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
ck007 = load_mod("ck007", os.path.join("CK", "CK007-pident.py"))

REPORT_PATH = os.path.join(i011.REL_DIR, "C011b_report.json")
C011A_ALPHA = 0.50
CNN_W = 0.2
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
A_GRID = (0.30, 0.50, 0.70)
EPS = 1e-6


def logit(p):
    x = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    return np.log(x) - np.log(1.0 - x)


def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def mix_prob(a, b, alpha):
    return ((1.0 - alpha) * a + alpha * b).astype(np.float32)


def mix_logit(a, b, alpha):
    return sigmoid((1.0 - alpha) * logit(a) + alpha * logit(b))


def fuse(deep, sim, y_train, y_val):
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


def pack_slices(rec, pred, y, twilight, remote, strong):
    rec["f1_twilight"] = round(i011.macro_f1(pred[twilight], y[twilight]), 6) if twilight.any() else 0.0
    rec["f1_remote"] = round(i011.macro_f1(pred[remote], y[remote]), 6) if remote.any() else 0.0
    rec["f1_strong"] = round(i011.macro_f1(pred[strong], y[strong]), 6) if strong.any() else 0.0
    return rec


def f1_05(p, y):
    return i011.macro_f1((p >= 0.5).astype(np.int8), y)


def main():
    t0 = time.time()
    print("=" * 60)
    print("C011b: C002b×C005 logit average into C011a")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v, pid_v = ck007.parse_aln_pid(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    pident = pid_v[:, 0].copy()
    pident[miss] = 0.0
    twilight = hit & (pident < 80) & (pident >= 50)
    remote = hit & (pident < 50)
    strong = hit & (pident >= 80)

    cnn42 = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))["val"]
    y5 = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))["y_true"]
    if y5.shape != y_val.shape or not np.array_equal(y5, y_val):
        raise RuntimeError("C005 y_true 与当前切分不对齐")

    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep_rest = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"]

    cnn_lock = mix_prob(cnn42, c005, C011A_ALPHA)
    deep_lock = deep_rest + CNN_W * cnn_lock
    s_lock = fuse(deep_lock, sim, y_train, y_val)
    pred0 = (s_lock >= i011.search_t(s_lock, y_val)).astype(np.int8)
    rec0 = pack_slices(i011.eval_ens(s_lock, y_val, hit, miss, "c011a"),
                       pred0, y_val, twilight, remote, strong)
    rec0.update({
        "alpha": C011A_ALPHA, "mix": "prob",
        "f1_05": round(f1_05(cnn_lock, y_val), 6),
        "delta_c011a": 0.0,
    })
    print(f"  C011a F1={rec0['f1_100']:.4f} F1(0.5)={rec0['f1_05']:.4f}")

    configs = {"c011a": s_lock}
    rows = [rec0]
    for a in A_GRID:
        for mix_name, mixer in (("prob", mix_prob), ("logit", mix_logit)):
            if mix_name == "prob" and abs(a - C011A_ALPHA) < 1e-12:
                continue
            cnn = mixer(cnn42, c005, a)
            deep = deep_rest + CNN_W * cnn
            s = fuse(deep, sim, y_train, y_val)
            pred = (s >= i011.search_t(s, y_val)).astype(np.int8)
            name = f"{mix_name}_a{a:.2f}"
            rec = pack_slices(i011.eval_ens(s, y_val, hit, miss, name),
                              pred, y_val, twilight, remote, strong)
            rec.update({
                "alpha": a, "mix": mix_name,
                "f1_05": round(f1_05(cnn, y_val), 6),
                "delta_c011a": round(rec["f1_100"] - rec0["f1_100"], 6),
            })
            rows.append(rec)
            configs[name] = s
            print(f"  {name:14s} F1(0.5)={rec['f1_05']:.4f} "
                  f"F1={rec['f1_100']:.4f} d={rec['delta_c011a']:+.4f}")

    rng = np.random.RandomState(i011.RANDOM_STATE)
    perm = rng.permutation(len(y_val))
    mid = len(y_val) // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            th = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= th).astype(np.int8)
        return pred

    oof_rows, oof_lock = [], None
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "c011a":
            oof_lock = rec["f1_100"]
        print(f"  {rec['name']:22s} F1={rec['f1_100']:.4f} miss={rec['f1_miss']:.4f}")
    for rec in oof_rows:
        rec["delta_c011a"] = round(rec["f1_100"] - oof_lock, 6)
    cand = [r for r in oof_rows if r["name"] != "oof_c011a"]
    best = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best["f1_100"] - oof_lock
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best['name']} vs C011a {gain:+.4f} → "
          f"{'有效' if useful else '无增益'}")
    report = {
        "baseline_c011a": rec0,
        "oof_c011a": oof_lock,
        "best_oof": best,
        "gain_oof_vs_c011a": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "logits 平均 OOF≥+0.001"
            if useful else "logits 平均无增益，关闭；CNN 路锁概率平均"
        ),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}  用时 {report['seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
