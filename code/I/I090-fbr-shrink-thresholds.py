"""
I090 — FBR 收缩阈值（完整版补测，Lin & Lin, CIKM 2023, doi:10.1145/3583780.3614996）
=====================================================================================
问题：现行 SCut = 逐标签 100 档裸 argmax（简化版），n_l 小的标签阈值噪声大，直接吃 Macro F1。
完整版（FBR 收缩）：t_l = (1 - γ_l)·t_global + γ_l·t_l*, γ_l = n_l / (n_l + κ)，
  κ ∈ {5, 10, 20, 50}；t_global = 全池微 F1 最优单阈值；n_l 取训练折支持数（避免折内噪声）。
协议：8:2 验证切分（random_state=42）上的 canonical 融合分（I064/I065 口径，锚点 F1=0.4475）；
  判据 = 半折 OOF 融合 Macro F1 较裸 SCut ≥ +0.001 才算有效（换锁线）。
附带诊断：头/中/尾（标签频次四分位）分带 F1 与阈值漂移，验证作用点在尾部。
文献：阈值收缩 / FBR（Lin & Lin 2023）；逐类阈值（Quevedo et al., Pattern Recognition 2012）。
输出：rel/I090_report.json
"""
import os
import io
import json
import time

import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
KAPPAS = [5.0, 10.0, 20.0, 50.0]
ANCHOR_F1 = 0.4475

i064 = SourceFileLoader(
    "i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")
).load_module()
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe


def per_label_f1(S, y, ths):
    preds = S >= ths
    yb = y.astype(bool)
    tp = (preds & yb).sum(0)
    fp = (preds & ~yb).sum(0)
    fn = (~preds & yb).sum(0)
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0.0)


def global_threshold(S, y):
    """全池微 F1 最优单阈值（100 档网格）。"""
    cands = np.linspace(0.01, 0.99, 100)
    yb = y.astype(bool)
    best, best_f1 = 0.5, -1.0
    for c in cands:
        pr = S >= c
        tp = float((pr & yb).sum())
        fp = float((pr & ~yb).sum())
        fn = float((~pr & yb).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if f1 > best_f1:
            best_f1, best = f1, c
    return best


def fbr_thresholds(S_fit, y_fit, n_tr, kappa):
    t_star = search_scut(S_fit, y_fit)
    t_g = global_threshold(S_fit, y_fit)
    gamma = n_tr / (n_tr + kappa)
    return (1.0 - gamma) * t_g + gamma * t_star, t_star, t_g


def build_sim(y_tr, n_va, val_pids, train_map, val_map):
    def parse_aln(tsv, qids, idmap, ev=0.001, mx=5):
        idx = np.full((len(qids), mx), -1, np.int64)
        bits = np.zeros((len(qids), mx), np.float32)
        qp = {q: i for i, q in enumerate(qids)}
        with open(tsv) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 5:
                    continue
                q, t = p[0], p[1]
                if q not in qp or t not in idmap:
                    continue
                i = qp[q]
                try:
                    bs, e = float(p[2]), float(p[3])
                except ValueError:
                    continue
                if e > ev:
                    continue
                j = 0
                while j < mx and bits[i, j] > bs:
                    j += 1
                if j < mx:
                    idx[i, j:] = np.roll(idx[i, j:], 1)
                    bits[i, j:] = np.roll(bits[i, j:], 1)
                    idx[i, j] = idmap[t]
                    bits[i, j] = bs
        return idx, bits

    def agg(idx, w, ys, tk=3):
        n = idx.shape[0]
        valid = idx[:, :tk] >= 0
        lab = np.zeros((n, tk, ys.shape[1]), np.float32)
        for j in range(tk):
            lab[:, j] = ys[idx[:, j].clip(min=0)]
        lab[~valid] = 0.0
        ww = np.maximum(w[:, :tk], 1e-7)
        ww[~valid] = 0.0
        return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)

    idx_v, bits_v = parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                              val_pids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim1 = agg(idx_v, bits_v, y_tr)
    v2v = {}
    with open(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            q, t = p[0], p[1]
            if q == t or q not in val_map or t not in val_map:
                continue
            qi, ti = val_map[q], val_map[t]
            v2v.setdefault(qi, [])
            if len(v2v[qi]) < 5:
                v2v[qi].append((ti, float(p[2])))
    sim2 = sim1.copy()
    for qi in range(n_va):
        if miss[qi] and qi in v2v:
            acc = np.zeros(N_LABELS, np.float32)
            aw = 0.0
            for ti, bs in v2v[qi]:
                if hit[ti]:
                    acc += sim1[ti] * bs
                    aw += bs
            if aw > 0:
                sim2[qi] = acc / aw
    kmer = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        pre = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(REL_DIR, f"{pre}_neigh_k{k}.npz"))
        n = d["idx_val"].shape[0]
        lab = y_tr[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        sm = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(sm, 1e-7) ** 2
        kmer += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return 0.4 * sim2 + 0.6 * kmer


def main():
    t0 = time.time()
    print("=" * 88)
    print("  I090: FBR shrinkage thresholds (Lin & Lin, CIKM 2023) — full version")
    print("=" * 88)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}
    n_tr_support = y_tr.sum(0).astype(np.float64)  # γ 用训练折支持数，避免折内噪声

    cnn = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn + 0.5 * c005)
    print("  building sim (DIAMOND + k-mer)...")
    sim = build_sim(y_tr, n_va, val_pids, train_map, val_map)
    plm1664 = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]

    f1c, aucc, mixed, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=plm1664, w=0.35)
    assert abs(f1c - ANCHOR_F1) < 0.0005, f"canonical anchor broken: {f1c}"
    print(f"  canonical anchor OK: F1={f1c:.4f} AUC={aucc:.4f}")

    variants = {"plain_scut": None}
    for kappa in KAPPAS:
        variants[f"fbr_k{kappa:g}"] = kappa

    # ---- 样本内（全 val 拟合并评估，与 0.4475 同口径）+ 频带诊断 ----
    bands = np.array_split(np.argsort(-n_tr_support), 4)  # 头/次头/次尾/尾 各 125 标签
    band_names = ["head-Q4", "Q3", "Q2", "tail-Q1"]
    in_sample = {}
    t_star_full = search_scut(mixed, y_va)
    for name, kappa in tqdm(variants.items(), desc="in-sample"):
        if kappa is None:
            ths = t_star_full
        else:
            ths, _, _ = fbr_thresholds(mixed, y_va, n_tr_support, kappa)
        f1 = macro_f1(mixed, y_va, ths)
        fl = per_label_f1(mixed, y_va, ths)
        preds = mixed >= ths
        yb = y_va.astype(bool)
        tp = (preds & yb).sum(0)
        fp = (preds & ~yb).sum(0)
        fn = (~preds & yb).sum(0)
        rr = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        pp = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        bands_f1 = {band_names[i]: float(fl[b].mean()) for i, b in enumerate(bands)}
        drift = np.abs(ths - t_star_full)
        bands_drift = {band_names[i]: float(drift[b].mean()) for i, b in enumerate(bands)}
        in_sample[name] = {"f1": f1, "recall": float(rr.mean()), "precision": float(pp.mean()),
                           "band_f1": bands_f1, "band_drift": bands_drift}
        print(f"  in-sample {name:12s} F1={f1:.4f} R={rr.mean():.4f} P={pp.mean():.4f}  "
              f"drift(head/tail)={bands_drift['head-Q4']:.4f}/{bands_drift['tail-Q1']:.4f}")

    # ---- 半折 OOF：A 拟合 → B 评估，B 拟合 → A 评估，拼全 val 算 Macro F1 ----
    a_idx, b_idx = train_test_split(np.arange(n_va), test_size=0.5, random_state=42)
    oof = {}
    for name, kappa in tqdm(variants.items(), desc="half-fold OOF"):
        preds = np.zeros_like(y_va, dtype=bool)
        for fit_idx, eval_idx in ((a_idx, b_idx), (b_idx, a_idx)):
            Sf, yf = mixed[fit_idx], y_va[fit_idx]
            if kappa is None:
                ths = search_scut(Sf, yf)
            else:
                ths, _, _ = fbr_thresholds(Sf, yf, n_tr_support, kappa)
            preds[eval_idx] = mixed[eval_idx] >= ths
        yb = y_va.astype(bool)
        tp = (preds & yb).sum(0)
        fp = (preds & ~yb).sum(0)
        fn = (~preds & yb).sum(0)
        p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        fl = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0.0)
        oof[name] = {
            "f1": float(fl.mean()),
            "band_f1": {band_names[i]: float(fl[b].mean()) for i, b in enumerate(bands)},
        }
        print(f"  OOF       {name:12s} F1={oof[name]['f1']:.4f}")

    base_oof = oof["plain_scut"]["f1"]
    verdict = {}
    for name, kappa in variants.items():
        if kappa is None:
            continue
        d = oof[name]["f1"] - base_oof
        verdict[name] = {"delta_oof": d, "valid": bool(d >= 0.001)}
    best_name = max((n for n in variants if n != "plain_scut"), key=lambda n: oof[n]["f1"])
    best_d = verdict[best_name]["delta_oof"]
    conclusion = "有效" if best_d >= 0.001 else "关闭（未过 +0.001 换锁线）"

    rep = {
        "experiment": "I090",
        "method": "FBR shrinkage thresholds (Lin & Lin, CIKM 2023)",
        "anchor_f1": f1c,
        "anchor_auc": aucc,
        "in_sample": in_sample,
        "oof": oof,
        "verdict": verdict,
        "best": best_name,
        "best_delta_oof": best_d,
        "conclusion": conclusion,
    }
    with io.open(os.path.join(REL_DIR, "I090_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 88)
    print(f"  best = {best_name}  ΔOOF = {best_d:+.4f}  → {conclusion}")
    print("=" * 88)
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
