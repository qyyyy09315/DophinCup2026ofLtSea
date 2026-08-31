"""
I044-lowcard — 低基数半折专家（冻 C011a 分数）
================================================================
依据:
  - A001：card≤10 共 7240/22760，C011a F1=0.2017（全局 0.4453）。
    这是实例基数，不是稀有 GO 项。
  - I009：hit/miss 两套 SCut，miss 阈值降到 0.32，全局 −0.015。
    本实验只允许抬阈值 / 截断，禁止降低阈值。
  - I008：半折 Logistic 门控后再截；门控不能用真基数。
  - Yang Y, SIGIR 2001, doi:10.1145/383952.383975（SCut）。
自己的改法:
  冻 C011a 分数。半折 Logistic 学 P(真基数≤10)，特征不含标签。
  P≥τ 的行才换专家：抬阈值 t_low=max(t_low, t_g)，或只删 SCut 正例
  里分数最低的（clip_k / clip_ratio）。高基数 hit 不动。
  真基数门控只作 oracle 上界。门槛：相对 C011a 半折 OOF ≥ +0.001。
用法:
  python code/I/I044-lowcard.py
输出:
  rel/I044_report.json
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I044_report.json")
C011A_ALPHA = 0.50
CNN_W = 0.2
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
CARD_CUT = 10
TAU_GRID = (0.45, 0.55, 0.65, 0.75)
K_GRID = (6, 8, 10, 12)
R_GRID = (0.40, 0.55, 0.70)


def build_c011a(deep, sim, y_train, y_val):
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


def make_features(score, hit, val_len, neigh_card, bit1):
    p70 = np.quantile(score, 0.70, axis=1)
    p80 = np.quantile(score, 0.80, axis=1)
    return np.column_stack([
        score.sum(1),
        np.log1p(val_len),
        (score >= 0.4).sum(1),
        (score >= 0.5).sum(1),
        (score >= 0.6).sum(1),
        p70, p80, score.max(1),
        hit.astype(np.float32),
        neigh_card.astype(np.float32),
        np.log1p(bit1),
    ]).astype(np.float32)


def search_t_raise(ens, y, t_floor):
    t = i011.search_t(ens, y)
    return np.maximum(t, t_floor)


def clip_k_rows(score, scut, mask, k):
    pred = scut.copy()
    if not mask.any():
        return pred
    masked = np.where(scut.astype(bool), score, -np.inf)
    order = np.argsort(-masked, axis=1)
    keep = np.zeros_like(scut)
    idx = np.arange(scut.shape[0])
    for j in range(int(k)):
        cols = order[:, j]
        valid = np.isfinite(masked[idx, cols])
        keep[idx[valid], cols[valid]] = 1
    pred[mask] = keep[mask]
    return pred


def clip_ratio_rows(score, scut, mask, ratio):
    pred = scut.copy()
    if not mask.any():
        return pred
    scut_b = scut.astype(bool)
    masked = np.where(scut_b, score, -np.inf)
    order = np.argsort(-masked, axis=1)
    scard = scut_b.sum(1)
    keep_n = np.minimum(np.maximum(1, np.round(scard * ratio).astype(np.int32)), scard)
    ranks = np.empty_like(scut, dtype=np.int32)
    idx = np.arange(scut.shape[0])[:, None]
    ranks[idx, order] = np.arange(scut.shape[1], dtype=np.int32)
    keep = (ranks < keep_n[:, None]) & scut_b
    pred[mask] = keep[mask].astype(np.int8)
    return pred


def pack(pred, y, hit, miss, low, name, extra=None, score=None):
    extra = dict(extra) if extra else {}
    r, p = i011.macro_rp(pred, y)
    rec = {
        "name": name,
        "f1_100": round(i011.macro_f1(pred, y), 6),
        "recall": round(r, 6),
        "precision": round(p, 6),
        "f1_miss": round(i011.macro_f1(pred[miss], y[miss]), 6) if miss.any() else 0.0,
        "f1_hit": round(i011.macro_f1(pred[hit], y[hit]), 6) if hit.any() else 0.0,
        "f1_low": round(i011.macro_f1(pred[low], y[low]), 6) if low.any() else 0.0,
        "f1_high": round(i011.macro_f1(pred[~low], y[~low]), 6) if (~low).any() else 0.0,
        "pred_card": round(float(pred.sum(1).mean()), 3),
        "pred_card_low": round(float(pred[low].sum(1).mean()), 3) if low.any() else 0.0,
        "n_gated": int(extra.pop("n_gated", 0)),
    }
    if score is not None:
        rec["auc"] = round(i011.macro_auc(y, score), 6)
    rec.update(extra)
    return rec


def apply_expert(score, scut, t_low, gated, kind, param):
    if not gated.any():
        return scut.copy()
    if kind == "raise":
        pred = scut.copy()
        pred[gated] = (score[gated] >= t_low).astype(np.int8)
        return pred
    if kind == "clip_k":
        return clip_k_rows(score, scut, gated, param)
    if kind == "clip_r":
        return clip_ratio_rows(score, scut, gated, param)
    raise ValueError(kind)


def main():
    t0 = time.time()
    print("=" * 60)
    print("I044: 低基数半折专家（冻 C011a）")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    val_len = np.array([len(s) for s in train_df["sequence"].values[idx_val]], np.float32)
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    card = y_val.sum(1)
    low = card <= CARD_CUT
    print(f"  miss={int(miss.sum())}  card≤{CARD_CUT}={int(low.sum())}/{len(y_val)}")

    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    cnn42 = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_val = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))["val"]
    cnn_lock = (1.0 - C011A_ALPHA) * cnn42 + C011A_ALPHA * c005_val
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + CNN_W * cnn_lock
    score = build_c011a(deep, sim, y_train, y_val)

    neigh_card = np.zeros(len(y_val), np.float32)
    neigh_card[hit] = y_train[idx_v[hit, 0]].sum(1).astype(np.float32)
    bit1 = bits_v[:, 0].copy()
    bit1[miss] = 0.0
    X_base = make_features(score, hit, val_len, neigh_card, bit1)
    y_gate = low.astype(np.int8)

    rec0 = i011.eval_ens(score, y_val, hit, miss, "c011a")
    pred0 = (score >= i011.search_t(score, y_val)).astype(np.int8)
    rec0["f1_low"] = round(i011.macro_f1(pred0[low], y_val[low]), 6)
    rec0["f1_high"] = round(i011.macro_f1(pred0[~low], y_val[~low]), 6)
    rec0["pred_card_low"] = round(float(pred0[low].sum(1).mean()), 3)
    print(f"  C011a F1={rec0['f1_100']:.4f} low={rec0['f1_low']:.4f} "
          f"high={rec0['f1_high']:.4f} pred_card_low={rec0['pred_card_low']:.1f}")

    rng = np.random.RandomState(i011.RANDOM_STATE)
    perm = rng.permutation(len(y_val))
    mid = len(y_val) // 2
    folds = [perm[:mid], perm[mid:]]

    def act_key(kind, param):
        return kind if param is None else f"{kind}_{param}"

    actions = [("raise", None)]
    actions += [("clip_k", k) for k in K_GRID]
    actions += [("clip_r", r) for r in R_GRID]

    oof_lock = np.zeros_like(score, dtype=np.int8)
    oof_oracle = {act_key(kind, param): np.zeros_like(score, dtype=np.int8)
                  for kind, param in actions}
    oof_gate = {f"t{tau:.2f}_{act_key(kind, param)}": np.zeros_like(score, dtype=np.int8)
                for tau in TAU_GRID for kind, param in actions}
    proba = np.zeros(len(y_val), np.float32)
    n_gated = {k: 0 for k in oof_gate}
    gate_aucs = []

    print("[1/2] 半折门控 + 专家 ...")
    for i, hold in enumerate(folds):
        fit = folds[1 - i]
        t_g = i011.search_t(score[fit], y_val[fit])
        scut_hold = (score[hold] >= t_g).astype(np.int8)
        oof_lock[hold] = scut_hold

        scard_fit = (score[fit] >= t_g).sum(1).astype(np.float32)
        scard_hold = scut_hold.sum(1).astype(np.float32)
        X_fit = np.column_stack([X_base[fit], scard_fit])
        X_hold = np.column_stack([X_base[hold], scard_hold])
        clf = LogisticRegression(max_iter=400, class_weight="balanced")
        clf.fit(X_fit, y_gate[fit])
        proba[hold] = clf.predict_proba(X_hold)[:, 1]
        if y_gate[hold].min() != y_gate[hold].max():
            gate_aucs.append(roc_auc_score(y_gate[hold], proba[hold]))

        low_fit = low[fit]
        if int(low_fit.sum()) < 50:
            t_low = t_g
        else:
            t_low = search_t_raise(score[fit][low_fit], y_val[fit][low_fit], t_g)

        oracle = low[hold]
        for kind, param in actions:
            key = act_key(kind, param)
            oof_oracle[key][hold] = apply_expert(
                score[hold], scut_hold, t_low, oracle, kind, param)

        for tau in TAU_GRID:
            gated = proba[hold] >= tau
            for kind, param in actions:
                key = f"t{tau:.2f}_{act_key(kind, param)}"
                n_gated[key] += int(gated.sum())
                oof_gate[key][hold] = apply_expert(
                    score[hold], scut_hold, t_low, gated, kind, param)

    gate_auc = float(np.mean(gate_aucs)) if gate_aucs else 0.5
    print(f"  门控 OOF AUC={gate_auc:.3f}  Pmean={proba.mean():.3f}  "
          f"P(low)={proba[low].mean():.3f}  P(high)={proba[~low].mean():.3f}")

    rec_lock = pack(oof_lock, y_val, hit, miss, low, "oof_c011a", score=score)
    print(f"  {rec_lock['name']:28s} F1={rec_lock['f1_100']:.4f} "
          f"low={rec_lock['f1_low']:.4f} high={rec_lock['f1_high']:.4f}")

    oof_rows = [rec_lock]
    print("[2/2] 评估网格 ...")
    for kind, param in actions:
        rec = pack(
            oof_oracle[act_key(kind, param)], y_val, hit, miss, low,
            f"oof_oracle_{act_key(kind, param)}",
            {"n_gated": int(low.sum()), "kind": kind, "param": param,
             "oracle": True},
            score=score)
        rec["delta_c011a"] = round(rec["f1_100"] - rec_lock["f1_100"], 6)
        rec["delta_high"] = round(rec["f1_high"] - rec_lock["f1_high"], 6)
        oof_rows.append(rec)
        print(f"  {rec['name']:28s} F1={rec['f1_100']:.4f} d={rec['delta_c011a']:+.4f} "
              f"low={rec['f1_low']:.4f} high={rec['f1_high']:.4f}")

    for tau in tqdm(TAU_GRID, desc="    τ", unit="t"):
        for kind, param in actions:
            key = f"t{tau:.2f}_{act_key(kind, param)}"
            rec = pack(
                oof_gate[key], y_val, hit, miss, low,
                f"oof_{key}",
                {"n_gated": n_gated[key], "tau": tau, "kind": kind,
                 "param": param, "oracle": False},
                score=score)
            rec["delta_c011a"] = round(rec["f1_100"] - rec_lock["f1_100"], 6)
            rec["delta_high"] = round(rec["f1_high"] - rec_lock["f1_high"], 6)
            oof_rows.append(rec)
            print(f"  {rec['name']:28s} F1={rec['f1_100']:.4f} d={rec['delta_c011a']:+.4f} "
                  f"low={rec['f1_low']:.4f} high={rec['f1_high']:.4f} "
                  f"gated={rec['n_gated']}")

    legal = [r for r in oof_rows if r["name"] != "oof_c011a" and not r.get("oracle")]
    best = max(legal, key=lambda r: r["f1_100"]) if legal else rec_lock
    gain = best["f1_100"] - rec_lock["f1_100"]
    high_ok = best["f1_high"] + 1e-12 >= rec_lock["f1_high"] - 0.0005
    useful = bool(gain >= 0.001 and high_ok)
    oracle_rows = [r for r in oof_rows if r.get("oracle")]
    best_oracle = max(oracle_rows, key=lambda r: r["f1_100"]) if oracle_rows else rec_lock
    print(f"  OOF 最优 {best['name']} vs C011a {gain:+.4f} "
          f"high {best['delta_high']:+.4f} → {'有效' if useful else '无增益'}")
    print(f"  oracle 上界 {best_oracle['name']} d={best_oracle['delta_c011a']:+.4f}")

    report = {
        "n_low": int(low.sum()),
        "card_cut": CARD_CUT,
        "gate_auc": round(gate_auc, 6),
        "baseline_c011a": rec0,
        "oof_c011a": rec_lock,
        "best_oof": best,
        "best_oracle": best_oracle,
        "gain_oof_vs_c011a": round(gain, 6),
        "useful": useful,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "C011a OOF≥+0.001 且高基数不掉，可换低基数专家"
            if useful else "低基数半折专家无增益，关闭；不重开 I009 分层 SCut"
        ),
    }
    os.makedirs(i011.REL_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}")
    print(f"  用时 {report['seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
