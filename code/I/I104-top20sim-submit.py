"""
I104 — top20 同源聚合提交版（I085 翻案增益的平台检验）
============================================================
背景:
  - I105 重复半折配对检验证实 I085 top20 sim 为统计显著真增益
    （配对 Δ=+0.0008 ± 0.0001，CI95 下界 +0.0006，P(Δ>0)=0.95）；
  - I097 平台先例: 本地微增益在平台可放大（本地 −0.0024 -> 平台 +0.0090）。
  - 本刀把 sim 比对支从 top-3 聚合升级为 top-20 聚合（top-100 比对已备），
    val/test 同口径重校阈值，生成新平台提交。
链路（I097/I081 同约定）:
  val 侧: deep5 = 0.65*deep4 + 0.35*plm_multiseed; sim = 0.4*(top20聚合+2hop) + 0.6*kmer
          full_pipe -> t_tmp(s级) + opt_ths(mixed级)
  test 侧: plm = I097 三种子全量 78ep 头均值; sim_te = 0.4*(top20聚合+2hop) + 0.6*kmer
          s_te = deep5 + 0.75*sim + 0.5*col_norm(RRF); row_m 用 t_tmp; Sinkhorn(τ=0.20);
          final = 0.4*s + 0.6*p; 二值化 opt_ths
输出: submit/submission_I104_top20sim.csv (+桌面副本), rel/I104_report.json
用法: python code/I/I104-top20sim-submit.py
"""

import os
import json
import shutil
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
SEED = 42

i096 = SourceFileLoader("i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")).load_module()
i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
i085 = SourceFileLoader("i085", os.path.join(PROJECT_ROOT, "code", "I", "I085-top100-gated-agg.py")).load_module()
g004 = SourceFileLoader("g004", os.path.join(PROJECT_ROOT, "code", "G", "G004-latent-flow-matching.py")).load_module()
MLP = i064.MLP
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe
ranks, col_norm, sinkhorn = i064.ranks, i064.col_norm, i064.sinkhorn
load_1664d_features = g004.load_1664d_features


@torch.no_grad()
def mlp_test_mean(device):
    """I097 三种子全量 78ep 头的 test 概率均值"""
    df_e = pd.read_parquet(os.path.join(REL_DIR, "test_esm2_embeddings.parquet"))
    df_p = pd.read_parquet(os.path.join(REL_DIR, "test_embeddings.parquet"))
    ec = [c for c in df_e.columns if c.startswith("emb_")]
    pc = [c for c in df_p.columns if c.startswith("emb_")]
    X = np.concatenate([df_e[ec].values, df_p[pc].values], axis=1).astype(np.float32)
    preds = []
    for s in (11, 22, 33):
        m = MLP(X.shape[1]).to(device)
        m.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"i097_fulltrain_s{s}.pt"),
                                     map_location=device, weights_only=True))
        m.eval()
        out = []
        for k in range(0, len(X), 512):
            bx = torch.from_numpy(X[k:k + 512]).to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                out.append(torch.sigmoid(m(bx).float()).cpu().numpy())
        preds.append(np.vstack(out))
        print(f"  seed {s} 推理完成")
    return np.mean(preds, axis=0)


def parse_2hop(path, qmap, tmap):
    """val_to_val / test_to_test 的 top-5 邻居（2-hop 用）"""
    nbr = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            q, t = p[0], p[1]
            if q == t or q not in qmap or t not in tmap:
                continue
            qi, ti = qmap[q], tmap[t]
            nbr.setdefault(qi, [])
            if len(nbr[qi]) < 5:
                nbr[qi].append((ti, float(p[2])))
    return nbr


def two_hop(sim1, hit, nbr, n):
    out = sim1.copy()
    for qi in range(n):
        if not hit[qi] and qi in nbr:
            acc = np.zeros(N_LABELS, np.float32); aw = 0.0
            for ti, bs in nbr[qi]:
                if hit[ti]:
                    acc += sim1[ti] * bs; aw += bs
            if aw > 0:
                out[qi] = acc / aw
    return out


def build_kmer(y_src, split):
    s = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        if split == "val":
            pre = "CK003" if k >= 7 else "CK002"
            d = np.load(os.path.join(REL_DIR, f"{pre}_neigh_k{k}.npz"))
            idx = d["idx_val"][:, :5]; sm = d["sim_val"][:, :5].astype(np.float32)
        else:
            f = "TCK002a_neigh_full.npz" if k == 6 else f"CK004_neigh_k{k}_full.npz"
            d = np.load(os.path.join(REL_DIR, f))
            idx = d["idx_test"][:, :5]; sm = d["sim_test"][:, :5].astype(np.float32)
        n = idx.shape[0]
        lab = y_src[idx.ravel()].reshape(n, 5, -1).astype(np.float32)
        ww = np.maximum(sm, 1e-7) ** 2
        s += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return s


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print("  I104: top-20 Homology Aggregation Platform Submission")
    print("=" * 88)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_all = np.arange(len(train_df))
    idx_tr, idx_va = train_test_split(idx_all, test_size=0.2, random_state=SEED)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}
    full_map = {p: i for i, p in enumerate(train_df["protein_id"].values)}
    test_pids = test_df["protein_id"].values

    print("[1/5] 解析 top-100 比对（val→train80 / test→train_full）...")
    idx100_v, bits100_v = i085.parse_topk(
        os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001_top100.tsv"),
        val_pids, train_map)
    idx100_t, bits100_t = i085.parse_topk(
        os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001_top100.tsv"),
        test_pids, full_map)

    print("[2/5] sim 支: top20 聚合 + 2-hop + kmer（val/test 同口径）...")
    sim1_v = i085.agg_topk(idx100_v, bits100_v, y_tr, 20)
    hit_v = idx100_v[:, 0] >= 0
    v2v = parse_2hop(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv"), val_map, val_map)
    sim_align_v = two_hop(sim1_v, hit_v, v2v, n_va)
    sim_v = 0.4 * sim_align_v + 0.6 * build_kmer(y_tr, "val")

    sim1_t = i085.agg_topk(idx100_t, bits100_t, y_full, 20)
    hit_t = idx100_t[:, 0] >= 0
    te_map = {p: i for i, p in enumerate(test_pids)}
    t2t = parse_2hop(os.path.join(CK005A_TMP, "align_test_to_test_e0.001.tsv"), te_map, te_map)
    sim_align_t = two_hop(sim1_t, hit_t, t2t, len(test_pids))
    sim_t = 0.4 * sim_align_t + 0.6 * build_kmer(y_full, "test")
    n_rescued = int(np.sum(np.any(sim_align_t != sim1_t, axis=1)))
    print(f"  test 2-hop 拯救孤儿 {n_rescued} / {int((~hit_t).sum())}")

    print("[3/5] val 阈值管线（top20 口径 full_pipe）...")
    cnn_raw = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn_raw + 0.5 * c005)
    plm_weak = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    deep5_v = 0.65 * deep4 + 0.35 * plm_weak
    f1_val, auc_val, mixed_v, opt_ths, s_v = full_pipe(deep5_v, sim_v, y_tr, y_va, n_va)
    t_tmp = search_scut(s_v, y_va)
    print(f"  val: F1={f1_val:.4f} (I085 top20 口径 0.4482) | t_tmp 均值 {t_tmp.mean():.4f}")

    print("[4/5] test 侧: I097 头均值 + top20 sim 融合...")
    plm_te = mlp_test_mean(device)
    df_e = pd.read_parquet(os.path.join(REL_DIR, "test_embeddings.parquet"))  # 占位校验维度
    deep4_te = (wc["t001a"] * np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_test"]
                + wc["t002"] * np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
                + wc["cnn"] * (0.5 * np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
                               + 0.5 * np.load(os.path.join(REL_DIR, "C005_test_probs.npz"))["test"]))
    deep5_t = 0.65 * deep4_te + 0.35 * plm_te
    s_te = deep5_t + 0.75 * sim_t + 0.50 * col_norm(
        1.0 / (30 + ranks(deep5_t) + 1) + 1.0 / (30 + ranks(sim_t) + 1))
    row_m = np.maximum((s_te >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_full.mean(0) * len(s_te), 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s_te - s_te.max(axis=1, keepdims=True)) / 0.20)
    p_te = sinkhorn(ker, row_m, col_m)
    final = 0.40 * s_te + 0.60 * p_te
    y_pred = (final >= opt_ths[None, :]).astype(np.int8)

    print("[5/5] 合规校验与落盘...")
    sub = pd.concat([test_df[["protein_id"]].reset_index(drop=True),
                     pd.DataFrame(y_pred, columns=label_cols)], axis=1)
    assert sub.shape == (28450, 501), f"尺寸不匹配: {sub.shape}"
    assert set(np.unique(y_pred)).issubset({0, 1}), "非 0/1"
    out = os.path.join(SUB_DIR, "submission_I104_top20sim.csv")
    sub.to_csv(out, index=False)
    mb = os.path.getsize(out) / 1e6
    assert mb <= 50, f"超 50MB: {mb}"
    old_path = os.path.join(SUB_DIR, "submission_I097_fulltrain_80ep.csv")
    diff_info = {}
    if os.path.isfile(old_path):
        old = pd.read_csv(old_path)[label_cols].values.astype(np.int8)
        diff = int((old != y_pred).sum())
        turn_on = int(((old == 0) & (y_pred == 1)).sum())
        diff_info = {"diff_vs_I097": diff, "turn_on": turn_on,
                     "ratio": round(diff / old.size, 6)}
        print(f"  与 I097 提交差异: {diff} 格 ({diff / old.size:.3%}, 开 {turn_on})")
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if os.path.isdir(desktop):
        shutil.copy2(out, os.path.join(desktop, "submission_I104_top20sim.csv"))
        print("  桌面副本已放")

    report = {
        "experiment": "I104",
        "method": "top-20 homology aggregation platform submission (I105-vindicated)",
        "val_top20_f1": round(f1_val, 4),
        "cardinality": round(float(y_pred.sum(1).mean()), 2),
        "total_pos": int(y_pred.sum()),
        "file_mb": round(mb, 2),
        **diff_info,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with Path(os.path.join(REL_DIR, "I104_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 88)
    print(f"  I104 完成! 基数均值 {report['cardinality']} | {out} ({mb:.1f} MB)")
    print("=" * 88)


if __name__ == "__main__":
    main()
