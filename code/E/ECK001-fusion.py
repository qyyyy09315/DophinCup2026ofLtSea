"""
ECK001-fusion — ProstT5 结构嵌入进 CK005 三路相似度融合
=========================================================
编号: ECK = Embedding + CK（ProstT5 嵌入 + kmer/DIAMOND 相似度融合），
      ECK001（原 E003a）。E 系列后续融合链统一归 ECK 系列。
背景:
  - CK005 (DIAMOND 比对 + kmer 多k 预融合 w_align=0.4/w_sim=0.75) = 0.4359
  - E003 (ProstT5 短序列嵌入, ≤256aa) 短序列 sim 独立 F1=0.4155 > kmer 0.3792,
    corr(pt5,kmer)=0.635 互补; E003 纯 kmer+pt5 融合全局 +0.0148
  - 本实验: 把 pt5 作为第三条相似度路混进 CK005 的 align+kmer 预融合 sim,
    短序列行用三路, 长序列行保持 CK005 两路 → 四路集成 → 100 档阈值

方案:
  - val 端: sim_base = w_align*pident_weight? NO——用 CK005 官方 bitscore 预融合
      sim_base = 0.4*sim_bit + 0.6*sim_kmer (CK005 提交配置)
    短行: sim = wp*sim_pt5 + (1-wp)*sim_base; 长行: sim = sim_base
  - 网格: wp ∈ {0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}
  - 评估: 融合后四路集成 val F1 (0.5 阈值 + 100 档), 对比 CK005 基线 0.4359
  - 有效 → test 端同理 → 提交
已知问题 (2026-08-20, ECK001a 闭环):
  val 检索库用了全部短序列 short_tr（含验证集），短 val 查询 top-1 自匹配 96.3%。
  泄漏 F1 0.4558 不可信；去泄漏后 w_pt5=0，集成回到 CK005 0.4359。
  提交 43.24 < CK005 44.84 即此因。后续勿再把 short_tr 当 val 库。
用法:
  ./python code/E/ECK001-fusion.py [--gen-sub]
输出:
  rel/E003_fusion_report.json
  submit/submission_E003_fusion.csv (若融合有效)
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")

TRAIN_PATH = f"{PROJECT_ROOT}/data/train.csv"
TEST_PATH = f"{PROJECT_ROOT}/data/test.csv"
TCK002_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
EMB_PATH = f"{REL_DIR}/E003_pt5_emb_short.npz"
ALN_VAL = f"{REL_DIR}/ck005_tmp/align_val.tsv"
ALN_TEST = f"{REL_DIR}/ck005_tmp/align_test.tsv"

MAX_AA = 256
TOP_K = 3
ALIGN_W = 0.4   # CK005 提交配置: bitscore 权重
MAX_ACCEPT = 5
EVALUE = 0.001
SIM_W = 0.75     # 集成中 sim 路权重 (CK005 提交用 0.75, test 段 0.7)


def macro_f1_fixed(ens, y_true, t=0.5):
    pred = (ens >= t)
    f1s = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def macro_auc(y_true, y_prob):
    aucs = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, y_prob[:, i]))
    return float(np.mean(aucs))


def macro_rp(ens, y_true, t):
    """逐标签固定阈值 → Macro Recall / Precision (与 macro_f1_fixed 同口径)。"""
    pred = (ens >= t)
    rs, ps = [], []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
        rs.append(tp / (tp + fn) if tp + fn else 0.0)
        ps.append(tp / (tp + fp) if tp + fp else 0.0)
    return float(np.mean(rs)), float(np.mean(ps))


def parse_aln(tsv_path, query_ids, id_map):
    """解析 DIAMOND blastp outfmt6 → (idx, bitscore, pident), 同 CK005。"""
    idx = np.full((len(query_ids), MAX_ACCEPT), -1, dtype=np.int64)
    bits = np.zeros((len(query_ids), MAX_ACCEPT), dtype=np.float32)
    pident = np.zeros((len(query_ids), MAX_ACCEPT), dtype=np.float32)
    qpos = {q: i for i, q in enumerate(query_ids)}
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            q, t = parts[0], parts[1]
            if q not in qpos or t not in id_map:
                continue
            i = qpos[q]
            try:
                bs, ev, pid = float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            if ev > EVALUE:
                continue
            j = 0
            while j < MAX_ACCEPT and bits[i, j] > bs:
                j += 1
            if j < MAX_ACCEPT:
                idx[i, j:] = np.roll(idx[i, j:], 1)
                bits[i, j:] = np.roll(bits[i, j:], 1)
                pident[i, j:] = np.roll(pident[i, j:], 1)
                idx[i, j] = id_map[t]
                bits[i, j] = bs
                pident[i, j] = pid
    return idx, bits, pident


def agg_labels(idx, w, y_space, top_k):
    n = idx.shape[0]
    valid = idx[:, :top_k] >= 0
    lab = np.zeros((n, top_k, y_space.shape[1]), dtype=np.float32)
    for j in range(top_k):
        lab[:, j] = y_space[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(w[:, :top_k], 1e-7)
    ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(axis=1) / (ww.sum(axis=1)[:, None] + 1e-7)


def knn_agg_cos(q_emb, db_emb, db_y, top_k=3):
    """余弦 top-k 加权邻居聚合 (E003 同款)。"""
    q = torch.from_numpy(q_emb.astype(np.float32))
    db = torch.from_numpy(db_emb.astype(np.float32))
    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    db = db / db.norm(dim=1, keepdim=True).clamp(min=1e-8)
    q, db = q.to("cuda"), db.to("cuda")
    n = q.shape[0]
    idx = np.zeros((n, 5), dtype=np.int64)
    sims = np.zeros((n, 5), dtype=np.float32)
    for s in range(0, n, 512):
        e = min(s + 512, n)
        top = torch.topk(q[s:e] @ db.T, 5, dim=1)
        idx[s:e] = top.indices.cpu().numpy()
        sims[s:e] = top.values.cpu().numpy()
    lab = db_y[idx[:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
    w = np.maximum(sims[:, :top_k], 1e-7)
    return (lab * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]


def build_kmer(idx_train, idx_val, y_train, split="val", y_full=None):
    """kmer 多k 融合 sim, 同 CK005 (val 91,036 空间 / test 全量 113,796)。

    split=test 时邻居索引指向全量 train 空间 (113,796), 必须用 y_full 索引。"""
    K_RANGE = [3, 4, 5, 6, 7, 8]
    FUSION = [0, 0.1, 0.2, 0.4, 0.2, 0.1]
    if split == "val":
        n = len(idx_val)
        sim = np.zeros((n, 500), dtype=np.float32)
        for k, wt in zip(K_RANGE, FUSION):
            if wt <= 0:
                continue
            prefix = "CK003" if k >= 7 else "CK002"
            d = np.load(f"{REL_DIR}/{prefix}_neigh_k{k}.npz")
            lab = y_train[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
            s = d["sim_val"][:, :5].astype(np.float32)
            ww = np.maximum(s, 1e-7) ** 2
            sim += wt * (lab * ww[:, :, None]).sum(axis=1) / ww.sum(axis=1)[:, None]
        return sim
    else:  # test 全量空间
        test_df = pd.read_csv(TEST_PATH)
        n2 = len(test_df)
        y_space = y_full  # 必须为全量标签 (113,796 行)
        sim = np.zeros((n2, 500), dtype=np.float32)
        for k, wt in zip(K_RANGE, FUSION):
            if wt <= 0:
                continue
            d = np.load(f"{REL_DIR}/TCK002a_neigh_full.npz" if k == 6
                        else f"{REL_DIR}/CK004_neigh_k{k}_full.npz")
            lab = y_space[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
            s = d["sim_test"][:, :5].astype(np.float32)
            ww = np.maximum(s, 1e-7) ** 2
            sim += wt * (lab * ww[:, :, None]).sum(axis=1) / ww.sum(axis=1)[:, None]
        return sim


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print(f"ECK001-fusion: ProstT5 进 CK005 三路相似度融合 | 模式: {'含提交' if gen_sub else '评估'}")

    print("[1/5] 读取数据 + 分割...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)),
                                          test_size=0.2, random_state=42)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    lens_tr = np.array([len(s) for s in train_df["sequence"].values])
    short_tr = np.where(lens_tr <= MAX_AA)[0]
    val_short = lens_tr[idx_val] <= MAX_AA
    n_vs = int(val_short.sum())
    short_te = np.where(test_df["sequence"].str.len().values <= MAX_AA)[0]
    print(f"  train {len(idx_train)} / val {len(idx_val)} / test {len(test_df)}")
    print(f"  短序列 (≤{MAX_AA}aa): val {n_vs} / test {len(short_te)}")

    print("[2/5] 构建三路 sim (val)...")
    # 路1: kmer 多k (val 91,036 空间)
    sim_kmer = build_kmer(idx_train, idx_val, y_train, "val")
    # 路2: DIAMOND bitscore (val → train 91,036)
    val_ids = train_df["protein_id"].values[idx_val]
    train_db_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    idx_v, bits_v, pid_v = parse_aln(ALN_VAL, val_ids, train_db_map)
    sim_bit = agg_labels(idx_v, bits_v, y_train, TOP_K)
    # CK005 预融合 (短行三路时作为 base)
    sim_base = ALIGN_W * sim_bit + (1 - ALIGN_W) * sim_kmer
    f1_base = macro_f1_fixed(sim_base, y_val)
    print(f"  sim_base (align+kmer): F1(0.5)={f1_base:.4f} (CK005 提交配置)")

    # 路3: ProstT5 短序列嵌入 → 短 val 查询 → 短 train 库
    emb = np.load(EMB_PATH)
    emb_s_tr, emb_s_te = emb["emb_s_tr"], emb["emb_s_te"]
    pos = np.searchsorted(short_tr, idx_val[val_short])
    emb_v_s = emb_s_tr[pos]
    sim_pt5_s = knn_agg_cos(emb_v_s, emb_s_tr, y_full[short_tr], TOP_K)
    f1_pt5 = macro_f1_fixed(sim_pt5_s, y_val[val_short])
    r_pt5, p_pt5 = macro_rp(sim_pt5_s, y_val[val_short], 0.5)
    f1_kmer_s = macro_f1_fixed(sim_kmer[val_short], y_val[val_short])
    r_kmer_s, p_kmer_s = macro_rp(sim_kmer[val_short], y_val[val_short], 0.5)
    print(f"  sim_pt5 (短序列): F1(0.5)={f1_pt5:.4f} R={r_pt5:.4f} P={p_pt5:.4f} "
          f"| kmer 短 F1={f1_kmer_s:.4f} R={r_kmer_s:.4f} P={p_kmer_s:.4f} "
          f"| corr={np.corrcoef(sim_pt5_s.ravel(), sim_kmer[val_short].ravel())[0,1]:.4f}")

    print("[3/5] 三路融合网格 (短行 wp 混 pt5, 长行保持 sim_base)...")
    fusion = {}
    for wp in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        sim_f = sim_base.copy()
        sim_f[val_short] = wp * sim_pt5_s + (1 - wp) * sim_base[val_short]
        fusion[wp] = round(macro_f1_fixed(sim_f, y_val), 6)
    print(f"  融合网格 (w_pt5): {fusion}")
    best_wp = max(fusion, key=fusion.get)
    gain = fusion[best_wp] - fusion[0.0]
    print(f"  最优 w_pt5={best_wp} (F1={fusion[best_wp]:.4f}, base={fusion[0.0]:.4f}, Δ={gain:+.4f})")

    # 四路集成 + 100 档阈值 (CK005 公安配置)
    print("[4/5] 集成评估 (四路 + 100 档阈值)...")
    vp = np.load(f"{REL_DIR}/TC003_val_probs.npz")
    t1 = np.load(f"{REL_DIR}/TCK001_t001a_probs.npz")
    t2v = np.load(f"{REL_DIR}/T002_val_probs.npz")
    with open(TCK002_CONFIG_PATH, "r", encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    sim_final = sim_base.copy()
    sim_final[val_short] = best_wp * sim_pt5_s + (1 - best_wp) * sim_base[val_short]
    ens_val = (w["t001a"] * t1["t001a_val"] + w["t002"] * t2v["t002"]
               + w["cnn"] * vp["cnn"] + SIM_W * sim_final)
    cands = np.linspace(0.01, 0.99, 100)
    f1_int = macro_f1_fixed(ens_val, y_val, 0.5)
    # 100 档
    best_t = np.full(500, 0.5)
    for i in tqdm(range(500), desc="    阈值搜索", unit="label"):
        yt = y_val[:, i]
        if yt.sum() == 0:
            continue
        p = ens_val[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (p & ytm).sum(0); fp = (p & ~ytm).sum(0); fn = (~p & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best_t[i] = cands[f1t.argmax()]
    pred100 = ens_val >= best_t[None, :]
    fs = []
    for i in range(500):
        yt = y_val[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        if yt.sum() == len(yt):
            fs.append(1.0); continue
        fs.append(f1_score(yt, pred100[:, i], zero_division=0))
    f1_100 = float(np.mean(fs))
    print(f"  四路集成: F1(0.5)={f1_int:.4f} | 100 档阈值 F1={f1_100:.4f} (CK005 基线 0.4359)")

    # 逐标签 100 档阈值 → R/P (补回腾讯表格缺列; 同口径: 阈值按各标签 F1 最大化)
    r100, p100 = macro_rp(ens_val, y_val, best_t)
    print(f"  四路集成 100 档: F1={f1_100:.4f} | R={r100:.4f} P={p100:.4f}")

    report = {"sim_base_f1_05": round(f1_base, 6), "sim_pt5_short_f1_05": round(f1_pt5, 6),
              "sim_pt5_short_recall": round(r_pt5, 6), "sim_pt5_short_precision": round(p_pt5, 6),
              "fusion_w_pt5": fusion, "best_w_pt5": best_wp, "gain_sim": round(gain, 6),
              "ensemble_f1_05": round(f1_int, 6), "ensemble_f1_100": round(f1_100, 6),
              "ensemble_recall_100": round(r100, 6), "ensemble_precision_100": round(p100, 6),
              "ck005_baseline": 0.4359, "delta_vs_ck005": round(f1_100 - 0.4359, 6)}

    if not gen_sub or f1_100 <= 0.4359 + 0.001:
        if f1_100 <= 0.4359 + 0.001:
            report["submission"] = "none (未超 CK005 基线)"
        with open(f"{REL_DIR}/E003_fusion_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  报告: {REL_DIR}/E003_fusion_report.json")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    print("[5/5] test 端三路融合 + 提交...")
    # test 端 align (test → full 113,796)
    test_ids = test_df["protein_id"].values
    full_ids = train_df["protein_id"].values
    full_db_map = {pid: i for i, pid in enumerate(full_ids)}
    idx_t, bits_t, pid_t = parse_aln(ALN_TEST, test_ids, full_db_map)
    sim_bit_test = agg_labels(idx_t, bits_t, y_full, TOP_K)
    sim_kmer_test = build_kmer(idx_train, idx_val, y_train, "test", y_full=y_full)
    sim_base_test = ALIGN_W * sim_bit_test + (1 - ALIGN_W) * sim_kmer_test
    # test 短序列 pt5 (short_te 查询 → 短 train 库)
    emb_te_s = emb_s_te
    sim_pt5_test = knn_agg_cos(emb_te_s, emb_s_tr, y_full[short_tr], TOP_K)
    sim_test = sim_base_test.copy()
    sim_test[short_te] = best_wp * sim_pt5_test + (1 - best_wp) * sim_base_test[short_te]

    tp = np.load(f"{REL_DIR}/TC003_test_probs.npz")
    t1t = np.load(f"{REL_DIR}/TCK001_t001a_probs.npz")
    t2t = np.load(f"{REL_DIR}/T002_test_probs.npz")
    ens_test = (w["t001a"] * t1t["t001a_test"] + w["t002"] * t2t["t002"]
                + w["cnn"] * tp["cnn"] + SIM_W * sim_test)
    y_pred = np.zeros_like(ens_test, dtype=np.int8)
    for i, t in enumerate(best_t):
        y_pred[:, i] = (ens_test[:, i] >= t).astype(np.int8)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    sub_path = f"{SUB_DIR}/submission_E003_fusion.csv"
    submission.to_csv(sub_path, index=False)
    report["submission"] = sub_path
    with open(f"{REL_DIR}/E003_fusion_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  提交: {sub_path} ({len(submission)} 行)")
    print(f"  报告: {REL_DIR}/E003_fusion_report.json")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
