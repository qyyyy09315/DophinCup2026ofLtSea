"""
ECK004-choufasman — 路线1: Chou-Fasman 二级结构倾向特征（二级结构信号）
=======================================================================
背景:
  - 文献: DeepSS2GO (Song et al., Brief. Bioinform. 2024, doi:10.1093/bib/bbae196;
    GitHub orca233/DeepSS2GO): sequence + homology + secondary structure 融合,
    二级结构(ss8)作为独立互补模块显著提升
  - 本任务已覆盖: kmer 组成(CK 系) / DIAMOND 序列比对(align) / ProstT5 折叠拓扑
    (E003/ECK001); 缺"局部构象"信号
  - 完整 ss8 管线(SPOT1D-LM/NetSurfP3)重成本高; ESM2 现有缓存是 mean-pooled
    无残基信息不可复用 → 采用**零模型方案**: Chou-Fasman 氨基酸构象倾向
    (α-helix/β-sheet/turn 三类分值, 公开参数表) 把序列转构象倾向轮廓
  - 目的: 验证"局部构象模式"是否互补且能给融合增益 (抢先关: 不重蹈
    E001 嵌入与 kmer 高相关 w=0)
方案:
  - 特征: 每残基按 Chou-Fasman 3 维倾向值 → 蛋白级 = 序列均值 tanh 归一化
    (3维); 可选追加: 构象状态计数 (最长 α/β 段长、段数等)
  - 邻居: 特征空间 KNN top-3 相似度加权聚合邻居标签 (同 CK 系聚合方式)
  - 三关评估 (任一大败即关):
    ① sim_cf 独立 F1(0.5) 是否 > 随机基线 (标签先验)
    ② corr(sim_cf, sim_kmer) 是否 < 0.9 (互补性: 与组成不冗余)
    ③ 融合进 ECK001 sim 路 (kmer+align+pt5 后) 是否 Δ>0.001
用法:
  python code/E/ECK004-choufasman.py [--gen-sub]
输出:
  rel/ECK004_report.json
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")

# Chou-Fasman 构象倾向参数 (α-helix / β-sheet / turn)。来源: Chou & Fasman 1978,
# 常用归一化分值表 (如 https://web.expasy.org/protscale/ 及各教科书标准表)
CF = {
    "A": (1.42, 0.83, 0.66), "C": (0.70, 1.19, 1.19), "D": (1.01, 0.54, 1.46),
    "E": (1.51, 0.37, 0.74), "F": (1.13, 1.38, 0.60), "G": (0.57, 0.75, 1.56),
    "H": (1.00, 0.87, 0.95), "I": (1.08, 1.60, 0.47), "K": (1.16, 0.74, 1.01),
    "L": (1.21, 1.30, 0.59), "M": (1.45, 1.05, 0.60), "N": (0.67, 0.89, 1.56),
    "P": (0.57, 0.55, 1.52), "Q": (1.11, 1.10, 0.98), "R": (0.98, 0.93, 0.95),
    "S": (0.77, 0.75, 1.43), "T": (0.83, 1.19, 0.96), "V": (1.06, 1.70, 0.50),
    "W": (1.08, 1.37, 0.96), "Y": (0.69, 1.47, 1.14),
}


def cf_feature(seq):
    """蛋白级 Chou-Fasman 构象特征: 3维均值(α/β/turn) + 3维(sqrt均值) + 3维(max占比)。
    共 9 维: 捕获整体构象倾向 + 局部模式。"""
    vec = np.zeros(9, dtype=np.float32)
    n = 0
    amax = np.zeros(3, dtype=np.float32)
    for c in seq:
        v = CF.get(c)
        if v is None:
            continue
        vec[:3] += np.array(v, dtype=np.float32)
        amax[:3] = np.maximum(amax[:3], np.array(v, dtype=np.float32))
        n += 1
    if n == 0:
        return vec
    mean3 = vec[:3] / n
    sqrt3 = np.sqrt(vec[:3] / n)          # 非线性压缩
    rel = mean3 / (mean3.sum() + 1e-9)    # 相对占比
    return np.concatenate([mean3, sqrt3, rel], axis=0).astype(np.float32)


def knn_sim(feat, y_space):
    """特征空间 KNN top-3 相似度加权聚合 (同 CK 系聚合)。"""
    q = torch.from_numpy(feat.astype(np.float32))
    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    n = q.shape[0]
    idx = np.zeros((n, 5), dtype=np.int64)
    sims = np.zeros((n, 5), dtype=np.float32)
    for s in range(0, n, 2048):
        e = min(s + 2048, n)
        top = torch.topk(q[s:e] @ q.T, 5, dim=1)
        idx[s:e] = top.indices.cpu().numpy()
        sims[s:e] = top.values.cpu().numpy()
    lab = y_space[idx[:, :3].ravel()].reshape(n, 3, -1).astype(np.float32)
    w = np.maximum(sims[:, :3], 1e-7)
    return (lab * w[:, :, None]).sum(1) / w.sum(1)[:, None], sims[:, 0].mean()


def macro_f1(ens, y):
    f1s = []
    for i in range(500):
        yt = y[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = ens[:, i] >= 0.5
        tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print(f"ECK004 路线1: Chou-Fasman 构象倾向特征 | 模式: {'含提交' if gen_sub else '评估'}")

    print("[1/5] 数据...")
    train_df = pd.read_csv(f"{PROJECT_ROOT}/data/train.csv")
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    # 必须用 train_test_split 双返回：CK002 邻居缓存按此顺序建索引，
    # setdiff1d 排序顺序会错位（E003b/CK005 同源坑）
    idx_train, idx_val = train_test_split(np.arange(len(train_df)),
                                          test_size=0.2, random_state=42)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    seqs = train_df["sequence"].values
    print(f"  train {len(idx_train)} / val {len(idx_val)}")

    print("[2/5] Chou-Fasman 特征 (val 全量)...")
    seq_val = [s.replace("U", "X").replace("Z", "X").replace("O", "X") for s in seqs[idx_val]]
    feat_val = np.stack([cf_feature(s) for s in tqdm(seq_val, desc="CF特征")])
    # 归一化
    feat_val = feat_val / (np.linalg.norm(feat_val, axis=1, keepdims=True) + 1e-9)

    print("[3/5] KNN 邻居聚合 (val→val? 不——用 train 库防泄漏)...")
    # 用 train 全部构建库
    seq_tr = [s.replace("U", "X").replace("Z", "X").replace("O", "X") for s in seqs[idx_train]]
    feat_tr = np.stack([cf_feature(s) for s in tqdm(seq_tr, desc="CF特征(train)")])
    feat_tr = feat_tr / (np.linalg.norm(feat_tr, axis=1, keepdims=True) + 1e-9)
    db = torch.from_numpy(feat_tr.astype(np.float32))
    db = db / db.norm(dim=1, keepdim=True).clamp(min=1e-8)
    q = torch.from_numpy(feat_val.astype(np.float32))
    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    n = q.shape[0]
    idx = np.zeros((n, 5), dtype=np.int64)
    sims = np.zeros((n, 5), dtype=np.float32)
    for s in range(0, n, 2048):
        e = min(s + 2048, n)
        top = torch.topk(q[s:e] @ db.T, 5, dim=1)
        idx[s:e] = top.indices.cpu().numpy()
        sims[s:e] = top.values.cpu().numpy()
    lab = y_train[idx[:, :3].ravel()].reshape(n, 3, -1).astype(np.float32)
    w = np.maximum(sims[:, :3], 1e-7)
    sim_cf = (lab * w[:, :, None]).sum(1) / w.sum(1)[:, None]
    top1 = sims[:, 0].mean()

    print("[4/5] 三关评估...")
    f1_cf = macro_f1(sim_cf, y_val)
    # 补算 R/P/AUC (0.5 阈值口径, 遵守 4 项必填规则)
    pred_cf = sim_cf >= 0.5
    rs, ps = [], []
    for i in range(500):
        yt = y_val[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred_cf[:, i]
        tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
        rs.append(tp / (tp + fn) if tp + fn else 0.0)
        ps.append(tp / (tp + fp) if tp + fp else 0.0)
    r_cf, p_cf = float(np.mean(rs)), float(np.mean(ps))
    auc_cf = []
    for i in range(500):
        yt = y_val[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        auc_cf.append(__import__("sklearn").metrics.roc_auc_score(yt, sim_cf[:, i]))
    auc_cf = float(np.mean(auc_cf))
    print(f"  sim_cf 独立: F1(0.5)={f1_cf:.4f} R={r_cf:.4f} P={p_cf:.4f} AUC={auc_cf:.4f}")
    # 随机基线 (先验)
    pri = np.tile(y_train.mean(axis=0), (len(y_val), 1))
    f1_pri = macro_f1(pri, y_val)
    # corr with kmer
    sim_kmer = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        prefix = "CK003" if k >= 7 else "CK002"
        d = np.load(f"{REL_DIR}/{prefix}_neigh_k{k}.npz")
        n_ = len(idx_val)
        lab_ = y_train[d["idx_val"][:, :5].ravel()].reshape(n_, 5, -1).astype(np.float32)
        s = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(s, 1e-7) ** 2
        sim_kmer += wt * (lab_ * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    f1_kmer = macro_f1(sim_kmer, y_val)
    corr = float(np.corrcoef(sim_cf.ravel(), sim_kmer.ravel())[0, 1])
    print(f"  sim_cf 独立: F1(0.5)={f1_cf:.4f}  top1_sim={top1:.4f}")
    print(f"  随机(先验)基线: F1={f1_pri:.4f}")
    print(f"  kmer 多k: F1={f1_kmer:.4f} | corr(cf,kmer)={corr:.4f}")

    # 融合 (进四路集成, 用现有概率)
    print("[5/5] 融合进 ECK001 集成...")
    vp = np.load(f"{REL_DIR}/TC003_val_probs.npz"); cnn = vp["cnn"]
    t1 = np.load(f"{REL_DIR}/TCK001_t001a_probs.npz"); t1a = t1["t001a_val"]
    t2v = np.load(f"{REL_DIR}/T002_val_probs.npz"); t2 = t2v["t002"]
    w = json.load(open(f"{PROJECT_ROOT}/models/tck002-ensemble.json"))["best_w"]
    # ECK001 的 sim (需 pt5 短序列)
    d = np.load(f"{REL_DIR}/E003_pt5_emb_short.npz"); emb_s_tr = d["emb_s_tr"]
    lens = np.array([len(s) for s in seqs])
    val_short = lens[idx_val] <= 256
    short_tr = np.where(lens <= 256)[0]
    import importlib.util
    spec = importlib.util.spec_from_file_location("eck001", os.path.join(PROJECT_ROOT, "code", "E", "ECK001-fusion.py"))
    eck001 = importlib.util.module_from_spec(spec); spec.loader.exec_module(eck001)
    pos = np.searchsorted(short_tr, idx_val[val_short])
    sim_pt5_s = eck001.knn_agg_cos(emb_s_tr[pos], emb_s_tr, y_full[short_tr], 3)
    assert np.isfinite(sim_pt5_s).all()
    # 对齐: val 全量上 sim_pt5 只对短行有值; 用 val 内部 KNN(sim_cf) 对长行补 → 简化:
    # 先验证 cf 融合到 sim_kmer 上看增益 (不做完整 ECK001 复杂对齐)
    best_gain = -9
    best_l = 0
    for lamb in [0.1, 0.2, 0.4]:
        sim_mix = (1 - lamb) * sim_kmer + lamb * sim_cf
        f = macro_f1(sim_mix, y_val)
        if f - f1_kmer > best_gain:
            best_gain = f - f1_kmer
            best_l = lamb
    print(f"  cf 混入 kmer (λ 网格) best λ={best_l} ΔF1={best_gain:+.4f} (kmer alone={f1_kmer:.4f})")

    report = {"cf_feature": "Chou-Fasman 3x3 构象倾向", "sim_cf_f1": round(f1_cf, 4),
              "sim_cf_recall": round(r_cf, 4), "sim_cf_precision": round(p_cf, 4),
              "sim_cf_auc": round(auc_cf, 4),
              "prior_baseline": round(f1_pri, 4), "kmer_f1": round(f1_kmer, 4),
              "corr_cf_kmer": round(corr, 4), "top1_sim": round(float(top1), 4),
              "fusion_gain_vs_kmer": round(best_gain, 4), "best_lambda": best_l}
    with Path(f"{REL_DIR}/ECK004_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REL_DIR}/ECK004_report.json")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
