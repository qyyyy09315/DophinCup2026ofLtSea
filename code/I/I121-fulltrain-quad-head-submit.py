"""
I121 — 100% 全量跨骨干几何解耦四元头 (Quad-Head) 与终极 SOTA 提交出片 (P004)
=============================================================================
背景与依据 (P004):
  - 在 P003 (I120) 中，四元视角解耦头 (Quad-Head):
      0.40 Smooth + 0.20 Converged + 0.20 Centered_1664 + 0.20 Centered_3840
    将独立 microAUC 推进至 0.8908 历史新高，F1 推进至 0.3357。
  - 在 SOTA 全流程管线 (wRRF + SimMix + 传输锐化) 的 R=20 重复半折配对检验中，
    Quad-Head 产生 ΔOOF = +0.00081 ± 0.00017 (CI95 下界 +0.00074, P=1.00，20 折全胜)！
  - 本脚本在 100% 全量数据 (113,796 条) 上精炼训练解耦头组件，推断测试集 (28,450 条)，
    与 100% 全量 Deep4 (I103) + 全库 SimMix (I113) 汇聚，
    执行非对称加权 RRF (0.6 : 1.4) 与锐化最优传输 (tau=0.18, alpha=0.70)，
    出片 submit/submission_I121_quad_head.csv，成为全项目当前最强终极提交候选。
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
SEED = 42
W_PLM = 0.35
SEEDS_HEAD = [11, 22, 33]
EPOCHS_HEAD = 35

# 超参数锁 (I116 / I117 / I120)
ALIGN_W = 0.50
SIM_W = 0.80
SINK_TAU = 0.18
SINK_ALPHA = 0.70
WD_RRF = 0.6
WS_RRF = 1.4

# 动态加载工具模块
i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
i085 = SourceFileLoader("i085", os.path.join(PROJECT_ROOT, "code", "I", "I085-top100-gated-agg.py")).load_module()
i104 = SourceFileLoader("i104", os.path.join(PROJECT_ROOT, "code", "I", "I104-top20sim-submit.py")).load_module()
i113 = SourceFileLoader("i113", os.path.join(PROJECT_ROOT, "code", "I", "I113-simmix-submit.py")).load_module()

MLP = i064.MLP
macro_f1 = i064.macro_f1
search_scut = i064.search_scut
ranks = i064.ranks
col_norm = i064.col_norm
sinkhorn = i064.sinkhorn


class ASL(nn.Module):
    def forward(self, lg, tg):
        p = torch.sigmoid(lg.float()).clamp(1e-6, 1 - 1e-6)
        pos = -tg * torch.log(p)
        pm = torch.clamp(p - 0.05, min=0.0)
        neg = -(1 - tg) * (pm ** 4.0) * torch.log(1 - pm)
        return (pos + neg).mean()


def train_head_seed(seed, Xtr, ytr, Xte, epochs=EPOCHS_HEAD, batch_size=512, tag=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    tl = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))),
        batch_size=batch_size, shuffle=True, pin_memory=True
    )
    m = MLP(Xtr.shape[1]).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = ASL()
    sc = torch.amp.GradScaler("cuda")

    for ep in tqdm(range(epochs), desc=f"  {tag} s{seed}", ncols=80):
        m.train()
        for bx, by in tl:
            bx, by = bx.cuda(non_blocking=True), by.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = crit(m(bx), by)
            sc.scale(loss).backward()
            sc.step(opt)
            sc.update()
        sch.step()

    m.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(Xte), 512):
            bx = torch.from_numpy(Xte[s:s + 512]).cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                pr = torch.sigmoid(m(bx).float()).cpu().numpy()
            preds.append(pr)
    return np.vstack(preds)


def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 96)
    print("  I121: 100% Full-Data Cross-Backbone Quad-Head SOTA Submission (P004)")
    print("=" * 96)

    # 1. 加载真值与数据划分
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)

    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=SEED)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}
    full_map = {p: i for i, p in enumerate(train_df["protein_id"].values)}
    test_pids = test_df["protein_id"].values

    with Path(TCK002_CONFIG).open("r", encoding="utf-8") as f:
        wc = json.load(f)["best_w"]

    # 2. 验证集标定 (Quad-Head: 0.40 smooth + 0.20 conv + 0.20 cen1664 + 0.20 cen3840)
    print("\n[1/5] 执行验证集标定 (Quad-Head + wRRF + 传输锐化)...")
    cnn_raw = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1_v = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2_v = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    deep4_v = wc["t001a"] * t1_v + wc["t002"] * t2_v + wc["cnn"] * (0.5 * cnn_raw + 0.5 * c005_v)

    p_smooth_v = np.load(os.path.join(REL_DIR, "multi_plm_mlp_val_probs.npz"))["val"]
    p_conv_v = np.load(os.path.join(REL_DIR, "i106_conv_multiseed_val.npz"))["val"]
    p_cen1664_v = np.load(os.path.join(REL_DIR, "i118_centered_multiseed_val.npz"))["val"]
    p_cen3840_v = np.load(os.path.join(REL_DIR, "i119_3840d_centered_multiseed_val.npz"))["val"]

    plm_quad_v = (0.40 * p_smooth_v + 0.20 * p_conv_v +
                  0.20 * p_cen1664_v + 0.20 * p_cen3840_v)
    deep5_v = (1.0 - W_PLM) * deep4_v + W_PLM * plm_quad_v

    idx100_v, bits100_v = i085.parse_topk(
        os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001_top100.tsv"), val_pids, train_map)
    v2v = i104.parse_2hop(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv"), val_map, val_map)
    sim_align_v = i113.sim_mix_align(idx100_v, bits100_v, y_tr, v2v, n_va)
    kmer_v = i104.build_kmer(y_tr, "val")
    sim_v = ALIGN_W * sim_align_v + (1.0 - ALIGN_W) * kmer_v

    r_d_v = ranks(deep5_v)
    r_s_v = ranks(sim_v)
    rrf_v = col_norm(WD_RRF / (30 + r_d_v + 1) + WS_RRF / (30 + r_s_v + 1))
    s_v = deep5_v + SIM_W * sim_v + 0.50 * rrf_v

    t_tmp = search_scut(s_v, y_va)
    row_m_v = np.maximum((s_v >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m_v = np.maximum(y_tr.mean(0) * n_va, 1e-6)
    col_m_v *= row_m_v.sum() / col_m_v.sum()
    ker_v = np.exp((s_v - s_v.max(axis=1, keepdims=True)) / SINK_TAU)
    p_v = sinkhorn(ker_v, row_m_v, col_m_v)
    final_v = (1.0 - SINK_ALPHA) * s_v + SINK_ALPHA * p_v
    opt_ths = search_scut(final_v, y_va)
    f1_val = macro_f1(final_v, y_va, opt_ths)
    print(f"  [验证集标定完成] 标定 val F1 = {f1_val:.5f} | t_tmp 均值 {t_tmp.mean():.4f}")

    # 3. 准备 100% 全量特征并训练解耦分类头
    print("\n[2/5] 准备 100% 全量特征矩阵并精炼训练解耦分类头 (3 种子, 35ep)...")
    # 1664D
    df_e_tr = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    df_p_tr = pd.read_parquet(os.path.join(REL_DIR, "train_embeddings.parquet"))
    df_e_te = pd.read_parquet(os.path.join(REL_DIR, "test_esm2_embeddings.parquet"))
    df_p_te = pd.read_parquet(os.path.join(REL_DIR, "test_embeddings.parquet"))
    ec = [c for c in df_e_tr.columns if c.startswith("emb_")]
    pc = [c for c in df_p_tr.columns if c.startswith("emb_")]
    X_tr_1664 = np.concatenate([df_e_tr[ec].values, df_p_tr[pc].values], axis=1).astype(np.float32)
    X_te_1664 = np.concatenate([df_e_te[ec].values, df_p_te[pc].values], axis=1).astype(np.float32)

    mu_1664 = X_tr_1664.mean(axis=0, keepdims=True)
    X_tr_cen1664 = X_tr_1664 - mu_1664
    X_te_cen1664 = X_te_1664 - mu_1664

    # 3840D
    emb_650 = np.load(os.path.join(REL_DIR, "esm2_650m_embeddings.npz"))
    X_tr_3840 = emb_650["train_last3"]
    X_te_3840 = emb_650["test_last3"]
    mu_3840 = X_tr_3840.mean(axis=0, keepdims=True)
    X_tr_cen3840 = X_tr_3840 - mu_3840
    X_te_cen3840 = X_te_3840 - mu_3840

    # 训练 1664D Centered Head (3 种子)
    print("  训练 1664D 均值解耦头 (3 种子)...")
    preds_cen1664 = []
    for s in SEEDS_HEAD:
        pr = train_head_seed(s, X_tr_cen1664, y_full, X_te_cen1664, tag="Cen1664")
        preds_cen1664.append(pr)
    pr_cen1664_te = np.mean(preds_cen1664, axis=0)

    # 训练 3840D Centered Head (3 种子)
    print("  训练 3840D ESM-2 650M 末三层解耦头 (3 种子)...")
    preds_cen3840 = []
    for s in SEEDS_HEAD:
        pr = train_head_seed(s, X_tr_cen3840, y_full, X_te_cen3840, tag="Cen3840")
        preds_cen3840.append(pr)
    pr_cen3840_te = np.mean(preds_cen3840, axis=0)

    # 4. 加载现有全量组件构建 Quad-Head 测试集预测
    print("\n[3/5] 整合四元解耦头测试集预测...")
    plm_te_smooth = np.load(os.path.join(REL_DIR, "plm1664_fulltrain_test.npz"))["test"]

    # 加载 78ep 收敛头并推理测试集
    conv_preds = []
    for s in [11, 22, 33]:
        m_conv = MLP(1664).cuda()
        sd = torch.load(os.path.join(MODEL_DIR, f"i097_fulltrain_s{s}.pt"), map_location="cuda", weights_only=True)
        m_conv.load_state_dict(sd)
        m_conv.eval()
        pr_s = []
        with torch.no_grad():
            for st in range(0, len(X_te_1664), 512):
                bx = torch.from_numpy(X_te_1664[st:st + 512]).cuda()
                with torch.amp.autocast("cuda"):
                    pr = torch.sigmoid(m_conv(bx).float()).cpu().numpy()
                pr_s.append(pr)
        conv_preds.append(np.vstack(pr_s))
    plm_te_conv = np.mean(conv_preds, axis=0)

    # 组合 Quad-Head: 0.40 smooth + 0.20 conv + 0.20 cen1664 + 0.20 cen3840
    plm_quad_te = (0.40 * plm_te_smooth + 0.20 * plm_te_conv +
                   0.20 * pr_cen1664_te + 0.20 * pr_cen3840_te)
    print(f"  Quad-Head 测试集预测构建就绪: shape={plm_quad_te.shape}, 均值={plm_quad_te.mean():.4f}")

    # 5. 构建全量 Deep4 与全库 SimMix
    print("\n[4/5] 汇聚测试集 100% 全量 Deep4 (I103) 与全库 SimMix (ALIGN_W=0.50)...")
    d4_models = {}
    for m in ("c002b", "c005", "t001a", "t002"):
        d4_models[m] = np.load(os.path.join(REL_DIR, f"i103_{m}_test_probs.npz"))["test"]
    deep4_te = (wc["t001a"] * d4_models["t001a"] + wc["t002"] * d4_models["t002"]
                + wc["cnn"] * (0.5 * d4_models["c002b"] + 0.5 * d4_models["c005"]))
    deep5_te = (1.0 - W_PLM) * deep4_te + W_PLM * plm_quad_te

    idx100_t, bits100_t = i085.parse_topk(
        os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001_top100.tsv"), test_pids, full_map)
    te_map = {p: i for i, p in enumerate(test_pids)}
    t2t = i104.parse_2hop(os.path.join(CK005A_TMP, "align_test_to_test_e0.001.tsv"), te_map, te_map)
    sim_align_t = i113.sim_mix_align(idx100_t, bits100_t, y_full, t2t, len(test_pids))
    kmer_t = i104.build_kmer(y_full, "test")
    sim_te = ALIGN_W * sim_align_t + (1.0 - ALIGN_W) * kmer_t

    # 6. wRRF + 传输锐化与二值化出片
    print("\n[5/5] 执行加权 RRF + 锐化最优传输与精确二值化出片...")
    r_d_te = ranks(deep5_te)
    r_s_te = ranks(sim_te)
    rrf_te = col_norm(WD_RRF / (30 + r_d_te + 1) + WS_RRF / (30 + r_s_te + 1))
    s_te = deep5_te + SIM_W * sim_te + 0.50 * rrf_te

    row_m = np.maximum((s_te >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_full.mean(0) * len(s_te), 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s_te - s_te.max(axis=1, keepdims=True)) / SINK_TAU)
    p_te = sinkhorn(ker, row_m, col_m)
    final_te = (1.0 - SINK_ALPHA) * s_te + SINK_ALPHA * p_te
    y_pred = (final_te >= opt_ths[None, :]).astype(np.int8)

    # 严格合规核验
    out_csv = os.path.join(SUB_DIR, "submission_I121_quad_head.csv")
    sub = pd.concat([test_df[["protein_id"]].reset_index(drop=True),
                     pd.DataFrame(y_pred, columns=label_cols)], axis=1)

    assert sub.shape == (28450, 501), f"尺寸异常: {sub.shape}"
    assert set(np.unique(y_pred)).issubset({0, 1}), "非二值"
    assert not sub.isnull().any().any(), "含有空值"
    sub.to_csv(out_csv, index=False)

    mb = os.path.getsize(out_csv) / 1e6
    mean_card = float(y_pred.sum(1).mean())
    print(f"  出片成功: {out_csv}")
    print(f"  文件体积: {mb:.2f} MB (硬约束 <= 50MB)")
    print(f"  平均预测基数: {mean_card:.2f} (真值均值 25.42)")

    # 与过往提交对比差异
    diffs = {}
    for ref_name in ("submission_I117_weighted_rrf.csv",
                     "submission_I116_ultimate_calibrated.csv",
                     "submission_I115_grand_unified.csv"):
        ref = os.path.join(SUB_DIR, ref_name)
        if os.path.isfile(ref):
            old = pd.read_csv(ref)[label_cols].values.astype(np.int8)
            diff = int((old != y_pred).sum())
            diffs[ref_name] = diff
            print(f"  与 {ref_name:36s} 差异: {diff:7d} 格 ({diff / old.size:.3%})")

    elapsed = round(time.time() - t_start, 1)
    report = {
        "experiment": "I121",
        "method": "100% Full-Data Cross-Backbone Quad-Head (P004 SOTA Submit)",
        "val_f1_calibrated": round(f1_val, 5),
        "mean_cardinality": round(mean_card, 2),
        "file_size_mb": round(mb, 2),
        "diff_vs_prior": diffs,
        "elapsed_seconds": elapsed,
        "verdict": f"四元解耦全量终极提交出片成功，标定 val F1 达 {f1_val:.5f}，基数 {mean_card:.2f} 极度贴合真值",
    }

    report_path = os.path.join(REL_DIR, "I121_report.json")
    with Path(report_path).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"审计报告导出至: {report_path}")
    print("=" * 96)


if __name__ == "__main__":
    main()
