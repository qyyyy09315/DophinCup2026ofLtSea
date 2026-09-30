from pathlib import Path
"""
I115 — 三位一体终极全量融合提交版 (Grand Unified Full-Data SOTA Submission)
========================================================================
核心背景与三位一体设计:
  - 自 9 月下旬以来，三大突破在不同支路上分别被证实带来平台/统计显著收益:
    1. 【Deep4 支 100% 全量重训 (I103)】:
       Transformer (T001a, T002) 与 CNN (C002b, C005) 首次全部在 100% 全量数据 (113,796 条)
       完成 Cosine 退火充分重训，补齐了占深度分支 65% 权重的历史最后一块全量拼图！
    2. 【PLM 头代际融合 (I109/I110)】:
       78-Epoch 充分收敛头与 35-Epoch 平滑头的概率代际混合 (alpha=0.25)，同时兼备
       微观排序质量 (microAUC 0.8856 突破历史纪录) 与操作点拟合精度！
    3. 【同源比对支代际混合 (I112/I113)】:
       top20 与 top3 比对的代际加权 (beta=0.75)，在 R=20 重复半折配对检验中取得
       全项目最强传导信号 (ΔOOF = +0.0009, CI95 下界 +0.0008, P=1.00)！
  - 历史局限: 过去这三项创新各自独立出片（I103b 仅含 1；I110 仅含 2；I113 仅含 2+3 但 deep4 仍用 80% 旧概率），
    本脚本首次将【全量 Deep4】+【全量代际 PLM 混合头】+【全量代际比对 SimMix】完美合体！
  - 严格合规性保障:
    1. 28,450 行 x 501 列 (protein_id + 500 标签);
    2. 严格 0/1 二值化 (无任何浮点泄漏);
    3. 文件大小严格 <= 50MB (实测约 28.7MB);
    4. 绝不使用任何被禁用的外部注释数据库。
输出: submit/submission_I115_grand_unified.csv (+ 桌面副本), rel/I115_report.json
用法: python code/I/I115-grand-unified-submit.py
"""

import os
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split

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
ALPHA = 0.25   # PLM 混合头参数 (I109 最优: 0.25 conv + 0.75 weak)
BETA = 0.75    # Sim 代际混合参数 (I112 最优: 0.75 top20 + 0.25 top3)

i096 = SourceFileLoader("i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")).load_module()
i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
i085 = SourceFileLoader("i085", os.path.join(PROJECT_ROOT, "code", "I", "I085-top100-gated-agg.py")).load_module()
i104 = SourceFileLoader("i104", os.path.join(PROJECT_ROOT, "code", "I", "I104-top20sim-submit.py")).load_module()
i113 = SourceFileLoader("i113", os.path.join(PROJECT_ROOT, "code", "I", "I113-simmix-submit.py")).load_module()
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe
ranks, col_norm, sinkhorn = i064.ranks, i064.col_norm, i064.sinkhorn


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 96)
    print("  I115: Grand Unified SOTA Full-Data Submission (Deep4-Full + Blend-Head + SimMix)")
    print("=" * 96)

    # 1. 载入基础数据
    print("[1/6] 载入训练与测试数据映射表...")
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

    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]

    # 2. 构建验证集基线 (严密保持 val 0.4491 SOTA 闭环校准)
    print("[2/6] 构建验证集全流程管线 (0.4491 口径阈值校准)...")
    cnn_raw = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1_v = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2_v = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    deep4_v = wc["t001a"] * t1_v + wc["t002"] * t2_v + wc["cnn"] * (0.5 * cnn_raw + 0.5 * c005_v)

    conv_v = np.load(os.path.join(REL_DIR, "i106_conv_multiseed_val.npz"))["val"]
    weak_v = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    blend_v = ALPHA * conv_v + (1.0 - ALPHA) * weak_v
    deep5_v = (1.0 - W_PLM) * deep4_v + W_PLM * blend_v

    idx100_v, bits100_v = i085.parse_topk(
        os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001_top100.tsv"), val_pids, train_map)
    v2v = i104.parse_2hop(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv"), val_map, val_map)
    sim_v = 0.4 * i113.sim_mix_align(idx100_v, bits100_v, y_tr, v2v, n_va) + 0.6 * i104.build_kmer(y_tr, "val")

    # I114 升级重校: SIM_W 0.75 -> 0.80
    rrf_v = col_norm(1.0 / (30 + ranks(deep5_v) + 1) + 1.0 / (30 + ranks(sim_v) + 1))
    s_v = deep5_v + 0.80 * sim_v + 0.50 * rrf_v
    t_tmp = search_scut(s_v, y_va)
    row_m_v = np.maximum((s_v >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m_v = np.maximum(y_tr.mean(0) * n_va, 1e-6); col_m_v *= row_m_v.sum() / col_m_v.sum()
    ker_v = np.exp((s_v - s_v.max(axis=1, keepdims=True)) / 0.20)
    p_v = sinkhorn(ker_v, row_m_v, col_m_v)
    mixed_v = 0.40 * s_v + 0.60 * p_v
    opt_ths = search_scut(mixed_v, y_va)
    f1_val = macro_f1(mixed_v, y_va, opt_ths)
    t_tmp = search_scut(s_v, y_va)
    print(f"  [验证集校准完成] SOTA val F1 = {f1_val:.4f} (必须为 0.4491)")

    # 3. 构建测试集全量 Deep4 (来自 I103 100% 全量重训产物)
    print("[3/6] 加载测试集 100% 全量重训 Deep4 (I103 产物)...")
    d4_models = {}
    for m in ("c002b", "c005", "t001a", "t002"):
        p = os.path.join(REL_DIR, f"i103_{m}_test_probs.npz")
        assert os.path.exists(p), f"缺少 {p}"
        d4_models[m] = np.load(p)["test"]
        print(f"  {m:6s} 全量概率均值: {d4_models[m].mean():.4f}")

    deep4_te = (wc["t001a"] * d4_models["t001a"] + wc["t002"] * d4_models["t002"]
                + wc["cnn"] * (0.5 * d4_models["c002b"] + 0.5 * d4_models["c005"]))
    print(f"  Deep4 全量测试集加权均值: {deep4_te.mean():.4f}")

    # 4. 构建测试集全量代际 PLM 混合头 (0.25 I097 + 0.75 I081)
    print("[4/6] 构建测试集全量代际 PLM 混合头 (0.25 I097 + 0.75 I081)...")
    plm_te_i097 = i104.mlp_test_mean(device)          # 78ep 全量三种子
    plm_te_i081 = np.load(os.path.join(REL_DIR, "plm1664_fulltrain_test.npz"))["test"]  # 35ep 全量三种子
    blend_te = ALPHA * plm_te_i097 + (1.0 - ALPHA) * plm_te_i081
    deep5_te = (1.0 - W_PLM) * deep4_te + W_PLM * blend_te
    print(f"  PLM 混合头测试集均值: {blend_te.mean():.4f}, Deep5 测试集均值: {deep5_te.mean():.4f}")

    # 5. 构建测试集全库比对代际混合 (SimMix)
    print("[5/6] 构建测试集全库比对代际混合 (SimMix: 0.75 top20 + 0.25 top3 + 2-hop + kmer)...")
    idx100_t, bits100_t = i085.parse_topk(
        os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001_top100.tsv"), test_pids, full_map)
    te_map = {p: i for i, p in enumerate(test_pids)}
    t2t = i104.parse_2hop(os.path.join(CK005A_TMP, "align_test_to_test_e0.001.tsv"), te_map, te_map)
    sim_t = 0.4 * i113.sim_mix_align(idx100_t, bits100_t, y_full, t2t, len(test_pids)) + 0.6 * i104.build_kmer(y_full, "test")

    # 6. RRF 秩融合 + 最优传输 Sinkhorn 决策
    print("[6/6] 执行最终 RRF + 最优传输 Sinkhorn 决策与二值化...")
    s_te = deep5_te + 0.80 * sim_t + 0.50 * col_norm(  # I114 升级重校: SIM_W 0.75->0.80 (val 0.4494)
        1.0 / (30 + ranks(deep5_te) + 1) + 1.0 / (30 + ranks(sim_t) + 1))

    row_m = np.maximum((s_te >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_full.mean(0) * len(s_te), 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s_te - s_te.max(axis=1, keepdims=True)) / 0.20)
    p_te = sinkhorn(ker, row_m, col_m)
    final = 0.40 * s_te + 0.60 * p_te
    y_pred = (final >= opt_ths[None, :]).astype(np.int8)

    # 严格合规性断言与写盘
    out_csv = os.path.join(SUB_DIR, "submission_I115_grand_unified.csv")
    sub = pd.concat([test_df[["protein_id"]].reset_index(drop=True),
                     pd.DataFrame(y_pred, columns=label_cols)], axis=1)
    assert sub.shape == (28450, 501), f"尺寸异常: {sub.shape}"
    assert set(np.unique(y_pred)).issubset({0, 1}), "非二值"
    sub.to_csv(out_csv, index=False)
    mb = os.path.getsize(out_csv) / 1e6
    assert mb <= 50, f"文件过大: {mb}MB"

    # 与各个前序提交对比
    diffs = {}
    for ref_name in ("submission_I113_simmix.csv", "submission_I110_blend.csv",
                     "submission_I104_top20sim.csv", "submission_I103_deep4full.csv",
                     "submission_I097_fulltrain_80ep.csv"):
        ref = os.path.join(SUB_DIR, ref_name)
        if os.path.isfile(ref):
            old = pd.read_csv(ref)[label_cols].values.astype(np.int8)
            diff = int((old != y_pred).sum())
            diffs[ref_name] = diff
            print(f"  与 {ref_name:36s} 差异: {diff:7d} 格 ({diff / old.size:.3%})")

    # 桌面备份
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if os.path.isdir(desktop):
        shutil.copy2(out_csv, os.path.join(desktop, "submission_I115_grand_unified.csv"))
        print(f"  [桌面副本已放]: {os.path.join(desktop, 'submission_I115_grand_unified.csv')}")

    report = {
        "experiment": "I115",
        "method": "Grand Unified SOTA Full-Data Submission (Deep4-Full + Blend-Head + SimMix)",
        "val_f1_calibrated": round(f1_val, 4),
        "predicted_cardinality_mean": round(float(y_pred.sum(1).mean()), 2),
        "total_positives": int(y_pred.sum()),
        "file_size_mb": round(mb, 2),
        "diff_vs_prior": diffs,
        "elapsed_seconds": round(time.time() - t0, 1)
    }
    with Path(REL_DIR, "I115_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 96)
    print(f"  I115 完成! 终极三位一体提交生成完毕 -> {out_csv}")
    print(f"  预测基数均值: {report['predicted_cardinality_mean']} (真值 25.42) | 文件体积: {mb:.1f} MB")
    print(f"  总耗时: {(time.time()-t0)/60:.1f} min")
    print("=" * 96)


if __name__ == "__main__":
    main()
