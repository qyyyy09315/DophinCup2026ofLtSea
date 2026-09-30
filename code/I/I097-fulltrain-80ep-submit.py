"""
I097 — 100% 全量数据充分收敛重训 (80-Epoch) 与新平台提交生成
============================================================
核心依据与设计:
  - 依据周志华老师《机器学习》第 2.2 节交付原则:
    在验证集完成超参数与轮次锁定后，必须并回验证集 (100% 全量数据 113,796 条)，
    最大化样本有效性 (Sample Efficiency) 与长尾/稀疏标签统计支撑。
  - 前置实测支撑:
    1. I081 在 35 轮下的全量重训在双盲平台取得 45.9546 (+0.1881)，证明全量数据有真增益；
    2. 刚刚实测证明：解除 35 轮硬截断至 80 轮后，PLM-MLP 独立 F1 从 0.2629 暴涨至 0.3028 (+0.0399)；
       最佳验证轮次锁定在 epoch 78！
  - 本脚本在 100% 全量标注数据 (113,796 条) 上执行三种子 (11, 22, 33) 的 78-Epoch 充分收敛重训，
    推断测试集 (28,450 条)，接入全量同源库 (DIAMOND + k-mer) 与 deep4 展开 RRF + Sinkhorn 决策，
    生成完全合规的二值提交文件 submit/submission_I097_fulltrain_80ep.csv。

严格提交约束核验:
  1. 值域严格为 0 和 1 二值;
  2. 文件大小 <= 50MB (实测约 28.7MB);
  3. 维度严格为 28,450 行 x 500 标签 + 1 列 protein_id;
  4. 绝不使用任何被禁用的外部注释数据库 (GO DAG、PPI、UniProt 注释等)。

用法:
  python code/I/I097-fulltrain-80ep-submit.py            # 正式全量重训与提交生成
  python code/I/I097-fulltrain-80ep-submit.py --smoke    # 快速冒烟校验
输出:
  submit/submission_I097_fulltrain_80ep.csv (+ 桌面副本)
  rel/I097_report.json
"""

import os
import sys
import json
import time
import argparse
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
W_PLM = 0.35
SEEDS = [11, 22, 33]
OPTIMAL_EPOCHS = 78  # 由 I096 探针在全量验证曲线上测定的最优收敛轮次

i064 = SourceFileLoader(
    "i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")
).load_module()
MLP, ASL = i064.MLP, i064.ASL
macro_f1, search_scut = i064.macro_f1, i064.search_scut
ranks, col_norm, sinkhorn = i064.ranks, i064.col_norm, i064.sinkhorn


def train_seed_full(seed, X_all, y_all, epochs, device):
    """在 100% 全量数据上训练单种子模型至充分收敛轮次"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_all), torch.from_numpy(y_all.astype(np.float32))),
        batch_size=512, shuffle=True, pin_memory=True
    )
    model = MLP(X_all.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = ASL()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"\n  [Seed {seed}] 开始 100% 全量数据重训 ({epochs} epochs)...")
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        loss_sum, n_seen = 0.0, 0
        for bx, by in loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(bx)
                loss = crit(logits, by)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * bx.size(0)
            n_seen += bx.size(0)
        sch.step()

        if ep % 10 == 0 or ep == epochs:
            print(f"    seed {seed} ep{ep:02d}/{epochs:02d} | loss={loss_sum/n_seen:.4f} ({time.time()-t_ep:.0f}s)", flush=True)

    return model


@torch.no_grad()
def predict_batches(model, X, device, batch_size=512):
    model.eval()
    preds = []
    for s in range(0, len(X), batch_size):
        bx = torch.from_numpy(X[s:s + batch_size]).to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            pr = torch.sigmoid(model(bx).float()).cpu().numpy()
        preds.append(pr)
    return np.vstack(preds)


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


def build_test_sim(y_full, test_pids, full_map):
    print("  构建测试集同源表征 (DIAMOND 100% 全库比对 + 2-hop + k-mer)...")
    idx_te, bits_te = parse_aln(os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv"),
                                test_pids, full_map)
    hit_te = idx_te[:, 0] >= 0
    miss_te = ~hit_te
    sim1_te = agg(idx_te, bits_te, y_full)

    # 2-hop 传递
    t2t = {}
    with open(os.path.join(CK005A_TMP, "align_test_to_test_e0.001.tsv")) as f:
        te_map = {p: i for i, p in enumerate(test_pids)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            q, t = p[0], p[1]
            if q == t or q not in te_map or t not in te_map:
                continue
            qi, ti = te_map[q], te_map[t]
            t2t.setdefault(qi, [])
            if len(t2t[qi]) < 5:
                t2t[qi].append((ti, float(p[2])))
    sim2_te = sim1_te.copy()
    for qi in range(len(test_pids)):
        if miss_te[qi] and qi in t2t:
            acc = np.zeros(N_LABELS, np.float32)
            aw = 0.0
            for ti, bs in t2t[qi]:
                if hit_te[ti]:
                    acc += sim1_te[ti] * bs
                    aw += bs
            if aw > 0:
                sim2_te[qi] = acc / aw

    # k-mer 流形
    kmer_te = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        fname = "TCK002a_neigh_full.npz" if k == 6 else f"CK004_neigh_k{k}_full.npz"
        d = np.load(os.path.join(REL_DIR, fname))
        idx = d["idx_test"][:, :5]
        sm = d["sim_test"][:, :5].astype(np.float32)
        n = idx.shape[0]
        lab = y_full[idx.ravel()].reshape(n, 5, -1).astype(np.float32)
        ww = np.maximum(sm, 1e-7) ** 2
        kmer_te += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]

    return 0.4 * sim2_te + 0.6 * kmer_te


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="沙箱快速冒烟模式 (~30s)")
    parser.add_argument("--epochs", type=int, default=OPTIMAL_EPOCHS, help="全量重训轮数 (默认 78)")
    args = parser.parse_args()

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 84)
    print(f"  I097: 100% Full-Data Converged (80ep) Refit & Platform Submission Rebuild ({'SMOKE' if args.smoke else '正式'})")
    print(f"  Device: {device} | Refit Epochs: {args.epochs if not args.smoke else 2} | Seeds: {SEEDS}")
    print("=" * 84)

    # 1. 加载 100% 全量特征矩阵
    print("[1/5] 加载 100% 全量训练特征 (113,796) 与测试特征 (28,450)...")
    df_e_tr = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    df_p_tr = pd.read_parquet(os.path.join(REL_DIR, "train_embeddings.parquet"))
    df_e_te = pd.read_parquet(os.path.join(REL_DIR, "test_esm2_embeddings.parquet"))
    df_p_te = pd.read_parquet(os.path.join(REL_DIR, "test_embeddings.parquet"))

    ec = [c for c in df_e_tr.columns if c.startswith("emb_")]
    pc = [c for c in df_p_tr.columns if c.startswith("emb_")]
    X_train_100 = np.concatenate([df_e_tr[ec].values, df_p_tr[pc].values], axis=1).astype(np.float32)
    X_test_100 = np.concatenate([df_e_te[ec].values, df_p_te[pc].values], axis=1).astype(np.float32)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_train_100 = train_df[label_cols].values.astype(np.int8)

    full_map = {p: i for i, p in enumerate(train_df["protein_id"].values)}
    test_pids = test_df["protein_id"].values
    n_train = len(X_train_100)
    n_test = len(X_test_100)

    if args.smoke:
        X_train_100 = X_train_100[:2048]
        y_train_100 = y_train_100[:2048]
        X_test_100 = X_test_100[:512]
        test_df = test_df.iloc[:512]
        test_pids = test_pids[:512]
        epochs = 2
        seeds = [11]
    else:
        epochs = args.epochs
        seeds = SEEDS

    print(f"  100% 训练集规模: {len(X_train_100)} 条, 测试集: {len(X_test_100)} 条, 特征维度: {X_train_100.shape[1]}")

    # 2. 100% 全量数据三种子训练
    print(f"\n[2/5] 在 100% 全量数据上训练 {len(seeds)} 个种子 (epochs={epochs})...")
    test_preds_seeds = []
    for s in seeds:
        m = train_seed_full(s, X_train_100, y_train_100, epochs, device)
        pr_te = predict_batches(m, X_test_100, device)
        test_preds_seeds.append(pr_te)
        torch.save(m.state_dict(), os.path.join(MODEL_DIR, f"i097_fulltrain_s{s}.pt"))

    plm_test_mean = np.mean(test_preds_seeds, axis=0)
    print(f"  PLM 三种子预测均值就绪: shape={plm_test_mean.shape}, 均值={plm_test_mean.mean():.4f}")

    # 3. 构建全量测试集 deep4 与 sim
    print("\n[3/5] 构建测试集全量 deep4 与 sim 支...")
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]

    if not args.smoke:
        t1_te = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_test"]
        t2_te = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]  # 与 I065/I081 同约定
        cnn_te = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
        c005_te = np.load(os.path.join(REL_DIR, "C005_test_probs.npz"))["test"]

        # 构建 deep4 测试集
        deep4_te = wc["t001a"] * t1_te + wc["t002"] * t2_te + wc["cnn"] * (0.5 * cnn_te + 0.5 * c005_te)
        sim_te = build_test_sim(y_train_100, test_pids, full_map)
    else:
        deep4_te = np.random.rand(len(X_test_100), N_LABELS).astype(np.float32)
        sim_te = np.random.rand(len(X_test_100), N_LABELS).astype(np.float32)

    # 4. 融合推理：RRF 秩融合 + Sinkhorn 双边际正则化
    print("\n[4/5] 执行全量测试集 RRF + Sinkhorn 最优传输决策...")
    d_te = (1.0 - W_PLM) * deep4_te + W_PLM * plm_test_mean
    s_te = d_te + 0.75 * sim_te + 0.50 * col_norm(
        1.0 / (30 + ranks(d_te) + 1) + 1.0 / (30 + ranks(sim_te) + 1)
    )

    # 从 80% 验证集重新精确搜索最稳健的 SCut 阈值向量
    train_idx, val_idx = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_va_ref = train_df[label_cols].values[val_idx].astype(np.int8)

    # 阈值搜索的 PLM 参考须与测试侧同配方（78ep 收敛头）；35ep multiseed 分布失配（诊断 mean|Δth|≈0.071）
    plm_val_ref = np.load(os.path.join(REL_DIR, "i096_plm_val_probs.npz"))["val"]
    cnn_v = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1_v = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2_v = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    deep4_v = wc["t001a"] * t1_v + wc["t002"] * t2_v + wc["cnn"] * (0.5 * cnn_v + 0.5 * c005_v)

    # 获取全验证集融合分与最优 SCut 阈值
    i096_mod = SourceFileLoader(
        "i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")
    ).load_module()
    val_pids_ref = train_df["protein_id"].values[val_idx]
    train_map_ref = {p: i for i, p in enumerate(train_df["protein_id"].values[train_idx])}
    val_map_ref = {p: i for i, p in enumerate(val_pids_ref)}
    sim_v = i096_mod.build_sim(train_df[label_cols].values[train_idx].astype(np.int8), len(val_idx), val_pids_ref, train_map_ref, val_map_ref)

    _, _, mixed_v_ref, opt_ths, s_v_ref = i064.full_pipe(deep4_v, sim_v, train_df[label_cols].values[train_idx].astype(np.int8), y_va_ref, len(val_idx), plm=plm_val_ref, w=0.35)
    # I081 同约定：row 边际用 s 级阈值 t_tmp，最终二值化用 mixed 级阈值 opt_ths
    t_tmp = search_scut(s_v_ref, y_va_ref)
    print(f"  锁定的最优逐标签 SCut 阈值: 均值={opt_ths.mean():.4f}, 最小值={opt_ths.min():.4f}, 最大值={opt_ths.max():.4f}")

    # Sinkhorn 边际推断
    row_m_te = np.maximum((s_te >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m_te = np.maximum(y_train_100.mean(0) * len(s_te), 1e-6)
    col_m_te *= row_m_te.sum() / col_m_te.sum()
    ker_te = np.exp((s_te - s_te.max(axis=1, keepdims=True)) / 0.20)
    p_te = sinkhorn(ker_te, row_m_te, col_m_te)
    final_scores_te = 0.40 * s_te + 0.60 * p_te

    # 5. 0/1 二值化与合规性核验
    print("\n[5/5] 生成 0/1 二值预测并执行严格合规性审计...")
    final_binary_preds = (final_scores_te >= opt_ths[None, :]).astype(np.int8)

    sub_file = os.path.join(SUB_DIR, "submission_I097_fulltrain_80ep.csv")
    sub_df = pd.concat([
        test_df[["protein_id"]].reset_index(drop=True),
        pd.DataFrame(final_binary_preds, columns=label_cols)
    ], axis=1)
    sub_df.to_csv(sub_file, index=False)

    # 严格合规性断言
    assert sub_df.shape == (28450 if not args.smoke else 512, 501), f"尺寸不匹配: {sub_df.shape}"
    assert set(np.unique(final_binary_preds)).issubset({0, 1}), "非 0/1 二值提交!"
    file_size_mb = os.path.getsize(sub_file) / (1024 * 1024)
    assert file_size_mb <= 50.0, f"文件大小超过 50MB: {file_size_mb:.1f}MB"
    print(f"  [合规验证通过] 行数: {len(sub_df)}, 列数: {sub_df.shape[1]}, 文件体积: {file_size_mb:.2f} MB, 值域全为 0/1")

    # 对比与 I081 的逐格差异
    diff_report = {}
    i081_sub_file = os.path.join(SUB_DIR, "submission_I081_fulltrain.csv")
    if os.path.isfile(i081_sub_file) and not args.smoke:
        df_old = pd.read_csv(i081_sub_file)
        diff_cells = int((sub_df[label_cols].values != df_old[label_cols].values).sum())
        total_cells = sub_df.shape[0] * 500
        diff_ratio = diff_cells / total_cells
        diff_report = {
            "diff_cells_vs_I081": diff_cells,
            "total_cells": total_cells,
            "diff_ratio": round(diff_ratio, 6)
        }
        print(f"  与 I081 当前最佳提交差异格数: {diff_cells} / {total_cells} ({diff_ratio:.3%})")

    # 桌面备份
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if os.path.isdir(desktop):
        dest = os.path.join(desktop, "submission_I097_fulltrain_80ep.csv")
        shutil.copy2(sub_file, dest)
        print(f"  桌面副本已放: {dest}")

    # 保存报告
    report = {
        "experiment": "I097",
        "method": "100% Full-Data Converged (78ep) Refit & Platform Submission Rebuild",
        "smoke": args.smoke,
        "n_train": n_train,
        "n_test": n_test,
        "seeds": seeds,
        "refit_epochs": epochs,
        "submission_file": sub_file,
        "file_size_mb": round(file_size_mb, 2),
        "total_predicted_positives": int(final_binary_preds.sum()),
        "average_cardinality": round(float(final_binary_preds.sum() / len(final_binary_preds)), 2),
        "diff_vs_I081": diff_report,
        "elapsed_seconds": round(time.time() - t_start, 1)
    }

    report_path = os.path.join(REL_DIR, "I097_report.json")
    with Path(report_path).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 84)
    print(f"  I097 完成! 提交生成完毕 -> {sub_file}")
    print(f"  报告写入: {report_path} | 总用时: {(time.time()-t_start)/60:.1f} min")
    print("=" * 84)


if __name__ == "__main__":
    main()
