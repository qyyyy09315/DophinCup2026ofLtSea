"""
I059-transductive-diffusion — 基于全量直推式序列图的二阶同源标签扩散
====================================================================
依据:
  - SPROF-GO: Yuan et al., Briefings in Bioinformatics 2023, doi:10.1093/bib/bbad117。
    在由训练集与测试集联合构成的同源序列图 G=(V_tr U V_te, E) 上做拉普拉斯标签扩散。
  - Learning with Local and Global Consistency: Zhou et al., NeurIPS 2003。
    F^(t+1) = α S F^(t) + (1-α) Y_prior，指数收敛到闭式解。
机制:
  1. 突破现有单向 Test->Train 的 1-hop 局部比对局限；
  2. 构造联合邻接矩阵 W (包含跨集比对与测试集内部自同源 align_test_to_test)；
  3. 对于 2,140 条在训练集原本无命中的孤儿蛋白 (Miss 序列)，借助测试集内部的连通图分量实现
     2-hop / 多跳标签间接传递与近邻平滑；
  4. 扩散后与现有深度模型 S_deep 实施 RRF 秩融合与 Sinkhorn 正则化。
用法:
  python code/I/I059-transductive-diffusion.py
  python code/I/I059-transductive-diffusion.py --gen-sub
输出:
  rel/I059_report.json
  submit/submission_I059_diffusion.csv (--gen-sub)
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
REPORT_PATH = os.path.join(REL_DIR, "I059_report.json")

N_LABELS = 500

def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=42
    )
    return train_df, test_df, y_full, idx_train, idx_val, label_cols

def parse_aln_edges(tsv_path, q_map, t_map, evalue=0.001, min_bit=40.0):
    """从 DIAMOND tsv 提取稀疏图的边 (u, v, weight)"""
    rows, cols, data = [], [], []
    if not os.path.exists(tsv_path):
        print(f"Warning: {tsv_path} not found")
        return rows, cols, data
    with open(tsv_path, 'r') as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            q, t = p[0], p[1]
            if q not in q_map or t not in t_map: continue
            try:
                bs, ev = float(p[2]), float(p[3])
            except ValueError: continue
            if ev > evalue or bs < min_bit: continue
            u = q_map[q]
            v = t_map[t]
            rows.append(u)
            cols.append(v)
            data.append(bs)
    return rows, cols, data

def run_label_diffusion(W_sparse, Y_init, alpha=0.60, max_iter=15, tol=1e-5):
    """
    对称归一化拉普拉斯矩阵扩散:
    S = D^{-1/2} W D^{-1/2}
    F^(t+1) = alpha * S @ F^(t) + (1 - alpha) * Y_init
    """
    n_nodes = W_sparse.shape[0]
    # 对称化
    W_sym = (W_sparse + W_sparse.T) * 0.5
    # 对角度矩阵
    d = np.array(W_sym.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(np.maximum(d, 1e-12), -0.5)
    d_inv_sqrt[d == 0] = 0.0
    D_mat = sp.diags(d_inv_sqrt)
    S = D_mat @ W_sym @ D_mat
    
    F = Y_init.copy()
    for it in range(max_iter):
        F_old = F.copy()
        F = alpha * (S @ F) + (1.0 - alpha) * Y_init
        diff = np.max(np.abs(F - F_old))
        if diff < tol:
            break
    return F

def ranks(score):
    return np.argsort(np.argsort(-score, axis=1, kind="mergesort"), axis=1)

def rrf_pair(deep, sim, k=30):
    return 1.0 / (k + ranks(deep) + 1) + 1.0 / (k + ranks(sim) + 1)

def col_norm(x):
    lo = x.min(axis=0)
    hi = x.max(axis=0)
    return (x - lo) / np.maximum(hi - lo, 1e-9)

def sinkhorn(K, row_m, col_m, max_iter=100, tol=1e-5):
    u = np.ones_like(row_m)
    v = np.ones_like(col_m)
    for _ in range(max_iter):
        u_new = row_m / np.maximum(K @ v, 1e-12)
        v_new = col_m / np.maximum(K.T @ u_new, 1e-12)
        if np.max(np.abs(u_new - u)) < tol and np.max(np.abs(v_new - v)) < tol:
            break
        u, v = u_new, v_new
    P = u[:, None] * K * v[None, :]
    lo = P.min(axis=0)
    hi = P.max(axis=0)
    return (P - lo) / np.maximum(hi - lo, 1e-9)

def search_scut(ens, y):
    cands = np.linspace(0.01, 0.99, 100)
    best = np.full(N_LABELS, 0.5, dtype=np.float64)
    for i in range(N_LABELS):
        yt = y[:, i]
        if yt.sum() == 0: continue
        pr = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0)
        fp = (pr & ~ytm).sum(0)
        fn = (~pr & ytm).sum(0)
        f1t = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
        best[i] = cands[f1t.argmax()]
    return best

def macro_f1(probs, targets, ths):
    preds = (probs >= ths).astype(np.int8)
    tp = (preds & targets).sum(0)
    fp = (preds & ~targets).sum(0)
    fn = (~preds & targets).sum(0)
    p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return float(np.mean(f)), float(np.mean(p)), float(np.mean(r))

def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 65)
    print("  I059: Transductive Homology Graph Label Diffusion")
    print("=" * 65)
    
    train_df, test_df, y_full, idx_train, idx_val, label_cols = load_data()
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    n_tr, n_va = len(idx_train), len(idx_val)
    n_te = len(test_df)
    
    val_pids = train_df["protein_id"].values[idx_val]
    train_pids = train_df["protein_id"].values[idx_train]
    
    # 1. 验证集上的直推式图构建 (V = V_train_split U V_val_split)
    print("\n[1/4] Building validation graph: Train_split (91k) + Val_split (22.7k)...")
    # 统一节点 ID 空间: 0..n_tr-1 为训练集，n_tr..n_tr+n_va-1 为验证集
    tr_map = {pid: i for i, pid in enumerate(train_pids)}
    va_map = {pid: n_tr + i for i, pid in enumerate(val_pids)}
    total_val_nodes = n_tr + n_va
    
    # a. val -> train 比对边
    tsv_val_tr = os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    r1, c1, d1 = parse_aln_edges(tsv_val_tr, va_map, tr_map)
    
    # b. val -> val 自比对边 (内部簇)
    tsv_val_val = os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv")
    r2, c2, d2 = parse_aln_edges(tsv_val_val, va_map, va_map)
    
    all_r = r1 + c1 + r2
    all_c = c1 + r1 + c2
    all_d = d1 + d1 + d2
    
    # 添加自环保证非奇异
    for i in range(total_val_nodes):
        all_r.append(i)
        all_c.append(i)
        all_d.append(100.0)
        
    W_val_graph = sp.csr_matrix((all_d, (all_r, all_c)), shape=(total_val_nodes, total_val_nodes), dtype=np.float32)
    print(f"  Validation graph built: {total_val_nodes} nodes, {W_val_graph.nnz} non-zero edges")
    
    # 加载深度模型预测先验
    cnn_val = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_val = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    mix_val = 0.5 * cnn_val + 0.5 * c005_val
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep_v = w["t001a"] * t1 + w["t002"] * t2 + w["cnn"] * mix_val
    
    # 构造初始先验矩阵 Y_init: 训练集为真实硬标签，验证集为深度模型先验
    Y_val_init = np.zeros((total_val_nodes, N_LABELS), dtype=np.float32)
    Y_val_init[:n_tr] = y_train.astype(np.float32)
    Y_val_init[n_tr:] = deep_v.astype(np.float32)
    
    print("\n[2/4] Running Transductive Graph Diffusion on Validation Graph...")
    # 扫描扩散系数 alpha
    best_diff_f1 = -1.0
    best_alpha = 0.50
    best_diff_sim_v = None
    
    for a in [0.20, 0.35, 0.50, 0.65, 0.80]:
        F_val_all = run_label_diffusion(W_val_graph, Y_val_init, alpha=a, max_iter=15)
        # 提取验证集节点的扩散特征
        sim_diff_v = F_val_all[n_tr:]
        
        # 接入 RRF + Sinkhorn
        s_rrf = deep_v + 0.75 * sim_diff_v + 0.50 * col_norm(rrf_pair(deep_v, sim_diff_v, 30))
        # 搜临时 SCut 估基数
        t_tmp = search_scut(s_rrf, y_val)
        row_m = np.maximum((s_rrf >= t_tmp).sum(1).astype(np.float64), 1.0)
        col_m = np.maximum(y_train.mean(0) * n_va * 1.0, 1e-6)
        col_m *= row_m.sum() / col_m.sum()
        ker = np.exp((s_rrf - s_rrf.max(axis=1, keepdims=True)) / 0.20)
        p_v = sinkhorn(ker, row_m, col_m)
        mixed = 0.40 * s_rrf + 0.60 * p_v
        
        t_final = search_scut(mixed, y_val)
        f1, p_val, r_val = macro_f1(mixed, y_val, t_final)
        print(f"  alpha={a:.2f} -> Val Macro F1: {f1:.4f} (P={p_val:.4f}, R={r_val:.4f})")
        
        if f1 > best_diff_f1:
            best_diff_f1 = f1
            best_alpha = a
            best_diff_sim_v = sim_diff_v
            
    # 计算孤儿蛋白与命中切片
    tsv_val_tr = os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    seen_hits = set()
    with open(tsv_val_tr, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts and parts[0] in va_map: seen_hits.add(parts[0])
    hit_mask = np.array([pid in seen_hits for pid in val_pids])
    miss_mask = ~hit_mask
    
    print("\n[3/4] Validation Assessment & Comparison with Locked C011a (0.4453):")
    print(f"  Best Transductive Graph Diffusion F1: {best_diff_f1:.4f} (alpha={best_alpha:.2f})")
    print(f"  Locked Baseline C011a                : 0.4453")
    print(f"  Delta F1 vs C011a                    : {best_diff_f1 - 0.4453:+.4f}")
    
    report_dict = {
        "experiment": "I059",
        "best_alpha": float(best_alpha),
        "val_macro_f1": float(best_diff_f1),
        "delta_vs_c011a": float(best_diff_f1 - 0.4453),
        "passed_gate": bool(best_diff_f1 >= 0.4453 + 0.001)
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"  Wrote report to {REPORT_PATH}")
    
    # 4. 生成测试集提交 (如果满足或用户明确要求)
    if gen_sub or best_diff_f1 >= 0.4453 + 0.001:
        print("\n[4/4] Generating Full Test Submission with Transductive Graph Diffusion...")
        # 构建全量图: Train_full (113.8k) + Test_full (28.45k)
        full_tr_pids = train_df["protein_id"].values
        full_te_pids = test_df["protein_id"].values
        n_full_tr = len(full_tr_pids)
        n_full_te = len(full_te_pids)
        total_test_nodes = n_full_tr + n_full_te
        
        full_tr_map = {pid: i for i, pid in enumerate(full_tr_pids)}
        full_te_map = {pid: n_full_tr + i for i, pid in enumerate(full_te_pids)}
        
        # a. test -> train 比对
        tsv_test_tr = os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
        tr_r, tr_c, tr_d = parse_aln_edges(tsv_test_tr, full_te_map, full_tr_map)
        
        # b. test -> test 自比对
        tsv_test_te = os.path.join(CK005A_TMP, "align_test_to_test_e0.001.tsv")
        te_r, te_c, te_d = parse_aln_edges(tsv_test_te, full_te_map, full_te_map)
        
        tot_r = tr_r + tr_c + te_r
        tot_c = tr_c + tr_r + te_c
        tot_d = tr_d + tr_d + te_d
        for i in range(total_test_nodes):
            tot_r.append(i); tot_c.append(i); tot_d.append(100.0)
            
        W_test_graph = sp.csr_matrix((tot_d, (tot_r, tot_c)), shape=(total_test_nodes, total_test_nodes), dtype=np.float32)
        print(f"  Test graph built: {total_test_nodes} nodes, {W_test_graph.nnz} non-zero edges")
        
        # 加载测试集深度预测
        c005_test = np.load(os.path.join(REL_DIR, "C005_test_probs.npz"))["test"]
        cnn_test = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
        mix_te = 0.5 * cnn_test + 0.5 * c005_test
        t1_test = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_test"]
        t2_test = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
        deep_t = w["t001a"] * t1_test + w["t002"] * t2_test + w["cnn"] * mix_te
        
        Y_test_init = np.zeros((total_test_nodes, N_LABELS), dtype=np.float32)
        Y_test_init[:n_full_tr] = y_full.astype(np.float32)
        Y_test_init[n_full_tr:] = deep_t.astype(np.float32)
        
        F_test_all = run_label_diffusion(W_test_graph, Y_test_init, alpha=best_alpha, max_iter=15)
        sim_diff_t = F_test_all[n_full_tr:]
        
        s011_t = deep_t + 0.75 * sim_diff_t + 0.50 * col_norm(rrf_pair(deep_t, sim_diff_t, 30))
        
        # 验证集学到的最优 SCut 阈值
        row_t = np.maximum((s011_t >= t_tmp).sum(1).astype(np.float64), 1.0)
        col_t = np.maximum(y_full.mean(0) * n_full_te * 1.0, 1e-6)
        col_t *= row_t.sum() / col_t.sum()
        ker_t = np.exp((s011_t - s011_t.max(axis=1, keepdims=True)) / 0.20)
        p_t = sinkhorn(ker_t, row_t, col_t)
        mixed_t = 0.40 * s011_t + 0.60 * p_t
        
        y_pred = (mixed_t >= t_final).astype(np.int8)
        
        out_csv = os.path.join(SUB_DIR, "submission_I059_diffusion.csv")
        sub_df = pd.concat([test_df[["protein_id"]], pd.DataFrame(y_pred, columns=label_cols)], axis=1)
        sub_df.to_csv(out_csv, index=False)
        print(f"  Generated submission: {out_csv} ({len(sub_df)} rows)")
    
    print(f"\nCompleted in {time.time()-t0:.1f}s")
    print("=" * 65)

if __name__ == "__main__":
    main()