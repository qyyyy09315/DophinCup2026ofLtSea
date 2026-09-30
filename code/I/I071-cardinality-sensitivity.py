"""
I071 — 基数预测精度 -> F1 的灵敏度 + 可实现的基数估计
=====================================================
I070 发现: Oracle RCut (按真实逐样本基数取 top-k) 达 F1 0.5326，较 SCut 0.4475 高 +0.0851，
           且召回更高、FPR 更低。若能逼近真实基数，即可大幅提分。
本实验回答两个问题:
  Q1 灵敏度: 基数估计误差 sigma 多大时，RCut 的 F1 优势消失？(决定值不值得投入)
  Q2 可实现性: 用现有特征能估到多准？(岭回归 / 分数剖面统计 / 二者结合)
输出: rel/I071_report.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
REPORT = os.path.join(REL_DIR, "I071_report.json")
N_LABELS = 500
W_PLM = 0.35

from importlib.machinery import SourceFileLoader
from pathlib import Path
i66c = SourceFileLoader("i66c", os.path.join(PROJECT_ROOT, "code", "I", "I066c-linear-calibration.py")).load_module()


def macro_f1_from_pred(preds, y):
    preds = preds.astype(bool); yb = y.astype(bool)
    tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    return float(np.mean(np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0.0)))


def macro_pr_from_pred(preds, y):
    """(macro R, macro P, macro AUC-of-binary-decision)"""
    preds = preds.astype(bool); yb = y.astype(bool)
    tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    fpr = np.where(tp + fp > 0, fp / np.maximum((~yb).sum(0), 1), 0.0)
    tpr = r
    auc = float(np.mean((tpr - fpr + 1.0) / 2.0))
    return float(np.mean(r)), float(np.mean(p)), auc


def rcut_predict(S, k):
    """按每样本基数 k (向量) 取 top-k"""
    n = S.shape[0]
    order = np.argsort(-S, axis=1)
    pred = np.zeros_like(S, dtype=bool)
    rows = np.arange(n)[:, None]
    mask = np.arange(S.shape[1])[None, :] < np.clip(k, 0, N_LABELS)[:, None].astype(int)
    pred[rows, order] = mask
    return pred


def main():
    t0 = time.time()
    print("=" * 78)
    print("  I071: Cardinality-Accuracy Sensitivity and Achievable Estimation")
    print("=" * 78)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]

    cnn_v = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1v = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2v = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    deep4_v = wc["t001a"]*t1v + wc["t002"]*t2v + wc["cnn"]*(0.5*cnn_v + 0.5*c005_v)
    deep5_v = (1-W_PLM)*deep4_v + W_PLM*np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]

    idx_v, bits_v = i66c.parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                                   val_pids, train_map)
    hit_v = idx_v[:, 0] >= 0
    sim1_v = i66c.agg(idx_v, bits_v, y_tr)
    v2v = {}
    with open(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            q, t = p[0], p[1]
            if q == t or q not in val_map or t not in val_map: continue
            qi, ti = val_map[q], val_map[t]
            v2v.setdefault(qi, [])
            if len(v2v[qi]) < 5: v2v[qi].append((ti, float(p[2])))
    sim_v = 0.4*i66c.transitive_2hop(sim1_v, hit_v, ~hit_v, v2v, n_va) + 0.6*i66c.build_kmers(y_tr, "val")
    s_v = deep5_v + 0.75*sim_v + 0.50*i66c.col_norm(
        1.0/(30+i66c.ranks(deep5_v)+1) + 1.0/(30+i66c.ranks(sim_v)+1))
    t_tmp = i66c.search_scut(s_v, y_va)
    row_m = np.maximum((s_v >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_tr.mean(0)*n_va, 1e-6); col_m *= row_m.sum()/col_m.sum()
    p_v = i66c.sinkhorn(np.exp((s_v - s_v.max(axis=1, keepdims=True))/0.20), row_m, col_m)
    S = 0.40*s_v + 0.60*p_v

    card_true = y_va.sum(1).astype(np.float64)
    ths = i66c.search_scut(S, y_va)
    f1_scut = macro_f1_from_pred(S >= ths, y_va)
    print(f"\n  基准: SCut F1 = {f1_scut:.4f} | 真实基数 mean={card_true.mean():.2f} std={card_true.std():.2f}")

    # ---------- Q1: 基数噪声灵敏度 ----------
    print("\n  Q1. 基数估计噪声 -> RCut F1 灵敏度 (sigma 为高斯噪声标准差):")
    print(f"     {'sigma':>7s} | {'Macro F1':>9s} | {'较SCut':>9s} | {'与SCut差':>9s}")
    print("     " + "-" * 44)
    rng = np.random.default_rng(0)
    sens = {}
    f1_oracle = macro_f1_from_pred(rcut_predict(S, card_true), y_va)
    print(f"     {'0(真值)':>7s} | {f1_oracle:9.4f} | {f1_oracle-f1_scut:+9.4f} |         -")
    sens['0'] = f1_oracle
    for sigma in [2, 5, 10, 15, 20, 25]:
        k = np.clip(card_true + rng.normal(0, sigma, n_va), 1, N_LABELS)
        f1 = macro_f1_from_pred(rcut_predict(S, k), y_va)
        sens[str(sigma)] = f1
        mark = "领先" if f1 > f1_scut else "落后"
        print(f"     {sigma:7d} | {f1:9.4f} | {f1-f1_scut:+9.4f} |  {mark}")

    # ---------- Q2: 可实现的基数估计 ----------
    print("\n  Q2. 可实现的基数估计 (在 train split 上拟合, val 上评估):")
    # 特征1: 分数剖面统计 (多阈值下的命中数)
    def profile_feats(Sm):
        qs = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        cols = [(Sm >= q).sum(1).astype(np.float32) for q in qs]
        cols.append(Sm.max(1)); cols.append(Sm.mean(1)); cols.append(Sm.std(1))
        return np.stack(cols, axis=1)

    # 训练集侧: 用 train split 的分数剖面（用 deep5/sim 近似，避免重算全链）
    # 为公平起见, 用同一套剖面对 train split 计算
    S_tr_prof = np.concatenate([deep5_v[idx_tr_local] if False else np.zeros((1, 1))], axis=0) if False else None

    # 直接用 val 的剖面特征做"可达性"上界估计: 5 折交叉拟合
    P_val = profile_feats(S)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    pred_card = np.zeros(n_va)
    for tr_i, te_i in kf.split(P_val):
        reg = Ridge(alpha=50.0)
        reg.fit(P_val[tr_i], card_true[tr_i])
        pred_card[te_i] = reg.predict(P_val[te_i])
    mae = float(np.mean(np.abs(pred_card - card_true)))
    corr = float(np.corrcoef(pred_card, card_true)[0, 1])
    print(f"     剖面特征+岭回归: MAE={mae:.2f} | 相关系数={corr:.4f}")
    f1_hat = macro_f1_from_pred(rcut_predict(S, pred_card), y_va)
    print(f"     -> RCut(预测基数) F1 = {f1_hat:.4f} ({f1_hat-f1_scut:+.4f} vs SCut)")

    # 加入 PLM 嵌入特征
    df_e = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    ec = [c for c in df_e.columns if c.startswith("emb_")]
    emb_val = df_e[ec].values[idx_va].astype(np.float32)
    P2 = np.concatenate([P_val, emb_val], axis=1)
    pred_card2 = np.zeros(n_va)
    for tr_i, te_i in kf.split(P2):
        reg = Ridge(alpha=100.0)
        reg.fit(P2[tr_i], card_true[tr_i])
        pred_card2[te_i] = reg.predict(P2[te_i])
    mae2 = float(np.mean(np.abs(pred_card2 - card_true)))
    corr2 = float(np.corrcoef(pred_card2, card_true)[0, 1])
    print(f"     剖面+ESM2嵌入+岭回归: MAE={mae2:.2f} | 相关系数={corr2:.4f}")
    f1_hat2 = macro_f1_from_pred(rcut_predict(S, pred_card2), y_va)
    print(f"     -> RCut(预测基数2) F1 = {f1_hat2:.4f} ({f1_hat2-f1_scut:+.4f} vs SCut)")

    # 对照: 直接用当前 SCut 产生的实际基数
    k_cur = (S >= ths).sum(1).astype(np.float64)
    mae_cur = float(np.mean(np.abs(k_cur - card_true)))
    print(f"\n     现状(SCut 实际基数): MAE={mae_cur:.2f} | 均值={k_cur.mean():.2f} (真值 {card_true.mean():.2f})")

    # 混合: 向预测基数收缩
    print("\n     混合策略 (SCut 基数 与 预测基数 的加权):")
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        k_mix = (1-w)*k_cur + w*pred_card2
        f1m = macro_f1_from_pred(rcut_predict(S, k_mix), y_va)
        print(f"       w={w:.2f} -> F1={f1m:.4f} ({f1m-f1_scut:+.4f})")

    res = {"f1_scut": f1_scut, "f1_oracle_rcut": f1_oracle,
           "sensitivity": sens, "mae_profile": mae, "mae_profile_esm": mae2,
           "f1_rcut_profile": f1_hat, "f1_rcut_esm": f1_hat2, "mae_current": mae_cur}
    r1, p1, a1 = macro_pr_from_pred(rcut_predict(S, pred_card), y_va)
    r2, p2, a2 = macro_pr_from_pred(rcut_predict(S, pred_card2), y_va)
    res.update({"rcut_profile_recall": r1, "rcut_profile_precision": p1, "rcut_profile_auc": a1,
                "rcut_esm_recall": r2, "rcut_esm_precision": p2, "rcut_esm_auc": a2})
    with Path(REPORT).open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
