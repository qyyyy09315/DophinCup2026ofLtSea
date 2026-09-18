"""
I070 — 决策层上限诊断 (标签必须 0/1，故只能在阈值策略上挖空间)
==============================================================
背景:
  - 平台要求标签列必须为 0/1 => 浮点提交作废，只能在决策层提分。
  - 历史线索(I006/CK005e, I006b): oracle RCut 可达 0.52，实务仅 0.43~0.45，缺口 0.07
    与榜首差距(0.4577 -> 0.5286)量级一致，指向"决策层"而非"模型层"。
  - 榜首二值反推: AUC 0.8755 => TPR-FPR=0.751；配合 F1 0.5286 => 其每蛋白预测数远高于我们
    (估 ~50 vs 我们 ~28)，即更激进的高召回工作点。
本诊断在验证集上量化当前分数的四种决策上限:
  A. SCut 逐标签阈值 (现状)
  B. Oracle RCut: 每样本取 top-k_i，k_i = 该样本真实标签数 (上限，不可实现)
  C. Oracle Fmax: 全标签共用一个全局阈值的最优值
  D. 调整工作点: 全局松弛 SCut 阈值以提升召回，观察 F1 曲线
输出: rel/I070_report.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
REPORT = os.path.join(REL_DIR, "I070_report.json")
N_LABELS = 500
W_PLM = 0.35

from importlib.machinery import SourceFileLoader
i66c = SourceFileLoader("i66c", os.path.join(PROJECT_ROOT, "code", "I", "I066c-linear-calibration.py")).load_module()


def macro_f1_from_pred(preds, y):
    preds = preds.astype(bool); yb = y.astype(bool)
    tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    return float(np.mean(np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0.0)))


def recall_fpr(preds, y):
    preds = preds.astype(bool); yb = y.astype(bool)
    tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0); tn = (~preds & ~yb).sum(0)
    rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    fpr = np.where(fp + tn > 0, fp / np.maximum(fp + tn, 1), 0.0)
    return float(rec.mean()), float(fpr.mean()), float(tp.sum()), float(fp.sum())


def main():
    t0 = time.time()
    print("=" * 76)
    print("  I070: Decision-Layer Ceiling Diagnosis (binary-only constraint)")
    print("=" * 76)

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

    auc_cont = roc_auc_score(y_va, S, average="macro")
    print(f"\n  当前分数矩阵: 连续 AUC = {auc_cont:.6f}")

    print("\n" + "=" * 76)
    print(f"{'决策策略':40s} | {'Macro F1':>9s} | {'Recall':>7s} | {'FPR':>7s} | {'每蛋白':>7s}")
    print("=" * 76)

    res = {}
    # A. SCut 现状
    ths = i66c.search_scut(S, y_va)
    predA = (S >= ths)
    f1A = macro_f1_from_pred(predA, y_va)
    rA, fA, tpA, fpA = recall_fpr(predA, y_va)
    print(f"{'A. SCut 逐标签阈值 (现状)':40s} | {f1A:9.4f} | {rA:7.4f} | {fA:7.4f} | {predA.sum(1).mean():7.2f}")
    res['A_SCut'] = {"f1": f1A, "recall": rA, "fpr": fA, "card": float(predA.sum(1).mean())}

    # B. Oracle RCut (上限)
    card_true = y_va.sum(1)
    order = np.argsort(-S, axis=1)
    predB = np.zeros_like(y_va, dtype=bool)
    rows = np.arange(n_va)[:, None]
    mask = np.arange(S.shape[1])[None, :] < card_true[:, None]
    predB[rows, order] = mask
    f1B = macro_f1_from_pred(predB, y_va)
    rB, fB, _, _ = recall_fpr(predB, y_va)
    print(f"{'B. Oracle RCut (真实基数, 不可实现上限)':40s} | {f1B:9.4f} | {rB:7.4f} | {fB:7.4f} | {predB.sum(1).mean():7.2f}")
    res['B_oracle_rcut'] = {"f1": f1B, "recall": rB, "fpr": fB, "card": float(predB.sum(1).mean())}

    # C. Oracle Fmax (全局单阈值最优)
    bestC, bestT = -1, 0
    for th in np.linspace(0.05, 0.95, 91):
        f = macro_f1_from_pred(S >= th, y_va)
        if f > bestC: bestC, bestT = f, th
    predC = (S >= bestT)
    rC, fC, _, _ = recall_fpr(predC, y_va)
    print(f"{'C. Oracle Fmax 全局阈值 (最优 t=%.2f)' % bestT:40s} | {bestC:9.4f} | {rC:7.4f} | {fC:7.4f} | {predC.sum(1).mean():7.2f}")
    res['C_oracle_fmax'] = {"f1": bestC, "t": float(bestT), "recall": rC, "fpr": fC,
                            "card": float(predC.sum(1).mean())}

    # D. 工作点扫描: SCut 阈值全局偏移, 观察 F1 与召回
    print("\n  D. 工作点扫描 (SCut 阈值全局偏移):")
    print(f"     {'偏移':>8s} | {'Macro F1':>9s} | {'Recall':>7s} | {'FPR':>7s} | {'每蛋白':>7s}")
    work = {}
    for delta in [-0.05, -0.02, 0.0, 0.05, 0.10, 0.15, 0.20]:
        th_d = np.clip(ths - delta, 0.01, 0.99)   # delta>0 => 阈值降低 => 召回升高
        predD = (S >= th_d)
        f = macro_f1_from_pred(predD, y_va)
        r, fp_, _, _ = recall_fpr(predD, y_va)
        work[delta] = {"f1": f, "recall": r, "fpr": fp_, "card": float(predD.sum(1).mean())}
        print(f"     {delta:+8.2f} | {f:9.4f} | {r:7.4f} | {fp_:7.4f} | {predD.sum(1).mean():7.2f}")
    res['D_workpoint'] = {str(k): v for k, v in work.items()}

    # E. 按标签频次分层的 SCut 松弛 (长尾类降低阈值)
    print("\n  E. 长尾类定向松弛 (仅对低频标签降阈值):")
    col_counts = y_tr.sum(0)
    print(f"     {'分位阈值':>10s} | {'拓扑标签数':>10s} | {'Macro F1':>9s} | {'Recall':>7s} | {'每蛋白':>7s}")
    tail_res = {}
    for q in [0.3, 0.5, 0.7]:
        cut = np.quantile(col_counts, q)
        tail = col_counts <= cut
        th_e = ths.copy()
        th_e[tail] = np.clip(th_e[tail] - 0.10, 0.01, 0.99)
        predE = (S >= th_e)
        f = macro_f1_from_pred(predE, y_va)
        r, _, _, _ = recall_fpr(predE, y_va)
        tail_res[q] = {"f1": f, "n_tail": int(tail.sum()), "recall": r,
                       "card": float(predE.sum(1).mean())}
        print(f"     {q:10.1f} | {int(tail.sum()):10d} | {f:9.4f} | {r:7.4f} | {predE.sum(1).mean():7.2f}")
    res['E_tail_relax'] = {str(k): v for k, v in tail_res.items()}

    print("\n" + "=" * 76)
    print(f"  决策层上限: Oracle RCut {f1B:.4f} | Oracle Fmax {bestC:.4f} | 现状 SCut {f1A:.4f}")
    print(f"  空间: {f1B - f1A:+.4f} (RCut) / {bestC - f1A:+.4f} (Fmax)")
    print("=" * 76)

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
