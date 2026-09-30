"""
I069 — 边界保持型低位量化 (≤50MB) ：F1 不降、AUC 提升
======================================================
问题: 2 位小数校准概率为 54.5MB，超 50MB 上限。
      普通 1 位小数四舍五入会把 0.49 抬到 0.5，产生假阳性 -> F1@0.5 从 0.4475 掉到 0.4411。
方案: **边界保持量化 (Boundary-Preserving Quantization)**
      对每个标签，按该标签分数在其 [lo, t) 与 [t, hi] 两个区间内分别等频分箱:
        s <  t_j  ->  量化到 {0.05, 0.15, 0.25, 0.35, 0.45}  (5 级, 全部 < 0.5)
        s >= t_j  ->  量化到 {0.55, 0.65, 0.75, 0.85, 0.95}  (5 级, 全部 >= 0.5)
      性质: ① 0.5 两侧严格不混 -> 平台若按 0.5 卡阈值，决策与二值完全一致(F1 不降)
            ② 保留 10 级排序信息 -> AUC 优于二值 (10 级 vs 2 级)
            ③ 每格 3 字符(".5")，估算 ~41MB，满足 50MB
输出: submit/submission_I069_bp_1dec.csv (+桌面), rel/I069_report.json
"""
import os, sys, json, time, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
REPORT = os.path.join(REL_DIR, "I069_report.json")
N_LABELS = 500
W_PLM = 0.35
SIZE_LIMIT_MB = 50.0
N_BINS = 4      # 低侧级数 (0.1..0.4); 高侧为 N_BINS+1 (0.5..0.9)

from importlib.machinery import SourceFileLoader
from pathlib import Path
i66c = SourceFileLoader("i66c", os.path.join(PROJECT_ROOT, "code", "I", "I066c-linear-calibration.py")).load_module()


def bp_quantize(S, ths, n_bins=N_BINS):
    """边界保持量化: 保留 0.5 决策边界，两侧各 n_bins 级。"""
    out = np.empty_like(S, dtype=np.float64)
    # 低侧代表值 0.05..0.45 ; 高侧 0.55..0.95
    # 1 位小数级别: 低侧 0.1..0.4 (4 级), 高侧 0.5..0.9 (5 级) -> 每格仅 2 字符
    lo_vals = np.arange(1, n_bins + 1, dtype=np.float64) / 10.0            # 0.1,0.2,0.3,0.4
    hi_vals = 0.5 + np.arange(n_bins, dtype=np.float64) / 10.0             # 0.5,0.6,0.7,0.8,0.9
    for j in range(S.shape[1]):
        col = S[:, j].astype(np.float64)
        t = float(ths[j])
        low = col < t
        # 低侧按分位数分箱
        if low.sum() > 0:
            qs = np.quantile(col[low], np.linspace(0, 1, n_bins + 1)[1:-1])
            bins = np.searchsorted(qs, col[low], side="right")
            out[low, j] = lo_vals[np.clip(bins, 0, n_bins - 1)]
        # 高侧按分位数分箱
        if (~low).sum() > 0:
            qs2 = np.quantile(col[~low], np.linspace(0, 1, n_bins + 1)[1:-1])
            bins2 = np.searchsorted(qs2, col[~low], side="right")
            out[~low, j] = hi_vals[np.clip(bins2, 0, n_bins - 1)]
    return out


def fmt_bp(v):
    """给定量化值(1位小数, 如 0.05/0.15/.../0.95) -> 字符串(省略前导0)"""
    s = f"{v:.2f}"          # 0.05 / 0.15 ...
    # 保留 2 位语义但只在需要时保留第二位；统一用 2 位避免歧义
    return s[1:] if s.startswith("0.") else s


def build_matrices():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va, n_te = len(idx_va), len(test_df)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}
    full_map = {p: i for i, p in enumerate(train_df["protein_id"].values)}
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
    mixed_v = 0.40*s_v + 0.60*p_v
    ths = i66c.search_scut(mixed_v, y_va)
    cal_v = i66c.calibrate_linear(mixed_v, ths)

    c005_t = np.load(os.path.join(REL_DIR, "C005_test_probs.npz"))["test"]
    cnn_t = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
    t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_test"]
    t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
    deep4_t = wc["t001a"]*t1t + wc["t002"]*t2t + wc["cnn"]*(0.5*cnn_t + 0.5*c005_t)
    deep5_t = (1-W_PLM)*deep4_t + W_PLM*np.load(os.path.join(REL_DIR, "plm1664_multiseed_test.npz"))["test"]
    idx_t, bits_t = i66c.parse_aln(os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv"),
                                   test_df["protein_id"].values, full_map)
    hit_t = idx_t[:, 0] >= 0
    sim1_t = i66c.agg(idx_t, bits_t, y_full)
    test_map = {p: i for i, p in enumerate(test_df["protein_id"].values)}
    te2te = {}
    with open(os.path.join(CK005A_TMP, "align_test_to_test_e0.001.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            q, t = p[0], p[1]
            if q == t or q not in test_map or t not in test_map: continue
            qi, ti = test_map[q], test_map[t]
            te2te.setdefault(qi, [])
            if len(te2te[qi]) < 5: te2te[qi].append((ti, float(p[2])))
    sim_t = 0.4*i66c.transitive_2hop(sim1_t, hit_t, ~hit_t, te2te, n_te) + 0.6*i66c.build_kmers(y_full, "test")
    s_t = deep5_t + 0.75*sim_t + 0.50*i66c.col_norm(
        1.0/(30+i66c.ranks(deep5_t)+1) + 1.0/(30+i66c.ranks(sim_t)+1))
    row_t = np.maximum((s_t >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_t = np.maximum(y_full.mean(0)*n_te, 1e-6); col_t *= row_t.sum()/col_t.sum()
    p_t = i66c.sinkhorn(np.exp((s_t - s_t.max(axis=1, keepdims=True))/0.20), row_t, col_t)
    mixed_t = 0.40*s_t + 0.60*p_t
    cal_t = i66c.calibrate_linear(mixed_t, ths)
    return cal_v, y_va, cal_t, test_df, label_cols, ths


def main():
    t0 = time.time()
    print("=" * 74)
    print("  I069: Boundary-Preserving Low-Bit Quantization (<= 50MB)")
    print("=" * 74)

    print("\n[1/3] Building calibrated matrices...")
    cal_v, y_va, cal_t, test_df, label_cols, ths = build_matrices()
    auc_ref = roc_auc_score(y_va, cal_v, average="macro")
    f1_ref = i66c.macro_f1(cal_v, y_va, np.full(N_LABELS, 0.5))
    print(f"  连续 4 位参考: AUC={auc_ref:.6f} | F1@0.5={f1_ref:.4f}")

    print("\n[2/3] Comparing quantizers on validation...")
    variants = {}
    for n_bins in [4]:
        q = bp_quantize(cal_v, np.full(N_LABELS, 0.5), n_bins)
        auc = roc_auc_score(y_va, q, average="macro")
        f1 = i66c.macro_f1(q, y_va, np.full(N_LABELS, 0.5))
        # 决策一致性: 与二值(基于 0.5)是否完全一致
        same = int(((q >= 0.5) == (cal_v >= 0.5)).all())
        variants[n_bins] = {"auc": auc, "f1": f1, "decision_identical": bool(same)}
        print(f"  低{n_bins}级/高{n_bins+1}级 ({2*n_bins+1} 级总) : AUC={auc:.6f} (损失 {auc_ref-auc:+.6f}) | "
              f"F1@0.5={f1:.4f} (损失 {f1_ref-f1:+.4f}) | 决策与二值一致={bool(same)}")

    print("\n[3/3] Writing test file...")
    chosen_bins = 4
    q_t = bp_quantize(cal_t, np.full(N_LABELS, 0.5), chosen_bins)
    arr = np.round(q_t, 1)
    pids = test_df["protein_id"].values
    lines = ["protein_id," + ",".join(label_cols)]
    for i in range(arr.shape[0]):
        cells = []
        for v in arr[i]:
            s = f"{v:.1f}"
            cells.append(s[1:] if s.startswith("0.") else s)
        lines.append(str(pids[i]) + "," + ",".join(cells))
    out = os.path.join(SUB_DIR, "submission_I069_bp_quant.csv")
    with Path(out).open("w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")
    mb = os.path.getsize(out) / 1024 / 1024
    print(f"  {out}")
    print(f"  Size: {mb:.1f} MB -> {'✅ 合规' if mb <= SIZE_LIMIT_MB else '❌ 超限'}")

    chk = pd.read_csv(out, nrows=3)
    print(f"  解析校验: shape={chk.shape}, 样本={chk.iloc[0,1:4].tolist()}")

    if mb <= SIZE_LIMIT_MB:
        shutil.copy(out, r"C:\Users\YKSHb\Desktop\submission_I069_bp_quant.csv")
        print(f"  已复制到桌面: submission_I069_bp_quant.csv")

    rep = {"auc_ref": auc_ref, "f1_ref": f1_ref, "variants": {str(k): v for k, v in variants.items()},
           "chosen_bins": chosen_bins, "size_mb": mb}
    with Path(REPORT).open("w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
