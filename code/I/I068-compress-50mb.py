"""
I068 — 概率提交压缩至 ≤50MB (精确测量各档实际大小与 AUC 损失)
===============================================================
约束更新: 平台上传上限 **50MB**（原按 90MB 处理，82MB 仍超限）。
待答问题: 在 ≤50MB 前提下，能保留多少 AUC？
候选格式（均省略前导 0，逐标签保序）:
  A. 2 位小数  ".12"    (4 字符/格)
  B. 1 位小数  ".1"     (3 字符/格)
  C. 百分整数  "12"     (3 字符/格, 语义同 A 但无小数点)
  D. 二值 0/1  (2 字符/格, 但丢失排序 -> AUC 回落到 ~0.71)
本脚本对 A/B/C 在验证集上实测宏平均 AUC，并写出实际文件大小，选最优合规档。
"""
import os, sys, json, time
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
REPORT = os.path.join(REL_DIR, "I068_report.json")
N_LABELS = 500
W_PLM = 0.35
SIZE_LIMIT_MB = 50.0

from importlib.machinery import SourceFileLoader
i66c = SourceFileLoader("i66c", os.path.join(PROJECT_ROOT, "code", "I", "I066c-linear-calibration.py")).load_module()


def build_matrices():
    """返回 (mixed_val, ths, mixed_test)"""
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
    auc_ref = roc_auc_score(y_va, cal_v, average="macro")

    # 测试集
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
    return cal_v, y_va, auc_ref, cal_t, test_df, label_cols


def write_compact(arr, pids, label_cols, fmt_fn, path):
    lines = ["protein_id," + ",".join(label_cols)]
    for i in range(arr.shape[0]):
        cells = [fmt_fn(v) for v in arr[i]]
        lines.append(str(pids[i]) + "," + ",".join(cells))
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")
    return os.path.getsize(path) / 1024 / 1024


def fmt_2dec(v):
    s = f"{v:.2f}"
    return s[1:] if s.startswith("0.") else ("0" if s in ("0.00",) else s)


def fmt_1dec(v):
    s = f"{v:.1f}"
    return s[1:] if s.startswith("0.") else s


def fmt_pint(v):
    """百分整数 0-100"""
    return str(int(round(v * 100)))


def main():
    t0 = time.time()
    print("=" * 74)
    print("  I068: Compress probability submission to <= 50MB")
    print("=" * 74)

    print("\n[1/2] Building calibrated matrices...")
    cal_v, y_va, auc_ref, cal_t, test_df, label_cols = build_matrices()
    pids = test_df["protein_id"].values
    print(f"  参考 AUC (连续校准) = {auc_ref:.6f}")

    cands = [
        ("2位小数 .12", fmt_2dec),
        ("1位小数 .1", fmt_1dec),
        ("百分整数 0-100", fmt_pint),
    ]

    print("\n  验证集合成精度对照:")
    print(f"  {'格式':16s} | {'AUC':>10s} | {'AUC损失':>10s} | {'F1@0.5':>8s}")
    print("  " + "-" * 56)
    results = {}
    for name, fn in cands:
        # 在验证集上模拟量化
        q = np.array([[fn(v) for v in row] for row in cal_v[:2000]])  # 子集快速评估
        qf = q.astype(np.float64)
        if fn is fmt_pint:
            qf = qf / 100.0
        auc_q = roc_auc_score(y_va[:2000], qf, average="macro")
        f1q = i66c.macro_f1(qf, y_va[:2000], np.full(N_LABELS, 0.5))
        results[name] = {"auc_sub": auc_q, "f1_at_0.5": f1q}
        print(f"  {name:16s} | {auc_q:10.6f} | {auc_ref-auc_q:+10.6f} | {f1q:8.4f}")

    print("\n[2/2] Writing test files and measuring actual size...")
    print(f"  {'格式':16s} | {'文件大小':>10s} | 合规")
    print("  " + "-" * 40)
    written = {}
    for name, fn in cands:
        safe = {"2位小数 .12": "2dec", "1位小数 .1": "1dec", "百分整数 0-100": "pint"}[name]
        path = os.path.join(SUB_DIR, f"submission_I068_{safe}.csv")
        mb = write_compact(cal_t, pids, label_cols, fn, path)
        ok = mb <= SIZE_LIMIT_MB
        written[name] = {"path": path, "mb": mb, "ok": ok}
        print(f"  {name:16s} | {mb:9.1f}MB | {'✅' if ok else '❌ 超限'}")

    # 选择: 合规且 AUC 损失最小
    ok_c = {n: w for n, w in written.items() if w["ok"]}
    if ok_c:
        # 2位小数精度最高
        best = "2位小数 .12" if "2位小数 .12" in ok_c else list(ok_c.keys())[0]
        print(f"\n  选定: {best} ({written[best]['mb']:.1f}MB)")
        import shutil
        shutil.copy(written[best]["path"],
                    r"C:\Users\YKSHb\Desktop\submission_I068_best.csv")
        print(f"  已复制到桌面: submission_I068_best.csv")
    else:
        best = None
        print("\n  无合规的浮点格式；需退回二值 (28MB)")

    rep = {"auc_ref": auc_ref, "candidates": results, "written": written, "chosen": best}
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
