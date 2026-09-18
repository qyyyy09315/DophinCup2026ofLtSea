"""
I067 — 概率提交的尺寸压缩 (≤90MB) 且尽量保序
=============================================
约束: 平台上传上限 90MB。当前 4 位小数(带前导0) 的校准概率为 94MB，超限。
手段(保序性从强到弱):
  A. 去掉前导 0: "0.1234" -> ".1234"，每格省 1 字符 (~14MB)，**零信息损失**
  B. 降位数: 4 位 -> 3 位 -> 2 位，会引入并列，损 AUC
本脚本在验证集上量化各方案的宏平均 AUC 损失，选择满足 ≤90MB 且损失最小的方案。
输出: submit/submission_I067_prob_compact.csv + 桌面副本 + rel/I067_report.json
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
REPORT = os.path.join(REL_DIR, "I067_report.json")
N_LABELS = 500
W_PLM = 0.35
MARGIN = 0.002
SIZE_LIMIT_MB = 90.0


# ---------------- 复用 I066c 的管线 ----------------
from importlib.machinery import SourceFileLoader
_i66c = SourceFileLoader("i66c", os.path.join(PROJECT_ROOT, "code", "I", "I066c-linear-calibration.py")).load_module()


def main():
    t0 = time.time()
    print("=" * 74)
    print("  I067: Compact Probability Submission (size <= 90MB)")
    print("=" * 74)

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

    parse_aln = _i66c.parse_aln
    agg = _i66c.agg
    t2h = _i66c.transitive_2hop
    build_kmers = _i66c.build_kmers
    ranks = _i66c.ranks
    col_norm = _i66c.col_norm
    sinkhorn = _i66c.sinkhorn
    search_scut = _i66c.search_scut
    macro_f1 = _i66c.macro_f1
    calibrate_linear = _i66c.calibrate_linear

    # ---------- 验证集: 量化各精度 AUC ----------
    print("\n[1/3] Validation: quantifying AUC loss at each precision...")
    cnn_v = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1v = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2v = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    deep4_v = wc["t001a"] * t1v + wc["t002"] * t2v + wc["cnn"] * (0.5 * cnn_v + 0.5 * c005_v)
    multi_v = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    deep5_v = (1 - W_PLM) * deep4_v + W_PLM * multi_v

    idx_v, bits_v = parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                              val_pids, train_map)
    hit_v = idx_v[:, 0] >= 0
    miss_v = ~hit_v
    sim1_v = agg(idx_v, bits_v, y_tr)
    v2v = {}
    with open(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            q, t = p[0], p[1]
            if q == t or q not in val_map or t not in val_map:
                continue
            qi, ti = val_map[q], val_map[t]
            v2v.setdefault(qi, [])
            if len(v2v[qi]) < 5:
                v2v[qi].append((ti, float(p[2])))
    sim_v = 0.4 * t2h(sim1_v, hit_v, miss_v, v2v, n_va) + 0.6 * build_kmers(y_tr, "val")

    s_v = deep5_v + 0.75 * sim_v + 0.50 * col_norm(1.0/(30+ranks(deep5_v)+1) + 1.0/(30+ranks(sim_v)+1))
    t_tmp = search_scut(s_v, y_va)
    row_m = np.maximum((s_v >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_tr.mean(0) * n_va, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    p_v = sinkhorn(np.exp((s_v - s_v.max(axis=1, keepdims=True)) / 0.20), row_m, col_m)
    mixed_v = 0.40 * s_v + 0.60 * p_v
    ths = search_scut(mixed_v, y_va)

    auc_ref = roc_auc_score(y_va, mixed_v, average="macro")
    cal_v = calibrate_linear(mixed_v, ths)
    print(f"  连续校准 (未舍入) AUC = {auc_ref:.6f}")

    variants = {}
    for dec in [4, 3, 2]:
        r = np.round(cal_v, dec)
        auc = roc_auc_score(y_va, r, average="macro")
        f1_half = macro_f1(r, y_va, np.full(N_LABELS, 0.5))
        # 估算文件大小: 每格字符数 = 1(前导0省略) + 1(.) + dec + 1(逗号)
        chars = (1 + 1 + dec + 1) if dec > 0 else 2
        est_mb = n_te * N_LABELS * chars / 1024 / 1024
        variants[dec] = {"auc": auc, "auc_loss": auc_ref - auc, "f1_at_0.5": f1_half,
                         "est_mb": est_mb}
        print(f"  {dec} 位小数(省略前导0): AUC={auc:.6f} (损失 {auc_ref-auc:+.6f}) | "
              f"F1@0.5={f1_half:.4f} | 估算 {est_mb:.1f}MB")

    # 选择: 满足 ≤ 90MB 且损失最小的
    feasible = {d: v for d, v in variants.items() if v["est_mb"] <= SIZE_LIMIT_MB}
    if not feasible:
        print("\n  !! 2 位小数仍超限，需改用二值提交")
        chosen = None
    else:
        chosen = max(feasible.keys())   # 位数最多者 = 损失最小
        print(f"\n  选定: {chosen} 位小数 (省略前导0), AUC 损失 {variants[chosen]['auc_loss']:+.6f}, "
              f"估算 {variants[chosen]['est_mb']:.1f}MB")

    # ---------- 测试集 ----------
    print("\n[2/3] Test: building calibrated matrix...")
    c005_t = np.load(os.path.join(REL_DIR, "C005_test_probs.npz"))["test"]
    cnn_t = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
    t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_test"]
    t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
    deep4_t = wc["t001a"] * t1t + wc["t002"] * t2t + wc["cnn"] * (0.5 * cnn_t + 0.5 * c005_t)
    multi_t = np.load(os.path.join(REL_DIR, "plm1664_multiseed_test.npz"))["test"]
    deep5_t = (1 - W_PLM) * deep4_t + W_PLM * multi_t

    idx_t, bits_t = parse_aln(os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv"),
                              test_df["protein_id"].values, full_map)
    hit_t = idx_t[:, 0] >= 0
    miss_t = ~hit_t
    sim1_t = agg(idx_t, bits_t, y_full)
    test_map = {p: i for i, p in enumerate(test_df["protein_id"].values)}
    te2te = {}
    with open(os.path.join(CK005A_TMP, "align_test_to_test_e0.001.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            q, t = p[0], p[1]
            if q == t or q not in test_map or t not in test_map:
                continue
            qi, ti = test_map[q], test_map[t]
            te2te.setdefault(qi, [])
            if len(te2te[qi]) < 5:
                te2te[qi].append((ti, float(p[2])))
    sim_t = 0.4 * t2h(sim1_t, hit_t, miss_t, te2te, n_te) + 0.6 * build_kmers(y_full, "test")

    s_t = deep5_t + 0.75 * sim_t + 0.50 * col_norm(1.0/(30+ranks(deep5_t)+1) + 1.0/(30+ranks(sim_t)+1))
    row_t = np.maximum((s_t >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_t = np.maximum(y_full.mean(0) * n_te, 1e-6)
    col_t *= row_t.sum() / col_t.sum()
    p_t = sinkhorn(np.exp((s_t - s_t.max(axis=1, keepdims=True)) / 0.20), row_t, col_t)
    mixed_t = 0.40 * s_t + 0.60 * p_t
    cal_t = calibrate_linear(mixed_t, ths)

    # ---------- 写出紧凑格式 ----------
    print("\n[3/3] Writing compact submission...")
    if chosen is not None:
        arr = np.round(cal_t, chosen)
        # 手工拼接: 省略前导 0, 减少字符数
        fmt = "{:." + str(chosen) + "f}"
        lines = []
        lines.append("protein_id," + ",".join(label_cols))
        pids = test_df["protein_id"].values
        for i in range(arr.shape[0]):
            row = arr[i]
            cells = []
            for v in row:
                s = fmt.format(v)
                if s.startswith("0."):
                    s = s[1:]          # "0.1234" -> ".1234"
                elif s.startswith("-0."):
                    s = "-" + s[2:]
                cells.append(s)
            lines.append(str(pids[i]) + "," + ",".join(cells))
        out = os.path.join(SUB_DIR, "submission_I067_prob_compact.csv")
        with open(out, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(lines) + "\n")
        size_mb = os.path.getsize(out) / 1024 / 1024
        print(f"  Written: {out}")
        print(f"  Actual size: {size_mb:.1f} MB (limit {SIZE_LIMIT_MB:.0f} MB) -> "
              f"{'OK' if size_mb <= SIZE_LIMIT_MB else 'STILL OVER!'}")

        # 校验可被 pandas 正确解析
        chk = pd.read_csv(out, nrows=3)
        print(f"  Parse check: shape={chk.shape}, first row sample="
              f"{chk.iloc[0, 1:4].tolist()}")

        if size_mb <= SIZE_LIMIT_MB:
            dst = r"C:\Users\YKSHb\Desktop\submission_I067_prob_compact.csv"
            with open(dst, "w", encoding="utf-8", newline="") as f:
                f.write("\n".join(lines) + "\n")
            print(f"  Copied to Desktop: submission_I067_prob_compact.csv")
            rep = {"decimals": chosen, "auc_loss": float(variants[chosen]["auc_loss"]),
                   "f1_at_0.5": float(variants[chosen]["f1_at_0.5"]),
                   "size_mb": float(size_mb), "auc_ref": float(auc_ref),
                   "variants": {str(k): v for k, v in variants.items()}}
        else:
            rep = {"decimals": chosen, "size_mb": float(size_mb), "over_limit": True}
    else:
        rep = {"decimals": None, "note": "2位小数仍超限，改用二值"}

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
