"""
CK005a-diamond-sensitive — DIAMOND 敏感度 / e-value 网格（打无同源覆盖）
================================================================
依据:
  Zhang & Freddolino, bioRxiv 2023, doi:10.1101/2023.11.14.567021
  默认参数下 BLASTp/MMseqs2 功能转移优于 DIAMOND；DIAMOND 调参后可追上。
  CK005 用 DeepGOPlus 默认 -e 0.001、未开 --sensitive。
方案:
  复用 CK005 fasta / dmnd；只重跑 val→train blastp。
  网格: sensitivity ∈ {default, sensitive, very-sensitive}
        × e ∈ {0.001, 0.01, 0.1}
  已有 rel/ck005_tmp/align_val.tsv（default, e=0.001）与
  align_val_e{0.01,0.1,1.0}.tsv 可 --skip-search 复用对应档。
  主指标: miss 覆盖率 + miss 子集 Macro F1（100档阈值），不是只看全局 F1。
  有效（miss F1 升且全局不降）→ 重跑 test 并生成提交。
用法:
  python code/CK/CK005a-diamond-sensitive.py
  python code/CK/CK005a-diamond-sensitive.py --skip-search   # 只评已有 tsv
  python code/CK/CK005a-diamond-sensitive.py --gen-sub
输出:
  rel/CK005a_report.json
  rel/ck005a_tmp/align_val_*.tsv
  submit/submission_CK005a_ensemble.csv（仅 --gen-sub 且有效）
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
DIAMOND = os.path.join(PROJECT_ROOT, "tools", "diamond", "diamond.exe")
CK005_TMP = os.path.join(REL_DIR, "ck005_tmp")
TMP_DIR = os.path.join(REL_DIR, "ck005a_tmp")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
REPORT_PATH = os.path.join(REL_DIR, "CK005a_report.json")
TCK002_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

MAX_ACCEPT = 5
TOP_K = 3
ALIGN_W = 0.4
SIM_W = 0.75
VAL_SIZE = 0.2
RANDOM_STATE = 42
N_LABELS = 500

SENS_FLAGS = {
    "default": [],
    "sensitive": ["--sensitive"],
    "very-sensitive": ["--very-sensitive"],
}
EVALUES = [0.001, 0.01, 0.1]


def run_diamond(args, desc):
    print(f"  {desc}...", flush=True)
    r = subprocess.run([DIAMOND] + args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print((r.stdout or "")[-2000:])
        print((r.stderr or "")[-2000:])
        raise RuntimeError(f"diamond 失败: {args[:3]}")


def parse_aln(tsv_path, query_ids, id_map, evalue):
    idx = np.full((len(query_ids), MAX_ACCEPT), -1, dtype=np.int64)
    bits = np.zeros((len(query_ids), MAX_ACCEPT), dtype=np.float32)
    qpos = {q: i for i, q in enumerate(query_ids)}
    if not os.path.exists(tsv_path):
        return idx, bits
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            q, t = parts[0], parts[1]
            if q not in qpos or t not in id_map:
                continue
            i = qpos[q]
            try:
                bs, ev = float(parts[2]), float(parts[3])
            except ValueError:
                continue
            if ev > evalue:
                continue
            j = 0
            while j < MAX_ACCEPT and bits[i, j] > bs:
                j += 1
            if j < MAX_ACCEPT:
                idx[i, j:] = np.roll(idx[i, j:], 1)
                bits[i, j:] = np.roll(bits[i, j:], 1)
                idx[i, j] = id_map[t]
                bits[i, j] = bs
    return idx, bits


def agg_labels(idx, w, y_space, top_k):
    n = idx.shape[0]
    valid = idx[:, :top_k] >= 0
    lab = np.zeros((n, top_k, y_space.shape[1]), dtype=np.float32)
    for j in range(top_k):
        lab[:, j] = y_space[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(w[:, :top_k], 1e-7)
    ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(axis=1) / (ww.sum(axis=1)[:, None] + 1e-7)


def macro_f1_fixed(ens, y_true, t=0.5):
    pred = ens >= t
    f1s = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum())
        fp = int((p & ~yt).sum())
        fn = int((~p & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def macro_rp(ens, y_true, t):
    pred = ens >= t
    rs, ps = [], []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum())
        fp = int((p & ~yt).sum())
        fn = int((~p & yt).sum())
        rs.append(tp / (tp + fn) if tp + fn else 0.0)
        ps.append(tp / (tp + fp) if tp + fp else 0.0)
    return float(np.mean(rs)), float(np.mean(ps))


def macro_auc(y_true, y_prob):
    aucs = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, y_prob[:, i]))
    return float(np.mean(aucs))


def search_thresholds(ens, y_val):
    cands = np.linspace(0.01, 0.99, 100)
    best_t = np.full(N_LABELS, 0.5)
    for i in range(N_LABELS):
        yt = y_val[:, i]
        if yt.sum() == 0:
            continue
        p = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (p & ytm).sum(0)
        fp = (p & ~ytm).sum(0)
        fn = (~p & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best_t[i] = cands[f1t.argmax()]
    return best_t


def eval_mask(ens, y, t, mask):
    return macro_f1_fixed(ens[mask], y[mask], t)


def existing_tsv(sens, ev):
    if sens == "default" and abs(ev - 0.001) < 1e-12:
        p = os.path.join(CK005_TMP, "align_val.tsv")
        if os.path.exists(p):
            return p
    if sens == "default":
        tag = f"{ev:g}"
        p = os.path.join(CK005_TMP, f"align_val_e{tag}.tsv")
        if os.path.exists(p):
            return p
    p = os.path.join(TMP_DIR, f"align_val_{sens}_e{ev:g}.tsv")
    if os.path.exists(p):
        return p
    return None


def build_kmer_val(idx_val, y_train):
    k_range = [3, 4, 5, 6, 7, 8]
    fusion = [0, 0.1, 0.2, 0.4, 0.2, 0.1]
    sim_f = 0.0
    n = len(idx_val)
    for k, wt in zip(k_range, fusion):
        if wt <= 0:
            continue
        prefix = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(REL_DIR, f"{prefix}_neigh_k{k}.npz"))
        lab = y_train[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        s = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(s, 1e-7) ** 2
        sim_f += wt * (lab * ww[:, :, None]).sum(axis=1) / ww.sum(axis=1)[:, None]
    return sim_f


def build_kmer_test(y_full):
    k_range = [3, 4, 5, 6, 7, 8]
    fusion = [0, 0.1, 0.2, 0.4, 0.2, 0.1]
    sim_f = 0.0
    for k, wt in zip(k_range, fusion):
        if wt <= 0:
            continue
        d = np.load(os.path.join(REL_DIR, "TCK002a_neigh_full.npz" if k == 6
                                 else f"CK004_neigh_k{k}_full.npz"))
        n2 = d["idx_test"].shape[0]
        lab = y_full[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
        s = d["sim_test"][:, :5].astype(np.float32)
        ww = np.maximum(s, 1e-7) ** 2
        sim_f += wt * (lab * ww[:, :, None]).sum(axis=1) / ww.sum(axis=1)[:, None]
    return sim_f


def main():
    skip_search = "--skip-search" in sys.argv
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    os.makedirs(TMP_DIR, exist_ok=True)
    print("=" * 60)
    print("CK005a: DIAMOND sensitivity × e-value 网格（盯 miss 覆盖 / miss F1）")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_db_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    fa_val = os.path.join(CK005_TMP, "val.fasta")
    db_train = os.path.join(CK005_TMP, "train.dmnd")
    if not os.path.exists(fa_val) or not os.path.exists(db_train):
        raise FileNotFoundError("缺少 CK005 fasta/dmnd，先跑 code/CK/CK005-alignment-neighbor.py")

    print("[1/4] 构建 kmer 多k + 深度模型概率...")
    sim_kmer = build_kmer_val(idx_val, y_train)
    vp = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(TCK002_CONFIG_PATH, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]

    print("[2/4] DIAMOND 网格...")
    rows = []
    baseline = None
    for sens, flags in SENS_FLAGS.items():
        for ev in EVALUES:
            tag = f"{sens}_e{ev:g}"
            tsv = existing_tsv(sens, ev)
            need = tsv is None
            if need and skip_search:
                print(f"  skip {tag}（无缓存）")
                continue
            if need:
                out = os.path.join(TMP_DIR, f"align_val_{tag}.tsv")
                args = ["blastp", "-d", db_train, "-q", fa_val, "-o", out,
                        "--max-target-seqs", str(MAX_ACCEPT), "-e", str(ev),
                        "--outfmt", "6", "qseqid", "sseqid", "bitscore", "evalue", "pident",
                        "--threads", "8"] + flags
                try:
                    run_diamond(args, f"blastp {tag}")
                    tsv = out
                except RuntimeError as e:
                    print(f"  {tag} 失败（可能不支持该 flag）: {e}")
                    continue
            idx_v, bits_v = parse_aln(tsv, val_ids, train_db_map, ev)
            hit = idx_v[:, 0] >= 0
            miss = ~hit
            sim_bit = agg_labels(idx_v, bits_v, y_train, TOP_K)
            sim = ALIGN_W * sim_bit + (1 - ALIGN_W) * sim_kmer
            ens = deep + SIM_W * sim
            best_t = search_thresholds(ens, y_val)
            f1_all = macro_f1_fixed(ens, y_val, best_t)
            r_all, p_all = macro_rp(ens, y_val, best_t)
            auc_all = macro_auc(y_val, ens)
            f1_hit = eval_mask(ens, y_val, best_t, hit) if hit.any() else 0.0
            f1_miss = eval_mask(ens, y_val, best_t, miss) if miss.any() else 0.0
            rec = {
                "sens": sens, "evalue": ev, "tsv": tsv,
                "hit_rate": round(float(hit.mean()), 6),
                "n_hit": int(hit.sum()), "n_miss": int(miss.sum()),
                "f1_100": round(f1_all, 6),
                "recall_100": round(r_all, 6),
                "precision_100": round(p_all, 6),
                "auc": round(auc_all, 6),
                "f1_hit": round(f1_hit, 6),
                "f1_miss": round(f1_miss, 6),
            }
            rows.append(rec)
            print(f"  {tag}: hit={hit.mean():.3f} F1={f1_all:.4f} "
                  f"hitF1={f1_hit:.4f} missF1={f1_miss:.4f}", flush=True)
            if sens == "default" and abs(ev - 0.001) < 1e-12:
                baseline = rec

    if not rows:
        raise RuntimeError("网格为空")
    if baseline is None:
        baseline = rows[0]

    def key(r):
        return (r["f1_miss"], r["f1_100"], r["hit_rate"])

    best = max(rows, key=key)
    print("[3/4] 对照 CK005 default/e=0.001:")
    print(f"  基线 hit={baseline['hit_rate']:.3f} F1={baseline['f1_100']:.4f} missF1={baseline['f1_miss']:.4f}")
    print(f"  最优 {best['sens']} e={best['evalue']}: "
          f"hit={best['hit_rate']:.3f} F1={best['f1_100']:.4f} missF1={best['f1_miss']:.4f}")
    gain_miss = best["f1_miss"] - baseline["f1_miss"]
    gain_all = best["f1_100"] - baseline["f1_100"]
    useful = (gain_miss >= 0.005 and gain_all >= -0.001)
    print(f"  ΔmissF1={gain_miss:+.4f} ΔF1={gain_all:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline": baseline, "best": best, "grid": rows,
        "gain_miss_f1": round(gain_miss, 6),
        "gain_all_f1": round(gain_all, 6),
        "useful": useful,
    }

    if not (gen_sub and useful):
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  报告: {REPORT_PATH}")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    print("[4/4] test 端重跑最优配置 + 提交...")
    fa_test = os.path.join(CK005_TMP, "test.fasta")
    db_full = os.path.join(CK005_TMP, "full.dmnd")
    sens, ev = best["sens"], best["evalue"]
    flags = SENS_FLAGS[sens]
    out_test = os.path.join(TMP_DIR, f"align_test_{sens}_e{ev:g}.tsv")
    if not os.path.exists(out_test):
        args = ["blastp", "-d", db_full, "-q", fa_test, "-o", out_test,
                "--max-target-seqs", str(MAX_ACCEPT), "-e", str(ev),
                "--outfmt", "6", "qseqid", "sseqid", "bitscore", "evalue", "pident",
                "--threads", "8"] + flags
        run_diamond(args, f"blastp test {sens} e={ev}")
    full_db_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
    idx_t, bits_t = parse_aln(out_test, test_df["protein_id"].values, full_db_map, ev)
    sim_bit_test = agg_labels(idx_t, bits_t, y_full, TOP_K)
    sim_kmer_test = build_kmer_test(y_full)
    sim_test = ALIGN_W * sim_bit_test + (1 - ALIGN_W) * sim_kmer_test
    tp = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))
    t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))
    ens_test = (w["t001a"] * t1t["t001a_test"] + w["t002"] * t2t["t002"]
                + w["cnn"] * tp["cnn"] + SIM_W * sim_test)

    idx_v, bits_v = parse_aln(best["tsv"], val_ids, train_db_map, ev)
    sim_bit_val = agg_labels(idx_v, bits_v, y_train, TOP_K)
    ens_val = deep + SIM_W * (ALIGN_W * sim_bit_val + (1 - ALIGN_W) * sim_kmer)
    best_t = search_thresholds(ens_val, y_val)
    y_pred = (ens_test >= best_t[None, :]).astype(np.int8)
    os.makedirs(SUB_DIR, exist_ok=True)
    sub = pd.concat([test_df[["protein_id"]],
                     pd.DataFrame(y_pred, columns=label_cols)], axis=1)
    sub_path = os.path.join(SUB_DIR, "submission_CK005a_ensemble.csv")
    sub.to_csv(sub_path, index=False)
    report["submission"] = sub_path
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  提交: {sub_path} ({len(sub)} 行)")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
