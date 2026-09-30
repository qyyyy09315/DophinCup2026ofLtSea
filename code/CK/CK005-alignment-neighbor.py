"""
CK005-alignment-neighbor — DIAMOND 序列比对邻居特征
====================================================
依据 (DeepGOPlus 全文深读, bioRxiv 10.1101/615260):
  - DeepGOPlus 序列相似性组件是 DIAMOND (BLAST 家族) 比对检索:
    e-value≤0.001 过滤, bitscore 归一化加权聚合 (DiamondScore)
  - 本项目 CK 系列一直用 k-mer hashing 近似 (composition-based),
    未实现比对组件; 比对捕捉真同源 (进化), 与 k-mer 组成相似度互补
方案:
  - DIAMOND (2.1.11, tools/diamond) makedb + blastp
    (val 查询→train 库 91,036 防泄漏; test 查询→全量库 113,796)
    --max-target-seqs 5, -e 0.001 (DeepGOPlus 同款)
  - 聚合: bitscore 加权 top-3 邻居标签 (DiamondScore 式),
    对照 pident 加权
  - 评估: sim_align 独立 F1(0.5)/AUC、corr(sim_align, kmer 多k融合)、
    融合网格 → 进 CK004a 集成 (w_sim 0.7)
文献:
  - DeepGOPlus: Kulmanov & Hoehndorf, Bioinformatics 2019,
    doi:10.1093/bioinformatics/btz595; 方法详见 bioRxiv 10.1101/615260
  - DIAMOND: Buchfink et al., Nat Methods 2015, doi:10.1038/nmeth.3176
输出:
  rel/CK005_report.json
  rel/CK005_align_{val,test}.tsv (DIAMOND 原始输出)
  submit/submission_CK005_ensemble.csv (若融合有效)
"""

import json
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
DIAMOND = os.path.join(PROJECT_ROOT, "tools", "diamond", "diamond.exe")
TMP_DIR = os.path.join(REL_DIR, "ck005_tmp")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
REPORT_PATH = f"{REL_DIR}/CK005_report.json"
SUB_PATH = f"{SUB_DIR}/submission_CK005_ensemble.csv"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
T001A_PROBS_PATH = f"{REL_DIR}/TCK001_t001a_probs.npz"
T002_VAL_PROBS_PATH = f"{REL_DIR}/T002_val_probs.npz"
T002_TEST_PROBS_PATH = f"{REL_DIR}/T002_test_probs.npz"
TCK002_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

EVALUE = 0.001
MAX_ACCEPT = 5
TOP_K = 3
VAL_SIZE = 0.2
RANDOM_STATE = 42
# 最终参数 (val 联合扫描 2026-08-16): w_align=0.4, w_sim=0.75 → F1=0.4359
ALIGN_W = 0.4
SIM_W = 0.75


def run_diamond(args, desc):
    print(f"  {desc}...")
    r = subprocess.run([DIAMOND] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"diamond 失败: {args[:2]}")


def write_fasta(df, path, seq_col="sequence", id_col="protein_id"):
    with Path(path).open("w") as f:
        for pid, seq in zip(df[id_col].values, df[seq_col].values):
            f.write(f">{pid}\n{seq}\n")


def parse_aln(tsv_path, query_ids, id_map):
    """解析 DIAMOND blastp outfmt6 输出 → (idx, bitscore, pident)。
    id_map: target protein_id → 标签空间行号。"""
    idx = np.full((len(query_ids), MAX_ACCEPT), -1, dtype=np.int64)
    bits = np.zeros((len(query_ids), MAX_ACCEPT), dtype=np.float32)
    pident = np.zeros((len(query_ids), MAX_ACCEPT), dtype=np.float32)
    qpos = {q: i for i, q in enumerate(query_ids)}
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:  # qseqid,sseqid,bitscore,evalue,pident
                continue
            q, t = parts[0], parts[1]
            if q not in qpos or t not in id_map:
                continue
            i = qpos[q]
            try:
                bs, ev, pid = float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            if ev > EVALUE:
                continue
            # 插入排序维护 top-MAX_ACCEPT (按 bitscore)
            j = 0
            while j < MAX_ACCEPT and bits[i, j] > bs:
                j += 1
            if j < MAX_ACCEPT:
                idx[i, j:] = np.roll(idx[i, j:], 1)
                bits[i, j:] = np.roll(bits[i, j:], 1)
                pident[i, j:] = np.roll(pident[i, j:], 1)
                idx[i, j] = id_map[t]
                bits[i, j] = bs
                pident[i, j] = pid
    return idx, bits, pident


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
    pred = (ens >= t)
    f1s = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def macro_auc(y_true, y_prob):
    aucs = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, y_prob[:, i]))
    return float(np.mean(aucs))


def main():
    skip_search = "--skip-search" in sys.argv
    t0 = time.time()
    os.makedirs(TMP_DIR, exist_ok=True)
    print("=" * 60)
    print(f"DIAMOND: {DIAMOND}")

    print("[1/7] 读取数据, 写 fasta...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)),
                                          test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]

    fa_train = f"{TMP_DIR}/train.fasta"
    fa_val = f"{TMP_DIR}/val.fasta"
    fa_test = f"{TMP_DIR}/test.fasta"
    fa_full = f"{TMP_DIR}/full.fasta"
    write_fasta(train_df.iloc[idx_train], fa_train)
    write_fasta(train_df.iloc[idx_val], fa_val)
    write_fasta(test_df, fa_test)
    write_fasta(train_df, fa_full)
    print(f"  fasta: train {len(idx_train)} / val {len(idx_val)} / test {len(test_df)} / full {len(train_df)}")

    if not skip_search:
        print("[2/7] DIAMOND makedb + blastp (val→train, test→full)...")
        run_diamond(["makedb", "--in", fa_train, "-d", f"{TMP_DIR}/train.dmnd"], "makedb train")
        run_diamond(["makedb", "--in", fa_full, "-d", f"{TMP_DIR}/full.dmnd"], "makedb full")
        fmt = "qseqid,sseqid,bitscore,evalue,pident"
        for q, db, out in [("val", "train.dmnd", "align_val"),
                           ("test", "full.dmnd", "align_test")]:
            run_diamond(["blastp", "-d", f"{TMP_DIR}/{db}", "-q", f"{TMP_DIR}/{q}.fasta",
                         "-o", f"{TMP_DIR}/{out}.tsv", "--max-target-seqs", str(MAX_ACCEPT),
                         "-e", str(EVALUE), "--outfmt", "6",
                         "qseqid", "sseqid", "bitscore", "evalue", "pident",
                         "--threads", "8"], f"blastp {q}")

    print("[3/7] 解析比对结果 + 聚合...")
    val_ids = train_df["protein_id"].values[idx_val]
    test_ids = test_df["protein_id"].values
    full_ids = train_df["protein_id"].values
    # DIAMOND 输出 sseqid = protein_id → 标签空间行号映射
    train_db_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    full_db_map = {pid: i for i, pid in enumerate(full_ids)}
    idx_v, bits_v, pid_v = parse_aln(f"{TMP_DIR}/align_val.tsv", val_ids, train_db_map)
    print(f"  val 命中: {np.count_nonzero(idx_v >= 0)}/{idx_v.size}")

    # 映射 mmseqs target 序号 → train 行: createdb 按 fasta 顺序编号 (0-based)
    # train_db 顺序 = idx_train 顺序
    sim_bit_val = agg_labels(idx_v, bits_v, y_train, TOP_K)
    sim_pid_val = agg_labels(idx_v, pid_v, y_train, TOP_K)
    f1_bit = macro_f1_fixed(sim_bit_val, y_val)
    f1_pid = macro_f1_fixed(sim_pid_val, y_val)
    auc_bit = macro_auc(y_val, sim_bit_val)
    print(f"  sim_align(bitscore): F1(0.5)={f1_bit:.4f} AUC={auc_bit:.4f}")
    print(f"  sim_align(pident):   F1(0.5)={f1_pid:.4f}")

    print("[4/7] 与 kmer 多 k 融合 sim 对比...")
    K_RANGE = [3, 4, 5, 6, 7, 8]
    FUSION = [0, 0.1, 0.2, 0.4, 0.2, 0.1]

    def build_kmer(top_k=5, mode="sim2"):
        sim_f = 0.0
        for k, wt in zip(K_RANGE, FUSION):
            if wt <= 0:
                continue
            prefix = "CK003" if k >= 7 else "CK002"
            d = np.load(f"{REL_DIR}/{prefix}_neigh_k{k}.npz")
            n = len(idx_val)
            lab = y_train[d["idx_val"][:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
            s = d["sim_val"][:, :top_k].astype(np.float32)
            ww = np.maximum(s, 1e-7) ** 2 if mode == "sim2" else np.maximum(s, 1e-7)
            sim_f += wt * (lab * ww[:, :, None]).sum(axis=1) / ww.sum(axis=1)[:, None]
        return sim_f

    sim_kmer = build_kmer()
    f1_kmer = macro_f1_fixed(sim_kmer, y_val)
    corr_bit = float(np.corrcoef(sim_bit_val.ravel(), sim_kmer.ravel())[0, 1])
    print(f"  kmer 多k融合: F1(0.5)={f1_kmer:.4f} | corr(bitscore, kmer)={corr_bit:.4f}")

    fusion = {}
    for wa in [0.0, 0.2, 0.5, 0.8, 1.0]:
        sim_f = wa * sim_bit_val + (1 - wa) * sim_kmer
        fusion[wa] = round(macro_f1_fixed(sim_f, y_val), 6)
    print(f"  融合网格 (w_align): {fusion}")
    best_wa = max(fusion, key=fusion.get)
    print(f"  最优 w_align={best_wa} (F1={fusion[best_wa]:.4f}, kmer alone={f1_kmer:.4f})")

    report = {"sim_align_bitscore": {"f1_05": round(f1_bit, 6), "auc": round(auc_bit, 6)},
              "sim_align_pident": {"f1_05": round(f1_pid, 6)},
              "kmer_multik": {"f1_05": round(f1_kmer, 6)},
              "corr_align_kmer": corr_bit, "fusion_w_align": fusion,
              "best_w_align": best_wa, "evalue": EVALUE, "max_accept": MAX_ACCEPT}

    if fusion[best_wa] <= f1_kmer + 0.001:
        report["submission"] = "none"
        with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  报告: {REPORT_PATH}")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    print("[5/7] test 端聚合 + 集成...")
    idx_t, bits_t, pid_t = parse_aln(f"{TMP_DIR}/align_test.tsv", test_ids, full_db_map)
    sim_bit_test = agg_labels(idx_t, bits_t, y_full, TOP_K)

    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, cnn_test = vp["cnn"], tp["cnn"]
    t1 = np.load(T001A_PROBS_PATH)
    t001a_val, t001a_test = t1["t001a_val"], t1["t001a_test"]
    t2v = np.load(T002_VAL_PROBS_PATH)
    t2t = np.load(T002_TEST_PROBS_PATH)
    t002_val, t002_test = t2v["t002"], t2t["t002"]
    with open(TCK002_CONFIG_PATH, "r", encoding="utf-8") as f:
        w = json.load(f)["best_w"]

    # test 端 kmer 多 k 融合 (全量缓存)
    sim_kmer_test = 0.0
    for k, wt in zip(K_RANGE, FUSION):
        if wt <= 0:
            continue
        d = np.load(f"{REL_DIR}/TCK002a_neigh_full.npz" if k == 6
                    else f"{REL_DIR}/CK004_neigh_k{k}_full.npz")
        n2 = d["idx_test"].shape[0]
        lab = y_full[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
        s = d["sim_test"][:, :5].astype(np.float32)
        ww = np.maximum(s, 1e-7) ** 2
        sim_kmer_test += wt * (lab * ww[:, :, None]).sum(axis=1) / ww.sum(axis=1)[:, None]

    sim_val = ALIGN_W * sim_bit_val + (1 - ALIGN_W) * sim_kmer
    sim_test = ALIGN_W * sim_bit_test + (1 - ALIGN_W) * sim_kmer_test
    ens_val = (w["t001a"] * t001a_val + w["t002"] * t002_val
               + w["cnn"] * cnn_val + SIM_W * sim_val)
    ens_test = (w["t001a"] * t001a_test + w["t002"] * t002_test
                + w["cnn"] * cnn_test + 0.7 * sim_test)

    print("[6/7] 100 档阈值 + 提交...")
    cands = np.linspace(0.01, 0.99, 100)
    best_t = np.full(500, 0.5)
    for i in tqdm(range(500), desc="    阈值搜索", unit="label"):
        yt = y_val[:, i]
        if yt.sum() == 0:
            continue
        p = ens_val[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (p & ytm).sum(0); fp = (p & ~ytm).sum(0); fn = (~p & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best_t[i] = cands[f1t.argmax()]
    y_pred = np.zeros_like(ens_test, dtype=np.int8)
    for i, t in enumerate(best_t):
        y_pred[:, i] = (ens_test[:, i] >= t).astype(np.int8)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    report["submission"] = SUB_PATH
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    print("[7/7] 报告...")
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
