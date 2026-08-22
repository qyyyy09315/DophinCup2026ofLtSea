"""
CK005k-ultra-sens — DIAMOND more/ultra-sensitive 与 iterate
================================================================
依据:
  Zhang & Freddolino, bioRxiv 2023, doi:10.1101/2023.11.14.567021
  DIAMOND 调参后可追上 BLASTp。CK005a 只扫了
  default / sensitive / very-sensitive × e∈{0.001,0.01,0.1}，
  最优 very-sensitive e=0.001（平台 45.3039）。
  2.1.11 还有 --more-sensitive / --ultra-sensitive / --iterate，
  改的是命中集合，不是融合权重（CK005g 已关）。
方案:
  复用 CK005 fasta/dmnd 与 CK005a 评估。e 固定 0.001（CK005a 放宽 e 有害）。
  网格: more-sensitive / ultra-sensitive / iterate
  基线: 已有 very-sensitive e=0.001 tsv，不重搜。
  主指标: 全局 100 档 Macro F1，门槛较 CK005a +0.001。
用法:
  python code/CK/CK005k-ultra-sens.py
  python code/CK/CK005k-ultra-sens.py --skip-search
  python code/CK/CK005k-ultra-sens.py --gen-sub
输出:
  rel/CK005k_report.json
  rel/ck005k_tmp/align_val_*.tsv
  submit/submission_CK005k_ensemble.csv（仅 --gen-sub 且有效）
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    "ck005a", os.path.join(PROJECT_ROOT, "code", "CK", "CK005a-diamond-sensitive.py"))
a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a)

DATA_DIR = a.DATA_DIR
REL_DIR = a.REL_DIR
SUB_DIR = a.SUB_DIR
CK005_TMP = a.CK005_TMP
TMP_DIR = os.path.join(REL_DIR, "ck005k_tmp")
REPORT_PATH = os.path.join(REL_DIR, "CK005k_report.json")
TRAIN_PATH = a.TRAIN_PATH
TEST_PATH = a.TEST_PATH
TCK002_CONFIG_PATH = a.TCK002_CONFIG_PATH

MAX_ACCEPT = a.MAX_ACCEPT
TOP_K = a.TOP_K
ALIGN_W = a.ALIGN_W
SIM_W = a.SIM_W
VAL_SIZE = a.VAL_SIZE
RANDOM_STATE = a.RANDOM_STATE
EVALUE = 0.001

# more-sensitive 介于 sensitive 与 very-sensitive 之间，作对照
SENS_FLAGS = {
    "more-sensitive": ["--more-sensitive"],
    "ultra-sensitive": ["--ultra-sensitive"],
    "iterate": ["--iterate"],
}
BASE_TSV = os.path.join(a.TMP_DIR, "align_val_very-sensitive_e0.001.tsv")


def tsv_path(sens):
    return os.path.join(TMP_DIR, f"align_val_{sens}_e{EVALUE:g}.tsv")


def eval_one(tsv, val_ids, train_db_map, y_train, y_val, deep, sim_kmer):
    idx_v, bits_v = a.parse_aln(tsv, val_ids, train_db_map, EVALUE)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim_bit = a.agg_labels(idx_v, bits_v, y_train, TOP_K)
    sim = ALIGN_W * sim_bit + (1.0 - ALIGN_W) * sim_kmer
    ens = deep + SIM_W * sim
    best_t = a.search_thresholds(ens, y_val)
    f1_all = a.macro_f1_fixed(ens, y_val, best_t)
    r_all, p_all = a.macro_rp(ens, y_val, best_t)
    auc_all = a.macro_auc(y_val, ens)
    f1_hit = a.eval_mask(ens, y_val, best_t, hit) if hit.any() else 0.0
    f1_miss = a.eval_mask(ens, y_val, best_t, miss) if miss.any() else 0.0
    return {
        "tsv": tsv,
        "hit_rate": round(float(hit.mean()), 6),
        "n_hit": int(hit.sum()),
        "n_miss": int(miss.sum()),
        "f1_100": round(f1_all, 6),
        "recall_100": round(r_all, 6),
        "precision_100": round(p_all, 6),
        "auc": round(auc_all, 6),
        "f1_hit": round(f1_hit, 6),
        "f1_miss": round(f1_miss, 6),
    }, ens, best_t


def main():
    skip_search = "--skip-search" in sys.argv
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    os.makedirs(TMP_DIR, exist_ok=True)
    print("=" * 60)
    print("CK005k: DIAMOND more/ultra-sensitive / iterate（e=0.001）")

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
        raise FileNotFoundError("缺少 CK005 fasta/dmnd")
    if not os.path.exists(BASE_TSV):
        raise FileNotFoundError("缺少 CK005a very-sensitive val tsv")

    print("[1/3] kmer + deep...")
    sim_kmer = a.build_kmer_val(idx_val, y_train)
    vp = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(TCK002_CONFIG_PATH, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]

    print("[2/3] 基线 very-sensitive + 新档网格...")
    rec, _, _ = eval_one(BASE_TSV, val_ids, train_db_map, y_train, y_val, deep, sim_kmer)
    rec["sens"] = "very-sensitive"
    rec["evalue"] = EVALUE
    rows = [rec]
    print(f"  very-sensitive e=0.001: hit={rec['hit_rate']:.3f} F1={rec['f1_100']:.4f} "
          f"missF1={rec['f1_miss']:.4f}", flush=True)
    baseline = rec

    for sens, flags in SENS_FLAGS.items():
        tag = f"{sens}_e{EVALUE:g}"
        out = tsv_path(sens)
        if os.path.exists(out):
            tsv = out
        elif skip_search:
            print(f"  skip {tag}（无缓存）")
            continue
        else:
            args = ["blastp", "-d", db_train, "-q", fa_val, "-o", out,
                    "--max-target-seqs", str(MAX_ACCEPT), "-e", str(EVALUE),
                    "--outfmt", "6", "qseqid", "sseqid", "bitscore", "evalue", "pident",
                    "--threads", "8"] + flags
            try:
                a.run_diamond(args, f"blastp {tag}")
                tsv = out
            except RuntimeError as e:
                print(f"  {tag} 失败（可能不支持该 flag）: {e}")
                continue
        rec, _, _ = eval_one(tsv, val_ids, train_db_map, y_train, y_val, deep, sim_kmer)
        rec["sens"] = sens
        rec["evalue"] = EVALUE
        rows.append(rec)
        print(f"  {tag}: hit={rec['hit_rate']:.3f} F1={rec['f1_100']:.4f} "
              f"hitF1={rec['f1_hit']:.4f} missF1={rec['f1_miss']:.4f}", flush=True)

    best = max(rows, key=lambda r: (r["f1_100"], r["f1_miss"]))
    gain_all = best["f1_100"] - baseline["f1_100"]
    gain_miss = best["f1_miss"] - baseline["f1_miss"]
    useful = bool(gain_all >= 0.001)
    print(f"  最优 {best['sens']} F1={best['f1_100']:.4f} Δ={gain_all:+.4f} "
          f"Δmiss={gain_miss:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline": baseline,
        "best": best,
        "grid": rows,
        "gain_all_f1": round(gain_all, 6),
        "gain_miss_f1": round(gain_miss, 6),
        "useful": useful,
        "seconds": round(time.time() - t0, 1),
    }

    if gen_sub and useful:
        print("[3/3] test 端最优配置 + 提交...")
        fa_test = os.path.join(CK005_TMP, "test.fasta")
        db_full = os.path.join(CK005_TMP, "full.dmnd")
        sens = best["sens"]
        flags = [] if sens == "very-sensitive" else SENS_FLAGS[sens]
        if sens == "very-sensitive":
            out_test = os.path.join(a.TMP_DIR, "align_test_very-sensitive_e0.001.tsv")
        else:
            out_test = os.path.join(TMP_DIR, f"align_test_{sens}_e{EVALUE:g}.tsv")
        if not os.path.exists(out_test):
            args = ["blastp", "-d", db_full, "-q", fa_test, "-o", out_test,
                    "--max-target-seqs", str(MAX_ACCEPT), "-e", str(EVALUE),
                    "--outfmt", "6", "qseqid", "sseqid", "bitscore", "evalue", "pident",
                    "--threads", "8"] + flags
            a.run_diamond(args, f"blastp test {sens}")
        full_db_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
        idx_t, bits_t = a.parse_aln(out_test, test_df["protein_id"].values, full_db_map, EVALUE)
        sim_bit_test = a.agg_labels(idx_t, bits_t, y_full, TOP_K)
        sim_kmer_test = a.build_kmer_test(y_full)
        sim_test = ALIGN_W * sim_bit_test + (1.0 - ALIGN_W) * sim_kmer_test
        tp = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))
        t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
        t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))
        ens_test = (w["t001a"] * t1t["t001a_test"] + w["t002"] * t2t["t002"]
                    + w["cnn"] * tp["cnn"] + SIM_W * sim_test)
        rec_v, _, best_t = eval_one(best["tsv"], val_ids, train_db_map,
                                    y_train, y_val, deep, sim_kmer)
        y_pred = (ens_test >= best_t[None, :]).astype(np.int8)
        os.makedirs(SUB_DIR, exist_ok=True)
        sub = pd.concat([test_df[["protein_id"]],
                         pd.DataFrame(y_pred, columns=label_cols)], axis=1)
        sub_path = os.path.join(SUB_DIR, "submission_CK005k_ensemble.csv")
        sub.to_csv(sub_path, index=False)
        report["submission"] = sub_path
        print(f"  提交: {sub_path} ({len(sub)} 行)")
    elif gen_sub:
        print("  --gen-sub 忽略（未过 +0.001）")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
