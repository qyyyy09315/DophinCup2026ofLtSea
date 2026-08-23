"""
CK005e-mmseqs — MMseqs2 比对邻居接到 CK005a（新检索引擎）
================================================================
依据:
  - MMseqs2: Steinegger & Söding, Nat Biotechnol 2017, doi:10.1038/nbt.3988
    k-mer 预过滤 + 无空位延伸，与 DIAMOND 双索引种子不同，远程同源命中集合
    不必相同。
  - Zhang & Freddolino, bioRxiv 2023, doi:10.1101/2023.11.14.567021
    默认参数下 MMseqs2/BLASTp 功能转移优于未调参 DIAMOND；CK005a/d 已把
    DIAMOND 调到 very-sensitive，从未换引擎。
  - DeepGOPlus DiamondScore: Kulmanov & Hoehndorf, Bioinformatics 2019,
    doi:10.1093/bioinformatics/btz595（bitscore 加权聚合，本脚本沿用）
方案:
  复用 CK005 fasta。val→train（防泄漏）跑 MMseqs2 easy-search，e=0.001，
  输出 query/target/bits/evalue/pident，解析与 CK005a 相同。
  网格: -s 5.7 / 7.5
  接入: replace（整路换 MMseqs）/ fill（只补 DIAMOND miss）/ merge（按 bits 并 top-5）
  四路权重不变。主指标全局 100 档 Macro F1，门槛较 CK005a +0.001。
用法:
  python code/CK/CK005e-mmseqs.py
  python code/CK/CK005e-mmseqs.py --skip-search
  python code/CK/CK005e-mmseqs.py --gen-sub
输出:
  rel/CK005e_report.json
  rel/ck005e_tmp/mmseqs_val_s{sens}.tsv
  submit/submission_CK005e_ensemble.csv（仅 --gen-sub 且有效）
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    "ck005a", os.path.join(PROJECT_ROOT, "code", "CK", "CK005a-diamond-sensitive.py"))
a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a)

DATA_DIR = a.DATA_DIR
REL_DIR = a.REL_DIR
SUB_DIR = a.SUB_DIR
CK005_TMP = a.CK005_TMP
TMP_DIR = os.path.join(REL_DIR, "ck005e_tmp")
REPORT_PATH = os.path.join(REL_DIR, "CK005e_report.json")
TRAIN_PATH = a.TRAIN_PATH
TEST_PATH = a.TEST_PATH
TCK002_CONFIG_PATH = a.TCK002_CONFIG_PATH
MMSEQS_BIN = os.path.join(PROJECT_ROOT, "tools", "mmseqs2", "mmseqs", "bin")
MMSEQS = os.path.join(MMSEQS_BIN, "mmseqs.exe")
BUSYBOX = os.path.join(MMSEQS_BIN, "busybox.exe")
DIAMOND_VAL = os.path.join(a.TMP_DIR, "align_val_very-sensitive_e0.001.tsv")
DIAMOND_TEST = os.path.join(a.TMP_DIR, "align_test_very-sensitive_e0.001.tsv")

MAX_ACCEPT = a.MAX_ACCEPT
TOP_K = a.TOP_K
ALIGN_W = a.ALIGN_W
SIM_W = a.SIM_W
VAL_SIZE = a.VAL_SIZE
RANDOM_STATE = a.RANDOM_STATE
EVALUE = 0.001
SENS_LIST = [5.7, 7.5]
MODES = ("replace", "fill", "merge")


def tsv_path(sens, split="val"):
    tag = str(sens).replace(".", "p")
    return os.path.join(TMP_DIR, f"mmseqs_{split}_s{tag}.tsv")


def ensure_busybox():
    bash = os.path.join(MMSEQS_BIN, "bash")
    if os.path.exists(bash) or os.path.exists(bash + ".exe"):
        return
    if not os.path.exists(BUSYBOX):
        raise FileNotFoundError(BUSYBOX)
    subprocess.run([BUSYBOX, "--install", "-s", MMSEQS_BIN],
                   capture_output=True, text=True)


def run_cmd(args, desc):
    print(f"  {desc}...", flush=True)
    r = subprocess.run(args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print((r.stdout or "")[-3000:])
        print((r.stderr or "")[-3000:])
        raise RuntimeError(f"mmseqs 失败: {args[1] if len(args) > 1 else args}")
    return r


def createdb(fasta, db_prefix):
    if os.path.exists(db_prefix + ".dbtype"):
        return db_prefix
    run_cmd([MMSEQS, "createdb", fasta, db_prefix, "--threads", "8"],
            f"createdb {os.path.basename(fasta)}")
    return db_prefix


def run_mmseqs(query_fa, target_fa, out_tsv, sens, desc, qdb=None, tdb=None):
    ensure_busybox()
    work = os.path.join(TMP_DIR, f"work_{os.path.basename(out_tsv)}")
    if os.path.isdir(work):
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    qdb = qdb or createdb(query_fa, os.path.join(TMP_DIR, f"qdb_{os.path.basename(out_tsv)}"))
    tdb = tdb or createdb(target_fa, os.path.join(TMP_DIR, "train_db" if "train" in target_fa else "full_db"))
    res = os.path.join(work, "result")
    search_tmp = os.path.join(work, "search_tmp")
    run_cmd(
        [MMSEQS, "search", qdb, tdb, res, search_tmp,
         "-e", str(EVALUE), "-s", str(sens),
         "--max-accept", str(MAX_ACCEPT), "--threads", "8",
         "--remove-tmp-files", "1"],
        desc,
    )
    run_cmd(
        [MMSEQS, "convertalis", qdb, tdb, res, out_tsv,
         "--format-output", "query,target,bits,evalue,pident",
         "--threads", "8"],
        f"convertalis s={sens}",
    )
    shutil.rmtree(work, ignore_errors=True)


def merge_hits(idx_a, bits_a, idx_b, bits_b):
    n, k = idx_a.shape
    out_idx = np.full((n, k), -1, np.int64)
    out_bits = np.zeros((n, k), np.float32)
    for i in tqdm(range(n), desc="    merge 邻居", unit="q"):
        seen = {}
        for src_idx, src_bits in ((idx_a[i], bits_a[i]), (idx_b[i], bits_b[i])):
            for j, t in enumerate(src_idx):
                if t < 0:
                    continue
                bs = float(src_bits[j])
                if t not in seen or bs > seen[t]:
                    seen[t] = bs
        if not seen:
            continue
        ranked = sorted(seen.items(), key=lambda x: -x[1])[:k]
        for j, (t, bs) in enumerate(ranked):
            out_idx[i, j] = t
            out_bits[i, j] = bs
    return out_idx, out_bits


def fill_miss(idx_d, bits_d, idx_m, bits_m):
    hit = idx_d[:, 0] >= 0
    idx = idx_d.copy()
    bits = bits_d.copy()
    miss = ~hit
    idx[miss] = idx_m[miss]
    bits[miss] = bits_m[miss]
    return idx, bits


def eval_hits(idx_v, bits_v, y_train, y_val, deep, sim_kmer):
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
    rec = {
        "hit_rate": round(float(hit.mean()), 6),
        "n_hit": int(hit.sum()),
        "n_miss": int(miss.sum()),
        "f1_100": round(f1_all, 6),
        "recall_100": round(r_all, 6),
        "precision_100": round(p_all, 6),
        "auc": round(auc_all, 6),
        "f1_hit": round(f1_hit, 6),
        "f1_miss": round(f1_miss, 6),
    }
    return rec, ens, best_t


def main():
    skip_search = "--skip-search" in sys.argv
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    os.makedirs(TMP_DIR, exist_ok=True)
    print("=" * 60)
    print("CK005e: MMseqs2 替换 / 补 miss / 合并（e=0.001）")

    if not os.path.exists(MMSEQS):
        raise FileNotFoundError(MMSEQS)
    if not os.path.exists(DIAMOND_VAL):
        raise FileNotFoundError("缺少 CK005a very-sensitive val tsv")

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
    fa_train = os.path.join(CK005_TMP, "train.fasta")
    if not os.path.exists(fa_val) or not os.path.exists(fa_train):
        raise FileNotFoundError("缺少 CK005 fasta")

    print("[1/4] kmer + deep + DIAMOND 基线...")
    sim_kmer = a.build_kmer_val(idx_val, y_train)
    vp = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(TCK002_CONFIG_PATH, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]

    idx_d, bits_d = a.parse_aln(DIAMOND_VAL, val_ids, train_db_map, EVALUE)
    baseline, _, _ = eval_hits(idx_d, bits_d, y_train, y_val, deep, sim_kmer)
    baseline["sens"] = "diamond-very-sensitive"
    baseline["mode"] = "diamond"
    print(f"  diamond: hit={baseline['hit_rate']:.3f} F1={baseline['f1_100']:.4f} "
          f"missF1={baseline['f1_miss']:.4f}", flush=True)

    print("[2/4] MMseqs2 搜索...")
    val_db = os.path.join(TMP_DIR, "val_db")
    train_db = os.path.join(TMP_DIR, "train_db")
    if not skip_search:
        createdb(fa_val, val_db)
        createdb(fa_train, train_db)
    rows = [baseline]
    for sens in SENS_LIST:
        out = tsv_path(sens, "val")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            tsv = out
        elif skip_search:
            print(f"  skip s={sens}（无缓存）")
            continue
        else:
            run_mmseqs(fa_val, fa_train, out, sens, f"search s={sens}",
                       qdb=val_db, tdb=train_db)
            tsv = out
        idx_m, bits_m = a.parse_aln(tsv, val_ids, train_db_map, EVALUE)
        packed = {
            "replace": (idx_m, bits_m),
            "fill": fill_miss(idx_d, bits_d, idx_m, bits_m),
            "merge": merge_hits(idx_d, bits_d, idx_m, bits_m),
        }
        for mode in MODES:
            rec, _, _ = eval_hits(*packed[mode], y_train, y_val, deep, sim_kmer)
            rec["sens"] = sens
            rec["mode"] = mode
            rec["tsv"] = tsv
            rows.append(rec)
            print(f"  s={sens} {mode}: hit={rec['hit_rate']:.3f} F1={rec['f1_100']:.4f} "
                  f"hitF1={rec['f1_hit']:.4f} missF1={rec['f1_miss']:.4f}", flush=True)

    cand = [r for r in rows if r.get("mode") != "diamond"]
    if not cand:
        raise RuntimeError("没有 MMseqs 结果")
    best = max(cand, key=lambda r: (r["f1_100"], r["f1_miss"]))
    gain_all = best["f1_100"] - baseline["f1_100"]
    gain_miss = best["f1_miss"] - baseline["f1_miss"]
    useful = bool(gain_all >= 0.001)
    print(f"  最优 s={best['sens']} {best['mode']} F1={best['f1_100']:.4f} "
          f"Δ={gain_all:+.4f} Δmiss={gain_miss:+.4f} → {'有效' if useful else '无增益'}")

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
        print("[3/4] test 端最优配置 + 提交...")
        fa_test = os.path.join(CK005_TMP, "test.fasta")
        fa_full = os.path.join(CK005_TMP, "full.fasta")
        if not os.path.exists(DIAMOND_TEST):
            raise FileNotFoundError("缺少 CK005a test tsv")
        out_test = tsv_path(best["sens"], "test")
        if not os.path.exists(out_test):
            test_db = createdb(fa_test, os.path.join(TMP_DIR, "test_db"))
            full_db = createdb(fa_full, os.path.join(TMP_DIR, "full_db"))
            run_mmseqs(fa_test, fa_full, out_test, best["sens"],
                       f"search test s={best['sens']}", qdb=test_db, tdb=full_db)
        full_db_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
        idx_td, bits_td = a.parse_aln(DIAMOND_TEST, test_df["protein_id"].values, full_db_map, EVALUE)
        idx_tm, bits_tm = a.parse_aln(out_test, test_df["protein_id"].values, full_db_map, EVALUE)
        if best["mode"] == "replace":
            idx_t, bits_t = idx_tm, bits_tm
        elif best["mode"] == "fill":
            idx_t, bits_t = fill_miss(idx_td, bits_td, idx_tm, bits_tm)
        else:
            idx_t, bits_t = merge_hits(idx_td, bits_td, idx_tm, bits_tm)
        sim_bit_test = a.agg_labels(idx_t, bits_t, y_full, TOP_K)
        sim_kmer_test = a.build_kmer_test(y_full)
        sim_test = ALIGN_W * sim_bit_test + (1.0 - ALIGN_W) * sim_kmer_test
        tp = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))
        t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
        t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))
        ens_test = (w["t001a"] * t1t["t001a_test"] + w["t002"] * t2t["t002"]
                    + w["cnn"] * tp["cnn"] + SIM_W * sim_test)
        packed_val = {
            "replace": a.parse_aln(best["tsv"], val_ids, train_db_map, EVALUE),
            "fill": fill_miss(idx_d, bits_d, *a.parse_aln(best["tsv"], val_ids, train_db_map, EVALUE)),
            "merge": merge_hits(idx_d, bits_d, *a.parse_aln(best["tsv"], val_ids, train_db_map, EVALUE)),
        }
        _, _, best_t = eval_hits(*packed_val[best["mode"]], y_train, y_val, deep, sim_kmer)
        y_pred = (ens_test >= best_t[None, :]).astype(np.int8)
        os.makedirs(SUB_DIR, exist_ok=True)
        sub = pd.concat([test_df[["protein_id"]],
                         pd.DataFrame(y_pred, columns=label_cols)], axis=1)
        sub_path = os.path.join(SUB_DIR, "submission_CK005e_ensemble.csv")
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
