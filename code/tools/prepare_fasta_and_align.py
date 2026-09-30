import os, sys, time, subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
os.makedirs(CK005A_TMP, exist_ok=True)

DIAMOND_EXE = os.path.join(PROJECT_ROOT, "tools", "diamond", "diamond.exe")

print("1. Exporting fasta files from train.csv and test.csv...")
t0 = time.time()
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# 导出 train.fasta 和 test.fasta
train_fasta = os.path.join(DATA_DIR, "train.fasta")
test_fasta = os.path.join(DATA_DIR, "test.fasta")

if not os.path.exists(train_fasta):
    with Path(train_fasta).open("w") as f:
        for pid, seq in zip(train_df["protein_id"], train_df["sequence"]):
            f.write(f">{pid}\n{seq}\n")
    print(f"  Wrote {train_fasta} ({len(train_df)} sequences)")
else:
    print(f"  {train_fasta} already exists.")

if not os.path.exists(test_fasta):
    with Path(test_fasta).open("w") as f:
        for pid, seq in zip(test_df["protein_id"], test_df["sequence"]):
            f.write(f">{pid}\n{seq}\n")
    print(f"  Wrote {test_fasta} ({len(test_df)} sequences)")
else:
    print(f"  {test_fasta} already exists.")

# 导出 8:2 验证集切分 fasta
idx_train, idx_val = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)

tr_split_fasta = os.path.join(DATA_DIR, "train_split.fasta")
va_split_fasta = os.path.join(DATA_DIR, "val_split.fasta")

if not os.path.exists(tr_split_fasta):
    with Path(tr_split_fasta).open("w") as f:
        for pid, seq in zip(train_df["protein_id"].values[idx_train], train_df["sequence"].values[idx_train]):
            f.write(f">{pid}\n{seq}\n")
    print(f"  Wrote {tr_split_fasta} ({len(idx_train)} sequences)")

if not os.path.exists(va_split_fasta):
    with Path(va_split_fasta).open("w") as f:
        for pid, seq in zip(train_df["protein_id"].values[idx_val], train_df["sequence"].values[idx_val]):
            f.write(f">{pid}\n{seq}\n")
    print(f"  Wrote {va_split_fasta} ({len(idx_val)} sequences)")

print("2. Running DIAMOND self-alignments...")

def run_diamond_self(fasta_path, out_tsv, db_name):
    dmnd_db = os.path.join(CK005A_TMP, f"{db_name}.dmnd")
    if not os.path.exists(dmnd_db):
        print(f"  Makedb: {dmnd_db} ...")
        cmd_db = [DIAMOND_EXE, "makedb", "--in", fasta_path, "-d", dmnd_db, "--quiet"]
        subprocess.run(cmd_db, check=True)
    
    if not os.path.exists(out_tsv):
        print(f"  Aligning: {fasta_path} against {dmnd_db} -> {out_tsv} ...")
        # 输出格式 6: qseqid sseqid bitscore evalue pident length
        cmd_aln = [DIAMOND_EXE, "blastp", "-q", fasta_path, "-d", dmnd_db, "-o", out_tsv,
                   "--very-sensitive", "-e", "0.001", "--outfmt",
                   "6 qseqid sseqid bitscore evalue pident length",
                   "--max-target-seqs", "10", "--threads", "8", "--quiet"]
        subprocess.run(cmd_aln, check=True)
        print(f"  Finished: {out_tsv} ({os.path.getsize(out_tsv)/1024/1024:.2f} MB)")
    else:
        print(f"  {out_tsv} already exists.")

# 1) 验证集内部自比对 (val-to-val)
run_diamond_self(va_split_fasta, os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv"), "val_split")

# 2) 测试集内部自比对 (test-to-test)
run_diamond_self(test_fasta, os.path.join(CK005A_TMP, "align_test_to_test_e0.001.tsv"), "test_full")

print(f"All fasta and DIAMOND alignments completed in {time.time()-t0:.1f}s")
PY