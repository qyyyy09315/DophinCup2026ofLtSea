"""
C011a-gensub — 用 C011 α=0.50 接到 I019 生成提交（不重训）
================================================================
C011 已关（OOF +0.0006 未过 +0.001），但在已关实验里最接近门槛，
且从未交过平台。I019 8.25 已交 45.7201。本脚本只推理 C005 test
概率，与 C002b 平均后重建 I019，写出提交。编号 C011a，不改 C011 行。
用法:
  python code/C/C011a-gensub.py
输出:
  rel/C005_test_probs.npz
  submit/submission_C011a_ensemble.csv
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ALPHA = 0.50
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
CNN_W = 0.2


def main():
    t0 = time.time()
    print("=" * 60)
    print("C011a: C002b×C005 α=0.50 接到 I019 出提交")

    c002b = load_mod("c002b", os.path.join("C", "C002b-cnn.py"))
    c005m = load_mod("c005", os.path.join("C", "C005-csra.py"))
    i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
    i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(i011.DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]

    test_cache = os.path.join(i011.REL_DIR, "C005_test_probs.npz")
    if os.path.exists(test_cache):
        c005_test = np.load(test_cache)["test"]
        print(f"  复用 {test_cache}")
    else:
        print("  推理 C005 test ...")
        model = c005m.MultiScaleCNNCSRA(n_labels=len(label_cols)).to(c002b.DEVICE)
        c002b.load_checkpoint(model, c005m.MODEL_PATH)
        test_ds = c002b.ProteinDataset(
            test_df["sequence"].values,
            np.zeros((len(test_df), len(label_cols)), dtype=np.int8),
            c002b.MAX_LEN)
        test_loader = DataLoader(
            test_ds, batch_size=c002b.BATCH_SIZE, shuffle=False,
            collate_fn=c002b.collate_onehot, pin_memory=True, num_workers=0)
        c005_test, _ = c002b.get_probs(model, test_loader)
        np.savez_compressed(test_cache, test=c005_test.astype(np.float32))
        print(f"  写入 {test_cache}")

    cnn_val = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_val = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))["val"]
    mix_val = (1.0 - ALPHA) * cnn_val + ALPHA * c005_val
    cnn_te = np.load(os.path.join(i011.REL_DIR, "TC003_test_probs.npz"))["cnn"]
    mix_te = (1.0 - ALPHA) * cnn_te + ALPHA * c005_test

    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    sim_v = i011.ALIGN_W * i011.agg_labels(idx_v, bits_v, y_train) + (
        1.0 - i011.ALIGN_W) * i011.build_kmer_val(y_train)
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep_v = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * mix_val
    s011_v = deep_v + i011.SIM_W * sim_v + BETA_RRF * i011.col_norm(
        i011.rrf_pair(deep_v, sim_v, K_RRF))
    n = s011_v.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t_i011 = i011.search_t(s011_v, y_val)
    row_m = np.maximum((s011_v >= t_i011).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011_v - s011_v.max(axis=1, keepdims=True)) / TAU_SK)
    p_v = i019.sinkhorn(ker, row_m, col_m)
    mixed_v = i019.mix(s011_v, p_v, ALPHA_SK)
    t_sub = i011.search_t(mixed_v, y_val)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    rec = i011.eval_ens(mixed_v, y_val, hit, miss, "c011a")
    print(f"  val F1={rec['f1_100']:.4f} miss={rec['f1_miss']:.4f} "
          f"(C011 报告 0.4453)")

    tsv_te = os.path.join(i011.CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
    full_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
    idx_t, bits_t = i011.parse_aln(tsv_te, test_df["protein_id"].values, full_map)
    sim_bit_t = i011.agg_labels(idx_t, bits_t, y_full)
    sim_kmer_t = 0.0
    for kk, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        d = np.load(os.path.join(
            i011.REL_DIR, "TCK002a_neigh_full.npz" if kk == 6
            else f"CK004_neigh_k{kk}_full.npz"))
        n2 = d["idx_test"].shape[0]
        lab = y_full[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
        s = d["sim_test"][:, :5].astype(np.float32)
        ww = np.maximum(s, 1e-7) ** 2
        sim_kmer_t = sim_kmer_t + wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    sim_t = i011.ALIGN_W * sim_bit_t + (1.0 - i011.ALIGN_W) * sim_kmer_t
    t2t = np.load(os.path.join(i011.REL_DIR, "T002_test_probs.npz"))
    deep_t = (w["t001a"] * t1["t001a_test"] + w["t002"] * t2t["t002"]
              + w["cnn"] * mix_te)
    s011_t = deep_t + i011.SIM_W * sim_t + BETA_RRF * i011.col_norm(
        i011.rrf_pair(deep_t, sim_t, K_RRF))
    n_t = s011_t.shape[0]
    row_t = np.maximum((s011_t >= t_i011).sum(1).astype(np.float64), 1.0)
    col_t = np.maximum(pi * n_t * RHO, 1e-6)
    col_t *= row_t.sum() / col_t.sum()
    ker_t = np.exp((s011_t - s011_t.max(axis=1, keepdims=True)) / TAU_SK)
    p_t = i019.sinkhorn(ker_t, row_t, col_t)
    mixed_t = i019.mix(s011_t, p_t, ALPHA_SK)
    y_pred = (mixed_t >= t_sub).astype(np.int8)

    os.makedirs(i011.SUB_DIR, exist_ok=True)
    sub = pd.concat([test_df[["protein_id"]],
                     pd.DataFrame(y_pred, columns=label_cols)], axis=1)
    sub_path = os.path.join(i011.SUB_DIR, "submission_C011a_ensemble.csv")
    sub.to_csv(sub_path, index=False)
    print(f"  提交: {sub_path} ({len(sub)} 行)")
    print(f"  用时 {time.time()-t0:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
