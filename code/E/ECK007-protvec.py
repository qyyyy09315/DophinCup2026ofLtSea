"""
ECK007-protvec — ProtVec 分布 3-mer 嵌入接到 CK005a
================================================================
依据:
  - Asgari & Mofrad, PLoS ONE 2015, doi:10.1371/journal.pone.0141287
    蛋白当句子、重叠 3-mer 当词，skip-gram 学 100 维词向量，序列取均值。
    与精确 k-mer 哈希（CK002–CK004）不同：相近 3-mer 可软匹配。
  - E001 PLM 邻居因嵌入坍缩失败；ECK006 PseAAC 组成特征失败。
    ProtVec 是语料共现，不是 PLM、不是比对、不是精确谱核。
方案:
  Word2Vec skip-gram 只在 idx_train 上拟合（防泄漏）。蛋白 = 3-mer 均值 + L2。
  val→train 余弦 top-5，bitscore² 式加权聚合 sim_pv。
  接入: 全局加权 / 只加在 DIAMOND miss。对照 Ridge 线性头。
  四路权重不变。主指标全局 100 档 Macro F1，门槛 +0.001。
用法:
  python code/E/ECK007-protvec.py
  python code/E/ECK007-protvec.py --skip-train   # 复用已训向量
输出:
  rel/ECK007_report.json
  rel/eck007_tmp/protvec.kv
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    "ck005a", os.path.join(PROJECT_ROOT, "code", "CK", "CK005a-diamond-sensitive.py"))
a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a)

DATA_DIR = a.DATA_DIR
REL_DIR = a.REL_DIR
CK005A_TMP = a.TMP_DIR
TMP_DIR = os.path.join(REL_DIR, "eck007_tmp")
REPORT_PATH = os.path.join(REL_DIR, "ECK007_report.json")
KV_PATH = os.path.join(TMP_DIR, "protvec.kv")
EMB_PATH = os.path.join(TMP_DIR, "emb_train_val.npz")
TCK002_CONFIG_PATH = a.TCK002_CONFIG_PATH
DIAMOND_VAL = os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")

ALIGN_W = a.ALIGN_W
SIM_W = a.SIM_W
TOP_K = a.TOP_K
RANDOM_STATE = a.RANDOM_STATE
N_LABELS = a.N_LABELS
KMER = 3
VEC_SIZE = 100
WINDOW = 25
EPOCHS = 5
MIN_COUNT = 2
BLOCK = 500
W_GRID = [0.05, 0.1, 0.2, 0.3]
MISS_W_GRID = [0.2, 0.4, 0.6]


def kmers(seq, k=KMER):
    s = "".join(c if ("A" <= c <= "Z") else "X" for c in seq)
    if len(s) < k:
        return [s] if s else []
    return [s[i:i + k] for i in range(len(s) - k + 1)]


def embed_seqs(seqs, wv):
    dim = wv.vector_size
    out = np.zeros((len(seqs), dim), np.float32)
    for i, seq in enumerate(tqdm(seqs, desc="    ProtVec 均值", unit="seq")):
        vecs = [wv[t] for t in kmers(seq) if t in wv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-8)
    return out / nrm


def neighbors(xq, xt, top=5, block=BLOCK):
    n = xq.shape[0]
    idx = np.zeros((n, top), np.int64)
    sim = np.zeros((n, top), np.float32)
    n_block = (n + block - 1) // block
    for b in tqdm(range(n_block), desc="    邻居", unit="blk"):
        s = b * block
        e = min(n, s + block)
        scores = xq[s:e] @ xt.T
        part = np.argpartition(-scores, top, axis=1)[:, :top]
        for i in range(e - s):
            row = part[i]
            order = np.argsort(-scores[i, row])
            idx[s + i] = row[order]
            sim[s + i] = scores[i, row[order]]
    return idx, sim


def eval_ens(ens, y_val, hit):
    best_t = a.search_thresholds(ens, y_val)
    f1 = a.macro_f1_fixed(ens, y_val, best_t)
    rec, prec = a.macro_rp(ens, y_val, best_t)
    auc = a.macro_auc(y_val, ens)
    miss = ~hit
    return {
        "f1_100": round(f1, 6),
        "recall_100": round(rec, 6),
        "precision_100": round(prec, 6),
        "auc": round(auc, 6),
        "f1_hit": round(a.eval_mask(ens, y_val, best_t, hit) if hit.any() else 0.0, 6),
        "f1_miss": round(a.eval_mask(ens, y_val, best_t, miss) if miss.any() else 0.0, 6),
    }


def main():
    skip_train = "--skip-train" in os.sys.argv
    t0 = time.time()
    os.makedirs(TMP_DIR, exist_ok=True)
    print("=" * 60)
    print("ECK007: ProtVec 3-mer skip-gram 接到 CK005a")

    train_df = pd.read_csv(a.TRAIN_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=a.VAL_SIZE, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    seqs = train_df["sequence"].values
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    print("[1/4] CK005a 四路基线...")
    sim_kmer = a.build_kmer_val(idx_val, y_train)
    idx_v, bits_v = a.parse_aln(DIAMOND_VAL, val_ids, train_map, 0.001)
    hit = idx_v[:, 0] >= 0
    sim_bit = a.agg_labels(idx_v, bits_v, y_train, TOP_K)
    sim = ALIGN_W * sim_bit + (1.0 - ALIGN_W) * sim_kmer
    vp = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(TCK002_CONFIG_PATH, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    ens0 = deep + SIM_W * sim
    baseline = eval_ens(ens0, y_val, hit)
    baseline["mode"] = "ck005a"
    print(f"  CK005a: F1={baseline['f1_100']:.4f} missF1={baseline['f1_miss']:.4f}", flush=True)

    print("[2/4] ProtVec skip-gram ...")
    if skip_train and os.path.exists(KV_PATH) and os.path.exists(EMB_PATH):
        wv = KeyedVectors.load(KV_PATH)
        d = np.load(EMB_PATH)
        x_tr, x_va = d["x_tr"], d["x_va"]
        print(f"  复用 {KV_PATH}")
    else:
        sentences = [kmers(s) for s in tqdm(seqs[idx_train], desc="    切 3-mer", unit="seq")]
        model = Word2Vec(
            sentences=sentences, vector_size=VEC_SIZE, window=WINDOW,
            min_count=MIN_COUNT, sg=1, workers=8, epochs=EPOCHS, seed=RANDOM_STATE)
        wv = model.wv
        wv.save(KV_PATH)
        x_tr = embed_seqs(seqs[idx_train], wv)
        x_va = embed_seqs(seqs[idx_val], wv)
        np.savez_compressed(EMB_PATH, x_tr=x_tr, x_va=x_va)
        print(f"  词表 {len(wv)}  保存 {KV_PATH}")

    print("[3/4] ProtVec 邻居 + Ridge ...")
    idx_pv, sim_pv_raw = neighbors(x_va, x_tr, top=5)
    sim_pv = a.agg_labels(idx_pv, np.maximum(sim_pv_raw, 0) ** 2, y_train, TOP_K)
    corr = float(np.corrcoef(sim_pv.ravel()[::50], sim.ravel()[::50])[0, 1])
    print(f"  corr(sim_pv, sim_align+kmer)={corr:.3f}", flush=True)

    scaler = StandardScaler()
    xr = scaler.fit_transform(x_tr)
    xv = scaler.transform(x_va)
    ridge_p = np.zeros((len(idx_val), N_LABELS), np.float32)
    for j in tqdm(range(N_LABELS), desc="    Ridge", unit="lab"):
        clf = Ridge(alpha=10.0)
        clf.fit(xr, y_train[:, j].astype(np.float32))
        ridge_p[:, j] = clf.predict(xv)
    ridge_p = np.clip(ridge_p, 0.0, 1.0)

    rows = [baseline]
    rec_pv = eval_ens(sim_pv, y_val, hit)
    rec_pv["mode"] = "pv_alone"
    rows.append(rec_pv)
    print(f"  pv_alone F1={rec_pv['f1_100']:.4f}", flush=True)
    rec_rd = eval_ens(ridge_p, y_val, hit)
    rec_rd["mode"] = "ridge_alone"
    rows.append(rec_rd)
    print(f"  ridge_alone F1={rec_rd['f1_100']:.4f}", flush=True)

    for wt in W_GRID:
        ens = ens0 + wt * sim_pv
        rec = eval_ens(ens, y_val, hit)
        rec["mode"] = f"add_w{wt:g}"
        rec["w"] = wt
        rows.append(rec)
        print(f"  add w={wt:g}: F1={rec['f1_100']:.4f} missF1={rec['f1_miss']:.4f}", flush=True)

    miss = ~hit
    for wt in MISS_W_GRID:
        ens = ens0.copy()
        ens[miss] = ens0[miss] + wt * sim_pv[miss]
        rec = eval_ens(ens, y_val, hit)
        rec["mode"] = f"miss_w{wt:g}"
        rec["w"] = wt
        rows.append(rec)
        print(f"  miss w={wt:g}: F1={rec['f1_100']:.4f} missF1={rec['f1_miss']:.4f}", flush=True)

    for wt in W_GRID:
        ens = ens0 + wt * ridge_p
        rec = eval_ens(ens, y_val, hit)
        rec["mode"] = f"ridge_w{wt:g}"
        rec["w"] = wt
        rows.append(rec)
        print(f"  ridge w={wt:g}: F1={rec['f1_100']:.4f}", flush=True)

    cand = [r for r in rows if r["mode"] != "ck005a"]
    best = max(cand, key=lambda r: r["f1_100"])
    gain = best["f1_100"] - baseline["f1_100"]
    useful = bool(gain >= 0.001)
    print(f"  最优 {best['mode']} F1={best['f1_100']:.4f} Δ={gain:+.4f} → "
          f"{'有效' if useful else '无增益'}")

    report = {
        "baseline": baseline,
        "best": best,
        "grid": rows,
        "corr_pv_sim": round(corr, 6),
        "vocab": int(len(wv)),
        "gain_all_f1": round(gain, 6),
        "useful": useful,
        "seconds": round(time.time() - t0, 1),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import sys
    os.sys = sys
    main()
