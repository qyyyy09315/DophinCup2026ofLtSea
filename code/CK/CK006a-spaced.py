"""
CK006a-spaced — 间隔种子 k-mer 邻居，混进 I019
================================================================
依据:
  - PatternHunter: Ma, Tromp & Li, Bioinformatics 2002,
    doi:10.1093/bioinformatics/18.3.440。非连续种子比连续 k-mer
    对替换更敏感。
  - CK006：Murphy-5 替换字母，与精确 k-mer corr 0.91，关闭。
    本实验改位置通配，字母表仍 20。
自己的改法:
  冻 I019。掩码 1110111（重量 6 跨度 7）与 11011011（重量 6 跨度 8）。
  HashingVectorizer 自定义 analyzer 抽间隔种子，val→idx_train top-5、
  sim² 加权传标签。ens'=(1-γ)·I019 + γ·col_norm(sp)。
  γ∈{0.15,0.30}。门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/CK/CK006a-spaced.py
输出:
  rel/CK006a_report.json
  rel/CK006a_neigh_{name}.npz
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))

REPORT_PATH = os.path.join(i011.REL_DIR, "CK006a_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
G_GRID = (0.15, 0.30)
TOP_K = 5
N_FEATURES = 2 ** 20
BLOCK = 2048
MASKS = {
    "w6s7": (1, 1, 1, 0, 1, 1, 1),
    "w6s8": (1, 1, 0, 1, 1, 0, 1, 1),
}


def make_analyzer(mask):
    keep = [i for i, b in enumerate(mask) if b]
    w = len(mask)

    def analyzer(seq):
        n = len(seq)
        if n < w:
            return []
        return ["".join(seq[i + j] for j in keep) for i in range(n - w + 1)]

    return analyzer


def neigh_sp(seqs_tr, seqs_q, name, mask, cache):
    if os.path.exists(cache):
        d = np.load(cache)
        return d["idx_val"], d["sim_val"]
    vec = HashingVectorizer(
        analyzer=make_analyzer(mask), n_features=N_FEATURES, dtype=np.float32)
    x_tr = normalize(sparse.csr_matrix(vec.transform(seqs_tr)))
    x_q = normalize(sparse.csr_matrix(vec.transform(seqs_q)))
    n_q = x_q.shape[0]
    idx = np.zeros((n_q, TOP_K), np.int64)
    sim = np.zeros((n_q, TOP_K), np.float32)
    for s in tqdm(range(0, n_q, BLOCK), desc=f"    {name} 邻居", unit="blk"):
        e = min(s + BLOCK, n_q)
        block = (x_q[s:e] @ x_tr.T).toarray()
        part = np.argpartition(-block, TOP_K, axis=1)[:, :TOP_K]
        idx[s:e] = part
        sim[s:e] = np.take_along_axis(block, part, axis=1)
    np.savez_compressed(cache, idx_val=idx, sim_val=sim)
    return idx, sim


def transfer(idx, sim, y_train):
    n = idx.shape[0]
    lab = y_train[idx[:, :TOP_K].ravel()].reshape(n, TOP_K, -1).astype(np.float32)
    ww = np.maximum(sim[:, :TOP_K], 1e-7) ** 2
    return (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]


def build_i019(train_df, y_full, idx_train, idx_val):
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1 - i011.ALIGN_W) * sim_kmer
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    s011 = deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(i011.rrf_pair(deep, sim, K_RRF))
    n = s011.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t_i011 = i011.search_t(s011, y_val)
    row_m = np.maximum((s011 >= t_i011).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    s019 = i019.mix(s011, p, ALPHA_SK)
    return s019, hit, y_train, y_val, sim_kmer


def main():
    t0 = time.time()
    print("=" * 60)
    print("CK006a: 间隔种子 k-mer")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s019, hit, y_train, y_val, sim_kmer = build_i019(
        train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    seqs = train_df["sequence"].values
    seqs_tr = seqs[idx_train]
    seqs_va = seqs[idx_val]
    ck6 = np.load(os.path.join(i011.REL_DIR, "CK002_neigh_k6.npz"))

    rows = [rec0]
    configs = {"i019": s019}
    overlap = {}

    print("[1/2] 间隔种子邻居 + 混合 ...")
    for name, mask in tqdm(MASKS.items(), desc="    mask", unit="m"):
        cache = os.path.join(i011.REL_DIR, f"CK006a_neigh_{name}.npz")
        idx_n, sim_n = neigh_sp(seqs_tr, seqs_va, name, mask, cache)
        sp = transfer(idx_n, sim_n, y_train)
        rec_s = i011.eval_ens(sp, y_val, hit, miss, f"sp_{name}")
        rec_s["delta_i019"] = round(rec_s["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec_s)
        print(f"  sp_{name} 独立 F1={rec_s['f1_100']:.4f}")

        top1_ov = float((idx_n[:, 0] == ck6["idx_val"][:, 0]).mean())
        set_ov = []
        for i in range(len(idx_n)):
            a, b = set(idx_n[i, :5]), set(ck6["idx_val"][i, :5])
            set_ov.append(len(a & b) / 5.0)
        ov5 = float(np.mean(set_ov))
        corr = float(np.corrcoef(sp.ravel(), sim_kmer.ravel())[0, 1])
        overlap[name] = {"top1": round(top1_ov, 4), "top5": round(ov5, 4),
                         "corr_kmer": round(corr, 4)}
        print(f"  vs CK002 k6: top1重叠={top1_ov:.3f} top5={ov5:.3f} corr={corr:.3f}")

        sp_n = i011.col_norm(sp)
        for g in G_GRID:
            mixed = (1.0 - g) * s019 + g * sp_n
            mname = f"mix_{name}_g{g:.2f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, mname,
                                {"mask": name, "gamma": g})
            rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            configs[mname] = mixed
            print(f"  {mname:22s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

    print("[2/2] 半折核对...")
    rng = np.random.RandomState(i011.RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            t = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= t).astype(np.int8)
        return pred

    oof_rows = []
    oof_i019 = None
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "i019":
            oof_i019 = rec["f1_100"]
        print(f"  oof_{name:22s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    mix_oof = [r for r in oof_rows if r["name"].startswith("oof_mix_")]
    best_oof = max(mix_oof, key=lambda r: r["f1_100"]) if mix_oof else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline_i019": rec0,
        "best_oof": best_oof,
        "gain_oof_vs_i019": round(gain, 6),
        "useful": useful,
        "overlap": overlap,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(i011.REL_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  用时 {report['seconds']}s")


if __name__ == "__main__":
    main()
