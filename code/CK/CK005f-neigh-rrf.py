"""
CK005f-neigh-rrf — 同源邻居名单 RRF（DIAMOND × k-mer）
================================================================
依据:
  - RRF: Cormack et al., SIGIR 2009, doi:10.1145/1571941.1572114
    原义是融两路检索的**文档列表**，不校准分数。
  - 稀疏+稠密混合检索普遍用 RRF 拼 BM25 与向量召回。
  - I011：把 RRF 用在 500 标签排序上已有效；I011a 在标签维拆比对/k-mer 未超。
自己的改法:
  训练蛋白当文档。DIAMOND bitscore 与 k-mer 相似是两路检索。
  对每条查询把邻居名单做 RRF，再用 RRF 分加权转移标签。
  短名单（≤5）不用网页检索的 k=60，改 k∈{0,1,5,10}。
  对照：k=6 单路；多 k 名单先 RRF 再与 DIAMOND 融。
  替换 CK005a 的 sim，或与旧 sim 凸组合。不改 deep。
  门槛相对 CK005a ≥ +0.001。
用法:
  python code/CK/CK005f-neigh-rrf.py
输出:
  rel/CK005f_report.json
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    "i011", os.path.join(PROJECT_ROOT, "code", "I", "I011-rrf.py"))
i011 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i011)

REPORT_PATH = os.path.join(i011.REL_DIR, "CK005f_report.json")
K_RRF = (0, 1, 5, 10)
TOP_OUT = 5
KMER_KS = (3, 4, 5, 6, 7, 8)


def load_kmer_lists():
    lists = []
    for k in KMER_KS:
        prefix = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(i011.REL_DIR, f"{prefix}_neigh_k{k}.npz"))
        lists.append(d["idx_val"].astype(np.int64))
    return lists


def rrf_lists(list_arrs, k_rrf, top_out=TOP_OUT):
    """list_arrs: 若干 (n, L) 邻居下标，-1 为空。返回 idx, w。"""
    n = list_arrs[0].shape[0]
    out_idx = np.full((n, top_out), -1, np.int64)
    out_w = np.zeros((n, top_out), np.float32)
    for i in range(n):
        scores = {}
        for arr in list_arrs:
            for r, j in enumerate(arr[i]):
                if j < 0:
                    continue
                j = int(j)
                scores[j] = scores.get(j, 0.0) + 1.0 / (k_rrf + r + 1)
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_out]
        for t, (j, s) in enumerate(ranked):
            out_idx[i, t] = j
            out_w[i, t] = s
    return out_idx, out_w


def main():
    t0 = time.time()
    print("=" * 60)
    print("CK005f: 邻居名单 RRF（DIAMOND × k-mer）")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_d, bits_d = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_d[:, 0] >= 0
    miss = ~hit
    print(f"  hit {int(hit.sum())}/{len(hit)} miss {int(miss.sum())}")

    kmer_lists = load_kmer_lists()
    idx_k6 = kmer_lists[KMER_KS.index(6)]
    sim_bit = i011.agg_labels(idx_d, bits_d, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim_old = i011.ALIGN_W * sim_bit + (1 - i011.ALIGN_W) * sim_kmer

    overlap = []
    for i in np.where(hit)[0][:]:
        d0 = int(idx_d[i, 0])
        overlap.append(d0 in set(idx_k6[i][idx_k6[i] >= 0].tolist()))
    print(f"  DIAMOND top-1 ∈ k6 top-5: {float(np.mean(overlap)):.3f} (hit)")

    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    base = deep + i011.SIM_W * sim_old

    rows = []
    rec = i011.eval_ens(base, y_val, hit, miss, "ck005a")
    rows.append(rec)
    print(f"  ck005a F1={rec['f1_100']:.4f} miss={rec['f1_miss']:.4f}")

    print("[1/2] 替换 sim：diamond×k6 / diamond×多k ...")
    configs = {"ck005a": base}
    for k_rrf in tqdm(K_RRF, desc="    k", unit="k"):
        for tag, lists in (
            ("k6", [idx_d, idx_k6]),
            ("mk", [idx_d] + kmer_lists),
        ):
            idx_r, w_r = rrf_lists(lists, k_rrf)
            sim_r = i011.agg_labels(idx_r, w_r, y_train, top_k=TOP_OUT)
            ens = deep + i011.SIM_W * sim_r
            name = f"rep_{tag}_k{k_rrf}"
            rec = i011.eval_ens(ens, y_val, hit, miss, name, {"k_rrf": k_rrf, "lists": tag})
            rec["delta"] = round(rec["f1_100"] - rows[0]["f1_100"], 6)
            rows.append(rec)
            configs[name] = ens
            print(f"  {name:18s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f} "
                  f"miss={rec['f1_miss']:.4f}")

    print("[2/2] 旧 sim 与 RRF-sim 凸组合（k_rrf=1, k6）...")
    idx_r, w_r = rrf_lists([idx_d, idx_k6], 1)
    sim_r = i011.agg_labels(idx_r, w_r, y_train, top_k=TOP_OUT)
    for g in (0.25, 0.50, 0.75):
        ens = deep + i011.SIM_W * ((1 - g) * sim_old + g * sim_r)
        name = f"mix_k6_k1_g{g:.2f}"
        rec = i011.eval_ens(ens, y_val, hit, miss, name, {"gamma": g})
        rec["delta"] = round(rec["f1_100"] - rows[0]["f1_100"], 6)
        rows.append(rec)
        configs[name] = ens
        print(f"  {name:18s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f}")

    rng = np.random.RandomState(i011.RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(ens):
        pred = np.zeros_like(ens, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            t = i011.search_t(ens[fit], y_val[fit])
            pred[hold] = (ens[hold] >= t).astype(np.int8)
        return pred

    print("半折核对...")
    oof_rows = []
    oof_base = None
    for name, ens in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(ens)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "ck005a":
            oof_base = rec["f1_100"]
        print(f"  oof_{name:18s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta"] = round(rec["f1_100"] - oof_base, 6)

    best_oof = max(oof_rows, key=lambda r: r["f1_100"])
    gain = best_oof["f1_100"] - oof_base
    useful = bool(gain >= 0.001 and best_oof["name"] != "oof_ck005a")
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"Δ={gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "diamond_in_k6": round(float(np.mean(overlap)), 6),
        "baseline": rows[0],
        "best_oof": best_oof,
        "gain_oof": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
