"""
C002c-tta — C002b 测试时多窗平均（不重训）
================================================================
依据:
  - Shanmugam et al., ICCV 2021, doi:10.1109/iccv48922.2021.00125
    测试时增强：对输入做不变变换后聚合概率。
  - C002b 把序列截到 1500 再全局池化；val 中 >1500aa 约 4.4%。
    CK005a miss 几乎不在长序列（>1500 miss 率 2.2%），TTA 主要打 hit 长蛋白。
方案:
  加载 models/cnn-c002b.pt。len≤1500 复用 TC003 的 cnn 概率（与 CK005a 一致）。
  len>1500：头 / 中 / 尾各 1500aa，sigmoid 概率算术平均，替换 CNN 路。
  再接 CK005a 四路（ALIGN_W=0.4, SIM_W=0.75）。
  主指标: 全局 F1；辅：>1500 子集 F1。不重训。
用法:
  python code/C/C002c-tta.py
  python code/C/C002c-tta.py --gen-sub
输出:
  rel/C002c_report.json
  rel/C002c_tta_cnn_probs.npz
  submit/submission_C002c_ensemble.csv（仅 --gen-sub 且有效）
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "C"))
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "c002b", os.path.join(PROJECT_ROOT, "code", "C", "C002b-cnn.py"))
c002b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c002b)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cnn-c002b.pt")
REPORT_PATH = os.path.join(REL_DIR, "C002c_report.json")
PROBS_PATH = os.path.join(REL_DIR, "C002c_tta_cnn_probs.npz")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
MAX_LEN = 1500
BATCH = 256
ALIGN_W = 0.4
SIM_W = 0.75
TOP_K = 3
MAX_ACCEPT = 5
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def windows(seq, max_len=MAX_LEN):
    n = len(seq)
    if n <= max_len:
        return [seq]
    out = [seq[:max_len], seq[-max_len:]]
    mid = seq[(n - max_len) // 2:(n - max_len) // 2 + max_len]
    if mid != out[0] and mid != out[-1]:
        out.append(mid)
    return out


class CropDS(Dataset):
    def __init__(self, seqs):
        self.seqs = c002b.encode_sequences(seqs, MAX_LEN)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return torch.from_numpy(self.seqs[i])


def collate_x(batch):
    seqs = torch.stack(batch).long()
    return F.one_hot(seqs, num_classes=c002b.N_CHANNELS).permute(0, 2, 1).float()


@torch.no_grad()
def predict_crops(model, seqs):
    ds = CropDS(seqs)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=collate_x,
                        pin_memory=DEVICE.type == "cuda")
    out = []
    model.eval()
    for x in loader:
        x = x.to(DEVICE)
        if DEVICE.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(x)
        else:
            logits = model(x)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def tta_probs(model, sequences, cache_short=None):
    """短序列用 cache；长序列头中尾平均。"""
    n = len(sequences)
    probs = np.zeros((n, N_LABELS), np.float32)
    long_idx, long_crops, owners = [], [], []
    n_short = 0
    for i, s in enumerate(sequences):
        crops = windows(s)
        if len(crops) == 1:
            n_short += 1
            if cache_short is not None:
                probs[i] = cache_short[i]
            else:
                long_idx.append(i)
                long_crops.append(crops[0])
                owners.append(i)
        else:
            for c in crops:
                long_crops.append(c)
                owners.append(i)
            long_idx.append(i)
    print(f"  短序列 {n_short} 复用缓存 | 长序列 {n - n_short} 共 {len(long_crops)} 窗")
    if long_crops:
        crop_p = predict_crops(model, long_crops)
        acc = np.zeros((n, N_LABELS), np.float32)
        cnt = np.zeros(n, np.float32)
        for j, oi in enumerate(owners):
            acc[oi] += crop_p[j]
            cnt[oi] += 1
        long_mask = cnt > 0
        probs[long_mask] = acc[long_mask] / cnt[long_mask, None]
    return probs, n - n_short


def parse_aln(tsv, query_ids, id_map, evalue=0.001):
    idx = np.full((len(query_ids), MAX_ACCEPT), -1, np.int64)
    bits = np.zeros((len(query_ids), MAX_ACCEPT), np.float32)
    qpos = {q: i for i, q in enumerate(query_ids)}
    with open(tsv) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            q, t = p[0], p[1]
            if q not in qpos or t not in id_map:
                continue
            i = qpos[q]
            try:
                bs, ev = float(p[2]), float(p[3])
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


def agg_labels(idx, w, y_space, top_k=TOP_K):
    n = idx.shape[0]
    valid = idx[:, :top_k] >= 0
    lab = np.zeros((n, top_k, y_space.shape[1]), np.float32)
    for j in range(top_k):
        lab[:, j] = y_space[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(w[:, :top_k], 1e-7)
    ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)


def build_kmer_val(y_train):
    sim = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        prefix = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(REL_DIR, f"{prefix}_neigh_k{k}.npz"))
        n = d["idx_val"].shape[0]
        lab = y_train[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        s = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(s, 1e-7) ** 2
        sim += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return sim


def build_kmer_test(y_full):
    sim = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        d = np.load(os.path.join(REL_DIR, "TCK002a_neigh_full.npz" if k == 6
                                 else f"CK004_neigh_k{k}_full.npz"))
        n2 = d["idx_test"].shape[0]
        lab = y_full[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
        s = d["sim_test"][:, :5].astype(np.float32)
        ww = np.maximum(s, 1e-7) ** 2
        sim += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return sim


def macro_f1(ens, y, t):
    pred = ens >= t
    f1s = []
    for i in range(y.shape[1]):
        yt = y[:, i].astype(bool)
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


def macro_rp(ens, y, t):
    pred = ens >= t
    rs, ps = [], []
    for i in range(y.shape[1]):
        yt = y[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum())
        fp = int((p & ~yt).sum())
        fn = int((~p & yt).sum())
        rs.append(tp / (tp + fn) if tp + fn else 0.0)
        ps.append(tp / (tp + fp) if tp + fp else 0.0)
    return float(np.mean(rs)), float(np.mean(ps))


def search_t(ens, y):
    cands = np.linspace(0.01, 0.99, 100)
    best = np.full(N_LABELS, 0.5)
    for i in tqdm(range(N_LABELS), desc="    阈值", unit="lab", leave=False):
        yt = y[:, i]
        if yt.sum() == 0:
            continue
        pr = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0)
        fp = (pr & ~ytm).sum(0)
        fn = (~pr & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best[i] = cands[f1t.argmax()]
    return best


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print(f"C002c: C002b 测试时多窗平均  device={DEVICE}")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    seqs = train_df["sequence"].values
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_seq = seqs[idx_val]
    val_len = np.array([len(s) for s in val_seq])
    long = val_len > MAX_LEN
    print(f"  val 长序列 >{MAX_LEN}: {int(long.sum())}/{len(val_seq)} ({long.mean():.3f})")

    print("[1/3] 加载 C002b + TTA 推理...")
    model = c002b.MultiScaleCNNv3(n_labels=N_LABELS).to(DEVICE)
    c002b.load_checkpoint(model, MODEL_PATH)
    model.eval()
    cache = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"].astype(np.float32)
    cnn_tta, n_long = tta_probs(model, val_seq, cache_short=cache)
    delta = np.abs(cnn_tta[~long] - cache[~long]).mean()
    print(f"  短序列与缓存 MAE={delta:.6f}（应为 ~0）")

    tsv = os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    idx_v, bits_v = parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim = ALIGN_W * agg_labels(idx_v, bits_v, y_train) + (1 - ALIGN_W) * build_kmer_val(y_train)
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]

    def ens_of(cnn):
        return (w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"]
                + w["cnn"] * cnn + SIM_W * sim)

    def eval_ens(ens, t=None):
        if t is None:
            t = search_t(ens, y_val)
        f_all = macro_f1(ens, y_val, t)
        r, p = macro_rp(ens, y_val, t)
        auc = float(np.mean([
            roc_auc_score(y_val[:, i], ens[:, i])
            for i in range(N_LABELS)
            if 0 < y_val[:, i].sum() < len(y_val)]))
        f_long = macro_f1(ens[long], y_val[long], t) if long.any() else 0.0
        f_miss = macro_f1(ens[miss], y_val[miss], t)
        return f_all, r, p, auc, f_long, f_miss, t

    print("[2/3] CK005a 对照 vs TTA CNN")
    b_all, b_r, b_p, b_auc, b_long, b_miss, _ = eval_ens(ens_of(cache))
    print(f"  基线 F1={b_all:.4f} longF1={b_long:.4f} missF1={b_miss:.4f}")
    f_all, r, p, auc, f_long, f_miss, t_best = eval_ens(ens_of(cnn_tta))
    print(f"  TTA   F1={f_all:.4f} longF1={f_long:.4f} missF1={f_miss:.4f} "
          f"Δ={f_all - b_all:+.4f} Δlong={f_long - b_long:+.4f}")

    useful = (f_all >= b_all + 0.001) or (f_long - b_long >= 0.005 and f_all >= b_all - 0.001)
    print(f"  → {'有效' if useful else '无增益'}")

    np.savez_compressed(PROBS_PATH, cnn_val=cnn_tta)
    report = {
        "n_long_val": int(long.sum()),
        "short_cache_mae": round(float(delta), 8),
        "baseline": {
            "f1_100": round(b_all, 6), "recall": round(b_r, 6),
            "precision": round(b_p, 6), "auc": round(b_auc, 6),
            "f1_long": round(b_long, 6), "f1_miss": round(b_miss, 6),
        },
        "tta": {
            "f1_100": round(f_all, 6), "recall": round(r, 6),
            "precision": round(p, 6), "auc": round(auc, 6),
            "f1_long": round(f_long, 6), "f1_miss": round(f_miss, 6),
        },
        "gain_all": round(f_all - b_all, 6),
        "gain_long": round(f_long - b_long, 6),
        "useful": useful,
        "seconds": round(time.time() - t0),
    }

    if not (gen_sub and useful):
        with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  报告: {REPORT_PATH}")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    print("[3/3] test TTA + 提交...")
    cache_te = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"].astype(np.float32)
    cnn_te, _ = tta_probs(model, test_df["sequence"].values, cache_short=cache_te)
    tsv_t = os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
    full_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
    idx_t, bits_t = parse_aln(tsv_t, test_df["protein_id"].values, full_map)
    sim_t = ALIGN_W * agg_labels(idx_t, bits_t, y_full) + (1 - ALIGN_W) * build_kmer_test(y_full)
    t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))
    ens_t = (w["t001a"] * t1t["t001a_test"] + w["t002"] * t2t["t002"]
             + w["cnn"] * cnn_te + SIM_W * sim_t)
    pred = (ens_t >= t_best).astype(np.int8)
    os.makedirs(SUB_DIR, exist_ok=True)
    sub = pd.concat([test_df[["protein_id"]],
                     pd.DataFrame(pred, columns=label_cols)], axis=1)
    sub_path = os.path.join(SUB_DIR, "submission_C002c_ensemble.csv")
    sub.to_csv(sub_path, index=False)
    np.savez_compressed(PROBS_PATH, cnn_val=cnn_tta, cnn_test=cnn_te)
    report["submission"] = sub_path
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  提交: {sub_path} ({len(sub)} 行)")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
