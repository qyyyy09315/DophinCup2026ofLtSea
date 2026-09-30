"""
E003-probe — ProstT5 嵌入区分度快速探针 (短序列子集)
====================================================
背景:
  基准 (E003-bench) 显示全量推理不可行: 256-512aa 1.4s/条, 512+aa 6s+/条且 OOM,
  全量 11 万条预估 70-100h。故先只嵌入短序列子集 (<MAX_AA), 检验:
    1. 余弦分布 (对照 E001 ESM2 的 mean≈0.94 全体聚集)
    2. top-3 邻居聚合 F1 vs 随机基线 vs kmer 全库参照
  若余弦有分散度且 F1 > 随机, 结构嵌入有信号, 再考虑子集化方案;
  否则直接关闭 E003 (与 E001/E002 同结论)。

用法:
  ./.venv/Scripts/python.exe code/E/E003-probe.py [--q 400] [--db 2000]
                                              [--max-aa 256] [--skip-emb]

输出:
  stdout 诊断; 嵌入缓存 rel/E003_probe_emb.npz
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
os.chdir(PROJECT_ROOT)  # transformers 4.46.3 绝对路径 from_pretrained 返回 bool 的 bug

MAX_LEN = 2000
TOKEN_BUDGET = 6000
BATCH_CAP = 32


def embed(model, tokenizer, seqs, desc):
    lens = np.array([len(s) for s in seqs])
    order = np.argsort(lens)
    emb = np.zeros((len(seqs), 1024), dtype=np.float32)
    pbar = tqdm(total=len(seqs), desc=desc)
    i = 0
    while i < len(seqs):
        j = i
        while (j < len(seqs) and (j - i) < BATCH_CAP
               and lens[order[j]] * (j - i + 1) <= TOKEN_BUDGET):
            j += 1
        bidx = order[i:j]
        texts = ["<AA2fold> " + " ".join(list(seqs[k][:MAX_LEN])) for k in bidx]
        enc = tokenizer(texts, padding="longest", truncation=True,
                        max_length=MAX_LEN + 2, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb[bidx] = ((out * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
        i = j
        pbar.update(len(bidx))
    pbar.close()
    return emb


def knn_agg_f1(q_emb, db_emb, db_y, y_true, top_k=3):
    """归一化嵌入 → 余弦 top-k 加权邻居聚合 → F1(0.5)。"""
    q = torch.from_numpy(q_emb.astype(np.float32))
    db = torch.from_numpy(db_emb.astype(np.float32))
    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    db = db / db.norm(dim=1, keepdim=True).clamp(min=1e-8)
    q, db = q.to("cuda"), db.to("cuda")
    n = q.shape[0]
    idx = np.zeros((n, 5), dtype=np.int64)
    sims = np.zeros((n, 5), dtype=np.float32)
    for s in range(0, n, 512):
        e = min(s + 512, n)
        top = torch.topk(q[s:e] @ db.T, 5, dim=1)
        idx[s:e] = top.indices.cpu().numpy()
        sims[s:e] = top.values.cpu().numpy()
    lab = db_y[idx[:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
    w = np.maximum(sims[:, :top_k], 1e-7)
    sim = (lab * w[:, :, None]).sum(1) / w.sum(1)[:, None]
    thr = np.arange(0.05, 1.0, 0.05)
    f1s = [f1_score(y_true, sim > t, average="macro") for t in thr]
    auc = roc_auc_score(y_true, sim, average="macro")
    return max(f1s), auc, float(np.percentile(sims, 50))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=400)
    ap.add_argument("--db", type=int, default=2000)
    ap.add_argument("--max-aa", type=int, default=256)
    ap.add_argument("--skip-emb", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda"

    from transformers import T5Tokenizer, T5EncoderModel
    print(f"加载 ProstT5 (models/prostt5)...")
    tokenizer = T5Tokenizer.from_pretrained("models/prostt5", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("models/prostt5", torch_dtype=torch.bfloat16).to(dev)
    model.eval()

    train_df = pd.read_csv("data/train.csv")
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    seqs = train_df["sequence"].values
    lens = np.array([len(s) for s in seqs])
    _, idx_val = train_test_split(np.arange(len(train_df)), test_size=0.2,
                                  random_state=42)
    short_val = idx_val[lens[idx_val] <= args.max_aa]
    short_tr = np.where(lens <= args.max_aa)[0]
    rng = np.random.RandomState(0)
    q_idx = rng.choice(short_val, min(args.q, len(short_val)), replace=False)
    db_idx = rng.choice(short_tr, min(args.db, len(short_tr)), replace=False)
    print(f"查询: {len(q_idx)} 条 (val, ≤{args.max_aa}aa) | 库: {len(db_idx)} 条 (train)")
    y_q, y_db = y_full[q_idx], y_full[db_idx]

    emb_path = f"{REL_DIR}/E003_probe_emb.npz"
    if args.skip_emb and os.path.exists(emb_path):
        d = np.load(emb_path)
        emb_q, emb_db = d["emb_q"], d["emb_db"]
    else:
        seq_q = [seqs[i].replace("U", "X").replace("Z", "X").replace("O", "X") for i in q_idx]
        seq_db = [seqs[i].replace("U", "X").replace("Z", "X").replace("O", "X") for i in db_idx]
        print(f"[1/2] 嵌入查询 {len(seq_q)} + 库 {len(seq_db)} 条...")
        emb_q = embed(model, tokenizer, seq_q, "嵌入查询")
        emb_db = embed(model, tokenizer, seq_db, "嵌入库")
        np.savez_compressed(emb_path, emb_q=emb_q, emb_db=emb_db)
        print(f"  缓存: {emb_path}")

    # 1) 余弦分布诊断 (对照 E001 mean≈0.94)
    qn = emb_q / np.linalg.norm(emb_q, axis=1, keepdims=True).clip(min=1e-8)
    dn = emb_db / np.linalg.norm(emb_db, axis=1, keepdims=True).clip(min=1e-8)
    diag = qn[:200] @ dn[:200].T
    off = diag[~np.eye(200, dtype=bool)]
    print(f"[2/2] 余弦分布: mean={off.mean():.4f} std={off.std():.4f} "
          f"p50={np.median(off):.4f} p99={np.percentile(off, 99):.4f} "
          f"(E001 ESM2 对照 mean≈0.94, p99≈0.97)")
    print(f"  嵌入范数 mean={np.linalg.norm(emb_q, axis=1).mean():.1f}")

    # 2) top-3 邻居聚合 F1 vs 基线
    f1_pt5, auc_pt5, med_sim = knn_agg_f1(emb_q, emb_db, y_db, y_q, 3)
    # 随机基线: 同库随机 3 邻居
    rng2 = np.random.RandomState(1)
    ridx = rng2.randint(0, len(db_idx), size=(len(q_idx), 3))
    rlab = y_db[ridx].mean(axis=1)
    thr = np.arange(0.05, 1.0, 0.05)
    f1_rand = max(f1_score(y_q, rlab > t, average="macro") for t in thr)
    auc_rand = roc_auc_score(y_q, rlab, average="macro")
    # 自身标签先验 (库标签均值)
    pri = np.tile(y_db.mean(axis=0), (len(q_idx), 1))
    f1_prior = max(f1_score(y_q, pri > t, average="macro") for t in thr)

    print(f"\nProstT5 嵌入 top-3 邻居: F1(best)={f1_pt5:.4f} AUC={auc_pt5:.4f} med_sim={med_sim:.4f}")
    print(f"随机邻居基线:          F1(best)={f1_rand:.4f} AUC={auc_rand:.4f}")
    print(f"标签先验基线:          F1(best)={f1_prior:.4f}")

    # 3) 有标签重合的邻居比例 (更强信号: 相似性是否对应功能相似)
    sim_m = qn[:200] @ dn.T
    top1 = sim_m.argmax(axis=1)
    y_overlap = (y_q[:200] & y_db[top1]).sum(axis=1)
    pos_n = y_q[:200].sum(axis=1).clip(min=1)
    print(f"top-1 邻居标签重合率: {y_overlap.mean():.3f} (随机期望 {pos_n.mean()/500:.3f})")

    rep = {"args": vars(args), "diag_cosine": {"mean": round(float(off.mean()), 4),
                                               "std": round(float(off.std()), 4),
                                               "p99": round(float(np.percentile(off, 99)), 4)},
           "pt5_top3": {"f1": round(f1_pt5, 4), "auc": round(auc_pt5, 4)},
           "random_baseline": {"f1": round(f1_rand, 4), "auc": round(auc_rand, 4)},
           "prior_baseline": {"f1": round(f1_prior, 4)},
           "top1_overlap": round(float(y_overlap.mean()), 4),
           "seconds": round(time.time() - t0)}
    with Path(f"{REL_DIR}/E003_probe_report.json").open("w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {REL_DIR}/E003_probe_report.json | 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
