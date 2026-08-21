"""
E004a-attn-pool — 冻结 ESM2-150M + 残基 attention pooling 分类头
================================================================
依据:
  SPROF-GO (Yuan et al., Brief Bioinform 2023, doi:10.1093/bib/bbad117)
  PLM 嵌入 + self-attention pooling（对准功能域）对 non-homologous 蛋白仍能泛化。
  本赛 E001 mean-pool 失败；E004 LoRA 微调太慢。本实验只训读出头、不训 30 层。
  SPROF-GO 的同源 label diffusion 不采用（TCK002b/ECK002 已证事后传播无效）。
方案:
  冻全部 ESM2 encoder；last_hidden_state → mask 内 attention pooling → Linear(640,500)
  残基嵌入缓存 rel/E004a_esm_hidden_{split}.{npy,mask.npy}（memmap，约 37GB train / 9GB test）
  对照: 同一缓存上 mean-pool 头（复现 E001/B005 读出）
  主指标: 独立 F1(0.5)/100档 + CK005 miss 子集 F1（应对 >0.17）
用法:
  python code/E/E004a-attn-pool.py
  python code/E/E004a-attn-pool.py --skip-embed   # 已有残基缓存
  python code/E/E004a-attn-pool.py --gen-sub
输出:
  models/e004a-attn.pt
  rel/E004a_report.json
  rel/E004a_probs.npz
  submit/submission_E004a_attn.csv（--gen-sub）
"""

import json
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import EsmModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

ESM2_PATH = os.path.join(PROJECT_ROOT, "models", "esm2_150m")
N_LABELS = 500
MAX_LEN = 256
HIDDEN = 640
EMBED_BS = 8
HEAD_BS = 256
N_EPOCHS = 30
LR = 1e-3
PATIENCE = 5


def macro_f1(ens, y, t=0.5):
    pred = ens >= t
    f1s = []
    for i in range(N_LABELS):
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
    for i in range(N_LABELS):
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


def macro_auc(ens, y):
    aucs = []
    for i in range(N_LABELS):
        yt = y[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, ens[:, i]))
    return float(np.mean(aucs))


def search_thresholds(ens, y):
    cands = np.linspace(0.01, 0.99, 100)
    best_t = np.full(N_LABELS, 0.5)
    for i in tqdm(range(N_LABELS), desc="    阈值搜索", unit="label"):
        yt = y[:, i]
        if yt.sum() == 0:
            continue
        p = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (p & ytm).sum(0)
        fp = (p & ~ytm).sum(0)
        fn = (~p & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best_t[i] = cands[f1t.argmax()]
    return best_t


class TokenDS(Dataset):
    def __init__(self, ids, mask):
        self.ids = ids
        self.mask = mask

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        m = max(int(self.mask[i].sum()), 1)
        return self.ids[i, :m], self.mask[i, :m]


def collate_tok(batch):
    ids, mask = zip(*batch)
    max_l = max(x.size(0) for x in ids)
    b = len(ids)
    ids_p = torch.zeros(b, max_l, dtype=torch.long)
    mask_p = torch.zeros(b, max_l, dtype=torch.long)
    for i, (a, m) in enumerate(zip(ids, mask)):
        n = a.size(0)
        ids_p[i, :n] = a
        mask_p[i, :n] = m
    return ids_p, mask_p


class HiddenDS(Dataset):
    def __init__(self, hidden, mask, y):
        self.hidden = hidden
        self.mask = mask
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (torch.from_numpy(self.hidden[i]),
                torch.from_numpy(self.mask[i]),
                torch.from_numpy(self.y[i]))


class AttnPoolHead(nn.Module):
    """SPROF-GO 式残基 attention pooling + 线性头。mean_pool=True 时退化为 mean。"""

    def __init__(self, hidden=HIDDEN, n_labels=N_LABELS, dropout=0.1, mean_pool=False):
        super().__init__()
        self.mean_pool = mean_pool
        self.score = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_labels)

    def forward(self, hidden, mask):
        # hidden: (B, L, H) 已按 MAX_LEN pad；mask: (B, L)
        if self.mean_pool:
            m = mask.unsqueeze(-1).float()
            pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            logits = self.score(hidden).squeeze(-1)
            logits = logits.masked_fill(mask == 0, -1e4)
            w = torch.softmax(logits, dim=-1).unsqueeze(-1)
            pooled = (hidden * w).sum(1)
        return self.fc(self.drop(pooled))


def load_esm():
    cwd = os.getcwd()
    os.chdir(os.path.dirname(ESM2_PATH))
    model = EsmModel.from_pretrained(os.path.basename(ESM2_PATH), local_files_only=True)
    os.chdir(cwd)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.cuda()


@torch.no_grad()
def extract_hidden(ids_np, mask_np, cache_stem):
    hid_path = cache_stem + ".npy"
    msk_path = cache_stem + ".mask.npy"
    n = len(ids_np)
    if os.path.exists(hid_path) and os.path.exists(msk_path):
        hid = np.lib.format.open_memmap(hid_path, mode="r")
        msk = np.lib.format.open_memmap(msk_path, mode="r")
        print(f"  加载 memmap {hid_path} {hid.shape}")
        return hid, msk
    print(f"  提取 ESM2 last_hidden → {hid_path}（memmap，约 {n * MAX_LEN * HIDDEN * 2 / 1e9:.1f} GB）")
    hid = np.lib.format.open_memmap(
        hid_path, mode="w+", dtype=np.float16, shape=(n, MAX_LEN, HIDDEN))
    msk = np.lib.format.open_memmap(
        msk_path, mode="w+", dtype=np.int8, shape=(n, MAX_LEN))
    model = load_esm()
    ds = TokenDS(torch.from_numpy(ids_np).long(), torch.from_numpy(mask_np).long())
    loader = DataLoader(ds, batch_size=EMBED_BS, shuffle=False, num_workers=0, collate_fn=collate_tok)
    off = 0
    for ids, mask in tqdm(loader, desc="    ESM2 前向", unit="batch"):
        b, l = ids.shape
        ids, mask = ids.cuda(), mask.cuda()
        with torch.amp.autocast("cuda"):
            h = model(input_ids=ids, attention_mask=mask).last_hidden_state
        hid[off:off + b, :l] = h.float().cpu().numpy().astype(np.float16)
        msk[off:off + b, :l] = mask.cpu().numpy().astype(np.int8)
        off += b
        del ids, mask, h
        if off % (EMBED_BS * 200) == 0:
            hid.flush()
            msk.flush()
            torch.cuda.empty_cache()
    hid.flush()
    msk.flush()
    del model
    torch.cuda.empty_cache()
    print(f"  已缓存 {hid_path}")
    return hid, msk


def ck005_miss_mask(train_df, idx_val):
    tsv = os.path.join(REL_DIR, "ck005_tmp", "align_val.tsv")
    val_ids = train_df["protein_id"].values[idx_val]
    hit = np.zeros(len(val_ids), dtype=bool)
    if not os.path.exists(tsv):
        return ~hit
    pos = {q: i for i, q in enumerate(val_ids)}
    with open(tsv) as f:
        for line in f:
            p = line.split("\t")
            if len(p) < 4:
                continue
            q = p[0]
            if q not in pos:
                continue
            try:
                ev = float(p[3])
            except ValueError:
                continue
            if ev <= 0.001:
                hit[pos[q]] = True
    return ~hit


def train_head(hidden, mask, y_train, y_val, mean_pool, tag):
    head = AttnPoolHead(mean_pool=mean_pool).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    tr_ds = HiddenDS(hidden[0], mask[0], y_train.astype(np.float32))
    va_ds = HiddenDS(hidden[1], mask[1], y_val.astype(np.float32))
    tr_loader = DataLoader(tr_ds, batch_size=HEAD_BS, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=HEAD_BS, shuffle=False, num_workers=0)
    best_f1, bad, best_state = -1.0, 0, None
    t0 = time.time()
    for ep in range(N_EPOCHS):
        head.train()
        total = 0.0
        for h, m, y in tr_loader:
            h, m, y = h.cuda().float(), m.cuda().long(), y.cuda()
            opt.zero_grad(set_to_none=True)
            loss = bce(head(h, m), y)
            loss.backward()
            opt.step()
            total += loss.item() * h.size(0)
        head.eval()
        probs = []
        with torch.no_grad():
            for h, m, _ in va_loader:
                h, m = h.cuda().float(), m.cuda().long()
                probs.append(torch.sigmoid(head(h, m)).cpu().numpy())
        probs = np.concatenate(probs)
        f1 = macro_f1(probs, y_val, 0.5)
        print(f"  {tag} ep{ep + 1}: loss={total / len(tr_ds):.4f} F1(0.5)={f1:.4f}", flush=True)
        if f1 > best_f1 + 1e-4:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"    {tag} 早停", flush=True)
                break
    head.load_state_dict(best_state)
    head.eval()
    def predict(hid, msk):
        ds = HiddenDS(hid, msk, np.zeros((len(hid), N_LABELS), np.float32))
        ld = DataLoader(ds, batch_size=HEAD_BS, shuffle=False, num_workers=0)
        out = []
        with torch.no_grad():
            for h, m, _ in ld:
                h, m = h.cuda().float(), m.cuda().long()
                out.append(torch.sigmoid(head(h, m)).cpu().numpy())
        return np.concatenate(out)
    print(f"  {tag} 最佳 F1(0.5)={best_f1:.4f} 耗时 {time.time() - t0:.0f}s")
    return head, best_f1, predict


def main():
    skip_embed = "--skip-embed" in sys.argv
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print("E004a: 冻结 ESM2 + attention pooling 头（不训 30 层）")

    print("[1/4] token 缓存...")
    tr_tok = np.load(os.path.join(REL_DIR, "esm2_tokens_train.npz"))
    te_tok = np.load(os.path.join(REL_DIR, "esm2_tokens_test.npz"))
    ids_tr = tr_tok["input_ids"][:, :MAX_LEN]
    mask_tr = tr_tok["attention_mask"][:, :MAX_LEN]
    ids_te = te_tok["input_ids"][:, :MAX_LEN]
    mask_te = te_tok["attention_mask"][:, :MAX_LEN]

    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train.csv"))
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_train, y_val = y_full[idx_train], y_full[idx_val]

    cache_tr = os.path.join(REL_DIR, "E004a_esm_hidden_train")
    cache_te = os.path.join(REL_DIR, "E004a_esm_hidden_test")
    if skip_embed and not (os.path.exists(cache_tr + ".npy") and os.path.exists(cache_te + ".npy")):
        raise FileNotFoundError("无残基缓存，去掉 --skip-embed")
    hid_all, msk_all = extract_hidden(ids_tr, mask_tr, cache_tr)
    hid_te, msk_te = extract_hidden(ids_te, mask_te, cache_te)
    hid_tr, msk_tr = hid_all[idx_train], msk_all[idx_train]
    hid_va, msk_va = hid_all[idx_val], msk_all[idx_val]

    print("[2/4] 训练 mean-pool 对照头 + attention pooling 头...")
    _, f1_mean, pred_mean = train_head(
        (hid_tr, hid_va), (msk_tr, msk_va), y_train, y_val, True, "mean")
    head_attn, f1_attn, pred_attn = train_head(
        (hid_tr, hid_va), (msk_tr, msk_va), y_train, y_val, False, "attn")
    torch.save(head_attn.state_dict(), os.path.join(MODEL_DIR, "e004a-attn.pt"))

    print("[3/4] 评估...")
    p_mean = pred_mean(hid_va, msk_va)
    p_attn = pred_attn(hid_va, msk_va)
    t_mean = search_thresholds(p_mean, y_val)
    t_attn = search_thresholds(p_attn, y_val)
    miss = ck005_miss_mask(train_df, idx_val)

    def pack(name, p, t):
        f05 = macro_f1(p, y_val, 0.5)
        f100 = macro_f1(p, y_val, t)
        r, prec = macro_rp(p, y_val, t)
        auc = macro_auc(p, y_val)
        f_miss = macro_f1(p[miss], y_val[miss], t) if miss.any() else 0.0
        print(f"  {name}: F1(0.5)={f05:.4f} F1(100)={f100:.4f} "
              f"R={r:.4f} P={prec:.4f} AUC={auc:.4f} missF1={f_miss:.4f}")
        return {"f1_05": round(f05, 6), "f1_100": round(f100, 6),
                "recall_100": round(r, 6), "precision_100": round(prec, 6),
                "auc": round(auc, 6), "f1_miss": round(f_miss, 6)}

    rec_mean = pack("mean-pool", p_mean, t_mean)
    rec_attn = pack("attn-pool", p_attn, t_attn)
    beat_c002b = rec_attn["f1_05"] > 0.2714
    beat_miss = rec_attn["f1_miss"] > 0.17
    print(f"  vs C002b 0.2714: {'超过' if beat_c002b else '未超'} | "
          f"miss>0.17: {'是' if beat_miss else '否'}")

    np.savez_compressed(os.path.join(REL_DIR, "E004a_probs.npz"),
                        val=p_attn, mean_val=p_mean)
    report = {
        "mean_pool": rec_mean, "attn_pool": rec_attn,
        "delta_f1_05": round(rec_attn["f1_05"] - rec_mean["f1_05"], 6),
        "beat_c002b": beat_c002b, "beat_miss": beat_miss,
        "seconds": round(time.time() - t0),
    }

    if gen_sub:
        p_test = pred_attn(hid_te, msk_te)
        np.savez_compressed(os.path.join(REL_DIR, "E004a_probs.npz"),
                            val=p_attn, mean_val=p_mean, test=p_test)
        y_pred = (p_test >= t_attn[None, :]).astype(np.int8)
        os.makedirs(SUB_DIR, exist_ok=True)
        sub = pd.concat([test_df[["protein_id"]],
                         pd.DataFrame(y_pred, columns=label_cols)], axis=1)
        sub_path = os.path.join(SUB_DIR, "submission_E004a_attn.csv")
        sub.to_csv(sub_path, index=False)
        report["submission"] = sub_path
        print(f"  提交: {sub_path}")

    with open(os.path.join(REL_DIR, "E004a_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {os.path.join(REL_DIR, 'E004a_report.json')}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
