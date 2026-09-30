"""
E006d-esm2-650-mah-asl-len — E006c 同配方长序列版（MAX_LEN 可配，Colab/A100 目标）
=================================================================================
依据:
  - E006c：E006a 同骨干同 MAH + ASL（γ_neg=4, γ_pos=0, clip=0.05），MAX_LEN=256 为
    8GB 显存妥协；历史记录「MAX_LEN 256→1500 会爆显存且混长度」的限制在 A100 40GB 上解除。
  - I083 实测长度分布：p50=420、p99=3171；>1022aa 占 10.7%（截断损失 14.1% 残基）。
    W=512 覆盖中位数（复用现有 esm2_tokens 缓存，宽 512）；W=1024 将截断面压到 ~10.7%。
  - 判据不变：独立 F1(0.5) > C002b 0.2714 才接融合。
  - ASL: Ben-Baruch et al., ICCV 2021, arXiv:2009.14119；SeqProFT 残基读出同 E006c。
与 E006c 的差异（其余逐位沿用，含步级断点续训 / 400 步存盘 / Ctrl-C 存盘）：
  1. MAX_LEN 可配（--max-len=，默认 512）；
  2. token 缓存按宽选择：优先 rel/esm2_tokens_w{W}_{train,test}.npz，
     W=512 时回退复用 rel/esm2_tokens_{train,test}.npz（现缓存宽 512）；
     --build-tokens 从 data/*.csv 重新分词建宽缓存（W=1024 需先建）；
  3. 模型来源：优先本地 models/esm2_650m；设 E006D_ALLOW_HUB=1 时允许在线拉取
     facebook/esm2_t33_650M_UR50D（Colab 首跑场景）。
用法:
  python code/E/E006d-esm2-650-mah-asl-len.py --max-len=512
  python code/E/E006d-esm2-650-mah-asl-len.py --build-tokens --max-len=1024   # 先建缓存
  python code/E/E006d-esm2-650-mah-asl-len.py --max-len=1024 --bs=2           # A100 长序列
  python code/E/E006d-esm2-650-mah-asl-len.py --smoke / --fresh / --epochs=N
输出:
  models/e006d-650-mah-asl-len.pt / .last.pt
  rel/E006d_report.json
  rel/E006d_probs.npz（始终写 val；test 仅独立过门槛时写）
"""

import json
import os
import sys
import time
from itertools import islice

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from peft import LoraConfig, TaskType, get_peft_model
from transformers import EsmModel, AutoTokenizer
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ESM2_PATH = os.path.join(PROJECT_ROOT, "models", "esm2_650m")
ESM2_HUB_ID = "facebook/esm2_t33_650M_UR50D"
STEM = os.path.join(MODEL_DIR, "e006d-650-mah-asl-len")
LAST_PATH = STEM + ".last.pt"
BEST_PATH = STEM + ".pt"
REPORT_PATH = os.path.join(REL_DIR, "E006d_report.json")
PROBS_PATH = os.path.join(REL_DIR, "E006d_probs.npz")

N_LABELS = 500
MAX_LEN = 512
HIDDEN = 1280
N_HEADS = 8
N_EPOCHS = 6
BATCH = 4
EVAL_BATCH = 4
GRAD_ACCUM = 4
LR = 1e-4
PATIENCE = 2
VALIDATE_EVERY = 1
LORA_R = 8
LORA_ALPHA = 16
FREEZE_FIRST = 27
C002B_F05 = 0.2714
GAMMA_NEG = 4.0
GAMMA_POS = 0.0
ASL_CLIP = 0.05


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


class TokenDS(Dataset):
    def __init__(self, ids, mask, y):
        self.ids = ids
        self.mask = mask
        self.y = y

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        m = int(self.mask[i].sum())
        m = max(m, 1)
        return self.ids[i, :m], self.mask[i, :m], self.y[i]


def collate_dyn(batch):
    ids, mask, y = zip(*batch)
    max_l = max(x.size(0) for x in ids)
    b = len(ids)
    ids_p = torch.zeros(b, max_l, dtype=torch.long)
    mask_p = torch.zeros(b, max_l, dtype=torch.long)
    for i, (a, m) in enumerate(zip(ids, mask)):
        n = a.size(0)
        ids_p[i, :n] = a
        mask_p[i, :n] = m
    return ids_p, mask_p, torch.stack(y)


class AsymmetricLoss(nn.Module):
    """C002b 同参 ASL。Ben-Baruch et al., ICCV 2021, arXiv:2009.14119。"""

    def __init__(self, gamma_neg=GAMMA_NEG, gamma_pos=GAMMA_POS,
                 clip=ASL_CLIP, eps=1e-6):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits.float()).clamp(self.eps, 1.0 - self.eps)
        pos_loss = -targets * ((1.0 - p) ** self.gamma_pos) * torch.log(p)
        p_m = torch.clamp(p - self.clip, min=0.0)
        neg_loss = -(1.0 - targets) * (p_m ** self.gamma_neg) * torch.log(1.0 - p_m)
        return (pos_loss + neg_loss).mean()


class ResidueMAHHead(nn.Module):
    """SeqProFT 式残基多头注意力池化；不用接触图。"""

    def __init__(self, hidden=HIDDEN, n_labels=N_LABELS, n_heads=N_HEADS, dropout=0.1):
        super().__init__()
        if hidden % n_heads != 0:
            raise ValueError(f"hidden {hidden} 不能整除 n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.query = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        self.proj = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_labels)

    def forward(self, hidden, mask):
        x = self.proj(hidden)
        b, length, _ = x.shape
        x = x.view(b, length, self.n_heads, self.head_dim)
        scores = torch.einsum("blhd,hd->bhl", x, self.query)
        scores = scores.masked_fill(mask[:, None, :] == 0, -1e4)
        w = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bhl,blhd->bhd", w, x).reshape(b, -1)
        return self.fc(self.drop(pooled))


class EsmMah(nn.Module):
    def __init__(self, esm, hidden=HIDDEN, n_labels=N_LABELS, n_heads=N_HEADS):
        super().__init__()
        self.esm = esm
        self.head = ResidueMAHHead(hidden, n_labels, n_heads)

    def forward(self, input_ids, attention_mask):
        out = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return self.head(out.last_hidden_state, attention_mask)


def _with_local_cwd(path, fn):
    """transformers 4.46.x 绝对路径加载怪癖的统一绕法（同 E006c）。"""
    cwd = os.getcwd()
    os.chdir(os.path.dirname(path))
    try:
        return fn(os.path.basename(path))
    finally:
        os.chdir(cwd)


def get_esm_source():
    if os.path.isdir(ESM2_PATH):
        return ("local", ESM2_PATH)
    if os.environ.get("E006D_ALLOW_HUB") == "1":
        return ("hub", ESM2_HUB_ID)
    raise FileNotFoundError(
        f"缺少 {ESM2_PATH}；在线拉取需显式设 E006D_ALLOW_HUB=1（Colab 首跑）")


def load_esm():
    kind, src = get_esm_source()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if kind == "local":
        model = _with_local_cwd(src, lambda name: EsmModel.from_pretrained(
            name, torch_dtype=torch.float32, local_files_only=True))
    else:
        os.environ.pop("HF_HUB_OFFLINE")
        os.environ.pop("TRANSFORMERS_OFFLINE")
        model = EsmModel.from_pretrained(src, torch_dtype=torch.float32)
    return model


def load_tokenizer():
    kind, src = get_esm_source()
    if kind == "local":
        return _with_local_cwd(src, lambda name: AutoTokenizer.from_pretrained(
            name, local_files_only=True))
    return AutoTokenizer.from_pretrained(src)


def token_cache_paths(w):
    wide = (os.path.join(REL_DIR, f"esm2_tokens_w{w}_train.npz"),
            os.path.join(REL_DIR, f"esm2_tokens_w{w}_test.npz"))
    legacy = (os.path.join(REL_DIR, "esm2_tokens_train.npz"),
              os.path.join(REL_DIR, "esm2_tokens_test.npz"))
    if all(os.path.isfile(p) for p in wide):
        return wide
    if w <= 512 and all(os.path.isfile(p) for p in legacy):
        return legacy  # 现有缓存宽 512
    return wide


def build_tokens(w):
    """从 data/*.csv 重建宽 w 的分词缓存（CPU，一次性）。"""
    print(f"[build-tokens] 重建宽 {w} 分词缓存 ...", flush=True)
    tok = load_tokenizer()
    for split in ("train", "test"):
        df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", f"{split}.csv"))
        seqs = df["sequence"].values
        n = len(seqs)
        ids = np.zeros((n, w), dtype=np.int32)
        mask = np.zeros((n, w), dtype=np.int8)
        bs = 512
        for s in tqdm(range(0, n, bs), desc=f"tokenize {split}"):
            batch = [x[:w * 2] for x in seqs[s:s + bs]]  # 粗截，防超长拖速
            enc = tok(batch, padding="max_length", truncation=True,
                      max_length=w, return_tensors="np")
            ids[s:s + bs] = enc["input_ids"].astype(np.int32)
            mask[s:s + bs] = enc["attention_mask"].astype(np.int8)
        out = os.path.join(REL_DIR, f"esm2_tokens_w{w}_{split}.npz")
        np.savez_compressed(out, input_ids=ids, attention_mask=mask)
        print(f"  写出 {out} ({ids.shape})", flush=True)


def attach_lora(base, freeze_first):
    n_layers = len(base.encoder.layer)
    freeze_first = min(freeze_first, n_layers)
    for i, layer in enumerate(base.encoder.layer):
        if i < freeze_first:
            layer.requires_grad_(False)
    if freeze_first > 0:
        base.embeddings.requires_grad_(False)
    base.gradient_checkpointing_enable()
    if hasattr(base, "config"):
        base.config.use_cache = False
    cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
        bias="none",
        layers_to_transform=list(range(freeze_first, n_layers)),
    )
    esm = get_peft_model(base, cfg)
    model = EsmMah(esm, HIDDEN, N_LABELS, N_HEADS)
    return model, n_layers, freeze_first


def save_last(path, model, opt, sched, scaler, epoch, step, best_f1, best_epoch, bad):
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "best_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "patience_counter": int(bad),
    }, path)


def load_ckpt(path, map_location="cuda"):
    return torch.load(path, map_location=map_location, weights_only=True)


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    torch.cuda.empty_cache()
    out = []
    for ids, mask, _ in loader:
        ids, mask = ids.cuda(non_blocking=True), mask.cuda(non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(input_ids=ids, attention_mask=mask)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
        del ids, mask, logits
    torch.cuda.empty_cache()
    return np.concatenate(out)


def parse_args():
    opt = {"n_epochs": N_EPOCHS, "bs": BATCH, "lr": LR, "accum": GRAD_ACCUM,
           "freeze": FREEZE_FIRST, "max_len": MAX_LEN,
           "smoke": "--smoke" in sys.argv, "fresh": "--fresh" in sys.argv,
           "build_tokens": "--build-tokens" in sys.argv}
    for a in sys.argv[1:]:
        for key, cast in (("epochs", int), ("bs", int), ("lr", float),
                          ("accum", int), ("freeze", int), ("max-len", int)):
            pfx = f"--{key}="
            if a.startswith(pfx):
                opt[key.replace("-", "_")] = cast(a.split("=", 1)[1])
    return opt


def main():
    a = parse_args()
    n_epochs, bs, lr, accum = a["n_epochs"], a["bs"], a["lr"], a["accum"]
    freeze_first, max_len = a["freeze"], a["max_len"]
    smoke, fresh = a["smoke"], a["fresh"]
    for d in (REL_DIR, SUB_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        dev = torch.cuda.get_device_name(0)
    else:
        dev = "CPU"

    t0 = time.time()
    print("=" * 60)
    print(f"E006d LoRA-ESM2-650M MAH+ASL 长序列: W={max_len} epochs={n_epochs} "
          f"bs={bs} accum={accum} lr={lr} freeze={freeze_first}/33 "
          f"smoke={smoke} fresh={fresh} dev={dev}")

    if a["build_tokens"]:
        build_tokens(max_len)

    print("[1/4] 加载数据...")
    tp = token_cache_paths(max_len)
    if not all(os.path.isfile(p) for p in tp):
        raise FileNotFoundError(
            f"缺宽 {max_len} 的分词缓存（{tp[0]}），先跑 --build-tokens --max-len={max_len}")
    if max_len > 512 and "esm2_tokens_w" not in tp[0]:
        raise RuntimeError("宽缓存与 max_len 不匹配")
    tr_tok = np.load(tp[0])
    te_tok = np.load(tp[1])
    ids_tr = tr_tok["input_ids"][:, :max_len]
    mask_tr = tr_tok["attention_mask"][:, :max_len]
    ids_te = te_tok["input_ids"][:, :max_len]
    mask_te = te_tok["attention_mask"][:, :max_len]
    cov = float((mask_tr.sum(1) < max_len - 1).mean())
    print(f"  缓存 {os.path.basename(tp[0])} 宽 {tr_tok['input_ids'].shape[1]} → 截取 {max_len}")
    print(f"  未被截断序列占比 ≈ {cov:.1%}")
    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    tr_ds = TokenDS(torch.from_numpy(ids_tr[idx_train]).long(),
                    torch.from_numpy(mask_tr[idx_train]).long(),
                    torch.from_numpy(y_train.astype(np.float32)))
    va_ds = TokenDS(torch.from_numpy(ids_tr[idx_val]).long(),
                    torch.from_numpy(mask_tr[idx_val]).long(),
                    torch.from_numpy(y_val.astype(np.float32)))
    te_ds = TokenDS(torch.from_numpy(ids_te).long(),
                    torch.from_numpy(mask_te).long(),
                    torch.zeros(len(ids_te), N_LABELS))
    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0,
                           collate_fn=collate_dyn, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=0,
                           collate_fn=collate_dyn, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=0,
                           collate_fn=collate_dyn, pin_memory=True)
    print(f"  步/epoch={len(tr_loader)} 有效 batch={bs * accum}")

    print("[2/4] 加载 ESM2-650M + LoRA + MAH 头...")
    base = load_esm()
    model, n_layers, freeze_first = attach_lora(base, freeze_first)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  层数 {n_layers} 冻结前 {freeze_first} 可训练 {n_trainable / 1e6:.2f}M")
    model = model.cuda()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.01)
    n_steps = max(1, (len(tr_loader) * n_epochs + accum - 1) // accum)
    warmup = max(1, int(n_steps * 0.06))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, 0.5 * (1 + np.cos(np.pi * s / n_steps))))
    scaler = torch.amp.GradScaler("cuda")
    criterion = AsymmetricLoss()

    start_ep = 0
    start_step = 0
    best_f1 = -1.0
    best_epoch = 0
    bad = 0
    do_resume = (not fresh) and os.path.isfile(LAST_PATH)
    if fresh and os.path.isfile(LAST_PATH):
        print("  --fresh：忽略已有 last.pt")
    if do_resume:
        ckpt = load_ckpt(LAST_PATH, map_location="cuda")
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_ep = int(ckpt.get("epoch", 0))
        start_step = int(ckpt.get("step", 0))
        best_f1 = float(ckpt.get("best_f1", -1))
        best_epoch = int(ckpt.get("best_epoch", 0))
        bad = int(ckpt.get("patience_counter", 0))
        n_loader = len(tr_loader)
        if start_step >= n_loader:
            start_ep += 1
            start_step = 0
        print(f"  resume {LAST_PATH}: ep{start_ep + 1} step>{start_step} "
              f"best_f1={best_f1:.4f} patience={bad}", flush=True)

    if smoke:
        print("[smoke] 2 步前向+反传...")
        model.train()
        it = iter(tr_loader)
        for i in range(2):
            ids, mask, y = next(it)
            ids, mask, y = ids.cuda(), mask.cuda(), y.cuda()
            with torch.amp.autocast("cuda"):
                loss = criterion(model(input_ids=ids, attention_mask=mask), y) / accum
            scaler.scale(loss).backward()
            mem = torch.cuda.max_memory_allocated() / 1024 ** 2
            print(f"  step{i + 1} loss={loss.item() * accum:.4f} peak={mem:.0f} MiB")
        opt.zero_grad(set_to_none=True)
        print(f"  smoke ok  peak={torch.cuda.max_memory_allocated() / 1024 ** 2:.0f} MiB")
        print("=" * 60)
        return

    print(f"[3/4] 训练 {n_epochs} epoch...")
    ep, step = start_ep, start_step
    try:
        for ep in range(start_ep, n_epochs):
            model.train()
            total_loss = 0.0
            seen = 0
            t_ep = time.time()
            opt.zero_grad(set_to_none=True)
            skip = start_step if ep == start_ep else 0
            it = tr_loader
            if skip:
                print(f"    skip first {skip} steps of ep{ep + 1} "
                      f"(shuffle 顺序与中断前不完全相同)", flush=True)
                it = islice(tr_loader, skip, None)
            pbar = tqdm(it, desc=f"ep{ep + 1}/{n_epochs}",
                        initial=skip, total=len(tr_loader),
                        unit="step", dynamic_ncols=True, mininterval=1.0)
            for step, (ids, mask, y) in enumerate(pbar, start=skip):
                ids = ids.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(input_ids=ids, attention_mask=mask)
                    loss = criterion(logits, y) / accum
                scaler.scale(loss).backward()
                if (step + 1) % accum == 0 or (step + 1) == len(tr_loader):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                bs_now = ids.size(0)
                total_loss += loss.item() * accum * bs_now
                seen += bs_now
                if (step + 1) % 20 == 0 or (step + 1) == len(tr_loader):
                    pbar.set_postfix(loss=f"{total_loss / max(seen, 1):.4f}",
                                     lr=f"{opt.param_groups[0]['lr']:.1e}")
                if (step + 1) % 400 == 0:
                    save_last(LAST_PATH, model, opt, sched, scaler,
                              ep, step + 1, best_f1, best_epoch, bad)
                    torch.cuda.empty_cache()
            save_last(LAST_PATH, model, opt, sched, scaler,
                      ep, len(tr_loader), best_f1, best_epoch, bad)
            dt = time.time() - t_ep
            msg = (f"  ep{ep + 1}: loss={total_loss / max(seen, 1):.4f} "
                   f"{dt:.0f}s ({seen / max(dt, 1e-6):.1f} seq/s)")
            do_val = ((ep + 1) % VALIDATE_EVERY == 0) or (ep == n_epochs - 1) or (ep < 1)
            if do_val:
                torch.cuda.empty_cache()
                f1 = macro_f1(predict_probs(model, va_loader), y_val, 0.5)
                msg += f" F1(0.5)={f1:.4f}"
                print(msg, flush=True)
                if f1 > best_f1 + 1e-4:
                    best_f1 = f1
                    best_epoch = ep + 1
                    bad = 0
                    torch.save({"state_dict": model.state_dict(), "f1": best_f1,
                                "epoch": best_epoch}, BEST_PATH)
                    print(f"    保存 best F1={best_f1:.4f}", flush=True)
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        print(f"    早停 (patience={PATIENCE})", flush=True)
                        break
            else:
                print(msg, flush=True)
            save_last(LAST_PATH, model, opt, sched, scaler,
                      ep, len(tr_loader), best_f1, best_epoch, bad)
            start_step = 0
    except KeyboardInterrupt:
        save_last(LAST_PATH, model, opt, sched, scaler,
                  ep, step + 1, best_f1, best_epoch, bad)
        print(f"\n  Ctrl+C 已保存 {LAST_PATH}  ep{ep + 1} step={step + 1}", flush=True)
        return
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            save_last(LAST_PATH, model, opt, sched, scaler,
                      ep, step + 1, best_f1, best_epoch, bad)
            print(f"  异常退出已保存 {LAST_PATH}  ep{ep + 1} step={step + 1}", flush=True)
        except Exception:
            pass
        raise

    print("[4/4] 评估...")
    if os.path.isfile(BEST_PATH):
        ckpt = load_ckpt(BEST_PATH, map_location="cuda")
        model.load_state_dict(ckpt["state_dict"])
    probs_val = predict_probs(model, va_loader)
    f1_05 = macro_f1(probs_val, y_val, 0.5)
    cands = np.linspace(0.01, 0.99, 100)
    best_t = np.full(N_LABELS, 0.5)
    for i in tqdm(range(N_LABELS), desc="    阈值", unit="label"):
        yt = y_val[:, i]
        if yt.sum() == 0:
            continue
        p = probs_val[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (p & ytm).sum(0)
        fp = (p & ~ytm).sum(0)
        fn = (~p & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best_t[i] = cands[f1t.argmax()]
    f1_100 = np.mean([
        f1_score(y_val[:, i].astype(bool), probs_val[:, i] >= best_t[i], zero_division=0)
        for i in range(N_LABELS) if y_val[:, i].sum() > 0])
    r100, p100 = macro_rp(probs_val, y_val, best_t)
    auc = macro_auc(probs_val, y_val)
    useful_ind = bool(f1_05 > C002B_F05)
    print(f"  E006d(W={max_len}): F1(0.5)={f1_05:.4f} F1(100)={f1_100:.4f} "
          f"R={r100:.4f} P={p100:.4f} AUC={auc:.4f}")
    print(f"  对照 C002b F1(0.5)={C002B_F05}: {'超过' if useful_ind else '未超'}")
    save_kw = {"val": probs_val}
    if useful_ind:
        probs_test = predict_probs(model, te_loader)
        save_kw["test"] = probs_test
        test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test.csv"))
        pred = (probs_test >= best_t[None, :]).astype(np.int8)
        sub = pd.concat([test_df[["protein_id"]], pd.DataFrame(pred, columns=label_cols)], axis=1)
        sub.to_csv(os.path.join(SUB_DIR, "submission_E006d_mah_asl_len.csv"), index=False)
    np.savez_compressed(PROBS_PATH, **save_kw)
    report = {
        "model": "ESM2-650M + LoRA r=8 last6 + residue MAH + ASL (long-seq)",
        "max_len": max_len,
        "untruncated_frac": round(cov, 4),
        "f1_05": round(float(f1_05), 4),
        "f1_100": round(float(f1_100), 4),
        "recall": round(r100, 4),
        "precision": round(p100, 4),
        "auc": round(auc, 4),
        "best_f1_05": round(float(best_f1), 4),
        "useful_independent": useful_ind,
        "c002b_f05": C002B_F05,
        "loss": "ASL",
        "epochs": n_epochs, "bs": bs, "accum": accum, "lr": lr,
        "freeze_first": freeze_first, "n_heads": N_HEADS,
        "device": dev,
        "seconds": round(time.time() - t0),
        "conclusion": (
            f"650M LoRA+MAH+ASL 长序列(W={max_len}) 独立超 C002b，可接到 C011a"
            if useful_ind else
            f"650M LoRA+MAH+ASL 长序列(W={max_len}) 未超 C002b 0.2714，读出/损失刀收口"
        ),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}  用时 {report['seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
